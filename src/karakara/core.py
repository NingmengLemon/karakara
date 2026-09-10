from __future__ import annotations

import tempfile
from logging import getLogger
from pathlib import Path
from typing import Literal

from lemony_lrc_parser import BasicLyricLine, LyricLine, Lyrics, LyricToken
from lemony_lrc_parser.offset import apply_delta, iter_all_timestamps
from numpy.typing import NDArray

from karakara.aligner.abc import AbstractAligner, AlignedWord, LangCode
from karakara.debug import AudioDumper
from karakara.offset import build_energy_curve, estimate_offset, score_vocal_activity
from karakara.preprocess import (
    AudioPreprocessConfig,
    compress_dynamic_range,
    normalize_loudness,
    suppress_vibrato,
)
from karakara.separator.abc import AbstractStemSeparator
from karakara.utils.io import load_audio_native, ms2sample
from karakara.utils.lang import detect_dominant_lang, detect_lang
from karakara.utils.metadata import MetadataFilter

logger = getLogger(__name__)

ExistingBywordPolicy = Literal["realign", "preserve"]


def _preprocess_vocals(
    audio: str | Path,
    *,
    separator: AbstractStemSeparator,
    config: AudioPreprocessConfig,
    dumper: AudioDumper,
    separate_work_dir: str | Path | None = None,
) -> tuple[NDArray, int]:
    """分离人声并应用配置的预处理步骤。

    人声轨先由分离器写到临时目录，读回内存后立即删除。采样率取音轨文件的原生值，
    不再假设某个固定值——不同分离模型（Demucs / MDX / RoFormer…）各有自己的
    采样率，主进程没有必要、也不应该替它决定。
    """
    work_root = Path(separate_work_dir) if separate_work_dir is not None else None
    if work_root is not None:
        work_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix="karakara-sep-", dir=str(work_root) if work_root else None
    ) as tmp:
        stems = separator.separate(audio, tmp, stems=(separator.VOCAL_STEM_NAME,))
        vocal_np, sample_rate = load_audio_native(stems[separator.VOCAL_STEM_NAME])

    dumper.dump("01_vocal_stem", vocal_np, sample_rate)

    if vocal_np.ndim != 2:
        raise ValueError(
            f"Vocal stem must have shape (channels, samples), got {vocal_np.shape}"
        )
    if config.normalize:
        vocal_np = normalize_loudness(vocal_np, config.target_dbfs)
        dumper.dump("02_normalized", vocal_np, sample_rate)
    if config.suppress_vibrato:
        vocal_np = suppress_vibrato(
            vocal_np,
            sample_rate,
            threshold_hz=config.vibrato_threshold_hz,
            smooth_window_ms=config.vibrato_smooth_window_ms,
        )
        dumper.dump("03_vibrato_suppressed", vocal_np, sample_rate)
    if config.compress:
        vocal_np = compress_dynamic_range(
            vocal_np,
            sample_rate,
            threshold_dbfs=config.comp_threshold_dbfs,
            ratio=config.comp_ratio,
            attack_ms=config.comp_attack_ms,
            release_ms=config.comp_release_ms,
        )
        dumper.dump("04_compressed", vocal_np, sample_rate)
    return vocal_np, sample_rate


def _apply_offset(
    lyrics: Lyrics,
    audio: NDArray,
    sample_rate: int,
    *,
    metadata_filter: MetadataFilter,
    offset_ms: float | None,
) -> None:
    """估计并就地应用全局时间偏移，必要时避免负时间戳。"""
    applied_offset = offset_ms
    if applied_offset is None:
        applied_offset = estimate_offset(
            audio, lyrics, sample_rate, metadata_filter=metadata_filter
        )
    if applied_offset == 0:
        return

    apply_delta(lyrics, int(applied_offset))
    logger.info(f"Applied global offset: {applied_offset:+.0f}ms")
    min_ts = min(iter_all_timestamps(lyrics), default=0)
    if min_ts < 0:
        apply_delta(lyrics, -min_ts)
        applied_offset += -min_ts
        logger.info(
            f"Shifted by {-min_ts}ms to avoid negative timestamps "
            f"(final offset={applied_offset:+.0f}ms)"
        )


def _is_byword_line(line: LyricLine) -> bool:
    """判断行中是否已有词级时间标签。"""
    return any(
        token.start is not None or token.end is not None for token in line.content
    )


def _should_preserve_line(
    line: LyricLine,
    text: str,
    *,
    index: int,
    target_lang: LangCode | None,
    metadata_filter: MetadataFilter,
    existing_byword_policy: ExistingBywordPolicy,
) -> bool:
    """判断该行是否应跳过对齐并保留原样。"""
    if not text:
        return True
    if existing_byword_policy == "preserve" and _is_byword_line(line):
        logger.info(f"preserve existing byword line {index}: {text!r}")
        return True
    language = detect_lang(text)
    if target_lang is not None and language != target_lang:
        logger.debug(f"skip line {index}: lang={language!r} != target={target_lang!r}")
        return True
    if metadata_filter(text):
        logger.info(f"skip metadata line {index}: {text!r}")
        return True
    return False


def _line_sample_range(
    lyrics: Lyrics,
    index: int,
    sample_rate: int,
) -> tuple[int, int | None]:
    """计算一行歌词对应的样本区间。"""
    line = lyrics[index]
    start = ms2sample(max(line.start, 0), sample_rate)
    if line.end is not None:
        return start, ms2sample(line.end, sample_rate)
    if index < len(lyrics) - 1 and lyrics[index + 1].start is not None:
        return start, ms2sample(lyrics[index + 1].start, sample_rate)
    return start, None


def _build_aligned_content(
    text: str,
    words: list[AlignedWord],
    *,
    line_start: int,
    line_index: int,
) -> BasicLyricLine | None:
    """将对齐器输出映射回原始歌词文本，失败时返回 ``None``。"""
    tokens: list[LyricToken] = []
    text_index = 0
    for word in words:
        if word.position is None:
            continue
        next_index = text.find(word.word, text_index)
        if next_index == -1:
            logger.warning(
                f"aligned word {word.word!r} not found in text "
                f"at pos {text_index}, skipping"
            )
            continue
        start, end = word.position
        if next_index > text_index:
            tokens.append(
                LyricToken(
                    start=tokens[-1].end if tokens else None,
                    end=start + line_start,
                    content=text[text_index:next_index],
                )
            )
        logger.debug(
            f"got aligned word: {word.word!r}, at time {word.position!r}ms "
            f"at line {line_index} [{next_index}, {next_index + len(word.word)}]"
        )
        tokens.append(
            LyricToken(
                start=start + line_start, end=end + line_start, content=word.word
            )
        )
        text_index = next_index + len(word.word)

    tail = text[text_index:]
    if tail:
        tokens.append(
            LyricToken(
                start=tokens[-1].end if tokens else None,
                end=None,
                content=tail,
            )
        )
    return BasicLyricLine(tokens) if tokens else None


def _align_line(
    line: LyricLine,
    text: str,
    *,
    index: int,
    audio: NDArray,
    sample_rate: int,
    aligner: AbstractAligner,
    dumper: AudioDumper,
    language: LangCode | None,
) -> LyricLine | None:
    """对单行执行对齐，不能可靠映射时返回 ``None``。"""
    audio_piece = audio
    dumper.dump(f"05_line_{index}", audio_piece, sample_rate)
    try:
        words = aligner.align(audio_piece, text, sample_rate, language=language)
    except Exception as exc:
        logger.error(f"Error occurred while aligning line {index}: {exc}")
        return None

    content = _build_aligned_content(
        text, words, line_start=line.start, line_index=index
    )
    if content is None:
        logger.warning(f"Line {index}: no usable alignment result, preserving original")
        return None
    end = content[-1].end if content and content[-1].end is not None else line.end
    if content and content[-1].end is not None:
        content[-1].end = None
    return LyricLine(
        start=line.start,
        end=end,
        content=content,
        reference_lines=[reference.copy() for reference in line.reference_lines],
    )


def gen_kara(
    lyrics: Lyrics,
    audio: str | Path,
    aligner: AbstractAligner,
    separator: AbstractStemSeparator,
    *,
    metadata_filter: MetadataFilter,
    target_lang: LangCode | None = None,
    aligner_language: Literal["auto"] | LangCode = "auto",
    preprocess_config: AudioPreprocessConfig | None = None,
    dump_dir: str | Path | None = None,
    separate_work_dir: str | Path | None = None,
    offset_ms: float | None = None,
    min_vocal_activity: float = 0.01,
    existing_byword_policy: ExistingBywordPolicy = "realign",
) -> Lyrics:
    """根据音频和行级歌词生成词级逐字歌词。

    已有逐字时间标签的行由 ``existing_byword_policy`` 控制：``"realign"`` 会
    将所有 token 文本拼接后重新对齐；``"preserve"`` 则原样保留。

    ``target_lang`` 用于**跳过**语言不符的行（例如夹在日文歌词里的中文翻译行）；
    ``aligner_language`` 决定**送给对齐器**的语言，``"auto"`` 时按整首歌的
    行级多数票判定。

    ``separate_work_dir`` 是分离中间产物的落盘位置；``None`` 用系统临时目录。
    """
    if min_vocal_activity < 0:
        raise ValueError("min_vocal_activity must be non-negative")
    if existing_byword_policy not in ("realign", "preserve"):
        raise ValueError(f"Unknown existing_byword_policy: {existing_byword_policy!r}")

    working_lyrics = lyrics.copy()
    dumper = AudioDumper(dump_dir)
    config = (
        preprocess_config if preprocess_config is not None else AudioPreprocessConfig()
    )
    vocal_np, sample_rate = _preprocess_vocals(
        audio,
        separator=separator,
        config=config,
        dumper=dumper,
        separate_work_dir=separate_work_dir,
    )
    total_samples = vocal_np.shape[-1]
    logger.info(f"Total samples: {total_samples}")
    language: LangCode | None
    if aligner_language == "auto":
        language = detect_dominant_lang(
            line.text
            for line in working_lyrics
            if line.text and not metadata_filter(line.text)
        )
        if language is None:
            logger.info("aligner language unresolved, falling back to aligner default")
    else:
        language = aligner_language
    logger.info(f"aligner language: {language or 'aligner default'}")
    _apply_offset(
        working_lyrics,
        vocal_np,
        sample_rate,
        metadata_filter=metadata_filter,
        offset_ms=offset_ms,
    )
    energy_curve = build_energy_curve(vocal_np, sample_rate)

    result = Lyrics(metadata=working_lyrics.metadata)
    for index, line in enumerate(working_lyrics):
        text = line.text
        if _should_preserve_line(
            line,
            text,
            index=index,
            target_lang=target_lang,
            metadata_filter=metadata_filter,
            existing_byword_policy=existing_byword_policy,
        ):
            result.append(line.copy())
            continue

        start, end = _line_sample_range(working_lyrics, index, sample_rate)
        logger.info(f"aligning line {index}: sample_point[{start}, {end}] {text!r}")
        if end is not None and start > end:
            logger.warning(
                f"Line {index}: invalid audio segment: [{start}, {end}], preserving original"
            )
            result.append(line.copy())
            continue
        if (end is not None and end >= total_samples) or start >= total_samples:
            logger.warning(
                f"Line {index}: audio segment out of bounds: [{start}, {end}] "
                f"(total samples: {total_samples}), preserving original"
            )
            result.append(line.copy())
            continue

        segment_end = end if end is not None else total_samples
        activity = score_vocal_activity(
            energy_curve,
            start_ms=start / sample_rate * 1000,
            end_ms=segment_end / sample_rate * 1000,
        )
        if min_vocal_activity > 0 and activity < min_vocal_activity:
            logger.info(
                f"skip low vocal activity line {index}: activity={activity:.3f} "
                f"< threshold={min_vocal_activity:.3f}: {text!r}"
            )
            result.append(line.copy())
            continue

        audio_piece = vocal_np[:, start:end] if end is not None else vocal_np[:, start:]
        aligned_line = _align_line(
            line,
            text,
            index=index,
            audio=audio_piece,
            sample_rate=sample_rate,
            aligner=aligner,
            dumper=dumper,
            language=language,
        )
        result.append(aligned_line if aligned_line is not None else line.copy())
    return result
