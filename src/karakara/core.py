from __future__ import annotations

from copy import deepcopy
from logging import getLogger
from pathlib import Path
from typing import Literal

import numpy as np
from lemony_lrc_parser import BasicLyricLine, LyricLine, Lyrics, LyricToken
from lemony_lrc_parser.offset import apply_delta, iter_all_timestamps
from numpy.typing import NDArray

from karakara.aligner.abc import AbstractAligner, AlignedWord
from karakara.debug import AudioDumper
from karakara.offset import build_energy_curve, estimate_offset, score_vocal_activity
from karakara.preprocess import (
    AudioPreprocessConfig,
    compress_dynamic_range,
    normalize_loudness,
    suppress_vibrato,
)
from karakara.separator.abc import AbstractStemSeparator
from karakara.utils.io import load_audio, ms2sample
from karakara.utils.lang import detect_lang
from karakara.utils.metadata import MetadataFilter

logger = getLogger(__name__)


def gen_kara(
    lyrics: Lyrics,
    audio: str | Path,
    aligner: AbstractAligner,
    separator: AbstractStemSeparator,
    *,
    metadata_filter: MetadataFilter,
    target_lang: Literal["en", "ja", "zh"] | None = None,
    preprocess_config: AudioPreprocessConfig | None = None,
    dump_dir: str | Path | None = None,
    offset_ms: float | None = None,
    min_vocal_activity: float = 0.01,
) -> Lyrics:
    """根据音频和行级歌词生成词级逐字歌词。

    流水线：加载音频 → 人声分离 → 音频预处理 → 偏移估计 → 按行对齐 → 替换 LyricToken。

    Args:
        lyrics: 已解析的 Lyrics 对象（行级歌词）
        audio: 音频文件路径
        aligner: 对齐器实例
        separator: 人声分离器实例
        metadata_filter: 元数据行过滤器，用于跳过作词/作曲等非歌词行。
        target_lang: 目标处理语言，None 时处理所有检测到的语言
        preprocess_config: 音频预处理配置，None 时使用默认值
        dump_dir: 调试音频导出目录，None 时不导出
        offset_ms: 全局时间偏移 (ms)。
            * 正值：LRC 偏早，延迟 LRC 后再切音频
            * 负值：LRC 偏晚，提前 LRC 后再切音频
            * None：自动估计偏移量
        min_vocal_activity: 人声活动度低于该阈值时跳过强制对齐并保留原行；
            取 ``0`` 可关闭。活动度是相对于整首人声音频峰值归一化的 RMS 均值。

    Returns:
        词级歌词的 Lyrics 对象（content 中每个 LyricToken 带有 start/end）
    """
    if min_vocal_activity < 0:
        raise ValueError("min_vocal_activity must be non-negative")

    lyrics = deepcopy(lyrics)
    dumper = AudioDumper(dump_dir)

    # ---------- 加载 & 分离人声 ----------
    sample_rate = separator.samplerate
    audio_np = load_audio(audio, sample_rate=sample_rate)
    dumper.dump("00_input", audio_np, sample_rate)
    stems = separator.separate(audio_np)
    vocal_stem = stems[separator.VOCAL_STEM_NAME]
    dumper.dump("01_vocal_stem", vocal_stem, sample_rate)

    # ---------- 音频预处理 ----------
    config = (
        preprocess_config if preprocess_config is not None else AudioPreprocessConfig()
    )
    # 分离器接口约定返回 (channels, samples)。保留全部声道而非固定取左声道。
    vocal_np: NDArray[np.float32] = vocal_stem
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

    total_samples = vocal_np.shape[-1]
    logger.info(f"Total samples: {total_samples}")

    # ---------- 偏移估计 ----------
    if offset_ms is None:
        offset_ms = estimate_offset(
            vocal_np, lyrics, sample_rate, metadata_filter=metadata_filter
        )

    if offset_ms != 0:
        apply_delta(lyrics, int(offset_ms))
        logger.info(f"Applied global offset: {offset_ms:+.0f}ms")

        # 偏移可能导致负时间戳——找出最小时间戳，
        # 若为负则整体再平移，使最小值为 0，保持相对时序不变
        min_ts = min(iter_all_timestamps(lyrics), default=0)
        if min_ts < 0:
            apply_delta(lyrics, -min_ts)
            offset_ms += -min_ts
            logger.info(
                f"Shifted by {-min_ts}ms to avoid negative timestamps "
                f"(final offset={offset_ms:+.0f}ms)"
            )

    # 偏移校正后才计算每行活动度。该曲线不参与首次偏移估计，避免低活动度
    # 判定和偏移校正之间形成反馈循环。
    energy_curve = build_energy_curve(vocal_np, sample_rate)

    # ---------- 逐行对齐 ----------
    result = Lyrics(metadata=lyrics.metadata)
    for idx, line in enumerate(lyrics):
        # 语言过滤
        text = ""
        if len(line.content) == 1 and (text := line.content[0].content):
            lang = detect_lang(text)
            if target_lang is not None and lang != target_lang:
                logger.debug(
                    f"skip line {idx}: lang={lang!r} != target={target_lang!r}"
                )
                result.append(deepcopy(line))
                continue
            if metadata_filter(text):
                logger.info(f"skip metadata line {idx}: {text!r}")
                result.append(deepcopy(line))
                continue

        if not text:
            result.append(deepcopy(line))
            continue

        # 确定音频片段边界（时间戳已应用偏移）
        start = ms2sample(max(line.start or 0, 0), sample_rate)
        end: int | None = None

        if line.end is not None:
            end = ms2sample(line.end, sample_rate)
        elif idx < len(lyrics) - 1 and (next_line := lyrics[idx + 1]).start is not None:
            end = ms2sample(next_line.start, sample_rate)

        logger.info(f"aligning line {idx}: sample_point[{start}, {end}] {text!r}")
        if end is not None and start > end:
            logger.warning(
                f"Line {idx}: invalid audio segment: [{start}, {end}], preserving original"
            )
            result.append(deepcopy(line))
            continue
        if (end is not None and end >= total_samples) or (start >= total_samples):
            logger.warning(
                f"Line {idx}: audio segment out of bounds: [{start}, {end}] "
                f"(total samples: {total_samples}), preserving original"
            )
            result.append(deepcopy(line))
            continue

        segment_end = end if end is not None else total_samples
        if min_vocal_activity > 0:
            activity = score_vocal_activity(
                energy_curve,
                start_ms=start / sample_rate * 1000,
                end_ms=segment_end / sample_rate * 1000,
            )
            if activity < min_vocal_activity:
                logger.info(
                    f"skip low vocal activity line {idx}: activity={activity:.3f} "
                    f"< threshold={min_vocal_activity:.3f}: {text!r}"
                )
                result.append(deepcopy(line))
                continue

        # 音频约定为 (channels, samples)；时间范围必须沿最后一维切片。
        audio_piece = vocal_np[:, start:end] if end is not None else vocal_np[:, start:]
        dumper.dump(f"05_line_{idx}", audio_piece, sample_rate)
        try:
            words: list[AlignedWord] = aligner.align(audio_piece, text, sample_rate)
        except Exception as e:
            logger.error(f"Error occurred while aligning line {idx}: {e}")
            words = [AlignedWord(word=text, position=None)]

        # 组装逐字 LyricToken
        words_kara: list[LyricToken] = []
        iidx = 0
        for word in words:
            if not (pos := word.position):
                continue
            next_idx = text.find(word.word, iidx)
            if next_idx == -1:
                logger.warning(
                    f"aligned word {word.word!r} not found in text "
                    f"at pos {iidx}, skipping"
                )
                continue
            logger.debug(
                f"got aligned word: {word.word!r}, at time {pos!r}ms "
                f"at line {idx} [{next_idx}, {next_idx + len(word.word)}]"
            )
            if next_idx > iidx:
                # 补上前一个单词和当前单词间的空隙
                words_kara.append(
                    LyricToken(
                        start=words_kara[-1].end if words_kara else None,
                        end=pos[0] + (line.start or 0),
                        content=text[iidx:next_idx],
                    )
                )

            words_kara.append(
                LyricToken(
                    start=pos[0] + (line.start or 0),
                    end=pos[1] + (line.start or 0),
                    content=word.word,
                )
            )
            iidx = next_idx + len(word.word)

        # 尾部剩余文本（仅在有内容时添加）
        tail = text[iidx:]
        if tail:
            words_kara.append(
                LyricToken(
                    start=words_kara[-1].end if words_kara else None,
                    end=None,
                    content=tail,
                )
            )

        # 对齐服务失败、未返回位置，或返回文本无法对应原歌词时，不能以空行
        # 覆盖原歌词；保留原始行可确保失败降级不会造成数据丢失。
        if not words_kara:
            logger.warning(
                f"Line {idx}: no usable alignment result, preserving original"
            )
            result.append(deepcopy(line))
            continue

        new_line = LyricLine(
            start=line.start,
            end=line.end,
            content=BasicLyricLine(words_kara),
            reference_lines=line.reference_lines,
        )
        if words_kara and words_kara[-1].end is not None:
            new_line.end = words_kara[-1].end
            new_line.content[-1].end = None
        result.append(new_line)

    return result
