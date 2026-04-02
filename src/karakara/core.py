from __future__ import annotations

from logging import getLogger
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from karakara.aligner.abc import AbstractAligner
from karakara.debug import AudioDumper
from karakara.preprocess import AudioPreprocessConfig
from karakara.separator.abc import AbstractStemSeparator
from karakara.spl_parser import Lyrics, LyricWord
from karakara.utils.io import load_audio, ms2sample
from karakara.utils.lang import detect_lang
from karakara.utils.metadata import is_metadataline

logger = getLogger(__name__)

# 支持对齐的语言列表（与 aligner 实现对应）
SUPPORTED_LANGS: set[Literal["en", "ja", "zh"]] = {"en", "ja", "zh"}


def gen_kara(
    lyrics: Lyrics,
    audio: str | Path,
    aligner: AbstractAligner,
    separator: AbstractStemSeparator,
    *,
    target_lang: Literal["en", "ja", "zh"] | None = None,
    preprocess_config: AudioPreprocessConfig | None = None,
    dump_dir: str | Path | None = None,
) -> Lyrics:
    """根据音频和行级歌词生成词级逐字歌词。

    流水线：加载音频 → 人声分离 → 音频预处理 → 按行对齐 → 替换 LyricWord。

    Args:
        lyrics: 已解析的 Lyrics 对象（行级歌词）
        audio: 音频文件路径
        aligner: 对齐器实例，None 时默认使用 GentleAligner
        target_lang: 目标处理语言，None 时处理所有检测到的语言
        preprocess_config: 音频预处理配置，None 时使用默认值
        dump_dir: 调试音频导出目录，None 时不导出

    Returns:
        词级歌词的 Lyrics 对象（content 中每个 LyricWord 带有 start/end）
    """
    lyrics = lyrics.model_copy(deep=True)
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
    vocal_np: NDArray[np.float32] = vocal_stem[0]

    if config.normalize:
        from karakara.preprocess import normalize_loudness

        vocal_np = normalize_loudness(vocal_np, config.target_dbfs)
        dumper.dump("02_normalized", vocal_np, sample_rate)

    if config.suppress_vibrato:
        from karakara.preprocess import suppress_vibrato

        vocal_np = suppress_vibrato(
            vocal_np,
            sample_rate,
            threshold_hz=config.vibrato_threshold_hz,
            smooth_window_ms=config.vibrato_smooth_window_ms,
        )
        dumper.dump("03_vibrato_suppressed", vocal_np, sample_rate)

    if config.compress:
        from karakara.preprocess import compress_dynamic_range

        vocal_np = compress_dynamic_range(
            vocal_np,
            sample_rate,
            threshold_dbfs=config.comp_threshold_dbfs,
            ratio=config.comp_ratio,
            attack_ms=config.comp_attack_ms,
            release_ms=config.comp_release_ms,
        )
        dumper.dump("04_compressed", vocal_np, sample_rate)

    # ---------- 逐行对齐 ----------
    for idx, line in enumerate(lyrics.lines):
        # 语言过滤
        text = ""
        if len(line.content) == 1 and (text := line.content[0].content):
            lang = detect_lang(text)
            if target_lang is not None and lang != target_lang:
                logger.debug(
                    f"skip line {idx}: lang={lang!r} != target={target_lang!r}"
                )
                continue
            if is_metadataline(text):
                logger.info(f"skip metadata line {idx}: {text!r}")
                continue

        if not text:
            continue

        # 确定音频片段边界
        start = ms2sample(line.start or 0, sample_rate)
        end: int | None = None
        if idx < len(lyrics.lines) - 1:
            if line.end:
                end = ms2sample(line.end, sample_rate)
            elif (next_line := lyrics.lines[idx + 1]).start:
                end = ms2sample(next_line.start, sample_rate)

        logger.info(f"aligning line {idx}: sample_point[{start}, {end}] {text!r}")
        if end is not None and start > end:
            continue

        audio_piece = vocal_np[start:end] if end else vocal_np[start:]
        dumper.dump(f"05_line_{idx}", audio_piece, sample_rate)
        words = aligner.align(audio_piece, text, sample_rate)

        # 组装逐字 LyricWord
        words_kara: list[LyricWord] = []
        iidx = 0
        for word in words:
            if not (pos := word.position):
                continue
            next_idx = text.find(word.word, iidx)
            logger.debug(
                f"got aligned word: {word.word!r}, at time {pos!r}ms "
                f"at line {idx} [{next_idx}, {next_idx + len(word.word)}]"
            )
            if next_idx > iidx:
                # 补上前一个单词和当前单词间的空隙
                words_kara.append(
                    LyricWord(
                        start=words_kara[-1].end if words_kara else None,
                        end=pos[0] + (line.start or 0),
                        content=text[iidx:next_idx],
                    )
                )

            words_kara.append(
                LyricWord(
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
                LyricWord(
                    start=words_kara[-1].end if words_kara else None,
                    end=None,
                    content=tail,
                )
            )

        line.content = words_kara

    return lyrics
