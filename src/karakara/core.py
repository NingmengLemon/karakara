from __future__ import annotations

import tempfile
from logging import getLogger
from pathlib import Path
from typing import Literal

from lemony_lrc_parser import BasicLyricLine, LyricLine, Lyrics, LyricToken
from lemony_lrc_parser.offset import apply_delta
from numpy.typing import NDArray

from karakara.aligner.abc import AbstractAligner, AlignedWord, LangCode
from karakara.aligner.postprocess import (
    DEGENERATE_RATIO_WARN,
    ZeroLengthStats,
    count_zero_length,
    refine_collapsed_words,
)
from karakara.debug import AudioDumper
from karakara.offset import (
    DEFAULT_ANCHOR_TOLERANCE_MS,
    build_energy_curve,
    estimate_offset,
    score_vocal_activity,
    suggest_offset_from_onset,
    validate_estimated_offset,
)
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


def _validate_auto_offset(
    lyrics: Lyrics,
    energy_curve: NDArray,
    candidate_ms: float,
    *,
    metadata_filter: MetadataFilter,
    anchor_tolerance_ms: float = DEFAULT_ANCHOR_TOLERANCE_MS,
) -> float:
    """校验自动估计出来的偏移：不通过就返回 ``0.0``（不偏移）。

    两道校验互相独立，各自都能单独否决：

    1. **行区间对比度**：估计值必须比「不偏移」更能让整段行区间落在人声里。
    2. **首次人声锚点**：``第一次持续人声`` 与 ``第一条歌词行`` 之差给出一个物理
       锚点，它与估计值分歧过大时说明两者无法调和——典型情形是 LRC 与音频属于
       不同剪辑（实测有一首需要 +17 秒），此时任何全局常量偏移都是错的。

    判据退化（算不出对比度 / 检不出人声）时不否决，交由另一道校验决定。
    """
    accepted, base, updated = validate_estimated_offset(
        energy_curve, lyrics, candidate_ms, metadata_filter=metadata_filter
    )
    if not accepted:
        logger.warning(
            f"自动偏移估计 {candidate_ms:+.0f}ms 未通过校验"
            f"（行区间对比度 {updated:.4f} 不优于不偏移的 {base:.4f}），"
            f"按不偏移处理；如需强制，请显式传 --offset {candidate_ms:+.0f}"
        )
        return 0.0

    anchor = suggest_offset_from_onset(
        lyrics, energy_curve, metadata_filter=metadata_filter
    )
    if anchor is not None and abs(anchor - candidate_ms) > anchor_tolerance_ms:
        logger.warning(
            f"自动偏移估计 {candidate_ms:+.0f}ms 与「首次持续人声」锚点 "
            f"{anchor:+.0f}ms 分歧超过 {anchor_tolerance_ms:.0f}ms，按不偏移处理。"
            f"这通常意味着该 LRC 与音频属于不同剪辑（此时任何全局偏移都不成立），"
            f"建议用 --dump-dir 检查产物后再用 --offset 手动指定"
        )
        return 0.0

    logger.info(
        f"自动偏移 {candidate_ms:+.0f}ms 通过校验"
        + (f"（锚点 {anchor:+.0f}ms）" if anchor is not None else "（锚点不可用）")
    )
    return candidate_ms


def _clamp_negative_timestamps(
    lyrics: Lyrics, *, metadata_filter: MetadataFilter
) -> tuple[int, list[int]]:
    """把所有负时间戳夹到 0，返回 ``(被夹个数, 被夹且参与对齐的行号)``。

    ``lemony-lrc-parser`` 的 ``format_timetag`` 遇到负值直接抛
    ``TimestampUnderflowError``，所以输出前必须保证时间轴非负。

    "参与对齐的行" 只用于告警：连它们都被夹住，说明偏移估计本身就过大。
    """

    def clamp_tokens(words: BasicLyricLine) -> int:
        count = 0
        for word in words:
            if word.start is not None and word.start < 0:
                word.start = 0
                count += 1
            if word.end is not None and word.end < 0:
                word.end = 0
                count += 1
        return count

    clamped = 0
    alignable_lines: list[int] = []
    for index, line in enumerate(lyrics):
        line_clamped = 0
        if line.start < 0:
            line.start = 0
            line_clamped += 1
        if line.end is not None and line.end < 0:
            line.end = 0
            line_clamped += 1
        line_clamped += clamp_tokens(line.content)
        for reference in line.reference_lines:
            line_clamped += clamp_tokens(reference)
        if line_clamped:
            clamped += line_clamped
            if line.text and not metadata_filter(line.text):
                alignable_lines.append(index)
    return clamped, alignable_lines


def _apply_offset(
    lyrics: Lyrics,
    audio: NDArray,
    sample_rate: int,
    *,
    metadata_filter: MetadataFilter,
    offset_ms: float | None,
    energy_curve: NDArray | None = None,
) -> float:
    """估计并就地应用全局时间偏移，返回实际生效的偏移量（ms）。

    三处刻意的行为：

    1. **负时间戳逐个夹到 0**，而不是把整条时间轴回退。回退会让修正彻底失效：
       LRC 里几乎总有一条 ``[00:00.000]`` 的元数据行（如「作词 : xxx」），它不
       参与对齐，却会把任何负偏移完整抵消掉——连 ``--offset`` 手动指定的负值也
       会被静默丢弃（实测：含 0ms 行的文件上 ``--offset -200`` 的净偏移为 0）。
    2. **自动估计要过两道互相独立的校验**（见下），任一不通过就按不偏移处理：
       行区间对比度校验（:func:`~karakara.offset.validate_estimated_offset`）与
       「第一次持续人声」锚点校验（:func:`~karakara.offset.suggest_offset_from_onset`）。
    3. 手动 ``--offset`` 不受校验约束——用户的显式意图优先。

    为什么自动偏移这么保守：实测三首真实曲目，两条能量判据**各自都会错、且错在
    不同的歌上**（见 `validate_estimated_offset` 的表格），其中一首 LRC 与音频
    属于不同剪辑、需要 +17 秒的偏移，任何能量判据都救不回来。自动偏移宁可不动，
    也不要动错——动错的代价是把整首歌的切段推离人声。
    """
    applied_offset = offset_ms
    if applied_offset is None:
        applied_offset = estimate_offset(
            audio, lyrics, sample_rate, metadata_filter=metadata_filter
        )
        if applied_offset != 0 and energy_curve is not None:
            applied_offset = _validate_auto_offset(
                lyrics,
                energy_curve,
                applied_offset,
                metadata_filter=metadata_filter,
            )
    if applied_offset == 0:
        return 0.0

    delta = int(applied_offset)
    apply_delta(lyrics, delta)
    clamped, alignable_lines = _clamp_negative_timestamps(
        lyrics, metadata_filter=metadata_filter
    )
    logger.info(f"Applied global offset: {delta:+d}ms")
    if clamped:
        logger.info(
            f"Clamped {clamped} negative timestamp(s) to 0 "
            f"(the offset is kept instead of reverting the whole timeline)"
        )
    if alignable_lines:
        # 「偏移估计过大」的信号：参与对齐的行本不该被推出歌外。
        logger.warning(
            f"{len(alignable_lines)} line(s) that participate in alignment were "
            f"pushed before 0ms by offset {delta:+d}ms and clamped: "
            f"{alignable_lines[:10]}"
            + (" ..." if len(alignable_lines) > 10 else "")
            + " — the offset estimate is probably too large"
        )
    return float(delta)


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
    tokens = _merge_zero_length_tokens(tokens, line_start=line_start)
    return BasicLyricLine(tokens) if tokens else None


def _canonicalise(
    words: BasicLyricLine, *, line_start: int | None = None
) -> BasicLyricLine:
    """合并 ``words`` 里不贡献时长的 token，返回（必要时新的）``BasicLyricLine``。

    既用于我们生成的行，也用于**原样保留的行**：输入 LRC 本身可能就是逐字文件、
    自己就带着重复时间标签（实测 `samples/ashen.lrc` 里有 47 处），原样吐出去只会把
    同一份毛病传下去。
    """
    tokens = _merge_zero_length_tokens(list(words), line_start=line_start)
    return words if len(tokens) == len(words) else BasicLyricLine(tokens)


def _preserved_line(line: LyricLine) -> LyricLine:
    """原样保留一行——但仍然把不贡献时长的 token 合并掉，保证产物规范。"""
    copied = line.copy()
    copied.content = _canonicalise(copied.content, line_start=copied.start)
    copied.reference_lines = [
        _canonicalise(reference, line_start=copied.start)
        for reference in copied.reference_lines
    ]
    return copied


def _merge_zero_length_tokens(
    tokens: list[LyricToken], *, line_start: int | None = None
) -> list[LyricToken]:
    """把「不贡献任何时长」的 token 并进相邻 token，保证产物里不出现重复时间标签。

    为什么必须做：序列化器按 ``[start]文本[end]`` 写标签，并对"与前一词相接的起点"
    做省略优化；于是任何"写出来与前一个标签相同"的时间戳都会变成**重复时间标签**
    （实测某首歌 67 处，形如 ``[00:15.990]  [00:15.990]君``）。解析器会把这种重复
    标签当异常丢掉（自带 ``Unordered time tag dropped`` 告警），丢掉之后该 token 的
    文本会并进前一个 token——也就是说"并进前一个"正是产物本来的语义。

    这里只是把它显式化：**文本一个字符不变**（这些 token 本来就不贡献时长），
    但文件变规范、告警消失、round-trip 无损。

    两类 token 会写出重复标签，都要合并：

    1. ``start == end``——时长不足一帧的单元（本模块存在的主因）；
    2. ``start is None`` 且 ``end <= 上一个时间戳``——解析器对「已有逐字标签的行」
       会产出这种 token（例如 ``(None, 0)``），它的 ``end`` 标签同样会与行首/前一个
       标签重复。

    合并规则：并进前一个 token；它是行首 token 时并进后一个。孤立零长度 token
    （与前后都不相接，约 20%）并进前一个后会丢掉它那个**瞬时**标记——它本来就没有
    时长，因此不损失任何区间，只损失一个时间点。
    """

    def collapsible(index: int) -> bool:
        token = tokens[index]
        if token.start is not None and token.end is not None:
            return token.start == token.end
        if token.start is None and token.end is not None:
            reference = tokens[index - 1].end if index > 0 else line_start
            return reference is not None and token.end <= reference
        return False

    if not any(collapsible(index) for index in range(len(tokens))):
        return tokens

    merged: list[LyricToken] = []
    prefix_text: list[str] = []
    for index, token in enumerate(tokens):
        if collapsible(index):
            if merged:
                merged[-1].content += token.content
            else:
                # 行首就是这类 token：先攒着，等第一个正常 token 到来时并进去
                prefix_text.append(token.content)
            continue
        if prefix_text:
            token.content = "".join(prefix_text) + token.content
            prefix_text = []
        merged.append(token)

    if prefix_text:
        # 整行全是这类 token（实测：某行两个词都塌成 0 长度，且对齐器还多吐了几个
        # 文本里根本没有的词）。合成**一个** token：时间取行首与最后一个已知终点，
        # 行首标签与行末标签会分别兜住它们，因此不会产生重复标签。
        last_end = next(
            (item.end for item in reversed(tokens) if item.end is not None), None
        )
        start = line_start if line_start is not None else tokens[0].start
        merged.append(
            LyricToken(content="".join(prefix_text), start=start, end=last_end)
        )
    return merged


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
    refine_collapsed: bool = False,
) -> tuple[LyricLine | None, ZeroLengthStats]:
    """对单行执行对齐，不能可靠映射时返回 ``(None, stats)``。

    这里刻意捕获所有异常：单行失败只应降级为该行保留原样，不该中断整首歌。
    "整首歌都失败" 由 :func:`_check_alignment_health` 兜住并升级为硬错误。

    ``refine_collapsed`` 打开时，对齐器返回的零长度单元会被摊进其后的空隙
    （见 :func:`karakara.aligner.postprocess.refine_collapsed_words`）；无论开关与否，
    零长度单元的文本都会在 ``_build_aligned_content`` 里被并进相邻 token。
    """
    audio_piece = audio
    dumper.dump(f"05_line_{index}", audio_piece, sample_rate)
    try:
        words = aligner.align(audio_piece, text, sample_rate, language=language)
    except Exception as exc:  # noqa: BLE001 - 见 docstring：逐行降级是设计行为
        logger.error(f"Error occurred while aligning line {index}: {exc}")
        return None, ZeroLengthStats()

    stats = count_zero_length(words)
    if refine_collapsed and stats.zero_length:
        segment_ms = audio_piece.shape[-1] / sample_rate * 1000
        words, refined = refine_collapsed_words(words, total_ms=segment_ms)
        stats = ZeroLengthStats(
            units=stats.units, zero_length=stats.zero_length, refined=refined
        )
        if refined:
            logger.debug(
                f"line {index}: refined {refined}/{stats.zero_length} zero-length unit(s)"
            )

    content = _build_aligned_content(
        text, words, line_start=line.start, line_index=index
    )
    if content is None:
        logger.warning(f"Line {index}: no usable alignment result, preserving original")
        return None, stats
    end = content[-1].end if content and content[-1].end is not None else line.end
    if content and content[-1].end is not None:
        content[-1].end = None
    return (
        LyricLine(
            start=line.start,
            end=end,
            content=content,
            reference_lines=[
                _canonicalise(reference.copy(), line_start=line.start)
                for reference in line.reference_lines
            ],
        ),
        stats,
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
    refine_collapsed_words: bool = False,
) -> Lyrics:
    """根据音频和行级歌词生成词级逐字歌词。

    已有逐字时间标签的行由 ``existing_byword_policy`` 控制：``"realign"`` 会
    将所有 token 文本拼接后重新对齐；``"preserve"`` 则原样保留。

    ``target_lang`` 用于**跳过**语言不符的行（例如夹在日文歌词里的中文翻译行）；
    ``aligner_language`` 决定**送给对齐器**的语言，``"auto"`` 时按整首歌的
    行级多数票判定。

    ``separate_work_dir`` 是分离中间产物的落盘位置；``None`` 用系统临时目录。

    ``refine_collapsed_words`` 对应 ``--refine-collapsed-words``：把对齐器返回的
    零长度单元摊进其后的空隙（详见 :mod:`karakara.aligner.postprocess`）。默认关闭是
    因为那是**推断值**——模型只说了"这两个边界落在同一个 80ms 帧里"。无论开关与否，
    零长度单元的**文本**都不会丢，只会并进相邻 token。
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
    # 能量曲线只建一次：偏移校验与逐行人声活动度都用它。
    energy_curve = build_energy_curve(vocal_np, sample_rate)
    _apply_offset(
        working_lyrics,
        vocal_np,
        sample_rate,
        metadata_filter=metadata_filter,
        offset_ms=offset_ms,
        energy_curve=energy_curve,
    )

    result = Lyrics(metadata=working_lyrics.metadata)
    attempted = 0
    failed = 0
    stats = ZeroLengthStats()
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
            result.append(_preserved_line(line))
            continue

        start, end = _line_sample_range(working_lyrics, index, sample_rate)
        logger.info(f"aligning line {index}: sample_point[{start}, {end}] {text!r}")
        if end is not None and start > end:
            logger.warning(
                f"Line {index}: invalid audio segment: [{start}, {end}], preserving original"
            )
            result.append(_preserved_line(line))
            continue
        if (end is not None and end >= total_samples) or start >= total_samples:
            logger.warning(
                f"Line {index}: audio segment out of bounds: [{start}, {end}] "
                f"(total samples: {total_samples}), preserving original"
            )
            result.append(_preserved_line(line))
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
            result.append(_preserved_line(line))
            continue

        audio_piece = vocal_np[:, start:end] if end is not None else vocal_np[:, start:]
        aligned_line, line_stats = _align_line(
            line,
            text,
            index=index,
            audio=audio_piece,
            sample_rate=sample_rate,
            aligner=aligner,
            dumper=dumper,
            language=language,
            refine_collapsed=refine_collapsed_words,
        )
        stats = stats.merged_with(line_stats)
        attempted += 1
        if aligned_line is None:
            failed += 1
        result.append(
            aligned_line if aligned_line is not None else _preserved_line(line)
        )

    _check_alignment_health(attempted=attempted, failed=failed)
    _report_zero_length(stats)
    return result


def _report_zero_length(stats: ZeroLengthStats) -> None:
    """把零长度词占比当质量信号报出来。

    它是"这首歌是否已经顶到本工具的能力边界"的直接信号：对齐器的边界量化到 80ms
    （见 :mod:`karakara.aligner.postprocess`），单元时长不足一帧就会变成零长度，
    在播放器里无法单独高亮。实测逐首 10.7% / 15.3% / 24.6% 属正常范围，
    41.8% 那首则明显是模型吃力。
    """
    if not stats.units:
        return
    message = (
        f"aligner returned {stats.units} unit(s), {stats.zero_length} zero-length "
        f"({stats.ratio:.1%})"
    )
    if stats.refined:
        message += f", refined {stats.refined} into the following gap"
    if stats.ratio >= DEGENERATE_RATIO_WARN:
        logger.warning(
            f"{message} — above the {DEGENERATE_RATIO_WARN:.0%} warn threshold; "
            f"those units cannot be highlighted individually (try "
            f"--refine-collapsed-words, or a different aligner)"
        )
    else:
        logger.info(message)


def _check_alignment_health(*, attempted: int, failed: int) -> None:
    """把「整体对齐失效」从静默成功变成显式失败。

    逐行失败是设计好的降级（该行原样保留），但**每一行都失败**只可能是环境问题：
    对齐服务没起、``--aligner-url`` 指错、服务端与客户端的响应契约不一致……此时
    若照常返回，调用方会拿到一个内容与输入完全相同、却「生成成功」的文件。实测这个
    静默失效真实发生过（服务端返回数组而客户端读 ``["words"]``），因此这里直接报错。
    """
    if attempted == 0:
        return
    if failed == attempted:
        raise RuntimeError(
            f"全部 {attempted} 行的对齐都失败了，没有任何一行拿到词级时间戳。"
            f"这通常不是歌词问题，而是对齐环境问题：请检查对齐服务是否在 "
            f"--aligner-url 上运行、服务端版本是否与客户端契约一致（见上方日志里的"
            f"每行错误）。已放弃写盘，避免产出一个「没有词级时间」的假成功文件。"
        )
    if attempted >= 4 and failed * 2 >= attempted:
        logger.warning(
            f"{failed}/{attempted} 行对齐失败，产物中这些行会保留原样，请检查对齐服务"
        )
