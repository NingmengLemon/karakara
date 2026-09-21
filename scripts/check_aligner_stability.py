"""对齐后端的扰动稳定性检查（V4）。

实测结论见 `docs/records/2026-09-18-aligner-choice-and-stability.md`；
判据设计与候选评估方法见 `docs/archive/aligner-backends-research.md`。

用法::

    # 默认后端（hfa）；缺分离人声时加 --separate 跑一次并缓存
    python scripts/check_aligner_stability.py --lrc song.lrc --audio song.flac --separate

    # 与另一个后端对照
    python scripts/check_aligner_stability.py --lrc song.lrc --audio song.flac \
        --aligner-backend qwen3 --tag _qwen3

    # 阈值可调（判据是「行内边界漂移中位数 < 一个帧移」）
    python scripts/check_aligner_stability.py --lrc ... --audio ... --frame-ms 80

为什么这样度量
--------------
我们没有人工标注，所以不去假装能算 BER/IOU（那是 V5 的事），而是量一个**零标注成本**
的性质：**扰动不该改变结果**。三次扰动，每次只改一个变量：

| 条件 | 改什么 | 想回答的问题 |
|---|---|---|
| ``pad+`` / ``pad-`` | 片段两端各加 / 各减 100ms | 对齐器认不认片段边界？生产路径的紧切片段够不够？ |
| ``mix`` | 同一时间区间，音频换成原始混音 | 分离质量对结果的影响有多大？ |
| ``drop1`` / ``add1`` | 文本删掉 / 重复一个字符 | 歌词错字会让整行糊掉，还是只毁错处之后？ |

三条口径，先说清再看数字：

1. **``pad`` 条件要先扣掉已知平移**。+100ms 的片段里，同一条边界理应出现在
   ``base+100ms``；不扣掉的话量到的是"我加的 padding"，而不是"对齐器的敏感度"。
2. **只比行内边界**（首尾各去掉一条）。首条边界是"人声从哪开始"、末条边界贴着片段尾，
   它们的差属于**锚定误差**，单独报。注意这个指标对**输出是整段连续划分**的后端
   （HubertFA 就是）恒等于 padding，不具判别力——报告里会提示。
3. **单元数变了（``drop1``/``add1``）就不做逐边界比较**，改报"是否还成立"，
   并用**单元文本的最长公共前缀**把漂移拆成前缀/后缀两段。

原始边界（每行每条件的全部边界与单元文本）落盘到 ``<out-dir>/<tag>.json``，
判定可重算；本脚本只读音频与歌词，不写任何产物。
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from karakara import backends
from karakara.aligner import HttpAligner
from karakara.core import _line_sample_range
from karakara.separator import SubprocessStemSeparator
from karakara.typ import NpAudioData
from karakara.utils.io import load_audio, save_audio
from karakara.utils.lang import detect_dominant_lang
from karakara.utils.lrc import load_lyrics
from karakara.utils.metadata import MetadataFilter

SAMPLE_RATE = 44100
DEFAULT_PAD_MS = 100
#: 判据：行内边界漂移的中位数小于一个帧移。HubertFA 是 10ms，Qwen3 是 80ms。
DEFAULT_FRAME_MS = 10.0
DEFAULT_MAX_LINES = 20
DEFAULT_OUT_DIR = Path("tmp/perturbation")
DEFAULT_SEPARATOR_MODEL = "UVR_Demucs_Model_1"
DEFAULT_METADATA_FILTER = (
    Path(__file__).resolve().parent.parent / "metadata_filter.toml"
)


# --------------------------------------------------------------------------
# 度量（纯函数，便于用已知答案自检）
# --------------------------------------------------------------------------


def drop_char(text: str) -> str | None:
    """删掉中间那个非空白字符；字符太少时返回 ``None``。"""
    indexes = [i for i, ch in enumerate(text) if not ch.isspace()]
    if len(indexes) < 2:
        return None
    middle = indexes[len(indexes) // 2]
    return text[:middle] + text[middle + 1 :]


def dup_char(text: str) -> str | None:
    """把中间那个非空白字符重复一次；没有内容字符时返回 ``None``。"""
    indexes = [i for i, ch in enumerate(text) if not ch.isspace()]
    if not indexes:
        return None
    middle = indexes[len(indexes) // 2]
    return text[:middle] + text[middle] + text[middle:]


def boundaries_of(units: list[tuple[str, int, int]]) -> list[int]:
    """单元序列 → 边界序列（n 个单元给 n+1 条边界）。"""
    if not units:
        return []
    return [unit[1] for unit in units] + [units[-1][2]]


def coverage(boundaries: list[int], segment_ms: float) -> float:
    """单元时长之和 ÷ 片段时长（与文档 §9 的"覆盖率"同口径）。"""
    if len(boundaries) < 2 or segment_ms <= 0:
        return 0.0
    spans = sum(boundaries[i + 1] - boundaries[i] for i in range(len(boundaries) - 1))
    return round(spans / segment_ms, 4)


def interior_drift(base: list[int], other: list[int], shift_ms: int) -> list[float]:
    """``other`` 扣掉已知平移后，与 ``base`` 的**行内**边界差（ms）。

    行内 = 首尾各去掉一条（见模块 docstring 第 2 条）。单元数不同或边界太少时返回空表。
    """
    if len(base) != len(other) or len(base) < 3:
        return []
    return [abs((other[i] - shift_ms) - base[i]) for i in range(1, len(base) - 1)]


def local_drift(
    base: list[int],
    base_texts: list[str],
    other: list[int],
    other_texts: list[str],
    delta: int,
) -> tuple[list[float], list[float]]:
    """把文本扰动造成的漂移拆成**前缀**与**后缀**两段（ms）。

    ``delta`` 是单元数变化（``drop1`` 为 -1、``add1`` 为 +1）。切分点是**单元文本的最长
    公共前缀**，所以对"逐字"（中文）与"按词"（日文）两种粒度都成立。

    意义：若错一个字只让局部边界移动，前缀漂移应接近 0（对齐器重新同步了）；若前缀
    也跟着漂，说明一个错字把整行都带偏了。
    """
    limit = min(len(base_texts), len(other_texts))
    pivot = 0
    while pivot < limit and base_texts[pivot] == other_texts[pivot]:
        pivot += 1
    prefix: list[float] = [
        float(abs(other[i] - base[i])) for i in range(1, min(pivot, len(other) - 1))
    ]
    suffix: list[float] = []
    for j in range(pivot, len(other_texts)):
        source = j - delta
        if 0 < source < len(base_texts) and j < len(other) - 1:
            suffix.append(abs(other[j] - base[source]))
    return prefix, suffix


def summarize(values: list[float]) -> dict[str, float] | None:
    """``中位/p90/最大``；空表返回 ``None``。"""
    if not values:
        return None
    ordered = sorted(values)
    return {
        "n": float(len(values)),
        "median": statistics.median(values),
        "p90": ordered[min(len(ordered) - 1, int(len(ordered) * 0.9))],
        "max": ordered[-1],
    }


# --------------------------------------------------------------------------
# 采集
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ConditionResult:
    """一行歌词在某一个扰动条件下的对齐结果。"""

    text: str
    shift_ms: int
    boundaries_ms: list[int]
    unit_texts: list[str]
    coverage: float

    @property
    def units(self) -> int:
        return len(self.unit_texts)


@dataclass(frozen=True)
class LineRecord:
    """一行歌词的全部条件结果。"""

    index: int
    text: str
    segment_ms: tuple[int, int]
    language: str
    conditions: dict[str, ConditionResult]

    def condition(self, name: str) -> ConditionResult | None:
        return self.conditions.get(name)


def align_units(
    aligner: HttpAligner, clip: NpAudioData, text: str, language: str
) -> list[tuple[str, int, int]]:
    """送一段音频，返回 ``[(单元文本, 起点ms, 终点ms), ...]``。"""
    words = aligner.align(clip, text, SAMPLE_RATE, language=language)
    return [
        (word.word, word.position[0], word.position[1])
        for word in words
        if word.position is not None
    ]


def build_conditions(
    vocal: NpAudioData,
    mix: NpAudioData,
    text: str,
    start: int,
    end: int,
    total: int,
    pad_ms: int,
) -> dict[str, tuple[NpAudioData, str, int]]:
    """构造一行上的全部扰动条件：``{条件名: (音频, 文本, 已知平移ms)}``。"""
    pad = pad_ms * SAMPLE_RATE // 1000
    padded_start = max(0, start - pad)
    padded_end = min(total, end + pad)
    conditions: dict[str, tuple[NpAudioData, str, int]] = {
        "base": (vocal[:, start:end], text, 0),
        # 平移按**毫秒**算：切点取整会带来 <1ms 的余数，不能直接用样本数换算。
        "pad+": (
            vocal[:, padded_start:padded_end],
            text,
            round((start - padded_start) / SAMPLE_RATE * 1000),
        ),
        "pad-": (vocal[:, start + pad : end - pad], text, -pad_ms),
        "mix": (mix[:, start:end], text, 0),
    }
    dropped = drop_char(text)
    if dropped:
        conditions["drop1"] = (vocal[:, start:end], dropped, 0)
    added = dup_char(text)
    if added:
        conditions["add1"] = (vocal[:, start:end], added, 0)
    return conditions


def ensure_vocals(audio: Path, vocals: Path, *, separate: bool, model: str) -> Path:
    """拿到分离人声：已有缓存就直接用，否则（``--separate``）跑一次并缓存。"""
    if vocals.is_file():
        print(f"分离人声缓存命中: {vocals}")
        return vocals
    if not separate:
        raise SystemExit(
            f"缺少分离人声 {vocals}；加 --separate 跑一次分离（结果会缓存）"
        )
    print(f"跑分离: {audio.name}")
    vocals.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="stability-sep-") as work:
        separator = SubprocessStemSeparator(model=model)
        try:
            stems = separator.separate(audio, work, stems=["vocals"])
            # 注意 load_audio 只返回数组；返回 (数组, 采样率) 的是 load_audio_native。
            # 解包写错的话，立体声会被沿 axis=0 拆成两个声道而不报错。
            vocal = load_audio(stems["vocals"], sample_rate=SAMPLE_RATE)
        finally:
            separator.close()
    save_audio(vocals, vocal, SAMPLE_RATE)
    print(f"已缓存: {vocals}")
    return vocals


def collect(
    *,
    lrc: Path,
    audio: Path,
    vocals: Path,
    url: str,
    language_arg: str,
    metadata_filter_path: Path,
    max_lines: int,
    pad_ms: int,
    separate: bool,
    separator_model: str,
) -> tuple[list[LineRecord], str]:
    """跑完所有行与所有条件，返回 ``(记录, 实际使用的语言)``。"""
    vocals_path = ensure_vocals(audio, vocals, separate=separate, model=separator_model)
    vocal = load_audio(vocals_path, sample_rate=SAMPLE_RATE)
    mix = load_audio(audio, sample_rate=SAMPLE_RATE)
    print(
        f"人声 {vocal.shape[-1] / SAMPLE_RATE:.3f}s / 混音 {mix.shape[-1] / SAMPLE_RATE:.3f}s"
        f"（差 {abs(vocal.shape[-1] - mix.shape[-1]) / SAMPLE_RATE * 1000:.0f}ms）"
    )
    total = min(vocal.shape[-1], mix.shape[-1])

    lyrics = load_lyrics(lrc)
    metadata_filter = MetadataFilter.from_file(metadata_filter_path)
    texts = [
        line.text for line in lyrics if line.text and not metadata_filter(line.text)
    ]
    language: str
    if language_arg == "auto":
        language = detect_dominant_lang(texts) or "zh"
        print(f"语言判定: {language}（{len(texts)} 行参与判定）")
    else:
        language = language_arg

    aligner = HttpAligner(base_url=url, timeout=None)
    records: list[LineRecord] = []
    try:
        for index in range(len(lyrics)):
            if len(records) >= max_lines:
                break
            line = lyrics[index]
            text = line.text
            if not text or metadata_filter(text):
                continue
            start, end = _line_sample_range(lyrics, index, SAMPLE_RATE)
            if end is None or end > total or start >= total:
                continue
            if end - start < 3 * pad_ms * SAMPLE_RATE // 1000:
                continue

            conditions: dict[str, ConditionResult] = {}
            for name, (clip, cond_text, shift) in build_conditions(
                vocal, mix, text, start, end, total, pad_ms
            ).items():
                units = align_units(aligner, clip, cond_text, language)
                boundary_list = boundaries_of(units)
                conditions[name] = ConditionResult(
                    text=cond_text,
                    shift_ms=shift,
                    boundaries_ms=boundary_list,
                    unit_texts=[unit[0] for unit in units],
                    coverage=coverage(
                        boundary_list, (end - start) / SAMPLE_RATE * 1000
                    ),
                )
            records.append(
                LineRecord(
                    index=index,
                    text=text,
                    segment_ms=(
                        round(start / SAMPLE_RATE * 1000),
                        round(end / SAMPLE_RATE * 1000),
                    ),
                    language=language,
                    conditions=conditions,
                )
            )
            counts = {name: item.units for name, item in conditions.items()}
            print(
                f"[{len(records):>2}] line {index:>3} {text[:22]:<24} 单元数 {counts}"
            )
    finally:
        aligner.close()
    return records, language


# --------------------------------------------------------------------------
# 报告
# --------------------------------------------------------------------------


def _drift_rows(records: list[LineRecord], name: str) -> tuple[int, list[float]]:
    """``(可用行数, 全部行内漂移)``。"""
    drifts: list[float] = []
    rows = 0
    for record in records:
        base = record.condition("base")
        other = record.condition(name)
        if base is None or other is None:
            continue
        drift = interior_drift(base.boundaries_ms, other.boundaries_ms, other.shift_ms)
        if drift:
            rows += 1
            drifts.extend(drift)
    return rows, drifts


def report(records: list[LineRecord], out_path: Path, frame_ms: float) -> None:
    print("\n" + "=" * 78)
    print(f"行内边界漂移（ms）——中位数 < 1 个帧移({frame_ms:.0f}ms) 视为稳定")
    print("=" * 78)
    print(f"{'条件':<7}{'行数':>5}{'n':>6}{'中位':>9}{'p90':>9}{'最大':>9}   判读")
    for name in ("pad+", "pad-", "mix"):
        rows, drifts = _drift_rows(records, name)
        summary = summarize(drifts)
        if summary is None:
            print(f"{name:<7}{rows:>5}{'-':>6}{'-':>9}{'-':>9}{'-':>9}   无可用行")
            continue
        verdict = "稳定" if summary["median"] < frame_ms else "不稳定"
        print(
            f"{name:<7}{rows:>5}{int(summary['n']):>6}{summary['median']:>9.1f}"
            f"{summary['p90']:>9.1f}{summary['max']:>9.1f}   {verdict}"
        )

    print("\n逐行：行内漂移（找离群行）")
    for record in records:
        base = record.condition("base")
        if base is None:
            continue
        parts: list[str] = []
        for name in ("pad+", "pad-", "mix"):
            other = record.condition(name)
            if other is None:
                continue
            drift = interior_drift(
                base.boundaries_ms, other.boundaries_ms, other.shift_ms
            )
            if drift:
                worst = max(range(len(drift)), key=lambda i: drift[i])
                parts.append(
                    f"{name} 中位 {statistics.median(drift):>5.1f}"
                    f" 最大 {drift[worst]:>6.1f}@{worst + 1}"
                )
        print(f"  line {record.index:>3} {record.text[:18]:<20} " + " | ".join(parts))

    print("\n锚定误差（首条边界，ms）")
    print("  注意：输出是**对整段的连续划分**的后端（HubertFA 就是），首条单元边界在")
    print(
        "  结构上等于片段起点，于是这个指标对它恒等于「我加的 padding」，不具判别力。"
    )
    print("  真正有信息量的是上面的行内漂移。")
    for name in ("pad+", "pad-", "mix"):
        errors: list[float] = []
        for record in records:
            base = record.condition("base")
            other = record.condition(name)
            if base is None or other is None:
                continue
            if base.boundaries_ms and other.boundaries_ms:
                errors.append(
                    abs(
                        (other.boundaries_ms[0] - other.shift_ms)
                        - base.boundaries_ms[0]
                    )
                )
        summary = summarize(errors)
        if summary:
            print(
                f"  {name:<6} 中位 {summary['median']:>7.1f}  p90 {summary['p90']:>7.1f}"
                f"  最大 {summary['max']:>7.1f}"
            )

    print("\n文本扰动：错一个字，误差是局部的还是全局的？")
    print("  （前缀 = 扰动点之前的边界漂移，后缀 = 之后的；用单元文本的公共前缀切分）")
    print("  （「单元数符合预期」只对**逐字**粒度有意义：日文是按词聚合的，删一个字")
    print("    可能让某个词整块消失，条数变化不固定，所以那一列在日文上偏低属正常）")
    for name, delta in (("drop1", -1), ("add1", 1)):
        prefix_all: list[float] = []
        suffix_all: list[float] = []
        survived = monotonic = covered = 0
        for record in records:
            base = record.condition("base")
            other = record.condition(name)
            if base is None or other is None:
                continue
            if other.boundaries_ms and all(
                other.boundaries_ms[i] <= other.boundaries_ms[i + 1]
                for i in range(len(other.boundaries_ms) - 1)
            ):
                monotonic += 1
            if other.coverage > 0.5:
                covered += 1
            if other.units == base.units + delta:
                survived += 1
            prefix, suffix = local_drift(
                base.boundaries_ms,
                base.unit_texts,
                other.boundaries_ms,
                other.unit_texts,
                delta,
            )
            prefix_all.extend(prefix)
            suffix_all.extend(suffix)
        print(
            f"  {name:<6} 单元数符合预期 {survived}/{len(records)}"
            f"  单调 {monotonic}/{len(records)}  覆盖率>50% {covered}/{len(records)}"
        )
        for label, values in (("前缀", prefix_all), ("后缀", suffix_all)):
            summary = summarize(values)
            if summary:
                print(
                    f"         {label} 中位 {summary['median']:>7.1f}"
                    f"  p90 {summary['p90']:>7.1f}  最大 {summary['max']:>7.1f}"
                    f"  (n={int(summary['n'])})"
                )

    print(f"\n原始结果: {out_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="对齐后端的扰动稳定性检查（±pad / 混音 vs 分离 / 删字）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "指标口径与实测结论见 docs/records/2026-09-18-aligner-choice-and-stability.md。"
        ),
    )
    parser.add_argument("--lrc", type=Path, required=True, help="行级 LRC")
    parser.add_argument("--audio", type=Path, required=True, help="原始音频（混音）")
    parser.add_argument(
        "--vocals",
        type=Path,
        default=None,
        help="分离人声路径（缺省为 <out-dir>/<音频名>_vocals.wav；不存在时需 --separate）",
    )
    parser.add_argument(
        "--separate", action="store_true", help="缺分离人声时跑一次并缓存"
    )
    parser.add_argument(
        "--separator-model",
        default=DEFAULT_SEPARATOR_MODEL,
        help=f"分离模型（默认: {DEFAULT_SEPARATOR_MODEL}）",
    )
    parser.add_argument(
        "--aligner-backend",
        choices=tuple(backends.ALIGNER_BACKENDS),
        default=backends.DEFAULT_ALIGNER_BACKEND,
        help="用哪个后端的默认地址与语言能力（默认: %(default)s）",
    )
    parser.add_argument("--url", default=None, help="对齐服务地址；缺省按后端取")
    parser.add_argument(
        "--language",
        choices=("auto", "zh", "ja", "en", "yue", "ko"),
        default="auto",
        help="送给对齐器的语言（默认: auto=按整首歌判定）",
    )
    parser.add_argument(
        "--pad-ms", type=int, default=DEFAULT_PAD_MS, help="pad 扰动量（默认: 100）"
    )
    parser.add_argument(
        "--frame-ms",
        type=float,
        default=DEFAULT_FRAME_MS,
        help="判据用的帧移（默认: 10=HubertFA；qwen3 是 80）",
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=DEFAULT_MAX_LINES,
        help=f"最多测几行（默认: {DEFAULT_MAX_LINES}）",
    )
    parser.add_argument(
        "--metadata-filter",
        type=Path,
        default=DEFAULT_METADATA_FILTER,
        help="元数据行过滤配置（随本脚本定位）",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="JSON 与日志目录"
    )
    parser.add_argument("--tag", default="", help="输出文件名后缀（换后端对照时用）")
    args = parser.parse_args(argv)

    try:
        backends.ensure_aligner_language(
            args.aligner_backend, None if args.language == "auto" else args.language
        )
    except backends.UnsupportedAlignerLanguage as exc:
        parser.error(str(exc))

    vocals: Path = args.vocals or (args.out_dir / f"{args.audio.stem}_vocals.wav")
    records, _language = collect(
        lrc=args.lrc,
        audio=args.audio,
        vocals=vocals,
        url=backends.resolve_aligner_url(args.aligner_backend, args.url),
        language_arg=args.language,
        metadata_filter_path=args.metadata_filter,
        max_lines=args.max_lines,
        pad_ms=args.pad_ms,
        separate=args.separate,
        separator_model=args.separator_model,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"{args.audio.stem}{args.tag}.json"
    out_path.write_text(
        json.dumps(
            {
                "lrc": str(args.lrc),
                "audio": str(args.audio),
                "backend": args.aligner_backend,
                "url": backends.resolve_aligner_url(args.aligner_backend, args.url),
                "pad_ms": args.pad_ms,
                "frame_ms": args.frame_ms,
                "records": [
                    {
                        **asdict(record),
                        "conditions": {
                            name: asdict(condition)
                            for name, condition in record.conditions.items()
                        },
                    }
                    for record in records
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    report(records, out_path, args.frame_ms)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
