"""分离后端 / 模型的人声质量 A/B 对比。

用法::

    # 本体：对比本地已有的 demucs 家族模型
    python scripts/compare_separators.py --songs-dir tmp/e2e \
        --configs demucs:htdemucs_6s demucs:htdemucs_ft demucs:mdx_extra_q \
        --out tmp/ab-report.json

    # 加上 audio-separator（需要它自己的环境，先装好）
    python scripts/compare_separators.py --songs-dir tmp/e2e \
        --configs demucs:htdemucs_6s audio-separator:UVR_MDXNET_KARA_2.onnx

为什么这样度量
--------------
我们没有「干净人声」的 ground truth，所以不去假装能算 SDR，而是量三个**与本项目
目标直接相关**的代理指标。本项目的目标是把每行歌词切出来送进强制对齐器，因此人声轨
里最有害的两件事是：(1) 混进了伴奏，(2) 把该有的人声削掉了。

1. ``leakage``（间隙泄漏）：歌词行区间**之外**的平均能量 ÷ 区间**之内**的平均能量。
   越低越好。外部区域按定义是伴奏/静音，那里能量越低说明伴奏混得越少。
2. ``contrast``（区间对比度）：区间内平均能量 − 区间外平均能量。越高越好。
3. ``coverage``（人声占位率）：区间内能量超过全曲 RMS 中位数的窗口占比。太低说明
   人声被削掉了（该有人的地方没能量）。

为了让指标只反映**分离质量**而不掺杂偏移估计的差异，所有对比都使用**同一个
固定偏移**（由基准配置估得），而不是各自的估计值。

这些是代理指标。最终裁决仍然是下游的对齐质量——真正的零假设检验需要人工试听，
或者用对齐器返回的零长度词比例做交叉验证（见 ``--report-degenerate`` 的设计留白）。
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import statistics
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from lemony_lrc_parser import Lyrics
from lemony_lrc_parser.offset import apply_delta

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from karakara.backends import SEPARATOR_BACKENDS
from karakara.offset import build_energy_curve, estimate_offset
from karakara.separator import SubprocessStemSeparator
from karakara.utils.io import load_audio_native
from karakara.utils.metadata import MetadataFilter

WINDOW_MS = 50.0

#: 判定「该处有人声」的固定能量下限。能量曲线已按全曲峰值归一化到 [0,1]，
#: 0.1 相当于距最响处约 -20dB。
AUDIBLE_FLOOR = 0.1

#: 后端 → worker 脚本。直接读主程序的登记表，避免两处各写一份而漂移。
_WORKERS = {name: backend.script for name, backend in SEPARATOR_BACKENDS.items()}


@dataclass(frozen=True)
class SeparatorConfig:
    """一个待对比的 (后端, 模型) 组合。"""

    backend: str
    model: str
    model_dir: str | None = None

    @property
    def label(self) -> str:
        return f"{self.backend}:{self.model}"

    @property
    def slug(self) -> str:
        return f"{self.backend}__{self.model}".replace(":", "_")

    def command(self, python: str | None = None) -> list[str]:
        """返回 worker 启动命令。

        ``python`` 非空时直接用该解释器运行脚本，跳过 ``uv run --script`` 的环境
        准备——已经有一个装好依赖的环境时这样更快，也便于离线/受控环境。
        """
        script = str(Path(_WORKERS[self.backend]).resolve())
        if python:
            return [python, script]
        return ["uv", "run", "--script", _WORKERS[self.backend]]


def parse_config(text: str) -> SeparatorConfig:
    """解析 ``后端:模型`` 或 ``后端:模型@模型目录``。"""
    backend, _, rest = text.partition(":")
    if not backend or not rest:
        raise argparse.ArgumentTypeError(f"配置格式应为 后端:模型，收到 {text!r}")
    if backend not in _WORKERS:
        raise argparse.ArgumentTypeError(
            f"未知后端 {backend!r}，可选: {sorted(_WORKERS)}"
        )
    model, _, model_dir = rest.partition("@")
    return SeparatorConfig(backend, model, model_dir or None)


@dataclass
class Metrics:
    """单个 (歌曲, 配置) 的度量结果。"""

    song: str
    config: str
    seconds: float
    samplerate: int
    frames: int
    leakage: float
    contrast: float
    coverage: float
    peak: float
    rms_dbfs: float
    #: 该配置的第一首歌包含 worker 冷启动（``uv run --script`` 准备环境 +
    #: ``import torch`` + 模型加载）。汇总时必须把它排除，否则耗时对比量的主要
    #: 是冷启动常数而不是分离本身。
    cold_start: bool = False


def vocal_regions(
    lyrics: Lyrics,
    total_duration_ms: float,
    offset_ms: float,
    *,
    metadata_filter: MetadataFilter,
) -> list[tuple[int, int]]:
    """返回歌词行区间 ``[行i起点, 行i+1起点)``，单位 ms。"""
    shifted = lyrics.copy()
    if offset_ms:
        apply_delta(shifted, int(offset_ms))
    starts = sorted(
        line.start
        for line in shifted
        if line.start is not None
        and 0 <= line.start < total_duration_ms
        and not metadata_filter(line.text)
    )
    return [(int(a), int(b)) for a, b in itertools.pairwise(starts) if int(b) > int(a)]


def measure(
    vocal: np.ndarray,
    sample_rate: int,
    spans: list[tuple[int, int]],
    *,
    window_ms: float = WINDOW_MS,
    audible_floor: float = AUDIBLE_FLOOR,
) -> tuple[float, float, float, float, float]:
    """计算 (leakage, contrast, coverage, peak, rms_dbfs)。

    ``coverage`` 是行区间内「可听见」窗口的占比，阈值取全曲峰值归一化能量上的固定
    下限（约 -20dB）。用固定下限而不是区间内中位数——后者会让该指标恒等于 0.5，
    变成废指标。
    """
    energy = build_energy_curve(vocal, sample_rate, window_ms)
    n = len(energy)
    inside = np.zeros(n, dtype=bool)
    for begin_ms, end_ms in spans:
        lo = max(0, min(int(begin_ms / window_ms), n - 1))
        hi = max(lo, min(int(end_ms / window_ms), n - 1))
        inside[lo : hi + 1] = True

    if inside.sum() < 2 or (~inside).sum() < 2:
        raise ValueError("行区间覆盖了整个音频，指标无信息量")

    in_mean = float(energy[inside].mean())
    out_mean = float(energy[~inside].mean())
    leakage = out_mean / in_mean if in_mean > 1e-9 else float("inf")
    contrast = in_mean - out_mean

    coverage = float((energy[inside] > audible_floor).mean())

    peak = float(np.abs(vocal).max())
    rms = float(np.sqrt(np.mean(vocal**2)))
    rms_dbfs = 20 * np.log10(rms) if rms > 1e-12 else float("-inf")
    return leakage, contrast, coverage, peak, rms_dbfs


def discover_songs(songs_dir: Path) -> list[tuple[Path, Path]]:
    """找出同名的 lrc/音频对。"""
    pairs: list[tuple[Path, Path]] = []
    for lrc in sorted(songs_dir.glob("*.lrc")):
        if lrc.stem.endswith(".kara"):
            continue
        for suffix in (".mp3", ".flac", ".wav", ".m4a"):
            audio = lrc.with_suffix(suffix)
            if audio.is_file():
                pairs.append((lrc, audio))
                break
    return pairs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="分离后端 / 模型的人声质量 A/B 对比")
    parser.add_argument(
        "--songs-dir", type=Path, required=True, help="含 lrc+音频对的目录"
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        required=True,
        help="形如 demucs:htdemucs_6s 或 audio-separator:模型文件名.onnx",
    )
    parser.add_argument(
        "--worker-python",
        default=os.environ.get("KARAKARA_WORKER_PYTHON"),
        help=(
            "用指定解释器直接运行 worker 脚本，跳过 uv run --script。"
            "注意：它作用于**所有**配置，所以该解释器必须同时具备所有后端的依赖；"
            "只对比单一后端时才用得上"
        ),
    )
    parser.add_argument("--out", type=Path, default=None, help="把结果写入该 JSON")
    parser.add_argument(
        "--limit", type=int, default=None, help="只用前 N 首歌（慢后端时很有用）"
    )
    parser.add_argument(
        "--work-dir", type=Path, default=None, help="分离产物目录（缺省用系统临时目录）"
    )
    parser.add_argument(
        "--metadata-filter", type=Path, default=Path("metadata_filter.toml")
    )
    parser.add_argument(
        "--reference-config",
        default=None,
        help="用哪个配置估计固定偏移（缺省用第一个）；所有配置共用它",
    )
    args = parser.parse_args(argv)

    configs = [parse_config(text) for text in args.configs]
    reference = (
        parse_config(args.reference_config) if args.reference_config else configs[0]
    )
    metadata_filter = MetadataFilter.from_file(args.metadata_filter)
    songs = discover_songs(args.songs_dir)
    if not songs:
        print(f"在 {args.songs_dir} 下没有找到 lrc/音频对", file=sys.stderr)
        return 1
    if args.limit is not None:
        songs = songs[: args.limit]

    print(
        f"曲目 {len(songs)} 首，配置 {len(configs)} 个，"
        f"固定偏移来自 {reference.label}\n"
    )

    results: list[Metrics] = []
    errors: list[str] = []

    work_root = (
        Path(tempfile.mkdtemp(prefix="karakara-ab-", dir=str(args.work_dir)))
        if args.work_dir is not None
        else Path(tempfile.mkdtemp(prefix="karakara-ab-"))
    )
    print(f"工作目录: {work_root}\n")

    # ---------- 每个配置只起一个 worker，跨歌曲复用 ----------
    # 本文档开头就写了「模型只加载一次」是这套架构的要点；对比脚本自己更不能
    # 每首歌 popen 一次——那样每首歌的 seconds 里都混着一次 ``uv run --script``
    # 的环境解析 + import torch + 模型加载（实测冷启动约 2.6s），耗时对比量的
    # 主要就是这个常数。这里显式复用，并把每个配置的首曲标成 cold_start。
    to_separate: list[SeparatorConfig] = list(configs)
    if reference.label not in {config.label for config in to_separate}:
        to_separate.append(reference)
    separators: dict[str, SubprocessStemSeparator] = {
        config.label: SubprocessStemSeparator(
            command=config.command(args.worker_python),
            model=config.model,
            model_dir=config.model_dir,
        )
        for config in to_separate
    }
    cold_start = dict.fromkeys(separators, True)

    try:
        for lrc_path, audio_path in songs:
            lyrics = Lyrics.loads(lrc_path.read_text(encoding="utf-8"))

            # ---------- 先用所有配置分离（含基准），再用基准确定固定偏移 ----------
            vocals: dict[str, tuple[Path, float, bool]] = {}
            for config in to_separate:
                try:
                    vocal_path, seconds = separate_with(
                        separators[config.label],
                        audio_path,
                        work_root / config.slug / audio_path.stem,
                    )
                    vocals[config.label] = (
                        vocal_path,
                        seconds,
                        cold_start[config.label],
                    )
                except Exception as exc:  # noqa: BLE001 - 单个配置失败不该中断整体对比
                    errors.append(f"{audio_path.name} / {config.label}: {exc}")
                    print(f"    {config.label:<34} SEPARATION FAILED: {exc}")

            if reference.label not in vocals:
                print(f"=== {audio_path.name}\n    基准配置分离失败，跳过该曲")
                continue

            ref_vocal_path, ref_seconds, ref_cold = vocals[reference.label]
            ref_vocal, sample_rate = load_audio_native(ref_vocal_path)
            total_ms = ref_vocal.shape[-1] / sample_rate * 1000
            offset_ms = estimate_offset(
                ref_vocal,
                lyrics,
                sample_rate,
                metadata_filter=metadata_filter,
                window_ms=WINDOW_MS,
            )
            spans = vocal_regions(
                lyrics, total_ms, offset_ms, metadata_filter=metadata_filter
            )
            print(
                f"=== {audio_path.name}\n"
                f"    基准 {reference.label}: {ref_seconds:.1f}s"
                f"{'（含冷启动）' if ref_cold else ''}, "
                f"采样率 {sample_rate}, 固定偏移 {offset_ms:+.0f}ms, "
                f"行区间 {len(spans)} 段"
            )

            for config in configs:
                label = config.label
                if label not in vocals:
                    continue
                try:
                    vocal_path, seconds, cold = vocals[label]
                    vocal, rate = load_audio_native(vocal_path)
                    leakage, contrast, coverage, peak, rms_dbfs = measure(
                        vocal, rate, spans
                    )
                    results.append(
                        Metrics(
                            song=audio_path.name,
                            config=label,
                            seconds=round(seconds, 3),
                            samplerate=rate,
                            frames=int(vocal.shape[-1]),
                            leakage=round(leakage, 4),
                            contrast=round(contrast, 4),
                            coverage=round(coverage, 4),
                            peak=round(peak, 4),
                            rms_dbfs=round(rms_dbfs, 2),
                            cold_start=cold,
                        )
                    )
                    print(
                        f"    {label:<34} {seconds:6.1f}s"
                        f"{' (cold)' if cold else '       '}  "
                        f"泄漏 {leakage:.3f}  对比度 {contrast:.4f}  "
                        f"占位 {coverage:.2f}"
                    )
                except Exception as exc:  # noqa: BLE001 - 单个配置失败不该中断整体对比
                    errors.append(f"{audio_path.name} / {label}: {exc}")
                    print(f"    {label:<34} FAILED: {exc}")
            # 每个配置的首曲已经跑过，后续不再算冷启动
            for config in to_separate:
                cold_start[config.label] = False
    finally:
        import shutil

        for separator in separators.values():
            separator.close()
        shutil.rmtree(work_root, ignore_errors=True)

    print_report(results, errors)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {
                    "reference_config": reference.label,
                    "results": [asdict(item) for item in results],
                    "errors": errors,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\n已写入 {args.out}")
    return 1 if errors else 0


def separate_with(
    separator: SubprocessStemSeparator, audio: Path, dest: Path
) -> tuple[Path, float]:
    """用**已构造好的**分离器分离一首歌，返回 (人声轨路径, 耗时秒)。

    ``separator`` 由调用方跨歌曲复用（见 ``main()``）：worker 进程常驻、模型只
    加载一次，所以耗时里不含重复的冷启动开销。
    """
    started = time.perf_counter()
    stems = separator.separate(audio, dest, stems=["vocals"])
    return stems["vocals"], time.perf_counter() - started


def print_report(results: list[Metrics], errors: list[str]) -> None:
    """按配置汇总并打印。"""
    if not results:
        print("\n没有任何成功结果")
        return

    by_config: dict[str, list[Metrics]] = {}
    for item in results:
        by_config.setdefault(item.config, []).append(item)

    def timing_rows(items: list[Metrics]) -> list[Metrics]:
        """算耗时时排除冷启动那一首（有其它样本可用时）。"""
        warm = [item for item in items if not item.cold_start]
        return warm or items

    print("\n" + "=" * 92)
    print("按配置汇总（多首取中位数；泄漏越低越好，对比度/占位越高越好）")
    print("耗时列已排除每个配置的首曲冷启动（worker 环境准备 + import torch + 载模型）")
    print("=" * 92)
    print(
        f"{'config':<34}{'泄漏':>9}{'对比度':>10}{'占位':>8}"
        f"{'峰值':>9}{'RMS dBFS':>11}{'耗时s':>8}"
    )
    print("-" * 92)
    for label, items in sorted(
        by_config.items(), key=lambda kv: statistics.median([i.leakage for i in kv[1]])
    ):
        print(
            f"{label:<34}"
            f"{statistics.median([i.leakage for i in items]):>9.3f}"
            f"{statistics.median([i.contrast for i in items]):>10.4f}"
            f"{statistics.median([i.coverage for i in items]):>8.2f}"
            f"{statistics.median([i.peak for i in items]):>9.3f}"
            f"{statistics.median([i.rms_dbfs for i in items]):>11.2f}"
            f"{statistics.median([i.seconds for i in timing_rows(items)]):>8.1f}"
        )
    print("-" * 92)
    print(f"曲目数 {len({i.song for i in results})}，结果行 {len(results)}")
    single_song_configs = [
        label for label, items in by_config.items() if len(timing_rows(items)) < 2
    ]
    if single_song_configs:
        print(
            "注意：以下配置只有 1 首可用样本，其耗时仍含冷启动，"
            f"不具可比性：{', '.join(single_song_configs)}"
        )
    if errors:
        print(f"\n失败 {len(errors)} 项：")
        for message in errors:
            print(f"  - {message}")


if __name__ == "__main__":
    raise SystemExit(main())
