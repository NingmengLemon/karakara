from __future__ import annotations

import argparse
import gc
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Literal

from lemony_lrc_parser import Lyrics, SerializationOptions

from karakara.aligner import Qwen3ForcedAligner
from karakara.core import ExistingBywordPolicy, gen_kara
from karakara.logging import setup_logging
from karakara.preprocess import AudioPreprocessConfig
from karakara.separator import SubprocessStemSeparator
from karakara.utils.metadata import MetadataFilter

_AUDIO_SUFFIXES = frozenset({".wav", ".mp3", ".flac", ".m4a"})

#: 后端 → worker 脚本。两者使用同一套行协议，所以主程序只认命令、不认后端。
_SEPARATOR_WORKERS = {
    "demucs": "scripts/separator_worker.py",
    "audio-separator": "scripts/separator_worker_audio_separator.py",
}

#: 仓库自带的元数据过滤配置。刻意相对**本文件**定位而不是 CWD，
#: 否则从别的目录运行就会去找那个目录下的同名文件（然后 FileNotFoundError）。
_DEFAULT_METADATA_FILTER = Path(__file__).resolve().parent / "metadata_filter.toml"


@dataclass(frozen=True)
class BatchJob:
    """一组同名歌词与音频的批处理任务。"""

    lyrics_path: Path
    audio_path: Path
    output_path: Path


def ask_for_input_file(type_: Literal["lyrics", "audio"]) -> str:
    """使用文件对话框请求输入文件。"""
    import tkinter as tk
    from tkinter import filedialog as fd

    root = tk.Tk()
    root.withdraw()
    if type_ == "lyrics":
        filetypes = [("LRC files", "*.lrc"), ("All files", "*.*")]
    else:
        filetypes = [("Audio files", "*.wav *.mp3 *.flac *.m4a"), ("All files", "*.*")]
    file_path = fd.askopenfilename(
        parent=root,
        title=f"Select an {type_} file",
        filetypes=filetypes,
    )
    root.destroy()
    return file_path


def ask_for_output_path(default_file: PathLike[str]) -> str:
    """使用文件对话框请求输出路径。"""
    import tkinter as tk
    from tkinter import filedialog as fd

    default_file = Path(default_file).resolve()
    root = tk.Tk()
    root.withdraw()
    file_path = fd.asksaveasfilename(
        parent=root,
        title="Select output file path",
        defaultextension=".lrc",
        filetypes=[("LRC files", "*.lrc"), ("All files", "*.*")],
        initialfile=default_file.name,
        initialdir=default_file.parent,
    )
    root.destroy()
    return file_path


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description="Karaoke lyrics alignment tool — 根据音频和行级 LRC 歌词生成词级逐字歌词",
    )
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--lyrics",
        "-l",
        help="单文件模式：LRC 歌词文件路径（未提供时进入交互模式）",
    )
    input_group.add_argument(
        "--batch-dir",
        type=Path,
        help="批处理模式：递归查找同目录同名的 .lrc 与音频文件",
    )
    parser.add_argument(
        "--audio",
        "-a",
        help="单文件模式：音频文件路径（支持 wav/mp3/flac/m4a；未提供时进入交互模式）",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="单文件模式：输出 .lrc 文件路径（默认为输入歌词同目录下的 .kara.lrc）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="批处理模式：输出根目录；默认与每个输入 LRC 同目录",
    )
    parser.add_argument(
        "--dump-dir",
        "-d",
        type=Path,
        default=None,
        help="调试音频导出目录（批处理时在其下按相对路径创建子目录）",
    )
    parser.add_argument(
        "--offset",
        type=float,
        default=None,
        help="手动指定全局时间偏移（ms）。正值=LRC偏早需延迟, 负值=LRC偏晚需提前。不指定时自动估计",
    )
    parser.add_argument(
        "--no-offset-estimate",
        action="store_true",
        help="禁用自动偏移估计（相当于 --offset 0）",
    )
    parser.add_argument(
        "--aligner-url",
        default="http://localhost:8787",
        help=(
            "Qwen3ForcedAligner 服务地址（默认: http://localhost:8787，"
            "与 scripts/qwen3aligner_server.py 的默认监听端口一致）"
        ),
    )
    parser.add_argument(
        "--aligner-timeout",
        type=float,
        default=120.0,
        help="单次对齐请求超时秒数（默认: 120；0 或负数表示不超时）",
    )
    parser.add_argument(
        "--metadata-filter",
        type=Path,
        default=_DEFAULT_METADATA_FILTER,
        help=f"元数据行过滤配置（默认: {_DEFAULT_METADATA_FILTER.name}，随本文件定位）",
    )
    parser.add_argument(
        "--strict-pairs",
        action="store_true",
        help="批处理时遇到无法配对的 LRC 立即失败（默认跳过并汇总）",
    )
    parser.add_argument(
        "--aligner-language",
        choices=("auto", "zh", "ja", "en", "yue", "ko"),
        default="auto",
        help="送给对齐器的语言；auto=按整首歌的行级多数票判定（默认: auto）",
    )
    parser.add_argument(
        "--target-lang",
        choices=("zh", "ja", "en"),
        default=None,
        help="只对齐该语言的行，其余行原样保留（默认: 不限制）",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="禁用响度归一化",
    )
    parser.add_argument(
        "--no-vibrato-suppress",
        action="store_true",
        help="禁用颤音抑制",
    )
    parser.add_argument(
        "--compress",
        action="store_true",
        help="启用动态范围压缩（默认关闭）",
    )
    parser.add_argument(
        "--min-vocal-activity",
        type=float,
        default=0.01,
        help="低于该归一化人声活动度的行不对齐（默认: 0.01；0=关闭）",
    )
    parser.add_argument(
        "--refine-collapsed-words",
        action="store_true",
        help=(
            "把对齐器返回的零长度词摊进其后的空隙（默认关闭）。对齐器的边界量化到 "
            "80ms，时长不足一帧的词会拿到 start==end、在播放器里无法单独高亮（实测占 "
            "22.5%%）；打开后按文本长度加权把它摊进后面的空隙，且不改动任何被模型报告"
            "过的边界。这是**推断值**，所以默认关闭"
        ),
    )
    parser.add_argument(
        "--existing-byword-policy",
        choices=("realign", "preserve"),
        default="realign",
        help="已有逐字时间标签的处理方式（默认: realign；preserve=原样保留）",
    )
    parser.add_argument(
        "--separator-backend",
        choices=("demucs", "audio-separator"),
        default="demucs",
        help=(
            "分离后端（默认: demucs）。两者都是独立 worker 进程，主环境都不需要 torch。"
            "audio-separator 的价值是人声质量（MDX/VR/RoFormer 等模型），代价是它的"
            "环境更重；需要配合 --separator-model 指定模型文件名"
        ),
    )
    parser.add_argument(
        "--separator-cmd",
        nargs="+",
        default=None,
        help=(
            "自定义 worker 启动命令，优先级最高（缺省用 KARAKARA_SEPARATOR_CMD，"
            "再缺省按 --separator-backend 选择脚本）"
        ),
    )
    parser.add_argument(
        "--separator-model",
        default=None,
        help=(
            "分离模型名（默认交给 worker；Demucs 后端为 UVR_Demucs_Model_1，"
            "可用脚本的 --info 查看本地仓库里全部可选模型）"
        ),
    )
    parser.add_argument(
        "--separator-device",
        default=None,
        help="分离设备，如 cuda:0 / cpu（默认交给 worker 自动选择）",
    )
    parser.add_argument(
        "--separator-model-dir",
        type=Path,
        default=None,
        help="分离模型仓库目录（默认交给 worker 自带的路径）",
    )
    parser.add_argument(
        "--separator-timeout",
        type=float,
        default=900.0,
        help=(
            "单次分离请求超时秒数（默认: 900）。默认有限是刻意的：批处理里一个卡住的"
            "worker 不该让主程序永久挂住。首次运行还要等 uv 准备 worker 环境，"
            "必要时把它调大；0 或负数表示不超时"
        ),
    )
    parser.add_argument(
        "--sep-work-dir",
        type=Path,
        default=None,
        help="分离中间产物目录（默认用系统临时目录；每首歌用完即删）",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="批处理时遇到第一个失败任务立即停止",
    )
    return parser


@dataclass(frozen=True)
class UnpairedLyrics:
    """一个找不到唯一同名音频、因而无法处理的 LRC。"""

    lyrics_path: Path
    reason: str


@dataclass(frozen=True)
class DiscoveryResult:
    """批处理发现的结果：可处理的配对 + 被跳过的 LRC。"""

    jobs: list[BatchJob]
    skipped: list[UnpairedLyrics]


def discover_batch_jobs(
    input_dir: Path,
    output_dir: Path | None = None,
    *,
    strict: bool = False,
) -> DiscoveryResult:
    """递归发现同目录、同文件名的 LRC/音频对。

    已生成的 ``*.kara.lrc`` 会被排除。找不到音频或同名音频不唯一的 LRC 默认
    **跳过并汇总**（``strict=True`` 时改为直接抛 ``ValueError``）。

    默认必须宽松：真实曲库里总会有少量 LRC 没有配套音频（人工泄漏、翻译稿、
    只有 instrumental 等）。实测在 6622 首的曲库上有 86 个这样的文件——若按
    「一个坏配对就抛异常」，整个批处理连能配对的 6536 首都不会处理。
    """
    if not input_dir.is_dir():
        raise ValueError(f"batch directory does not exist: {input_dir}")

    # 每个目录只枚举一次：大曲库里几千个 LRC 常挤在少数几个目录（实测某曲库
    # 6600+ 个 LRC 集中在 ~1000 文件级的根目录），逐行 iterdir() 会把同一批目录
    # 条目重复枚举几百万次，Windows 上足以让「发现阶段」从秒级涨到分钟级。
    audio_index: dict[Path, dict[str, list[Path]]] = {}

    def audio_by_stem(directory: Path) -> dict[str, list[Path]]:
        index = audio_index.get(directory)
        if index is None:
            index = {}
            for entry in directory.iterdir():
                if entry.is_file() and entry.suffix.lower() in _AUDIO_SUFFIXES:
                    index.setdefault(entry.stem, []).append(entry)
            for paths in index.values():
                paths.sort()
            audio_index[directory] = index
        return index

    jobs: list[BatchJob] = []
    skipped: list[UnpairedLyrics] = []
    for lyrics_path in sorted(input_dir.rglob("*.lrc")):
        if lyrics_path.stem.endswith(".kara"):
            continue
        candidates = audio_by_stem(lyrics_path.parent).get(lyrics_path.stem, [])
        if len(candidates) != 1:
            description = (
                "none"
                if not candidates
                else ", ".join(str(path) for path in candidates)
            )
            if strict:
                raise ValueError(
                    f"Expected exactly one audio file for {lyrics_path}, "
                    f"found: {description}"
                )
            skipped.append(UnpairedLyrics(lyrics_path, description))
            continue
        if output_dir is None:
            output_path = lyrics_path.with_suffix(".kara.lrc")
        else:
            relative = lyrics_path.relative_to(input_dir)
            output_path = output_dir / relative.with_suffix(".kara.lrc")
        jobs.append(BatchJob(lyrics_path, candidates[0], output_path))
    return DiscoveryResult(jobs=jobs, skipped=skipped)


def save_lyrics(lyrics: Lyrics, output_path: Path) -> None:
    """按兼容 foobar2000 的格式写入逐字 LRC。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        lyrics.dumps(
            options=SerializationOptions(
                use_bracket_for_byword_tag=True,
                line_tag_decimal_length=3,
                word_tag_decimal_length=3,
            )
        ),
        encoding="utf-8",
    )


def build_separator(args: argparse.Namespace) -> SubprocessStemSeparator:
    """按 CLI 选项构造分离器。

    ``--separator-cmd`` 优先；否则按 ``--separator-backend`` 选脚本，用
    ``uv run --script`` 拉起（worker 脚本头部自带 PEP 723 内联依赖）。
    """
    command = args.separator_cmd
    if command is None:
        command = ["uv", "run", "--script", _SEPARATOR_WORKERS[args.separator_backend]]
    return SubprocessStemSeparator(
        command=command,
        model=args.separator_model,
        device=args.separator_device,
        model_dir=args.separator_model_dir,
        request_timeout=_resolve_timeout(args.separator_timeout),
    )


def process_job(
    job: BatchJob,
    *,
    aligner: Qwen3ForcedAligner,
    separator: SubprocessStemSeparator,
    metadata_filter: MetadataFilter,
    preprocess_config: AudioPreprocessConfig,
    dump_dir: Path | None,
    separate_work_dir: Path | None,
    offset_ms: float | None,
    min_vocal_activity: float,
    existing_byword_policy: ExistingBywordPolicy,
    aligner_language: str = "auto",
    target_lang: str | None = None,
    refine_collapsed_words: bool = False,
) -> None:
    """处理一组输入，并在函数返回时释放该任务的大型音频对象。"""
    lyrics = Lyrics.loads(job.lyrics_path.read_text(encoding="utf-8"))
    aligned = gen_kara(
        lyrics,
        job.audio_path,
        aligner=aligner,
        separator=separator,
        metadata_filter=metadata_filter,
        preprocess_config=preprocess_config,
        dump_dir=dump_dir,
        separate_work_dir=separate_work_dir,
        offset_ms=offset_ms,
        min_vocal_activity=min_vocal_activity,
        existing_byword_policy=existing_byword_policy,
        aligner_language=aligner_language,
        target_lang=target_lang,
        refine_collapsed_words=refine_collapsed_words,
    )
    save_lyrics(aligned, job.output_path)


def release_item_resources() -> None:
    """回收单个任务产生的 CPU/GPU 临时对象。

    GPU 侧的资源现在由分离 worker 进程独自持有，主进程不再初始化 CUDA 上下文，
    因此这里只需要回收 Python 对象。
    """
    gc.collect()


def resolve_offset(args: argparse.Namespace) -> float | None:
    """将 CLI 偏移选项转为流水线参数。"""
    if args.no_offset_estimate:
        return 0.0
    offset = args.offset
    return offset if isinstance(offset, float) else None


def _resolve_timeout(value: float | None) -> float | None:
    """把 CLI 的超时值转成 ``requests``/``Popen`` 语义：``<=0`` 表示不超时。"""
    if value is None or value <= 0:
        return None
    return value


def run_batch(args: argparse.Namespace) -> int:
    """批量执行任务，共享模型和 HTTP 客户端，逐项回收临时对象。"""
    assert args.batch_dir is not None
    input_dir = args.batch_dir.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir is not None else None
    discovery = discover_batch_jobs(input_dir, output_dir, strict=args.strict_pairs)
    jobs = discovery.jobs
    if discovery.skipped:
        print(
            f"Skipped {len(discovery.skipped)} LRC file(s) without a unique "
            f"same-name audio (use --strict-pairs to fail instead):"
        )
        for item in discovery.skipped[:10]:
            relative = item.lyrics_path.relative_to(input_dir)
            print(f"  - {relative} (found: {item.reason})")
        if len(discovery.skipped) > 10:
            print(f"  ... and {len(discovery.skipped) - 10} more")
    if not jobs:
        print(f"No matching LRC/audio pairs found under: {input_dir}")
        return 0

    preprocess_config = AudioPreprocessConfig(
        normalize=not args.no_normalize,
        suppress_vibrato=not args.no_vibrato_suppress,
        compress=args.compress,
    )
    metadata_filter = MetadataFilter.from_file(args.metadata_filter)
    aligner = Qwen3ForcedAligner(
        base_url=args.aligner_url, timeout=_resolve_timeout(args.aligner_timeout)
    )
    separator = build_separator(args)
    failures = 0
    try:
        for index, job in enumerate(jobs, start=1):
            relative = job.lyrics_path.relative_to(input_dir)
            item_dump_dir = (
                args.dump_dir / relative.with_suffix("") if args.dump_dir else None
            )
            print(f"[{index}/{len(jobs)}] {relative}")
            try:
                process_job(
                    job,
                    aligner=aligner,
                    separator=separator,
                    metadata_filter=metadata_filter,
                    preprocess_config=preprocess_config,
                    dump_dir=item_dump_dir,
                    separate_work_dir=args.sep_work_dir,
                    offset_ms=resolve_offset(args),
                    min_vocal_activity=args.min_vocal_activity,
                    existing_byword_policy=args.existing_byword_policy,
                    aligner_language=args.aligner_language,
                    target_lang=args.target_lang,
                    refine_collapsed_words=args.refine_collapsed_words,
                )
            except Exception as exc:
                failures += 1
                print(f"FAILED {relative}: {exc}")
                if args.fail_fast:
                    raise
            finally:
                release_item_resources()
    finally:
        aligner.close()
        separator.close()

    print(f"Batch finished: {len(jobs) - failures} succeeded, {failures} failed")
    return 1 if failures else 0


def run_single(args: argparse.Namespace) -> int:
    """交互式或显式路径的单文件处理入口。"""
    lyrics_src = (
        args.lyrics or input("lyrics: ").strip() or ask_for_input_file("lyrics")
    )
    audio_src = (
        args.audio or input("audio file: ").strip() or ask_for_input_file("audio")
    )
    output_src = args.output
    if not output_src:
        output_src = input("output: ").strip() or ask_for_output_path(
            Path(lyrics_src).with_suffix(".kara.lrc")
        )
    if not output_src:
        print("No output file specified. Exiting...")
        return 0

    job = BatchJob(Path(lyrics_src), Path(audio_src), Path(output_src))
    preprocess_config = AudioPreprocessConfig(
        normalize=not args.no_normalize,
        suppress_vibrato=not args.no_vibrato_suppress,
        compress=args.compress,
    )
    aligner = Qwen3ForcedAligner(
        base_url=args.aligner_url, timeout=_resolve_timeout(args.aligner_timeout)
    )
    separator = build_separator(args)
    try:
        process_job(
            job,
            aligner=aligner,
            separator=separator,
            metadata_filter=MetadataFilter.from_file(args.metadata_filter),
            preprocess_config=preprocess_config,
            dump_dir=args.dump_dir,
            separate_work_dir=args.sep_work_dir,
            offset_ms=resolve_offset(args),
            min_vocal_activity=args.min_vocal_activity,
            existing_byword_policy=args.existing_byword_policy,
            aligner_language=args.aligner_language,
            target_lang=args.target_lang,
            refine_collapsed_words=args.refine_collapsed_words,
        )
    finally:
        aligner.close()
        separator.close()
    print(f"saved: {job.output_path}")
    return 0


def main(argv: list[str] | None = None) -> None:
    """解析 CLI 参数后执行单文件或批量对齐。"""
    setup_logging()
    args = build_parser().parse_args(argv)
    if args.batch_dir is not None:
        raise SystemExit(run_batch(args))
    raise SystemExit(run_single(args))


if __name__ == "__main__":
    main()
