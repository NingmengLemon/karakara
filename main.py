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
from karakara.separator.demucs import DemucsSeparator
from karakara.utils.metadata import MetadataFilter

_AUDIO_SUFFIXES = frozenset({".wav", ".mp3", ".flac", ".m4a"})


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
        help="Qwen3ForcedAligner 服务地址（默认: http://localhost:8787）",
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
        "--existing-byword-policy",
        choices=("realign", "preserve"),
        default="realign",
        help="已有逐字时间标签的处理方式（默认: realign；preserve=原样保留）",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="批处理时遇到第一个失败任务立即停止",
    )
    return parser


def discover_batch_jobs(
    input_dir: Path, output_dir: Path | None = None
) -> list[BatchJob]:
    """递归发现同目录、同文件名的 LRC/音频对。

    已生成的 ``*.kara.lrc`` 会被排除。找不到音频或同名音频不唯一时抛出
    ``ValueError``，避免批处理时错误地配对文件。
    """
    if not input_dir.is_dir():
        raise ValueError(f"batch directory does not exist: {input_dir}")

    jobs: list[BatchJob] = []
    for lyrics_path in sorted(input_dir.rglob("*.lrc")):
        if lyrics_path.stem.endswith(".kara"):
            continue
        candidates = sorted(
            path
            for path in lyrics_path.parent.iterdir()
            if path.is_file()
            and path.stem == lyrics_path.stem
            and path.suffix.lower() in _AUDIO_SUFFIXES
        )
        if len(candidates) != 1:
            description = (
                "none"
                if not candidates
                else ", ".join(str(path) for path in candidates)
            )
            raise ValueError(
                f"Expected exactly one audio file for {lyrics_path}, found: {description}"
            )
        if output_dir is None:
            output_path = lyrics_path.with_suffix(".kara.lrc")
        else:
            relative = lyrics_path.relative_to(input_dir)
            output_path = output_dir / relative.with_suffix(".kara.lrc")
        jobs.append(BatchJob(lyrics_path, candidates[0], output_path))
    return jobs


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


def process_job(
    job: BatchJob,
    *,
    aligner: Qwen3ForcedAligner,
    separator: DemucsSeparator,
    metadata_filter: MetadataFilter,
    preprocess_config: AudioPreprocessConfig,
    dump_dir: Path | None,
    offset_ms: float | None,
    min_vocal_activity: float,
    existing_byword_policy: ExistingBywordPolicy,
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
        offset_ms=offset_ms,
        min_vocal_activity=min_vocal_activity,
        existing_byword_policy=existing_byword_policy,
    )
    save_lyrics(aligned, job.output_path)


def release_item_resources() -> None:
    """回收单个任务产生的 CPU/GPU 临时对象，保留可复用的模型实例。"""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def resolve_offset(args: argparse.Namespace) -> float | None:
    """将 CLI 偏移选项转为流水线参数。"""
    if args.no_offset_estimate:
        return 0.0
    offset = args.offset
    return offset if isinstance(offset, float) else None


def run_batch(args: argparse.Namespace) -> int:
    """批量执行任务，共享模型和 HTTP 客户端，逐项回收临时对象。"""
    assert args.batch_dir is not None
    input_dir = args.batch_dir.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir is not None else None
    jobs = discover_batch_jobs(input_dir, output_dir)
    if not jobs:
        print(f"No matching LRC/audio pairs found under: {input_dir}")
        return 0

    preprocess_config = AudioPreprocessConfig(
        normalize=not args.no_normalize,
        suppress_vibrato=not args.no_vibrato_suppress,
        compress=args.compress,
    )
    metadata_filter = MetadataFilter.from_file("metadata_filter.toml")
    aligner = Qwen3ForcedAligner(base_url=args.aligner_url)
    separator = DemucsSeparator()
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
                    offset_ms=resolve_offset(args),
                    min_vocal_activity=args.min_vocal_activity,
                    existing_byword_policy=args.existing_byword_policy,
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
    aligner = Qwen3ForcedAligner(base_url=args.aligner_url)
    try:
        process_job(
            job,
            aligner=aligner,
            separator=DemucsSeparator(),
            metadata_filter=MetadataFilter.from_file("metadata_filter.toml"),
            preprocess_config=preprocess_config,
            dump_dir=args.dump_dir,
            offset_ms=resolve_offset(args),
            min_vocal_activity=args.min_vocal_activity,
            existing_byword_policy=args.existing_byword_policy,
        )
    finally:
        aligner.close()
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
