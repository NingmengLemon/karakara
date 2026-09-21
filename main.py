"""CLI 入口：只做参数解析与装配，逻辑在 :mod:`karakara` 包里。

这里刻意保留三样东西：参数解析器、两个运行模式（单文件/批处理）的**编排**、
以及仓库自带配置文件的定位。真正的逻辑（发现配对、跑一组输入、后端登记表、
文件对话框）都在包本体里，可以被直接测试。
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

from karakara import backends
from karakara.aligner import HttpAligner
from karakara.backends import (
    ALIGNER_BACKENDS,
    DEFAULT_ALIGNER_BACKEND,
    DEFAULT_SEPARATOR_BACKEND,
    SEPARATOR_BACKENDS,
    UnsupportedAlignerLanguage,
)
from karakara.batch import (
    BatchJob,
    discover_batch_jobs,
    process_job,
    release_item_resources,
)
from karakara.interactive import ask_for_input_file, ask_for_output_path
from karakara.logging import setup_logging
from karakara.preprocess import AudioPreprocessConfig
from karakara.separator import SubprocessStemSeparator
from karakara.utils.metadata import MetadataFilter

#: 仓库自带的元数据过滤配置。刻意相对**本文件**定位而不是 CWD，
#: 否则从别的目录运行就会去找那个目录下的同名文件（然后 FileNotFoundError）。
_DEFAULT_METADATA_FILTER = Path(__file__).resolve().parent / "metadata_filter.toml"


def _describe(descriptions: Iterable[str]) -> str:
    """把登记表里的描述渲染成 argparse 的 help 片段（同一段话只写一遍）。"""
    return "；".join(descriptions)


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
        "--aligner-backend",
        choices=tuple(ALIGNER_BACKENDS),
        default=DEFAULT_ALIGNER_BACKEND,
        help=(
            f"对齐后端（默认: {DEFAULT_ALIGNER_BACKEND}）。两者是独立服务、共用同一套 "
            f"/align 契约：{_describe(b.description for b in ALIGNER_BACKENDS.values())}"
        ),
    )
    parser.add_argument(
        "--aligner-url",
        default=None,
        help=(
            "对齐服务地址；缺省按 --aligner-backend 选（"
            + "，".join(f"{n} → {b.default_url}" for n, b in ALIGNER_BACKENDS.items())
            + "）"
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
        # 选项集合是两个后端语言能力的**并集**（yue/ko 只有 qwen3 支持），
        # 所以选了 hfa 又给 yue/ko 时由 main() 在开跑前拦下。
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
            "把对齐器返回的零长度词摊进其后的空隙（默认关闭）。时长不足一个对齐"
            "帧的词会拿到 start==end，在播放器里无法单独高亮：qwen3 后端帧移 80ms 时"
            "实测占 22.5%%，默认的 HubertFA 是 10ms、约 1.9%%。打开后按文本长度加权"
            "把它摊进后面的空隙，且不改动任何被模型报告过的边界。这是**推断值**，"
            "所以默认关闭"
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
        choices=tuple(SEPARATOR_BACKENDS),
        default=DEFAULT_SEPARATOR_BACKEND,
        help=(
            f"分离后端（默认: {DEFAULT_SEPARATOR_BACKEND}）。两者都是独立 worker 进程，"
            f"主环境都不需要 torch："
            f"{_describe(b.description for b in SEPARATOR_BACKENDS.values())}"
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


def build_separator(args: argparse.Namespace) -> SubprocessStemSeparator:
    """按 CLI 选项构造分离器（``--separator-cmd`` 优先级最高）。"""
    return backends.build_separator(
        args.separator_backend,
        command=args.separator_cmd,
        model=args.separator_model,
        device=args.separator_device,
        model_dir=args.separator_model_dir,
        request_timeout=backends.resolve_timeout(args.separator_timeout),
    )


def build_aligner(args: argparse.Namespace) -> HttpAligner:
    """按 CLI 选项构造对齐客户端（地址按后端选，超时 ``<=0`` 视为不限）。"""
    return HttpAligner(
        base_url=backends.resolve_aligner_url(args.aligner_backend, args.aligner_url),
        timeout=backends.resolve_timeout(args.aligner_timeout),
    )


def build_preprocess_config(args: argparse.Namespace) -> AudioPreprocessConfig:
    """按 CLI 选项构造预处理配置。"""
    return AudioPreprocessConfig(
        normalize=not args.no_normalize,
        suppress_vibrato=not args.no_vibrato_suppress,
        compress=args.compress,
    )


def resolve_offset(args: argparse.Namespace) -> float | None:
    """将 CLI 偏移选项转为流水线参数（``--no-offset-estimate`` 等价于 ``--offset 0``）。"""
    if args.no_offset_estimate:
        return 0.0
    offset = args.offset
    return offset if isinstance(offset, float) else None


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

    preprocess_config = build_preprocess_config(args)
    metadata_filter = MetadataFilter.from_file(args.metadata_filter)
    aligner = build_aligner(args)
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
    aligner = build_aligner(args)
    separator = build_separator(args)
    try:
        process_job(
            job,
            aligner=aligner,
            separator=separator,
            metadata_filter=MetadataFilter.from_file(args.metadata_filter),
            preprocess_config=build_preprocess_config(args),
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
    parser = build_parser()
    args = parser.parse_args(argv)
    # 语言能力在**开跑之前**检查：不受支持的语言若留到服务端才报 400，
    # 人声分离（几十秒）已经白跑完了。
    try:
        backends.ensure_aligner_language(
            args.aligner_backend,
            None if args.aligner_language == "auto" else args.aligner_language,
        )
    except UnsupportedAlignerLanguage as exc:
        parser.error(str(exc))
    if args.batch_dir is not None:
        raise SystemExit(run_batch(args))
    raise SystemExit(run_single(args))


if __name__ == "__main__":
    main()
