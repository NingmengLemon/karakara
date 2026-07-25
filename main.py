from __future__ import annotations

import argparse
from os import PathLike
from pathlib import Path
from typing import Literal

from lemony_lrc_parser import Lyrics, SerializationOptions

from karakara.aligner import Qwen3ForcedAligner
from karakara.core import gen_kara
from karakara.utils.metadata import MetadataFilter
from karakara.logging import setup_logging
from karakara.preprocess import AudioPreprocessConfig
from karakara.separator.demucs import DemucsSeparator


def ask_for_input_file(type_: Literal["lyrics", "audio"]) -> str:
    # use tkiner dialog to ask for file path
    import tkinter as tk
    from tkinter import filedialog as fd

    root = tk.Tk()
    root.withdraw()  # Hide the main window
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


def ask_for_output_path(default_file: PathLike) -> str:
    # use tkiner dialog to ask for output file path
    import tkinter as tk
    from tkinter import filedialog as fd

    default_file = Path(default_file).resolve()
    root = tk.Tk()
    root.withdraw()  # Hide the main window
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
    # --- 文件 I/O ---
    parser.add_argument(
        "--lyrics",
        "-l",
        help="LRC 歌词文件路径（未提供时进入交互模式）",
    )
    parser.add_argument(
        "--audio",
        "-a",
        help="音频文件路径（支持 wav/mp3/flac/m4a；未提供时进入交互模式）",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="输出 .lrc 文件路径（默认为输入歌词同目录下的 .kara.lrc）",
    )

    # --- 调试 ---
    parser.add_argument(
        "--dump-dir",
        "-d",
        default=None,
        help="调试音频导出目录（不指定则不导出中间结果）",
    )

    # --- 偏移预测 ---
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

    # --- 对齐器配置 ---
    parser.add_argument(
        "--aligner-url",
        default="http://localhost:8787",
        help="Qwen3ForcedAligner 服务地址（默认: http://localhost:8787）",
    )

    # --- 预处理开关 ---
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

    return parser


def main(argv: list[str] | None = None) -> None:
    """入口：解析命令行参数并运行 Karaoke 对齐流水线。

    命令行参数优先；未通过 CLI 提供的必选参数将回退到交互式输入。
    """
    setup_logging()

    parser = build_parser()
    args = parser.parse_args(argv)

    # -------- 歌词文件 --------
    lyrics_src: str = args.lyrics or ""
    if not lyrics_src:
        lyrics_src = input("lyrics: ").strip() or ask_for_input_file("lyrics")
    print("lrc src:", lyrics_src)
    with open(lyrics_src, "r", encoding="utf-8") as fp:
        lyrics = Lyrics.loads(fp.read())

    # -------- 音频文件 --------
    audio_src: str = args.audio or ""
    if not audio_src:
        audio_src = input("audio file: ").strip() or ask_for_input_file("audio")
    print("audio src:", audio_src)

    # -------- 预处理配置 --------
    preprocess_cfg = AudioPreprocessConfig(
        normalize=not args.no_normalize,
        suppress_vibrato=not args.no_vibrato_suppress,
        compress=args.compress,
    )

    # -------- dump 目录 --------
    dump_dir = args.dump_dir or input("dump dir (blank=skip): ").strip() or None

    # -------- 偏移参数 --------
    if args.no_offset_estimate:
        offset_ms: float | None = 0.0
    elif args.offset is not None:
        offset_ms = args.offset
    else:
        offset_ms = None  # 自动估计

    # -------- 加载元数据过滤器 --------
    metadata_filter = MetadataFilter.from_file("metadata_filter.toml")

    # -------- 执行流水线 --------
    lyrics = gen_kara(
        lyrics,
        audio_src,
        aligner=Qwen3ForcedAligner(base_url=args.aligner_url),
        separator=DemucsSeparator(),
        metadata_filter=metadata_filter,
        preprocess_config=preprocess_cfg,
        dump_dir=dump_dir,
        offset_ms=offset_ms,
    )

    # -------- 输出路径 --------
    save_as: str = args.output or ""
    if not save_as:
        save_as = input("output: ").strip() or ask_for_output_path(
            Path(lyrics_src).with_suffix(".kara.lrc")
        )
    if not save_as:
        print("No output file specified. Exiting...")
        return
    print("saving as:", save_as)
    with open(save_as, "w+", encoding="utf-8") as fp:
        fp.write(
            lyrics.dumps(
                options=SerializationOptions(
                    use_bracket_for_byword_tag=True,
                    # compatible with foobar2000
                    line_tag_decimal_length=3,
                    word_tag_decimal_length=3,
                )
            )
        )


if __name__ == "__main__":
    main()
