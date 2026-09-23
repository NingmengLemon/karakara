"""偏移对照工具：人工看波形与歌词对齐，把偏移写回歌词文件。

与 CLI 并列的另一个入口。它**不**做分离、不做对齐、不生成逐字产物；它只解决一件事：
「这份歌词整体偏了多少」。看波形（有人声的地方）与歌词标记（每行的起点）对不对得上，
用微调按钮或键盘挪一挪，确认后原地写回。

用法::

    # 歌词 + 音频（音频缺省时找同目录同名的）
    uv run --group gui python scripts/offset_gui.py song.lrc --audio song.flac

    # 只看波形/标记，不播放（没装 sounddevice 时的自动状态）
    uv run --group gui python scripts/offset_gui.py song.lrc --audio song.flac

    # 起手就带一个初值（例如 CLI 提示的 --offset）
    uv run --group gui python scripts/offset_gui.py song.lrc --offset -120

操作：空格播放/暂停、``停止``、``上一行``/``下一行``（或 ``[`` ``]``）、``,`` ``.`` 微调
10ms、滚轮缩放、拖动平移、点画布定位、``循环本行`` 反复听当前行、``写回歌词文件`` 落地
（写入前的内容会备份成 ``<文件名>.bak``）。

``--selftest`` 会建好窗口、画一帧就退出，用来在没有人工介入的情况下确认界面能起来。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from karakara.batch import AUDIO_SUFFIXES  # noqa: E402
from karakara.gui.session import OffsetSession, find_vocals  # noqa: E402


def find_audio(lyrics_path: Path) -> Path | None:
    """找同目录同名的音频（与批处理的配对规则一致）。"""
    candidates = sorted(
        entry
        for entry in lyrics_path.parent.iterdir()
        if entry.is_file()
        and entry.suffix.lower() in AUDIO_SUFFIXES
        and entry.stem == lyrics_path.stem
    )
    return candidates[0] if len(candidates) == 1 else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="人工对照波形与歌词、调全局偏移、原地写回 LRC",
    )
    parser.add_argument("lyrics", type=Path, help="歌词文件（会被原地写回）")
    parser.add_argument(
        "--audio",
        type=Path,
        default=None,
        help="音频文件；缺省时在同目录找同名的 wav/mp3/flac/m4a",
    )
    parser.add_argument(
        "--vocals",
        type=Path,
        default=None,
        help=(
            "分离人声轨，只用于画波形（混音波形看不出起唱点）；"
            "缺省时自动找同目录的 <名字>_vocals.wav 或 tmp/perturbation/ 里的缓存"
        ),
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="起手偏移（ms），正值表示歌词偏早、需要延后",
    )
    parser.add_argument(
        "--bucket-ms",
        type=float,
        default=1.0,
        help="波形包络的分辨率（默认 1ms 一个桶）",
    )
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="建好窗口画一帧就退出（不需要人工介入，用于自检）",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    lyrics_path: Path = args.lyrics
    if not lyrics_path.is_file():
        print(f"歌词文件不存在: {lyrics_path}", file=sys.stderr)
        return 2

    audio_path: Path | None = args.audio
    if audio_path is None:
        audio_path = find_audio(lyrics_path)
        if audio_path is not None:
            print(f"自动配到音频: {audio_path.name}")
        else:
            print("没有找到同名音频，只显示歌词（可用界面上的「打开音频…」指定）")
    elif not audio_path.is_file():
        print(f"音频文件不存在: {audio_path}", file=sys.stderr)
        return 2

    session = OffsetSession(
        lyrics_path,
        audio_path=audio_path,
        vocals_path=args.vocals,
        bucket_ms=args.bucket_ms,
    )
    if audio_path is not None:
        track = session.load_track()
        print(
            f"音频已载入: {track.sample_rate} Hz / {track.channels}ch / "
            f"{track.duration_ms / 1000:.1f}s"
        )
        if args.vocals is None:
            discovered = find_vocals(audio_path)
            if discovered is not None:
                session.vocals_path = discovered
                print(f"自动配到分离人声: {discovered}")
        vocals = session.load_vocals()
        if vocals is not None:
            print(f"人声波形已载入: {vocals.path.name}")
        else:
            print("没有分离人声，波形用混音画（起唱点会看不清）")
    session.set_offset(args.offset)

    from karakara.gui.app import OffsetApp, run_app  # 延迟到需要 GUI 时再 import

    if args.selftest:
        import tkinter as tk

        root = tk.Tk()
        app = OffsetApp(root, session)
        root.update_idletasks()
        root.update()
        print(
            f"selftest OK: {len(session.lines)} 行, "
            f"播放可用={app.player.available if app.player else False}, "
            f"视图 {app.view_start_ms:.0f}-{app.view_end_ms:.0f}ms"
        )
        if app.player is not None:
            app.player.close()
        root.destroy()
        return 0

    run_app(session)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
