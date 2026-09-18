"""LRC 的读与写。

写出格式刻意选 **foobar2000 兼容**的逐字标签（``[mm:ss.xxx]`` 包住每个词）加 3 位
小数——那是本项目产物的主要消费方。读入侧直接交给 ``lemony_lrc_parser``。
"""

from __future__ import annotations

from pathlib import Path

from lemony_lrc_parser import Lyrics, SerializationOptions


def load_lyrics(path: str | Path) -> Lyrics:
    """读入一个 UTF-8 的 LRC 文件。"""
    return Lyrics.loads(Path(path).read_text(encoding="utf-8"))


def save_lyrics(lyrics: Lyrics, path: str | Path) -> None:
    """写入逐字 LRC；父目录不存在时一并创建。"""
    output_path = Path(path)
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
