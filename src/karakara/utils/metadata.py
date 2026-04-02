"""
metadata.py

歌词文本相关的工具函数。
"""

from __future__ import annotations

import re

# 常见的元数据行关键词（中文 / 繁体中文 / 日文）
METADATA_KEYWORDS: list[str] = [
    "作词",
    "作曲",
    "编曲",
    "演唱",
    "专辑",
    "歌手",
    "制作人",
    "和声",
    "混音",
    "录音",
    "监制",
    "策划",
    "封面设计",
    "文案",
    "出品",
    "OP",
    "SP",
    "翻译",
    "PV",
    "母带",
    "调教",
    "调校",
    "曲绘",
    "原曲",
    "編曲",
    "作詞",
    "唄",
    "呗",
]

_METADATA_PATTERN = re.compile(
    rf"({'|'.join(re.escape(k) for k in METADATA_KEYWORDS)})\s*[：\:].+"
)


def is_metadataline(s: str) -> bool:
    """判断字符串是否为元数据行（如「作词: xxx」「作曲: xxx」）。

    Args:
        s: 待检测的字符串。

    Returns:
        若疑似元数据行则返回 True，否则返回 False。
    """
    return bool(_METADATA_PATTERN.search(s))
