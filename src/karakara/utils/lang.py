from __future__ import annotations

import re
from logging import getLogger
from typing import Literal

logger = getLogger(__name__)

# --------------------------------------------------------------------------
# 语言检测
# --------------------------------------------------------------------------

_JA_PATTERN = re.compile(r"[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FBF\u3000-\u303F]")
_ZH_PATTERN = re.compile(r"[\u4E00-\u9FFF]")
_EN_PATTERN = re.compile(r"[A-Za-z]")


def detect_lang(s: str) -> Literal["ja", "zh", "en"] | None:
    """根据字符串内容判断语言类型。

    Args:
        s: 输入字符串。

    Returns:
        "ja"（日语）、"zh"（中文）、"en"（英语）或 None（无法判断）。
    """
    ja_count = len(_JA_PATTERN.findall(s))
    zh_count = len(_ZH_PATTERN.findall(s))
    en_count = len(_EN_PATTERN.findall(s))

    counts: dict[str, int] = {"ja": ja_count, "zh": zh_count, "en": en_count}
    max_lang = max(counts, key=counts.__getitem__)
    if counts[max_lang] == 0:
        logger.info(f"unknown language: {s!r}")
        return None
    logger.debug(f"language={max_lang!r}: {s!r}")
    return max_lang  # type: ignore[return-value]
