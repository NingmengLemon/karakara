"""
metadata.py

歌词文本相关的元数据行过滤工具。

提供 ``MetadataFilter`` 类，支持多种检测策略通过 **逻辑或** 组合：
  - 关键字匹配（作词/作曲/编曲 等）
  - ID3 风格标签（[ti:...] / [ar:...] 等）
  - 括号段落标记（(Prelude) / (间奏) 等）
  - 纯数字行
  - 自定义正则模式

所有参数必须显式提供（无隐式默认值）。推荐通过 ``MetadataFilter.from_file()``
从 TOML 配置文件加载。
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Any


class MetadataFilter:
    """可组合的元数据行过滤器。

    每种检测策略可独立开关，所有启用的策略通过 **逻辑或** 组合：
    任意一项命中即判定为元数据行。

    Attributes:
        keywords: 关键字列表，匹配形如 ``关键字: 值`` 的行。
        detect_id3_tags: 启用 ID3 风格标签检测。
        detect_parenthetical: 启用括号段落标记检测。
        detect_pure_numbers: 启用纯数字行检测。
        custom_patterns: 额外的正则模式列表（字符串形式，大小写不敏感）。
    """

    # ---------- 构造 ----------

    def __init__(
        self,
        *,
        keywords: list[str],
        id3_tags: frozenset[str],
        parenthetical_markers: list[str],
        detect_id3_tags: bool,
        detect_parenthetical: bool,
        detect_pure_numbers: bool,
        custom_patterns: list[str],
    ) -> None:
        """初始化过滤器（所有参数必填）。

        Args:
            keywords: 关键字列表。传入空列表可关闭关键字检测。
            id3_tags: ID3 风格标签名集合，如 ``frozenset({"ti", "ar", "al"})``。
            parenthetical_markers: 括号段落标记列表，如 ``["间奏", "Prelude"]``。
            detect_id3_tags: 是否匹配形如 ``[ti:...]`` 的 ID3 标签行。
            detect_parenthetical: 是否匹配形如 ``(Prelude)`` / ``(间奏)`` 的段落标记。
            detect_pure_numbers: 是否匹配仅含数字的行（如倒计时）。
            custom_patterns: 自定义正则列表，每项为完整的正则表达式字符串。
        """
        self._keywords = list(keywords)
        self._id3_tags = id3_tags
        self._parenthetical_markers = list(parenthetical_markers)
        self._detect_id3_tags = detect_id3_tags
        self._detect_parenthetical = detect_parenthetical
        self._detect_pure_numbers = detect_pure_numbers
        self._custom_patterns = list(custom_patterns)

        self._pattern: re.Pattern[str] = self._build_pattern()

    # ---------- 工厂方法 ----------

    @classmethod
    def from_file(cls, path: str | Path) -> MetadataFilter:
        """从 TOML 配置文件加载过滤器。

        配置格式参见 ``metadata_filter.toml``。

        Args:
            path: TOML 配置文件路径。

        Returns:
            按配置构造的 ``MetadataFilter`` 实例。
        """
        with Path(path).open("rb") as f:
            data: dict[str, Any] = tomllib.load(f)

        enabled = data.get("enabled", {})
        detect_id3: bool = enabled.get("id3_tags", False)
        detect_paren: bool = enabled.get("parenthetical", False)
        detect_numbers: bool = enabled.get("pure_numbers", False)

        # 默认情况下 keywords 开启（向后兼容）
        keywords_enabled: bool = enabled.get("keywords", True)
        keywords: list[str] = (
            list(data.get("keywords", {}).get("items", [])) if keywords_enabled else []
        )

        id3_tags: frozenset[str] = frozenset(data.get("id3_tags", {}).get("tags", []))
        paren_markers: list[str] = list(
            data.get("parenthetical", {}).get("markers", [])
        )
        custom_patterns: list[str] = list(data.get("custom", {}).get("patterns", []))

        return cls(
            keywords=keywords,
            id3_tags=id3_tags,
            parenthetical_markers=paren_markers,
            detect_id3_tags=detect_id3,
            detect_parenthetical=detect_paren,
            detect_pure_numbers=detect_numbers,
            custom_patterns=custom_patterns,
        )

    # ---------- 公共 API ----------

    def is_metadata(self, s: str) -> bool:
        """判断字符串是否为元数据行。

        Args:
            s: 待检测的字符串。

        Returns:
            若任意一种启用的检测策略命中则返回 ``True``。
        """
        return bool(self._pattern.search(s))

    def __call__(self, s: str) -> bool:
        """可用作 ``filter()`` 的回调。"""
        return self.is_metadata(s)

    # ---------- 内部 ----------

    def _build_pattern(self) -> re.Pattern[str]:
        parts: list[str] = []

        # 1) 关键字匹配：以关键字开头，后接冒号和值
        if self._keywords:
            escaped = "|".join(re.escape(k) for k in self._keywords)
            parts.append(rf"^\s*(?:{escaped})\s*[：:].+")

        # 2) ID3 风格标签（如 [ti:标题] [ar:歌手]）
        if self._detect_id3_tags:
            tags = "|".join(re.escape(t) for t in self._id3_tags)
            parts.append(rf"\[(?:{tags}):[^\]]*\]")

        # 3) 括号段落标记（如 (Prelude) (间奏)）—— 要求整行匹配
        if self._detect_parenthetical:
            markers = "|".join(re.escape(m) for m in self._parenthetical_markers)
            parts.append(rf"^\s*\(\s*(?:{markers})\s*\)\s*$")

        # 4) 纯数字行
        if self._detect_pure_numbers:
            parts.append(r"^\s*\d+\s*$")

        # 5) 自定义模式
        parts.extend(self._custom_patterns)

        if not parts:
            # 永不匹配
            return re.compile(r"(?!)")

        combined = "|".join(parts)
        return re.compile(combined, re.IGNORECASE)
