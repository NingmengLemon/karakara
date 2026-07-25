"""metadata.py 单元测试。"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from karakara.utils.metadata import MetadataFilter


# ===========================================================================
# 辅助函数：构建默认的"仅关键字匹配" filter
# ===========================================================================

_DEFAULT_KEYWORDS: list[str] = [
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

_DEFAULT_ID3_TAGS: frozenset[str] = frozenset(
    {
        "ti",
        "ar",
        "al",
        "by",
        "re",
        "ve",
        "offset",
        "length",
        "tool",
    }
)

_DEFAULT_PAREN_MARKERS: list[str] = [
    "前奏",
    "间奏",
    "尾奏",
    "副歌",
    "主歌",
    "过渡",
    "桥段",
    "前奏曲",
    "间奏曲",
    "Prelude",
    "Interlude",
    "Outro",
    "Intro",
    "Chorus",
    "Verse",
    "Bridge",
    "Instrumental",
    "Ending",
    "Solo",
    "Opening",
    "Break",
    "Refrain",
    "イントロ",
    "アウトロ",
    "間奏",
]


def _mk_filter(
    *,
    keywords: list[str] | None = None,
    id3_tags: frozenset[str] | None = None,
    parenthetical_markers: list[str] | None = None,
    detect_id3_tags: bool = False,
    detect_parenthetical: bool = False,
    detect_pure_numbers: bool = False,
    custom_patterns: list[str] | None = None,
) -> MetadataFilter:
    """创建 MetadataFilter 的辅助函数，未传入时使用默认值。"""
    return MetadataFilter(
        keywords=keywords if keywords is not None else list(_DEFAULT_KEYWORDS),
        id3_tags=id3_tags if id3_tags is not None else _DEFAULT_ID3_TAGS,
        parenthetical_markers=(
            parenthetical_markers
            if parenthetical_markers is not None
            else list(_DEFAULT_PAREN_MARKERS)
        ),
        detect_id3_tags=detect_id3_tags,
        detect_parenthetical=detect_parenthetical,
        detect_pure_numbers=detect_pure_numbers,
        custom_patterns=custom_patterns if custom_patterns is not None else [],
    )


# ===========================================================================
# 关键字匹配（默认行为）
# ===========================================================================


class TestKeywords:
    _filter = _mk_filter()

    @pytest.mark.parametrize(
        "text",
        [
            "作词: Hiro",
            "作曲 : Sho/Nob",
            "编曲: someone",
            "作词: A 作曲: B",
            "演唱：Alice",
            "混音 : Bob",
            "OP: original",
            "SP: secondary",
            "母带 : master",
            "调教: tune",
        ],
    )
    def test_matches_known_keywords(self, text: str) -> None:
        assert self._filter.is_metadata(text)

    @pytest.mark.parametrize(
        "text",
        [
            "",
            "We don't want to be like them",
            "ファミファミファミーマ",
            "普通歌词行",
            "ただ灰になって",
        ],
    )
    def test_does_not_match_lyrics(self, text: str) -> None:
        assert not self._filter.is_metadata(text)

    def test_callable_interface(self) -> None:
        """__call__ 可用作 filter() 的回调。"""
        f = _mk_filter(keywords=["作词", "作曲"])
        items = ["作词: A", "hello", "作曲: B", "world"]
        assert list(filter(f, items)) == ["作词: A", "作曲: B"]


# ===========================================================================
# ID3 标签检测
# ===========================================================================


class TestID3Tags:
    _filter = _mk_filter(keywords=[], detect_id3_tags=True)

    @pytest.mark.parametrize(
        "text",
        [
            "[ti: 标题]",
            "[ar:歌手]",
            "[al:Album Name]",
            "[by:Author]",
            "[re: Subtitle Edit]",
            "[ve:4.0.14.0]",
            "[offset: +500]",
            "[length: 03:45]",
            "[tool: SomeTool]",
        ],
    )
    def test_matches_standard_id3(self, text: str) -> None:
        assert self._filter.is_metadata(text)

    @pytest.mark.parametrize(
        "text",
        [
            "[notag:val]",  # 非标准标签
            "[xx:abc]",
            "作词: xxx",  # 关键字未开启
            "普通歌词行",
        ],
    )
    def test_does_not_match_non_id3(self, text: str) -> None:
        assert not self._filter.is_metadata(text)


# ===========================================================================
# 括号段落标记检测
# ===========================================================================


class TestParenthetical:
    _filter = _mk_filter(keywords=[], detect_parenthetical=True)

    @pytest.mark.parametrize(
        "text",
        [
            "(Prelude)",
            "( 间奏 )",
            "(Interlude)",
            "(Outro)",
            "(Intro)",
            "(Chorus)",
            "(前奏)",
            "(间奏)",
            "(尾奏)",
            "(桥段)",
            "(イントロ)",
            "(アウトロ)",
            "(間奏)",
        ],
    )
    def test_matches_known_markers(self, text: str) -> None:
        assert self._filter.is_metadata(text)

    @pytest.mark.parametrize(
        "text",
        [
            "Not in parens",
            "(unknown marker)",
            "普通(间奏)文本",  # 嵌入文本中的括号不算独立标记
            "作词: xxx",
        ],
    )
    def test_does_not_match_other(self, text: str) -> None:
        assert not self._filter.is_metadata(text)


# ===========================================================================
# 纯数字行检测
# ===========================================================================


class TestPureNumbers:
    _filter = _mk_filter(keywords=[], detect_pure_numbers=True)

    @pytest.mark.parametrize(
        "text",
        ["5", "  42  ", "100", "0", "9999"],
    )
    def test_matches_pure_numbers(self, text: str) -> None:
        assert self._filter.is_metadata(text)

    @pytest.mark.parametrize(
        "text",
        ["abc", "5a", "a1", "", "   ", "普通"],
    )
    def test_does_not_match_non_pure(self, text: str) -> None:
        assert not self._filter.is_metadata(text)


# ===========================================================================
# 组合策略
# ===========================================================================


class TestCombined:
    _filter = MetadataFilter(
        keywords=["作词", "作曲"],
        id3_tags=_DEFAULT_ID3_TAGS,
        parenthetical_markers=list(_DEFAULT_PAREN_MARKERS),
        detect_id3_tags=True,
        detect_parenthetical=True,
        detect_pure_numbers=False,
        custom_patterns=[],
    )

    def test_combined_or_logic(self) -> None:
        assert self._filter.is_metadata("作词: Hiro")  # keyword
        assert self._filter.is_metadata("[ti:Title]")  # ID3
        assert self._filter.is_metadata("(间奏)")  # paren

    def test_combined_excludes_unselected(self) -> None:
        assert not self._filter.is_metadata("编曲: someone")  # 不在 keywords
        assert not self._filter.is_metadata("普通歌词")


# ===========================================================================
# 自定义正则
# ===========================================================================


class TestCustomPatterns:
    _filter = _mk_filter(
        keywords=[],
        custom_patterns=[r"^【.*】$", r"^\[注\d+\]", r"---\s*\w+\s*---"],
    )

    @pytest.mark.parametrize(
        "text",
        ["【Verse A】", "[注1]", "[注99]", "--- Chorus ---", "---Bridge---"],
    )
    def test_matches_custom(self, text: str) -> None:
        assert self._filter.is_metadata(text)

    def test_does_not_match_unrelated(self) -> None:
        assert not self._filter.is_metadata("作词: xxx")
        assert not self._filter.is_metadata("普通歌词")


# ===========================================================================
# 空 filter
# ===========================================================================


class TestEmptyFilter:
    _filter = _mk_filter(
        keywords=[],
        detect_id3_tags=False,
        detect_parenthetical=False,
        detect_pure_numbers=False,
    )

    @pytest.mark.parametrize(
        "text",
        ["作词: xxx", "[ti:X]", "(Prelude)", "5", "hello"],
    )
    def test_empty_filter_matches_nothing(self, text: str) -> None:
        assert not self._filter.is_metadata(text)


# ===========================================================================
# 边界 / 回归
# ===========================================================================


class TestEdgeCases:
    def test_empty_string(self) -> None:
        assert not _mk_filter().is_metadata("")

    def test_only_keyword_no_colon(self) -> None:
        """仅关键字、无冒号——不应匹配。"""
        assert not _mk_filter().is_metadata("作词")

    def test_whitespace_only(self) -> None:
        assert not _mk_filter().is_metadata("   ")

    def test_keyword_in_middle(self) -> None:
        """关键字出现在歌词中间不应匹配。"""
        assert not _mk_filter().is_metadata("他的演唱很精彩")

    def test_all_default_keywords_in_list(self) -> None:
        """确保 _DEFAULT_KEYWORDS 非空且全部可被默认 filter 识别。"""
        assert len(_DEFAULT_KEYWORDS) > 0
        f = _mk_filter()
        for keyword in _DEFAULT_KEYWORDS:
            assert f.is_metadata(f"{keyword}: test"), f"keyword {keyword!r}"

    def test_regex_injection_safety(self) -> None:
        """含正则特殊字符的关键字不应导致错误或误匹配。"""
        f = _mk_filter()
        assert not f.is_metadata("(作词)")
        assert not f.is_metadata("作词+作曲")

    def test_soft_hyphen_in_text(self) -> None:
        """Unicode 软连字符不应触发生错误。"""
        _mk_filter(
            keywords=[],
            custom_patterns=[r"\u00ad"],
        ).is_metadata("\u00ad")

    def test_id3_tag_regex_escape(self) -> None:
        """ID3 标签中的正则特殊字符不会产生意外匹配。"""
        f = MetadataFilter(
            keywords=[],
            id3_tags=frozenset({"x|ti"}),
            parenthetical_markers=[],
            detect_id3_tags=True,
            detect_parenthetical=False,
            detect_pure_numbers=False,
            custom_patterns=[],
        )
        assert f.is_metadata("[x|ti: val]")
        assert not f.is_metadata("[ti: unexpected]")

        # 来自 TOML 文件的自定义标签也应经过转义
        import tempfile

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".toml", delete=False, encoding="utf-8"
        ) as tf:
            tf.write(
                textwrap.dedent("""\
                    [enabled]
                    keywords = false
                    id3_tags = true
                    parenthetical = false
                    pure_numbers = false

                    [keywords]
                    items = []

                    [id3_tags]
                    tags = ["x|ti", "dot.me"]

                    [parenthetical]
                    markers = []

                    [custom]
                    patterns = []
                """)
            )
            tf.flush()
            f2 = MetadataFilter.from_file(tf.name)

        assert f2.is_metadata("[x|ti: val]")
        assert f2.is_metadata("[dot.me: val]")
        assert not f2.is_metadata("[ti: unexpected]")
        assert not f2.is_metadata("[dotXme: val]")


# ===========================================================================
# from_file() — TOML 配置文件加载
# ===========================================================================


class TestFromFile:
    """MetadataFilter.from_file() 测试。"""

    @pytest.fixture
    def tmp_toml(self, tmp_path: Path) -> Path:
        """返回临时目录，方便各测试写入自己的 toml。"""
        return tmp_path

    def test_default_toml_keywords_only(self, tmp_toml: Path) -> None:
        """默认 TOML（仅 keywords enabled）。"""
        toml_path = tmp_toml / "filter.toml"
        toml_path.write_text(
            textwrap.dedent("""\
                [keywords]
                items = ["作词", "作曲", "编曲"]

                [custom]
                patterns = []
            """),
            encoding="utf-8",
        )
        f = MetadataFilter.from_file(toml_path)
        assert f.is_metadata("作词: Hiro")
        assert f.is_metadata("作曲: Bob")
        assert not f.is_metadata("普通歌词")
        assert not f.is_metadata("演唱: Alice")  # 不在列表中

    def test_enable_all_strategies(self, tmp_toml: Path) -> None:
        """所有策略开启。"""
        toml_path = tmp_toml / "all.toml"
        toml_path.write_text(
            textwrap.dedent("""\
                [enabled]
                keywords = true
                id3_tags = true
                parenthetical = true
                pure_numbers = true

                [keywords]
                items = ["作词"]

                [id3_tags]
                tags = ["ti", "ar"]

                [parenthetical]
                markers = ["间奏"]

                [custom]
                patterns = ["^【.*】$"]
            """),
            encoding="utf-8",
        )
        f = MetadataFilter.from_file(toml_path)
        assert f.is_metadata("作词: A")  # keywords
        assert f.is_metadata("[ti:Title]")  # ID3
        assert f.is_metadata("(间奏)")  # paren
        assert f.is_metadata("42")  # numbers
        assert f.is_metadata("【Verse】")  # custom
        assert not f.is_metadata("real lyric")

    def test_keywords_disabled(self, tmp_toml: Path) -> None:
        """关闭 keywords 后不应匹配。"""
        toml_path = tmp_toml / "nokw.toml"
        toml_path.write_text(
            textwrap.dedent("""\
                [enabled]
                keywords = false

                [custom]
                patterns = []
            """),
            encoding="utf-8",
        )
        f = MetadataFilter.from_file(toml_path)
        assert not f.is_metadata("作词: Hiro")

    def test_custom_keywords(self, tmp_toml: Path) -> None:
        """显式指定 keywords items。"""
        toml_path = tmp_toml / "custom_kw.toml"
        toml_path.write_text(
            textwrap.dedent("""\
                [keywords]
                items = ["总监制", "执行制作"]

                [custom]
                patterns = []
            """),
            encoding="utf-8",
        )
        f = MetadataFilter.from_file(toml_path)
        assert f.is_metadata("总监制: Alice")
        assert f.is_metadata("执行制作: Bob")
        assert not f.is_metadata("作词: Hiro")
        assert not f.is_metadata("作曲: Charlie")

    def test_custom_id3_tags(self, tmp_toml: Path) -> None:
        """显式指定 ID3 tags。"""
        toml_path = tmp_toml / "custom_id3.toml"
        toml_path.write_text(
            textwrap.dedent("""\
                [enabled]
                keywords = false
                id3_tags = true

                [keywords]
                items = []

                [id3_tags]
                tags = ["custom_tag"]

                [parenthetical]
                markers = []

                [custom]
                patterns = []
            """),
            encoding="utf-8",
        )
        f = MetadataFilter.from_file(toml_path)
        assert f.is_metadata("[custom_tag: val]")
        assert not f.is_metadata("[ti:Title]")

    def test_custom_parenthetical_markers(self, tmp_toml: Path) -> None:
        """显式指定 parenthetical markers。"""
        toml_path = tmp_toml / "custom_paren.toml"
        toml_path.write_text(
            textwrap.dedent("""\
                [enabled]
                keywords = false
                parenthetical = true

                [keywords]
                items = []

                [id3_tags]
                tags = []

                [parenthetical]
                markers = ["Special Section"]

                [custom]
                patterns = []
            """),
            encoding="utf-8",
        )
        f = MetadataFilter.from_file(toml_path)
        assert f.is_metadata("(Special Section)")
        assert not f.is_metadata("(Interlude)")

    def test_missing_sections_default_to_disabled(self, tmp_toml: Path) -> None:
        """缺失的 section 默认为 disabled。"""
        toml_path = tmp_toml / "minimal.toml"
        toml_path.write_text(
            textwrap.dedent("""\
                [keywords]
                items = ["作词"]

                [custom]
                patterns = []
            """),
            encoding="utf-8",
        )
        f = MetadataFilter.from_file(toml_path)
        assert f.is_metadata("作词: A")
        assert not f.is_metadata("[ti:X]")
        assert not f.is_metadata("(Prelude)")
        assert not f.is_metadata("5")
