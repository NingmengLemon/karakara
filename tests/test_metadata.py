"""metadata.py 单元测试。"""

from __future__ import annotations

import re
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

    def test_keyword_and_colon_in_lyric_text(self) -> None:
        """歌词正文中的“关键字: 内容”不能被误判为元数据。"""
        assert not _mk_filter().is_metadata("我想写下作词: 未完的故事")

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


# ===========================================================================
# 仓库自带的那份配置（metadata_filter.toml）
#
# 这些断言把一个真实曲库扫描的结论固化下来：哪些形态必须被抓到，哪些形态
# **绝对不能**抓（假阳性会直接删掉一行歌词，是这套过滤最严重的失败模式）。
# 规模数字来自 6622 个真实 .lrc 的统计。
# ===========================================================================


def repo_filter() -> MetadataFilter:
    path = Path(__file__).resolve().parent.parent / "metadata_filter.toml"
    return MetadataFilter.from_file(path)


class TestShippedConfigCatches:
    """仓库配置必须抓到的形态。"""

    @pytest.mark.parametrize(
        "line",
        [
            # 缩写写法（真实曲库里的漏网主力）
            "曲：シャノン",
            "词：みきとP",
            "歌：GUMI",
            "词曲：COP",
            "词、曲：一二三",
            "译：某人",
            "绘:王刃、唯Tu（封面）",
            "调：坐标P",
            # 职位
            "原唱：乐正绫",
            "人声 : Danny Sweet",
            "和声编写：梁丹郡",
            "混音工程师：张之晨",
            "插画：贝贝-web-",
            "映像：スタジオごはん",
            "発売日：2017 01 18",
            # 乐器
            "吉他 : Phil Solem",
            "贝斯：成元",
            "弦乐：瀚樂集 Han Ensemble",
            # 英文（IGNORECASE）
            "Vocal：Cryu",
            "vocal: someone",
            "Lyrics: Kizuna AI",
            "Music:NceS",
            "Album: Halozy - Starry Presto (C77)",
            "Arrange：KOBATYU",
            # 空值行（`+.` 时期抓不到，见 test_metadata.py 的 .+ → .* 注释）
            "Singer：",
            "Rap:",
            # ID3 标签（过滤器层面必须认；注意管线里这些标签会被解析器先收进
            # lyrics.metadata，根本不会以行的形态到达这里——所以这条断言保护的是
            # MetadataFilter 独立使用时的正确性，不是召回率。）
            "[by: 某人]",
            "[ar:歌手]",
            "[ly: 苍十三 / 阿良良木健]",
            "[total: 274005]",
            # 全角括号的段落标记（本次才支持）
            "（间奏）",
            "【サビ】",
            "(Bridge)",
            # 自定义模式
            "————————————",
            "……",
            "♪♪",
            "End",
            "(END)",
            "終わり",
            "undefined",
            "Vocals by Hannah Crowley",
            "编曲 Arranger：宫奇Gon(HOYO-MiX)",
            "出品 Produced by：HOYO-MiX",
            "（翻译鸣谢：弓野笃祯）",
            # 纯段落词裸行（自定义模式 ④/⑥）
            "Interlude",
            "Instrumental",
            "music",
            "サビ",
            "間奏",
        ],
    )
    def test_is_metadata(self, line: str) -> None:
        assert repo_filter().is_metadata(line), f"应当被判为元数据行: {line!r}"


class TestShippedConfigMustNotCatch:
    """仓库配置**绝对不能**抓到的形态（每一条都有真实命中的歌词作证据）。"""

    @pytest.mark.parametrize(
        ("line", "why"),
        [
            ("合：赔盏茶才算周到", "对唱分句标记，冒号后就是歌词正文（48 行）"),
            ("洛：麻酱韭花对垒 转眼用光满桌调料", "角色分句标记（28 行）"),
            ("雏：才会这样慌不择路急不择途地迷失吗", "角色分句标记（27 行）"),
            ("8:07に君を待ってる", "「时间 + 歌词」，不是时间戳元数据"),
            ("8:00 二号车二节 被占的特等座", "同上"),
            ("_(:з」∠)_", "颜文字歌词"),
            ("By the way, do you like baseball?", "英文歌词里以 By 开头"),
            ("Bye for now", "同上"),
            ("谢谢 想说谢谢你", "「谢谢」开头的歌词"),
            ("谢谢，我吃饱了", "同上"),
            ("Thanks for the meal", "英文歌词"),
            ("Thanks a lot 君のsongが好きそう", "英文歌词"),
            ("**你是什么垃圾？**", "装饰符号开头的歌词"),
            ("****ed get right now", "同上"),
            ("（与你同在）", "整行被括号包裹的歌词（10,163 行 / 1,887 文件）"),
            ("(By your side)", "同上"),
            ("「よっしゃー行くぞ」", "同上"),
            ("1", "报数演唱（コンコンきつね）"),
            ("3", "报数演唱"),
            ("1234", "连打演唱"),
            ("11111111111111111111111111111111", "连打演唱"),
            ("[間奏]", "方括号形态刻意不认，避免与 LRC 时间标签语法打架"),
            ("Music を止めないで", "段落词后面还有正文——整行锚定就是为了不误伤这种"),
            ("Solo で踊ろう", "同上"),
            ("Interlude of my heart", "同上（英文歌词）"),
        ],
    )
    def test_is_not_metadata(self, line: str, why: str) -> None:
        assert not repo_filter().is_metadata(line), f"误判为元数据行（{why}）: {line!r}"


class TestShippedConfigInvariants:
    """配置本身的不变量。"""

    def test_pure_numbers_stays_disabled(self) -> None:
        """纯数字必须保持关闭：真实曲库里有报数/连打演唱。"""
        floater = repo_filter()
        assert not floater.is_metadata("1")
        assert not floater.is_metadata("1234")

    def test_config_patterns_compile_without_bare_inline_flag(self) -> None:
        """自定义正则不能写裸 `(?i)`：所有分支会被 `|` 拼成一条正则。

        解析 TOML 再查（而不是在原文里搜字符串），否则注释里提到 `(?i)` 也会被算进去。
        """
        import tomllib

        path = Path(__file__).resolve().parent.parent / "metadata_filter.toml"
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        patterns: list[str] = list(data.get("custom", {}).get("patterns", []))
        assert patterns, "仓库配置里应当有自定义模式"
        for pattern in patterns:
            # 裸 (?i) = 紧跟的不是冒号（作用域写法是 (?i:...)）
            scoped = pattern.replace("(?i:", "").replace("(?m:", "")
            assert "(?i)" not in scoped, (
                "裸 (?i) 会让 re.compile 抛 "
                "'global flags not at the start of the expression'"
            )
            assert "(?m)" not in scoped
            # 每条都必须真的能编译
            re.compile(pattern)

    def test_shipped_config_still_catches_every_legacy_keyword(self) -> None:
        """新增条目不能把原有 28 个关键字挤掉（零召回损失）。"""
        floater = repo_filter()
        for keyword in _DEFAULT_KEYWORDS:
            assert floater.is_metadata(f"{keyword}: 某人"), keyword
