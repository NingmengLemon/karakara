"""零长度词的处理：统计、可选细分、以及产物里不出现重复时间标签。

背景见 :mod:`karakara.aligner.postprocess`：对齐器的边界量化到 80ms
（模型 config 的 ``timestamp_segment_time = 80``），时长不足一帧的单元会拿到
``start == end``；实测 6 首歌 1497 个单元里 337 个（22.5%）如此，最高一首 41.8%。

这里盯三件事：

1. :func:`refine_collapsed_words` 的三条不变量（不改动被报告过的边界、不越过下一个
   正常单元、不摊出比本行典型单元还长）；
2. ``_merge_zero_length_tokens`` 把零长度 token 的文本并进相邻 token —— 文本一个字符
   都不能丢；
3. **序列化后的产物里没有连续重复的时间标签**（这是"零长度"在文件层面的唯一表现）。
"""

from __future__ import annotations

import itertools
import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from lemony_lrc_parser import Lyrics, LyricToken, SerializationOptions

from karakara.aligner.abc import AbstractAligner, AlignedWord, LangCode
from karakara.aligner.postprocess import (
    DEGENERATE_RATIO_WARN,
    count_zero_length,
    refine_collapsed_words,
)
from karakara.core import _merge_zero_length_tokens, gen_kara
from karakara.separator.abc import AbstractStemSeparator
from karakara.typ import NpAudioData, NpAudioSamples
from karakara.utils.io import DEFAULT_SAMPLE_RATE
from karakara.utils.metadata import MetadataFilter

SERIALIZE = SerializationOptions(
    use_bracket_for_byword_tag=True,
    line_tag_decimal_length=3,
    word_tag_decimal_length=3,
)
TAG = re.compile(r"\[(\d+):(\d+)\.(\d+)\]")


def words(*specs: tuple[str, int, int]) -> list[AlignedWord]:
    return [AlignedWord(word=text, position=(start, end)) for text, start, end in specs]


def span(word: AlignedWord) -> tuple[int, int]:
    """取一个单元的时间区间，顺便把 ``None`` 断言掉（类型检查器也看得懂）。"""
    assert word.position is not None
    return word.position


def duplicated_tags(text: str) -> list[tuple[int, int]]:
    """返回所有「连续两个相同时间标签」的位置（毫秒值对）。"""
    stamps = [
        int(m.group(1)) * 60_000 + int(m.group(2)) * 1000 + int(m.group(3))
        for m in TAG.finditer(text)
    ]
    return [
        (previous, current)
        for previous, current in itertools.pairwise(stamps)
        if previous == current
    ]


def plain_text(text: str) -> str:
    """去掉全部时间标签与换行，得到播放器真正渲染出来的文本。"""
    return TAG.sub("", text).replace("\n", "").strip()


# --------------------------------------------------------------------------
# 统计
# --------------------------------------------------------------------------


def test_count_zero_length() -> None:
    stats = count_zero_length(
        words(("a", 0, 100), ("b", 100, 100), ("c", 100, 200), ("d", 200, 200))
    )

    assert stats.units == 4
    assert stats.zero_length == 2
    assert stats.ratio == 0.5


def test_count_zero_length_ignores_units_without_position() -> None:
    stats = count_zero_length(
        [AlignedWord(word="x"), AlignedWord(word="y", position=(0, 0))]
    )

    assert stats.units == 1
    assert stats.zero_length == 1


# --------------------------------------------------------------------------
# 细分：三条不变量
# --------------------------------------------------------------------------


def test_refine_borrows_from_the_following_gap() -> None:
    refined, count = refine_collapsed_words(
        words(("a", 0, 200), ("b", 200, 200), ("c", 800, 1000)),
        total_ms=1000,
    )

    assert count == 1
    assert [w.position for w in refined] == [(0, 200), (200, 400), (800, 1000)]


def test_refine_never_touches_reported_boundaries() -> None:
    """正常单元的时间必须原封不动（只是折叠的那个拿到时长）。"""
    original = words(("a", 0, 200), ("b", 200, 200), ("c", 800, 1000))
    refined, _ = refine_collapsed_words(original, total_ms=1000)

    assert refined[0].position == original[0].position
    assert refined[2].position == original[2].position


def test_refine_does_not_pass_the_next_unit() -> None:
    refined, _ = refine_collapsed_words(
        words(("a", 0, 200), ("b", 200, 200), ("c", 260, 400)), total_ms=1000
    )

    start, end = span(refined[1])
    assert 200 <= start < end <= 260


def test_refine_is_capped_by_the_lines_median_duration() -> None:
    """空隙很大时也不能摊出比这一行典型单元还长的时长。"""
    refined, _ = refine_collapsed_words(
        words(("a", 0, 200), ("b", 200, 200), ("c", 800, 1000)), total_ms=5000
    )

    # 正常跨度的中位数是 200ms → 折叠单元拿到的不能超过 200ms
    assert refined[1].position == (200, 400)


def test_refine_explicit_cap_wins() -> None:
    refined, _ = refine_collapsed_words(
        words(("a", 0, 200), ("b", 200, 200), ("c", 800, 1000)),
        total_ms=5000,
        max_duration_ms=50,
    )

    assert refined[1].position == (200, 250)


def test_refine_splits_a_run_by_text_length() -> None:
    """多词折叠段按文本长度加权分配（长词占更长区间）。"""
    refined, count = refine_collapsed_words(
        words(("a", 0, 100), ("bbbb", 100, 100), ("c", 100, 100), ("d", 500, 600)),
        total_ms=1000,
    )

    assert count == 2
    long_start, long_end = span(refined[1])
    short_start, short_end = span(refined[2])
    assert long_end - long_start > short_end - short_start
    assert long_start >= 100 and short_end <= 500


def test_refine_uses_the_segment_end_for_a_line_final_run() -> None:
    refined, count = refine_collapsed_words(
        words(("a", 0, 200), ("b", 200, 200)), total_ms=600
    )

    assert count == 1
    assert refined[1].position == (200, 400)  # 上限 200ms（本行中位跨度）


def test_refine_leaves_collapsed_when_there_is_no_gap() -> None:
    """后面紧跟一个同刻开始的正常单元时无空隙可分 —— 原样保留，交给合并。"""
    refined, count = refine_collapsed_words(
        words(("a", 0, 200), ("b", 200, 200), ("c", 200, 400)), total_ms=1000
    )

    assert count == 0
    assert refined[1].position == (200, 200)


def test_refine_keeps_non_zero_length_units_untouched() -> None:
    original = words(("a", 0, 100), ("b", 100, 300))
    refined, count = refine_collapsed_words(original, total_ms=1000)

    assert count == 0
    assert refined == original


# --------------------------------------------------------------------------
# 合并：文本一个字符都不能丢
# --------------------------------------------------------------------------


def test_merge_zero_length_token_into_the_previous_one() -> None:
    tokens = [
        LyricToken(content="a", start=0, end=100),
        LyricToken(content="b", start=100, end=100),  # 零长度
        LyricToken(content="c", start=100, end=200),
    ]

    merged = _merge_zero_length_tokens(tokens)

    assert [t.content for t in merged] == ["ab", "c"]
    assert [t.start for t in merged] == [0, 100]


def test_merge_first_token_into_the_following_one() -> None:
    tokens = [
        LyricToken(content="a", start=0, end=0),  # 行首零长度
        LyricToken(content="b", start=0, end=100),
    ]

    merged = _merge_zero_length_tokens(tokens)

    assert [t.content for t in merged] == ["ab"]


def test_merge_keeps_a_lone_zero_length_token() -> None:
    tokens = [LyricToken(content="a", start=0, end=0)]

    assert _merge_zero_length_tokens(tokens) == tokens


def test_merge_preserves_all_text() -> None:
    tokens = [
        LyricToken(content="君", start=0, end=100),
        LyricToken(content="が", start=100, end=100),
        LyricToken(content="好き", start=100, end=400),
        LyricToken(content="だ", start=400, end=400),
        LyricToken(content="から", start=400, end=700),
    ]

    merged = _merge_zero_length_tokens(tokens)

    assert "".join(t.content for t in merged) == "君が好きだから"


def test_merge_is_a_no_op_without_zero_length_tokens() -> None:
    tokens = [
        LyricToken(content="a", start=0, end=100),
        LyricToken(content="b", start=100, end=200),
    ]

    assert _merge_zero_length_tokens(tokens) is tokens


# --------------------------------------------------------------------------
# 端到端：产物里不能有连续重复的时间标签
# --------------------------------------------------------------------------


class ZeroLengthAligner(AbstractAligner):
    """对每一行都返回「词 + 零长度词 + 词」的对齐器替身。"""

    def align(
        self,
        audio: NpAudioData | NpAudioSamples,
        text: str,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        *,
        language: LangCode | None = None,
    ) -> list[AlignedWord]:
        return words(
            (text[:1], 0, 200),
            (text[1:2], 200, 200),  # 零长度
            (text[2:], 200, 600),
        )


class NoCollapseAligner(AbstractAligner):
    """所有单元都有正时长。"""

    def align(
        self,
        audio: NpAudioData | NpAudioSamples,
        text: str,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        *,
        language: LangCode | None = None,
    ) -> list[AlignedWord]:
        return words((text[:1], 0, 200), (text[1:], 200, 600))


class FixedSeparator(AbstractStemSeparator):
    """把一段静音写成 32 位浮点 WAV 当作人声轨。"""

    def separate(
        self,
        audio_path: str | Path,
        dest_dir: str | Path,
        *,
        stems: Sequence[str] | None = None,
    ) -> dict[str, Path]:
        dest = Path(dest_dir)
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / f"{self.VOCAL_STEM_NAME}.wav"
        sf.write(str(path), np.zeros(8000, dtype=np.float32), 8000, subtype="FLOAT")
        return {self.VOCAL_STEM_NAME: path}


def _run(aligner: AbstractAligner, lyrics_text: str, *, refine: bool) -> str:
    result = gen_kara(
        Lyrics.loads(lyrics_text),
        "ignored.wav",
        aligner=aligner,
        separator=FixedSeparator(),
        metadata_filter=MetadataFilter(
            keywords=[],
            id3_tags=frozenset(),
            parenthetical_markers=[],
            detect_id3_tags=False,
            detect_parenthetical=False,
            detect_pure_numbers=False,
            custom_patterns=[],
        ),
        offset_ms=0,
        min_vocal_activity=0,
        refine_collapsed_words=refine,
    )
    return result.dumps(options=SERIALIZE)


def test_product_has_no_duplicate_consecutive_tags() -> None:
    """零长度词合并之后，产物里不应再有连续重复的时间标签。"""
    text = _run(ZeroLengthAligner(), "[00:00.00]あいうえお\n", refine=False)

    assert duplicated_tags(text) == []
    # 文本完整保留（标签会把文本切开，所以按"去掉标签后的纯文本"比较）
    assert plain_text(text) == "あいうえお"


def test_refine_also_leaves_no_duplicate_tags() -> None:
    text = _run(ZeroLengthAligner(), "[00:00.00]あいうえお\n", refine=True)

    assert duplicated_tags(text) == []
    assert plain_text(text) == "あいうえお"


def test_refine_gives_every_unit_a_positive_duration() -> None:
    """打开细分后，产物里每个词级标签区间都应为正（词能被单独高亮）。"""
    text = _run(ZeroLengthAligner(), "[00:00.00]あいうえお\n", refine=True)

    stamps = [
        int(m.group(1)) * 60_000 + int(m.group(2)) * 1000 + int(m.group(3))
        for m in TAG.finditer(text)
    ]
    assert stamps == sorted(stamps)
    assert len(stamps) == len(set(stamps))


def test_zero_length_ratio_warns_above_the_threshold(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """1/3 的单元是零长度 → 高于 30% 阈值，应当告警。"""
    from logging import WARNING

    with caplog.at_level(WARNING):
        _run(ZeroLengthAligner(), "[00:00.00]あいうえお\n", refine=False)

    warnings = [r.message for r in caplog.records if r.levelno == WARNING]
    assert any("zero-length" in message for message in warnings)
    assert any("refine-collapsed-words" in message for message in warnings)


def test_zero_length_ratio_below_the_threshold_only_logs_info(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """没有零长度单元时只记一行 info，不告警。"""
    from logging import INFO, WARNING

    with caplog.at_level(INFO):
        _run(NoCollapseAligner(), "[00:00.00]あいうえお\n", refine=False)

    messages = [r.message for r in caplog.records if "zero-length" in r.message]
    assert messages, "统计信息应当被记录"
    assert not [
        r for r in caplog.records if r.levelno == WARNING and "zero-length" in r.message
    ]


def test_shipped_threshold_is_30_percent() -> None:
    assert DEGENERATE_RATIO_WARN == 0.30
