"""`karakara.gui.lrctext` 的已知答案测试。

这一层决定「原地写回」会不会毁掉用户的歌词文件，所以断言要钉死两件事：
**只有时间戳的数字变了**，以及**小数位数不会被悄悄砍掉精度**。
"""

from __future__ import annotations

import pytest

from karakara.gui.lrctext import iter_timetags, shift_lrc_text

SAMPLE = """[ti:测试]
[ar:某人]
[00:01.500]第一行
[00:12.000][00:20.000]重复标签行
<00:30.25>逐字标签行
[00:40]整秒行
"""


def test_only_timestamps_change() -> None:
    result = shift_lrc_text(SAMPLE, 1000)

    assert result.tags == 5
    assert (
        result.text
        == """[ti:测试]
[ar:某人]
[00:02.500]第一行
[00:13.000][00:21.000]重复标签行
<00:31.25>逐字标签行
[00:41]整秒行
"""
    )


def test_metadata_tags_are_untouched() -> None:
    """`[ti:]`/`[ar:]`/`[offset:]` 不是时间，绝不能被平移。"""
    text = "[offset:+250]\n[ti:歌名]\n[00:01.000]词\n"

    result = shift_lrc_text(text, 500)

    assert "[offset:+250]" in result.text
    assert "[ti:歌名]" in result.text
    assert "[00:01.500]词" in result.text
    assert result.tags == 1


def test_fraction_digits_are_preserved_when_precise_enough() -> None:
    """整秒平移不该改变原文件的书写风格。"""
    result = shift_lrc_text("[00:01]a\n[00:02.50]b\n", 1000)
    assert result.text == "[00:02]a\n[00:03.50]b\n"


def test_fraction_digits_grow_when_needed_to_keep_precision() -> None:
    """原位数不够表示新值时必须加位，不能四舍五入丢掉 20ms。

    1620ms 用 2 位小数就能精确表示，所以只加到位数够用为止（不是无脑补到 3 位）。
    """
    result = shift_lrc_text("[00:01.5]a\n", 120)
    assert result.text == "[00:01.62]a\n"
    assert result.changes == [(1500, 1620)]


def test_fraction_digits_go_to_three_when_only_three_can_express_it() -> None:
    """平移量落在毫秒位上时，必须补到 3 位小数。"""
    result = shift_lrc_text("[00:01.5]a\n", 123)
    assert result.text == "[00:01.623]a\n"


def test_colon_fraction_separator_is_supported() -> None:
    result = shift_lrc_text("[00:01:25]a\n", 1000)
    assert result.text == "[00:02.25]a\n"


def test_negative_result_is_clamped_and_counted() -> None:
    result = shift_lrc_text("[00:01.000]a\n[00:05.000]b\n", -2000)

    assert result.clamped == 1
    assert result.text == "[00:00.000]a\n[00:03.000]b\n"
    assert result.min_ms == 1000


def test_refusing_to_clamp_raises_with_the_minimum_delta() -> None:
    with pytest.raises(ValueError, match=r"最小的可用偏移是 -1000ms"):
        shift_lrc_text("[00:01.000]a\n", -2000, clamp_at_zero=False)


def test_zero_delta_is_a_pure_no_op() -> None:
    result = shift_lrc_text(SAMPLE, 0)

    assert result.text == SAMPLE
    assert result.changed is False
    assert result.tags == 5


def test_text_without_tags_is_returned_unchanged() -> None:
    result = shift_lrc_text("没有时间标签\n", 1000)

    assert result.text == "没有时间标签\n"
    assert result.tags == 0
    assert result.min_ms is None


def test_mismatched_brackets_are_not_treated_as_timestamps() -> None:
    """`[00:01.000>` 这种畸形标签不该被改写（宁可不动，也不要造出更怪的东西）。"""
    result = shift_lrc_text("[00:01.000>a\n", 1000)
    assert result.text == "[00:01.000>a\n"
    assert result.tags == 0


def test_crlf_is_preserved() -> None:
    """原地写回必须保住原文件的换行风格（Windows 上的歌词文件常见 CRLF）。"""
    result = shift_lrc_text("[00:01.000]a\r\n[00:02.000]b\r\n", 500)
    assert result.text == "[00:01.500]a\r\n[00:02.500]b\r\n"


def test_iter_timetags_reports_spans_and_digits() -> None:
    tags = list(iter_timetags("[00:01.5]x[01:02:25]y"))

    assert [(tag.ms, tag.digits) for tag in tags] == [(1500, 1), (62250, 2)]
    assert tags[0].span == (0, 9)
    assert tags[1].span == (10, 20)


def test_minute_field_can_grow_past_two_digits() -> None:
    result = shift_lrc_text("[59:30.000]a\n", 60_000)
    assert result.text == "[60:30.000]a\n"
