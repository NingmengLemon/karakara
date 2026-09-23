"""LRC 文本的时间戳平移：原地改数，不动别的。

为什么不用「解析成 ``Lyrics`` 再序列化」：那是**产出**新歌词的路径（本项目生成
``*.kara.lrc`` 用的就是它），它会规范化格式、重排 metadata、丢掉解析器不认识的东西。
而 GUI 要写回的是**用户自己的歌词文件**，里面可能有翻译行、特殊标签、奇怪的空白与
CRLF。原地写回必须只改时间戳的数字，其余一个字节都不动。

因此这里做的是**文本级**替换：找出所有真正的时间标签（``[mm:ss.xxx]`` / ``<mm:ss.xxx>``），
把数值加上 delta 再按原位写回去。

不碰的东西：``[ti:...]`` / ``[ar:...]`` / ``[offset:...]`` 这类元数据标签（它们不是时间），
以及所有非标签文本。
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, field

#: 时间标签：``[mm:ss]`` / ``[mm:ss.xxx]`` / ``[mm:ss:xx]`` / ``<mm:ss.xxx>``。
#: 开闭括号分别捕获，代码里要求配对；``ti:`` 这类因分钟位不是数字而天然不匹配。
_TIMETAG = re.compile(
    r"(?P<open>[\[<])(?P<minute>\d{1,3}):(?P<second>\d{1,2})"
    r"(?:(?P<sep>[.:])(?P<frac>\d{1,3}))?(?P<close>[\]>])"
)

#: 分数位的最大位数（毫秒精度）。
_MAX_DIGITS = 3


@dataclass(frozen=True)
class TimeTag:
    """文本里的一个时间标签。"""

    #: 在原文中的区间 ``[start, end)``。
    span: tuple[int, int]
    #: 解析出来的毫秒值。
    ms: int
    #: 原本的分数位数（``[00:01]`` 是 0，``[00:01.5]`` 是 1）。
    digits: int


@dataclass
class ShiftResult:
    """平移结果。"""

    text: str
    tags: int = 0
    clamped: int = 0
    delta_ms: int = 0
    #: 原文里最小的时间戳（ms），没有标签时为 ``None``。
    min_ms: int | None = None
    #: 逐标签的 ``(原文值, 新值)``，供界面预览。
    changes: list[tuple[int, int]] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        """是否有任何标签的值真的变了。"""
        return any(old != new for old, new in self.changes)


def iter_timetags(text: str) -> Iterator[TimeTag]:
    """按出现顺序产出文本里所有合法的时间标签。"""
    for match in _TIMETAG.finditer(text):
        if not _brackets_match(match.group("open"), match.group("close")):
            continue
        yield TimeTag(
            span=match.span(),
            ms=_to_ms(
                match.group("minute"), match.group("second"), match.group("frac")
            ),
            digits=len(match.group("frac") or ""),
        )


def shift_lrc_text(
    text: str, delta_ms: int, *, clamp_at_zero: bool = True
) -> ShiftResult:
    """把文本里所有时间标签加上 ``delta_ms``，其余内容原样保留。

    Args:
        text: LRC 原文。
        delta_ms: 平移量（毫秒）。正值表示歌词偏早、需要延后。
        clamp_at_zero: 平移后为负的时间戳夹到 0（默认）。关掉时保留负值，交给调用方
            自己处理（本项目的产物路径不允许负值，见 ``core._clamp_negative_timestamps``）。

    Returns:
        平移后的文本与统计。``clamped`` 是**被夹到 0 的标签个数**，不是行数。
    """
    if delta_ms == 0:
        tags = list(iter_timetags(text))
        return ShiftResult(
            text=text,
            tags=len(tags),
            min_ms=min((tag.ms for tag in tags), default=None),
            changes=[(tag.ms, tag.ms) for tag in tags],
        )

    pieces: list[str] = []
    cursor = 0
    result = ShiftResult(text="", delta_ms=delta_ms)

    for tag in iter_timetags(text):
        start, end = tag.span
        if start < cursor:  # 理论上不会发生（finditer 不重叠），保险
            continue
        shifted = tag.ms + delta_ms
        if shifted < 0:
            if not clamp_at_zero:
                raise ValueError(
                    f"偏移 {delta_ms:+d}ms 会让 {text[start:end]} 变成负时间戳；"
                    f"对这份歌词最小的可用偏移是 {-tag.ms:+d}ms"
                )
            shifted = 0
            result.clamped += 1

        pieces.append(text[cursor:start])
        pieces.append(_format_tag(text[start], text[end - 1], shifted, tag.digits))
        cursor = end

        result.tags += 1
        result.min_ms = tag.ms if result.min_ms is None else min(result.min_ms, tag.ms)
        result.changes.append((tag.ms, shifted))

    pieces.append(text[cursor:])
    result.text = "".join(pieces)
    return result


def _brackets_match(open_bracket: str, close_bracket: str) -> bool:
    return (open_bracket, close_bracket) in (("[", "]"), ("<", ">"))


def _to_ms(minute: str, second: str, frac: str | None) -> int:
    """把标签的三段解析成毫秒。分数位按**左对齐**补齐到 3 位（``.5`` 是 500ms）。"""
    milliseconds = int(minute) * 60_000 + int(second) * 1_000
    if frac:
        milliseconds += int(frac.ljust(_MAX_DIGITS, "0"))
    return milliseconds


def _needed_digits(ms: int) -> int:
    """表示 ``ms`` 至少需要几位小数。"""
    if ms % 1_000 == 0:
        return 0
    if ms % 100 == 0:
        return 1
    if ms % 10 == 0:
        return 2
    return _MAX_DIGITS


def _format_tag(
    open_bracket: str, close_bracket: str, ms: int, original_digits: int
) -> str:
    """按原标签的括号与**足够表示新值**的小数位数重新格式化。

    刻意保留原文件的小数位数风格（``[00:01]`` 就还是 0 位），只在原位数不足以表示新值
    时才加位——平移 120ms 却把 ``[00:01.5]`` 写成 ``[00:01.6]`` 会悄悄丢掉 20ms。
    """
    digits = max(original_digits, _needed_digits(ms))
    minutes, remainder = divmod(ms, 60_000)
    seconds, milliseconds = divmod(remainder, 1_000)
    if digits == 0:
        body = f"{minutes:02d}:{seconds:02d}"
    else:
        tail = f"{milliseconds:03d}"[:digits]
        body = f"{minutes:02d}:{seconds:02d}.{tail}"
    return f"{open_bracket}{body}{close_bracket}"
