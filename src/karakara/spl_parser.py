from __future__ import annotations

from enum import Enum, auto
from logging import getLogger
from typing import Any

import regex
from pydantic import BaseModel, Field
from typing_extensions import TypeAlias, TypeIs

logger = getLogger()

_REGEX_PATTERN_CACHE: dict[str, regex.Pattern[str]] = {}


def compile_regex(r: str):
    if r not in _REGEX_PATTERN_CACHE:
        _REGEX_PATTERN_CACHE[r] = regex.compile(r, flags=regex.VERBOSE | regex.UNICODE)
    return _REGEX_PATTERN_CACHE[r]


compile_regex(
    TIMETAG_REGEX := r"""    
    \[
        \s*
            (?P<min>\d{1,4})  
        \s*       
        :  
        \s*
            (?P<sec>\d{1,2})   
        \s*
        (?:
            [:\.]
            \s*
                (?P<tail>\d{1,6}) 
            \s*
        )?
    \]
"""
)
compile_regex(
    WORD_TIMETAG_REGEX := TIMETAG_REGEX.replace(r"\[", r"\<").replace(r"\]", r"\>")
)
compile_regex(
    TIMETAG_REGEX_STRICT := r"""
    \[
        (?P<min>\d{1,4})
        :
        (?P<sec>\d{1,2})
        \.
        (?P<tail>\d{1,3})
    \]
"""
)
compile_regex(
    METATAG_REGEX := r"""
    \[
        \s*
        (?P<key>[a-zA-Z]{2,5})
        \s*
        :
        \s*
        (?P<value>.+?)
        \s*
    \]
"""
)
compile_regex(GENERIC_TIMETAG_REGEX := f"{TIMETAG_REGEX} | {WORD_TIMETAG_REGEX}")


class LyricsParserError(Exception):
    pass


class InvalidLyricsError(LyricsParserError):
    pass


def validate_timetag_strict(s: str) -> None | int:
    ma = compile_regex(rf"^{TIMETAG_REGEX_STRICT}$").match(s)
    return match2ms(ma) if ma else None


def match2ms(match: regex.Match[str]) -> int:
    match_dict = match.groupdict()
    mi = int(match_dict.get("min", "0"))
    sec = int(match_dict.get("sec", "0"))
    if tail := match_dict.get("tail"):
        ms = int(int(tail) * (10 ** (3 - len(tail))))
    else:
        ms = 0
    return int(ms + sec * 1000 + mi * 60 * 1000)


class LyricLineType(Enum):
    # just a list of situations, not intended for use
    EMPTY = auto()
    # ``
    # literally empty
    PURE_TEXT = auto()
    # `如果时间逃走了 还有谁会记得我`
    # no time tag found, regarded as reference line
    BYLINE = auto()
    # `[00:47.07]如果时间逃走了 还有谁会记得我[00:51.90]`
    #  ^^^^^^^^^^ start time tag         ^^^^^^^^^^ end time tag
    # ellipsis of end time tag is ok
    # ellipsis of start time tag is also ok, but regarded as reference line
    BYWORD = auto()
    # `[00:47.07]如果时间逃走了<00:49.45>还有谁会记得我[00:51.90]`
    #  ^^^^^^^^^^ by-line   ^^^^^^^^^^ by-word timetag
    # use `[...]` to wrap by-word time tag is okay, but should give warning
    # `<...><...>` is also a valid word, but content is ""


class LyricWord(BaseModel):
    content: str = ""
    start: int | None = None
    end: int | None = None


BasicLyricLine: TypeAlias = list[LyricWord]


class LyricLine(BaseModel):
    content: BasicLyricLine = Field(default_factory=list)
    reference_lines: list[BasicLyricLine] = Field(default_factory=list)


class Lyrics(BaseModel):
    lyrics: list[LyricLine] = Field(default_factory=list)
    metadata: dict[str, str] = Field(default_factory=dict)


def split_to_sequence(
    pattern: str | regex.Pattern[str], text: str
) -> list[str | regex.Match[str]]:
    if isinstance(pattern, str):
        pattern = compile_regex(pattern)
    result = []
    last_index = 0

    for ma in pattern.finditer(text):
        # 上一个匹配结束到当前匹配开始之间
        # 如果 match.start() > last_index, 说明中间有普通文本
        # if ma.start() > last_index:
        # 小巧思: 如果把空文本也算上的话, 那么生成序列就必然是 文本-匹配-文本 的交替出现
        # 这样后续的状态管理能省很多事
        result.append(text[last_index : ma.start()])

        result.append(ma)
        last_index = ma.end()

    # 处理最后一个匹配项到字符串末尾
    # 如果 last_index 小于字符串总长度, 说明末尾还有文本
    # if last_index < len(text):
    result.append(text[last_index:])
    # 不做判定, 这样一来序列的首尾都会是字符串

    return result


def parse_line(line: str) -> BasicLyricLine:
    # 因为折叠行的特性将在上层预处理, 所以我们可以将行内的所有时间标签一视同仁
    seq = split_to_sequence(
        GENERIC_TIMETAG_REGEX,
        line,
    )
    if not seq:
        return []

    if len(seq) % 2 != 1:
        raise LyricsParserError(
            f"未预料的情况: 匹配序列的长度预期为奇数, 而不是: {len(seq)}"
        )
    # 一些特殊情况
    if len(seq) == 1:
        if not is_str(seq[0]):
            raise LyricsParserError(
                f"未预料的情况: 匹配序列长度为 1 时, 其中的唯一元素的类型预期为 str, 而不是 {type(seq[0])!r}"
            )
        return [LyricWord(content=seq[0])]

    result: BasicLyricLine = []
    for idx in range(0, len(seq), 2):
        text = seq[idx]
        if not is_str(text):
            raise LyricsParserError(
                f"未预料的情况: 匹配序列在 [{idx}] 处的元素的类型预期为 str, 而不是 {type(text)}"
            )
        word = LyricWord(content=text)
        if idx > 0:
            if not is_match((t := seq[idx - 1])):
                raise LyricsParserError(
                    f"未预料的情况: 匹配序列在 [{idx - 1}] 处的元素的类型预期为 Match, 而不是 {type(text)}"
                )
            word.start = match2ms(t)
        if idx < len(seq) - 1:
            if not is_match((t := seq[idx + 1])):
                raise LyricsParserError(
                    f"未预料的情况: 匹配序列在 [{idx + 1}] 处的元素的类型预期为 Match, 而不是 {type(text)}"
                )
            word.end = match2ms(t)
        result.append(word)

    if len(result) < 2:
        raise LyricsParserError(
            f"未预料的情况: 预处理结果序列的预期长度 >= 2, 而不是 {len(result)}"
        )

    # pop 掉空的首尾, 这样 [0] 的 start 就是整行的 start, [-1] 同理
    if not result[0].content:
        result.pop(0)
    if not result[-1].content and len(result) > 1:
        result.pop(-1)
    return result


def is_match(obj: Any) -> TypeIs[regex.Match]:
    return isinstance(obj, regex.Match)


def is_str(obj: Any) -> TypeIs[str]:
    return isinstance(obj, str)


def parse_file(lrc: str) -> Lyrics:
    lines: list[LyricLine] = []
    current: LyricLine | None = None
    for raw_line in lrc.strip().splitlines():
        line_str = raw_line.strip()
        line = parse_line(line_str)

        if line is None:
            if current:
                lines.append(current)
            current = None
        if line:
            if current is None:
                current = LyricLine(content=line)
            else:
                current.reference_lines.append(line)

    if current is not None:
        lines.append(current)

    raise NotImplementedError


def construct_lrc(lyrics: Lyrics):
    pass
