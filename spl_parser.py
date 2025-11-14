from __future__ import annotations

from enum import Enum, auto
from typing import Any, Generic, NamedTuple, TypedDict, TypeVar, overload

import regex
from pydantic import BaseModel, Field
from typing_extensions import TypeIs

T = TypeVar("T")
TV = TypeVar("TV")
TVV = TypeVar("TVV")
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
    ^
    \[
        (?P<min>\d{1,4})
        :
        (?P<sec>\d{1,2})
        \.
        (?P<tail>\d{1,3})
    \]
    $
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


class InvalidLyricsError(Exception):
    pass


def validate_timetag_strict(s: str) -> None | int:
    ma = compile_regex(TIMETAG_REGEX_STRICT).match(s)
    return match_result_to_ms(ma) if ma else None


def match_result_to_ms(match: regex.Match[str]) -> int:
    match_dict = match.groupdict()
    mi = int(match_dict.get("min", "0"))
    sec = int(match_dict.get("sec", "0"))
    if tail := match_dict.get("tail"):
        ms = int(int(tail) * (10 ** (3 - len(tail))))
    else:
        ms = 0
    return int(ms + sec * 1000 + mi * 60 * 1000)


class LyricLineType(Enum):
    EMPTY = auto()  # ``
    NO_TEXT = auto()  # `[00:00.00]`
    PURE_TEXT = auto()  # `如果时间逃走了 还有谁会记得我`
    BYLINE = auto()  # `[00:47.07]如果时间逃走了 还有谁会记得我`
    BYWORD = auto()  # `[00:47.07]如果时间逃走了<00:49.45>还有谁会记得我[00:51.90]`


class LyricWord(BaseModel):
    content: str = ""
    start: int | None = None
    end: int | None = None


class BasicLyricLine(BaseModel):
    start: int | None = None  # ms
    end: int | None = None  # ms
    words: list[LyricWord] = Field(default_factory=list)


class LyricLine(BasicLyricLine):
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
        if ma.start() > last_index:
            result.append(text[last_index : ma.start()])

        result.append(ma)
        last_index = ma.end()

    # 处理最后一个匹配项到字符串末尾
    # 如果 last_index 小于字符串总长度, 说明末尾还有文本
    if last_index < len(text):
        result.append(text[last_index:])

    return result


def _match_instance_type(objs: tuple[object, ...], types: tuple[type, ...]) -> bool:
    if len(objs) != len(types):
        return False
    return all([isinstance(o, t) for o, t in zip(objs, types)])


def _parse_line(line: str) -> tuple[BasicLyricLine, LyricLineType]:
    """
    start_time, words, type
    """

    start = None
    end = None
    seq: list[str | regex.Match[str]] = split_to_sequence(
        compile_regex(f"{TIMETAG_REGEX} | {WORD_TIMETAG_REGEX}"),
        line.strip(),
    )

    if not seq:
        return BasicLyricLine(), LyricLineType.EMPTY
    elif len(seq) == 1:
        item = seq[0]
        if isinstance(item, str):
            return BasicLyricLine(
                words=[LyricWord(content=item)]
            ), LyricLineType.PURE_TEXT
        else:
            return BasicLyricLine(), LyricLineType.NO_TEXT
    elif len(seq) == 2:
        a, b = seq
        if isinstance(a, regex.Match) and isinstance(b, str):
            return BasicLyricLine(
                start=match_result_to_ms(a), words=[LyricWord(content=b)]
            ), LyricLineType.BYLINE
        raise InvalidLyricsError(f"invalid lyric sequence: {seq!r}")

    result = BasicLyricLine()

    for idx, obj in enumerate(seq):
        pass

    return result, LyricLineType.BYWORD


def parse_lrc(lrc: str):
    raw_lines = lrc.splitlines()
    semi_result: list[LyricLine | None] = [None]

    for raw_line in raw_lines:
        pass
