from __future__ import annotations

from enum import Enum
from functools import lru_cache
from typing import Any

import regex
from pydantic import BaseModel, Field

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
                (?P<ms>\d{1,6}) 
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
        (?P<min>\d{1,3})
        :
        (?P<sec>\d{1,2})
        \.
        (?P<cs>\d{2})
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
    if ma is None:
        return None
    mi = int(ma.group("min"))
    sec = int(ma.group("sec"))
    cs = int(ma.group("cs"))
    return cs * 10 + sec * 1000 + mi * 60 * 1000


def match_result_to_ms(ma: dict[str, Any]) -> int:
    mi = int(ma.get("min", "0"))
    sec = int(ma.get("sec", "0"))
    tail = ma.get("ms")
    if tail is None:
        ms = 0
    else:
        ms = int(int(tail) * (10 ** (3 - len(tail))))
    return ms + sec * 1000 + mi * 60 * 1000


class LyricWord(BaseModel):
    content: str
    start: int
    end: int | None = None


class BasicLyricLine(BaseModel):
    content: str
    words: list[LyricWord] | None = None


class LyricLine(BasicLyricLine):
    start: int  # ms
    end: int | None = None  # ms
    reference_lines: list[BasicLyricLine] = Field(default_factory=list)


class Lyrics(BaseModel):
    lyrics: list[LyricLine] = Field(default_factory=list)
    metadata: dict[str, str] = Field(default_factory=dict)


def _parse_line(line: str) -> tuple[int | None, int | None, list[LyricWord]]:
    start = None
    end = None
    if m := compile_regex(TIMETAG_REGEX).match(line):
        start = match_result_to_ms(m.groupdict())
    if m := compile_regex(TIMETAG_REGEX + r"$").search(line):
        end = match_result_to_ms(m.groupdict())
    # TODO:
    return start, end, []


def parse_lrc(lrc: str):
    raw_lines = list(map(lambda s: s.strip(), lrc.splitlines(keepends=False)))
    semi_result: list[LyricLine | None] = [None]

    for raw_line in raw_lines:
        if m := compile_regex(TIMETAG_REGEX).match(raw_line):
            pass
        elif semi_result[-1] is not None:
            semi_result[-1].reference_lines.append(BasicLyricLine())
