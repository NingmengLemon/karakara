from __future__ import annotations

from enum import Enum, auto
from logging import getLogger

import regex
from pydantic import BaseModel, Field

logger = getLogger()

_REGEX_PATTERN_CACHE: dict[str, regex.Pattern[str]] = {}


def compile_regex(r: str):
    if r not in _REGEX_PATTERN_CACHE:
        _REGEX_PATTERN_CACHE[r] = regex.compile(r, flags=regex.VERBOSE | regex.UNICODE)
    return _REGEX_PATTERN_CACHE[r]


class Counter:
    def __init__(self) -> None:
        self._n = 0

    @property
    def value(self):
        return self._n

    def increase(self):
        self._n += 1
        return self._n

    def __next__(self):
        return self.increase()

    def __iter__(self):
        return self


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
    #                       ^^^^^^^^^^ by-word timetag
    # use `[...]` to wrap by-word time tag is okay, but should give warning
    # `<...><...>` is also a valid word, but content is ""


_id_counter = Counter()


class LyricWord(BaseModel):
    id: int = Field(default_factory=_id_counter.increase)
    content: str = ""
    start: int | None = None
    end: int | None = None


class BasicLyricLine(BaseModel):
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
        # if ma.start() > last_index:
        # 小巧思: 如果把空文本也算上的话, 那么生成序列就必然是 文本-匹配-文本 的交替出现
        # 这样后续的状态管理能省很多事
        result.append(text[last_index : ma.start()])

        result.append(ma)
        last_index = ma.end()

    # 处理最后一个匹配项到字符串末尾
    # 如果 last_index 小于字符串总长度, 说明末尾还有文本
    if last_index < len(text):
        result.append(text[last_index:])

    return result


def parse_line(line: str) -> BasicLyricLine | None:
    # 因为折叠行的特性将在上层预处理, 所以我们可以将行内的所有时间标签一视同仁
    seq = split_to_sequence(
        compile_regex(f"{TIMETAG_REGEX} | {WORD_TIMETAG_REGEX}"),
        line.strip(),
    )
    # 为空行时直接返回 None
    if not seq:
        return None

    result = BasicLyricLine()
    if len(seq) == 1:
        obj = seq[0]
        if isinstance(obj, str):
            return BasicLyricLine(words=[LyricWord(content=obj)])
        elif isinstance(obj, regex.Match):
            return BasicLyricLine(words=[LyricWord(content="", start=match2ms(obj))])

    current_word = LyricWord()
    for idx, obj in enumerate(seq):
        if isinstance(obj, regex.Match):
            ts = match2ms(obj)
            if idx == len(seq) - 1:
                pass
            elif idx == 0:
                pass
            elif obj.group().startswith("["):
                s, e = obj.span()
                logger.warning(
                    (
                        "square-brackets-wrapped by-word time tag found"
                        # ", this may confuse by-word time tag with folded by-line time tag"
                        ", nearby objs = %s"
                    ),
                    seq[idx - 1 : idx + 2],
                )

            if idx > 0:
                # 总之是一个 word 结束了
                last_obj = seq[idx - 1]
                if isinstance(last_obj, str):
                    # `还有谁会记得我 [00:51.90]`
                    current_word.content = last_obj
                    current_word.end = ts
                elif isinstance(last_obj, regex.Match):
                    # `<00:49.45> <00:50.00>`
                    current_word.content = ""
                    current_word.start = match2ms(last_obj)
                    current_word.end = ts
                result.words.append(current_word)
                current_word = LyricWord()
        elif isinstance(obj, str):
            if idx > 0:
                last_obj = seq[idx - 1]
                if isinstance(last_obj, str):
                    # `如果时间逃走了 还有谁会记得我`
                    # 如果中间没有标签的话这俩不可能被切开
                    # 那么就是出错了
                    raise InvalidLyricsError(
                        f"unexpected condition, sequence = {seq!r}"
                    )
                elif isinstance(last_obj, regex.Match):
                    # `<00:49.45> 还有谁会记得我`
                    # 一个 word 开始了
                    current_word.start = match2ms(last_obj)
                    current_word.content = obj
    if current_word.content:
        result.words.append(current_word)

    # 检查时间顺序
    words: list[LyricWord] = []
    for idx, word in enumerate(result.words):
        if not word.content:
            continue
        if idx > 0 and word.start is None:
            raise LyricsParserError(
                f"unexpected condition, start time of the first word should not be None: {word!r}; full sequence: {seq!r}"
            )
        if idx < len(result.words) - 1 and word.end is None:
            raise LyricsParserError(
                f"unexpected condition, end time of the last word should not be None: {word!r}; full sequence: {seq!r}"
            )
        if words:
            last_word = words[-1]
            if not (
                (last_word.start or float("-inf"))
                < (last_word.end or 0)
                <= (word.start or 0)
                < (word.end or float("+inf"))
            ):
                last_word.end = max(
                    [
                        o
                        for o in [last_word.start, last_word.end, word.start, word.end]
                        if o is not None
                    ],
                )
                last_word.content += word.content
                continue
        words.append(word)
    result.words = words

    return result


def parse_file(lrc: str) -> Lyrics:
    lines: list[LyricLine] = []
    current: LyricLine | None = None
    for raw_line in lrc.strip().splitlines():
        seq = split_to_sequence(
            compile_regex(f"{TIMETAG_REGEX} | {WORD_TIMETAG_REGEX}"),
            raw_line.strip(),
        )
        line = parse_line(seq)

        if line is None:
            if current:
                lines.append(current)
            current = None
        if line:
            if current is None:
                current = LyricLine.model_validate(line.model_dump())
            else:
                current.reference_lines.append(line)

    if current is not None:
        lines.append(current)

    raise NotImplementedError


def construct_lrc(lyrics: Lyrics):
    pass
