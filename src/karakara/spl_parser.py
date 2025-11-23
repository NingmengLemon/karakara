from __future__ import annotations

from enum import Enum, auto
from io import StringIO
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
    (?:
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
    )
"""
)
compile_regex(
    WORD_TIMETAG_REGEX := r"""
    (?:
        \<
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
        \>
    )
"""
)
compile_regex(
    TIMETAG_REGEX_STRICT := r"""
    (?:
        \[
            (?P<min>\d{1,4})
            :
            (?P<sec>\d{1,2})
            \.
            (?P<tail>\d{1,3})
        \]
    )
"""
)
compile_regex(
    METATAG_REGEX := r"""
    (?:
        \[
            \s*
            (?P<key>[a-zA-Z]{2,16})
            \s*
            :
            \s*
            (?P<value>.+?)
            \s*
        \]
    )
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
        # 将毫秒部分规范化到3位长
        if len(tail) > 3:  # 截断
            tail = tail[:3]  # "123456" -> "123"
        elif len(tail) < 3:  # 补齐
            tail = tail.ljust(3, "0")  # "1" -> "100", "12" -> "120"
        ms = int(tail)
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


class NullableStartEndModel(BaseModel):
    start: int | None = None
    end: int | None = None


class StartEndModel(BaseModel):
    start: int
    end: int


class LyricWord(NullableStartEndModel):
    content: str = ""


BasicLyricLine: TypeAlias = list[LyricWord]


class LyricLine(NullableStartEndModel):
    content: BasicLyricLine = Field(default_factory=list)
    reference_lines: list[BasicLyricLine] = Field(default_factory=list)


class Lyrics(BaseModel):
    lines: list[LyricLine] = Field(default_factory=list)
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


def parse_line(line: str) -> BasicLyricLine | None:
    # 因为折叠行的特性将在上层预处理, 所以我们可以将行内的所有时间标签一视同仁
    if not line.strip():
        return None
    seq = split_to_sequence(
        GENERIC_TIMETAG_REGEX,
        line,
    )
    if not seq or (len(seq) == 1 and not seq[0]):
        return None

    if len(seq) % 2 != 1:
        raise LyricsParserError(
            f"未预料的情况: 匹配序列的长度预期为奇数, 而不是: {len(seq)}"
        )

    # 总之先 unzip 成两个序列, 然后就会变成这样:
    # text_seq: [0] [1] [2] [3] [4]
    #            | / | / | / | /
    # time_seq: [0] [1] [2] [3]
    text_seq: list[str] = []
    time_seq: list[int] = []
    for idx in range(len((seq))):
        if is_match(m := seq[idx]):
            time_seq.append(match2ms(m))
        elif is_str(s := seq[idx]):
            text_seq.append(s)
        else:
            raise LyricsParserError(
                f"未预料的情况: 匹配序列中的元素的类型预期为 str 或 Match, 而不是 {type(seq[idx])}"
            )

    # 特殊情况
    if len(text_seq) == 1:
        if time_seq:
            raise LyricsParserError(
                f"未预料的情况: 匹配序列长度为 1 时, 其中的唯一元素的类型预期为 str, 而不是 {type(seq[0])}"
            )
        return [LyricWord(content=text_seq[0])]

    if (d := (len(text_seq) - len(time_seq))) != 1:
        raise LyricsParserError(
            f"未预料的情况: 文本序列的长度 减去 时间序列的长度 的值预期为 1, 而不是 {d}"
        )

    offset = 0
    for idx in range(0, len(time_seq)):
        idx -= offset
        if idx > 0:
            now_time = time_seq[idx]
            prev_time = time_seq[idx - 1]
            if not (prev_time < now_time):
                logger.warning(
                    f"unordered time tag found: [prev={prev_time}, now={now_time}]"
                )
                text_seq[idx] += text_seq[idx + 1]
                text_seq.pop(idx + 1)
                time_seq.pop(idx)
                offset += 1

    result: BasicLyricLine = []
    for idx in range(0, len(text_seq)):
        word = LyricWord(content=text_seq[idx])
        if 0 < idx <= len(text_seq) - 1:
            word.start = time_seq[idx - 1]
        if 0 <= idx < len(text_seq) - 1:
            word.end = time_seq[idx]

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


def split_line_timetags(raw_line: str) -> tuple[list[regex.Match[str]], str]:
    matches = []
    m = compile_regex(f"^{TIMETAG_REGEX}").match(raw_line)
    while m is not None:
        matches.append(m)
        raw_line = raw_line.removeprefix(m.group())
        m = compile_regex(f"^{TIMETAG_REGEX}").match(raw_line)
    return matches, raw_line


def extract_metadata(s: str) -> dict[str, str]:
    matches = [m.groupdict() for m in compile_regex(METATAG_REGEX).scanner(s)]
    return {m["key"]: m["value"] for m in matches}


def parse_file(lrc: str) -> Lyrics:
    metadata = {}
    line_pool: dict[int, LyricLine] = {}
    last_tag: int | None = None
    for raw_line in lrc.strip().splitlines():
        line_str = raw_line.strip()

        if m := extract_metadata(line_str):
            metadata.update(m)
            logger.debug(f"元数据行, 不解析歌词: {line_str!r}")
            continue

        logger.debug(f"开始解析歌词行: {line_str!r}")
        matches, line_str = split_line_timetags(line_str)
        time_tags = [match2ms(m) for m in matches]
        line = parse_line(line_str)
        if not line:
            last_tag = None
            logger.debug("参照行标记已重置")
            if time_tags and (t := time_tags[0]) not in line_pool:
                line_pool[t] = LyricLine(start=t, content=[LyricWord(content="")])
            continue
        if not time_tags:
            if last_tag is None:
                logger.warning(f"孤立的歌词行: {line!r} (raw={raw_line!r})")
                continue
            logger.debug(f"添加 {line!r} 作为 {line_pool[last_tag]!r} 的参照行")
            line_pool[last_tag].reference_lines.append(line)
            continue

        word_start = line[0].start
        for tag in time_tags:
            if word_start is not None and word_start < tag:
                logger.warning(
                    f"无效的重复行时间标签: {tag}ms, 因为它开始于首字开始 ({word_start}ms) 之后"
                )
                continue
            if tag in line_pool:
                # 同一时间点有两行不同的歌词文本,
                # 新来的这行作为参照行
                line_pool[tag].reference_lines.append(line)
            else:
                # 创建一个新的 LyricLine 对象, 而不是引用同一个 line 对象
                # 需要深拷贝 line_content, 以免后续修改互相影响
                line_pool[tag] = LyricLine(
                    content=[word.model_copy(deep=True) for word in line]
                )
        if len(time_tags) == 1:
            last_tag = time_tags[0]
        else:
            # 重复行不应该成为后续无时间戳行的参照
            last_tag = None

    lyrics = Lyrics(metadata=metadata)
    sorted_lines = sorted(list(line_pool.items()), key=lambda o: o[0])
    for idx, (start, line) in enumerate(sorted_lines):
        line.start = start
        if (lw := line.content[-1]).end is not None:
            line.end, lw.end = lw.end, None

        # 或许应该加个开关...?

        # optional feature: 填充隐式结尾
        # if line.end is None and idx + 1 < len(sorted_lines):
        #     # 隐式结尾：持续到下一行开始
        #     line.end = sorted_lines[idx + 1][0]

        # optional feature: 填充首字起始和末字结束
        # if (fw := line.content[0]).start is None:
        #     fw.start = start
        # if lw.end is None:
        #     lw.end = line.end

        lyrics.lines.append(line)

    return lyrics


def ms_to_tag(ms: int, byword: bool = False) -> str:
    mi = ms // 60000
    sec = (ms % 60000) // 1000
    ms = ms % 1000
    return (
        f"<{mi:02d}:{sec:02d}.{ms:03d}>" if byword else f"[{mi:02d}:{sec:02d}.{ms:03d}]"
    )


def construct_line(line: BasicLyricLine) -> str:
    result = ""
    for idx, word in enumerate(line):
        prefix = ""
        suffix = "" if word.end is None else ms_to_tag(word.end, byword=True)
        if word.start is not None and (
            (idx > 0 and line[idx - 1].end != word.start) or idx == 0
        ):
            prefix = ms_to_tag(word.start, byword=True)
        result += f"{prefix}{word.content}{suffix}"

    return result


def construct_lrc(lyrics: Lyrics) -> str:
    buffer = StringIO()

    # metadata
    for mk, mv in lyrics.metadata.items():
        buffer.write(f"[{mk}: {mv}]\n")

    for line in lyrics.lines:
        buffer.write("\n")
        line_start = line.start
        if line_start is None:
            logger.warning(f"未知的行起始时间: {line}")
            continue
        buffer.write(ms_to_tag(line_start))
        buffer.write(construct_line(line.content))
        if line.end:
            buffer.write(ms_to_tag(line.end))
        buffer.write("\n")

        for refline in line.reference_lines:
            buffer.write(ms_to_tag(line_start))
            buffer.write(construct_line(refline))
            buffer.write("\n")

    return buffer.getvalue()
