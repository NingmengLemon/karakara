# models.py
import re
from typing import List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator


class Word(BaseModel):
    """代表逐字歌词中的一个片段"""

    text: str
    start_time_ms: int = Field(..., description="片段开始时间的毫秒数")


class LyricLine(BaseModel):
    """代表一行歌词"""

    start_time_ms: int = Field(..., description="歌词行开始时间的毫秒数")
    end_time_ms: Optional[int] = Field(
        None, description="歌词行结束时间的毫秒数 (可选)"
    )
    text: str = Field(..., description="主歌词文本")
    translations: List[str] = Field(default_factory=list, description="翻译文本列表")
    words: Optional[List[Word]] = Field(None, description="逐字歌词片段列表 (可选)")

    @field_validator("words")
    @classmethod
    def validate_words(cls, words: Optional[List[Word]], values):
        """
        验证并过滤逐字歌词。根据规范，逐字时间戳必须在行时间范围内且递增。
        """
        if not words:
            return None

        start_time = values.data.get("start_time_ms")
        end_time = values.data.get("end_time_ms")

        valid_words = []
        last_word_time = start_time - 1  # 保证第一个词的时间戳有效

        for word in words:
            # 检查时间戳是否有效
            is_valid = word.start_time_ms > last_word_time
            if end_time is not None:
                is_valid = is_valid and (word.start_time_ms < end_time)

            if is_valid:
                valid_words.append(word)
                last_word_time = word.start_time_ms
            else:
                # 根据规范，忽略无效的逐字时间戳
                # 可以在这里添加日志记录来提醒用户
                print(
                    f"警告: 忽略无效的逐字时间戳 {word.start_time_ms} (行范围: {start_time}-{end_time})"
                )

        return valid_words if valid_words else None


class Lyrics(BaseModel):
    """代表解析后的整个歌词对象"""

    lines: List[LyricLine] = Field(..., description="歌词行列表")


# parser.py

# --- 正则表达式定义 ---
# 匹配时间戳 [mm:ss.xx]
TIMESTAMP_RE = re.compile(r"\[(\d{1,3}):(\d{1,2})\.(\d{1,6})\]")
# 匹配一个或多个行首时间戳
LINE_START_RE = re.compile(r"^((\[[^\]]+\])+)(.*)")
# 匹配逐字歌词中的时间戳标记 [mm:ss.xx] 或 <mm:ss.xx>
WORD_TAG_RE = re.compile(r"[<\[](\d{1,3}:\d{1,2}\.\d{1,6})[>\]]")
# 用于分割逐字歌词文本和标记
WORD_SPLIT_RE = re.compile(r"([<\[]\d{1,3}:\d{1,2}\.\d{1,6}[>\]])")


def _parse_timestamp_str(ts_str: str) -> Optional[int]:
    """将 'mm:ss.xxx' 格式的字符串解析为总毫秒数"""
    try:
        minutes_str, rest = ts_str.split(":", 1)
        seconds_str, ms_part = rest.split(".", 1)

        minutes = int(minutes_str)
        seconds = int(seconds_str)

        # 规范: "不足 3 位的写法将视为在后位省略了 0"
        # 例如: .5 -> 500ms, .12 -> 120ms. 这等同于将其作为小数处理
        # 规范: "毫秒限制 1 至 6 位数字"，这种方式也能正确处理
        milliseconds = int(float(f"0.{ms_part}") * 1000)

        return minutes * 60 * 1000 + seconds * 1000 + milliseconds
    except (ValueError, IndexError):
        return None


class SPLParser:
    """Salt Player 歌词格式 (SPL) 解析器"""

    def parse(self, text: str) -> Lyrics:
        """
        解析 SPL 格式的歌词文本。

        Args:
            text: 包含 SPL 歌词的字符串。

        Returns:
            一个包含所有歌词信息的 Lyrics 对象。
        """
        lines = text.splitlines()
        raw_lines: List[Tuple[List[int], str]] = []

        # --- 阶段 1: 初步行解析 ---
        # 遍历文本行，将带时间戳的行与其后的翻译行分组
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue

            match = LINE_START_RE.match(line)
            if match:
                timestamps_block, text_content = match.groups()[0], match.groups()[2]

                start_timestamps_ms = [
                    t
                    for ts_str in TIMESTAMP_RE.findall(timestamps_block)
                    if (
                        t := _parse_timestamp_str(
                            f"{ts_str[0]}:{ts_str[1]}.{ts_str[2]}"
                        )
                    )
                    is not None
                ]

                if not start_timestamps_ms:  # 如果时间戳格式错误，则跳过
                    i += 1
                    continue

                # 合并主歌词行和紧随其后的翻译行
                full_text_block = [text_content.strip()]
                j = i + 1
                while j < len(lines):
                    next_line = lines[j].strip()
                    if not next_line or LINE_START_RE.match(next_line):
                        break
                    full_text_block.append(next_line)
                    j += 1

                raw_lines.append((start_timestamps_ms, "\n".join(full_text_block)))
                i = j
            else:
                # 忽略不以时间戳开头的非空行（它们应已被前一步作为翻译处理）
                i += 1

        # --- 阶段 2: 结构化组装 ---
        # 处理 raw_lines, 创建 LyricLine 对象
        lyric_lines: List[LyricLine] = []
        for start_times, block in raw_lines:
            parts = block.split("\n")
            main_lyric_text = parts[0]
            translations = parts[1:]

            # 解析逐字歌词和显式行结尾
            clean_text, words, explicit_end_time = self._parse_line_content(
                main_lyric_text
            )

            # 处理显式换行结尾
            if not main_lyric_text and translations:
                # 这种情况是 [start]\n[end]
                # 我们需要在前一行找到这个 'start'
                if lyric_lines:
                    ts_match = TIMESTAMP_RE.match(translations[0])
                    if ts_match:
                        end_time = _parse_timestamp_str(
                            f"{ts_match.group(1)}:{ts_match.group(2)}.{ts_match.group(3)}"
                        )
                        if end_time is not None:
                            # 为上一组时间戳相同的所有行设置结束时间
                            last_start_time = lyric_lines[-1].start_time_ms
                            for line in reversed(lyric_lines):
                                if line.start_time_ms == last_start_time:
                                    if line.end_time_ms is None:
                                        line.end_time_ms = end_time
                                else:
                                    break
                continue

            # 处理重复行 [ts1][ts2]...
            for start_time in start_times:
                line_obj = LyricLine(
                    start_time_ms=start_time,
                    text=clean_text,
                    translations=translations,
                    words=words,
                    end_time_ms=explicit_end_time,
                )
                lyric_lines.append(line_obj)

        # --- 阶段 3: 排序和二次处理 ---
        # 按开始时间排序
        lyric_lines.sort(key=lambda x: x.start_time_ms)

        # 计算隐式行结尾时间
        for idx, current_line in enumerate(lyric_lines):
            if current_line.end_time_ms is None:
                if idx + 1 < len(lyric_lines):
                    next_line = lyric_lines[idx + 1]
                    current_line.end_time_ms = next_line.start_time_ms

        return Lyrics(lines=lyric_lines)

    def _parse_line_content(
        self, text: str
    ) -> Tuple[str, Optional[List[Word]], Optional[int]]:
        """
        从单行文本中解析出干净的文本、逐字列表和显式结束时间。
        """
        if not WORD_TAG_RE.search(text):
            return text, None, None

        # 检查行尾是否存在显式结束时间戳 [..:.. ..]
        explicit_end_time = None
        # 注意：结尾时间戳必须是 [ ] 形式
        m = re.search(r"(\[(\d{1,3}:\d{1,2}\.\d{1,6})\])\s*$", text)
        if m:
            tag_str, ts_str = m.groups()
            explicit_end_time = _parse_timestamp_str(ts_str)
            # 从文本中移除结尾时间戳以便处理逐字部分
            text = text[: -len(tag_str)].rstrip()

        split_parts = WORD_SPLIT_RE.split(text)

        words: List[Word] = []
        clean_text_parts: List[str] = []

        # 第一个片段的文本
        if split_parts[0]:
            # 延迟逐字标记：如果文本以 < > 开头，则第一个文本片段为空
            # [start]<t1>word1... 此时 start 时间没有对应文本
            clean_text_parts.append(split_parts[0])

        # 迭代处理 (时间戳, 文本) 对
        for i in range(1, len(split_parts), 2):
            ts_tag = split_parts[i]
            word_text = split_parts[i + 1] if i + 1 < len(split_parts) else ""

            ts_str_match = WORD_TAG_RE.match(ts_tag)
            if ts_str_match:
                ts_str = ts_str_match.group(1)
                word_start_time = _parse_timestamp_str(ts_str)
                if word_start_time is not None and word_text:
                    words.append(Word(text=word_text, start_time_ms=word_start_time))

            if word_text:
                clean_text_parts.append(word_text)

        # 兼容延迟逐字：[start]<t1>word1..在这种情况下，第一个文本片段是word1
        # 但我们还需要处理 [start]word0<t1>word1.. 的情况
        if not words and split_parts[0]:  # 如果没有逐字标记，则返回原始文本
            return "".join(clean_text_parts), None, explicit_end_time

        # 如果第一个片段在任何时间戳之前，它没有自己的逐字条目
        # 它属于行的开始时间戳
        # `validate_words` 之后会清理不合逻辑的时间戳

        return "".join(clean_text_parts), words, explicit_end_time


def main():
    # 示例文本，涵盖了规范中的大部分特性
    spl_text = """
    [00:10.5]这是第一行歌词
    [00:12.500]This is the first line of lyrics
    こんにちは

    [00:15.1]这是第二行，有显式结尾[00:18.2]
    [00:20.0][00:40.0]这是一句重复出现的歌词
    This is a repeated line
    这是第二种语言的翻译

    [00:25.33]这行歌词使用换行来标记结尾
    [00:28.0]

    [00:30.123]这行歌词将隐式地在下一行开始时结束
    [00:35.45]天天开心

    [01:00.0]这是一行[01:01.0]逐字[01:02.5]歌词[01:04.0]
    [01:05.0]兼容性<01:06.0>逐字<01:07.5>标记[01:09.0]

    [01:10.0]<01:11.0>延迟开始的<01:12.5>逐字歌词[01:14.0]

    [02:00.0][02:10.0]重复行与<02:02.0>逐字标记<02:03.0>结合[02:04.0]
    """

    # 创建解析器实例
    parser = SPLParser()

    # 解析歌词
    parsed_lyrics = parser.parse(spl_text)

    # Pydantic的 .model_dump_json() 方法可以方便地将其转换为格式化的JSON字符串
    print(parsed_lyrics.model_dump_json(indent=2, exclude_none=True))

    # 你也可以直接访问数据对象
    print("\n--- 直接访问数据 ---")
    first_line = parsed_lyrics.lines[0]
    print(f"第一行歌词: '{first_line.text}'")
    print(f"开始时间: {first_line.start_time_ms}ms")
    print(f"结束时间: {first_line.end_time_ms}ms")
    print(f"翻译: {first_line.translations}")

    word_line = parsed_lyrics.lines[-3]  # 获取延迟开始的逐字歌词行
    print(f"\n逐字歌词行: '{word_line.text}'")
    if word_line.words is not None:
        for word in word_line.words:
            print(f"  - 片段: '{word.text}', 开始于: {word.start_time_ms}ms")

    # 检查重复行与逐字标记的局限性
    print("\n--- 检查重复行与逐字标记的兼容性 ---")
    invalid_word_line = next(
        line for line in parsed_lyrics.lines if line.start_time_ms == 70000
    )  # 02:10.0
    print(f"行开始于 70000ms (01:10.0) 的歌词: '{invalid_word_line.text}'")
    if invalid_word_line.words:
        print("逐字标记: ", invalid_word_line.words)
    else:
        # `validate_words` 已经清除了无效的逐字标记
        print(
            "逐字标记: 无 (因为时间戳 <02:02.0> 在行开始时间 <02:10.0> 之前，已被自动过滤)"
        )


if __name__ == "__main__":
    main()
