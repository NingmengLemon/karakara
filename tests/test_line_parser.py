import pytest

from karakara.spl_parser import (
    TIMETAG_REGEX,
    WORD_TIMETAG_REGEX,
    LyricWord,
    split_to_sequence,
)
from karakara.spl_parser import parse_line as parse_line


def parse_line(line: str):
    return parse_line(
        split_to_sequence(f"{TIMETAG_REGEX} | {WORD_TIMETAG_REGEX}", line.strip())
    )


def pydantic_eq(a: list[LyricWord], b: list[LyricWord]) -> bool:
    if len(a) != len(b):
        return False
    for j, k in zip(a, b):
        j.id = k.id = 0
        if j.model_dump_json() != k.model_dump_json():
            return False
    return True


class TestParseLine:
    """测试 parse_line 函数的各种情况"""

    def test_basic_line(self):
        """测试基础歌词行"""
        result = parse_line("[00:05.000]今天天气真好")
        expected_words = [LyricWord(id=1, content="今天天气真好", start=5000, end=None)]

        assert result is not None
        assert result.start == 5000
        assert result.end is None
        assert pydantic_eq(result.words, expected_words)

    def test_explicit_end_same_line(self):
        """测试同一行内的显式行结尾"""
        result = parse_line("[00:15.000]这句歌词只持续两秒哦[00:17.000]")
        expected_words = [
            LyricWord(id=2, content="这句歌词只持续两秒哦", start=15000, end=17000)
        ]

        assert result is not None
        assert result.start == 15000
        assert result.end == 17000
        assert pydantic_eq(result.words, expected_words)

    def test_empty_line_timestamp(self):
        """测试空行时间戳（清空歌词）"""
        result = parse_line("[00:10.500]")
        expected_words = [LyricWord(id=3, content="", start=10500, end=None)]

        assert result is not None
        assert result.start == 10500
        assert result.end is None
        assert pydantic_eq(result.words, expected_words)

    def test_timestamp_format_variations(self):
        """测试时间戳格式变体"""
        # 测试 .5 → 500 毫秒
        result1 = parse_line("[00:22.5]喵~ 喵~ 喵~")

        assert result1 is not None
        assert result1.start == 22500

        # 测试单数字分钟
        result2 = parse_line("[1:22.500]Meow~ Meow~ Meow~")

        assert result2 is not None
        assert result2.start == 82500

        # 测试两位秒数
        result3 = parse_line("[01:02.50]测试")

        assert result3 is not None
        assert result3.start == 62500

    def test_byword_angle_brackets(self):
        """测试使用 <> 标记的逐字歌词"""
        result = parse_line(
            "[00:40.000]要<00:41.000>吃<00:41.500>小<00:42.000>鱼<00:42.500>干[00:44.000]"
        )
        expected_words = [
            LyricWord(id=4, content="要", start=40000, end=41000),
            LyricWord(id=5, content="吃", start=41000, end=41500),
            LyricWord(id=6, content="小", start=41500, end=42000),
            LyricWord(id=7, content="鱼", start=42000, end=42500),
            LyricWord(id=8, content="干", start=42500, end=44000),
        ]

        assert result is not None
        assert result.start == 40000
        assert result.end == 44000
        assert pydantic_eq(result.words, expected_words)

    def test_delayed_byword_start(self):
        """测试延迟开始的逐字标记"""
        result = parse_line(
            "[00:40.000]<00:41.000>要<00:41.500>吃<00:42.000>小<00:42.500>鱼<00:43.000>干[00:44.000]"
        )
        expected_words = [
            LyricWord(id=9, content="要", start=41000, end=41500),
            LyricWord(id=10, content="吃", start=41500, end=42000),
            LyricWord(id=11, content="小", start=42000, end=42500),
            LyricWord(id=12, content="鱼", start=42500, end=43000),
            LyricWord(id=13, content="干", start=43000, end=44000),
        ]

        assert result is not None
        assert result.start == 40000
        assert result.end == 44000
        assert pydantic_eq(result.words, expected_words)

    def test_mixed_byword_brackets(self):
        """测试混合使用 [] 和 <> 的逐字标记"""
        # 这里应该会产生 logging 消息
        result = parse_line(
            "[00:40.000]要[00:41.000]吃<00:41.500>小[00:42.000]鱼<00:42.500>干[00:44.000]"
        )

        expected_words = [
            LyricWord(id=14, content="要", start=40000, end=41000),
            LyricWord(id=15, content="吃", start=41000, end=41500),
            LyricWord(id=16, content="小", start=41500, end=42000),
            LyricWord(id=17, content="鱼", start=42000, end=42500),
            LyricWord(id=18, content="干", start=42500, end=44000),
        ]

        assert result is not None
        assert result.start == 40000
        assert result.end == 44000
        assert pydantic_eq(result.words, expected_words)

    def test_special_characters(self):
        """测试特殊字符和颜文字"""
        result = parse_line("[00:12.000]好呀！=^._.^= inte")
        expected_words = [
            LyricWord(id=19, content="好呀！=^._.^= inte", start=12000, end=None)
        ]

        assert result is not None
        assert pydantic_eq(result.words, expected_words)

    def test_invalid_byword_timestamp(self):
        """测试错误的逐字时间戳（时间倒序）"""
        result = parse_line("[00:50.000]第一遍<00:49.000>正常<00:52.000>")
        # 应该忽略 <00:49.000>，因为小于开始时间
        expected_words = [
            LyricWord(id=20, content="第一遍正常", start=50000, end=52000)
        ]

        assert result is not None
        assert pydantic_eq(result.words, expected_words)

    def test_only_start_and_end(self):
        """测试只有开始和结束时间戳的情况"""
        result = parse_line("[00:50.000]简单的歌词行[00:52.000]")
        expected_words = [
            LyricWord(id=21, content="简单的歌词行", start=50000, end=52000)
        ]

        assert result is not None
        assert pydantic_eq(result.words, expected_words)

    # 新增的边界测试用例
    def test_whitespace_handling(self):
        """测试前后空格的处理"""
        result = parse_line("[00:30.000]  前后有空格  ")
        expected_words = [
            LyricWord(id=22, content="  前后有空格  ", start=30000, end=None)
        ]

        assert result is not None
        assert pydantic_eq(result.words, expected_words)

    def test_multiple_spaces_in_text(self):
        """测试文本中的多个连续空格"""
        result = parse_line("[00:31.000]这里    有    很多空格")
        expected_words = [
            LyricWord(id=23, content="这里    有    很多空格", start=31000, end=None)
        ]

        assert result is not None
        assert pydantic_eq(result.words, expected_words)

    def test_unicode_characters(self):
        """测试Unicode字符"""
        result = parse_line("[00:32.000]🎵音乐🎶和😺表情")
        expected_words = [
            LyricWord(id=24, content="🎵音乐🎶和😺表情", start=32000, end=None)
        ]

        assert result is not None
        assert pydantic_eq(result.words, expected_words)

    def test_edge_case_timestamps(self):
        """测试边界情况的时间戳"""
        # 最大毫秒数
        result1 = parse_line("[99:59.999999]边界测试")

        assert result1 is not None
        assert result1.start == (99 * 60 + 59) * 1000 + 999

        # 最小时间
        result2 = parse_line("[0:0.001]开始")

        assert result2 is not None
        assert result2.start == 1

    def test_no_text_after_byword(self):
        """测试逐字标记后没有文本的情况"""
        result = parse_line("[00:45.000]测试<00:46.000>")
        expected_words = [LyricWord(id=25, content="测试", start=45000, end=46000)]

        assert result is not None
        assert pydantic_eq(result.words, expected_words)

    def test_only_byword_no_text(self):
        """测试只有逐字标记没有实际文本的情况"""
        result = parse_line("[00:47.000]<00:48.000>")
        expected_words = [LyricWord(id=26, content="", start=48000, end=None)]

        assert result is not None
        assert result.start == 47000
        assert pydantic_eq(result.words, expected_words)

    def test_complex_mixed_scenario(self):
        """测试复杂的混合场景"""
        result = parse_line(
            "[01:30.500]<01:31.000>开始<01:32.000>唱歌[01:33.000]谢谢[01:34.000]"
        )
        expected_words = [
            LyricWord(id=27, content="开始", start=31000, end=32000),
            LyricWord(id=28, content="唱歌", start=32000, end=33000),
            LyricWord(id=29, content="谢谢", start=33000, end=34000),
        ]

        assert result is not None
        assert result.start == 90500  # 01:30.500
        assert result.end == 94000  # 01:34.000
        assert pydantic_eq(result.words, expected_words)


# 运行测试的便捷函数
def run_tests():
    """运行所有测试"""
    import os
    import sys

    sys.path.append(os.path.dirname(__file__))

    # 使用 pytest 运行测试
    pytest.main([__file__, "-v"])


if __name__ == "__main__":
    run_tests()
