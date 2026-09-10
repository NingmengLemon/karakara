"""歌词语言判定测试。"""

from __future__ import annotations

from karakara.utils.lang import detect_dominant_lang, detect_lang


def test_detect_lang_single_line() -> None:
    assert detect_lang("あいつも") == "ja"
    assert detect_lang("你好世界") == "zh"
    assert detect_lang("hello world") == "en"
    assert detect_lang("1234 !!!") is None


def test_detect_lang_cannot_separate_kanji_only_japanese_from_chinese() -> None:
    """单行判据的已知局限：纯汉字行只能判成中文。"""
    assert detect_lang("畜生") == "zh"


def test_dominant_lang_kana_is_decisive_even_in_minority() -> None:
    """只要出现假名就是日语，即使纯汉字行更多。

    回归测试：逐行判据会把「畜生」判成中文，按行计票时日文歌里密集的纯汉字行
    可能翻盘，导致整首日文歌被按中文送进对齐器。
    """
    texts = ["畜生", "苛立つ　臓器の末端", "糞みたいだ", "あいつも"]

    assert all(detect_lang(t) in ("zh", "ja") for t in texts)
    assert detect_dominant_lang(texts) == "ja"


def test_dominant_lang_picks_chinese_when_no_kana() -> None:
    assert detect_dominant_lang(["你好世界", "再见", "畜生"]) == "zh"


def test_dominant_lang_picks_english_for_latin_lyrics() -> None:
    assert detect_dominant_lang(["hello", "world", "shine on"]) == "en"


def test_dominant_lang_returns_none_when_no_usable_text() -> None:
    assert detect_dominant_lang([]) is None
    assert detect_dominant_lang(["", "!!!", "123"]) is None
