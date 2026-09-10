from __future__ import annotations

import re
from collections.abc import Iterable
from logging import getLogger
from typing import Literal

logger = getLogger(__name__)

# --------------------------------------------------------------------------
# 语言检测
# --------------------------------------------------------------------------

# 日语检测：仅匹配假名（平假名 + 片假名），不包含 CJK 汉字以避免与中文重叠
# 但是这也意味着包含汉字的日语歌词可能被误判为中文, 这里还需要改
_JA_PATTERN = re.compile(r"[\u3040-\u309F\u30A0-\u30FF]")
_ZH_PATTERN = re.compile(r"[\u4E00-\u9FFF]")
_EN_PATTERN = re.compile(r"[A-Za-z]")

LangCode = Literal["ja", "zh", "en"]


def detect_lang(s: str) -> LangCode | None:
    """根据字符串内容判断语言类型。

    注意：仅凭单行文本无法区分「纯汉字的日语行」与中文行——两者都会判成
    ``"zh"``。需要整首歌的判据时请用 :func:`detect_dominant_lang`。

    Args:
        s: 输入字符串。

    Returns:
        "ja"（日语）、"zh"（中文）、"en"（英语）或 None（无法判断）。
    """
    ja_count = len(_JA_PATTERN.findall(s))
    zh_count = len(_ZH_PATTERN.findall(s))
    en_count = len(_EN_PATTERN.findall(s))

    # 显式比较而不是 max(dict)：后者只能推成 str，返回值要靠 type: ignore 压住，
    # 而且平局顺序隐含在 dict 的字面顺序里。这里把顺序（ja > zh > en）明说出来。
    if ja_count > 0 and ja_count >= zh_count and ja_count >= en_count:
        logger.debug(f"language='ja': {s!r}")
        return "ja"
    if zh_count > 0 and zh_count >= en_count:
        logger.debug(f"language='zh': {s!r}")
        return "zh"
    if en_count > 0:
        logger.debug(f"language='en': {s!r}")
        return "en"

    logger.info(f"unknown language: {s!r}")
    return None


def detect_dominant_lang(texts: Iterable[str]) -> LangCode | None:
    """对整个文本集合（一首歌的歌词行）做语言判定。

    判定规则，按优先级：

    1. **出现任意假名（平假名/片假名）⇒ 日语。**
       假名基本是日语独有的，中文歌词不含假名。这条规则直接解决本项目最要命的
       误判：日文歌里的**纯汉字行**（如「畜生」「苛立つ　臓器の末端」）在逐行判据
       下会被判成中文。在真实语料上（``D:\\MUSIC`` 的日文 VOCALOID 曲库）这类行
       约占 5%，按行数计票时随时可能翻盘，于是整首日文歌被按中文送进对齐器。
    2. 否则出现 CJK 汉字 ⇒ 中文。
    3. 否则出现拉丁字母 ⇒ 英语。
    4. 全无可用字符 ⇒ ``None``（调用方决定退回哪个默认语言）。

    规则的已知代价：一首以中文或英文为主、但夹杂了假名的歌会被判成日语。这类
    情况罕见，且可用 ``--aligner-language`` 显式覆盖。

    Args:
        texts: 该首歌参与对齐的文本行。调用方应先剔除元数据行。

    Returns:
        "ja" / "zh" / "en"，无法判定时为 None。
    """
    kana_total = 0
    cjk_total = 0
    latin_total = 0
    line_count = 0
    for text in texts:
        if not text:
            continue
        line_count += 1
        kana_total += len(_JA_PATTERN.findall(text))
        cjk_total += len(_ZH_PATTERN.findall(text))
        latin_total += len(_EN_PATTERN.findall(text))

    result: LangCode
    if kana_total > 0:
        result = "ja"
    elif cjk_total > 0:
        result = "zh"
    elif latin_total > 0:
        result = "en"
    else:
        logger.info(f"dominant language unknown over {line_count} line(s)")
        return None

    logger.info(
        f"dominant language: {result!r} "
        f"(lines={line_count}, kana={kana_total}, cjk={cjk_total}, latin={latin_total})"
    )
    return result
