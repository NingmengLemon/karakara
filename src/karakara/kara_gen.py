from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from karakara.aligner.abc import AbstractAligner
from karakara.aligner.gentle import GentleAligner
from karakara.spl_parser import LyricLine, Lyrics, LyricWord
from karakara.stem_separator import get_vocal_stem, separator
from karakara.typ import NpAudioData
from karakara.utils import load_audio, ms2sample


def judge_lang(s: str) -> Literal["ja", "zh", "en"] | None:
    """根据字符串内容判断语言类型。

    Args:
        s (str): 输入字符串。

    Returns:
        Literal["ja", "zh", "en"] | None: 返回语言类型，可能的值为 "ja"（日语）、"zh"（中文）、"en"（英语）或 None（无法判断）。
    """
    ja_pattern = re.compile(
        r"[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FBF\u3000-\u303F]"
    )  # 包含平假名、片假名、汉字和日文标点符号
    zh_pattern = re.compile(r"[\u4E00-\u9FFF]")  # 包含常用汉字
    en_pattern = re.compile(r"[A-Za-z]")  # 包含英文字母

    ja_count = len(ja_pattern.findall(s))
    zh_count = len(zh_pattern.findall(s))
    en_count = len(en_pattern.findall(s))

    counts = {"ja": ja_count, "zh": zh_count, "en": en_count}
    max_lang = max(counts.keys(), key=counts.get)
    if counts[max_lang] == 0:
        return None
    return max_lang


class KaraGen:
    def __init__(
        self,
        lyrics: str | Path,
        audio: str | Path,
        aligner: AbstractAligner | None = None,
        target_lang: Literal["en"] = "en",
    ) -> None:
        self.aligner: AbstractAligner = aligner or GentleAligner()
        with open(lyrics, "r") as fp:
            self.lyrics = Lyrics.loads(fp.read())
        vocal, self.sample_rate = get_vocal_stem(audio)
        self.vocal: NDArray[np.float32] = vocal[0]
        self.tlang = target_lang

    def __call__(self) -> Lyrics:
        for idx, line in enumerate(self.lyrics.lines):
            do = False
            text = ""
            if len(line.content) == 1 and (text := line.content[0].content):
                if judge_lang(text) == self.tlang:
                    do = True
            if not do or not text:
                continue
            if re.search(
                r"(作?词|作?曲|编曲|演?唱|专辑|歌手?|制作人?|和声|混音?|录音?|监制|策划|封面设计|文案|出品|OP|SP|翻译|PV|母带|调教|调校|曲?绘|原曲|編曲|作詞|唄|呗)\s*[：\:].+",
                text,
            ):
                continue
            start = ms2sample(line.start or 0, self.sample_rate)
            end = None
            if idx < len(self.lyrics.lines) - 1:
                if line.end:
                    end = ms2sample(line.end, self.sample_rate)
                elif (next_line := self.lyrics.lines[idx + 1]).start:
                    end = ms2sample(next_line.start, self.sample_rate)

            print("line:", start, end)
            if start and end and start > end:
                continue
            audio_piece = self.vocal[start:end] if end else self.vocal[start:]
            words = self.aligner.align(audio_piece, text, self.sample_rate)
            iidx = 0
            words_kara: list[LyricWord] = []
            for word in words:
                if not (pos := word.position):
                    continue
                next_idx = text.find(word.word, iidx)
                print("word:", idx, iidx, next_idx, word)
                if next_idx > iidx:
                    words_kara.append(
                        LyricWord(
                            start=words_kara[-1].end if words_kara else None,
                            end=pos[0] + (line.start or 0),
                            content=text[iidx:next_idx],
                        )
                    )

                words_kara.append(
                    LyricWord(
                        start=pos[0] + (line.start or 0),
                        end=pos[1] + (line.start or 0),
                        content=word.word,
                    )
                )
                iidx = next_idx + len(word.word)
            words_kara.append(
                LyricWord(
                    start=words_kara[-1].end if words_kara else None,
                    end=None,
                    content=text[iidx:],
                )
            )
            line.content = words_kara
        return self.lyrics
