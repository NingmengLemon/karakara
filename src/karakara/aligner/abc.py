from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from karakara.typ import NpAudioData, NpAudioSamples
from karakara.utils.io import DEFAULT_SAMPLE_RATE


@dataclass
class AlignedWord:
    word: str
    position: tuple[int, int] | None = None


#: 本项目内部使用的语言代码（ISO 639-1）。实现负责映射成各自服务的形式。
LangCode = str


class AbstractAligner(ABC):
    @abstractmethod
    def align(
        self,
        audio: NpAudioData | NpAudioSamples,
        text: str,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        *,
        language: LangCode | None = None,
    ) -> list[AlignedWord]:
        """对一段音频与对应文本做强制对齐。

        Args:
            audio: 音频，shape (channels, samples) 或 (samples,)
            text: 与音频对应的参考文本
            sample_rate: 采样率 (Hz)
            language: ISO 639-1 语言代码（如 ``"ja"`` / ``"zh"`` / ``"en"``）。
                ``None`` 表示使用实现自身的默认语言。

        Returns:
            对齐后的词列表；无法确定位置的词其 ``position`` 为 ``None``。
        """
        raise NotImplementedError
