from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from karakara.typ import NpAudioData, NpAudioSamples


class AbstractStemSeparator(ABC):
    VOCAL_STEM_NAME: ClassVar[str] = "vocal"

    @abstractmethod
    def separate(self, audio: NpAudioData | NpAudioSamples) -> dict[str, NpAudioData]:
        raise NotImplementedError

    @property
    @abstractmethod
    def samplerate(self) -> int:
        """返回分离器期望的采样率。"""
        return 44100
