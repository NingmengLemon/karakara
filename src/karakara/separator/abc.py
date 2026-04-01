from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from karakara.typ import NpAudioData, NpAudioSamples


class StemSeparator(ABC):
    VOCAL_STEM_NAME: ClassVar[str] = "vocal"

    @abstractmethod
    def separate(self, audio: NpAudioData | NpAudioSamples) -> dict[str, NpAudioData]:
        raise NotImplementedError
