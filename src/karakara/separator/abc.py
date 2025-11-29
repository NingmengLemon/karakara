from __future__ import annotations

from abc import ABC, abstractmethod

from karakara.typ import NpAudioData, NpAudioSamples


class StemSeparator(ABC):
    @abstractmethod
    def separate(self, audio: NpAudioData | NpAudioSamples) -> dict[str, NpAudioData]:
        raise NotImplementedError
