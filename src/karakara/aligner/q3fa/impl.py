from __future__ import annotations

from io import BytesIO

import numpy as np
import requests
from typing_extensions import override

from karakara.typ import NpAudioData, NpAudioSamples
from karakara.utils.io import DEFAULT_SAMPLE_RATE, save_audio

from ..abc import AbstractAligner, AlignedWord
from .client import Q3FAClient


class Qwen3ForcedAligner(AbstractAligner):
    """基于 Qwen3-ForcedAligner 服务的对齐器实现。"""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        language: str = "Chinese",
        session: requests.Session | None = None,
    ) -> None:
        self._client = Q3FAClient(base_url=base_url, timeout=None, session=session)
        self._language = language

    @override
    def align(
        self,
        audio: NpAudioData | NpAudioSamples,
        text: str,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
    ) -> list[AlignedWord]:
        if audio.ndim == 1:
            audio = np.expand_dims(audio, axis=0)
        elif audio.ndim == 2:
            pass
        else:
            raise ValueError(f"bad audio ndarray dim: {audio.ndim}, 1 or 2 expected")

        buffer = BytesIO()
        save_audio(buffer, audio, sample_rate)
        response = self._client.align_bytes(
            buffer.getvalue(), text=text, language=self._language
        )

        result: list[AlignedWord] = []
        for w in response["words"]:
            result.append(
                AlignedWord(
                    word=w["text"],
                    position=(
                        int(w["start_time"] * 1e3),
                        int(w["end_time"] * 1e3),
                    ),
                )
            )
        return result
