from __future__ import annotations

from io import BytesIO
from typing import override

import numpy as np
import requests

from karakara.typ import NpAudioData, NpAudioSamples
from karakara.utils.io import DEFAULT_SAMPLE_RATE, save_audio

from ..abc import AbstractAligner, AlignedWord, LangCode
from .client import GentleClient


class GentleAligner(AbstractAligner):
    def __init__(
        self,
        base_url: str = "http://localhost:8765",
        session: requests.Session | None = None,
    ) -> None:
        self._client = GentleClient(base_url=base_url, timeout=None, session=session)

    @override
    def align(
        self,
        audio: NpAudioData | NpAudioSamples,
        text: str,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        *,
        language: LangCode | None = None,
    ) -> list[AlignedWord]:
        # Gentle 只支持英语对齐，``language`` 在本实现中被忽略。
        if audio.ndim == 1:
            audio = np.expand_dims(audio, axis=0)
        elif audio.ndim == 2:
            pass
        else:
            raise ValueError(f"bad audio ndarray dim: {audio.ndim}, 1 or 2 expected")

        buffer = BytesIO()
        save_audio(buffer, audio, sample_rate)
        result: list[AlignedWord] = []
        for w in self._client.submit_bytes_sync(buffer.getvalue(), transcript=text)[
            "words"
        ]:
            result.append(
                AlignedWord(
                    word=w["word"],
                    position=(
                        (int(w["start"] * 1e3), int(w["end"] * 1e3))
                        if w["case"] == "success"
                        else None
                    ),
                )
            )
        return result
