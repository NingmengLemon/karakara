from __future__ import annotations

from io import BytesIO
from typing import Annotated

import numpy as np
import requests
from numpy.typing import NDArray
from typing_extensions import override

from karakara.gentle_client import GentleClient
from karakara.utils import DEFAULT_SAMPLE_RATE, save_audio

from .abc import AbstractAligner, AlignedWord


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
        audio: Annotated[NDArray[np.float32], "Shape[*,]"],
        text: str,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
    ) -> list[AlignedWord]:
        buffer = BytesIO()

        save_audio(buffer, np.expand_dims(audio, axis=0), sample_rate)
        result: list[AlignedWord] = []
        for w in self._client.submit_bytes_sync(buffer.getvalue(), transcript=text)[
            "words"
        ]:
            result.append(
                AlignedWord(
                    word=w["word"],
                    position=(int(w["start"] * 1e3), int(w["end"] * 1e3))
                    if w["case"] == "success"
                    else None,
                )
            )
        return result
