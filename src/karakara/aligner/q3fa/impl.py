from __future__ import annotations

from io import BytesIO

import numpy as np
import requests
from typing_extensions import override

from karakara.typ import NpAudioData, NpAudioSamples
from karakara.utils.io import DEFAULT_SAMPLE_RATE, save_audio

from ..abc import AbstractAligner, AlignedWord, LangCode
from .client import Q3FAClient

#: Qwen3-ForcedAligner-0.6B 支持的 11 种语言（模型卡），ISO 639-1 → 服务名。
_LANGUAGE_NAMES: dict[str, str] = {
    "zh": "Chinese",
    "en": "English",
    "yue": "Cantonese",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "pt": "Portuguese",
    "ru": "Russian",
    "es": "Spanish",
}

#: 服务端 ``language`` 字段的默认值。
DEFAULT_LANGUAGE = "Chinese"


class Qwen3ForcedAligner(AbstractAligner):
    """基于 Qwen3-ForcedAligner 服务的对齐器实现。"""

    def __init__(
        self,
        base_url: str = "http://localhost:8787",
        language: str = DEFAULT_LANGUAGE,
        session: requests.Session | None = None,
    ) -> None:
        self._client = Q3FAClient(base_url=base_url, timeout=None, session=session)
        self._language = language

    def close(self) -> None:
        """关闭底层 HTTP 会话并释放连接池。"""
        self._client.close()

    @staticmethod
    def resolve_language(language: LangCode | None) -> str:
        """把 ISO 639-1 代码映射为服务端语言名。

        未知代码按原样透传，留出让服务端报错而不是静默退回默认语言的余地。
        """
        if language is None:
            return DEFAULT_LANGUAGE
        return _LANGUAGE_NAMES.get(language, language)

    @override
    def align(
        self,
        audio: NpAudioData | NpAudioSamples,
        text: str,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        *,
        language: LangCode | None = None,
    ) -> list[AlignedWord]:
        if audio.ndim == 1:
            audio = np.expand_dims(audio, axis=0)
        elif audio.ndim == 2:
            pass
        else:
            raise ValueError(f"bad audio ndarray dim: {audio.ndim}, 1 or 2 expected")

        # ``language`` 为 None 时退回构造时配置的默认语言。
        service_language = (
            self._language if language is None else self.resolve_language(language)
        )

        buffer = BytesIO()
        save_audio(buffer, audio, sample_rate)
        response = self._client.align_bytes(
            buffer.getvalue(), text=text, language=service_language
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
