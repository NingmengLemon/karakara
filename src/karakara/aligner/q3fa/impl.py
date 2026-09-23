from __future__ import annotations

from io import BytesIO
from typing import Any, override

import numpy as np
import requests

from karakara.typ import NpAudioData, NpAudioSamples
from karakara.utils.io import DEFAULT_SAMPLE_RATE, save_audio

from ..abc import AbstractAligner, AlignedWord, LangCode
from .client import Q3FAAlignedWord, Q3FAClient

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

#: 单次对齐请求的默认超时（秒）。置 ``None`` 表示不超时。
#: 默认**有限**是刻意的：批处理时一个卡住的服务不应该让主程序永久挂住。
DEFAULT_TIMEOUT_S: float | None = 120.0


class Q3FAProtocolError(RuntimeError):
    """服务端响应不符合契约形状。"""


def extract_words(response: Any) -> list[Q3FAAlignedWord]:
    """从 ``/align`` 的响应里取出词列表。

    契约是「单个文件 → 对象，多个文件 → 数组」。这里同时容忍「长度为 1 的数组」
    ——旧版服务端用联合类型签名时对单文件也会返回数组。这一点必须显式处理而不是
    想当然：否则客户端会在 ``response["words"]`` 上抛 ``TypeError``，而
    ``core._align_line`` 会把它当成「这一行对齐失败」逐行吞掉，最终产出一个
    **没有任何词级时间戳、退出码却是 0** 的文件。

    Raises:
        Q3FAProtocolError: 响应既不是对象、也不是长度为 1 的数组。
    """
    if isinstance(response, dict):
        words = response.get("words")
        if not isinstance(words, list):
            raise Q3FAProtocolError(f"响应缺少 words 数组: {response!r}")
        return words
    if isinstance(response, list):
        if len(response) != 1:
            raise Q3FAProtocolError(
                f"只提交了一个音频，服务端却返回了 {len(response)} 组结果；"
                f"请检查服务端版本是否与客户端匹配"
            )
        return extract_words(response[0])
    raise Q3FAProtocolError(
        f"无法识别的响应类型 {type(response).__name__}: {response!r}"
    )


class HttpAligner(AbstractAligner):
    """通过 HTTP `/align` 契约访问对齐服务的适配器。

    模块路径里的 ``q3fa`` 是历史名字：这个适配器**不绑定具体后端**——Qwen3-ForcedAligner
    与 HubertFA 的服务提供同一套契约（音频 + 文本 + 语言 → 逐单元 `(text, start, end)`），
    由 ``main.py --aligner-backend`` 决定连哪个端口。旧类名 :class:`Qwen3ForcedAligner`
    保留为别名。
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8787",
        language: str = DEFAULT_LANGUAGE,
        session: requests.Session | None = None,
        timeout: float | None = DEFAULT_TIMEOUT_S,
    ) -> None:
        """
        Args:
            base_url: 服务根地址（例如 http://localhost:8787）。
            language: 未显式指定语言时使用的服务端语言名。
            session: 可选的 requests.Session。
            timeout: 单次对齐请求超时（秒）。``None`` = 不超时（默认 120s）。
        """
        self._client = Q3FAClient(base_url=base_url, timeout=timeout, session=session)
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
        # 送 float32 WAV：分离链路刻意全程 32 位浮点（见 docs/current/pipeline.md），
        # 在这里退回 16 位等于在下游预处理之前又引入一次量化。已实测两种载荷
        # 在 HubertFA 上的逐单元边界完全一致（见该 commit 的 A/B 记录）。
        save_audio(buffer, audio, sample_rate, subtype="float32")
        response = self._client.align_bytes(
            buffer.getvalue(), text=text, language=service_language
        )

        result: list[AlignedWord] = []
        for w in extract_words(response):
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


#: 旧类名（这个适配器最初只为 Qwen3-ForcedAligner 服务而写）。
Qwen3ForcedAligner = HttpAligner
