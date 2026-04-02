"""
q3fa_client.py

Qwen3-ForcedAligner HTTP API 客户端（基于 requests），带类型注解。

API 端点：
- POST /align  — 提交音频+文本进行强制对齐，返回逐词时间戳
- GET  /health — 健康检查

供其他脚本或程序 import 使用。
"""

from __future__ import annotations

import io
import logging
from os import PathLike
from pathlib import Path

import requests
from typing_extensions import TypedDict

logger = logging.getLogger(__name__)


class Q3FAAlignedWord(TypedDict):
    """API 返回的单个对齐词。"""

    text: str
    start_time: float
    end_time: float


class Q3FAResponse(TypedDict):
    """POST /align 的 JSON 响应体。"""

    words: list[Q3FAAlignedWord]


class Q3FAClient:
    """
    Qwen3-ForcedAligner HTTP API client.

    Example:
        client = Q3FAClient("http://localhost:8000")
        result = client.align_bytes(audio_bytes, "你好世界", language="Chinese")
        for w in result["words"]:
            print(w["text"], w["start_time"], w["end_time"])
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: float | None = 120.0,
        session: requests.Session | None = None,
    ) -> None:
        """
        :param base_url: 服务根地址（例如 http://localhost:8000）
        :param timeout: 默认单次请求超时（秒）。对可能耗时很长的对齐请求，可传 None。
        :param session: 可选的 requests.Session（便于复用连接）
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = session or requests.Session()

    def health(self, timeout: float | None = None) -> dict[str, str]:
        """
        健康检查。

        :return: 服务器返回的 JSON（例如 {"status": "ok"}）
        :raises: requests.HTTPError
        """
        url = f"{self.base_url}/health"
        resp = self.session.get(
            url, timeout=(self.timeout if timeout is None else timeout)
        )
        resp.raise_for_status()
        return resp.json()  # type: ignore[no-any-return]

    def align(
        self,
        audio_path: PathLike,
        text: str,
        language: str = "Chinese",
        timeout: float | None = None,
    ) -> Q3FAResponse:
        """
        提交本地音频文件进行强制对齐。

        :param audio_path: 本地音频文件路径（wav/mp3/flac/m4a）
        :param text: 与音频对应的参考文本
        :param language: 语言 (Chinese/English/French/German/...)
        :param timeout: 覆盖默认超时
        :return: AlignResponse
        :raises: requests.HTTPError
        """
        url = f"{self.base_url}/align"
        with open(str(audio_path), "rb") as f:
            files = {"audio": (Path(audio_path).name, f)}
            data: dict[str, str] = {"text": text, "language": language}
            resp = self.session.post(
                url,
                files=files,
                data=data,
                timeout=(self.timeout if timeout is None else timeout),
            )
        resp.raise_for_status()
        return resp.json()  # type: ignore[no-any-return]

    def align_bytes(
        self,
        audio_bytes: bytes,
        text: str,
        language: str = "Chinese",
        filename: str = "upload.wav",
        timeout: float | None = None,
    ) -> Q3FAResponse:
        """
        提交内存中的音频字节进行强制对齐。

        :param audio_bytes: 音频文件的原始字节
        :param text: 与音频对应的参考文本
        :param language: 语言 (Chinese/English/French/German/...)
        :param filename: 上传时使用的文件名
        :param timeout: 覆盖默认超时
        :return: AlignResponse
        :raises: requests.HTTPError
        """
        url = f"{self.base_url}/align"
        bio = io.BytesIO(audio_bytes)
        files = {"audio": (filename, bio)}
        data: dict[str, str] = {"text": text, "language": language}
        resp = self.session.post(
            url,
            files=files,
            data=data,
            timeout=(self.timeout if timeout is None else timeout),
        )
        resp.raise_for_status()
        return resp.json()  # type: ignore[no-any-return]
