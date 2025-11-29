"""
gentle_client.py

轻量的 Gentle HTTP API 客户端（基于 requests），带类型注解。
- 提交音频（同步或异步）
- 轮询 status.json
- 下载 align.json

不包含 CLI，供其他脚本或程序 import 使用。
"""

from __future__ import annotations

import io
import logging
import time
from pathlib import Path
from typing import Any, Union

import requests
from typing_extensions import TypeAlias

logger = logging.getLogger(__name__)
PathLike: TypeAlias = Union[str, Path]


class GentleClient:
    """
    Gentle HTTP API client.

    Example:
        client = GentleClient("http://localhost:8765")
        uid, location = client.submit_async("audio.mp3", "this is transcript")
        status = client.poll_status(uid)
        client.download_align_json(uid, "out/align.json")
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8765",
        timeout: float | None = 30.0,
        session: requests.Session | None = None,
    ) -> None:
        """
        :param base_url: 服务根地址（例如 http://localhost:8765）
        :param timeout: 默认单次请求超时（秒）。对可能耗时很长的同步提交，可传 None。
        :param session: 可选的 requests.Session（便于复用连接）
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = session or requests.Session()

    def _build_submit_fields(
        self, transcript: str = "", disfluency: bool = False, conservative: bool = False
    ) -> dict[str, str]:
        data: dict[str, str] = {}
        if transcript is not None:
            data["transcript"] = transcript
        if disfluency:
            # 服务器只检查字段是否存在
            data["disfluency"] = "1"
        if conservative:
            data["conservative"] = "1"
        return data

    def submit_async(
        self,
        audio_path: PathLike,
        transcript: str = "",
        disfluency: bool = False,
        conservative: bool = False,
        timeout: float | None = None,
    ) -> tuple[str, str]:
        """
        异步提交文件。默认不跟随重定向，以便读取 Location header 获取 uid。

        :param audio_path: 本地音频文件路径
        :param transcript: 可选的参考转录文本
        :param disfluency: 若为 True，会在表单中包含 disfluency 字段
        :param conservative: 若为 True，会在表单中包含 conservative 字段
        :param timeout: 覆盖默认超时；若为 None 则不设置超时（阻塞直到响应或网络错误）
        :return: (uid, location) 例如 ("abcd1234", "/transcriptions/abcd1234")
        :raises: requests.HTTPError 或 RuntimeError
        """
        url = f"{self.base_url}/transcriptions"
        files = {}
        with open(str(audio_path), "rb") as f:
            files = {"audio": (Path(audio_path).name, f)}
            data = self._build_submit_fields(transcript, disfluency, conservative)
            resp = self.session.post(
                url,
                files=files,
                data=data,
                allow_redirects=False,
                timeout=(self.timeout if timeout is None else timeout),
            )
        resp.raise_for_status()
        if resp.status_code in (302,):
            location = resp.headers.get("Location")
            if not location:
                raise RuntimeError("Missing Location header on async submission")
            uid = location.rstrip("/").split("/")[-1]
            return uid, location
        # 某些服务器可能返回 200 + body; 兼容处理:
        if resp.status_code == 200:
            # 尝试解析 body 中的 uid（不保证存在）
            text = resp.text
            # 如果 body 为空或不是预期格式，抛出错误
            raise RuntimeError(f"Unexpected 200 response for async submit: {text}")
        raise RuntimeError(f"Unexpected response code: {resp.status_code}")

    def submit_bytes_async(
        self,
        audio_bytes: bytes,
        filename: str = "upload",
        transcript: str = "",
        disfluency: bool = False,
        conservative: bool = False,
        timeout: float | None = None,
    ) -> tuple[str, str]:
        """
        与 submit_async 等价，但使用内存中的字节作为上传内容。
        """
        url = f"{self.base_url}/transcriptions"
        bio = io.BytesIO(audio_bytes)
        files = {"audio": (filename, bio)}
        data = self._build_submit_fields(transcript, disfluency, conservative)
        resp = self.session.post(
            url,
            files=files,
            data=data,
            allow_redirects=False,
            timeout=(self.timeout if timeout is None else timeout),
        )
        resp.raise_for_status()
        if resp.status_code in (302,):
            location = resp.headers.get("Location")
            if not location:
                raise RuntimeError("Missing Location header on async submission")
            uid = location.rstrip("/").split("/")[-1]
            return uid, location
        if resp.status_code == 200:
            raise RuntimeError(f"Unexpected 200 response for async submit: {resp.text}")
        raise RuntimeError(f"Unexpected response code: {resp.status_code}")

    def submit_sync(
        self,
        audio_path: PathLike,
        transcript: str = "",
        disfluency: bool = False,
        conservative: bool = False,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """
        同步提交：会在服务器完成后直接返回 JSON 结果。将表单字段 async=false 发送给服务器。

        :param audio_path: 本地音频文件路径
        :param transcript: 可选参考转录文本
        :param timeout: 超时时间（秒）。若设置为 None，则请求不会超时（阻塞直到完成或连接错误）
        :return: 服务器返回的 JSON 解析结果
        :raises: requests.HTTPError
        """
        url = f"{self.base_url}/transcriptions"
        with open(str(audio_path), "rb") as f:
            files = {"audio": (Path(audio_path).name, f)}
            data = self._build_submit_fields(transcript, disfluency, conservative)
            data["async"] = "false"
            resp = self.session.post(
                url,
                files=files,
                data=data,
                timeout=(self.timeout if timeout is None else timeout),
            )
        resp.raise_for_status()
        return resp.json()  # type: ignore

    def submit_bytes_sync(
        self,
        audio_bytes: bytes,
        filename: str = "upload",
        transcript: str = "",
        disfluency: bool = False,
        conservative: bool = False,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """
        与 submit_sync 类似，但使用字节数据上传。
        """
        url = f"{self.base_url}/transcriptions"
        bio = io.BytesIO(audio_bytes)
        files = {"audio": (filename, bio)}
        data = self._build_submit_fields(transcript, disfluency, conservative)
        data["async"] = "false"
        resp = self.session.post(
            url,
            files=files,
            data=data,
            timeout=(self.timeout if timeout is None else timeout),
        )
        resp.raise_for_status()
        data = resp.json()
        return data  # type: ignore

    def get_status(self, uid: str, timeout: float | None = None) -> dict[str, Any]:
        """
        直接获取 status.json 的内容（如果存在）。

        :param uid: transcription uid
        :return: status 字典（JSON 解析结果）
        """
        url = f"{self.base_url}/transcriptions/{uid}/status.json"
        resp = self.session.get(
            url, timeout=(self.timeout if timeout is None else timeout)
        )
        resp.raise_for_status()
        return resp.json()  # type: ignore

    def poll_status(
        self, uid: str, interval: float = 2.0, timeout: float = 300.0
    ) -> dict[str, Any]:
        """
        轮询 status.json，直到 status 字段为 OK 或 ERROR，或超时。

        :param uid: transcription uid
        :param interval: 轮询间隔（秒）
        :param timeout: 最大等待时间（秒）
        :return: 最终的 status JSON 字典
        :raises: TimeoutError
        """
        url = f"{self.base_url}/transcriptions/{uid}/status.json"
        start = time.time()
        while True:
            resp = self.session.get(url, timeout=self.timeout)
            if resp.status_code == 200:
                status = resp.json()
                st = str(status.get("status", "")).upper()
                if st in ("OK", "ERROR"):
                    return status  # type: ignore
            else:
                resp.raise_for_status()
            if (time.time() - start) > timeout:
                raise TimeoutError(
                    f"Polling status for {uid} timed out after {timeout} seconds"
                )
            time.sleep(interval)

    def download_align_json(
        self, uid: str, dest_path: PathLike, timeout: float | None = None
    ) -> Path:
        """
        下载 align.json 到本地。

        :param uid: transcription uid
        :param dest_path: 本地目标路径
        :return: Path 对象
        """
        url = f"{self.base_url}/transcriptions/{uid}/align.json"
        resp = self.session.get(
            url, timeout=(self.timeout if timeout is None else timeout)
        )
        resp.raise_for_status()
        p = Path(dest_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(resp.content)
        return p
