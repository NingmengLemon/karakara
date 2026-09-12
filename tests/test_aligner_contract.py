"""对齐服务的响应契约测试（客户端 × 服务端）。

这里盯的是一个真实发生过的静默失效：

* 服务端用 ``audio: list[UploadFile] | UploadFile`` 这样的联合类型签名时，
  FastAPI 对**单个**上传也会走 list 分支，于是 ``/align`` 返回**数组**；
* 客户端读 ``response["words"]``，直接 ``TypeError``；
* ``core._align_line`` 会逐行吞掉这个异常并把该行原样保留，最终产出一个
  「没有任何词级时间戳、退出码却是 0」的文件。

服务端那半边用 stub 模块导入真实脚本（不需要 fastapi / qwen-asr / torch），
这样 ``is_batch`` 的判定规则本身也能被回归测试盯住。
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from karakara.aligner.q3fa.impl import (
    Q3FAProtocolError,
    Qwen3ForcedAligner,
    extract_words,
)

_SERVER_PATH = (
    Path(__file__).resolve().parent.parent / "scripts" / "qwen3aligner_server.py"
)


# ==========================================================================
# 客户端：响应形状归一化
# ==========================================================================


def test_extract_words_accepts_the_object_form() -> None:
    words = extract_words(
        {"words": [{"text": "a", "start_time": 0.0, "end_time": 0.5}]}
    )

    assert [w["text"] for w in words] == ["a"]


def test_extract_words_accepts_a_single_element_array() -> None:
    """旧/新版服务端返回数组时也要能用——这正是线上旧的 8787 服务的行为。"""
    words = extract_words(
        [{"words": [{"text": "a", "start_time": 0.0, "end_time": 0.5}]}]
    )

    assert [w["text"] for w in words] == ["a"]


def test_extract_words_rejects_a_multi_element_array() -> None:
    with pytest.raises(Q3FAProtocolError, match="只提交了一个音频"):
        extract_words([{"words": []}, {"words": []}])


def test_extract_words_rejects_a_missing_words_key() -> None:
    with pytest.raises(Q3FAProtocolError, match="缺少 words"):
        extract_words({"nope": []})


def test_extract_words_rejects_an_unexpected_type() -> None:
    with pytest.raises(Q3FAProtocolError, match="无法识别"):
        extract_words("not json")


def test_aligner_handles_an_array_response(monkeypatch: pytest.MonkeyPatch) -> None:
    """端到端形状：客户端收到数组时不应再抛 TypeError。"""
    aligner = Qwen3ForcedAligner()
    monkeypatch.setattr(
        aligner._client,
        "align_bytes",
        lambda *_args, **_kwargs: [
            {"words": [{"text": "你", "start_time": 0.5, "end_time": 0.75}]}
        ],
    )
    try:
        words = aligner.align(np.zeros((1, 1000), dtype=np.float32), "你", 1000)
    finally:
        aligner.close()

    assert [(w.word, w.position) for w in words] == [("你", (500, 750))]


def test_aligner_handles_the_object_response(monkeypatch: pytest.MonkeyPatch) -> None:
    aligner = Qwen3ForcedAligner()
    monkeypatch.setattr(
        aligner._client,
        "align_bytes",
        lambda *_args, **_kwargs: {
            "words": [{"text": "hi", "start_time": 0.0, "end_time": 0.5}]
        },
    )
    try:
        words = aligner.align(np.zeros((1, 1000), dtype=np.float32), "hi", 1000)
    finally:
        aligner.close()

    assert [(w.word, w.position) for w in words] == [("hi", (0, 500))]


def test_aligner_has_a_finite_default_timeout() -> None:
    """默认必须有限超时：一个卡住的服务不该让批处理永久挂住。"""
    aligner = Qwen3ForcedAligner()
    try:
        assert aligner._client.timeout == 120.0
    finally:
        aligner.close()


def test_aligner_timeout_can_be_disabled_explicitly() -> None:
    aligner = Qwen3ForcedAligner(timeout=None)
    try:
        assert aligner._client.timeout is None
    finally:
        aligner.close()


# ==========================================================================
# 服务端：单文件必须返回对象
# ==========================================================================


class _FakeUpload:
    """够用的 UploadFile 替身。"""

    def __init__(self, name: str, payload: bytes = b"RIFF") -> None:
        import io

        self.filename = name
        self.file = io.BytesIO(payload)


def _stub_module(name: str, **attrs: Any) -> types.ModuleType:
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def _load_server(monkeypatch: pytest.MonkeyPatch) -> Any:
    """导入真实的 aligner 服务脚本，并 stub 掉它那份重量级依赖。

    对齐器的 ``align()`` 被换成一个返回固定结果的替身：这里要验证的是**响应形状
    契约**，与模型权重无关。已安装的真实依赖（pydantic）优先使用。
    """
    stubs: dict[str, types.ModuleType] = {}

    if "fastapi" not in sys.modules:

        class _FakeHTTPException(Exception):
            def __init__(self, status_code: int = 500, detail: str = "") -> None:
                super().__init__(detail)
                self.status_code = status_code
                self.detail = detail

        class _FakeApp:
            def post(self, *_a: Any, **_k: Any) -> Any:
                return lambda fn: fn

            def get(self, *_a: Any, **_k: Any) -> Any:
                return lambda fn: fn

        stubs["fastapi"] = _stub_module(
            "fastapi",
            FastAPI=lambda **_k: _FakeApp(),
            File=lambda default=None, **_k: default,
            Form=lambda default=None, **_k: default,
            HTTPException=_FakeHTTPException,
            UploadFile=object,
        )
    if "qwen_asr" not in sys.modules:
        stubs["qwen_asr"] = _stub_module("qwen_asr", Qwen3ForcedAligner=object)
    if "torch" not in sys.modules:
        stubs["torch"] = _stub_module(
            "torch",
            __version__="0.0.0+stub",
            bfloat16="bfloat16",
            float32="float32",
            version=types.SimpleNamespace(cuda=None),
            cuda=types.SimpleNamespace(is_available=lambda: False),
        )
    if "uvicorn" not in sys.modules:
        stubs["uvicorn"] = _stub_module("uvicorn", run=lambda *_a, **_k: None)

    sys.modules.update(stubs)
    module_name = "_aligner_server_under_test"
    try:
        spec = importlib.util.spec_from_file_location(module_name, _SERVER_PATH)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        # 必须先登记进 sys.modules：pydantic 解析 AlignResponse 里的 AlignedWord
        # 前向引用时要靠它回查模块命名空间。
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    finally:
        for name in stubs:
            sys.modules.pop(name, None)

    # 用替身顶掉真模型
    class _Word:
        def __init__(self, text: str) -> None:
            self.text = text
            self.start_time = 0.1
            self.end_time = 0.2

    class _StubAligner:
        def align(
            self, *, audio: list[str], text: list[str], language: list[str]
        ) -> Any:
            assert len(audio) == len(text) == len(language), "长度必须已经对齐"
            return [[_Word("w")] for _ in audio]

    monkeypatch.setattr(module, "_aligner", _StubAligner())
    return module


@pytest.mark.asyncio
async def test_server_returns_object_for_a_single_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """单个文件 → 对象。这正是客户端 ``response["words"]`` 依赖的形状。"""
    server = _load_server(monkeypatch)

    result = server.align(audio=_FakeUpload("a.wav"), text="hello")

    # FastAPI 会把它序列化成 {"words": [...]}，即客户端 response["words"] 需要的形状。
    assert not isinstance(result, list), (
        "单文件必须返回对象；返回数组会让客户端的 response['words'] 抛 TypeError"
    )
    assert isinstance(result, server.AlignResponse)
    assert [w.text for w in result.words] == ["w"]


@pytest.mark.asyncio
async def test_server_returns_array_for_multiple_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = _load_server(monkeypatch)

    result = server.align(
        audio=[_FakeUpload("a.wav"), _FakeUpload("b.wav")],
        text=["one", "two"],
        language="Chinese",
    )

    assert isinstance(result, list)
    assert len(result) == 2


@pytest.mark.asyncio
async def test_server_broadcasts_single_language_and_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """单个 language 应被广播到每个音频，而不是只作用于第一个。"""
    server = _load_server(monkeypatch)

    result = server.align(
        audio=[_FakeUpload("a.wav"), _FakeUpload("b.wav")],
        text="same text",
        language="Japanese",
    )

    assert len(result) == 2


@pytest.mark.asyncio
async def test_server_rejects_mismatched_text_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = _load_server(monkeypatch)

    with pytest.raises(Exception, match="不匹配"):
        server.align(
            audio=[_FakeUpload("a.wav"), _FakeUpload("b.wav")],
            text=["one", "two", "three"],
        )


def test_align_endpoint_is_sync_so_inference_cannot_wedge_the_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """端点必须是同步 `def`。

    推理是阻塞的：写成 ``async def`` 就会在事件循环里跑完整个生成过程，于是实测
    一个异常耗时的请求把 ``/health`` 和所有后续请求一起堵死，客户端超时断开后服务端
    还在算，最后连监听套接字都废掉（WinError 64）。同步端点由 FastAPI 丢进线程池。
    """
    import inspect

    server = _load_server(monkeypatch)

    assert not inspect.iscoroutinefunction(server.align), (
        "align 端点退化回 async def 会让一个慢请求堵死整个服务"
    )
    # 推理本身要串行（GPU 并发只会互相抢显存），但锁不能放在事件循环里。
    assert hasattr(server, "_INFERENCE_LOCK")
