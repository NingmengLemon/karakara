"""worker 脚本的协议循环单元测试。

这些测试直接 import `scripts/` 下的 worker 脚本并替换掉后端，因此不需要 torch、
不需要 GPU、不下载任何模型。它们盯的是几个只有真实运行才会暴露的行为：

* 依赖用 ``sys.exit()`` 报错时（demucs 缺 diffq 就是这样）worker 必须回一条
  可读的错误，而不是直接死掉——``SystemExit`` 继承自 ``BaseException``，
  ``except Exception`` 抓不到它。
* 分离期间往 stdout 漏的任何东西都必须被赶去 stderr，否则协议通道会被污染。
"""

from __future__ import annotations

import importlib.util
import io
import json
import sys
from pathlib import Path
from typing import Any

import pytest

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def load_script(name: str) -> Any:
    """把 ``scripts/<name>`` 作为独立模块导入。"""
    path = _SCRIPTS / name
    spec = importlib.util.spec_from_file_location(f"_worker_{path.stem}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def worker() -> Any:
    return load_script("separator_worker.py")


@pytest.fixture(scope="module")
def audio_worker() -> Any:
    return load_script("separator_worker_audio_separator.py")


class FakeBackend:
    """可编程的假后端。"""

    def __init__(self, behaviour: str, payload: dict[str, Any] | None = None) -> None:
        self.behaviour = behaviour
        self.payload = payload or {"stems": {"vocals": "x.wav"}, "samplerate": 8000}
        self.seen: list[dict[str, Any]] = []

    def separate(self, **kwargs: Any) -> dict[str, Any]:
        self.seen.append(kwargs)
        if self.behaviour == "systemexit":
            raise SystemExit(1)
        if self.behaviour == "exception":
            raise ValueError("模型不存在")
        if self.behaviour == "noisy":
            # 模拟第三方库往 stdout 乱打印
            print("library banner on stdout")
        return dict(self.payload)

    def info(self, model_dir: str | None) -> dict[str, Any]:
        return {"backend": "fake", "model_dir": model_dir}


def run_requests(
    worker: Any,
    backend: Any,
    requests: list[dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[dict[str, Any]], str]:
    """喂入若干请求，返回 (响应列表, 协议 stdout 的原始文本)。"""
    stdin = io.StringIO("\n".join(json.dumps(item) for item in requests) + "\n")
    captured = io.StringIO()
    monkeypatch.setattr(sys, "stdin", stdin)
    monkeypatch.setattr(sys, "stdout", captured)
    worker.serve(backend)
    lines = [json.loads(line) for line in captured.getvalue().splitlines() if line]
    return lines, captured.getvalue()


def test_system_exit_becomes_a_readable_error(
    worker: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """回归：依赖 ``sys.exit()`` 时 worker 不能静默死掉。"""
    backend = FakeBackend("systemexit")
    responses, _ = run_requests(
        worker,
        backend,
        [{"id": 1, "cmd": "separate", "audio": "a.mp3", "dest_dir": "d"}],
        monkeypatch,
    )

    assert len(responses) == 1
    assert responses[0]["ok"] is False
    assert "sys.exit" in responses[0]["error"]


def test_failed_request_does_not_stop_the_loop(
    worker: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """一首歌失败不能让整批任务陪葬：后续请求仍要被处理。"""
    backend = FakeBackend("exception")
    responses, _ = run_requests(
        worker,
        backend,
        [
            {"id": 1, "cmd": "separate", "audio": "a.mp3", "dest_dir": "d"},
            {"id": 2, "cmd": "info"},
        ],
        monkeypatch,
    )

    assert [item["ok"] for item in responses] == [False, True]
    assert "模型不存在" in responses[0]["error"]


def test_stray_stdout_is_redirected_off_the_protocol_channel(
    worker: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """分离期间第三方库的 stdout 输出不能污染协议通道。"""
    backend = FakeBackend("noisy")
    responses, raw = run_requests(
        worker,
        backend,
        [{"id": 1, "cmd": "separate", "audio": "a.mp3", "dest_dir": "d"}],
        monkeypatch,
    )

    assert len(responses) == 1 and responses[0]["ok"] is True
    assert "library banner" not in raw


def test_unknown_command_and_bad_payload_are_rejected(
    worker: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = FakeBackend("ok")
    responses, _ = run_requests(
        worker,
        backend,
        [
            {"id": 1, "cmd": "frobnicate"},
            {"id": 2, "cmd": "separate"},
            {"id": 3, "cmd": "separate", "audio": "a.mp3", "dest_dir": "d", "stems": "vocals"},
        ],
        monkeypatch,
    )

    assert all(item["ok"] is False for item in responses)
    assert "未知命令" in responses[0]["error"]
    assert "需要 audio" in responses[1]["error"]
    assert "stems 必须是字符串数组" in responses[2]["error"]


def test_shutdown_stops_the_loop_and_replies(
    worker: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = FakeBackend("ok")
    responses, _ = run_requests(
        worker,
        backend,
        [
            {"id": 1, "cmd": "shutdown"},
            {"id": 2, "cmd": "info"},
        ],
        monkeypatch,
    )

    assert len(responses) == 1, "收到 shutdown 之后不应再处理后续请求"
    assert responses[0]["ok"] is True


def test_invalid_json_and_non_object_requests_are_reported(
    worker: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    stdin = io.StringIO('not json\n[1, 2, 3]\n{"id": 9, "cmd": "shutdown"}\n')
    captured = io.StringIO()
    monkeypatch.setattr(sys, "stdin", stdin)
    monkeypatch.setattr(sys, "stdout", captured)

    worker.serve(FakeBackend("ok"))

    responses = [json.loads(line) for line in captured.getvalue().splitlines() if line]
    assert responses[0]["error"].startswith("非法 JSON")
    assert responses[1]["error"] == "请求必须是 JSON 对象"
    assert responses[2]["ok"] is True


def test_separate_forwards_request_fields(
    worker: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """worker 必须把模型/设备/目录/音轨白名单原样转给后端。"""
    backend = FakeBackend("ok")
    responses, _ = run_requests(
        worker,
        backend,
        [
            {
                "id": 1,
                "cmd": "separate",
                "audio": "a.mp3",
                "dest_dir": "d",
                "stems": ["vocals"],
                "model": "htdemucs_6s",
                "device": "cpu",
                "model_dir": "models/x",
            }
        ],
        monkeypatch,
    )

    assert responses[0]["ok"] is True
    assert responses[0]["elapsed_s"] >= 0
    assert backend.seen[0]["model"] == "htdemucs_6s"
    assert backend.seen[0]["device"] == "cpu"
    assert backend.seen[0]["model_dir"] == "models/x"
    assert backend.seen[0]["stems"] == ["vocals"]


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("track_(Vocals)_model.wav", "vocals"),
        ("track_(Instrumental)_model.wav", "instrumental"),
        ("song_(Drums)_htdemucs.wav", "drums"),
        ("mystery_output.wav", None),
    ],
)
def test_audio_separator_stem_classification(
    audio_worker: Any, filename: str, expected: str | None
) -> None:
    """输出文件名的归类不能依赖某个固定命名模板（各版本模板不同）。"""
    assert audio_worker.classify_stem(filename) == expected
