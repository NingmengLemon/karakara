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
            {
                "id": 3,
                "cmd": "separate",
                "audio": "a.mp3",
                "dest_dir": "d",
                "stems": "vocals",
            },
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


# --------------------------------------------------------------------------
# 解码容错
# --------------------------------------------------------------------------

_MP3_RATE = 44100


def _write_mp3(path: Path, *, seconds: float = 1.5) -> Path:
    """写一个确定性的合成 mp3（固定随机种子，内容可复现）。"""
    import av
    import numpy as np

    rng = np.random.default_rng(0)
    samples = (rng.standard_normal((2, int(_MP3_RATE * seconds))) * 0.2).astype(
        np.float32
    )
    with av.open(str(path), "w", format="mp3") as container:
        stream = container.add_stream("libmp3lame", rate=_MP3_RATE)
        # add_stream 的返回类型是 Video/Audio/SubtitleStream 的联合；按模板名收窄，
        # 免得 mypy 与 ty 都对着 .layout/.encode 报 union-attr。
        assert isinstance(stream, av.AudioStream)
        stream.layout = "stereo"
        frame = av.AudioFrame.from_ndarray(samples, format="fltp", layout="stereo")
        frame.sample_rate = _MP3_RATE
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    return path


@pytest.fixture(scope="module")
def corrupt_mp3(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """一个中段被覆写成垃圾的 mp3，用来模拟真实曲库里的损坏文件。

    损坏方式**是实测选的**：单纯截断尾部不会让 PyAV 报错（只是少解出几帧），
    只有在中段写入非法字节才会稳定产生 ``av.InvalidDataError``
    （实测 ok=58 bad=2）。因此下面的「跳过」断言若因 fixture 失效而失败，
    说明损坏方式不再能造出坏包，而不是产品行为回归。
    """
    av = pytest.importorskip("av", reason="解码容错测试需要 PyAV")
    assert "libmp3lame" in av.codecs_available

    path = _write_mp3(tmp_path_factory.mktemp("decode") / "source.mp3")
    data = bytearray(path.read_bytes())
    middle = len(data) // 2
    data[middle : middle + 400] = b"\xff" * 400
    corrupt = path.with_name("corrupt.mp3")
    corrupt.write_bytes(bytes(data))
    return corrupt


def test_corrupt_packets_are_skipped_not_fatal(
    worker: Any, corrupt_mp3: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """回归：零星坏包只能让那一小段音频缺失，不能让整首歌分离失败。

    实测来源：`D:\\MUSIC` 某首 mp3 有 5 个坏包 / 9569 个好包，旧实现直接抛
    ``InvalidDataError`` 导致整首歌失败，而主进程侧的 ``io._decode`` 却能正常解码
    （``skip_invalid`` 默认开），两侧行为不一致。
    """
    with caplog.at_level("WARNING", logger="karakara.separator_worker"):
        audio = worker.decode_audio(corrupt_mp3, _MP3_RATE)

    assert audio.ndim == 2
    assert audio.shape[0] == 2
    assert audio.shape[1] > 0
    assert "跳过" in caplog.text, "坏包被静默吞掉了，应当留下可追溯的告警"


def test_undecodable_file_still_fails_loudly(worker: Any, tmp_path: Path) -> None:
    """不可用的文件仍要报错：容错不能变成「静默产出空音频」。

    实测（``tmp/probe_allbad_mp3.py``）：纯垃圾字节、以及「合法头部 + 其余全覆写」
    这几种构造都在 ``av.open`` 阶段就抛 ``InvalidDataError``，走不到 decode_audio
    末尾那条「所有包都无法解码」的兜底分支。所以这里断言的是用户真正在意的那条
    性质：**失败必须响亮**，而不是返回空数组。
    """
    path = tmp_path / "garbage.mp3"
    path.write_bytes(b"\xff" * 8192)

    with pytest.raises(Exception) as excinfo:
        worker.decode_audio(path, _MP3_RATE)

    assert not isinstance(excinfo.value, AssertionError)
