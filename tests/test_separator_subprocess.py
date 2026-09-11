"""`SubprocessStemSeparator` 的行协议与生命周期测试。

这些测试用「假 worker」而不是真 Demucs：协议契约必须能在没有 torch、没有 GPU、
不下载任何模型的机器上被验证。
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from karakara.separator import StemSeparationError, SubprocessStemSeparator

VOCALS = "vocals"
STEM_SAMPLES = np.linspace(-0.5, 0.5, 64, dtype=np.float32)


@pytest.fixture
def audio(tmp_path: Path) -> Path:
    """一个真实存在的最小音频文件。

    分离器会在启动 worker **之前**就校验源文件存在（避免白起一个进程），
    因此测试必须给出真文件；假 worker 只关心路径，不关心内容。
    """
    path = tmp_path / "source.wav"
    sf.write(str(path), np.zeros(128, dtype=np.float32), 8000, subtype="FLOAT")
    return path


def write_fake_worker(tmp_path: Path, body: str) -> Path:
    """生成一个假 worker 脚本，返回其路径。"""
    script = tmp_path / "fake_worker.py"
    script.write_text(
        textwrap.dedent(
            """
            import json
            import sys
            from pathlib import Path

            import numpy as np
            import soundfile as sf

            STEM_SAMPLES = np.linspace(-0.5, 0.5, 64, dtype=np.float32)
            """
        ).lstrip()
        + textwrap.dedent(body),
        encoding="utf-8",
    )
    return script


def worker_command(script: Path) -> list[str]:
    return [sys.executable, str(script)]


COMPLIANT_WORKER = """
    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        request = json.loads(raw)
        if request["cmd"] == "shutdown":
            print(json.dumps({"id": request["id"], "ok": True}), flush=True)
            break
        if request["cmd"] == "info":
            print(json.dumps({"id": request["id"], "ok": True, "backend": "fake"}), flush=True)
            continue
        dest = Path(request["dest_dir"])
        dest.mkdir(parents=True, exist_ok=True)
        wanted = request.get("stems") or ["vocals", "drums"]
        written = {}
        for name in wanted:
            path = dest / f"{name}.wav"
            sf.write(str(path), STEM_SAMPLES, 8000, subtype="FLOAT")
            written[name] = str(path)
        print(
            json.dumps(
                {
                    "id": request["id"],
                    "ok": True,
                    "stems": written,
                    "samplerate": 8000,
                    "elapsed_s": 0.01,
                }
            ),
            flush=True,
        )
"""


def test_separate_returns_readable_stem_paths(tmp_path: Path, audio: Path) -> None:
    script = write_fake_worker(tmp_path, COMPLIANT_WORKER)
    separator = SubprocessStemSeparator(worker_command(script))
    try:
        stems = separator.separate(audio, tmp_path / "out", stems=[VOCALS])

        assert set(stems) == {VOCALS}
        data, rate = sf.read(str(stems[VOCALS]), dtype="float32")
        assert rate == 8000
        np.testing.assert_allclose(data, STEM_SAMPLES, rtol=0, atol=1e-6)
    finally:
        separator.close()


def test_worker_process_is_reused_across_requests(tmp_path: Path, audio: Path) -> None:
    """模型只加载一次的前提：进程必须跨请求常驻，而不是每首歌重启。"""
    script = write_fake_worker(tmp_path, COMPLIANT_WORKER)
    separator = SubprocessStemSeparator(worker_command(script))
    try:
        separator.separate(audio, tmp_path / "o1", stems=[VOCALS])
        process = separator._process
        assert process is not None, "首次请求之后 worker 必须已经启动"
        pid_after_first = process.pid

        separator.separate(audio, tmp_path / "o2", stems=[VOCALS])

        # 断言同一个进程对象仍然存活：这才是"模型只加载一次"的前提
        assert separator._process is process, "worker 不应在两次请求之间被重建"
        assert process.poll() is None, "worker 不应在两次请求之间退出"
        assert process.pid == pid_after_first
    finally:
        separator.close()


def test_separate_forwards_whitelist_and_model_options(
    tmp_path: Path, audio: Path
) -> None:
    script = write_fake_worker(
        tmp_path,
        COMPLIANT_WORKER.replace(
            'wanted = request.get("stems") or ["vocals", "drums"]',
            "wanted = request.get('stems') or ['vocals', 'drums']\n"
            "        assert request.get('model') == 'htdemucs_6s'\n"
            "        assert request.get('device') == 'cpu'\n"
            "        assert request.get('model_dir') == 'models/x'",
        ),
    )
    separator = SubprocessStemSeparator(
        worker_command(script), model="htdemucs_6s", device="cpu", model_dir="models/x"
    )
    try:
        stems = separator.separate(audio, tmp_path / "out", stems=[VOCALS])
        assert set(stems) == {VOCALS}
    finally:
        separator.close()


def test_missing_vocal_stem_is_an_error(tmp_path: Path, audio: Path) -> None:
    """只输出了 drums 时必须报错，而不是让下游拿到 KeyError。"""
    script = write_fake_worker(
        tmp_path,
        COMPLIANT_WORKER.replace(
            'wanted = request.get("stems") or ["vocals", "drums"]',
            'wanted = ["drums"]  # 故意不给 vocals',
        ),
    )
    separator = SubprocessStemSeparator(worker_command(script))
    try:
        with pytest.raises(StemSeparationError, match="缺少 'vocals'"):
            separator.separate(audio, tmp_path / "out", stems=[VOCALS])
    finally:
        separator.close()


def test_worker_reported_failure_propagates(tmp_path: Path, audio: Path) -> None:
    script = write_fake_worker(
        tmp_path,
        """
        for raw in sys.stdin:
            request = json.loads(raw)
            if request["cmd"] == "shutdown":
                break
            print(
                json.dumps(
                    {"id": request["id"], "ok": False, "error": "模型不存在"}
                ),
                flush=True,
            )
        """,
    )
    separator = SubprocessStemSeparator(worker_command(script))
    try:
        with pytest.raises(StemSeparationError, match="模型不存在"):
            separator.separate(audio, tmp_path / "out")
    finally:
        separator.close()


def test_worker_exiting_without_response_is_an_error(
    tmp_path: Path, audio: Path
) -> None:
    """worker 被 OOM 干掉时必须立刻报错，而不是永久挂住。

    同时要求报出**真实退出码**：不 reap 进程的话 ``poll()`` 常常还返回 ``None``，
    错误信息会退化成没用的 ``returncode=None``。
    """
    script = write_fake_worker(tmp_path, "")
    separator = SubprocessStemSeparator(worker_command(script))
    try:
        with pytest.raises(StemSeparationError, match=r"提前退出（returncode=0）"):
            separator.separate(audio, tmp_path / "out")
    finally:
        separator.close()


def test_worker_failing_with_nonzero_code_reports_that_code(
    tmp_path: Path, audio: Path
) -> None:
    script = write_fake_worker(tmp_path, "raise SystemExit(3)\n")
    separator = SubprocessStemSeparator(worker_command(script))
    try:
        with pytest.raises(StemSeparationError, match=r"returncode=3"):
            separator.separate(audio, tmp_path / "out")
    finally:
        separator.close()


def test_claimed_but_missing_stem_file_is_an_error(tmp_path: Path, audio: Path) -> None:
    """worker 谎报写出了文件时不能当成成功。"""
    script = write_fake_worker(
        tmp_path,
        """
        for raw in sys.stdin:
            request = json.loads(raw)
            if request["cmd"] == "shutdown":
                break
            print(
                json.dumps(
                    {
                        "id": request["id"],
                        "ok": True,
                        "stems": {"vocals": str(Path(request["dest_dir"]) / "nope.wav")},
                    }
                ),
                flush=True,
            )
        """,
    )
    separator = SubprocessStemSeparator(worker_command(script))
    try:
        with pytest.raises(StemSeparationError, match="文件不存在"):
            separator.separate(audio, tmp_path / "out")
    finally:
        separator.close()


def test_stray_non_protocol_stdout_is_tolerated(tmp_path: Path, audio: Path) -> None:
    """第三方库往 stdout 漏一行日志不应让整个分离失败。"""
    script = write_fake_worker(
        tmp_path,
        """
        for raw in sys.stdin:
            request = json.loads(raw)
            if request["cmd"] == "shutdown":
                break
            print("some library banner on stdout", flush=True)
            dest = Path(request["dest_dir"])
            dest.mkdir(parents=True, exist_ok=True)
            path = dest / "vocals.wav"
            sf.write(str(path), STEM_SAMPLES, 8000, subtype="FLOAT")
            print(
                json.dumps(
                    {"id": request["id"], "ok": True, "stems": {"vocals": str(path)}}
                ),
                flush=True,
            )
        """,
    )
    separator = SubprocessStemSeparator(worker_command(script))
    try:
        stems = separator.separate(audio, tmp_path / "out", stems=[VOCALS])
        assert set(stems) == {VOCALS}
    finally:
        separator.close()


def test_corrupted_json_response_is_a_protocol_error(
    tmp_path: Path, audio: Path
) -> None:
    script = write_fake_worker(
        tmp_path,
        """
        for raw in sys.stdin:
            request = json.loads(raw)
            if request["cmd"] == "shutdown":
                break
            print('{"id": 1, "ok": tru', flush=True)
        """,
    )
    separator = SubprocessStemSeparator(worker_command(script))
    try:
        with pytest.raises(StemSeparationError, match="JSON 无法解析"):
            separator.separate(audio, tmp_path / "out")
    finally:
        separator.close()


def test_missing_source_audio_is_rejected_before_spawning(
    tmp_path: Path, audio: Path
) -> None:
    separator = SubprocessStemSeparator([sys.executable, "does-not-matter.py"])
    try:
        with pytest.raises(StemSeparationError, match="源音频不存在"):
            separator.separate(tmp_path / "nope.mp3", tmp_path / "out")
        assert separator._process is None, "校验失败时不应启动 worker"
    finally:
        separator.close()


def test_missing_executable_reports_actionable_error(
    tmp_path: Path, audio: Path
) -> None:
    separator = SubprocessStemSeparator(["definitely-not-a-real-binary-xyz"])
    try:
        with pytest.raises(StemSeparationError, match="--separator-cmd"):
            separator.separate(audio, tmp_path / "out")
    finally:
        separator.close()


def test_info_command_round_trip(tmp_path: Path, audio: Path) -> None:
    script = write_fake_worker(tmp_path, COMPLIANT_WORKER)
    separator = SubprocessStemSeparator(worker_command(script))
    try:
        assert separator.info()["backend"] == "fake"
    finally:
        separator.close()


def test_close_is_idempotent_and_kills_worker(tmp_path: Path, audio: Path) -> None:
    script = write_fake_worker(tmp_path, COMPLIANT_WORKER)
    separator = SubprocessStemSeparator(worker_command(script))
    separator.separate(audio, tmp_path / "out", stems=[VOCALS])
    process = separator._process
    assert process is not None

    separator.close()
    separator.close()

    assert process.poll() is not None, "close() 之后 worker 必须已退出"


def test_request_after_close_is_rejected(tmp_path: Path, audio: Path) -> None:
    script = write_fake_worker(tmp_path, COMPLIANT_WORKER)
    separator = SubprocessStemSeparator(worker_command(script))
    separator.close()

    with pytest.raises(StemSeparationError, match="已关闭"):
        separator.separate(audio, tmp_path / "out")


def test_command_comes_from_env_when_not_given(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = write_fake_worker(tmp_path, COMPLIANT_WORKER)
    monkeypatch.setenv("KARAKARA_SEPARATOR_CMD", f'"{sys.executable}" "{script}"')

    separator = SubprocessStemSeparator()
    try:
        assert separator.command == [sys.executable, str(script)]
    finally:
        separator.close()
