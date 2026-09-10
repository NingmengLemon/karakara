"""端到端集成测试：真实分离 worker + 真实对齐服务。

默认跳过——它们需要模型文件、GPU 以及（对对齐部分而言）一个正在运行的服务。
需要时手动运行：

    $env:KARAKARA_RUN_INTEGRATION = "1"
    uv run pytest -m integration

分离 worker 默认用 ``uv run --script`` 启动。若已经有一个装好 demucs 的解释器，
用 ``KARAKARA_SEPARATOR_CMD`` 指过去可以避免 uv 准备环境：

    $env:KARAKARA_SEPARATOR_CMD = '"E:\\path\\to\\python.exe" "scripts/separator_worker.py"'
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from karakara.separator import SubprocessStemSeparator
from karakara.utils.io import load_audio_native

pytestmark = pytest.mark.integration

_SAMPLE_AUDIO = Path("samples/ashen.mp3")

requires_integration = pytest.mark.skipif(
    os.environ.get("KARAKARA_RUN_INTEGRATION") != "1",
    reason="set KARAKARA_RUN_INTEGRATION=1 to run local separator integration tests",
)


@requires_integration
def test_worker_reports_backend_info(tmp_path: Path) -> None:
    """worker 能被拉起并自报后端信息。"""
    separator = SubprocessStemSeparator()
    try:
        info = separator.info()
    finally:
        separator.close()

    assert info["backend"] == "demucs"


@requires_integration
@pytest.mark.skipif(
    not _SAMPLE_AUDIO.is_file(), reason=f"sample audio missing: {_SAMPLE_AUDIO}"
)
def test_worker_separates_sample_audio_to_float_wav(tmp_path: Path) -> None:
    """真实分离一次，并验证输出是可直接读回的浮点人声轨。"""
    separator = SubprocessStemSeparator()
    try:
        stems = separator.separate(_SAMPLE_AUDIO, tmp_path, stems=["vocals"])
    finally:
        separator.close()

    assert set(stems) == {"vocals"}
    vocal_path = stems["vocals"]
    assert vocal_path.is_file() and vocal_path.stat().st_size > 0

    info = sf.info(str(vocal_path))
    assert info.subtype == "FLOAT", "必须是 32 位浮点，避免下游预处理前的量化损失"
    assert info.samplerate == 44100

    audio, sample_rate = load_audio_native(vocal_path)
    assert sample_rate == 44100
    assert audio.ndim == 2
    assert audio.shape[0] in (1, 2)
    assert audio.shape[1] > 0
    assert np.abs(audio).max() > 0, "人声轨不应是静音"
