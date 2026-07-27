"""可选的 Demucs 实机集成测试。

运行前需要准备模型仓库与可用的 PyTorch 设备。默认跳过，避免日常测试
下载模型或占用 GPU；手动运行：

    $env:KARAKARA_RUN_INTEGRATION = "1"
    uv run pytest -m integration
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from karakara.separator.demucs.impl import DemucsSeparator
from karakara.utils.io import load_audio


pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    os.environ.get("KARAKARA_RUN_INTEGRATION") != "1",
    reason="set KARAKARA_RUN_INTEGRATION=1 to run local Demucs integration tests",
)
def test_demucs_separates_sample_audio_with_channel_sample_shape() -> None:
    """实机验证当前 Demucs API 输出被适配为二维 stem。"""
    audio_path = Path("samples/ashen.mp3")
    separator = DemucsSeparator()
    audio = load_audio(audio_path, sample_rate=separator.samplerate)

    stems = separator.separate(audio)

    assert separator.VOCAL_STEM_NAME in stems
    for name, stem in stems.items():
        assert stem.ndim == 2, f"unexpected shape for {name!r}: {stem.shape}"
        assert stem.shape[0] in (1, 2), (
            f"unexpected channel count for {name!r}: {stem.shape}"
        )
        assert stem.shape[1] > 0
