"""`karakara.utils.io` 的读写往返测试。"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from karakara.utils.io import load_audio_native, save_audio


def _stereo(seconds: float = 0.05, sr: int = 8000) -> NDArray[np.float32]:
    t = np.linspace(0.0, seconds, int(sr * seconds), endpoint=False, dtype=np.float32)
    left: NDArray[np.float32] = (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    right: NDArray[np.float32] = (0.25 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    return np.stack([left, right])


def test_int16_round_trip_is_close_but_lossy(tmp_path: Path) -> None:
    data = _stereo()
    path = tmp_path / "i16.wav"
    save_audio(path, data, 8000)

    decoded, rate = load_audio_native(path)

    assert rate == 8000
    assert decoded.shape == data.shape
    # 16 位量化的量化步长是 1/32767，误差必须落在这半个步长以内
    assert float(np.abs(decoded - data).max()) < 1.5 / 32767


def test_int16_clips_out_of_range_samples(tmp_path: Path) -> None:
    """超出 [-1, 1] 的样本必须被裁掉，否则整数转换会绕回反面。"""
    data = np.array([[2.0, -2.0, 0.0]], dtype=np.float32)
    path = tmp_path / "clip.wav"
    save_audio(path, data, 8000)

    decoded, _ = load_audio_native(path)

    assert float(decoded.max()) <= 1.0
    assert float(decoded.min()) >= -1.0


def test_load_audio_native_accepts_an_in_memory_buffer() -> None:
    """回归：`load_audio_native` 要能读内存缓冲区。

    它内部把容器打开**两次**（先取采样率、再解码），而同一个 `BytesIO` 被打开一次
    之后就到末尾了，第二次必然 `InvalidDataError`——与音频是否合法无关。实测
    int16 与 float32 两种载荷都会踩到，所以这里只用一个合法 WAV 就能复现。
    """
    data = _stereo()
    buffer = BytesIO()
    save_audio(buffer, data, 8000)

    decoded, rate = load_audio_native(BytesIO(buffer.getvalue()))

    assert rate == 8000
    assert decoded.shape == data.shape


def test_more_than_two_channels_is_rejected(tmp_path: Path) -> None:
    data = np.zeros((3, 100), dtype=np.float32)
    with pytest.raises(ValueError, match="仅支持 1 或 2 声道"):
        save_audio(tmp_path / "x.wav", data, 8000)
