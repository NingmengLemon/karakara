from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import av
import numpy as np
import torch
from numpy.typing import NDArray

from karakara.typ import NpAudioData

DEFAULT_SAMPLE_RATE = 16000


def load_audio(
    src: str | Path | BytesIO,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    audiotrack_idx: int = 0,
) -> NpAudioData:
    """
    dim: 2
    axis: (channels, samples)
    """
    resampler = av.AudioResampler("fltp", rate=sample_rate)
    frames_np: list[np.ndarray] = []
    with av.open(src, "r") as container:
        for raw_frame in container.decode(container.streams.audio[audiotrack_idx]):
            for frame in resampler.resample(raw_frame):
                frames_np.append(frame.to_ndarray())
        for frame in resampler.resample(None):
            frames_np.append(frame.to_ndarray())

    wf_np: NDArray[np.float32] = np.concatenate(frames_np, axis=1).astype(np.float32)
    return wf_np


def save_audio(
    dst: str | Path | BytesIO,
    data: NpAudioData,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> None:
    data_np_i16 = (data * 32767).astype(np.int16)
    channel_n = data_np_i16.shape[0]
    layout = "stereo" if channel_n == 2 else "mono"

    with av.open(dst, "w", format="wav") as container:
        stream = container.add_stream(
            "pcm_s16le",
            rate=sample_rate,
            layout=layout,
        )

        frame = av.AudioFrame.from_ndarray(data_np_i16, format="s16p", layout=layout)
        frame.sample_rate = sample_rate

        # 编码并写入文件
        for packet in stream.encode(frame):
            container.mux(packet)

        # Flush a-v stream
        for packet in stream.encode(None):
            container.mux(packet)


def ndarray2tensor(array: NDArray) -> torch.Tensor:
    return torch.from_numpy(array)


def tensor2ndarray(tensor: torch.Tensor) -> Any:
    return tensor.cpu().contiguous().numpy()


def ms2sample(ms: int | float, sample_rate: int = DEFAULT_SAMPLE_RATE) -> int:
    return int(ms * sample_rate / 1000)


def sample2ms(sample: int, sample_rate: int = DEFAULT_SAMPLE_RATE) -> int:
    return int(sample / sample_rate * 1000)
