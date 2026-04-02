from __future__ import annotations

import warnings
from io import BytesIO
from pathlib import Path

import av
import numpy as np
import torch
from numpy.typing import NDArray

from karakara.typ import NpAudioData

DEFAULT_SAMPLE_RATE = 44100


def load_audio(
    src: str | Path | BytesIO,
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    audiotrack_idx: int = 0,
    skip_invalid: bool = True,
) -> NpAudioData:
    """
    dim: 2
    axis: (channels, samples)
    """
    resampler = av.AudioResampler("fltp", rate=sample_rate)
    frames_np: list[np.ndarray] = []

    with av.open(src, "r") as container:
        audio_stream = container.streams.audio[audiotrack_idx]

        # 手动 demux + decode 以捕获单帧错误
        for packet in container.demux(audio_stream):
            try:
                for raw_frame in packet.decode():
                    assert isinstance(raw_frame, av.AudioFrame), (
                        f"Expected AudioFrame from audio stream, got {type(raw_frame).__name__}"
                    )
                    for frame in resampler.resample(raw_frame):
                        frames_np.append(frame.to_ndarray())
            except av.InvalidDataError as e:
                if skip_invalid:
                    warnings.warn(f"跳过损坏的音频帧 @ {packet.pts}: {e}")
                    continue
                raise

            # 流末尾 flush resampler 里的残留
            if packet.is_corrupt or packet.is_discard:
                # 标记为 corrupt 的 packet 通常已经 decode 失败，上面已经处理
                pass

        # 最后 flush resampler
        for frame in resampler.resample(None):
            frames_np.append(frame.to_ndarray())
    if not frames_np:
        raise ValueError("未能读取任何有效音频帧，文件可能已严重损坏")

    wf_np: NDArray[np.float32] = np.concatenate(frames_np, axis=1).astype(np.float32)
    return wf_np


def save_audio(
    dst: str | Path | BytesIO,
    data: NpAudioData,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> None:
    # 防止溢出：先裁剪到 [-1.0, 1.0] 范围
    data_clipped = np.clip(data, -1.0, 1.0)
    data_np_i16 = (data_clipped * 32767).astype(np.int16)
    channel_n = data_np_i16.shape[0]
    if channel_n == 1:
        layout = "mono"
    elif channel_n == 2:
        layout = "stereo"
    else:
        raise ValueError(f"save_audio 仅支持 1 或 2 声道, 收到 {channel_n} 声道")

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


def ndarray2tensor(array: NDArray[np.float32]) -> torch.Tensor:
    return torch.from_numpy(array)


def tensor2ndarray(tensor: torch.Tensor) -> NDArray[np.float32]:
    return tensor.cpu().contiguous().numpy()  # type: ignore[return-value]


def ms2sample(ms: int | float, sample_rate: int = DEFAULT_SAMPLE_RATE) -> int:
    return int(ms * sample_rate / 1000)


def sample2ms(sample: int, sample_rate: int = DEFAULT_SAMPLE_RATE) -> int:
    return int(sample / sample_rate * 1000)
