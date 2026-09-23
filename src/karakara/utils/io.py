from __future__ import annotations

import warnings
from io import BytesIO
from pathlib import Path
from typing import Literal

import av
import numpy as np
from numpy.typing import NDArray

from karakara.typ import NpAudioData

DEFAULT_SAMPLE_RATE = 44100

#: 解码时的目标采样格式。``fltp`` 是平面 float32，``to_ndarray()`` 会得到
#: (channels, samples)。
_DECODE_FORMAT = "fltp"

#: 写出 WAV 时可选的采样格式：``subtype -> (编码器, PyAV 帧格式)``。
_SAMPLE_SUBTYPES: dict[str, tuple[str, str]] = {
    "int16": ("pcm_s16le", "s16p"),
    "float32": ("pcm_f32le", "fltp"),
}

#: :func:`save_audio` 的 ``subtype`` 取值。
SampleSubtype = Literal["int16", "float32"]


def _decode(
    src: str | Path | BytesIO,
    *,
    audiotrack_idx: int,
    skip_invalid: bool,
    resampler: av.AudioResampler,
) -> NpAudioData:
    """手动 demux + decode，返回 (channels, samples) 的 float32 数组。"""
    frames_np: list[np.ndarray] = []

    with av.open(src, "r") as container:
        audio_stream = container.streams.audio[audiotrack_idx]

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
                    warnings.warn(f"跳过损坏的音频帧 @ {packet.pts}: {e}", stacklevel=2)
                    continue
                raise

        # 最后 flush resampler
        for frame in resampler.resample(None):
            frames_np.append(frame.to_ndarray())

    if not frames_np:
        raise ValueError("未能读取任何有效音频帧，文件可能已严重损坏")

    wf_np: NDArray[np.float32] = np.concatenate(frames_np, axis=1).astype(np.float32)
    return wf_np


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

    重采样到 ``sample_rate``。
    """
    return _decode(
        src,
        audiotrack_idx=audiotrack_idx,
        skip_invalid=skip_invalid,
        resampler=av.AudioResampler(_DECODE_FORMAT, rate=sample_rate),
    )


def load_audio_native(
    src: str | Path | BytesIO,
    *,
    audiotrack_idx: int = 0,
    skip_invalid: bool = True,
) -> tuple[NpAudioData, int]:
    """解码音频并**保持原采样率**。

    分离 worker 写出的音轨采样率由它选定的模型决定，主进程不该再去假设一个
    固定值，因此直接把文件的原生采样率一并返回。

    容器在这里会被打开**两次**（先取采样率、再解码），所以非路径来源必须在第二次
    打开前回到起点：实测同一个 ``BytesIO`` 直接开第二次必然
    ``InvalidDataError``（缓冲区已读到末尾），与音频是否合法无关。

    Returns:
        ``(audio, sample_rate)``，audio 为 (channels, samples) 的 float32。
    """
    with av.open(src, "r") as container:
        stream = container.streams.audio[audiotrack_idx]
        sample_rate = stream.rate or DEFAULT_SAMPLE_RATE
    seek = getattr(src, "seek", None)
    if callable(seek):
        seek(0)
    # rate=None 表示沿用输入采样率，只做采样格式归一化
    resampler = av.AudioResampler(_DECODE_FORMAT)
    return _decode(
        src,
        audiotrack_idx=audiotrack_idx,
        skip_invalid=skip_invalid,
        resampler=resampler,
    ), sample_rate


def save_audio(
    dst: str | Path | BytesIO,
    data: NpAudioData,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    *,
    subtype: SampleSubtype = "int16",
) -> None:
    """把 ``(channels, samples)`` 的音频写成 WAV。

    Args:
        dst: 目标路径或内存缓冲区。
        data: ``(channels, samples)`` 的 float32 音频。
        sample_rate: 采样率。
        subtype: ``"int16"``（默认）先裁剪到 ``[-1, 1]`` 再量化，兼容性最好；
            ``"float32"`` 写 32 位浮点且**不裁剪**，供不希望引入量化的下游使用
            （送给对齐器的音频就是这种：分离链路刻意全程 32 位浮点，
            在最后一步退回 16 位会把前面保住的那点动态范围白白丢掉）。
    """
    if subtype not in _SAMPLE_SUBTYPES:
        raise ValueError(
            f"未知的采样格式: {subtype!r}，可选: {sorted(_SAMPLE_SUBTYPES)}"
        )
    codec, frame_format = _SAMPLE_SUBTYPES[subtype]
    channel_n = data.shape[0]
    if channel_n == 1:
        layout = "mono"
    elif channel_n == 2:
        layout = "stereo"
    else:
        raise ValueError(f"save_audio 仅支持 1 或 2 声道, 收到 {channel_n} 声道")

    if subtype == "int16":
        samples: NDArray = (np.clip(data, -1.0, 1.0) * 32767).astype(np.int16)
    else:
        samples = np.asarray(data, dtype=np.float32)

    with av.open(dst, "w", format="wav") as container:
        stream = container.add_stream(
            codec,
            rate=sample_rate,
            layout=layout,
        )
        # 编码器名现在来自变量而不是字面量，重载解析拿不到 AudioStream，
        # 于是显式收窄一次（ty 认 isinstance）。
        assert isinstance(stream, av.AudioStream)

        frame = av.AudioFrame.from_ndarray(samples, format=frame_format, layout=layout)
        frame.sample_rate = sample_rate

        # 编码并写入文件
        for packet in stream.encode(frame):
            container.mux(packet)

        # Flush a-v stream
        for packet in stream.encode(None):
            container.mux(packet)


def ms2sample(ms: float, sample_rate: int = DEFAULT_SAMPLE_RATE) -> int:
    return int(ms * sample_rate / 1000)


def sample2ms(sample: int, sample_rate: int = DEFAULT_SAMPLE_RATE) -> int:
    return int(sample / sample_rate * 1000)
