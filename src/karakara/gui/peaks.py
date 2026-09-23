"""波形包络：把整首歌压成固定分辨率的 min/max 桶，供界面按需聚合绘制。

为什么单独一个模块：GUI 里最容易写错、也最值得单测的就是「时间 ↔ 像素」的换算。
这里只做纯 numpy 的数学，不 import 任何 GUI 库，因此可以在没有显示器的环境下测。

用法::

    envelope = compute_peaks(audio, sample_rate, bucket_ms=1.0)
    minima, maxima, times = peaks_for_range(envelope, 12000, 18000, columns=1200)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

#: 默认的包络分辨率。1ms 一个桶：5 分钟的歌约 30 万个桶（两份 float32 约 2.4MB）。
DEFAULT_BUCKET_MS = 1.0


@dataclass(frozen=True)
class PeakEnvelope:
    """整首歌的 min/max 包络。

    Attributes:
        bucket_ms: 每个桶覆盖的毫秒数。
        minima: 每桶最小值，shape ``(bucket_count,)``。
        maxima: 每桶最大值，shape ``(bucket_count,)``。
        duration_ms: 音频总时长（毫秒）。
        sample_rate: 采样率，仅作记录。
        channels: 声道数，仅作记录。
    """

    bucket_ms: float
    minima: NDArray[np.float32]
    maxima: NDArray[np.float32]
    duration_ms: float
    sample_rate: int
    channels: int

    @property
    def bucket_count(self) -> int:
        """桶的数量。"""
        return int(self.minima.size)

    @property
    def peak_amplitude(self) -> float:
        """全曲最大绝对值，用来把绘制归一化到固定高度。"""
        if self.maxima.size == 0:
            return 1.0
        return max(
            float(np.abs(self.maxima).max()),
            float(np.abs(self.minima).max()),
            1e-6,
        )


def compute_peaks(
    audio: NDArray[np.float32],
    sample_rate: int,
    *,
    bucket_ms: float = DEFAULT_BUCKET_MS,
) -> PeakEnvelope:
    """把音频压成 min/max 包络。

    Args:
        audio: ``(channels, samples)`` 或 ``(samples,)`` 的 float32 音频。
        sample_rate: 采样率。
        bucket_ms: 每个桶的毫秒数，必须为正。

    Returns:
        包络对象。音频短于一个桶时返回单桶。

    Raises:
        ValueError: ``bucket_ms`` 非正，或音频为空。
    """
    if bucket_ms <= 0:
        raise ValueError(f"bucket_ms must be positive, got {bucket_ms}")
    if audio.size == 0:
        raise ValueError("audio is empty")

    mono = audio.mean(axis=0) if audio.ndim == 2 else audio
    channels = audio.shape[0] if audio.ndim == 2 else 1
    total = int(mono.size)
    duration_ms = total / sample_rate * 1000.0

    bucket_samples = max(1, round(sample_rate * bucket_ms / 1000.0))
    bucket_count = max(1, -(-total // bucket_samples))  # 向上取整
    padded = np.zeros(bucket_count * bucket_samples, dtype=np.float32)
    padded[:total] = mono
    frames = padded.reshape(bucket_count, bucket_samples)

    return PeakEnvelope(
        bucket_ms=bucket_ms,
        minima=frames.min(axis=1).astype(np.float32),
        maxima=frames.max(axis=1).astype(np.float32),
        duration_ms=duration_ms,
        sample_rate=sample_rate,
        channels=channels,
    )


def peaks_for_range(
    envelope: PeakEnvelope,
    start_ms: float,
    end_ms: float,
    *,
    columns: int,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float64]]:
    """把 ``[start_ms, end_ms)`` 聚合到 ``columns`` 列，供画布逐列画竖线。

    区间外的部分被裁掉；区间非法（``end <= start``）或 ``columns < 1`` 时返回三个空数组。

    Returns:
        ``(minima, maxima, column_start_ms)``，长度均为实际列数（``<= columns``）。
        空音频或非法区间返回长度 0 的数组，调用方不需要特判。
    """
    empty_f = np.empty(0, dtype=np.float32)
    empty_t = np.empty(0, dtype=np.float64)
    if columns < 1 or end_ms <= start_ms or envelope.bucket_count == 0:
        return empty_f, empty_f.copy(), empty_t

    first = max(0, int(start_ms / envelope.bucket_ms))
    last = min(envelope.bucket_count, int(-(-end_ms // envelope.bucket_ms)))
    if last <= first:
        return empty_f, empty_f.copy(), empty_t

    span = last - first
    if span <= columns:
        # 区间比画布还窄：一桶一列，不合并
        minima = envelope.minima[first:last]
        maxima = envelope.maxima[first:last]
        times = (np.arange(first, last) * envelope.bucket_ms).astype(np.float64)
        return minima, maxima, times

    # 每列覆盖的桶数（向上取整，最后一列可能短一点）
    step = -(-span // columns)
    groups = -(-span // step)
    minima = np.empty(groups, dtype=np.float32)
    maxima = np.empty(groups, dtype=np.float32)
    times = np.empty(groups, dtype=np.float64)
    for index in range(groups):
        low = first + index * step
        high = min(last, low + step)
        minima[index] = envelope.minima[low:high].min()
        maxima[index] = envelope.maxima[low:high].max()
        times[index] = low * envelope.bucket_ms
    return minima, maxima, times
