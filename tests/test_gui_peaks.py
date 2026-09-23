"""`karakara.gui.peaks` 的已知答案测试。

包络是「时间 ↔ 像素」换算的地基，所以用**手算就能得出答案**的合成信号来钉：
一段满幅、一段静音、一段半幅，桶边界与聚合列都必须落在预期值上。
"""

from __future__ import annotations

import numpy as np
import pytest

from karakara.gui.peaks import compute_peaks, peaks_for_range

RATE = 1000  # 1kHz：1ms 正好 1 个采样，桶边界好算


def _ramp_signal() -> np.ndarray:
    """构造：0-100ms 满幅 ±1.0、100-200ms 静音、200-300ms ±0.5。"""
    samples = np.zeros(300, dtype=np.float32)
    samples[0:100] = np.tile([1.0, -1.0], 50)
    samples[200:300] = np.tile([0.5, -0.5], 50)
    return samples


def test_peaks_match_hand_computed_envelope() -> None:
    envelope = compute_peaks(_ramp_signal(), RATE, bucket_ms=100.0)

    assert envelope.bucket_count == 3
    np.testing.assert_allclose(envelope.maxima, [1.0, 0.0, 0.5])
    np.testing.assert_allclose(envelope.minima, [-1.0, 0.0, -0.5])
    assert envelope.duration_ms == pytest.approx(300.0)


def test_short_audio_still_yields_one_bucket() -> None:
    """音频短于一个桶时不能返回空包络（界面会除以 0）。"""
    envelope = compute_peaks(np.ones(3, dtype=np.float32), RATE, bucket_ms=100.0)

    assert envelope.bucket_count == 1
    assert envelope.duration_ms == pytest.approx(3.0)


def test_stereo_is_downmixed_for_the_envelope() -> None:
    left = np.full(200, 0.25, dtype=np.float32)
    right = np.full(200, 0.75, dtype=np.float32)

    envelope = compute_peaks(np.stack([left, right]), RATE, bucket_ms=100.0)

    assert envelope.channels == 2
    # 均值 0.5，两个桶都是常数
    np.testing.assert_allclose(envelope.maxima, [0.5, 0.5])
    np.testing.assert_allclose(envelope.minima, [0.5, 0.5])


def test_peak_amplitude_is_never_zero() -> None:
    """全静音时归一化分母不能是 0。"""
    envelope = compute_peaks(np.zeros(100, dtype=np.float32), RATE, bucket_ms=10.0)
    assert envelope.peak_amplitude > 0


def test_range_aggregation_keeps_extremes() -> None:
    """聚合到较少列时，每列必须保住区间内的极值（不能抽样丢峰）。"""
    envelope = compute_peaks(_ramp_signal(), RATE, bucket_ms=1.0)

    minima, maxima, times = peaks_for_range(envelope, 0, 300, columns=3)

    assert len(maxima) == 3
    np.testing.assert_allclose(maxima, [1.0, 0.0, 0.5])
    np.testing.assert_allclose(minima, [-1.0, 0.0, -0.5])
    np.testing.assert_allclose(times, [0.0, 100.0, 200.0])


def test_range_narrower_than_canvas_returns_one_column_per_bucket() -> None:
    envelope = compute_peaks(_ramp_signal(), RATE, bucket_ms=1.0)

    _, maxima, times = peaks_for_range(envelope, 0, 5, columns=1000)

    assert len(maxima) == 5
    np.testing.assert_allclose(times, [0.0, 1.0, 2.0, 3.0, 4.0])


def test_range_is_clipped_to_the_audio() -> None:
    envelope = compute_peaks(_ramp_signal(), RATE, bucket_ms=1.0)

    _, maxima, _ = peaks_for_range(envelope, -500, 10_000, columns=10_000)

    assert len(maxima) == 300


def test_invalid_ranges_return_empty_arrays() -> None:
    envelope = compute_peaks(_ramp_signal(), RATE, bucket_ms=1.0)

    for start, end, columns in ((100, 100, 10), (200, 100, 10), (0, 100, 0)):
        minima, maxima, times = peaks_for_range(envelope, start, end, columns=columns)
        assert len(minima) == len(maxima) == len(times) == 0


def test_bucket_ms_must_be_positive() -> None:
    with pytest.raises(ValueError, match="bucket_ms"):
        compute_peaks(np.ones(10, dtype=np.float32), RATE, bucket_ms=0)


def test_empty_audio_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        compute_peaks(np.empty(0, dtype=np.float32), RATE)
