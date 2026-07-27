"""人声音频能量与活动度评分测试。"""

from __future__ import annotations

import numpy as np

from karakara.offset import build_energy_curve, score_vocal_activity


def test_build_energy_curve_and_score_vocal_activity() -> None:
    """活动度应反映指定时间段的人声能量。"""
    audio = np.concatenate(
        (
            np.zeros(10, dtype=np.float32),
            np.ones(10, dtype=np.float32),
        )
    )
    energy = build_energy_curve(audio, sample_rate=10, window_ms=500)

    assert energy.tolist() == [0.0, 0.0, 1.0, 1.0]
    assert score_vocal_activity(energy, 0, 1000, window_ms=500) == 0.0
    assert score_vocal_activity(energy, 1000, 2000, window_ms=500) == 1.0


def test_score_vocal_activity_handles_empty_or_out_of_range_ranges() -> None:
    energy = np.array([0.5, 1.0], dtype=np.float32)

    assert score_vocal_activity(energy, 100, 100, window_ms=100) == 0.0
    assert score_vocal_activity(energy, 500, 600, window_ms=100) == 0.0
