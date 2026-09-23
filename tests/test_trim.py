"""`karakara.trim` 与 `karakara.offset.detect_last_vocal_activity` 的已知答案测试。

裁剪会**动到送给对齐器的音频**，所以每条守卫都要用手算得出的合成曲线钉住：
该裁的裁、不该裁的一条都不许裁。
"""

from __future__ import annotations

import numpy as np
import pytest

from karakara.offset import detect_last_vocal_activity
from karakara.trim import TailTrimConfig, trim_window_end

WINDOW_MS = 20.0
CONFIG = TailTrimConfig()


def curve(*spans: tuple[float, float, float], total_ms: float = 20_000) -> np.ndarray:
    """按 ``(起点ms, 终点ms, 幅度)`` 拼一条能量曲线（其余为 0）。"""
    size = int(total_ms / WINDOW_MS)
    energy = np.zeros(size, dtype=np.float32)
    for start_ms, end_ms, level in spans:
        low = int(start_ms / WINDOW_MS)
        high = int(end_ms / WINDOW_MS)
        energy[low:high] = level
    return energy


def add_burst(
    energy: np.ndarray,
    start_ms: float,
    end_ms: float,
    level: float,
    *,
    on_ms: float = 100,
    off_ms: float = 100,
) -> np.ndarray:
    """往曲线里叠一段**突发式**人声：每 ``on_ms`` 有声、随后 ``off_ms`` 静音。

    这是判据真正会漏掉的东西：每个 run 都短于持续门槛（200ms），所以
    :func:`detect_last_vocal_activity` 找不到它，但那段区间里明显有人在唱。
    """
    cursor = start_ms
    while cursor < end_ms:
        on_end = min(cursor + on_ms, end_ms)
        energy[int(cursor / WINDOW_MS) : int(on_end / WINDOW_MS)] = level
        cursor = on_end + off_ms
    return energy


# --------------------------------------------------------------------------
# 检测器
# --------------------------------------------------------------------------


def test_detects_the_end_of_the_last_run() -> None:
    energy = curve((0, 3_000, 0.8), (5_000, 8_000, 0.9))

    assert detect_last_vocal_activity(energy, window_ms=WINDOW_MS) == 8_000


def test_ignores_runs_shorter_than_the_minimum() -> None:
    """短于持续门槛的爆发不算「有人在唱」。"""
    energy = curve((0, 3_000, 0.8), (5_000, 5_100, 0.9))

    assert (
        detect_last_vocal_activity(energy, window_ms=WINDOW_MS, min_run_ms=200) == 3_000
    )


def test_returns_none_when_nothing_is_active() -> None:
    assert detect_last_vocal_activity(curve(), window_ms=WINDOW_MS) is None


def test_threshold_is_relative_to_the_peak() -> None:
    """安静的人声（峰值的 3%）在 5% 阈值下不算活跃，2% 阈值下算。"""
    energy = curve((0, 2_000, 1.0), (4_000, 6_000, 0.03))

    assert (
        detect_last_vocal_activity(energy, window_ms=WINDOW_MS, relative_threshold=0.05)
        == 2_000
    )
    assert (
        detect_last_vocal_activity(energy, window_ms=WINDOW_MS, relative_threshold=0.02)
        == 6_000
    )


# --------------------------------------------------------------------------
# 裁剪判定：该裁的
# --------------------------------------------------------------------------


def test_trims_a_long_silent_tail_keeping_the_margin() -> None:
    """唱到 3s，窗口到 20s：裁到 3s + 500ms 余量。"""
    energy = curve((0, 3_000, 0.8), total_ms=20_000)

    decision = trim_window_end(energy, 0, 20_000, config=CONFIG, window_ms=WINDOW_MS)

    assert decision.trimmed is True
    assert decision.end_ms == 3_300
    assert decision.trimmed_ms == 16_700


def test_window_offset_is_respected() -> None:
    """窗口不从 0 开始时，判定要按绝对时间算。"""
    energy = curve((10_000, 13_000, 0.8), total_ms=40_000)

    decision = trim_window_end(
        energy, 10_000, 30_000, config=CONFIG, window_ms=WINDOW_MS
    )

    assert decision.end_ms == 13_300


# --------------------------------------------------------------------------
# 裁剪判定：不该裁的（每条守卫一个用例）
# --------------------------------------------------------------------------


def test_disabled_config_never_trims() -> None:
    energy = curve((0, 3_000, 0.8), total_ms=20_000)

    decision = trim_window_end(energy, 0, 20_000, config=None, window_ms=WINDOW_MS)

    assert decision.trimmed is False
    assert decision.end_ms == 20_000


def test_short_tail_is_left_alone() -> None:
    """尾部只有 800ms（< min_tail_ms 1000）：不值得裁。"""
    energy = curve((0, 4_200, 0.8), total_ms=5_000)

    decision = trim_window_end(energy, 0, 5_000, config=CONFIG, window_ms=WINDOW_MS)

    assert decision.trimmed is False
    assert "不值得裁" in decision.reason


def test_guard_refuses_when_bursty_vocals_are_missed() -> None:
    """漏检场景：判据说人声 860ms 结束，但之后还有一段**突发式**的唱。

    这是评估里唯一那例真漏检的复现（活跃度 0.2045）：人声在 860ms 之后继续，但每个
    连续段都短于 200ms 的持续门槛，所以检测器看不到它。守卫必须拒绝裁剪。
    """
    energy = curve((0, 860, 0.8), total_ms=2_030)
    add_burst(energy, 1_360, 2_030, 0.5)

    decision = trim_window_end(energy, 0, 2_030, config=CONFIG, window_ms=WINDOW_MS)

    assert decision.trimmed is False
    assert "漏检" in decision.reason


def test_late_continuous_vocals_are_detected_not_missed() -> None:
    """反过来：后面那段人声**连续**够长时，检测器会找到它，裁到它结束是正确行为。"""
    energy = curve((0, 3_000, 0.8), (4_000, 6_000, 0.5))

    decision = trim_window_end(energy, 0, 20_000, config=CONFIG, window_ms=WINDOW_MS)

    assert decision.trimmed is True
    assert decision.end_ms == 6_300


def test_quiet_audio_after_the_cut_is_fine() -> None:
    """裁剪点之后只剩混响级别的能量（0.03 < 守卫阈值）：照裁。"""
    energy = curve((0, 3_000, 0.8), (3_500, 4_500, 0.03), total_ms=20_000)

    decision = trim_window_end(energy, 0, 20_000, config=CONFIG, window_ms=WINDOW_MS)

    assert decision.trimmed is True
    assert decision.end_ms == 3_300


def test_no_activity_at_all_is_left_alone() -> None:
    """整段检不出人声：不裁（交给 --min-vocal-activity 去跳过这一行）。"""
    decision = trim_window_end(curve(), 0, 20_000, config=CONFIG, window_ms=WINDOW_MS)

    assert decision.trimmed is False
    assert "检不出持续人声" in decision.reason


def test_margin_larger_than_the_tail_is_left_alone() -> None:
    """人声结束得很晚，余量已经盖住整个尾部：没必要裁。"""
    energy = curve((0, 9_800, 0.8), total_ms=10_000)

    decision = trim_window_end(energy, 0, 10_000, config=CONFIG, window_ms=WINDOW_MS)

    assert decision.trimmed is False


def test_tiny_window_is_left_alone() -> None:
    decision = trim_window_end(
        curve((0, 20, 0.8)), 0, 20, config=CONFIG, window_ms=WINDOW_MS
    )

    assert decision.trimmed is False
    assert "窗口太短" in decision.reason


def test_inverted_window_is_left_alone() -> None:
    decision = trim_window_end(curve((0, 3_000, 0.8)), 5_000, 5_000, config=CONFIG)

    assert decision.trimmed is False


@pytest.mark.parametrize("level", [0.1, 0.3, 0.8])
def test_guard_fires_for_any_audible_bursty_tail(level: float) -> None:
    """守卫对「突发式但明显可听」的尾巴都拒绝裁剪（幅度从 0.1 到 0.8）。

    这三档的占空比都是 50%（100ms 有声 / 100ms 静音），远超「超过阈值的窗口占比 5%」，
    所以无论幅度是刚过阈值还是很大，守卫都要拦下。
    """
    energy = curve((0, 860, 0.8), total_ms=2_030)
    add_burst(energy, 1_360, 2_030, level)

    decision = trim_window_end(energy, 0, 2_030, config=CONFIG, window_ms=WINDOW_MS)

    assert decision.trimmed is False
    assert "漏检" in decision.reason


def test_a_single_short_burst_does_not_block_the_trim() -> None:
    """孤立的 100ms 小爆发（占区域 1%）不算「还在唱」，不拦裁剪。

    这是守卫的边界：占比判据要能放过真正的零星噪声，否则一点点响动就永远裁不了。
    """
    energy = curve((0, 3_000, 0.8), total_ms=20_000)
    add_burst(energy, 6_000, 6_100, 0.5)

    decision = trim_window_end(energy, 0, 20_000, config=CONFIG, window_ms=WINDOW_MS)

    assert decision.trimmed is True
    assert decision.end_ms == 3_300


def test_bursty_tail_below_the_guard_threshold_still_trims() -> None:
    """突发式的尾巴若整体低于守卫阈值（混响级别），仍然照裁。"""
    energy = curve((0, 3_000, 0.8), total_ms=20_000)
    add_burst(energy, 4_000, 6_000, 0.03)

    decision = trim_window_end(energy, 0, 20_000, config=CONFIG, window_ms=WINDOW_MS)

    assert decision.trimmed is True
    assert decision.end_ms == 3_300
