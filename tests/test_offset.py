"""人声音频能量、活动度评分与全局偏移估计测试。"""

from __future__ import annotations

import numpy as np
import pytest
from lemony_lrc_parser import Lyrics
from numpy.typing import NDArray

from karakara.offset import (
    _build_lrc_presence_curve,
    _score_at_offset,
    build_energy_curve,
    estimate_offset,
    score_vocal_activity,
)
from karakara.utils.metadata import MetadataFilter

RATE = 1000
WINDOW_MS = 50.0
PHRASES = [(16.0, 24.0), (32.0, 40.0), (48.0, 56.0), (64.0, 72.0)]
VARIED_PHRASES = [(16.0, 21.0), (28.0, 36.0), (44.0, 47.0), (58.0, 68.0), (80.0, 84.0)]


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


def permissive_filter() -> MetadataFilter:
    """不做任何过滤的 MetadataFilter。"""
    return MetadataFilter(
        keywords=[],
        id3_tags=frozenset(),
        parenthetical_markers=[],
        detect_id3_tags=False,
        detect_parenthetical=False,
        detect_pure_numbers=False,
        custom_patterns=[],
    )


def build_song(
    *,
    duration_s: float,
    phrases: list[tuple[float, float]],
    line_step_s: float,
    shift_s: float,
) -> tuple[NDArray[np.float32], Lyrics, float]:
    """构造 ground truth 为 ``+shift_s`` 的合成场景。

    真实人声位于 ``phrases`` 各区间；歌词行铺在区间内，时间戳整体比真实位置
    早 ``shift_s``（``shift_s`` 为负即歌词偏晚），因此期望估计值就是
    ``shift_s * 1000`` ms。
    """
    audio = np.full(int(duration_s * RATE), 0.05, dtype=np.float32)
    for start, end in phrases:
        audio[int(start * RATE) : int(end * RATE)] = 1.0

    stamps: list[float] = []
    for start, end in phrases:
        t = start - shift_s
        while t < end - shift_s - 1e-9:
            stamps.append(t)
            t += line_step_s
    stamps.sort()
    assert stamps, "构造的用例没有产生任何歌词行"
    lyrics = Lyrics.loads(
        "\n".join(f"[{int(s // 60):02d}:{s % 60:06.3f}]line" for s in stamps)
    )
    return audio, lyrics, shift_s * 1000.0


def run_estimate(audio: NDArray[np.float32], lyrics: Lyrics) -> float:
    return estimate_offset(
        audio, lyrics, RATE, metadata_filter=permissive_filter(), window_ms=WINDOW_MS
    )


# --------------------------------------------------------------------------
# 回归：修复前会凭空注入偏移
# --------------------------------------------------------------------------


def test_keeps_zero_offset_when_lyrics_already_aligned() -> None:
    """歌词本来就对齐时必须返回 0。

    回归测试：旧的「整行方块 + 裸点积」实现会在这首歌上返回 **−1800ms**，
    凭空给一首已经对齐的歌注入 1.8 秒偏移。
    """
    audio, lyrics, _ = build_song(
        duration_s=90, phrases=PHRASES, line_step_s=2.0, shift_s=0.0
    )

    assert run_estimate(audio, lyrics) == 0.0


@pytest.mark.parametrize(
    ("line_step_s", "shift_s", "duration_s", "phrases"),
    [
        # 密集行距 + 正偏移：旧实现分别偏 −2600 / −1500 / −2400ms
        (2.0, 3.0, 90.0, PHRASES),
        (1.0, 1.5, 90.0, PHRASES),
        (2.0, 8.0, 110.0, PHRASES),
        # 单个长人声块：旧实现偏 −2000ms
        (3.0, 3.0, 120.0, [(16.0, 64.0)]),
        # 稀疏行距
        (8.0, 3.0, 90.0, PHRASES),
        # 真实感的参差乐句
        (2.0, 2.0, 100.0, VARIED_PHRASES),
        (1.0, 3.0, 100.0, VARIED_PHRASES),
    ],
)
def test_estimates_positive_shift_exactly(
    line_step_s: float, shift_s: float, duration_s: float,
    phrases: list[tuple[float, float]],
) -> None:
    """正偏移（歌词偏早）应被精确估计到窗口精度。"""
    audio, lyrics, truth = build_song(
        duration_s=duration_s,
        phrases=phrases,
        line_step_s=line_step_s,
        shift_s=shift_s,
    )

    assert run_estimate(audio, lyrics) == truth


@pytest.mark.parametrize(
    ("line_step_s", "shift_s"),
    [(2.0, -2.0), (2.0, -1.0), (1.0, -1.5)],
)
def test_negative_shift_error_stays_within_shortest_phrase(
    line_step_s: float, shift_s: float
) -> None:
    """负偏移（歌词偏晚）只能约束到乐句长度级别，但误差必须有界。

    这是可辨识性限制而非缺陷：歌词行落在持续人声内部时，「这一行从人声中间
    开始」与「这一行正好在人声起点」在能量曲线上不可区分。修复前这类用例会
    整体塌到 0（误差 +2000 ~ +5000ms）；现在误差被压到最短乐句（3 秒）以内。
    """
    audio, lyrics, truth = build_song(
        duration_s=100, phrases=VARIED_PHRASES, line_step_s=line_step_s,
        shift_s=shift_s,
    )

    assert abs(run_estimate(audio, lyrics) - truth) <= 1000.0


# --------------------------------------------------------------------------
# 行首指示曲线
# --------------------------------------------------------------------------


def test_presence_curve_marks_only_line_onsets() -> None:
    """只在行首标记窄窗，而不是整行区间。"""
    lyrics = Lyrics.loads("[00:01.000]a\n[00:05.000]b")

    presence = _build_lrc_presence_curve(
        lyrics, 400, 20_000.0, metadata_filter=permissive_filter(),
        window_ms=WINDOW_MS, onset_ms=300.0,
    )

    # 行首窗 300ms / 窗口 50ms = 6 个窗口
    assert np.flatnonzero(presence).tolist() == [
        20, 21, 22, 23, 24, 25, 100, 101, 102, 103, 104, 105,
    ]


def test_presence_curve_skips_metadata_and_out_of_range_lines() -> None:
    """元数据行与超出音频范围的行都不参与（固定分母必须干净）。"""
    lyrics = Lyrics.loads("[00:00.000]作词 : someone\n[00:01.000]a\n[00:05.000]b")
    metadata_filter = MetadataFilter(
        keywords=["作词"], id3_tags=frozenset(), parenthetical_markers=[],
        detect_id3_tags=False, detect_parenthetical=False,
        detect_pure_numbers=False, custom_patterns=[],
    )

    presence = _build_lrc_presence_curve(
        lyrics, 200, 4_000.0, metadata_filter=metadata_filter,
        window_ms=WINDOW_MS, onset_ms=300.0,
    )

    # 只剩 [00:01.000] 一行（[00:05.000] 的起点已在 4 秒音频之外）
    assert np.flatnonzero(presence).tolist() == [20, 21, 22, 23, 24, 25]


# --------------------------------------------------------------------------
# 打分函数
# --------------------------------------------------------------------------


def test_score_penalizes_offsets_that_push_lyrics_out_of_audio() -> None:
    """分母固定：越界行首按 0 计入，位移越大得分越低。"""
    energy = np.ones(200, dtype=np.float32)
    presence = np.zeros(200, dtype=np.float32)
    presence[100:106] = 1.0

    score_at_zero = _score_at_offset(energy, presence, 0, total_presence=6.0)
    score_at_large = _score_at_offset(energy, presence, 190, total_presence=6.0)

    assert score_at_zero > score_at_large


def test_score_returns_neg_inf_when_coverage_too_low() -> None:
    energy = np.ones(200, dtype=np.float32)
    presence = np.zeros(200, dtype=np.float32)
    presence[0:6] = 1.0

    # 右移 195 个窗口后 6 个行首窗只剩 5 个，低于 0.8 的保留率
    assert _score_at_offset(energy, presence, 195, total_presence=6.0) == float(
        "-inf"
    )


def test_score_rejects_degenerate_presence() -> None:
    """行首窗覆盖全部窗口（或一个都没有）时无法做判别，应判为无效。"""
    energy = np.ones(10, dtype=np.float32)

    assert _score_at_offset(
        energy, np.ones(10, dtype=np.float32), 0, total_presence=10.0
    ) == float("-inf")
    assert _score_at_offset(
        energy, np.zeros(10, dtype=np.float32), 0, total_presence=0.0
    ) == float("-inf")


# --------------------------------------------------------------------------
# 退化输入
# --------------------------------------------------------------------------


def test_returns_zero_when_no_timed_lines() -> None:
    audio = np.ones(10_000, dtype=np.float32)
    lyrics = Lyrics.loads("no timestamps here\nnor here")

    assert run_estimate(audio, lyrics) == 0.0


def test_returns_zero_when_audio_too_short() -> None:
    audio = np.ones(10, dtype=np.float32)
    lyrics = Lyrics.loads("[00:00.00]a")

    assert run_estimate(audio, lyrics) == 0.0
