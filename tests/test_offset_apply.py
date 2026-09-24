"""`_apply_offset()` 的时间轴夹取行为、自动偏移校验与对齐健康检查。

重点回归：防负时间戳的**回退**会抵消负偏移。LRC 里几乎总有一条
``[00:00.000]`` 的元数据行（「作词 : xxx」），它不参与对齐，却能让任何负偏移
被完整撤销——包括 ``--offset`` 手动指定的负值。
"""

from __future__ import annotations

from logging import INFO, WARNING

import numpy as np
import pytest
from lemony_lrc_parser import Lyrics
from lemony_lrc_parser.offset import iter_all_timestamps

from karakara.core import _apply_offset, _check_alignment_health
from karakara.offset import (
    detect_first_vocal_onset,
    score_line_intervals,
    suggest_offset_from_onset,
    validate_estimated_offset,
)
from karakara.utils.metadata import MetadataFilter


def filter_with_keywords(*keywords: str) -> MetadataFilter:
    return MetadataFilter(
        keywords=list(keywords),
        id3_tags=frozenset(),
        parenthetical_markers=[],
        detect_id3_tags=False,
        detect_parenthetical=False,
        detect_pure_numbers=False,
        custom_patterns=[],
    )


SILENT = np.zeros(1000, dtype=np.float32)


def line_starts(lyrics: Lyrics) -> list[int]:
    return [line.start for line in lyrics]


def test_negative_offset_survives_a_zero_timestamp_metadata_line() -> None:
    """含 ``[00:00.000]`` 元数据行时，负偏移必须真的生效。

    回归测试：旧实现会把整条时间轴回退 200ms，净偏移为 0。
    """
    lyrics = Lyrics.loads(
        "[00:00.000] 作词 : someone\n[00:13.590]first lyric\n[00:19.580]second"
    )

    applied = _apply_offset(
        lyrics,
        SILENT,
        1000,
        metadata_filter=filter_with_keywords("作词"),
        offset_ms=-200.0,
    )

    assert applied == -200.0
    # 元数据行被夹到 0，歌词行保留修正
    assert line_starts(lyrics) == [0, 13390, 19380]
    assert min(iter_all_timestamps(lyrics)) >= 0


def test_negative_offset_without_zero_line_is_unchanged() -> None:
    lyrics = Lyrics.loads("[00:13.590]first lyric\n[00:19.580]second")

    applied = _apply_offset(
        lyrics,
        SILENT,
        1000,
        metadata_filter=filter_with_keywords(),
        offset_ms=-200.0,
    )

    assert applied == -200.0
    assert line_starts(lyrics) == [13390, 19380]


def test_positive_offset_is_applied_as_before() -> None:
    lyrics = Lyrics.loads("[00:00.000] 作词 : someone\n[00:13.590]first")

    applied = _apply_offset(
        lyrics,
        SILENT,
        1000,
        metadata_filter=filter_with_keywords("作词"),
        offset_ms=500.0,
    )

    assert applied == 500.0
    assert line_starts(lyrics) == [500, 14090]


def test_clamping_keeps_the_output_serializable() -> None:
    """夹取之后必须还能 dumps：序列化器对负时间戳会直接抛异常。"""
    lyrics = Lyrics.loads("[00:00.000] 作词 : someone\n[00:00.500]short line")
    _apply_offset(
        lyrics,
        SILENT,
        1000,
        metadata_filter=filter_with_keywords("作词"),
        offset_ms=-4000.0,
    )

    assert min(iter_all_timestamps(lyrics)) >= 0
    assert "[00:00.000]" in lyrics.dumps()


def test_clamping_alignable_lines_warns_about_an_oversized_offset(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """连参与对齐的行都被夹住时，说明估计过大，必须告警而不是静默。"""
    lyrics = Lyrics.loads("[00:00.000] 作词 : someone\n[00:01.000]real lyric")

    with caplog.at_level(WARNING):
        _apply_offset(
            lyrics,
            SILENT,
            1000,
            metadata_filter=filter_with_keywords("作词"),
            offset_ms=-5000.0,
        )

    assert any("probably too large" in record.message for record in caplog.records)
    assert line_starts(lyrics) == [0, 0]


def test_offset_is_applied_to_word_tokens_too() -> None:
    lyrics = Lyrics.loads("[00:00.000] 作词 : x\n[00:13.590]he[00:14.070]llo")

    _apply_offset(
        lyrics,
        SILENT,
        1000,
        metadata_filter=filter_with_keywords("作词"),
        offset_ms=-400.0,
    )

    assert line_starts(lyrics) == [0, 13190]
    # 第一个词的起点与行时间戳相同，解析器用 ``start=None`` 表示「继承行时间戳」，
    # 因此只有显式带标签的那个词能看到位移。
    assert [token.start for token in lyrics[1].content] == [None, 13670]
    assert min(iter_all_timestamps(lyrics)) >= 0


# --------------------------------------------------------------------------
# 对齐健康检查
# --------------------------------------------------------------------------


def test_all_lines_failing_alignment_is_a_hard_error() -> None:
    """整体对齐失效必须报错，而不是产出一个「没有词级时间」的假成功文件。"""
    with pytest.raises(RuntimeError, match="全部 3 行的对齐都失败了"):
        _check_alignment_health(attempted=3, failed=3)


def test_partial_failure_is_still_a_soft_degradation() -> None:
    _check_alignment_health(attempted=10, failed=1)
    _check_alignment_health(attempted=1, failed=0)


def test_no_attempts_is_not_an_error() -> None:
    """整首歌都被跳过（全都保留原样）时不该报错。"""
    _check_alignment_health(attempted=0, failed=0)


def test_majority_failure_logs_a_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(WARNING):
        _check_alignment_health(attempted=10, failed=6)

    assert any("6/10 行对齐失败" in record.message for record in caplog.records)


# --------------------------------------------------------------------------
# 自动偏移的校验闸门
# --------------------------------------------------------------------------


def _energy_with_band(low: int, high: int, n: int = 400) -> np.ndarray:
    """只有 [low, high) 这些窗口有能量，其余为 0。"""
    energy = np.zeros(n, dtype=np.float32)
    energy[low:high] = 1.0
    return energy


#: 两行歌词 → 一个行区间 [5000ms, 10000ms) → 窗口 [100, 200)。
TWO_LINES = "[00:05.000]a\n[00:10.000]b"


def test_score_line_intervals_is_high_when_the_interval_sits_on_energy() -> None:
    energy = _energy_with_band(100, 200)
    lyrics = Lyrics.loads(TWO_LINES)

    aligned = score_line_intervals(
        energy, lyrics, 0.0, metadata_filter=filter_with_keywords(), window_ms=50.0
    )
    shifted = score_line_intervals(
        energy, lyrics, -2000.0, metadata_filter=filter_with_keywords(), window_ms=50.0
    )

    assert aligned is not None and shifted is not None
    assert aligned > shifted


def test_score_line_intervals_is_none_when_intervals_cover_everything() -> None:
    """区间铺满全曲时判据没有信息量，必须返回 None（而不是 0 或 nan）。"""
    energy = np.ones(400, dtype=np.float32)
    lyrics = Lyrics.loads("[00:00.000]a\n[00:20.000]b")  # [0, 20000) → 覆盖全部窗口

    assert (
        score_line_intervals(
            energy, lyrics, 0.0, metadata_filter=filter_with_keywords(), window_ms=50.0
        )
        is None
    )


def test_validate_rejects_an_offset_that_makes_intervals_worse() -> None:
    energy = _energy_with_band(100, 200)
    lyrics = Lyrics.loads(TWO_LINES)

    accepted, base, candidate = validate_estimated_offset(
        energy,
        lyrics,
        -2000.0,
        metadata_filter=filter_with_keywords(),
        window_ms=50.0,
    )

    assert not accepted
    assert base is not None and candidate is not None and candidate < base


def test_validate_accepts_an_offset_that_improves_intervals() -> None:
    # 能量在第 200–300 窗口：把歌词往后推 5000ms 正好对上
    energy = _energy_with_band(200, 300)
    lyrics = Lyrics.loads("[00:00.000]a\n[00:05.000]b")

    accepted, base, candidate = validate_estimated_offset(
        energy,
        lyrics,
        5000.0,
        metadata_filter=filter_with_keywords(),
        window_ms=50.0,
    )

    assert accepted
    assert base is not None and candidate is not None and candidate > base


def test_validate_trusts_the_estimator_when_the_criterion_is_degenerate() -> None:
    """判据无信息量（区间几乎铺满全曲）时不能否决估计值，只能信任它。"""
    energy = np.ones(400, dtype=np.float32)
    lyrics = Lyrics.loads("[00:00.000]a\n[00:20.000]b")

    accepted, base, candidate = validate_estimated_offset(
        energy,
        lyrics,
        -100.0,  # 区间仍然覆盖几乎全部窗口 → 两侧都无法计算
        metadata_filter=filter_with_keywords(),
        window_ms=50.0,
    )

    assert accepted and base is None and candidate is None


def test_validate_trusts_the_estimator_when_only_the_baseline_is_degenerate() -> None:
    """不偏移时区间铺满全曲、偏移后才有区间外窗口：无从比较，按信任处理。"""
    energy = np.ones(400, dtype=np.float32)
    lyrics = Lyrics.loads("[00:00.000]a\n[00:20.000]b")

    accepted, base, candidate = validate_estimated_offset(
        energy,
        lyrics,
        1234.0,
        metadata_filter=filter_with_keywords(),
        window_ms=50.0,
    )

    assert accepted and base is None and candidate is not None


def test_auto_offset_is_dropped_when_it_fails_validation(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """自动估计不能盲信：行区间对比度没变好就退回不偏移，并明确告警。"""
    from karakara import core

    monkeypatch.setattr(core, "estimate_offset", lambda *_a, **_k: -2000.0)
    energy = _energy_with_band(100, 200)
    lyrics = Lyrics.loads(TWO_LINES)

    with caplog.at_level(WARNING):
        applied = _apply_offset(
            lyrics,
            SILENT,
            1000,
            metadata_filter=filter_with_keywords(),
            estimate=True,
            energy_curve=energy,
        )

    assert applied == 0.0
    assert line_starts(lyrics) == [5000, 10000]
    assert any("未通过校验" in record.message for record in caplog.records)


def test_manual_offset_bypasses_the_validation_gate(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``--offset`` 是用户的显式意图，不该被自动校验拦住。"""
    energy = _energy_with_band(100, 200)
    lyrics = Lyrics.loads(TWO_LINES)

    with caplog.at_level(WARNING):
        applied = _apply_offset(
            lyrics,
            SILENT,
            1000,
            metadata_filter=filter_with_keywords(),
            offset_ms=-2000.0,
            energy_curve=energy,
        )

    assert applied == -2000.0
    assert line_starts(lyrics) == [3000, 8000]
    assert not any("未通过校验" in record.message for record in caplog.records)


def test_auto_offset_is_applied_when_it_passes_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """两道校验都通过时才真的施加：这里 +5000ms 既提升对比度，又与首次人声锚点一致。"""
    from karakara import core

    monkeypatch.setattr(core, "estimate_offset", lambda *_a, **_k: 5000.0)
    energy = _energy_with_band(200, 300)  # 人声在 10s–15s
    lyrics = Lyrics.loads("[00:05.000]a\n[00:10.000]b")  # 加 +5000 正好对上

    applied = _apply_offset(
        lyrics,
        SILENT,
        1000,
        metadata_filter=filter_with_keywords(),
        estimate=True,
        energy_curve=energy,
    )

    assert applied == 5000.0
    assert line_starts(lyrics) == [10000, 15000]


def test_auto_offset_is_dropped_when_the_vocal_anchor_disagrees(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """与「首次持续人声」锚点分歧过大时按不偏移处理。

    真实场景就是这一条救下的：某首歌的 LRC 与音频属于不同剪辑（需要 +17 秒），
    而能量判据给出 −600ms —— 两者无法调和，此时**任何**全局常量偏移都不成立。
    """
    from karakara import core

    monkeypatch.setattr(core, "estimate_offset", lambda *_a, **_k: 5000.0)
    energy = _energy_with_band(200, 300)  # 人声在 10s，即锚点指向 +10000ms
    lyrics = Lyrics.loads("[00:00.000]a\n[00:05.000]b")

    with caplog.at_level(WARNING):
        applied = _apply_offset(
            lyrics,
            SILENT,
            1000,
            metadata_filter=filter_with_keywords(),
            estimate=True,
            energy_curve=energy,
        )

    assert applied == 0.0
    assert line_starts(lyrics) == [0, 5000]
    assert any("分歧超过" in record.message for record in caplog.records)
    assert any("不同剪辑" in record.message for record in caplog.records)


def test_estimation_is_off_by_default(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """回归：不显式要求时**绝不**自动估计。

    实测自动估计误判偏多（两条能量判据各自都会错、且错在不同的歌上），所以默认按不偏移
    处理。这里把估计器换成一个「一调用就炸」的替身：只要默认路径碰它，用例立刻失败。
    """
    from karakara import core

    def explode(*_args: object, **_kwargs: object) -> float:
        raise AssertionError("默认路径不该调用 estimate_offset")

    monkeypatch.setattr(core, "estimate_offset", explode)
    lyrics = Lyrics.loads(TWO_LINES)

    with caplog.at_level(INFO):
        applied = _apply_offset(
            lyrics,
            SILENT,
            1000,
            metadata_filter=filter_with_keywords(),
            energy_curve=_energy_with_band(200, 300),
        )

    assert applied == 0.0
    assert line_starts(lyrics) == [5000, 10000]
    assert any("自动偏移估计未启用" in record.message for record in caplog.records)


def test_manual_offset_and_estimation_are_mutually_exclusive() -> None:
    """同时给手动偏移与自动估计是调用方的 bug，要立刻报错而不是静默取其一。"""
    with pytest.raises(ValueError, match="只能给一个"):
        _apply_offset(
            Lyrics.loads(TWO_LINES),
            SILENT,
            1000,
            metadata_filter=filter_with_keywords(),
            offset_ms=500.0,
            estimate=True,
        )


# --------------------------------------------------------------------------
# 首次持续人声锚点
# --------------------------------------------------------------------------


def test_detect_first_vocal_onset_finds_the_first_sustained_run() -> None:
    energy = np.zeros(200, dtype=np.float32)
    energy[80:120] = 1.0  # 4s 处开始持续人声

    assert detect_first_vocal_onset(energy, window_ms=50.0) == 4000.0


def test_detect_first_vocal_onset_ignores_a_short_blip() -> None:
    """单个窗口的尖峰不算「持续人声」（分离残留/打击乐）。"""
    energy = np.zeros(200, dtype=np.float32)
    energy[40] = 1.0  # 一个孤立窗口
    energy[120:160] = 1.0

    assert detect_first_vocal_onset(energy, window_ms=50.0) == 6000.0


def test_detect_first_vocal_onset_returns_none_for_silence() -> None:
    assert detect_first_vocal_onset(np.zeros(200, dtype=np.float32)) is None


def test_suggest_offset_from_onset_sign_convention() -> None:
    """锚点符号与 estimate_offset 一致：正值 = 歌词偏早、需要延后。"""
    energy = np.zeros(400, dtype=np.float32)
    energy[200:300] = 1.0  # 人声从 10s 开始
    early = Lyrics.loads("[00:08.000]a\n[00:12.000]b")  # 歌词偏早 2s

    anchor = suggest_offset_from_onset(
        early, energy, metadata_filter=filter_with_keywords(), window_ms=50.0
    )

    assert anchor == 2000.0


def test_suggest_offset_from_onset_returns_none_without_lyrics() -> None:
    energy = np.zeros(100, dtype=np.float32)
    energy[50:80] = 1.0
    only_metadata = Lyrics.loads("[00:00.000] 作词 : x")

    assert (
        suggest_offset_from_onset(
            only_metadata,
            energy,
            metadata_filter=filter_with_keywords("作词"),
            window_ms=50.0,
        )
        is None
    )
