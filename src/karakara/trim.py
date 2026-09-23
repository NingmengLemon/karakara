"""把行的音频窗口裁到「人声结束」：参数、守卫与判定。

问题：送给对齐器的片段是 ``[本行起点, 下一行起点)``。一行唱完之后如果还有长间奏，这段
窗口的后半截全是间奏；而对齐器（HubertFA）的输出是对整段的**连续划分**，最后一个单元会
把剩下的时间吃掉，产物里的行尾正是它的结束时刻。于是「唱完之后剩下的内容全是间奏」变成
「这一行的高亮一直拖到间奏结束」。

实测规模（8 首歌 333 行，见 ``docs/records/2026-09-23-line-tail-trimming.md``）：绝大多数
行的行尾是准的（中位比人声结束早 10ms），但 1.5% 的行晚于 1s，最严重的一例晚了 88 秒；
**最后一行**尤其容易命中（它的窗口右端是音频末尾，命中率是其余行的 30 倍）。

本模块只做「该不该裁、裁到哪」的判定，不碰音频。默认关闭（``--trim-line-tail``），因为
判据有一个已知的失败模式：**安静间隙之后的短促人声会被漏掉**，那会导致把真唱切掉。

裁了之后对产物有什么影响（同样 8 首歌，每个会被裁的行各跑一次裁剪后的对齐）：15 行里
行尾变化中位 **−11ms**，3 行大幅提前（−2.9s / −3.8s / −88.3s），11 行落在 ±300ms 内，
**没有一行被推后超过 300ms**。也就是说它几乎只动该动的那些行。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from karakara.offset import detect_last_vocal_activity

#: 裁剪判定用的能量曲线分辨率（毫秒）。比偏移估计用的 50ms 细，因为它要定位「最后一个
#: 音节什么时候结束」。
TRIM_WINDOW_MS = 20.0


@dataclass(frozen=True)
class TailTrimConfig:
    """裁剪参数。默认值来自那次评估，改动前请重跑同一套度量。

    Attributes:
        relative_threshold: 判定「有人声」的阈值，相对全曲峰值（能量曲线归一化到 1.0）。
        min_run_ms: 判定「持续有人声」的最短时长。
        margin_ms: 在人声结束之后保留的余量，给混响尾巴、呼吸与检测误差留一点空间。
            实测（8 首歌里会被裁的 15 行，各跑一次裁剪后的对齐）：余量 300ms 时行尾变化
            中位 **−11ms**、最大推后 +121ms，没有一行被推后超过 300ms；也就是说这个余量
            不会把本来准的行明显推后，而它挡住了检测偏早的风险。
        min_tail_ms: 尾部无人声段短于这个值就不裁（裁了没意义，见评估里的对照组）。
        guard_active_ratio: 裁剪点之后「超过阈值」的窗口占比达到这个值就**放弃裁剪**。
            这条守卫专门对付「漏检」：判据只认连续 ≥ ``min_run_ms`` 的活跃段，所以
            **突发式**的人声（每段都短于门槛）会被漏掉，而它的能量明明很高。用「占比」
            而不是「平均能量」是因为后者会被很长的静音尾巴摊薄——余量越小、检查区域越长，
            均值就越钝。
        min_keep_ms: 裁剪后至少要保留这么多音频，避免判据抽风时产出退化片段。
    """

    relative_threshold: float = 0.05
    min_run_ms: float = 200.0
    margin_ms: int = 300
    min_tail_ms: int = 1000
    guard_active_ratio: float = 0.05
    min_keep_ms: int = 500


@dataclass(frozen=True)
class TrimDecision:
    """一次裁剪判定的结果。

    Attributes:
        end_ms: 最终采用的窗口右端（没裁就是原值）。
        trimmed_ms: 裁掉了多少毫秒（0 表示没裁）。
        reason: 人话说明，进日志用。
    """

    end_ms: int
    trimmed_ms: int = 0
    reason: str = ""

    @property
    def trimmed(self) -> bool:
        return self.trimmed_ms > 0


def trim_window_end(
    energy: NDArray[np.float32],
    start_ms: float,
    end_ms: float,
    *,
    config: TailTrimConfig | None = None,
    window_ms: float = TRIM_WINDOW_MS,
) -> TrimDecision:
    """按 :class:`TailTrimConfig` 决定这一行的窗口右端要不要提前。

    Args:
        energy: :func:`karakara.offset.build_energy_curve` 产出的能量曲线（归一化到 1.0）。
        start_ms: 本行窗口左端。
        end_ms: 本行窗口右端（通常是下一行的起点，最后一行是音频末尾）。
        config: 参数；``None`` 表示关闭（原样返回）。
        window_ms: ``energy`` 的分辨率，必须与建曲线时一致。

    Returns:
        判定结果。任何一条守卫不通过都返回「不裁」，并把原因写在 ``reason`` 里。
    """
    if config is None or end_ms <= start_ms:
        return TrimDecision(end_ms=int(end_ms), reason="未启用")

    low = max(0, int(start_ms / window_ms))
    high = min(len(energy), int(-(-end_ms // window_ms)))
    if high - low < 2:
        return TrimDecision(end_ms=int(end_ms), reason="窗口太短")

    segment = energy[low:high]
    vocal_end = detect_last_vocal_activity(
        segment,
        window_ms=window_ms,
        relative_threshold=config.relative_threshold,
        min_run_ms=config.min_run_ms,
    )
    if vocal_end is None:
        return TrimDecision(end_ms=int(end_ms), reason="窗口内检不出持续人声")

    window_ms_total = end_ms - start_ms
    tail_ms = window_ms_total - vocal_end
    if tail_ms < config.min_tail_ms:
        return TrimDecision(
            end_ms=int(end_ms), reason=f"尾部只有 {tail_ms:.0f}ms，不值得裁"
        )

    cut_ms = vocal_end + config.margin_ms
    if cut_ms < config.min_keep_ms:
        return TrimDecision(
            end_ms=int(end_ms), reason=f"裁剪后会只剩 {cut_ms:.0f}ms，太短"
        )
    if cut_ms >= window_ms_total:
        return TrimDecision(end_ms=int(end_ms), reason="余量已覆盖整个尾部")

    # 守卫：裁剪点之后必须真的安静，否则说明判据漏检了。
    # 用「超过阈值的窗口占比」而不是平均能量：突发式人声的平均值会被长静音摊薄，占比不会。
    guard_low = max(0, int((start_ms + cut_ms) / window_ms))
    if high > guard_low:
        beyond = segment[guard_low - low : high - low]
        loud_ratio = float((beyond > config.relative_threshold).mean())
        if loud_ratio > config.guard_active_ratio:
            return TrimDecision(
                end_ms=int(end_ms),
                reason=f"裁剪点之后 {loud_ratio:.0%} 的时间仍有人声"
                f"（超过 {config.guard_active_ratio:.0%}），判据可能漏检",
            )

    return TrimDecision(
        end_ms=int(start_ms + cut_ms),
        trimmed_ms=int(tail_ms - config.margin_ms),
        reason=f"人声结束于 {vocal_end:.0f}ms，尾部 {tail_ms:.0f}ms 间奏，"
        f"保留 {config.margin_ms}ms 余量",
    )
