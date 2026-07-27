"""
offset.py

基于音频能量曲线与 LRC 时间戳的滑动窗口互相关，估计歌词全局时间偏移量。

核心思路：
  1. 从人声音频构建 RMS 能量曲线（下采样到每 window_ms 一个值）
  2. 从 LRC 歌词构建二值「人声指示」曲线（有歌词行的时间区间标记为 1）
  3. 滑动窗口互相关搜索，找到使两条曲线最匹配的时间偏移量
"""

from __future__ import annotations

from logging import getLogger

import numpy as np
from lemony_lrc_parser import Lyrics
from numpy.typing import NDArray

from karakara.utils.metadata import MetadataFilter

logger = getLogger(__name__)


def build_energy_curve(
    audio: NDArray[np.float32],
    sample_rate: int,
    window_ms: float = 50.0,
) -> NDArray[np.float32]:
    """构建音频 RMS 能量曲线。

    将音频下采样到每窗口一个 RMS 值，并归一化到 [0, 1]。

    Args:
        audio: 单声道或立体声音频，shape (samples,) 或 (channels, samples)
        sample_rate: 采样率 (Hz)
        window_ms: 窗口长度 (ms)

    Returns:
        归一化 RMS 能量曲线，shape (n_windows,)，值域 [0, 1]
    """
    if audio.ndim == 2:
        audio = audio.mean(axis=0)  # 多通道取均值 → 单声道

    total_samples = audio.shape[-1]
    window_samples = max(1, int(sample_rate * window_ms / 1000))
    n_windows = total_samples // window_samples

    if n_windows < 2:
        return np.zeros(1, dtype=np.float32)

    # reshape → 每行一个窗口，向量化计算 RMS
    trimmed = audio[: n_windows * window_samples]
    frames = trimmed.reshape(n_windows, window_samples)
    energy = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1)).astype(np.float32)

    e_max = float(energy.max())
    if e_max > 1e-10:
        energy = energy / e_max  # type: ignore[assignment]

    return energy  # type: ignore[no-any-return]


def score_vocal_activity(
    energy: NDArray[np.float32],
    start_ms: float,
    end_ms: float,
    *,
    window_ms: float = 50.0,
) -> float:
    """计算歌词时间段内的归一化人声活动度。

    ``energy`` 应由 :func:`build_energy_curve` 生成。返回值为该时间段内
    RMS 能量的均值，范围为 ``[0, 1]``；区间为空或超出音频范围时返回 ``0``。
    """
    if end_ms <= start_ms or energy.size == 0:
        return 0.0

    start_window = max(0, int(start_ms / window_ms))
    end_window = min(len(energy), int(np.ceil(end_ms / window_ms)))
    if end_window <= start_window:
        return 0.0
    return float(energy[start_window:end_window].mean())


def _build_lrc_presence_curve(
    lyrics: Lyrics,
    n_windows: int,
    total_duration_ms: float,
    *,
    metadata_filter: MetadataFilter,
    window_ms: float = 50.0,
) -> NDArray[np.float32]:
    """构建 LRC 时间戳「人声指示」曲线。

    对每个歌词行，将该行的时间区间标记为 1（有人声），其余为 0。
    跳过元数据行和无时间戳行。

    Args:
        lyrics: 已解析的 LRC 歌词对象
        n_windows: 与能量曲线相同的窗口数
        total_duration_ms: 音频总时长 (ms)，用于裁剪超出行尾
        window_ms: 窗口长度 (ms)

    Returns:
        二值指示曲线，shape (n_windows,)，dtype float32
    """
    presence = np.zeros(n_windows, dtype=np.float32)

    for line in lyrics:
        if line.start is None:
            continue

        # 跳过元数据行
        text = ""
        if len(line.content) == 1:
            text = line.content[0].content
        if metadata_filter(text):
            continue

        start_ms = line.start
        end_ms = (
            line.end
            if line.end is not None
            else min(start_ms + 5000, total_duration_ms)
        )

        start_win = int(start_ms / window_ms)
        end_win = int(end_ms / window_ms)
        start_win = max(0, min(start_win, n_windows - 1))
        end_win = max(start_win, min(end_win, n_windows - 1))
        presence[start_win : end_win + 1] = 1.0

    return presence


def _score_at_offset(
    energy: NDArray[np.float32],
    presence: NDArray[np.float32],
    offset_windows: int,
) -> float:
    """计算在给定窗口偏移下两条曲线的重叠得分（内积）。

    offset_windows > 0 表示将 presence 曲线右移（LRC 延迟），
    即 energy 取后半截、presence 取前半截做内积。

    Args:
        energy: 音频能量曲线
        presence: LRC 指示曲线
        offset_windows: 窗口级偏移量

    Returns:
        内积得分，值越大表示匹配越好
    """
    if offset_windows > 0:
        return float(np.dot(energy[offset_windows:], presence[:-offset_windows]))
    elif offset_windows < 0:
        o = -offset_windows
        return float(np.dot(energy[:-o], presence[o:]))
    else:
        return float(np.dot(energy, presence))


def estimate_offset(
    audio: NDArray[np.float32],
    lyrics: Lyrics,
    sample_rate: int,
    *,
    metadata_filter: MetadataFilter,
    window_ms: float = 50.0,
    max_offset_s: float = 30.0,
    coarse_step_ms: float = 200.0,
) -> float:
    """估计 LRC 时间戳与音频实际人声位置之间的全局时间偏移。

    采用两级搜索：
      1. 粗搜索：以 coarse_step_ms 步长在 ±max_offset_s 范围内扫一遍
      2. 精搜索：在最佳粗搜索点附近逐窗口搜索（window_ms 步长）

    Args:
        audio: 人声分离后的音频，shape (channels, samples) 或 (samples,)
        lyrics: 已解析的 LRC 歌词
        sample_rate: 采样率 (Hz)
        metadata_filter: 元数据行过滤器。
        window_ms: 能量计算与精搜索窗口 (ms)，默认 50ms
        max_offset_s: 最大搜索偏移范围 (秒)，默认 ±30s
        coarse_step_ms: 粗搜索步长 (ms)，默认 200ms

    Returns:
        偏移量 (ms)。
        * 正值：LRC 时间戳偏早（音频中人声晚于 LRC 标记），需延迟 LRC
        * 负值：LRC 时间戳偏晚，需提前 LRC
        * 0.0：无需调整或无法估计
    """
    # ---------- 构建两条曲线 ----------
    energy = build_energy_curve(audio, sample_rate, window_ms)
    n_windows = len(energy)

    if n_windows < 2:
        logger.warning("Audio too short for offset estimation")
        return 0.0

    total_duration_ms = (audio.shape[-1] / sample_rate) * 1000
    presence = _build_lrc_presence_curve(
        lyrics,
        n_windows,
        total_duration_ms,
        metadata_filter=metadata_filter,
        window_ms=window_ms,
    )

    if presence.sum() < 1:
        logger.warning(
            "No LRC lines with valid timestamps found, skipping offset estimation"
        )
        return 0.0

    max_offset_windows = int(max_offset_s * 1000 / window_ms)
    coarse_step_windows = max(1, int(coarse_step_ms / window_ms))

    # ---------- 粗搜索 ----------
    best_offset = 0
    best_score = _score_at_offset(energy, presence, 0)

    for offset in range(
        -max_offset_windows, max_offset_windows + 1, coarse_step_windows
    ):
        score = _score_at_offset(energy, presence, offset)
        if score > best_score:
            best_score = score
            best_offset = offset

    # ---------- 精搜索 ----------
    fine_start = max(-max_offset_windows, best_offset - coarse_step_windows)
    fine_end = min(max_offset_windows, best_offset + coarse_step_windows)
    for offset in range(fine_start, fine_end + 1):
        score = _score_at_offset(energy, presence, offset)
        if score > best_score:
            best_score = score
            best_offset = offset

    offset_ms = best_offset * window_ms
    logger.info(
        f"Offset estimation result: {offset_ms:+.0f}ms "
        f"(score={best_score:.4f}, search_range=±{max_offset_s:.0f}s, "
        f"coarse_step={coarse_step_ms:.0f}ms, window={window_ms:.0f}ms)"
    )
    return offset_ms
