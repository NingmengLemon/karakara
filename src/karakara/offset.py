"""
offset.py

基于音频能量曲线与 LRC 行首（onset）的全局时间偏移估计。

核心思路：
  1. 从人声音频构建 RMS 能量曲线（下采样到每 window_ms 一个值）
  2. 从 LRC 歌词构建「行首指示」曲线：**只在每行时间戳处**标记一个窄窗
  3. 两级搜索，找到使「行首窗内人声能量」相对「其余位置人声能量」对比度最大的偏移

三个关键设计，各自都在合成用例上做过前后对比：

1. **行首窗，而不是整行区间。**
   LRC 的行级时间戳断言的是「这一行的人声从这里开始」，而不是「这一段时间里
   一直有人声」。早期实现按 ``[start, start + 5000ms)`` 的方块标记，密集排布的
   歌词会让指示曲线退化成近乎全 1 的实心块，此时任何打分函数都只能对齐两条曲线
   的重心/边缘而非起唱点。实测（行距 2–4s）偏差达 −1.0 ~ −3.1 秒。

2. **归一化的判别式对比度得分，而不是裸点积。**
   ``sum(energy * presence)`` 会随重叠长度单调增长，等于在惩罚偏移量本身而不是
   衡量匹配度。改用的得分是

       (行首窗内平均能量 − 其余位置平均能量)

   且分母固定为未偏移时的行首窗总数，被推出音频范围的行首窗按 0 计入。这样
   「把歌词推出歌外」这类偏移会被连续扣分，而不是靠一个小样本均值刷高分。

3. **平局优先取更小的偏移量。**
   早期实现保留扫描中第一个严格最大值，于是平局时结果被推到搜索区间的极端。

已知局限：本估计器只有**全局常量偏移**一个自由度，无法处理逐行漂移。而且当密集
排布的行首「梳齿」整体落在同一个长人声区间内时，区间内存在一段真实无法区分的
平台；此时由平局规则给出结果。这类平台对下游影响有限——行首仍落在人声区间内，
逐行对齐拿到的音频片段依然正确。
"""

from __future__ import annotations

import itertools
from logging import getLogger

import numpy as np
from lemony_lrc_parser import Lyrics
from numpy.typing import NDArray

from karakara.utils.metadata import MetadataFilter

logger = getLogger(__name__)

#: 行首窗的默认长度 (ms)。略长于能量曲线的窗口，以容忍 LRC 时间戳的量化误差。
DEFAULT_ONSET_MS = 300.0

#: 行首窗的最小保留率。低于该值的偏移直接判为无效，作为兜底。
DEFAULT_MIN_PRESENCE_COVERAGE = 0.8

#: 得分差小于该阈值即视为平局，按「优先取更小偏移量」处理。
#: 目的是让结果稳定可复现，而不是由浮点累加噪声决定。
TIE_EPSILON = 1e-6


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


def build_line_intervals(
    lyrics: Lyrics,
    offset_ms: float,
    n_windows: int,
    *,
    metadata_filter: MetadataFilter,
    window_ms: float = 50.0,
) -> NDArray[np.bool_]:
    """标记「非元数据歌词行所覆盖的窗口」，返回布尔掩码。

    区间取 ``[行 i 起点 + offset, 行 i+1 起点 + offset)``，与 ``core`` 切段送对齐
    的口径一致；越界部分被裁到 ``[0, n_windows)``。
    """
    inside = np.zeros(n_windows, dtype=bool)
    starts = sorted(
        line.start
        for line in lyrics
        if line.start is not None and not metadata_filter(line.text)
    )
    for begin, end in itertools.pairwise(starts):
        low = max(0, min(int((begin + offset_ms) / window_ms), n_windows - 1))
        high = max(0, min(int((end + offset_ms) / window_ms), n_windows - 1))
        if high > low:
            inside[low : high + 1] = True
    return inside


def score_line_intervals(
    energy: NDArray[np.float32],
    lyrics: Lyrics,
    offset_ms: float,
    *,
    metadata_filter: MetadataFilter,
    window_ms: float = 50.0,
) -> float | None:
    """行区间对比度：区间内平均能量 − 区间外平均能量。

    这是与 :func:`estimate_offset` 的内部判据**互补**的第二个统计量：内部判据只看
    每行的行首窄窗（「这一行是不是从这里开始唱」），本函数看整行区间（「这一段时间
    里是不是都有人声」）。后者正是下游逐行切段去对齐所依赖的性质。

    两条判据在真实曲目上确实会分歧：实测 3 首里 2 首，估计器偏好的偏移按本判据
    **比完全不偏移更差**（见 :func:`validate_estimated_offset`）。

    区间覆盖过少或过多时判据没有信息量（例如区间几乎铺满全曲），返回 ``None``。
    """
    n_windows = len(energy)
    if n_windows < 4:
        return None
    inside = build_line_intervals(
        lyrics,
        offset_ms,
        n_windows,
        metadata_filter=metadata_filter,
        window_ms=window_ms,
    )
    n_inside = int(inside.sum())
    if n_inside < 2 or n_windows - n_inside < 2:
        return None
    return float(energy[inside].mean() - energy[~inside].mean())


def validate_estimated_offset(
    energy: NDArray[np.float32],
    lyrics: Lyrics,
    offset_ms: float,
    *,
    metadata_filter: MetadataFilter,
    window_ms: float = 50.0,
) -> tuple[bool, float | None, float | None]:
    """自动估计出来的偏移是否**真的**优于「一点都不偏移」。

    为什么需要这道闸门：``estimate_offset`` 的打分只看行首窄窗，因此它可能给出一个
    让行首仍然落在人声里、却把整段行区间推出人声的偏移。三首真实曲目的实测
    （``UVR_Demucs_Model_1`` 实物分离；「物理真值」= 第一次持续人声出现的位置与
    第一条歌词行时间戳之差，见 :func:`suggest_offset_from_onset`）：

    ==============  ==========  ==========  ===========
    曲目             物理真值    onset 判据   interval 判据
    ==============  ==========  ==========  ===========
    Saya - 失う      ≈ −1040ms     −4000ms      −1000ms
    ReoNa - SACRA    ≈   +270ms      +200ms      +4600ms
    Rick Astley      ≈ +17160ms      −600ms      +7600ms
    ==============  ==========  ==========  ===========

    两条判据**各自都会错，而且错在不同的歌上**：Saya 上 onset 差 3 秒而 interval
    很准；SACRA 上 interval 差 4 秒而 onset 很准。所以这道闸门的目的不是「选出正确
    的偏移」（做不到），而是**在证据不足时不动**——自动偏移宁可不动，也不要动错。
    手动 ``--offset`` 不受这道闸门影响（用户的显式意图优先）。

    Returns:
        ``(是否采纳, 不偏移时的对比度, 估计值下的对比度)``。
        判据退化（无信息量）时返回 ``(True, None, None)``——即信任估计器。
    """
    base = score_line_intervals(
        energy, lyrics, 0.0, metadata_filter=metadata_filter, window_ms=window_ms
    )
    candidate = score_line_intervals(
        energy, lyrics, offset_ms, metadata_filter=metadata_filter, window_ms=window_ms
    )
    if base is None or candidate is None:
        return True, base, candidate
    return candidate > base, base, candidate


#: 默认的「持续人声」判据：能量连续超过峰值这个比例，且至少持续这么多窗口。
DEFAULT_VOCAL_RELATIVE_THRESHOLD = 0.12
DEFAULT_VOCAL_MIN_RUN_WINDOWS = 6

#: 能量判据与「首次人声锚点」之间允许的分歧（ms）。超过它说明两者无法调和
#: ——通常是 LRC 与音频属于不同剪辑，此时**任何**全局常量偏移都是错的。
#: 2 秒是"量级级"的阈值（锚点本身的精度约 ±0.3s），不是质量门槛。
DEFAULT_ANCHOR_TOLERANCE_MS = 2000.0


def detect_first_vocal_onset(
    energy: NDArray[np.float32],
    *,
    window_ms: float = 50.0,
    relative_threshold: float = DEFAULT_VOCAL_RELATIVE_THRESHOLD,
    min_run_windows: int = DEFAULT_VOCAL_MIN_RUN_WINDOWS,
) -> float | None:
    """第一次「持续有人声」的时刻（ms），检不出返回 ``None``。

    这个判据只依赖一件事——**人声从哪一刻开始**，不依赖能量判据的形式，因此可以
    当作 :func:`estimate_offset` 之外的独立锚点。歌曲开头一般先是器乐前奏，所以
    「第一条歌词行的时间戳」应当落在它附近。
    """
    if energy.size == 0:
        return None
    threshold = float(energy.max()) * relative_threshold
    streak = 0
    for index, value in enumerate(energy):
        streak = streak + 1 if float(value) > threshold else 0
        if streak >= min_run_windows:
            return float(index - min_run_windows + 1) * window_ms
    return None


def first_lyric_timestamp(
    lyrics: Lyrics, *, metadata_filter: MetadataFilter
) -> int | None:
    """第一条**参与对齐**的歌词行的时间戳（ms），没有则 ``None``。"""
    starts = [
        line.start
        for line in lyrics
        if line.start is not None and line.text and not metadata_filter(line.text)
    ]
    return min(starts) if starts else None


def detect_last_vocal_activity(
    energy: NDArray[np.float32],
    *,
    window_ms: float = 50.0,
    relative_threshold: float = DEFAULT_VOCAL_RELATIVE_THRESHOLD,
    min_run_ms: float = 200.0,
) -> int | None:
    """最后一次「持续有人声」的结束时刻（ms），检不出返回 ``None``。

    与 :func:`detect_first_vocal_onset` 对称：只看「人声到哪一刻还在」，不依赖某个能量
    判据的形式。它服务的是「一行唱完之后还剩多少间奏」这个问题（见 :mod:`karakara.trim`）。

    判据是「连续 ``min_run_ms`` 内每窗都超过阈值」，因此**安静间隙之后的短促人声会被
    漏掉**。这是已知的失败模式，调用方需要自己加守卫（``karakara.trim`` 里有一条）。
    """
    if energy.size == 0:
        return None
    threshold = float(energy.max()) * relative_threshold
    min_run = max(1, round(min_run_ms / window_ms))

    index = len(energy) - 1
    while index >= 0:
        if float(energy[index]) <= threshold:
            index -= 1
            continue
        end = index
        while index >= 0 and float(energy[index]) > threshold:
            index -= 1
        if end - index >= min_run:
            return int((end + 1) * window_ms)
    return None


def suggest_offset_from_onset(
    lyrics: Lyrics,
    energy: NDArray[np.float32],
    *,
    metadata_filter: MetadataFilter,
    window_ms: float = 50.0,
    relative_threshold: float = DEFAULT_VOCAL_RELATIVE_THRESHOLD,
    min_run_windows: int = DEFAULT_VOCAL_MIN_RUN_WINDOWS,
) -> float | None:
    """用「第一条歌词行 ↔ 第一次持续人声」给出的偏移锚点（ms）。

    正负号与 :func:`estimate_offset` 一致：正值表示歌词偏早、需要延后。

    已知偏差来源（因此它只做**粗锚点**，不做精细估计）：副歌前的呼吸/哼唱、
    第一行之前的无词人声、以及分离残留的伴奏都会被算成「人声已开始」。
    """
    first_line = first_lyric_timestamp(lyrics, metadata_filter=metadata_filter)
    if first_line is None:
        return None
    onset = detect_first_vocal_onset(
        energy,
        window_ms=window_ms,
        relative_threshold=relative_threshold,
        min_run_windows=min_run_windows,
    )
    if onset is None:
        return None
    return onset - float(first_line)


def _build_lrc_presence_curve(
    lyrics: Lyrics,
    n_windows: int,
    total_duration_ms: float,
    *,
    metadata_filter: MetadataFilter,
    window_ms: float = 50.0,
    onset_ms: float = DEFAULT_ONSET_MS,
) -> NDArray[np.float32]:
    """构建 LRC「行首指示」曲线。

    对每个有效的歌词行，只在其时间戳开始的 ``onset_ms`` 窄窗内标记 1。
    跳过元数据行、无时间戳行，以及起点已经落在音频范围之外的行（后者无法被
    任何偏移量救回，纳入只会让固定分母失真）。

    Args:
        lyrics: 已解析的 LRC 歌词对象
        n_windows: 与能量曲线相同的窗口数
        total_duration_ms: 音频总时长 (ms)，用于剔除越界行
        metadata_filter: 元数据行过滤器
        window_ms: 窗口长度 (ms)
        onset_ms: 行首窗长度 (ms)

    Returns:
        二值行首指示曲线，shape (n_windows,), dtype float32
    """
    presence = np.zeros(n_windows, dtype=np.float32)
    onset_windows = max(1, round(onset_ms / window_ms))

    kept = 0
    for line in lyrics:
        if line.start is None:
            continue
        if metadata_filter(line.text):
            continue
        if line.start >= total_duration_ms:
            continue

        start_win = int(line.start / window_ms)
        if start_win >= n_windows:
            continue
        start_win = max(0, start_win)
        end_win = min(n_windows - 1, start_win + onset_windows - 1)
        presence[start_win : end_win + 1] = 1.0
        kept += 1

    logger.debug(
        f"built onset curve: {kept} lines, {onset_windows} windows each "
        f"({onset_ms:.0f}ms), {int(presence.sum())} marked windows"
    )
    return presence


def _score_at_offset(
    energy: NDArray[np.float32],
    presence: NDArray[np.float32],
    offset_windows: int,
    *,
    total_presence: float,
    min_presence_coverage: float = DEFAULT_MIN_PRESENCE_COVERAGE,
) -> float:
    """计算在给定窗口偏移下「行首位置有人声」的判别式对比度得分。

    offset_windows > 0 表示将 presence 曲线右移（LRC 延迟），
    即 energy 取后半截、presence 取前半截比较。

    得分为 ``行首窗内平均能量 − 其余位置平均能量``。``energy`` 由
    :func:`build_energy_curve` 产生、值域 [0, 1]，因此得分天然落在 [-1, 1]。

    两处刻意的设计，都是为了避免「靠减少参与统计的窗口数刷分」：

    1. **分母固定为 ``total_presence``**（未偏移时的行首窗总数），而不是重叠区
       内实际保留的行首窗数。被推出音频范围的行首窗按 0 计入，于是
       「把歌词推出歌外」这类偏移会被连续地扣分，而不是靠一个小样本均值拿高分。
    2. **不做按尺度（标准差）归一化**。某个偏移量下的切片统计量不能当作全局
       尺度，否则排除掉信息区的偏移量会因为切片标准差趋近 0 而刷出虚高得分
       （实测可导致数十秒的错误估计）。

    ``min_presence_coverage`` 只作为兜底：保留率低于该值的偏移直接判为无效，
    避免在歌词几乎全部越界时给出无意义的结果。

    Args:
        energy: 由 :func:`build_energy_curve` 生成的能量曲线
        presence: LRC 行首指示曲线
        offset_windows: 窗口级偏移量
        total_presence: 未偏移时 presence 的总和，即固定分母
        min_presence_coverage: 行首窗最小保留率（兜底检查）

    Returns:
        对比度得分，无效偏移返回 ``-inf``
    """
    if offset_windows > 0:
        e = energy[offset_windows:]
        p = presence[:-offset_windows]
    elif offset_windows < 0:
        k = -offset_windows
        e = energy[:-k]
        p = presence[k:]
    else:
        e = energy
        p = presence

    if e.size < 2 or e.size != p.size or total_presence < 1:
        return float("-inf")

    hit = p > 0.5
    n_hit = int(hit.sum())
    if n_hit < 1 or n_hit == p.size:
        return float("-inf")
    if n_hit < min_presence_coverage * total_presence:
        return float("-inf")

    on_mean = float(e[hit].sum()) / total_presence
    off_mean = float(e[~hit].mean())
    return on_mean - off_mean


def estimate_offset(
    audio: NDArray[np.float32],
    lyrics: Lyrics,
    sample_rate: int,
    *,
    metadata_filter: MetadataFilter,
    window_ms: float = 50.0,
    max_offset_s: float = 30.0,
    coarse_step_ms: float = 200.0,
    onset_ms: float = DEFAULT_ONSET_MS,
    min_presence_coverage: float = DEFAULT_MIN_PRESENCE_COVERAGE,
) -> float:
    """估计 LRC 时间戳与音频实际起唱位置之间的全局时间偏移。

    采用两级搜索：
      1. 粗搜索：以 coarse_step_ms 步长在 ±max_offset_s 范围内扫一遍
      2. 精搜索：在最佳粗搜索点附近逐窗口搜索（window_ms 步长）

    匹配目标是原始能量曲线：每一行的行首都应该落在有人声的位置上，因此每一行
    都在贡献约束。得分近似平局时优先取绝对值更小的偏移量。

    Args:
        audio: 人声分离后的音频，shape (channels, samples) 或 (samples,)
        lyrics: 已解析的 LRC 歌词
        sample_rate: 采样率 (Hz)
        metadata_filter: 元数据行过滤器。
        window_ms: 能量计算与精搜索窗口 (ms)，默认 50ms
        max_offset_s: 最大搜索偏移范围 (秒)，默认 ±30s
        coarse_step_ms: 粗搜索步长 (ms)，默认 200ms
        onset_ms: 行首窗长度 (ms)，默认 300ms
        min_presence_coverage: 行首窗最小保留率

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
        onset_ms=onset_ms,
    )

    total_presence = float(presence.sum())
    if total_presence < 1:
        logger.warning(
            "No LRC lines with valid timestamps found, skipping offset estimation"
        )
        return 0.0

    def score(offset: int) -> float:
        return _score_at_offset(
            energy,
            presence,
            offset,
            total_presence=total_presence,
            min_presence_coverage=min_presence_coverage,
        )

    def better(candidate: float, best: float, offset: int, best_offset: int) -> bool:
        """更优则胜出；平局（含近似平局）时取绝对值更小的偏移量。

        偏移量的先验集中在 0 附近，因此把近平分交给这条规则，比让浮点累加
        噪声决定结果更稳定、也更可复现。
        """
        if candidate > best + TIE_EPSILON:
            return True
        if candidate < best - TIE_EPSILON:
            return False
        return abs(offset) < abs(best_offset)

    max_offset_windows = int(max_offset_s * 1000 / window_ms)
    coarse_step_windows = max(1, int(coarse_step_ms / window_ms))

    # ---------- 粗搜索 ----------
    best_offset = 0
    best_score = score(0)

    for offset in range(
        -max_offset_windows, max_offset_windows + 1, coarse_step_windows
    ):
        candidate = score(offset)
        if better(candidate, best_score, offset, best_offset):
            best_score = candidate
            best_offset = offset

    # ---------- 精搜索 ----------
    fine_start = max(-max_offset_windows, best_offset - coarse_step_windows)
    fine_end = min(max_offset_windows, best_offset + coarse_step_windows)
    for offset in range(fine_start, fine_end + 1):
        candidate = score(offset)
        if better(candidate, best_score, offset, best_offset):
            best_score = candidate
            best_offset = offset

    if best_score == float("-inf"):
        logger.warning(
            f"Offset estimation failed: no valid offset within ±{max_offset_s:.0f}s "
            f"(lyrics would fall outside the audio)"
        )
        return 0.0

    offset_ms = best_offset * window_ms
    logger.info(
        f"Offset estimation result: {offset_ms:+.0f}ms "
        f"(score={best_score:.4f}, search_range=±{max_offset_s:.0f}s, "
        f"coarse_step={coarse_step_ms:.0f}ms, window={window_ms:.0f}ms, "
        f"onset={onset_ms:.0f}ms)"
    )
    return offset_ms
