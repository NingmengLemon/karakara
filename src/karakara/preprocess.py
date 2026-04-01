"""
audio.py

音频预处理模块：响度归一化、颤音抑制、动态范围压缩。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from karakara.typ import NpAudioData, NpAudioSamples


# --------------------------------------------------------------------------
# 预处理配置
# --------------------------------------------------------------------------

@dataclass
class AudioPreprocessConfig:
    """音频预处理配置。"""

    # ---------- 响度归一化 ----------
    normalize: bool = True
    target_dbfs: float = -20.0  # 目标响度 (dBFS)，-20 是播客/流媒体常用值

    # ---------- 颤音抑制 ----------
    suppress_vibrato: bool = True
    vibrato_threshold_hz: float = 5.0  # 频率 > 此值视为颤音成分
    vibrato_smooth_window_ms: float = 50.0  # 抑制颤音的平滑窗口 (ms)

    # ---------- 动态范围压缩 ----------
    compress: bool = False
    comp_threshold_dbfs: float = -18.0  # 压缩阈值 (dBFS)
    comp_ratio: float = 4.0  # 压缩比 (ratio)
    comp_attack_ms: float = 10.0  # 启动时间 (ms)
    comp_release_ms: float = 100.0  # 释放时间 (ms)


# --------------------------------------------------------------------------
# 辅助函数
# --------------------------------------------------------------------------

def _db_to_linear(db: float) -> float:
    return float(10 ** (db / 20))


def _linear_to_db(x: float | NDArray[np.float32]) -> float | NDArray:
    eps = 1e-10
    result: NDArray = 20 * np.log10(np.maximum(np.abs(x), eps))  # type: ignore[assignment]
    return result  # type: ignore[return-value]


def _envelope(
    samples: NDArray[np.float32],
    sample_rate: int,
    window_ms: float,
) -> NDArray[np.float32]:
    """计算信号的 RMS 包络。

    Args:
        samples: 输入样本 (shape: [samples], 即 1D)
        sample_rate: 采样率 (Hz)
        window_ms: 窗口长度 (ms)

    Returns:
        与输入等长的 RMS 包络数组。
    """
    window_samples = max(1, int(sample_rate * window_ms / 1000))
    # 简单的移动 RMS
    padded = np.pad(samples, window_samples, mode="edge")
    square = padded**2
    window = np.ones(window_samples) / window_samples
    rms = np.sqrt(np.convolve(square, window, mode="valid"))
    # 截断/填充到原始长度
    if len(rms) < len(samples):
        rms = np.pad(rms, (0, len(samples) - len(rms)))
    return rms[: len(samples)].astype(np.float32)


# --------------------------------------------------------------------------
# 各预处理步骤
# --------------------------------------------------------------------------

def normalize_loudness(
    audio: NpAudioData | NpAudioSamples,
    target_dbfs: float = -20.0,
) -> NpAudioData | NpAudioSamples:
    """RMS 响度归一化。

    将音频的 RMS 响度调整到目标 dBFS 水平。
    对立体声信号同时作用于所有通道，对单声道同理。

    Args:
        audio: 输入音频，shape (channels, samples) 或 (samples,)
        target_dbfs: 目标响度 (dBFS)，值越大声音越响

    Returns:
        归一化后的音频，形状不变。
    """
    if audio.size == 0:
        return audio

    # 计算 RMS（多通道平均）
    rms_linear = np.sqrt(np.mean(audio**2))
    if rms_linear < 1e-10:
        return audio  # 信号太弱，跳过

    current_dbfs = _linear_to_db(rms_linear)
    gain_db = target_dbfs - float(current_dbfs)
    gain_linear = _db_to_linear(gain_db)

    return audio * gain_linear


def suppress_vibrato(
    audio: NpAudioData | NpAudioSamples,
    sample_rate: int,
    threshold_hz: float = 5.0,
    smooth_window_ms: float = 50.0,
) -> NpAudioData | NpAudioSamples:
    """颤音抑制。

    通过对信号的幅度包络进行低频波动检测并平滑，
    减小由颤音产生的幅度振荡（通常在 5–8 Hz 范围）。

    这是一种轻量级的包络级处理，不做完整基频估计。

    Args:
        audio: 输入音频，shape (channels, samples) 或 (samples,)
        sample_rate: 采样率 (Hz)
        threshold_hz: 颤音频率下限阈值 (Hz)
        smooth_window_ms: 抑制颤音的平滑窗口 (ms)

    Returns:
        颤音抑制后的音频，形状不变。
    """
    if audio.size == 0:
        return audio

    # 颤音抑制窗口：需要足够大以平滑掉高频波动
    # 颤音周期约 125–200ms，故窗口取 2–3 倍周期
    vib_period_ms = 1000.0 / threshold_hz if threshold_hz > 0 else 200.0
    actual_window_ms = max(smooth_window_ms, vib_period_ms * 2)

    if audio.ndim == 2:
        result = np.zeros_like(audio)
        for ch in range(audio.shape[0]):
            env = _envelope(audio[ch], sample_rate, actual_window_ms)
            # 低频包络（原始 / 包络 得到归一化残差，再乘以平滑包络）
            normalized = audio[ch] / (env + 1e-10)
            smoothed_env = _envelope(env, sample_rate, actual_window_ms)
            result[ch] = normalized * smoothed_env
        return result
    else:
        env = _envelope(audio, sample_rate, actual_window_ms)
        normalized = audio / (env + 1e-10)
        smoothed_env = _envelope(env, sample_rate, actual_window_ms)
        return (normalized * smoothed_env).astype(np.float32)


def compress_dynamic_range(
    audio: NpAudioData | NpAudioSamples,
    sample_rate: int,
    threshold_dbfs: float = -18.0,
    ratio: float = 4.0,
    attack_ms: float = 10.0,
    release_ms: float = 100.0,
) -> NpAudioData | NpAudioSamples:
    """动态范围压缩（Dynamic Range Compression, DRC）。

    基于阈值的软拐点压缩，模拟标准音频压缩器行为：
    - attack: 超过阈值后快速增益减小
    - release: 低于阈值后缓慢恢复

    Args:
        audio: 输入音频，shape (channels, samples) 或 (samples,)
        sample_rate: 采样率 (Hz)
        threshold_dbfs: 压缩阈值 (dBFS)
        ratio: 压缩比 (>1 表示压缩)
        attack_ms: 启动时间 (ms)
        release_ms: 释放时间 (ms)

    Returns:
        压缩后的音频，形状不变。
    """
    if audio.size == 0 or ratio <= 1.0:
        return audio

    threshold_linear = _db_to_linear(threshold_dbfs)

    # 计算信号包络（用于检测阈值穿越）
    if audio.ndim == 2:
        env = np.max(np.abs(audio), axis=0)  # shape: (samples,)
    else:
        env = np.abs(audio)

    # 计算增益包络（attack / release 平滑）
    # attack: 超过阈值 → 增益下降
    # release: 低于阈值 → 增益恢复
    attack_samples = max(1, int(sample_rate * attack_ms / 1000))
    release_samples = max(1, int(sample_rate * release_ms / 1000))

    gain = np.ones(len(env), dtype=np.float32)

    # 计算目标增益（理想值）
    over_threshold = env > threshold_linear
    target_gain = np.where(
        over_threshold,
        threshold_linear + (env - threshold_linear) / ratio,
        env,
    ) / (env + 1e-10)

    # 用 attack/release 系数平滑增益
    a_attack = 1.0 - np.exp(-1.0 / attack_samples)
    a_release = 1.0 - np.exp(-1.0 / release_samples)

    for i in range(1, len(gain)):
        coeff = a_attack if env[i] > env[i - 1] else a_release
        gain[i] = gain[i - 1] + coeff * (target_gain[i] - gain[i - 1])

    gain = gain.astype(np.float32)

    # 应用增益
    if audio.ndim == 2:
        return audio * gain[np.newaxis, :]
    return audio * gain


# --------------------------------------------------------------------------
# 统一预处理入口
# --------------------------------------------------------------------------

def preprocess(
    audio: NpAudioData | NpAudioSamples,
    sample_rate: int,
    config: AudioPreprocessConfig | None = None,
) -> NpAudioData | NpAudioSamples:
    """对音频应用预处理（归一化 → 颤音抑制 → DRC）。

    Args:
        audio: 输入音频
        sample_rate: 采样率 (Hz)
        config: 预处理配置，None 时使用默认值

    Returns:
        预处理后的音频，形状不变。
    """
    if config is None:
        config = AudioPreprocessConfig()

    result = audio

    if config.normalize:
        result = normalize_loudness(result, config.target_dbfs)

    if config.suppress_vibrato:
        result = suppress_vibrato(
            result,
            sample_rate,
            threshold_hz=config.vibrato_threshold_hz,
            smooth_window_ms=config.vibrato_smooth_window_ms,
        )

    if config.compress:
        result = compress_dynamic_range(
            result,
            sample_rate,
            threshold_dbfs=config.comp_threshold_dbfs,
            ratio=config.comp_ratio,
            attack_ms=config.comp_attack_ms,
            release_ms=config.comp_release_ms,
        )

    return result
