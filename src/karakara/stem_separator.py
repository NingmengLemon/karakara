"""
stem_separator.py

向后兼容的 facade — 委托给 ``separator.demucs.DemucseSeparator``。
新代码推荐直接使用 ``karakara.separator`` 包。
"""

from __future__ import annotations

from io import BytesIO
from logging import getLogger
from pathlib import Path

from karakara.separator.demucs import DemucsSeparator
from karakara.separator.demucs.impl import DEFAULT_DEVICE, DEFAULT_MODEL, DEFAULT_REPO
from karakara.typ import NpAudioData
from karakara.utils.io import load_audio

logger = getLogger(__name__)


def get_vocal_stem(
    src: str | Path | BytesIO,
    model: str = DEFAULT_MODEL,
    repo: Path = DEFAULT_REPO,
    device: str = DEFAULT_DEVICE,
) -> tuple[NpAudioData, int]:
    """从音频中提取人声轨道（向后兼容接口）。

    推荐新代码直接使用 ``DemucseSeparator`` 类。

    Args:
        src: 音频文件路径或 BytesIO
        model: Demucs 模型名称
        repo: 模型仓库路径
        device: 计算设备

    Returns:
        (vocal_stem, sample_rate) 元组
    """
    sep = DemucsSeparator(model=model, repo=repo, device=device)
    sample_rate = sep.samplerate
    audio_np = load_audio(src, sample_rate=sample_rate)
    stems = sep.separate(audio_np)
    return stems[sep.VOCAL_STEM_NAME], sample_rate
