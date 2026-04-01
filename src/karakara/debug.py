"""
debug.py

调试工具：将流水线各阶段的中间音频保存到指定目录，供事后检查。
"""

from __future__ import annotations

from logging import getLogger
from pathlib import Path

from karakara.typ import NpAudioData, NpAudioSamples
from karakara.utils.io import save_audio

logger = getLogger(__name__)


class AudioDumper:
    """流水线中间音频的调试导出器。

    用法::

        dumper = AudioDumper("tmp/debug_dump")
        dumper.dump("01_raw_vocal", vocal_np, sample_rate)
        dumper.dump("02_normalized", normalized_np, sample_rate)

    如果不需要 dump，传 ``dump_dir=None`` 即可，``AudioDumper(None)`` 的
    :meth:`dump` 调用是无操作的。
    """

    def __init__(self, dump_dir: str | Path | None) -> None:
        if dump_dir is not None:
            self._dir = Path(dump_dir)
            self._dir.mkdir(parents=True, exist_ok=True)
            self._enabled = True
            logger.info(f"AudioDumper enabled, dir={self._dir}")
        else:
            self._dir = Path(".")  # 占位，不会真正使用
            self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    def dump(
        self,
        name: str,
        audio: NpAudioData | NpAudioSamples,
        sample_rate: int,
        *,
        fmt: str = "wav",
    ) -> Path | None:
        """保存一段音频到 dump 目录。

        Args:
            name: 文件名（不含扩展名），如 ``"01_raw_vocal"``
            audio: 音频数据，shape (channels, samples) 或 (samples,)
            sample_rate: 采样率
            fmt: 输出格式后缀（默认 wav）

        Returns:
            保存的文件路径，若 dump 未启用则返回 None。
        """
        if not self._enabled:
            return None

        import numpy as np

        # 确保 2D
        if audio.ndim == 1:
            audio_2d = np.expand_dims(audio, axis=0)
        else:
            audio_2d = audio

        path = self._dir / f"{name}.{fmt}"
        save_audio(path, audio_2d, sample_rate)
        logger.debug(f"dumped audio: {path} (shape={audio_2d.shape}, sr={sample_rate})")
        return path
