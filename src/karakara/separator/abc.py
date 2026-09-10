from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import ClassVar


class AbstractStemSeparator(ABC):
    """音轨分离器契约。

    接口刻意设计为**基于文件**而不是基于内存数组：

    * 分离一定跑在别的进程里（可能是另一个 Python 环境、甚至另一台机器），
      跨进程传整首歌的 float32 数组（一首 4 分钟立体声约 40MB）不划算；
    * 让 worker 自己解码源文件，可以省掉主进程里那份「解码完立刻交出去」的
      整首歌副本；
    * 分离结果本来就要落盘（Demucs 的命令行接口就是这么工作的），主进程只读
      回自己需要的那条音轨。
    """

    #: 主进程需要的人声轨名称。实现负责把自家命名映射到这个键。
    VOCAL_STEM_NAME: ClassVar[str] = "vocals"

    @abstractmethod
    def separate(
        self,
        audio_path: str | Path,
        dest_dir: str | Path,
        *,
        stems: Sequence[str] | None = None,
    ) -> dict[str, Path]:
        """分离 ``audio_path`` 并把结果写入 ``dest_dir``。

        Args:
            audio_path: 源音频文件（wav/mp3/flac/m4a 等，由实现负责解码）。
            dest_dir: 输出目录，调用方负责创建与清理。
            stems: 只保留这些音轨名；``None`` 表示实现自行决定。
                实现可以借此跳过写出用不到的音轨。

        Returns:
            音轨名 → 已落盘的音频文件路径。至少包含 :attr:`VOCAL_STEM_NAME`。

        Raises:
            StemSeparationError: 分离失败（worker 崩溃、模型缺失、音轨缺失等）。
        """
        raise NotImplementedError


class StemSeparationError(RuntimeError):
    """音轨分离失败。"""
