"""播放器：整段音频常驻内存，支持定位与区间循环。

`sounddevice` 是**可选依赖**（``uv run --group gui``）：没装或没有输出设备时
:attr:`AudioPlayer.available` 为 ``False``，界面退化成「只能看波形、不能听」，
而不是崩掉。

回调里的窗口/循环/定位逻辑是这个模块最容易出错的部分，所以它被写成不依赖硬件的样子：
测试可以注入一个假的 ``sounddevice``，手工调用回调来检查样本与位置。
"""

from __future__ import annotations

import threading
from logging import getLogger
from typing import Any

import numpy as np
from numpy.typing import NDArray

logger = getLogger(__name__)

try:  # pragma: no cover - 依赖是否装好由运行环境决定
    import sounddevice as sd
except Exception as exc:  # noqa: BLE001 - 缺依赖不是错误，只是没有播放能力
    sd = None
    _IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
else:
    _IMPORT_ERROR = ""


class AudioPlayer:
    """把 ``(channels, samples)`` 的音频放进一个常驻输出流里播放。

    Attributes:
        available: 是否能播放（依赖装好且有输出设备）。
        error: 不能播放的原因，能给用户看的一句话。
    """

    def __init__(
        self,
        audio: NDArray[np.float32],
        sample_rate: int,
        *,
        block_ms: float = 50.0,
    ) -> None:
        data = audio if audio.ndim == 2 else audio[np.newaxis, :]
        # sounddevice 要 (frames, channels)
        self._data = np.ascontiguousarray(data.T, dtype=np.float32)
        self._rate = int(sample_rate)
        self._total = int(self._data.shape[0])
        self._channels = int(self._data.shape[1])

        self._lock = threading.Lock()
        self._index = 0
        self._window_start = 0
        self._window_end = self._total
        self._loop = False
        self._playing = False

        self._stream: Any = None
        self.error: str | None = None
        self._block_ms = block_ms

        if sd is None:
            self.error = f"sounddevice 不可用（{_IMPORT_ERROR}）"
        elif not self._has_output_device():
            self.error = "没有可用的音频输出设备"

    # ---------- 状态 ----------

    @property
    def available(self) -> bool:
        """能播放时为真。"""
        return self.error is None

    @property
    def playing(self) -> bool:
        """是否正在出声。"""
        return self._playing

    @property
    def duration_ms(self) -> float:
        """音频总时长（毫秒）。"""
        return self._total / self._rate * 1000.0

    @property
    def position_ms(self) -> int:
        """当前播放位置（毫秒）。"""
        with self._lock:
            return int(self._index / self._rate * 1000.0)

    @property
    def loop(self) -> bool:
        """是否在窗口内循环。"""
        return self._loop

    # ---------- 控制 ----------

    def seek_ms(self, position_ms: float) -> int:
        """把播放位置挪到 ``position_ms``（自动夹进音频范围），返回落点。"""
        with self._lock:
            self._index = self._clamp_sample(position_ms / 1000.0 * self._rate)
            return int(self._index / self._rate * 1000.0)

    def play(
        self,
        *,
        start_ms: float | None = None,
        end_ms: float | None = None,
        loop: bool = False,
    ) -> bool:
        """从 ``start_ms`` 播到 ``end_ms``（``None`` 表示音频末尾），可循环。

        Returns:
            是否真的开始播放（不可用时返回 ``False``）。
        """
        if not self.available:
            return False
        with self._lock:
            if start_ms is not None:
                self._window_start = self._clamp_sample(start_ms / 1000.0 * self._rate)
            if end_ms is not None:
                self._window_end = self._clamp_sample(end_ms / 1000.0 * self._rate)
            if self._window_end <= self._window_start:
                self._window_end = min(
                    self._total, self._window_start + self._rate // 10
                )
            if self._index < self._window_start or self._index >= self._window_end:
                self._index = self._window_start
            self._loop = loop
            self._playing = True
        self._ensure_stream()
        return True

    def pause(self) -> None:
        """暂停（位置保留）。"""
        with self._lock:
            self._playing = False

    def toggle(self) -> bool:
        """在播放与暂停之间切换，返回切换后是否在播放。"""
        if self._playing:
            self.pause()
        else:
            self.play()
        return self._playing

    def stop(self) -> None:
        """停止并回到窗口起点。"""
        with self._lock:
            self._playing = False
            self._index = self._window_start

    def close(self) -> None:
        """关掉输出流（退出前调用）。"""
        with self._lock:
            self._playing = False
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception as exc:  # noqa: BLE001 - 关流失败不该阻断退出
                logger.debug(f"closing audio stream failed: {exc}")

    # ---------- 内部 ----------

    def _has_output_device(self) -> bool:
        module = sd
        if module is None:
            return False
        try:
            devices = module.query_devices()
        except Exception as exc:  # noqa: BLE001 - PortAudio 后端缺失等
            logger.debug(f"query_devices failed: {exc}")
            return False
        return any(int(device.get("max_output_channels", 0)) > 0 for device in devices)

    def _ensure_stream(self) -> None:
        if self._stream is not None:
            if not self._stream.active:
                self._stream.start()
            return
        # 取到局部变量再判空：模块级 `sd` 是可选的，类型检查器不会跨函数收窄它。
        module = sd
        if module is None:
            raise RuntimeError(self.error or "sounddevice 不可用")
        blocksize = max(64, int(self._rate * self._block_ms / 1000.0))
        self._stream = module.OutputStream(
            samplerate=self._rate,
            channels=self._channels,
            dtype="float32",
            blocksize=blocksize,
            callback=self._callback,
        )
        self._stream.start()

    def _clamp_sample(self, sample: float) -> int:
        return int(max(0, min(self._total, sample)))

    def _callback(self, outdata: Any, frames: int, _time: Any, status: Any) -> None:
        """输出回调：把窗口内的样本填进 ``outdata``，窗口外填静音。"""
        if status:  # pragma: no cover - 取决于硬件
            logger.debug(f"audio callback status: {status}")
        with self._lock:
            if not self._playing:
                outdata.fill(0.0)
                return

            index = self._index
            if index >= self._window_end:
                if self._loop:
                    index = self._window_start
                else:
                    self._playing = False
                    outdata.fill(0.0)
                    return

            remaining = self._window_end - index
            take = min(frames, remaining)
            outdata[:take] = self._data[index : index + take]
            if take < frames:
                outdata[take:].fill(0.0)

            index += take
            if index >= self._window_end:
                if self._loop:
                    index = self._window_start
                else:
                    self._playing = False
            self._index = index
