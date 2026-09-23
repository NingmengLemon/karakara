"""`karakara.gui.player` 的测试：注入一个假的 sounddevice，手工驱动回调。

播放器的窗口/循环/定位逻辑不该只在有声卡的机器上才被验证，所以这里把 ``sd`` 换掉，
直接调用输出回调检查填进去的样本。样本值就等于它的采样序号（0..999），
因此「听到的是哪一段」可以被精确断言。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from karakara.gui import player as player_module
from karakara.gui.player import AudioPlayer

RATE = 1000  # 1kHz：1ms = 1 个采样
TOTAL = 1000


class FakeStream:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.callback = kwargs["callback"]
        self.active = False
        self.start_calls = 0
        self.stop_calls = 0
        self.closed = False

    def start(self) -> None:
        self.active = True
        self.start_calls += 1

    def stop(self) -> None:
        self.active = False
        self.stop_calls += 1

    def close(self) -> None:
        self.closed = True


class FakeSd:
    def __init__(self, devices: list[dict[str, Any]] | None = None) -> None:
        self.devices = devices if devices is not None else [{"max_output_channels": 2}]
        self.streams: list[FakeStream] = []

    def query_devices(self) -> list[dict[str, Any]]:
        return self.devices

    def OutputStream(self, **kwargs: Any) -> FakeStream:
        stream = FakeStream(**kwargs)
        self.streams.append(stream)
        return stream


@pytest.fixture
def ramp() -> np.ndarray:
    """样本值 = 采样序号（0..999），便于精确断言播放到了哪一段。"""
    return np.arange(TOTAL, dtype=np.float32).reshape(1, TOTAL)


@pytest.fixture
def fake_sd(monkeypatch: pytest.MonkeyPatch) -> FakeSd:
    fake = FakeSd()
    monkeypatch.setattr(player_module, "sd", fake)
    monkeypatch.setattr(player_module, "_IMPORT_ERROR", "")
    return fake


def _pull(player: AudioPlayer, frames: int) -> np.ndarray:
    """手工跑一次回调，返回它填出来的样本（单声道）。"""
    stream = player._stream
    assert stream is not None
    out = np.zeros((frames, 1), dtype=np.float32)
    stream.callback(out, frames, None, None)
    return out[:, 0]


def test_unavailable_without_an_output_device(
    monkeypatch: pytest.MonkeyPatch, ramp: np.ndarray
) -> None:
    monkeypatch.setattr(player_module, "sd", FakeSd([{"max_output_channels": 0}]))
    player = AudioPlayer(ramp, RATE)

    assert player.available is False
    assert player.error is not None and "输出设备" in player.error
    assert player.play() is False


def test_unavailable_without_sounddevice(
    monkeypatch: pytest.MonkeyPatch, ramp: np.ndarray
) -> None:
    monkeypatch.setattr(player_module, "sd", None)
    monkeypatch.setattr(player_module, "_IMPORT_ERROR", "ModuleNotFoundError: x")

    player = AudioPlayer(ramp, RATE)

    assert player.available is False
    assert player.error is not None and "sounddevice" in player.error


def test_playback_window_is_respected(fake_sd: FakeSd, ramp: np.ndarray) -> None:
    player = AudioPlayer(ramp, RATE)
    assert player.play(start_ms=200, end_ms=400) is True

    first = _pull(player, 100)
    np.testing.assert_array_equal(first, np.arange(200, 300, dtype=np.float32))
    assert player.position_ms == 300

    second = _pull(player, 100)
    np.testing.assert_array_equal(second, np.arange(300, 400, dtype=np.float32))
    assert player.playing is False, "播到窗口末尾应自动停"


def test_audio_after_the_window_is_silence(fake_sd: FakeSd, ramp: np.ndarray) -> None:
    player = AudioPlayer(ramp, RATE)
    player.play(start_ms=0, end_ms=150)

    out = _pull(player, 200)

    np.testing.assert_array_equal(out[:150], np.arange(150, dtype=np.float32))
    np.testing.assert_array_equal(out[150:], np.zeros(50, dtype=np.float32))
    assert player.playing is False


def test_loop_wraps_back_to_the_window_start(fake_sd: FakeSd, ramp: np.ndarray) -> None:
    player = AudioPlayer(ramp, RATE)
    player.play(start_ms=200, end_ms=400, loop=True)

    _pull(player, 100)  # 200..299
    _pull(player, 100)  # 300..399，然后绕回
    third = _pull(player, 100)

    np.testing.assert_array_equal(third, np.arange(200, 300, dtype=np.float32))
    assert player.loop is True


def test_paused_player_outputs_silence(fake_sd: FakeSd, ramp: np.ndarray) -> None:
    player = AudioPlayer(ramp, RATE)
    player.play(start_ms=100, end_ms=200)
    _pull(player, 50)
    player.pause()

    out = _pull(player, 50)

    np.testing.assert_array_equal(out, np.zeros(50, dtype=np.float32))
    assert player.position_ms == 150, "暂停不应改变位置"


def test_seek_clamps_to_the_audio(fake_sd: FakeSd, ramp: np.ndarray) -> None:
    player = AudioPlayer(ramp, RATE)

    assert player.seek_ms(-500) == 0
    assert player.seek_ms(250) == 250
    assert player.seek_ms(999_999) == 1000


def test_playing_from_the_start_uses_the_whole_audio(
    fake_sd: FakeSd, ramp: np.ndarray
) -> None:
    player = AudioPlayer(ramp, RATE)
    player.play()
    _pull(player, 400)

    assert player.position_ms == 400
    assert player.playing is True


def test_stream_is_reused_across_plays(fake_sd: FakeSd, ramp: np.ndarray) -> None:
    player = AudioPlayer(ramp, RATE)
    player.play(start_ms=0, end_ms=100)
    player.stop()
    player.play(start_ms=0, end_ms=100)

    assert len(fake_sd.streams) == 1, "反复播放不该反复建流"


def test_close_stops_the_stream(fake_sd: FakeSd, ramp: np.ndarray) -> None:
    player = AudioPlayer(ramp, RATE)
    player.play()

    player.close()

    assert fake_sd.streams[0].stop_calls == 1
    assert fake_sd.streams[0].closed is True
