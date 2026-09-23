"""`karakara.gui.app` 的测试：纯函数 + 有显示时的整体冒烟。

界面本身没法在这里点，所以拆成两半：
* 不依赖显示器的格式化/刻度函数，直接断言；
* 建窗口、画一帧的整体冒烟，在没有 Tk 显示时自动跳过。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from karakara.gui.app import OffsetApp, format_ms, ruler_step_ms
from karakara.gui.session import OffsetSession

LYRICS = "[00:01.000]第一行\n[00:05.500]第二行\n"


def _tk_available() -> bool:
    try:
        import tkinter

        root = tkinter.Tk()
        root.destroy()
    except Exception:  # noqa: BLE001 - 没有显示/没有 Tk 都算不可用
        return False
    return True


TK_OK = _tk_available()


@pytest.mark.parametrize(
    ("ms", "expected"),
    [
        (0, "00:00.000"),
        (1500, "00:01.500"),
        (62_250, "01:02.250"),
        (-250, "-00:00.250"),
        (3_600_000, "60:00.000"),
    ],
)
def test_format_ms(ms: int, expected: str) -> None:
    assert format_ms(ms) == expected


@pytest.mark.parametrize(
    ("span_ms", "expected"),
    [
        (500, 100.0),  # 半秒视图：100ms 一格
        (5_000, 500.0),
        (60_000, 5_000.0),
        (245_000, 30_000.0),
        (10_000_000, 300_000.0),  # 极端跨度兜底
    ],
)
def test_ruler_step_keeps_tick_count_sane(span_ms: float, expected: float) -> None:
    step = ruler_step_ms(span_ms)
    assert step == expected
    assert span_ms / step <= 12 or step == 300_000.0


@pytest.mark.skipif(not TK_OK, reason="没有可用的 Tk 显示")
def test_app_builds_and_draws(tmp_path: Path) -> None:
    """整体冒烟：建窗口、画一帧、切一行、改偏移、关掉。"""
    import tkinter as tk

    lyrics = tmp_path / "song.lrc"
    lyrics.write_text(LYRICS, encoding="utf-8")
    audio = tmp_path / "song.wav"
    samples = np.zeros(8000 * 3, dtype=np.float32)
    samples[8000:16000] = 0.5
    sf.write(str(audio), samples, 8000, subtype="FLOAT")

    session = OffsetSession(lyrics, audio_path=audio)
    session.load_track()

    root = tk.Tk()
    root.withdraw()
    try:
        app = OffsetApp(root, session)
        root.update_idletasks()
        app.draw()
        app.select_line(1, seek=False)
        app.set_offset(250)
        assert session.offset_ms == 250
        assert app.current_line == 1
        # 画布上应当至少有包络与刻度画出来的元素
        assert len(app.canvas.find_all()) > 0
    finally:
        if "app" in locals() and app.player is not None:
            app.player.close()
        root.destroy()
