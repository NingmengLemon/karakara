"""``karakara.gui``：与 CLI 并列的图形工具（人工对照波形与歌词调偏移）。

子模块分工：

* :mod:`karakara.gui.peaks` —— 波形包络与「时间 ↔ 像素」聚合（纯 numpy，可测）
* :mod:`karakara.gui.lrctext` —— 文本级时间戳平移（原地写回的地基，可测）
* :mod:`karakara.gui.session` —— 会话状态、音频载入、安全写回（不 import GUI 库）
* :mod:`karakara.gui.player` —— 可选的声音播放（``sounddevice``，缺了也能用）
* :mod:`karakara.gui.app` —— tkinter 界面

依赖边界：``tkinter`` 是标准库，``sounddevice`` 在 ``gui`` 依赖组里
（``uv run --group gui python scripts/offset_gui.py ...``），主环境的依赖闭包不受影响。
"""

from __future__ import annotations

from karakara.gui.session import AudioTrack, LineView, OffsetSession, SaveReport

__all__ = ["AudioTrack", "LineView", "OffsetSession", "SaveReport"]
