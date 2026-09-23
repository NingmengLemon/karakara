"""人工对照波形与歌词、调偏移、写回 LRC 的桌面界面。

设计要点：

* **只做一件事**：让「歌词标记」和「波形上人声出现的位置」对上，然后把偏移写回文件。
  对齐本身、分离、生成逐字产物仍然由 CLI 与两个 worker 负责，这里不重复它们。
* 逻辑都在 :mod:`karakara.gui.session` / :mod:`karakara.gui.peaks` /
  :mod:`karakara.gui.lrctext` 里，本模块只负责画和收事件，因此可以在没有显示器的
  环境里把逻辑测完。
* 播放是**可选能力**：没有 ``sounddevice`` 或没有输出设备时只禁用播放按钮，
  波形与偏移照常可用。
"""

from __future__ import annotations

import tkinter as tk
from functools import partial
from logging import getLogger
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from karakara.gui.peaks import peaks_for_range
from karakara.gui.player import AudioPlayer
from karakara.gui.session import AudioTrack, OffsetSession, SaveReport

logger = getLogger(__name__)

#: 配色：深底、蓝波形、灰标记、红当前行、黄播放头。
BG = "#111318"
WAVE = "#4aa3ff"
WAVE_MIX = "#2c3a4a"
WAVE_CENTER = "#2a2f3a"
LINE = "#8b93a1"
LINE_CURRENT = "#ff6b6b"
PLAYHEAD = "#ffd166"
RULER = "#5a6270"
TEXT = "#e8e8e8"
CURRENT_WINDOW = "#1d2733"

#: 偏移调节的上下限（毫秒）。超出这个范围的输入基本说明是另一份剪辑。
OFFSET_LIMIT_MS = 30_000

#: 视图最小跨度（毫秒）：再窄就看不出波形了。
MIN_VIEW_SPAN_MS = 200.0

#: 判定「点击」而不是「拖拽」的像素阈值。
CLICK_SLOP_PX = 3

#: 状态刷新间隔（毫秒）。
TICK_MS = 50


def format_ms(ms: float) -> str:
    """把毫秒格式化成 ``mm:ss.mmm``（负数带负号）。"""
    negative = ms < 0
    value = abs(round(ms))
    minutes, remainder = divmod(value, 60_000)
    seconds, milliseconds = divmod(remainder, 1_000)
    sign = "-" if negative else ""
    return f"{sign}{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def ruler_step_ms(span_ms: float) -> float:
    """按视图跨度挑一个刻度间隔，让画面上大约 6 到 12 条刻度线。"""
    for candidate in (100, 250, 500, 1_000, 2_000, 5_000, 10_000, 30_000, 60_000):
        if span_ms / candidate <= 12:
            return float(candidate)
    return 300_000.0


class OffsetApp:
    """波形 + 歌词的偏移调节器。"""

    def __init__(
        self,
        root: tk.Tk,
        session: OffsetSession,
        *,
        player: AudioPlayer | None = None,
    ) -> None:
        self.root = root
        self.session = session
        self.player = player
        if self.player is None and session.track is not None:
            self.player = AudioPlayer(session.track.data, session.track.sample_rate)

        duration = session.track.duration_ms if session.track is not None else 60_000.0
        self.view_start_ms = 0.0
        self.view_end_ms = max(duration, MIN_VIEW_SPAN_MS)
        self.current_line = 0
        self._drag_origin: tuple[int, float, float] | None = None
        self._drag_moved = False

        self._offset_var = tk.IntVar(value=session.offset_ms)
        self._loop_var = tk.BooleanVar(value=False)
        self._status_var = tk.StringVar(value="")
        self._position_var = tk.StringVar(value="00:00.000 / " + format_ms(duration))
        self._file_var = tk.StringVar(value="")

        self._build()
        self._bind_keys()
        self.refresh_all()
        self.root.after(TICK_MS, self._tick)

    # ------------------------------------------------------------------
    # 构建
    # ------------------------------------------------------------------

    def _build(self) -> None:
        self.root.title(f"karakara 偏移对照 — {self.session.lyrics_path.name}")
        self.root.configure(bg=BG)
        self.root.geometry("1100x760")

        self._build_toolbar()
        self._build_canvas()
        self._build_transport()
        self._build_offset_bar()
        self._build_lyrics()
        self._build_status()

    def _build_toolbar(self) -> None:
        bar = ttk.Frame(self.root, padding=(8, 6))
        bar.pack(fill="x")
        ttk.Label(bar, textvariable=self._file_var).pack(side="left")
        ttk.Button(bar, text="重新载入", command=self.reload).pack(side="right")
        ttk.Button(bar, text="打开音频…", command=self.open_audio).pack(
            side="right", padx=4
        )
        ttk.Button(bar, text="打开歌词…", command=self.open_lyrics).pack(
            side="right", padx=4
        )

    def _build_canvas(self) -> None:
        self.canvas = tk.Canvas(
            self.root, height=240, bg=BG, highlightthickness=0, cursor="crosshair"
        )
        self.canvas.pack(fill="both", expand=True, padx=8, pady=(0, 4))
        self.canvas.bind("<Configure>", lambda _event: self.draw())
        self.canvas.bind("<Button-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<MouseWheel>", self._on_wheel)

    def _build_transport(self) -> None:
        bar = ttk.Frame(self.root, padding=(8, 0))
        bar.pack(fill="x")
        self._play_button = ttk.Button(bar, text="播放", command=self.toggle_play)
        self._play_button.pack(side="left")
        ttk.Button(bar, text="停止", command=self.stop).pack(side="left", padx=4)
        ttk.Button(bar, text="上一行", command=lambda: self.step_line(-1)).pack(
            side="left", padx=(12, 2)
        )
        ttk.Button(bar, text="下一行", command=lambda: self.step_line(1)).pack(
            side="left", padx=2
        )
        ttk.Checkbutton(bar, text="循环本行", variable=self._loop_var).pack(
            side="left", padx=12
        )
        ttk.Label(bar, textvariable=self._position_var).pack(side="right")

    def _build_offset_bar(self) -> None:
        bar = ttk.Frame(self.root, padding=(8, 6))
        bar.pack(fill="x")
        ttk.Label(bar, text="偏移 (ms)").pack(side="left")
        for label, delta in (("-100", -100), ("-10", -10), ("+10", 10), ("+100", 100)):
            ttk.Button(
                bar, text=label, width=5, command=partial(self.nudge, delta)
            ).pack(side="left", padx=2)
        self._offset_spin = ttk.Spinbox(
            bar,
            from_=-OFFSET_LIMIT_MS,
            to=OFFSET_LIMIT_MS,
            increment=10,
            width=8,
            textvariable=self._offset_var,
            command=self._on_spin,
        )
        self._offset_spin.pack(side="left", padx=6)
        self._offset_spin.bind("<Return>", lambda _event: self._on_spin())
        self._offset_spin.bind("<FocusOut>", lambda _event: self._on_spin())
        ttk.Button(bar, text="归零", command=lambda: self.set_offset(0)).pack(
            side="left", padx=2
        )
        ttk.Button(bar, text="试听本行", command=self.audition_line).pack(
            side="left", padx=(12, 2)
        )
        ttk.Button(bar, text="写回歌词文件", command=self.save_in_place).pack(
            side="left", padx=2
        )
        ttk.Button(bar, text="另存为…", command=self.save_as).pack(side="left", padx=2)

    def _build_lyrics(self) -> None:
        frame = ttk.Frame(self.root, padding=(8, 0))
        frame.pack(fill="both", expand=True, pady=(4, 0))
        scroll = ttk.Scrollbar(frame, orient="vertical")
        self.lyrics_list = tk.Listbox(
            frame,
            activestyle="none",
            bg=BG,
            fg=TEXT,
            selectbackground=CURRENT_WINDOW,
            selectforeground=LINE_CURRENT,
            highlightthickness=0,
            font=("Consolas", 10),
            yscrollcommand=scroll.set,
        )
        scroll.config(command=self.lyrics_list.yview)
        scroll.pack(side="right", fill="y")
        self.lyrics_list.pack(side="left", fill="both", expand=True)
        self.lyrics_list.bind("<<ListboxSelect>>", self._on_line_selected)

    def _build_status(self) -> None:
        bar = ttk.Frame(self.root, padding=(8, 4))
        bar.pack(fill="x")
        ttk.Label(bar, textvariable=self._status_var).pack(side="left")

    def _bind_keys(self) -> None:
        self.root.bind("<space>", lambda _event: self.toggle_play())
        self.root.bind("<comma>", lambda _event: self.nudge(-10))
        self.root.bind("<period>", lambda _event: self.nudge(10))
        self.root.bind("<bracketleft>", lambda _event: self.step_line(-1))
        self.root.bind("<bracketright>", lambda _event: self.step_line(1))
        self.root.bind("<Left>", lambda _event: self.seek_relative(-1000))
        self.root.bind("<Right>", lambda _event: self.seek_relative(1000))
        self.root.bind(
            "<Key-l>", lambda _event: self._loop_var.set(not self._loop_var.get())
        )
        self.root.bind("<Key-s>", lambda _event: self.save_in_place())
        self.root.bind("<Control-s>", lambda _event: self.save_as())

    # ------------------------------------------------------------------
    # 坐标换算
    # ------------------------------------------------------------------

    def _width(self) -> int:
        return max(1, int(self.canvas.winfo_width()))

    def _time_to_x(self, ms: float) -> float:
        span = max(1e-6, self.view_end_ms - self.view_start_ms)
        return (ms - self.view_start_ms) / span * self._width()

    def _x_to_time(self, x: float) -> float:
        span = self.view_end_ms - self.view_start_ms
        return self.view_start_ms + x / self._width() * span

    # ------------------------------------------------------------------
    # 绘制
    # ------------------------------------------------------------------

    def draw(self) -> None:
        """重画整块画布（包络、行标记、播放头、时间刻度）。"""
        canvas = self.canvas
        canvas.delete("all")
        width = self._width()
        height = max(1, int(canvas.winfo_height()))
        center = height / 2

        canvas.create_line(0, center, width, center, fill=WAVE_CENTER)
        self._draw_envelope(width, height, center)
        self._draw_current_window(height)
        self._draw_line_markers(height)
        self._draw_ruler(height)
        self._draw_playhead(height)

    def _draw_envelope(self, width: int, height: int, center: float) -> None:
        track = self.session.track
        if track is None:
            self.canvas.create_text(
                width / 2,
                center,
                text="没有音频：用「打开音频…」选一个文件",
                fill=RULER,
            )
            return

        # 有分离人声时：混音画暗做背景，人声画亮。调偏移靠的是「人声什么时候进来」，
        # 只看混音的话波形几乎铺满全曲，看不出起唱点。
        vocals = self.session.vocals
        if vocals is not None:
            self._draw_one_envelope(track, width, height, center, color=WAVE_MIX)
            self._draw_one_envelope(vocals, width, height, center, color=WAVE)
        else:
            self._draw_one_envelope(track, width, height, center, color=WAVE)

    def _draw_one_envelope(
        self,
        track: AudioTrack,
        width: int,
        height: int,
        center: float,
        *,
        color: str,
    ) -> None:
        envelope = track.envelope
        minima, maxima, _ = peaks_for_range(
            envelope, self.view_start_ms, self.view_end_ms, columns=width
        )
        scale = (height / 2 - 10) / envelope.peak_amplitude
        for column in range(len(maxima)):
            self.canvas.create_line(
                column,
                center - float(maxima[column]) * scale,
                column,
                center - float(minima[column]) * scale,
                fill=color,
            )

    def _draw_current_window(self, height: int) -> None:
        start, end = self.session.window_ms(self.current_line)
        left = self._time_to_x(start)
        right = self._time_to_x(end)
        if right < 0 or left > self._width():
            return
        self.canvas.create_rectangle(
            left, 0, right, height, fill=CURRENT_WINDOW, outline=""
        )

    def _draw_line_markers(self, height: int) -> None:
        width = self._width()
        for view in self.session.lines:
            shifted = view.start_ms + self.session.offset_ms
            x = self._time_to_x(shifted)
            if x < -1 or x > width + 1:
                continue
            current = view.index == self.current_line
            self.canvas.create_line(
                x,
                0,
                x,
                height,
                fill=LINE_CURRENT if current else LINE,
                dash=() if current else (2, 4),
            )

    def _draw_ruler(self, height: int) -> None:
        step = ruler_step_ms(self.view_end_ms - self.view_start_ms)
        start = int(self.view_start_ms // step) * step
        time = start
        while time <= self.view_end_ms:
            x = self._time_to_x(time)
            self.canvas.create_line(x, height - 6, x, height, fill=RULER)
            self.canvas.create_text(
                x + 3, height - 16, text=format_ms(time)[:5], fill=RULER, anchor="w"
            )
            time += step

    def _draw_playhead(self, height: int) -> None:
        x = self._time_to_x(self.position_ms())
        self.canvas.create_line(x, 0, x, height, fill=PLAYHEAD)

    # ------------------------------------------------------------------
    # 播放
    # ------------------------------------------------------------------

    def position_ms(self) -> int:
        """当前播放位置；没有播放能力时用视图起点，画布仍能画出播放头。"""
        if self.player is not None and self.player.available:
            return self.player.position_ms
        return int(self.view_start_ms)

    def toggle_play(self) -> None:
        if self.player is None or not self.player.available:
            self._warn_no_player()
            return
        if self.player.playing:
            self.player.pause()
            self._play_button.config(text="播放")
            return
        self._start_playback(from_current=True)

    def _start_playback(self, *, from_current: bool) -> None:
        assert self.player is not None
        if self._loop_var.get():
            start, end = self.session.window_ms(self.current_line)
            self.player.seek_ms(max(0, start))
            self.player.play(start_ms=max(0, start), end_ms=max(0, end), loop=True)
        else:
            if not from_current:
                self.player.seek_ms(
                    max(0, self.session.shifted_start_ms(self.current_line))
                )
            self.player.play(loop=False)
        self._play_button.config(text="暂停")

    def stop(self) -> None:
        if self.player is not None:
            self.player.stop()
        self._play_button.config(text="播放")

    def audition_line(self) -> None:
        """循环播放当前行（按当前偏移）。"""
        if self.player is None or not self.player.available:
            self._warn_no_player()
            return
        start, end = self.session.window_ms(self.current_line)
        start = max(0, start)
        end = max(start + 100, end)
        self._loop_var.set(True)
        self.player.play(start_ms=start, end_ms=end, loop=True)
        self._play_button.config(text="暂停")

    def seek_relative(self, delta_ms: int) -> None:
        if self.player is not None and self.player.available:
            self.player.seek_ms(self.player.position_ms + delta_ms)
        else:
            self._pan(delta_ms)
        self.draw()

    def _warn_no_player(self) -> None:
        reason = self.player.error if self.player is not None else "没有音频"
        messagebox.showwarning("无法播放", reason or "没有可用的播放器")

    # ------------------------------------------------------------------
    # 行与偏移
    # ------------------------------------------------------------------

    def step_line(self, delta: int) -> None:
        count = len(self.session.lines)
        if count == 0:
            return
        self.select_line(max(0, min(count - 1, self.current_line + delta)), seek=True)

    def _on_line_selected(self, _event: tk.Event) -> None:
        """点歌词列表：切到那一行并把播放位置挪过去。"""
        # tkinter 的桩里 curselection 没有注解，而本项目开着 disallow_untyped_calls。
        selection = self.lyrics_list.curselection()
        if not selection:
            return
        index = int(selection[0])
        if index != self.current_line:
            self.select_line(index, seek=True)

    def select_line(self, index: int, *, seek: bool = True) -> None:
        self.current_line = index
        if seek and self.player is not None and self.player.available:
            self.player.seek_ms(max(0, self.session.shifted_start_ms(index)))
        self._sync_list_selection()
        self._update_status()
        self.draw()

    def set_offset(self, offset_ms: int) -> None:
        """设置偏移并立即重画（波形不动，标记与当前行窗口跟着动）。"""
        clamped = max(-OFFSET_LIMIT_MS, min(OFFSET_LIMIT_MS, int(offset_ms)))
        self.session.set_offset(clamped)
        if int(self._offset_var.get()) != clamped:
            self._offset_var.set(clamped)
        self._update_status()
        self.draw()

    def nudge(self, delta_ms: int) -> None:
        self.set_offset(self.session.offset_ms + delta_ms)

    def _on_spin(self) -> None:
        try:
            value = int(self._offset_var.get())
        except (tk.TclError, ValueError):
            self._offset_var.set(self.session.offset_ms)
            return
        self.set_offset(value)

    # ------------------------------------------------------------------
    # 画布交互
    # ------------------------------------------------------------------

    def _on_press(self, event: tk.Event) -> None:
        self._drag_origin = (event.x, self.view_start_ms, self.view_end_ms)
        self._drag_moved = False

    def _on_drag(self, event: tk.Event) -> None:
        if self._drag_origin is None:
            return
        origin_x, start, end = self._drag_origin
        if abs(event.x - origin_x) > CLICK_SLOP_PX:
            self._drag_moved = True
        span = end - start
        shift = (origin_x - event.x) / self._width() * span
        self._set_view(start + shift, end + shift)

    def _on_release(self, event: tk.Event) -> None:
        if self._drag_origin is not None and not self._drag_moved:
            target = max(0.0, self._x_to_time(event.x))
            if self.player is not None and self.player.available:
                self.player.seek_ms(target)
            self._sync_current_line(target)
        self._drag_origin = None
        self.draw()

    def _on_wheel(self, event: tk.Event) -> None:
        factor = 0.8 if event.delta > 0 else 1.25
        anchor = self._x_to_time(event.x)
        self._set_view(
            anchor - (anchor - self.view_start_ms) * factor,
            anchor + (self.view_end_ms - anchor) * factor,
        )

    def _set_view(self, start: float, end: float) -> None:
        duration = (
            self.session.track.duration_ms if self.session.track is not None else end
        )
        span = max(MIN_VIEW_SPAN_MS, min(end - start, max(duration, MIN_VIEW_SPAN_MS)))
        start = max(0.0, min(start, max(0.0, duration - span)))
        self.view_start_ms = start
        self.view_end_ms = start + span
        self.draw()

    def _pan(self, delta_ms: float) -> None:
        self._set_view(self.view_start_ms + delta_ms, self.view_end_ms + delta_ms)

    def _sync_current_line(self, position_ms: float) -> None:
        """把当前行切到包含 ``position_ms`` 的那一行。"""
        lines = self.session.lines
        for view in lines:
            start, end = self.session.window_ms(view.index)
            if start <= position_ms < end:
                self.select_line(view.index, seek=False)
                return

    # ------------------------------------------------------------------
    # 文件
    # ------------------------------------------------------------------

    def open_lyrics(self) -> None:
        chosen = filedialog.askopenfilename(
            title="选择歌词文件", filetypes=[("LRC", "*.lrc"), ("全部文件", "*.*")]
        )
        if chosen:
            self.load(lyrics_path=Path(chosen))

    def open_audio(self) -> None:
        chosen = filedialog.askopenfilename(
            title="选择音频文件",
            filetypes=[("音频", "*.wav *.mp3 *.flac *.m4a"), ("全部文件", "*.*")],
        )
        if chosen:
            self.load(audio_path=Path(chosen))

    def reload(self) -> None:
        """从磁盘重新读一遍（文件被别的程序改过时用）。"""
        self.load(
            lyrics_path=self.session.lyrics_path, audio_path=self.session.audio_path
        )

    def load(
        self, *, lyrics_path: Path | None = None, audio_path: Path | None = None
    ) -> None:
        """换文件并重建播放器。任一步失败都只弹错误、不改变当前状态。"""
        try:
            session = OffsetSession(
                lyrics_path or self.session.lyrics_path,
                audio_path=audio_path or self.session.audio_path,
                vocals_path=self.session.vocals_path,
            )
            if session.audio_path is not None:
                session.load_track()
            session.load_vocals()
        except Exception as exc:  # noqa: BLE001 - 载入失败要弹给用户看
            messagebox.showerror("载入失败", f"{type(exc).__name__}: {exc}")
            return

        if self.player is not None:
            self.player.close()
        self.session = session
        self.player = None
        if session.track is not None:
            self.player = AudioPlayer(session.track.data, session.track.sample_rate)
        self.view_start_ms = 0.0
        self.view_end_ms = max(
            session.track.duration_ms if session.track else 0, MIN_VIEW_SPAN_MS
        )
        self.current_line = 0
        self._offset_var.set(0)
        self.refresh_all()

    def save_in_place(self) -> None:
        """把当前偏移写回打开的那个歌词文件。"""
        if self.session.offset_ms == 0:
            messagebox.showinfo("无需写回", "当前偏移是 0，文件不会有任何变化。")
            return
        target = self.session.lyrics_path
        answer = messagebox.askyesno(
            "写回歌词文件",
            f"把 {len(self.session.lines)} 行的时间戳整体平移 "
            f"{self.session.offset_ms:+d} ms 并写回：\n{target}\n\n"
            f"写入前的内容会备份成 {target.name}.bak。",
        )
        if not answer:
            return
        try:
            report = self.session.save()
        except Exception as exc:  # noqa: BLE001 - 写盘失败必须让用户看到
            messagebox.showerror("写回失败", f"{type(exc).__name__}: {exc}")
            return
        self._after_save(report)

    def save_as(self) -> None:
        """把平移后的歌词另存为别的文件。"""
        if self.session.offset_ms == 0:
            messagebox.showinfo("无需另存", "当前偏移是 0，内容与原文相同。")
            return
        chosen = filedialog.asksaveasfilename(
            title="另存为",
            defaultextension=".lrc",
            initialfile=self.session.lyrics_path.name,
            filetypes=[("LRC", "*.lrc"), ("全部文件", "*.*")],
        )
        if not chosen:
            return
        try:
            report = self.session.save(target=Path(chosen))
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("另存失败", f"{type(exc).__name__}: {exc}")
            return
        self._after_save(report)

    def _after_save(self, report: SaveReport) -> None:
        self._offset_var.set(self.session.offset_ms)
        self.refresh_all()
        detail = f"已写回 {report.tags} 个时间戳（偏移 {report.delta_ms:+d} ms）"
        if report.clamped:
            detail += f"\n其中 {report.clamped} 个被夹到 0（原来的时间戳不够减）"
        if report.backup is not None:
            detail += f"\n备份：{report.backup}"
        messagebox.showinfo("完成", detail)

    # ------------------------------------------------------------------
    # 刷新
    # ------------------------------------------------------------------

    def refresh_all(self) -> None:
        """文件变了或载入完成后整体刷新。"""
        self._refresh_file_label()
        self._refresh_list()
        self._update_status()
        self.draw()

    def _refresh_file_label(self) -> None:
        audio = self.session.audio_path
        text = f"歌词：{self.session.lyrics_path.name}"
        if audio is not None and self.session.track is not None:
            track = self.session.track
            text += (
                f"    音频：{audio.name}"
                f"（{track.sample_rate} Hz / {track.channels}ch / "
                f"{format_ms(track.duration_ms)}）"
            )
        elif audio is not None:
            text += f"    音频：{audio.name}（未载入）"
        else:
            text += "    音频：未指定"
        if self.session.vocals is not None and self.session.vocals_path is not None:
            text += f"    人声：{self.session.vocals_path.name}"
        else:
            text += "    人声：无（用 --vocals 指定分离人声可看清起唱点）"
        if self.player is not None and not self.player.available:
            text += f"    ⚠ {self.player.error}"
        self._file_var.set(text)

    def _refresh_list(self) -> None:
        self.lyrics_list.delete(0, "end")
        for view in self.session.lines:
            shifted = view.start_ms + self.session.offset_ms
            mark = "*" if view.has_byword else " "
            self.lyrics_list.insert("end", f"{format_ms(shifted)} {mark} {view.text}")
        self._sync_list_selection()

    def _sync_list_selection(self) -> None:
        if not self.session.lines:
            return
        self.lyrics_list.selection_clear(0, "end")
        self.lyrics_list.selection_set(self.current_line)
        self.lyrics_list.see(max(0, self.current_line - 3))

    def _update_status(self) -> None:
        lines = self.session.lines
        if not lines:
            self._status_var.set("没有歌词行")
            return
        view = lines[self.current_line]
        shifted = view.start_ms + self.session.offset_ms
        self._status_var.set(
            f"第 {self.current_line + 1}/{len(lines)} 行    "
            f"原文 {format_ms(view.start_ms)}    "
            f"偏移后 {format_ms(shifted)}    "
            f"偏移 {self.session.offset_ms:+d} ms    "
            f"（空格播放 / , . 微调 / [ ] 换行 / 滚轮缩放 / 拖动平移 / 双击画布定位）"
        )

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------

    def _tick(self) -> None:
        """周期性刷新播放头与当前行。"""
        if self.player is not None and self.player.available:
            position = self.player.position_ms
            self._position_var.set(
                f"{format_ms(position)} / {format_ms(self.session.track.duration_ms if self.session.track else 0)}"
            )
            self._sync_current_line(position)
            if not self.player.playing:
                self._play_button.config(text="播放")
        self.draw()
        self.root.after(TICK_MS, self._tick)


def run_app(session: OffsetSession) -> None:
    """创建窗口并进入主循环。"""
    root = tk.Tk()
    app = OffsetApp(root, session)

    def on_close() -> None:
        if app.player is not None:
            app.player.close()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()
