"""GUI 的状态层：歌词、音频轨、当前偏移、写回。

这一层刻意**不 import 任何 GUI 库**，因为「写回用户的歌词文件」是这里最容易出事、
也最值得单测的部分：

* 只动时间戳的数字（见 :mod:`karakara.gui.lrctext`），其余字节原样保留；
* 写回是**原子替换**（先写同目录临时文件再 ``os.replace``），中途失败不会留下半截文件；
* 写回前把**本次写入之前**的内容复制成 ``<文件名>.bak``，等于一次撤销；
* 编码按原文件来（UTF-8 / GB18030 / CP932 依次严格试解），写回时用同一个编码，
  不把 GBK 的歌词悄悄转成 UTF-8。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path

import numpy as np
from lemony_lrc_parser import Lyrics
from numpy.typing import NDArray

from karakara.gui.lrctext import shift_lrc_text
from karakara.gui.peaks import DEFAULT_BUCKET_MS, PeakEnvelope, compute_peaks
from karakara.paths import REPO_ROOT
from karakara.utils.io import load_audio_native

logger = getLogger(__name__)

#: 尝试解码歌词文件的编码顺序。UTF-8 优先（本项目自己的产物都是 UTF-8），
#: 其次是中文歌词常见的 GB18030 与日文歌词的 CP932。
_ENCODINGS = ("utf-8", "gb18030", "cp932")


@dataclass(frozen=True)
class LineView:
    """界面要显示的一行。时间戳是**原文**的值，偏移由界面叠加。"""

    index: int
    start_ms: int
    end_ms: int | None
    text: str
    #: 这一行是否已经带逐字时间标签。
    has_byword: bool = False


@dataclass(frozen=True)
class AudioTrack:
    """解码好的音频轨：播放用原始波形，绘制用包络。"""

    path: Path
    data: NDArray[np.float32]
    sample_rate: int
    envelope: PeakEnvelope

    @property
    def duration_ms(self) -> float:
        return self.data.shape[-1] / self.sample_rate * 1000.0

    @property
    def channels(self) -> int:
        return self.data.shape[0] if self.data.ndim == 2 else 1


@dataclass(frozen=True)
class SaveReport:
    """一次写回的结果。"""

    path: Path
    backup: Path | None
    delta_ms: int
    tags: int
    clamped: int
    bytes_written: int

    @property
    def wrote(self) -> bool:
        return self.bytes_written > 0


def decode_lrc_bytes(raw: bytes, path: Path) -> tuple[str, str]:
    """按候选编码严格解码歌词字节，返回 ``(文本, 编码名)``。

    Raises:
        ValueError: 所有候选编码都解不出来（不猜，直接报错让用户自己转）。
    """
    for encoding in _ENCODINGS:
        try:
            return raw.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    raise ValueError(
        f"{path} 不是 UTF-8/GB18030/CP932 中的任何一种，无法安全地原地写回；"
        f"请先把它转成 UTF-8"
    )


def load_track(path: str | Path, *, bucket_ms: float = DEFAULT_BUCKET_MS) -> AudioTrack:
    """解码音频并建好包络（播放与绘制共用这一次解码）。"""
    resolved = Path(path)
    data, sample_rate = load_audio_native(resolved)
    return AudioTrack(
        path=resolved,
        data=data,
        sample_rate=sample_rate,
        envelope=compute_peaks(data, sample_rate, bucket_ms=bucket_ms),
    )


#: 分离人声的候选命名（放在音频同目录时）。`audio-separator` 与 Demucs 的命名习惯不同。
_VOCALS_SUFFIXES = ("_vocals.wav", ".vocals.wav", " (Vocals).wav")


def find_vocals(
    audio_path: str | Path, *, cache_dir: str | Path | None = None
) -> Path | None:
    """找现成的分离人声轨，找不到返回 ``None``。

    先看音频同目录的常见命名，再看扰动稳定性工具的缓存目录
    （``tmp/perturbation/<音频名>_vocals.wav``，见 ``scripts/check_aligner_stability.py``）。
    GUI 需要它是因为**混音波形看不出人声在哪**，而调偏移靠的正是「人声什么时候进来」。
    """
    audio = Path(audio_path)
    candidates = [audio.with_name(audio.stem + suffix) for suffix in _VOCALS_SUFFIXES]
    if cache_dir is None:
        cache_dir = REPO_ROOT / "tmp" / "perturbation"
    candidates.append(Path(cache_dir) / f"{audio.stem}_vocals.wav")

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def write_shifted_text(
    target: str | Path,
    text: str,
    delta_ms: int,
    *,
    encoding: str = "utf-8",
    backup_raw: bytes | None = None,
    clamp_at_zero: bool = True,
) -> SaveReport:
    """把 ``text`` 的时间戳平移后写到 ``target``（原子替换）。

    这是**内容已在手上**的那条路径（「另存为」用它）；原地改文件走
    :func:`write_shifted_lrc`。

    Args:
        target: 目标文件，不存在时创建。
        text: 待平移的 LRC 文本。
        delta_ms: 平移量，正值表示歌词偏早、需要延后。
        encoding: 写出用的编码，必须与 ``text`` 的来源一致。
        backup_raw: 非 ``None`` 时先把这些字节写成 ``<目标名>.bak``（原地写回时传原文）。
        clamp_at_zero: 平移后为负的时间戳夹到 0；关掉时直接报错、不写任何东西。

    Returns:
        写回报告；平移量为 0 或没有标签时 ``bytes_written`` 为 0（不动文件）。
    """
    destination = Path(target)
    result = shift_lrc_text(text, delta_ms, clamp_at_zero=clamp_at_zero)

    if not result.changed:
        return SaveReport(
            path=destination,
            backup=None,
            delta_ms=delta_ms,
            tags=result.tags,
            clamped=result.clamped,
            bytes_written=0,
        )

    backup_path: Path | None = None
    if backup_raw is not None:
        backup_path = destination.with_name(destination.name + ".bak")
        backup_path.write_bytes(backup_raw)

    payload = result.text.encode(encoding)
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()

    logger.info(
        f"shifted {result.tags} timestamp(s) by {delta_ms:+d}ms in {destination} "
        f"(clamped {result.clamped}, encoding {encoding})"
    )
    return SaveReport(
        path=destination,
        backup=backup_path,
        delta_ms=delta_ms,
        tags=result.tags,
        clamped=result.clamped,
        bytes_written=len(payload),
    )


def write_shifted_lrc(
    path: str | Path,
    delta_ms: int,
    *,
    backup: bool = True,
    clamp_at_zero: bool = True,
) -> SaveReport:
    """把 ``path`` 里的时间戳整体平移 ``delta_ms`` 并**原地写回**。

    Args:
        path: 目标歌词文件（必须已存在，会被覆盖）。
        delta_ms: 平移量，正值表示歌词偏早、需要延后。
        backup: 是否把写入前的内容复制成 ``<文件名>.bak``。
        clamp_at_zero: 平移后为负的时间戳夹到 0；关掉时直接报错、不写任何东西。

    Returns:
        写回报告。
    """
    target = Path(path)
    raw = target.read_bytes()
    text, encoding = decode_lrc_bytes(raw, target)
    return write_shifted_text(
        target,
        text,
        delta_ms,
        encoding=encoding,
        backup_raw=raw if backup else None,
        clamp_at_zero=clamp_at_zero,
    )


class OffsetSession:
    """一次「对照波形与歌词调偏移」的会话。

    Attributes:
        lyrics_path: 正在编辑的歌词文件。
        offset_ms: 当前偏移（毫秒），正值表示歌词偏早、需要延后。
    """

    def __init__(
        self,
        lyrics_path: str | Path,
        *,
        audio_path: str | Path | None = None,
        vocals_path: str | Path | None = None,
        bucket_ms: float = DEFAULT_BUCKET_MS,
    ) -> None:
        self.lyrics_path = Path(lyrics_path)
        self._raw, self.encoding = decode_lrc_bytes(
            self.lyrics_path.read_bytes(), self.lyrics_path
        )
        self.lyrics: Lyrics = Lyrics.loads(self._raw)
        self.audio_path = Path(audio_path) if audio_path is not None else None
        self.vocals_path = Path(vocals_path) if vocals_path is not None else None
        self.track: AudioTrack | None = None
        self.vocals: AudioTrack | None = None
        self.offset_ms: int = 0
        self._bucket_ms = bucket_ms

    # ---------- 歌词 ----------

    @property
    def original_text(self) -> str:
        """打开时的原文（未被任何平移改动）。"""
        return self._raw

    @property
    def lines(self) -> list[LineView]:
        """所有行（含元数据行）。"""
        return [
            LineView(
                index=index,
                start_ms=line.start,
                end_ms=line.end,
                text=line.text,
                has_byword=any(
                    token.start is not None or token.end is not None
                    for token in line.content
                ),
            )
            for index, line in enumerate(self.lyrics)
        ]

    def line_texts(self) -> list[str]:
        """所有行的纯文本，顺序与 :attr:`lines` 一致。"""
        return [line.text for line in self.lyrics]

    def shifted_start_ms(self, index: int) -> int:
        """第 ``index`` 行**加上当前偏移**后的起点（可能为负，界面自行夹取显示）。"""
        return self.lyrics[index].start + self.offset_ms

    def window_ms(self, index: int, *, shifted: bool = True) -> tuple[int, int]:
        """第 ``index`` 行的播放窗口 ``[起点, 下一行起点)``。

        与 ``core._line_sample_range`` 的口径一致：没有 ``end`` 时用下一行的起点，
        最后一行用音频末尾（没有音频时用起点 + 默认 5 秒）。
        """
        delta = self.offset_ms if shifted else 0
        start = self.lyrics[index].start + delta
        line = self.lyrics[index]
        if line.end is not None:
            return start, line.end + delta
        if index + 1 < len(self.lyrics):
            return start, self.lyrics[index + 1].start + delta
        if self.track is not None:
            return start, int(self.track.duration_ms)
        return start, start + 5_000

    # ---------- 音频 ----------

    def load_track(self, path: str | Path | None = None) -> AudioTrack:
        """解码音频（默认用构造时给的路径）并缓存。"""
        target = Path(path) if path is not None else self.audio_path
        if target is None:
            raise ValueError("没有指定音频文件")
        self.audio_path = target
        self.track = load_track(target, bucket_ms=self._bucket_ms)
        return self.track

    def load_vocals(self, path: str | Path | None = None) -> AudioTrack | None:
        """解码分离人声（只用于画波形，不用于播放）并缓存；没有就返回 ``None``。"""
        target = Path(path) if path is not None else self.vocals_path
        if target is None or not Path(target).is_file():
            return None
        self.vocals_path = Path(target)
        self.vocals = load_track(target, bucket_ms=self._bucket_ms)
        return self.vocals

    # ---------- 偏移 ----------

    def set_offset(self, offset_ms: int) -> None:
        """设置当前偏移（毫秒）。"""
        self.offset_ms = int(offset_ms)

    def nudge(self, delta_ms: int) -> int:
        """在当前偏移上再叠加一点，返回新值。"""
        self.offset_ms += int(delta_ms)
        return self.offset_ms

    # ---------- 写回 ----------

    def save(
        self, *, target: str | Path | None = None, backup: bool = True
    ) -> SaveReport:
        """把当前偏移写回歌词文件（默认就是打开的那个文件）。

        写回成功后**偏移清零**，因为文件本身已经落在新的时间轴上了：不清零的话第二次
        保存会把同一个偏移再叠一次（数据损坏）。界面上表现为「标记仍在原处、文件已对齐」。

        另存为（``target`` 指向别的文件）时用内存里的原文平移，不读目标文件、也不建
        ``.bak``；原地写回时才建备份。
        """
        destination = Path(target) if target is not None else self.lyrics_path
        in_place = destination == self.lyrics_path
        if in_place:
            report = write_shifted_lrc(destination, self.offset_ms, backup=backup)
        else:
            report = write_shifted_text(
                destination, self._raw, self.offset_ms, encoding=self.encoding
            )
        if report.wrote:
            self.lyrics_path = destination
            self._raw, self.encoding = decode_lrc_bytes(
                destination.read_bytes(), destination
            )
            self.lyrics = Lyrics.loads(self._raw)
            self.offset_ms = 0
        return report
