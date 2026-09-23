"""`karakara.gui.session` 的测试：写回必须安全。

重点不是「能不能写」，而是**写坏的可能性**：累加平移、留下半截文件、把 GBK 悄悄转成
UTF-8、把 .bak 覆盖掉。每条都用手算得出的输入输出钉住。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from karakara.gui.session import (
    OffsetSession,
    decode_lrc_bytes,
    load_track,
    write_shifted_lrc,
)

LYRICS = "[ti:测试]\n[00:01.000]第一行\n[00:05.500]第二行\n"


def _write_lyrics(path: Path, text: str = LYRICS, encoding: str = "utf-8") -> Path:
    path.write_bytes(text.encode(encoding))
    return path


def _write_wav(path: Path, seconds: float = 8.0, rate: int = 8000) -> Path:
    samples = np.zeros(int(rate * seconds), dtype=np.float32)
    samples[rate : 2 * rate] = 0.5  # 第 1 秒到第 2 秒有声
    sf.write(str(path), samples, rate, subtype="FLOAT")
    return path


# --------------------------------------------------------------------------
# 写回
# --------------------------------------------------------------------------


def test_write_back_shifts_and_keeps_everything_else(tmp_path: Path) -> None:
    path = _write_lyrics(tmp_path / "song.lrc")

    report = write_shifted_lrc(path, 250)

    assert report.tags == 2
    assert report.clamped == 0
    assert report.wrote
    assert path.read_text(encoding="utf-8") == (
        "[ti:测试]\n[00:01.250]第一行\n[00:05.750]第二行\n"
    )


def test_backup_holds_the_content_from_before_the_write(tmp_path: Path) -> None:
    path = _write_lyrics(tmp_path / "song.lrc")

    report = write_shifted_lrc(path, 250)

    assert report.backup == tmp_path / "song.lrc.bak"
    assert report.backup is not None
    assert report.backup.read_text(encoding="utf-8") == LYRICS


def test_backup_is_overwritten_by_the_next_save(tmp_path: Path) -> None:
    """`.bak` 是「上一次写回之前」的内容，也就是一层撤销。"""
    path = _write_lyrics(tmp_path / "song.lrc")

    write_shifted_lrc(path, 250)
    write_shifted_lrc(path, 250)

    backup = tmp_path / "song.lrc.bak"
    assert backup.read_text(encoding="utf-8") == (
        "[ti:测试]\n[00:01.250]第一行\n[00:05.750]第二行\n"
    )


def test_zero_delta_does_not_touch_the_file(tmp_path: Path) -> None:
    path = _write_lyrics(tmp_path / "song.lrc")
    before = path.stat().st_mtime_ns

    report = write_shifted_lrc(path, 0)

    assert report.wrote is False
    assert report.backup is None
    assert path.stat().st_mtime_ns == before
    assert not (tmp_path / "song.lrc.bak").exists()


def test_refusing_to_clamp_leaves_the_file_untouched(tmp_path: Path) -> None:
    """不能夹取时必须在**动文件之前**失败。"""
    path = _write_lyrics(tmp_path / "song.lrc")

    with pytest.raises(ValueError, match="负时间戳"):
        write_shifted_lrc(path, -2000, clamp_at_zero=False)

    assert path.read_text(encoding="utf-8") == LYRICS
    assert not (tmp_path / "song.lrc.bak").exists()
    assert not (tmp_path / "song.lrc.tmp").exists()


def test_no_temporary_file_survives_a_successful_write(tmp_path: Path) -> None:
    path = _write_lyrics(tmp_path / "song.lrc")
    write_shifted_lrc(path, 250)
    assert not (tmp_path / "song.lrc.tmp").exists()


def test_gbk_lyrics_stay_gbk(tmp_path: Path) -> None:
    """原地写回不能顺手改编码：GBK 的歌词写回去还得是 GBK。"""
    path = _write_lyrics(
        tmp_path / "gbk.lrc", "[00:01.000]中文歌词\n", encoding="gb18030"
    )

    write_shifted_lrc(path, 500)

    raw = path.read_bytes()
    assert raw.decode("gb18030") == "[00:01.500]中文歌词\n"
    with pytest.raises(UnicodeDecodeError):
        raw.decode("utf-8")


def test_crlf_lyrics_keep_crlf(tmp_path: Path) -> None:
    path = tmp_path / "crlf.lrc"
    path.write_bytes(b"[00:01.000]a\r\n[00:02.000]b\r\n")

    write_shifted_lrc(path, 1000)

    assert path.read_bytes() == b"[00:02.000]a\r\n[00:03.000]b\r\n"


def test_bom_is_preserved(tmp_path: Path) -> None:
    path = tmp_path / "bom.lrc"
    path.write_bytes(b"\xef\xbb\xbf[00:01.000]a\n")

    write_shifted_lrc(path, 1000)

    assert path.read_bytes() == b"\xef\xbb\xbf[00:02.000]a\n"


def test_undecodable_file_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "weird.lrc"
    path.write_bytes(b"\x81\x30\x81\x30\xff\xfe\x00\x00")

    with pytest.raises(ValueError, match="无法安全地原地写回"):
        decode_lrc_bytes(path.read_bytes(), path)


# --------------------------------------------------------------------------
# 会话
# --------------------------------------------------------------------------


def test_session_lines_and_shifted_positions(tmp_path: Path) -> None:
    session = OffsetSession(_write_lyrics(tmp_path / "song.lrc"))
    session.set_offset(250)

    # `[ti:测试]` 被解析器收进 metadata，不是行；所以这里只有两行歌词
    assert [line.text for line in session.lines] == ["第一行", "第二行"]
    assert session.lyrics.metadata.get("ti") == "测试"
    assert session.lines[0].start_ms == 1000
    assert session.shifted_start_ms(0) == 1250
    assert session.nudge(-250) == 0


def test_session_window_follows_the_offset(tmp_path: Path) -> None:
    session = OffsetSession(_write_lyrics(tmp_path / "song.lrc"))
    session.set_offset(100)

    assert session.window_ms(0) == (1100, 5600)
    assert session.window_ms(0, shifted=False) == (1000, 5500)


def test_saving_twice_does_not_accumulate(tmp_path: Path) -> None:
    """回归：写回成功后偏移必须清零，否则第二次保存会把同一个偏移再叠一次。"""
    session = OffsetSession(_write_lyrics(tmp_path / "song.lrc"))
    session.set_offset(250)

    first = session.save()
    assert first.wrote
    assert session.offset_ms == 0

    second = session.save()

    assert second.wrote is False, "偏移已清零，第二次保存不该再动文件"
    assert (tmp_path / "song.lrc").read_text(encoding="utf-8") == (
        "[ti:测试]\n[00:01.250]第一行\n[00:05.750]第二行\n"
    )


def test_session_can_save_to_another_path_without_a_backup(tmp_path: Path) -> None:
    source = _write_lyrics(tmp_path / "song.lrc")
    target = tmp_path / "copy.lrc"
    session = OffsetSession(source)
    session.set_offset(-500)

    report = session.save(target=target)

    assert report.wrote
    assert report.backup is None, "另存为不该在原文件旁边造 .bak"
    assert target.read_text(encoding="utf-8") == (
        "[ti:测试]\n[00:00.500]第一行\n[00:05.000]第二行\n"
    )
    assert source.read_text(encoding="utf-8") == LYRICS


def test_session_reports_byword_lines(tmp_path: Path) -> None:
    path = _write_lyrics(
        tmp_path / "byword.lrc", "[00:01.000][00:01.500]逐字[00:02.000]行\n"
    )
    session = OffsetSession(path)
    assert session.lines[0].has_byword is True


def test_load_track_reports_duration_and_channels(tmp_path: Path) -> None:
    track = load_track(_write_wav(tmp_path / "a.wav", seconds=8.0))

    assert track.sample_rate == 8000
    assert track.channels == 1
    assert track.duration_ms == pytest.approx(8000.0)
    assert track.envelope.bucket_count > 0


def test_session_loads_track_and_uses_it_for_the_last_line_window(
    tmp_path: Path,
) -> None:
    session = OffsetSession(
        _write_lyrics(tmp_path / "song.lrc"),
        audio_path=_write_wav(tmp_path / "a.wav", seconds=8.0),
    )
    track = session.load_track()

    assert track.duration_ms == pytest.approx(8000.0)
    # 最后一行（第二行）没有 end、后面也没有行，窗口右端取音频末尾
    assert session.window_ms(1) == (5500, 8000)
