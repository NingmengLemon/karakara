"""批处理：配对发现与「跑一组输入」。

发现规则是这个项目里最容易静默出错的一环（一个坏配对曾让整个曲库一首都不处理），
所以规则本身住在包里、被直接测试，而不是藏在 CLI 里。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from karakara.batch import BatchJob, discover_batch_jobs


def _write_pair(directory: Path, stem: str, suffix: str = ".flac") -> None:
    (directory / f"{stem}.lrc").write_text("[00:00.00]test", encoding="utf-8")
    (directory / f"{stem}{suffix}").write_bytes(b"audio")


def test_discover_batch_jobs_preserves_relative_output_paths(tmp_path: Path) -> None:
    nested = tmp_path / "album"
    nested.mkdir()
    _write_pair(tmp_path, "first")
    _write_pair(nested, "second")
    (tmp_path / "old.kara.lrc").write_text("ignored", encoding="utf-8")

    result = discover_batch_jobs(tmp_path, tmp_path / "out")

    assert result.skipped == []
    assert [(job.lyrics_path.name, job.audio_path.name) for job in result.jobs] == [
        ("second.lrc", "second.flac"),
        ("first.lrc", "first.flac"),
    ]
    assert [job.output_path.relative_to(tmp_path / "out") for job in result.jobs] == [
        Path("album/second.kara.lrc"),
        Path("first.kara.lrc"),
    ]


def test_discover_batch_jobs_outputs_next_to_input_by_default(tmp_path: Path) -> None:
    _write_pair(tmp_path, "song")

    result = discover_batch_jobs(tmp_path)

    assert [job.output_path for job in result.jobs] == [tmp_path / "song.kara.lrc"]


def test_discover_batch_jobs_rejects_ambiguous_audio_in_strict_mode(
    tmp_path: Path,
) -> None:
    _write_pair(tmp_path, "song", ".flac")
    (tmp_path / "song.mp3").write_bytes(b"audio")

    with pytest.raises(ValueError, match="exactly one audio"):
        discover_batch_jobs(tmp_path, strict=True)


def test_discover_batch_jobs_skips_bad_pairs_by_default(tmp_path: Path) -> None:
    """默认必须宽松：一个坏配对不能拖垮整个曲库。

    回归测试：真实曲库（6622 个 LRC）里有 86 个 LRC 没有同名音频，而此前
    ``discover_batch_jobs`` 会在第一个这样的文件上抛异常，于是连能配对的
    6536 首也一首都不会处理。
    """
    _write_pair(tmp_path, "good")
    (tmp_path / "orphan.lrc").write_text("[00:00.00]no audio", encoding="utf-8")

    result = discover_batch_jobs(tmp_path)

    assert [job.lyrics_path.name for job in result.jobs] == ["good.lrc"]
    assert [(item.lyrics_path.name, item.reason) for item in result.skipped] == [
        ("orphan.lrc", "none")
    ]


def test_discover_batch_jobs_lists_each_directory_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """发现阶段每个目录只枚举一次。

    回归：大曲库里几千个 LRC 常挤在同一个目录，逐行 ``iterdir()`` 会把同一批目录
    条目重复枚举几百万次——实测某 6600 个 LRC 的真实曲库上，发现阶段因此超过
    10 分钟未跑完（缓存后 1.8s）。
    """
    for index in range(20):
        _write_pair(tmp_path, f"song{index}")

    calls: list[Path] = []
    real_iterdir = Path.iterdir

    def counting_iterdir(self: Path) -> object:
        calls.append(self)
        return real_iterdir(self)

    monkeypatch.setattr(Path, "iterdir", counting_iterdir)

    result = discover_batch_jobs(tmp_path)

    assert len(result.jobs) == 20
    assert calls.count(tmp_path) == 1


def test_discover_batch_jobs_rejects_a_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        discover_batch_jobs(tmp_path / "nope")


def test_batch_job_is_hashable_and_ordered(tmp_path: Path) -> None:
    """`BatchJob` 是 frozen dataclass：能被放进集合、也能按字段比较。"""
    job = BatchJob(tmp_path / "a.lrc", tmp_path / "a.flac", tmp_path / "a.kara.lrc")
    assert job == BatchJob(
        tmp_path / "a.lrc", tmp_path / "a.flac", tmp_path / "a.kara.lrc"
    )
    assert len({job, job}) == 1
