"""批处理 CLI 的文件发现与资源复用测试。"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pytest

_MAIN_PATH = Path(__file__).parent.parent / "main.py"
_MAIN_SPEC = importlib.util.spec_from_file_location("karakara_main", _MAIN_PATH)
assert _MAIN_SPEC is not None and _MAIN_SPEC.loader is not None
main = importlib.util.module_from_spec(_MAIN_SPEC)
sys.modules[_MAIN_SPEC.name] = main
_MAIN_SPEC.loader.exec_module(main)


def _write_pair(directory: Path, stem: str, suffix: str = ".flac") -> None:
    (directory / f"{stem}.lrc").write_text("[00:00.00]test", encoding="utf-8")
    (directory / f"{stem}{suffix}").write_bytes(b"audio")


def test_discover_batch_jobs_preserves_relative_output_paths(tmp_path: Path) -> None:
    nested = tmp_path / "album"
    nested.mkdir()
    _write_pair(tmp_path, "first")
    _write_pair(nested, "second")
    (tmp_path / "old.kara.lrc").write_text("ignored", encoding="utf-8")

    jobs = main.discover_batch_jobs(tmp_path, tmp_path / "out")

    assert [(job.lyrics_path.name, job.audio_path.name) for job in jobs] == [
        ("second.lrc", "second.flac"),
        ("first.lrc", "first.flac"),
    ]
    assert [job.output_path.relative_to(tmp_path / "out") for job in jobs] == [
        Path("album/second.kara.lrc"),
        Path("first.kara.lrc"),
    ]


def test_discover_batch_jobs_rejects_ambiguous_audio(tmp_path: Path) -> None:
    _write_pair(tmp_path, "song", ".flac")
    (tmp_path / "song.mp3").write_bytes(b"audio")

    with pytest.raises(ValueError, match="exactly one audio"):
        main.discover_batch_jobs(tmp_path)


def test_run_batch_reuses_workers_and_releases_after_each_job(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _write_pair(tmp_path, "one")
    _write_pair(tmp_path, "two")
    processed: list[object] = []
    released: list[None] = []
    created_aligners: list[object] = []
    created_separators: list[object] = []

    class FakeAligner:
        def __init__(self, **_kwargs: object) -> None:
            created_aligners.append(self)

        def close(self) -> None:
            pass

    class FakeSeparator:
        pass

    def create_separator() -> FakeSeparator:
        separator = FakeSeparator()
        created_separators.append(separator)
        return separator

    monkeypatch.setattr(main, "Qwen3ForcedAligner", FakeAligner)
    monkeypatch.setattr(main, "DemucsSeparator", create_separator)
    monkeypatch.setattr(main.MetadataFilter, "from_file", lambda _path: object())
    monkeypatch.setattr(
        main, "process_job", lambda job, **_kwargs: processed.append(job)
    )
    monkeypatch.setattr(main, "release_item_resources", lambda: released.append(None))

    args = argparse.Namespace(
        batch_dir=tmp_path,
        output_dir=None,
        dump_dir=None,
        no_normalize=False,
        no_vibrato_suppress=False,
        compress=False,
        aligner_url="http://test",
        fail_fast=False,
        no_offset_estimate=True,
        offset=None,
        min_vocal_activity=0.01,
    )

    assert main.run_batch(args) == 0
    assert len(processed) == 2
    assert len(released) == 2
    assert len(created_aligners) == 1
    assert len(created_separators) == 1
