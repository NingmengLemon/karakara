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
        def __init__(self, **_kwargs: object) -> None:
            pass

        def close(self) -> None:
            pass

    def create_separator(**_kwargs: object) -> FakeSeparator:
        separator = FakeSeparator()
        created_separators.append(separator)
        return separator

    monkeypatch.setattr(main, "Qwen3ForcedAligner", FakeAligner)
    monkeypatch.setattr(main, "SubprocessStemSeparator", create_separator)
    monkeypatch.setattr(main.MetadataFilter, "from_file", lambda _path: object())
    monkeypatch.setattr(
        main, "process_job", lambda job, **_kwargs: processed.append(job)
    )
    monkeypatch.setattr(main, "release_item_resources", lambda: released.append(None))

    args = argparse.Namespace(
        batch_dir=tmp_path,
        output_dir=None,
        dump_dir=None,
        sep_work_dir=None,
        no_normalize=False,
        no_vibrato_suppress=False,
        compress=False,
        aligner_url="http://test",
        aligner_language="auto",
        target_lang=None,
        separator_cmd=None,
        separator_backend="demucs",
        separator_model=None,
        separator_device=None,
        separator_model_dir=None,
        separator_timeout=None,
        fail_fast=False,
        no_offset_estimate=True,
        offset=None,
        min_vocal_activity=0.01,
        existing_byword_policy="realign",
    )

    assert main.run_batch(args) == 0
    assert len(processed) == 2
    assert len(released) == 2
    assert len(created_aligners) == 1
    assert len(created_separators) == 1


def _separator_args(**overrides: object) -> argparse.Namespace:
    base: dict[str, object] = {
        "separator_cmd": None,
        "separator_backend": "demucs",
        "separator_model": None,
        "separator_device": None,
        "separator_model_dir": None,
        "separator_timeout": None,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def test_build_separator_uses_uv_script_for_demucs() -> None:
    """默认后端走 uv 管理的独立环境，主环境的依赖里因此不需要 torch。"""
    separator = main.build_separator(_separator_args())

    assert separator.command[:3] == ["uv", "run", "--script"]
    assert separator.command[3].endswith("scripts/separator_worker.py")
    assert "separator_worker_audio_separator.py" not in separator.command[3]


def test_build_separator_selects_audio_separator_backend() -> None:
    separator = main.build_separator(
        _separator_args(
            separator_backend="audio-separator",
            separator_model="UVR_MDXNET_KARA_2.onnx",
        )
    )

    assert separator.command[3].endswith("scripts/separator_worker_audio_separator.py")


def test_explicit_separator_cmd_wins_over_backend() -> None:
    """显式命令优先级最高：便于接自有环境或远程 worker。"""
    separator = main.build_separator(
        _separator_args(
            separator_cmd=["C:/some/python.exe", "my_worker.py"],
            separator_backend="audio-separator",
        )
    )

    assert separator.command == ["C:/some/python.exe", "my_worker.py"]


def test_all_separator_backends_have_a_worker_script() -> None:
    """每个后端选项都必须指向真实存在的 worker 脚本。"""
    for backend, script in main._SEPARATOR_WORKERS.items():
        assert Path(script).is_file(), f"后端 {backend} 的 worker 脚本不存在: {script}"
