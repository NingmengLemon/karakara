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

    result = main.discover_batch_jobs(tmp_path, tmp_path / "out")

    assert result.skipped == []
    assert [(job.lyrics_path.name, job.audio_path.name) for job in result.jobs] == [
        ("second.lrc", "second.flac"),
        ("first.lrc", "first.flac"),
    ]
    assert [job.output_path.relative_to(tmp_path / "out") for job in result.jobs] == [
        Path("album/second.kara.lrc"),
        Path("first.kara.lrc"),
    ]


def test_discover_batch_jobs_rejects_ambiguous_audio_in_strict_mode(
    tmp_path: Path,
) -> None:
    _write_pair(tmp_path, "song", ".flac")
    (tmp_path / "song.mp3").write_bytes(b"audio")

    with pytest.raises(ValueError, match="exactly one audio"):
        main.discover_batch_jobs(tmp_path, strict=True)


def test_discover_batch_jobs_skips_bad_pairs_by_default(tmp_path: Path) -> None:
    """默认必须宽松：一个坏配对不能拖垮整个曲库。

    回归测试：真实曲库（6622 个 LRC）里有 86 个 LRC 没有同名音频，而此前
    ``discover_batch_jobs`` 会在第一个这样的文件上抛异常，于是连能配对的
    6536 首也一首都不会处理。
    """
    _write_pair(tmp_path, "good")
    (tmp_path / "orphan.lrc").write_text("[00:00.00]no audio", encoding="utf-8")

    result = main.discover_batch_jobs(tmp_path)

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

    result = main.discover_batch_jobs(tmp_path)

    assert len(result.jobs) == 20
    assert calls.count(tmp_path) == 1


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

    monkeypatch.setattr(main, "HttpAligner", FakeAligner)
    monkeypatch.setattr(main, "SubprocessStemSeparator", create_separator)
    monkeypatch.setattr(main.MetadataFilter, "from_file", lambda _path: object())
    monkeypatch.setattr(
        main, "process_job", lambda job, **_kwargs: processed.append(job)
    )
    monkeypatch.setattr(main, "release_item_resources", lambda: released.append(None))

    assert main.run_batch(_batch_args(tmp_path)) == 0
    assert len(processed) == 2
    assert len(released) == 2
    assert len(created_aligners) == 1
    assert len(created_separators) == 1


def _batch_args(batch_dir: Path, **overrides: object) -> argparse.Namespace:
    """构造 ``run_batch`` 需要的 CLI 命名空间（字段与 build_parser() 对齐）。"""
    base: dict[str, object] = {
        "batch_dir": batch_dir,
        "output_dir": None,
        "dump_dir": None,
        "sep_work_dir": None,
        "no_normalize": False,
        "no_vibrato_suppress": False,
        "compress": False,
        "aligner_url": "http://test",
        "aligner_backend": "hfa",
        "aligner_timeout": 120.0,
        "aligner_language": "auto",
        "target_lang": None,
        "metadata_filter": Path("metadata_filter.toml"),
        "strict_pairs": False,
        "separator_cmd": None,
        "separator_backend": "demucs",
        "separator_model": None,
        "separator_device": None,
        "separator_model_dir": None,
        "separator_timeout": 900.0,
        "fail_fast": False,
        "no_offset_estimate": True,
        "offset": None,
        "min_vocal_activity": 0.01,
        "existing_byword_policy": "realign",
        "refine_collapsed_words": False,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


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
