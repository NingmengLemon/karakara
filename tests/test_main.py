"""CLI 入口的装配与编排。

逻辑本身住在包里（见 ``test_batch.py`` / ``test_aligner_backends.py``），这里只测
``main.py`` 自己的职责：把参数解成包的入参、复用 worker、逐项回收资源。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pytest

from karakara import backends
from karakara.backends import ALIGNER_BACKENDS, SEPARATOR_BACKENDS

_ROOT = Path(__file__).resolve().parent.parent


def _write_pair(directory: Path, stem: str, suffix: str = ".flac") -> None:
    (directory / f"{stem}.lrc").write_text("[00:00.00]test", encoding="utf-8")
    (directory / f"{stem}{suffix}").write_bytes(b"audio")


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


def _aligner_args(**overrides: object) -> argparse.Namespace:
    base: dict[str, object] = {
        "aligner_backend": "hfa",
        "aligner_url": None,
        "aligner_timeout": 120.0,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


# --------------------------------------------------------------------------
# run_batch：共享 worker、逐项回收
# --------------------------------------------------------------------------


def test_run_batch_reuses_workers_and_releases_after_each_job(
    main_module: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
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

    monkeypatch.setattr(main_module, "HttpAligner", FakeAligner)
    # 分离器的构造在包里（`karakara.backends.build_separator`），所以要打在它身上。
    monkeypatch.setattr(backends, "SubprocessStemSeparator", create_separator)
    monkeypatch.setattr(main_module.MetadataFilter, "from_file", lambda _path: object())
    monkeypatch.setattr(
        main_module, "process_job", lambda job, **_kwargs: processed.append(job)
    )
    monkeypatch.setattr(
        main_module, "release_item_resources", lambda: released.append(None)
    )

    assert main_module.run_batch(_batch_args(tmp_path)) == 0
    assert len(processed) == 2
    assert len(released) == 2
    assert len(created_aligners) == 1
    assert len(created_separators) == 1


def test_run_batch_reports_failures_without_stopping(
    main_module: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """一个任务失败不该终止整批：退出码非 0、但其余任务照跑。"""
    _write_pair(tmp_path, "one")
    _write_pair(tmp_path, "two")
    attempted: list[str] = []

    class FakeAligner:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def close(self) -> None:
            pass

    class FakeSeparator:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def close(self) -> None:
            pass

    def fake_process(job: object, **_kwargs: object) -> None:
        attempted.append(str(job))
        if len(attempted) == 1:
            raise RuntimeError("boom")

    monkeypatch.setattr(main_module, "HttpAligner", FakeAligner)
    monkeypatch.setattr(backends, "SubprocessStemSeparator", FakeSeparator)
    monkeypatch.setattr(main_module.MetadataFilter, "from_file", lambda _path: object())
    monkeypatch.setattr(main_module, "process_job", fake_process)

    assert main_module.run_batch(_batch_args(tmp_path)) == 1
    assert len(attempted) == 2


# --------------------------------------------------------------------------
# build_separator / build_aligner
# --------------------------------------------------------------------------


def test_build_separator_uses_uv_script_for_demucs(main_module: Any) -> None:
    """默认后端走 uv 管理的独立环境，主环境的依赖里因此不需要 torch。"""
    separator = main_module.build_separator(_separator_args())

    assert separator.command[:3] == ["uv", "run", "--script"]
    # 用 Path 比较而不是字符串后缀：解析出来的是 Windows 绝对路径（反斜杠），
    # 「以后缀判断」会被路径分隔符坑掉。
    assert (
        Path(separator.command[3]).resolve()
        == (_ROOT / SEPARATOR_BACKENDS["demucs"].script).resolve()
    )


def test_build_separator_selects_audio_separator_backend(main_module: Any) -> None:
    separator = main_module.build_separator(
        _separator_args(
            separator_backend="audio-separator",
            separator_model="UVR_MDXNET_KARA_2.onnx",
        )
    )

    assert (
        Path(separator.command[3]).resolve()
        == (_ROOT / SEPARATOR_BACKENDS["audio-separator"].script).resolve()
    )


def test_explicit_separator_cmd_wins_over_backend(main_module: Any) -> None:
    """显式命令优先级最高：便于接自有环境或远程 worker。"""
    separator = main_module.build_separator(
        _separator_args(
            separator_cmd=["C:/some/python.exe", "my_worker.py"],
            separator_backend="audio-separator",
        )
    )

    assert separator.command == ["C:/some/python.exe", "my_worker.py"]


def test_build_aligner_uses_the_backend_default_url(
    main_module: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorded: dict[str, object] = {}

    class FakeAligner:
        def __init__(self, **kwargs: object) -> None:
            recorded.update(kwargs)

    monkeypatch.setattr(main_module, "HttpAligner", FakeAligner)

    main_module.build_aligner(_aligner_args(aligner_backend="qwen3"))
    assert recorded["base_url"] == ALIGNER_BACKENDS["qwen3"].default_url
    assert recorded["timeout"] == 120.0

    main_module.build_aligner(
        _aligner_args(aligner_backend="qwen3", aligner_url="http://other:9000")
    )
    assert recorded["base_url"] == "http://other:9000"


def test_build_aligner_turns_non_positive_timeout_into_none(
    main_module: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorded: dict[str, object] = {}

    class FakeAligner:
        def __init__(self, **kwargs: object) -> None:
            recorded.update(kwargs)

    monkeypatch.setattr(main_module, "HttpAligner", FakeAligner)

    main_module.build_aligner(_aligner_args(aligner_timeout=0.0))
    assert recorded["timeout"] is None


# --------------------------------------------------------------------------
# main()：语言能力在开跑前拦截
# --------------------------------------------------------------------------


def test_main_rejects_a_language_the_backend_lacks(
    main_module: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    """`hfa` 不支持 yue/ko：必须在开跑前以退出码 2 拦下，并指出能用的后端。

    回归：此前这条路径要跑到服务端才拿到 400，而那时人声分离已经白跑完了。
    """
    with pytest.raises(SystemExit) as info:
        main_module.main(
            ["--lyrics", "a.lrc", "--audio", "a.flac", "--aligner-language", "ko"]
        )

    assert info.value.code == 2
    stderr = capsys.readouterr().err
    assert "不支持语言" in stderr
    assert "qwen3" in stderr


def test_main_accepts_the_same_language_on_qwen3(
    main_module: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """同一个语言在 qwen3 上不该被拦（拦的是能力，不是语言本身）。"""
    seen: list[argparse.Namespace] = []

    def fake_run_single(args: argparse.Namespace) -> int:
        seen.append(args)
        return 0

    monkeypatch.setattr(main_module, "run_single", fake_run_single)

    with pytest.raises(SystemExit):
        main_module.main(
            [
                "--lyrics",
                "a.lrc",
                "--audio",
                "a.flac",
                "--aligner-backend",
                "qwen3",
                "--aligner-language",
                "ko",
            ]
        )

    assert len(seen) == 1
    assert seen[0].aligner_language == "ko"
