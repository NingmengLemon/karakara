"""`karakara.paths` 的路径解析测试。

存在的理由：worker 脚本要用 ``uv run --script <路径>`` 拉起，而相对路径由 uv 按
**当前工作目录**解析。这一层把「仓库内文件」的定位收成一个出口，所以自己也要被钉住：
解析结果必须是绝对路径、必须真的存在、且**与 CWD 无关**。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from karakara.paths import REPO_ROOT, repo_file

WORKER = "scripts/separator_worker.py"


def test_repo_root_points_at_the_repository() -> None:
    assert (REPO_ROOT / "pyproject.toml").is_file()
    assert (REPO_ROOT / "src" / "karakara" / "paths.py").is_file()


def test_repo_file_returns_an_existing_absolute_path() -> None:
    path = repo_file(WORKER)
    assert path.is_absolute()
    assert path.is_file()
    assert path == REPO_ROOT / WORKER


def test_repo_file_is_independent_of_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """回归：换到别的目录后解析结果不能变。"""
    monkeypatch.chdir(tmp_path)
    assert repo_file(WORKER) == REPO_ROOT / WORKER
    assert repo_file(WORKER).is_file()


def test_repo_file_accepts_an_absolute_path() -> None:
    absolute = REPO_ROOT / WORKER
    assert repo_file(absolute) == absolute


def test_repo_file_rejects_a_missing_file() -> None:
    """缺文件时要给出可执行的补救信息，而不是让 uv 抛一句难懂的错。"""
    with pytest.raises(FileNotFoundError) as excinfo:
        repo_file("scripts/definitely-not-here.py")

    message = str(excinfo.value)
    assert "definitely-not-here.py" in message
    assert "KARAKARA_SEPARATOR_CMD" in message
