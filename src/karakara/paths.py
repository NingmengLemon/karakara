"""仓库内文件的定位。

存在的理由：worker 脚本要靠 ``uv run --script <路径>`` 拉起，而**相对路径是按当前
工作目录解析的**。实测从仓库外运行主程序时，uv 去找 ``E:\\scripts\\separator_worker.py``
并失败，而主程序拿到的只是「分离 worker 提前退出（returncode=2）」，看不出是路径问题。
``main.py`` 里的 ``metadata_filter.toml`` 早就改成相对 ``__file__`` 定位了，worker
脚本路径当时漏了。

统一走 :func:`repo_file`，这类问题就只会有一个出口。
"""

from __future__ import annotations

from pathlib import Path

#: 仓库根目录。本文件位于 ``<root>/src/karakara/paths.py``。
REPO_ROOT = Path(__file__).resolve().parents[2]


def repo_file(relative: str | Path) -> Path:
    """把仓库内的相对路径解析成绝对路径，并校验它确实存在。

    Args:
        relative: 相对仓库根的路径（如 ``scripts/separator_worker.py``）。
            绝对路径按原样校验。

    Returns:
        绝对路径。

    Raises:
        FileNotFoundError: 目标文件不存在。以非 editable 方式安装时 ``scripts/``
            不会随包分发，这时应当改用 ``--separator-cmd`` 或
            ``KARAKARA_SEPARATOR_CMD`` 指定 worker。
    """
    path = Path(relative)
    resolved = path if path.is_absolute() else REPO_ROOT / path
    if not resolved.is_file():
        raise FileNotFoundError(
            f"仓库内文件不存在: {resolved}（相对仓库根 {REPO_ROOT}）。"
            f"若本项目不是以 editable 方式安装的，scripts/ 不会随包分发，"
            f"请用 --separator-cmd 或 KARAKARA_SEPARATOR_CMD 指定 worker"
        )
    return resolved
