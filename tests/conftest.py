"""共用夹具。

``main.py`` 位于仓库根目录，**不是** ``karakara`` 包的一部分（它是 CLI 入口），
所以 ``import main`` 在 pytest 的默认导入模式下找不到它——只能按路径加载。
以前每个测试文件各写一份加载代码，现在收在这里。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load_main() -> Any:
    spec = importlib.util.spec_from_file_location("karakara_cli_main", ROOT / "main.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def main_module() -> Any:
    """已加载的 ``main.py`` 模块对象。"""
    return _load_main()
