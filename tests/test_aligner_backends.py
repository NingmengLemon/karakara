"""对齐后端的登记表与脚本之间的一致性。

起因：这个项目踩过一次「客户端默认指向 8787、服务端默认监听 8000」的坑
（见 docs/known-issues.md）。现在后端有多个、各自有默认端口，更需要一道机械检查：
登记表里的脚本要真实存在、脚本里的默认端口要与登记表一致、两个后端不能撞端口。
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from pathlib import Path
from typing import Any

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_MAIN = _ROOT / "main.py"


def _load_main() -> Any:
    spec = importlib.util.spec_from_file_location("karakara_main_backends", _MAIN)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def main_module() -> Any:
    return _load_main()


def test_every_backend_script_exists(main_module: Any) -> None:
    for backend, (script, _url) in main_module._ALIGNER_BACKENDS.items():
        assert (_ROOT / script).is_file(), f"后端 {backend} 的服务脚本不存在: {script}"


def test_backend_ports_are_unique(main_module: Any) -> None:
    ports = [
        url.rsplit(":", 1)[-1] for _, url in main_module._ALIGNER_BACKENDS.values()
    ]
    assert len(ports) == len(set(ports)), f"两个后端撞了同一个端口: {ports}"


@pytest.mark.parametrize("backend", ["hfa", "qwen3"])
def test_script_default_port_matches_the_registry(
    main_module: Any, backend: str
) -> None:
    """服务脚本里的 `DEFAULT_PORT` 必须等于登记表里的地址端口。"""
    script, url = main_module._ALIGNER_BACKENDS[backend]
    source = (_ROOT / script).read_text(encoding="utf-8")
    match = re.search(r"^DEFAULT_PORT\s*=\s*(\d+)", source, re.MULTILINE)
    assert match, f"{script} 里没有 DEFAULT_PORT"
    assert match.group(1) == url.rsplit(":", 1)[-1], (
        f"{script} 的默认端口与 main.py 登记表不一致：{match.group(1)} vs {url}"
    )


@pytest.mark.parametrize("backend", ["hfa", "qwen3"])
def test_script_default_host_is_localhost(main_module: Any, backend: str) -> None:
    script, url = main_module._ALIGNER_BACKENDS[backend]
    source = (_ROOT / script).read_text(encoding="utf-8")
    match = re.search(r'^DEFAULT_HOST\s*=\s*"([^"]+)"', source, re.MULTILINE)
    assert match, f"{script} 里没有 DEFAULT_HOST"
    assert match.group(1) in url, (
        f"{script} 的默认监听地址({match.group(1)})没出现在登记表地址里({url})"
    )


def test_resolve_aligner_url_prefers_the_explicit_url(
    main_module: Any,
) -> None:
    explicit = argparse.Namespace(
        aligner_backend="hfa", aligner_url="http://example:9999"
    )
    assert main_module.resolve_aligner_url(explicit) == "http://example:9999"

    by_default = argparse.Namespace(aligner_backend="hfa", aligner_url=None)
    assert (
        main_module.resolve_aligner_url(by_default)
        == (main_module._ALIGNER_BACKENDS["hfa"][1])
    )


def test_default_backend_is_registered(main_module: Any) -> None:
    """CLI 的默认后端必须真的在登记表里（help 里的默认值与实现不能脱节）。"""
    parser = main_module.build_parser()
    default = parser.get_default("aligner_backend")
    assert default in main_module._ALIGNER_BACKENDS
