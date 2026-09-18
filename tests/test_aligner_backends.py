"""对齐/分离后端登记表、脚本、CLI 三者之间的一致性。

起因：这个项目踩过一次「客户端默认指向 8787、服务端默认监听 8000」的坑
（见 docs/known-issues.md）。现在后端有多个、各自有默认端口与**语言能力**，
更需要一道机械检查：登记表里的脚本要真实存在、脚本里的默认端口/地址要与登记表
一致、两个后端不能撞端口、登记表声称支持的语言要与服务端真正接受的一致。

这些检查都是**离线**的（读源码、不比网络），所以可以在 CI 里跑。
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from karakara import backends
from karakara.backends import (
    ALIGNER_BACKENDS,
    SEPARATOR_BACKENDS,
    UnsupportedAlignerLanguage,
    ensure_aligner_language,
    resolve_aligner_url,
    resolve_timeout,
)

_ROOT = Path(__file__).resolve().parent.parent


def _module_constant(script: str, name: str) -> Any:
    """把脚本里的一个**字面量**模块级常量取出来（不 import，避免拉起重依赖）。"""
    source = (_ROOT / script).read_text(encoding="utf-8")
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(
            getattr(target, "id", None) == name for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{script} 里没有字面量常量 {name}")


def _language_choices(main_module: Any) -> tuple[str, ...]:
    """从 CLI 解析器里取出 `--aligner-language` 的可选值。"""
    action = next(
        action
        for action in main_module.build_parser()._actions
        if action.dest == "aligner_language"
    )
    return tuple(action.choices or ())


# --------------------------------------------------------------------------
# 脚本与端口
# --------------------------------------------------------------------------


@pytest.mark.parametrize("backend", sorted(ALIGNER_BACKENDS))
def test_every_aligner_backend_script_exists(backend: str) -> None:
    script = ALIGNER_BACKENDS[backend].script
    assert (_ROOT / script).is_file(), f"后端 {backend} 的服务脚本不存在: {script}"


@pytest.mark.parametrize("backend", sorted(SEPARATOR_BACKENDS))
def test_every_separator_backend_script_exists(backend: str) -> None:
    script = SEPARATOR_BACKENDS[backend].script
    assert (_ROOT / script).is_file(), f"后端 {backend} 的 worker 脚本不存在: {script}"


def test_aligner_backend_ports_are_unique() -> None:
    urls = [backend.default_url for backend in ALIGNER_BACKENDS.values()]
    ports = [url.rsplit(":", 1)[-1] for url in urls]
    assert len(ports) == len(set(ports)), f"两个后端撞了同一个端口: {ports}"


@pytest.mark.parametrize("backend", sorted(ALIGNER_BACKENDS))
def test_script_default_port_matches_the_registry(backend: str) -> None:
    """服务脚本里的 `DEFAULT_PORT` 必须等于登记表里的地址端口。"""
    script = ALIGNER_BACKENDS[backend].script
    url = ALIGNER_BACKENDS[backend].default_url
    port = _module_constant(script, "DEFAULT_PORT")
    assert str(port) == url.rsplit(":", 1)[-1], (
        f"{script} 的默认端口与登记表不一致：{port} vs {url}"
    )


@pytest.mark.parametrize("backend", sorted(ALIGNER_BACKENDS))
def test_script_default_host_matches_the_registry(backend: str) -> None:
    """服务脚本的 `DEFAULT_HOST` 必须出现在登记表地址里。

    用 ``127.0.0.1`` 而不是 ``localhost``：Windows 上 ``localhost`` 可能先解析到
    ``::1``，于是客户端连不上只绑了 IPv4 的服务。
    """
    script = ALIGNER_BACKENDS[backend].script
    url = ALIGNER_BACKENDS[backend].default_url
    host = _module_constant(script, "DEFAULT_HOST")
    assert host in url, f"{script} 的默认监听地址({host})没出现在登记表地址里({url})"


# --------------------------------------------------------------------------
# 语言能力
# --------------------------------------------------------------------------


def test_hfa_registry_languages_match_the_server() -> None:
    """HubertFA 登记表里的语言 = 服务端 `_LANGUAGE_ALIASES` 的值域。

    这是最要紧的一条：主程序靠登记表在开跑前拦下不支持的语言，表若比服务端宽松
    就拦不住（服务端会 400，但人声分离已经白跑完）。
    """
    aliases = _module_constant(ALIGNER_BACKENDS["hfa"].script, "_LANGUAGE_ALIASES")
    assert ALIGNER_BACKENDS["hfa"].languages == frozenset(aliases.values())


def test_qwen3_registry_languages_are_all_mappable_by_the_client() -> None:
    """Qwen3 登记表里的语言都必须能被客户端映射成服务端语言名。

    客户端 ``_LANGUAGE_NAMES`` 是「ISO 代码 → 服务名」的唯一映射表；登记表声称支持
    但映射表里没有的代码，会被原样透传给服务端（服务端只认名字）→ 400。
    """
    from karakara.aligner.q3fa.impl import _LANGUAGE_NAMES

    unmappable = ALIGNER_BACKENDS["qwen3"].languages - set(_LANGUAGE_NAMES)
    assert not unmappable, f"登记表里有客户端映射不了的语言: {sorted(unmappable)}"


def test_registry_languages_are_offered_by_the_cli(main_module: Any) -> None:
    """登记表里的语言必须都是 `--aligner-language` 的可选值，否则等于摆设。"""
    offered = set(_language_choices(main_module))
    for name, backend in ALIGNER_BACKENDS.items():
        missing = backend.languages - offered
        assert not missing, f"{name} 声称支持但 CLI 选不到: {sorted(missing)}"


def test_every_cli_language_choice_has_at_least_one_backend(
    main_module: Any,
) -> None:
    """反向检查：CLI 给的每个语言都得有后端能处理（`auto` 除外）。"""
    for language in _language_choices(main_module):
        if language == "auto":
            continue
        assert backends.aligner_backends_for(language), (
            f"--aligner-language {language} 没有任何后端支持"
        )


def test_default_backend_is_registered(main_module: Any) -> None:
    """CLI 的默认后端必须真的在登记表里（help 里的默认值与实现不能脱节）。"""
    parser = main_module.build_parser()
    assert parser.get_default("aligner_backend") in ALIGNER_BACKENDS
    assert parser.get_default("separator_backend") in SEPARATOR_BACKENDS


# --------------------------------------------------------------------------
# ensure_aligner_language
# --------------------------------------------------------------------------


@pytest.mark.parametrize("language", ["zh", "ja", "en"])
def test_ensure_language_accepts_what_hfa_supports(language: str) -> None:
    ensure_aligner_language("hfa", language)  # 不抛就是通过


@pytest.mark.parametrize("language", ["yue", "ko"])
def test_ensure_language_rejects_what_hfa_lacks(language: str) -> None:
    with pytest.raises(UnsupportedAlignerLanguage) as info:
        ensure_aligner_language("hfa", language)
    message = str(info.value)
    # 报错必须**可执行**：说清支持什么、以及换哪个后端能跑。
    assert language in message
    assert "qwen3" in message
    for supported in ALIGNER_BACKENDS["hfa"].languages:
        assert supported in message


@pytest.mark.parametrize("language", ["zh", "ja", "en", "yue", "ko"])
def test_ensure_language_accepts_everything_qwen3_supports(language: str) -> None:
    ensure_aligner_language("qwen3", language)


def test_ensure_language_skips_auto() -> None:
    """``None`` = auto，由整首歌的文本判定（值域只有 ja/zh/en），不该被拦。"""
    ensure_aligner_language("hfa", None)


# --------------------------------------------------------------------------
# 地址与超时
# --------------------------------------------------------------------------


def test_resolve_aligner_url_prefers_the_explicit_url() -> None:
    assert resolve_aligner_url("hfa", "http://example:9999") == "http://example:9999"
    assert resolve_aligner_url("hfa", None) == ALIGNER_BACKENDS["hfa"].default_url
    assert resolve_aligner_url("hfa", "") == ALIGNER_BACKENDS["hfa"].default_url


@pytest.mark.parametrize(
    ("value", "expected"),
    [(120.0, 120.0), (0.0, None), (-1.0, None), (None, None)],
)
def test_resolve_timeout(value: float | None, expected: float | None) -> None:
    """``<=0`` 统一表示「不超时」，因为 requests/Popen 就是这么理解 None 的。"""
    assert resolve_timeout(value) == expected


def test_build_separator_registry_command_uses_the_backend_script() -> None:
    separator = backends.build_separator("audio-separator")
    assert separator.command == [
        "uv",
        "run",
        "--script",
        SEPARATOR_BACKENDS["audio-separator"].script,
    ]


def test_build_separator_explicit_command_wins() -> None:
    separator = backends.build_separator(
        "demucs", command=["C:/some/python.exe", "my_worker.py"]
    )
    assert separator.command == ["C:/some/python.exe", "my_worker.py"]
