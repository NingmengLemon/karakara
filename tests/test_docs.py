"""文档的机械检查：相对链接必须指向真实存在的文件。

起因：2026-09-21 把 `docs/` 拆成 `current/` / `records/` / `archive/` 三层，一次性移动了
七八个文件。这类重构靠人眼核对链接必然漏（`grep` 只能找到写死的路径，找不到指向已删除
文件的链接），所以做成闸门。

只查**相对链接**：外部 URL、纯锚点、以及 `mailto:` 之类不查。
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
_LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
_SKIP_PREFIXES = ("http://", "https://", "mailto:", "#")


def _markdown_files() -> list[Path]:
    files = [ROOT / "README.md"]
    files.extend(sorted((ROOT / "docs").rglob("*.md")))
    return [path for path in files if path.is_file()]


def _relative_links(path: Path) -> list[tuple[int, str]]:
    found: list[tuple[int, str]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        for target in _LINK.findall(line):
            if target.startswith(_SKIP_PREFIXES):
                continue
            found.append((number, target.split("#", 1)[0]))
    return [(number, target) for number, target in found if target]


def test_every_relative_doc_link_resolves() -> None:
    broken: list[str] = []
    for path in _markdown_files():
        for number, target in _relative_links(path):
            resolved = (path.parent / target).resolve()
            if not resolved.exists():
                broken.append(f"{path.relative_to(ROOT)}:{number} -> {target}")

    assert not broken, "文档里有死链：\n" + "\n".join(broken)


def test_docs_root_has_the_three_layers() -> None:
    """三层结构是这次重构的约定，别在后续改动里悄悄退回平铺。"""
    for name in ("current", "records", "archive"):
        directory = ROOT / "docs" / name
        assert directory.is_dir(), f"缺少文档层: docs/{name}"
        assert list(directory.glob("*.md")), f"docs/{name}/ 里没有任何文档"


@pytest.mark.parametrize(
    "stale", ["aligner-backends.md", "known-issues.md", "benchmarks.md"]
)
def test_old_flat_doc_paths_are_gone(stale: str) -> None:
    """旧路径不能复活：内容已经拆到 current/records/archive 里了。"""
    assert not (ROOT / "docs" / stale).exists()
