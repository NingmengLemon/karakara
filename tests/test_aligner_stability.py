"""扰动稳定性检查的度量口径（已知答案自检）。

这些函数是 `scripts/check_aligner_stability.py` 的判据本身，所以必须先用**手算得出
答案**的样本验证一遍——否则后面所有"稳定/不稳定"的结论都建在不明地基上。

`pad` 条件的漂移**要先扣掉已知平移**、`drop1`/`add1` 的漂移要按**单元文本的公共前缀**
切成前后两段，这两条最容易写错，各有专门的用例。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

_SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent / "scripts" / "check_aligner_stability.py"
)


def _load() -> Any:
    """按路径加载脚本（`scripts/` 不在包里，`import` 找不到它）。"""
    spec = importlib.util.spec_from_file_location("_aligner_stability", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def script() -> Any:
    return _load()


# --------------------------------------------------------------------------
# 边界与覆盖率
# --------------------------------------------------------------------------


def test_boundaries_are_unit_starts_plus_the_final_end(script: Any) -> None:
    units = [("a", 0, 100), ("b", 100, 250), ("c", 250, 400)]
    assert script.boundaries_of(units) == [0, 100, 250, 400]


def test_boundaries_of_empty_is_empty(script: Any) -> None:
    assert script.boundaries_of([]) == []


def test_coverage_is_span_sum_over_segment(script: Any) -> None:
    # 单元跨 0-100 与 100-250，共 250ms；片段 500ms
    assert script.coverage([0, 100, 250], 500.0) == 0.5


@pytest.mark.parametrize(
    ("boundaries", "segment_ms"),
    [([], 500.0), ([0], 500.0), ([0, 100], 0.0), ([0, 100], -1.0)],
)
def test_coverage_degrades_to_zero(
    script: Any, boundaries: list[int], segment_ms: float
) -> None:
    assert script.coverage(boundaries, segment_ms) == 0.0


# --------------------------------------------------------------------------
# 行内漂移：必须扣掉已知平移
# --------------------------------------------------------------------------


def test_interior_drift_drops_the_first_and_last_boundary(script: Any) -> None:
    """首尾在结构上被片段边界钉住，不该混进稳定性里。"""
    base = [0, 100, 200, 300]
    # 首尾故意差 50ms：不该出现在结果里
    other = [50, 110, 190, 350]
    assert script.interior_drift(base, other, 0) == [10, 10]


def test_interior_drift_subtracts_the_padding_shift(script: Any) -> None:
    """+100ms 的片段里同一条边界理应出现在 base+100ms，扣掉后漂移应为 0。"""
    base = [0, 100, 200, 300]
    padded = [100, 200, 300, 400]
    assert script.interior_drift(base, padded, 100) == [0, 0]
    # 不扣平移就会量到"我加的 padding"
    assert script.interior_drift(base, padded, 0) == [100, 100]


def test_interior_drift_requires_matching_unit_counts(script: Any) -> None:
    assert script.interior_drift([0, 100, 200], [0, 100, 150, 200], 0) == []


def test_interior_drift_requires_at_least_three_boundaries(script: Any) -> None:
    """只有两条边界时没有"行内"可言。"""
    assert script.interior_drift([0, 100], [0, 110], 0) == []


# --------------------------------------------------------------------------
# 文本扰动：按单元文本的公共前缀切分
# --------------------------------------------------------------------------


def test_local_drift_drop_splits_at_the_perturbation(script: Any) -> None:
    """删掉中间那个单元：前缀不动，后缀按索引偏移对齐。"""
    base_texts = ["a", "b", "c", "d"]
    base = [0, 10, 20, 30, 40]
    other_texts = ["a", "c", "d"]  # b 被删掉
    other = [0, 10, 25, 40]

    prefix, suffix = script.local_drift(base, base_texts, other, other_texts, -1)

    assert prefix == []
    # other 的 c ↔ base 的 c（索引 2）：|10-20|；other 的 d ↔ base 的 d（索引 3）：|25-30|
    assert suffix == [10.0, 5.0]


def test_local_drift_add_splits_at_the_perturbation(script: Any) -> None:
    """插入一个单元：前缀不动，后缀按索引偏移对齐。"""
    base_texts = ["a", "b", "c"]
    base = [0, 10, 20, 30]
    other_texts = ["a", "b", "b", "c"]
    other = [0, 10, 15, 20, 30]

    prefix, suffix = script.local_drift(base, base_texts, other, other_texts, 1)

    # 前缀 = 公共前缀 ["a","b"] 之内、且排除索引 0 的边界 → 只有索引 1
    assert prefix == [0.0]
    # other 的第二个 b（索引 2）↔ base 的 b（索引 1）；other 的 c（索引 3）↔ base 的 c（索引 2）
    assert suffix == [5.0, 0.0]


def test_local_drift_prefix_is_unchanged_when_the_aligner_resyncs(script: Any) -> None:
    """前缀完全一致时漂移必须是 0（这正是实测里日文上看到的现象）。"""
    base_texts = ["a", "b", "c", "d", "e"]
    base = [0, 10, 20, 30, 40, 50]
    other_texts = ["a", "b", "c", "e"]
    other = [0, 10, 20, 30, 50]

    prefix, _suffix = script.local_drift(base, base_texts, other, other_texts, -1)

    assert prefix == [0.0, 0.0]


# --------------------------------------------------------------------------
# 文本扰动本身
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [("abc", "ac"), ("ab", "a"), ("a", None), ("", None), ("あいう", "あう")],
)
def test_drop_char_removes_the_middle_content_char(
    script: Any, text: str, expected: str | None
) -> None:
    assert script.drop_char(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("abc", "abbc"),
        ("a", "aa"),
        ("", None),
        # 两个内容字符时取 `indexes[len // 2]` = 第二个 → 重复的是「い」
        ("あい", "あいい"),
    ],
)
def test_dup_char_duplicates_the_middle_content_char(
    script: Any, text: str, expected: str | None
) -> None:
    assert script.dup_char(text) == expected


def test_drop_and_dup_ignore_whitespace_when_picking_the_middle(script: Any) -> None:
    """扰动点是中间那个**非空白**字符，而空格在日文歌词里很常见（全角空格）。

    ``"a b"`` 的内容字符是索引 0 与 2，所以取索引 2 的 ``b``：删掉它剩下 ``"a "``。
    若不忽略空白，取的是 ``len("a b") // 2 == 1`` 即那个空格，结果会变成 ``"ab"``
    ——这个用例正是用来区分这两种实现的。
    """
    assert script.drop_char("a b") == "a "
    assert script.dup_char("a b") == "a bb"


# --------------------------------------------------------------------------
# 汇总
# --------------------------------------------------------------------------


def test_summarize_reports_median_p90_max(script: Any) -> None:
    summary = script.summarize([float(v) for v in range(1, 11)])
    assert summary == {"n": 10.0, "median": 5.5, "p90": 10.0, "max": 10.0}


def test_summarize_of_empty_is_none(script: Any) -> None:
    assert script.summarize([]) is None


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def test_help_mentions_the_documented_criteria(
    script: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as info:
        script.main(["--help"])
    assert info.value.code == 0
    # 判据必须写在 help 里：这是脚本唯一的"事先定好"的地方
    assert "扰动稳定性" in capsys.readouterr().out


def test_cli_rejects_a_language_the_backend_lacks(
    script: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    """与主程序同一道闸门：hfa 不支持 ko，必须在开跑前拦下。"""
    with pytest.raises(SystemExit) as info:
        script.main(
            [
                "--lrc",
                "a.lrc",
                "--audio",
                "a.flac",
                "--aligner-backend",
                "hfa",
                "--language",
                "ko",
            ]
        )
    assert info.value.code == 2
    assert "不支持语言" in capsys.readouterr().err


def test_cli_requires_lrc_and_audio(script: Any) -> None:
    with pytest.raises(SystemExit) as info:
        script.main([])
    assert info.value.code == 2
