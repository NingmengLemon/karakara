"""HubertFA 对齐服务的逻辑测试（不需要模型、不需要 GPU）。

服务脚本跑在自己的环境里（PEP 723），所以这里用 stub 顶掉它那份重量级依赖
（`fastapi`、`pykakasi`），只测**纯逻辑**——而那正是最该盯的部分：

* 日文 kana → 音节键的规则（拗音合并、促音丢弃、长音重复前元音、助词发音）；
* 文本 → 片段（G2P）以及**按词典过滤**（上游 G2P 会静默丢词，不过滤就会整体错位）；
* TextGrid → 原文片段的聚合（单元的 `text` 必须是**原行文本的子串**，否则主程序的
  `_build_aligned_content` 映射不上）；
* `/align` 的响应契约（单文件 → 对象，多文件 → 数组）与不支持语言的 400。
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

import pytest

_SERVER_PATH = (
    Path(__file__).resolve().parent.parent / "scripts" / "hubertfa_aligner_server.py"
)

#: 伪 pykakasi 的转换结果由测试注入（`install_fake_kakasi`）。
_FAKE_CONVERSION: dict[str, list[dict[str, str]]] = {}


def _make_fake_kakasi() -> types.ModuleType:
    def kakasi() -> Any:
        class _K:
            @staticmethod
            def convert(text: str) -> list[dict[str, str]]:
                return _FAKE_CONVERSION.get(text, [{"orig": text, "hira": text}])

        return _K()

    return _stub_module("pykakasi", kakasi=kakasi)


def _stub_module(name: str, **attrs: Any) -> types.ModuleType:
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def _load_server(monkeypatch: pytest.MonkeyPatch) -> Any:
    """导入真实的服务脚本，stub 掉 fastapi / pykakasi（pydantic、pypinyin 用真的）。"""
    stubs: dict[str, types.ModuleType] = {}
    if "fastapi" not in sys.modules:

        class _FakeHTTPException(Exception):
            def __init__(self, status_code: int = 500, detail: str = "") -> None:
                super().__init__(detail)
                self.status_code = status_code
                self.detail = detail

        class _FakeApp:
            def post(self, *_a: Any, **_k: Any) -> Any:
                return lambda fn: fn

            def get(self, *_a: Any, **_k: Any) -> Any:
                return lambda fn: fn

        stubs["fastapi"] = _stub_module(
            "fastapi",
            FastAPI=lambda **_k: _FakeApp(),
            File=lambda default=None, **_k: default,
            Form=lambda default=None, **_k: default,
            HTTPException=_FakeHTTPException,
            UploadFile=object,
        )
    if "pykakasi" not in sys.modules:
        stubs["pykakasi"] = _make_fake_kakasi()

    sys.modules.update(stubs)
    name = "_hubertfa_server_under_test"
    try:
        spec = importlib.util.spec_from_file_location(name, _SERVER_PATH)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    finally:
        for key in stubs:
            sys.modules.pop(key, None)
    return module


@pytest.fixture
def server(monkeypatch: pytest.MonkeyPatch) -> Any:
    _FAKE_CONVERSION.clear()
    return _load_server(monkeypatch)


def install_conversion(text: str, pairs: list[tuple[str, str]]) -> None:
    _FAKE_CONVERSION[text] = [
        {"orig": original, "hira": hira} for original, hira in pairs
    ]


# --------------------------------------------------------------------------
# 1. kana → 音节键的规则
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kana", "expected"),
    [
        ("そら", ["so", "ra"]),
        ("きゃく", ["kya", "ku"]),  # 拗音合并
        ("しゃしん", ["sha", "shi", "n"]),  # ん → n
        ("がっこう", ["ga", "ko", "u"]),  # 促音被丢弃（上游 ja/cl 的 bug）
        (
            "コーヒー",
            ["ko", "o", "hi", "i"],
        ),  # 长音 → 重复前一个元音（且片假名要归一化）
        ("を", ["wo"]),
        ("ちっちゃい", ["chi", "cha", "i"]),
        ("じゃあ", ["ja", "a"]),
        # 外来语的小写元音 → 与前一音节合成一个モーラ
        ("ヴァイオリン", ["va", "i", "o", "ri", "n"]),
        ("ファイト", ["fa", "i", "to"]),
        ("パーティー", ["pa", "a", "ti", "i"]),
        ("ウィスキー", ["wi", "su", "ki", "i"]),
        ("シェア", ["she", "a"]),
    ],
)
def test_kana_to_syllables(server: Any, kana: str, expected: list[str]) -> None:
    assert server.kana_to_syllables(kana) == expected


def test_kana_to_syllables_skips_non_kana(server: Any) -> None:
    assert server.kana_to_syllables("「あ」") == ["a"]


def test_kana_syllables_all_exist_in_the_shipped_dictionary(server: Any) -> None:
    """规则产出的键必须都在真实词典里（词典缺键会被 G2P 静默丢弃）。"""
    dictionary = (
        Path(__file__).resolve().parent.parent
        / "models"
        / "aligner"
        / "HubertFA"
        / "japanese_dict_full.txt"
    )
    if not dictionary.is_file():
        pytest.skip("本地没有 HubertFA 词典（models/ 不进仓库）")
    keys = {
        line.split("\t")[0].strip()
        for line in dictionary.read_text(encoding="utf-8").strip().split("\n")
        if "\t" in line
    }

    sample = "きゃくしゃしんがっこうコーヒーをちっちゃいじゃあヴァイオリンファイトパーティーウィスキーシェア"
    produced = server.kana_to_syllables(sample)

    assert produced, "样例应当产出音节"
    assert set(produced) <= keys, f"词典里没有的键: {sorted(set(produced) - keys)}"


# --------------------------------------------------------------------------
# 2. 文本 → 片段（G2P）
# --------------------------------------------------------------------------


def test_split_ja_keeps_original_substrings(server: Any) -> None:
    install_conversion("空の音", [("空", "そら"), ("の", "の"), ("音", "おと")])

    chunks = server.split_ja("空の音")

    assert [chunk.text for chunk in chunks] == ["空", "の", "音"]
    assert chunks[0].syllables == ["so", "ra"]
    assert chunks[2].syllables == ["o", "to"]
    # 单元的 text 必须是原行文本的子串（主程序靠 text.find 定位）
    for chunk in chunks:
        assert chunk.text in "空の音"


def test_split_ja_uses_sung_pronunciation_for_particles(server: Any) -> None:
    install_conversion("花は", [("花", "はな"), ("は", "は")])

    chunks = server.split_ja("花は")

    assert chunks[-1].syllables == ["wa"]  # 助词 は 唱作 wa


def test_split_zh_is_per_character(server: Any) -> None:
    chunks = server.split_zh("风一下子")

    assert [chunk.text for chunk in chunks] == ["风", "一", "下", "子"]
    assert all(len(chunk.syllables) == 1 for chunk in chunks)
    assert chunks[0].syllables == ["feng"]


def test_split_en_is_per_word(server: Any) -> None:
    chunks = server.split_en("Hello, World!")

    assert [chunk.text for chunk in chunks] == ["Hello", "World"]
    assert [chunk.syllables for chunk in chunks] == [["hello"], ["world"]]


def test_g2p_drops_chunks_missing_from_the_dictionary(
    server: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """词典里没有的音节必须**在送模型前**剔掉，否则音节数对不上、整体错位。"""
    (tmp_path / "japanese_dict_full.txt").write_text(
        "so\tso\nra\tra\nno\tno\n", encoding="utf-8"
    )
    monkeypatch.setattr(server, "HUBERTFA_ROOT", tmp_path)
    monkeypatch.setattr(server, "_DICTIONARY_KEYS", {})
    install_conversion("空の音", [("空", "そら"), ("の", "の"), ("音", "おと")])

    chunks = server.g2p("空の音", "ja")

    # 「音」= o + to 两个键都不在词典里 → 整块被丢弃；空/の 保留
    assert [chunk.text for chunk in chunks] == ["空", "の"]


# --------------------------------------------------------------------------
# 3. TextGrid → 原文片段的聚合
# --------------------------------------------------------------------------

TEXTGRID = """File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = 4.0
tiers? <exists>
size = 2
item []:
\titem [1]:
\t\tclass = "IntervalTier"
\t\tname = "words"
\t\txmin = 0.0
\t\txmax = 4.0
\t\tintervals: size = 6
\t\t\tintervals [1]:
\t\t\t\txmin = 0.0
\t\t\t\txmax = 0.3
\t\t\t\ttext = "SP"
\t\t\tintervals [2]:
\t\t\t\txmin = 0.3
\t\t\t\txmax = 1.0
\t\t\t\ttext = "so"
\t\t\tintervals [3]:
\t\t\t\txmin = 1.0
\t\t\t\txmax = 1.4
\t\t\t\ttext = "ra"
\t\t\tintervals [4]:
\t\t\t\txmin = 1.4
\t\t\t\txmax = 1.6
\t\t\t\ttext = "AP"
\t\t\tintervals [5]:
\t\t\t\txmin = 1.6
\t\t\t\txmax = 2.2
\t\t\t\ttext = "no"
\t\t\tintervals [6]:
\t\t\t\txmin = 2.2
\t\t\t\txmax = 4.0
\t\t\t\ttext = "o"
\t\titem [2]:
\t\tclass = "IntervalTier"
\t\tname = "phones"
\t\txmin = 0.0
\t\txmax = 4.0
\t\tintervals: size = 1
"""


def test_parse_textgrid_reads_intervals(server: Any, tmp_path: Path) -> None:
    path = tmp_path / "x.TextGrid"
    path.write_text(TEXTGRID, encoding="utf-8")

    tiers = server.parse_textgrid(path)

    assert [label for _, _, label in tiers["words"]][:3] == ["SP", "so", "ra"]
    assert tiers["words"][2] == (1.0, 1.4, "ra")


def test_build_units_aggregates_morae_into_chunks(server: Any, tmp_path: Path) -> None:
    path = tmp_path / "x.TextGrid"
    path.write_text(TEXTGRID, encoding="utf-8")
    tiers = server.parse_textgrid(path)

    chunks = [
        server.G2PChunk(text="空", syllables=["so", "ra"]),
        server.G2PChunk(text="の", syllables=["no"]),
        server.G2PChunk(text="音", syllables=["o"]),
    ]

    units = server.build_units(chunks, tiers)

    # 空 = so+ra → 覆盖两个モーラ；静音/呼吸（SP/AP）不占位
    assert units == [
        ("空", 300.0, 1400.0),
        ("の", 1600.0, 2200.0),
        ("音", 2200.0, 4000.0),
    ]


# --------------------------------------------------------------------------
# 4. /align 契约
# --------------------------------------------------------------------------


class _FakeUpload:
    def __init__(self, payload: bytes = b"RIFF") -> None:
        import io

        self.filename = "clip.wav"
        self.file = io.BytesIO(payload)


def test_align_returns_object_for_a_single_file(
    server: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(server, "align_one", lambda *_a, **_k: [("空", 300.0, 1400.0)])

    result = server.align(audio=_FakeUpload(), text="空", language="Japanese")

    assert not isinstance(result, list)
    assert [w.text for w in result.words] == ["空"]
    assert result.words[0].start_time == 0.3


def test_align_returns_array_for_multiple_files(
    server: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(server, "align_one", lambda *_a, **_k: [("a", 0.0, 100.0)])

    result = server.align(
        audio=[_FakeUpload(), _FakeUpload()], text=["a", "b"], language="Chinese"
    )

    assert isinstance(result, list)
    assert len(result) == 2


def test_align_rejects_unsupported_language(
    server: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(server, "align_one", lambda *_a, **_k: [])

    with pytest.raises(Exception, match="不支持的语言"):
        server.align(audio=_FakeUpload(), text="안녕", language="Korean")


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("Chinese", "zh"),
        ("chinese", "zh"),
        ("zh", "zh"),
        ("Japanese", "ja"),
        ("JA", "ja"),
        ("English", "en"),
    ],
)
def test_language_aliases(server: Any, given: str, expected: str) -> None:
    assert server._LANGUAGE_ALIASES[given.strip().lower()] == expected


def test_endpoint_is_sync(server: Any) -> None:
    """与 qwen 服务同样的理由：阻塞推理不能跑在事件循环里。"""
    import inspect

    assert not inspect.iscoroutinefunction(server.align)
