# /// script
# requires-python = "==3.12.*"
# dependencies = [
#     "click",
#     "fastapi>=0.135.3",
#     "librosa<0.10.0",
#     "matplotlib",
#     "numpy<2",
#     "onnxruntime>=1.19",
#     "pydantic>=2.12.4",
#     "pykakasi>=2.3.0",
#     "pypinyin>=0.55.0",
#     "python-multipart>=0.0.20",
#     "PyYAML",
#     "setuptools<81",
#     "soundfile>=0.13.1",
#     "textgrid~=1.5",
#     "tqdm",
#     "uvicorn>=0.35.0",
# ]
# ///
"""HubertFA 对齐服务：与 `qwen3aligner_server.py` **同一套 `/align` 契约**。

HubertFA 是**为歌声训练**的强制对齐器（SOFA 血统，ONNX，10ms 帧）。实机验证见
`docs/aligner-backends.md` §9：在 Qwen3-ForcedAligner 把歌词压扁的行上，它是对的
（逐行覆盖率 84–94% vs Qwen 的 6.7%–93.3%；零长度单元 1.9% vs 41.8%）。

为什么需要这一层而不是直接调 CLI：HubertFA 是**文件级**的（每个 wav 配一个 `.lab`
音素序列），而本项目的契约是「音频片段 + 行文本 → 该行的逐单元时间」。所以本服务负责

1. **G2P**：行文本 → 音节序列（zh 用 `pypinyin`，ja 用 `pykakasi` + 音节规则），并**记住
   每个音节来自哪一段原文**；
2. 跑模型（复用上游 `onnx_infer.py` 的代码，见 `HUBERTFA_ROOT`）；
3. **把音素/モーラ 聚合回原文的片段**——返回单元的 `text` 必须是原行文本的**子串序列**，
   否则主程序的 `_build_aligned_content`（靠 `text.find` 定位）会映射失败。

契约（与 Qwen 服务一致）：

* 单文件 → 对象 `{"words": [{"text", "start_time", "end_time"}]}`；多文件 → 数组。
* `language` 接受 Qwen 那套名字（Chinese/Japanese/English）**也**接受 `zh`/`ja`/`en`。
"""

from __future__ import annotations

import argparse
import logging
import re
import shutil
import sys
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from pykakasi import kakasi
from pypinyin import Style
from pypinyin import pinyin as pypinyin_pinyin

logger = logging.getLogger("karakara.hubertfa_server")

#: 项目根目录（本脚本在 <root>/scripts/ 下）。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
#: 模型与上游代码位置（`models/` 不进仓库；来源与下载命令见该目录的 SOURCE.md）。
HUBERTFA_ROOT = PROJECT_ROOT / "models" / "aligner" / "HubertFA"
DEFAULT_HOST = "127.0.0.1"
#: 与 qwen 服务的 8787 区分开，便于同时起两个后端做 A/B。
DEFAULT_PORT = 8788

#: 与 Qwen 客户端一致的默认语言名 → HubertFA 的词典 key。
_LANGUAGE_ALIASES = {
    "chinese": "zh",
    "zh": "zh",
    "mandarin": "zh",
    "japanese": "ja",
    "ja": "ja",
    "jp": "ja",
    "english": "en",
    "en": "en",
}

app = FastAPI(title="HubertFA Aligner Service", version="1.0.0")


class AlignedWord(BaseModel):
    text: str
    start_time: float
    end_time: float


class AlignResponse(BaseModel):
    words: list[AlignedWord]


# --------------------------------------------------------------------------
# G2P：文本 → 音节序列 + 每个音节来自原文的哪一段
# --------------------------------------------------------------------------


@dataclass
class G2PChunk:
    """一段原文（连续子串）与它对应的音节序列。"""

    text: str
    syllables: list[str]


#: 五十音 → HubertFA 的日文词典键（`japanese_dict_full.txt`）。
_KANA_BASE: dict[str, str] = {
    "あ": "a",
    "い": "i",
    "う": "u",
    "え": "e",
    "お": "o",
    "か": "ka",
    "き": "ki",
    "く": "ku",
    "け": "ke",
    "こ": "ko",
    "が": "ga",
    "ぎ": "gi",
    "ぐ": "gu",
    "げ": "ge",
    "ご": "go",
    "さ": "sa",
    "し": "shi",
    "す": "su",
    "せ": "se",
    "そ": "so",
    "ざ": "za",
    "じ": "ji",
    "ず": "zu",
    "ぜ": "ze",
    "ぞ": "zo",
    "た": "ta",
    "ち": "chi",
    "つ": "tsu",
    "て": "te",
    "と": "to",
    "だ": "da",
    "ぢ": "ji",
    "づ": "zu",
    "で": "de",
    "ど": "do",
    "な": "na",
    "に": "ni",
    "ぬ": "nu",
    "ね": "ne",
    "の": "no",
    "は": "ha",
    "ひ": "hi",
    "ふ": "fu",
    "へ": "he",
    "ほ": "ho",
    "ば": "ba",
    "び": "bi",
    "ぶ": "bu",
    "べ": "be",
    "ぼ": "bo",
    "ぱ": "pa",
    "ぴ": "pi",
    "ぷ": "pu",
    "ぺ": "pe",
    "ぽ": "po",
    "ま": "ma",
    "み": "mi",
    "む": "mu",
    "め": "me",
    "も": "mo",
    "や": "ya",
    "ゆ": "yu",
    "よ": "yo",
    "ら": "ra",
    "り": "ri",
    "る": "ru",
    "れ": "re",
    "ろ": "ro",
    "わ": "wa",
    "ゐ": "i",
    "ゑ": "e",
    "を": "wo",
    "ゔ": "vu",
    "ぁ": "a",
    "ぃ": "i",
    "ぅ": "u",
    "ぇ": "e",
    "ぉ": "o",
    "ゎ": "wa",
    "ゕ": "ka",
    "ゖ": "ke",
    "ん": "n",
}
_YOON: dict[str, str] = {"ゃ": "a", "ゅ": "u", "ょ": "o"}
#: 小写元音：与前一音节合成**一个**モーラ（外来语：ヴァ=va、ファ=fa、ティ=ti、ウィ=wi…）。
_SMALL_VOWEL: dict[str, str] = {"ぁ": "a", "ぃ": "i", "ぅ": "u", "ぇ": "e", "ぉ": "o"}
#: 这些声母的拗音不再写 ``y``（ヘボン式：sha / cha / ja）。
_YOON_NO_Y = {"sh", "ch", "j"}
_VOWELS = "aiueo"


def _to_hiragana(text: str) -> str:
    """片假名 → 平假名（同区偏移 0x60；长音符 ``ー`` 原样保留）。"""
    out: list[str] = []
    for ch in text:
        code = ord(ch)
        if 0x30A1 <= code <= 0x30F6:
            out.append(chr(code - 0x60))
        else:
            out.append(ch)
    return "".join(out)


def kana_to_syllables(kana: str) -> list[str]:
    """（平/片假名）读音 → HubertFA 的音节键序列。

    ``っ``（促音）**刻意丢弃**：上游 `tools/g2p.py` 的 ``BaseG2P.__call__`` 会把除 ``SP``
    以外的音素一律加上 ``{language}/`` 前缀，而 ``cl`` 在 vocab 里属于无前缀的静音组，
    于是会拼出不存在的 ``ja/cl`` 并 KeyError（上游 bug，本文件不改它的代码）。
    """
    out: list[str] = []
    index = 0
    kana = _to_hiragana(kana)
    while index < len(kana):
        ch = kana[index]
        if ch in _YOON:
            vowel = _YOON[ch]
            if out and out[-1] and out[-1][-1] in _VOWELS:
                head = out[-1][:-1]
                out[-1] = head + (vowel if head in _YOON_NO_Y else "y" + vowel)
            else:
                out.append(vowel)
        elif ch in _SMALL_VOWEL:
            # 外来语的小写元音：与前一音节合并（ヴァ→va、ティ→ti、ウィ→wi）
            vowel = _SMALL_VOWEL[ch]
            if out and out[-1] and out[-1][-1] in _VOWELS:
                head = out[-1][:-1]
                if head:
                    out[-1] = head + vowel
                else:
                    out[-1] = "w" + vowel if vowel in "ieo" else vowel
            else:
                out.append(vowel)
        elif ch in ("っ", "ッ"):
            pass
        elif ch in ("ー", "―", "‐", "‑", "−"):
            if out and out[-1] and out[-1][-1] in _VOWELS:
                out.append(out[-1][-1])  # 长音 → 重复前一个元音
        elif ch in _KANA_BASE:
            out.append(_KANA_BASE[ch])
        index += 1
    return out


def split_ja(text: str) -> list[G2PChunk]:
    """日文：按 pykakasi 的分块给出「原文片段 → 音节」，片段是原文的连续子串。"""
    chunks: list[G2PChunk] = []
    for item in kakasi().convert(text):
        original = item["orig"]
        hira = item["hira"]
        # 助词的歌唱发音（单独成块的 は/へ/を）
        if len(hira) == 1 and hira in ("は", "へ", "を"):
            hira = {"は": "わ", "へ": "え", "を": "お"}[hira]
        syllables = kana_to_syllables(hira)
        if original and syllables:
            chunks.append(G2PChunk(text=original, syllables=syllables))
    return chunks


def split_zh(text: str) -> list[G2PChunk]:
    """中文：逐字给拼音（`pypinyin` 的逐字结果天然带原文位置）。"""
    chunks: list[G2PChunk] = []
    for char, readings in zip(
        text, pypinyin_pinyin(text, style=Style.NORMAL), strict=False
    ):
        if not char.strip():
            continue
        syllable = readings[0] if readings else ""
        if not syllable:
            continue
        chunks.append(G2PChunk(text=char, syllables=[syllable]))
    return chunks


def split_en(text: str) -> list[G2PChunk]:
    """英文：按词切；HubertFA 的英文词典（CMUdict）键是单词本身。"""
    chunks: list[G2PChunk] = []
    for match in re.finditer(r"[A-Za-z']+", text):
        chunks.append(G2PChunk(text=match.group(0), syllables=[match.group(0).lower()]))
    return chunks


def g2p(text: str, language: str) -> list[G2PChunk]:
    """文本 → 片段列表（**已按词典过滤**）。

    必须过滤：上游 G2P 对不在词典里的词只打一条 warning 然后**丢弃**，那样送进模型的
    音节数就与我这里的片段对不上，后面「按顺序把音节归回片段」会整体错位。宁可在这里
    明确丢掉（并记日志），也不要在产物里给出错位的时间。
    """
    if language == "ja":
        chunks = split_ja(text)
    elif language == "zh":
        chunks = split_zh(text)
    elif language == "en":
        chunks = split_en(text)
    else:
        raise HTTPException(
            status_code=400, detail=f"HubertFA 不支持的语言: {language}"
        )

    keys = _dictionary_keys(language)
    kept: list[G2PChunk] = []
    dropped: list[str] = []
    for chunk in chunks:
        if all(syllable in keys for syllable in chunk.syllables):
            kept.append(chunk)
        else:
            dropped.append(chunk.text)
    if dropped:
        logger.warning(
            "以下片段不在 HubertFA 的 %s 词典里，已跳过（不会出现在结果中）: %s",
            language,
            dropped[:10],
        )
    return kept


# --------------------------------------------------------------------------
# TextGrid → 行文本的单元
# --------------------------------------------------------------------------


def parse_textgrid(path: Path) -> dict[str, list[tuple[float, float, str]]]:
    """极简 TextGrid 读取：``{tier: [(xmin_s, xmax_s, text)]}``。"""
    text = path.read_text(encoding="utf-8")
    tiers: dict[str, list[tuple[float, float, str]]] = {}
    for block in re.split(r"item \[\d+\]:", text)[1:]:
        name = re.search(r'name = "([^"]+)"', block)
        if not name:
            continue
        intervals: list[tuple[float, float, str]] = []
        for item in re.split(r"intervals \[\d+\]:", block)[1:]:
            xmin = re.search(r"xmin = ([\d.eE+-]+)", item)
            xmax = re.search(r"xmax = ([\d.eE+-]+)", item)
            label = re.search(r'text = "(.*)"', item)
            if xmin and xmax:
                intervals.append(
                    (
                        float(xmin.group(1)),
                        float(xmax.group(1)),
                        label.group(1) if label else "",
                    )
                )
        tiers[name.group(1)] = intervals
    return tiers


def build_units(
    chunks: list[G2PChunk], tiers: dict[str, list[tuple[float, float, str]]]
) -> list[tuple[str, float, float]]:
    """把音节/モーラ的时间按顺序聚合回 G2P 片段，返回 ``(原文片段, 起点ms, 终点ms)``。

    TextGrid 的 words 层里，非静音（``SP``/``AP``/``EP``）区间按**输入顺序**对应 `.lab`
    里送进去的音节序列，所以这里用「按顺序消耗」的方式把音节归回片段——多音节的片段
    （``空`` = ``so``+``ra``）拿到覆盖全部音节的跨度。
    """
    words = [
        (xmin * 1000, xmax * 1000, label)
        for xmin, xmax, label in tiers.get("words", [])
        if label not in {"SP", "AP", "EP", "sil", ""}
    ]
    units: list[tuple[str, float, float]] = []
    cursor = 0
    for chunk in chunks:
        wanted = len(chunk.syllables)
        taken = words[cursor : cursor + wanted]
        cursor += wanted
        if not taken:
            continue
        units.append((chunk.text, taken[0][0], taken[-1][1]))
    return units


# --------------------------------------------------------------------------
# 服务
# --------------------------------------------------------------------------

_inference: Any = None
_INFERENCE_LOCK = threading.Lock()


def _load_inference() -> Any:
    """惰性加载模型（第一次请求时），并复用上游 `onnx_infer.py` 的实现。"""
    global _inference
    if _inference is not None:
        return _inference
    model_path = HUBERTFA_ROOT / "model.onnx"
    upstream = HUBERTFA_ROOT / "upstream"
    if not model_path.is_file() or not upstream.is_dir():
        raise HTTPException(
            status_code=503,
            detail=(
                f"HubertFA 模型/代码不存在：{HUBERTFA_ROOT}。"
                f"下载方式见 models/aligner/HubertFA/SOURCE.md"
            ),
        )
    sys.path.insert(0, str(upstream))
    from onnx_infer import InferenceOnnx  # type: ignore[import-not-found]

    inference = InferenceOnnx(onnx_path=model_path)
    inference.load_config()
    inference.init_decoder()
    inference.load_model()
    _inference = inference
    logger.info("HubertFA model loaded: %s", model_path)
    return inference


def _dictionary_for(language: str) -> Path:
    name = {
        "zh": "ds-zh-pinyin-lite.txt",
        "ja": "japanese_dict_full.txt",
        "en": "ds_cmudict-07b.txt",
    }[language]
    return HUBERTFA_ROOT / name


_DICTIONARY_KEYS: dict[str, set[str]] = {}


def _dictionary_keys(language: str) -> set[str]:
    """词典的键集合（音节/词）。用来在送模型之前把不认识的部分剔掉。"""
    cached = _DICTIONARY_KEYS.get(language)
    if cached is None:
        path = _dictionary_for(language)
        if not path.is_file():
            raise HTTPException(status_code=503, detail=f"HubertFA 词典缺失: {path}")
        cached = {
            line.split("\t")[0].strip()
            for line in path.read_text(encoding="utf-8").strip().split("\n")
            if "\t" in line
        }
        _DICTIONARY_KEYS[language] = cached
    return cached


def align_one(
    audio_bytes: bytes, text: str, language: str
) -> list[tuple[str, float, float]]:
    """单个片段：写临时 wav + lab → 跑模型 → 解析 TextGrid → 聚合回原文片段。"""
    chunks = g2p(text, language)
    if not chunks:
        return []
    inference = _load_inference()
    work = Path(tempfile.mkdtemp(prefix="hfa-align-"))
    try:
        wav_path = work / "clip.wav"
        wav_path.write_bytes(audio_bytes)
        (work / "clip.lab").write_text(
            " ".join(s for chunk in chunks for s in chunk.syllables), encoding="utf-8"
        )
        with _INFERENCE_LOCK:
            # 上游的 dataset 是**累加**的（__init__ 里初始化一次，get_dataset 只 append），
            # 常驻服务里必须每次清空，否则会重复处理上一批请求的（已删除的）临时路径。
            inference.dataset.clear()
            inference.get_dataset(
                wav_folder=work,
                language=language,
                g2p="dictionary",
                dictionary_path=_dictionary_for(language),
            )
            inference.infer(non_lexical_phonemes="AP")
            inference.export(output_folder=work, output_format=["textgrid"])
        grid = work / "TextGrid" / "clip.TextGrid"
        if not grid.is_file():
            raise RuntimeError(f"HubertFA 没有产出 TextGrid: {list(work.rglob('*'))}")
        return build_units(chunks, parse_textgrid(grid))
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _normalise_list(value: list[str] | str, count: int, field: str) -> list[str]:
    values = value if isinstance(value, list) else [value]
    if len(values) == 1:
        return values * count
    if len(values) != count:
        raise HTTPException(
            status_code=400,
            detail=f"{field} 的数量({len(values)})与音频数量({count})不匹配",
        )
    return values


@app.post("/align", response_model=list[AlignResponse] | AlignResponse)
def align(
    audio: list[UploadFile] | UploadFile = File(
        ..., description="音频文件 (wav/mp3/flac/m4a)"
    ),
    text: list[str] | str = Form(..., description="与音频对应的参考文本"),
    language: list[str] | str = Form(
        "Chinese", description="语言 (Chinese/Japanese/English 或 zh/ja/en)"
    ),
) -> list[AlignResponse] | AlignResponse:
    """对音频和文本做强制对齐，返回逐单元的 `(text, start, end)`。

    契约与 Qwen 服务一致：**单文件返回对象，多文件返回数组**（详见
    `qwen3aligner_server.py` 的说明）。`language` 既接受 Qwen 那套名字也接受短代码。
    """
    audios = audio if isinstance(audio, list) else [audio]
    is_batch = len(audios) > 1
    texts = _normalise_list(text, len(audios), "text")
    raw_languages = _normalise_list(language, len(audios), "language")

    languages = []
    for item in raw_languages:
        key = _LANGUAGE_ALIASES.get(item.strip().lower())
        if key is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"HubertFA 不支持的语言 {item!r}；"
                    f"可用: {sorted(set(_LANGUAGE_ALIASES.values()))}"
                ),
            )
        languages.append(key)

    responses: list[AlignResponse] = []
    for item, line_text, lang in zip(audios, texts, languages, strict=True):
        try:
            units = align_one(item.file.read(), line_text, lang)
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("对齐失败")
            raise HTTPException(
                status_code=500, detail=f"{type(exc).__name__}: {exc}"
            ) from exc
        responses.append(
            AlignResponse(
                words=[
                    AlignedWord(
                        text=chunk_text,
                        start_time=round(start / 1000.0, 4),
                        end_time=round(end / 1000.0, 4),
                    )
                    for chunk_text, start, end in units
                ]
            )
        )
    if not responses:
        raise HTTPException(status_code=500, detail="对齐器没有返回任何结果")
    return responses if is_batch else responses[0]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def main(argv: list[str] | None = None) -> int:
    import uvicorn

    parser = argparse.ArgumentParser(description="HubertFA 对齐服务（歌声专用）")
    parser.add_argument("--host", default=DEFAULT_HOST, help="监听地址")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="监听端口")
    parser.add_argument(
        "--preload",
        action="store_true",
        help="启动时就把模型加载好（否则首次请求才加载）",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="调试日志")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s [hfa] %(name)s: %(message)s",
    )
    if args.preload:
        _load_inference()
    logger.info("serving on http://%s:%d", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
