# server.py
from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from qwen_asr import Qwen3ForcedAligner

logger = logging.getLogger(__name__)

# ── 模型初始化 ──────────────────────────────────────────────
MODEL_PATH = "./models/aligner/Qwen3-ForcedAligner-0.6B"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE.startswith("cuda") else torch.float32

logger.info(f"Loading model from {MODEL_PATH} on {DEVICE} ...")
aligner = Qwen3ForcedAligner.from_pretrained(
    MODEL_PATH,
    dtype=DTYPE,
    device_map=DEVICE,
)
logger.info("Model loaded successfully!")

# ── FastAPI 应用 ────────────────────────────────────────────
app = FastAPI(title="Qwen3-ForcedAligner Service", version="1.0.0")


class AlignedWord(BaseModel):
    text: str
    start_time: float
    end_time: float


class AlignResponse(BaseModel):
    words: list[AlignedWord]


@app.post("/align", response_model=list[AlignResponse] | AlignResponse)
async def align(
    audio: list[UploadFile] | UploadFile = File(
        ..., description="音频文件 (wav/mp3/flac/m4a)"
    ),
    text: list[str] | str = Form(..., description="与音频对应的参考文本"),
    language: list[str] | str = Form(
        "Chinese", description="语言 (Chinese/English/French/German/...)"
    ),
) -> list[AlignResponse] | AlignResponse:
    """对音频和文本进行强制对齐，返回逐词时间戳。支持单样本和批量对齐。"""
    is_batch = isinstance(audio, list)

    audios = audio if isinstance(audio, list) else [audio]
    texts = text if isinstance(text, list) else [text]
    languages = language if isinstance(language, list) else [language]

    if len(languages) == 1 and len(audios) > 1:
        languages = languages * len(audios)

    if len(audios) != len(texts):
        raise HTTPException(status_code=400, detail="音频和文本的数量必须匹配")

    tmp_paths = []
    for a in audios:
        suffix = Path(a.filename or "audio.wav").suffix
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            shutil.copyfileobj(a.file, tmp)
            tmp_paths.append(tmp.name)

    try:
        results = aligner.align(
            audio=tmp_paths,
            text=texts,
            language=languages,
        )

        responses = []
        for res in results:
            words: list[AlignedWord] = []
            for word in res:
                words.append(
                    AlignedWord(
                        text=word.text,
                        start_time=round(word.start_time, 4),
                        end_time=round(word.end_time, 4),
                    )
                )
            responses.append(AlignResponse(words=words))

        if not is_batch:
            return responses[0]
        return responses
    finally:
        for p in tmp_paths:
            Path(p).unlink(missing_ok=True)


@app.get("/supported_languages")
async def supported_languages() -> list[str] | None:
    """获取模型支持的语言列表。"""
    if hasattr(aligner, "get_supported_languages"):
        return aligner.get_supported_languages()  # type: ignore
    return None


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(
        "__main__:app",
        host="0.0.0.0",  # 监听所有网络接口
        port=8000,
        workers=1,  # GPU 模型通常单 worker
    )
