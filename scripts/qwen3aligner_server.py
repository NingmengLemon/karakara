# /// script
# requires-python = "==3.12.*"
# dependencies = [
#     "fastapi>=0.135.3",
#     "pydantic>=2.12.4",
#     "python-multipart>=0.0.20",
#     "qwen-asr>=0.0.6",
#     "torch>=2.9.0",
#     "uvicorn>=0.35.0",
# ]
#
# # PyTorch 的 CUDA 索引。**torch 必须显式列在上面**：它在 qwen-asr 里只是间接
# # 依赖，而 [tool.uv.sources] 只对直接依赖生效；不给它建索引的话 torch 会从 PyPI
# # 解析到 CPU 版，于是 GPU 机器上也在 CPU 上跑，不看日志根本发现不了
# # （worker 侧 scripts/separator_worker.py 是同样的写法）。
# # 换 CUDA 版本只需改 url；只用 CPU 则把这两段一起删掉、并去掉上面的 torch。
# [[tool.uv.index]]
# name = "pytorch_cu130"
# url = "https://download.pytorch.org/whl/cu130"
# explicit = true
#
# [tool.uv.sources]
# torch = { index = "pytorch_cu130" }
# ///
"""Qwen3-ForcedAligner HTTP 服务（由主程序按 ``--aligner-url`` 调用）。

启动::

    uv run --script scripts/qwen3aligner_server.py                  # 127.0.0.1:8787
    uv run --script scripts/qwen3aligner_server.py --port 9000
    uv run --script scripts/qwen3aligner_server.py --host 0.0.0.0   # 跨机（服务无鉴权！）

默认监听地址与 ``main.py --aligner-url`` 的默认值一致（8787），因此从仓库根目录
按上面第一条命令启动后，主程序不需要任何额外参数。

**响应形状是这份契约的一部分**：``Q3FAClient`` 读的是 ``response["words"]``，所以

* 单个文件 → 返回**对象** ``{"words": [...]}``
* 多个文件 → 返回**数组** ``[{"words": [...]}, ...]``

``is_batch`` 必须按**实际收到的文件数**判断，而不能按形参的运行时类型判断：
FastAPI 对 ``list[UploadFile] | UploadFile`` 这种联合类型，对单个上传也会走 list
分支（实测），于是单文件请求拿到数组响应，客户端随即抛
``TypeError: list indices must be integers or slices, not str``；主程序那边则表现为
「每一行都对齐失败、退出码却为 0」的静默失效。
"""

from __future__ import annotations

import argparse
import logging
import shutil
import tempfile
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from qwen_asr import Qwen3ForcedAligner

logger = logging.getLogger("karakara.aligner_server")

#: 项目根目录（本脚本位于 <root>/scripts/ 下）。模型路径以此为基准，
#: 否则从别的目录启动就会去别处找模型。
PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "aligner" / "Qwen3-ForcedAligner-0.6B"

#: 与 main.py 的 --aligner-url 默认值保持一致。
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8787

# ── FastAPI 应用 ────────────────────────────────────────────
app = FastAPI(title="Qwen3-ForcedAligner Service", version="1.1.0")


class AlignedWord(BaseModel):
    text: str
    start_time: float
    end_time: float


class AlignResponse(BaseModel):
    words: list[AlignedWord]


#: 由 ``main()`` 填充；模块级持有以便整个进程复用同一个模型实例。
_aligner: Qwen3ForcedAligner | None = None


def current_aligner() -> Qwen3ForcedAligner:
    """取出已加载的模型；尚未就绪时返回 503 而不是 AttributeError。"""
    if _aligner is None:
        raise HTTPException(status_code=503, detail="模型尚未加载完成")
    return _aligner


def _normalise_list(value: list[str] | str, count: int, field: str) -> list[str]:
    """把表单字段规范成长度等于 ``count`` 的列表。

    单个值会被广播；长度既不是 1 也不是 ``count`` 时直接报 400，而不是让两个
    长度错位的列表在模型里静默对上错的行。
    """
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
    audios = audio if isinstance(audio, list) else [audio]
    # 契约：只有真的收到多个文件才算批量（详见模块 docstring）。
    is_batch = len(audios) > 1
    texts = _normalise_list(text, len(audios), "text")
    languages = _normalise_list(language, len(audios), "language")

    tmp_paths: list[str] = []
    try:
        for item in audios:
            suffix = Path(item.filename or "audio.wav").suffix
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                shutil.copyfileobj(item.file, tmp)
                tmp_paths.append(tmp.name)

        try:
            results = current_aligner().align(
                audio=tmp_paths,
                text=texts,
                language=languages,
            )
        except HTTPException:
            raise
        except Exception as exc:
            # 统一转成可读的 500（BLE001 不报这里，因为异常被重新抛出）
            logger.exception("对齐失败")
            raise HTTPException(
                status_code=500, detail=f"{type(exc).__name__}: {exc}"
            ) from exc

        responses = [
            AlignResponse(
                words=[
                    AlignedWord(
                        text=word.text,
                        start_time=round(word.start_time, 4),
                        end_time=round(word.end_time, 4),
                    )
                    for word in result
                ]
            )
            for result in results
        ]
        if not responses:
            raise HTTPException(status_code=500, detail="对齐器没有返回任何结果")

        # 契约要点：单文件返回对象，多文件返回数组。
        return responses if is_batch else responses[0]
    finally:
        for path in tmp_paths:
            Path(path).unlink(missing_ok=True)


@app.get("/supported_languages")
async def supported_languages() -> list[str] | None:
    """获取模型支持的语言列表。"""
    model = current_aligner()
    if hasattr(model, "get_supported_languages"):
        return model.get_supported_languages()  # type: ignore[no-any-return]
    return None


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


def _pick_device(requested: str | None) -> str:
    """决定推理设备，并且**不静默降级**到 CPU。"""
    available = bool(torch.cuda.is_available())
    if requested is not None:
        if requested.startswith("cuda") and not available:
            raise SystemExit(
                f"指定了 {requested}，但当前 torch 看不到 CUDA："
                f"torch={torch.__version__}, cuda_build={torch.version.cuda}"
            )
        return requested
    if available:
        return "cuda:0"
    # 与 separator_worker 同样的坑：worker/服务各自跑在独立环境里，torch 从 PyPI
    # 解析时会拿到 CPU 版，GPU 机器上也会悄悄跑在 CPU 上。
    logger.warning(
        "CUDA 不可用，对齐将在 CPU 上运行（会慢很多）。当前 torch=%s。"
        "若要用 GPU，请在 scripts/qwen3aligner_server.py 的 PEP 723 头部加入 "
        '\'[[tool.uv.index]] name="pytorch_cu130" '
        'url="https://download.pytorch.org/whl/cu130" explicit=true\' 与 '
        "'[tool.uv.sources] torch = { index = \"pytorch_cu130\" }'，"
        "然后清掉 uv 对该脚本的缓存环境再重建"
        "（uv 的脚本环境缓存键不含索引配置，只改索引会照旧复用 CPU 环境）。",
        torch.__version__,
    )
    return "cpu"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Qwen3-ForcedAligner HTTP 服务（与 main.py --aligner-url 对接）"
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help=f"监听地址（默认 {DEFAULT_HOST}；跨机访问才用 0.0.0.0，注意服务无鉴权）",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"监听端口（默认 {DEFAULT_PORT}，与 main.py 的默认值一致）",
    )
    parser.add_argument("--model", default=str(MODEL_PATH), help="模型目录")
    parser.add_argument("--device", default=None, help="推理设备，如 cuda:0 / cpu")
    parser.add_argument("--verbose", "-v", action="store_true", help="调试日志")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s [aligner] %(name)s: %(message)s",
    )

    model_path = Path(args.model)
    if not model_path.is_dir():
        raise SystemExit(f"模型目录不存在: {model_path}")

    device = _pick_device(args.device)
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32

    global _aligner
    logger.info("loading model from %s on %s ...", model_path, device)
    _aligner = Qwen3ForcedAligner.from_pretrained(
        str(model_path),
        dtype=dtype,
        device_map=device,
    )
    logger.info("model loaded successfully on %s", device)

    logger.info("serving on http://%s:%d", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
