# /// script
# requires-python = "==3.12.*"
# dependencies = [
#     "av>=15.1.0",
#     "demucs @ git+https://github.com/adefossez/demucs@eeac1d15891af95b1288d2884b95baa3e5baa96c",
#     "numpy<2",
#     "soundfile>=0.13.1",
#     "torch>=2.9.0",
# ]
#
# # PyTorch 的 CUDA 索引。不加这一段的话 torch 会从 PyPI 解析到 CPU 版，
# # 分离会悄悄退化到 CPU（慢一个数量级）。索引是 explicit 的，必须由下面的
# # [tool.uv.sources] 显式引用才会生效。
# # 换 CUDA 版本只需改 url；只用 CPU 则把这两段一起删掉。
# [[tool.uv.index]]
# name = "pytorch_cu130"
# url = "https://download.pytorch.org/whl/cu130"
# explicit = true
#
# [tool.uv.sources]
# torch = { index = "pytorch_cu130" }
# ///
"""Demucs 分离 worker：常驻子进程 + stdin/stdout 行分隔 JSON 协议。

用法（由主程序 ``SubprocessStemSeparator`` 自动拉起，一般不需要手动运行）::

    uv run --script scripts/separator_worker.py
    uv run --script scripts/separator_worker.py --info      # 自检

为什么独立于主环境
------------------
Demucs 依赖 PyTorch（CUDA 版约 2.7GB）。把它放在独立环境里，主程序就不再需要
torch/demucs，也不用强制 GPU。本脚本头部的 PEP 723 内联依赖让 ``uv run --script``
自动准备并缓存这个环境。

协议
----
请求（每行一个 JSON 对象）::

    {"id": 1, "cmd": "separate", "audio": "<abs>", "dest_dir": "<abs>",
     "stems": ["vocals"], "model": "htdemucs_6s", "device": "cuda:0",
     "model_dir": "<abs>"}
    {"id": 2, "cmd": "info"}
    {"id": 3, "cmd": "shutdown"}

响应::

    {"id": 1, "ok": true, "stems": {"vocals": "<abs>"}, "samplerate": 44100,
     "elapsed_s": 12.3}
    {"id": 1, "ok": false, "error": "<message>"}

约定
----
* **stdout 只允许出现协议 JSON。** Demucs 与第三方库可能往 stdout 打东西，因此
  分离过程会把 stdout 临时重定向到 stderr。
* 日志与 Demucs 进度条走 stderr，父进程直接继承，不做解析。
* 模型按 (model, model_dir, device) 缓存，**只加载一次**并在整个进程生命周期内
  复用；进程在收到 ``shutdown`` 或 stdin EOF 时退出。
* 音轨写成 **32 位浮点 WAV**：16 位量化会在下游的归一化/颤音抑制之前就引入
  量化噪声，白白损失动态范围。
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger("karakara.separator_worker")

#: 项目根目录（本脚本位于 <root>/scripts/ 下），用于解析默认模型仓库。
PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: 与主程序一致的默认值。
#:
#: 默认模型选 ``UVR_Demucs_Model_1`` 而不是 ``htdemucs_6s``：在 4 首真实歌曲上
#: 实测前者人声泄漏更低（0.481 vs 0.533）、区间对比度更高（0.1771 vs 0.1650），
#: 而且更快（6.7s vs 8.0s，GPU）。度量方法与局限见 scripts/compare_separators.py。
DEFAULT_MODEL = "UVR_Demucs_Model_1"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "sep" / "Demucs_Models" / "v3_v4_repo"
DEFAULT_DEVICE = "cuda:0"
DEFAULT_FORMAT = "FLOAT"
DEFAULT_SUBTYPE = "FLOAT"

_separator_cache: dict[tuple[str, str, str], Any] = {}


# --------------------------------------------------------------------------
# 音频 I/O
#
# 刻意内联一份最小实现，让 worker 完全自包含：它跑在与主程序不同的虚拟环境里，
# 不能假设 karakara 包可用。
# --------------------------------------------------------------------------


def decode_audio(path: Path, sample_rate: int) -> Any:
    """解码为 (channels, samples) 的 float32 数组，并重采样到 sample_rate。

    单个损坏的 packet 只跳过并告警，不让整首歌失败。真实曲库里确实存在这种文件
    （实测 `D:\\MUSIC` 某首 mp3 有 5 个坏包 / 9569 个好包，位置在文件尾部），为它们
    整体报错等于把一首本来能对齐的歌白白丢掉。主进程侧的
    ``karakara.utils.io._decode`` 早就是这个行为（``skip_invalid`` 默认开），
    这里与它对齐。

    **全部**包都解不出来仍然报错：那是文件本身不可用，不是零星损坏。
    """
    import av
    import numpy as np

    resampler = av.AudioResampler("fltp", rate=sample_rate)
    frames: list[Any] = []
    skipped = 0
    with av.open(str(path), "r") as container:
        stream = container.streams.audio[0]
        for packet in container.demux(stream):
            try:
                decoded = packet.decode()
            except av.InvalidDataError as exc:
                skipped += 1
                if skipped == 1:
                    LOGGER.warning(f"跳过损坏的音频包 @ {packet.pts}: {exc}")
                continue
            for frame in decoded:
                for resampled in resampler.resample(frame):
                    frames.append(resampled.to_ndarray())
        for resampled in resampler.resample(None):
            frames.append(resampled.to_ndarray())

    if skipped:
        LOGGER.warning(f"{path.name}: 共跳过 {skipped} 个损坏的音频包")
    if not frames:
        raise ValueError(
            f"未能从 {path} 解码出任何音频帧"
            + (f"（{skipped} 个包全部无法解码）" if skipped else "")
        )
    return np.concatenate(frames, axis=1).astype(np.float32)


def write_float_wav(path: Path, audio: Any, sample_rate: int) -> None:
    """把 (channels, samples) 的浮点音频写成 32 位浮点 WAV。"""
    import soundfile as sf

    if audio.ndim != 2:
        raise ValueError(f"音轨维度应为 (channels, samples)，实际 {audio.shape}")
    # soundfile 期望 (samples, channels)
    sf.write(str(path), audio.T, sample_rate, format="WAV", subtype=DEFAULT_SUBTYPE)


# --------------------------------------------------------------------------
# Demucs 后端
# --------------------------------------------------------------------------


class DemucsBackend:
    """Demucs 分离后端。"""

    name = "demucs"

    def __init__(
        self,
        *,
        default_model: str = DEFAULT_MODEL,
        default_model_dir: Path | None = DEFAULT_MODEL_DIR,
        default_device: str | None = None,
    ) -> None:
        self._default_model = default_model
        self._default_model_dir = default_model_dir
        self._default_device = default_device or self._pick_device()

    @staticmethod
    def _pick_device() -> str:
        try:
            import torch
        except ImportError:  # pragma: no cover - 依赖缺失时给出明确报错
            raise RuntimeError("缺少 torch，无法运行 demucs 后端") from None
        if torch.cuda.is_available():
            return "cuda:0"
        # 绝不静默降级：本项目原先从 pytorch 的 cu130 索引装 torch，分离跑在 GPU 上；
        # 改成 worker 独立环境后，torch 默认从 PyPI 解析（Windows 上是 CPU 版），
        # 于是设备悄悄变成 cpu、分离慢一个数量级。这里必须明确告警。
        LOGGER.warning(
            "CUDA 不可用，分离将在 CPU 上运行（会慢很多）。"
            "当前 torch=%s。若要用 GPU，请让 worker 环境改用 PyTorch 的 CUDA 索引："
            "在 scripts/separator_worker.py 的 PEP 723 头部加入 "
            '\'[[tool.uv.index]] name="pytorch_cu130" url="https://download.pytorch.org/whl/cu130"'
            " explicit=true' 与 '[tool.uv.sources] torch = { index = \"pytorch_cu130\" }'，"
            "或用 --separator-cmd 指向一个已装 CUDA 版 torch 的解释器",
            getattr(torch, "__version__", "?"),
        )
        return "cpu"

    def _resolve_model_dir(self, model_dir: str | None) -> Path | None:
        if model_dir is not None:
            path = Path(model_dir)
            return path if path.exists() else None
        if self._default_model_dir is not None and self._default_model_dir.exists():
            return self._default_model_dir
        # 本地没有模型仓库时交给 demucs 自己下载
        return None

    def get_separator(self, model: str, model_dir: Path | None, device: str) -> Any:
        import demucs.api

        key = (model, str(model_dir), device)
        cached = _separator_cache.get(key)
        if cached is not None:
            return cached

        LOGGER.info(
            "loading demucs model: model=%s dir=%s device=%s", model, model_dir, device
        )
        separator = demucs.api.Separator(
            model=model,
            repo=model_dir,
            device=device,
            progress=False,
        )
        _separator_cache[key] = separator
        LOGGER.info("demucs model loaded, samplerate=%d", separator.samplerate)
        return separator

    def list_models(self, model_dir: Path | None) -> dict[str, list[str]] | None:
        import demucs.api

        if model_dir is None:
            return None
        return {
            name: sorted(values)
            for name, values in demucs.api.list_models(model_dir).items()
        }

    def separate(
        self,
        *,
        audio: Path,
        dest_dir: Path,
        stems: list[str] | None,
        model: str | None,
        device: str | None,
        model_dir: str | None,
    ) -> dict[str, Any]:
        import numpy as np

        resolved_model = model or self._default_model
        resolved_dir = self._resolve_model_dir(model_dir)
        resolved_device = device or self._default_device

        separator = self.get_separator(resolved_model, resolved_dir, resolved_device)
        sample_rate = int(separator.samplerate)

        waveform = decode_audio(audio, sample_rate)
        if waveform.ndim == 1:
            waveform = np.expand_dims(waveform, axis=0)

        import torch

        _, raw_stems = separator.separate_tensor(
            torch.from_numpy(waveform), sr=sample_rate
        )

        wanted = list(stems) if stems else None
        if wanted:
            missing = [name for name in wanted if name not in raw_stems]
            if missing:
                raise ValueError(
                    f"模型 {resolved_model!r} 没有这些音轨: {missing}；"
                    f"可用: {sorted(raw_stems)}"
                )

        dest_dir.mkdir(parents=True, exist_ok=True)
        written: dict[str, str] = {}
        for name, tensor in raw_stems.items():
            if wanted and name not in wanted:
                continue
            data = tensor.detach().cpu().contiguous().numpy()
            if data.ndim == 3:
                if data.shape[0] != 1:
                    raise ValueError(f"音轨 {name!r} 返回了意外的批次形状 {data.shape}")
                data = data[0]
            out_path = dest_dir / f"{name}.wav"
            write_float_wav(out_path, data, sample_rate)
            written[name] = str(out_path)

        return {"stems": written, "samplerate": sample_rate, "model": resolved_model}

    def info(self, model_dir: str | None) -> dict[str, Any]:
        resolved_dir = self._resolve_model_dir(model_dir)
        models: dict[str, list[str]] | None
        try:
            models = self.list_models(resolved_dir)
        except Exception as exc:  # noqa: BLE001 - 模型仓库不可读不应让自检整体失败
            LOGGER.warning("无法枚举模型仓库: %s", exc)
            models = None
        return {
            "backend": self.name,
            "device": self._default_device,
            "model": self._default_model,
            "model_dir": str(resolved_dir) if resolved_dir is not None else None,
            "models": models,
            "loaded": [list(key) for key in _separator_cache],
            "torch": self._torch_info(),
        }

    @staticmethod
    def _torch_info() -> dict[str, Any]:
        """报告 torch/CUDA 情况。

        "分离悄悄退化到 CPU" 是这套架构最容易踩的坑：worker 是独立环境，它的 torch
        依赖由脚本头部的内联依赖决定；若从 PyPI 解析到 CPU 版，分离会慢一个数量级
        而且看不出原因。把这些信息放进 ``--info`` 是最省事的排查手段。
        """
        try:
            import torch
        except ImportError:
            return {"installed": False}
        cuda_available = False
        error: str | None = None
        try:
            cuda_available = bool(torch.cuda.is_available())
        except Exception as exc:  # noqa: BLE001 - CUDA 初始化失败本身就是要报告的信息
            error = f"{type(exc).__name__}: {exc}"
        return {
            "installed": True,
            "version": getattr(torch, "__version__", None),
            "cuda_build": getattr(torch.version, "cuda", None),
            "cuda_available": cuda_available,
            "device_count": (torch.cuda.device_count() if cuda_available else 0),
            "error": error,
        }


# --------------------------------------------------------------------------
# 协议循环
# --------------------------------------------------------------------------


def _write_response(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _handle(request: dict[str, Any], backend: DemucsBackend) -> dict[str, Any]:
    cmd = request.get("cmd")
    if cmd == "shutdown":
        return {"ok": True, "shutdown": True}
    if cmd == "info":
        return {"ok": True, **backend.info(request.get("model_dir"))}
    if cmd != "separate":
        return {"ok": False, "error": f"未知命令: {cmd!r}"}

    audio = request.get("audio")
    dest_dir = request.get("dest_dir")
    if not audio or not dest_dir:
        return {"ok": False, "error": "separate 需要 audio 与 dest_dir"}

    stems = request.get("stems")
    if stems is not None and not isinstance(stems, list):
        return {"ok": False, "error": "stems 必须是字符串数组或省略"}

    started = time.perf_counter()
    result = backend.separate(
        audio=Path(str(audio)),
        dest_dir=Path(str(dest_dir)),
        stems=[str(s) for s in stems] if stems else None,
        model=request.get("model"),
        device=request.get("device"),
        model_dir=request.get("model_dir"),
    )
    result["elapsed_s"] = round(time.perf_counter() - started, 3)
    return {"ok": True, **result}


def serve(backend: DemucsBackend) -> int:
    """读 stdin 的行请求并逐条回应，直到 shutdown 或 EOF。"""
    LOGGER.info("separator worker ready, waiting for requests on stdin")
    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            request = json.loads(raw)
        except json.JSONDecodeError as exc:
            _write_response({"id": None, "ok": False, "error": f"非法 JSON: {exc}"})
            continue
        if not isinstance(request, dict):
            _write_response({"id": None, "ok": False, "error": "请求必须是 JSON 对象"})
            continue

        request_id = request.get("id")
        try:
            # 把任何往 stdout 的输出赶去 stderr，保证协议通道纯净
            with contextlib.redirect_stdout(sys.stderr):
                response = _handle(request, backend)
        except SystemExit as exc:
            # 依赖里有人用 sys.exit() 报错（例如 demucs 发现缺 diffq 时）。
            # SystemExit 继承自 BaseException，``except Exception`` 抓不到它，
            # 那样 worker 会直接死掉、父进程只能看到"提前退出"。这里把它当成
            # 一次普通的请求失败，让批量处理能继续跑下一首。
            LOGGER.error("依赖以 SystemExit 终止了本次请求: %s", exc)
            _write_response(
                {
                    "id": request_id,
                    "ok": False,
                    "error": f"依赖调用了 sys.exit({exc.code!r})，"
                    f"请检查 worker 环境的依赖是否完整",
                }
            )
            continue
        except Exception as exc:
            # 单条请求失败不能让 worker 退出，否则批量处理时一首歌会拖垮整批
            LOGGER.exception("处理请求失败: %s", request.get("cmd"))
            _write_response(
                {
                    "id": request_id,
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue

        shutdown = bool(response.pop("shutdown", False))
        _write_response({"id": request_id, **response})
        if shutdown:
            LOGGER.info("收到 shutdown，退出")
            break

    LOGGER.info("stdin 已关闭，退出")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Demucs 分离 worker（由主程序拉起）")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="默认模型名")
    parser.add_argument(
        "--model-dir",
        default=str(DEFAULT_MODEL_DIR),
        help="模型仓库目录（默认指向项目的 models/sep/Demucs_Models/v3_v4_repo）",
    )
    parser.add_argument("--device", default=None, help="默认设备，缺省自动选择")
    parser.add_argument(
        "--info",
        action="store_true",
        help="只打印后端信息后退出（自检用，不需要父进程）",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="调试日志")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s [worker] %(name)s: %(message)s",
        stream=sys.stderr,
    )

    model_dir = Path(args.model_dir) if args.model_dir else None
    backend = DemucsBackend(
        default_model=args.model,
        default_model_dir=model_dir,
        default_device=args.device,
    )

    if args.info:
        payload = backend.info(str(model_dir) if model_dir else None)
        # 自检走 stdout 是给人看的，这里不用协议格式
        sys.stdout.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        return 0

    return serve(backend)


if __name__ == "__main__":
    raise SystemExit(main())
