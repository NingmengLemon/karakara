# /// script
# requires-python = "==3.12.*"
# dependencies = [
#     "audio-separator[cpu]>=0.47",
#     # librosa 在部分格式/版本下会退回 audioread，缺它会直接 ImportError
#     "audioread>=3.0",
# ]
# ///
"""audio-separator 分离 worker：与 separator_worker.py 使用完全相同的行协议。

用法（由主程序 ``SubprocessStemSeparator`` 拉起，一般无需手动运行）::

    uv run --script scripts/separator_worker_audio_separator.py
    uv run --script scripts/separator_worker_audio_separator.py --info
    uv run --script scripts/separator_worker_audio_separator.py --list-models --list-filter vocals

为什么值得多一个后端
--------------------
``python-audio-separator`` 把 UVR 社区的模型库（MDX-Net / VR-Arch / MDX23C /
RoFormer / Demucs）统一到一个接口下。对本项目而言，它的价值是**人声质量**，
不是依赖体积——恰恰相反，它比只用 demucs 更重（librosa / scipy / onnx /
onnx2torch / resampy 等，而且同样无条件需要 torch）。

质量差距是实打实的：Demucs ``htdemucs_6s`` 的人声 SDR 约 9.7，而 BS-RoFormer
人声模型约 12.9；此外它自带 ``karaoke`` / ``vocal_balanced`` 等 ensemble 预设。
逐行切段送给强制对齐器时，人声轨越干净，对齐越准。

模型目录
--------
``--model-dir`` 必须是 audio-separator 期望的**扁平**目录（模型文件直接躺在里面）。
注意 UVR5 GUI 的 ``models/sep`` 是 ``MDX_Net_Models/`` + ``VR_Models/`` +
``Demucs_Models/`` 的嵌套布局，audio-separator **不认**；缺省的模型它会自己下载到
``--model-dir``。本 worker **绝不写入** ``models/sep``。
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger("karakara.separator_worker")

PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: 已知音轨名 → 本项目的规范化键。audio-separator 的输出文件名里带括号化的音轨名，
#: 这里按关键字识别，而不是依赖它具体的命名模板（模板随版本变过）。
_STEM_KEYWORDS: tuple[tuple[str, str], ...] = (
    ("vocals", "vocals"),
    ("instrumental", "instrumental"),
    ("drums", "drums"),
    ("bass", "bass"),
    ("guitar", "guitar"),
    ("piano", "piano"),
    ("other", "other"),
)

#: 支持 ``--list-filter`` 的音轨名（audio-separator 的取值）。
LIST_FILTERS = ("vocals", "instrumental", "drums", "bass", "guitar", "piano", "other")

#: 默认模型目录。与 separator_worker.py 一样按**项目根目录**定位，而不是 CWD：
#: worker 是被主程序以继承来的工作目录拉起的，用相对路径会在别的目录下找错地方。
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "sep-audio-separator"


def classify_stem(filename: str) -> str | None:
    """从输出文件名判断它属于哪条音轨。

    audio-separator 的命名模板随版本变化（``<track>_(Vocals)_<model>.wav`` 之类），
    所以按关键字识别比按模板解析更稳。``instrumental`` 必须排在前面判断：它同时
    包含 ``...tal``，不会和其它关键字冲突，但顺序仍然显式固定以便推理。
    """
    lowered = filename.lower()
    for keyword, canonical in _STEM_KEYWORDS:
        if keyword in lowered:
            return canonical
    return None


def find_main_model_file(model_dir: Path) -> Path | None:
    """找出被 junction/软链指到别处的模型目录本体（只读用途）。"""
    try:
        return model_dir.resolve(strict=True)
    except OSError:
        return None


def prepare_float32_input(audio: Path, work_dir: Path) -> Path:
    """把源音频解码成 32 位浮点 WAV，供 audio-separator 使用。

    为什么必须这么做：audio-separator 会「按输入位深决定输出位深」，遇到它认不出
    位深的容器（典型如 mp3，日志里是 ``Unknown audio subtype MPEG_LAYER_III,
    defaulting to 16-bit output``）就退回 16 位。那意味着在我们的归一化/颤音抑制
    **之前**就多了一次量化，白白损失动态范围——而这正是本后端存在的意义（质量）。

    喂 32 位浮点输入能让它的位深保持落在浮点上，避免这次量化。对已经是浮点的输入
    直接沿用原文件，不做无谓的转码。
    """
    import librosa
    import numpy as np
    import soundfile as sf

    try:
        info = sf.info(str(audio))
        if info.subtype in {"FLOAT", "DOUBLE"}:
            return audio
    except Exception as exc:  # noqa: BLE001 - 读不出信息就老实转码，格式支持面更广
        LOGGER.debug("无法读取 %s 的位深信息，将统一转码: %s", audio.name, exc)

    work_dir.mkdir(parents=True, exist_ok=True)
    target = work_dir / f"{audio.stem}.float32.wav"

    # librosa 是 audio-separator 自己的依赖，格式支持面与它一致；sr=None 保持原生采样率
    data, sample_rate = librosa.load(str(audio), sr=None, mono=False)
    data = np.asarray(data, dtype=np.float32)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    sf.write(str(target), data.T, int(sample_rate), format="WAV", subtype="FLOAT")
    LOGGER.info(
        "把输入转成 32 位浮点以保留位深: %s -> %s (%dHz, %dch)",
        audio.name,
        target.name,
        int(sample_rate),
        data.shape[0],
    )
    return target


class AudioSeparatorBackend:
    """audio-separator 分离后端。"""

    name = "audio-separator"

    def __init__(
        self,
        *,
        default_model: str | None = None,
        default_model_dir: Path | None = None,
    ) -> None:
        self._default_model = default_model
        self._default_model_dir = default_model_dir

    def _model_dir(self, requested: str | None) -> Path:
        if requested:
            return Path(requested)
        if self._default_model_dir is not None:
            return self._default_model_dir
        return DEFAULT_MODEL_DIR

    def _make_separator(
        self, model_dir: Path, output_dir: Path, stems: list[str] | None
    ) -> Any:
        from audio_separator.separator import Separator

        model_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 只要 vocals 时只写 vocals：省掉多余的写盘与编码
        single_stem: str | None = None
        if stems and len(stems) == 1:
            single_stem = _canonical_to_display(stems[0])

        return Separator(
            log_level=logging.WARNING,
            model_file_dir=str(model_dir),
            output_dir=str(output_dir),
            output_format="WAV",
            output_single_stem=single_stem,
            sample_rate=44100,
        )

    def list_models(self, model_dir: str | None, list_filter: str | None) -> Any:
        """列出 audio-separator 认识的模型。

        直接调用它 CLI 的入口函数（``audio_separator.utils.cli:main``，也就是
        ``audio-separator`` 这个 console script 背后的东西），并把 stdout 抓下来。
        不要用 ``python -m audio_separator.utils.cli``：那个模块**没有**
        ``if __name__ == "__main__"`` 块，``-m`` 跑它只会静默退出、返回码为 0、
        什么都不打印。
        """
        from audio_separator.utils import cli

        resolved = self._model_dir(model_dir)
        argv = [
            "--list_models",
            "--list_format=json",
            "--model_file_dir",
            str(resolved),
        ]
        if list_filter:
            argv.append(f"--list_filter={list_filter}")

        buffer = io.StringIO()
        previous_argv = sys.argv
        # cli.main() 自己解析 sys.argv，不接受参数——按 console script 的真实调用方式
        # 临时替换 argv 再调用它。
        sys.argv = ["audio-separator", *argv]
        try:
            with contextlib.redirect_stdout(buffer):
                cli.main()
        except SystemExit as exc:
            if exc.code not in (0, None):
                raise RuntimeError(
                    f"audio-separator CLI 列出模型失败（exit={exc.code!r}）"
                ) from exc
        finally:
            sys.argv = previous_argv

        stdout = buffer.getvalue()
        start = min(
            (index for index in (stdout.find("["), stdout.find("{")) if index >= 0),
            default=-1,
        )
        if start < 0:
            raise RuntimeError(f"CLI 没有输出 JSON: {stdout.strip()[-300:]!r}")
        return json.loads(stdout[start:])

    def separate(
        self,
        *,
        audio: Path,
        dest_dir: Path,
        stems: list[str] | None,
        model: str | None,
        model_dir: str | None,
        **_: Any,
    ) -> dict[str, Any]:
        resolved_model = model or self._default_model
        if not resolved_model:
            raise ValueError(
                "audio-separator 必须显式指定模型（--separator-model），"
                "可用 --list-models 查看全部可选值"
            )

        resolved_dir = self._model_dir(model_dir)
        # 每次都用一个干净子目录，避免上一首的产物混进来
        out_dir = dest_dir / "audio_separator"
        separator = self._make_separator(resolved_dir, out_dir, stems)

        LOGGER.info(
            "loading audio-separator model: model=%s dir=%s",
            resolved_model,
            resolved_dir,
        )
        loaded = separator.load_model(model_filename=resolved_model)
        LOGGER.info("model loaded: %s", loaded or resolved_model)

        source = prepare_float32_input(audio, dest_dir / "float32-input")
        produced = separator.separate(str(source))
        if isinstance(produced, str):
            produced = [produced]

        wanted = set(stems) if stems else None
        written: dict[str, str] = {}
        for item in produced:
            path = Path(item)
            if not path.is_absolute():
                path = out_dir / path
            canonical = classify_stem(path.name)
            if canonical is None:
                LOGGER.warning("无法归类 audio-separator 的输出文件: %s", path.name)
                continue
            if wanted is not None and canonical not in wanted:
                continue
            written[canonical] = str(path)

        if not written:
            raise ValueError(
                f"audio-separator 没有产出可识别的音轨；"
                f"返回的文件: {[Path(p).name for p in produced]}"
            )

        return {
            "stems": written,
            "samplerate": int(getattr(separator, "sample_rate", 44100)),
            "model": resolved_model,
        }

    def info(self, model_dir: str | None) -> dict[str, Any]:
        resolved_dir = self._model_dir(model_dir)
        local_files: list[str] = []
        if resolved_dir.is_dir():
            local_files = sorted(
                p.name
                for p in resolved_dir.iterdir()
                if p.suffix.lower() in {".onnx", ".pth", ".ckpt", ".yaml", ".th"}
            )
        payload: dict[str, Any] = {
            "backend": self.name,
            "model_dir": str(resolved_dir),
            "resolved_model_dir": (
                str(target)
                if (target := find_main_model_file(resolved_dir)) is not None
                else None
            ),
            "local_model_files": local_files,
            "note": (
                "model_dir 必须是扁平目录；UVR5 GUI 的 models/sep 是嵌套布局，"
                "audio-separator 不认，本 worker 也不会写入那里。"
            ),
        }
        try:
            payload["available"] = self.list_models(model_dir, None)
        except Exception as exc:  # noqa: BLE001 - 自检不应因枚举失败而崩
            payload["available_error"] = f"{type(exc).__name__}: {exc}"
        return payload


def _canonical_to_display(canonical: str) -> str:
    """本项目规范键 → audio-separator 的音轨显示名。"""
    mapping = {
        "vocals": "Vocals",
        "instrumental": "Instrumental",
        "drums": "Drums",
        "bass": "Bass",
        "guitar": "Guitar",
        "piano": "Piano",
        "other": "Other",
    }
    return mapping.get(canonical, canonical.capitalize())


# --------------------------------------------------------------------------
# 协议循环（与 separator_worker.py 完全一致）
# --------------------------------------------------------------------------


def _write_response(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _handle(request: dict[str, Any], backend: AudioSeparatorBackend) -> dict[str, Any]:
    cmd = request.get("cmd")
    if cmd == "shutdown":
        return {"ok": True, "shutdown": True}
    if cmd == "info":
        return {"ok": True, **backend.info(request.get("model_dir"))}
    if cmd == "list_models":
        return {
            "ok": True,
            "models": backend.list_models(
                request.get("model_dir"), request.get("list_filter")
            ),
        }
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
        model_dir=request.get("model_dir"),
        device=request.get("device"),
    )
    result["elapsed_s"] = round(time.perf_counter() - started, 3)
    return {"ok": True, **result}


def serve(backend: AudioSeparatorBackend) -> int:
    LOGGER.info("audio-separator worker ready, waiting for requests on stdin")
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
            # audio-separator 及其依赖会往 stdout 打东西，协议通道必须保持纯净
            with contextlib.redirect_stdout(sys.stderr):
                response = _handle(request, backend)
        except SystemExit as exc:
            # SystemExit 继承自 BaseException，``except Exception`` 抓不到；
            # 依赖用 sys.exit() 报错时若不拦，worker 会直接死掉。
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
                {"id": request_id, "ok": False, "error": f"{type(exc).__name__}: {exc}"}
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
    parser = argparse.ArgumentParser(
        description="audio-separator 分离 worker（由主程序拉起）"
    )
    parser.add_argument("--model", default=None, help="默认模型文件名")
    parser.add_argument(
        "--model-dir",
        default=None,
        help="扁平的模型目录（缺省 models/sep-audio-separator）",
    )
    parser.add_argument(
        "--info", action="store_true", help="打印后端信息后退出（自检用）"
    )
    parser.add_argument(
        "--list-models", action="store_true", help="打印可用模型列表后退出"
    )
    parser.add_argument(
        "--list-filter", default=None, choices=LIST_FILTERS, help="按音轨过滤模型列表"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="调试日志")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s [worker] %(name)s: %(message)s",
        stream=sys.stderr,
    )

    backend = AudioSeparatorBackend(
        default_model=args.model,
        default_model_dir=Path(args.model_dir) if args.model_dir else None,
    )

    if args.info:
        payload = backend.info(args.model_dir)
        sys.stdout.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        return 0
    if args.list_models:
        models = backend.list_models(args.model_dir, args.list_filter)
        sys.stdout.write(json.dumps(models, ensure_ascii=False, indent=2) + "\n")
        return 0

    return serve(backend)


if __name__ == "__main__":
    raise SystemExit(main())
