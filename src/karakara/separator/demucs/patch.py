import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import demucs.repo
import demucs.states
import torch

# ── PyTorch 2.6+ 兼容性修复 ────────────────────────────────────
# 此问题在 demucs 4.1.0 正式版已修复, 在此仅作存档......
# demucs 4.1.0a 的 checkpoint 使用 pickle 序列化了完整的模型类对象，
# 而 PyTorch 2.6+ 默认 weights_only=True 会拒绝反序列化。
# 这里精确地 patch demucs 的 load_model，不影响其他代码的 torch.load。
_original_load_model: Callable[..., Any] = demucs.states.load_model


def _patched_load_model(
    path_or_package: dict[str, Any] | str | Path,
    strict: bool = False,
) -> Any:
    if isinstance(path_or_package, (str, Path)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            path_or_package = torch.load(
                path_or_package, map_location="cpu", weights_only=False
            )
    return _original_load_model(path_or_package, strict)


demucs.states.load_model = _patched_load_model
demucs.repo.load_model = _patched_load_model  # repo.py 持有独立引用
# ── 修复结束 ──────────────────────────────────────────────────
