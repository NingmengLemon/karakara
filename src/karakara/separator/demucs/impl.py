from __future__ import annotations

import warnings
from collections.abc import Callable
from functools import lru_cache
from logging import getLogger
from pathlib import Path
from typing import Any

import demucs.api
import demucs.repo
import demucs.states
import torch
from typing_extensions import override

from karakara.typ import NpAudioData, NpAudioSamples
from karakara.utils.io import ndarray2tensor, tensor2ndarray

from ..abc import AbstractStemSeparator

DEFAULT_REPO = Path("models/sep/Demucs_Models/v3_v4_repo")
DEFAULT_MODEL = "htdemucs_6s"
DEFAULT_DEVICE = "cuda:0"

logger = getLogger(__name__)

# ── PyTorch 2.6+ 兼容性修复 ────────────────────────────────────
# demucs 的 checkpoint 使用 pickle 序列化了完整的模型类对象，
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


@lru_cache(maxsize=4)
def _get_demucs_separator(
    model: str,
    repo: str,
    device: str,
) -> demucs.api.Separator:
    """获取（可能已缓存的）Demucs Separator 实例。"""
    models = demucs.api.list_models(Path(repo))
    assert model in models["single"] or model in models["bag"], (
        f"model {model!r} not found in repo {repo!r}"
    )
    sep = demucs.api.Separator(
        model=model,
        repo=Path(repo),
        device=device,
    )
    logger.debug(f"Demucs Separator created: model={model!r}, device={device!r}")
    return sep


class DemucsSeparator(AbstractStemSeparator):
    """基于 Meta Demucs 的音轨分离器。"""

    VOCAL_STEM_NAME = "vocals"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        repo: Path = DEFAULT_REPO,
        device: str = DEFAULT_DEVICE,
    ) -> None:
        self._model = model
        self._repo = repo
        self._device = device

    @property
    def samplerate(self) -> int:
        return self._separator.samplerate  # type: ignore[no-any-return]

    @property
    def _separator(self) -> demucs.api.Separator:
        return _get_demucs_separator(
            model=self._model,
            repo=str(self._repo),
            device=self._device,
        )

    @override
    def separate(self, audio: NpAudioData | NpAudioSamples) -> dict[str, NpAudioData]:
        """分离音轨。

        Args:
            audio: 输入音频，shape (channels, samples)。
                   如果是 1D (samples,) 会自动升维。

        Returns:
            stem_name → NpAudioData 的字典，如 {"vocals": ..., "drums": ..., ...}
        """
        import numpy as np

        if audio.ndim == 1:
            audio = np.expand_dims(audio, axis=0)

        tensor = ndarray2tensor(audio)
        sr = self.samplerate
        logger.info(f"separating with Demucs, sr={sr}")
        _, stems = self._separator.separate_tensor(tensor, sr=sr)

        result: dict[str, NpAudioData] = {}
        for name, stem_tensor in stems.items():
            result[name] = tensor2ndarray(stem_tensor)
        logger.info(f"separation done, stems: {list(result.keys())}")
        return result
