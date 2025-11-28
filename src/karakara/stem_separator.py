from __future__ import annotations

from io import BytesIO
from logging import getLogger
from pathlib import Path

import demucs
import demucs.api

from karakara.typ import NpAudioData
from karakara.utils import (
    load_audio,
    ndarray2tensor,
    tensor2ndarray,
)

DEFAULT_REPO = Path("models/sep/Demucs_Models/v3_v4_repo")
DEFAULT_MODEL = "htdemucs_6s"
DEFAULT_DEVICE = "cuda:0"

VOCAL_STEM_NAME = "vocals"

separator: demucs.api.Separator | None = None
logger = getLogger(__name__)


def get_separator(
    model: str = DEFAULT_MODEL,
    repo: Path = DEFAULT_REPO,
    device: str = DEFAULT_DEVICE,
) -> demucs.api.Separator:
    models = demucs.api.list_models(repo)
    assert model in models["single"] or model in models["bag"], (
        "model not found in repo"
    )
    separator = demucs.api.Separator(
        model=model,
        repo=repo,
        device=device,
        # segment=44,
    )
    return separator


def get_vocal_stem(
    src: str | Path | BytesIO,
) -> tuple[NpAudioData, int]:
    global separator
    if separator is None:
        separator = get_separator()
    sample_rate = separator.samplerate
    logger.info(f"separating vocal stem, model={separator.model!r}, sr={sample_rate!r}")
    audio_np = load_audio(src, sample_rate=sample_rate)
    audio_tensor = ndarray2tensor(audio_np)
    _, stems = separator.separate_tensor(audio_tensor, sr=sample_rate)
    vocal_stem = stems[VOCAL_STEM_NAME]
    vocal_np: NpAudioData = tensor2ndarray(vocal_stem)
    logger.info(f"separate done, sample_points={len(vocal_np[0])}")
    return vocal_np, sample_rate
