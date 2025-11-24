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

logger = getLogger(__name__)
repo = Path("models/sep/Demucs_Models/v3_v4_repo")
models = demucs.api.list_models(repo)
model = "htdemucs_6s"
assert model in models["single"] or model in models["bag"], "model not found in repo"
separator = demucs.api.Separator(
    model=model,
    repo=repo,
    device="cuda:0",
    # segment=44,
)


def get_vocal_stem(
    src: str | Path | BytesIO,
) -> tuple[NpAudioData, int]:
    sample_rate = separator.samplerate
    logger.info(f"ready to separate vocal stem, model={model!r}, sr={sample_rate!r}")
    audio_np = load_audio(src, sample_rate=sample_rate)
    audio_tensor = ndarray2tensor(audio_np)
    _, stems = separator.separate_tensor(audio_tensor, sr=sample_rate)
    vocal_stem = stems["vocals"]
    vocal_np: NpAudioData = tensor2ndarray(vocal_stem)
    logger.info(f"separate done, sample_points={len(vocal_np[0])}")
    return vocal_np, sample_rate
