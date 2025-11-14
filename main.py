from pathlib import Path

import demucs
import demucs.api

repo = Path("models/sep/Demucs_Models/v3_v4_repo")
models = demucs.api.list_models(repo)
sep = demucs.api.Separator(
    model=str(models["bag"]["htdemucs_6s"]),
    repo=repo,
    device="cuda",
    # segment=44,
)
