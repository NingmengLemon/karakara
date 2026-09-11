from __future__ import annotations

from typing import Annotated

import numpy as np
from numpy.typing import NDArray

#: (channels, samples) 的 float32 音频。
type NpAudioData = Annotated[NDArray[np.float32], "Shape[*, *]"]
#: (samples,) 的 float32 音频。
type NpAudioSamples = Annotated[NDArray[np.float32], "Shape[*,]"]
