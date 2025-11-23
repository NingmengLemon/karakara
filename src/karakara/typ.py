from __future__ import annotations

from typing import Annotated

import numpy as np
from numpy.typing import NDArray
from typing_extensions import TypeAlias

NpAudioData: TypeAlias = Annotated[NDArray[np.float32], "Shape[*, *]"]
