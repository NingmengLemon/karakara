from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Annotated

import numpy as np
from numpy.typing import NDArray

from karakara.utils import DEFAULT_SAMPLE_RATE


@dataclass
class AlignedWord:
    word: str
    position: tuple[int, int] | None = None


class AbstractAligner(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def align(
        self,
        audio: Annotated[NDArray[np.float32], "Shape[*,]"],
        text: str,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
    ) -> list[AlignedWord]:
        raise NotImplementedError
