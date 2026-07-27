"""Demucs 分离器输出形状适配测试。"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from karakara.separator.demucs.impl import DemucsSeparator


class _FakeDemucsApiSeparator:
    samplerate = 44_100

    def __init__(self, stem_shape: tuple[int, ...]) -> None:
        self._stem_shape = stem_shape

    def separate_tensor(
        self, audio: torch.Tensor, *, sr: int
    ) -> tuple[None, dict[str, torch.Tensor]]:
        assert sr == self.samplerate
        assert tuple(audio.shape) == (2, 20)
        return None, {"vocals": torch.ones(self._stem_shape, dtype=torch.float32)}


@pytest.mark.parametrize("stem_shape", [(2, 20), (1, 2, 20)])
def test_demucs_separator_normalizes_supported_stem_shapes(
    monkeypatch: pytest.MonkeyPatch,
    stem_shape: tuple[int, ...],
) -> None:
    fake_separator = _FakeDemucsApiSeparator(stem_shape)
    monkeypatch.setattr(
        "karakara.separator.demucs.impl._get_demucs_separator",
        lambda **_kwargs: fake_separator,
    )

    stems = DemucsSeparator().separate(np.zeros((2, 20), dtype=np.float32))

    assert stems["vocals"].shape == (2, 20)
    np.testing.assert_array_equal(stems["vocals"], np.ones((2, 20), dtype=np.float32))
