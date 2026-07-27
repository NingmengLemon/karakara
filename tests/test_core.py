"""`gen_kara()` 的失败降级与音频形状回归测试。"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from lemony_lrc_parser import Lyrics

from karakara.aligner.abc import AbstractAligner, AlignedWord
from karakara.core import gen_kara
from karakara.preprocess import AudioPreprocessConfig
from karakara.separator.abc import AbstractStemSeparator
from karakara.utils.metadata import MetadataFilter


class FakeSeparator(AbstractStemSeparator):
    VOCAL_STEM_NAME = "vocals"

    @property
    def samplerate(self) -> int:
        return 10

    def __init__(self, vocal: np.ndarray) -> None:
        self._vocal = vocal

    def separate(self, audio: np.ndarray) -> dict[str, np.ndarray]:
        return {self.VOCAL_STEM_NAME: self._vocal}


class FakeAligner(AbstractAligner):
    def __init__(
        self, align_impl: Callable[[np.ndarray, str], list[AlignedWord]]
    ) -> None:
        self._align_impl = align_impl
        self.received_audio: list[np.ndarray] = []

    def align(
        self, audio: np.ndarray, text: str, sample_rate: int = 44100
    ) -> list[AlignedWord]:
        assert sample_rate == 10
        self.received_audio.append(audio)
        return self._align_impl(audio, text)


@pytest.fixture
def metadata_filter() -> MetadataFilter:
    return MetadataFilter(
        keywords=[],
        id3_tags=frozenset(),
        parenthetical_markers=[],
        detect_id3_tags=False,
        detect_parenthetical=False,
        detect_pure_numbers=False,
        custom_patterns=[],
    )


def _line_text(lyrics: Lyrics, index: int = 0) -> str:
    return "".join(token.content for token in lyrics[index].content)


def test_preserves_original_line_when_alignment_raises(
    monkeypatch: pytest.MonkeyPatch, metadata_filter: MetadataFilter
) -> None:
    monkeypatch.setattr(
        "karakara.core.load_audio",
        lambda _path, sample_rate: np.zeros((2, 20), np.float32),
    )
    separator = FakeSeparator(np.zeros((2, 20), np.float32))

    def raise_alignment(_audio: np.ndarray, _text: str) -> list[AlignedWord]:
        raise RuntimeError("aligner unavailable")

    result = gen_kara(
        Lyrics.loads("[00:00.00]hello"),
        "ignored.wav",
        aligner=FakeAligner(raise_alignment),
        separator=separator,
        metadata_filter=metadata_filter,
        offset_ms=0,
    )

    assert _line_text(result) == "hello"


def test_preserves_original_line_when_alignment_has_no_positions(
    monkeypatch: pytest.MonkeyPatch, metadata_filter: MetadataFilter
) -> None:
    monkeypatch.setattr(
        "karakara.core.load_audio",
        lambda _path, sample_rate: np.zeros((2, 20), np.float32),
    )
    separator = FakeSeparator(np.zeros((2, 20), np.float32))
    aligner = FakeAligner(lambda _audio, text: [AlignedWord(word=text)])

    result = gen_kara(
        Lyrics.loads("[00:00.00]hello"),
        "ignored.wav",
        aligner=aligner,
        separator=separator,
        metadata_filter=metadata_filter,
        offset_ms=0,
    )

    assert _line_text(result) == "hello"


def test_preserves_out_of_bounds_line_without_calling_aligner(
    monkeypatch: pytest.MonkeyPatch, metadata_filter: MetadataFilter
) -> None:
    monkeypatch.setattr(
        "karakara.core.load_audio",
        lambda _path, sample_rate: np.zeros((2, 10), np.float32),
    )
    separator = FakeSeparator(np.zeros((2, 10), np.float32))
    aligner = FakeAligner(lambda _audio, text: [AlignedWord(text, (0, 100))])

    result = gen_kara(
        Lyrics.loads("[00:02.00]hello"),
        "ignored.wav",
        aligner=aligner,
        separator=separator,
        metadata_filter=metadata_filter,
        preprocess_config=AudioPreprocessConfig(
            normalize=False, suppress_vibrato=False, compress=False
        ),
        offset_ms=0,
    )

    assert _line_text(result) == "hello"
    assert aligner.received_audio == []


def test_passes_all_vocal_channels_to_aligner(
    monkeypatch: pytest.MonkeyPatch, metadata_filter: MetadataFilter
) -> None:
    monkeypatch.setattr(
        "karakara.core.load_audio",
        lambda _path, sample_rate: np.zeros((2, 20), np.float32),
    )
    vocal = np.vstack((np.zeros(20, np.float32), np.ones(20, np.float32)))
    separator = FakeSeparator(vocal)
    aligner = FakeAligner(lambda _audio, text: [AlignedWord(text, (0, 100))])

    result = gen_kara(
        Lyrics.loads("[00:00.50]hello"),
        "ignored.wav",
        aligner=aligner,
        separator=separator,
        metadata_filter=metadata_filter,
        preprocess_config=AudioPreprocessConfig(
            normalize=False, suppress_vibrato=False, compress=False
        ),
        offset_ms=0,
    )

    assert aligner.received_audio[0].shape == (2, 15)
    np.testing.assert_array_equal(aligner.received_audio[0][1], np.ones(15, np.float32))
    assert _line_text(result) == "hello"


def test_skips_alignment_for_silent_lyric_line(
    monkeypatch: pytest.MonkeyPatch, metadata_filter: MetadataFilter
) -> None:
    monkeypatch.setattr(
        "karakara.core.load_audio",
        lambda _path, sample_rate: np.zeros((2, 20), np.float32),
    )
    separator = FakeSeparator(np.zeros((2, 20), np.float32))
    aligner = FakeAligner(lambda _audio, text: [AlignedWord(text, (0, 100))])

    result = gen_kara(
        Lyrics.loads("[00:00.00]silent line"),
        "ignored.wav",
        aligner=aligner,
        separator=separator,
        metadata_filter=metadata_filter,
        preprocess_config=AudioPreprocessConfig(
            normalize=False, suppress_vibrato=False, compress=False
        ),
        offset_ms=0,
        min_vocal_activity=0.01,
    )

    assert aligner.received_audio == []
    assert _line_text(result) == "silent line"


def test_vocal_activity_check_can_be_disabled(
    monkeypatch: pytest.MonkeyPatch, metadata_filter: MetadataFilter
) -> None:
    monkeypatch.setattr(
        "karakara.core.load_audio",
        lambda _path, sample_rate: np.zeros((2, 20), np.float32),
    )
    separator = FakeSeparator(np.zeros((2, 20), np.float32))
    aligner = FakeAligner(lambda _audio, text: [AlignedWord(text, (0, 100))])

    gen_kara(
        Lyrics.loads("[00:00.00]silent line"),
        "ignored.wav",
        aligner=aligner,
        separator=separator,
        metadata_filter=metadata_filter,
        preprocess_config=AudioPreprocessConfig(
            normalize=False, suppress_vibrato=False, compress=False
        ),
        offset_ms=0,
        min_vocal_activity=0,
    )

    assert len(aligner.received_audio) == 1


def test_preserves_existing_byword_line_when_configured(
    monkeypatch: pytest.MonkeyPatch, metadata_filter: MetadataFilter
) -> None:
    monkeypatch.setattr(
        "karakara.core.load_audio",
        lambda _path, sample_rate: np.ones((2, 20), np.float32),
    )
    source = Lyrics.loads("[00:00.00]he[00:00.10]llo[00:00.20]")
    separator = FakeSeparator(np.ones((2, 20), np.float32))
    aligner = FakeAligner(lambda _audio, text: [AlignedWord(text, (0, 200))])

    result = gen_kara(
        source,
        "ignored.wav",
        aligner=aligner,
        separator=separator,
        metadata_filter=metadata_filter,
        preprocess_config=AudioPreprocessConfig(
            normalize=False, suppress_vibrato=False, compress=False
        ),
        offset_ms=0,
        min_vocal_activity=0,
        existing_byword_policy="preserve",
    )

    assert len(result[0].content) == 2
    assert aligner.received_audio == []


def test_realigns_existing_byword_line_by_default(
    monkeypatch: pytest.MonkeyPatch, metadata_filter: MetadataFilter
) -> None:
    monkeypatch.setattr(
        "karakara.core.load_audio",
        lambda _path, sample_rate: np.ones((2, 20), np.float32),
    )
    source = Lyrics.loads("[00:00.00]he[00:00.10]llo[00:00.20]")
    separator = FakeSeparator(np.ones((2, 20), np.float32))
    aligner = FakeAligner(lambda _audio, text: [AlignedWord(text, (0, 200))])

    result = gen_kara(
        source,
        "ignored.wav",
        aligner=aligner,
        separator=separator,
        metadata_filter=metadata_filter,
        preprocess_config=AudioPreprocessConfig(
            normalize=False, suppress_vibrato=False, compress=False
        ),
        offset_ms=0,
        min_vocal_activity=0,
    )

    assert len(aligner.received_audio) == 1
    assert len(result[0].content) == 1
    assert _line_text(result) == "hello"
