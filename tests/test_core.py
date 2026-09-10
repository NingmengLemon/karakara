"""`gen_kara()` 的失败降级与音频形状回归测试。"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
import soundfile as sf
from lemony_lrc_parser import Lyrics

from karakara.aligner.abc import AbstractAligner, AlignedWord, LangCode
from karakara.core import gen_kara
from karakara.preprocess import AudioPreprocessConfig
from karakara.separator.abc import AbstractStemSeparator
from karakara.utils.metadata import MetadataFilter

#: 测试用的分离采样率。刻意取一个非默认值（不是 44100），这样"主进程到底有
#: 没有从音轨文件读采样率"是能被测出来的。
TEST_SAMPLE_RATE = 10


class FakeSeparator(AbstractStemSeparator):
    """把预先准备好的数组写成 32 位浮点 WAV 当作人声轨。

    接口是「文件进、文件出」，所以假实现也必须真的落盘——这样 `_preprocess_vocals`
    的读取路径（含采样率嗅探）才真的被覆盖到。
    """

    VOCAL_STEM_NAME = "vocals"

    def __init__(self, vocal: np.ndarray) -> None:
        self._vocal = vocal
        self.calls: list[tuple[Path, Path, list[str] | None]] = []

    def separate(
        self,
        audio_path: str | Path,
        dest_dir: str | Path,
        *,
        stems: Sequence[str] | None = None,
    ) -> dict[str, Path]:
        requested = list(stems) if stems is not None else [self.VOCAL_STEM_NAME]
        assert self.VOCAL_STEM_NAME in requested, (
            f"主进程必须索取人声轨，实际索取: {requested}"
        )
        dest = Path(dest_dir)
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / f"{self.VOCAL_STEM_NAME}.wav"
        vocal = self._vocal
        data = vocal.T if vocal.ndim == 2 else vocal
        sf.write(str(path), data, TEST_SAMPLE_RATE, subtype="FLOAT")
        self.calls.append((Path(audio_path), dest, requested))
        return {self.VOCAL_STEM_NAME: path}


class FakeAligner(AbstractAligner):
    def __init__(
        self, align_impl: Callable[[np.ndarray, str], list[AlignedWord]]
    ) -> None:
        self._align_impl = align_impl
        self.received_audio: list[np.ndarray] = []
        self.received_languages: list[str | None] = []

    def align(
        self,
        audio: np.ndarray,
        text: str,
        sample_rate: int = 44100,
        *,
        language: str | None = None,
    ) -> list[AlignedWord]:
        assert sample_rate == 10
        self.received_audio.append(audio)
        self.received_languages.append(language)
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


def _run_single_line(
    monkeypatch: pytest.MonkeyPatch,
    metadata_filter: MetadataFilter,
    *,
    lyrics: str,
    aligner_language: Literal["auto"] | LangCode = "auto",
    target_lang: LangCode | None = None,
) -> FakeAligner:
    """跑一次单行对齐，返回收集到调用的 FakeAligner。

    显式声明透传的参数而不是 ``**kwargs: object``：后者对类型检查器是
    ``object``，等于把类型信息丢掉了（ty / mypy 都会报参数类型不符），
    只能靠 ``type: ignore`` 压住。
    """
    aligner = FakeAligner(lambda _audio, text: [AlignedWord(text, (0, 200))])
    gen_kara(
        Lyrics.loads(lyrics),
        "ignored.wav",
        aligner=aligner,
        separator=FakeSeparator(np.ones((2, 200), np.float32)),
        metadata_filter=metadata_filter,
        preprocess_config=AudioPreprocessConfig(
            normalize=False, suppress_vibrato=False, compress=False
        ),
        offset_ms=0,
        min_vocal_activity=0,
        aligner_language=aligner_language,
        target_lang=target_lang,
    )
    return aligner


def test_detects_japanese_song_and_passes_language_to_aligner(
    monkeypatch: pytest.MonkeyPatch, metadata_filter: MetadataFilter
) -> None:
    """日语歌词必须按 ja 送对齐器。

    回归测试：此前 ``Qwen3ForcedAligner`` 的语言在构造时硬编码为 Chinese，
    且 ``gen_kara`` 的语言判定结果从未被使用，导致日文歌全部按中文对齐。
    """
    aligner = _run_single_line(
        monkeypatch, metadata_filter, lyrics="[00:00.00]あいつも　こいつも"
    )

    assert aligner.received_languages == ["ja"]


def test_kanji_only_lines_do_not_flip_song_language(
    monkeypatch: pytest.MonkeyPatch, metadata_filter: MetadataFilter
) -> None:
    """纯汉字行（单行判据会误判为中文）不能把整首歌翻成 zh。"""
    aligner = _run_single_line(
        monkeypatch,
        metadata_filter,
        lyrics="[00:00.00]畜生\n[00:01.00]あいつもこいつも",
    )

    assert aligner.received_languages == ["ja", "ja"]


def test_aligner_language_can_be_forced(
    monkeypatch: pytest.MonkeyPatch, metadata_filter: MetadataFilter
) -> None:
    aligner = _run_single_line(
        monkeypatch,
        metadata_filter,
        lyrics="[00:00.00]hello",
        aligner_language="ja",
    )

    assert aligner.received_languages == ["ja"]


def test_target_lang_skips_other_languages(
    monkeypatch: pytest.MonkeyPatch, metadata_filter: MetadataFilter
) -> None:
    """target_lang 只过滤不对齐的行，不影响送给对齐器的语言判定。"""
    aligner = FakeAligner(lambda _audio, text: [AlignedWord(text, (0, 200))])

    result = gen_kara(
        Lyrics.loads("[00:00.00]hello\n[00:01.00]あいつも"),
        "ignored.wav",
        aligner=aligner,
        separator=FakeSeparator(np.ones((2, 200), np.float32)),
        metadata_filter=metadata_filter,
        preprocess_config=AudioPreprocessConfig(
            normalize=False, suppress_vibrato=False, compress=False
        ),
        offset_ms=0,
        min_vocal_activity=0,
        target_lang="ja",
    )

    assert aligner.received_languages == ["ja"]
    assert _line_text(result, 0) == "hello"
    assert _line_text(result, 1) == "あいつも"
