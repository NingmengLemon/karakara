"""批处理：发现「同目录同名的 LRC + 音频」对，并把每一对跑成产物。

这个模块刻意不依赖 ``argparse``：CLI 只负责把参数解成这里的入参，于是
「发现规则」和「跑一组输入」都能被直接测试与复用。
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path

from .aligner import HttpAligner
from .core import ExistingBywordPolicy, gen_kara
from .preprocess import AudioPreprocessConfig
from .separator import AbstractStemSeparator
from .utils.lrc import load_lyrics, save_lyrics
from .utils.metadata import MetadataFilter

#: 视为「音频」的后缀。刻意与 README/`--audio` 的说法一致（wav/mp3/flac/m4a）。
AUDIO_SUFFIXES = frozenset({".wav", ".mp3", ".flac", ".m4a"})


@dataclass(frozen=True)
class BatchJob:
    """一组同名歌词与音频的批处理任务。"""

    lyrics_path: Path
    audio_path: Path
    output_path: Path


@dataclass(frozen=True)
class UnpairedLyrics:
    """一个找不到唯一同名音频、因而无法处理的 LRC。"""

    lyrics_path: Path
    reason: str


@dataclass(frozen=True)
class DiscoveryResult:
    """批处理发现的结果：可处理的配对 + 被跳过的 LRC。"""

    jobs: list[BatchJob]
    skipped: list[UnpairedLyrics]


def discover_batch_jobs(
    input_dir: Path,
    output_dir: Path | None = None,
    *,
    strict: bool = False,
) -> DiscoveryResult:
    """递归发现同目录、同文件名的 LRC/音频对。

    已生成的 ``*.kara.lrc`` 会被排除。找不到音频或同名音频不唯一的 LRC 默认
    **跳过并汇总**（``strict=True`` 时改为直接抛 ``ValueError``）。

    默认必须宽松：真实曲库里总会有少量 LRC 没有配套音频（人工泄漏、翻译稿、
    只有 instrumental 等）。实测在 6622 首的曲库上有 86 个这样的文件——若按
    「一个坏配对就抛异常」，整个批处理连能配对的 6536 首都不会处理。
    """
    if not input_dir.is_dir():
        raise ValueError(f"batch directory does not exist: {input_dir}")

    # 每个目录只枚举一次：大曲库里几千个 LRC 常挤在少数几个目录（实测某曲库
    # 6600+ 个 LRC 集中在 ~1000 文件级的根目录），逐行 iterdir() 会把同一批目录
    # 条目重复枚举几百万次，Windows 上足以让「发现阶段」从秒级涨到分钟级。
    audio_index: dict[Path, dict[str, list[Path]]] = {}

    def audio_by_stem(directory: Path) -> dict[str, list[Path]]:
        index = audio_index.get(directory)
        if index is None:
            index = {}
            for entry in directory.iterdir():
                if entry.is_file() and entry.suffix.lower() in AUDIO_SUFFIXES:
                    index.setdefault(entry.stem, []).append(entry)
            for paths in index.values():
                paths.sort()
            audio_index[directory] = index
        return index

    jobs: list[BatchJob] = []
    skipped: list[UnpairedLyrics] = []
    for lyrics_path in sorted(input_dir.rglob("*.lrc")):
        if lyrics_path.stem.endswith(".kara"):
            continue
        candidates = audio_by_stem(lyrics_path.parent).get(lyrics_path.stem, [])
        if len(candidates) != 1:
            description = (
                "none"
                if not candidates
                else ", ".join(str(path) for path in candidates)
            )
            if strict:
                raise ValueError(
                    f"Expected exactly one audio file for {lyrics_path}, "
                    f"found: {description}"
                )
            skipped.append(UnpairedLyrics(lyrics_path, description))
            continue
        if output_dir is None:
            output_path = lyrics_path.with_suffix(".kara.lrc")
        else:
            relative = lyrics_path.relative_to(input_dir)
            output_path = output_dir / relative.with_suffix(".kara.lrc")
        jobs.append(BatchJob(lyrics_path, candidates[0], output_path))
    return DiscoveryResult(jobs=jobs, skipped=skipped)


def process_job(
    job: BatchJob,
    *,
    aligner: HttpAligner,
    separator: AbstractStemSeparator,
    metadata_filter: MetadataFilter,
    preprocess_config: AudioPreprocessConfig,
    dump_dir: Path | None,
    separate_work_dir: Path | None,
    offset_ms: float | None,
    min_vocal_activity: float,
    existing_byword_policy: ExistingBywordPolicy,
    aligner_language: str = "auto",
    target_lang: str | None = None,
    refine_collapsed_words: bool = False,
) -> None:
    """处理一组输入，并在函数返回时释放该任务的大型音频对象。"""
    lyrics = load_lyrics(job.lyrics_path)
    aligned = gen_kara(
        lyrics,
        job.audio_path,
        aligner=aligner,
        separator=separator,
        metadata_filter=metadata_filter,
        preprocess_config=preprocess_config,
        dump_dir=dump_dir,
        separate_work_dir=separate_work_dir,
        offset_ms=offset_ms,
        min_vocal_activity=min_vocal_activity,
        existing_byword_policy=existing_byword_policy,
        aligner_language=aligner_language,
        target_lang=target_lang,
        refine_collapsed_words=refine_collapsed_words,
    )
    save_lyrics(aligned, job.output_path)


def release_item_resources() -> None:
    """回收单个任务产生的 CPU/GPU 临时对象。

    GPU 侧的资源由分离 worker 进程独自持有，主进程不初始化 CUDA 上下文，
    因此这里只需要回收 Python 对象。
    """
    gc.collect()
