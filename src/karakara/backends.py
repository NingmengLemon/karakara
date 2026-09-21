"""后端登记表：分离与对齐各有哪些后端、默认地址、以及**语言能力**。

主程序只认**命令与地址**、不认后端：两侧都定义了后端无关的契约（分离是行分隔
JSON 的 stdin/stdout 协议，对齐是 HTTP ``/align``），所以"换后端"在代码路径上
只是换个脚本或换个端口。

这张表放在包本体而不是 ``main.py``，有两个理由：

* ``tests/`` 不必再用 importlib 的绕法加载 CLI 就能检查它；
* "某个后端不支持某个语言"这类**能力**信息属于后端本身，不属于命令行——
  命令行只是把它呈现给用户。

能力信息是**必需**的：``--aligner-language`` 的选项集合是两个后端语言集合的并集
（``yue``/``ko`` 只有 Qwen3 支持），没有这张表的话，选了 HubertFA 又指定 ``--aligner-language ko``
会一路跑到服务端才拿到 400——那时人声分离已经白跑完了。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .paths import repo_file
from .separator import SubprocessStemSeparator


@dataclass(frozen=True)
class SeparatorBackend:
    """一个分离后端：worker 脚本 + 一句人话说明。"""

    name: str
    script: str
    description: str


@dataclass(frozen=True)
class AlignerBackend:
    """一个对齐后端：服务脚本 + 默认地址 + 能处理的语言。"""

    name: str
    script: str
    default_url: str
    #: 该后端能处理的语言代码。主程序据此在开跑前拦截不受支持的语言。
    languages: frozenset[str]
    description: str


#: 分离后端 → worker 脚本。两者使用同一套行协议，所以主程序只认命令、不认后端。
SEPARATOR_BACKENDS: dict[str, SeparatorBackend] = {
    "demucs": SeparatorBackend(
        name="demucs",
        script="scripts/separator_worker.py",
        description="Demucs（默认模型 UVR_Demucs_Model_1）",
    ),
    "audio-separator": SeparatorBackend(
        name="audio-separator",
        script="scripts/separator_worker_audio_separator.py",
        description=(
            "audio-separator（人声质量更好，环境更重；需配合 --separator-model 指定模型文件名）"
        ),
    ),
}

#: 对齐后端 → (服务脚本, 默认地址, 语言能力)。两者提供**同一套 `/align` 契约**，
#: 所以主程序只认地址、不认后端：换后端 = 换个端口，代码路径完全一致。
#:
#: * ``hfa``  —— HubertFA（歌声专用，10ms 帧，逐字/逐词都有真实时长）。实测在
#:   Qwen 把歌词压扁的行上是对的，见 docs/aligner-backends.md §9。语言来自它自带的
#:   三份词典（`ds-zh-pinyin-lite` / `japanese_dict_full` / `ds_cmudict-07b`）。
#: * ``qwen3``—— Qwen3-ForcedAligner（多语言通用，边界量化在 80ms，零长度词多）。
ALIGNER_BACKENDS: dict[str, AlignerBackend] = {
    "hfa": AlignerBackend(
        name="hfa",
        script="scripts/hubertfa_aligner_server.py",
        default_url="http://127.0.0.1:8788",
        languages=frozenset({"zh", "ja", "en"}),
        description="HubertFA（歌声专用，10ms 帧，实测在 Qwen 压扁歌词的行上更准）",
    ),
    "qwen3": AlignerBackend(
        name="qwen3",
        script="scripts/qwen3aligner_server.py",
        default_url="http://127.0.0.1:8787",
        languages=frozenset({"zh", "ja", "en", "yue", "ko"}),
        description="Qwen3-ForcedAligner（多语言通用，80ms 量化、零长度词多）",
    ),
}

DEFAULT_SEPARATOR_BACKEND = "demucs"
DEFAULT_ALIGNER_BACKEND = "hfa"


class UnsupportedAlignerLanguage(ValueError):
    """所选对齐后端不支持请求的语言。"""


def aligner_backends_for(language: str) -> list[str]:
    """支持该语言的登记后端名，按名字排序。用来给出可执行的补救建议。"""
    return sorted(
        name
        for name, backend in ALIGNER_BACKENDS.items()
        if language in backend.languages
    )


def ensure_aligner_language(backend: str, language: str | None) -> None:
    """语言不受支持时**在开跑之前**报错。

    ``language`` 为 ``None`` 时不做检查，调用方用 ``None`` 表示 ``--aligner-language auto``：
    ``auto`` 由 :func:`karakara.utils.lang.detect_dominant_lang` 判定，值域只有
    ja/zh/en，是所有登记后端都支持的。这里刻意不去"猜"它会判成什么——猜错会把本来
    能跑的歌拦下来。

    Args:
        backend: ``--aligner-backend`` 的名字。
        language: 送给对齐器的语言代码；``None`` 表示由后端自行判定。

    Raises:
        UnsupportedAlignerLanguage: 该后端不支持这个语言。
    """
    if language is None:
        return
    supported = ALIGNER_BACKENDS[backend].languages
    if language in supported:
        return
    alternatives = aligner_backends_for(language)
    hint = (
        f"可用 --aligner-backend {' 或 '.join(alternatives)} 处理该语言"
        if alternatives
        else "没有任何登记后端支持该语言"
    )
    raise UnsupportedAlignerLanguage(
        f"对齐后端 {backend!r} 不支持语言 {language!r}"
        f"（支持: {', '.join(sorted(supported))}）。{hint}"
    )


def resolve_aligner_url(backend: str, explicit_url: str | None = None) -> str:
    """对齐服务地址：显式地址优先，否则按后端取默认值。"""
    if explicit_url:
        return explicit_url
    return ALIGNER_BACKENDS[backend].default_url


def resolve_timeout(value: float | None) -> float | None:
    """把 CLI 的超时值转成 ``requests``/``Popen`` 语义：``<=0`` 表示不超时。"""
    if value is None or value <= 0:
        return None
    return value


def build_separator(
    backend: str,
    *,
    command: list[str] | None = None,
    model: str | None = None,
    device: str | None = None,
    model_dir: str | Path | None = None,
    request_timeout: float | None = None,
) -> SubprocessStemSeparator:
    """按后端构造分离器。

    ``command`` 优先（便于接自有环境或远程 worker）；否则按 ``backend`` 选脚本，用
    ``uv run --script`` 拉起（worker 脚本头部自带 PEP 723 内联依赖）。

    脚本路径取**绝对路径**（:func:`karakara.paths.repo_file`）：相对路径由 uv 按当前
    工作目录解析，实测从仓库外运行时它会去找 ``E:\\scripts\\separator_worker.py``。
    """
    if command is None:
        script = repo_file(SEPARATOR_BACKENDS[backend].script)
        command = ["uv", "run", "--script", str(script)]
    return SubprocessStemSeparator(
        command=command,
        model=model,
        device=device,
        model_dir=model_dir,
        request_timeout=request_timeout,
    )
