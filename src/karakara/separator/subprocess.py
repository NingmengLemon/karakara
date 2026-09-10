from __future__ import annotations

import atexit
import json
import os
import shutil
import subprocess
import threading
from collections.abc import Mapping, Sequence
from logging import getLogger
from pathlib import Path
from queue import Empty, Queue
from typing import Any

from karakara.separator.abc import AbstractStemSeparator, StemSeparationError

logger = getLogger(__name__)

#: worker 在 argv 与协议中都使用 UTF-8；Windows 上必须显式指定，否则默认走
#: locale 编码（GBK/CP932），中文路径与日志都会出问题。
_ENCODING = "utf-8"

#: 优雅退出（发送 shutdown 后）等待 worker 结束的秒数。
_SHUTDOWN_TIMEOUT_S = 10.0

#: 启动 worker 时允许的最大等待秒数（仅用于校验进程是否立刻退出）。
_STARTUP_GRACE_S = 0.5

#: 给 uv 管理的 worker 使用的默认命令。
DEFAULT_WORKER_SCRIPT = Path("scripts/separator_worker.py")


def _default_command() -> list[str]:
    """构造默认的 worker 启动命令。

    ``uv run --script`` 会按脚本头部的 PEP 723 内联依赖准备一个缓存环境，
    因此主环境的依赖里不需要出现 torch / demucs。
    """
    return ["uv", "run", "--script", str(DEFAULT_WORKER_SCRIPT)]


class _WorkerProtocolError(StemSeparationError):
    """worker 违反了行协议（输出非法 JSON 或使用了非法结构）。"""


class _WorkerReader(threading.Thread):
    """把 worker stdout 的每一行塞进队列，并负责识别 EOF。

    用独立线程而不是直接阻塞 ``readline()``，是为了让请求可以有超时，并且在
    worker 被 OOM killer 之类干掉时能立刻拿到"流已结束"而不是永久挂住。
    """

    _EOF = object()

    def __init__(self, stream: Any) -> None:
        super().__init__(name="separator-worker-reader", daemon=True)
        self._stream = stream
        self._lines: Queue[Any] = Queue()
        # 注意不要命名为 ``_started``：那是 ``threading.Thread`` 自己用的内部
        # Event，遮蔽它会让 ``start()`` 直接崩掉。
        self._started_once = False

    def run(self) -> None:
        try:
            for line in self._stream:
                self._lines.put(line)
        except Exception as exc:  # noqa: BLE001 - 读线程绝不能把异常抛给解释器
            logger.debug(f"worker stdout reader stopped: {exc}")
        finally:
            self._lines.put(self._EOF)

    def start_once(self) -> None:
        if not self._started_once:
            self.start()
            self._started_once = True

    def read_line(self, timeout: float | None) -> str | None:
        """读取一行；返回 ``None`` 表示 worker 的 stdout 已关闭。"""
        self.start_once()
        try:
            item = self._lines.get(timeout=timeout)
        except Empty:
            raise TimeoutError("等待分离 worker 响应超时") from None
        if item is self._EOF:
            # 放回哨兵，后续读取仍然立刻得知 EOF
            self._lines.put(self._EOF)
            return None
        return str(item)


class SubprocessStemSeparator(AbstractStemSeparator):
    """通过在独立进程中运行 worker 脚本来分离音轨。

    为什么必须另起进程，而不是 ``import demucs``：

    * Demucs 依赖 PyTorch，而 CUDA 版 PyTorch 单独就占约 2.7GB（本项目实测占
      虚拟环境总大小的 77%）。走子进程后主环境的依赖里不再需要 torch/demucs，
      整个环境可以降到百兆级，也不再强制要求 GPU。
    * 分离依赖与主程序依赖的版本诉求会打架（例如 numpy 大版本），进程隔离后
      两边各用各的，互不影响。
    * 主进程不再初始化 CUDA 上下文，批量处理时显存/内存都更干净。

    模型在 worker 内**加载一次并复用**：进程按需启动、跑完整批任务、最后优雅
    退出。刻意不做"每首歌 popen 一次"——实测每次冷启动约 2.6 秒（主要是
    ``import torch``），几十首的批量就会白白多花一两分钟。

    协议：stdin/stdout 走行分隔 JSON，stderr 直接继承给父进程（worker 的日志与
    Demucs 进度条都从这里透出）。
    """

    def __init__(
        self,
        command: Sequence[str] | None = None,
        *,
        model: str | None = None,
        device: str | None = None,
        model_dir: str | Path | None = None,
        stems: Sequence[str] | None = None,
        request_timeout: float | None = None,
        extra_env: Mapping[str, str] | None = None,
    ) -> None:
        """
        Args:
            command: worker 启动命令。``None`` 时依次尝试环境变量
                ``KARAKARA_SEPARATOR_CMD``，再退回 ``uv run --script``。
            model: 传给 worker 的模型名；``None`` 用 worker 自身默认值。
            device: 传给 worker 的设备；``None`` 用 worker 自身默认值。
            model_dir: 模型仓库目录；``None`` 用 worker 自身默认值。
            stems: 默认只保留这些音轨。``None`` 表示 worker 自行决定。
            request_timeout: 单次分离请求的超时秒数；``None`` 表示不超时。
            extra_env: 追加到 worker 环境变量上的额外项。
        """
        env_command = os.environ.get("KARAKARA_SEPARATOR_CMD")
        if command is not None:
            self._command = list(command)
        elif env_command:
            self._command = split_command(env_command)
        else:
            self._command = _default_command()

        self._model = model
        self._device = device
        self._model_dir = str(model_dir) if model_dir is not None else None
        self._stems = list(stems) if stems is not None else None
        self._request_timeout = request_timeout
        self._extra_env = dict(extra_env or {})

        self._process: subprocess.Popen[str] | None = None
        self._reader: _WorkerReader | None = None
        self._next_id = 0
        self._lock = threading.Lock()
        self._closed = False
        atexit.register(self.close)

    # ---------- 生命周期 ----------

    @property
    def command(self) -> list[str]:
        """当前使用的 worker 启动命令。"""
        return list(self._command)

    def _spawn(self) -> None:
        executable = self._command[0]
        if shutil.which(executable) is None and not Path(executable).exists():
            raise StemSeparationError(
                f"找不到分离 worker 的可执行文件 {executable!r}；"
                f"可用 --separator-cmd 指定，或设置 KARAKARA_SEPARATOR_CMD"
            )

        env = os.environ.copy()
        env.update(self._extra_env)
        env.setdefault("PYTHONIOENCODING", _ENCODING)
        env.setdefault("PYTHONUTF8", "1")

        logger.info(f"starting separator worker: {' '.join(self._command)}")
        try:
            # stderr 继承给父进程：worker 的日志与 Demucs 进度条直接可见
            process = subprocess.Popen(
                self._command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=None,
                text=True,
                encoding=_ENCODING,
                bufsize=1,
                env=env,
            )
        except OSError as exc:
            raise StemSeparationError(f"启动分离 worker 失败: {exc}") from exc

        assert process.stdout is not None
        self._process = process
        self._reader = _WorkerReader(process.stdout)

    def _ensure_started(self) -> None:
        if self._closed:
            raise StemSeparationError("SubprocessStemSeparator 已关闭")
        if self._process is None:
            self._spawn()
        elif self._process.poll() is not None:
            raise StemSeparationError(
                f"分离 worker 已退出（returncode={self._process.returncode}）"
            )

    def close(self) -> None:
        """请求 worker 优雅退出；必要时强制终止。可重复调用。"""
        process, self._process = self._process, None
        reader, self._reader = self._reader, None
        self._closed = True
        if process is None:
            return

        try:
            if process.poll() is None and process.stdin is not None:
                self._next_id += 1
                try:
                    process.stdin.write(
                        json.dumps({"id": self._next_id, "cmd": "shutdown"}) + "\n"
                    )
                    process.stdin.flush()
                except (BrokenPipeError, ValueError, OSError):
                    pass
            # 关闭 stdin 让 worker 在 EOF 时也能自行退出（含 uv 这一层）
            if process.stdin is not None:
                try:
                    process.stdin.close()
                except OSError:
                    pass
            try:
                process.wait(timeout=_SHUTDOWN_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                logger.warning("separator worker 未在超时内退出，强制终止")
                process.terminate()
                try:
                    process.wait(timeout=_SHUTDOWN_TIMEOUT_S)
                except subprocess.TimeoutExpired:
                    process.kill()
        finally:
            for stream in (process.stdout,):
                if stream is not None:
                    try:
                        stream.close()
                    except OSError:
                        pass
            if reader is not None:
                reader.start_once()

    # ---------- 协议 ----------

    def _reap(self) -> int | None:
        """等 worker 结束并返回真实退出码；超时则退回 ``poll()`` 的结果。"""
        process = self._process
        if process is None:
            return None
        try:
            return process.wait(timeout=_SHUTDOWN_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            return process.poll()

    def _request(self, payload: dict[str, Any], timeout: float | None) -> dict[str, Any]:
        assert self._process is not None
        assert self._process.stdin is not None
        assert self._reader is not None

        self._next_id += 1
        request_id = self._next_id
        payload = {"id": request_id, **payload}

        try:
            self._process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self._process.stdin.flush()
        except (BrokenPipeError, ValueError, OSError) as exc:
            raise StemSeparationError(f"向分离 worker 发送请求失败: {exc}") from exc

        while True:
            try:
                line = self._reader.read_line(timeout)
            except TimeoutError as exc:
                raise StemSeparationError(str(exc)) from exc
            if line is None:
                # stdout 已关闭：把进程 reap 掉才能拿到真实退出码。不等的话
                # poll() 往往还返回 None，报错信息会变成没用的 "returncode=None"。
                code = self._reap()
                raise StemSeparationError(
                    f"分离 worker 提前退出（returncode={code}），未返回结果"
                )
            line = line.strip()
            if not line:
                continue
            try:
                response = json.loads(line)
            except json.JSONDecodeError as exc:
                # 以 "{" 开头却不是合法 JSON：协议真的坏了，直接报错。
                # 其它情况更可能是第三方库往 stdout 漏了一行日志，跳过并告警即可
                # ——如果它其实是本该来的响应，后面的 EOF 检测会报"worker 提前退出"。
                if line.startswith("{"):
                    raise _WorkerProtocolError(
                        f"worker 输出的 JSON 无法解析: {line[:200]!r}"
                    ) from exc
                logger.warning(f"忽略 worker stdout 上的非协议输出: {line[:200]!r}")
                continue
            if not isinstance(response, dict):
                logger.warning(f"忽略 worker stdout 上的非对象 JSON: {line[:200]!r}")
                continue
            if response.get("id") != request_id:
                logger.warning(
                    f"忽略 id 不匹配的 worker 响应: "
                    f"expected={request_id}, got={response.get('id')!r}"
                )
                continue
            if not response.get("ok", False):
                raise StemSeparationError(
                    f"worker 分离失败: {response.get('error', 'unknown error')}"
                )
            return response

    # ---------- 公共 API ----------

    def separate(
        self,
        audio_path: str | Path,
        dest_dir: str | Path,
        *,
        stems: Sequence[str] | None = None,
    ) -> dict[str, Path]:
        audio = Path(audio_path).resolve()
        if not audio.is_file():
            raise StemSeparationError(f"源音频不存在: {audio}")
        dest = Path(dest_dir).resolve()
        dest.mkdir(parents=True, exist_ok=True)

        requested = list(stems) if stems is not None else self._stems

        with self._lock:
            self._ensure_started()
            payload: dict[str, Any] = {
                "cmd": "separate",
                "audio": str(audio),
                "dest_dir": str(dest),
            }
            if self._model is not None:
                payload["model"] = self._model
            if self._device is not None:
                payload["device"] = self._device
            if self._model_dir is not None:
                payload["model_dir"] = self._model_dir
            if requested is not None:
                payload["stems"] = requested

            response = self._request(payload, self._request_timeout)

        raw_stems = response.get("stems")
        if not isinstance(raw_stems, dict) or not raw_stems:
            raise _WorkerProtocolError(f"worker 未返回任何音轨: {response!r}")

        resolved: dict[str, Path] = {}
        for name, value in raw_stems.items():
            path = Path(str(value))
            if not path.is_file():
                raise _WorkerProtocolError(
                    f"worker 声称已写出音轨 {name!r}，但文件不存在: {path}"
                )
            resolved[str(name)] = path

        if self.VOCAL_STEM_NAME not in resolved:
            raise StemSeparationError(
                f"worker 结果缺少 {self.VOCAL_STEM_NAME!r} 音轨，"
                f"实际得到: {sorted(resolved)}"
            )

        elapsed = response.get("elapsed_s")
        logger.info(
            f"separated {audio.name} -> {sorted(resolved)}"
            + (f" in {elapsed:.1f}s" if isinstance(elapsed, (int, float)) else "")
        )
        return resolved

    def info(self) -> dict[str, Any]:
        """询问 worker 的后端信息（后端名、可用模型、设备等）。"""
        with self._lock:
            self._ensure_started()
            response = self._request({"cmd": "info"}, self._request_timeout)
        return response


def split_command(command: str) -> list[str]:
    """切分 ``KARAKARA_SEPARATOR_CMD`` 这类命令字符串。

    Windows 路径里全是反斜杠，所以不能用 ``posix=True``（它会把 ``\\`` 当转义符
    吃掉）。改用 ``posix=False`` 再自己剥掉成对的引号。
    """
    import shlex

    parts = shlex.split(command, posix=False)
    stripped: list[str] = []
    for part in parts:
        if len(part) >= 2 and part[0] == part[-1] and part[0] in "\"'":
            part = part[1:-1]
        stripped.append(part)
    return [part for part in stripped if part]
