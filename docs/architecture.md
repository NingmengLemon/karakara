# 架构：两个外部 worker，主环境不带 torch

分离与对齐各自跑在**独立进程**里，主程序只通过进程边界与它们交互：

| 角色 | 位置 | 依赖怎么来的 |
|---|---|---|
| 对齐 | `scripts/qwen3aligner_server.py`（HTTP 服务） | 脚本头部的 PEP 723 内联依赖，`uv run --script` 自动准备 |
| 分离 | `scripts/separator_worker.py`（子进程 + 行分隔 JSON） | 同上 |

这样做的三个直接收益：

- **主环境极小。** Demucs 依赖 PyTorch，CUDA 版单独就占约 2.7GB（实测占原虚拟环境总大小的 77%）。分离移出主进程后，运行时依赖闭包只剩 11 个包：`av / eval-type-backport / lemony-lrc-parser / numpy / pydantic / pypinyin / regex / requests / soundfile / typing-extensions / zhconv`。
- **版本诉求不再打架。** 例如分离侧要 `numpy>=2`，主程序锁 `numpy<2`，进程隔离后各用各的。
- **不再强制 GPU。** 主进程不初始化 CUDA 上下文；分离 worker 可以指向另一台机器或另一个环境。

## worker 协议

`SubprocessStemSeparator` 与 worker 之间是 **stdin/stdout 行分隔 JSON**，stderr 直接继承（worker 日志与 Demucs 进度条从那里透出）：

```jsonc
// 请求
{"id": 1, "cmd": "separate", "audio": "<abs>", "dest_dir": "<abs>",
 "stems": ["vocals"], "model": "htdemucs_6s", "device": "cuda:0"}
{"id": 2, "cmd": "info"}
{"id": 3, "cmd": "shutdown"}
// 响应
{"id": 1, "ok": true, "stems": {"vocals": "<abs>"}, "samplerate": 44100, "elapsed_s": 12.3}
{"id": 1, "ok": false, "error": "<message>"}
```

要点：

- **进程常驻、模型只加载一次。** 刻意不做「每首歌 popen 一次」——实测每次冷启动约 2.6 秒（主要是 `import torch`），几十首的批量会白白多花一两分钟。
- **worker 的 stdout 只允许协议 JSON。** 分离时会把 stdout 临时重定向到 stderr，隔离第三方库乱打印。
- **音轨写成 32 位浮点 WAV。** 16 位量化会在下游归一化/颤音抑制**之前**就引入量化噪声。
- **单条请求失败不会杀掉 worker。** `SystemExit` 也会被拦下转成 `ok:false`（demucs 在缺 `diffq` 时会 `sys.exit(1)`，`except Exception` 抓不到它）。
- 退出走 `shutdown` + 关闭 stdin（EOF 兜底），不依赖杀进程。
- 请求默认有超时（见 README 的开关表）；默认值有限是刻意的——一个卡住的 worker 不该让批处理永久挂住。

## 分离后端选择

`--separator-backend` 只决定用哪个 worker 脚本，主程序本身完全不知道后端是什么。

- `demucs`（默认，模型默认 `UVR_Demucs_Model_1`）：模型在 `models/sep/Demucs_Models/v3_v4_repo`（该目录是指向 UVR5 GUI 的 junction，**只读**）。
- `audio-separator`：`uv run --script scripts/separator_worker_audio_separator.py`，需要 `--separator-model` 指定模型文件名。它的价值是**人声质量**（MDX-Net / VR-Arch / MDX23C / RoFormer / ensemble 预设），**不是**依赖体积——它比只用 demucs 更重（多出 librosa / scipy / onnx / onnx2torch / resampy 等，而且同样无条件需要 torch；实测内联依赖下载约 240MB）。它的 `--model-dir` 必须是**扁平**目录；UVR5 GUI 的嵌套布局它不认，worker 也不会写入 `models/sep`。

两个 worker 的模型目录都按**项目根目录**定位（不是 CWD），因为 worker 是被主程序以继承来的工作目录拉起的。

## 代码布局

CLI 只做参数解析与装配，逻辑都在包里（`src/karakara/`），因此都能被直接测试。

| 位置 | 职责 |
|---|---|
| `main.py` | 参数表 + 单文件/批处理两种模式的编排（`build_parser` / `run_single` / `run_batch`） |
| `karakara/core.py` | 流水线主体：预处理 → 偏移 → 逐行切段 → 对齐 → 词级产物 |
| `karakara/backends.py` | 后端登记表（分离/对齐）、默认地址、**对齐后端的语言能力** |
| `karakara/batch.py` | 批处理：配对发现、跑一组输入、逐项回收 |
| `karakara/offset.py` | 全局偏移估计与两道校验 |
| `karakara/preprocess.py` | 归一化 / 颤音抑制 / 可选压缩 |
| `karakara/aligner/` | 对齐器契约与后端适配（`q3fa` 是那套 HTTP `/align` 契约的客户端，HubertFA 也走它） |
| `karakara/separator/` | 分离器契约与子进程 worker 适配 |
| `karakara/utils/` | 音频 IO、LRC 读写、语言判定、元数据过滤 |
| `karakara/interactive.py` | 交互模式的文件对话框（tkinter） |
| `scripts/` | 跑在**独立环境**里的 worker 与服务（PEP 723 内联依赖） |
| `third_party/` | 上游代码的 git submodule（HubertFA；权重在 gitignored 的 `models/`） |
