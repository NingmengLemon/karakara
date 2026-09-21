# 主流程与架构

这一页讲清楚两件事：**一次运行里数据怎么流动**，以及**为什么有两块东西跑在别的进程里**。
想直接跑起来看 [README](../../README.md)；想改某个环节，这里有它在代码里的位置。

## 一次运行

```
源音频 ──> 分离人声（独立 worker 进程）──> 预处理 ──┬──> 全局偏移估计
                                                  └──> 逐行切段 ──> 对齐（HTTP 服务）──> 词级 LRC
```

行级 LRC 与音频按**同目录同名**配对；产物默认写成同目录下的 `<名字>.kara.lrc`。

各环节的现状：

| 环节 | 做什么 | 在哪 |
|---|---|---|
| 分离 | 从混音里取出人声轨，写 32 位浮点 WAV 到临时目录 | `separator/`，worker 见 `scripts/separator_worker*.py` |
| 预处理 | 响度归一化（默认开）、颤音抑制（默认开）、动态范围压缩（默认关） | `preprocess.py` |
| 全局偏移 | 估计一个常量偏移，过不了校验就不动 | `offset.py`，现状见 [offset.md](offset.md) |
| 逐行切段 | 每行取 `[本行起点, 下一行起点)` 交给对齐器 | `core._line_sample_range` |
| 对齐 | 音频片段 + 行文本 → 逐单元 `(文本, 起, 止)` | `aligner/`，现状见 [aligner.md](aligner.md) |
| 产物 | 合并零长度单元后序列化，foobar2000 兼容的逐字标签、3 位小数 | `utils/lrc.py`、`core._merge_zero_length_tokens` |

## 为什么要拆成两个独立进程

分离（Demucs 系）和对齐（HubertFA / qwen3）各自需要 PyTorch 与模型，而主程序不需要。
拆开之后：

- **主环境极小**：运行时依赖闭包只有 11 个包
  （`av / eval-type-backport / lemony-lrc-parser / numpy / pydantic / pypinyin / regex / requests / soundfile / typing-extensions / zhconv`）。
  CUDA 版 PyTorch 单独就占约 2.7GB，装上它主环境会大一个数量级。
- **版本诉求不再打架**：分离侧要 `numpy>=2`、主程序锁 `numpy<2`，进程隔离后各用各的。
- **不强制 GPU**：主进程不初始化 CUDA 上下文；两个 worker 都可以指向另一台机器或另一个环境。

环境怎么准备、CUDA 索引怎么配、缓存有什么坑，见 [environments.md](environments.md)。

## 与分离 worker 的协议

`SubprocessStemSeparator` 用 **stdin/stdout 行分隔 JSON** 跟 worker 说话；stderr 直接继承给
父进程（worker 日志与 Demucs 进度条从那里透出）。

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

几条契约，改 worker 时必须守住：

- **进程常驻，模型只加载一次。** 不做「每首歌起一次进程」：冷启动约 2.6 秒（主要是
  `import torch`），几十首的批量会白白多花一两分钟。
- **stdout 只允许出现协议 JSON。** 分离期间 worker 会把 stdout 临时重定向到 stderr，
  免得第三方库乱打印污染通道；父进程对非 JSON 行也会告警跳过。
- **单条请求失败不会杀掉 worker。** `SystemExit` 也会被拦下转成 `ok: false`
  （demucs 缺 `diffq` 时会 `sys.exit(1)`，`except Exception` 抓不到它）。
- **损坏的音频包跳过并告警**，全部包都解不出才报错。
- **退出走 `shutdown` + 关闭 stdin（EOF 兜底）**，不依赖杀进程。
- **单次请求默认 900s 超时**（`--separator-timeout`），一个卡住的 worker 不该让批处理永久挂住。

## 分离后端

`--separator-backend` 只决定用哪个 worker 脚本，主程序不知道后端是什么。

| 后端 | 脚本 | 说明 |
|---|---|---|
| `demucs`（默认） | `scripts/separator_worker.py` | 默认模型 `UVR_Demucs_Model_1`，实测比 `htdemucs_6s` 一致更好 |
| `audio-separator` | `scripts/separator_worker_audio_separator.py` | 人声质量取向（MDX-Net / VR-Arch / MDX23C / RoFormer / ensemble），依赖更重，需要 `--separator-model` 指定模型文件名 |

模型目录按**项目根目录**定位，不依赖 CWD。`demucs` 的模型在
`models/sep/Demucs_Models/v3_v4_repo`（该目录是指向 UVR5 GUI 的 junction，**只读**）；
`audio-separator` 的 `--model-dir` 必须是**扁平**目录，UVR5 GUI 的嵌套布局它不认。

**想比较分离质量**时用 `scripts/compare_separators.py`：它量泄漏、区间对比度、人声占位率三个
代理指标（没有干净人声的真值，所以不假装能算 SDR），所有配置共用同一个固定偏移以免被偏移
估计的差异污染。已测结果与失效过的度量见
[records/2026-09-12-separator-ab.md](../records/2026-09-12-separator-ab.md)。

## 代码布局

CLI 只做参数解析与装配，逻辑都在包里，因此都能被直接测试。

| 位置 | 职责 |
|---|---|
| `main.py` | 参数表 + 单文件/批处理两种模式的编排 |
| `karakara/core.py` | 流水线主体：预处理 → 偏移 → 逐行切段 → 对齐 → 产物 |
| `karakara/backends.py` | 后端登记表（分离/对齐）、默认地址、对齐后端的语言能力 |
| `karakara/batch.py` | 批处理：配对发现、跑一组输入、逐项回收 |
| `karakara/offset.py` | 全局偏移估计与两道校验 |
| `karakara/preprocess.py` | 归一化 / 颤音抑制 / 可选压缩 |
| `karakara/paths.py` | 仓库内文件定位（worker 脚本路径等，一律绝对路径） |
| `karakara/aligner/` | 对齐器契约与 HTTP `/align` 客户端（两个后端共用） |
| `karakara/separator/` | 分离器契约与子进程 worker 适配 |
| `karakara/utils/` | 音频 IO、LRC 读写、语言判定、元数据过滤 |
| `karakara/interactive.py` | 交互模式的文件对话框（tkinter，只在交互模式 import） |
| `scripts/` | 跑在**独立环境**里的 worker 与服务（PEP 723 内联依赖） |
| `third_party/` | 上游代码的 git submodule（HubertFA；权重在 gitignored 的 `models/`） |
