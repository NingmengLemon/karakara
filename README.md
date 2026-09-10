# karakara

> **Kara**oke & [**KARA**KARA](https://moegirl.icu/KARAKARA)

只是 1 个神秘的 Playground (

> 已知一首歌的音频文件和行级别精度的歌词文件, 能否用程序做出词级别精度的歌词文件呢...?

## 它做什么

读入「音频 + 行级 LRC」，产出「词级逐字 LRC」：

```
源音频 ──> 人声分离 ──> 预处理(归一化/颤音抑制/可选压缩)
                        └──> 全局偏移估计
                        └──> 逐行切段 ──> 强制对齐 ──> 词级 LRC
```

唯一花了心思的部分是一个简陋的 lrc 歌词解析器: [lemony-lrc-parser](https://github.com/NingmengLemon/lemony-lrc-parser)，参考 [SPL 歌词格式](https://moriafly.com/standards/spl.html) (2025-11-14 版) ~~, 但其实并没有严格遵循~~

## 架构：两个外部 worker，主环境不带 torch

分离与对齐各自跑在**独立进程**里，主程序只通过进程边界与它们交互：

| 角色 | 位置 | 依赖怎么来的 |
|---|---|---|
| 对齐 | `scripts/qwen3aligner_server.py`（HTTP 服务） | 脚本头部的 PEP 723 内联依赖，`uv run --script` 自动准备 |
| 分离 | `scripts/separator_worker.py`（子进程 + 行分隔 JSON） | 同上 |

这样做的三个直接收益：

- **主环境极小。** Demucs 依赖 PyTorch，CUDA 版单独就占约 2.7GB（实测占原虚拟环境总大小的 77%）。分离移出主进程后，运行时依赖闭包只剩 11 个包：`av / eval-type-backport / lemony-lrc-parser / numpy / pydantic / pypinyin / regex / requests / soundfile / typing-extensions / zhconv`。
- **版本诉求不再打架。** 例如分离侧要 `numpy>=2`，主程序锁 `numpy<2`，进程隔离后各用各的。
- **不再强制 GPU。** 主进程不初始化 CUDA 上下文；分离 worker 可以指向另一台机器或另一个环境。

### worker 协议

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

## 用法

```bash
# 分离 worker 自检（不需要主程序）
uv run --script scripts/separator_worker.py --info
uv run --script scripts/separator_worker_audio_separator.py --list-models   # 见下方说明

# 单文件
uv run main.py -l song.lrc -a song.flac -o song.kara.lrc

# 批量（递归找同名 lrc/音频对）
uv run main.py --batch-dir /path/to/album --output-dir /path/to/out

# 已经有一个装好 demucs 的环境时，跳过 uv 的环境准备
uv run main.py --batch-dir ... --separator-cmd /path/to/python.exe scripts/separator_worker.py
# 或用环境变量
KARAKARA_SEPARATOR_CMD='"C:\sep-env\Scripts\python.exe" "scripts/separator_worker.py"'
```

对齐服务需要先起在 `http://localhost:8787`（`--aligner-url` 可改）。

常用开关：`--separator-backend {demucs,audio-separator}`、`--separator-model`（默认 `UVR_Demucs_Model_1`）、`--separator-device`、`--aligner-language {auto,zh,ja,en,...}`、`--target-lang`、`--min-vocal-activity`、`--offset` / `--no-offset-estimate`、`--dump-dir`、`--sep-work-dir`、`--existing-byword-policy`、`--fail-fast`。

### 分离后端选择

`--separator-backend` 只决定用哪个 worker 脚本，主程序本身完全不知道后端是什么。

- `demucs`（默认，模型默认 `UVR_Demucs_Model_1`）：模型在 `models/sep/Demucs_Models/v3_v4_repo`（该目录是指向 UVR5 GUI 的 junction，**只读**）。
- `audio-separator`：`uv run --script scripts/separator_worker_audio_separator.py`，需要 `--separator-model` 指定模型文件名。它的价值是**人声质量**（MDX-Net / VR-Arch / MDX23C / RoFormer / ensemble 预设），**不是**依赖体积——它比只用 demucs 更重（多出 librosa / scipy / onnx / onnx2torch / resampy 等，而且同样无条件需要 torch；实测内联依赖下载约 240MB）。它的 `--model-dir` 必须是**扁平**目录；UVR5 GUI 的嵌套布局它不认，worker 也不会写入 `models/sep`。

#### worker 环境里的 torch 版本（重要）

分离 worker 有自己的环境，它的 torch 由**脚本头部的 PEP 723 内联依赖**决定。若不加
索引，torch 会从 PyPI 解析到 CPU 版（`torch+x.y.z+cpu`），于是分离**跑在 CPU 上**、
慢一个数量级，而且不看日志根本发现不了。

仓库当前的 `scripts/separator_worker.py` **已经带上 PyTorch 的 CUDA 索引**，实测
`torch 2.14.0+cu130` / `device=cuda:0`（同一首歌：GPU 4.2s，CPU 22.3s）。随时可以用

```bash
uv run --script scripts/separator_worker.py --info
```

的 `device` 与 `torch` 字段确认；worker 在 CUDA 不可用时也会打印显式告警。

要换 CUDA 版本或改回 CPU，编辑脚本头部那两段的 `url` 或直接删掉它们：

```toml
# [[tool.uv.index]]
# name = "pytorch_cu130"
# url = "https://download.pytorch.org/whl/cu130"
# explicit = true
#
# [tool.uv.sources]
# torch = { index = "pytorch_cu130" }
```

> ⚠️ **改完索引必须清掉 worker 的缓存环境。** uv 对 PEP 723 脚本的环境是按**依赖
> 清单**做缓存键的，`[tool.uv]` 里的索引配置**不参与缓存键**。所以只改索引而不清理的话，
> uv 会照旧复用那个装着 CPU 版 torch 的环境，看起来像"改了没用"。用 `uv cache clean`
> 或删掉 uv 缓存下对应的 `separator-worker-*` 环境目录，再跑一次让它重建。

不想动仓库的话，也可以用 `--separator-cmd`（或 `KARAKARA_SEPARATOR_CMD`）指向任何已装
CUDA 版 torch 的解释器：

```bash
uv run main.py --batch-dir ... \
  --separator-cmd /path/to/cuda-env/python.exe scripts/separator_worker.py
```

### 人声质量对比

```bash
uv run python scripts/compare_separators.py --songs-dir /path/to/pairs \
  --configs demucs:htdemucs_6s demucs:UVR_Demucs_Model_1 \
  --worker-python .venv/Scripts/python.exe --out report.json
```

指标说明见脚本 docstring。要注意它们是**代理指标**，不是 SDR：没有「干净人声」的 ground truth，所以量的是与「切段送去对齐」这一目标直接相关的三件事——行区间外的泄漏、区间对比度、区间内可听见窗口占比。所有配置共用**同一个固定偏移**，以免指标被偏移估计的差异污染。

已经量到的结果（都在 CPU 上跑，质量对比与设备无关）：

| 对比 | 曲目数 | 结论 |
|---|---|---|
| `UVR_Demucs_Model_1` vs `htdemucs_6s` | 4 | **前者一致更好**：泄漏 0.481 vs 0.533、对比度 0.1771 vs 0.1650，而且更快（6.7s vs 8.0s）。这是零下载就能拿到的提升 |
| `audio-separator:UVR_MDXNET_KARA_2.onnx` vs `htdemucs_6s` | 2 | 基本持平（泄漏 0.312 vs 0.311）。注意 `UVR_MDXNET_KARA_2` 是**卡拉OK（去主唱）**模型，它的 "vocals" 是残差而非干净主唱，并不是这个用途的合适模型 |

要把「人声质量」这个决策真正定下来，还需要：**(a)** 明显更多的曲目；**(b)** 专用人声模型（BS-RoFormer 人声 SDR ≈12.9、VR HP-UVR 等），而不是卡拉OK模型；**(c)** GPU 以免每次测量都要等一分钟以上。

## 状态与已知问题

### 已修

- **全局偏移估计**（`src/karakara/offset.py`）。原实现把每行按 `[start, start+5000ms)` 的方块标记，密集歌词会让指示曲线退化成近乎全 1 的实心块，于是任何打分函数都只能对齐两条曲线的重心而非起唱点；而且用的是裸点积（随重叠长度单调增长，等于惩罚偏移量本身），平局时还会被扫描顺序推到区间极端。
  现在改成**行首窄窗** + **分母固定的判别式对比度** + **平局取更小偏移量**。在真实曲库（4 首，人声实机分离）上用独立判据（行区间能量对比度）裁决：**新实现 4/4 胜出**；旧实现在其中 3 首上给出的偏移**比完全不偏移还差**。
- **对齐语言**。此前 `Qwen3ForcedAligner` 的语言在构造时硬编码为 `Chinese`，`gen_kara` 算出的语言判定结果从未被使用，于是英文/日文歌全部按中文对齐。现在 `--aligner-language auto` 会按整首歌判定（假名出现即判日语——纯汉字行会被逐行判据误判为中文，这是关键原因），也可显式指定；`--target-lang` 用于跳过语言不符的行。
- **端口不一致**：`--aligner-url` 默认值曾指向 8787，而客户端/实现内部默认 8000。
- **`separator/demucs/patch.py` 已删除**：它 module-level 去 patch `demucs.states.load_model`，但全仓库没有任何地方 import 它，完全没生效。
- **`scripts/demucs_separator_server.py` 已退役**（移到 `tmp/demucs_separator_server.py.retired`，gitignored、可恢复）：`src/` 下从来没有客户端实现它，而它的能力已被 `scripts/separator_worker.py` 覆盖——常驻子进程同样只加载一次模型，还不需要额外起一个常驻 HTTP 服务。以后若确实要跨机分离，更干净的做法是把 HTTP 变体放在同一套 worker 协议之后，而不是另立一套接口。
- **主进程不再 `import torch`**，`release_item_resources()` 只剩 `gc.collect()`。

### 已知问题

- **LRC 与音频版本不匹配。** 曲库中存在歌词与音频属于不同剪辑的样本。此时任何全局常量偏移都不成立——实测一例需要 +21 秒偏移，而其行首分布仍能骗过对齐判据。**这类输入需要人工发现**（`--offset` 手动指定，或先看 `--dump-dir` 的产物）。
- **偏移估计的可辨识性平台。** 当密集排布的行首整体落在同一个长人声区间内时，区间内存在一段真实无法区分的平台，结果由平局规则决定。影响有限（行首仍落在人声区间内），但它意味着该估计器只有「全局常量偏移」一个自由度，无法处理逐行漂移。
- **`audio-separator` 后端已实机验证**（`UVR_MDXNET_KARA_2.onnx` 端到端跑通）。两点注意：它的模型清单查询（`--list-models`）需要访问 `raw.githubusercontent.com`，被墙时只能直接给 `--separator-model`；它默认按输入位深决定输出位深，遇到认不出位深的容器（如 mp3）会退回 16 位，所以 worker **会先把输入解码成 32 位浮点 WAV** 再交给它，避免在下游归一化之前多一次量化。
- `--worker-python` 作用于**所有**配置，因此该解释器必须同时具备所有后端的依赖；只对比单一后端时才用得上。
- `--dump-dir` 会为**每一行歌词**导出一个 WAV（`05_line_<i>.wav`），一首歌几百个文件。
- `min_vocal_activity` 用的是按峰值归一化到 1 的能量曲线，所以阈值是**相对**的：一首歌只要有一个响的副歌，安静的主歌就可能整段被跳过。

## 参考资料

- ~~[FunASR](https://github.com/modelscope/FunASR)~~
- [SPL Format](https://moriafly.com/standards/spl.html)
- [UVR Models](https://github.com/TRvlvr/model_repo/releases/)
- [Demucs](https://github.com/adefossez/demucs)
- [python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator)
- ~~[whisper](https://github.com/openai/whisper)~~
- ~~[faster-whisper](https://github.com/AIXerum/faster-whisper)~~
- ~~[mfa](https://mfa-models.readthedocs.io/en/latest/index.html)~~
- [Gentle container](https://hub.docker.com/r/lowerquality/gentle)
- [Qwen3ForcedAligner](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B)

位于 [samples/](samples/) 下的样本文件们的版权归其各自的原始创作者们所有
