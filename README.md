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
# ① 先起对齐服务（默认 127.0.0.1:8787，与 main.py 的 --aligner-url 默认值一致）
uv run --script scripts/qwen3aligner_server.py
# 换端口/地址：--port 9000 / --host 0.0.0.0（注意服务无鉴权，跨机再开）

# ② 分离 worker 自检（不需要主程序）
uv run --script scripts/separator_worker.py --info
uv run --script scripts/separator_worker_audio_separator.py --list-models   # 见下方说明

# 单文件
uv run main.py -l song.lrc -a song.flac -o song.kara.lrc

# 批量（递归找同名 lrc/音频对；无法配对的 LRC 会被跳过并汇总）
uv run main.py --batch-dir /path/to/album --output-dir /path/to/out

# 已经有一个装好 demucs 的环境时，跳过 uv 的环境准备
uv run main.py --batch-dir ... --separator-cmd /path/to/python.exe scripts/separator_worker.py
# 或用环境变量
KARAKARA_SEPARATOR_CMD='"C:\sep-env\Scripts\python.exe" "scripts/separator_worker.py"'
```

**对齐服务的响应形状是契约的一部分**：单个文件必须返回对象 `{"words": [...]}`，
多个文件才返回数组。服务端与客户端任意一侧改错都会让主程序「每行都对齐失败、
退出码却是 0」——所以两侧各有一条回归测试，且 `gen_kara` 在**整首歌全部对齐失败**
时会直接报错，而不是写出一个没有词级时间的文件。

常用开关：`--separator-backend {demucs,audio-separator}`、`--separator-model`（默认 `UVR_Demucs_Model_1`）、`--separator-device`、`--separator-timeout`（默认 900s）、`--aligner-url`、`--aligner-timeout`（默认 120s，`0` 表示不超时）、`--aligner-language {auto,zh,ja,en,...}`、`--target-lang`、`--min-vocal-activity`、`--offset` / `--no-offset-estimate`、`--metadata-filter`、`--strict-pairs`、`--dump-dir`、`--sep-work-dir`、`--existing-byword-policy`、`--fail-fast`。

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

> ⚠️ **改完索引之后必须清掉该脚本的缓存环境——而且理由比"缓存键"更细一层。**
> 实测 uv 的 PEP 723 脚本环境是**按脚本名固定**的：环境目录名里的哈希**既不包含**
> `[tool.uv]` 里的索引配置，也**不随依赖清单的变化而改变**（往清单里加一个包，uv 只是
> 把那个包装进同一个环境里）。所以
>
> * 只改索引 → uv 认为"什么都没变"，照旧复用那个装着 CPU 版 torch 的环境；
> * 甚至**把 `torch` 加进依赖清单也没用**：CPU 版 torch 已经满足 `torch>=2.9.0`，
>   uv 判定无需改动，于是索引依旧不会被查询。
>
> 结论：**必须删掉该脚本对应的环境目录**（`uv cache dir` / `environments-v2/<脚本名>-<哈希>`）
> 或 `uv cache clean`，再跑一次让它重建。实测删掉后重建约 10 秒（cached wheel 直接硬链）。

不想动仓库的话，也可以用 `--separator-cmd`（或 `KARAKARA_SEPARATOR_CMD`）指向任何已装
CUDA 版 torch 的解释器：

```bash
uv run main.py --batch-dir ... \
  --separator-cmd /path/to/cuda-env/python.exe scripts/separator_worker.py
```

**对齐服务同理，且已经配好了**：`scripts/qwen3aligner_server.py` 的头部现在也带 CUDA
索引 + `torch` 显式依赖（`torch` 在 `qwen-asr` 里只是间接依赖，而 `[tool.uv.sources]`
只对直接依赖生效，所以必须显式列出）。win32 实测可用：

```
torch 2.14.0+cu130 / cuda_available=True / RTX 4060 Laptop (cc 8.9) / bf16 ok
model loaded successfully on cuda:0
```

服务启动时会打印它实际用的设备；CUDA 不可用时给出显式告警而不是静默降级。
若只想要 CPU 版，把头部那两段索引删掉、并去掉依赖里的 `torch`，再删掉缓存环境重建。

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
| `UVR_Demucs_Model_1` vs `htdemucs_6s` | 4 | **前者一致更好**：泄漏 0.481 vs 0.533、对比度 0.1771 vs 0.1650。这是零下载就能拿到的提升 |
| `audio-separator:UVR_MDXNET_KARA_2.onnx` vs `htdemucs_6s` | 2 | 基本持平（泄漏 0.312 vs 0.311）。注意 `UVR_MDXNET_KARA_2` 是**卡拉OK（去主唱）**模型，它的 "vocals" 是残差而非干净主唱，并不是这个用途的合适模型 |

> ⚠️ **耗时列（原表里的 6.7s vs 8.0s）已不可用。** 脚本此前为每个 (歌曲, 配置) 组合
> 各起一个 worker，于是每首歌的耗时里都混着一次冷启动（`uv run --script` 环境解析 +
> `import torch` + 模型加载，实测约 2.6s）；那个 1.3s 的差距与这个常数同量级。现在
> worker 跨歌曲复用、汇总时排除首曲冷启动（表中会把仅 1 首样本的配置单独标出）。
> 需要重新测一次再下结论。对比度/泄漏/占位这三列不受影响——它们与冷启动无关。

要把「人声质量」这个决策真正定下来，还需要：**(a)** 明显更多的曲目；**(b)** 专用人声模型（BS-RoFormer 人声 SDR ≈12.9、VR HP-UVR 等），而不是卡拉OK模型；**(c)** GPU 以免每次测量都要等一分钟以上。

## 状态与已知问题

### 已修

- **全局偏移估计**（`src/karakara/offset.py`）。原实现把每行按 `[start, start+5000ms)` 的方块标记，密集歌词会让指示曲线退化成近乎全 1 的实心块，于是任何打分函数都只能对齐两条曲线的重心而非起唱点；而且用的是裸点积（随重叠长度单调增长，等于惩罚偏移量本身），平局时还会被扫描顺序推到区间极端。
  现在改成**行首窄窗** + **分母固定的判别式对比度** + **平局取更小偏移量**。在真实曲库（4 首，人声实机分离）上用独立判据（行区间能量对比度）裁决：**新实现 4/4 胜出**；旧实现在其中 3 首上给出的偏移**比完全不偏移还差**。
- **负偏移被静默抵消**（`src/karakara/core.py`）。此前 `_apply_offset` 在出现负时间戳时会把**整条时间轴回退**，而 LRC 里几乎总有一条 `[00:00.000]` 的元数据行（「作词 : xxx」）——它不参与对齐，却会让任何负偏移被精确抵消：实测含 0ms 行的文件上 `--offset -200` 的净偏移为 0，`--offset` 这条人工兜底因此**对负值完全失效**（实测某首歌唱段估计出 −4000ms，最终净偏移 0）。现在改为**逐个把负时间戳夹到 0**，偏移本身保留；被夹到的是参与对齐的行时会额外告警（那是「估计过大」的信号）。
- **自动偏移要过两道互相独立的校验**。行首窄窗判据只看「这一行是不是从这里开始唱」，可能给出让行首仍落在人声里、却把整段行区间推出人声的偏移。现在自动估计必须同时满足：(a) 在**行区间对比度**上优于「不偏移」；(b) 与**「第一次持续人声」锚点**（第一次持续人声的位置 − 第一条歌词行时间戳）分歧不超过 2 秒。任一不通过就退回 0 并告警（附可复制的 `--offset` 命令）。手动 `--offset` 不受校验约束。实测：SACRA 采纳 +450ms（锚点 +270ms 相符，物理真值 ≈ +270ms），Saya / Rick Astley 退回 0 从而避开 −4000ms / −600ms 的错误修正。
- **对齐服务的响应契约**（`scripts/qwen3aligner_server.py` + `src/karakara/aligner/q3fa/`）。服务端签名是 `audio: list[UploadFile] | UploadFile`，而 FastAPI 对**单个**上传也会走 list 分支，于是 `/align` 返回数组、客户端读 `response["words"]` 抛 `TypeError`——而 `core._align_line` 逐行吞掉异常并保留原行，最终写出一个**没有任何词级时间戳、退出码却是 0** 的文件。现在 `is_batch` 按实际文件数判定（单文件返回对象），客户端同时容忍「长度为 1 的数组」并在形状异常时抛 `Q3FAProtocolError`，`gen_kara` 在**整首歌全部对齐失败**时直接报错并拒绝写盘。
- **对齐服务端口与启动方式**。`__main__` 此前硬编码 `0.0.0.0:8000`，与 `--aligner-url` 默认值（8787）和 README 都不一致，而且没有 `--port`。现在默认 `127.0.0.1:8787`（与客户端默认值一致），可 `--host/--port/--model/--device` 覆盖；模型路径改为按项目根目录定位，不再依赖 CWD。
- **批处理容错**：`discover_batch_jobs` 此前在第一个「找不到唯一同名音频」的 LRC 上抛异常，导致**整个曲库一首都不处理**（实测某 6622 首的曲库里有 86 个这样的 LRC，能配对的 6536 首全被拖死）。现在默认**跳过并汇总**，`--strict-pairs` 才恢复「立即失败」。
- **不再无限等待**：对齐请求默认 120s 超时（`--aligner-timeout`），分离请求默认 900s（`--separator-timeout`）；两者此前都是 `None` = 不超时，一个卡住的 worker 会让批处理永久挂住。`0` 仍可显式表示不超时。
- **配置路径**：`metadata_filter.toml` 此前按 CWD 加载（换目录运行即 `FileNotFoundError`），现在按项目根目录定位并可用 `--metadata-filter` 覆盖；`audio-separator` worker 的默认模型目录同样改为按项目根目录定位。对齐服务的模型路径同理。
- **元数据行滤除**（`metadata_filter.toml` + `src/karakara/utils/metadata.py`）。此前只有「28 个关键字 + 冒号」一条规则在生效，真实曲库里相当一批署名行逃过过滤、被当成歌词送进对齐器。现在：关键字补到约 200 项（含 `曲`/`词`/`歌`/`唱`/`译`/`绘`/`调`/`混` 这类单字缩写，以及 `词曲编调绘` 这类复合写法），`parenthetical` 与 `id3_tags` 打开，括号形态支持全角（`（间奏）`/`【サビ】`），并新增 7 条自定义正则覆盖分隔线/纯段落词/结束标记/空值占位/「X by Y」署名/关键字与冒号之间夹英文注释等形态。

  度量走的是**真实解析路径**（`Lyrics.loads` 之后的行文本，而不是裸文件文本）：6622 个 lrc 上旧配置命中 11,586 行 → 新配置 12,735 行，**丢失 0 行**、新增 1,149 行；新增项中 45% 位于文件中段，最常见的是 `music...`(94)、`♪`(73)、`...`(69)、`終わり`(52)、`-END-`(37)、`undefined`(19)——没有一条像歌词。这些行此前会被真的送去对齐（既浪费请求，也可能产出无意义的词级时间）。

  假阳性是这套过滤最严重的失败模式（删掉一行就等于丢掉一句歌词），所以把「绝对不能命中」的形态连同真实命中证据一起固化成了回归测试：`合：赔盏茶才算周到`（对唱分句标记，冒号后就是歌词正文，48 行）、`8:07に君を待ってる`、`_(:з」∠)_`、`By the way, do you like baseball?`、`谢谢 想说谢谢你`、`Thanks for the meal`、`**你是什么垃圾？**`、`（与你同在）`（整行被括号包裹的歌词，10,163 行 / 1,887 文件——所以 `parenthetical` 必须是白名单 + 整行匹配，绝不能做成形状规则），以及 `1`/`2`/`3`/`1234`/`1111…`（报数与连打演唱，故 `pure_numbers` 保持关闭）。

  顺带修掉两处实现问题：关键字分支用 `.+` 抓不到空值行（`Singer：`、`Rap:`），已改为 `.*`；`parenthetical` 此前只认半角括号。

  一个反直觉的实测结论：**`id3_tags` 打开后在本管线里是空转的**。`[ti:] [ar:] [by:] [offset:]` 这类标签会被 `lemony-lrc-parser` 先收进 `lyrics.metadata`，根本不会以「行」的形态到达过滤器——裸文本里有 495 行这种标签，到达过滤器的 **0 行**。保留开关只是为了 `MetadataFilter` 独立使用时仍然正确，不要指望它带来召回提升。
- **对齐语言**。此前 `Qwen3ForcedAligner` 的语言在构造时硬编码为 `Chinese`，`gen_kara` 算出的语言判定结果从未被使用，于是英文/日文歌全部按中文对齐。现在 `--aligner-language auto` 会按整首歌判定（假名出现即判日语——纯汉字行会被逐行判据误判为中文，这是关键原因），也可显式指定；`--target-lang` 用于跳过语言不符的行。
- **端口不一致**：`--aligner-url` 默认值曾指向 8787，而客户端/实现内部默认 8000。
- **`separator/demucs/patch.py` 已删除**：它 module-level 去 patch `demucs.states.load_model`，但全仓库没有任何地方 import 它，完全没生效。
- **`scripts/demucs_separator_server.py` 已退役**（移到 `tmp/demucs_separator_server.py.retired`，gitignored、可恢复）：`src/` 下从来没有客户端实现它，而它的能力已被 `scripts/separator_worker.py` 覆盖——常驻子进程同样只加载一次模型，还不需要额外起一个常驻 HTTP 服务。以后若确实要跨机分离，更干净的做法是把 HTTP 变体放在同一套 worker 协议之后，而不是另立一套接口。
- **主进程不再 `import torch`**，`release_item_resources()` 只剩 `gc.collect()`。

### 已知问题

- **全局偏移估计：两条判据各自都会错，而且错在不同的歌上。** 用「第一次持续人声出现的位置 − 第一条歌词行时间戳」当**物理真值**（它与任何能量判据的形式无关），三首真实曲目的实测：

  | 曲目 | 物理真值 | `onset` 判据（现状） | `interval` 判据 |
  |---|---|---|---|
  | Saya - 失う | ≈ −1040ms | −4000ms ✗ | −1000ms ✓ |
  | ReoNa - SACRA | ≈ +270ms | +200ms ✓ | +4600ms ✗ |
  | Rick Astley - Never Gonna Give You Up | ≈ +17160ms（LRC 与音频属于不同剪辑） | −600ms ✗ | +7600ms ✗ |

  `interval` 判据（行区间内外的能量对比度）看似更"下游相关"，其实有**响度偏置**：把行区间整体挪到更响的段落就能刷高分数，所以它在 Rick Astley 上给出 +7.6 秒、在 SACRA 上给出 +4.6 秒。**因此不能用它替换 `onset` 判据**，只能当"是否值得动"的旁证。

  当前的处理不是"选出正确答案"（在只有全局常量一个自由度的前提下做不到），而是**证据不足时不动**：自动估计要同时通过 ①行区间对比度校验 ②「首次持续人声」锚点校验，任一不通过就按不偏移处理并给出可复制的 `--offset`。实测结果：SACRA 采纳 +450ms（锚点 +270ms 与之相符，真值 ≈ +270ms）；Saya 与 Rick Astley 都退回 0，从而避开了 −4000ms / −600ms 这两个错误修正。
  **根因仍未解决**——真正的出路不是继续调能量判据，而是换建模方式：把「行首」对齐到**音频里检测出的人声起音（onset strength / spectral flux）**，而不是对齐到"能量高的地方"。现有合成回归用例把歌词行均匀铺在人声区间**内部**（而不是区间起点），因此换成起音判据会与那套用例的构造前提冲突，需要连用例一起重新设计。
- **LRC 与音频版本不匹配。** 曲库中存在歌词与音频属于不同剪辑的样本。此时任何全局常量偏移都不成立——实测一例需要 +21 秒偏移，而其行首分布仍能骗过对齐判据。**这类输入需要人工发现**（`--offset` 手动指定，或先看 `--dump-dir` 的产物）。
- **偏移估计的可辨识性平台。** 当密集排布的行首整体落在同一个长人声区间内时，区间内存在一段真实无法区分的平台，结果由平局规则决定。影响有限（行首仍落在人声区间内），但它意味着该估计器只有「全局常量偏移」一个自由度，无法处理逐行漂移。
- **零长度词**：对齐器自身会返回 `start == end` 的词（实测某首歌 252 个词里 27 个，约 10.7%），产物序列化时会与后一词合并（时间语义无损）。目前没有任何检测——建议按此比例做退化告警。
- **元数据行仍然会充当切段边界。** `_line_sample_range` 在行没有 `end` 时一律取下一行的时间戳，并不看那一行是不是元数据。实测「处理前把元数据行整体 pop 掉」会让 219/4971 个文件里的 261 个切段发生变化，其中 **250 个是变长**（`[上一行, 署名行]` 变成 `[上一行, 下一行歌词]`，把器乐段也圈了进去）——所以**刻意不做这件事**：留着署名行反而让切段更紧。真正需要 pop 的情形（署名行被塞在中段、形成错误边界）在本库里没有观察到。
- **「X by Y」自定义模式有理论上的误伤面**：英文歌词 `Sound by the sea` 这类会被判成署名。本库实测 0 例，但它终究是形状规则；真遇到就在 `metadata_filter.toml` 的 `[custom] patterns` 里删掉那一条。
- **`audio-separator` 后端已实机验证**（`UVR_MDXNET_KARA_2.onnx` 端到端跑通）。两点注意：它的模型清单查询（`--list-models`）需要访问 `raw.githubusercontent.com`，被墙时只能直接给 `--separator-model`；它默认按输入位深决定输出位深，遇到认不出位深的容器（如 mp3）会退回 16 位，所以 worker **会先把输入解码成 32 位浮点 WAV** 再交给它，避免在下游归一化之前多一次量化。
- `--worker-python` 作用于**所有**配置，因此该解释器必须同时具备所有后端的依赖；只对比单一后端时才用得上。
- `--dump-dir` 会为**每一行歌词**导出一个 WAV（`05_line_<i>.wav`），一首歌几百个文件。
- `min_vocal_activity` 用的是按峰值归一化到 1 的能量曲线，所以阈值是**相对**的：一首歌只要有一个响的副歌，安静的主歌就可能整段被跳过。
- **`ruff`/`mypy`/`ty` 没有 CI**：`pyproject.toml` 里现在有 `[tool.ruff]` 配置，但仓库没有 `.github/workflows`，三个检查都得手动跑。

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
