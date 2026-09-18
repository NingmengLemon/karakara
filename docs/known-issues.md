# 已修 / 已知问题

## 已修（变更记录）

- **全局偏移估计**（`src/karakara/offset.py`）。原实现把每行按 `[start, start+5000ms)` 的方块标记，密集歌词会让指示曲线退化成近乎全 1 的实心块，于是任何打分函数都只能对齐两条曲线的重心而非起唱点；而且用的是裸点积（随重叠长度单调增长，等于惩罚偏移量本身），平局时还会被扫描顺序推到区间极端。现在改成**行首窄窗** + **分母固定的判别式对比度** + **平局取更小偏移量**。细节与后续发现见 [offset.md](offset.md)。
- **负偏移被静默抵消**（`src/karakara/core.py`）。此前 `_apply_offset` 在出现负时间戳时会把**整条时间轴回退**，而 LRC 里几乎总有一条 `[00:00.000]` 的元数据行（「作词 : xxx」），它不参与对齐，却会让任何负偏移被精确抵消：实测含 0ms 行的文件上 `--offset -200` 的净偏移为 0，`--offset` 这条人工兜底因此**对负值完全失效**。现在改为**逐个把负时间戳夹到 0**，偏移本身保留。
- **自动偏移要过两道互相独立的校验**（行区间对比度 + 「第一次持续人声」锚点）。见 [offset.md](offset.md)。
- **对齐服务的响应契约**（`scripts/qwen3aligner_server.py` + `src/karakara/aligner/q3fa/`）。服务端用 `audio: list[UploadFile] | UploadFile` 时，FastAPI 对**单个**上传也会走 list 分支，于是 `/align` 返回数组、客户端读 `response["words"]` 抛 `TypeError`——而 `core._align_line` 逐行吞掉异常并保留原行，最终写出一个**没有任何词级时间戳、退出码却是 0** 的文件。现在按实际文件数判定 `is_batch`，客户端容忍「长度为 1 的数组」，`gen_kara` 在整首歌全部对齐失败时直接报错并拒绝写盘。见 [aligner.md](aligner.md)。
- **对齐服务曾被单个慢请求拖死**。`/align` 原本是 `async def` 而推理是阻塞的，于是整段生成跑在事件循环里：一个耗时请求会把 `/health` 和所有后续请求一起堵死，客户端放弃后服务端还在算——实测最终连监听套接字都废掉（`WinError 64 指定的网络名不再可用`），只能重启。现在端点改成同步 `def`（FastAPI 会丢进线程池），推理本身用一个锁串行化（GPU 并发只会互相抢显存，而主程序本来也是逐行调用）。实测：客户端 1.5s 超时断开后 `/health` 立刻仍能响应、下一个请求照常成功。
- **对齐服务端口与启动方式**。`__main__` 此前硬编码 `0.0.0.0:8000`，与 `--aligner-url` 默认值（8787）和文档都不一致，而且没有 `--port`。现在默认 `127.0.0.1:8787`，可 `--host/--port/--model/--device` 覆盖。
- **批处理容错**：`discover_batch_jobs` 此前在第一个「找不到唯一同名音频」的 LRC 上抛异常，导致**整个曲库一首都不处理**（实测某 6622 首的曲库里有 86 个这样的 LRC，能配对的 6536 首全被拖死）。现在默认**跳过并汇总**，`--strict-pairs` 才恢复「立即失败」。发现阶段还改成每个目录只枚举一次：同样的曲库从 >10 分钟降到 **1.8 秒**。
- **不再无限等待**：对齐请求默认 120s 超时（`--aligner-timeout`），分离请求默认 900s（`--separator-timeout`）；两者此前都是 `None` = 不超时，一个卡住的 worker 会让批处理永久挂住。
- **配置路径**：`metadata_filter.toml` 此前按 CWD 加载（换目录运行即 `FileNotFoundError`），现在按项目根目录定位并可用 `--metadata-filter` 覆盖；`audio-separator` worker 的默认模型目录与对齐服务的模型路径同理。
- **元数据行滤除**：关键字从 28 项扩到约 200 项、打开 `parenthetical`/`id3_tags`、支持全角括号、新增 7 条自定义正则。度量与全部假阳性证据见 [metadata-filter.md](metadata-filter.md)。
- **对齐语言**。此前语言在构造时硬编码为 `Chinese`，`gen_kara` 算出的语言判定结果从未被使用，于是英文/日文歌全部按中文对齐。见 [aligner.md](aligner.md)。
- **对齐后端的语言能力此前只存在于服务端**。`--aligner-language` 的选项是两个后端语言集合的并集（`yue`/`ko` 只有 qwen3 支持），于是 `--aligner-language ko` 配 `hfa` 会一路跑到服务端才拿到 400——那时人声分离（几十秒）已经白跑完。现在能力进了 `src/karakara/backends.py` 的登记表，`main()` 在开跑前检查并以退出码 2 报错，消息里直接给出能用的后端；`tests/test_aligner_backends.py` 另有一道防漂移检查，保证登记表与服务端真正接受的语言一致。
- **HubertFA 的上游代码从 zip 快照改为 git submodule**（`third_party/HubertFA`，钉在 `b3f0869`）。快照没有出处与提交号、换台机器就没了（`models/` 在 .gitignore 内）；submodule 只让仓库多一个 gitlink，且能把第三方代码排除在 mypy/ty/ruff 之外。模型权重仍在 gitignored 的 `models/aligner/HubertFA/`。
- **`separator/demucs/patch.py` 已删除**：它 module-level 去 patch `demucs.states.load_model`，但全仓库没有任何地方 import 它，完全没生效。
- **`scripts/demucs_separator_server.py` 已退役**（移到 `tmp/demucs_separator_server.py.retired`，gitignored、可恢复）：`src/` 下从来没有客户端实现它，而它的能力已被 `scripts/separator_worker.py` 覆盖——常驻子进程同样只加载一次模型，还不需要额外起一个常驻 HTTP 服务。以后若确实要跨机分离，更干净的做法是把 HTTP 变体放在同一套 worker 协议之后，而不是另立一套接口。
- **主进程不再 `import torch`**，`release_item_resources()` 只剩 `gc.collect()`。
- **工程卫生**：补了 `[tool.ruff]` 配置（此前 ruff 是 dev 依赖却没有配置，`ruff check .` 在干净检出上就报 12 条）、统一格式、加 `.gitattributes` 把行尾钉成 LF。

## 已知问题

### 会影响产物质量

- **零长度单元与精度下限取决于后端**。默认的 HubertFA 帧移 **10ms**，实测日文零长度单元 **1.9%**、行区间覆盖率 88.8%；可选的 `qwen3` 后端量化在 **80ms**，零长度词 **22.5%**（逐首 10.7%～41.8%），根因是模型的时间戳词表（`timestamp_segment_time = 80`），且**无法靠改对齐粒度绕过**（日文走 nagisa 词切分，插空格会被重新切回去）。两者共用同一套后处理：每首歌记录零长度占比、超过 **30%** 告警；零长度单元的文本总是被并进相邻 token（产物里不再有重复时间标签，解析器告警 62 → 0）；`--refine-collapsed-words` 可把它们摊进其后的空隙（实测 63% 可细分，属**推断值**，默认关闭）。详见 [aligner.md](aligner.md) 与 [aligner-backends.md](aligner-backends.md) §9。
- **歌词错一个字的代价是"从错处往后全毁"**。V4 扰动实测（[aligner-backends.md](aligner-backends.md) §9.7）：扰动点**之前**的边界漂移在日文上恒等于 0，而错处**之后**的中位漂移 329–519ms、p90 约 2.1s。HubertFA 是从左到右锚定的，没有 SOFA `match` 模式那种"只取最优连续子序列"的容错，所以 VOCALOID 的错字/重复段落会命中这个坑，而且**产物上看不出异常**（覆盖率与单调性都正常）。真遇到就手工修歌词，或用 `--target-lang` 跳过该行。
- **分离质量比片段边界重要得多**。V4 实测：片段两端各加减 100ms，HubertFA 的行内边界中位只漂 **1ms**；但把分离人声换成原始混音，中位漂 18.5ms、p90 **398ms**（日文）。所以 [aligner-backends.md](aligner-backends.md) §4 里"逐行送片段时两端留 100–200ms padding"的建议对本项目**不需要**，要再提高质量应该优先换分离模型。
- **全局偏移估计不可靠**：两条判据各自都会错、且错在不同的歌上，因此自动偏移极度保守（证据不足就不动）。换建模方式（对齐到人声**起音**而不是"能量高的地方"）是未做的正解。见 [offset.md](offset.md)。
- **LRC 与音频版本不匹配**：歌词与音频属于不同剪辑时，任何全局常量偏移都不成立——实测一例需要 +21 秒偏移，而其行首分布仍能骗过对齐判据。**这类输入需要人工发现**（`--offset` 手动指定，或先看 `--dump-dir` 的产物）；现在的锚点校验会把明显不可调和的情形拦下来并告警。
- **偏移估计的可辨识性平台**：密集行首整体落在同一个长人声区间内时，区间内存在真实无法区分的平台，结果由平局规则决定。该估计器只有「全局常量偏移」一个自由度，无法处理逐行漂移。

### 行为与副作用

- **元数据行仍然充当切段边界。** 这是刻意的：实测摘掉它们会让 219/4971 个文件里的 261 个切段变长（把器乐段圈进送给对齐器的片段）。见 [metadata-filter.md](metadata-filter.md)。
- **`--dump-dir` 会为每一行歌词导出一个 WAV**（`05_line_<i>.wav`），一首歌几百个文件。
- **`min_vocal_activity` 用的是按峰值归一化到 1 的能量曲线**，所以阈值是**相对**的：一首歌只要有一个响的副歌，安静的主歌就可能整段被跳过。
- **「X by Y」自定义模式有理论上的误伤面**：英文歌词 `Sound by the sea` 这类会被判成署名。本库实测 0 例，真遇到就删掉 `[custom] patterns` 里那一条。

### 工程

- **`audio-separator` 后端已实机验证**（`UVR_MDXNET_KARA_2.onnx` 端到端跑通）。它的模型清单查询（`--list-models`）需要访问 `raw.githubusercontent.com`，被墙时只能直接给 `--separator-model`；它默认按输入位深决定输出位深，遇到认不出位深的容器（如 mp3）会退回 16 位，所以 worker **会先把输入解码成 32 位浮点 WAV** 再交给它，避免在下游归一化之前多一次量化。
- **没有 CI**：`pyproject.toml` 里有 `[tool.ruff]` 配置，`mypy`/`ty`/`pytest` 也都配好了，但仓库没有 `.github/workflows`，四个检查都得手动跑。
- **`samples/` 下有音频二进制入库**（约 19MB），版权归其各自的原始创作者所有。
