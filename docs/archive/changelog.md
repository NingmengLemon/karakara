# 归档：变更记录

> 这里是**历史**。它回答「这个项目以前踩过什么坑、怎么修的」，不描述现状。
> 现状请看 [../current/](../current/)；每条修复对应的当前行为散落在相应的现状文档里。
>
> 早期条目按主题归并（当时的文档是「已修 + 已知问题」混排的），日期是它们进入仓库的时间。

## 2026-09-21

- **分离 worker 解码容错**（`scripts/separator_worker.py`）。单个损坏的音频包会让整个
  分离请求失败（实测某曲库文件有 5 个坏包 / 9569 个好包，位置在文件尾部），而主进程侧的
  `utils.io._decode` 默认开着 `skip_invalid`、同一个文件能正常解码。现在 worker 与它对齐：
  坏包跳过并告警，全部包都解不出时仍然报错。
- **worker 脚本路径改为绝对路径**（`karakara/paths.py`）。此前默认命令里是相对路径
  `scripts/separator_worker.py`，由 uv 按当前工作目录解析；从仓库外运行时它去找
  `E:\scripts\separator_worker.py`，以 `returncode=2` 退出，而主程序只报「分离 worker
  提前退出」。同一次改动给这个异常补上了启动命令与工作目录。
- **对齐请求改用 32 位浮点 WAV**（`utils.io.save_audio` 新增 `subtype`）。分离链路刻意全程
  32 位浮点，最后一步却把它量化成 int16 才上传，等于在终点丢掉前面保住的动态范围。A/B 实测
  影响被限制在一个帧以内（40/58 行逐单元完全一致）。
- **`load_audio_native` 支持内存缓冲区**：它内部把容器打开两次（先取采样率、再解码），同一个
  `BytesIO` 第二次打开必然失败，与音频是否合法无关。
- **补三条「只在生产里疼」的测试**：分离请求超时链路、过期 id 响应被忽略、`run_batch` 的
  `--fail-fast` 与 `finally` 语义。
- **代码内过期的后端数字**：`--refine-collapsed-words` 的帮助文本、`postprocess.py` 与
  `core._report_zero_length` 的说明都还把 qwen3 的 80ms / 22.5% 当成通用事实，默认后端
  早已是 HubertFA（10ms / 1.9%）。

## 2026-09-18

- **HubertFA 上游代码从 zip 快照改为 git submodule**（`third_party/HubertFA`，钉在 `b3f0869`）。
  快照没有出处与提交号、换台机器就没了；submodule 只让仓库多一个 gitlink，还能把第三方代码
  排除在 mypy/ty/ruff 之外。模型权重仍在 gitignored 的 `models/aligner/HubertFA/`。
- **V4 扰动稳定性检查进仓库**（`scripts/check_aligner_stability.py` + `tests/test_aligner_stability.py`），
  实测结论见 [../records/2026-09-18-aligner-choice-and-stability.md](../records/2026-09-18-aligner-choice-and-stability.md)。

## 2026-09-13

- **接入 HubertFA 后端并设为默认**（`--aligner-backend {hfa,qwen3}`）。对齐侧照搬分离侧的形状：
  一个 `AbstractAligner` 契约、每后端一个服务脚本、能力登记在 `backends.py`。契约不变，差异
  全部关在服务端。选型依据见上面那份记录。

## 2026-09-12（初版整理）

- **全局偏移估计重写**（`src/karakara/offset.py`）。原实现把每行按 `[start, start+5000ms)` 的
  方块标记，密集歌词会让指示曲线退化成近乎全 1 的实心块，任何打分函数都只能对齐两条曲线的
  重心而非起唱点；而且用的是裸点积（随重叠长度单调增长，等于惩罚偏移量本身），平局时还会被
  扫描顺序推到区间极端。改成行首窄窗 + 分母固定的判别式对比度 + 平局取更小偏移量。
- **负偏移被静默抵消**（`core._apply_offset`）。此前出现负时间戳时会把**整条时间轴回退**，
  而 LRC 里几乎总有一条 `[00:00.000]` 的元数据行，它不参与对齐却会让任何负偏移被精确抵消
  （实测含 0ms 行的文件上 `--offset -200` 的净偏移为 0，人工兜底对负值完全失效）。现在改为
  逐个把负时间戳夹到 0，偏移本身保留；被夹住的是参与对齐的行时额外告警。
- **自动偏移加两道互相独立的校验**（行区间对比度 + 首次持续人声锚点）。
- **对齐服务的响应契约**。服务端写成 `audio: list[UploadFile] | UploadFile` 时，FastAPI 对
  单个上传也会走 list 分支，于是单文件请求拿到数组响应，客户端读 `response["words"]` 抛
  `TypeError`，而逐行降级会把异常吞掉，最终写出一个**没有任何词级时间戳、退出码却是 0** 的
  文件。现在按实际文件数判定 `is_batch`，客户端容忍「长度为 1 的数组」，整首歌全部对齐失败时
  直接报错并拒绝写盘。
- **对齐服务曾被单个慢请求拖死**。`/align` 原本是 `async def` 而推理是阻塞的，一个耗时请求会
  把 `/health` 与所有后续请求一起堵死，客户端放弃后服务端还在算，最终连监听套接字都废掉。
  现在端点改成同步 `def`（FastAPI 丢进线程池），推理用一个锁串行化。
- **对齐服务端口与启动方式**：`__main__` 此前硬编码 `0.0.0.0:8000`，与 `--aligner-url` 的
  默认值（8787）和文档都不一致。现在默认 `127.0.0.1:8787`，可 `--host/--port/--model/--device`。
- **批处理容错**：`discover_batch_jobs` 此前在第一个「找不到唯一同名音频」的 LRC 上抛异常，
  导致整个曲库一首都不处理（实测 6622 首的曲库里有 86 个这样的 LRC，能配对的 6536 首全被拖死）。
  现在默认跳过并汇总，`--strict-pairs` 才恢复立即失败；发现阶段改成每个目录只枚举一次，
  同样的曲库从 >10 分钟降到 **1.8 秒**。
- **不再无限等待**：对齐请求默认 120s、分离请求默认 900s 超时（此前都是不超时）。
- **配置路径**：`metadata_filter.toml` 此前按 CWD 加载（换目录即 `FileNotFoundError`），现在按
  项目根目录定位并可用 `--metadata-filter` 覆盖。
- **元数据行滤除**：关键字从 28 项扩到约 200 项、打开 `parenthetical` 与 `id3_tags`、支持全角
  括号、新增自定义正则。度量与假阳性证据见 [../records/2026-09-12-metadata-corpus-scan.md](../records/2026-09-12-metadata-corpus-scan.md)。
- **对齐语言**：此前语言在构造时硬编码为 `Chinese`，语言判定结果从未被使用，于是英文与日文歌
  全部按中文对齐。现在 `--aligner-language auto` 按整首歌判定。
- **对齐后端的语言能力进了登记表**：`--aligner-language` 的选项是两个后端语言集合的并集
  （`yue`/`ko` 只有 qwen3 支持），此前选 `hfa` 又给 `ko` 会一路跑到服务端才拿到 400，那时人声
  分离（几十秒）已经白跑完。现在 `main()` 开跑前检查并以退出码 2 报错，另有一道机械防漂移
  测试保证登记表与服务端真正接受的语言一致。
- **`separator/demucs/patch.py` 已删除**：它在 module-level 去 patch `demucs.states.load_model`，
  但全仓库没有任何地方 import 它，完全没生效。
- **`scripts/demucs_separator_server.py` 已退役**（移到 `tmp/`，gitignored、可恢复）：`src/` 下
  从来没有客户端实现它，而它的能力已被 `scripts/separator_worker.py` 覆盖。以后若确实要跨机
  分离，更干净的做法是把 HTTP 变体放在同一套 worker 协议之后，而不是另立一套接口。
- **主进程不再 `import torch`**，`release_item_resources()` 只剩 `gc.collect()`。
- **工程卫生**：补 `[tool.ruff]` 配置（此前 ruff 是 dev 依赖却没有配置，干净检出上就报 12 条）、
  统一格式、加 `.gitattributes` 把行尾钉成 LF。
