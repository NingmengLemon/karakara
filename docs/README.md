# 文档地图

这里的文档分三层，**按「你现在想知道什么」来选**，而不是按主题平铺：

| 层 | 回答的问题 | 会不会过期 |
|---|---|---|
| [current/](current/) | **现在是什么样**：契约、默认值、限制 | 跟着代码走，改行为就改它 |
| [records/](records/) | **凭什么这么说**：某次实测的原始数字与判据 | 永不修改（写下来就是历史） |
| [archive/](archive/) | **以前是什么样、试过什么**：被取代的调研与变更记录 | 永远不更新 |

一条纪律：**现状文档里不写实验过程，记录文档里不写「当前应当如何」**。想引用某个数字时，
现状文档给出结论并链接到记录；记录给出数字与复现命令，不规定行为。

## 我该看哪一篇

- 想跑起来、想知道开关有哪些 → [../README.md](../README.md)
- 想知道一次运行里数据怎么流动、worker 协议、代码在哪 → [current/pipeline.md](current/pipeline.md)
- 对齐精度为什么是这样、零长度单元怎么处理 → [current/aligner.md](current/aligner.md)
- 时间轴整体偏了怎么办 → [current/offset.md](current/offset.md) 与 [current/offset-gui.md](current/offset-gui.md)
- 某一行字幕被当成署名吃掉了 / 想改过滤规则 → [current/metadata-filter.md](current/metadata-filter.md)
- 环境装不上、GPU 没生效、uv 缓存作怪 → [current/environments.md](current/environments.md)
- 想知道哪些坑还在 → [current/limits.md](current/limits.md)
- 想换一个对齐后端，先看别人怎么评估的 → [records/2026-09-18-aligner-choice-and-stability.md](records/2026-09-18-aligner-choice-and-stability.md) 与 [archive/aligner-backends-research.md](archive/aligner-backends-research.md)

## current/

| 文档 | 内容 |
|---|---|
| [pipeline.md](current/pipeline.md) | 主流程、两个 worker 的架构与协议、分离后端、代码布局 |
| [aligner.md](current/aligner.md) | `/align` 契约、两个后端、语言判定、零长度单元的三层处理 |
| [offset.md](current/offset.md) | 全局偏移的算法、两道校验、负时间戳夹取、已知限制 |
| [offset-gui.md](current/offset-gui.md) | 人工对照波形与歌词调偏移的桌面工具、写回歌词文件的行为 |
| [metadata-filter.md](current/metadata-filter.md) | 五种过滤策略与默认值、假阳性纪律、为什么不预先 pop |
| [environments.md](current/environments.md) | 三套环境、CUDA 索引、uv 脚本环境缓存的坑 |
| [limits.md](current/limits.md) | 现在还成立的限制与副作用（产物质量 / 行为 / 工程） |

## records/

| 记录 | 内容 |
|---|---|
| [2026-09-12-qwen3-80ms-zero-length.md](records/2026-09-12-qwen3-80ms-zero-length.md) | 80ms 量化的根因、22.5% 零长度实测、试过但走不通的绕法 |
| [2026-09-12-offset-criteria.md](records/2026-09-12-offset-criteria.md) | 三首歌上两条偏移判据的对照、早期实现错在哪 |
| [2026-09-12-metadata-corpus-scan.md](records/2026-09-12-metadata-corpus-scan.md) | 6622 个 LRC 的扫描、各开关的判据、假阳性清单 |
| [2026-09-12-separator-ab.md](records/2026-09-12-separator-ab.md) | 分离后端的代理指标与已测结果、一次失效的度量 |
| [2026-09-18-aligner-choice-and-stability.md](records/2026-09-18-aligner-choice-and-stability.md) | 人耳裁决、V4 扰动稳定性（两个后端）、复现命令 |

新记录的文件名用 `YYYY-MM-DD-主题.md`，开头写清「这是记录，不是规范」，并链接对应的现状文档。

## archive/

| 文档 | 为什么在这 |
|---|---|
| [changelog.md](archive/changelog.md) | 修过的坑与当时的现象、根因、改法 |
| [aligner-backends-research.md](archive/aligner-backends-research.md) | 2026-09-12 的后端调研（含候选清单与「怎么评估一个新后端」的方法）；结论已落地为默认后端 `hfa` |
