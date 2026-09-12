# 对齐服务（Qwen3-ForcedAligner）

```bash
uv run --script scripts/qwen3aligner_server.py                 # 127.0.0.1:8787
uv run --script scripts/qwen3aligner_server.py --port 9000
uv run --script scripts/qwen3aligner_server.py --host 0.0.0.0  # 跨机（服务无鉴权！）
uv run --script scripts/qwen3aligner_server.py --device cpu    # 强制设备
```

默认监听 `127.0.0.1:8787`，与 `main.py --aligner-url` 的默认值一致，所以从仓库根目录按第一条命令启动后主程序不需要任何额外参数。模型路径按脚本所在仓库根目录定位，不依赖 CWD。

## 响应形状是契约的一部分

客户端 `Q3FAClient` 读的是 `response["words"]`，所以：

- **单个文件 → 对象** `{"words": [...]}`
- 多个文件 → 数组 `[{"words": [...]}, ...]`

这里踩过一个会导致**静默失效**的坑：服务端签名若写成 `audio: list[UploadFile] | UploadFile`，FastAPI 对**单个**上传也会走 list 分支（实测），于是单文件请求拿到数组响应，客户端在 `response["words"]` 上抛 `TypeError`；而 `core._align_line` 会逐行吞掉异常、保留原行，最终写出一个**没有任何词级时间戳、退出码却是 0** 的文件。

因此现在：

- `is_batch` 按**实际收到的文件数**判定（不是按形参的运行时类型）；
- 客户端同时容忍「长度为 1 的数组」，其他形状抛 `Q3FAProtocolError`；
- `gen_kara` 在**整首歌全部对齐失败**时直接报错并拒绝写盘；
- 两侧各有一条回归测试（`tests/test_aligner_contract.py`）。

## 语言

`--aligner-language auto`（默认）按**整首歌**判定：出现任意假名即判日语，否则有 CJK 汉字判中文，否则有拉丁字母判英文。关键原因是纯汉字行（如「畜生」）在逐行判据下会被误判为中文，按行计票时随时可能把整首日文歌翻盘。`--target-lang` 用于跳过语言不符的行（例如夹在日文歌词里的中文翻译行）。

## ⚠️ 词级时间的下限是 80ms，以及零长度词

这是这套方案的真实精度上限，值得先知道：

- 模型 config 里 `timestamp_segment_time = 80`，时间戳来自**离散 token**（`timestamp_ms = token_id * 80`），所以**每一个词边界都被量化到 80ms 的格子上**。`parse_timestamp` 把相邻两个边界 token 配成一个词的 `(start, end)`。
- 于是「真实时长不足一个 80ms 帧」的词会拿到 `start == end`，即**零长度词**。
- 库自带的 `fix_timestamp` 只用最长递增子序列修**递减**的异常，判据是 `data[j] <= data[i]`——**允许相等**，所以零长度词会被原样保留，不会被它修掉。
- 实测零长度词的占比（6 首歌、1497 个词的真实运行日志）：

  | 歌 | 词数 | 零长度 | 占比 | 时间戳栅格 |
  |---|---|---|---|---|
  | Saya - 失う | 252 | 27 | 10.7% | 80ms |
  | ReoNa - SACRA | 222 | 34 | 15.3% | 20ms（fix_timestamp 插值过） |
  | Cryu - 蒲公英 | 141 | 59 | **41.8%** | 80ms |
  | samples/ 三首 | 882 | 217 | 24.6% | — |
  | **合计** | **1497** | **337** | **22.5%** | |

- 位置与上下文：**70% 落在行中**（不是行首/行末），**80% 紧跟前一个词的结束时刻**——也就是"被前一个词吞掉"。所有词跨度都是 80ms 的整数倍（0/80/160/240/320/400/480/560/640/800/880/960…）。

**这意味着什么**：零长度词在序列化时会与相邻词合并（时间语义无损，见 `docs/metadata-filter.md` 之外的那条 round-trip 结论），歌词文本不会丢，但**这些词在播放器里无法被单独高亮**——而在「蒲公英」这样的歌上，这个比例高达 42%。这是当前唯一一处会明显影响"逐字"体验的缺陷，根因在模型的时间戳词表，不在本仓库的代码。
