# karakara

> **Kara**oke & [**KARA**KARA](https://moegirl.icu/KARAKARA)

只是 1 个神秘的 Playground (

> 已知一首歌的音频文件和行级别精度的歌词文件, 能否用程序做出词级别精度的歌词文件呢...?

读入「音频 + 行级 LRC」，产出「词级逐字 LRC」：

```
源音频 ──> 人声分离 ──> 预处理(归一化/颤音抑制/可选压缩)
                        └──> 全局偏移估计
                        └──> 逐行切段 ──> 强制对齐 ──> 词级 LRC
```

唯一花了心思的部分是一个简陋的 lrc 歌词解析器: [lemony-lrc-parser](https://github.com/NingmengLemon/lemony-lrc-parser)，参考 [SPL 歌词格式](https://moriafly.com/standards/spl.html) (2025-11-14 版) ~~, 但其实并没有严格遵循~~

分离与对齐各自跑在**独立进程**里（主环境的依赖闭包因此只有 11 个包、不含 torch）。这套架构、worker 协议与后端选择见 [docs/architecture.md](docs/architecture.md)。

## 快速开始

```bash
# ① 先起对齐服务（默认后端 hfa → 127.0.0.1:8788）
uv run --script scripts/hubertfa_aligner_server.py
# 用旧的 Qwen3 后端： uv run --script scripts/qwen3aligner_server.py
#                    uv run main.py ... --aligner-backend qwen3

# ② 单文件
uv run main.py -l song.lrc -a song.flac -o song.kara.lrc

# ③ 批量（递归找同名 lrc/音频对；无法配对的 LRC 会被跳过并汇总）
uv run main.py --batch-dir /path/to/album --output-dir /path/to/out
```

分离 worker 可以单独自检（不需要主程序、不需要对齐服务）：

```bash
uv run --script scripts/separator_worker.py --info
uv run --script scripts/separator_worker_audio_separator.py --list-models
```

对齐后端也可以自检「扰动稳定性」（片段两端 ±100ms / 混音 vs 分离 / 删一个字；需要先起对齐服务）：

```bash
uv run python scripts/check_aligner_stability.py \
  --lrc samples/countdown_to_zero_luotianyi.lrc \
  --audio samples/countdown_to_zero_luotianyi.mp3 --separate
```

常用开关：

| 开关 | 说明 |
|---|---|
| `--lyrics/-l`、`--audio/-a`、`--output/-o` | 单文件模式的输入与输出 |
| `--batch-dir`、`--output-dir` | 批处理模式的输入目录与输出根目录 |
| `--separator-backend {demucs,audio-separator}` | 用哪个分离 worker；默认 `demucs` |
| `--separator-model` | 默认 `UVR_Demucs_Model_1` |
| `--separator-device`、`--separator-model-dir` | 分离设备（如 `cuda:0` / `cpu`）、模型仓库目录 |
| `--separator-timeout` | 单次分离请求超时（默认 900s，`0` = 不限） |
| `--separator-cmd` / `KARAKARA_SEPARATOR_CMD` | 用自己的解释器跑 worker（跳过 uv 环境准备） |
| `--aligner-backend {hfa,qwen3}` | 用哪个对齐后端；默认 **hfa**（歌声专用，10ms 帧）。两者是独立服务、共用同一套 `/align` 契约 |
| `--aligner-url` / `--aligner-timeout` | 对齐服务地址（缺省按后端：hfa→8788、qwen3→8787）/ 单次请求超时（默认 120s，`0` = 不限） |
| `--aligner-language {auto,zh,ja,en,...}` / `--target-lang` | 送给对齐器的语言 / 只对齐该语言的行 |
| `--offset` / `--no-offset-estimate` | 手动全局偏移（ms）/ 完全不偏移 |
| `--min-vocal-activity` | 低于该归一化人声活动度的行不对齐（默认 0.01） |
| `--refine-collapsed-words` | 把对齐器的零长度词摊进其后的空隙（**推断值**，默认关闭；见 [docs/aligner.md](docs/aligner.md)） |
| `--metadata-filter` / `--strict-pairs` | 元数据行配置 / 要求所有 LRC 都有配对音频 |
| `--dump-dir` / `--sep-work-dir` | 调试音频导出 / 分离中间产物目录 |
| `--existing-byword-policy {realign,preserve}` | 已有逐字标签的行：重新对齐 / 原样保留 |
| `--fail-fast` | 批处理遇到第一个失败立刻停 |

`uv run main.py --help` 是完整的清单。

## 文档

| 文档 | 内容 |
|---|---|
| [docs/architecture.md](docs/architecture.md) | 两个 worker 的架构、行分隔 JSON 协议、分离后端选择 |
| [docs/environments.md](docs/environments.md) | PEP 723 内联依赖、PyTorch CUDA 索引、**uv 脚本环境缓存的坑** |
| [docs/aligner.md](docs/aligner.md) | 对齐服务的启动/端口/响应契约、语言判定、**80ms 精度与零长度词** |
| [docs/aligner-backends.md](docs/aligner-backends.md) | 换对齐后端的候选调研（WhisperX / HubertFA / 自建 CTC…）：分辨率、许可、Windows 可行性、验证方法 |
| [docs/offset.md](docs/offset.md) | 全局偏移估计的设计、为什么它不可靠、两道校验 |
| [docs/metadata-filter.md](docs/metadata-filter.md) | 元数据行滤除的开关与证据、为什么不预先 pop 掉 |
| [docs/benchmarks.md](docs/benchmarks.md) | 人声质量 A/B 的代理指标与已测结果 |
| [docs/known-issues.md](docs/known-issues.md) | 已修变更记录 + 已知问题清单 |

## 最要紧的三条限制

1. **对齐精度取决于后端**：默认的 HubertFA 帧移 **10ms**（日文实测零长度单元 1.9%、行区间覆盖率 88.8%），代价是日文路径会丢掉促音、汉字读音靠猜；可选的 Qwen3 后端量化在 **80ms**、零长度词 22.5%（最高一首 41.8%）。见 [docs/aligner.md](docs/aligner.md) 与 [docs/aligner-backends.md](docs/aligner-backends.md)。
2. **全局偏移估计不可靠**：两条能量判据各自都会错、且错在不同的歌上，所以自动偏移极度保守（证据不足就不动）。见 [docs/offset.md](docs/offset.md)。
3. **LRC 与音频版本不匹配的输入无解**：此时任何全局常量偏移都不成立，需要人工发现。见 [docs/known-issues.md](docs/known-issues.md)。

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
