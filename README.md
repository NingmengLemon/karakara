# karakara

把「音频 + 行级 LRC」变成「词级逐字 LRC」。

> **Kara**oke & [**KARA**KARA](https://moegirl.icu/KARAKARA)。一个 Playground：
> 已知一首歌的音频和行级精度的歌词，能不能用程序做出词级精度的歌词？

```
源音频 ──> 人声分离 ──> 预处理（归一化 / 颤音抑制 / 可选压缩）
                        ├──> 全局偏移估计
                        └──> 逐行切段 ──> 强制对齐 ──> 词级 LRC
```

顺手写了一个简陋的 LRC 歌词解析器：[lemony-lrc-parser](https://github.com/NingmengLemon/lemony-lrc-parser)，
参考 [SPL 歌词格式](https://moriafly.com/standards/spl.html)。

## 跑起来

需要 `uv`、一块能跑 Demucs 的机器（CPU 也行，慢），以及对齐服务要用的模型。

```bash
# ① 起对齐服务（默认后端 hfa → 127.0.0.1:8788）
uv run --script scripts/hubertfa_aligner_server.py --preload
# 首次使用 HubertFA 还要准备模型与上游代码：
#   git submodule update --init third_party/HubertFA   （模型来源见 models/aligner/HubertFA/SOURCE.md）

# ② 单文件
uv run main.py -l song.lrc -a song.flac -o song.kara.lrc

# ③ 批量：递归找同目录同名的 lrc/音频对；配不上对的 LRC 跳过并汇总
uv run main.py --batch-dir /path/to/album --output-dir /path/to/out
```

分离 worker 与服务都可以单独自检，不需要主程序：

```bash
uv run --script scripts/separator_worker.py --info
uv run --script scripts/separator_worker_audio_separator.py --list-models
```

时间轴整体偏了（自动估计不可靠、需要人工判断），用图形工具对照波形调偏移并写回歌词：

```bash
uv run --group gui python scripts/offset_gui.py song.lrc --audio song.flac
```

`uv run main.py --help` 是完整的开关清单。

## 常用开关

| 开关 | 说明 |
|---|---|
| `--lyrics/-l`、`--audio/-a`、`--output/-o` | 单文件模式的输入输出 |
| `--batch-dir`、`--output-dir` | 批处理的输入目录与输出根目录 |
| `--separator-backend {demucs,audio-separator}`、`--separator-model` | 用哪个分离 worker、哪个模型（默认 `demucs` + `UVR_Demucs_Model_1`） |
| `--separator-device`、`--separator-model-dir`、`--separator-timeout` | 分离设备、模型仓库目录、单次请求超时（默认 900s） |
| `--aligner-backend {hfa,qwen3}`、`--aligner-url`、`--aligner-timeout` | 对齐后端（默认 `hfa`）、地址、超时（默认 120s） |
| `--aligner-language {auto,zh,ja,en,yue,ko}`、`--target-lang` | 送给对齐器的语言 / 只对齐该语言的行 |
| `--offset`、`--no-offset-estimate` | 手动全局偏移（ms）/ 完全不偏移 |
| `--min-vocal-activity` | 低于该归一化人声活动度的行不对齐（默认 0.01） |
| `--refine-collapsed-words` | 把零长度的词摊进其后的空隙（**推断值**，默认关闭） |
| `--trim-line-tail` | 把每行的音频窗口裁到「人声结束 + 300ms」，免得行尾拖进间奏（默认关闭） |
| `--metadata-filter`、`--strict-pairs` | 元数据行配置 / 要求所有 LRC 都有配对音频 |
| `--dump-dir`、`--sep-work-dir` | 调试音频导出目录 / 分离中间产物目录 |
| `--existing-byword-policy {realign,preserve}`、`--fail-fast` | 已有逐字标签的行怎么办 / 批处理遇错即停 |

## 现在到哪了

- **默认对齐后端是 HubertFA**（为歌声训练，10ms 帧）。日文实测零长度单元 1.9%、行区间覆盖率
  88.8%；代价是日文路径会丢促音、汉字读音靠猜。可选的 `qwen3` 后端量化在 80ms、零长度单元
  22.5%（最高一首 41.8%）。
- **全局偏移估计不可靠**，所以自动偏移极度保守：证据不足就不动，宁可不动也不要动错。
- **歌词错一个字会「从错处往后全毁」**，而且产物上看不出异常。
- **LRC 与音频版本不匹配的输入无解**，需要人工发现。

完整的限制清单在 [docs/current/limits.md](docs/current/limits.md)。

## 文档怎么读

文档分三层：**现状**（现在是什么样）、**记录**（凭什么这么说，带日期的实测）、**归档**
（以前是什么样）。入口是 [docs/README.md](docs/README.md)。

| 我想知道 | 去哪 |
|---|---|
| 数据怎么流动、worker 协议、代码布局 | [docs/current/pipeline.md](docs/current/pipeline.md) |
| 对齐契约、两个后端、零长度单元怎么处理 | [docs/current/aligner.md](docs/current/aligner.md) |
| 时间轴偏了怎么办 | [docs/current/offset.md](docs/current/offset.md) |
| 用图形工具调偏移、写回歌词 | [docs/current/offset-gui.md](docs/current/offset-gui.md) |
| 某行字幕被当署名吃掉了 | [docs/current/metadata-filter.md](docs/current/metadata-filter.md) |
| 环境装不上、GPU 没生效 | [docs/current/environments.md](docs/current/environments.md) |
| 哪些坑还在 | [docs/current/limits.md](docs/current/limits.md) |
| 某个数字是怎么测出来的 | [docs/records/](docs/records/) |
| 修过什么、试过什么 | [docs/archive/](docs/archive/) |

## 开发

```bash
uv run ruff check . && uv run ruff format --check .
uv run ty check
uv run pytest                  # 需要模型/GPU 的用例默认跳过，见 pytest.ini 的 integration 标记
```

仓库没有 CI，这四件事都是手动跑。

## 参考资料

- [SPL Format](https://moriafly.com/standards/spl.html)
- [UVR Models](https://github.com/TRvlvr/model_repo/releases/)
- [Demucs](https://github.com/adefossez/demucs)
- [python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator)
- [HubertFA](https://github.com/wolfgitpr/HubertFA)
- [Qwen3ForcedAligner](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B)
- [Gentle container](https://hub.docker.com/r/lowerquality/gentle)（早期尝试，未接入）

`samples/` 下的样本文件版权归各自的原始创作者所有。
