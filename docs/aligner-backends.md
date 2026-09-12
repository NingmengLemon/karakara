# 强制对齐（forced alignment）候选后端调研

> **这是一份调研记录，不是实现决策**：仓库当前仍然只用 `Qwen3-ForcedAligner-0.6B`，下面没有任何候选被集成。留在这里是因为它回答了一个具体问题——**80ms 精度与零长度词是不是选型问题**。结论是「是架构问题，不是配置疏漏」（见 §0 第 1 条），因此换后端才是唯一的出路，这份调研列出了候选与验证方法。
>
> 调研过程未改动项目任何文件；临时产物只写在 `E:\Projects\Python\karakara\tmp\aligner_research_*`；`D:\MUSIC` 未访问。
> 调研日期：**2026-09-12**。本机：Windows / Python 3.12.13 / RTX 4060 Laptop 8GB（驱动 610.62）/ 主 `.venv` 无 torch（已复核）。

**证据标注约定**（全文适用）：

| 标记 | 含义 |
|---|---|
| 【我实测】 | 本次调研在本机真实跑过的命令/脚本，附日志路径 |
| 【官方】 | 官方 README / 文档 / 模型卡 / 源码 |
| 【论文】 | 论文中的实验数字 |
| 【他人实测】 | 第三方博客、issue、项目文档中的经验报告 |
| 【未确认】 | 没能核实，**不要当作事实使用** |

> **未完成项（诚实记录）**：两个最被看好的候选（WhisperX align-only、HubertFA ONNX）都没能端到端真跑——卡在模型下载速度（GitHub Releases ~50KB/s、HuggingFace ~0.65MB/s）。所以它们的**质量数字全部标为未确认**；已核实的是**结构、帧移、语言覆盖、依赖、许可**这些源码/元数据事实。待验证清单与判定标准见 §8。

---

## 0. 结论速览

1. **现任后端（Qwen3-ForcedAligner-0.6B）就是当前文献里最新的那一类 LLM 非自回归对齐器**。`arXiv:2601.18220` 的 "LLM-ForcedAligner" 明确写 "Checkpoint and inference code are available at QwenLM/Qwen3-ASR"，而该仓库 README 介绍的就是「一个新颖的非自回归语音强制对齐模型，覆盖 11 种语言」【论文+官方】。模型 config 实测 `timestamp_segment_time: 80`、`classify_num: 5000`（5000 × 80ms = 400s 上限）【我实测，见 §1】。**所以「换个更新的 Qwen 对齐器」这条路不存在**，80ms 是"离散时间槽（slot-filling）"这一设计的固有量化，不是配置疏漏。
2. 要突破 80ms，**必须换架构**。技术上可达 10–20ms 的候选有：WhisperX(wav2vec2 CTC)、torchaudio `forced_align` 自建、ctc-forced-aligner(MMS)、MFA 3.x、SOFA/HubertFA（歌声专用）、Charsiu（仅中英）。
3. 结合本项目约束（日文 VOCALOID 为主、**逐行 2–8s 片段 + 行文本**、Windows、主环境不装 torch、私人项目），**最值得先做原型的三个**是：
   - **① WhisperX 的 align-only 路径**（20ms、ja/zh 额外给到**字级**、逐行调用天然支持、Windows 同构环境已验证过 torch）；
   - **② HubertFA**（唯一「为歌声训练 + 内置日文词典 + 10ms 帧 + ONNX 推理不需要 torch」的组合）；
   - **③ 自建 torchaudio `forced_align` + 自选 CTC 模型**（把 ctc-forced-aligner 的思路自己实现，因为那个包**在本机 Windows 上构建失败**【我实测】）。
4. **唱歌这件事是有实测证据的**：STARS（ACL Findings 2025）在中文歌声上给了直接对照——**MFA BER 40.3 / IOU 56.8，SOFA BER 20.9 / IOU 80.0**【论文】。朗读训练的通用对齐器在歌声上确实退化，**MFA 是其中最差的那一档**。
5. 一个容易被忽略的**架构性风险**：现在「逐行切段 + 行文本」的做法隐含假设「这一行的文本与这一行音频逐字对应」。日文 VOCALOID 里长音拖腔、英语化发音、重复段落很常见，错配时 CTC 对齐器会把误差**平均摊派**到整行（不会报错，只会变模糊）。所以对比实验里必须包含「错字鲁棒性」和「片段边界扰动稳定性」两项，见 §6。

---

## 1. 基线复核：现任后端的 80ms 到底锁在哪

| 项 | 值 | 证据 |
|---|---|---|
| `Qwen/Qwen3-ForcedAligner-0.6B` config | `"timestamp_segment_time": 80`、`"timestamp_token_id": 151705` | 【我实测】直接拉 HF `raw/main/config.json` |
| 对齐槽位数量 | `thinker_config.classify_num: 5000` → 5000 × 80ms = **400s 音频上限** | 【我实测】同上 |
| 支持语言（config 自述） | Chinese, Cantonese, English, German, Spanish, French, Italian, Portuguese, Russian, Korean, **Japanese** | 【我实测】config `support_languages` |
| 许可 | Apache-2.0（`Qwen/Qwen3-ForcedAligner-0.6B-hf` 模型卡） | 【我实测】HF API |
| 是否可调细 | **否**：时间戳是离散 token 索引（slot-filling），80ms = 400s/5000，改不了 | 【论文 arXiv:2601.18220 + 我实测 config】 |

> 结论：**这条线上的努力到此为止**。80ms 与「零长度词」是同源问题（真实时长 <1 个槽位的单元必然塌成 0 长度），`docs/aligner.md` 的判断是准确的。

---

## 2. 候选总表

「分辨率」= 模型帧移 / 输出粒度。**这是本次调研的核心指标。**

| # | 后端 | 许可 | 中/日/英 | 帧移 | 输出粒度 | 唱歌证据 | torch | Windows | 逐行片段集成 |
|---|---|---|---|---|---|---|---|---|---|
| 0 | Qwen3-ForcedAligner-0.6B（现状） | Apache-2.0 | ✅/✅/✅ | **80ms** | 日=nagisa 词、中=字 | 未确认 | 需要 | ✅ 已验证 | ✅（HTTP 服务） |
| 1 | **WhisperX**（只用 align） | BSD-2 | ✅/✅/✅ | **20ms** | 词；**ja/zh 另给字级** | 他人实测（卡拉OK 工具在用） | 需要 | ✅ 同构环境可用 | ✅ 按 segment 切音频 |
| 2 | **HubertFA** | Apache-2.0 | ✅/✅/✅/粤 | **10ms** | 音素 + 词 | **为歌声训练** | **不需要**（ONNX） | ✅ onnxruntime 实测可装 | ✅ 文件级（wav+lab） |
| 3 | **torchaudio `forced_align`**（自建） | BSD-2 | 取决于所选模型 | **20ms**（wav2vec2/MMS/HuBERT） | 音素/字/词（自己聚合） | 无（取决于模型） | 需要 | ✅ 已验证 | ✅ 进程内函数 |
| 4 | ctc-forced-aligner | BSD-2 代码 + 默认模型 **CC-BY-NC-4.0** | 1130 语言（含 zh/ja） | **20ms** | 句/词/**字** | 未确认 | 1.0.2 已改 ONNX | ❌ **构建失败**【我实测】 | ⚠️ 需自己拼函数 |
| 5 | MFA 3.x | MIT（代码）/ **CC-BY-4.0**（模型） | ✅/✅/✅ | **10ms**（`--frame_shift 1` 可 1ms，实验性） | 音素 + 词 | ❌ **歌声上 BER 40.3**（最差） | 不需要 | ⚠️ 必须 conda（pip 后 import 失败【我实测】） | ⚠️ 语料库式 CLI |
| 6 | SOFA | MIT | 中文（拼音词典）；日文需自建 | 10ms（mel 441/44100；**自身 ckpt 未核对**） | 音素 + 词 | **强**（歌声专用；STARS BER 20.9） | 训练用 3.8 conda；有 ONNX 推理 | ⚠️ 模型只在 GitHub Discussions 分发 | ✅ 文件级 |
| 7 | stable-ts | MIT（**仓库已归档**） | ✅/✅/✅ | 20ms（whisper 交叉注意力 DTW） | 词 | 官方提示 `demucs=True, vad=True` for music | 需要 + whisper 权重 | ✅ sdist 构建/导入成功【我实测】 | ⚠️ 每次调用跑一遍 whisper |
| 8 | NFA（NeMo） | Apache-2.0 | 中文有 CTC 模型；**日文无** | 40ms（Conformer 4×）/ 80ms（FastConformer 8×） | token/词/段 | 未确认 | 需要（NeMo 很重） | ❌ 官方不支持 | ⚠️ manifest 批处理 |
| 9 | Charsiu / CharsiuG2P | MIT | 仅 **en + zh**（日文不在路线图） | **10ms**（`*_10ms` 模型） | 音素/字 | 未确认 | 需要（老版 s3prl） | 未确认（2022 年停更） | ⚠️ |
| 10 | SpeechBrain | Apache-2.0 | 需自备模型 | 随模型（20ms） | 音素 | 未确认 | 需要 | 未确认 | ⚠️ |
| 11 | k2 / icefall | Apache-2.0 | 英文为主 | 10ms×降采样 | 音素/token | 未确认 | 需要 | ❌ **k2 无 win 轮子**【我实测】 | ❌ 整句 + FST |
| 12 | aeneas | **AGPL-3.0** | 靠 TTS 参考（espeak） | 粗（片段级设计） | **片段级** | 未确认 | 不需要 | ⚠️ 仅 sdist，需编译 C 扩展；最后版本 2017 | ❌ 文本片段级 |
| 13 | gentle | MIT | 英 | — | 音素 | — | kaldi | ❌ | 已退役 |
| 14 | lyric-align | MIT | ✅/✅/✅（CJK-first） | 继承 ASR（≈20ms） | 字/行（**非强制对齐**） | 官方定位=歌声/说唱 | 不需要（core 零依赖） | ✅ 实测 import OK | ✅ 支持喂外部时间戳 |

---

## 3. 逐个候选详述

### 3.1 WhisperX（推荐：只取 `align()`）

- 链接：[github.com/m-bain/whisperX](https://github.com/m-bain/whisperX)（PyPI 3.8.6，2026-08-30 仍在提交）、[arXiv:2303.00747](https://arxiv.org/abs/2303.00747)、双语对齐模型表见 `whisperx/alignment.py`。
- **许可**：仓库 **BSD-2-Clause**（GitHub API）【我实测】。默认 ja/zh 对齐模型 `jonatasgrosman/wav2vec2-large-xlsr-53-japanese` / `-chinese-zh-cn` 是 **Apache-2.0**（HF API）【我实测】。→ 私人项目毫无问题。
- **语言**：`DEFAULT_ALIGN_MODELS_HF` 覆盖 30+ 语言，含 ja/zh/ko（源码）【我实测】。非英语不靠 G2P 词典，而是**直接用该语言的 wav2vec2 CTC 模型**。
- **分辨率（核心）**：wav2vec2 特征 320 采样 @16kHz = **20ms/帧**；`align()` 内部用 `torchaudio.functional.forced_align` 建 trellis，再按 `ratio = duration * waveform_segment.size(0) / (trellis.size(0)-1)` 把帧索引换成秒（源码）【我实测】。**关键**：源码里 `LANGUAGES_WITHOUT_SPACES = ["ja","zh"]`，对中日文**额外返回 `chars` 字段（字级）**——这正好命中「中文按字、日文按 mora/字」的需求。
- **唱歌适配**：官方没有任何唱歌声明。他人实测层面：UltraSinger / USKMaker / nightingale 等卡拉OK 工具用 WhisperX 做歌词对齐；nightingale 明确对 CJK 做 per-character 对齐。已知问题：issue #1308（想用**自己的歌词**做强制对齐时找不到干净入口）、issue #1127（v3.3.3 日语对齐比旧版退化、score=0）。**风险点**：ja 对齐模型是朗读数据训练的，歌声退化程度未知。
- **依赖 / Windows**：torch + torchaudio + transformers + nltk（会下载 punkt）+ ffmpeg（`load_audio` 走 ffmpeg）。diarization 才需要 pyannote，对齐不需要。**本项目已有跑通 CUDA torch PEP 723 worker 的成熟模式**，所以 Windows 不是障碍。
- **集成方式**：`whisperx.align(segments, model_a, metadata, audio, device)` 本身就是**按 segment 切音频**的循环 → 可以「一行 = 一个 segment（`start=0, end=片段时长`）」，与现有 `AbstractAligner.align(audio_segment, text)` 契约同构。**不需要跑 ASR**（自己构造 segments 即可），但安装 whisperx 会连带拉 faster-whisper。
- **证据强度**：源码级【我实测读了 alignment.py】+ 官方文档；分辨率与字级行为是**源码事实**；唱歌表现只有【他人实测】。
- **落地成本**：中。一个 PEP 723 worker（模型加载 + HTTP 服务）+ 一个 `aligner/whisperx` 客户端实现；模型首次下载 ~1.2GB（ja large-xlsr）/ 360MB（en base）。

### 3.2 HubertFA（推荐：歌声专用 + 10ms + 免 torch）

- 链接：[github.com/wolfgitpr/HubertFA](https://github.com/wolfgitpr/HubertFA)（Apache-2.0，61★，2026-03 最后提交）；模型在 [Releases v0.0.7](https://github.com/wolfgitpr/HubertFA/releases)（`1218_hfa_model_new_dict.zip`，**256MB**，下载量 1107）【我实测】。
- **定位**：README 原话 "A lightweight forced alignment tool specifically designed for singing voice, compatible with non-singing voice alignment. Based on SOFA and FoxBreatheLabeler"。额外做**非乐音音素识别**（呼吸声 AP/EP），对"分离后残留呼吸/齿音"这种实际场景是加分项。
- **语言**：`--language zh | ja | en | yue`，内置词典 `dictionaries/{opencpop-extension, japanese_dict_full, ds_cmudict-07b, jyutping_dict}.txt`（仓库内，已核对）【我实测】。
  - ⚠️ **日文词典是"罗马字音节 → 音素"**（`ka → k a`、`sha → sh a`、`cl`=促音、`N`=拨音）→ 送进去的 `.lab` 必须是**空格分隔的罗马字音节**，需要你自己做 kana/汉字→罗马字（例如 pykakasi）+ 长音/促音处理。这是本项目集成日文时**最容易低估的工作量**。
- **分辨率（核心）**：`configs/force_alignment.yaml` 的 `mel_spec_config: sample_rate 44100, hop_size 441` → **10ms/帧**（441/44100 = 0.01s）【我实测读源码】；`tools/infer_base.py` 用 `pad_length / hop_size` 做帧↔秒换算，确认输出时间单位就是 mel 帧。输出 TextGrid/HTK，含**词层 + 音素层**。
- **唱歌适配**：**这是它的设计目标本身**（不是"能不能用"的问题）。血统上继承 SOFA，而 SOFA 在 STARS 的歌声对齐评测里 BER 20.9 / IOU 80.0，显著优于 MFA 的 40.3 / 56.8【论文】。
- **依赖 / Windows**：ONNX 推理路径**不需要 torch**：`requirements_onnx.txt` = click / librosa<0.10.0 / matplotlib / numpy~=1.26.4 / PyYAML / tqdm / textgrid~=1.5 / pandas / onnx / onnxsim / onnxscript / **onnxruntime-gpu==1.19.0**（钉死 CUDA 12）【我实测读文件】。本机 `onnxruntime 1.30.0` 在 py3.12/Windows 安装并 import 成功（provider: Azure/CPU）【我实测】。
- **集成方式**：`python onnx_infer.py --onnx_path ... --wav_folder ... --language ja --dictionary ...` → **文件级**（目录下 wav + 同名的 .lab）。对「逐行片段」很自然（每行一个 wav），但要自己写 worker 包一层 + 把音素/词边界聚合回歌词单元。
- **证据强度**：**结构/分辨率是官方源码事实**（hop、词典、语言参数我逐个读过）；唱歌质量只有 SOFA 系的血统证据，HubertFA 自身没有第三方 benchmark（**未确认**）。代码出自个人项目，需要自己审一遍。
- **⚠️ 模型获取是真实的落地成本（我实测）**：官方 ONNX 模型只在 GitHub Releases，本机实测下载 **~50KB/s**（11.5 分钟只下到 34.6MB / 256MB）→ **端到端试跑本次未能完成**。脚本已写好待网络好转时一键复跑：`tmp\aligner_research_hfa_probe.ps1`（会下载模型 → 解压 → 用 `tmp\05_line_0.wav` + 拼音 lab 跑 `onnx_infer.py --language zh` → 打印 TextGrid 并落日志到 `tmp\aligner_research_hfa.log`）。
- **额外发现**：HF 上有**第三方 HubertFA 重训权重**——`Silasimo/HubertFA-SynthGT`、`Silasimo/HubertFA-GTSinger`、`Silasimo/HubertFA-combined`（2026-08，含 `config.yaml`/`vocab.yaml`/`.ckpt`），**在歌声数据集 SynthGT / GTSinger 上训练**，说明"HubertFA 系可以做唱歌专向微调"这条路是成立的。但它们的许可是 **CC-BY-NC-SA-4.0**（比官方代码的 Apache-2.0 更严，带 ShareAlike），且是 `.ckpt` 需要 torch 训练路径而非 ONNX。
- **落地成本**：中。新 PEP 723 worker（onnxruntime-gpu + librosa + 文本→罗马字预处理）+ 音素层→字/mora 的聚合 + 模型分发（GitHub Releases，256MB，非 HF，需要固定版本和校验和；本机实测该下载很慢）。

### 3.3 torchaudio `forced_align`（自建 CTC 对齐）

- 链接：[torchaudio forced alignment 教程](https://docs.pytorch.org/audio/stable/tutorials/forced_alignment_tutorial.html)、[多语言版（含中文）](https://docs.pytorch.org/audio/2.4.0/tutorials/forced_alignment_for_multilingual_data_tutorial.html)、PyPI `torchaudio` 2.11.0（BSD-2，有 win 轮子）【我实测 PyPI】。
- **语言**：取决于你挂哪个 CTC 模型。可选：`jonatasgrosman/wav2vec2-large-xlsr-53-japanese`（**字级**，Apache-2.0）、`-chinese-zh-cn`（字级）、MMS-300M（1130 语言，但模型许可 CC-BY-NC-4.0，且非拉丁文字要先罗马化）。
- **分辨率**：wav2vec2 / HuBERT / MMS 的 `inputs_to_logits_ratio = 320` → **20ms/帧**（`ctc-forced-aligner` 源码里显式断言 `window % ratio == 0`，`ratio = 320`）【我实测源码】。torchaudio 官方多语言教程**明确说明**：中文不需要分词即可做**字级**对齐，要词级才需要先分词。
- **唱歌适配**：无。这就是"自己拿一个模型跑 CTC Viterbi"，唱歌表现完全取决于模型选择。
- **依赖 / Windows**：torch + torchaudio。**本项目已有在 win32 + CUDA 13 上跑通 torch 2.14.0+cu130 的实证**（`docs/environments.md`），无 flash-attn / deep_grad 之类 Linux-only 依赖。
- **集成方式**：进程内函数，天然适配「短片段 + 文本」。你要自己补：文本→模型词表 id（`<star>`/blank 处理）、`torchaudio.functional.forced_align` 调用、frame→time、span 合并、词/字聚合。
- **证据强度**：官方教程（分辨率/字级行为）。
- **落地成本**：中（代码量比 ① 多一些，但**没有新框架、没有第三方包风险、许可最干净**）。本质上是**避开 ctc-forced-aligner 的 Windows 构建问题，自己实现它的核心**。

### 3.4 ctc-forced-aligner（思路很好，但本机装不上）

- 链接：[github.com/MahmoudAshraf97/ctc-forced-aligner](https://github.com/MahmoudAshraf97/ctc-forced-aligner)（BSD-2，560★，2026-09-07 仍在提交）、默认模型 `MahmoudAshraf/mms-300m-1130-forced-aligner`。
- **许可（重要）**：代码 BSD-2，但**默认模型是 CC-BY-NC-4.0**（官方 README 明写 "note that the default model has CC-BY-NC 4.0 License"）→ 私人项目可用，**任何商业化/分发都要换模型**。
- **语言 / 分辨率**：1130 语言（含 zh/ja 的 ISO-639-3）。非拉丁文字必须 `--romanize`（走 uroman）→ 中文/日文都会被罗马化，**日文汉字的读音还原是明确风险**（uroman 对汉字读音的处理能力**未确认**，但多音字/熟字训必然不可靠）。`--split_size sentence|word|char` → **可以直接要字级**。
- **已知内部行为**：`get_spans()` 对空 token 会 append `(seg_idx, seg_idx)` → **它自己也会产生零长度 span**（源码）【我实测】。也就是说换过来并不能天然消灭零长度词，需要你后处理。
- **Windows 可用性**：**❌ 本机实测失败**。PyPI 上 `ctc-forced-aligner` 只有 sdist（无 wheel）；用 uv 构建 1.0.2 时 MSVC 链接失败：
  ```
  LINK : error LNK2001: 无法解析的外部符号 PyInit_align_ops
  fatal error LNK1120: 1 个无法解析的外部命令
  ```
  日志：`tmp\aligner_research_install_probe.log`。**注意这不是"没装编译器"**——本机有 VS BuildTools，`cl.exe` 已被正常调用、`.obj` 已生成，是包自身的 `pybind11` 模块名/导出与 `main.cpp` 不匹配（main branch 看起来正在重构，`forced_align_impl.cpp` 与 `main.cpp` 并存）。另外注意 pip 元数据里 1.0.2 已把 torch 降为 extra，主依赖变成 `onnxruntime`（想走 ONNX 路线），但**构建这一关先卡住了**。
- **结论**：**不要直接依赖这个包**；但要抄它的三件事——`--split_size char` 的粒度设计、`<star>` token 处理非对应文本、以及"20ms + 字级"的组合。【我实测】

### 3.5 MFA 3.x（10ms 很诱人，但两处硬伤）

- 链接：[Montreal-Forced-Aligner](https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner)（**MIT**，2026-08 活跃）、[MFA 3.x 文档](https://montreal-forced-aligner.readthedocs.io/)、模型库 [mfa-models](https://mfa-models.readthedocs.io/)、[arXiv:2606.18466](https://arxiv.org/html/2606.18466v1)（Interspeech 2026）。
- **语言**：官方预训练含 **`japanese_mfa`（声学 + 词典，日语）、`mandarin_mfa`、`english_mfa`、韩语**等。论文明确 "evaluates MFA's performance across English, Japanese, and Korean"，**mean boundary error < 15ms**（TIMIT 12.1ms / Buckeye 13.9ms / 韩语 14.8ms；日语是新增模型，无 1.0 对照）【论文】。模型许可 **CC BY 4.0**（japanese_mfa 3.0.0 模型卡）【官方】。
- **分辨率（核心）**：MFCC 25ms 窗 / **10ms 帧移**（MFA 原始论文）；**phone 最短时长 = 1 帧 = 10ms**（2.0 changelog）；`mfa align --frame_shift 1` 可到 **1ms**（官方标注 experimental）。→ **理论上精度是本次调研里最好的之一**。
- **硬伤 1：歌声退化（有直接实测数字）**。STARS 论文在**中文歌声**上对照：**MFA BER 40.3 / IOU 56.8**，SOFA 20.9 / 80.0。MFA 是这一栏里最差的【论文，Table 1】。
- **硬伤 2：Windows 只能走 conda，且工作流是"语料库式"**。本机实测：
  ```
  pip 安装 montreal-forced-aligner 后 import →
  ModuleNotFoundError: No module named '_kalpy'
  ```
  （日志 `tmp\aligner_research_install_probe2.log`）→ 必须用 conda 装 `kalpy`/`kaldi`（官方 Windows 路径：`conda create -c conda-forge -p C:\...\envs\aligner montreal-forced-aligner=3.2.1`）【官方/他人实测文档】。这会**打破本项目「uv + PEP 723 独立环境」的既有模式**。
  集成上它是 `mfa align <corpus_dir> <dict> <acoustic> <out_dir>` 的**文件→文件 + 需要独立输出目录（会清空）**工作流，不是 Python 函数 API；且社区经验是「wav 末尾要留 20–50ms 余量，否则容易跑偏」【他人实测】，对"逐行紧切片段"是直接相关的坑。
- **证据强度**：分辨率与日文支持=官方/论文；Windows 安装失败=【我实测】；歌声退化=【论文】。
- **落地成本**：**高**（新 conda 环境 + 语料库 I/O + 日文 OOV/词典处理）。**建议只作为对照基线，不作为主后端。**

### 3.6 SOFA（唱歌对齐的"祖师爷"）

- 链接：[github.com/qiuqiao/SOFA](https://github.com/qiuqiao/SOFA)（**MIT**，238★，**2026-09-02 仍在提交**）。README 自述：针对歌声设计，相比 MFA「安装更容易、效果更好、推理更快」。
- **分辨率**：mel 谱驱动，DiffSinger 系约定 hop 441 / 44100 = **10ms**（同族 HubertFA 已从源码确认 441/44100）；**SOFA 自身 ckpt 的 mel 配置我没能直接核对（模型只以 `.ckpt` 形式在 Discussions 分发）→ 标为未确认**。
- **语言**：**词典驱动**，默认 `dictionary/opencpop-extension.txt`（**中文拼音音节**）。日文需要自建词典（社区做法：HubertFA 的 `japanese_dict_full.txt` 就是这条路）。`infer.py --g2p` 可换 G2P 模块（Dictionary/Phoneme/None，文档已核对）。
- **集成**：`python infer.py --ckpt ... --folder segments/`，目录里 `*.wav` + 同名 `*.lab`（一行空格分隔的转写）→ 输出 TextGrid / HTK(lab,nnsvs,sinsy) / transcriptions.csv(diffsinger)，**含音素与词两级**；另有 `onnx_infer.py`（onnxruntime）与 `-m match` 模式（只取最优连续子序列，容忍文本与音频不完全对应）。
- **最大不确定性**：**模型分发不规范**——官方仓库里 `ckpt/` 是空的，模型在 GitHub Discussions 的帖子里以附件形式共享，没有固定 URL/校验和/许可声明。**这是把它放第一位的主要障碍**（可复现性/供应链）。
- **落地成本**：中（新环境 + 词典/文本预处理 + 音素→词聚合 + 模型获取流程固化）。

### 3.7 其他「唱歌向」方案（了解即可）

| 名称 | 许可 | 要点 | 结论 |
|---|---|---|---|
| [colstone/SOFA_AI](https://github.com/colstone/SOFA_AI) | MIT | FunASR（ASR 自动出文本/拼音）+ SOFA：无标注干声 → 音素标注；中文/英文；README 含针对某开源组织的对抗性声明、且自述"代码由 ChatGPT 辅助生成" | 作为**思路参考**（ASR 先给文本/拼音再对齐）；不建议直接依赖 |
| [schufo/lyrics-aligner](https://github.com/schufo/lyrics-aligner) | MIT | **专门在 MUSDB18 歌声数据上训练**的词/音素 onset 预测 DNN；官方称"最适合类似音乐，也能用于纯人声" | **仅英文 ARPAbet**，预处理要 CMUdict/espeak，2021 后未更新 → 参考价值 > 落地价值 |
| [STARS](https://github.com/gwx314/STARS)（ACL Findings 2025） | MIT（代码） | 统一歌声标注（对齐/音符/技法/风格），多级粒度；本报告的 MFA vs SOFA 数字来自它 | 研究框架，非即插即用；**是评估口径的权威参考**（BER/IOU） |
| VocalParse（arXiv:2605.04613）/ LLM-ForcedAligner | — | 前者基于 SOFA2 重训做歌声词级时间戳；后者见 §0 | 论文阶段，**未见可直接用的权重**（未确认） |

### 3.8 明确不建议的

| 候选 | 理由 |
|---|---|
| **aeneas** | **AGPL-3.0**；最后版本 1.7.3（2017）；PyPI 仅 sdist（无 win wheel，需编译 C 扩展）【我实测】；设计目标是**文本片段级**（行级）同步，不是词级；非英语靠 TTS 参考（espeak） |
| **gentle** | 本项目已试过并退役；最后 release 0.11.0（2023-03）【我实测 GitHub API】；依赖 kaldi(DNN) + Docker，Windows 上历来困难 |
| **k2 / icefall** | **PyPI 上 k2 只有 macOS 轮子**（`k2-1.24.1-cp3xx-macosx_*.whl`，无 win）【我实测】→ Windows 需自行编译，实际不可行；且 icefall 官方文档对 forced alignment 的做法是"用 MFA"，自身是整句 FST/CTC 对齐 |
| **NFA（NeMo）** | **日文缺**（NeMo 没有官方日语 CTC 模型；有他人实测报错 `Model is not an instance of NeMo EncDecCTCModel`——拿 transducer 模型做不了 NFA）；帧移取决于模型（Conformer 4× → 40ms，FastConformer 8× → 80ms）【论文/官方】；NeMo toolkit 依赖极重（lightning/hydra/wandb/datasets/sacrebleu…，PyPI 元数据已核对）+ 官方不支持 Windows |
| **Charsiu** | MIT，但 README 的语言进度表**只实现了 English + Mandarin**，日文不在路线图；仓库 2022-09 后停更（老版 transformers/s3prl，py3.12 风险）→ 只做中文曲目时才值得回头看 |
| **whisper-timestamped** | **AGPL-3.0**（GitHub API 实测）+ `dtw-python` 是 GPL-3。私人自用可以，但 AGPL 的传染性会影响将来任何分发形态 → 有同等效果但许可干净的 WhisperX，没必要选它 |
| **stable-ts** | 上游仓库已 **archived**（`archived=True`，2.19.1，2025-08）【我实测】；且 `model.align()` 每次调用都要重跑 whisper 解码，逐行 2–8s 片段 × N 行的成本模型不划算 |
| **ctc-forced-aligner（包本身）** | Windows 构建失败【我实测】；默认模型 CC-BY-NC-4.0；自己实现核心更省事（见 §3.3） |

---

## 4. 集成方式对照（本项目最关心的那条约束）

本项目现有契约是 `align(audio_segment, text, language) -> [(start_ms, end_ms)]`（逐行短片段）。

| 后端 | 能否"逐行片段 + 行文本" | 备注 |
|---|---|---|
| Qwen3-ForcedAligner（现状） | ✅ | HTTP 服务，天然支持 |
| **WhisperX** | ✅✅ | `align()` 按 segment 切音频，**本来就是逐段循环**；且 ja/zh 额外给字级 |
| **HubertFA / SOFA** | ✅ | 文件级（每行一个 wav + .lab）；自己写 worker 包一层 |
| **torchaudio forced_align** | ✅✅ | 进程内函数，最贴合 |
| ctc-forced-aligner | ⚠️ | CLI 是"整文件 + 整文本"；函数级可拼装，但包装不上 |
| MFA | ⚠️ | 语料库式 CLI + 独立输出目录；短片段要两端留余量 |
| NFA | ⚠️ | NeMo manifest（audio_filepath + text）批处理 |
| stable-ts / whisper-timestamped | ⚠️ | 设计成"整段音频 + 文本"，逐行要反复重跑解码器 |
| aeneas | ❌ | 文本片段列表 ↔ 单个音频文件的全局同步 |
| k2/icefall | ❌ | 整句 + FST/LG 结构 |

**重要提示**：CTC 类对齐器对「片段边界切在音素中间」和「文本与音频不完全对应」都敏感。逐行送片段时建议两端各留 100–200ms padding，并把返回的越界单元裁剪/合并（`match` 模式、`<star>` token、`merge_threshold` 这类参数就是为此存在的）。

---

## 5. 推荐排序 + 各自的最小验证实验

> **⚠️ 排序已经被实证推翻了：见 §9。** 2026-09-13 补做的验证显示，**"是不是为歌声训练"才是决定性因素**，而不是架构（CTC / 离散时间槽）：朗读训练的 wav2vec2 CTC 在歌声片段上完全退化（99.5% 的音素只占 1 帧、覆盖率 2.5%、贪心转写是乱码），而歌声训练的 HubertFA 在同一类材料上给出连续、可用的区间（零长度 0%、覆盖率 80%）。所以**实际优先级应是 HubertFA（及同源的 SOFA 系）> WhisperX/自建 CTC**。
>
> 所有实验都应复用同一批输入：同一首歌、同一份分离人声、同一份行切分、同一份行文本。中间产物写到 `tmp\aligner_research_*` 或 `tmp\*_eval`。

### ① WhisperX align-only（首选：改动最小、许可最干净、字级）

**最小验证实验**：取现有 6 首（`失う`/`SACRA`/`蒲公英`/samples 三首）中的**任意 1 首**，把 `tmp\05_line_*.wav`（或现流程重新切的行片段）逐行送 `whisperx.align()`（每行一个 segment；ja 用 `jonatasgrosman/wav2vec2-large-xlsr-53-japanese`），输出 `words` + `chars`，然后：
1. 统计**边界栅格**是否 20ms（以及取值是否只有 20ms 的整数倍）；
2. 统计**零长度单元比例**（与现状 22.5% 对照，尤其看「蒲公英」41.8% 那首）；
3. 统计**首/末单元锚定误差**（首个单元的 start 是否 ≈0、末单元 end 是否 ≈片段时长）；
4. 与现后端做**同位置边界差**（中位数 / P90），但**只当作"差异"不当作"精度"**（见 §6）。

**风险**：ja 模型是朗读数据；若 1 首上零长度率没有明显下降，就换 ② 或 ③。

### ② HubertFA ONNX（首选：唯一"为歌声训练 + 日文就绪 + 10ms + 免 torch"）

**最小验证实验**：
1. 用一首**日文** VOCALOID（如 `P丸様。 - メズマライザー`）的一行：kana→罗马字（pykakasi 等）→ `.lab`，跑 `onnx_infer.py --language ja`；
2. 同时用一首**中文**（`洛天依 - step on your heart`）跑 `--language zh`（pypinyin 出拼音，不需要罗马字转换，链路更短）；
3. 量：**10ms 栅格比例**、**零长度音素比例**、**音素→字/mora 聚合后的零长度比例**、以及**与现后端字级边界的差异分布**；
4. 顺带量**耗时**（ONNX CPU vs CUDA）——逐行调用下每行 2–8s，ONNX CPU 可能也够快。

**风险**：日文罗马字预处理（长音/促音/汉字多音）会直接决定成败；模型分发在 GitHub Releases（要固定版本 + 校验和）；个人项目，代码要自己审。

### ③ 自建 torchaudio `forced_align` + 自选 CTC 模型（首选：完全可控、许可最干净）

**最小验证实验**：同一行片段，两条模型路线并行跑：
- **A：`jonatasgrosman/wav2vec2-large-xlsr-53-japanese`（字级，Apache-2.0）** → 直接得字级，无需 G2P；
- **B：`MahmoudAshraf/mms-300m-1130-forced-aligner`（1130 语言）** → 但非拉丁文字要罗马化，**用来验证"日文汉字罗马化"这条路是否可行**（我预判风险高）。

量同一组四项指标（栅格/零长度/锚定/与现后端差异），外加**扰动稳定性**（见 §6 的 D2），因为这条路线的模型/参数全在你手里，最值得花时间调。

**备选**：不想碰 torch 时，可用 `onnx-community/mms-300m-1130-forced-aligner-ONNX`（int8/fp16/quantized 多个变体，**实测 onnxruntime 在 win/py3.12 可装**）自写 Viterbi——注意该 ONNX 转换的许可仍是 **CC-BY-NC-4.0**（继承 MMS 微调模型）。

---

## 6. 方法论提醒 + 可复现的对比口径

### 6.1 评估对齐器时最容易顺手用的三条指标，哪条其实没判别力

> 这一节针对的是「评估**新后端**时该量什么」。项目**当前**并没有把这三条做成自动指标——现在只统计零长度占比（见 `docs/aligner.md`），所以下面是对"要不要加、加什么"的建议。

| 顺手的指标 | 问题 | 建议 |
|---|---|---|
| **与行首时间戳的一致性** | 本项目是**逐行送片段**，片段内时间原点就是行首 → 这个指标几乎**恒等于 0**，对区分后端毫无判别力（它其实在检验"行切分"而不是"对齐器"） | 改成 **片段首/末单元锚定误差**：`first_unit.start - 0`、`segment_duration - last_unit.end`（单位 ms），这才反映"对齐器认不认得片段边界" |
| **零长度词比例** | 不同后端的零长度**定义不同**：有的返回 `NaN`/未放置，有的把 0 长度 span 显式吐出（ctc-forced-aligner 源码里 `get_spans` 就会有 `(i, i)`），有的自动合并 | 统一定义为 **`duration_ms < EPS` 的单元占比**（EPS 取 10ms），并把"未放置/NaN"单列一栏，不要混进零长度 |
| **时间分辨率** | 只看"栅格"会漏掉两种情形：① 20ms 栅格的后端也可能输出非整倍数（浮点插值）② 80ms 栅格经过 `fix_timestamp` 插值后看着像 20ms（`docs/aligner.md` 的 SACRA 那行就是例子） | 同时记 **(a) 边界值分布的 gcd/众数间隔**、**(b) 相邻边界的最小非零间隔**、**(c) 有多少边界是"非栅格值"（插值痕迹）** |

### 6.2 建议的对比口径（一个后端 = 一份 JSON，字段固定）

```
run_id, backend, model_id, model_rev, device, language, song_id,
segment_index, segment_start_ms, segment_end_ms, segment_source(vocals|mix),
text_units_in (输入单元数/粒度: char|mora|word),
n_units_out, n_units_zero_len, zero_len_ratio,             # 见 6.1 的统一定义
n_units_unplaced,                                          # NaN/未放置（单列）
grid_step_ms_mode, non_grid_boundary_ratio, min_gap_ms,     # 分辨率三件套
first_unit_start_ms, last_unit_end_ms, anchor_err_head_ms, anchor_err_tail_ms,
monotonic_violations, text_mismatch_units,                 # 漏字/多字
median_abs_diff_vs_ref_ms, p90_abs_diff_vs_ref_ms,         # 与基线（现 Qwen）逐边界差
runtime_s_total, runtime_s_per_segment, cold_start_s
```
**四条硬性可比条件**（否则数字不可比）：
1. **同一批输入**：同歌、同分离人声产物、同行切分、同行文本；
2. **同一后处理**：只允许"单调性修复 + 零长度合并"，**禁止插值细分**（插值会伪造分辨率，`fix_timestamp` 就是先例）；
3. **同一统计粒度**：一律换算到**字/mora 级**再比（各后端原生粒度不同：音素/字/词）；
4. **同一计时口径**：每首歌单独计总时长，首曲冷启动单列（`docs/benchmarks.md` 已经踩过这个坑）。

### 6.3 没有 ground truth 时，用三个代理指标

- **C1 交叉一致性**：两个**架构不同**的后端（例如 WhisperX-CTC 与 HubertFA-音素）在同一行上边界差的中位数 → 一致性高说明两者都对（或都错），一致性极高（<20ms）反而说明可能共享同一偏置。
- **C2 扰动稳定性（最能暴露问题，且零标注成本）**：
  - 片段两端各加/减 **±100ms padding**，看行内边界漂移多少 ms；
  - **混音 vs 分离人声** 各跑一次；
  - 把同一行的文本**故意删掉一个字/多一个字**，看是否崩（→ 反映 VOCALOID 错字/重复段落的鲁棒性）。
- **C3 下游代理（直接对应用户体验）**：`可高亮单元占比` = 时长 ≥ 100ms 的单元比例；`逐字高亮停留时长` 的 p10/p50/p90；`同一单元边界抖动` 在渲染器里表现为"跳字"的比例。
- **只有愿意花 ~30 分钟手工标注一首歌的一行**（Praat/Vlabeler）时，才能算 **BER / IOU**，也**只有这时数字才能和 STARS 论文的 MFA 40.3 / SOFA 20.9 直接对话**。强烈建议至少在 1 行上做这件事——它是唯一能判定"20ms 后端是否真的比 80ms 后端更接近真值"的证据。

### 6.4 两个容易踩的解释陷阱

1. **量化误差陷阱**：拿新后端与现后端比"边界差中位数"时，现后端本身被量化在 80ms 栅格上，**这个差值天然会很大（期望 ~20–40ms 的量化噪声）**，它说明"两者不同"，**不能**说明"新后端更准"。要证明更准，只能靠 C2/C3 或手工标注。
2. **零长度≠不可用**：本项目当前把 0 长度词合并进相邻词（时间语义无损、文本不丢），所以"零长度率"下降不自动等于"高亮体验变好"；请同时看 **可高亮单元占比**（C3）和**高亮停留时长分布**——一个把 80ms 塌缩变成"每个字 5ms"的后端，零长度率很漂亮但体验更差。

### 6.5 一条架构性的替代思路（不作为本次排序，但值得知道）

[lyric-align](https://github.com/ijuinryukichi/lyric-align)（**MIT**，2026-07 活跃，PyPI 可装，core **零依赖**，py3.12 `import` 成功【我实测】）走的是**另一条路**：不做强制对齐，而是"**用已知歌词按字符级模糊匹配去锚定 ASR 输出的时间**"，输出 LRC/SRT/ASS(逐字 `\k`)/JSON，自称 CJK-first、为歌声和说唱设计，还支持 `--segments` 直接喂入外部词级时间戳。它的价值在于：**当"文本与音频不完全对应"时（VOCALOID 长音拖腔、非标准发音、重复段落），强制对齐会静默摊派误差，而"只借时间不借文本"的做法更稳**。可以把它当成"零成本的对照实现"，用来交叉验证 C1。它自己不是对齐器，所以不解决 80ms 分辨率问题（除非换掉上游 ASR）。

---

## 7. 本次实测记录（可复核）

| 探针 | 结果 | 证据 |
|---|---|---|
| 主 `.venv` 是否含 torch | torch/torchaudio 均**未安装**（符合设计） | 实测 `importlib.util.find_spec` |
| Qwen3-ForcedAligner config | `timestamp_segment_time=80`、`classify_num=5000` | HF `raw/main/config.json` |
| 各仓库许可 | WhisperX BSD-2 / stable-ts MIT(**archived**) / whisper-timestamped **AGPL-3.0** / ctc-forced-aligner BSD-2 / MFA MIT / NeMo Apache-2.0 / torchaudio BSD-2 / speechbrain Apache-2.0 / icefall Apache-2.0 / charsiu MIT / SOFA MIT / HubertFA Apache-2.0 / SOFA_AI MIT / aeneas **AGPL-3.0** / gentle MIT | GitHub API，输出见正文 |
| ctc-forced-aligner 构建 | **失败**：MSVC `LNK2001 PyInit_align_ops`；PyPI 仅 sdist | `tmp\aligner_research_install_probe.log` |
| MFA pip 安装 | **import 失败**：`No module named '_kalpy'` | `tmp\aligner_research_install_probe2.log` |
| k2 Windows 轮子 | PyPI 只有 `macosx_*` 轮子 | PyPI API |
| onnxruntime（win/py3.12） | 1.30.0 安装+导入成功 | `tmp\aligner_research_install_probe2.log` |
| lyric-align（win/py3.12） | import 成功（core 零依赖） | 同上 |
| stable-ts（win/py3.12） | sdist 构建成功 + import 成功 | 同上 |
| 关键源码核对 | WhisperX `alignment.py`（ja/zh=字级、20ms trellis）；HubertFA `force_alignment.yaml`（441/44100=10ms、ja/en/zh/yue 词典）；ctc-forced-aligner `get_spans`（会产出零长度 span，ratio=320） | 见正文引用 |
| **网络吞吐（决定"能不能马上试"）** | GitHub Releases **~50KB/s**（HubertFA 256MB 模型 11.5 分钟仅 34.6MB，端到端试跑中止）；HuggingFace **~0.65MB/s**（→ 1.2GB 的 ja 对齐模型约需 **31 分钟**） | `tmp\aligner_research_hfa.log` |

> **未完成项（诚实记录）**：两个"端到端真跑"的候选（HubertFA ONNX、WhisperX align-only）都因为**模型下载速度**没能在本次调研窗口内跑完，因此本文中它们的**质量数字都标为未确认**；已核实的是它们的**结构、帧移、语言覆盖、依赖与许可**（均为源码/元数据事实）。HubertFA 的复跑脚本已就绪（`tmp\aligner_research_hfa_probe.ps1`）；WhisperX 建议先手动把模型拉到本地（`huggingface-cli download jonatasgrosman/wav2vec2-large-xlsr-53-japanese`）再跑，避免每次探针都重下。

---

## 8. 待验证清单（网络好转后按序做，投入很小时限内）

| 序 | 动作 | 判定标准（先定好，避免事后解释） |
|---|---|---|
| V1 | HubertFA ONNX：中文（pypinyin 链路最短）跑 1 行 | TextGrid 边界是否 10ms 整数倍；零长度音素比例；音素→字聚合后零长度比例 |
| V2 | HubertFA ONNX：日文（kana→罗马字）跑同行 | 与 V1 同口径 + 罗马字预处理是否吃掉了精度（对比"人工核对过的罗马字"与"自动转换"） |
| V3 | WhisperX align-only（ja 字级模型）跑同一批 6 首歌 | 零长度率 vs 现状 22.5%（重点看「蒲公英」41.8%）；边界栅格是否 20ms；锚定误差 |
| V4 | §6.3-C2 扰动测试（±100ms padding / 混音 vs 分离 / 删一个字） | 行内边界漂移中位数 < 1 个帧移（10/20ms）视为稳定 |
| V5 | 手工标注 1 行（Vlabeler/Praat）算 BER/IOU | 与 STARS 论文的 MFA 40.3 / SOFA 20.9 同口径对照——**唯一能证明"更准"的证据** |

---

## 9. 追加验证（2026-09-13）：V1、V3 已做，结论反转

### 9.0 下载通道（原本卡住的唯一原因）

模型下载慢是**通道**问题，不是墙：实测同一份 GitHub Releases 资产

| 通道 | 速度 |
|---|---|
| 直连 GitHub | **47 KB/s** |
| 本地代理 `http://127.0.0.1:7890` | 94 KB/s |
| `ghfast.top/<原 URL>` | 4.6 MB/s |
| **`gh-proxy.com/<原 URL>`** | **23 MB/s**（256MB 模型 11 秒下完） |

HuggingFace 直连本次实测 **7.8 MB/s**（早先的 0.65 MB/s 是瞬时波动），`hf-mirror.com` 3.0 MB/s，代理 3.7 MB/s。→ **HF 走直连、GitHub 走 gh-proxy.com** 是当前最省事的组合。

### 9.1 V3 的变体：「换成 20ms 的 CTC 就该更好」——**错了**

按 §6.2 的四条硬性可比条件，用**同一首歌（蒲公英）、同一批 17 个行片段**跑了两条路：

| 指标 | Qwen3-ForcedAligner（现状） | wav2vec2 CTC ja（`jonatasgrosman/…-japanese`，20ms 帧） |
|---|---|---|
| 单元数 | 141（nagisa 词级） | 211（字级） |
| 零长度 (<10ms) | 59（41.8%） | **0（0.0%）** |
| 恰好 1 帧的单元 | — | **210 / 211（99.5%）** |
| 单元跨度之和 ÷ 片段时长 | 49.5% | **2.5%** |
| 最小非零跨度 | 80ms | 20ms |

**那个 0% 零长度是假象**：CTC 把每个字都放在**单帧**上、其余全是 blank，于是"零长度"消失了、"时长"也一起消失了（覆盖率 2.5%）。再往下查一层——**模型在这些歌声片段上根本听不出内容**：

| 参考 | 贪心转写（模型听到的） | blank 占比 |
|---|---|---|
| `土の色　花が魅せた世界` | `水中のいルカダーダーミスーンャートャた` | 93.1% |
| `高く遠く羽を伸ばし届けと放つ種` | `入ターカャクタ空花ボのバ子糸がケト庭` | 96.7% |
| `風の中　空が背を押す` | `非グラーのかそラーが末こををます` | 94.6% |

→ **朗读数据训练的模型对歌唱无能为力**，这正是 §0 第 5 条与 STARS 论文警告的情形。**如果只看"零长度率"，这个后端会被误判为巨大改进。**

（本次踩到两个实现坑，记下来免得复现时再花时间：① 输入必须**重采样到 16kHz** 再送模型——喂 44.1kHz 会让帧数多出 2.76 倍；② 必须过 `processor.feature_extractor` 做零均值/单位方差归一化，直接喂裸波形会让 emission 变成噪声、Viterbi 退化成"每字一帧"。）

### 9.2 V1：HubertFA（歌声专用）在中文上是**可用**的

同样按 §6.2，用**另一首歌（countdown_to_zero_luotianyi，中文）的 10 个行片段**，`.lab` 用项目自带的 `pypinyin` 生成音节序列，跑官方 `onnx_infer.py -l zh`（ONNX CPU，**10 段 2 秒**）：

| 指标（音节级） | HubertFA（歌声训练 + 显式呼吸/静音建模） | Qwen3-ForcedAligner（同片段） |
|---|---|---|
| 单元数 | 81 | 81 |
| 零长度 (<10ms) | **0（0.0%）** | 5（6.2%） |
| 最小非零跨度 | **19.8ms** | 80.0ms |
| 跨度中位数 | 440.8ms | 320.0ms |
| 覆盖率 | **80.0%** | 68.9% |
| 首单元锚定 | 0.0ms | 0.0ms |

音节层之上还有音素层（161 段，中位 146.7ms），并且**显式给出 15 段呼吸音（AP）与 12 段静音（SP）**——对"分离后残留呼吸/齿音"这个实际场景是加分项。举一行实例（`风一下子停住了`）：

```
feng 0.00–0.64 | SP 0.64–0.98 | yi 0.98–1.08 | xia 1.08–1.79 | zi 1.79–2.27
   | AP 2.27–2.60 | ting 2.60–2.90 | zhu 2.90–3.96 | …
```

时长是**有变化、像人唱**的（`zhu` 拖了 1.06 秒），不是"每单位一帧"。

### 9.3 怎么读这两个结果（诚实边界）

- **"是不是为歌声训练"比架构更重要。** CTC 与离散时间槽都能用，但朗读模型在歌声上会退化到不可用；这也解释了为什么现任后端（Qwen3-ForcedAligner 是多语言通用模型）在日文 VOCALOID 上零长度率能到 41.8%。
- **HubertFA 的零长度率不能与 Qwen 直接比**：HubertFA 的输出是**对整段的连续划分**（静音/呼吸也占区间），结构上就不可能产生零长度单元。真正有信息量的是**覆盖率（80% vs 69%）**、**最小非零跨度（19.8ms vs 80ms）**与**跨度中位数**。
- **仍未证明"更准"**：本轮没有人工标注，所以没有 BER/IOU；§6.3 的扰动测试（V4）与人工标注（V5）仍然待做。判定"HubertFA 更准"需要它们。**因此现在不适合直接替换默认后端**——更适合先做成"多后端之一"，等 V4/V5 有结论再决定默认值（见 §9.5）。
- **日文路径已验证**（V2 完成，见 §9.2b）：罗马字预处理可用、词典 0 遗漏；代价是促音被丢（上游 `ja/cl` 的 bug）与汉字读音靠猜（假名歌词可绕开）。
- **工程代价**：模型 415MB（ONNX）、仓库个人项目（Apache-2.0，代码需要自己审）、官方 ONNX 推理依赖 `onnxruntime-gpu==1.19.0`（钉 CUDA 12）；本轮用**CPU 版 onnxruntime** 跑通，10 段 2 秒——按"逐行片段"的调用方式，CPU 很可能就够。

### 9.2b V2：日文（罗马字）路径**也跑通了**

日文要先把行文本转成**罗马字音节 `.lab`**（词典 179 个键：CV/CyV 音节 + `cl` 促音 + `n` 撥音 + 单元音）。转换链：文本 --`pykakasi`--> 平假名读音 --自写规则--> 音节键（拗音合并、`っ→cl`、`ん→n`、长音 `ー` 重复前元音、助词 `は/へ/を → wa/e/o`）。

对**蒲公英的同一批 17 个片段**（与 9.1 完全相同的输入，Qwen 数字已在上表）跑 `-l ja`：

| 指标 | HubertFA ja（モーラ级，10ms 帧） | Qwen3-ForcedAligner（词级，同片段） |
|---|---|---|
| 单元数 | 260 | 141 |
| 零长度 (<10ms) | 5（1.9%） | **59（41.8%）** |
| 最小非零跨度 | **6.8ms** | 80ms |
| 跨度中位数 | 440.1ms | 160ms |
| **覆盖率** | **88.8%** | **46.1%** |

逐行覆盖率对照（HFA vs Qwen）：`91.1% vs 38.8%`、`84.8% vs 64.6%`、`94.1% vs 59.3%`、`85.9% vs 78.2%`、`84.0% vs 14.2%`、`94.0% vs 43.3%`、`88.8% vs 6.7%`、`89.5% vs 93.3%` —— **HFA 稳定在 84–94%，Qwen 则在 6.7%–93.3% 之间剧烈波动**（后者正是零长度词密集出现的行）。

mora 级输出实例（`空の音　風が伝えた`，预期 12 音节）：

```
SP 300-896 | so 896-1320 | ra 1320-1767 | no 1767-1859 | o 1859-3470
   | to 3700-4127 | AP 4127-4547 | ka 4547-4909 | ze 4909-5149 | ga 5149-5920
```

**日文路径的实际代价与坑**：

1. **促音（っ）会被丢掉**：上游 `tools/g2p.py` 的 `BaseG2P.__call__` 把除 `SP` 外的音素一律加 `{language}/` 前缀，而 `cl` 在 `vocab.json` 里属于**无前缀的静音组**（`silent_phonemes` / `merged_phoneme_groups`）→ 会拼出不存在的 `ja/cl` 并 KeyError。**这是上游的 bug，本仓库没有改它**；绕法是送 `.lab` 前去掉促音（代价：促音/长音的边界会糊一点）。若将来正式集成，更干净的做法是在自己的 worker 里修掉前缀逻辑（对 `silent_phonemes` 不加前缀）。
2. **汉字读音靠猜**：`pykakasi` 给的是猜测读音，实测 `放つ` 被读成 `ho u tsu`（正确 `ha na tsu`）——"唱的字"与"送进去的音素"不一致会直接拉低对齐质量。**用假名歌词可以完全绕开这一步**（本库里就有 `samples/tsurupettan_sun3_kana.lrc` 这类）。
3. 词典覆盖：本轮 260 个音节**全部命中词典键**（0 遗漏），无法转成假名的字符只有 `「」`。
4. 老 `librosa`（`<0.10`）需要 `setuptools<81`（`pkg_resources` 在 setuptools 81 之后被移除）。

### 9.2c 模型已迁到 `models/`

`models/aligner/HubertFA/`（`models/` 在 .gitignore 内，不进仓库）：`model.onnx` + `vocab.json` + `config.json` + 三个词典 + 上游代码快照 `upstream/`，共 416MB；来源、版本、下载通道与全部实测结论记录在同目录的 `SOURCE.md`。

### 9.4 复现方式

模型与上游代码现在放在 **`models/aligner/HubertFA/`**（`models/` 在 .gitignore 内；来源与坑见该目录的 `SOURCE.md`），中间产物由脚本重新生成到 `tmp/`。

```powershell
# 通道：HF 直连，GitHub 走 gh-proxy.com
curl.exe -sL -o hfa.zip "https://gh-proxy.com/https://github.com/wolfgitpr/HubertFA/releases/download/v0.0.7/1218_hfa_model_new_dict.zip"

# 准备片段 + .lab（用项目自己的切段逻辑；中文走 pypinyin，日文走 pykakasi）
uv run --with pykakasi python tmp/hfa_prepare_ja.py     # 生成 tmp/hfa/segments_ja/
uv run python tmp/hfa_prepare_zh.py                     # 生成 tmp/hfa/segments/

# 跑 HubertFA（注意 setuptools<81：老 librosa 需要 pkg_resources）
cd models/aligner/HubertFA/upstream
uv run --no-project --with "setuptools<81" --with click --with "librosa<0.10.0" --with textgrid `
  --with pandas --with pyyaml --with tqdm --with onnxruntime --with soundfile --with "numpy<2" `
  python onnx_infer.py -m ../model.onnx -wf <wav 目录> -l ja -d ../japanese_dict_full.txt

# 对比（同一批片段跑 Qwen，并解析 TextGrid）
uv run python tmp/hfa_compare_ja.py
```

对比脚本：`tmp/hfa_prepare_zh.py`、`tmp/hfa_compare.py`、`tmp/hfa_prepare_ja.py`、`tmp/hfa_compare_ja.py`；CTC 那条路的脚本：`tmp/ctc_eval_prepare.py`、`tmp/ctc_eval_qwen.py`、`tmp/ctc_eval_w2v2.py`、`tmp/ctc_eval_transcript.py`、`tmp/ctc_eval_compare.py`；V5 听力材料：`tmp/v5_make_listening.py`、`tmp/v5_verify_material.py`。

### 9.6 V5：人耳裁决（2026-09-13，已出一部分结论）

`tmp/v5_listen/` 是 4 行（line 0 / 4 / 7 / 8，取自 Qwen 覆盖率最低的几行）的**立体声 A/B**——
**左耳 = HubertFA 的边界咔哒，右耳 = Qwen 的边界咔哒**，两声道都有人声；配套 PNG（波形 + 两套边界）
与 `README.md`（逐单元时间表）。咔哒位置已自检：100% 落在预期边界上（±2 ms）。

**line 007 的裁决（人耳确认）**：`何もない世界それは降り立つと`，片段 13.1 秒——

| | 覆盖率 | 判读 |
|---|---|---|
| Qwen | 6.7%（8 个词全挤在前 2 秒） | ❌ **错** |
| HubertFA | 88.8%（16 个モーラ铺满整段） | ✅ **对** |

人耳确认**整段 13 秒都在唱这一行**（该行开头有大幅度混音造成的分离缺陷，但不影响"是否在唱"的判断）。
于是两件事被定下来：

1. **HubertFA 的高覆盖率不是"连续划分"这个输出形态的副产品**，而是真实的对齐能力——它在
   Qwen 把歌词压扁的地方是对的。§9.3 里"覆盖率高也可能是摊派误差"的怀疑，在这个样本上被排除。
2. **Qwen 在这类行上是真失败**（不是"更稀疏但更准"）：8 个词被放进 13 秒演唱的前 2 秒里。这类
   行正是产物里零长度词密集、逐字高亮崩掉的行。

**因此默认后端的建议改成**：日文（以及中文）走 **HubertFA**，Qwen3 保留为可选后端（它在
line 008 这类"没有压扁"的行上与 HFA 一致，作为对照/回退仍有价值）。§9.5 的多后端设计不变，
只是把默认值从此前的"先不换"改为 **hfa**。仍未做完整 V5（人工标一行算 BER/IOU），但对
**默认值这个决策**来说，"两种结构性冲突里 HFA 对、Qwen 错"已经足够；真要出 BER/IOU 数字，
可以在这 4 行的材料上继续做。

### 9.5 接入方式：**多后端，已实现**

项目在分离侧本来就有同构的先例（`--separator-backend {demucs,audio-separator}` +
`AbstractStemSeparator` + 每后端一个 worker），对齐侧照搬了同一套形状：

```
--aligner-backend {hfa,qwen3}     # 默认 hfa
--aligner-url <http://host:port>  # 显式地址优先；缺省按后端取（hfa→8788、qwen3→8787）
```

**契约不变**：请求 = 音频 + 文本 + 语言，响应 = `{"words": [{"text","start_time","end_time"}]}`，
单文件返回对象、多文件返回数组。差异全部关在新的服务端里：

| 关注点 | 谁负责 | 说明 |
|---|---|---|
| 音频切段、偏移、元数据过滤、产物序列化 | **主程序（未改动）** | 现有流水线一行都没改，只多了一个开关 |
| 文本 → `.lab`（G2P） | `scripts/hubertfa_aligner_server.py` | zh 用 `pypinyin` 逐字、ja 用 `pykakasi` + 音节规则、en 按词 |
| 音素/モーラ → **行文本的单元** | 同上 | 单元的 `text` 必须是原行文本的**子串**（主程序靠 `text.find` 定位）：按 G2P 片段聚合，`空` = `so`+`ra` → 一个片段 |
| 后端特有的坑 | 同上 | 促音 `ja/cl` 的前缀 bug（丢弃促音）、词典过滤、上游 `dataset` 累加需清空、`setuptools<81` |

实现里几个**必须知道**的点（都是被测试/实测逼出来的）：

1. **上游 G2P 会静默丢词**（不在词典里只打 warning 然后丢弃）。不过滤的话，送模型的音节数
   与我这里的片段数就对不上，后面「按顺序把音节归回片段」会**整体错位**——所以送模型前先按
   词典过滤，并记日志。
2. **上游 `InferenceBase.dataset` 是累加的**（`__init__` 初始化一次，`get_dataset` 只 append）。
   常驻服务里必须每次 `clear()`，否则会重复处理上一批请求的（已删除的）临时路径。
3. **kana → 音节键的规则踩了两个坑**，都由单元测试抓住：拗音不能直接拼 `y`（`ち+ゃ` 会得到
   错误的 `chya`，正确是 `cha`；`sh/ch/j` 不写 `y`），以及片假名要先归一化成平假名；
   另外外来语的小写元音要与前一音节合成**一个**モーラ（`ヴァ`→`va`、`ティ`→`ti`、`ウィ`→`wi`）。
4. **一道机械防漂移测试**（`tests/test_aligner_backends.py`）：登记表里的脚本要存在、
   脚本里的 `DEFAULT_PORT` 要与登记表一致、两个后端不能撞端口。这个项目踩过
   「客户端默认 8787 / 服务端默认 8000」的坑，多后端之后更该有这道检查。

**实机结果**（`Cryu - 蒲公英`，同一首歌、同一偏移、两个后端各跑一遍完整流水线）：

| | 词级 token | 每行中位 | 行内重复标签 | 行区间覆盖率 | 对齐器报告的零长度单元 |
|---|---|---|---|---|---|
| **hfa** | 167 | 10 | 0 | **100.0%** | **0 / 135（0.0%）** |
| qwen3 | 109 | 6 | 0 | 77.3% | 59 / 141（41.8%） |

（两者产物都是 0 重复标签、0 零时长 token——那是 `_merge_zero_length_tokens` 的效果；
上表最后一列是**对齐器原始输出**里的零长度比例。日文路径**没有出现任何词典未覆盖的片段**。）

**默认后端 = hfa**（理由见 §9.6 的人耳裁决）。Qwen3 保留为可选后端：它在"没有压扁歌词"的行上
与 HFA 一致，作为对照与回退仍有价值，而且它不依赖 `models/aligner/HubertFA/` 那 416MB。

