# 环境

这个项目同时跑在**三套环境**里。这一页讲它们各自怎么来的、CUDA 索引怎么配，以及一个会让你
「改了配置却没生效」的 uv 缓存坑。

| 谁 | 环境从哪来 | 需要 torch 吗 |
|---|---|---|
| 主程序 | 项目 `.venv`（`uv sync`） | **不需要** |
| 分离 worker | 脚本头部的 PEP 723 内联依赖 | 需要（CUDA 版更好） |
| 对齐服务 | 同上 | 需要（CUDA 版更好） |

## CUDA 索引

`scripts/separator_worker.py`、`scripts/separator_worker_audio_separator.py` 与
`scripts/qwen3aligner_server.py` 的头部都带了 PyTorch 的 CUDA 索引：

```toml
# [[tool.uv.index]]
# name = "pytorch_cu130"
# url = "https://download.pytorch.org/whl/cu130"
# explicit = true
#
# [tool.uv.sources]
# torch = { index = "pytorch_cu130" }
```

两个要点：

- **索引是 `explicit` 的**，必须由 `[tool.uv.sources]` 显式引用才生效。不加这一段，torch 会
  从 PyPI 解析到 CPU 版，于是在 GPU 机器上也跑在 CPU 上，慢一个数量级，而且**不看日志根本
  发现不了**（实测同一首歌：GPU 4.2s / CPU 22.3s）。
- **`[tool.uv.sources]` 只对直接依赖生效**，所以 `torch` 必须被显式列进 `dependencies`。
  qwen3 服务里 torch 只是 `qwen-asr` 的间接依赖，漏了它就索引不到。
- 两个 worker 在 CUDA 不可用时都会**打印显式告警**，不会静默降级。

自检：

```bash
uv run --script scripts/separator_worker.py --info    # 看 device 与 torch 字段
```

## 改了索引之后必须删掉该脚本的缓存环境

这是实测出来的，比「缓存键」的一般说法更细一层。uv 给 PEP 723 脚本准备的环境目录名形如
`environments-v2/<脚本名>-<哈希>`，而那个哈希：

- **不包含** `[tool.uv]` 里的索引配置；
- **也不随依赖清单变化**：往清单里加一个包，uv 只是把那个包装进**同一个**环境里。

于是有两种「改了没用」：只改索引时 uv 认为什么都没变；把 `torch` 加进依赖清单也没用，
因为 CPU 版 torch 已经满足 `torch>=2.9.0`，uv 判定无需改动，索引根本不会被查询。

```powershell
uv cache dir
Remove-Item "<cache>\environments-v2\qwen3aligner-server-*" -Recurse -Force
# 或者粗暴一点：uv cache clean
```

删掉后重建约 10 秒（wheel 还在缓存里，直接硬链，不需要重新下载）。

## 不想动仓库？用 `--separator-cmd`

指向任何已经装好依赖（含 CUDA 版 torch）的解释器即可：

```bash
uv run main.py --batch-dir ... \
  --separator-cmd /path/to/cuda-env/python.exe scripts/separator_worker.py
# 或环境变量
KARAKARA_SEPARATOR_CMD='"C:\sep-env\Scripts\python.exe" "scripts\separator_worker.py"'
```

对齐服务没有等价的开关，但它可以起在任何地址上，用 `--aligner-url` 指过去。

## 主环境里有什么

运行时依赖闭包 11 个包，不含 torch：

```
av / eval-type-backport / lemony-lrc-parser / numpy / pydantic /
pypinyin / regex / requests / soundfile / typing-extensions / zhconv
```

开发工具（`ruff` / `ty` / `pytest`）在 dev 依赖组里；`uv sync` 会一并装上。
