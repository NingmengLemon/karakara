# 环境：PEP 723、CUDA 索引，以及 uv 缓存那个坑

三个跑在不同环境里的东西：

| 谁 | 环境从哪来 | 需要 torch 吗 |
|---|---|---|
| 主程序 | 项目 `.venv`（`uv sync`） | **不需要** |
| 分离 worker | 脚本头部的 PEP 723 内联依赖 | 需要（CUDA 版更好） |
| 对齐服务 | 同上 | 需要（CUDA 版更好） |

## 两个脚本都已经配好 CUDA 索引

`scripts/separator_worker.py` 与 `scripts/qwen3aligner_server.py` 的头部都有：

```toml
# [[tool.uv.index]]
# name = "pytorch_cu130"
# url = "https://download.pytorch.org/whl/cu130"
# explicit = true
#
# [tool.uv.sources]
# torch = { index = "pytorch_cu130" }
```

要点与坑：

- **索引是 `explicit` 的**，必须由 `[tool.uv.sources]` 显式引用才生效；不加这一段，torch 会从 PyPI 解析到 CPU 版，于是**在 GPU 机器上也跑在 CPU 上**，慢一个数量级，而且不看日志根本发现不了（实测同一首歌：GPU 4.2s / CPU 22.3s）。
- **`[tool.uv.sources]` 只对直接依赖生效。** 对齐服务里 `torch` 只是 `qwen-asr` 的间接依赖，所以它必须被**显式列进 `dependencies`**，否则索引指向不到它（分离 worker 本来就显式列了 torch，所以那边没这个问题）。
- 两个脚本都会在 CUDA 不可用时打印**显式告警**，不会静默降级。
- 自检：

  ```bash
  uv run --script scripts/separator_worker.py --info
  ```

  看 `device` 与 `torch` 字段。对齐服务启动时也会打印它实际用的设备。
- win32 实测可用（GPU 机器）：

  ```
  torch 2.14.0+cu130 / cuda_available=True / RTX 4060 Laptop (cc 8.9) / bf16 ok
  model loaded successfully on cuda:0
  ```

## ⚠️ 改了索引之后必须**删掉该脚本的缓存环境**

这条比"缓存键"的说法更细一层，是实测出来的：

uv 给 PEP 723 脚本准备的环境目录名形如 `environments-v2/<脚本名>-<哈希>`，而

- 那个哈希**不包含** `[tool.uv]` 里的索引配置；
- 它**也不随依赖清单变化**：往清单里加一个包，uv 只是把那个包装进**同一个**环境里（实测环境目录名不变）。

于是会出现两种"改了没用"：

1. **只改索引** → uv 认为什么都没变，照旧复用那个装着 CPU 版 torch 的环境；
2. **把 `torch` 加进依赖清单也没用** —— CPU 版 torch 已经满足 `torch>=2.9.0`，uv 判定无需改动，索引依旧不会被查询。（这个坑真踩过：改完头部后服务仍打印 `torch=2.14.0+cpu`。）

**结论：必须删掉该脚本对应的环境目录**，再跑一次让它重建：

```powershell
# 用 uv cache dir 找到缓存根目录，然后删掉 environments-v2/<脚本名>-<哈希>
uv cache dir
Remove-Item "<cache>\environments-v2\qwen3aligner-server-*" -Recurse -Force
# 或者粗暴一点
uv cache clean
```

实测删掉后重建约 10 秒（wheel 已在缓存里，直接硬链，不需要重新下载）。

## 不想动仓库？用 `--separator-cmd`

指向任何已经装好依赖（含 CUDA 版 torch）的解释器即可：

```bash
uv run main.py --batch-dir ... \
  --separator-cmd /path/to/cuda-env/python.exe scripts/separator_worker.py
# 或环境变量
KARAKARA_SEPARATOR_CMD='"C:\sep-env\Scripts\python.exe" "scripts/separator_worker.py"'
```

对齐服务没有等价的开关，但可以自己起在任何地址上，再用 `--aligner-url` 指过去。
