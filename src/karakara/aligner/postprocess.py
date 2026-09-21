"""对齐器输出的后处理：零长度词的统计、细分与合并依据。

**零长度词从哪来**（下面这一整段讲的是可选的 ``qwen3`` 后端，默认后端不这样）：

* 模型 config 里 ``timestamp_segment_time = 80``，时间戳来自**离散 token**
  （``timestamp_ms = token_id * 80``），所以每个单元边界都被量化到 80ms 格子；
* ``parse_timestamp`` 把相邻的两个边界 token 配成一个单元的 ``(start, end)``；
* 库自带的 ``fix_timestamp`` 只用最长递增子序列修**递减**的异常，判据是
  ``data[j] <= data[i]``：**允许相等**，于是 ``start == end``（语义上就是
  "这个单元的真实时长不足一帧"）会被原样保留。

qwen3 后端的实测规模（6 首歌、1497 个单元的真实运行日志）：337 个零长度，占
**22.5%**，逐首 10.7% / 15.3% / **41.8%** / 24.6%；所有单元跨度都是 80ms 的整数倍；
70% 落在行中，80% 紧跟前一个单元的结束时刻。

默认后端 ``hfa``（HubertFA）帧移 **10ms**，而且它的输出是对整段的**连续划分**
（静音/呼吸也占区间），结构上很难产生零长度单元，实测日文只有 **1.9%**。所以下面
这套「统计 + 可选细分」在默认路径上基本不会被触发；但产物规范（不出现重复时间标签）
与逐行降级路径是两个后端共用的。完整对照见
``docs/records/2026-09-18-aligner-choice-and-stability.md``。

零长度单元在产物里的表现是「与前一个标签完全重复的标签」，播放器无法单独高亮它。
本模块负责两件事：

* :func:`count_zero_length` / :class:`ZeroLengthStats`（含 ``ratio``）—— 把它作为
  **质量信号**报出来；
* :func:`refine_collapsed_words` —— **可选**的细分：把折叠段摊进它**后面**的空隙，
  且**绝不改动任何被模型报告过的边界**（对应 ``--refine-collapsed-words``）。

至于"更细的对齐粒度"那条路：已经试过，走不通——``language`` 只用于选择分词器
（日文走 nagisa 形态素词），插空格会被 nagisa 重新切回去（实测 ``"君が好きだから"``
与其逐字插空格版本返回完全相同的 5 个单元与时间戳），而谎称中文会把假名合成一个
token（假名不算 ``is_cjk_char``），比原来还粗。详见 ``docs/current/aligner.md``。
"""

from __future__ import annotations

from dataclasses import dataclass

from karakara.aligner.abc import AlignedWord

#: 零长度占比超过这个值就告警。下面这组刻度来自 **qwen3** 后端：Saya 10.7% /
#: SACRA 15.3% / samples 24.6% 属"正常"范围，蒲公英 41.8% 则明显是模型在这首歌上
#: 吃力。默认的 ``hfa`` 只有 1.9%，正常情况下永远够不到这条线——它更像是"某个后端
#: 在这首歌上塌了"的兜底信号，而不是质量门槛。
DEGENERATE_RATIO_WARN = 0.30

#: 细分时给每个折叠单元的最小长度（ms）。取 1 是因为这是**推断值**，
#: 再往下取整就变成 0 了，没有意义。
MIN_REFINED_MS = 1


@dataclass(frozen=True)
class ZeroLengthStats:
    """一次对齐（通常是一整首歌）的零长度统计。"""

    units: int = 0
    zero_length: int = 0
    refined: int = 0

    @property
    def ratio(self) -> float:
        """零长度单元占全部单元的比例。"""
        return self.zero_length / self.units if self.units else 0.0

    def merged_with(self, other: ZeroLengthStats) -> ZeroLengthStats:
        return ZeroLengthStats(
            units=self.units + other.units,
            zero_length=self.zero_length + other.zero_length,
            refined=self.refined + other.refined,
        )


def count_zero_length(words: list[AlignedWord]) -> ZeroLengthStats:
    """统计一批对齐单元里的零长度数量。"""
    units = sum(1 for word in words if word.position is not None)
    zero = sum(
        1
        for word in words
        if word.position is not None and word.position[0] == word.position[1]
    )
    return ZeroLengthStats(units=units, zero_length=zero)


def _median_positive_duration(words: list[AlignedWord]) -> int | None:
    """这一批单元里"正常"跨度的中位数，用作细分时的时长上限。"""
    durations = sorted(
        end - start
        for word in words
        if word.position is not None and word.position[1] > word.position[0]
        for start, end in (word.position,)
    )
    if not durations:
        return None
    return durations[len(durations) // 2]


def refine_collapsed_words(
    words: list[AlignedWord],
    *,
    total_ms: float,
    max_duration_ms: int | None = None,
) -> tuple[list[AlignedWord], int]:
    """把零长度（折叠）单元摊进它**后面**的空隙里，返回 ``(新列表, 被细分的数量)``。

    三条不变量：

    1. **不改动任何被模型报告过的边界。** 只给折叠单元分配它们原本拿不到的时长，
       借的是"折叠段之后到下一个正常单元起点"之间的空闲区间（行末折叠段则借到整段末尾）。
    2. **不越过下一个正常单元的起点。** 借不到（空隙为 0）就原样返回，交给上游合并。
    3. **不摊出比这一行典型单元还长的时长**：每个折叠单元的上限取本行正常跨度的
      中位数（可用 ``max_duration_ms`` 覆盖），总预算按单元文本长度加权分配。

    Args:
        words: 对齐器返回的单元列表（时间相对于本段起点，单位 ms）。
        total_ms: 本段的长度，用于给行末的折叠段划定可用区间。
        max_duration_ms: 单个折叠单元的时长上限；``None`` 时取本行正常跨度的中位数。

    Returns:
        新的单元列表与其中被细分的数量。**没有可用空隙的折叠段原样保留**。
    """
    out: list[AlignedWord] = []
    refined = 0
    cap = (
        max_duration_ms
        if max_duration_ms is not None
        else _median_positive_duration(words)
    )
    index = 0
    total = len(words)

    while index < total:
        word = words[index]
        position = word.position
        if position is None or position[0] != position[1]:
            out.append(word)
            index += 1
            continue

        # 收集连续的折叠段 [index, end)
        end = index
        while end < total:
            candidate = words[end].position
            if candidate is None or candidate[0] != candidate[1]:
                break
            end += 1
        run = end - index
        start_ms = position[0]

        # 可用空隙：段后第一个"正常单元"的起点，没有就用本段末尾
        limit_ms = float(total_ms)
        for later in range(end, total):
            later_position = words[later].position
            if later_position is not None and later_position[1] > later_position[0]:
                limit_ms = float(later_position[0])
                break

        available = limit_ms - start_ms
        if available < run * MIN_REFINED_MS:
            # 没有空隙可分——保持折叠，让上层把它并进相邻单元
            out.extend(words[index:end])
            index = end
            continue

        budget = available
        if cap is not None:
            budget = min(budget, float(cap) * run)

        weights = [max(1, len(item.word)) for item in words[index:end]]
        total_weight = sum(weights)
        cursor = float(start_ms)
        for offset, item in enumerate(words[index:end]):
            share = budget * weights[offset] / total_weight
            segment_start = int(cursor)
            cursor = min(limit_ms, cursor + max(MIN_REFINED_MS, share))
            segment_end = int(cursor)
            if offset == len(weights) - 1:
                segment_end = max(segment_end, segment_start + MIN_REFINED_MS)
            out.append(
                AlignedWord(word=item.word, position=(segment_start, segment_end))
            )
            refined += 1
        index = end

    return out, refined
