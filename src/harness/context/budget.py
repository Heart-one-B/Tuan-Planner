# harness/context/budget.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ContextBudget:
    """上下文预算配置,纯数据不含逻辑(逻辑在 should_compact 这类只读推导里)。

    两个预留概念(而不是原设计的单一 reserve_tokens):
      output_reserve      给模型本轮回复留的空间——没有它,窗口填满后
                          模型一个字都吐不出来
      compaction_reserve  给压缩摘要的生成留的空间——压缩本身是一次
                          LLM 调用,等真撑满了再压,连摘要都生成不了。
                          参考 Claude Code 的量级(其保留约 2 万 token)。

    min_compact_tokens:最小压缩门槛。太小的会话不值得压——摘要有固定
    开销,还必然丢信息。参考 Anthropic 原生压缩 API 的下限(50K)。

    卸载相关:
      default_tool_result_max_chars  单条工具结果超过此字符数即卸载
                                     (单个工具可用 ToolDefinition.max_result_chars 覆盖)
      offload_preview_chars          卸载后留在上下文里的预览长度(头尾各一半)
      offload_dir                    卸载文件根目录(用户可配)。
                                     刻意没有 ttl/auto_cleanup 参数——清理策略
                                     归 Phase 3 会话生命周期管理,本层只写不删。
      tool_result_clear_after_rounds 工具结果"沉底"超过 N 个工具调用轮次后
                                     清空为占位符(管"太旧",与卸载管"太大"互补)
    """

    max_tokens: int = 128_000
    compaction_trigger_margin: int = 13_000   # 触发线=effective_window减此绝对缓冲。
                                              # 弃用比例阈值(0.85那版)的原因:窗口越大,
                                              # 比例留出的空窗越大,但摘要任务实际需要的
                                              # 预算并不随窗口膨胀——绝对缓冲不随窗口
                                              # 大小漂移,1M窗口下不会浪费十几万token。
    output_reserve: int = 4_096
    compaction_reserve: int = 16_000          # 给压缩摘要输出的预留。如实说明:此数为
                                              # 估计值,未经我们自己的输出分布校准
                                              # (Claude Code的20k是p99.99=17,387实测
                                              # 推出的);校准任务记给Phase 5 evals。
    min_compact_tokens: int = 50_000

    default_tool_result_max_chars: int = 50_000
    offload_preview_chars: int = 2_000
    offload_dir: Path = Path("data/offload")
    tool_result_clear_after_rounds: int = 3
    keep_recent_rounds: int = 2    # 压缩时尾部保留的原文轮次数,收口进
                                   # budget 统一配置(原先是 Compactor.compact
                                   # 的独立默认参数,散落两处容易配置漂移)

    @property
    def effective_window(self) -> int:
        """真正可用于对话内容的窗口:标称窗口扣掉两个预留。"""
        return self.max_tokens - self.output_reserve - self.compaction_reserve

    @property
    def trigger_line(self) -> int:
        """压缩触发线:有效窗口再减一道绝对安全缓冲。"""
        return self.effective_window - self.compaction_trigger_margin

    def should_compact(self, current_tokens: int) -> bool:
        """两个条件都满足才压缩:
        ① 会话够大(超过最小门槛,压缩才划算)
        ② 占用越过触发线(距上限只剩固定缓冲了)
        """
        if current_tokens < self.min_compact_tokens:
            return False
        return current_tokens >= self.trigger_line