# harness/agent/state.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from harness.agent.permission import Allow, Deny, PermissionPolicy
from harness.llm.base import LLMClientBase
from harness.tools.tool_executor import ToolExecutor

if TYPE_CHECKING:
    # 只在静态类型检查时导入——loop.py(第四层)要从这里导入
    # LoopConfig/LoopState,这里的字段标注又要引用 loop.py 定义的
    # Budget/MessageStore,互相导入会在运行时成环。
    # `from __future__ import annotations` 让下面的类型标注在运行时
    # 保持字符串形态,不会真的触发这次导入,只有静态类型检查器才会
    # 沿着 TYPE_CHECKING 分支解析它们。这是 memory/extraction.py 打破
    # Agent<->提取 双向依赖同一个手法,这里是第二次应用。
    from harness.agent.loop import Budget, MessageStore


@dataclass(frozen=True)
class LoopConfig:
    """一次 run 期间不变的配置。frozen 是刻意的——循环不该改写它,
    这是"显式 State"原则里"配置和现场分离"的那一半(另一半是 LoopState)。

    tools 在构造时算好、之后每轮复用,不在循环内部重新拼装
    (tool_executor.schemas 不便宜,循环每轮都调用它是浪费)。
    """
    llm: LLMClientBase
    tool_executor: ToolExecutor
    budget: "Budget"
    permission_policy: PermissionPolicy
    tools: list
    require_terminal_tool: str | None = None
    max_terminal_nudges: int = 2


@dataclass
class ResumePoint:
    """从权限挂起点继续所需的全部信息。state.resume_point 非 None 时,
    run_query_loop 的入口会先补完"挂起前那一轮的后半段",再落入正常
    的 while 循环——一份循环代码,服务两种入口(全新 run / 恢复 run),
    不再需要 AgentLoop.resume() 那样整段重复的第二份实现。
    """
    tool_call_id: str
    decision: Allow | Deny


@dataclass
class LoopState:
    """跨轮传递的全部可变现场,集中管理(决策 E)。每轮开头读、结尾
    写回。第四层函数 run_query_loop(cfg, state, run_ctx) 的签名里
    只有这两个数据参数——"这一轮能碰什么"结构上一目了然,不靠自觉。

    store 放进 State 而不是单独传参:CC 的 State.messages 就在
    State 里,这是同一个设计的直接落地;放进去之后循环函数不再需要
    第三个可变数据参数。

    abort 曾经是这里的占位槽,第四刀实现真正的检查点时挪到了
    RunContext(见 run_context.py 的说明)——工具执行和权限检查都
    能摸到 RunContext,摸不到 LoopState,"中断信号"这类跨层都要
    感知的信息应该放在够得着的地方,不该锁死在只有 loop.py 内部
    流转的这个对象里。

    【第三刀改动】压缩防死循环状态从"整个 run 一个 bool"拆成两个字段
    (修 bug#1):
      compacted_this_round  每轮开头重置——防同一轮内反复压缩(CC 的
                            hasAttemptedReactiveCompact 语义)
      overflow_recovery_count  整个 run 累计,不清零——防长期烧压缩
                               预算,上限由 Budget.max_overflow_recoveries
                               控制。这个字段进快照,跨 resume 边界
                               存活,这正是 bug#1 真正被修复的地方
                               (第二刀只是搬了个家,行为没变;第三刀
                               这里才是语义真正被修正的地方)。
    round2 的 overflow_recovered(bool)字段被这两个字段取代,不再存在。

    output_truncation_count/output_upgraded 是任务 3.6/3.7/3.8 的落点:
    前者是"催续写"已经用了几次(上限见 Budget.max_output_truncation_
    recoveries);后者是"是否已经用过一次性的静默升档"(Budget.
    max_output_tokens_upgraded 配置时才有意义)——一次性且不会自动
    复位,升档之后这个 run 剩余部分都用升档后的预算,不会在某一轮
    "回退"到小预算(见 loop.py _call_model 的实现说明)。
    """
    store: "MessageStore"

    rounds: int = 0
    tool_calls_used: int = 0

    compacted_this_round: bool = False
    overflow_recovery_count: int = 0
    output_truncation_count: int = 0
    output_upgraded: bool = False
    terminal_nudge_count: int = 0

    resume_point: ResumePoint | None = None

    def to_dict(self) -> dict:
        """只导出计数器/标志位。store 不进(它的内容由
        RunSnapshot.messages 承载,不需要第二本账);resume_point 也
        不进(它是"这一次 resume 调用要做什么"的指令,不是"当前现场
        的事实");compacted_this_round 也不进——它是"这一轮内"的
        临时标志,每轮开头就会被重置,resume 恢复的永远是"上一轮已经
        结束"之后的现场,不存在"恢复到某一轮中途、这个标志还有意义"
        的情况(见 loop.py 对 Defer 只会发生在工具批处理阶段的说明)。
        """
        return {
            "rounds": self.rounds,
            "tool_calls_used": self.tool_calls_used,
            "overflow_recovery_count": self.overflow_recovery_count,
            "output_truncation_count": self.output_truncation_count,
            "output_upgraded": self.output_upgraded,
            "terminal_nudge_count": self.terminal_nudge_count,
        }

    @classmethod
    def from_dict(cls, data: dict, store: "MessageStore") -> "LoopState":
        """store 必须由调用方现场构造并传入——道理同 to_dict() 的
        注释,它不在被序列化的字段里。"""
        return cls(
            store=store,
            rounds=data.get("rounds", 0),
            tool_calls_used=data.get("tool_calls_used", 0),
            overflow_recovery_count=data.get("overflow_recovery_count", 0),
            output_truncation_count=data.get("output_truncation_count", 0),
            output_upgraded=data.get("output_upgraded", False),
            terminal_nudge_count=data.get("terminal_nudge_count", 0),
        )