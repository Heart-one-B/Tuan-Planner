# harness/agent/permission.py
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from harness.agent.run_context import RunContext


@dataclass
class Allow:
    """放行,工具照常执行。"""
    reason: str | None = None


@dataclass
class Deny:
    """拒绝执行。工具不会被调用,harness 合成一条工具结果反馈给模型
    (格式与 TerminationPolicy.Reject 一致——委婉否决,给模型一个可读
    的理由,让它据此调整后续行为,而不是抛异常打断整个 run)。"""
    reason: str


@dataclass
class Defer:
    """挂起,等待外部(通常是人工)审批。

    这次 run 到此为止:harness 会把"哪个工具调用在等审批"这件事存进
    快照、产出 status="awaiting_approval" 的 LoopOutcome,协程随之
    结束、进程可以退出——不会为了等一个可能长达几小时的人工审批,
    占着一个活的 await 空转。这是形态二(慢审批)专用的决策;如果
    宿主的审批总能很快拿到结果,直接在 check() 内部同步等、返回
    Allow/Deny 即可,不需要用到 Defer。

    approval_id 是宿主自己生成、自己管理含义的审批工单标识——
    harness 不解释它,只负责原样存进快照、resume 时原样带回来给
    宿主核对"你要恢复的是不是你认为正在处理的那个审批"。
    """
    approval_id: str
    reason: str | None = None


PermissionDecision = Allow | Deny | Defer


class PermissionPolicy(Protocol):
    """宿主实现的权限判断接口。harness 只定义"执行前问一次、三种
    结果分别怎么处理"这个机制,从不判断"这个工具危不危险"——那是
    纯粹的业务事实,只有宿主知道,harness 猜不出来也不该猜。

    check() 可以是任何实现:纯规则匹配、查一张风险等级表、调一个
    独立的风控模型、查询一个审批系统当前的状态——harness 一视同仁,
    只关心返回的是 Allow / Deny / Defer 里的哪一个。
    """
    async def check(
        self, tool_name: str, args: dict, run_ctx: RunContext,
    ) -> PermissionDecision:
        ...


class AllowAllPolicy:
    """默认策略:全部放行。"这项能力不配置就不该有感"的具体落实——
    不传 permission_policy 的 Agent,行为与没有这套机制之前完全一致。
    """
    async def check(self, tool_name: str, args: dict, run_ctx) -> PermissionDecision:
        return Allow()