# harness/agent/query.py
from __future__ import annotations

import logging
from typing import AsyncGenerator

from harness.agent.loop import LoopOutcome, run_query_loop
from harness.agent.run_context import RunContext
from harness.agent.state import LoopConfig, LoopState

logger = logging.getLogger(__name__)


async def query(
    cfg: LoopConfig, state: LoopState, run_ctx: RunContext,
) -> AsyncGenerator[dict, None]:
    """第三层:把 run_query_loop(第四层)包一层,负责"不管怎么退出都
    要做的收尾"。这一层存在的唯一理由:让第四层只管转圈(设计方案
    §4.1/任务 2.5)。

    和第一刀的差别:第一刀里 AgentLoop.run()/resume() 各自有一份
    try/except+span.end,第二刀把它收口成这一处唯一的实现——
    run_query_loop 内部不再捕获任何异常,全部冒泡到这里处理。

    与最初设计草稿的一个偏离:草稿里设想第三层要分 query()/
    resume_query() 两个函数,对应第四层的两种入口。实现时发现不需要——
    "是不是在恢复"这件事已经完全由 state.resume_point 是否为 None
    决定,对第三层和第四层都是透明的;只有第二层(Agent)才有真正不同
    的准备工作(从 task/history 组装消息,还是从快照读 messages),
    所以"两个入口"只需要在第二层体现(events()/resume_events()),
    第三层一份 query() 服务两种情况就够了——更简单,而不是更复杂。
    """
    saw_outcome = False
    sealed = False
    try:
        async for ev in run_query_loop(cfg, state, run_ctx):
            if ev["type"] == "outcome":
                saw_outcome = True
            yield ev
    except Exception as e:
        logger.error(f"[query] error: {e}", exc_info=True)
        from harness.agent.repair import repair_orphan_tool_calls
        repair_orphan_tool_calls(state.store.messages, reason="执行异常")
        run_ctx.span.end("", status="error")
        sealed = True
        raise
    finally:
        if not saw_outcome and not sealed:
            # 调用方(比如宿主的 UI)提前 break 了这个生成器,或者
            # run_query_loop 在没有异常、也没有产出 outcome 的情况下
            # 结束(理论上不应发生,但"没看到 outcome"本身就是一种
            # 需要如实记录的状态,不能默默吞掉)。这次 run 没有终态,
            # 不该被当成"完成"或"异常"——诚实语义,不是缺陷。
            # 孤儿 tool_use 修复留给第四刀,本刀只负责如实标注 span 状态。
            run_ctx.span.end("", status="abandoned")


async def run_to_outcome(
    cfg: LoopConfig, state: LoopState, run_ctx: RunContext,
) -> LoopOutcome:
    """把 query() 的事件流收敛成一个 LoopOutcome——给不需要流式事件的
    调用方(测试、或者宿主只想要最终结果时)用。

    第一刀有 run_to_outcome()/resume_to_outcome() 两个版本,分别对应
    AgentLoop.run()/resume()。第二刀只需要一个:resume 场景只是
    "state.resume_point 非 None 的 run",对这个函数完全透明——
    调用方在构造 state 时决定是不是 resume,这个函数不需要关心。
    """
    outcome: LoopOutcome | None = None
    async for ev in query(cfg, state, run_ctx):
        if ev["type"] == "outcome":
            outcome = ev["outcome"]
    assert outcome is not None, "query 未产出 outcome(不应发生)"
    return outcome