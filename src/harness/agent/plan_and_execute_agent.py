# harness/agent/plan_execute_agent.py
import json
import logging
import uuid
from typing import AsyncGenerator

from harness.agent.dag_scheduler import DagScheduler, StepExecutor, default_gate
from harness.agent.result import AgentResult
from harness.llm.base import LLMClientBase
from harness.tracing import tracer

logger = logging.getLogger(__name__)


PLANNER_SYSTEM_PROMPT = """你是一个任务规划器。给定用户目标,把它拆解成一系列可执行步骤,并标注依赖关系。

输出要求:
1. 只输出一个 JSON 对象,不要任何额外文字、不要用 markdown 代码块包裹。
2. 格式:{"steps": [{"id": 整数, "task": "这一步做什么(自然语言)", "depends_on": [依赖的步骤id]}]}
3. id 从 1 开始递增。
4. depends_on 表示本步必须等哪些步骤完成才能开始:
   - 需要用到另一步的结果,就把那一步 id 放进 depends_on。
   - 不依赖任何步骤(可立即执行)则留空数组 []。
5. 重要:不要制造不必要的依赖。两步只有在"后者真需要前者结果"时才标依赖,
   否则都留空,以便并行执行。
"""


class PlanExecuteAgent:
    """Plan-and-Execute 编排器。

    五步:Planner 画 DAG → Scheduler 调度执行 → Synthesizer 汇总。
    依赖全部注入,无全局单例;产出带类型的事件流,与 ReactAgent 同构。

    事件:
        {"type": "plan_ready", "steps": [...]}
        {"type": "step_done",  "id": int, "status": str, "summary": str}
        {"type": "token",      "content": str}
        {"type": "done"}
    """

    def __init__(
        self,
        llm_client: LLMClientBase,
        executor: StepExecutor,          # 节点执行器(如 StructuredAgentExecutor)
        gate=default_gate,
        name: str = "plan_execute",
    ):
        self.llm_client = llm_client
        self.executor = executor
        self.gate = gate
        self.name = name

    # ── ① Planner:目标 → 带依赖的 DAG ──
    async def _plan(self, goal: str, trace_id: str) -> list[dict]:
        resp = await self.llm_client.call(
            trace_id=trace_id,
            messages=[
                {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
                {"role": "user", "content": goal},
            ],
        )
        raw = resp.choices[0].message.content
        try:
            return json.loads(raw)["steps"]
        except (json.JSONDecodeError, KeyError) as e:
            # 规划失败是致命的(没有图就没法执行),如实抛出由顶层处理
            raise ValueError(f"Planner 输出无法解析为合法计划: {e}\n原始输出: {raw}")

    # ── ⑤ Synthesizer:所有步骤结果 → 最终答案 ──
    async def _synthesize(self, goal: str, results: dict[int, AgentResult], trace_id: str):
        # 把每步的结论喂给模型(只喂 summary/data,不喂内部细节)
        step_lines = [
            f"步骤{nid}({res.status}): {res.summary};数据: {res.data}"
            for nid, res in sorted(results.items())
        ]
        synth_prompt = (
            f"用户最初的目标是:{goal}\n\n"
            f"以下是各步骤的执行结果:\n" + "\n".join(step_lines) + "\n\n"
            f"请基于这些结果,给出面向用户的最终回答。"
            f"若某些步骤失败或被跳过,请如实说明缺了什么,不要编造。"
        )
        return await self.llm_client.call(
            trace_id=trace_id,
            messages=[{"role": "user", "content": synth_prompt}],
            stream=True,
        )

    # ── 顶层编排:串起五步 ──
    async def run(self, goal: str, session_id: str = None) -> AsyncGenerator[dict, None]:
        trace_id = str(uuid.uuid4())
        tracer.start_trace(trace_id, session_id or self.name, goal)

        try:
            # ①② 规划 → 拿到 DAG
            steps = await self._plan(goal, trace_id)
            yield {"type": "plan_ready", "steps": steps}

            # ③④ 调度执行(波次并行/串行 + 跳过策略 全在调度器里)
            scheduler = DagScheduler(executor=self.executor, gate=self.gate, tracer=tracer)
            results = await scheduler.run(steps, trace_id=trace_id)

            for nid, res in sorted(results.items()):
                yield {"type": "step_done", "id": nid, "status": res.status, "summary": res.summary}

            # ⑤ 汇总(流式输出最终答案)
            final_reply = ""
            stream = await self._synthesize(goal, results, trace_id)
            async for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    final_reply += content
                    yield {"type": "token", "content": content}

            yield {"type": "done"}
            tracer.end_trace(trace_id, final_reply, status="success")

        except Exception as e:
            logger.error(f"[PlanExecuteAgent] error: {e}", exc_info=True)
            tracer.end_trace(trace_id, "", status="error")
            raise