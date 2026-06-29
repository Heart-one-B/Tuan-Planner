from __future__ import annotations

import json
import uuid

from agents.orchestrator.schema import OrchestratorData
from agents.orchestrator.prompt import ORCHESTRATOR_SYSTEM_PROMPT
from agents.orchestrator.tools import build_orchestrator_context
from agents.fact.agent import FactAgent
from agents.planning.agent import PlanningAgent
from agents.evaluation.agent import EvaluationAgent
from harness.agent.result import AgentResult
from harness.llm.base import LLMClientBase
from harness.tools.tool_definition import ToolDefinition
from harness.tools.tool_executor import ToolExecutor
from harness.tracing import tracer


class OrchestratorAgent:
    """用户反馈调度 Agent。

    用 ToolExecutor + ToolDefinition 管理子 Agent 工具调用。
    ToolExecutor 已支持异步 func，子 Agent 的 async run() 可以直接注册。
    不用 StructuredAgent——Orchestrator 需要自己控制 ReAct loop，
    调用顺序和次数由它自己推理决定。
    """

    def __init__(
        self,
        llm_client: LLMClientBase,
        fact_agent: FactAgent,
        planning_agent: PlanningAgent,
        evaluation_agent: EvaluationAgent,
        max_rounds: int = 6,
    ):
        self._llm            = llm_client
        self._fact_agent     = fact_agent
        self._planning_agent = planning_agent
        self._eval_agent     = evaluation_agent
        self._max_rounds     = max_rounds

        # 运行时状态(每次 run() 重置)
        self._plan_context:       dict = {}
        self._current_fact_data:  dict = {}
        self._current_plan_data:  dict = {}
        self._updated_fact:       dict = {}
        self._updated_planning:   dict = {}
        self._updated_evaluation: dict = {}
        self._trace_id: str | None = None
        self._action_log: list[str] = []

    async def run(
        self,
        user_feedback: str,
        plan_context: dict,
        fact_data: dict,
        selected_plan: dict,
        trace_id: str | None = None,
    ) -> AgentResult:

        # 重置运行时状态
        self._plan_context       = plan_context
        self._current_fact_data  = dict(fact_data)
        self._current_plan_data  = {}
        self._updated_fact       = {}
        self._updated_planning   = {}
        self._updated_evaluation = {}
        self._trace_id           = trace_id or str(uuid.uuid4())
        self._action_log         = []

        tracer.start_trace(self._trace_id, "orchestrator", user_feedback)

        # 注册子 Agent 工具
        executor = ToolExecutor()
        for tool in self._build_tools():
            executor.register(tool)

        # 构建压缩上下文
        context = build_orchestrator_context(
            user_feedback=user_feedback,
            plan_context=plan_context,
            fact_data=fact_data,
            selected_plan=selected_plan,
        )

        tools = executor.schemas
        messages = [
            {"role": "system", "content": ORCHESTRATOR_SYSTEM_PROMPT},
            {"role": "user",   "content": context},
        ]

        try:
            for round_i in range(self._max_rounds):
                resp = await self._llm.call(
                    trace_id=self._trace_id,
                    messages=messages,
                    tools=tools,
                )
                msg = resp.choices[0].message

                # 没有工具调用 → 完成
                if not msg.tool_calls:
                    self._action_log.append("orchestrator: 推理完成，无更多工具调用")
                    break

                messages.append(msg)

                for tc in msg.tool_calls:
                    result_str = await executor.execute(tc, trace_id=self._trace_id)
                    self._action_log.append(f"{tc.function.name}: done")
                    messages.append({
                        "role":         "tool",
                        "tool_call_id": tc.id,
                        "content":      result_str,
                    })

        except Exception as e:
            tracer.end_trace(self._trace_id, str(e), status="error")
            return AgentResult(
                status="error",
                summary=f"Orchestrator 运行异常: {e}",
                data=OrchestratorData().model_dump(),
            )

        result_data = OrchestratorData(
            updated_fact=self._updated_fact,
            updated_planning=self._updated_planning,
            updated_evaluation=self._updated_evaluation,
            action_log=self._action_log,
        )

        summary = " → ".join(self._action_log) or "无操作"
        tracer.end_trace(self._trace_id, summary, status="success")

        return AgentResult(
            status="ok",
            summary=summary,
            data=result_data.model_dump(),
        )

    def _build_tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name="search_pois",
                description="搜索新的 POI 候选。当候选池里没有用户需要的类型时调用。",
                parameters={
                    "keywords": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "搜索关键词列表，如 ['烧烤', '烤肉']",
                    },
                    "is_restaurant": {
                        "type": "boolean",
                        "description": "True=搜餐厅，False=搜活动",
                    },
                },
                func=self._tool_search_pois,
                required=["keywords", "is_restaurant"],
            ),
            ToolDefinition(
                name="replan",
                description=(
                    "基于当前 POI 池重新生成 3 个候选方案。"
                    "用户反馈涉及方案内容修改时调用。"
                ),
                parameters={
                    "hint": {
                        "type": "string",
                        "description": "给规划 Agent 的调整提示，如'将餐厅替换为川菜类型'",
                    },
                },
                func=self._tool_replan,
                required=[],
            ),
            ToolDefinition(
                name="evaluate",
                description="对 replan 生成的新方案评估选优。replan 之后必须调用。",
                parameters={},
                func=self._tool_evaluate,
                required=[],
            ),
        ]

    async def _tool_search_pois(self, keywords: list, is_restaurant: bool) -> str:
        task = (
            f"出发地：{self._plan_context.get('origin_area')}\n"
            f"搜索关键词：{'、'.join(keywords)}\n"
            f"类型：{'餐厅' if is_restaurant else '活动'}\n"
            f"场景：{self._plan_context.get('scenario')}，"
            f"人数：{self._plan_context.get('people_count')}"
        )
        result = await self._fact_agent.run(task=task, trace_id=self._trace_id)
        if result.status == "ok":
            new_data = result.data
            for key in ("activities", "restaurants", "waypoints"):
                existing  = self._current_fact_data.get(key) or []
                new_items = new_data.get(key) or []
                merged_ids = {
                    p["id"] for p in existing
                    if isinstance(p, dict) and p.get("id")
                }
                self._current_fact_data[key] = existing + [
                    p for p in new_items
                    if isinstance(p, dict) and p.get("id") not in merged_ids
                ]
            self._updated_fact = {
                "status":  result.status,
                "summary": result.summary,
                "data":    self._current_fact_data,
            }
        return result.as_tool_output()

    async def _tool_replan(self, hint: str = "") -> str:
        updated_context = dict(self._plan_context)
        if hint:
            updated_context["raw_query"] = (
                f"{self._plan_context.get('raw_query', '')}\n[调整需求]{hint}"
            )
        result = await self._planning_agent.run(
            plan_context=updated_context,
            fact_data=self._current_fact_data,
            replan_reason=hint,
            trace_id=self._trace_id,
        )
        if result.status == "ok":
            self._current_plan_data = result.data
            self._updated_planning  = {
                "status":  result.status,
                "summary": result.summary,
                "data":    result.data,
            }
        return result.as_tool_output()

    async def _tool_evaluate(self) -> str:
        result = await self._eval_agent.run(
            plan_data=self._current_plan_data,
            plan_context=self._plan_context,
            fact_data=self._current_fact_data,
            trace_id=self._trace_id,
        )
        if result.status == "ok":
            self._updated_evaluation = {
                "status":  result.status,
                "summary": result.summary,
                "data":    result.data,
            }
        return result.as_tool_output()