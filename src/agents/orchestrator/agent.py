# agents/orchestrator/agent.py
from __future__ import annotations

import logging

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
from harness.tracing.span import Span

logger = logging.getLogger(__name__)


class OrchestratorAgent:
    """用户反馈调度 Agent。

    用 ToolExecutor + ToolDefinition 管理子 Agent 工具调用，
    自己控制 ReAct loop——调用顺序和次数由它自己推理决定。

    【tracing】原来直接 tracer.start_trace(trace_id, ...)，trace_id
    由调用方传入且等于 session_id，导致同一会话里每次调度都覆盖
    前一次的 trace。改成 Span.begin(parent=parent_span)。

    【为什么仍是手写循环】它调的是子 Agent，需要在工具之间维护
    _current_fact_data / _current_plan_data 这类跨调用累积状态。
    改成 harness Agent 是一件独立的、值得做的事（RunContext.state
    正是为此设计的），但不该和别的改动混在一起。
    """

    def __init__(
        self,
        llm_client: LLMClientBase,
        fact_agent: FactAgent,
        planning_agent: PlanningAgent,
        evaluation_agent: EvaluationAgent,
        max_rounds: int = 6,
    ):
        self._llm = llm_client
        self._fact_agent = fact_agent
        self._planning_agent = planning_agent
        self._eval_agent = evaluation_agent
        self._max_rounds = max_rounds

        self._plan_context: dict = {}
        self._current_fact_data: dict = {}
        self._current_plan_data: dict = {}
        self._updated_fact: dict = {}
        self._updated_planning: dict = {}
        self._updated_evaluation: dict = {}
        self._span: Span | None = None
        self._action_log: list[str] = []

    async def run(
        self,
        user_feedback: str,
        plan_context: dict,
        fact_data: dict,
        selected_plan: dict,
        parent_span: Span | None = None,
    ) -> AgentResult:

        self._plan_context = dict(plan_context)
        self._current_fact_data = dict(fact_data)
        self._current_plan_data = {}
        self._updated_fact = {}
        self._updated_planning = {}
        self._updated_evaluation = {}
        self._action_log = []

        span = Span.begin("orchestrator", user_feedback, parent=parent_span)
        self._span = span

        executor = ToolExecutor()
        for tool in self._build_tools():
            executor.register(tool)

        context = build_orchestrator_context(
            user_feedback=user_feedback,
            plan_context=plan_context,
            fact_data=fact_data,
            selected_plan=selected_plan,
        )
        # 当前方案各 POI 的车程，给模型判断"更近"时有个参照系。
        # 不给参照系，它无法把"太远了"翻译成一个具体的分钟数上限。
        current_max_eta = self._max_eta_of(selected_plan, fact_data)
        if current_max_eta is not None:
            context += (
                f"\n\n[当前方案最远的一站车程约 {current_max_eta} 分钟。"
                f"如果用户要求更近，调用 replan 时把 max_eta_minutes 设成一个"
                f"**小于 {current_max_eta}** 的值，系统会强制过滤掉超过它的候选。]"
            )

        tools = executor.schemas
        messages = [
            {"role": "system", "content": ORCHESTRATOR_SYSTEM_PROMPT},
            {"role": "user", "content": context},
        ]

        try:
            for _round in range(self._max_rounds):
                resp = await self._llm.call(trace_id=span.trace_id,
                                           messages=messages, tools=tools)
                msg = resp.choices[0].message

                if not msg.tool_calls:
                    self._action_log.append("orchestrator: 推理完成，无更多工具调用")
                    break

                messages.append(msg)
                for tc in msg.tool_calls:
                    result_str = await executor.execute(tc, trace_id=span.trace_id)
                    self._action_log.append(f"{tc.function.name}: done")
                    messages.append({"role": "tool", "tool_call_id": tc.id,
                                    "content": result_str})

        except Exception as e:
            span.end(str(e), status="error")
            return AgentResult(status="error",
                              summary=f"Orchestrator 运行异常: {e}",
                              data=OrchestratorData().model_dump())

        result_data = OrchestratorData(
            updated_fact=self._updated_fact,
            updated_planning=self._updated_planning,
            updated_evaluation=self._updated_evaluation,
            action_log=self._action_log,
        )
        summary = " → ".join(self._action_log) or "无操作"
        span.end(summary, status="success")
        return AgentResult(status="ok", summary=summary,
                          data=result_data.model_dump())

    # ── 工具 ──────────────────────────────────────────────────────

    def _build_tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name="search_pois",
                description="搜索新的 POI 候选。当候选池里没有用户需要的类型时调用。",
                parameters={
                    "keywords": {
                        "type": "array", "items": {"type": "string"},
                        "description": "搜索关键词列表，如 ['烧烤', '烤肉']",
                    },
                    "is_restaurant": {
                        "type": "boolean",
                        "description": "True=搜餐厅，False=搜活动",
                    },
                },
                func=self._tool_search_pois,
                required=["keywords", "is_restaurant"],
                read_only=True,
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
                        "description": "给规划 Agent 的调整提示，如'将餐厅替换为不辣的类型'",
                    },
                    "max_eta_minutes": {
                        "type": "integer",
                        "description": (
                            "距离上限（分钟）。**只在用户明确要求更近/嫌远时才传**。"
                            "传了之后系统会从候选池里**强制删除**车程超过这个值的"
                            "POI，规划 Agent 根本看不到它们——这不是建议，是硬过滤。"
                            "要传就传一个小于当前方案最远车程的值，否则没有效果。"
                        ),
                    },
                },
                func=self._tool_replan,
                required=[],
                read_only=True,
            ),
            ToolDefinition(
                name="evaluate",
                description="对 replan 生成的新方案评估选优。replan 之后必须调用。",
                parameters={},
                func=self._tool_evaluate,
                required=[],
                read_only=True,
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
        result = await self._fact_agent.run(
            task=task,
            origin_city=self._plan_context.get("origin_city") or "",
            parent_span=self._span,
        )
        if result.status in ("ok", "partial"):
            new_data = result.data
            for key in ("activities", "restaurants", "waypoints"):
                existing = self._current_fact_data.get(key) or []
                new_items = new_data.get(key) or []
                merged_ids = {p["id"] for p in existing
                             if isinstance(p, dict) and p.get("id")}
                self._current_fact_data[key] = existing + [
                    p for p in new_items
                    if isinstance(p, dict) and p.get("id") not in merged_ids
                ]
            self._updated_fact = {
                "status": result.status, "summary": result.summary,
                "data": self._current_fact_data,
            }
        return result.as_tool_output()

    async def _tool_replan(self, hint: str = "", max_eta_minutes: int | None = None) -> str:
        """重新规划。

        【max_eta_minutes 是硬过滤，修一个实测到的真 bug】
        用户说"这个餐厅太远了，换个近点的"，实测结果是 eta 从 7 分钟
        变成了 9 分钟——**方案确实换了，但换成了更远的**。

        根因：距离约束只以自由文本形式进了 replan 的 hint，而
        PlanningAgent 从整个候选池里挑，EvaluationAgent 的评分维度里
        距离只占"综合质量"20% 权重的一部分，压不过"整体体验"。
        用户说的"换个近点的"被理解成了"换一个"。

        修法不是调权重（那是另一个不可控的软约束），而是**从候选池
        里物理删除超限的 POI**——模型看不到它们，就不可能选中。
        这和 explicit 模式下强制覆盖搜索关键词是同一条思路：
        约束要有执行机构，不能指望模型自觉。

        由模型判断"用户是不是在要求更近、上限该是多少"（那是语义
        判断），由代码执行过滤（那是确定性操作）。分工清楚。

        【就地更新 self._plan_context】原实现只改了一份局部副本，
        导致 _tool_evaluate 用的还是旧的——"规划听懂了、评估没听懂"
        的分裂：方案确实换成了不辣的，评估器却拿旧需求打了 58 分。
        """
        if hint:
            base = (self._plan_context.get("current_request")
                    or self._plan_context.get("raw_query", ""))
            self._plan_context = {
                **self._plan_context,
                "current_request": f"{base}\n[本次调整]{hint}",
            }

        fact_data = self._current_fact_data
        if max_eta_minutes and max_eta_minutes > 0:
            fact_data, dropped = self._filter_by_eta(fact_data, max_eta_minutes)
            self._plan_context = {
                **self._plan_context,
                "max_traffic_minutes": max_eta_minutes,
            }
            self._action_log.append(f"按 eta≤{max_eta_minutes}min 过滤掉 {dropped} 个候选")
            logger.info(f"[Orchestrator] 距离硬约束 eta≤{max_eta_minutes}min，"
                       f"候选池删除 {dropped} 个")

        result = await self._planning_agent.run(
            plan_context=self._plan_context,
            fact_data=fact_data,
            replan_reason=hint,
            parent_span=self._span,
        )
        if result.status == "ok":
            self._current_plan_data = result.data
            self._updated_planning = {
                "status": result.status, "summary": result.summary,
                "data": result.data,
            }
        return result.as_tool_output()

    async def _tool_evaluate(self) -> str:
        result = await self._eval_agent.run(
            plan_data=self._current_plan_data,
            plan_context=self._plan_context,
            fact_data=self._current_fact_data,
            parent_span=self._span,
        )
        if result.status == "ok":
            self._updated_evaluation = {
                "status": result.status, "summary": result.summary,
                "data": result.data,
            }
        return result.as_tool_output()

    # ── 辅助 ──────────────────────────────────────────────────────

    @staticmethod
    def _max_eta_of(selected_plan: dict, fact_data: dict) -> int | None:
        """当前方案里最远的一站是多少分钟。给模型一个参照系——
        没有参照系它无法把"太远了"翻译成一个具体的分钟数。"""
        by_id: dict[str, dict] = {}
        for key in ("activities", "restaurants", "waypoints"):
            for p in fact_data.get(key) or []:
                if isinstance(p, dict) and p.get("id"):
                    by_id[p["id"]] = p
        etas = []
        for step in (selected_plan or {}).get("steps") or []:
            pid = step.get("poi_id") or step.get("id")
            poi = by_id.get(pid) if pid else None
            if poi and poi.get("eta_minutes") is not None:
                etas.append(poi["eta_minutes"])
        return max(etas) if etas else None

    @staticmethod
    def _filter_by_eta(fact_data: dict, max_eta: int) -> tuple[dict, int]:
        """从候选池里删除车程超限的 POI。

        eta 为 None 的**保留**：None 的语义是"没算出来"，不是"很远"。
        按"未知即超限"删掉，等于用缺失数据惩罚候选——这和把
        cost=None 当成 0 是同一类错误（伪造确定性）。
        代价是过滤不彻底，但方向上偏保守（宁可留下也不误删）。
        """
        out = dict(fact_data)
        dropped = 0
        for key in ("activities", "restaurants", "waypoints"):
            items = fact_data.get(key) or []
            kept = []
            for p in items:
                eta = p.get("eta_minutes") if isinstance(p, dict) else None
                if eta is not None and eta > max_eta:
                    dropped += 1
                    continue
                kept.append(p)
            # 某一类全被过滤光时不执行过滤——给一个空候选池，
            # Planning 只能产出空方案，用户得到的是"没有结果"而不是
            # "更近的结果"。宁可放宽也不交付空方案。
            out[key] = kept if kept else items
            if not kept and items:
                logger.warning(f"[Orchestrator] eta≤{max_eta}min 会清空 {key}，"
                              f"该类别不执行过滤")
        return out, dropped