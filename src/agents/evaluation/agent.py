# agents/evaluation/agent.py
from __future__ import annotations

import json

from agents.evaluation.schema import EvalData, PlanScore
from agents.evaluation.prompt import EVAL_SYSTEM_PROMPT
from agents.evaluation.tools import validate_candidate, build_candidate_summary
from agents.planning.agent import current_request_of
from harness.agent import Agent, Budget, build_finish_tool
from harness.agent.result import AgentResult
from harness.llm.base import LLMClientBase
from harness.tools.tool_executor import ToolExecutor
from harness.tracing.span import Span

FINISH_TOOL = "finish"


class EvaluationAgent:
    """方案评估 Agent。

    两个阶段：
      1. 代码校验（确定性）：过滤物理不可行的方案
      2. LLM 评分（语义）：对合格方案打分选优

    【harness 重构后的改动】StructuredAgent 已被删除，改用
    Agent + build_finish_tool(EvalData) + require_terminal_tool。
    行为等价：finish 工具的 terminal=True 让循环在参数通过 schema
    校验后立即收口，require_terminal_tool 保证模型不会用自由文本
    糊弄过去（会被 nudge 回来）。

    span：不自己开，直接把 parent_span 透传给 Agent——Agent 内部
    会以 name="evaluation" 开自己的 span 并正确挂到父节点上。
    这里再包一层只会在 trace 树上多一个没有自己 LLM 调用的空节点。
    """

    def __init__(self, llm_client: LLMClientBase):
        self._llm = llm_client

    def _build_eta_dict(self, fact_data: dict) -> dict:
        eta = {}
        all_pois = (
            (fact_data.get("activities") or [])
            + (fact_data.get("restaurants") or [])
            + (fact_data.get("waypoints") or [])
        )
        for poi in all_pois:
            if isinstance(poi, dict) and poi.get("id") and poi.get("eta_minutes") is not None:
                eta[poi["id"]] = poi["eta_minutes"]
        return eta

    async def run(
        self,
        plan_data: dict,
        plan_context: dict,
        fact_data: dict,
        parent_span: Span | None = None,
    ) -> AgentResult:

        candidates = plan_data.get("candidates") or []

        # ── 阶段1：代码校验 ────────────────────────────────────────────
        valid_candidates = []
        all_scores: list[PlanScore] = []

        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            violations = validate_candidate(candidate, plan_context)
            if violations:
                all_scores.append(PlanScore(
                    candidate_id=candidate.get("id", ""),
                    score=0.0,
                    reason="物理校验未通过",
                    passed_validation=False,
                    violations=violations,
                ))
            else:
                valid_candidates.append(candidate)

        if not valid_candidates:
            return AgentResult(
                status="error",
                summary="所有候选方案均未通过物理校验",
                data=EvalData(selected_plan_id="", all_scores=all_scores).model_dump(),
            )

        # ── 阶段2：LLM 评分 ────────────────────────────────────────────
        eta_dict = self._build_eta_dict(fact_data)
        summaries = [build_candidate_summary(c, eta_dict) for c in valid_candidates]

        # 读 current_request 而不是 raw_query。
        #
        # 【实测到的分裂】用户先说"晚上吃个火锅"，再说"不吃火锅了，
        # 吃不了辣"。结构化字段全部更新正确（restaurant_intent
        # explicit→open），Planning 也照做了、换成了不辣的餐厅，
        # 然后评估器打了 58 分，评语是：
        #
        #   三个方案均将用户明确点名的'火锅'替换为其他不辣餐食，
        #   核心需求未精确匹配，需求匹配度严重不足
        #
        # 系统照做了用户的要求，然后给自己打了不及格——因为它读的
        # 是 raw_query（最初原话，含已撤回的"要火锅"）。
        # 模型读原话比读结构化字段更直觉，所以旧的那套赢了。
        user_message = f"""\
用户当前需求：
{current_request_of(plan_context)}

场景：{plan_context.get("scenario")}，人数：{plan_context.get("people_count")}
时间：{plan_context.get("start_time")} - {plan_context.get("end_time")}
饮食偏好：{(plan_context.get("preferences") or {}).get("diet") or []}
明确避免：{(plan_context.get("preferences") or {}).get("avoid") or []}

候选方案（已通过物理校验，共 {len(summaries)} 个）：
{json.dumps(summaries, ensure_ascii=False, indent=2)}

评分时以"用户当前需求"为准。如果其中包含后续调整，说明用户已经
修改了最初的想法——被撤回的要求不再是需求，方案没有满足它不是
缺陷，不要因此扣分。

请对每个方案打分并选出最优方案，调用 {FINISH_TOOL} 工具输出结果。
"""

        executor = ToolExecutor()
        executor.register(build_finish_tool(EvalData, name=FINISH_TOOL))

        agent = Agent(
            llm_client=self._llm,
            tool_executor=executor,
            system_prompt=EVAL_SYSTEM_PROMPT,
            # max_tool_calls=3：正常只需 1 次 finish，留 2 次给参数
            # 校验失败后的重试（ToolExecutor 的【参数错误】反馈路径）。
            budget=Budget(max_tool_calls=3, max_rounds=6, exhausted_action="stop"),
            require_terminal_tool=FINISH_TOOL,
            name="evaluation",
        )

        result = await agent.run_result(task=user_message, parent_span=parent_span)

        if result.status == "error":
            return AgentResult(
                status="error",
                summary=f"评分失败: {result.summary}",
                data=EvalData(all_scores=all_scores).model_dump(),
            )

        # ⚠️ 必须显式校验，不能直接 model_validate。
        # harness 的 run_result() 在"模型始终没调 finish、nudge 用尽
        # 之后放弃"的情况下，返回的是 status="ok" +
        # data={"text": "<自由文本>"}（outcome.status == "completed"
        # 时一律当成功，不区分 reason == "terminal_tool_not_called"）。
        # 那份 data 不是 EvalData 形状，直接 model_validate 会抛异常，
        # 而且异常长得像 schema 问题，掩盖了真实原因。
        try:
            eval_data = EvalData.model_validate(result.data)
        except Exception as e:
            return AgentResult(
                status="error",
                summary=f"评分结果不是合法的 EvalData（模型可能未调用 {FINISH_TOOL}）: {e}",
                data=EvalData(all_scores=all_scores).model_dump(),
            )

        if not eval_data.selected_plan_id:
            return AgentResult(
                status="error",
                summary="评分完成但未选出方案（selected_plan_id 为空）",
                data=EvalData(all_scores=all_scores).model_dump(),
            )

        eval_data.all_scores = all_scores + [
            s for s in eval_data.all_scores
            if s.candidate_id not in {x.candidate_id for x in all_scores}
        ]

        return AgentResult(
            status="ok",
            summary=f"选中方案 {eval_data.selected_plan_id}，得分 {eval_data.selected_score}",
            data=eval_data.model_dump(),
        )