from __future__ import annotations

import json

from agents.evaluation.schema import EvalData, PlanScore
from agents.evaluation.prompt import EVAL_SYSTEM_PROMPT
from agents.evaluation.tools import validate_candidate, build_candidate_summary
from harness.agent.result import AgentResult
from harness.agent.structured_agent import StructuredAgent
from harness.llm.base import LLMClientBase
from harness.tools.tool_executor import ToolExecutor


class EvaluationAgent:
    """方案评估 Agent。

    两个阶段:
    1. 代码校验(确定性):过滤物理不可行的方案
    2. LLM 评分(语义):对合格方案打分选优

    用 StructuredAgent + finish tool 约束输出
    因为 EvalData 是扁平结构,适合 function calling 强约束。
    """

    def __init__(self, llm_client: LLMClientBase):
        self._llm = llm_client

    def _build_eta_dict(self, fact_data: dict) -> dict:
        """从 FactData 的 POI 列表里提取 eta,重建成 {poi_id: eta_minutes} 格式。"""
        eta = {}
        all_pois = (
                (fact_data.get("activities") or []) +
                (fact_data.get("restaurants") or []) +
                (fact_data.get("waypoints") or [])
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
        trace_id: str | None = None,
    ) -> AgentResult:

        candidates = plan_data.get("candidates") or []

        # ── 阶段1:代码校验 ────────────────────────────────────────────────
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

        # 全部不合格 → 直接返回,不调 LLM
        if not valid_candidates:
            eval_data = EvalData(
                selected_plan_id="",
                all_scores=all_scores,
            )
            return AgentResult(
                status="error",
                summary="所有候选方案均未通过物理校验",
                data=eval_data.model_dump(),
            )

        # ── 阶段2:LLM 评分 ────────────────────────────────────────────────
        eta_dict = self._build_eta_dict(fact_data)
        summaries = [build_candidate_summary(c, eta_dict) for c in valid_candidates]

        user_message = f"""\
用户原始需求：「{plan_context.get("raw_query") or ""}」

场景：{plan_context.get("scenario")}，人数：{plan_context.get("people_count")}
时间：{plan_context.get("start_time")} - {plan_context.get("end_time")}

候选方案（已通过物理校验，共 {len(summaries)} 个）：
{json.dumps(summaries, ensure_ascii=False, indent=2)}

请对每个方案打分并选出最优方案，调用 finish 工具输出结果。
"""
        # StructuredAgent + finish tool:EvalData 扁平,适合 function calling 强约束
        agent = StructuredAgent(
            llm_client=self._llm,
            tool_executor=ToolExecutor(),   # 无业务工具,只有 finish tool
            system_prompt=EVAL_SYSTEM_PROMPT,
            output_schema=EvalData,
            max_tool_calls=3,
            name="evaluation",
        )

        result = await agent.run(task=user_message, trace_id=trace_id)

        if result.status == "error":
            return AgentResult(
                status="error",
                summary=f"评分失败: {result.summary}",
                data=EvalData(all_scores=all_scores).model_dump(),
            )

        # 把代码校验阶段的分数合并进来
        eval_data = EvalData.model_validate(result.data)
        eval_data.all_scores = all_scores + [
            s for s in eval_data.all_scores
            if s.candidate_id not in {x.candidate_id for x in all_scores}
        ]

        return AgentResult(
            status="ok",
            summary=f"选中方案 {eval_data.selected_plan_id}，得分 {eval_data.selected_score}",
            data=eval_data.model_dump(),
        )
