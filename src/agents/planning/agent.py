# agents/planning/agent.py
from __future__ import annotations

import json
import logging
import re

from agents.planning.schema import PlanData, CandidatePlan
from agents.planning.prompt import PLANNING_SYSTEM_PROMPT
from agents.planning.tools import (
    assign_step_times,
    build_step_index,
    validate_and_normalize_steps,
    validate_meal_phase_consistency,
    calc_slot_budgets,
)
from harness.agent.result import AgentResult
from harness.llm.base import LLMClientBase
from harness.tracing.span import Span

logger = logging.getLogger(__name__)


def current_request_of(plan_context: dict) -> str:
    """取"用户当前有效的需求"。

    current_request 含最初原话 + 后续全部调整；raw_query 只有最初
    原话（含可能已被撤回的要求）。prompt 必须读前者——实测过读
    raw_query 的后果：用户说"不吃火锅了"，方案换成不辣的了，
    评估器却拿着"晚上吃个火锅"打了 58 分，评语是"核心需求未满足"。

    `or raw_query` 兜底：plan_context 可能来自旧 checkpoint，
    那时还没有 current_request 这个字段，退回原行为而不是变成空串。
    """
    return plan_context.get("current_request") or plan_context.get("raw_query") or ""


class PlanningAgent:
    """行程规划 Agent。

    不需要外部工具调用（POI 数据已由 Fact Agent 收集完毕），
    单次 LLM 调用生成 3 个候选方案，再由时刻求解器和后处理逻辑
    组装成完整的 PlanData。

    推理（模型决定放什么 POI、排几个步骤、判断 category）与计算
    （求解器算具体几点）严格分离：模型只输出 steps 结构、时长和
    category，求解器负责时刻计算。

    【tracing】自己开 span、自己 end，与 FactAgent / OrchestratorAgent
    对齐。原来复用调用方传入的 trace_id 且从不 start_trace，导致本
    Agent 的 LLM 调用被 record_llm_call 开头的
    `if trace is None: return` 静默丢弃。
    """

    def __init__(self, llm_client: LLMClientBase, max_retries: int = 1):
        self._llm = llm_client
        self._max_retries = max_retries

    async def run(
        self,
        plan_context: dict,
        fact_data: dict,
        replan_reason: str = "",
        parent_span: Span | None = None,
    ) -> AgentResult:
        span = Span.begin("planning", current_request_of(plan_context) or "规划",
                         parent=parent_span)
        try:
            result = await self._run_inner(plan_context, fact_data, replan_reason, span)
            span.end(result.summary, status="success" if result.status == "ok" else "error")
            return result
        except Exception as e:
            span.end(str(e), status="error")
            raise

    async def _run_inner(
        self, plan_context: dict, fact_data: dict, replan_reason: str, span: Span
    ) -> AgentResult:
        user_message = self._build_user_message(plan_context, fact_data, replan_reason)
        messages = [
            {"role": "system", "content": PLANNING_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]

        last_content = ""
        for attempt in range(self._max_retries + 1):
            try:
                resp = await self._llm.call(
                    trace_id=span.trace_id,
                    messages=messages,
                    response_format={"type": "json_object"},
                )
                last_content = resp.choices[0].message.content or ""
                plan_data = self._parse_and_postprocess(last_content, plan_context, fact_data)
                return AgentResult(
                    status="ok",
                    summary=f"生成 {len(plan_data.candidates)} 个候选方案",
                    data=plan_data.model_dump(),
                )
            except Exception as e:
                if attempt == self._max_retries:
                    logger.error(f"[PlanningAgent] 规划生成失败（已重试 {attempt} 次）: {e}")
                    return AgentResult(status="error",
                                      summary=f"规划生成失败: {e}", data={})
                logger.warning(f"[PlanningAgent] 第 {attempt + 1} 次输出不合格，回灌重试: {e}")
                messages.append({"role": "assistant", "content": last_content})
                messages.append({
                    "role": "user",
                    "content": f"输出格式不正确: {e}。请只输出 JSON，不要其他内容。",
                })

    def _parse_and_postprocess(
        self, content: str, plan_context: dict, fact_data: dict
    ) -> PlanData:
        # 原来这里有两句 print 直接把模型原始输出打到 stdout。改成
        # debug 日志：print 在生产里既关不掉也不带级别，而这段内容
        # 可能很长、还会混进实验脚本的输出里干扰结果判读。
        logger.debug(f"[PlanningAgent] raw output:\n{content}")

        match = re.search(r"\{.*\}", content, re.DOTALL)
        raw_text = match.group() if match else content
        payload = json.loads(raw_text)
        raw_candidates = payload.get("candidates") or []

        activities = fact_data.get("activities") or []
        restaurants = fact_data.get("restaurants") or []
        waypoints = fact_data.get("waypoints") or []
        eta = self._build_eta_dict(fact_data)

        activities_by_id = {a["id"]: a for a in activities if isinstance(a, dict) and a.get("id")}
        restaurants_by_id = {r["id"]: r for r in restaurants if isinstance(r, dict) and r.get("id")}
        waypoints_by_id = {w["id"]: w for w in waypoints if isinstance(w, dict) and w.get("id")}

        start_time = plan_context.get("start_time") or "10:00"
        end_time = plan_context.get("end_time") or ""

        normalized: list[CandidatePlan] = []
        # 变量名用 raw_candidate 而不是 raw：原代码在这个循环里用 raw
        # 覆盖了上面 `raw = match.group()` 那个变量，两个完全不同的
        # 东西共用一个名字，读的时候要回溯才能确定当前的 raw 是哪个。
        for i, raw_candidate in enumerate(raw_candidates[:3], 1):
            if not isinstance(raw_candidate, dict):
                continue

            valid_steps = validate_and_normalize_steps(
                raw_candidate.get("steps") or [],
                activities_by_id, restaurants_by_id, waypoints_by_id,
            )
            valid_steps = validate_meal_phase_consistency(valid_steps)
            timed_steps = assign_step_times(valid_steps, start_time, end_time, eta)
            index = build_step_index(
                timed_steps, activities_by_id, restaurants_by_id, waypoints_by_id,
            )

            normalized.append(CandidatePlan(
                id=raw_candidate.get("id") or f"plan_{i}",
                title=raw_candidate.get("title") or f"候选方案 {i}",
                steps=timed_steps,
                reasoning=[r for r in (raw_candidate.get("reasoning") or [])
                           if isinstance(r, str)][:3],
                **index,
            ))

        while len(normalized) < 3:
            normalized.append(CandidatePlan(id=f"plan_{len(normalized) + 1}"))

        return PlanData(
            plan_mode=plan_context.get("plan_mode") or "activity_plus_meal",
            candidates=normalized,
        )

    def _build_eta_dict(self, fact_data: dict) -> dict:
        """从 FactData 的 POI 列表里提取 eta，重建成 {poi_id: {eta_minutes: N}}。"""
        eta = {}
        for poi in (
            (fact_data.get("activities") or [])
            + (fact_data.get("restaurants") or [])
            + (fact_data.get("waypoints") or [])
        ):
            if isinstance(poi, dict) and poi.get("id") and poi.get("eta_minutes") is not None:
                eta[poi["id"]] = {"eta_minutes": poi["eta_minutes"]}
        return eta

    def _build_user_message(
        self, plan_context: dict, fact_data: dict, replan_reason: str
    ) -> str:
        activities = fact_data.get("activities") or []
        restaurants = fact_data.get("restaurants") or []
        waypoints = fact_data.get("waypoints") or []
        weather = fact_data.get("weather") or {}
        prefs = plan_context.get("preferences") or {}

        start_time = plan_context.get("start_time") or "10:00"
        end_time = plan_context.get("end_time") or ""
        budgets = calc_slot_budgets(start_time, end_time)
        _label_map = {"morning": "上午", "afternoon": "下午", "evening": "夜间"}
        _guide_map = {
            "morning": "< 90min → 最多1个short；90~180min → 1个medium；> 180min → 1个long",
            "afternoon": "90~180min → 1个medium；180~270min → 1个long；> 270min → 必须2个活动",
            "evening": "按剩余时间和用户需求自由安排",
        }
        slot_lines = [
            f"- {_label_map.get(s, s)}（{s}）：可用 {m} 分钟 → {_guide_map.get(s, '')}"
            for s, m in budgets.items()
        ] or ["- 无需安排活动时段"]

        explicit = (
            "、".join(prefs.get("activity") or [])
            or "（无明确点名，由你根据场景自由推荐）"
        )

        plan_mode = plan_context.get("plan_mode", "activity_plus_meal")
        need_activity = plan_mode != "meal_only"
        need_restaurant = plan_mode != "activity_only"

        waypoint_pool_text = json.dumps([
            {"id": w.get("id"), "name": w.get("name"), "type": w.get("type"),
             "rating": w.get("rating")}
            for w in waypoints
        ], ensure_ascii=False, indent=2) if waypoints else "无途径点候选"

        activity_pool_text = json.dumps([
            {"id": a.get("id"), "name": a.get("name"), "type": a.get("type"),
             "rating": a.get("rating")}
            for a in activities
        ], ensure_ascii=False, indent=2) if need_activity else "本次无需安排活动"

        restaurant_pool_text = json.dumps([
            {"id": r.get("id"), "name": r.get("name"), "type": r.get("type"),
             "rating": r.get("rating"), "cost": r.get("cost")}
            for r in restaurants
        ], ensure_ascii=False, indent=2) if need_restaurant else "本次无需安排餐厅"

        replan_section = (
            f"## 上一轮校验失败原因（必须修正）\n{replan_reason}"
            if replan_reason else ""
        )

        # 读 current_request 而不是 raw_query：后者是最初原话，
        # 含可能已被用户撤回的要求（见 current_request_of 的说明）。
        return f"""\
# 用户当前需求
{current_request_of(plan_context)}

## 用户明确点名的活动（硬约束）
{explicit}

## 时间信息
- 日期：{plan_context.get("date_label") or "未指定"}
- 开始时间：{start_time}
- 结束时间：{end_time or "未指定"}
- 时间描述：{plan_context.get("time_phrase") or ""}

## 各时段可用分钟数
{chr(10).join(slot_lines)}

## 人员信息
- 场景：{plan_context.get("scenario") or "friends"}
- 人数：{plan_context.get("people_count") or 2}
- 是否有孩子：{plan_context.get("child_friendly") or False}

## 出发地
{plan_context.get("origin_area") or "未指定"}

## 用户偏好
- 活动风格：{prefs.get("activity") or []}
- 明确避免：{prefs.get("avoid") or []}
- 饮食偏好：{prefs.get("diet") or []}

## 天气
{weather.get("day_weather", "未知")}，{weather.get("day_temp", "?")}℃，{weather.get("day_wind", "未知")}

## 可选活动候选池
{activity_pool_text}

## 可选餐厅候选池（cost 为人均消费，null 表示高德未提供）
{restaurant_pool_text}

## 可选途径点候选池（甜品/饮品类，可能是 meal，也可能是 snack_drink，
自行按 POI 名称和类型判断 category）
{waypoint_pool_text}

请生成 3 个候选方案，不要填写 start_time / end_time，系统自动计算时刻。
{replan_section}
"""