from datetime import datetime
import json
import re
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from src.agent.constraint_agent import ConstraintAgent
from src.agent.execution_agent import ExecutionAgent
from src.agent.intent_agent import IntentAgent
from src.agent.planning_agent import PlanningAgent
from src.agent.presentation_agent import PresentationAgent
from src.agent.retrieval_agent import RetrievalAgent
from src.graph.state import AgentState
from src.graph.time_contract import missing_required_time_fields
from src.model.factory import chat_model
from src.tools.mock_api import MockToolAPI


_LLM_ANSWER_FALLBACK = (
    "该问题不属于本地生活规划范畴，建议直接咨询通用助手或搜索引擎。"
)


class CandidatePlanDraftItem(BaseModel):
    id: str = Field(..., description="candidate id such as plan_1")
    title: str = Field(..., description="short plan title")
    steps: list[dict] = Field(default_factory=list, description="ordered plan steps")
    activity_id: str = Field(default="", description="legacy single activity id")
    restaurant_ids: list[str] = Field(default_factory=list, description="legacy restaurant ids")
    reasoning: list[str] = Field(default_factory=list, description="2-3 short reasons")


class CandidatePlanDraftEnvelope(BaseModel):
    candidates: list[CandidatePlanDraftItem] = Field(default_factory=list)


def _extract_step_phases(steps: list[dict]) -> set[str]:
    phases: set[str] = set()
    for step in steps:
        if not isinstance(step, dict):
            continue
        phase = step.get("phase")
        if isinstance(phase, str) and phase.strip():
            phases.add(phase.strip())
    return phases


def _steps_cover_daypart(steps: list[dict], daypart: str) -> bool:
    """Semantic guardrail for LLM-produced plan skeletons."""
    if daypart != "全天":
        return True
    phases = _extract_step_phases(steps)
    return {"morning", "lunch", "afternoon", "dinner"}.issubset(phases)


def _extract_json_object(raw_text: str) -> str | None:
    text = (raw_text or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()

    start_index = text.find("{")
    end_index = text.rfind("}")
    if start_index == -1 or end_index == -1 or end_index <= start_index:
        return None
    return text[start_index : end_index + 1]


def _append_error(state: AgentState, error: str) -> AgentState:
    errors = list(state.get("errors", []))
    errors.append(error)
    return {"errors": errors}


def intent_node(state: AgentState) -> AgentState:
    try:
        intent = IntentAgent().parse(
            state["user_input"],
            state.get("runtime_origin_area", ""),
        )
        return {
            "intent": intent,
            "is_leisure_planning": intent.get("is_leisure_planning"),
            "need_retrieval": intent.get("need_retrieval"),
            "clarification_needed": intent.get("clarification_needed"),
            "missing_slots": intent.get("missing_slots"),
            "follow_up_message": intent.get("follow_up_message"),
        }
    except Exception as exc:
        return _append_error(state, f"Intent node failed: {exc}")


def time_normalize_node(state: AgentState) -> AgentState:
    """统一时间规范器：把今天过时的计划顺延到明天，并输出统一时间字段。"""
    print("[Time Normalize Node] 规范时间语义...")
    try:
        intent = state.get("intent")
        if not isinstance(intent, dict):
            intent = {}

        time_info = intent.get("time")
        if not isinstance(time_info, dict):
            time_info = {}

        date_label = time_info.get("date_label") if isinstance(time_info.get("date_label"), str) else ""
        daypart = time_info.get("daypart") if isinstance(time_info.get("daypart"), str) else ""
        time_phrase = time_info.get("time_phrase") if isinstance(time_info.get("time_phrase"), str) else ""

        normalized_date_label = date_label
        normalized_daypart = daypart
        normalized_time_phrase = time_phrase or (f"{date_label}{daypart}" if date_label and daypart else "")

        now = datetime.now()
        now_minutes = now.hour * 60 + now.minute
        cutoff_minutes = 18 * 60 if normalized_daypart in {"下午", "上午", "全天"} else 21 * 60
        if normalized_date_label == "今天" and now_minutes > cutoff_minutes:
            normalized_date_label = "明天"
            if normalized_time_phrase:
                normalized_time_phrase = normalized_time_phrase.replace("今天", "明天", 1)

        if normalized_daypart == "上午":
            base_start_minutes = 9 * 60 + 30
        elif normalized_daypart == "晚上":
            base_start_minutes = 18 * 60 + 30
        elif normalized_daypart == "全天":
            base_start_minutes = 9 * 60 + 30
        else:
            base_start_minutes = 14 * 60
        if normalized_date_label == "今天" and normalized_daypart in {"下午", "晚上"} and now_minutes + 15 > cutoff_minutes:
            normalized_date_label = "明天"

        normalized_time = {
            "normalized_date_label": normalized_date_label,
            "normalized_daypart": normalized_daypart,
            "normalized_time_phrase": normalized_time_phrase,
            "base_start_minutes": base_start_minutes,
            "current_minutes": now_minutes,
            "current_time": now.strftime("%H:%M"),
        }
        print(
            f"[Time Normalize Node] date_label={date_label!r}, daypart={daypart!r}, "
            f"normalized_date_label={normalized_date_label!r}, base_start_minutes={base_start_minutes}"
        )
        return {
            "normalized_time": normalized_time,
            "time_normalization_result": normalized_time,
        }
    except Exception as exc:
        print(f"[Time Normalize Node][WARN] 节点异常，记录错误: {exc}")
        return _append_error(state, f"Time Normalize node failed: {exc}")


def location_permission_node(state: AgentState) -> AgentState:
    """定位权限节点：当用户未明确地点时，先询问是否允许自动定位。"""
    print("[Location Permission Node] 用户未明确地点，准备请求定位授权...")
    follow_up = "如果你没说具体地点，我可以先用当前定位继续规划。是否允许？(y/n): "
    confirm = input(follow_up)
    granted = confirm.strip().lower() == "y"
    return {
        "location_permission_granted": granted,
    }


def location_fallback_node(state: AgentState) -> AgentState:
    """定位兜底节点：自动定位失败时，改为询问用户所在城市/区域。"""
    print("[Location Fallback Node] 自动定位失败，改为询问城市/区域...")
    user_area = input("你大概在哪个城市或区域？\n> ").strip()
    while not user_area:
        user_area = input("你大概在哪个城市或区域？\n> ").strip()
    return {
        "runtime_origin_area": user_area,
        "location_lookup_result": {
            "status": "fallback_user_input",
            "city": user_area,
            "reason": "用户手动补充城市/区域",
        },
    }


def location_lookup_node(state: AgentState) -> AgentState:
    """定位查询节点：根据 IP 做粗略定位，回填 runtime_origin_area。"""
    print("[Location Lookup Node] 开始获取用户定位...")
    try:
        amap = MockToolAPI()._get_amap()
        if amap is None:
            return {
                "location_lookup_result": {
                    "status": "unavailable",
                    "reason": "MCP 客户端不可用",
                }
            }

        import requests

        ip = ""
        ip_sources = [
            ("https://ifconfig.me/ip", "text"),
        ]
        for url, mode in ip_sources:
            try:
                resp = requests.get(url, timeout=5, headers={"User-Agent": "Mozilla/5.0"})
                if not resp.ok:
                    continue
                if mode == "json":
                    ip = resp.json().get("ip", "") or ""
                else:
                    ip = resp.text.strip()
                if ip:
                    break
            except Exception as exc:
                print(f"[Location Lookup Node][WARN] 获取公网 IP 失败: source={url!r}, error={exc}")

        if not ip:
            print("[Location Lookup Node][WARN] 无法获取公网 IP，无法调用 maps_ip_location")
            return {
                "runtime_origin_coordinates": "",
                "location_lookup_result": {
                    "status": "unavailable",
                    "reason": "无法获取公网 IP",
                }
            }

        lookup = amap.maps_ip_location(ip)
        if not isinstance(lookup, dict):
            lookup = {"status": "unknown"}

        city = lookup.get("city") if isinstance(lookup.get("city"), str) else ""
        adcode = lookup.get("adcode") if isinstance(lookup.get("adcode"), str) else ""
        rectangle = lookup.get("rectangle") if isinstance(lookup.get("rectangle"), str) else ""
        runtime_origin_coordinates = ""
        if rectangle and ";" in rectangle:
            try:
                p1, p2 = rectangle.split(";", 1)
                lng1, lat1 = [float(x) for x in p1.split(",")]
                lng2, lat2 = [float(x) for x in p2.split(",")]
                runtime_origin_coordinates = f"{(lng1 + lng2) / 2:.6f},{(lat1 + lat2) / 2:.6f}"
            except Exception:
                runtime_origin_coordinates = ""
        runtime_origin_area = city or adcode or ""
        print(
            f"[Location Lookup Node] result status={lookup.get('status')!r}, "
            f"city={city!r}, adcode={adcode!r}, rectangle={rectangle!r}, lookup={lookup}"
        )
        return {
            "runtime_origin_area": runtime_origin_area,
            "runtime_origin_coordinates": runtime_origin_coordinates,
            "location_lookup_result": lookup,
        }
    except Exception as exc:
        print(f"[Location Lookup Node][WARN] 定位失败: {exc}")
        return {
            "runtime_origin_coordinates": "",
            "location_lookup_result": {
                "status": "error",
                "reason": str(exc),
            }
        }


def llm_answer_node(state: AgentState) -> AgentState:
    """非规划任务直答节点：让 LLM 直接回答用户输入。失败时写入固定 fallback 文案。"""
    user_input = state.get("user_input", "")
    print("[LLM Answer Node] 检测到非本地生活规划任务，调用 LLM 直接回答...")
    try:
        response = chat_model.invoke([HumanMessage(content=user_input)])
        answer = getattr(response, "content", None)
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("LLM 返回为空或不是字符串")
        print(f"[OK] LLM 直答完成，长度={len(answer)}")
        return {"llm_answer": answer}
    except Exception as exc:
        print(f"[WARN] LLM 直答失败，使用固定 fallback: {exc}")
        update = _append_error(state, f"LLM Answer node failed: {exc}")
        update["llm_answer"] = _LLM_ANSWER_FALLBACK
        return update


# DEPRECATED (T8 起替换为 plan_candidate_node)：保留实现以便向后兼容旧链路 / 测试。
# 不再被 workflow.py 主图连入，移除前请确认无外部调用方依赖。

def clarification_node(state: AgentState) -> AgentState:
    """图内澄清节点：读取追问并收集补充输入。"""
    follow_up = state.get("follow_up_message")
    if not isinstance(follow_up, str) or not follow_up.strip():
        follow_up = "为了继续规划，请补充一下关键信息。"

    clarification_round = state.get("clarification_round", 0)
    if isinstance(clarification_round, bool) or not isinstance(clarification_round, int):
        clarification_round = 0

    print(f"[Clarification Node] {follow_up}")
    extra_input = input("> ")
    while not extra_input.strip():
        extra_input = input("> ")
    extra_input = extra_input.strip()

    turns = list(state.get("conversation_turns", []))
    if not turns:
        first_turn = state.get("user_input", "")
        if isinstance(first_turn, str) and first_turn.strip():
            turns = [first_turn]

    turns.append(extra_input)
    combined_input = "\n".join([turn for turn in turns if isinstance(turn, str) and turn.strip()])

    return {
        "clarification_round": clarification_round + 1,
        "conversation_turns": turns,
        "user_input": combined_input,
        "clarification_needed": False,
        "missing_slots": {},
        "follow_up_message": "",
    }


def planning_node(state: AgentState) -> AgentState:
    try:
        plan = PlanningAgent().plan(state.get("intent", {}))
        return {"plan": plan}
    except Exception as exc:
        return _append_error(state, f"Planning node failed: {exc}")


def plan_candidate_node(state: AgentState) -> AgentState:
    """Plan Candidate Node：消费 6 个并行节点的输出，合成 primary + backup。

    设计：
        * 完全不调用 LLM / MockToolAPI；纯函数代理 PlanningAgent().compose。
        * 同时写入：
              - state.candidates = 完整 {primary, backup} 结构
              - state.plan = candidates["primary"]，向后兼容旧
                PresentationAgent / ExecutionAgent。
        * 防御：
              - 任意上游字段缺失 / 非法 → 由 compose 内部按空安全默认处理；
              - compose 返回 {} → state.candidates={}, state.plan={}，不写错误；
              - compose 抛异常 → 兜底 catch，写 errors。
    """
    print("[Plan Candidate Node] 合成 primary + backup 候选方案...")
    try:
        fact_gathering = state.get("fact_gathering_result")
        if not isinstance(fact_gathering, dict):
            fact_gathering = {}

        weather = fact_gathering.get("weather") if isinstance(fact_gathering.get("weather"), dict) else state.get("weather") or {}
        activities = fact_gathering.get("activities") if isinstance(fact_gathering.get("activities"), list) else state.get("activities") or []
        restaurants = fact_gathering.get("restaurants") if isinstance(fact_gathering.get("restaurants"), list) else state.get("restaurants") or []
        traffic = fact_gathering.get("traffic") if isinstance(fact_gathering.get("traffic"), dict) else state.get("traffic") or {}
        queue = fact_gathering.get("queue") if isinstance(fact_gathering.get("queue"), dict) else state.get("queue") or {}
        crowd = fact_gathering.get("crowd") if isinstance(fact_gathering.get("crowd"), dict) else state.get("crowd") or {}

        candidates = PlanningAgent().compose(
            constraints=state.get("constraints") or {},
            constraint_build=state.get("constraint_build") or {},
            weather=weather,
            activities=activities,
            restaurants=restaurants,
            traffic=traffic,
            queue=queue,
            crowd=crowd,
        )
        if not isinstance(candidates, dict):
            candidates = {}
        primary = candidates.get("primary") if candidates else None
        if isinstance(primary, dict):
            # state.plan = candidates["primary"] 的同时，补齐旧 PresentationAgent /
            # ExecutionAgent 期望的 activities 列表键，避免破坏向后兼容。
            plan_compat = dict(primary)
            activity = primary.get("activity")
            plan_compat["activities"] = [activity] if isinstance(activity, dict) else []
        else:
            plan_compat = {}
        return {"candidates": candidates, "plan": plan_compat}
    except Exception as exc:
        print(f"[Plan Candidate Node][WARN] 节点异常，跳过并记录错误: {exc}")
        return _append_error(state, f"Plan Candidate node failed: {exc}")


def candidate_planning_node(state: AgentState) -> AgentState:
    """新架构候选计划节点：LLM 在真实候选池中选择并组织 3 个结构化候选计划。"""
    print("[Candidate Planning Node] 基于真实候选池生成 3 个结构化候选方案...")
    try:
        constraint_build = state.get("constraint_build")
        if not isinstance(constraint_build, dict):
            constraint_build = {}

        fact_gathering_result = state.get("fact_gathering_result")
        if not isinstance(fact_gathering_result, dict):
            fact_gathering_result = {}

        request_type = constraint_build.get("request_type", "generic_local_plan")
        plan_mode = constraint_build.get("plan_mode", "activity_plus_meal")
        hard_constraints = constraint_build.get("hard_constraints") or {}
        soft_preferences = constraint_build.get("soft_preferences") or {}
        query_constraints = constraint_build.get("query_constraints") or {}
        activities = fact_gathering_result.get("activities")
        if not isinstance(activities, list):
            activities = []
        restaurants = fact_gathering_result.get("restaurants")
        if not isinstance(restaurants, list):
            restaurants = []

        daypart = hard_constraints.get("daypart") if isinstance(hard_constraints, dict) else ""
        date_label = hard_constraints.get("date_label") if isinstance(hard_constraints, dict) else ""
        time_phrase = f"{date_label}{daypart}" if date_label and daypart else (date_label or daypart or "待确认时间")

        def _timeline_item(
            *,
            time_label: str,
            item_name: str,
            item_type: str,
            ref_type: str,
            ref_id: str,
        ) -> dict:
            return {
                "time": time_label,
                "item": item_name,
                "type": item_type,
                "ref_type": ref_type,
                "ref_id": ref_id,
            }

        def _fallback_steps(
            fallback_activity_ids: list[str],
            fallback_restaurant_ids: list[str],
        ) -> list[list[dict]]:
            if daypart == "全天":
                activity_a = fallback_activity_ids[0] if fallback_activity_ids else ""
                activity_b = fallback_activity_ids[1] if len(fallback_activity_ids) > 1 else activity_a
                lunch_id = fallback_restaurant_ids[0] if fallback_restaurant_ids else ""
                dinner_id = fallback_restaurant_ids[1] if len(fallback_restaurant_ids) > 1 else lunch_id
                return [
                    [
                        {"step_id": "step_1", "phase": "morning", "poi_type": "activity", "poi_id": activity_a, "label": "上午活动", "duration_minutes": 90},
                        {"step_id": "step_2", "phase": "lunch", "poi_type": "restaurant", "poi_id": lunch_id, "label": "午餐", "duration_minutes": 75},
                        {"step_id": "step_3", "phase": "afternoon", "poi_type": "activity", "poi_id": activity_b, "label": "下午活动", "duration_minutes": 90},
                        {"step_id": "step_4", "phase": "dinner", "poi_type": "restaurant", "poi_id": dinner_id, "label": "晚餐", "duration_minutes": 75},
                    ],
                    [
                        {"step_id": "step_1", "phase": "morning", "poi_type": "activity", "poi_id": activity_b or activity_a, "label": "上午活动", "duration_minutes": 90},
                        {"step_id": "step_2", "phase": "lunch", "poi_type": "restaurant", "poi_id": dinner_id or lunch_id, "label": "午餐", "duration_minutes": 75},
                        {"step_id": "step_3", "phase": "afternoon", "poi_type": "activity", "poi_id": activity_a, "label": "下午活动", "duration_minutes": 90},
                        {"step_id": "step_4", "phase": "dinner", "poi_type": "restaurant", "poi_id": lunch_id, "label": "晚餐", "duration_minutes": 75},
                    ],
                    [
                        {"step_id": "step_1", "phase": "morning", "poi_type": "activity", "poi_id": activity_a, "label": "上午活动", "duration_minutes": 90},
                        {"step_id": "step_2", "phase": "lunch", "poi_type": "restaurant", "poi_id": lunch_id, "label": "午餐", "duration_minutes": 75},
                        {"step_id": "step_3", "phase": "afternoon", "poi_type": "activity", "poi_id": "", "label": "下午活动", "duration_minutes": 90},
                        {"step_id": "step_4", "phase": "dinner", "poi_type": "restaurant", "poi_id": dinner_id, "label": "晚餐", "duration_minutes": 75},
                    ],
                ]
            activity_id = fallback_activity_ids[0] if fallback_activity_ids else ""
            restaurant_id = fallback_restaurant_ids[0] if fallback_restaurant_ids else ""
            meal_phase = "dinner" if "晚餐" in (state.get("intent", {}).get("raw_query", "") if isinstance(state.get("intent"), dict) else "") else "meal"
            activity_phase = "afternoon" if daypart == "下午" else "evening" if daypart == "晚上" else "flex"
            return [
                [
                    {"step_id": "step_1", "phase": activity_phase, "poi_type": "activity", "poi_id": activity_id, "label": "主活动", "duration_minutes": 90},
                    {"step_id": "step_2", "phase": meal_phase, "poi_type": "restaurant", "poi_id": restaurant_id, "label": "用餐", "duration_minutes": 75},
                ],
                [
                    {"step_id": "step_1", "phase": activity_phase, "poi_type": "activity", "poi_id": fallback_activity_ids[1] if len(fallback_activity_ids) > 1 else activity_id, "label": "主活动", "duration_minutes": 90},
                    {"step_id": "step_2", "phase": meal_phase, "poi_type": "restaurant", "poi_id": fallback_restaurant_ids[1] if len(fallback_restaurant_ids) > 1 else restaurant_id, "label": "用餐", "duration_minutes": 75},
                ],
                [
                    {"step_id": "step_1", "phase": activity_phase, "poi_type": "activity", "poi_id": activity_id, "label": "主活动", "duration_minutes": 90},
                    {"step_id": "step_2", "phase": meal_phase, "poi_type": "restaurant", "poi_id": restaurant_id, "label": "用餐", "duration_minutes": 75},
                ],
            ]

        def _legacy_steps(draft: CandidatePlanDraftItem) -> list[dict]:
            activity_id = draft.activity_id if isinstance(draft.activity_id, str) else ""
            restaurant_ids = draft.restaurant_ids if isinstance(draft.restaurant_ids, list) else []
            restaurant_ids = [item for item in restaurant_ids if isinstance(item, str) and item]
            if not activity_id and not restaurant_ids:
                return []
            if daypart == "全天":
                lunch_id = restaurant_ids[0] if restaurant_ids else ""
                dinner_id = restaurant_ids[1] if len(restaurant_ids) > 1 else lunch_id
                return [
                    {"step_id": "step_1", "phase": "morning", "poi_type": "activity", "poi_id": activity_id, "label": "上午活动", "duration_minutes": 90},
                    {"step_id": "step_2", "phase": "lunch", "poi_type": "restaurant", "poi_id": lunch_id, "label": "午餐", "duration_minutes": 75},
                    {"step_id": "step_3", "phase": "afternoon", "poi_type": "activity", "poi_id": activity_id, "label": "下午活动", "duration_minutes": 90},
                    {"step_id": "step_4", "phase": "dinner", "poi_type": "restaurant", "poi_id": dinner_id, "label": "晚餐", "duration_minutes": 75},
                ]
            restaurant_id = restaurant_ids[0] if restaurant_ids else ""
            meal_phase = "dinner" if "晚餐" in (state.get("intent", {}).get("raw_query", "") if isinstance(state.get("intent"), dict) else "") else "meal"
            activity_phase = "afternoon" if daypart == "下午" else "evening" if daypart == "晚上" else "flex"
            return [
                {"step_id": "step_1", "phase": activity_phase, "poi_type": "activity", "poi_id": activity_id, "label": "主活动", "duration_minutes": 90},
                {"step_id": "step_2", "phase": meal_phase, "poi_type": "restaurant", "poi_id": restaurant_id, "label": "用餐", "duration_minutes": 75},
            ]

        def _normalize_steps(raw_steps: list[dict]) -> list[dict]:
            normalized_steps: list[dict] = []
            for index, step_item in enumerate(raw_steps, start=1):
                if not isinstance(step_item, dict):
                    continue
                poi_type = step_item.get("poi_type")
                if poi_type not in {"activity", "restaurant"}:
                    continue
                poi_id = step_item.get("poi_id") if isinstance(step_item.get("poi_id"), str) else ""
                phase = step_item.get("phase") if isinstance(step_item.get("phase"), str) and step_item.get("phase").strip() else "flex"
                label = step_item.get("label") if isinstance(step_item.get("label"), str) and step_item.get("label").strip() else ("活动" if poi_type == "activity" else "用餐")
                duration_minutes = step_item.get("duration_minutes")
                if isinstance(duration_minutes, bool) or not isinstance(duration_minutes, int) or duration_minutes <= 0:
                    duration_minutes = 90 if poi_type == "activity" else 75
                step_id = step_item.get("step_id") if isinstance(step_item.get("step_id"), str) and step_item.get("step_id").strip() else f"step_{index}"
                normalized_steps.append(
                    {
                        "step_id": step_id,
                        "phase": phase.strip(),
                        "poi_type": poi_type,
                        "poi_id": poi_id,
                        "label": label.strip(),
                        "duration_minutes": duration_minutes,
                    }
                )
            return normalized_steps

        def _phase_to_timeline_type(phase: str, poi_type: str) -> str:
            if poi_type == "activity":
                if phase == "morning":
                    return "activity_morning"
                if phase == "afternoon":
                    return "activity_afternoon"
                return "activity"
            if phase == "lunch":
                return "lunch"
            if phase == "dinner":
                return "dinner"
            return "restaurant"

        def _phase_to_time_label(phase: str, poi_type: str) -> str:
            if phase == "morning":
                return "上午"
            if phase == "noon":
                return "中午"
            if phase == "afternoon":
                return "下午"
            if phase == "evening":
                return "晚上"
            if phase == "lunch":
                return "午餐"
            if phase == "dinner":
                return "晚餐"
            if poi_type == "activity":
                return time_phrase
            return "用餐"

        def _build_candidate_from_steps(candidate_id: str, title: str, raw_steps: list[dict], reasoning: list[str]) -> dict:
            normalized_steps = _normalize_steps(raw_steps)
            timeline: list[dict] = []
            activities_in_order: list[dict] = []
            restaurants_in_order: list[dict] = []
            dinner_restaurant: dict = {}
            for step in normalized_steps:
                poi_type = step.get("poi_type") if isinstance(step.get("poi_type"), str) else ""
                poi_id = step.get("poi_id") if isinstance(step.get("poi_id"), str) else ""
                phase = step.get("phase") if isinstance(step.get("phase"), str) else "flex"
                if poi_type == "activity":
                    target = activities_by_id.get(poi_id, {})
                    if target:
                        activities_in_order.append(target)
                else:
                    target = restaurants_by_id.get(poi_id, {})
                    if target:
                        restaurants_in_order.append(target)
                        if phase == "dinner":
                            dinner_restaurant = target

                target_name = target.get("name") if isinstance(target, dict) else ""
                timeline.append(
                    _timeline_item(
                        time_label=_phase_to_time_label(phase, poi_type),
                        item_name=target_name or "待确认",
                        item_type=_phase_to_timeline_type(phase, poi_type),
                        ref_type=poi_type,
                        ref_id=poi_id,
                    )
                )

            primary_activity = activities_in_order[0] if activities_in_order else {}
            primary_restaurant = dinner_restaurant or (restaurants_in_order[0] if restaurants_in_order else {})
            secondary_activity = activities_in_order[1] if len(activities_in_order) > 1 else {}
            candidate = {
                "id": candidate_id,
                "title": title,
                "steps": normalized_steps,
                "timeline": timeline,
                "activity": primary_activity,
                "secondary_activity": secondary_activity,
                "activities": activities_in_order,
                "restaurant": primary_restaurant,
                "restaurants": restaurants_in_order,
                "reasoning": [item for item in reasoning if isinstance(item, str)][:3],
            }
            return candidate

        activities_by_id = {
            item.get("id"): dict(item)
            for item in activities
            if isinstance(item, dict) and isinstance(item.get("id"), str) and item.get("id")
        }
        restaurants_by_id = {
            item.get("id"): dict(item)
            for item in restaurants
            if isinstance(item, dict) and isinstance(item.get("id"), str) and item.get("id")
        }
        user_input = state.get("user_input", "")
        if not isinstance(user_input, str):
            user_input = ""
        conversation_turns = state.get("conversation_turns", [])
        if not isinstance(conversation_turns, list):
            conversation_turns = []

        prompt = f"""
你是本地生活规划助手。请基于真实候选池生成 3 个候选方案。
要求：
1. 只能从提供的 activity_id 和 restaurant_id 中选择，不能编造新地点。
2. 只输出 JSON。
3. 格式必须是:
{{
  "candidates": [
    {{
      "id": "plan_1",
      "title": "...",
      "steps": [
        {{"step_id": "step_1", "phase": "afternoon", "poi_type": "activity", "poi_id": "...", "label": "主活动", "duration_minutes": 90}},
        {{"step_id": "step_2", "phase": "dinner", "poi_type": "restaurant", "poi_id": "...", "label": "晚餐", "duration_minutes": 75}}
      ],
      "reasoning": ["...", "..."]
    }}
  ]
}}
4. steps 必须按真实顺序输出，不要编造没有给出的 poi_id。
5. 如果用户明确提到了某个时段的餐饮偏好，例如“晚上吃火锅”，应优先把该类型安排在更合适的 phase。
6. 如果要安排多餐，餐饮类型尽量不要完全重复；如果要安排多个活动，也尽量不要完全重复。
7. phase 可以使用 morning / noon / afternoon / evening / lunch / dinner / flex。
8. reasoning 只要 2-3 条简短理由。
9. 当 hard_constraints.daypart 为“全天”时，每个候选的 steps 必须包含 morning 活动、lunch 餐饮、afternoon 活动、dinner 餐饮；餐厅不足时晚餐可以复用午餐餐厅。

输入：
- request_type: {request_type}
- plan_mode: {plan_mode}
- hard_constraints: {hard_constraints}
- soft_preferences: {soft_preferences}
- query_constraints: {query_constraints}
- user_input: {user_input}
- conversation_turns: {conversation_turns}
- raw_query: {state.get("intent", {}).get("raw_query", "") if isinstance(state.get("intent"), dict) else ""}
- activity_candidates: {list(activities_by_id.values())}
- restaurant_candidates: {list(restaurants_by_id.values())}
"""

        normalized_candidates = []
        try:
            response = chat_model.invoke([HumanMessage(content=prompt)])
            content = getattr(response, "content", "") or ""
            json_text = _extract_json_object(content)
            payload = json.loads(json_text) if json_text else {}
            envelope = CandidatePlanDraftEnvelope.model_validate(payload)
            drafts = envelope.candidates
        except Exception:
            drafts = []

        if not drafts:
            fallback_activity_ids = list(activities_by_id.keys())
            fallback_restaurant_ids = list(restaurants_by_id.keys())
            fallback_step_groups = _fallback_steps(fallback_activity_ids, fallback_restaurant_ids)
            drafts = [
                CandidatePlanDraftItem(
                    id="plan_1",
                    title="主推荐方案",
                    steps=fallback_step_groups[0],
                    reasoning=["优先使用当前候选池中的高相关项"],
                ),
                CandidatePlanDraftItem(
                    id="plan_2",
                    title="备选近场方案",
                    steps=fallback_step_groups[1],
                    reasoning=["保留近场与低切换成本作为备选"],
                ),
                CandidatePlanDraftItem(
                    id="plan_3",
                    title="备选轻量方案",
                    steps=fallback_step_groups[2],
                    reasoning=["保留更轻量的候选结构"],
                ),
            ]

        fallback_activity_ids = list(activities_by_id.keys())
        fallback_restaurant_ids = list(restaurants_by_id.keys())
        fallback_step_groups = _fallback_steps(fallback_activity_ids, fallback_restaurant_ids)

        for index, draft in enumerate(drafts[:3], start=1):
            draft_steps = draft.steps if isinstance(draft.steps, list) else []
            if not draft_steps:
                draft_steps = _legacy_steps(draft)
            if not _steps_cover_daypart(draft_steps, daypart):
                draft_steps = fallback_step_groups[index - 1] if index - 1 < len(fallback_step_groups) else fallback_step_groups[0]
            normalized_candidates.append(
                _build_candidate_from_steps(
                    draft.id if isinstance(draft.id, str) and draft.id else f"plan_{index}",
                    draft.title if isinstance(draft.title, str) and draft.title else f"候选方案{index}",
                    draft_steps,
                    draft.reasoning if isinstance(draft.reasoning, list) else [],
                )
            )

        while len(normalized_candidates) < 3:
            normalized_candidates.append(
                {
                    "id": f"plan_{len(normalized_candidates) + 1}",
                    "title": "",
                    "steps": [],
                    "timeline": [],
                    "activity": {},
                    "secondary_activity": {},
                    "activities": [],
                    "restaurant": {},
                    "restaurants": [],
                    "reasoning": [],
                }
            )

        print(
            "[Candidate Planning Node] "
            f"plan_1_activity={normalized_candidates[0].get('activity', {}).get('name')!r}, "
            f"plan_1_restaurant={normalized_candidates[0].get('restaurant', {}).get('name')!r}, "
            f"plan_1_restaurants_n={len(normalized_candidates[0].get('restaurants', []))}"
        )

        return {
            "candidate_plans": {
                "request_type": request_type,
                "plan_mode": plan_mode,
                "candidates": normalized_candidates,
            }
        }
    except Exception as exc:
        print(f"[Candidate Planning Node][WARN] 节点异常，返回空骨架: {exc}")
        update = _append_error(state, f"Candidate Planning node failed: {exc}")
        update["candidate_plans"] = {
            "request_type": "generic_local_plan",
            "plan_mode": "activity_plus_meal",
            "candidates": [
                {"id": "plan_1", "title": "", "steps": [], "timeline": [], "activity": {}, "restaurant": {}, "reasoning": []},
                {"id": "plan_2", "title": "", "steps": [], "timeline": [], "activity": {}, "restaurant": {}, "reasoning": []},
                {"id": "plan_3", "title": "", "steps": [], "timeline": [], "activity": {}, "restaurant": {}, "reasoning": []},
            ],
        }
        return update


def rule_validation_node(state: AgentState) -> AgentState:
    """新架构规则校验节点骨架：逐个检查 candidate_plans，并输出合法/非法结果。"""
    print("[Rule Validation Node] 校验候选计划骨架...")
    try:
        candidate_plans = state.get("candidate_plans")
        if not isinstance(candidate_plans, dict):
            candidate_plans = {}

        constraint_build = state.get("constraint_build")
        if not isinstance(constraint_build, dict):
            constraint_build = {}

        fact_gathering_result = state.get("fact_gathering_result")
        if not isinstance(fact_gathering_result, dict):
            fact_gathering_result = {}

        validation_profile = constraint_build.get("validation_profile")
        if not isinstance(validation_profile, dict):
            validation_profile = {}
        hard_constraints = constraint_build.get("hard_constraints")
        if not isinstance(hard_constraints, dict):
            hard_constraints = {}
        daypart = hard_constraints.get("daypart") if isinstance(hard_constraints.get("daypart"), str) else ""

        candidates = candidate_plans.get("candidates")
        if not isinstance(candidates, list):
            candidates = []

        valid_plans = []
        invalid_plans = []

        weather = fact_gathering_result.get("weather") if isinstance(fact_gathering_result.get("weather"), dict) else {}
        weather_risk = weather.get("risk_level") or weather.get("risk") or ""

        for item in candidates:
            if not isinstance(item, dict):
                continue

            candidate_id = item.get("id") if isinstance(item.get("id"), str) else "unknown_plan"
            activity = item.get("activity") if isinstance(item.get("activity"), dict) else {}
            restaurant = item.get("restaurant") if isinstance(item.get("restaurant"), dict) else {}

            violations = []
            repair_instructions = []
            steps = item.get("steps") if isinstance(item.get("steps"), list) else []

            if not _steps_cover_daypart(steps, daypart):
                violations.append("daypart_coverage_conflict")
                repair_instructions.append(
                    {"type": "regenerate_steps", "constraint": "full_day_requires_morning_lunch_afternoon_dinner"}
                )

            if validation_profile.get("check_weather_compatibility") is True:
                if weather_risk in {"High", "high"} and activity.get("type") == "outdoor":
                    violations.append("weather_outdoor_conflict")
                    repair_instructions.append(
                        {"type": "replace_activity", "constraint": "indoor_only"}
                    )

            if validation_profile.get("check_party_fit") is True:
                if not restaurant:
                    violations.append("restaurant_missing")
                    repair_instructions.append(
                        {"type": "replace_restaurant", "constraint": "party_fit_required"}
                    )

            if violations:
                invalid_plans.append(
                    {
                        "candidate_id": candidate_id,
                        "violations": violations,
                        "repair_instructions": repair_instructions,
                    }
                )
            else:
                valid_plans.append(item)

        return {
            "rule_validation_result": {
                "request_type": candidate_plans.get("request_type", "generic_local_plan"),
                "plan_mode": candidate_plans.get("plan_mode", "activity_plus_meal"),
                "valid_plans": valid_plans,
                "invalid_plans": invalid_plans,
            }
        }
    except Exception as exc:
        print(f"[Rule Validation Node][WARN] 节点异常，返回空骨架: {exc}")
        update = _append_error(state, f"Rule Validation node failed: {exc}")
        update["rule_validation_result"] = {
            "request_type": "generic_local_plan",
            "plan_mode": "activity_plus_meal",
            "valid_plans": [],
            "invalid_plans": [],
        }
        return update


def repair_loop_node(state: AgentState) -> AgentState:
    """新架构修复回环节点骨架：把非法候选转成下一轮规划修正输入。"""
    print("[Repair Loop Node] 生成修复回环输入骨架...")
    try:
        rule_validation_result = state.get("rule_validation_result")
        if not isinstance(rule_validation_result, dict):
            rule_validation_result = {}

        constraint_build = state.get("constraint_build")
        if not isinstance(constraint_build, dict):
            constraint_build = {}

        invalid_plans = rule_validation_result.get("invalid_plans")
        if not isinstance(invalid_plans, list):
            invalid_plans = []

        repair_targets = []
        merged_repair_instructions = []
        merged_violations = []

        for item in invalid_plans:
            if not isinstance(item, dict):
                continue
            candidate_id = item.get("candidate_id") if isinstance(item.get("candidate_id"), str) else "unknown_plan"
            violations = item.get("violations") if isinstance(item.get("violations"), list) else []
            repair_instructions = item.get("repair_instructions") if isinstance(item.get("repair_instructions"), list) else []

            repair_targets.append(candidate_id)
            for violation in violations:
                if isinstance(violation, str) and violation not in merged_violations:
                    merged_violations.append(violation)
            for instruction in repair_instructions:
                if isinstance(instruction, dict):
                    merged_repair_instructions.append(instruction)

        hard_constraints = constraint_build.get("hard_constraints")
        if not isinstance(hard_constraints, dict):
            hard_constraints = {}
        soft_preferences = constraint_build.get("soft_preferences")
        if not isinstance(soft_preferences, dict):
            soft_preferences = {}
        context_memory = constraint_build.get("context_memory")
        if not isinstance(context_memory, dict):
            context_memory = {}

        next_hard_constraints = dict(hard_constraints)
        next_soft_preferences = dict(soft_preferences)
        next_context_memory = dict(context_memory)

        for instruction in merged_repair_instructions:
            instruction_type = instruction.get("type")
            constraint = instruction.get("constraint")
            if instruction_type == "replace_activity" and constraint == "indoor_only":
                next_hard_constraints["indoor_only"] = True
            if instruction_type == "replace_restaurant" and constraint == "party_fit_required":
                next_hard_constraints["restaurant_required"] = True
            if instruction_type == "reduce_eta":
                current_eta = next_hard_constraints.get("max_traffic_minutes", 40)
                if isinstance(current_eta, int) and current_eta > 20:
                    next_hard_constraints["max_traffic_minutes"] = current_eta - 10
            if instruction_type == "reduce_queue":
                current_queue = next_hard_constraints.get("max_queue_minutes", 30)
                if isinstance(current_queue, int) and current_queue > 10:
                    next_hard_constraints["max_queue_minutes"] = current_queue - 10

        previous_repair_round = context_memory.get("repair_round", 0)
        if not isinstance(previous_repair_round, int) or isinstance(previous_repair_round, bool):
            previous_repair_round = 0
        next_context_memory["repair_round"] = previous_repair_round + 1
        next_context_memory["repair_targets"] = repair_targets
        next_context_memory["repair_violations"] = merged_violations

        return {
            "repair_loop_result": {
                "repair_targets": repair_targets,
                "violations": merged_violations,
                "repair_instructions": merged_repair_instructions,
                "next_constraint_build": {
                    "request_type": constraint_build.get("request_type", "generic_local_plan"),
                    "plan_mode": constraint_build.get("plan_mode", "activity_plus_meal"),
                    "hard_constraints": next_hard_constraints,
                    "soft_preferences": next_soft_preferences,
                    "query_constraints": constraint_build.get("query_constraints", {}),
                    "validation_profile": constraint_build.get("validation_profile", {}),
                    "scoring_profile": constraint_build.get("scoring_profile", {}),
                    "context_memory": next_context_memory,
                },
            }
        }
    except Exception as exc:
        print(f"[Repair Loop Node][WARN] 节点异常，返回空骨架: {exc}")
        update = _append_error(state, f"Repair Loop node failed: {exc}")
        update["repair_loop_result"] = {
            "repair_targets": [],
            "violations": [],
            "repair_instructions": [],
            "next_constraint_build": {},
        }
        return update


def scoring_node(state: AgentState) -> AgentState:
    """新架构打分节点骨架：对合法候选计划做结构化打分。"""
    print("[Scoring Node] 对合法候选计划进行打分骨架...")
    try:
        rule_validation_result = state.get("rule_validation_result")
        if not isinstance(rule_validation_result, dict):
            rule_validation_result = {}

        constraint_build = state.get("constraint_build")
        if not isinstance(constraint_build, dict):
            constraint_build = {}

        fact_gathering_result = state.get("fact_gathering_result")
        if not isinstance(fact_gathering_result, dict):
            fact_gathering_result = {}

        valid_plans = rule_validation_result.get("valid_plans")
        if not isinstance(valid_plans, list):
            valid_plans = []

        scoring_profile = constraint_build.get("scoring_profile")
        if not isinstance(scoring_profile, dict):
            scoring_profile = {}

        weights = scoring_profile.get("weights")
        if not isinstance(weights, dict):
            weights = {
                "semantic_match": 0.30,
                "time_relaxation": 0.20,
                "weather_fit": 0.15,
                "distance_fit": 0.15,
                "queue_fit": 0.10,
                "review_quality": 0.10,
            }

        scored_candidates = []
        weather = fact_gathering_result.get("weather") if isinstance(fact_gathering_result.get("weather"), dict) else {}
        weather_risk = weather.get("risk_level") or weather.get("risk") or ""

        for item in valid_plans:
            if not isinstance(item, dict):
                continue

            candidate_id = item.get("id") if isinstance(item.get("id"), str) else "unknown_plan"
            activity = item.get("activity") if isinstance(item.get("activity"), dict) else {}
            restaurant = item.get("restaurant") if isinstance(item.get("restaurant"), dict) else {}

            semantic_match = 8.0 if activity or restaurant else 3.0
            time_relaxation = 7.0
            weather_fit = 9.0 if weather_risk not in {"High", "high"} or activity.get("type") == "indoor" else 4.0
            distance_fit = 8.0
            queue_fit = 7.0
            review_quality = 7.0

            final_score = (
                semantic_match * float(weights.get("semantic_match", 0.30)) * 10
                + time_relaxation * float(weights.get("time_relaxation", 0.20)) * 10
                + weather_fit * float(weights.get("weather_fit", 0.15)) * 10
                + distance_fit * float(weights.get("distance_fit", 0.15)) * 10
                + queue_fit * float(weights.get("queue_fit", 0.10)) * 10
                + review_quality * float(weights.get("review_quality", 0.10)) * 10
            )

            scored_candidates.append(
                {
                    "candidate_id": candidate_id,
                    "score_breakdown": {
                        "semantic_match": semantic_match,
                        "time_relaxation": time_relaxation,
                        "weather_fit": weather_fit,
                        "distance_fit": distance_fit,
                        "queue_fit": queue_fit,
                        "review_quality": review_quality,
                    },
                    "final_score": round(final_score, 2),
                }
            )

        scored_candidates.sort(key=lambda item: item.get("final_score", 0), reverse=True)

        return {
            "scoring_result": {
                "request_type": rule_validation_result.get("request_type", "generic_local_plan"),
                "plan_mode": rule_validation_result.get("plan_mode", "activity_plus_meal"),
                "weights": weights,
                "scored_candidates": scored_candidates,
            }
        }
    except Exception as exc:
        print(f"[Scoring Node][WARN] 节点异常，返回空骨架: {exc}")
        update = _append_error(state, f"Scoring node failed: {exc}")
        update["scoring_result"] = {
            "request_type": "generic_local_plan",
            "plan_mode": "activity_plus_meal",
            "weights": {},
            "scored_candidates": [],
        }
        return update


def final_plan_node(state: AgentState) -> AgentState:
    """新架构最终计划节点骨架：从评分结果中选出分最高的计划。"""
    print("[Final Plan Node] 选择分最高的候选计划...")
    try:
        scoring_result = state.get("scoring_result")
        if not isinstance(scoring_result, dict):
            scoring_result = {}

        rule_validation_result = state.get("rule_validation_result")
        if not isinstance(rule_validation_result, dict):
            rule_validation_result = {}

        scored_candidates = scoring_result.get("scored_candidates")
        if not isinstance(scored_candidates, list):
            scored_candidates = []

        valid_plans = rule_validation_result.get("valid_plans")
        if not isinstance(valid_plans, list):
            valid_plans = []

        best_candidate_score = None
        best_score = None
        for item in scored_candidates:
            if not isinstance(item, dict):
                continue
            score = item.get("final_score")
            if not isinstance(score, (int, float)):
                continue
            if best_score is None or score > best_score:
                best_candidate_score = item
                best_score = score

        best_candidate = {}
        best_candidate_id = ""
        if isinstance(best_candidate_score, dict):
            best_candidate_id = best_candidate_score.get("candidate_id") if isinstance(best_candidate_score.get("candidate_id"), str) else ""
            for item in valid_plans:
                if not isinstance(item, dict):
                    continue
                if item.get("id") == best_candidate_id:
                    best_candidate = dict(item)
                    break

        if not best_candidate and isinstance(best_candidate_score, dict):
            best_candidate = dict(best_candidate_score)

        print(
            f"[Final Plan Node] selected_candidate_id={best_candidate_id!r}, "
            f"has_activity={isinstance(best_candidate.get('activity'), dict) and bool(best_candidate.get('activity'))}, "
            f"has_restaurant={isinstance(best_candidate.get('restaurant'), dict) and bool(best_candidate.get('restaurant'))}, "
            f"final_score={best_score if best_score is not None else 0}"
        )

        return {
            "final_plan_result": {
                "selected_candidate": best_candidate or {},
                "selected_candidate_id": best_candidate_id,
                "final_score": best_score if best_score is not None else 0,
                "all_scored_candidates": scored_candidates,
            }
        }
    except Exception as exc:
        print(f"[Final Plan Node][WARN] 节点异常，返回空骨架: {exc}")
        update = _append_error(state, f"Final Plan node failed: {exc}")
        update["final_plan_result"] = {
            "selected_candidate": {},
            "selected_candidate_id": "",
            "final_score": 0,
            "all_scored_candidates": [],
        }
        return update


def _derive_plan_compat_from_state(state: AgentState) -> dict:
    """优先使用兼容 plan；缺失时从 candidates.primary 派生最小兼容视图。"""
    plan = state.get("plan")
    if isinstance(plan, dict) and plan:
        return plan

    candidates = state.get("candidates")
    primary = candidates.get("primary") if isinstance(candidates, dict) else None
    if not isinstance(primary, dict) or not primary:
        return {}

    plan_compat = dict(primary)
    activity = primary.get("activity")
    restaurant = primary.get("restaurant")
    plan_compat["activities"] = [activity] if isinstance(activity, dict) else []
    plan_compat["restaurant"] = restaurant if isinstance(restaurant, dict) else {}
    return plan_compat


def retrieval_node(state: AgentState) -> AgentState:
    """Retrieval Node：基于 intent 做 mock RAG 检索，写入 ``state.retrieval_context``。

    任意异常都被吞掉、写入 ``state.errors``，避免阻塞主链路。RetrievalAgent 自身
    设计为"任意失败返回空骨架"，因此正常路径不会抛；本 try 仅是兜底护栏。
    """
    print("[Retrieval Node] 开始 mock RAG 检索...")
    try:
        result = RetrievalAgent().retrieve(state.get("intent", {}))
        return {"retrieval_context": result}
    except Exception as exc:
        print(f"[Retrieval Node][WARN] 检索节点异常，跳过并记录错误: {exc}")
        return _append_error(state, f"Retrieval node failed: {exc}")


def constraint_collect_node(state: AgentState) -> AgentState:
    """Constraint Collect Node：把 intent + retrieval_context + replan_reason
    汇总为 ``state.constraints``，作为后续 6 个并行工具节点的统一输入。

    设计：
        * ConstraintAgent 是纯函数（不调用 LLM）。
        * 任意异常都被吞掉，写入 ``state.errors``，**不**写 ``constraints``，
          让下游节点以"缺少 constraints"路径继续兜底，不阻塞图执行。
    """
    print("[Constraint Collect Node] 汇总意图 / 检索 / 默认策略 / 重规划反馈...")
    try:
        result = ConstraintAgent().collect(
            state.get("intent", {}),
            state.get("retrieval_context", {}),
            state.get("replan_reason", ""),
            state.get("replan_reason_type", ""),
            state.get("runtime_origin_area", ""),
            state.get("runtime_origin_coordinates", ""),
        )
        constraints = result.get("constraints") if isinstance(result, dict) else {}
        constraint_build = result.get("constraint_build") if isinstance(result, dict) else {}
        normalized_time = state.get("normalized_time") if isinstance(state.get("normalized_time"), dict) else {}
        if normalized_time:
            constraints["date_label"] = normalized_time.get("normalized_date_label", constraints.get("date_label", ""))
            constraints["daypart"] = normalized_time.get("normalized_daypart", constraints.get("daypart", ""))
            constraints["time_phrase"] = normalized_time.get("normalized_time_phrase", constraints.get("time_phrase", ""))
            if not constraints.get("start_time") and normalized_time.get("base_start_minutes") is not None:
                base_start_minutes = normalized_time.get("base_start_minutes")
                if isinstance(base_start_minutes, int):
                    constraints["base_start_minutes"] = base_start_minutes
            constraint_build = dict(constraint_build) if isinstance(constraint_build, dict) else {}
            hard_constraints = constraint_build.get("hard_constraints") if isinstance(constraint_build.get("hard_constraints"), dict) else {}
            hard_constraints["date_label"] = constraints["date_label"]
            hard_constraints["daypart"] = constraints["daypart"]
            hard_constraints["time_phrase"] = constraints["time_phrase"]
            hard_constraints["base_start_minutes"] = normalized_time.get("base_start_minutes")
            if not hard_constraints.get("time_window") and constraints.get("time_window"):
                hard_constraints["time_window"] = constraints.get("time_window")
            constraint_build["hard_constraints"] = hard_constraints
            query_constraints = constraint_build.get("query_constraints") if isinstance(constraint_build.get("query_constraints"), dict) else {}
            query_constraints["time_window"] = constraints.get("time_window", query_constraints.get("time_window", ""))
            constraint_build["query_constraints"] = query_constraints
            context_memory = constraint_build.get("context_memory") if isinstance(constraint_build.get("context_memory"), dict) else {}
            defaults_applied = context_memory.get("defaults_applied")
            if not isinstance(defaults_applied, list):
                defaults_applied = []
            if normalized_time.get("normalized_date_label") and normalized_time.get("normalized_date_label") != state.get("intent", {}).get("time", {}).get("date_label"):
                marker = "date_label:normalized_by_time_node"
                if marker not in defaults_applied:
                    defaults_applied.append(marker)
            context_memory["defaults_applied"] = defaults_applied
            constraint_build["context_memory"] = context_memory
        print(
            f"[Constraint Collect Node][OK] scenario={constraints.get('scenario')}, "
            f"party={constraints.get('party')}, "
            f"origin_area={constraints.get('origin_area')}, "
            f"max_traffic_minutes={constraints.get('max_traffic_minutes')}, "
            f"max_queue_minutes={constraints.get('max_queue_minutes')}, "
            f"replan_hints_n={len(constraints.get('replan_hints', []))}, "
            f"retrieval_pois_n={len(constraints.get('retrieval_pois', []))}, "
            f"retrieval_notes_n={len(constraints.get('retrieval_notes', []))}"
        )
        return {
            "constraints": constraints,
            "constraint_build": constraint_build,
        }
    except Exception as exc:
        print(f"[Constraint Collect Node][WARN] 节点异常，跳过并记录错误: {exc}")
        return _append_error(state, f"Constraint Collect node failed: {exc}")


# ---------------------------------------------------------------------------
# T7：Constraint Collect 之后的 6 个并行工具节点
# ---------------------------------------------------------------------------
#
# 设计原则：
#   * 6 个节点之间互不读对方写入的字段，保持真正并行；
#   * 仅依赖 ``state.constraints``（缺失时使用安全默认）或 mock 数据全集；
#   * 任意异常都被吞掉，写入 ``state.errors``，不阻塞主链路；
#   * 每节点最多一两条 print，避免噪音。
#
# 这一阶段 PlanningAgent 仍然内部独立调工具（暂时桥接），下一任务 T8 会用
# plan_candidate_node 替换它，让 planning 节点直接消费这里写入的 6 个字段。


def _safe_constraints(state: AgentState) -> dict:
    constraints = state.get("constraints")
    return constraints if isinstance(constraints, dict) else {}


def _fact_query_constraints(state: AgentState) -> dict:
    """优先消费新 schema 中的 query_constraints，缺失时回退到旧 constraints。"""
    constraint_build = state.get("constraint_build")
    if isinstance(constraint_build, dict):
        query_constraints = constraint_build.get("query_constraints")
        if isinstance(query_constraints, dict) and query_constraints:
            merged = dict(_safe_constraints(state))
            merged.update(query_constraints)
            return merged
    return _safe_constraints(state)


def _has_required_time_fields(constraints: dict, node_name: str) -> bool:
    return len(missing_required_time_fields(constraints, node_name)) == 0


def _print_time_context(node_name: str, constraints: dict) -> None:
    print(
        f"[{node_name}] time="
        f"date_label={constraints.get('date_label')!r}, "
        f"daypart={constraints.get('daypart')!r}, "
        f"time_phrase={constraints.get('time_phrase')!r}, "
        f"time_window={constraints.get('time_window')!r}, "
        f"start_time={constraints.get('start_time')!r}, "
        f"duration_hours={constraints.get('duration_hours')!r}"
    )


def _infer_queue_time_slot(time_window: str) -> str:
    """time_window → estimate_restaurant_queue 所需 slot。

    规则：
        * `today_evening` / `weekend_evening` → `dinner`
        * `today_afternoon` / `weekend_afternoon` → `lunch`
        * 其它缺省回退 `lunch`
    """
    if isinstance(time_window, str) and time_window in {"today_evening", "weekend_evening"}:
        return "dinner"
    return "lunch"


def _infer_crowd_time_slot(time_window: str) -> str:
    """time_window → evaluate_crowd_risk 所需 slot。

    外部统一使用业务标签，内部映射到当前 mock 支持的 crowd slot。
    """
    mapping = {
        "today_afternoon": "weekend_morning",
        "weekend_afternoon": "weekend_morning",
        "today_evening": "weekday_evening",
        "weekend_evening": "weekend_evening",
    }
    if isinstance(time_window, str):
        return mapping.get(time_window, "weekend_morning")
    return "weekend_morning"


def _derive_weather_scenario_key(constraints: dict) -> str:
    explicit = constraints.get("weather_scenario")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()

    date_label = constraints.get("date_label")
    if not isinstance(date_label, str):
        date_label = ""
    daypart = constraints.get("daypart")
    if not isinstance(daypart, str):
        daypart = ""

    is_weekend = any(token in date_label for token in ("周末", "周六", "周日", "周天"))
    if daypart == "晚上":
        return "storm"
    if daypart == "下午" and is_weekend:
        return "sunny"
    return "default"


def _derive_traffic_depart_context(constraints: dict) -> str:
    date_label = constraints.get("date_label")
    if not isinstance(date_label, str):
        date_label = ""
    daypart = constraints.get("daypart")
    if not isinstance(daypart, str):
        daypart = ""
    if not date_label or not daypart:
        return ""
    return f"{date_label}:{daypart}"


def _parse_hhmm_to_minutes(text: str) -> int | None:
    if not isinstance(text, str) or ":" not in text:
        return None
    try:
        hour_text, minute_text = text.strip().split(":", 1)
        hour = int(hour_text)
        minute = int(minute_text)
    except (TypeError, ValueError):
        return None
    if hour < 0 or hour > 24 or minute < 0 or minute >= 60:
        return None
    return hour * 60 + minute


def _extract_time_ranges(text: str) -> list[tuple[int, int]]:
    if not isinstance(text, str) or not text.strip():
        return []
    ranges: list[tuple[int, int]] = []
    pattern = re.compile(r"(\d{1,2}:\d{2})\s*-\s*(\d{1,2}:\d{2})")
    for start_text, end_text in pattern.findall(text):
        start_minutes = _parse_hhmm_to_minutes(start_text)
        end_minutes = _parse_hhmm_to_minutes(end_text)
        if start_minutes is None or end_minutes is None:
            continue
        if end_minutes <= start_minutes:
            end_minutes += 24 * 60
        ranges.append((start_minutes, end_minutes))
    return ranges


def _daypart_window(daypart: str) -> tuple[int, int] | None:
    if daypart == "上午":
        return (9 * 60, 12 * 60)
    if daypart == "下午":
        return (12 * 60, 18 * 60)
    if daypart == "晚上":
        return (18 * 60, 22 * 60)
    if daypart == "全天":
        return (0, 24 * 60)
    return None


def _ranges_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return max(a_start, b_start) < min(a_end, b_end)


def _matches_daypart_by_time_windows(item: dict, daypart: str) -> bool:
    if daypart == "全天":
        return True

    window = _daypart_window(daypart)
    if window is None:
        return False
    window_start, window_end = window

    open_time = item.get("open_time")
    opentime2 = item.get("opentime2")
    open_window = f"{open_time or ''} {opentime2 or ''}".strip()
    time_ranges = _extract_time_ranges(open_window)
    for start_minutes, end_minutes in time_ranges:
        if _ranges_overlap(start_minutes, end_minutes, window_start, window_end):
            return True

    peak_hours = item.get("peak_hours")
    if not isinstance(peak_hours, list) or not peak_hours:
        return False
    for slot in peak_hours:
        for start_minutes, end_minutes in _extract_time_ranges(slot):
            if _ranges_overlap(start_minutes, end_minutes, window_start, window_end):
                return True
    return False


def _activity_matches_daypart(activity: dict, daypart: str) -> bool:
    return _matches_daypart_by_time_windows(activity, daypart)


def _restaurant_matches_daypart(restaurant: dict, daypart: str) -> bool:
    return _matches_daypart_by_time_windows(restaurant, daypart)


def _normalize_text_list(value) -> list[str]:
    if isinstance(value, str):
        value = value.strip()
        return [value] if value else []
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            if isinstance(item, str):
                text = item.strip()
                if text:
                    out.append(text)
        return out
    return []


def _poi_text(item: dict) -> str:
    tags = item.get("tags") or []
    tags_semantic = item.get("tags_semantic") or []
    parts = [
        item.get("name") or "",
        item.get("description") or "",
        item.get("location") or "",
    ]
    if isinstance(tags, list):
        parts.extend(str(tag) for tag in tags)
    if isinstance(tags_semantic, list):
        parts.extend(str(tag) for tag in tags_semantic)
    return " ".join(part for part in parts if isinstance(part, str)).lower()


def _matches_keywords(item: dict, keywords: list[str], *, require_all: bool = False) -> bool:
    if not keywords:
        return True
    haystack = _poi_text(item)
    normalized = [keyword.strip().lower() for keyword in keywords if isinstance(keyword, str) and keyword.strip()]
    if not normalized:
        return True
    if require_all:
        return all(keyword in haystack for keyword in normalized)
    return any(keyword in haystack for keyword in normalized)


def _contains_excluded_keywords(item: dict, exclude_keywords: list[str]) -> bool:
    if not exclude_keywords:
        return False
    haystack = _poi_text(item)
    normalized = [keyword.strip().lower() for keyword in exclude_keywords if isinstance(keyword, str) and keyword.strip()]
    return any(keyword in haystack for keyword in normalized)


def _activity_bucket_key_from_name(name: str) -> str:
    text = (name or "").lower()
    if any(token in text for token in ("商场", "购物中心", "步行街", "广场", "mall")):
        return "shopping"
    if any(token in text for token in ("公园", "绿道", "步道")):
        return "park_walk"
    if any(token in text for token in ("展览", "美术馆", "博物馆", "艺术馆")):
        return "exhibition"
    if any(token in text for token in ("亲子", "儿童", "乐园")):
        return "parent_child"
    if any(token in text for token in ("ktv", "剧本杀", "桌游", "轰趴")):
        return "indoor_entertainment"
    return "other"


def _rating_value(item: dict) -> float:
    value = item.get("rating")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return -1.0
    return -1.0


def _dedupe_activities_by_identity(items: list[dict]) -> list[dict]:
    seen: set[tuple[str, str]] = set()
    out: list[dict] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        name = item.get("name") if isinstance(item.get("name"), str) else ""
        item_id = item.get("id") if isinstance(item.get("id"), str) else ""
        key = (name.strip().lower(), item_id.strip().lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _shortlist_activities(items: list[dict], per_bucket_limit: int = 2) -> list[dict]:
    deduped = _dedupe_activities_by_identity(items)
    buckets: dict[str, list[dict]] = {}
    for item in deduped:
        bucket = _activity_bucket_key_from_name(item.get("name") if isinstance(item.get("name"), str) else "")
        buckets.setdefault(bucket, []).append(item)

    shortlisted: list[dict] = []
    for bucket_items in buckets.values():
        ranked = sorted(
            bucket_items,
            key=lambda item: (_rating_value(item), len((item.get("name") or ""))),
            reverse=True,
        )
        shortlisted.extend(ranked[:per_bucket_limit])

    return shortlisted


def _has_open_time_fields(item: dict) -> bool:
    if not isinstance(item, dict):
        return False
    open_time = item.get("open_time")
    opentime2 = item.get("opentime2")
    return (
        isinstance(open_time, str) and open_time.strip()
    ) or (
        isinstance(opentime2, str) and opentime2.strip()
    )


def _shortlist_with_detail_gate(
    items: list[dict],
    *,
    bucket_key_fn,
    api: MockToolAPI,
    daypart: str,
    daypart_match_fn,
    per_bucket_limit: int = 2,
    max_scan_per_bucket: int = 10,
    bucket_order: list[str] | None = None,
    bucket_key_from_item_fn=None,
    allow_unverified_fallback: bool = False,
    total_limit: int | None = None,
) -> list[dict]:
    buckets: dict[str, list[dict]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        if bucket_key_from_item_fn is not None:
            bucket = bucket_key_from_item_fn(item)
        else:
            bucket = bucket_key_fn(item.get("name") if isinstance(item.get("name"), str) else "")
        buckets.setdefault(bucket, []).append(dict(item))

    shortlisted: list[dict] = []
    ordered_bucket_keys: list[str] = []
    if bucket_order:
        for key in bucket_order:
            if key in buckets and key not in ordered_bucket_keys:
                ordered_bucket_keys.append(key)
    for key in buckets:
        if key not in ordered_bucket_keys:
            ordered_bucket_keys.append(key)

    accepted_by_bucket: dict[str, list[dict]] = {}
    for bucket_key in ordered_bucket_keys:
        bucket_items = buckets.get(bucket_key, [])
        ranked = sorted(
            bucket_items,
            key=lambda item: (_rating_value(item), len((item.get("name") or ""))),
            reverse=True,
        )
        accepted: list[dict] = []
        scanned = 0
        for candidate in ranked:
            if scanned >= max_scan_per_bucket or len(accepted) >= per_bucket_limit:
                break
            scanned += 1
            enriched_batch = api.enrich_poi_details([candidate])
            enriched = enriched_batch[0] if isinstance(enriched_batch, list) and enriched_batch else candidate
            if _has_open_time_fields(enriched) and daypart_match_fn(enriched, daypart):
                enriched["time_validation_status"] = "verified_open_time"
                accepted.append(enriched)
        if not accepted and allow_unverified_fallback:
            for candidate in ranked[:per_bucket_limit]:
                fallback_item = dict(candidate)
                fallback_item["time_validation_status"] = "unverified_open_time"
                accepted.append(fallback_item)
        accepted_by_bucket[bucket_key] = accepted

    if total_limit is None:
        for bucket_key in ordered_bucket_keys:
            shortlisted.extend(accepted_by_bucket.get(bucket_key, []))
        return shortlisted

    for offset in range(per_bucket_limit):
        for bucket_key in ordered_bucket_keys:
            bucket_items = accepted_by_bucket.get(bucket_key, [])
            if offset < len(bucket_items):
                shortlisted.append(bucket_items[offset])
                if len(shortlisted) >= total_limit:
                    return shortlisted

    return shortlisted


def _restaurant_bucket_key_from_name(name: str) -> str:
    text = (name or "").lower()
    if any(token in text for token in ("火锅", "串串", "hotpot")):
        return "hotpot"
    if any(token in text for token in ("轻食", "沙拉", "健康")):
        return "light_meal"
    if any(token in text for token in ("咖啡", "cafe", "coffee")):
        return "cafe"
    if any(token in text for token in ("烧烤", "烤肉", "bbq")):
        return "bbq"
    if any(token in text for token in ("日料", "寿司", "居酒屋")):
        return "japanese"
    if any(token in text for token in ("西餐", "牛排", "意面")):
        return "western"
    if any(token in text for token in ("简餐", "餐厅", "饭")):
        return "simple_meal"
    return "other"


def _dedupe_restaurants_by_identity(items: list[dict]) -> list[dict]:
    seen: set[tuple[str, str]] = set()
    out: list[dict] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        name = item.get("name") if isinstance(item.get("name"), str) else ""
        item_id = item.get("id") if isinstance(item.get("id"), str) else ""
        key = (name.strip().lower(), item_id.strip().lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _shortlist_restaurants(items: list[dict], per_bucket_limit: int = 2) -> list[dict]:
    deduped = _dedupe_restaurants_by_identity(items)
    buckets: dict[str, list[dict]] = {}
    for item in deduped:
        bucket = _restaurant_bucket_key_from_name(item.get("name") if isinstance(item.get("name"), str) else "")
        buckets.setdefault(bucket, []).append(item)

    shortlisted: list[dict] = []
    for bucket_items in buckets.values():
        ranked = sorted(
            bucket_items,
            key=lambda item: (_rating_value(item), len((item.get("name") or ""))),
            reverse=True,
        )
        shortlisted.extend(ranked[:per_bucket_limit])

    return shortlisted


def weather_check_node(state: AgentState) -> AgentState:
    """并行节点 1：查询天气，写 ``state.weather``。"""
    print("[Weather Check Node] 查询天气...")
    try:
        constraints = _fact_query_constraints(state)
        _print_time_context("Weather Check Node", constraints)
        if not _has_required_time_fields(constraints, "weather_check"):
            missing = ",".join(missing_required_time_fields(constraints, "weather_check"))
            update = _append_error(state, f"Weather Check node skipped: missing time fields [{missing}]")
            print(f"[Weather Check Node] skipped, missing_fields=[{missing}]")
            update["weather"] = {
                "target_id": "weather",
                "status": "unknown",
                "weather": "",
                "risk_level": "unknown",
                "advice": "",
            }
            return update

        scenario_key = _derive_weather_scenario_key(constraints)
        weather = MockToolAPI().get_weather(
            scenario_key,
            origin_area=constraints.get("origin_area") or "",
            runtime_origin_area=state.get("runtime_origin_area", "") or "",
        )
        if not isinstance(weather, dict):
            weather = {}
        weather["scenario_key_used"] = scenario_key
        weather["date_label_used"] = constraints.get("date_label")
        weather["daypart_used"] = constraints.get("daypart")
        print(
            f"[Weather Check Node] result scenario_key_used={scenario_key!r}, "
            f"source={weather.get('source')!r}, provider={weather.get('provider')!r}, "
            f"requested_city={weather.get('requested_city')!r}, "
            f"resolved_city={weather.get('resolved_city')!r}, "
            f"weather={weather.get('weather')!r}, "
            f"risk_level={weather.get('risk_level')!r}, advice={weather.get('advice')!r}"
        )
        return {"weather": weather}
    except Exception as exc:
        print(f"[Weather Check Node][WARN] 节点异常，跳过并记录错误: {exc}")
        return _append_error(state, f"Weather Check node failed: {exc}")


def activity_search_node(state: AgentState) -> AgentState:
    """并行节点 2：根据场景搜活动，写 ``state.activities``。"""
    print("[Activity Search Node] 搜索候选活动...")
    try:
        constraints = _fact_query_constraints(state)
        _print_time_context("Activity Search Node", constraints)
        if not _has_required_time_fields(constraints, "activity_search"):
            missing = ",".join(missing_required_time_fields(constraints, "activity_search"))
            update = _append_error(state, f"Activity Search node skipped: missing time fields [{missing}]")
            print(f"[Activity Search Node] skipped, missing_fields=[{missing}]")
            update["activities"] = []
            return update

        if constraints.get("need_activity") is False:
            print("[Activity Search Node] skipped, query_constraints.need_activity=False")
            return {"activities": []}

        scenario = constraints.get("scenario") or "family"
        daypart = constraints.get("daypart")
        if not isinstance(daypart, str):
            daypart = ""
        keywords_activity = _normalize_text_list(constraints.get("keywords_activity"))
        activity_search_keywords = _normalize_text_list(constraints.get("activity_search_keywords"))
        preferred_activity_tags = _normalize_text_list(constraints.get("preferred_activity_tags"))
        activities = MockToolAPI().search_activities(
            scenario,
            activity_keywords=activity_search_keywords or keywords_activity,
            origin_area=constraints.get("origin_area") or "",
            runtime_origin_area=state.get("runtime_origin_area", "") or "",
            runtime_origin_coordinates=state.get("runtime_origin_coordinates", "") or "",
            enrich_details=False,
        )
        if not isinstance(activities, list):
            activities = []

        child_friendly_required = constraints.get("child_friendly_required") is True
        child_friendly_preferred = constraints.get("child_friendly_preferred") is True
        normalized_activities: list[dict] = []
        for item in activities:
            if not isinstance(item, dict):
                continue
            normalized_activities.append(dict(item))

        if child_friendly_required or child_friendly_preferred:
            child_friendly_matches = [
                item for item in normalized_activities if item.get("child_friendly") is True
            ]
            if child_friendly_matches:
                normalized_activities = child_friendly_matches

        has_keyword_sourced_results = any(
            isinstance(item, dict) and (item.get("search_keyword_sources") or item.get("keyword_source"))
            for item in normalized_activities
        )
        if keywords_activity and not (activity_search_keywords and has_keyword_sourced_results):
            keyword_matched = [
                item for item in normalized_activities if _matches_keywords(item, keywords_activity)
            ]
            if keyword_matched:
                normalized_activities = keyword_matched
            elif activity_search_keywords:
                search_keyword_matched = [
                    item for item in normalized_activities if _matches_keywords(item, activity_search_keywords)
                ]
                normalized_activities = search_keyword_matched
            else:
                normalized_activities = []

        if preferred_activity_tags:
            tag_matched = [
                item for item in normalized_activities if _matches_keywords(item, preferred_activity_tags)
            ]
            if tag_matched:
                normalized_activities = tag_matched

        def _activity_search_keyword_bucket(item: dict) -> str:
            sources = item.get("search_keyword_sources")
            if isinstance(sources, list):
                for source in sources:
                    if isinstance(source, str) and source in activity_search_keywords:
                        return source
            source = item.get("keyword_source")
            if isinstance(source, str) and source:
                return source
            return _activity_bucket_key_from_name(item.get("name") if isinstance(item.get("name"), str) else "")

        matched = _shortlist_with_detail_gate(
            _dedupe_activities_by_identity(normalized_activities),
            bucket_key_fn=_activity_bucket_key_from_name,
            api=MockToolAPI(),
            daypart=daypart,
            daypart_match_fn=_activity_matches_daypart,
            per_bucket_limit=2,
            max_scan_per_bucket=10,
            bucket_order=activity_search_keywords,
            bucket_key_from_item_fn=_activity_search_keyword_bucket if activity_search_keywords else None,
            allow_unverified_fallback=bool(activity_search_keywords),
            total_limit=5 if activity_search_keywords else None,
        )
        if not matched:
            update = _append_error(state, f"Activity Search node found no activities with open_time/opentime2 for daypart [{daypart}] after detail enrichment")
            print(
                f"[Activity Search Node] no shortlisted activities with open_time/opentime2 for daypart={daypart!r} after detail enrichment, "
                f"origin_area={constraints.get('origin_area')!r}, "
                f"origin_coordinates={state.get('runtime_origin_coordinates', '')!r}"
            )
            update["activities"] = []
            return update

        for item in matched:
            item["daypart_used"] = daypart
        activity_source = matched[0].get("source") if matched else None
        activity_provider = matched[0].get("provider") if matched else None
        activity_requested_city = matched[0].get("requested_city") if matched else None
        activity_search_mode = matched[0].get("search_mode") if matched else None
        print(
            f"[Activity Search Node] matched_ids={[item.get('id') for item in matched]}, "
            f"daypart_used={daypart!r}, source={activity_source!r}, "
            f"provider={activity_provider!r}, requested_city={activity_requested_city!r}, "
            f"search_mode={activity_search_mode!r}, "
            f"keywords={keywords_activity!r}, search_keywords={activity_search_keywords!r}, preferred_tags={preferred_activity_tags!r}, "
            f"shortlisted_n={len(matched)}, matched_names={[item.get('name') for item in matched]}, "
            f"keyword_sources={[item.get('search_keyword_sources') or item.get('keyword_source') for item in matched]}"
        )
        return {"activities": matched}
    except Exception as exc:
        print(f"[Activity Search Node][WARN] 节点异常，跳过并记录错误: {exc}")
        return _append_error(state, f"Activity Search node failed: {exc}")


def restaurant_search_node(state: AgentState) -> AgentState:
    """并行节点 3：根据饮食偏好搜餐厅，写 ``state.restaurants``。"""
    print("[Restaurant Search Node] 搜索候选餐厅...")
    try:
        constraints = _fact_query_constraints(state)
        _print_time_context("Restaurant Search Node", constraints)
        if not _has_required_time_fields(constraints, "restaurant_search"):
            missing = ",".join(missing_required_time_fields(constraints, "restaurant_search"))
            update = _append_error(state, f"Restaurant Search node skipped: missing time fields [{missing}]")
            print(f"[Restaurant Search Node] skipped, missing_fields=[{missing}]")
            update["restaurants"] = []
            return update

        if constraints.get("need_restaurant") is False:
            print("[Restaurant Search Node] skipped, query_constraints.need_restaurant=False")
            return {"restaurants": []}

        query_keywords = _normalize_text_list(constraints.get("keywords_restaurant"))
        exclude_keywords = _normalize_text_list(constraints.get("exclude_keywords_restaurant"))
        diet_preference = query_keywords or constraints.get("diet_preference") or ""
        scenario = constraints.get("scenario") or ""
        if not isinstance(scenario, str):
            scenario = ""
        daypart = constraints.get("daypart")
        if not isinstance(daypart, str):
            daypart = ""
        api = MockToolAPI()
        if query_keywords:
            restaurants = []
            seen_ids: set[str] = set()
            for keyword in query_keywords:
                batch = api.search_restaurants(
                    [keyword],
                    origin_area=constraints.get("origin_area") or "",
                    runtime_origin_area=state.get("runtime_origin_area", "") or "",
                    runtime_origin_coordinates=state.get("runtime_origin_coordinates", "") or "",
                    enrich_details=False,
                )
                if not isinstance(batch, list):
                    continue
                for item in batch:
                    if not isinstance(item, dict):
                        continue
                    item_copy = dict(item)
                    item_copy["keyword_source"] = keyword
                    sources = item_copy.get("search_keyword_sources")
                    if not isinstance(sources, list):
                        sources = []
                    if keyword not in sources:
                        sources.append(keyword)
                    item_copy["search_keyword_sources"] = sources
                    item_id = item.get("id")
                    if isinstance(item_id, str) and item_id in seen_ids:
                        for existing in restaurants:
                            if isinstance(existing, dict) and existing.get("id") == item_id:
                                existing_sources = existing.get("search_keyword_sources")
                                if not isinstance(existing_sources, list):
                                    existing_sources = []
                                if keyword not in existing_sources:
                                    existing_sources.append(keyword)
                                existing["search_keyword_sources"] = existing_sources
                                break
                        continue
                    if isinstance(item_id, str):
                        seen_ids.add(item_id)
                    restaurants.append(item_copy)
        else:
            restaurants = api.search_restaurants(
                diet_preference,
                origin_area=constraints.get("origin_area") or "",
                runtime_origin_area=state.get("runtime_origin_area", "") or "",
                runtime_origin_coordinates=state.get("runtime_origin_coordinates", "") or "",
                enrich_details=False,
            )
        if not isinstance(restaurants, list):
            restaurants = []

        normalized_restaurants: list[dict] = []
        for item in restaurants:
            if not isinstance(item, dict):
                continue
            normalized_restaurants.append(dict(item))

        if query_keywords:
            keyword_matched = [
                item for item in normalized_restaurants if _matches_keywords(item, query_keywords)
            ]
            if keyword_matched:
                normalized_restaurants = keyword_matched

        if exclude_keywords:
            filtered = [
                item for item in normalized_restaurants
                if not _contains_excluded_keywords(item, exclude_keywords)
            ]
            if filtered:
                normalized_restaurants = filtered

        def _restaurant_search_keyword_bucket(item: dict) -> str:
            sources = item.get("search_keyword_sources")
            if isinstance(sources, list):
                for source in sources:
                    if isinstance(source, str) and source in query_keywords:
                        return source
            source = item.get("keyword_source")
            if isinstance(source, str) and source:
                return source
            return _restaurant_bucket_key_from_name(item.get("name") if isinstance(item.get("name"), str) else "")

        matched = _shortlist_with_detail_gate(
            _dedupe_restaurants_by_identity(normalized_restaurants),
            bucket_key_fn=_restaurant_bucket_key_from_name,
            api=api,
            daypart=daypart,
            daypart_match_fn=_restaurant_matches_daypart,
            per_bucket_limit=2,
            max_scan_per_bucket=10,
            bucket_order=query_keywords,
            bucket_key_from_item_fn=_restaurant_search_keyword_bucket if query_keywords else None,
            allow_unverified_fallback=bool(query_keywords),
            total_limit=5 if query_keywords else None,
        )
        if not matched:
            update = _append_error(state, f"Restaurant Search node found no restaurants with open_time/opentime2 for daypart [{daypart}] after detail enrichment")
            print(
                f"[Restaurant Search Node] no shortlisted restaurants with open_time/opentime2 for daypart={daypart!r} after detail enrichment, "
                f"origin_area={constraints.get('origin_area')!r}, "
                f"origin_coordinates={state.get('runtime_origin_coordinates', '')!r}"
            )
            update["restaurants"] = []
            return update

        def _scenario_rank(item: dict) -> int:
            tags = item.get("tags") or []
            tags_semantic = item.get("tags_semantic") or []
            merged = []
            if isinstance(tags, list):
                merged.extend(str(tag) for tag in tags)
            if isinstance(tags_semantic, list):
                merged.extend(str(tag) for tag in tags_semantic)
            text = " ".join(merged)
            if scenario == "friends" and any(token in text for token in ("聚会", "音乐", "氛围", "晚餐", "酒馆")):
                return 0
            if scenario == "family" and any(token in text for token in ("健康", "轻食", "有机", "简餐")):
                return 0
            return 1

        matched.sort(key=_scenario_rank)
        for item in matched:
            item["daypart_used"] = daypart
        restaurant_source = matched[0].get("source") if matched else None
        restaurant_provider = matched[0].get("provider") if matched else None
        restaurant_requested_city = matched[0].get("requested_city") if matched else None
        restaurant_search_mode = matched[0].get("search_mode") if matched else None
        print(
            f"[Restaurant Search Node] matched_ids={[item.get('id') for item in matched]}, "
            f"daypart_used={daypart!r}, source={restaurant_source!r}, "
            f"provider={restaurant_provider!r}, requested_city={restaurant_requested_city!r}, "
            f"search_mode={restaurant_search_mode!r}, "
            f"keywords={query_keywords or diet_preference!r}, excludes={exclude_keywords!r}, "
            f"shortlisted_n={len(matched)}, matched_names={[item.get('name') for item in matched]}, "
            f"keyword_sources={[item.get('search_keyword_sources') or item.get('keyword_source') for item in matched]}"
        )
        return {"restaurants": matched}
    except Exception as exc:
        print(f"[Restaurant Search Node][WARN] 节点异常，跳过并记录错误: {exc}")
        return _append_error(state, f"Restaurant Search node failed: {exc}")


def traffic_eta_node(state: AgentState) -> AgentState:
    """并行节点 4：批量查通勤 ETA，写 ``state.traffic``。

    为保持与其它并行节点真正独立（不读 state.activities / state.restaurants），
    直接遍历 mock_db 中的活动 (family+friends) 和餐厅全集 id 做 ETA 查询。
    """
    print("[Traffic ETA Node] 批量查询通勤 ETA...")
    try:
        constraints = _fact_query_constraints(state)
        _print_time_context("Traffic ETA Node", constraints)
        date_label = constraints.get("date_label")
        if not isinstance(date_label, str) or date_label.strip() != "今天":
            print("[Traffic ETA Node] skipped, only enabled for date_label='今天'")
            return {
                "traffic": {
                    "origin_area_used": constraints.get("origin_area") or "",
                    "origin_coordinates_used": state.get("runtime_origin_coordinates", "") or "",
                    "traffic_origin_used": "",
                    "depart_context_used": "",
                    "eta_by_target": {},
                    "enabled_for_today_only": True,
                }
            }
        if not _has_required_time_fields(constraints, "traffic_eta"):
            missing = ",".join(missing_required_time_fields(constraints, "traffic_eta"))
            update = _append_error(state, f"Traffic ETA node skipped: missing time fields [{missing}]")
            print(f"[Traffic ETA Node] skipped, missing_fields=[{missing}]")
            update["traffic"] = {
                "origin_area_used": constraints.get("origin_area") or "",
                "depart_context_used": "",
                "eta_by_target": {},
            }
            return update
        origin_area = constraints.get("origin_area") or "area_central"
        origin_coordinates = state.get("runtime_origin_coordinates", "") or ""
        traffic_origin = origin_coordinates if isinstance(origin_coordinates, str) and origin_coordinates.strip() else origin_area
        depart_context = _derive_traffic_depart_context(constraints)
        api = MockToolAPI()
        activity_ids: list[str] = []
        activities_by_scenario = api.db.get("activities", {}) or {}
        for scenario_key in ("family", "friends"):
            for item in activities_by_scenario.get(scenario_key, []) or []:
                if isinstance(item, dict) and item.get("id"):
                    activity_ids.append(item["id"])
        restaurant_ids = [
            item["id"]
            for item in (api.db.get("restaurants", []) or [])
            if isinstance(item, dict) and item.get("id")
        ]
        all_targets = activity_ids + restaurant_ids

        eta_by_target: dict[str, dict] = {}
        for target_id in all_targets:
            record = api.get_traffic_eta(traffic_origin, target_id, depart_context)
            eta_by_target[target_id] = {
                "eta_minutes": record.get("eta_minutes"),
                "congestion": record.get("congestion"),
                "fallback_hint": record.get("fallback_hint"),
                "depart_context_used": depart_context,
            }
        print(
            f"[Traffic ETA Node] origin_area_used={origin_area!r}, "
            f"origin_coordinates_used={origin_coordinates!r}, "
            f"traffic_origin_used={traffic_origin!r}, "
            f"depart_context_used={depart_context!r}, "
            f"target_count={len(eta_by_target)}"
        )
        return {
            "traffic": {
                "origin_area_used": origin_area,
                "origin_coordinates_used": origin_coordinates,
                "traffic_origin_used": traffic_origin,
                "depart_context_used": depart_context,
                "eta_by_target": eta_by_target,
                "enabled_for_today_only": True,
            }
        }
    except Exception as exc:
        print(f"[Traffic ETA Node][WARN] 节点异常，跳过并记录错误: {exc}")
        return _append_error(state, f"Traffic ETA node failed: {exc}")


def schedule_timing_node(state: AgentState) -> AgentState:
    """基于已选候选计划，为现有 timeline 补充具体时刻，不重写结构。"""
    print("[Schedule Timing Node] 计算计划链顺序通勤时间与时刻表...")
    try:
        final_plan_result = state.get("final_plan_result")
        if not isinstance(final_plan_result, dict):
            final_plan_result = {}
        selected_candidate = final_plan_result.get("selected_candidate")
        if not isinstance(selected_candidate, dict):
            selected_candidate = {}

        activity = selected_candidate.get("activity") if isinstance(selected_candidate.get("activity"), dict) else {}
        restaurant = selected_candidate.get("restaurant") if isinstance(selected_candidate.get("restaurant"), dict) else {}
        restaurants = selected_candidate.get("restaurants")
        if not isinstance(restaurants, list):
            restaurants = []
        raw_timeline = selected_candidate.get("timeline")
        if not isinstance(raw_timeline, list):
            raw_timeline = []
        constraints = state.get("constraints") if isinstance(state.get("constraints"), dict) else {}
        normalized_time = state.get("normalized_time") if isinstance(state.get("normalized_time"), dict) else {}
        date_label = constraints.get("date_label") if isinstance(constraints.get("date_label"), str) else ""
        daypart = constraints.get("daypart") if isinstance(constraints.get("daypart"), str) else ""
        time_phrase = constraints.get("time_phrase") if isinstance(constraints.get("time_phrase"), str) else ""
        raw_query = state.get("intent", {}).get("raw_query") if isinstance(state.get("intent"), dict) else ""
        origin_coordinates = state.get("runtime_origin_coordinates", "") or ""
        normalized_date_label = normalized_time.get("normalized_date_label") if isinstance(normalized_time.get("normalized_date_label"), str) else date_label
        now = datetime.now()
        now_minutes = now.hour * 60 + now.minute
        cutoff_minutes = 18 * 60 if daypart in {"上午", "下午", "全天"} else 21 * 60
        if normalized_date_label == "今天" and now_minutes > cutoff_minutes:
            normalized_date_label = "明天"
            if isinstance(time_phrase, str) and time_phrase:
                time_phrase = time_phrase.replace("今天", "明天", 1)

        api = MockToolAPI()

        def _detail_location(item: dict) -> str:
            coordinates = item.get("coordinates") if isinstance(item.get("coordinates"), str) else ""
            if coordinates.strip():
                return coordinates.strip()
            location = item.get("location") if isinstance(item.get("location"), str) else ""
            if location.strip():
                return location.strip()
            item_id = item.get("id")
            source = item.get("source")
            if not isinstance(item_id, str) or source != "mcp":
                return ""
            if item.get("detail_loaded") is True:
                return ""
            try:
                detail_item = api.enrich_poi_details([item])
                if isinstance(detail_item, list) and detail_item and isinstance(detail_item[0], dict):
                    detail = detail_item[0]
                    detail_location = detail.get("coordinates") if isinstance(detail.get("coordinates"), str) else ""
                    if detail_location.strip():
                        return detail_location.strip()
                    detail_location = detail.get("location") if isinstance(detail.get("location"), str) else ""
                    if detail_location.strip():
                        return detail_location.strip()
            except Exception:
                return ""
            return ""

        activity_by_id = {}
        if isinstance(activity.get("id"), str) and activity.get("id"):
            activity_by_id[activity["id"]] = activity
        restaurants_by_id = {
            item.get("id"): item
            for item in restaurants
            if isinstance(item, dict) and isinstance(item.get("id"), str) and item.get("id")
        }
        if isinstance(restaurant.get("id"), str) and restaurant.get("id") and restaurant["id"] not in restaurants_by_id:
            restaurants_by_id[restaurant["id"]] = restaurant

        segment_eta: dict[str, int] = {}
        amap = api._get_amap()

        def _distance_minutes(origin: str, destination: str) -> int | None:
            if not amap or not origin or not destination:
                return None
            try:
                result = amap.maps_distance(origin, destination, "1")
                if isinstance(result, dict):
                    results = result.get("results")
                    if isinstance(results, list) and results and isinstance(results[0], dict):
                        duration = results[0].get("duration")
                        if isinstance(duration, str) and duration.isdigit():
                            return max(1, int(duration) // 60)
                return None
            except Exception:
                return None

        def _minutes_to_hhmm(total_minutes: int) -> str:
            total_minutes = max(0, int(total_minutes))
            hour = total_minutes // 60
            minute = total_minutes % 60
            return f"{hour:02d}:{minute:02d}"

        def _default_activity_start_minutes(daypart_value: str) -> int:
            if daypart_value == "上午":
                return 9 * 60 + 30
            if daypart_value == "晚上":
                return 18 * 60 + 30
            if daypart_value == "全天":
                return 9 * 60 + 30
            return 14 * 60

        activity_duration_minutes = 90
        meal_duration_minutes = 75
        buffer_minutes = 15
        meal_floor_minutes = 0
        if isinstance(raw_query, str):
            if "晚餐" in raw_query or "晚饭" in raw_query:
                meal_floor_minutes = 18 * 60
            elif "午餐" in raw_query:
                meal_floor_minutes = 12 * 60

        def _item_floor_minutes(item_type: str) -> int:
            if item_type == "activity_morning":
                return 9 * 60 + 30
            if item_type == "activity_afternoon":
                return 14 * 60
            if item_type == "lunch":
                return max(12 * 60, meal_floor_minutes)
            if item_type == "dinner":
                return max(18 * 60, meal_floor_minutes)
            if item_type in {"restaurant", "meal"}:
                return meal_floor_minutes
            return _default_activity_start_minutes(daypart)

        def _item_duration_minutes(item_type: str) -> int:
            if item_type in {"activity", "activity_morning", "activity_afternoon", "indoor", "outdoor"}:
                return activity_duration_minutes
            if item_type in {"restaurant", "meal", "lunch", "dinner"}:
                return meal_duration_minutes
            return 60

        def _resolve_timeline_target(item: dict) -> dict:
            ref_type = item.get("ref_type")
            ref_id = item.get("ref_id")
            if ref_type == "activity" and isinstance(ref_id, str):
                return activity_by_id.get(ref_id, activity)
            if ref_type == "restaurant" and isinstance(ref_id, str):
                return restaurants_by_id.get(ref_id, restaurant)
            item_type = item.get("type")
            if isinstance(item_type, str) and item_type.startswith("activity"):
                return activity
            return restaurant

        def _resolve_item_location(item: dict) -> str:
            target = _resolve_timeline_target(item)
            return _detail_location(target) if target else ""

        if not raw_timeline:
            fallback_timeline = []
            activity_name = activity.get("name") or "待确认活动"
            restaurant_name = restaurant.get("name") or "待确认餐厅"
            if activity:
                fallback_timeline.append(
                    {
                        "time": time_phrase or "活动",
                        "item": activity_name,
                        "type": "activity",
                        "ref_type": "activity",
                        "ref_id": activity.get("id", ""),
                    }
                )
            if restaurant:
                fallback_timeline.append(
                    {
                        "time": "用餐",
                        "item": restaurant_name,
                        "type": "restaurant",
                        "ref_type": "restaurant",
                        "ref_id": restaurant.get("id", ""),
                    }
                )
            raw_timeline = fallback_timeline

        first_location = _resolve_item_location(raw_timeline[0]) if raw_timeline else ""
        first_eta_minutes = _distance_minutes(origin_coordinates, first_location) if first_location else None
        if first_eta_minutes is not None:
            segment_eta["origin_to_first_stop_minutes"] = first_eta_minutes

        base_start_minutes = _item_floor_minutes(raw_timeline[0].get("type", "")) if raw_timeline else _default_activity_start_minutes(daypart)
        if normalized_date_label == "今天":
            earliest_start_minutes = now_minutes + buffer_minutes + (first_eta_minutes or 0)
            base_start_minutes = max(base_start_minutes, earliest_start_minutes)
            if daypart in {"下午", "晚上"} and base_start_minutes > cutoff_minutes:
                normalized_date_label = "明天"
                if isinstance(time_phrase, str) and time_phrase:
                    time_phrase = time_phrase.replace("今天", "明天", 1)
                base_start_minutes = _item_floor_minutes(raw_timeline[0].get("type", "")) if raw_timeline else _default_activity_start_minutes(daypart)

        timeline = []
        cursor_minutes = base_start_minutes
        previous_location = ""
        previous_type = ""

        for index, raw_item in enumerate(raw_timeline):
            if not isinstance(raw_item, dict):
                continue

            item_copy = dict(raw_item)
            item_type = item_copy.get("type") if isinstance(item_copy.get("type"), str) else ""
            item_location = _resolve_item_location(item_copy)
            if index == 0:
                item_start_minutes = cursor_minutes
            else:
                travel_minutes = _distance_minutes(previous_location, item_location) if previous_location and item_location else None
                if travel_minutes is not None:
                    segment_eta[f"leg_{index}_minutes"] = travel_minutes
                item_start_minutes = cursor_minutes + (travel_minutes or 0) + buffer_minutes
                item_start_minutes = max(item_start_minutes, _item_floor_minutes(item_type))
                if previous_type in {"lunch", "dinner", "restaurant", "meal"} and item_type == "activity_afternoon":
                    item_start_minutes = max(item_start_minutes, 14 * 60)

            item_copy["time"] = _minutes_to_hhmm(item_start_minutes)
            timeline.append(item_copy)

            cursor_minutes = item_start_minutes + _item_duration_minutes(item_type)
            previous_location = item_location
            previous_type = item_type

        selected_candidate["timeline"] = timeline
        final_plan_result["selected_candidate"] = selected_candidate
        print(
            f"[Schedule Timing Node] date_label={date_label!r}, normalized_date_label={normalized_date_label!r}, "
            f"origin_to_first_stop_minutes={segment_eta.get('origin_to_first_stop_minutes')!r}, "
            f"timeline_items={len(timeline)}"
        )
        return {
            "final_plan_result": final_plan_result,
            "schedule_timing_result": {
                "segment_eta": segment_eta,
                "timeline": timeline,
                "normalized_date_label": normalized_date_label,
            },
        }
    except Exception as exc:
        print(f"[Schedule Timing Node][WARN] 节点异常，跳过并记录错误: {exc}")
        return _append_error(state, f"Schedule Timing node failed: {exc}")


def queue_check_node(state: AgentState) -> AgentState:
    """并行节点 5：批量查餐厅排队，写 ``state.queue``。"""
    print("[Queue Check Node] 批量查询餐厅排队...")
    try:
        constraints = _fact_query_constraints(state)
        time_window = constraints.get("time_window") or ""
        if not _has_required_time_fields(constraints, "queue_check"):
            missing = ",".join(missing_required_time_fields(constraints, "queue_check"))
            update = _append_error(state, f"Queue Check node skipped: missing time fields [{missing}]")
            update["queue"] = {"time_slot_used": "", "wait_by_restaurant": {}}
            return update
        time_slot = _infer_queue_time_slot(time_window)
        people_count = constraints.get("people_count")
        if not isinstance(people_count, int) or isinstance(people_count, bool) or people_count <= 0:
            people_count = 2

        api = MockToolAPI()
        restaurant_ids = [
            item["id"]
            for item in (api.db.get("restaurants", []) or [])
            if isinstance(item, dict) and item.get("id")
        ]

        wait_by_restaurant: dict[str, dict] = {}
        for rid in restaurant_ids:
            record = api.estimate_restaurant_queue(rid, time_slot, people_count)
            wait_by_restaurant[rid] = {
                "wait_minutes": record.get("wait_minutes"),
                "party_acceptable": record.get("party_acceptable"),
                "fallback_hint": record.get("fallback_hint"),
            }
        return {"queue": {"time_slot_used": time_slot, "wait_by_restaurant": wait_by_restaurant}}
    except Exception as exc:
        print(f"[Queue Check Node][WARN] 节点异常，跳过并记录错误: {exc}")
        return _append_error(state, f"Queue Check node failed: {exc}")


def crowd_risk_node(state: AgentState) -> AgentState:
    """并行节点 6：批量评估活动人流风险，写 ``state.crowd``。"""
    print("[Crowd Risk Node] 批量评估活动人流风险...")
    try:
        constraints = _fact_query_constraints(state)
        time_window = constraints.get("time_window") or ""
        if not _has_required_time_fields(constraints, "crowd_risk"):
            missing = ",".join(missing_required_time_fields(constraints, "crowd_risk"))
            update = _append_error(state, f"Crowd Risk node skipped: missing time fields [{missing}]")
            update["crowd"] = {"crowd_by_activity": {}}
            return update
        time_slot = _infer_crowd_time_slot(time_window)

        api = MockToolAPI()
        activity_ids: list[str] = []
        activities_by_scenario = api.db.get("activities", {}) or {}
        for scenario_key in ("family", "friends"):
            for item in activities_by_scenario.get(scenario_key, []) or []:
                if isinstance(item, dict) and item.get("id"):
                    activity_ids.append(item["id"])

        crowd_by_activity: dict[str, dict] = {}
        for aid in activity_ids:
            record = api.evaluate_crowd_risk(aid, time_slot)
            crowd_by_activity[aid] = {
                "risk_level": record.get("risk_level"),
                "fallback_hint": record.get("fallback_hint"),
            }
        return {"crowd": {"crowd_by_activity": crowd_by_activity}}
    except Exception as exc:
        print(f"[Crowd Risk Node][WARN] 节点异常，跳过并记录错误: {exc}")
        return _append_error(state, f"Crowd Risk node failed: {exc}")


def fact_gathering_node(state: AgentState) -> AgentState:
    """聚合事实采集层输出，同时保留旧字段兼容。"""
    query_constraints = {}
    constraint_build = state.get("constraint_build")
    if isinstance(constraint_build, dict):
        query_constraints = constraint_build.get("query_constraints") or {}
        if not isinstance(query_constraints, dict):
            query_constraints = {}

    result = {
        "query_constraints": query_constraints,
        "weather": state.get("weather") if isinstance(state.get("weather"), dict) else {},
        "activities": state.get("activities") if isinstance(state.get("activities"), list) else [],
        "restaurants": state.get("restaurants") if isinstance(state.get("restaurants"), list) else [],
        "traffic": state.get("traffic") if isinstance(state.get("traffic"), dict) else {},
        "queue": state.get("queue") if isinstance(state.get("queue"), dict) else {},
        "crowd": state.get("crowd") if isinstance(state.get("crowd"), dict) else {},
    }
    return {"fact_gathering_result": result}


def validate_plan_node(state: AgentState) -> AgentState:
    """Validate Plan Node：纯校验，不调 LLM。

    输入：``state.candidates.primary`` 与 ``state.constraints`` /
    ``state.traffic`` / ``state.queue`` / ``state.weather``。
    输出：``state.validation_result = {"passed": bool, "violations": [str],
    "suggested_fixes": [str]}``，**任何分支都必须写入** validation_result。

    校验项（命中即追加一条人类可读 violation 和一条对应 suggested_fix）：
        1. 若 primary.activity 非空：traffic.eta_by_target[id].eta_minutes
           ≤ constraints.max_traffic_minutes
        2. 若 primary.restaurant 非空：queue.wait_by_restaurant[id].wait_minutes
           ≤ constraints.max_queue_minutes
        3. weather.risk_level ∈ {"High","high"} 时 activity.type 必须为 "indoor"
        4. queue.wait_by_restaurant[id].party_acceptable 不能为 False
        * primary 为空 / 非 dict → 直接 passed=False，violations=["主方案为空"]

    异常时追加 errors，但 validation_result 仍以 passed=False 写入。
    """
    print("[Validate Plan Node] 校验主方案...")
    violations: list[str] = []
    suggested_fixes: list[str] = []
    try:
        candidates = state.get("candidates")
        primary = candidates.get("primary") if isinstance(candidates, dict) else None

        if not isinstance(primary, dict) or not primary:
            violations.append("主方案为空")
            suggested_fixes.append("重新生成候选方案或放宽约束")
        else:
            constraints = state.get("constraints") or {}
            constraints = constraints if isinstance(constraints, dict) else {}
            traffic = state.get("traffic") or {}
            traffic = traffic if isinstance(traffic, dict) else {}
            queue = state.get("queue") or {}
            queue = queue if isinstance(queue, dict) else {}
            weather = state.get("weather") or {}
            weather = weather if isinstance(weather, dict) else {}

            max_traffic = constraints.get("max_traffic_minutes")
            max_queue = constraints.get("max_queue_minutes")
            if not _has_required_time_fields(constraints, "validate_plan"):
                violations.append("时间语义未澄清完整，无法判断具体安排在哪一天的下午或晚上")
                suggested_fixes.append("先把日期/星期和下午或晚上都澄清清楚，再继续生成方案")

            def _is_num(v):
                return isinstance(v, (int, float)) and not isinstance(v, bool)

            activity = primary.get("activity")
            if isinstance(activity, dict) and activity:
                # 1) ETA 阈值
                aid = activity.get("id")
                eta_table = traffic.get("eta_by_target")
                eta_record = eta_table.get(aid) if isinstance(eta_table, dict) else None
                eta = eta_record.get("eta_minutes") if isinstance(eta_record, dict) else None
                if _is_num(eta) and _is_num(max_traffic) and eta > max_traffic:
                    violations.append(
                        f"首选活动 ETA {eta} 分钟超过阈值 {max_traffic} 分钟"
                    )
                    suggested_fixes.append("放宽 max_traffic_minutes 或更换更近活动")

                # 3) 天气 risk_level vs activity.type
                risk_level = weather.get("risk_level")
                if isinstance(risk_level, str) and risk_level in {"High", "high"}:
                    act_type = activity.get("type")
                    if act_type != "indoor":
                        violations.append(
                            f"天气风险等级 {risk_level} 与首选活动类型 {act_type!r} 冲突，应为 indoor"
                        )
                        suggested_fixes.append("更换 indoor 类型活动或调整时间窗口")

            restaurant = primary.get("restaurant")
            if isinstance(restaurant, dict) and restaurant:
                rid = restaurant.get("id")
                wait_table = queue.get("wait_by_restaurant")
                wait_record = wait_table.get(rid) if isinstance(wait_table, dict) else None
                if isinstance(wait_record, dict):
                    wait_minutes = wait_record.get("wait_minutes")
                    # 2) 排队阈值
                    if _is_num(wait_minutes) and _is_num(max_queue) and wait_minutes > max_queue:
                        violations.append(
                            f"首选餐厅排队 {wait_minutes} 分钟超过阈值 {max_queue} 分钟"
                        )
                        suggested_fixes.append("放宽 max_queue_minutes 或更换排队更短餐厅")
                    # 4) party_acceptable
                    if wait_record.get("party_acceptable") is False:
                        violations.append("首选餐厅不接受当前人数")
                        suggested_fixes.append("更换适合该人数的餐厅")

        passed = len(violations) == 0
        validation_result = {
            "passed": passed,
            "violations": violations,
            "suggested_fixes": suggested_fixes,
        }
        print(
            f"[Validate Plan Node][OK] passed={passed}, "
            f"violations_n={len(violations)}"
        )
        return {"validation_result": validation_result}
    except Exception as exc:
        print(f"[Validate Plan Node][WARN] 节点异常，记录错误并强制 passed=False: {exc}")
        errors = list(state.get("errors", []))
        errors.append(f"Validate Plan node failed: {exc}")
        if not violations:
            violations = [f"校验异常: {exc}"]
            suggested_fixes = ["请检查上游节点输出"]
        return {
            "errors": errors,
            "validation_result": {
                "passed": False,
                "violations": violations,
                "suggested_fixes": suggested_fixes,
            },
        }


def route_after_validate(state: AgentState) -> str:
    """Validate 之后的条件路由。

    * validation_result.passed 显式为 True → ``presentation``；
    * 其它（缺失 / 非 dict / passed 非 True） → ``replan``，由上游图把 ``replan``
      映射到合适的下游节点（T9 暂占位映射到 presentation，T10 改为真正的 replan 节点）。
    """
    result = state.get("validation_result")
    if isinstance(result, dict) and result.get("passed") is True:
        return "presentation"
    return "replan"


# ---------------------------------------------------------------------------
# T10：Replan Node + 路由
# ---------------------------------------------------------------------------
#
# 设计要点：
#   * 纯函数，不调 LLM。从 ``state.replan_reason`` 或 ``validation_result.violations``
#     取一段人类可读的失败原因，按关键字 → 约束变更映射收紧 constraints；
#   * 通过浅拷贝写新 dict / list，绝不 in-place 修改入参；
#   * ``replan_count`` 自增；命中上限（>= 3）时不再触发新一轮，写降级说明到
#     ``state.errors``，由路由把后续走向 ``final_message``（T15 占位 → presentation）。
#   * 节点输出**必须**把 ``replan_reason`` 重置为空串，避免下一轮 validate 失败时
#     旧 reason 累积污染；
#   * 异常时仅追加 errors，不修改 constraints / replan_count / replan_reason，
#     让上层评估到底是节点出错还是约束没法再收紧。

# 关键字 → 约束变更的安全默认：当 current 不是合法数值时回退此基线，
# 与 ConstraintAgent 默认 max_traffic_minutes / max_queue_minutes 保持一致。
_REPLAN_TRAFFIC_FALLBACK_BASE = 60
_REPLAN_QUEUE_FALLBACK_BASE = 60
_REPLAN_TRAFFIC_FLOOR = 20
_REPLAN_QUEUE_FLOOR = 10
_REPLAN_BUDGET = 3


def _is_pos_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _resolve_replan_reason(state: AgentState) -> str:
    """优先 state.replan_reason，否则把 validation_result.violations 拼成一段。"""
    reason = state.get("replan_reason")
    if isinstance(reason, str) and reason.strip():
        return reason
    result = state.get("validation_result")
    if isinstance(result, dict):
        violations = result.get("violations")
        if isinstance(violations, list) and violations:
            parts = [v for v in violations if isinstance(v, str) and v.strip()]
            if parts:
                return "; ".join(parts)
    return ""


def _normalize_replan_reason_type(state: AgentState) -> str:
    raw = state.get("replan_reason_type")
    if isinstance(raw, str) and raw in {
        "validation_failure",
        "user_feedback",
        "execution_core_change",
    }:
        return raw

    execution_result = state.get("execution_result")
    if isinstance(execution_result, dict) and execution_result.get("core_plan_changed") is True:
        return "execution_core_change"

    if state.get("user_confirmed") is False:
        return "user_feedback"

    return "validation_failure"


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in text for keyword in keywords)


def _append_unique_hint(hints: list[str], hint: str) -> None:
    if isinstance(hint, str) and hint and hint not in hints:
        hints.append(hint)


def replan_node(state: AgentState) -> AgentState:
    """Replan Node: map failure reasons into new constraints."""
    print("[Replan Node] 把失败原因 / 拒绝反馈映射成新约束...")
    try:
        prev_count = state.get("replan_count", 0)
        if not isinstance(prev_count, int) or isinstance(prev_count, bool):
            prev_count = 0
        new_count = prev_count + 1

        reason = _resolve_replan_reason(state)
        reason_type = _normalize_replan_reason_type(state)

        old_constraints = state.get("constraints")
        if not isinstance(old_constraints, dict):
            old_constraints = {}
        new_constraints = dict(old_constraints)

        old_hints = new_constraints.get("replan_hints")
        if not isinstance(old_hints, list):
            old_hints = []
        new_hints = list(old_hints)

        if reason_type == "execution_core_change":
            new_constraints["indoor_preferred"] = True
            cur = new_constraints.get("max_traffic_minutes")
            base = cur if _is_pos_num(cur) else _REPLAN_TRAFFIC_FALLBACK_BASE
            new_constraints["max_traffic_minutes"] = max(_REPLAN_TRAFFIC_FLOOR, int(base) - 5)
            _append_unique_hint(new_hints, "execution_core_change")

        if reason:
            if _contains_any(reason, ("ETA", "通勤", "太远", "too far", "路线太长")):
                cur = new_constraints.get("max_traffic_minutes")
                base = cur if _is_pos_num(cur) else _REPLAN_TRAFFIC_FALLBACK_BASE
                new_constraints["max_traffic_minutes"] = max(_REPLAN_TRAFFIC_FLOOR, int(base) - 10)
                if reason_type == "user_feedback":
                    _append_unique_hint(new_hints, "prefer_nearer_options")

            if _contains_any(reason, ("排队", "没位", "没位置", "wait", "等太久")):
                cur = new_constraints.get("max_queue_minutes")
                base = cur if _is_pos_num(cur) else _REPLAN_QUEUE_FALLBACK_BASE
                new_constraints["max_queue_minutes"] = max(_REPLAN_QUEUE_FLOOR, int(base) - 10)
                if reason_type == "user_feedback":
                    _append_unique_hint(new_hints, "prefer_shorter_queue")

            if _contains_any(reason, ("天气", "户外", "outdoor", "室内", "室外", "indoor")):
                new_constraints["indoor_preferred"] = True

            if reason_type == "user_feedback":
                if _contains_any(reason, ("轻松", "累", "不想太累", "relax")):
                    _append_unique_hint(new_hints, "prefer_relaxed_schedule")
                if _contains_any(reason, ("拍照", "好看", "氛围", "photo")):
                    _append_unique_hint(new_hints, "prefer_photo_friendly")

            _append_unique_hint(new_hints, reason)

        new_constraints["replan_hints"] = new_hints

        update: AgentState = {
            "constraints": new_constraints,
            "replan_count": new_count,
            "replan_reason": "",
            "replan_reason_type": reason_type,
        }

        if new_count >= _REPLAN_BUDGET:
            errors = list(state.get("errors", []))
            errors.append(f"已尝试重规划 {new_count} 次，仍无法满足全部约束，进入兜底展示")
            update["errors"] = errors
            print(f"[Replan Node][WARN] replan_count={new_count} 命中上限 {_REPLAN_BUDGET}，进入兜底")
        else:
            print(
                f"[Replan Node][OK] replan_count={new_count}, "
                f"max_traffic_minutes={new_constraints.get('max_traffic_minutes')}, "
                f"max_queue_minutes={new_constraints.get('max_queue_minutes')}, "
                f"indoor_preferred={new_constraints.get('indoor_preferred')}, "
                f"replan_hints_n={len(new_hints)}"
            )

        return update
    except Exception as exc:
        print(f"[Replan Node][WARN] 节点异常，跳过并记录错误: {exc}")
        return _append_error(state, f"Replan node failed: {exc}")
def route_after_replan(state: AgentState) -> str:
    """Replan 之后的条件路由。

    * replan_count >= 3 → ``final_message``（T15 之前由 workflow 占位映射到
      presentation，避免阻塞图编译）；
    * 否则 → ``constraint_collect``，由 6 个并行节点 + plan_candidate + validate
      重新执行（这是 LangGraph 中复跑前置并行节点最直接的做法）。
    """
    count = state.get("replan_count", 0)
    if isinstance(count, bool) or not isinstance(count, int):
        count = 0
    if count >= _REPLAN_BUDGET:
        return "final_message"
    return "constraint_collect"



def presentation_node(state: AgentState) -> AgentState:
    try:
        final_plan_result = state.get("final_plan_result")
        if isinstance(final_plan_result, dict) and final_plan_result:
            selected_candidate = final_plan_result.get("selected_candidate")
            if not isinstance(selected_candidate, dict):
                selected_candidate = {}
            plan = dict(selected_candidate)
            activities = plan.get("activities")
            if isinstance(activities, list):
                plan["activities"] = [item for item in activities if isinstance(item, dict)]
            else:
                activity = plan.get("activity")
                if isinstance(activity, dict) and activity:
                    plan["activities"] = [activity]
                else:
                    plan["activities"] = []
            if not isinstance(plan.get("restaurant"), dict):
                plan["restaurant"] = {}
            schedule_timing_result = state.get("schedule_timing_result")
            if isinstance(schedule_timing_result, dict) and schedule_timing_result:
                plan["schedule_timing_result"] = schedule_timing_result
            plan["selected_candidate_id"] = final_plan_result.get("selected_candidate_id", "")
            plan["final_score"] = final_plan_result.get("final_score", 0)
            plan["all_scored_candidates"] = final_plan_result.get("all_scored_candidates", [])
        else:
            plan = _derive_plan_compat_from_state(state)

        display_text = PresentationAgent().generate_plan_display(
            plan,
            state.get("intent", {}),
        )
        print("\n" + "=" * 20 + " 方案详情 " + "=" * 20)
        print(display_text)
        print("=" * 50)
        return {"display_text": display_text, "plan": plan}
    except Exception as exc:
        return _append_error(state, f"Presentation node failed: {exc}")
def confirmation_node(state: AgentState) -> AgentState:
    if "user_confirmed" in state:
        return {"user_confirmed": state["user_confirmed"]}
    confirm = input("\n[系统提示] 确定按照此方案执行一键下单吗？(y/n): ")
    confirmed = confirm.lower() == "y"
    if confirmed:
        return {"user_confirmed": True}
    return {
        "user_confirmed": False,
        "replan_reason": "用户未确认当前方案",
        "replan_reason_type": "user_feedback",
    }


def execution_node(state: AgentState) -> AgentState:
    try:
        plan = _derive_plan_compat_from_state(state)
        result = ExecutionAgent().execute(plan)
        if result is None:
            result = {"status": "success", "message": "Execution completed"}
        return {"execution_result": result, "plan": plan}
    except Exception as exc:
        return _append_error(state, f"Execution node failed: {exc}")


def reject_node(state: AgentState) -> AgentState:
    return {
        "execution_result": {
            "status": "cancelled",
            "message": "用户未确认方案，未执行任何预约或下单动作。",
        }
    }


def final_message_node(state: AgentState) -> AgentState:
    """Final Message Node：把执行结果转成可转发的最终消息。"""
    execution_result = state.get("execution_result") or {}
    plan = _derive_plan_compat_from_state(state)
    intent = state.get("intent") or {}
    status = execution_result.get("status")
    scenario = intent.get("scenario", "family")

    activity_name = "待确认活动"
    restaurant_name = "待确认餐厅"
    if isinstance(plan, dict):
        activities = plan.get("activities") or []
        if isinstance(activities, list) and activities and isinstance(activities[0], dict):
            activity_name = activities[0].get("name") or activity_name
        restaurant = plan.get("restaurant") or {}
        if isinstance(restaurant, dict):
            restaurant_name = restaurant.get("name") or restaurant_name

    if status == "success":
        orders = execution_result.get("orders") or []
        order_lines = []
        if isinstance(orders, list):
            for item in orders:
                if not isinstance(item, dict):
                    continue
                order_type = item.get("type", "order")
                order_id = item.get("order_id", "N/A")
                order_lines.append(f"- {order_type}: {order_id}")
        order_text = "\n".join(order_lines) if order_lines else "- 已完成关键预约"
        message = (
            f"搞定了。{scenario} 场景下的本次安排已经完成。\n"
            f"活动：{activity_name}\n"
            f"餐厅：{restaurant_name}\n"
            f"执行结果：\n{order_text}\n"
            f"如果你要，我也可以继续帮你整理成可直接发给家人/朋友的版本。"
        )
    elif status == "cancelled":
        message = (
            f"你刚才选择了不执行当前方案。"
            f"如果你想重新安排 {scenario} 场景的本地活动，我可以继续帮你重新规划。"
        )
    elif status == "error":
        msg = execution_result.get("message") or "执行阶段发生异常。"
        message = (
            f"本次执行没有完全完成：{msg}\n"
            f"如果你愿意，我可以基于当前结果帮你重新规划一版更稳妥的方案。"
        )
    else:
        errors = state.get("errors") or []
        if isinstance(errors, list) and errors:
            message = (
                "当前方案暂时没有收敛到可执行结果。\n"
                f"原因：{errors[-1]}"
            )
        else:
            message = (
                f"本次安排已走到执行后的结果整理阶段。"
                f"当前状态：{status or 'unknown'}。"
            )

    return {"final_message": message}


def route_after_confirmation(state: AgentState) -> str:
    if state.get("user_confirmed"):
        return "execute"
    return "replan"


def route_after_intent(state: AgentState) -> str:
    """Intent 节点之后的条件路由。

    判定逻辑：
        * 仅当 ``intent.is_leisure_planning`` 显式为 False 时，路由到直答 ``llm_answer``。
        * 字段缺失 / intent 缺失 / 任意异常 → 默认走 ``planning``，避免误把规划任务路由到直答。
    """
    intent = state.get("intent") or {}
    if not isinstance(intent, dict):
        return "planning"
    if intent.get("is_leisure_planning") is False:
        return "llm_answer"
    return "planning"


def route_after_intent_for_retrieval(state: AgentState) -> str:
    """Intent 节点之后的条件路由（T5 起启用，含 retrieval 分支）。

    判定逻辑：
        * intent 缺失 / 非 dict → 保守走 ``planning``（与旧路由保持一致的兜底策略）；
        * ``is_leisure_planning is False`` → ``llm_answer``；
        * ``is_leisure_planning is True`` 且 ``need_retrieval is True`` → ``retrieval``；
        * 其它（含 ``need_retrieval`` 缺失 / 非 True） → ``planning``，
          这样默认家庭场景不会被多绕一次检索，主链路速度不变。
    """
    intent = state.get("intent")
    if not isinstance(intent, dict):
        return "planning"
    if intent.get("is_leisure_planning") is False:
        return "llm_answer"
    if intent.get("is_leisure_planning") is True and intent.get("need_retrieval") is True:
        return "retrieval"
    return "planning"





