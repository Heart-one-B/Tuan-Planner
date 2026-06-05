from __future__ import annotations

import json
import re

from langchain_core.messages import HumanMessage, SystemMessage

from src.graph.state import AgentState
from src.model.factory import get_chat_model

_SYSTEM_PROMPT = """\
你是行程规划助手的对话路由模块。你需要判断用户这句话的意图。

只输出 JSON，格式如下，不输出任何其他内容：
{
  "route": "new_plan" | "adjust" | "chat",
  "new_activity_keywords": [],
  "new_restaurant_keywords": [],
  "new_waypoint_keywords": [],
  "feedback_summary": "用户意见的简短摘要"
}

## route 判断规则

**new_plan**：用户想要一个全新的规划，包括：
- 第一次提出规划需求
- 换了场景/日期/人员/出发地，和当前方案完全无关
- 明确说"重新规划""换一个""下次..."

**adjust**：用户在对当前方案提意见、补充需求、要求修改，包括：
- 修改某个活动或餐厅（"把午饭换成火锅""不想去那个美术馆"）
- 补充新需求（"我还想去看电影""能不能加一个活动"）
- 调整时间/顺序（"能不能晚点吃饭"）
- 表达不满意（"活动太少了""这个不好"）

**chat**：闲聊、确认、感谢、询问，不需要修改方案，包括：
- "好的谢谢""明白了""确认"
- "几点出发""在哪里"
- 单纯问答

## new_*_keywords 填写规则（只在 route=adjust 时填）
只填用户新提到的、需要额外搜索 POI 的关键词：
- 活动类（电影/剧本杀/台球/KTV）→ new_activity_keywords
- 餐饮类（火锅/烧烤/日料）→ new_restaurant_keywords
- 顺路小需求（奶茶/DQ/甜品）→ new_waypoint_keywords
- 纯修改意见（换掉某个/调整顺序）→ 三个列表都为空，只填 feedback_summary

## feedback_summary
一句话概括用户的意见，供重规划模块参考。new_plan/chat 时填空字符串。
"""


def feedback_router_node(state: AgentState) -> AgentState:
    """
    Feedback Router Node：所有用户输入的第一个节点。

    判断逻辑：
    - 没有已有方案（first_plan_result 为空）→ 直接 new_plan，不调 LLM
    - 有已有方案 → 调 LLM 判断 adjust / new_plan / chat

    路由结果写入 feedback_route，下游 workflow 按此路由。

    new_plan  → intent（重走完整流程，清空旧状态）
    adjust    → constraint_build（增量合并 + 增量搜索）
    chat      → llm_answer
    """
    print("[Feedback Router Node] 判断用户意图...")

    user_input = (state.get("user_input") or "").strip()
    final_plan = state.get("final_plan_result") or {}

    # ── 没有已有方案，直接走 new_plan，不调 LLM ──────────────────────────
    if not final_plan or not final_plan.get("selected_candidate"):
        print("[Feedback Router Node] 无已有方案，直接走 new_plan")
        return {
            "feedback_route":                  "new_plan",
            "feedback_summary":                "",
            "incremental_activity_keywords":   [],
            "incremental_restaurant_keywords": [],
            "incremental_waypoint_keywords":   [],
        }

    # ── 有已有方案，调 LLM 判断 ───────────────────────────────────────────
    selected = final_plan.get("selected_candidate") or {}
    timeline = selected.get("timeline") or []
    plan_summary = ""
    if timeline:
        items = [
            f"{t.get('time', '')} {t.get('item', '')}"
            for t in timeline if t.get("item")
        ]
        plan_summary = f"当前方案：{', '.join(items[:6])}"

    user_message = f"""\
{plan_summary}

用户说：「{user_input}」

请判断属于 new_plan / adjust / chat，并按格式输出 JSON。
"""

    route = "new_plan"
    new_activity_kws:   list[str] = []
    new_restaurant_kws: list[str] = []
    new_waypoint_kws:   list[str] = []
    feedback_summary = ""

    try:
        response = get_chat_model().invoke([
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ])
        match = re.search(r"\{.*\}", response.content, re.DOTALL)
        raw = match.group() if match else response.content
        result = json.loads(raw)

        route              = result.get("route") or "new_plan"
        new_activity_kws   = result.get("new_activity_keywords") or []
        new_restaurant_kws = result.get("new_restaurant_keywords") or []
        new_waypoint_kws   = result.get("new_waypoint_keywords") or []
        feedback_summary   = result.get("feedback_summary") or ""

        print(
            f"[Feedback Router Node] route={route} | "
            f"活动+={new_activity_kws} 餐厅+={new_restaurant_kws} waypoint+={new_waypoint_kws}"
        )
        print(f"[Feedback Router Node] feedback={feedback_summary!r}")

    except Exception as exc:
        print(f"[Feedback Router Node][ERROR] {exc}，默认走 new_plan")

    # ── new_plan 时清空旧规划状态 ─────────────────────────────────────────
    if route == "new_plan":
        return {
            "feedback_route":                  "new_plan",
            "feedback_summary":                "",
            "incremental_activity_keywords":   [],
            "incremental_restaurant_keywords": [],
            "incremental_waypoint_keywords":   [],
            # 清空上一轮规划状态，避免新一轮受旧数据污染
            "plan_context":           {},
            "fact_gathering_result":  {},
            "candidate_plans":        {},
            "rule_validation_result": {},
            "scoring_result":         {},
            "final_plan_result":      {},
            "replan_count":           0,
            "replan_reason":          "",
            "replan_reason_type":     "",
            "activities":             [],
            "restaurants":            [],
            "waypoints":              [],
            "eta":                    {},
        }

    return {
        "feedback_route":                  route,
        "feedback_summary":                feedback_summary,
        "incremental_activity_keywords":   new_activity_kws,
        "incremental_restaurant_keywords": new_restaurant_kws,
        "incremental_waypoint_keywords":   new_waypoint_kws,
    }