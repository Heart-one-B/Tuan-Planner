# src/graph/nodes/confirmation_node.py
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from src.graph.state import AgentState
from src.model.factory import get_chat_model
from src.utils.path_tool import get_abs_path

_PREFERENCE_DOC_PATH = get_abs_path("data/user_preference_profile.md")


def _text_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [item.strip() for item in value if isinstance(item, str) and item.strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _safe_dict(value: object) -> dict:
    return value if isinstance(value, dict) else {}


def _extract_plan_context(state: AgentState) -> dict:
    intent = _safe_dict(state.get("intent"))
    plan = _safe_dict(state.get("plan"))
    final_plan_result = _safe_dict(state.get("final_plan_result"))
    selected_candidate = _safe_dict(final_plan_result.get("selected_candidate"))

    if not plan and selected_candidate:
        plan = selected_candidate

    restaurant = _safe_dict(plan.get("restaurant"))
    if not restaurant:
        restaurant = _safe_dict(selected_candidate.get("restaurant"))

    activities = plan.get("activities")
    if not isinstance(activities, list) or not activities:
        activities = selected_candidate.get("activities")
    if not isinstance(activities, list):
        activities = []

    activity_items: list[dict] = []
    for item in activities:
        if isinstance(item, dict) and item:
            activity_items.append(item)

    if not activity_items and isinstance(plan.get("activity"), dict):
        activity_items.append(plan["activity"])

    restaurant_types = _text_list(restaurant.get("tags_semantic")) or _text_list(restaurant.get("tags"))
    activity_types: list[str] = []
    for item in activity_items:
        activity_types.extend(_text_list(item.get("tags_semantic")))
        activity_types.extend(_text_list(item.get("tags")))

    preferences = _safe_dict(intent.get("preferences"))

    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "scenario": intent.get("scenario") or "unknown",
        "restaurant": {
            "name": restaurant.get("name") or "",
            "types": restaurant_types,
            "description": restaurant.get("description") or "",
        },
        "activities": [
            {
                "name": item.get("name") or "",
                "types": _text_list(item.get("tags_semantic")) or _text_list(item.get("tags")),
                "description": item.get("description") or "",
            }
            for item in activity_items
        ],
        "activity_types": activity_types,
        "distance_preference": preferences.get("distance_preference"),
        "diet_preference": _text_list(preferences.get("diet_preference")),
        "activity_style": _text_list(preferences.get("activity_style")),
        "must_avoid": _text_list(preferences.get("must_avoid")),
        "display_text": state.get("display_text") or "",
        "history_document": _load_history_document(),
    }


def _load_history_document() -> str:
    path = Path(_PREFERENCE_DOC_PATH)
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    match = re.match(r"^```(?:markdown|md)?\s*(.*?)\s*```$", stripped, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return stripped.strip("`").strip()


def _build_fallback_document(context: dict) -> str:
    restaurant = context.get("restaurant") or {}
    activities = context.get("activities") or []
    activity_lines = []
    for item in activities:
        if not isinstance(item, dict):
            continue
        name = item.get("name") or "未命名活动"
        types = item.get("types") or []
        type_text = f"（{', '.join(types)}）" if types else ""
        activity_lines.append(f"- {name}{type_text}")

    lines = [
        "# 用户偏好档案",
        "",
        f"- 最近更新时间：{context.get('generated_at') or datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 当前场景：{context.get('scenario') or 'unknown'}",
        "",
        "## 综合偏好画像",
        f"- 餐厅偏好：{', '.join(restaurant.get('types') or []) or '待进一步积累'}",
        f"- 活动偏好：{', '.join(context.get('activity_types') or []) or '待进一步积累'}",
        f"- 距离偏好：{context.get('distance_preference') or '待进一步积累'}",
        f"- 饮食偏好：{', '.join(context.get('diet_preference') or []) or '待进一步积累'}",
        f"- 活动风格：{', '.join(context.get('activity_style') or []) or '待进一步积累'}",
    ]

    if restaurant.get("name"):
        lines.extend([
            "",
            "## 本次确认记录",
            f"- 餐厅：{restaurant.get('name')}",
        ])
        if restaurant.get("types"):
            lines.append(f"- 餐厅类型：{', '.join(restaurant.get('types') or [])}")
    if activity_lines:
        if "## 本次确认记录" not in lines:
            lines.extend(["", "## 本次确认记录"])
        lines.extend(activity_lines)

    history_document = context.get("history_document") or ""
    lines.extend([
        "",
        "## 历史偏好记录",
        "- 当前为首条偏好记录。" if not history_document.strip() else "- 已存在历史文档，当前版本已在其基础上更新。",
    ])

    return "\n".join(lines).strip() + "\n"


def _summarize_preference_document(context: dict) -> str:
    history_document = context.get("history_document") or ""
    system_prompt = (
        "你是本地生活规划助手的用户偏好归纳器。"
        "你的任务是根据本次用户已经确认的方案，以及已有的偏好记录，"
        "综合总结出稳定、可复用的用户偏好档案。"
        "只输出 Markdown 文档，不要输出解释、JSON 或代码围栏。\n\n"
        "文档必须包含以下部分：\n"
        "# 用户偏好档案\n"
        "## 综合偏好画像\n"
        "## 本次确认记录\n"
        "## 历史偏好记录\n\n"
        "规则：\n"
        "1. 优先从本次确认的餐厅类型、娱乐/活动设施类型、室内外倾向、距离倾向等提炼偏好。\n"
        "2. 将历史记录与本次确认结果综合起来，提炼稳定偏好；如果存在冲突，使用“更常选择”或“倾向于”等措辞。\n"
        "3. 不要虚构用户没有体现出来的偏好。\n"
        "4. 如果历史记录为空，明确写出当前为首条记录。\n"
    )
    user_prompt = (
        "请基于下面的信息生成并更新用户偏好档案：\n\n"
        f"【当前确认方案】\n{json.dumps(context, ensure_ascii=False, indent=2)}\n\n"
        f"【历史偏好文档】\n{history_document or '（空）'}\n"
    )

    response = get_chat_model().invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    )
    content = getattr(response, "content", "") or ""
    return _strip_code_fences(content).strip()


def _write_preference_document(content: str) -> None:
    path = Path(_PREFERENCE_DOC_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def _update_preference_document(state: AgentState) -> None:
    context = _extract_plan_context(state)
    try:
        content = _summarize_preference_document(context)
    except Exception as exc:
        print(f"[Confirmation Node] 偏好文档大模型总结失败，改用本地兜底：{exc}")
        content = ""

    if not content.strip():
        content = _build_fallback_document(context)

    _write_preference_document(content)
    print(f"[Confirmation Node] 用户偏好文档已更新：{_PREFERENCE_DOC_PATH}")


def confirmation_node(state: AgentState) -> AgentState:
    """确认节点：在终端阻塞等待用户输入以确认是否按照当前排版方案执行一键下单。"""
    if "user_confirmed" in state:
        return {"user_confirmed": state["user_confirmed"]}

    if state.get("web_preview_mode") is True:
        return {
            "user_confirmed": False,
            "pending_confirmation": {
                "title": "确认执行当前方案",
                "message": "当前方案已生成，请确认是否执行一键下单。",
                "confirm_label": "确认执行",
                "cancel_label": "暂不执行",
            },
        }

    confirm = input("\n[系统提示] 确定按照此方案执行一键下单吗？(y/n): ")
    confirmed = confirm.lower() == "y"
    if confirmed:
        try:
            _update_preference_document(state)
        except Exception as exc:
            print(f"[Confirmation Node] 用户偏好文档生成失败：{exc}")
        return {"user_confirmed": True}

    return {
        "user_confirmed": False,
        "replan_reason": "用户未确认当前方案",
        "replan_reason_type": "user_feedback",
    }