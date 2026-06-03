# src/graph/routers.py
from src.graph.state import AgentState

_REPLAN_BUDGET = 3


def route_after_validate(state: AgentState) -> str:
    """Validate 之后的条件路由。

    * validation_result.passed 显式为 True → ``presentation``；
    * 其它（缺失 / 非 dict / passed 非 True） → ``replan``。
    """
    result = state.get("validation_result")
    if isinstance(result, dict) and result.get("passed") is True:
        return "presentation"
    return "replan"


def route_after_replan(state: AgentState) -> str:
    """Replan 之后的条件路由。

    * replan_count >= 3 → ``final_message``；
    * 否则 → ``constraint_collect``。
    """
    count = state.get("replan_count", 0)
    if isinstance(count, bool) or not isinstance(count, int):
        count = 0
    if count >= _REPLAN_BUDGET:
        return "final_message"
    return "constraint_collect"


def route_after_confirmation(state: AgentState) -> str:
    """用户 CLI 确认一键预订下单之后的条件路由。

    * user_confirmed 为 True → ``execute``（对应节点注册的 execution）；
    * web_preview_mode 为 True → ``await_confirmation``（Web 端先展示结果，等待用户二次确认）；
    * 否则 → ``replan``（对应节点注册的 repair_loop/replan）。
    """
    if state.get("web_preview_mode") is True:
        return "await_confirmation"
    if state.get("user_confirmed"):
        return "execute"
    return "replan"


def route_after_intent(state: AgentState) -> str:
    """Intent 节点之后的条件路由。

    判定逻辑：
        * 仅当 ``intent.is_leisure_planning`` 显式为 False 时，路由到直答 ``llm_answer``。
        * 字段缺失 / intent 缺失 / 任意异常 → 默认走 ``planning``。
    """
    intent = state.get("intent") or {}
    if not isinstance(intent, dict):
        return "planning"
    if intent.get("is_leisure_planning") is False:
        return "llm_answer"
    return "planning"


def route_after_intent_for_retrieval(state: AgentState) -> str:
    """Intent 节点之后的条件路由（含 retrieval 分支）。

    判定逻辑：
        * intent 缺失 / 非 dict → 保守走 ``planning``；
        * ``is_leisure_planning is False`` → ``llm_answer``；
        * ``is_leisure_planning is True`` 且 ``need_retrieval is True`` → ``retrieval``；
        * 其它 → ``planning``。
    """
    intent = state.get("intent")
    if not isinstance(intent, dict):
        return "planning"
    if intent.get("is_leisure_planning") is False:
        return "llm_answer"
    if intent.get("is_leisure_planning") is True and intent.get("need_retrieval") is True:
        return "retrieval"
    return "planning"