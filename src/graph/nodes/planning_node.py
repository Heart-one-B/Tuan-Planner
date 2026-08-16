# src/graph/nodes/planning_node.py
from __future__ import annotations

from agents.planning.agent import PlanningAgent
from src.graph.state import AgentState
from src.graph.tracing import parent_span_of
from src.model.factory import build_llm_client


async def planning_node(state: AgentState) -> dict:
    task_log = list(state.get("task_log") or [])
    errors = list(state.get("errors") or [])
    agent_outputs = dict(state.get("agent_outputs") or {})

    plan_context = state.get("plan_context") or {}
    fact_output = agent_outputs.get("fact") or {}
    fact_data = fact_output.get("data") or {}

    # fact 失败则不规划
    if fact_output.get("status") == "error" or not fact_data:
        errors.append({"node": "planning", "error": "fact_data 缺失，跳过规划",
                       "recoverable": False})
        task_log.append("planning: skipped, fact_data missing")
        return {"task_log": task_log, "errors": errors}

    try:
        result = await PlanningAgent(llm_client=build_llm_client()).run(
            plan_context=plan_context,
            fact_data=fact_data,
            parent_span=parent_span_of(state),
        )
        agent_outputs["planning"] = {
            "status": result.status,
            "summary": result.summary,
            "data": result.data,
        }
        task_log.append(f"planning: status={result.status} summary={result.summary}")

        if result.status == "error":
            errors.append({"node": "planning", "error": result.summary, "recoverable": True})

    except Exception as e:
        # 原来这里写的是 AgentOutput(...)，而这个名字在本模块没有 import——
        # 异常处理分支自己会抛 NameError，把真实异常掩盖成一个
        # "name 'AgentOutput' is not defined"，排查时看到的错误和真正
        # 发生的事情完全无关。AgentOutput 是 TypedDict，本来也只是
        # 类型标注，构造用字面量 dict 就够。
        agent_outputs["planning"] = {"status": "error", "summary": str(e), "data": {}}
        errors.append({"node": "planning", "error": str(e), "recoverable": True})
        task_log.append(f"planning: exception {e}")

    return {"agent_outputs": agent_outputs, "task_log": task_log, "errors": errors}