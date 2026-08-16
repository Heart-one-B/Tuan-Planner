# src/graph/nodes/orchestrator_node.py
from __future__ import annotations

import logging

from agents.orchestrator.agent import OrchestratorAgent
from agents.fact.agent import FactAgent
from agents.planning.agent import PlanningAgent
from agents.evaluation.agent import EvaluationAgent
from src.graph.state import AgentState
from src.graph.tracing import parent_span_of
from src.model.factory import build_llm_client

logger = logging.getLogger(__name__)

# 这几类异常是代码缺陷，不是运行时的业务失败——它们每次调用都会
# 以完全相同的方式失败，重试、降级、换个说法都救不回来。
#
# 【为什么要单独列出来】这个节点原来把所有异常统一标成
# recoverable=True，于是一个 `OrchestratorAgent.run() got an
# unexpected keyword argument 'trace_id'`（签名不匹配，纯粹的
# 代码 bug）被当成了可恢复的业务失败：has_new_plan=False →
# 路由到 END → display_text 沿用上一轮的旧值 → **用户看到一个
# "成功"的响应，只是方案一个字没变**。
#
# 用户说"晚上不吃火锅，太辣了"，系统回了同一份火锅方案，
# 没有任何错误提示。这个 bug 活了很久，因为它每一层看起来都正常：
# routing 判对了（route=adjust），异常被捕获了，流程继续了。
#
# 把"永远不会成功的错误"标成可恢复，只是让 bug 活得更久。
_CODE_DEFECTS = (TypeError, AttributeError, NameError, ImportError, KeyError)


async def orchestrator_node(state: AgentState) -> dict:
    task_log = list(state.get("task_log") or [])
    errors = list(state.get("errors") or [])
    agent_outputs = dict(state.get("agent_outputs") or {})

    plan_context = state.get("plan_context") or {}
    fact_data = (agent_outputs.get("fact") or {}).get("data") or {}
    eval_data = (agent_outputs.get("evaluation") or {}).get("data") or {}
    selected_plan_id = eval_data.get("selected_plan_id") or ""

    planning_data = (agent_outputs.get("planning") or {}).get("data") or {}
    candidates = planning_data.get("candidates") or []
    selected_plan = next(
        (c for c in candidates if c.get("id") == selected_plan_id),
        candidates[0] if candidates else {},
    )

    user_feedback = state.get("user_input") or ""

    if not user_feedback:
        task_log.append("orchestrator: no feedback, skipped")
        return {"task_log": task_log, "errors": errors, "has_new_plan": False}

    has_new_plan = False

    try:
        llm = build_llm_client()
        result = await OrchestratorAgent(
            llm_client=llm,
            fact_agent=FactAgent(llm_client=llm),
            planning_agent=PlanningAgent(llm_client=llm),
            evaluation_agent=EvaluationAgent(llm_client=llm),
        ).run(
            user_feedback=user_feedback,
            plan_context=plan_context,
            fact_data=fact_data,
            selected_plan=selected_plan,
            parent_span=parent_span_of(state),
        )

        data = result.data
        if data.get("updated_fact"):
            agent_outputs["fact"] = data["updated_fact"]
        if data.get("updated_planning"):
            agent_outputs["planning"] = data["updated_planning"]
        if data.get("updated_evaluation"):
            agent_outputs["evaluation"] = data["updated_evaluation"]

        task_log.append(f"orchestrator: {result.summary}")

        # 只有真正产出了新的 evaluation 结果、且有选中方案，才走向
        # presentation 重新渲染；否则保持原有展示不变。
        new_eval = agent_outputs.get("evaluation") or {}
        has_new_plan = bool(
            data.get("updated_evaluation")
            and new_eval.get("status") == "ok"
            and (new_eval.get("data") or {}).get("selected_plan_id")
        )

        # 调度跑完了却没产出新方案：这不是异常，但对用户来说和
        # "说了不听"没有区别。如实记一条，让日志里能看出
        # "调整请求被处理了，但没有产生任何变化"这个事实，
        # 而不是只留下一条看起来正常的 orchestrator 日志。
        if not has_new_plan:
            task_log.append(
                "orchestrator: 未产出新方案（未触发 replan / evaluate 失败 / "
                "无合格候选），本轮展示维持不变"
            )

    except _CODE_DEFECTS as e:
        # 代码缺陷：recoverable=False。这会让 errors 里出现一条
        # 致命错误，调用方（main.py 的 _brief / 冒烟脚本的 ok 判定）
        # 据此知道"这次是真的失败了"，而不是把它当成一次正常的
        # "没什么可调整的"。
        logger.error(f"[orchestrator] 代码缺陷: {type(e).__name__}: {e}",
                    exc_info=True)
        errors.append({
            "node": "orchestrator",
            "error": f"{type(e).__name__}: {e}",
            "recoverable": False,
        })
        task_log.append(f"orchestrator: 代码缺陷 {type(e).__name__}: {e}")

    except Exception as e:
        # 业务/运行时异常（网络、超时、模型输出不合格）：可恢复，
        # 用户重试一次可能就好了，保持原方案展示是合理的降级。
        logger.warning(f"[orchestrator] 运行异常: {type(e).__name__}: {e}")
        errors.append({
            "node": "orchestrator",
            "error": f"{type(e).__name__}: {e}",
            "recoverable": True,
        })
        task_log.append(f"orchestrator: exception {type(e).__name__}: {e}")

    return {
        "agent_outputs": agent_outputs,
        "task_log": task_log,
        "errors": errors,
        "has_new_plan": has_new_plan,
    }