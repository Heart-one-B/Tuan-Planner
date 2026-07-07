# src/graph/nodes/presentation_node.py

from __future__ import annotations

from src.graph.state import AgentState


async def presentation_node(state: AgentState) -> dict:
    """把选中方案渲染成展示文本，纯代码渲染，不调用 LLM。

    数据已经是结构化的（CandidatePlan 的 timeline/activities/restaurants），
    不需要模型再生成一次自然语言描述 —— 这样能保证：
    1. 不消耗额外 LLM 调用
    2. 不会有渲染幻觉（模型编造不存在的细节）
    3. 渲染结果是确定性的，方便测试
    """
    task_log = list(state.get("task_log") or [])
    errors   = list(state.get("errors") or [])
    agent_outputs = state.get("agent_outputs") or {}

    eval_output = agent_outputs.get("evaluation") or {}
    eval_data   = eval_output.get("data") or {}
    selected_id = eval_data.get("selected_plan_id") or ""

    # ── 异常1：evaluation 没有选中方案（不该走到这里，但防御性检查）──────
    if eval_output.get("status") != "ok" or not selected_id:
        errors.append({
            "node": "presentation",
            "error": "evaluation 未产出合格方案，presentation 不应被调用",
            "recoverable": False,
        })
        task_log.append("presentation: 无selected_plan_id，跳过渲染")
        return {
            "display_text": "抱歉，没能生成合适的方案，要不要换个时间或地点再试试？",
            "task_log": task_log,
            "errors":   errors,
        }

    planning_output = agent_outputs.get("planning") or {}
    planning_data   = planning_output.get("data") or {}
    candidates      = planning_data.get("candidates") or []

    selected_plan = next(
        (c for c in candidates if isinstance(c, dict) and c.get("id") == selected_id),
        None
    )

    # ── 异常2：selected_id 在 candidates 里找不到对应方案 ──────────────
    # 理论上不该发生(evaluation只能从传入的candidates里选)，但数据流转
    # 出问题时(比如 Orchestrator 更新了 planning 但 evaluation 引用了旧id)
    # 必须有兜底，不能让整个流程崩溃
    if not selected_plan:
        errors.append({
            "node": "presentation",
            "error": f"selected_plan_id={selected_id} 在 candidates 中未找到",
            "recoverable": False,
        })
        task_log.append(f"presentation: 找不到方案{selected_id}，candidates数量={len(candidates)}")
        return {
            "display_text": "方案数据出现异常，请重新发起规划。",
            "task_log": task_log,
            "errors":   errors,
        }

    try:
        display_text, plan_summary = _render_plan(
            selected_plan,
            eval_data.get("selected_score"),
            eval_data.get("selected_reason"),
        )
        task_log.append(f"presentation: 渲染完成 plan_id={selected_id}")
        return {
            "display_text":          display_text,
            "selected_plan_summary": plan_summary,
            "pending_confirmation":  True,
            "task_log": task_log,
            "errors":   errors,
        }

    except Exception as e:
        # ── 异常3：渲染过程本身出错（数据字段缺失导致拼接失败等）────────
        errors.append({"node": "presentation", "error": str(e), "recoverable": False})
        task_log.append(f"presentation: 渲染异常 {e}")
        return {
            "display_text": "方案生成时出现了点问题，麻烦重新试一下。",
            "task_log": task_log,
            "errors":   errors,
        }


def _render_plan(plan: dict, score, reason: str) -> tuple[str, dict]:
    """纯函数渲染：把 CandidatePlan 转成 (展示文本, 结构化摘要)。

    所有字段访问都用 .get() 兜底，任何一个 POI 字段缺失都不应该
    让整个渲染崩溃 —— 缺了就显示"待确认"或留空，而不是抛异常。
    """
    title     = plan.get("title") or "为你推荐的方案"
    timeline  = plan.get("timeline") or []
    reasoning = plan.get("reasoning") or []

    lines = [f"# {title}", ""]

    if score is not None:
        lines.append(f"匹配度：{score}分")
        lines.append("")

    lines.append("## 行程安排")
    if not timeline:
        lines.append("（暂无具体安排）")
    else:
        for item in timeline:
            time_str = item.get("time") or "?"
            end_str  = item.get("end_time") or ""
            name     = item.get("item") or item.get("label") or "待确认"
            time_range = f"{time_str}-{end_str}" if end_str else time_str
            lines.append(f"- {time_range} {name}")

    if reasoning:
        lines.append("")
        lines.append("## 推荐理由")
        for r in reasoning:
            if isinstance(r, str) and r.strip():
                lines.append(f"- {r}")

    if reason:
        lines.append("")
        lines.append(f"## 综合评价\n{reason}")

    lines.append("")
    lines.append("这个方案可以吗？需要调整随时告诉我。")

    display_text = "\n".join(lines)

    # 结构化摘要，给前端按需渲染卡片用，不依赖文本解析
    plan_summary = {
        "id":        plan.get("id", ""),
        "title":     title,
        "timeline":  timeline,
        "reasoning": reasoning,
        "score":     score,
    }

    return display_text, plan_summary