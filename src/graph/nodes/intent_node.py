# src/graph/nodes/intent_node.py
from __future__ import annotations

from agents.intent.agent import IntentAgent
from src.graph.state import AgentState
from src.graph.tracing import parent_span_of
from src.memory.wiring import memory_enabled, recall_preference_context
from src.model.factory import build_llm_client


async def intent_node(state: AgentState) -> dict:
    """解析意图。记忆召回也在这里发生。

    【为什么召回接在 intent 而不是别处】
    偏好必须在 intent 解析时就在场，才能进入 IntentResult.preferences
    → plan_context.preferences → fact_task 的搜索关键词。接得晚一步
    （比如接在 planning）就只能影响"从候选池里挑哪个"，影响不了
    "候选池里有什么"——而记忆的价值恰恰在后者。

    【三条入口路径】
      new_plan      全新需求，直接解析 user_input
      clarify_reply 澄清回复，拼接原始需求 + 追问历史重新解析
      adjust        对当前方案提修改意见，拼接原始需求 + 全部调整
                    历史重新解析 ← 本次新增

    adjust 走这里的理由（原来它直接跳到 orchestrator）：
    用户说"晚上不吃火锅了我们都吃不了辣"，这句话**改变了约束本身**
    ——restaurant_intent 从 explicit(火锅) 变成 open，
    diet_preference 多了"不辣"。而原来的实现只把这句话当自由文本
    塞进 replan 的 hint，plan_context 一个字没变。

    后果实测到了：EvaluationAgent 拿着旧的 plan_context（要火锅）
    去评判新方案（不要辣的），给出 72 分并写下"用户明确要求晚上
    吃火锅，但方案提供的是羊肉汤锅"——**系统在用用户已经撤回的
    需求批评自己刚做对的事**。用户看到会困惑：我说不吃辣了，
    你却说我要火锅。

    多花一次 IntentAgent 调用，换来下游全部环节对"用户现在到底
    想要什么"有一致的理解。
    """
    task_log = list(state.get("task_log") or [])
    errors = list(state.get("errors") or [])
    surfaced = list(state.get("surfaced_memory_names") or [])
    adjust_history = list(state.get("adjust_history") or [])

    user_input = (state.get("user_input") or "").strip()
    feedback_route = state.get("feedback_route") or ""
    prev_intent = state.get("intent") or {}
    original_query = prev_intent.get("raw_query") or user_input

    is_adjust = feedback_route == "adjust"

    if feedback_route == "clarify_reply":
        history = state.get("clarification_history") or []
        history_text = "\n".join(
            f"[追问]{h.get('question','')} [回答]{h.get('answer','')}"
            for h in history
        )
        combined_input = (
            f"{original_query}\n{history_text}" if history_text else original_query
        )
        task_log.append(f"intent: clarify_reply模式，拼接{len(history)}轮历史重新解析")

    elif is_adjust:
        # 全部调整意见都带上，不只最后一条：用户可能先说"不吃辣"，
        # 再说"换个近点的"，两条都是有效约束。只带最后一条会让
        # 前面的调整被静默丢弃——那正是用户最容易察觉、也最恼火的
        # 那种"说了不听"。
        adjust_history.append(user_input)
        adjust_text = "\n".join(f"[调整{i}]{t}"
                               for i, t in enumerate(adjust_history, 1))
        combined_input = (
            f"[最初需求]{original_query}\n{adjust_text}\n\n"
            f"注意：后面的调整意见优先于最初需求。如果调整意见撤回或"
            f"改变了最初的某个要求（比如最初说想吃火锅、后来说不吃辣了），"
            f"以调整意见为准重新判断，不要保留已被撤回的要求。"
        )
        task_log.append(f"intent: adjust模式，累计{len(adjust_history)}条调整意见重新解析")

    else:
        combined_input = user_input

    if not combined_input.strip():
        errors.append({
            "node": "intent",
            "error": "用户输入为空，无法解析意图",
            "recoverable": False,
        })
        task_log.append("intent: combined_input为空，跳过解析")
        return {"task_log": task_log, "errors": errors}

    # ── 记忆召回 ──
    # 失败一律降级为"不注入"（fail open）：召回是增益，它坏了不该
    # 让一次本来能完成的规划失败。harness 内部已经这么做了，
    # 这里再包一层挡住装配层面的意外（比如记忆目录不可写）。
    preference_context = ""
    if memory_enabled():
        parent = parent_span_of(state)
        try:
            recall = await recall_preference_context(
                session_id=state.get("session_id") or "default",
                task=combined_input,
                trace_id=parent.trace_id if parent else "recall",
                already_surfaced_names=surfaced,
                llm_client=build_llm_client(),
            )
            preference_context = recall.preference_context
            surfaced = recall.surfaced_names
            task_log.append(
                f"memory: 召回 {'命中' if preference_context else '无'}"
                f"（已推送 {len(surfaced)} 条）"
            )
        except Exception as e:
            errors.append({"node": "memory_recall", "error": str(e),
                          "recoverable": True})
            task_log.append(f"memory: 召回异常 {e}，降级为不注入")

    try:
        result = await IntentAgent(llm_client=build_llm_client()).parse(
            user_input=combined_input,
            preference_context=preference_context,
            trace_id=state.get("session_id"),
        )
        intent = result.model_dump()

        # raw_query 保留最初的需求，防止被重写成只含本次回复的片段
        if feedback_route in ("clarify_reply", "adjust") and not intent.get("raw_query"):
            intent["raw_query"] = original_query
        if is_adjust:
            intent["raw_query"] = original_query
            # adjust 模式下不追问：已经有一份能用的 plan_context 了，
            # 用户在调整现有方案，不是从头提需求。这时候反过来问他
            # "你们几个人呀"是荒谬的——那些槽位第一轮就填好了。
            # 万一调整意见真的引入了新的信息缺口，交给下游用旧值
            # 兜底，比打断一次正在进行的调整体验更好。
            intent["clarification_needed"] = False
            intent["missing_slots"] = []
            intent["current_asking_slot"] = None
            intent["follow_up_message"] = None

        task_log.append(
            f"intent: scenario={intent.get('scenario')} "
            f"restaurant_intent={intent.get('restaurant_intent')} "
            f"clarification_needed={intent.get('clarification_needed')} "
            f"missing_slots={intent.get('missing_slots')}"
        )
        return {
            "intent": intent,
            "surfaced_memory_names": surfaced,
            "adjust_history": adjust_history,
            "task_log": task_log,
            "errors": errors,
        }

    except Exception as e:
        errors.append({"node": "intent", "error": str(e), "recoverable": False})
        task_log.append(f"intent: failed {e}")
        return {
            "surfaced_memory_names": surfaced,
            "adjust_history": adjust_history,
            "task_log": task_log,
            "errors": errors,
        }