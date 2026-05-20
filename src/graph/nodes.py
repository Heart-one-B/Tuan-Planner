from langchain_core.messages import HumanMessage

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
    """新架构候选计划节点骨架：使用 LLM 生成 3 个结构化候选计划。"""
    print("[Candidate Planning Node] 生成 3 个候选计划骨架...")
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

        prompt = f"""
你是本地生活行程规划助手。
请根据给定的结构化约束和事实数据，生成 3 个候选计划。

要求：
1. 只输出 JSON，不要输出任何额外解释。
2. 输出字段必须是：
{{
  "candidates": [
    {{
      "id": "plan_1",
      "title": "...",
      "timeline": [],
      "activity": {{}},
      "restaurant": {{}},
      "reasoning": []
    }},
    {{
      "id": "plan_2",
      "title": "...",
      "timeline": [],
      "activity": {{}},
      "restaurant": {{}},
      "reasoning": []
    }},
    {{
      "id": "plan_3",
      "title": "...",
      "timeline": [],
      "activity": {{}},
      "restaurant": {{}},
      "reasoning": []
    }}
  ]
}}
3. 必须只基于输入事实生成，不允许编造输入里不存在的事实。
4. timeline/activity/restaurant 可以先保持最小骨架，但结构必须完整。

输入：
- request_type: {request_type}
- plan_mode: {plan_mode}
- hard_constraints: {hard_constraints}
- soft_preferences: {soft_preferences}
- query_constraints: {query_constraints}
- fact_gathering_result: {fact_gathering_result}
"""

        response = chat_model.invoke([HumanMessage(content=prompt)])
        content = getattr(response, "content", "") or ""

        import json
        import re

        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        payload = json.loads(json_match.group() if json_match else content)
        if not isinstance(payload, dict):
            payload = {}

        candidates = payload.get("candidates")
        if not isinstance(candidates, list):
            candidates = []

        normalized_candidates = []
        for index in range(3):
            item = candidates[index] if index < len(candidates) and isinstance(candidates[index], dict) else {}
            normalized_candidates.append(
                {
                    "id": item.get("id") if isinstance(item.get("id"), str) and item.get("id") else f"plan_{index + 1}",
                    "title": item.get("title") if isinstance(item.get("title"), str) else "",
                    "timeline": item.get("timeline") if isinstance(item.get("timeline"), list) else [],
                    "activity": item.get("activity") if isinstance(item.get("activity"), dict) else {},
                    "restaurant": item.get("restaurant") if isinstance(item.get("restaurant"), dict) else {},
                    "reasoning": item.get("reasoning") if isinstance(item.get("reasoning"), list) else [],
                }
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
                {"id": "plan_1", "title": "", "timeline": [], "activity": {}, "restaurant": {}, "reasoning": []},
                {"id": "plan_2", "title": "", "timeline": [], "activity": {}, "restaurant": {}, "reasoning": []},
                {"id": "plan_3", "title": "", "timeline": [], "activity": {}, "restaurant": {}, "reasoning": []},
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

        next_context_memory["repair_round"] = state.get("replan_count", 0)
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

        scored_candidates = scoring_result.get("scored_candidates")
        if not isinstance(scored_candidates, list):
            scored_candidates = []

        best_candidate = None
        best_score = None
        for item in scored_candidates:
            if not isinstance(item, dict):
                continue
            score = item.get("final_score")
            if not isinstance(score, (int, float)):
                continue
            if best_score is None or score > best_score:
                best_candidate = item
                best_score = score

        return {
            "final_plan_result": {
                "selected_candidate": best_candidate or {},
                "selected_candidate_id": best_candidate.get("candidate_id") if isinstance(best_candidate, dict) else "",
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
        )
        constraints = result.get("constraints") if isinstance(result, dict) else {}
        constraint_build = result.get("constraint_build") if isinstance(result, dict) else {}
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


def _activity_matches_daypart(activity: dict, daypart: str) -> bool:
    peak_hours = activity.get("peak_hours")
    if not isinstance(peak_hours, list) or not peak_hours:
        return False

    for slot in peak_hours:
        if not isinstance(slot, str) or "-" not in slot:
            continue
        start_text, end_text = slot.split("-", 1)
        try:
            start_hour = int(start_text.split(":")[0])
            end_hour = int(end_text.split(":")[0])
        except (TypeError, ValueError):
            continue

        if daypart == "下午" and start_hour < 18:
            return True
        if daypart == "晚上" and end_hour >= 18:
            return True
    return False


def _restaurant_matches_daypart(restaurant: dict, daypart: str) -> bool:
    peak_hours = restaurant.get("peak_hours")
    if not isinstance(peak_hours, list) or not peak_hours:
        return False

    for slot in peak_hours:
        if not isinstance(slot, str) or "-" not in slot:
            continue
        start_text, end_text = slot.split("-", 1)
        try:
            start_hour = int(start_text.split(":")[0])
            end_hour = int(end_text.split(":")[0])
        except (TypeError, ValueError):
            continue

        if daypart == "下午" and start_hour < 18:
            return True
        if daypart == "晚上" and end_hour >= 18:
            return True
    return False


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
        weather = MockToolAPI().get_weather(scenario_key)
        if not isinstance(weather, dict):
            weather = {}
        weather["scenario_key_used"] = scenario_key
        weather["date_label_used"] = constraints.get("date_label")
        weather["daypart_used"] = constraints.get("daypart")
        print(
            f"[Weather Check Node] result scenario_key_used={scenario_key!r}, "
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

        scenario = constraints.get("scenario") or "family"
        daypart = constraints.get("daypart")
        if not isinstance(daypart, str):
            daypart = ""
        activities = MockToolAPI().search_activities(scenario)
        if not isinstance(activities, list):
            activities = []

        child_friendly_required = constraints.get("child_friendly_required") is True
        normalized_activities: list[dict] = []
        for item in activities:
            if not isinstance(item, dict):
                continue
            normalized_activities.append(dict(item))

        if child_friendly_required:
            child_friendly_matches = [
                item for item in normalized_activities if item.get("child_friendly") is True
            ]
            if child_friendly_matches:
                normalized_activities = child_friendly_matches

        matched = [
            item for item in normalized_activities if _activity_matches_daypart(item, daypart)
        ]
        if not matched:
            update = _append_error(state, f"Activity Search node found no activities for daypart [{daypart}]")
            print(f"[Activity Search Node] no matched activities for daypart={daypart!r}")
            update["activities"] = []
            return update

        for item in matched:
            item["daypart_used"] = daypart
        print(
            f"[Activity Search Node] matched_ids={[item.get('id') for item in matched]}, "
            f"daypart_used={daypart!r}"
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

        diet_preference = constraints.get("diet_preference") or ""
        scenario = constraints.get("scenario") or ""
        if not isinstance(scenario, str):
            scenario = ""
        daypart = constraints.get("daypart")
        if not isinstance(daypart, str):
            daypart = ""
        restaurants = MockToolAPI().search_restaurants(diet_preference)
        if not isinstance(restaurants, list):
            restaurants = []

        normalized_restaurants: list[dict] = []
        for item in restaurants:
            if not isinstance(item, dict):
                continue
            normalized_restaurants.append(dict(item))

        matched = [
            item for item in normalized_restaurants if _restaurant_matches_daypart(item, daypart)
        ]
        if not matched:
            update = _append_error(state, f"Restaurant Search node found no restaurants for daypart [{daypart}]")
            print(f"[Restaurant Search Node] no matched restaurants for daypart={daypart!r}")
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
        print(
            f"[Restaurant Search Node] matched_ids={[item.get('id') for item in matched]}, "
            f"daypart_used={daypart!r}"
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
            record = api.get_traffic_eta(origin_area, target_id, depart_context)
            eta_by_target[target_id] = {
                "eta_minutes": record.get("eta_minutes"),
                "congestion": record.get("congestion"),
                "fallback_hint": record.get("fallback_hint"),
                "depart_context_used": depart_context,
            }
        print(
            f"[Traffic ETA Node] origin_area_used={origin_area!r}, "
            f"depart_context_used={depart_context!r}, "
            f"target_count={len(eta_by_target)}"
        )
        return {
            "traffic": {
                "origin_area_used": origin_area,
                "depart_context_used": depart_context,
                "eta_by_target": eta_by_target,
            }
        }
    except Exception as exc:
        print(f"[Traffic ETA Node][WARN] 节点异常，跳过并记录错误: {exc}")
        return _append_error(state, f"Traffic ETA node failed: {exc}")


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





