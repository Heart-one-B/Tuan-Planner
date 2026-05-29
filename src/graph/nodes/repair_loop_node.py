# src/graph/nodes/replan_node.py
from src.graph.state import AgentState
from src.utils.state_utils import (
    _append_error, _resolve_replan_reason, _normalize_replan_reason_type,
    _is_pos_num, _contains_any, _append_unique_hint
)

_REPLAN_TRAFFIC_FALLBACK_BASE = 60
_REPLAN_QUEUE_FALLBACK_BASE = 60
_REPLAN_TRAFFIC_FLOOR = 20
_REPLAN_QUEUE_FLOOR = 10
_REPLAN_BUDGET = 3


def repair_loop_node(state: AgentState) -> AgentState:
    """Merged Replan Node / Repair Loop Node:
    统一合并了原：
      - repair_loop_node (自动规则校验失败，根据 repair_instructions 进行修复)
      - replan_node (人机交互拒绝，解析 user replan_reason 进行约束退让)

    职责：
      统一收集规划违约、失败与拒绝因子，对其进行空间/时间等约束收紧或退让。
    """
    print("[Replan Node / Repair Loop] 启动合并修复回环，解析违规指令与人工反馈...")
    try:
        # === 1. 统一维护迭代轮次 ===
        prev_count = state.get("replan_count", 0)
        if not isinstance(prev_count, int) or isinstance(prev_count, bool):
            prev_count = 0
        new_count = prev_count + 1

        # === 2. 读取基础结构与上下文 ===
        rule_validation_result = state.get("rule_validation_result") or {}
        constraint_build = state.get("constraint_build") or {}
        invalid_plans = rule_validation_result.get("invalid_plans") or []

        old_constraints = state.get("constraints") or {}
        new_constraints = dict(old_constraints)

        old_hints = new_constraints.get("replan_hints") or []
        new_hints = list(old_hints)

        # 分支分流判定：优先处理自动校验失败
        has_validation_failures = len(invalid_plans) > 0
        reason_type = "validation_failure" if has_validation_failures else _normalize_replan_reason_type(state)

        # 提取底层的各种约束细分结构以便同步更新
        hard_constraints = constraint_build.get("hard_constraints") or {}
        soft_preferences = constraint_build.get("soft_preferences") or {}
        context_memory = constraint_build.get("context_memory") or {}
        query_constraints = constraint_build.get("query_constraints") or {}

        next_hard_constraints = dict(hard_constraints)
        next_soft_preferences = dict(soft_preferences)
        next_query_constraints = dict(query_constraints)
        next_context_memory = dict(context_memory)

        # -------------------------------------------------------------------
        # 路径 A: 自动规则校验失败的修复流 (100% 对应原 repair_loop_node 逻辑)
        # -------------------------------------------------------------------
        if has_validation_failures:
            repair_targets = []
            merged_repair_instructions = []
            merged_violations = []

            for item in invalid_plans:
                if not isinstance(item, dict):
                    continue
                candidate_id = item.get("candidate_id") if isinstance(item.get("candidate_id"), str) else "unknown_plan"
                violations = item.get("violations") if isinstance(item.get("violations"), list) else []
                repair_instructions = item.get("repair_instructions") if isinstance(item.get("repair_instructions"),
                                                                                    list) else []

                repair_targets.append(candidate_id)
                for violation in violations:
                    if isinstance(violation, str) and violation not in merged_violations:
                        merged_violations.append(violation)
                for instruction in repair_instructions:
                    if isinstance(instruction, dict):
                        merged_repair_instructions.append(instruction)

            # 逐一解析违规修复指令，修改约束大底
            for instruction in merged_repair_instructions:
                instruction_type = instruction.get("type")
                constraint = instruction.get("constraint")
                if instruction_type == "replace_activity" and constraint == "indoor_only":
                    next_hard_constraints["indoor_only"] = True
                    next_query_constraints["indoor_preferred"] = True
                    next_query_constraints["weather_guard"] = "indoor_preferred"
                    new_constraints["indoor_preferred"] = True
                if instruction_type == "replace_restaurant" and constraint == "party_fit_required":
                    next_hard_constraints["restaurant_required"] = True
                if instruction_type == "reduce_eta":
                    current_eta = next_hard_constraints.get("max_traffic_minutes", 40)
                    if isinstance(current_eta, int) and current_eta > 20:
                        next_hard_constraints["max_traffic_minutes"] = current_eta - 10
                        new_constraints["max_traffic_minutes"] = current_eta - 10
                if instruction_type == "reduce_queue":
                    current_queue = next_hard_constraints.get("max_queue_minutes", 30)
                    if isinstance(current_queue, int) and current_queue > 10:
                        next_hard_constraints["max_queue_minutes"] = current_queue - 10
                        new_constraints["max_queue_minutes"] = current_queue - 10

            next_context_memory["repair_targets"] = repair_targets
            next_context_memory["repair_violations"] = merged_violations
            new_hints.extend(merged_violations)

        # -------------------------------------------------------------------
        # 路径 B: 人工交互拒绝的反馈流 (100% 对应原 replan_node 逻辑)
        # -------------------------------------------------------------------
        else:
            reason = _resolve_replan_reason(state)

            if reason_type == "execution_core_change":
                next_hard_constraints["indoor_only"] = True
                next_query_constraints["indoor_preferred"] = True
                new_constraints["indoor_preferred"] = True
                cur = new_constraints.get("max_traffic_minutes")
                base = cur if _is_pos_num(cur) else _REPLAN_TRAFFIC_FALLBACK_BASE
                next_hard_constraints["max_traffic_minutes"] = max(_REPLAN_TRAFFIC_FLOOR, int(base) - 5)
                new_constraints["max_traffic_minutes"] = max(_REPLAN_TRAFFIC_FLOOR, int(base) - 5)
                _append_unique_hint(new_hints, "execution_core_change")

            if reason:
                if _contains_any(reason, ("ETA", "通勤", "太远", "too far", "路线太长")):
                    cur = new_constraints.get("max_traffic_minutes")
                    base = cur if _is_pos_num(cur) else _REPLAN_TRAFFIC_FALLBACK_BASE
                    target_val = max(_REPLAN_TRAFFIC_FLOOR, int(base) - 10)
                    next_hard_constraints["max_traffic_minutes"] = target_val
                    new_constraints["max_traffic_minutes"] = target_val
                    if reason_type == "user_feedback":
                        _append_unique_hint(new_hints, "prefer_nearer_options")

                if _contains_any(reason, ("排队", "没位", "没位置", "wait", "等太久")):
                    cur = new_constraints.get("max_queue_minutes")
                    base = cur if _is_pos_num(cur) else _REPLAN_QUEUE_FALLBACK_BASE
                    target_val = max(_REPLAN_QUEUE_FLOOR, int(base) - 10)
                    next_hard_constraints["max_queue_minutes"] = target_val
                    new_constraints["max_queue_minutes"] = target_val
                    if reason_type == "user_feedback":
                        _append_unique_hint(new_hints, "prefer_shorter_queue")

                if _contains_any(reason, ("天气", "户外", "outdoor", "室内", "室外", "indoor")):
                    next_hard_constraints["indoor_only"] = True
                    next_query_constraints["indoor_preferred"] = True
                    new_constraints["indoor_preferred"] = True

                if reason_type == "user_feedback":
                    if _contains_any(reason, ("轻松", "累", "不想太累", "relax")):
                        _append_unique_hint(new_hints, "prefer_relaxed_schedule")
                    if _contains_any(reason, ("拍照", "好看", "氛围", "photo")):
                        _append_unique_hint(new_hints, "prefer_photo_friendly")

                _append_unique_hint(new_hints, reason)

        # === 3. 统一写回并序列化更新状态 ===
        new_constraints["replan_hints"] = new_hints
        next_context_memory["repair_round"] = new_count

        next_constraint_build = {
            "request_type": constraint_build.get("request_type", "generic_local_plan"),
            "plan_mode": constraint_build.get("plan_mode", "activity_plus_meal"),
            "hard_constraints": next_hard_constraints,
            "soft_preferences": next_soft_preferences,
            "query_constraints": next_query_constraints,
            "validation_profile": constraint_build.get("validation_profile", {}),
            "scoring_profile": constraint_build.get("scoring_profile", {}),
            "context_memory": next_context_memory,
        }

        retrieval_context = state.get("retrieval_context") or {}
        next_retrieval_context = dict(retrieval_context)
        next_retrieval_context["next_constraint_build"] = next_constraint_build

        update_dict = {
            "replan_count": new_count,
            "replan_reason": "",
            "replan_reason_type": reason_type,
            "constraints": new_constraints,
            "constraint_build": next_constraint_build,
            "retrieval_context": next_retrieval_context,
        }

        # 达到或超出预算上限时，标记报错引导至降级呈现
        if new_count >= _REPLAN_BUDGET:
            errors = list(state.get("errors", []))
            errors.append(f"已尝试重规划 {new_count} 次，仍无法满足全部约束，进入兜底展示")
            update_dict["errors"] = errors
            print(f"[Replan Node][WARN] replan_count={new_count} 命中上限 {_REPLAN_BUDGET}，进入兜底")
        else:
            print(
                f"[Replan Node][OK] replan_count={new_count}, "
                f"max_traffic_minutes={new_constraints.get('max_traffic_minutes')}, "
                f"max_queue_minutes={new_constraints.get('max_queue_minutes')}, "
                f"indoor_preferred={new_constraints.get('indoor_preferred')}"
            )

        return update_dict

    except Exception as exc:
        print(f"[Replan Node][WARN] 合并重规划异常，记录错误并跳过: {exc}")
        return _append_error(state, f"Merged Replan node failed: {exc}")