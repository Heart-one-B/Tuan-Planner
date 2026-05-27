# src/graph/nodes/constraint_build_node.py
import requests
from src.graph.state import AgentState
from src.agent.constraint_agent import ConstraintAgent
from src.tools.mock_api import MockToolAPI
from src.utils.state_utils import _append_error


def constraint_build_node(state: AgentState) -> AgentState:
    """Constraint Build Node：形成统一规划问题。
    合并了原：
      - location_permission_node (询问定位授权)
      - location_lookup_node (公网IP定位查询)
      - location_fallback_node (定位兜底询问)
      - constraint_collect_node (意图/检索汇总)
    """
    print("[Constraint Build Node] 形成统一规划问题 (处理定位并汇总约束)...")

    # 1. 物理定位与兜底交互逻辑
    runtime_area = state.get("runtime_origin_area", "")
    runtime_coordinates = state.get("runtime_origin_coordinates", "")

    if not runtime_area:
        print("[Constraint Build Node] 用户未明确地点，准备请求定位授权...")
        confirm = input("如果你没说具体地点，我可以先用当前定位继续规划。是否允许？(y/n): ")
        granted = confirm.strip().lower() == "y"

        if granted:
            print("[Constraint Build Node] 开始获取用户定位...")
            try:
                amap = MockToolAPI()._get_amap()
                lookup_success = False
                if amap is not None:
                    ip = ""
                    try:
                        resp = requests.get("https://ifconfig.me/ip", timeout=5, headers={"User-Agent": "Mozilla/5.0"})
                        if resp.ok:
                            ip = resp.text.strip()
                    except Exception:
                        pass

                    if ip:
                        lookup = amap.maps_ip_location(ip)
                        if isinstance(lookup, dict) and lookup.get("status") != "unknown":
                            city = lookup.get("city") or ""
                            adcode = lookup.get("adcode") or ""
                            rectangle = lookup.get("rectangle") or ""
                            if rectangle and ";" in rectangle:
                                try:
                                    p1, p2 = rectangle.split(";", 1)
                                    lng1, lat1 = [float(x) for x in p1.split(",")]
                                    lng2, lat2 = [float(x) for x in p2.split(",")]
                                    runtime_coordinates = f"{(lng1 + lng2) / 2:.6f},{(lat1 + lat2) / 2:.6f}"
                                except Exception:
                                    pass
                            runtime_area = city or adcode or ""
                            lookup_success = True

                if not lookup_success:
                    print("[Constraint Build Node] 自动定位查询失败，改为询问城市/区域...")
                    user_area = input("你大概在哪个城市或区域？\n> ").strip()
                    while not user_area:
                        user_area = input("你大概在哪个城市或区域？\n> ").strip()
                    runtime_area = user_area
            except Exception as exc:
                print(f"[Constraint Build Node][WARN] 定位失败，走兜底询问: {exc}")
                user_area = input("你大概在哪个城市或区域？\n> ").strip()
                while not user_area:
                    user_area = input("你大概在哪个城市或区域？\n> ").strip()
                runtime_area = user_area
        else:
            print("[Constraint Build Node] 用户拒绝定位授权，改为询问城市/区域...")
            user_area = input("你大概在哪个城市或区域？\n> ").strip()
            while not user_area:
                user_area = input("你大概在哪个城市或区域？\n> ").strip()
            runtime_area = user_area

    # 2. 约束条件汇总整合
    try:
        result = ConstraintAgent().collect(
            state.get("intent", {}),
            state.get("retrieval_context", {}),
            state.get("replan_reason", ""),
            state.get("replan_reason_type", ""),
            runtime_area,
            runtime_coordinates,
        )
        constraints = result.get("constraints") if isinstance(result, dict) else {}
        constraint_build = result.get("constraint_build") if isinstance(result, dict) else {}
        normalized_time = state.get("normalized_time") or {}
        if normalized_time:
            constraints["date_label"] = normalized_time.get("normalized_date_label", constraints.get("date_label", ""))
            constraints["daypart"] = normalized_time.get("normalized_daypart", constraints.get("daypart", ""))
            constraints["time_phrase"] = normalized_time.get("normalized_time_phrase",
                                                             constraints.get("time_phrase", ""))
            if not constraints.get("start_time") and normalized_time.get("base_start_minutes") is not None:
                base_start_minutes = normalized_time.get("base_start_minutes")
                if isinstance(base_start_minutes, int):
                    constraints["base_start_minutes"] = base_start_minutes

            hard_constraints = constraint_build.get("hard_constraints") or {}
            hard_constraints["date_label"] = constraints["date_label"]
            hard_constraints["daypart"] = constraints["daypart"]
            hard_constraints["time_phrase"] = constraints["time_phrase"]
            hard_constraints["base_start_minutes"] = normalized_time.get("base_start_minutes")
            if not hard_constraints.get("time_window") and constraints.get("time_window"):
                hard_constraints["time_window"] = constraints.get("time_window")
            constraint_build["hard_constraints"] = hard_constraints

            query_constraints = constraint_build.get("query_constraints") or {}
            query_constraints["time_window"] = constraints.get("time_window", query_constraints.get("time_window", ""))
            constraint_build["query_constraints"] = query_constraints

            context_memory = constraint_build.get("context_memory") or {}
            defaults_applied = context_memory.get("defaults_applied") or []
            if normalized_time.get("normalized_date_label") and normalized_time.get(
                    "normalized_date_label") != state.get("intent", {}).get("time", {}).get("date_label"):
                marker = "date_label:normalized_by_time_node"
                if marker not in defaults_applied:
                    defaults_applied.append(marker)
            context_memory["defaults_applied"] = defaults_applied
            constraint_build["context_memory"] = context_memory

        return {
            "runtime_origin_area": runtime_area,
            "runtime_origin_coordinates": runtime_coordinates,
            "constraints": constraints,
            "constraint_build": constraint_build,
        }
    except Exception as exc:
        print(f"[Constraint Build Node][WARN] 约束整合失败: {exc}")
        state_update = _append_error(state, f"Constraint Build failed: {exc}")
        state_update.update({
            "runtime_origin_area": runtime_area,
            "runtime_origin_coordinates": runtime_coordinates,
        })
        return state_update