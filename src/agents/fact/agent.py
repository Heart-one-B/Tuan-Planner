from __future__ import annotations

import json

from harness.agent.dag_scheduler import DagScheduler
from harness.agent.dag_executors import FunctionExecutor
from harness.agent.result import AgentResult
from harness.llm.base import LLMClientBase

from agents.fact.schema import FactData
from agents.fact.tools import FactToolset
from agents.fact.prompt import build_fact_planning_prompt, FACT_PLANNING_REVIEW_PROMPT


class FactAgent:
    """事实收集 Agent — Plan-and-Execute 版本（真异步）。

    搜索规划阶段用"两轮自我审查"代替代码层面的缺失类别检测：
    第一轮模型自由生成搜索计划，第二轮把产出原样喂回去让模型自己
    检查有没有遗漏或过度扩展——这是对 ReAct 范式"自省"能力的
    真实复用，不是用代码规则去模拟语义判断。代码不参与任何
    "这次任务需要什么品类""该不该精确匹配"这类判断，只负责编排。
    """

    def __init__(self, llm_client: LLMClientBase):
        self._llm = llm_client

    async def run(self, task: str, plan_mode: str = "", trace_id: str | None = None) -> AgentResult:
        toolset = FactToolset()

        # ── 第一步：geocode ──
        try:
            geocode_raw = await toolset.geocode(self._extract_origin_hint(task))
            geocode_result = json.loads(geocode_raw)
            if not geocode_result.get("coordinates"):
                return AgentResult(
                    status="error",
                    summary=f"出发地解析失败: {geocode_result}",
                    data={},
                )
        except Exception as e:
            return AgentResult(status="error", summary=f"geocode 失败: {e}", data={})

        # ── 第二步：天气查询 ──
        try:
            weather_raw = await toolset.get_weather(toolset.origin_city)
            weather_result = json.loads(weather_raw)
        except Exception:
            weather_result = {}

        # ── 第三步：LLM规划搜索关键词（两轮自我审查）──
        try:
            search_plan = await self._plan_searches(task, weather_result, plan_mode, trace_id)
        except Exception as e:
            return AgentResult(status="error", summary=f"搜索规划失败: {e}", data={})

        if not search_plan:
            return AgentResult(
                status="empty",
                summary="未规划出任何搜索任务，任务描述可能缺少关键信息",
                data={},
            )

        # ── 第四步：构建DAG，并行搜索 → 批量ETA ──
        steps = self._build_dag(search_plan)
        executor = FunctionExecutor(self._make_exec_func(toolset))

        def lenient_gate(upstream_results):
            """至少一个上游成功就放行，避免1个搜索失败拖累batch_eta整体跳过。"""
            return any(r.status == "ok" for r in upstream_results)

        scheduler = DagScheduler(executor=executor, gate=lenient_gate)

        try:
            results = await scheduler.run(steps, trace_id=trace_id)
        except Exception as e:
            return AgentResult(status="error", summary=f"DAG执行失败: {e}", data={})

        search_step_ids = [s["id"] for s in steps if s["type"] == "search"]
        failed_searches = [
            sid for sid in search_step_ids
            if results.get(sid) and results[sid].status != "ok"
        ]
        if len(failed_searches) == len(search_step_ids):
            return AgentResult(
                status="error",
                summary="所有搜索任务均失败，无法生成候选池",
                data={},
            )

        # ── 第五步：Synthesize（纯代码，不调LLM）──
        try:
            fact_data = self._synthesize(
                geocode_result=geocode_result,
                weather_result=weather_result,
                results=results,
                steps=steps,
            )
        except Exception as e:
            return AgentResult(status="error", summary=f"结果提炼失败: {e}", data={})

        status = "partial" if failed_searches else "ok"
        return AgentResult(
            status=status,
            summary=(
                f"搜到{len(fact_data.activities)}个活动候选、"
                f"{len(fact_data.restaurants)}家餐厅候选"
                + (f"，{len(failed_searches)}个搜索任务失败" if failed_searches else "")
            ),
            data=fact_data.model_dump(),
        )

    # ── Planner：两轮对话，第二轮做自我审查 ────────────────────────────

    async def _plan_searches(
        self, task: str, weather: dict, plan_mode: str, trace_id: str | None
    ) -> list[dict]:
        """两轮对话完成搜索规划。

        第一轮：模型根据任务描述、天气、plan_mode生成搜索计划。
        第二轮：把第一轮产出原样喂回去，让模型自己检查有没有遗漏
                或过度扩展——这是它自己的责任，不是代码替它做判断。
        """
        system_prompt = build_fact_planning_prompt(
            day_weather=weather.get("day_weather", ""),
            day_temp=weather.get("day_temp", ""),
            day_wind=weather.get("day_wind", ""),
            plan_mode=plan_mode,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task},
        ]

        resp1 = await self._llm.call(
            trace_id=trace_id or "fact_plan",
            messages=messages,
            response_format={"type": "json_object"},
        )
        first_draft = resp1.choices[0].message.content or "{}"

        messages.append({"role": "assistant", "content": first_draft})
        messages.append({"role": "user", "content": FACT_PLANNING_REVIEW_PROMPT})

        resp2 = await self._llm.call(
            trace_id=trace_id or "fact_plan_review",
            messages=messages,
            response_format={"type": "json_object"},
        )
        raw = resp2.choices[0].message.content or "{}"
        parsed = json.loads(raw)
        return parsed.get("searches") or []

    # ── DAG 构建 ─────────────────────────────────────────────────────────

    def _build_dag(self, search_plan: list[dict]) -> list[dict]:
        steps = []
        sid = 1
        search_ids = []

        for item in search_plan:
            steps.append({
                "id": sid, "type": "search",
                "task": json.dumps(item, ensure_ascii=False),
                "depends_on": [],
            })
            search_ids.append(sid)
            sid += 1

        steps.append({
            "id": sid, "type": "batch_eta",
            "task": "batch_eta",
            "depends_on": search_ids,
        })
        return steps

    def _make_exec_func(self, toolset: FactToolset):
        async def _exec(task: str, context: dict) -> dict:
            if task == "batch_eta":
                poi_ids = []
                for dep_result in context.values():
                    if dep_result.status != "ok":
                        continue
                    output = dep_result.data.get("output") or {}
                    for poi in output.get("pois") or []:
                        if poi.get("id"):
                            poi_ids.append(poi["id"])
                if not poi_ids:
                    return {"results": []}
                raw = await toolset.get_distance_batch(poi_ids)
                return json.loads(raw)
            else:
                item = json.loads(task)
                raw = await toolset.search_pois(
                    keywords=item["keywords"],
                    is_restaurant=item.get("is_restaurant", False),
                )
                return json.loads(raw)
        return _exec

    # ── Synthesize：纯代码，不调LLM ─────────────────────────────────────

    def _synthesize(
        self, geocode_result, weather_result, results, steps
    ) -> FactData:
        """从DAG执行结果里整理出FactData —— 纯代码，不调LLM。

        这一步做的全是确定性操作(分类/去重/格式转换)，不需要语义理解：
        - is_restaurant 字段在搜索阶段已经标注好，分类是个if判断
        - 去重按 poi_id 做 set 操作
        - 格式化输出就是字典拼装
        """
        activities: list[dict] = []
        restaurants: list[dict] = []
        seen_activity_ids: set[str] = set()
        seen_restaurant_ids: set[str] = set()
        eta_map: dict[str, int | None] = {}

        for step in steps:
            if step["type"] != "batch_eta":
                continue
            res = results.get(step["id"])
            if not res or res.status != "ok":
                continue
            output = res.data.get("output") or {}
            for r in output.get("results") or []:
                if r.get("poi_id"):
                    eta_map[r["poi_id"]] = r.get("eta_minutes")

        for step in steps:
            if step["type"] != "search":
                continue
            res = results.get(step["id"])
            if not res or res.status != "ok":
                continue

            output = res.data.get("output") or {}
            is_restaurant = json.loads(step["task"]).get("is_restaurant", False)
            pois = output.get("pois") or []

            for p in pois:
                pid = p.get("id")
                if not pid:
                    continue

                target_list = restaurants if is_restaurant else activities
                seen_ids = seen_restaurant_ids if is_restaurant else seen_activity_ids

                if pid in seen_ids:
                    continue
                seen_ids.add(pid)

                target_list.append({
                    "id": pid,
                    "name": p.get("name") or "",
                    "type": p.get("type") or "",
                    "location": p.get("location") or "",
                    "rating": p.get("rating") or "",
                    "eta_minutes": eta_map.get(pid),
                })

        return FactData(
            origin_city=geocode_result.get("city") or "",
            origin_coordinates=geocode_result.get("coordinates") or "",
            weather={
                "city": weather_result.get("city") or "",
                "day_weather": weather_result.get("day_weather") or "",
                "day_temp": weather_result.get("day_temp") or "",
                "day_wind": weather_result.get("day_wind") or "",
            },
            activities=activities,
            restaurants=restaurants,
            waypoints=[],
        )

    def _extract_origin_hint(self, task: str) -> str:
        for line in task.split("\n"):
            if "出发地" in line:
                return line.split("：", 1)[-1].strip()
        return task[:50]