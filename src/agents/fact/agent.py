# agents/fact/agent.py
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field

from agents.fact.amap_types import label_of, spice_level
from agents.fact.schema import FactData
from agents.fact.tools import FactToolset
from agents.fact.prompt import build_fact_planning_prompt, FACT_PLANNING_REVIEW_PROMPT
from harness.agent.result import AgentResult
from harness.llm.base import LLMClientBase
from harness.tracing.span import Span

logger = logging.getLogger(__name__)

_MODE_ACTIVITY_ONLY = "activity_only"
_MODE_MEAL_ONLY = "meal_only"
_KNOWN_MODES = {_MODE_ACTIVITY_ONLY, _MODE_MEAL_ONLY, "activity_plus_meal",
                "family_with_kids"}

_FALLBACK_RESTAURANT_KEYWORDS = ["餐厅", "中餐"]
_FALLBACK_ACTIVITY_KEYWORDS = ["商场", "公园"]

# 可选餐厅少于这个数就算缺口。取 3 的依据：PlanningAgent 固定产出
# 3 个候选方案，少于 3 个可选餐厅时三个方案在"吃什么"这维必然重复，
# 用户看到的"三个选择"是假的。
# 如实记账：有推理依据但未经校准（Phase 5 的 C 类）。
_MIN_VIABLE_RESTAURANTS = 3

_MAX_REPLAN_ROUNDS = 1

FACT_REPLAN_PROMPT = """\
你刚才规划的搜索已经执行完毕，下面是**代码统计出的结果缺口**。
注意：你看不到具体搜到了哪些店，只能看到数量和缺口——这是刻意的，
你需要的只是"哪里不够"，不是"搜到了什么"。

{gaps}

请只针对上述缺口补充搜索关键词，不要重复已经搜过的词，
也不要为已经足够的类别再加词。

补搜关键词的选择原则：
- 某个词搜到 0 家 → 换更通用、更常见的说法（如"独立小馆子"→"家常菜"）
- 符合约束的餐厅不够 → 补充明确符合该约束的品类
  （如需要不辣的 → 粤菜、江浙菜、日料、菌汤、清汤类）
- 类别整体为空 → 用该类别最通用的词

如果你判断当前结果已经够用、或者补搜也不会有更好的结果，
返回空列表。不要为了填满而凑数。

严格输出 JSON，格式与之前一致：
{{"searches": [{{"keywords": "粤菜", "is_restaurant": true}}]}}
"""


@dataclass
class GapReport:
    """搜索结果的缺口，**纯代码算出来的**。

    这是"ReWOO + 一轮反馈"这个折中的核心：Planner 拿到的是这份
    百来字的摘要，不是几千 token 的 POI 列表。ReWOO 想省的那部分
    （观测不进上下文）完整保住，同时拿回最关键的那点反馈——
    "哪里不够"。

    为什么这点信息就够：Planner 要做的决策是"补搜什么关键词"，
    这个决策只需要知道缺口在哪，不需要知道搜到的每一家店叫什么。
    """
    empty_keywords: list[str] = field(default_factory=list)
    missing_categories: list[str] = field(default_factory=list)
    constraint_shortfall: str = ""
    searched: list[tuple[str, bool, int]] = field(default_factory=list)

    def has_gaps(self) -> bool:
        return bool(self.empty_keywords or self.missing_categories
                    or self.constraint_shortfall)

    def render(self) -> str:
        lines = ["## 已执行的搜索"]
        for kw, is_rest, n in self.searched:
            lines.append(f"- [{'餐厅' if is_rest else '活动'}] {kw}: {n} 家")
        lines.append("\n## 缺口")
        if self.empty_keywords:
            lines.append(f"- 以下关键词一家都没搜到：{'、'.join(self.empty_keywords)}")
        if self.missing_categories:
            lines.append(f"- 以下类别候选池为空：{'、'.join(self.missing_categories)}")
        if self.constraint_shortfall:
            lines.append(f"- {self.constraint_shortfall}")
        if not self.has_gaps():
            lines.append("- 无")
        return "\n".join(lines)


class FactAgent:
    """事实收集 Agent —— ReWOO + 一轮代码驱动的反馈。

    【范式】标准 ReWOO：Planner 一次产出全部计划 → Worker 并行执行
    → Solver 汇总，观测**不回流**给 Planner。本实现比标准 ReWOO 更
    彻底：Solver（_synthesize）是纯代码，连汇总都不用模型。

    好处是 token 省、并行度高、延迟低；代价是零反馈。
    本实现在 Worker 和 Solver 之间插一个**纯代码的缺口检测**，
    只有检测到缺口时才回 Planner 一轮，回给它的是百来字的摘要
    （GapReport），不是观测本身：

        Plan ──→ Execute(并行) ──→ [代码检测缺口]
                                      │无缺口 → Solve（常见路径，零额外成本）
                                      └有缺口 → Replan(小 prompt) → Execute → Solve

    观测始终没有进入上下文，进入的只是对观测的**结构化断言**。
    """

    def __init__(self, llm_client: LLMClientBase):
        self._llm = llm_client

    # ── 入口 ──────────────────────────────────────────────────────────

    async def run(
        self,
        task: str,
        plan_mode: str = "",
        explicit_restaurant_keywords: list[str] | None = None,
        explicit_activity_keywords: list[str] | None = None,
        origin_city: str = "",
        no_spicy: bool = False,
        parent_span: Span | None = None,
    ) -> AgentResult:
        """origin_city：收窄 geocode 的检索范围。

        修实测 bug：「春熙路」→ 云南昭通、「陆家嘴」→ 昆明。
        中国重名地名极多，不给城市范围就是碰运气——而这个"运气"
        决定了整个候选池落在哪个省，错了之后下游全部工作都白做。

        no_spicy：刻意只暴露这一个偏好参数，不做通用 constraints
        字典——typecode 能可靠执行的偏好目前只有辣度（050102 川菜 /
        050108 湘菜 是确定的）。"不要连锁"没有可靠判据（品牌码只覆盖
        十几个大牌，店名括号实测 12/15 命中但「馨苑餐厅(四川大学
        江安校区店)」显然不是连锁）。开一个通用接口然后只有一个参数
        能用，是假装有能力。
        """
        span = Span.begin("fact", task, parent=parent_span)
        try:
            result = await self._run_inner(
                task, plan_mode,
                explicit_restaurant_keywords or [],
                explicit_activity_keywords or [],
                origin_city, no_spicy, span,
            )
            span.end(result.summary,
                     status="success" if result.status in ("ok", "partial") else "error")
            return result
        except Exception as e:
            span.end(str(e), status="error")
            raise

    async def _run_inner(
        self, task: str, plan_mode: str,
        explicit_restaurants: list[str], explicit_activities: list[str],
        origin_city: str, no_spicy: bool, span: Span,
    ) -> AgentResult:
        toolset = FactToolset()
        trace_id = span.trace_id

        # ── 第一步：geocode（带城市范围）──
        try:
            geocode_raw = await toolset.geocode(
                self._extract_origin(task), city=origin_city,
            )
            geocode_result = json.loads(geocode_raw)
            if not geocode_result.get("coordinates"):
                return AgentResult(status="error",
                                  summary=f"出发地解析失败: {geocode_result}", data={})
        except Exception as e:
            return AgentResult(status="error", summary=f"geocode 失败: {e}", data={})

        # ── 第二步：天气 ──
        try:
            weather_result = json.loads(await toolset.get_weather(toolset.origin_city))
        except Exception:
            # 天气拿不到不该让整次收集作废——它只影响 Planning 的
            # 室内外偏好，不影响候选池本身。
            weather_result = {}

        # ── 第三步：Plan（两轮自审，都在 Planner 内部）──
        try:
            raw_plan = await self._plan_searches(task, weather_result, plan_mode, trace_id)
        except Exception as e:
            return AgentResult(status="error", summary=f"搜索规划失败: {e}", data={})

        searches = self._flatten_plan(raw_plan)

        # ── 用户明确点名：代码强制覆盖，不指望模型自觉 ──────────────
        #
        # 【这不是提示，是命令】实测过"指望模型遵守 prompt 约束"会
        # 失败：记忆注入"不吃辣"之后，模型在同一次输出里生成过
        # restaurant_keywords=['清淡川菜']——它试图调和两个信号，
        # 办法是造一个自相矛盾的词。模型不是不听话，是约束以
        # "prompt 里的一行字"这种软形式存在时，它有裁量权去"融合"
        # 而不是"服从"。
        #
        # explicit 模式下没有裁量空间的理由：constraint_node 已经
        # 判定"用户点名了"，此时目标就是精确检索，不需要创造力。
        overrode_rest = self._override(searches, explicit_restaurants, True)
        overrode_act = self._override(searches, explicit_activities, False)
        searches = (
            [s for s in searches
             if not (explicit_restaurants and s["is_restaurant"])
             and not (explicit_activities and not s["is_restaurant"])]
            + [{"keywords": k, "is_restaurant": True} for k in explicit_restaurants]
            + [{"keywords": k, "is_restaurant": False} for k in explicit_activities]
        )

        searches, repairs = self._ensure_mode_coverage(
            searches, plan_mode, task,
            skip_restaurant=bool(explicit_restaurants),
            skip_activity=bool(explicit_activities),
        )
        if not searches:
            return AgentResult(status="empty",
                              summary="未规划出任何有效搜索关键词", data={})

        # ── 第四步：Execute（并行）──
        outcomes = await self._run_searches(toolset, searches)

        # ── 第四步半：代码检测缺口 → 必要时补一轮 ──
        replans: list[str] = []
        # explicit 模式不补搜：用户点名了要什么，搜不到就是搜不到，
        # 替他补一个别的是擅自改需求。此时正确的行为是如实报告，
        # 让用户自己决定。
        if not (overrode_rest and overrode_act):
            gaps = self._detect_gaps(outcomes, plan_mode, no_spicy,
                                    skip_restaurant=overrode_rest,
                                    skip_activity=overrode_act)
            if gaps.has_gaps():
                logger.info(f"[FactAgent] 检测到缺口，补搜一轮：\n{gaps.render()}")
                extra = self._dedupe_against(await self._replan(gaps, trace_id), searches)
                # 补搜也不许碰被 explicit 锁定的那一侧
                if overrode_rest:
                    extra = [s for s in extra if not s["is_restaurant"]]
                if overrode_act:
                    extra = [s for s in extra if s["is_restaurant"]]
                if extra:
                    replans = [s["keywords"] for s in extra]
                    outcomes += await self._run_searches(toolset, extra)
                    logger.info(f"[FactAgent] 补搜 {replans} 完成")

        failed = [o for o in outcomes if o["error"] is not None]
        if len(failed) == len(outcomes):
            detail = "; ".join(f"{o['plan']['keywords']}: {o['error']}"
                              for o in failed[:3])
            return AgentResult(status="error",
                              summary=f"所有搜索任务均失败（{detail}）", data={})

        # ETA 放在补搜合并之后：一次批量调用覆盖全部 POI
        eta_map = await self._run_batch_eta(toolset, outcomes)

        # ── 第五步：Solve（纯代码）──
        try:
            fact_data = self._synthesize(geocode_result, weather_result, outcomes, eta_map)
        except Exception as e:
            return AgentResult(status="error", summary=f"结果提炼失败: {e}", data={})

        return self._verdict(fact_data, plan_mode, failed, repairs, replans,
                             overrode_rest, explicit_restaurants,
                             overrode_act, explicit_activities, no_spicy)

    @staticmethod
    def _override(searches: list[dict], explicit: list[str], is_restaurant: bool) -> bool:
        """只做日志与返回标志，实际替换在调用处（一次性重建列表更清楚）。"""
        if not explicit:
            return False
        before = [s["keywords"] for s in searches
                  if s["is_restaurant"] == is_restaurant]
        logger.info(f"[FactAgent] {'餐饮' if is_restaurant else '活动'}"
                   f" explicit 模式，关键词 {before} → {explicit}")
        return True

    # ── plan_mode 一致性：校验 + 修复 ──────────────────────────────

    def _ensure_mode_coverage(
        self, searches: list[dict], plan_mode: str, task: str,
        skip_restaurant: bool = False, skip_activity: bool = False,
    ) -> tuple[list[dict], list[str]]:
        """搜索**执行之前**的盲补：模型标错 is_restaurant 时兜底。

        实测过的静默失败：用户明确要"吃个饭"、plan_mode=
        activity_plus_meal，最终候选池是「27 个活动 + 0 家餐厅」，
        而 status 仍然是 ok——模型把 is_restaurant 标成了 False，
        搜到的餐厅全被 _synthesize 归进了 activities。

        补搜而不是报错，理由和 repair_orphan_tool_calls 的"先认错，
        再继续"是同一条：宁可丑陋地补一次通用搜索，也不该让整次
        规划因为模型的一次标注失误而作废。但补救必须留痕。

        与 _detect_gaps 的分工：这里是"计划层面"的检查（连搜都没搜，
        类别就是空的），缺口检测是"结果层面"的（搜了但不够）。
        前者只能盲补通用词，后者能带着具体缺口让模型想。
        """
        if plan_mode not in _KNOWN_MODES:
            return searches, []

        needs_rest = plan_mode != _MODE_ACTIVITY_ONLY
        needs_act = plan_mode != _MODE_MEAL_ONLY
        has_rest = any(s["is_restaurant"] for s in searches)
        has_act = any(not s["is_restaurant"] for s in searches)
        repairs: list[str] = []

        if needs_rest and not has_rest and not skip_restaurant:
            kws = self._fallback(task, "餐饮搜索关键词", _FALLBACK_RESTAURANT_KEYWORDS)
            searches += [{"keywords": k, "is_restaurant": True} for k in kws]
            repairs.append(f"餐厅（补搜 {'、'.join(kws)}）")
            logger.warning(f"[FactAgent] plan_mode={plan_mode} 要求餐厅，但计划里"
                          f"一条餐厅搜索都没有。盲补 {kws}")

        if needs_act and not has_act and not skip_activity:
            kws = self._fallback(task, "活动搜索关键词", _FALLBACK_ACTIVITY_KEYWORDS)
            searches += [{"keywords": k, "is_restaurant": False} for k in kws]
            repairs.append(f"活动（补搜 {'、'.join(kws)}）")
            logger.warning(f"[FactAgent] plan_mode={plan_mode} 要求活动，但计划里"
                          f"一条活动搜索都没有。盲补 {kws}")

        return searches, repairs

    # ── 缺口检测（纯代码，不调 LLM）──────────────────────────────

    def _detect_gaps(
        self, outcomes: list[dict], plan_mode: str, no_spicy: bool,
        skip_restaurant: bool = False, skip_activity: bool = False,
    ) -> GapReport:
        report = GapReport()
        restaurants: list[dict] = []

        for o in outcomes:
            if o["error"] is not None:
                continue
            pois = (o["output"] or {}).get("pois") or []
            kw, is_rest = o["plan"]["keywords"], o["plan"]["is_restaurant"]
            report.searched.append((kw, is_rest, len(pois)))
            if not pois:
                # explicit 锁定的那一侧搜不到不算"缺口"——那是需要
                # 如实报告给用户的事实，不是要补搜的漏洞
                if not (is_rest and skip_restaurant) and not (not is_rest and skip_activity):
                    report.empty_keywords.append(kw)
            elif is_rest:
                restaurants.extend(pois)

        n_act = sum(n for _, is_r, n in report.searched if not is_r)
        n_rest = sum(n for _, is_r, n in report.searched if is_r)

        if plan_mode in _KNOWN_MODES:
            if plan_mode != _MODE_ACTIVITY_ONLY and n_rest == 0 and not skip_restaurant:
                report.missing_categories.append("餐厅")
            if plan_mode != _MODE_MEAL_ONLY and n_act == 0 and not skip_activity:
                report.missing_categories.append("活动")

        # 约束执行：用 typecode 判定，不用店名。
        # 店名判定已被证伪——云南的「爱尚菌·野生菌火锅」是菌汤、
        # 「正宗富源酸菜土猪脚火锅」是酸菜，靠"火锅"二字判辣必然误判。
        if no_spicy and restaurants and not skip_restaurant:
            uniq = {p.get("id"): p for p in restaurants if p.get("id")}
            viable = [p for p in uniq.values()
                     if spice_level(p.get("type") or "") != "spicy"]
            if len(viable) < _MIN_VIABLE_RESTAURANTS:
                spicy = [f"{p.get('name')}[{label_of(p.get('type') or '')}]"
                        for p in uniq.values()
                        if spice_level(p.get("type") or "") == "spicy"][:3]
                report.constraint_shortfall = (
                    f"用户不吃辣，但 {len(uniq)} 家餐厅里只有 {len(viable)} 家"
                    f"非辣系（需要至少 {_MIN_VIABLE_RESTAURANTS} 家）"
                    + (f"，辣系的有：{'、'.join(spicy)}" if spicy else "")
                )
        return report

    async def _replan(self, gaps: GapReport, trace_id: str) -> list[dict]:
        """带着缺口摘要回 Planner 一轮。

        独立的短对话，不带原来那轮规划的 messages——Planner 不需要
        重看自己之前的推理，它只需要知道现在缺什么。带上历史会让这次
        调用的成本接近第一轮，ReWOO 省下的就又还回去了。
        """
        try:
            resp = await self._llm.call(
                trace_id=trace_id,
                messages=[{"role": "user",
                          "content": FACT_REPLAN_PROMPT.format(gaps=gaps.render())}],
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content or "{}"
            return self._flatten_plan(json.loads(raw).get("searches") or [])
        except Exception as e:
            logger.warning(f"[FactAgent] 补搜规划失败，跳过: {e}")
            return []

    @staticmethod
    def _dedupe_against(extra: list[dict], done: list[dict]) -> list[dict]:
        seen = {(s["keywords"], s["is_restaurant"]) for s in done}
        out = []
        for s in extra:
            key = (s["keywords"], s["is_restaurant"])
            if key not in seen:
                seen.add(key)
                out.append(s)
        return out

    # ── 定级 ──────────────────────────────────────────────────────

    def _verdict(
        self, fact_data: FactData, plan_mode: str, failed: list,
        repairs: list[str], replans: list[str],
        overrode_rest: bool, explicit_rest: list[str],
        overrode_act: bool, explicit_act: list[str],
        no_spicy: bool,
    ) -> AgentResult:
        """定级看候选池实际内容，不看"搜索有没有报错"。
        status 描述的是"任务完成得怎么样"，不是"HTTP 调用成没成功"。"""
        n_act, n_rest = len(fact_data.activities), len(fact_data.restaurants)
        needs_rest = plan_mode in _KNOWN_MODES and plan_mode != _MODE_ACTIVITY_ONLY
        needs_act = plan_mode in _KNOWN_MODES and plan_mode != _MODE_MEAL_ONLY

        gaps: list[str] = []
        rest_miss = overrode_rest and n_rest == 0
        act_miss = overrode_act and n_act == 0
        if needs_rest and n_rest == 0 and not rest_miss:
            gaps.append("餐厅候选池为空")
        if needs_act and n_act == 0 and not act_miss:
            gaps.append("活动候选池为空")

        viable_note = ""
        if no_spicy and n_rest:
            viable = [r for r in fact_data.restaurants
                     if spice_level(r.type or "") != "spicy"]
            if len(viable) < _MIN_VIABLE_RESTAURANTS:
                viable_note = (f"，不辣的餐厅仅 {len(viable)}/{n_rest} 家，"
                              f"补搜后仍不足")

        parts = [f"搜到{n_act}个活动候选、{n_rest}家餐厅候选"]
        if overrode_rest:
            parts.append(f"（餐饮按点名精确检索：{'、'.join(explicit_rest)}）")
        if overrode_act:
            parts.append(f"（活动按点名精确检索：{'、'.join(explicit_act)}）")
        if replans:
            parts.append(f"（缺口补搜：{'、'.join(replans)}）")
        if repairs:
            parts.append(f"（类别盲补：{'；'.join(repairs)}）")
        if failed:
            parts.append(f"，{len(failed)}个搜索失败")
        if gaps:
            parts.append(f"，但{'、'.join(gaps)}")
        if viable_note:
            parts.append(viable_note)
        if rest_miss:
            parts.append(f"，用户点名的「{'、'.join(explicit_rest)}」附近未搜到")
        if act_miss:
            parts.append(f"，用户点名的「{'、'.join(explicit_act)}」附近未搜到")

        summary = "".join(parts)
        degraded = bool(gaps or rest_miss or act_miss or failed or repairs or viable_note)
        if degraded:
            logger.warning(f"[FactAgent] {summary}")
        return AgentResult(status="partial" if degraded else "ok",
                          summary=summary, data=fact_data.model_dump())

    # ── Planner ───────────────────────────────────────────────────

    async def _plan_searches(
        self, task: str, weather: dict, plan_mode: str, trace_id: str
    ) -> list[dict]:
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
        resp1 = await self._llm.call(trace_id=trace_id, messages=messages,
                                    response_format={"type": "json_object"})
        first_draft = resp1.choices[0].message.content or "{}"

        messages.append({"role": "assistant", "content": first_draft})
        messages.append({"role": "user", "content": FACT_PLANNING_REVIEW_PROMPT})
        resp2 = await self._llm.call(trace_id=trace_id, messages=messages,
                                    response_format={"type": "json_object"})
        return json.loads(resp2.choices[0].message.content or "{}").get("searches") or []

    # ── 计划归一 ──────────────────────────────────────────────────

    @staticmethod
    def _normalize_keywords(raw) -> list[str]:
        """归一成字符串列表。修的是一个不报错的数据损坏：原实现把
        模型返回的 list 直接传给要求 str 的 search_pois，高德收到的是
        "['川菜', '本帮菜']" 这种字面量。"""
        if isinstance(raw, str):
            s = raw.strip()
            return [s] if s else []
        if isinstance(raw, (list, tuple)):
            out: list[str] = []
            for k in raw:
                if isinstance(k, str) and k.strip() and k.strip() not in out:
                    out.append(k.strip())
            return out
        return []

    def _flatten_plan(self, raw_plan: list) -> list[dict]:
        searches: list[dict] = []
        seen: set[tuple[str, bool]] = set()
        for item in raw_plan or []:
            if not isinstance(item, dict):
                continue
            is_restaurant = bool(item.get("is_restaurant", False))
            for kw in self._normalize_keywords(item.get("keywords")):
                key = (kw, is_restaurant)
                if key in seen:
                    continue
                seen.add(key)
                searches.append({"keywords": kw, "is_restaurant": is_restaurant})
        return searches

    # ── Worker：并行执行 ──────────────────────────────────────────

    @staticmethod
    async def _run_searches(toolset: FactToolset, searches: list[dict]) -> list[dict]:
        """并发发出。单个失败就地捕获成 error 字段，不让 gather 整批炸。"""
        async def one(plan: dict) -> dict:
            try:
                raw = await toolset.search_pois(
                    keywords=plan["keywords"], is_restaurant=plan["is_restaurant"])
                return {"plan": plan, "output": json.loads(raw), "error": None}
            except Exception as e:
                return {"plan": plan, "output": None, "error": str(e)}

        return list(await asyncio.gather(*(one(p) for p in searches)))

    @staticmethod
    async def _run_batch_eta(toolset: FactToolset, outcomes: list[dict]) -> dict:
        poi_ids: list[str] = []
        for o in outcomes:
            if o["error"] is not None:
                continue
            for poi in (o["output"] or {}).get("pois") or []:
                if poi.get("id"):
                    poi_ids.append(poi["id"])
        if not poi_ids:
            return {}
        try:
            payload = json.loads(await toolset.get_distance_batch(poi_ids))
        except Exception:
            return {}
        return {r["poi_id"]: r.get("eta_minutes")
                for r in payload.get("results") or [] if r.get("poi_id")}

    # ── Solver：纯代码 ────────────────────────────────────────────

    def _synthesize(self, geocode_result: dict, weather_result: dict,
                   outcomes: list[dict], eta_map: dict) -> FactData:
        activities: list[dict] = []
        restaurants: list[dict] = []
        seen_act: set[str] = set()
        seen_rest: set[str] = set()

        for o in outcomes:
            if o["error"] is not None:
                continue
            is_restaurant = o["plan"]["is_restaurant"]
            target = restaurants if is_restaurant else activities
            seen = seen_rest if is_restaurant else seen_act

            for p in (o["output"] or {}).get("pois") or []:
                pid = p.get("id")
                if not pid or pid in seen:
                    continue
                seen.add(pid)
                target.append({
                    "id": pid,
                    "name": p.get("name") or "",
                    "type": p.get("type") or "",
                    "location": p.get("location") or "",
                    "rating": p.get("rating") or "",
                    # cost 不写 or ""：None 的语义是"高德没给价格"，
                    # 不是"免费"，混淆会让预算判断静默通过。
                    "cost": p.get("cost"),
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
            activities=activities, restaurants=restaurants, waypoints=[],
        )

    # ── 从 task 文本取字段 ────────────────────────────────────────

    @staticmethod
    def _extract_line(task: str, label: str) -> str:
        for line in task.split("\n"):
            if line.startswith(label):
                return line.split("：", 1)[-1].strip()
        return ""

    def _extract_origin(self, task: str) -> str:
        """取出发地。任务文本里的形式是「出发地：春熙路（成都）」——
        括号里的城市已经通过 origin_city 参数单独传了，这里要去掉，
        否则地名带上括号会影响检索。"""
        raw = self._extract_line(task, "出发地")
        if not raw:
            return task[:50]
        for sep in ("（", "("):
            if sep in raw:
                raw = raw.split(sep, 1)[0]
        return raw.strip()

    def _fallback(self, task: str, label: str, default: list[str]) -> list[str]:
        raw = self._extract_line(task, label)
        kws = [k.strip() for k in raw.split("、") if k.strip()]
        return kws[:2] if kws else list(default)