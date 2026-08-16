# evals/run.py
"""评估集 runner。

【设计原则】
1. **一条挂掉不能停整批**。一条挂掉就停摆的评估集没人会反复跑，
   不反复跑的评估集等于不存在。

2. **未知的断言键必须报错**。写错键名却静默跳过，那条用例会
   "通过"而其实什么都没测——本项目撞过四次的"schema 是静默
   过滤器"，不能在评估工具里再犯。

3. **"链路没跑完"归 error 不归 failed**。实测踩过：15 条用例因
   intent_node 抛异常导致链路没执行，obs 全是 null，被判成 failed
   ——看起来像系统很烂，实际是这次跑挂了。更糟的是
   `waypoint_handled: false` 会**假通过**（链路没跑，waypoints
   当然是空的）。空数据让断言意外满足，和被删掉的
   activity_not_contains 是同一类问题。

4. **失败时必须能诊断**。obs 里带 errors / task_log，
   否则一堆失败摆在面前看不出为什么。

5. **断言不许靠日志字符串匹配**。文案一改就静默失效，
   而且失效时恒假，看起来像系统坏了。

用法：
    python evals/run.py --layer l1
    python evals/run.py --layer l2 --case L2-01
    python evals/run.py --dry-run --layer all
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# 项目根 = 本文件的上一级。不依赖 cwd：PyCharm 默认把工作目录设成
# 脚本所在目录，命令行又通常在项目根——同一份代码在两种运行方式下
# 相对路径含义不同，这个项目已经为此撞过好几次。
_ROOT = Path(__file__).resolve().parent.parent
os.chdir(_ROOT)
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import yaml  # noqa: E402

CASES_FILE = Path("evals/cases.yaml")
OUT_DIR = Path("data/experiments")
DB_PATH = Path("data/trace.db")
CASE_TIMEOUT_S = 180


class PipelineNotRun(Exception):
    """链路没跑完（节点抛异常 / 追问未收敛）。

    判 error 而不是 failed：断言没有得到被判定的机会，此时任何
    "通过"都是空数据造成的假通过，任何"失败"都不指向系统缺陷。
    把这两种情况和真实的断言失败混在一起统计，分数就失去意义了。
    """


# ══════════════════════════════════════════════════════════════
# 搜索记录器
# ══════════════════════════════════════════════════════════════

@dataclass
class SearchRecorder:
    """记录实际发出的搜索。抓行为而不是模型输出的计划——
    _flatten_plan 会展开、去重，explicit 模式还会被代码强制覆盖，
    只看计划会看到一个和真实行为不一致的东西。"""
    calls: list = field(default_factory=list)

    def restaurant_keywords(self) -> list[str]:
        return [c["keywords"] for c in self.calls if c["is_restaurant"]]

    def activity_keywords(self) -> list[str]:
        return [c["keywords"] for c in self.calls if not c["is_restaurant"]]


class recording:
    def __init__(self, rec: SearchRecorder):
        self.rec = rec

    def __enter__(self):
        from agents.fact.tools import FactToolset
        self._orig = FactToolset.search_pois
        rec = self.rec

        async def wrapper(ts_self, keywords, is_restaurant=False):
            entry = {"keywords": keywords, "is_restaurant": bool(is_restaurant),
                     "count": 0, "error": None}
            try:
                raw = await self._orig(ts_self, keywords, is_restaurant)
                entry["count"] = len((json.loads(raw) or {}).get("pois") or [])
                return raw
            except Exception as e:
                entry["error"] = f"{type(e).__name__}: {e}"
                raise
            finally:
                rec.calls.append(entry)

        FactToolset.search_pois = wrapper
        return self.rec

    def __exit__(self, *exc):
        from agents.fact.tools import FactToolset
        FactToolset.search_pois = self._orig
        return False


# ══════════════════════════════════════════════════════════════
# 断言注册表
# ══════════════════════════════════════════════════════════════
# 准入判据：**这条断言失败时，能不能推出"系统有缺陷"？**
# 不能就不该存在——只会制造噪声，久了让人习惯性忽略红色。

CHECKERS: dict = {}


def checker(name: str):
    def deco(fn):
        CHECKERS[name] = fn
        return fn
    return deco


def _eq(expected, actual, label: str):
    return actual == expected, f"{label}: 期望={expected!r} 实际={actual!r}"


def _contains_all(expected: list, actual: list, label: str):
    actual = actual or []
    missing = [e for e in (expected or [])
               if not any(e in str(a) for a in actual)]
    return not missing, f"{label}: 期望含={expected} 实际={actual} 缺={missing}"


# ── 分类类（值域封闭，字面比较是稳的）──

@checker("restaurant_intent")
def _c_rest_intent(exp, obs):
    return _eq(exp, obs.get("restaurant_intent"), "restaurant_intent")


@checker("restaurant_intent_final")
def _c_rest_intent_final(exp, obs):
    return _eq(exp, obs.get("restaurant_intent"), "最终 restaurant_intent")


@checker("scenario")
def _c_scenario(exp, obs):
    return _eq(exp, obs.get("scenario"), "scenario")


@checker("plan_mode")
def _c_plan_mode(exp, obs):
    return _eq(exp, obs.get("plan_mode"), "plan_mode")


@checker("route")
def _c_route(exp, obs):
    return _eq(exp, obs.get("last_route"), "最后一轮 route")


@checker("people_count")
def _c_people(exp, obs):
    return _eq(exp, obs.get("people_count"), "people_count")


@checker("clarification_needed")
def _c_clar(exp, obs):
    return _eq(exp, bool(obs.get("clarification_needed")), "clarification_needed")


@checker("is_leisure_planning")
def _c_leisure(exp, obs):
    return _eq(exp, bool(obs.get("is_leisure_planning")), "is_leisure_planning")


@checker("origin_city")
def _c_city(exp, obs):
    return _eq(exp, obs.get("origin_city"), "origin_city")


@checker("adjust_history_len")
def _c_adjust_len(exp, obs):
    return _eq(exp, len(obs.get("adjust_history") or []), "adjust_history 长度")


# ── 包含类（用户原话会出现在结果里的场合才用）──

@checker("diet_preference_contains")
def _c_diet(exp, obs):
    return _contains_all(exp, obs.get("diet_preference"), "diet_preference")


@checker("must_avoid_contains")
def _c_avoid(exp, obs):
    return _contains_all(exp, obs.get("must_avoid"), "must_avoid")


@checker("missing_slots_contains")
def _c_slots(exp, obs):
    return _contains_all(exp, obs.get("missing_slots"), "missing_slots")


@checker("waypoint_keywords_contains")
def _c_waypoints(exp, obs):
    return _contains_all(exp, obs.get("waypoint_keywords"), "waypoint_keywords")


@checker("start_time")
def _c_start(exp, obs):
    return _eq(exp, obs.get("start_time"), "start_time")


@checker("end_time")
def _c_end(exp, obs):
    return _eq(exp, obs.get("end_time"), "end_time")


# ── 结构性 ──

@checker("has_restaurant")
def _c_has_rest(exp, obs):
    return _eq(exp, bool(obs.get("selected_restaurants")), "方案里有餐厅")


@checker("produced_plan")
def _c_produced(exp, obs):
    return _eq(exp, bool(obs.get("produced_plan")), "出了方案")


@checker("no_search_expansion")
def _c_no_expand(exp, obs):
    """explicit 模式下，实搜的餐厅关键词必须 ⊆ 用户点名的。

    和"搜没搜到"解耦：用户点"佛跳墙"、成都真有闽菜馆搜到了，
    status=ok 并不是缺陷。真正该断言的是**不擅自把用户点名的
    品类扩展成别的**。

    子串双向匹配而不是相等：模型可能把「日料」填成「日本料理」，
    实搜也可能带后缀（「火锅店」）——这些都不构成"扩展"。
    """
    if not exp:
        return True, "（不判定）"
    kws = obs.get("searched_restaurant_keywords") or []
    explicit = obs.get("restaurant_explicit_types") or []
    if not explicit:
        return False, f"restaurant_explicit_types 为空，无法判定；实搜={kws}"
    strayed = [k for k in kws if not any(e in k or k in e for e in explicit)]
    return not strayed, f"实搜={kws} 点名={explicit} 擅自扩展={strayed}"


@checker("time_coverage_min")
def _c_time_cov(exp, obs):
    """⚠️ 阈值未经校准。cases.yaml 统一取 0.3（最宽松），只用来抓
    极端情况。首轮实测分布 38%/57%/64%，看起来 0.3 偏松，
    但校准需要更多样本——先定一个漂亮的 0.6 然后发现大面积失败，
    那是在测我的想象。"""
    cov = obs.get("time_coverage")
    if cov is None:
        return False, "time_coverage 无法计算（方案缺时间字段）"
    return cov >= exp, f"时段覆盖率={cov:.0%} 要求≥{exp:.0%}（阈值未校准）"


@checker("max_eta")
def _c_max_eta(exp, obs):
    etas = [r.get("eta_minutes") for r in (obs.get("selected_restaurants") or [])
            if r.get("eta_minutes") is not None]
    if not etas:
        return True, "无 eta 数据（不判定）"
    return max(etas) <= exp, f"最大 eta={max(etas)} 要求≤{exp}"


@checker("waypoint_handled")
def _c_waypoint(exp, obs):
    """⚠️ 天然容易假通过：期望 false 时只要 waypoints 为空就满足，
    而链路没跑完时它也是空的。靠 PipelineNotRun 提前拦成 error
    才安全。"""
    actual = bool(obs.get("waypoints"))
    return actual == exp, f"waypoints={obs.get('waypoints')}"


@checker("searched_activity_contains_any")
def _c_searched_act(exp, obs):
    """断言**搜索关键词**含某类活动，而不是最终 POI 名。

    【为什么不断言 POI 名】实测：用户要"博物馆"，方案排的是
    「林跃艺术收藏馆」——它就是对的结果，只是叫"收藏馆"。
    补词（收藏馆/艺术馆/文化馆/科技馆/XX院…）只是把下次误报推迟，
    **用词表穷举 POI 的命名方式本质上做不到**。

    而"有没有去搜"是系统的责任，"搜出来叫什么"是高德的。
    断言该落在前者——这样它的失败才真的指向系统缺陷。
    """
    kws = obs.get("searched_activity_keywords") or []
    hit = [k for k in kws if any(w in k for w in (exp or []))]
    return bool(hit), f"期望搜索词含之一={exp} 实搜活动词={kws} 命中={hit}"


@checker("activity_contains_any")
def _c_act_any(exp, obs):
    """方案里至少有一项命中候选词之一。

    ⚠️ 保留但**不推荐新用例使用**——它有 POI 命名不可穷举的问题
    （见 searched_activity_contains_any 的说明）。只在"确实要断言
    最终方案内容"、且候选词覆盖面足够时才用。

    刻意**没有** activity_not_contains（否定式）：要求方案不含
    "商场"而实际排了「茂业天地」，名字里没这两个字，断言通过——
    但它就是个商场。假通过比误报危险得多。
    """
    names = obs.get("plan_item_names") or []
    hit = [n for n in names if any(w in n for w in (exp or []))]
    return bool(hit), f"期望命中之一={exp} 方案项={names} 命中={hit}"


# ── 偏好执行 ──

@checker("no_spicy")
def _c_no_spicy_flag(exp, obs):
    return _eq(exp, bool(obs.get("no_spicy")), "no_spicy 约束已启用")


@checker("no_spicy_violation")
def _c_no_spicy_viol(exp, obs):
    """选中餐厅里有没有**确定辛辣**的（050102 川菜 / 050108 湘菜）。
    存疑类目（火锅店 050117、云贵菜）不算违规——与生产代码
    violates_no_spicy(strict=False) 同一口径。评估用一套判定、
    生产用另一套是自欺。"""
    from agents.fact.amap_types import label_of, spice_level
    rests = obs.get("selected_restaurants") or []
    if not rests:
        return False, "方案里没有餐厅，无法判定"
    bad = [f"{r.get('name')}[{label_of(r.get('type') or '')}]"
           for r in rests if spice_level(r.get("type") or "") == "spicy"]
    return bool(bad) == exp, f"辛辣餐厅={bad or '无'}"


# ── 多轮 ──

@checker("plan_changed")
def _c_changed(exp, obs):
    """比逐轮的方案项，不比 plan_id——PlanningAgent 每次重新生成
    plan_1/2/3，换了方案 id 也可能不变。"""
    snapshots = obs.get("turn_items") or []
    changed = any(snapshots[i] != snapshots[i - 1] for i in range(1, len(snapshots)))
    return changed == exp, f"逐轮方案项={snapshots}"


@checker("eval_score_min")
def _c_score(exp, obs):
    score = obs.get("eval_score")
    if score is None:
        return False, "没有评分"
    return score >= exp, f"评分={score} 要求≥{exp}"


@checker("eval_reason_not_contains")
def _c_reason_not(exp, obs):
    """禁止词取的是修复前实际出现过的原文，不是编的措辞——
    命中即意味着回归。"""
    reason = obs.get("eval_reason") or ""
    hit = [t for t in (exp or []) if t in reason]
    return not hit, f"命中禁止词={hit}；评语={reason[:150]}"


@checker("eta_decreased")
def _c_eta_dec(exp, obs):
    before, after = obs.get("eta_before"), obs.get("eta_after")
    if before is None or after is None:
        return False, f"eta 数据不全 before={before} after={after}"
    return (after < before) == exp, f"eta {before} → {after}"


# ══════════════════════════════════════════════════════════════
# 观测提取
# ══════════════════════════════════════════════════════════════

def _hhmm(s) -> int | None:
    if not s or ":" not in str(s):
        return None
    try:
        h, m = str(s).split(":")[:2]
        return int(h) * 60 + int(m)
    except ValueError:
        return None


def _time_coverage(plan: dict, ctx: dict) -> float | None:
    start, end = _hhmm(ctx.get("start_time")), _hhmm(ctx.get("end_time"))
    if start is None or end is None or end <= start:
        return None
    covered = 0
    for item in plan.get("timeline") or []:
        a, b = _hhmm(item.get("time")), _hhmm(item.get("end_time"))
        if a is not None and b is not None and b > a:
            covered += b - a
    return min(1.0, covered / (end - start))


def _selected_plan(state: dict) -> dict:
    outputs = state.get("agent_outputs") or {}
    pid = ((outputs.get("evaluation") or {}).get("data") or {}).get("selected_plan_id") or ""
    cands = ((outputs.get("planning") or {}).get("data") or {}).get("candidates") or []
    return next((c for c in cands if isinstance(c, dict) and c.get("id") == pid), {})


def _selected_restaurants(state: dict) -> list[dict]:
    """从候选池按 id 取回完整 POI（含 typecode / cost）。
    不解析 display_text——那是渲染过的文本，靠它断言既脆弱，
    又会把渲染 bug 和业务 bug 混在一起。"""
    plan = _selected_plan(state)
    if not plan:
        return []
    fact = ((state.get("agent_outputs") or {}).get("fact") or {}).get("data") or {}
    by_id = {r["id"]: r for r in (fact.get("restaurants") or [])
             if isinstance(r, dict) and r.get("id")}
    out = []
    for step in plan.get("steps") or []:
        pid = step.get("poi_id") or step.get("id")
        if pid and pid in by_id:
            out.append(by_id[pid])
    return out


def _wants_no_spicy(prefs: dict) -> bool:
    """与 fact_node 同一判据。不靠日志字符串匹配——文案一改断言
    就静默失效，而且失效时恒假，看起来像系统坏了。"""
    signals = list(prefs.get("diet") or []) + list(prefs.get("avoid") or [])
    words = ("不辣", "不吃辣", "不能吃辣", "清淡", "微辣", "少辣", "忌辣", "重辣", "辣")
    return any(w in item for item in signals for w in words)


# evals/run.py 的 _guard_pipeline 替换版
#
# 直接替换 run.py 里同名函数即可，其余不变。

def _guard_pipeline(state: dict, case_id: str) -> None:
    """链路跑完了吗？没跑完直接抛，判 error 不判 failed。

    【v4 修：判据从"intent 是否为空"扩到"有没有不可恢复错误"】
    v3 只检查 intent。实测漏掉了 L2-13：403 配额耗尽发生在
    evaluation 阶段，此时 intent 已经成功产出，guard 放行——
    然后 `waypoint_handled: false` **假通过**了（waypoints 是空的，
    但那是因为链路半途死了，不是因为系统真的没处理 waypoint）。

    假通过是"彻底的沉默"：误判成失败还会被人追查，假通过什么都
    不会发生，而它会进 baseline，之后每次 diff 都带着它。

    两条判据，任一命中即判 error：

      ① errors 里有 recoverable=False 的条目
         不可恢复的定义就是"这次失败重试也不会成功"——配额耗尽、
         代码缺陷、模型输出不合格。断言没有得到公平判定的机会，
         此时"通过"是空数据造成的，"失败"不指向系统缺陷。

      ② intent 为空
         intent_node 的 except 分支返回的 dict 里没有 intent 键，
         链路从那里就断了。这条 ① 通常也能覆盖，保留是因为它
         判定更早、错误信息更直接。

    刻意**不**把"没出方案"当作判据：restaurant_intent=none 的
    用例、或输入本就是闲聊的用例，正常结束时就没有方案，那是
    合法结果。用"有没有产出"做门槛会把它们全误判成 error。
    """
    errs = state.get("errors") or []
    logs = state.get("task_log") or []

    fatal = [e for e in errs if not e.get("recoverable", False)]
    if fatal:
        detail = "; ".join(f"[{e.get('node')}] {str(e.get('error'))[:200]}"
                          for e in fatal)
        raise PipelineNotRun(f"链路含不可恢复错误：{detail}")

    if not state.get("intent"):
        detail = ("; ".join(f"[{e.get('node')}] {e.get('error')}" for e in errs)
                  or "（无错误记录）")
        raise PipelineNotRun(
            f"链路未执行完：intent 为空。errors={detail}；task_log={logs[-3:]}"
        )


def _obs_from_state(state: dict, rec: SearchRecorder, produced: bool) -> dict:
    intent = state.get("intent") or {}
    ctx = state.get("plan_context") or {}
    outputs = state.get("agent_outputs") or {}
    fact = (outputs.get("fact") or {}).get("data") or {}
    eval_data = (outputs.get("evaluation") or {}).get("data") or {}
    plan = _selected_plan(state)
    prefs_intent = intent.get("preferences") or {}

    return {
        "restaurant_intent": ctx.get("restaurant_intent") or intent.get("restaurant_intent"),
        "restaurant_explicit_types": intent.get("restaurant_explicit_types") or [],
        "scenario": ctx.get("scenario") or intent.get("scenario"),
        "plan_mode": ctx.get("plan_mode"),
        "clarification_needed": intent.get("clarification_needed"),
        "missing_slots": intent.get("missing_slots") or [],
        "people_count": (intent.get("participants") or {}).get("people_count"),
        "diet_preference": prefs_intent.get("diet_preference") or [],
        "must_avoid": prefs_intent.get("must_avoid") or [],
        "no_spicy": _wants_no_spicy(ctx.get("preferences") or {}),
        "origin_city": fact.get("origin_city"),
        "waypoints": fact.get("waypoints") or [],
        "fact_status": (outputs.get("fact") or {}).get("status"),
        "searched_restaurant_keywords": rec.restaurant_keywords(),
        "searched_activity_keywords": rec.activity_keywords(),
        "selected_plan_id": eval_data.get("selected_plan_id"),
        "selected_restaurants": _selected_restaurants(state),
        "plan_item_names": [i.get("item") or i.get("label") or ""
                            for i in (plan.get("timeline") or [])],
        "time_coverage": _time_coverage(plan, ctx),
        "eval_score": eval_data.get("selected_score"),
        "eval_reason": eval_data.get("selected_reason") or "",
        "adjust_history": state.get("adjust_history") or [],
        "last_route": state.get("feedback_route"),
        "produced_plan": produced,
        # 失败时最需要的诊断信息
        "errors": state.get("errors") or [],
        "task_log": state.get("task_log") or [],
    }


# ══════════════════════════════════════════════════════════════
# 三层执行
# ══════════════════════════════════════════════════════════════

async def run_l1(case: dict) -> dict:
    """只跑 IntentAgent。1 次 LLM 调用，不碰高德、不走全链路。"""
    from agents.intent.agent import IntentAgent
    from src.model.factory import build_llm_client

    d = (await IntentAgent(llm_client=build_llm_client()).parse(
        user_input=case["input"], trace_id=f"eval-{case['id']}",
    )).model_dump()

    prefs = d.get("preferences") or {}
    t = d.get("time") or {}
    p = d.get("participants") or {}
    return {
        "restaurant_intent": d.get("restaurant_intent"),
        "restaurant_explicit_types": d.get("restaurant_explicit_types") or [],
        "scenario": d.get("scenario"),
        "people_count": p.get("people_count"),
        "clarification_needed": d.get("clarification_needed"),
        "missing_slots": d.get("missing_slots") or [],
        "is_leisure_planning": d.get("is_leisure_planning"),
        "diet_preference": prefs.get("diet_preference") or [],
        "must_avoid": prefs.get("must_avoid") or [],
        "start_time": t.get("start_time"),
        "end_time": t.get("end_time"),
        "waypoint_keywords": [w.get("keyword") for w in (d.get("waypoint_requests") or [])],
    }


def _clarify_answer(case: dict) -> str:
    """澄清追问的自动回答：**把用例原话再说一遍**。

    早期版本用一句固定回答（"朋友聚会，四个人，下午两点到晚上十点，
    从四川大学江安校区出发"），它会**覆盖用例本身的设定**——
    L2-07 是"两个人约会"（couple）、L2-11 是"六个人同事聚餐"（team），
    自动答一句"朋友聚会四个人"，场景和人数就被改掉了，断言测的
    就不再是用例想测的东西。

    重复原话是安全的：用户原话里本来就包含全部他愿意提供的信息，
    重说一遍不会引入用例之外的设定。如果原话确实缺槽位，追问三轮后
    clarification_node 会强制填默认值放行——那时"这条用例的输入
    缺槽位"这个事实会体现在结果里，而不是被伪造的回答掩盖。
    """
    return case.get("input") or (case.get("turns") or [""])[0]


async def run_l2(case: dict) -> dict:
    from src.session_runner import Session

    rec = SearchRecorder()
    with recording(rec):
        s = Session(session_id=f"eval-{case['id']}-{uuid.uuid4().hex[:4]}",
                    auto_clarify_answer=_clarify_answer(case))
        r = await s.say(case["input"])
    _guard_pipeline(r.state, case["id"])
    return _obs_from_state(r.state, rec, produced=r.ok)


async def run_l3(case: dict) -> dict:
    """多轮。逐轮记录方案项，plan_changed 靠它判定。"""
    from src.session_runner import Session

    rec = SearchRecorder()
    turn_items, etas, obs = [], [], {}
    with recording(rec):
        s = Session(session_id=f"eval-{case['id']}-{uuid.uuid4().hex[:4]}",
                    auto_clarify_answer=_clarify_answer(case))
        for text in case["turns"]:
            r = await s.say(text)
            _guard_pipeline(r.state, case["id"])
            obs = _obs_from_state(r.state, rec, produced=r.ok)
            turn_items.append(list(obs.get("plan_item_names") or []))
            e = [x.get("eta_minutes") for x in (obs.get("selected_restaurants") or [])
                 if x.get("eta_minutes") is not None]
            etas.append(max(e) if e else None)

    obs["turn_items"] = turn_items
    obs["eta_before"], obs["eta_after"] = (
        (etas[0], etas[-1]) if len(etas) >= 2 else (None, None))
    return obs


RUNNERS = {"l1_intent": run_l1, "l2_e2e": run_l2, "l3_multiturn": run_l3}


# ══════════════════════════════════════════════════════════════
# 判定与汇总
# ══════════════════════════════════════════════════════════════

def judge(case: dict, obs: dict) -> tuple[str, list[dict]]:
    results, unknown = [], False
    for key, expected in (case.get("expect") or {}).items():
        fn = CHECKERS.get(key)
        if fn is None:
            unknown = True
            results.append({"key": key, "passed": False,
                            "detail": f"未知断言键 '{key}'，检查 cases.yaml 拼写"})
            continue
        try:
            passed, detail = fn(expected, obs)
        except Exception as e:
            passed, detail = False, f"断言执行异常: {type(e).__name__}: {e}"
        results.append({"key": key, "passed": bool(passed), "detail": detail})

    if unknown:
        return "error", results
    return ("passed" if all(r["passed"] for r in results) else "failed"), results


async def run_case(layer: str, case: dict) -> dict:
    rec = {"id": case["id"], "layer": layer,
           "origin": case.get("origin", "constructed"),
           "input": case.get("input") or case.get("turns"),
           "expect": case.get("expect") or {}}
    try:
        obs = await asyncio.wait_for(RUNNERS[layer](case), timeout=CASE_TIMEOUT_S)
    except asyncio.TimeoutError:
        rec.update(verdict="error", error=f"超时 >{CASE_TIMEOUT_S}s", checks=[])
        return rec
    except PipelineNotRun as e:
        rec.update(verdict="error", error=str(e), checks=[])
        return rec
    except Exception as e:
        rec.update(verdict="error", error=f"{type(e).__name__}: {e}",
                   traceback=traceback.format_exc()[-1500:], checks=[])
        return rec

    verdict, checks = judge(case, obs)
    rec.update(verdict=verdict, checks=checks, obs=obs)
    return rec


def summarize(records: list[dict]) -> dict:
    def stat(rows):
        return {"n": len(rows),
                "passed": sum(1 for r in rows if r["verdict"] == "passed"),
                "failed": sum(1 for r in rows if r["verdict"] == "failed"),
                "error": sum(1 for r in rows if r["verdict"] == "error")}

    out = {"overall": stat(records), "by_layer": {}, "by_origin": {}}
    for layer in sorted({r["layer"] for r in records}):
        out["by_layer"][layer] = stat([r for r in records if r["layer"] == layer])
    for origin in sorted({r["origin"] for r in records}):
        out["by_origin"][origin] = stat([r for r in records if r["origin"] == origin])
    return out


def diff_baseline(records: list[dict], baseline_path: Path) -> list[str]:
    if not baseline_path.is_file():
        return [f"  （没有 baseline：{baseline_path}）"]
    base = {r["id"]: r["verdict"]
            for r in json.loads(baseline_path.read_text(encoding="utf-8"))["records"]}
    lines = []
    for r in records:
        old = base.get(r["id"])
        if old is None:
            lines.append(f"  🆕 {r['id']}: {r['verdict']}（baseline 中没有）")
        elif old != r["verdict"]:
            mark = "✅" if r["verdict"] == "passed" else "🔴"
            lines.append(f"  {mark} {r['id']}: {old} → {r['verdict']}")
    for cid in base:
        if not any(r["id"] == cid for r in records):
            lines.append(f"  ⬜ {cid}: 本次未跑")
    return lines or ["  （与 baseline 完全一致）"]


# ══════════════════════════════════════════════════════════════

async def main(args) -> int:
    if not CASES_FILE.is_file():
        print(f"❌ 找不到 {CASES_FILE}")
        return 1
    data = yaml.safe_load(CASES_FILE.read_text(encoding="utf-8"))

    layers = (["l1_intent", "l2_e2e", "l3_multiturn"] if args.layer == "all"
              else {"l1": ["l1_intent"], "l2": ["l2_e2e"],
                    "l3": ["l3_multiturn"]}[args.layer])

    selected = [(layer, c) for layer in layers for c in (data.get(layer) or [])
                if not args.case or c["id"] in args.case]

    if args.dry_run:
        print(f"用例总数 {len(selected)}")
        bad = 0
        for _layer, case in selected:
            unknown = [k for k in (case.get("expect") or {}) if k not in CHECKERS]
            if unknown:
                bad += 1
                print(f"  ❌ {case['id']}: 未知断言键 {unknown}")
            if not (case.get("input") or case.get("turns")):
                bad += 1
                print(f"  ❌ {case['id']}: 既没有 input 也没有 turns")
            if not (case.get("expect") or {}):
                bad += 1
                print(f"  ❌ {case['id']}: expect 为空，这条什么都没测")
        print(f"\n已注册断言（{len(CHECKERS)}）：{sorted(CHECKERS)}")
        print("\n✅ 用例文件校验通过" if not bad else f"\n❌ {bad} 处问题")
        return 0 if not bad else 1

    from harness.tracing import SQLiteTraceStorage, configure_storage
    from src.memory.wiring import configure_memory
    configure_storage(SQLiteTraceStorage(db_path=DB_PATH))

    if args.memory:
        from src.memory.wiring import build_memory_config
        configure_memory(build_memory_config(memory_dir=Path("data/memory")))
        print("记忆：ON")
    else:
        configure_memory(None)
        print("记忆：OFF（默认——记忆会让同一输入的结果依赖此前跑过"
              "什么，用例之间不再独立）")

    records = []
    consecutive_errors = 0
    for i, (layer, case) in enumerate(selected, 1):
        print(f"\n[{i}/{len(selected)}] {case['id']} ({layer})")
        r = await run_case(layer, case)
        records.append(r)
        print({"passed": "  ✅ passed", "failed": "  ❌ failed",
               "error": "  💥 error"}[r["verdict"]])
        for c in r.get("checks") or []:
            if not c["passed"]:
                print(f"     · {c['detail']}")
        if r.get("error"):
            print(f"     · {r['error']}")

        # 连续 error 意味着环境出了问题（配额、网络、模型换了），
        # 不是用例在失败。继续跑只是把钱和时间烧在同一个错误上。
        consecutive_errors = consecutive_errors + 1 if r["verdict"] == "error" else 0
        if consecutive_errors >= args.abort_after:
            print(f"\n🛑 连续 {consecutive_errors} 条 error，中止本批。")
            print("   连续异常通常是环境问题（配额/网络/模型配置），")
            print("   不是用例在失败——继续跑只会把钱烧在同一个错误上。")
            print("   已跑完的结果仍会落盘。")
            break

    s = summarize(records)
    o = s["overall"]
    print("\n" + "=" * 70)
    print("汇总")
    print("=" * 70)
    print(f"  已跑 {o['n']}/{len(selected)}  ✅{o['passed']}  "
          f"❌{o['failed']}  💥{o['error']}")
    for name, st in s["by_layer"].items():
        print(f"    {name:<14} {st['passed']}/{st['n']} 通过"
              f"（失败 {st['failed']}，异常 {st['error']}）")
    print("\n  按来源（real=真实出现过；constructed=为覆盖维度构造）：")
    for name, st in s["by_origin"].items():
        print(f"    {name:<14} {st['passed']}/{st['n']}")

    print("\n  与 baseline 对比：")
    for line in diff_baseline(records, Path(args.baseline)):
        print(line)

    # 跑偏信号（见任务简报 §六）。n<5 时不提示"全部通过"——
    # 单条跑必然全过，那个警告是噪声。
    if o["n"] >= 5 and o["passed"] == o["n"]:
        print("\n  ⚠️ 全部通过 —— 断言可能太松，缺少区分度，值得复查")
    if o["error"] and o["error"] >= o["n"] * 0.3:
        print(f"\n  ⚠️ {o['error']}/{o['n']} 条 error —— 先修环境或 runner，"
              f"这一批的分数不可用")
    for name, st in s["by_layer"].items():
        if st["n"] >= 5 and st["passed"] / st["n"] < 0.5:
            print(f"\n  ⚠️ {name} 通过率 <50% —— 先确认不是 runner 或"
                  f"期望值本身有问题，再怀疑系统")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"eval_{datetime.now():%Y%m%dT%H%M%S}.json"
    out.write_text(json.dumps({
        "timestamp": datetime.now().isoformat(),
        "cases_version": (data.get("meta") or {}).get("version"),
        "layers": layers, "memory": args.memory,
        "selected": len(selected), "summary": s, "records": records,
    }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n  结果 → {out}")
    print(f"  设为基线：copy \"{out}\" \"{OUT_DIR / 'baseline.json'}\"")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", choices=["l1", "l2", "l3", "all"], default="l1")
    ap.add_argument("--case", nargs="*", help="只跑指定 id")
    ap.add_argument("--dry-run", action="store_true", help="不调模型，只校验用例文件")
    ap.add_argument("--memory", action="store_true", help="开启记忆（默认关闭）")
    ap.add_argument("--baseline", default=str(OUT_DIR / "baseline.json"))
    ap.add_argument("--abort-after", type=int, default=3,
                    help="连续多少条 error 就中止本批（默认 3）")
    sys.exit(asyncio.run(main(ap.parse_args())))