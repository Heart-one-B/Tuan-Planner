# src/session_runner.py
"""一次会话的运行器：多轮对话 + 记忆提取。

【为什么需要这一层】
LangGraph 每轮独立 invoke，节点看不到跨轮的完整对话——而记忆提取
需要的恰恰是"这次会话从头到尾说了什么"。拥有多轮循环的那一层
（也就是这里）是唯一知道完整对话的地方，所以 dialogue 的累积和
提取的触发都放在这里，不塞进节点。

这同时也是根 span 的正确位置：LangGraph 没有"整图结束"的钩子，
span 在节点里开就没有可靠的地方关，漏掉任何一条终止路径都会让
trace 永远停在 running（Tracer.active_count 只涨不落 = 泄漏）。
放在 invoke 外面，开和关天然对齐一次请求。

实验脚本和交互式使用共用这一层，保证两者跑的是同一条路径——
实验测的必须是真实代码路径，不是为实验另写的一条。
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field

from src.graph.tracing import ROOT_TRACE_ID_KEY, begin_request_span
from src.graph.workflow import build_workflow
from src.memory.wiring import extract_session, memory_enabled
from src.model.factory import build_llm_client

logger = logging.getLogger(__name__)

MAX_CLARIFY_TURNS = 3


@dataclass
class TurnResult:
    """一轮对话的产出。state 原样保留，供下一轮续跑和断言检查。"""
    reply: str
    state: dict
    root_trace_id: str
    task_log: list = field(default_factory=list)
    errors: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """有回复且没有不可恢复的错误。可恢复错误（比如召回失败
        降级为不注入）不算失败——那正是降级设计要达到的效果。"""
        fatal = [e for e in self.errors if not e.get("recoverable", False)]
        return bool(self.reply) and not fatal


class Session:
    """一次会话。多轮之间共享 session_id 和累积的 dialogue。

    刻意不持有 MemoryConfig：记忆开关是全局的
    （src.memory.wiring.configure_memory），对照组只需在实验脚本
    开头调一次 configure_memory(None)，会话代码一行不改。
    单变量对比要求的正是这个。
    """

    def __init__(self, session_id: str | None = None, auto_clarify_answer: str | None = None):
        self.session_id = session_id or f"s-{uuid.uuid4().hex[:8]}"
        self.graph = build_workflow()
        self.dialogue: list[dict] = []
        self.state: dict = {"session_id": self.session_id}
        self.auto_clarify_answer = auto_clarify_answer

    async def say(self, user_input: str) -> TurnResult:
        """说一句话，拿到回复。

        澄清追问最多自动应答 MAX_CLARIFY_TURNS 轮（仅当构造时给了
        auto_clarify_answer）。clarification_node 达到自己的轮次上限
        后会强制填默认值放行，所以有限轮内必定收敛；超过上限仍在
        追问说明 intent 的槽位判定有问题，那是需要单独查的 bug，
        不该让脚本无限陪聊烧钱。
        """
        span = begin_request_span(user_input)
        self.dialogue.append({"role": "user", "content": user_input})

        try:
            result = await self.graph.ainvoke({
                **self.state,
                "user_input": user_input,
                "session_id": self.session_id,
                ROOT_TRACE_ID_KEY: span.trace_id,
            })

            for _ in range(MAX_CLARIFY_TURNS):
                pending = result.get("pending_clarification") or ""
                if not pending or not self.auto_clarify_answer:
                    break
                self.dialogue.append({"role": "assistant", "content": pending})
                self.dialogue.append({"role": "user", "content": self.auto_clarify_answer})
                result = await self.graph.ainvoke({
                    **result,
                    "user_input": self.auto_clarify_answer,
                    "session_id": self.session_id,
                    ROOT_TRACE_ID_KEY: span.trace_id,
                })

            reply = result.get("display_text") or result.get("final_message") or ""
            span.end(reply[:200], status="success")

        except Exception as e:
            span.end(str(e), status="error")
            logger.error(f"[session] {self.session_id} 轮次异常: {e}", exc_info=True)
            raise

        self.dialogue.append({"role": "assistant", "content": reply})
        # 整个 state 带进下一轮：agent_outputs / intent / plan_context /
        # surfaced_memory_names 都要跨轮存活，routing_node 靠
        # agent_outputs.evaluation 判断"有没有已选中方案"来区分
        # new_plan / adjust。
        self.state = dict(result)

        return TurnResult(
            reply=reply,
            state=result,
            root_trace_id=span.trace_id,
            task_log=list(result.get("task_log") or []),
            errors=list(result.get("errors") or []),
        )

    async def close(self) -> int:
        """会话结束：触发记忆提取，返回写入条数。

        提取放在这里而不是每轮之后：一条偏好往往要跨几轮才暴露
        完整（"川菜算了" + "这家是连锁的吧"），逐轮提取会把半截
        信息记成一条记忆。会话级别的边界是这个应用里最自然的
        提取粒度。

        提取失败返回 0，不抛异常——它是增益不是主流程。
        """
        if not memory_enabled():
            return 0
        return await extract_session(
            session_id=self.session_id,
            dialogue=self.dialogue,
            llm_client=build_llm_client(),
        )

    def selected_restaurants(self) -> list[dict]:
        """当前选中方案里的餐厅，供断言使用。

        从 planning 的候选池里按 evaluation 选中的 id 取回完整
        POI（含 name / type / cost）——display_text 是渲染过的
        文本，靠解析文本做断言既脆弱又会把渲染 bug 和业务 bug
        混在一起。断言必须打在结构化数据上。
        """
        outputs = self.state.get("agent_outputs") or {}
        eval_data = (outputs.get("evaluation") or {}).get("data") or {}
        selected_id = eval_data.get("selected_plan_id") or ""
        if not selected_id:
            return []

        candidates = ((outputs.get("planning") or {}).get("data") or {}).get("candidates") or []
        plan = next((c for c in candidates
                     if isinstance(c, dict) and c.get("id") == selected_id), None)
        if not plan:
            return []

        fact = (outputs.get("fact") or {}).get("data") or {}
        by_id = {r["id"]: r for r in (fact.get("restaurants") or [])
                 if isinstance(r, dict) and r.get("id")}

        out = []
        for step in plan.get("steps") or []:
            pid = step.get("poi_id") or step.get("id")
            if pid and pid in by_id:
                out.append(by_id[pid])
        return out


async def run_dialogue(
    turns: list[str],
    session_id: str | None = None,
    auto_clarify_answer: str | None = None,
    extract: bool = True,
) -> Session:
    """跑一段预设对话，返回 Session（含完整 state 和 dialogue）。
    实验脚本的主入口。"""
    session = Session(session_id=session_id, auto_clarify_answer=auto_clarify_answer)
    for text in turns:
        result = await session.say(text)
        logger.info(f"[session] {session.session_id} turn ok={result.ok}")
    if extract:
        await session.close()
    return session