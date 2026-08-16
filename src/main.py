# main.py
"""本地行程规划助手 CLI。

【本次接入的 harness 能力】
  tracing   每轮一棵 Span 树，退出时汇总本次会话的真实 token 开销
  记忆      召回在 intent_node（每轮），提取在会话结束（一次）
  持久化    LangGraph SqliteSaver，跨进程续聊

【刻意没接的两个，以及原因 —— 不是遗漏】

  上下文管理（ContextManager）
    它管的是"一个 Agent 累积的消息历史"。本图的每个节点各自组装
    messages、调完即弃，跨轮传递的是 AgentState（结构化状态），
    不是消息列表。FactAgent 是 ReWOO：观测走数据结构不进上下文，
    从设计上就不累积。唯一会累积的是 OrchestratorAgent 的手写
    循环（工具返回是完整 FactData/PlanData JSON），但那需要先把
    它换成 harness Agent。
    在此之前接 ContextManager，只能得到一个永不触发的摆设。

  快照（RunSnapshot）
    它是为 harness Agent 循环设计的：messages + rounds +
    pending_tool_call_id。而这里要持久化的是 AgentState，形状
    完全不同。硬塞等于用错误的容器装数据，还会和 LangGraph 的
    checkpointer 形成同一件事的两本账——这个仓库已经为"两本账
    必然漂移"付过学费。
    对口的工具是 LangGraph 自己的 checkpointer，见下面
    _build_checkpointer()。
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import uuid
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s [%(name)s] %(message)s")

from harness.mcp.registry import close_all
from harness.tracing import SQLiteTraceStorage, configure_storage
from harness.tracing.analytics import request_cost
from src.graph.tracing import ROOT_TRACE_ID_KEY, begin_request_span
from src.graph.workflow import build_workflow
from src.memory.wiring import (
    build_memory_config, configure_memory, extract_session, list_memories,
)
from src.model.factory import build_llm_client

DB_PATH = Path("data/trace.db")
MEMORY_DIR = Path("data/memory")
CHECKPOINT_DB = Path("data/checkpoints.sqlite")

_DEFAULT_USER_INPUT = (
    "今天下午是空的，想和老婆孩子出去玩几个小时。老婆最近在减肥，孩子5岁。"
)

HELP = """\
可用命令：
  /memories        查看当前记忆库
  /trace           查看上一轮的 trace 树与开销
  /verbose         切换详细日志（各 Agent 的完整输出）
  /new             开一段新会话（清空对话，触发本段的记忆提取）
  q / exit         退出（退出前会提取本次会话的记忆）
直接输入需求开始规划；回车使用默认场景。
"""


def _section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def _build_checkpointer():
    """优先用 SqliteSaver（跨进程续聊），装不上就退回 MemorySaver。

    MemorySaver 的状态在进程退出时全部丢失——对一个"个人助手"
    来说这是真实的缺陷：昨天聊到一半的行程，今天打开就没了。
    SqliteSaver 需要额外的 langgraph-checkpoint-sqlite 包，
    装不上时**如实降级并告知**，不假装持久化成功了。
    """
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
        CHECKPOINT_DB.parent.mkdir(parents=True, exist_ok=True)
        cm = SqliteSaver.from_conn_string(str(CHECKPOINT_DB))
        saver = cm.__enter__()
        print(f"✅ 会话持久化：{CHECKPOINT_DB}（跨进程续聊可用）")
        return saver, cm
    except Exception as e:
        from langgraph.checkpoint.memory import MemorySaver
        print(f"⚠️  会话仅存内存，退出即丢失（SqliteSaver 不可用：{e}）")
        print("   需要跨进程续聊请安装：pip install langgraph-checkpoint-sqlite")
        return MemorySaver(), None


class Cli:
    def __init__(self, app, thread_id: str):
        self.app = app
        self.thread_id = thread_id
        self.verbose = False
        # 提取需要"这次会话从头到尾说了什么"。LangGraph 每轮独立
        # invoke，节点看不到跨轮对话；拥有多轮循环的这一层是唯一
        # 知道完整对话的地方，所以在这里累积。
        self.dialogue: list[dict] = []
        self.last_trace_id: str | None = None

    @property
    def thread(self) -> dict:
        return {"configurable": {"thread_id": self.thread_id}}

    async def turn(self, user_input: str) -> None:
        # 每轮一棵树。span 在 invoke 外面开和关：LangGraph 没有
        # "整图结束"的钩子，在节点里开就没有可靠的地方关，漏掉
        # 任何一条终止路径都会让 trace 停在 running（泄漏）。
        span = begin_request_span(user_input)
        self.last_trace_id = span.trace_id
        self.dialogue.append({"role": "user", "content": user_input})

        try:
            await self.app.ainvoke(
                {"user_input": user_input, ROOT_TRACE_ID_KEY: span.trace_id},
                config=self.thread,
            )
            state = self.app.get_state(self.thread).values

            # 澄清追问：同一次用户请求内的往返，共用同一个 root span
            while state.get("pending_clarification"):
                question = state["pending_clarification"]
                print(f"\n🤔 {question}")
                self.dialogue.append({"role": "assistant", "content": question})
                reply = input("> ").strip()
                while not reply:
                    reply = input("> ").strip()
                self.dialogue.append({"role": "user", "content": reply})
                await self.app.ainvoke(
                    {"user_input": reply, ROOT_TRACE_ID_KEY: span.trace_id},
                    config=self.thread,
                )
                state = self.app.get_state(self.thread).values

            reply = (state.get("display_text")
                     or state.get("final_message") or "")
            span.end(reply[:200], status="success")
        except Exception as e:
            span.end(str(e), status="error")
            raise

        self.dialogue.append({"role": "assistant", "content": reply})

        if self.verbose:
            self._dump(state)
        else:
            self._brief(state)

        _section("方案")
        print(reply or "（没有生成可展示的内容）")
        self._cost()

    # ── 展示 ──────────────────────────────────────────────────────

    def _brief(self, state: dict) -> None:
        """默认只打关键轨迹。完整输出用 /verbose——排错时有用，
        日常使用时几十 KB 的 JSON 会把方案本身淹掉。"""
        print("\n执行轨迹：")
        for line in state.get("task_log") or []:
            print(f"  · {line}")
        fatal = [e for e in (state.get("errors") or [])
                 if not e.get("recoverable", False)]
        if fatal:
            print("\n⚠️ 错误：")
            for e in fatal:
                print(f"  · [{e.get('node')}] {e.get('error')}")

    def _dump(self, state: dict) -> None:
        outputs = state.get("agent_outputs") or {}
        for name in ("fact", "planning", "evaluation"):
            o = outputs.get(name) or {}
            _section(f"[{name}]")
            print(f"status : {o.get('status')}")
            print(f"summary: {o.get('summary')}")
            if o.get("data"):
                print(json.dumps(o["data"], ensure_ascii=False, indent=2))
        self._brief(state)

    def _cost(self) -> None:
        if not self.last_trace_id:
            return
        try:
            cost = request_cost(SQLiteTraceStorage(db_path=DB_PATH),
                                self.last_trace_id)
            if cost and cost.get("total_tokens"):
                print(f"\n💰 本轮 {cost['total_tokens']} tokens / "
                      f"{cost['trace_count']} 个 Agent / "
                      f"{cost['duration_ms'] / 1000:.1f}s")
        except Exception:
            # 成本统计失败不该影响使用——观测层故障不拖垮主流程，
            # 这是 Tracer._safe_call 已经确立的原则，这里同样适用。
            pass

    def show_trace(self) -> None:
        if not self.last_trace_id:
            print("还没有可查看的 trace")
            return
        tree = SQLiteTraceStorage(db_path=DB_PATH).get_trace_tree(self.last_trace_id)
        if tree is None:
            print("找不到 trace")
            return

        def walk(node, depth=0):
            t = node.trace
            print(f"  {'  ' * depth}├─ {t.session_id:<14} "
                  f"llm={t.llm_call_count:<3} tool={t.tool_call_count:<3} "
                  f"{t.total_duration_ms}ms [{t.status}]")
            for c in node.children:
                walk(c, depth + 1)

        _section("上一轮 trace 树")
        walk(tree)
        p, c = tree.total_tokens()
        print(f"\n  prompt={p}  completion={c}  合计={p + c}")

    # ── 记忆 ──────────────────────────────────────────────────────

    async def flush_memory(self) -> None:
        """会话结束时提取一次。

        为什么是会话级而不是每轮：一条偏好常要跨几轮才暴露完整
        （"这个火锅算了" + "这家像连锁店"），逐轮提取会把半截信息
        记成一条记忆。
        """
        if len(self.dialogue) < 2:
            return
        print("\n💭 正在从本次对话提取记忆…")
        n = await extract_session(
            session_id=self.thread_id,
            dialogue=self.dialogue,
            llm_client=build_llm_client(),
        )
        mems = list_memories()
        print(f"   记忆库 {len(mems)} 条" + (f"（本次新增 {n}）" if n else ""))
        self.dialogue = []

    @staticmethod
    def show_memories() -> None:
        mems = list_memories()
        _section(f"记忆库（{len(mems)} 条）")
        if not mems:
            print("  （空）多聊几轮、说说你的偏好，结束时会自动记下来")
            return
        for m in mems:
            print(f"\n  ── {m.name} [{m.type}] {m.updated_at:%Y-%m-%d} ──")
            print(f"     {m.description}")
            body = (m.content or "").strip().replace("\n", "\n     ")
            print(f"     {body[:400]}")


async def main() -> None:
    print("=" * 60)
    print("本地行程规划助手")
    print("=" * 60)

    configure_storage(SQLiteTraceStorage(db_path=DB_PATH))
    configure_memory(build_memory_config(memory_dir=MEMORY_DIR))

    saver, saver_cm = _build_checkpointer()
    mems = list_memories()
    print(f"✅ 记忆库 {len(mems)} 条"
          + (f"：{[m.name for m in mems]}" if mems else "（空）"))
    print(f"\n{HELP}")

    app = build_workflow(saver)
    cli = Cli(app, thread_id=f"cli-{uuid.uuid4().hex[:8]}")

    try:
        while True:
            try:
                raw = input("\n你：\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if raw.lower() in ("q", "exit", "quit", "退出"):
                break
            if raw in ("/help", "?"):
                print(HELP)
                continue
            if raw == "/memories":
                cli.show_memories()
                continue
            if raw == "/trace":
                cli.show_trace()
                continue
            if raw == "/verbose":
                cli.verbose = not cli.verbose
                print(f"详细日志：{'开' if cli.verbose else '关'}")
                continue
            if raw == "/new":
                await cli.flush_memory()
                cli.thread_id = f"cli-{uuid.uuid4().hex[:8]}"
                print(f"已开始新会话 {cli.thread_id}")
                continue

            if not raw:
                raw = _DEFAULT_USER_INPUT
                print(f"（默认场景：{raw}）")

            try:
                await cli.turn(raw)
            except KeyboardInterrupt:
                print("\n（本轮已中断）")
            except Exception as e:
                print(f"\n[ERROR] {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
    finally:
        # 退出前提取：这次聊出来的偏好如果不落盘，下次就白聊了。
        # 放在 finally 里，Ctrl+C 退出也能保住。
        try:
            await cli.flush_memory()
        except Exception as e:
            print(f"（记忆提取失败，本次对话未能留下记忆：{e}）")
        await close_all()
        if saver_cm is not None:
            try:
                saver_cm.__exit__(None, None, None)
            except Exception:
                pass
        print("\n再见！")


if __name__ == "__main__":
    asyncio.run(main())