# harness/agent/dag_executors.py
import asyncio
import inspect
from typing import Awaitable, Callable

from harness.agent.result import AgentResult
from harness.agent.structured_agent import StructuredAgent


class StructuredAgentExecutor:
    """把一个 StructuredAgent 适配成 DAG 执行器。

    关键适配:DAG 给的是 task(str) + context(上游 AgentResult),
    而 StructuredAgent.run 只吃 task(str)。
    这里把上游结果拼进 task 文本,作为本步的已知前提,再交给子 agent——
    因为子 agent 接收的是自然语言,这是最自然的喂法。
    """

    def __init__(self, agent: StructuredAgent):
        self.agent = agent

    async def execute(self, task: str, context: dict[int, AgentResult]) -> AgentResult:
        if context:
            upstream_lines = [
                f"- 前置步骤{dep}的结论: {res.summary};数据: {res.data}"
                for dep, res in context.items()
            ]
            task = (
                f"{task}\n\n已知以下前置信息(供你完成本步参考):\n"
                + "\n".join(upstream_lines)
            )
        # StructuredAgent.run 本就返回 AgentResult,接口天然对齐
        return await self.agent.run(task)


class FunctionExecutor:
    """把一个普通函数(同步或异步)适配成 DAG 执行器。

    用于那些不需要模型智能、直接算个结果的轻量节点。
    函数签名约定:func(task: str, context: dict) -> Any
    返回值会被包成 status=ok 的 AgentResult;抛异常则包成 error。
    """

    def __init__(self, func: Callable[[str, dict], object | Awaitable[object]]):
        self.func = func

    async def execute(self, task: str, context: dict[int, AgentResult]) -> AgentResult:
        try:
            if inspect.iscoroutinefunction(self.func):
                raw = await self.func(task, context)
            else:
                loop = asyncio.get_event_loop()
                raw = await loop.run_in_executor(None, lambda: self.func(task, context))
            return AgentResult(status="ok", summary=str(raw)[:200], data={"output": raw})
        except Exception as e:
            return AgentResult(status="error", summary=f"函数执行失败: {e}", data={})