from __future__ import annotations

from harness.agent.result import AgentResult
from harness.agent.structured_agent import StructuredAgent
from harness.llm.base import LLMClientBase
from harness.tools.tool_executor import ToolExecutor

from agents.fact.schema import FactData
from agents.fact.tools import FactToolset, build_fact_tools
from agents.fact.prompt import FACT_SYSTEM_PROMPT


class FactAgent:
    """事实收集 Agent。

    职责:接到一个规划需求描述,自主完成
        geocode 出发地 → 查天气 → 按需多轮搜索活动/餐厅/途径点
        → 给候选算车程 → finish 交出精炼的 FactData。

    这是对 StructuredAgent 的一层业务封装:
    - 注入 Fact 专属的工具集、prompt、输出契约(FactData)
    - 每次 run() 用全新的 FactToolset,保证会话间状态隔离
      (pool 暂存仅供 get_distance 按 id 查坐标,不跨请求复用)

    对外暴露 run(task) -> AgentResult,result.data 即 FactData 的字典形式。
    上层(规划流程 / Orchestrator)只依赖这个契约,不关心内部多少轮搜索。
    """

    def __init__(
        self,
        llm_client: LLMClientBase,
        max_tool_calls: int = 15,
    ):
        # llm 是无状态的,可在实例间复用;工具集有状态,每次 run 新建
        self._llm_client = llm_client
        self._max_tool_calls = max_tool_calls

    async def run(self, task: str, trace_id: str | None = None) -> AgentResult:
        """执行一次完整的事实收集。

        task: 规划需求的自然语言描述(由上层拼好,包含出发地/场景/
              时间/人数/偏好/途径需求等已知信息)。
        """
        # ── 每次请求独立的工具集(隔离 pool / origin / weather 状态)──────────
        toolset = FactToolset()
        executor = ToolExecutor()
        for tool in build_fact_tools(toolset):
            executor.register(tool)

        # ── 用统一的子 Agent 内核组装 ────────────────────────────────────────
        agent = StructuredAgent(
            llm_client=self._llm_client,
            tool_executor=executor,
            system_prompt=FACT_SYSTEM_PROMPT,
            output_schema=FactData,
            max_tool_calls=self._max_tool_calls,
            name="fact",
        )

        return await agent.run(task, trace_id=trace_id)