# harness/tools/tool_definition.py
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class ToolDefinition:
    """Binds a JSON schema with the callable that implements it.

    timeout: 单个工具的超时秒数(覆盖 ToolExecutor 的全局默认值)。
             None 表示使用 ToolExecutor.default_timeout。
             同步函数超时后 execute() 会按时返回,但底层线程池里的
             调用本身不会被真正杀死(Python 不支持强制终止线程),
             只是不再等待它——这是已知的物理限制,不是 bug。

    【第一刀新增,任务 1.1】终止判定重构(见 agent/finish_tool.py 的
    模块头注释)新增的两个字段:
      terminal    这个工具是不是"结构化收口"工具。AgentLoop 处理完
                  一批工具调用后,据此检查 run_ctx.state["result"]
                  是否被写入,不再需要 TerminationPolicy.owns() 这层
                  单独的分拣接口。
      read_only   本次只加字段、不消费(见设计方案"明确不做的事":
                  流式并行工具执行与 Defer 链式审批语义冲突,留给
                  以后单独设计)。fail-closed 默认 False,照抄 CC
                  的保守默认。
    """
    name: str
    description: str
    parameters: dict
    func: Callable
    required: list[str] = field(default_factory=list)
    timeout: float | None = None
    max_result_chars: int | None = None   # 单工具的卸载阈值,None 用全局默认
                                          # (与 timeout 同构:工具级覆盖全局)
    exempt_from_offload: bool = False
    terminal: bool = False
    read_only: bool = False

    def to_openai_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters,
                    "required": self.required,
                },
            },
        }