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
    """
    name: str
    description: str
    parameters: dict
    func: Callable
    required: list[str] = field(default_factory=list)
    timeout: float | None = None
    max_result_chars: int | None = None   # 单工具的卸载阈值,None 用全局默认
                                          # (与 timeout 同构:工具级覆盖全局)

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