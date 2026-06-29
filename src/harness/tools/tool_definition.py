from dataclasses import dataclass, field
from typing import Callable


@dataclass
class ToolDefinition:
    """Binds a JSON schema with the callable that implements it.

    func 支持同步和异步 Callable:
    - 同步 func: 普通函数，ToolExecutor 直接调用
    - 异步 func: async 函数，ToolExecutor 用 await 调用
    """

    name: str
    description: str
    parameters: dict
    func: Callable
    required: list[str] = field(default_factory=list)

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