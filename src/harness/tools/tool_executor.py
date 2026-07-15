# harness/tool/tool_excutor.py

import asyncio
import inspect
import json
import logging
import time

from harness.tools.tool_definition import ToolDefinition
from harness.tools.exceptions import (
    ParamError, RetryableError, MissingInfoError, DataNotFoundError,
    ServiceUnavailableError, DegradedError, PartialResultError,
)

logger = logging.getLogger(__name__)


class ToolExecutor:
    """Tool registry and executor.

    default_timeout: 全局默认超时(秒),单个工具可通过 ToolDefinition.timeout
                     覆盖。超时不是框架自动重试——转成和其他工具异常同一套
                     【】前缀格式的消息喂回模型,由模型决定要不要换参数重试
                     (是否值得重试是语义判断,该留给模型,不该由框架替它决定)。
    """

    def __init__(self, default_timeout: float = 30.0):
        self._tools: dict[str, ToolDefinition] = {}
        self.default_timeout = default_timeout

    def register(self, tool: ToolDefinition) -> None:
        self._tools[tool.name] = tool
        logger.info(f"[ToolExecutor] registered: {tool.name}")

    def max_result_chars_for(self, name: str) -> int | None:
        """单工具卸载阈值查询(审查修复:此前 ToolDefinition.max_result_chars
        字段没有任何消费者,是死字段)。未注册的名字返回 None(用全局默认)。"""
        tool = self._tools.get(name)
        return tool.max_result_chars if tool else None

    @property
    def schemas(self) -> list[dict]:
        return [t.to_openai_schema() for t in self._tools.values()]

    async def execute(self, tool_call, trace_id: str = None, run_ctx=None) -> str:
        name = tool_call.function.name

        try:
            args = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError as e:
            msg = f"【参数错误,请调整后重试】参数不是合法 JSON:{e}"
            logger.warning(f"[ToolExecutor] bad JSON in {name}: {e}")
            return msg

        logger.info(f"[ToolExecutor] execute: {name}  args={args}")

        if name not in self._tools:
            msg = f"【参数错误，请调整后重试】工具 '{name}' 未注册"
            logger.error(f"[ToolExecutor] {msg}")
            return msg

        tool_def = self._tools[name]
        effective_timeout = tool_def.timeout if tool_def.timeout is not None else self.default_timeout

        tool_start = time.time()
        status = "success"
        result: str = ""

        try:
            func = tool_def.func
            call_args = dict(args)
            if run_ctx is not None and "run_ctx" in inspect.signature(func).parameters:
                call_args["run_ctx"] = run_ctx

            if asyncio.iscoroutinefunction(func):
                raw = await asyncio.wait_for(func(**call_args), timeout=effective_timeout)
            else:
                loop = asyncio.get_event_loop()
                raw = await asyncio.wait_for(
                    loop.run_in_executor(None, lambda: func(**call_args)),
                    timeout=effective_timeout,
                )

            result = str(raw)
            logger.info(f"[ToolExecutor] {name} ok  preview={result[:120]}")
            return result

        except asyncio.TimeoutError:
            status = "timeout"
            result = f"【执行超时,已中止,可尝试简化参数或换个方式重试】工具 '{name}' 在 {effective_timeout}s 内未完成"
            logger.warning(f"[ToolExecutor] TimeoutError in {name} (limit={effective_timeout}s)")
            return result

        except ParamError as e:
            status = "param_error"; result = f"【参数错误，请调整后重试】{e}"; return result
        except RetryableError as e:
            status = "retryable"; result = f"【暂时失败，可以重试一次】{e}"; return result
        except MissingInfoError as e:
            status = "missing_info"; result = f"【信息缺失，请询问用户】{e}"; return result
        except DataNotFoundError as e:
            status = "not_found"; result = f"【数据不存在】{e}"; return result
        except DegradedError as e:
            status = "degraded"; result = f"【服务降级，请使用兜底方案】{e}"; return result
        except ServiceUnavailableError as e:
            status = "unavailable"; result = f"【服务不可用，请勿重试】{e}"; return result
        except PartialResultError as e:
            status = "partial"; result = f"【结果不完整，请基于现有信息尽力回答】{e}"; return result
        except Exception as e:
            status = "error"; result = f"【系统错误，请勿重试】{e}"; return result
        finally:
            if trace_id:
                from harness.tracing import tracer
                duration = int((time.time() - tool_start) * 1000)
                tracer.record_tool_event(
                    trace_id=trace_id, tool_name=name, args=args,
                    result=result, duration_ms=duration, status=status,
                )