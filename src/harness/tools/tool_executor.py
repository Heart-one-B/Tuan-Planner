import asyncio
import json
import logging
import time

from harness.tools.tool_definition import ToolDefinition
from harness.tools.exceptions import (
    ParamError,
    RetryableError,
    MissingInfoError,
    DataNotFoundError,
    ServiceUnavailableError,
    DegradedError,
    PartialResultError,
)

logger = logging.getLogger(__name__)


class ToolExecutor:
    """Tool registry and executor.

    Responsibilities:
    - register()  — add a ToolDefinition at startup
    - schemas     — expose OpenAI function schemas to the LLM
    - execute()   — run a tool_call, always return an LLM-readable string

    No tools are pre-registered here. Business code calls register()
    after instantiation (see examples/robot_assistant/registry.py).
    """

    def __init__(self):
        self._tools: dict[str, ToolDefinition] = {}

    # ── registration ──────────────────────────────────────────────────────────

    def register(self, tool: ToolDefinition) -> None:
        self._tools[tool.name] = tool
        logger.info(f"[ToolExecutor] registered: {tool.name}")

    # ── schema exposure ───────────────────────────────────────────────────────

    @property
    def schemas(self) -> list[dict]:
        """All tool schemas in OpenAI function-calling format."""
        return [t.to_openai_schema() for t in self._tools.values()]

    # ── execution ─────────────────────────────────────────────────────────────

    async def execute(self, tool_call, trace_id: str = None) -> str:
        """Execute one tool_call object from an LLM response.

        Always returns a string — never raises. Exceptions are caught and
        converted to prefixed messages that guide the LLM's next action:

          【参数错误，请调整后重试】    ParamError
          【暂时失败，可以重试一次】    RetryableError
          【信息缺失，请询问用户】      MissingInfoError
          【数据不存在】               DataNotFoundError
          【服务降级，请使用兜底方案】  DegradedError
          【服务不可用，请勿重试】      ServiceUnavailableError
          【结果不完整，请基于现有信息尽力回答】  PartialResultError
          【系统错误，请勿重试】        unexpected Exception
        """
        name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)

        logger.info(f"[ToolExecutor] execute: {name}  args={args}")

        if name not in self._tools:
            msg = f"【参数错误，请调整后重试】工具 '{name}' 未注册"
            logger.error(f"[ToolExecutor] {msg}")
            return msg

        tool_start = time.time()
        status = "success"
        result: str = ""    # initialised before try — avoids UnboundLocalError in finally

        try:
            func = self._tools[name].func
            if asyncio.iscoroutinefunction(func):
                raw = await func(**args)
            else:
                loop = asyncio.get_event_loop()
                raw = await loop.run_in_executor(None, lambda: func(**args))

            result = str(raw)
            logger.info(f"[ToolExecutor] {name} ok  preview={result[:120]}")
            return result

        except ParamError as e:
            status = "param_error"
            result = f"【参数错误，请调整后重试】{e}"
            logger.warning(f"[ToolExecutor] ParamError in {name}: {e}")
            return result

        except RetryableError as e:
            status = "retryable"
            result = f"【暂时失败，可以重试一次】{e}"
            logger.warning(f"[ToolExecutor] RetryableError in {name}: {e}")
            return result

        except MissingInfoError as e:
            status = "missing_info"
            result = f"【信息缺失，请询问用户】{e}"
            logger.warning(f"[ToolExecutor] MissingInfoError in {name}: {e}")
            return result

        except DataNotFoundError as e:
            status = "not_found"
            result = f"【数据不存在】{e}"
            logger.warning(f"[ToolExecutor] DataNotFoundError in {name}: {e}")
            return result

        except DegradedError as e:
            status = "degraded"
            result = f"【服务降级，请使用兜底方案】{e}"
            logger.error(f"[ToolExecutor] DegradedError in {name}: {e}")
            return result

        except ServiceUnavailableError as e:
            status = "unavailable"
            result = f"【服务不可用，请勿重试】{e}"
            logger.error(f"[ToolExecutor] ServiceUnavailableError in {name}: {e}")
            return result

        except PartialResultError as e:
            status = "partial"
            result = f"【结果不完整，请基于现有信息尽力回答】{e}"
            logger.warning(f"[ToolExecutor] PartialResultError in {name}: {e}")
            return result

        except Exception as e:
            status = "error"
            result = f"【系统错误，请勿重试】{e}"
            logger.error(f"[ToolExecutor] unexpected error in {name}: {e}", exc_info=True)
            return result

        finally:
            if trace_id:
                from harness.tracing import tracer
                duration = int((time.time() - tool_start) * 1000)
                tracer.record_tool_event(
                    trace_id=trace_id,
                    tool_name=name,
                    args=args,
                    result=result,
                    duration_ms=duration,
                    status=status,
                )