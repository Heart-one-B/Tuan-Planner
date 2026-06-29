from __future__ import annotations

from harness.llm.openai_client import OpenAIClient
from harness.tracing import configure_storage, SQLiteTraceStorage
from src.utils.config_handler import model_conf

_DASHSCOPE_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
_tracing_done = False


def setup_tracing() -> None:
    global _tracing_done
    if not _tracing_done:
        configure_storage(SQLiteTraceStorage(db_path="data/trace.db"))
        _tracing_done = True


def build_llm_client() -> OpenAIClient:
    api_key = model_conf.get("dashscope_api_key")
    if not api_key:
        raise ValueError("缺少 dashscope_api_key，请检查 config/model.yml")
    return OpenAIClient(
        api_key=api_key,
        base_url=_DASHSCOPE_BASE,
        model_name=model_conf["chat_model_name"],
        extra_body={"enable_thinking": False},
    )