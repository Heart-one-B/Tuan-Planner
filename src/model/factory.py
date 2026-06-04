from __future__ import annotations

from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Optional

from langchain_community.chat_models import ChatTongyi
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from src.utils.config_handler import model_conf
from src.utils.path_tool import get_abs_path


# ── 抽象基类 ───────────────────────────────────────────────────────────────────

class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        pass


# ── Chat 模型工厂 ──────────────────────────────────────────────────────────────

class ChatModelFactory(BaseModelFactory):
    """
    通过检测配置文件中存在哪个 api_key 来决定使用哪个模型客户端。
    - 存在 deepseek_api_key → DeepSeek（ChatOpenAI + 自定义 base_url）
    - 存在 dashscope_api_key → 通义千问（ChatTongyi）
    切换 provider 只需修改 config/model.yml，代码不用动。
    """

    def generator(self) -> BaseChatModel:
        if model_conf.get("deepseek_api_key"):
            return ChatOpenAI(
                model=model_conf["deepseek_model_name"],
                api_key=model_conf["deepseek_api_key"],
                base_url=model_conf.get("deepseek_base_url"),
                extra_body={"thinking": {"type": "disabled"}},
            )
        if model_conf.get("dashscope_api_key"):
            return ChatTongyi(
                model=model_conf["chat_model_name"],
                dashscope_api_key=model_conf["dashscope_api_key"],
                model_kwargs={"enable_thinking": False},
            )
        raise ValueError(
            "未找到可用的 API Key，请在 config/model.yml 中配置 "
            "dashscope_api_key 或 deepseek_api_key"
        )

class EmbeddingsFactory(BaseModelFactory):
    def generator(self) -> Embeddings:
        api_key = model_conf.get("dashscope_api_key")
        if not api_key:
            raise ValueError("Missing dashscope_api_key in config/model.yml")
        return DashScopeEmbeddings(
            model=model_conf["embedding_model_name"],
            dashscope_api_key=api_key,
        )

class RerankModelFactory(BaseModelFactory):
    """本地 HuggingFace 交叉编码器，用于检索后重排序。"""

    def generator(self) -> HuggingFaceCrossEncoder:
        local_path = get_abs_path(model_conf["rerank_model_path"])
        return HuggingFaceCrossEncoder(model_name=local_path)


# ── 模块级懒加载单例 ─────────────────────────────────────────────────────────────────
#首次调用时才初始化，避免 import 时加载所有模型
@lru_cache(maxsize=1)
def get_chat_model() -> BaseChatModel:
    return ChatModelFactory().generator()


@lru_cache(maxsize=1)
def get_embed_model() -> Embeddings:
    return EmbeddingsFactory().generator()


@lru_cache(maxsize=1)
def get_rerank_model() -> HuggingFaceCrossEncoder:
    return RerankModelFactory().generator()