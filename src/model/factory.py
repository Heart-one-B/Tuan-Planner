from abc import ABC, abstractmethod
from typing import Optional

from langchain_community.chat_models import ChatTongyi
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from src.utils.config_handler import model_conf


class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        pass


class ChatModelFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        provider = model_conf.get("provider", "dashscope")
        if provider == "dashscope":
            return self._generator_dashscope()
        if provider == "deepseek":
            return self._generator_deepseek()
        raise ValueError(f"Unsupported model provider: {provider}")

    def _generator_dashscope(self) -> BaseChatModel:
        api_key = model_conf.get("dashscope_api_key")
        if not api_key:
            raise ValueError("Missing dashscope_api_key in config/model.yml")
        return ChatTongyi(
            model=model_conf["chat_model_name"],
            dashscope_api_key=api_key,
            model_kwargs={"enable_thinking": False}
        )

    def _generator_deepseek(self) -> BaseChatModel:
        api_key = model_conf.get("deepseek_api_key")
        if not api_key:
            raise ValueError("Missing deepseek_api_key in config/model.yml")
        return ChatOpenAI(
            model=model_conf.get("deepseek_model_name", "deepseek-v4-flash"),
            api_key=api_key,
            base_url=model_conf.get("deepseek_base_url", "https://api.deepseek.com"),
            extra_body={"thinking": {"type": "disabled"}}
        )


class LazyChatModel:
    def __init__(self):
        self._model: Optional[BaseChatModel] = None

    def _get_model(self) -> BaseChatModel:
        if self._model is None:
            self._model = ChatModelFactory().generator()
        return self._model

    def invoke(self, *args, **kwargs):
        return self._get_model().invoke(*args, **kwargs)


chat_model = LazyChatModel()
