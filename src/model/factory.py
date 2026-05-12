#模型较多时，用该代码文件封装模型
from abc import ABC, abstractmethod
from typing import Optional

from langchain_community.chat_models import ChatTongyi
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from src.utils.config_handler import model_conf


"""
    BaseModelFactory 继承自ABC，是一个抽象基类
    作用：定义规范，禁止直接实例化，唯一作用就是被继承， 是工厂类的模板
"""
class BaseModelFactory(ABC):
    """
        abstractmethod:强制子类实现
        强制约束：任何继承了 BaseModelFactory 的子类，必须自己写一个 generator 方法。

    """
    @abstractmethod
    # Optional[Embeddings | BaseChatModel] —— 返回值类型注解，这是 Python 的 Type Hinting（类型提示）。
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        pass

class ChatModelFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        return ChatTongyi(
            model=model_conf["chat_model_name"],
            dashscope_api_key=model_conf["dashscope_api_key"],
            model_kwargs={"enable_thinking": False}
        )


chat_model=ChatModelFactory().generator()




