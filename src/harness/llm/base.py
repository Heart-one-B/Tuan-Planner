from abc import ABC, abstractmethod


class LLMClientBase(ABC):
    """Abstract interface for LLM clients.

    Implement this to swap providers (OpenAI, Anthropic, local, mock).
    The harness only depends on this interface — never on a concrete client.
    """

    @abstractmethod
    async def call(self, trace_id: str, stream: bool = False, **kwargs) -> object:
        """Make a chat completion call.

        Args:
            trace_id: Correlates this call in the tracing layer.
            stream:   If True, return an async iterable stream instead of
                      a full response object.
            **kwargs: Passed through to the underlying API
                      (messages, tools, temperature, etc.).

        Returns:
            OpenAI-SDK-compatible response object with a .choices attribute.
        """
        ...