"""Provider interface.

The agent talks to an LLM only through this one method. To add another
provider (OpenAI, Anthropic, a local model...), implement ``generate`` in a
new module and pass the instance to the agent.
"""

from abc import ABC, abstractmethod

from pydantic import BaseModel


class LLMProvider(ABC):
    """Minimal text-generation interface."""

    @abstractmethod
    def generate(
        self,
        *,
        system_instruction: str,
        prompt: str,
        response_schema: type[BaseModel] | None = None,
        temperature: float = 0.2,
    ) -> str:
        """Generate a response and return it as text.

        Args:
            system_instruction: fixed rules for the model.
            prompt: the task for this call.
            response_schema: when given, the provider should ask the model for
                JSON matching this pydantic class. The caller still validates
                the result; providers do not guarantee valid output.
            temperature: sampling temperature.

        Raises:
            ProviderTimeoutError, ProviderRateLimitError,
            ProviderUnavailableError, ProviderAuthError, ProviderResponseError
            (all subclasses of ``ProviderError``).
        """
