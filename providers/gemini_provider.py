"""Gemini implementation of ``LLMProvider`` using the official Google Gen AI SDK.

All Gemini-specific details live in this file. SDK calls used (``google-genai``):

* ``genai.Client(api_key=..., http_options=types.HttpOptions(timeout=<ms>))``
* ``client.models.generate_content(model=..., contents=..., config={...})``
  where ``config`` may contain ``system_instruction``, ``temperature``,
  ``response_mime_type="application/json"`` and ``response_schema=<pydantic class>``.
* ``response.text``

The SDK is imported lazily, so the rest of the project (and the test suite)
works without it installed.
"""

import logging
from typing import Any

from pydantic import BaseModel

from errors import (
    ConfigError,
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitError,
    ProviderResponseError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from providers.base import LLMProvider

logger = logging.getLogger("mra.gemini")

_TIMEOUT_NAMES = {"timeout", "readtimeout", "connecttimeout", "timeoutexception",
                  "deadlineexceeded", "timeouterror"}
_CONNECTION_NAMES = {"connecterror", "connectionerror", "remoteprotocolerror"}


def translate_exception(exc: Exception) -> ProviderError:
    """Map an SDK/network exception to an application-level ``ProviderError``.

    Classification uses the HTTP status ``code`` attribute (set by the SDK's
    API errors) and exception class names, so this module does not depend on
    the exact exception classes of the installed SDK version. The user-facing
    message never includes the raw exception text.
    """
    name = type(exc).__name__.lower()
    code = getattr(exc, "code", None)
    if not isinstance(code, int):
        code = None

    if isinstance(exc, TimeoutError) or name in _TIMEOUT_NAMES or code in (408, 504):
        return ProviderTimeoutError(
            "The model request timed out. Try again, or raise REQUEST_TIMEOUT_SECONDS."
        )
    if code == 429:
        return ProviderRateLimitError(
            "The model API rate limit or quota was reached. Wait a moment and try again."
        )
    if code in (401, 403):
        return ProviderAuthError(
            "The model API rejected the credentials. Check GEMINI_API_KEY."
        )
    if (code is not None and code >= 500) or name in _CONNECTION_NAMES:
        return ProviderUnavailableError(
            "The model API is currently unavailable. Try again later."
        )
    detail = f" (HTTP {code})" if code else ""
    return ProviderError(f"The model API request failed{detail} [{type(exc).__name__}].")


class GeminiProvider(LLMProvider):
    """Generate text with a Gemini model."""

    def __init__(
        self,
        api_key: str,
        model: str,
        timeout_seconds: float = 30.0,
        client: Any | None = None,
    ) -> None:
        """
        Args:
            api_key: Gemini API key (never logged or placed in error messages).
            model: model name, e.g. ``gemini-3.6-flash``.
            timeout_seconds: per-request timeout.
            client: optional pre-built client (used by tests to inject a fake).
        """
        if not api_key and client is None:
            raise ConfigError("GEMINI_API_KEY is not set.")
        self._api_key = api_key
        self._model = model
        self._timeout_seconds = timeout_seconds
        self._client = client if client is not None else self._build_client()

    def _build_client(self) -> Any:
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise ConfigError(
                "The 'google-genai' package is not installed. "
                "Run: pip install -r requirements.txt"
            ) from exc
        # The SDK expresses the HTTP timeout in milliseconds.
        http_options = types.HttpOptions(timeout=int(self._timeout_seconds * 1000))
        return genai.Client(api_key=self._api_key, http_options=http_options)

    def generate(
        self,
        *,
        system_instruction: str,
        prompt: str,
        response_schema: type[BaseModel] | None = None,
        temperature: float = 0.2,
    ) -> str:
        config: dict[str, Any] = {
            "system_instruction": system_instruction,
            "temperature": temperature,
        }
        if response_schema is not None:
            config["response_mime_type"] = "application/json"
            config["response_schema"] = response_schema

        logger.info(
            "event=model_request model=%s structured=%s prompt_chars=%d",
            self._model,
            response_schema is not None,
            len(prompt),
        )
        try:
            response = self._client.models.generate_content(
                model=self._model, contents=prompt, config=config
            )
        except Exception as exc:  # translated, not swallowed: see translate_exception
            logger.warning(
                "event=model_request_failed error_type=%s code=%s",
                type(exc).__name__,
                getattr(exc, "code", None),
            )
            raise translate_exception(exc) from exc

        text = getattr(response, "text", None)
        if not isinstance(text, str) or not text.strip():
            logger.warning("event=model_response_empty")
            raise ProviderResponseError(
                "The model returned no text (the response may have been blocked or truncated)."
            )
        logger.info("event=model_response_received chars=%d", len(text))
        return text
