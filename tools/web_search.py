"""Web search tool backed by the Tavily API (``tavily-python``).

Tavily call used: ``TavilyClient(api_key=...).search(query, max_results=...)``,
which returns a dict with a ``results`` list of items containing ``title``,
``url``, ``content`` (snippet) and ``score``.

Search output is UNTRUSTED external content. This module only normalizes it
into a predictable shape; the agent is responsible for presenting it to the
model as data, never as instructions.
"""

import inspect
import logging
import re
from typing import Any
from urllib.parse import urlsplit

from errors import (
    InvalidToolRequestError,
    ToolError,
    ToolExecutionError,
    ToolTimeoutError,
    ToolUnavailableError,
)
from models import SearchResult, ToolRequest, ToolResult
from tools.base import Tool

logger = logging.getLogger("mra.tools.web_search")

MAX_SNIPPET_CHARS = 500
MAX_URL_CHARS = 2048

_TIMEOUT_NAMES = {"timeout", "readtimeout", "connecttimeout", "timeouterror"}
_AUTH_NAMES = {"invalidapikeyerror", "missingapikeyerror"}
_LIMIT_NAMES = {"usagelimitexceedederror"}


def is_safe_url(url: str) -> bool:
    """Accept only plain http(s) URLs with a host and no embedded credentials."""
    if not url or len(url) > MAX_URL_CHARS or re.search(r"\s", url):
        return False
    try:
        parts = urlsplit(url)
    except ValueError:
        return False
    return parts.scheme in ("http", "https") and bool(parts.netloc) and "@" not in parts.netloc


def _translate_exception(exc: Exception) -> ToolError:
    """Map a Tavily/network exception to a ``ToolError`` (by class name).

    Matching on names keeps this module importable without ``tavily-python``.
    The user-facing message never contains the raw exception text.
    """
    name = type(exc).__name__.lower()
    if isinstance(exc, TimeoutError) or name in _TIMEOUT_NAMES:
        return ToolTimeoutError("The search service timed out.")
    if name in _AUTH_NAMES:
        return ToolUnavailableError("The search service rejected the API key. Check TAVILY_API_KEY.")
    if name in _LIMIT_NAMES:
        return ToolExecutionError("The search service usage limit or rate limit was reached.")
    return ToolExecutionError(f"The search service failed [{type(exc).__name__}].")


def _accepts_named_parameter(func: Any, parameter: str) -> bool:
    """True if ``func`` declares a parameter with exactly this name."""
    try:
        return parameter in inspect.signature(func).parameters
    except (TypeError, ValueError):
        return False


class WebSearchTool(Tool):
    """Search the web and return normalized results."""

    name = "web_search"
    description = "Search the web for current information. Input: a short search query."

    def __init__(
        self,
        api_key: str | None,
        *,
        max_results: int = 5,
        timeout_seconds: float = 30.0,
        max_query_length: int = 200,
        client: Any | None = None,
    ) -> None:
        """
        Args:
            api_key: Tavily API key; ``None``/empty disables the tool.
            max_results: maximum number of results returned per search.
            timeout_seconds: request timeout (passed to the client when the
                client's ``search`` method supports a ``timeout`` parameter).
            max_query_length: longest accepted query.
            client: optional pre-built client with a ``search`` method (tests).
        """
        self._api_key = api_key
        self._max_results = max_results
        self._timeout_seconds = timeout_seconds
        self._max_query_length = max_query_length
        self._client = client

    @property
    def available(self) -> bool:
        return self._client is not None or bool(self._api_key)

    def run(self, request: ToolRequest) -> ToolResult:
        query = self._validate_query(request.query)
        client = self._get_client()

        kwargs: dict[str, Any] = {"max_results": self._max_results}
        if _accepts_named_parameter(client.search, "timeout"):
            kwargs["timeout"] = self._timeout_seconds

        try:
            raw = client.search(query, **kwargs)
        except Exception as exc:  # translated, not swallowed: see _translate_exception
            logger.warning("event=search_api_failed error_type=%s", type(exc).__name__)
            raise _translate_exception(exc) from exc

        results = self._normalize(raw)
        logger.info("event=search_normalized results=%d", len(results))
        return ToolResult(tool_name=self.name, query=query, results=results)

    # --- internals -----------------------------------------------------------
    def _validate_query(self, query: str) -> str:
        cleaned = " ".join((query or "").split())
        if not cleaned:
            raise InvalidToolRequestError("Search query is empty.")
        if len(cleaned) > self._max_query_length:
            raise InvalidToolRequestError(
                f"Search query is too long ({len(cleaned)} > {self._max_query_length} characters)."
            )
        return cleaned

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        if not self._api_key:
            raise ToolUnavailableError("TAVILY_API_KEY is not set.")
        try:
            from tavily import TavilyClient
        except ImportError as exc:
            raise ToolUnavailableError(
                "The 'tavily-python' package is not installed. "
                "Run: pip install -r requirements.txt"
            ) from exc
        self._client = TavilyClient(api_key=self._api_key)
        return self._client

    def _normalize(self, raw: Any) -> list[SearchResult]:
        if not isinstance(raw, dict) or not isinstance(raw.get("results"), list):
            raise ToolExecutionError("The search service returned an unexpected response format.")

        results: list[SearchResult] = []
        seen: set[str] = set()
        for item in raw["results"]:
            if not isinstance(item, dict):
                continue
            url = str(item.get("url") or "").strip()
            if not is_safe_url(url) or url in seen:
                continue
            seen.add(url)

            snippet = " ".join(str(item.get("content") or item.get("snippet") or "").split())
            if len(snippet) > MAX_SNIPPET_CHARS:
                snippet = snippet[: MAX_SNIPPET_CHARS - 1] + "…"
            title = " ".join(str(item.get("title") or "").split()) or urlsplit(url).netloc

            score = item.get("score")
            published = item.get("published_date")
            results.append(
                SearchResult(
                    title=title[:200],
                    url=url,
                    snippet=snippet,
                    score=float(score) if isinstance(score, (int, float)) else None,
                    published_date=str(published) if published else None,
                )
            )
            if len(results) >= self._max_results:
                break
        return results
