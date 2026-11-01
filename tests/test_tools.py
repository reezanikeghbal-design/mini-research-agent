"""Tests for the web search tool and the tool registry (fake client, no network)."""

import pytest

from conftest import FakeSearchTool
from errors import (
    InvalidToolRequestError,
    ToolExecutionError,
    ToolTimeoutError,
    ToolUnavailableError,
)
from models import ToolRequest
from tools.base import ToolRegistry
from tools.web_search import WebSearchTool, is_safe_url


class FakeClient:
    """Mimics the part of TavilyClient that the tool uses."""

    def __init__(self, response=None, error: Exception | None = None):
        self.response = response
        self.error = error
        self.calls: list[tuple] = []

    def search(self, query, max_results=5, **kwargs):
        self.calls.append((query, max_results, kwargs))
        if self.error:
            raise self.error
        return self.response


class FakeClientWithTimeout(FakeClient):
    def search(self, query, max_results=5, timeout=60):
        self.calls.append((query, max_results, {"timeout": timeout}))
        return self.response


def request(query="small business automation") -> ToolRequest:
    return ToolRequest(tool_name="web_search", query=query)


def tool_with(client, **kwargs) -> WebSearchTool:
    return WebSearchTool("key", client=client, **kwargs)


def raw(*items):
    return {"results": list(items)}


def item(n=1, **overrides):
    base = {
        "title": f"Title {n}",
        "url": f"https://example.com/{n}",
        "content": f"Content {n}",
        "score": 0.5,
        "published_date": "2026-01-01",
    }
    base.update(overrides)
    return base


# --- normalization ------------------------------------------------------------
def test_results_are_normalized_with_title_url_snippet_and_metadata():
    result = tool_with(FakeClient(raw(item(1)))).run(request())

    assert result.tool_name == "web_search"
    assert result.query == "small business automation"
    only = result.results[0]
    assert (only.title, only.url, only.snippet) == ("Title 1", "https://example.com/1", "Content 1")
    assert only.score == 0.5
    assert only.published_date == "2026-01-01"


def test_unsafe_duplicate_and_malformed_items_are_dropped():
    response = raw(
        item(1),
        item(1),  # duplicate URL
        item(2, url="javascript:alert(1)"),
        item(3, url="file:///etc/passwd"),
        item(4, url="https://user:pw@evil.example/"),
        item(5, url="not a url"),
        item(6, url=""),
        "a string, not a dict",
        item(7),
    )

    urls = [r.url for r in tool_with(FakeClient(response)).run(request()).results]

    assert urls == ["https://example.com/1", "https://example.com/7"]


def test_results_are_limited_and_snippets_truncated():
    response = raw(*[item(n, content="word " * 400) for n in range(1, 9)])
    results = tool_with(FakeClient(response), max_results=3).run(request()).results

    assert len(results) == 3
    assert all(len(r.snippet) <= 500 for r in results)
    assert results[0].snippet.endswith("…")


def test_missing_title_falls_back_to_host_and_score_is_optional():
    result = tool_with(FakeClient(raw({"url": "https://example.org/x", "content": "c"}))).run(request())

    assert result.results[0].title == "example.org"
    assert result.results[0].score is None


def test_empty_results_are_returned_not_raised():
    assert tool_with(FakeClient(raw())).run(request()).results == []


@pytest.mark.parametrize("bad", [None, "text", [], {"no_results_key": 1}, {"results": "x"}])
def test_unexpected_response_format_raises_tool_execution_error(bad):
    with pytest.raises(ToolExecutionError, match="unexpected response format"):
        tool_with(FakeClient(bad)).run(request())


# --- query validation -----------------------------------------------------------
def test_empty_query_is_rejected_before_calling_the_service():
    client = FakeClient(raw())
    with pytest.raises(InvalidToolRequestError, match="empty"):
        tool_with(client).run(request("   "))
    assert client.calls == []


def test_overlong_query_is_rejected():
    with pytest.raises(InvalidToolRequestError, match="too long"):
        tool_with(FakeClient(raw()), max_query_length=20).run(request("x" * 21))


# --- error mapping ---------------------------------------------------------------
def test_timeout_is_mapped_to_tool_timeout_error():
    class ReadTimeout(Exception):
        pass

    with pytest.raises(ToolTimeoutError):
        tool_with(FakeClient(error=ReadTimeout("secret details"))).run(request())
    with pytest.raises(ToolTimeoutError):
        tool_with(FakeClient(error=TimeoutError())).run(request())


def test_rate_limit_and_auth_errors_have_clear_messages():
    class UsageLimitExceededError(Exception):
        pass

    class InvalidAPIKeyError(Exception):
        pass

    with pytest.raises(ToolExecutionError, match="usage limit"):
        tool_with(FakeClient(error=UsageLimitExceededError())).run(request())
    with pytest.raises(ToolUnavailableError, match="TAVILY_API_KEY"):
        tool_with(FakeClient(error=InvalidAPIKeyError())).run(request())


def test_generic_failure_message_does_not_leak_exception_text():
    with pytest.raises(ToolExecutionError) as caught:
        tool_with(FakeClient(error=RuntimeError("Bearer tvly-SECRET-123"))).run(request())

    assert "SECRET" not in str(caught.value)
    assert "RuntimeError" in str(caught.value)


# --- availability / missing credentials -------------------------------------------
def test_tool_without_api_key_is_unavailable_and_refuses_to_run():
    tool = WebSearchTool(None)

    assert tool.available is False
    with pytest.raises(ToolUnavailableError, match="TAVILY_API_KEY"):
        tool.run(request())


def test_tool_with_blank_api_key_is_unavailable():
    assert WebSearchTool("").available is False


def test_timeout_is_passed_only_to_clients_that_declare_it():
    with_timeout = FakeClientWithTimeout(raw(item(1)))
    tool_with(with_timeout, timeout_seconds=12).run(request())
    assert with_timeout.calls[0][2] == {"timeout": 12}

    without = FakeClient(raw(item(1)))
    tool_with(without, timeout_seconds=12).run(request())
    assert without.calls[0][2] == {}  # not forced into **kwargs


@pytest.mark.parametrize(
    "url, ok",
    [
        ("https://example.com/a?b=1", True),
        ("http://example.com", True),
        ("ftp://example.com", False),
        ("https:///nohost", False),
        ("https://a.com/with space", False),
        ("https://" + "a" * 2100 + ".com", False),
    ],
)
def test_is_safe_url(url, ok):
    assert is_safe_url(url) is ok


# --- registry --------------------------------------------------------------------
def test_registry_only_executes_registered_tools():
    registry = ToolRegistry()
    registry.register(FakeSearchTool([[]]))

    with pytest.raises(InvalidToolRequestError, match="Unknown tool"):
        registry.execute(ToolRequest(tool_name="run_shell", query="rm -rf /"))
    assert registry.execute(request()).results == []


def test_registry_rejects_unavailable_tool_and_duplicate_registration():
    registry = ToolRegistry()
    registry.register(FakeSearchTool(available=False))

    assert registry.is_available("web_search") is False
    assert registry.is_available("unknown") is False
    with pytest.raises(ToolUnavailableError):
        registry.execute(request())
    with pytest.raises(ValueError):
        registry.register(FakeSearchTool())
