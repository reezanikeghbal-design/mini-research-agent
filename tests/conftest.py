"""Shared fakes and fixtures. No test uses real API keys or the network."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import ResearchAgent  # noqa: E402
from config import Settings  # noqa: E402
from errors import ToolError  # noqa: E402
from memory import ConversationMemory  # noqa: E402
from models import SearchResult, ToolRequest, ToolResult  # noqa: E402
from providers.base import LLMProvider  # noqa: E402
from tools.base import Tool, ToolRegistry  # noqa: E402


# --- builders for scripted model output --------------------------------------
def decision(action: str, query: str = "", reason: str = "test") -> str:
    return json.dumps({"action": action, "query": query, "reason": reason})


def answer(text: str, urls: list[str] | None = None) -> str:
    return json.dumps({"answer": text, "source_urls": urls or []})


def make_result(n: int = 1) -> SearchResult:
    return SearchResult(
        title=f"Source {n}",
        url=f"https://example.com/article-{n}",
        snippet=f"Snippet number {n} about small business automation.",
        score=0.9,
    )


# --- fakes -------------------------------------------------------------------
class FakeProvider(LLMProvider):
    """Returns scripted responses in order; raises if an item is an Exception.

    Calling it more often than scripted fails the test, which lets tests assert
    exact call counts (e.g. "no more than one regeneration").
    """

    def __init__(self, script: list) -> None:
        self.script = list(script)
        self.calls: list[dict] = []

    def generate(self, *, system_instruction, prompt, response_schema=None, temperature=0.2):
        self.calls.append(
            {"system": system_instruction, "prompt": prompt, "schema": response_schema}
        )
        if not self.script:
            raise AssertionError(f"Unexpected extra provider call #{len(self.calls)}")
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakeSearchTool(Tool):
    """Scripted search tool. Each run pops one item: list[SearchResult] or Exception."""

    name = "web_search"
    description = "fake"

    def __init__(self, script: list | None = None, available: bool = True) -> None:
        self.script = list(script or [])
        self._available = available
        self.requests: list[ToolRequest] = []

    @property
    def available(self) -> bool:
        return self._available

    def run(self, request: ToolRequest) -> ToolResult:
        self.requests.append(request)
        if not self.script:
            raise AssertionError("Unexpected extra search execution")
        item = self.script.pop(0)
        if isinstance(item, ToolError):
            raise item
        return ToolResult(tool_name=self.name, query=request.query, results=item)


# --- fixtures ----------------------------------------------------------------
@pytest.fixture
def settings(tmp_path) -> Settings:
    return make_settings(tmp_path)


def make_settings(tmp_path: Path, **overrides) -> Settings:
    base = dict(
        gemini_api_key="test-gemini-key",
        tavily_api_key="test-tavily-key",
        data_dir=tmp_path / "data",
        log_dir=tmp_path / "logs",
    )
    base.update(overrides)
    return Settings(**base)


@pytest.fixture
def build_agent(tmp_path):
    """Factory: build_agent(provider_script, search_script, **settings_overrides)."""

    def _build(provider_script, search_script=None, *, search_available=True, **overrides):
        settings = make_settings(tmp_path, **overrides)
        provider = FakeProvider(provider_script)
        tool = FakeSearchTool(search_script, available=search_available)
        registry = ToolRegistry()
        registry.register(tool)
        memory = ConversationMemory(settings.data_dir)
        agent = ResearchAgent(
            settings=settings, provider=provider, tools=registry, memory=memory
        )
        return agent, provider, tool, memory

    return _build
