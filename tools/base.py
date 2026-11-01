"""Tool interface and the explicit tool registry.

The agent can only run tools that were registered here. The LLM never chooses
code to run: it names a tool, and ``ToolRegistry.execute`` decides whether
that name is allowed.
"""

import logging
from abc import ABC, abstractmethod

from errors import InvalidToolRequestError, ToolUnavailableError
from models import ToolRequest, ToolResult

logger = logging.getLogger("mra.tools")


class Tool(ABC):
    """Common interface for every tool."""

    name: str
    description: str

    @property
    @abstractmethod
    def available(self) -> bool:
        """Whether the tool can run right now (e.g. its credentials exist)."""

    @abstractmethod
    def run(self, request: ToolRequest) -> ToolResult:
        """Execute the request and return normalized results.

        Raises:
            ToolError subclasses on invalid input or service failure.
        """


class ToolRegistry:
    """Explicit allow-list of tools the agent may execute."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool under its ``name``."""
        if tool.name in self._tools:
            raise ValueError(f"Tool {tool.name!r} is already registered.")
        self._tools[tool.name] = tool

    def names(self) -> list[str]:
        """Names of all registered tools."""
        return sorted(self._tools)

    def is_available(self, name: str) -> bool:
        """True if the tool is registered and currently usable."""
        tool = self._tools.get(name)
        return tool is not None and tool.available

    def execute(self, request: ToolRequest) -> ToolResult:
        """Validate the request against the registry and run it.

        Raises:
            InvalidToolRequestError: the tool name is not registered.
            ToolUnavailableError: the tool is registered but cannot run.
        """
        tool = self._tools.get(request.tool_name)
        if tool is None:
            raise InvalidToolRequestError(
                f"Unknown tool {request.tool_name[:40]!r}. "
                f"Registered tools: {', '.join(self.names()) or 'none'}."
            )
        if not tool.available:
            raise ToolUnavailableError(f"Tool {tool.name!r} is not available.")
        return tool.run(request)
