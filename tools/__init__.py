"""Tools the agent may execute."""

from tools.base import Tool, ToolRegistry
from tools.web_search import WebSearchTool

__all__ = ["Tool", "ToolRegistry", "WebSearchTool"]
