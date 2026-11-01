"""Command-line interface for Mini Research Agent.

This module only handles user interaction and wiring. All agent logic lives
in ``agent.py``.

Usage:
    python app.py                       # interactive session
    python app.py --ask "your question" # one-shot
    python app.py --conversation <id>   # resume a saved conversation

Interactive commands: /new  /help  /exit  (also: exit, quit)
"""

import argparse
import logging
import sys
from typing import Callable, Sequence

from dotenv import load_dotenv

from agent import ResearchAgent
from config import Settings, configure_logging
from errors import AgentError, ConfigError
from memory import ConversationMemory
from models import AgentResponse
from providers.gemini_provider import GeminiProvider
from tools.base import ToolRegistry
from tools.web_search import WebSearchTool

logger = logging.getLogger("mra.app")

EXIT_COMMANDS = {"/exit", "/quit", "exit", "quit"}
HELP_TEXT = (
    "Commands:\n"
    "  /new   start a new conversation (fresh history)\n"
    "  /help  show this help\n"
    "  /exit  quit (also: exit, quit)\n"
    "Anything else is treated as a question."
)
LIMITED_MODE_BANNER = (
    "[LIMITED MODE] TAVILY_API_KEY is not set (or MAX_SEARCH_CALLS=0): web search is\n"
    "disabled. Answers come from the model's general knowledge only, may be outdated,\n"
    "and are NOT based on external research."
)


def build_agent(settings: Settings) -> ResearchAgent:
    """Wire up provider, tools and memory from settings.

    Raises:
        ConfigError: the provider cannot be created (e.g. SDK not installed).
    """
    provider = GeminiProvider(
        api_key=settings.gemini_api_key,
        model=settings.gemini_model,
        timeout_seconds=settings.request_timeout_seconds,
    )
    registry = ToolRegistry()
    registry.register(
        WebSearchTool(
            settings.tavily_api_key,
            max_results=settings.max_search_results,
            timeout_seconds=settings.request_timeout_seconds,
            max_query_length=settings.max_query_length,
        )
    )
    return ResearchAgent(
        settings=settings,
        provider=provider,
        tools=registry,
        memory=ConversationMemory(settings.data_dir),
    )


def format_response(response: AgentResponse) -> str:
    """Render an ``AgentResponse`` as plain text for the terminal."""
    lines: list[str] = []
    if response.status == "failed":
        lines.append(f"[FAILED] {response.error or 'The request could not be completed.'}")
    else:
        if response.status == "partial":
            lines.append("[PARTIAL RESULT] The answer below is incomplete or unverified; see notes.")
        lines.append("")
        lines.append(response.answer or "")
        if response.sources:
            lines.append("")
            lines.append("Sources:")
            for index, source in enumerate(response.sources, start=1):
                lines.append(f"  {index}. {source.title} - {source.url}")

    if response.warnings:
        lines.append("")
        lines.append("Notes:")
        lines.extend(f"  - {warning}" for warning in response.warnings)

    lines.append("")
    lines.append(
        f"[status: {response.status} | searches: {response.searches_performed} | "
        f"limited mode: {'yes' if response.limited_mode else 'no'}]"
    )
    return "\n".join(lines)


def run_interactive(
    agent: ResearchAgent,
    conversation_id: str,
    *,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> None:
    """Interactive loop. ``input_fn``/``output_fn`` are injectable for tests."""
    output_fn("Mini Research Agent. Type /help for commands, /exit to quit.")
    output_fn(f"Conversation: {conversation_id}")
    while True:
        try:
            line = input_fn("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            output_fn("\nGoodbye.")
            return
        if not line:
            continue
        lowered = line.lower()
        if lowered in EXIT_COMMANDS:
            output_fn("Goodbye.")
            return
        if lowered == "/help":
            output_fn(HELP_TEXT)
            continue
        if lowered == "/new":
            conversation_id = agent.memory.new_conversation_id()
            output_fn(f"Started a new conversation: {conversation_id}")
            continue

        try:
            response = agent.run(line, conversation_id)
        except Exception:  # last-resort guard so one bug does not kill the session
            logger.exception("event=unexpected_error")
            output_fn("[FAILED] An unexpected internal error occurred. Details are in the log file.")
            continue
        output_fn(format_response(response))


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="mini-research-agent", description="A small AI research agent."
    )
    parser.add_argument("--ask", metavar="QUESTION", help="ask one question and exit")
    parser.add_argument(
        "--conversation", "-c", metavar="ID", help="resume a saved conversation by id"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns a process exit code (0 ok, 1 failed answer, 2 config error)."""
    args = parse_args(argv)
    load_dotenv()  # does not override variables already set in the environment

    try:
        settings = Settings.from_env()
        configure_logging(settings)
        agent = build_agent(settings)
        conversation_id = (
            ConversationMemory.validate_conversation_id(args.conversation)
            if args.conversation
            else agent.memory.new_conversation_id()
        )
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    except AgentError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    logger.info(
        "event=app_started model=%s search_enabled=%s", settings.gemini_model, settings.search_enabled
    )
    if not settings.search_enabled:
        print(LIMITED_MODE_BANNER)

    if args.ask:
        response = agent.run(args.ask, conversation_id)
        print(format_response(response))
        return 1 if response.status == "failed" else 0

    run_interactive(agent, conversation_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
