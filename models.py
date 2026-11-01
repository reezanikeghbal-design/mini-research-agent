"""Typed data models shared across the application.

NOTE: this module intentionally does not use ``from __future__ import
annotations``. ``ActionDecision`` and ``ModelAnswer`` are also handed to the
LLM SDK as response schemas, and SDKs inspect real (non-string) annotations.
For the same reason these two schema classes have no default values: every
field is required, which is the most widely supported schema shape.
"""

import json
import re
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

ResponseStatus = Literal["completed", "partial", "failed"]


# --- What the LLM returns ----------------------------------------------------
class ActionDecision(BaseModel):
    """Raw action chosen by the model: "answer" or "search".

    ``query`` is the search query when ``action == "search"`` and an empty
    string otherwise. The agent validates this before anything is executed.
    """

    action: str
    query: str
    reason: str


class ModelAnswer(BaseModel):
    """Structured final answer produced by the model."""

    answer: str
    source_urls: list[str]


# --- Tools -------------------------------------------------------------------
class ToolRequest(BaseModel):
    """A validated request to run a registered tool."""

    tool_name: str
    query: str


class SearchResult(BaseModel):
    """One normalized web search result."""

    title: str
    url: str
    snippet: str
    score: float | None = None
    published_date: str | None = None


class ToolResult(BaseModel):
    """Normalized output of a tool run."""

    tool_name: str
    query: str
    results: list[SearchResult]


# --- Evaluation --------------------------------------------------------------
class EvaluationResult(BaseModel):
    """Outcome of the deterministic output checks.

    ``parsed_answer`` is filled when the raw model output could be parsed into
    a ``ModelAnswer`` (even if later checks failed).
    """

    passed: bool
    checks: dict[str, bool]
    issues: list[str]
    parsed_answer: ModelAnswer | None = None


# --- Conversation memory -----------------------------------------------------
class ChatMessage(BaseModel):
    """One message of conversation history sent to the model."""

    role: Literal["user", "assistant"]
    content: str


def utc_now_iso() -> str:
    """Current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


class ConversationRecord(BaseModel):
    """One saved question/answer exchange."""

    record_id: str
    conversation_id: str
    question: str
    answer: str
    status: ResponseStatus
    created_at: str = Field(default_factory=utc_now_iso)
    tool_metadata: dict[str, Any] = Field(default_factory=dict)


class ConversationFile(BaseModel):
    """On-disk layout of one conversation's JSON file."""

    conversation_id: str
    created_at: str = Field(default_factory=utc_now_iso)
    records: list[ConversationRecord] = Field(default_factory=list)


# --- What the agent returns --------------------------------------------------
class AgentResponse(BaseModel):
    """Final result of one ``ResearchAgent.run`` call.

    status:
        completed - finished normally and passed evaluation.
        partial   - an answer exists but something went wrong (see warnings).
        failed    - no usable answer; see ``error``.
    """

    status: ResponseStatus
    conversation_id: str
    answer: str | None = None
    sources: list[SearchResult] = Field(default_factory=list)
    searches_performed: int = 0
    queries: list[str] = Field(default_factory=list)
    iterations: int = 0
    limited_mode: bool = False
    warnings: list[str] = Field(default_factory=list)
    evaluation: EvaluationResult | None = None
    error: str | None = None


# --- Helpers -----------------------------------------------------------------
_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL | re.IGNORECASE)


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse model output into a JSON object.

    Model output is never trusted to be valid JSON: this strips an optional
    Markdown code fence, parses, and requires the top-level value to be an
    object. Raises ``ValueError`` with a short reason otherwise.
    """
    cleaned = (text or "").strip()
    fence = _FENCE_RE.match(cleaned)
    if fence:
        cleaned = fence.group(1).strip()
    if not cleaned:
        raise ValueError("output is empty")
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"output is not valid JSON ({exc.msg})") from exc
    if not isinstance(data, dict):
        raise ValueError("output is valid JSON but not a JSON object")
    return data
