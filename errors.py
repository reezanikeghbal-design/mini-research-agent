"""Application-level exceptions.

Every exception that the agent can reasonably expect from an external service
or from invalid model output has its own class, so the agent can react to the
*kind* of failure instead of catching everything.

Messages on these exceptions are written for end users (they are shown in the
CLI) and must never contain API keys.
"""


class AgentError(Exception):
    """Base class for every expected application error."""


# --- Configuration -----------------------------------------------------------
class ConfigError(AgentError):
    """Configuration is missing or invalid."""


# --- LLM provider ------------------------------------------------------------
class ProviderError(AgentError):
    """The LLM provider failed."""


class ProviderTimeoutError(ProviderError):
    """The LLM request timed out."""


class ProviderRateLimitError(ProviderError):
    """The LLM provider rejected the request because of rate/quota limits."""


class ProviderUnavailableError(ProviderError):
    """The LLM provider is unreachable or returned a server error."""


class ProviderAuthError(ProviderError):
    """The LLM provider rejected the credentials."""


class ProviderResponseError(ProviderError):
    """The LLM provider answered, but the response was empty or unusable."""


# --- Tools -------------------------------------------------------------------
class ToolError(AgentError):
    """A tool could not be executed."""


class InvalidToolRequestError(ToolError):
    """The tool request is invalid (unknown tool, empty or oversized query)."""


class ToolUnavailableError(ToolError):
    """The tool cannot run (for example, its API key is missing)."""


class ToolTimeoutError(ToolError):
    """The tool's external service timed out."""


class ToolExecutionError(ToolError):
    """The tool's external service failed or returned an unusable response."""


# --- Agent workflow ----------------------------------------------------------
class InvalidActionError(AgentError):
    """The model produced an action the application refuses to execute."""


class WorkflowAbortedError(AgentError):
    """The workflow was stopped by a safety limit (e.g. repeated invalid actions)."""


# --- Memory ------------------------------------------------------------------
class MemoryStoreError(AgentError):
    """Conversation memory could not be read or written."""


class MemoryWriteError(MemoryStoreError):
    """Conversation memory could not be written to disk."""


class InvalidConversationIdError(MemoryStoreError):
    """The conversation identifier has an unsafe or invalid format."""
