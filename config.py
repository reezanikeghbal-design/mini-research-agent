"""Configuration loading, validation and logging setup.

Values come from environment variables (optionally loaded from a ``.env`` file
by ``app.py``). Empty values are treated as "not set", so the blank entries in
``.env.example`` fall back to sensible defaults.
"""

import logging
import os
import sys
from dataclasses import dataclass, field
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Mapping

from errors import ConfigError

PROJECT_ROOT = Path(__file__).resolve().parent
LOGGER_NAME = "mra"  # every module logs under "mra.<module>"
VALID_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


@dataclass(frozen=True)
class Settings:
    """Validated application settings.

    API keys are excluded from ``repr`` so they cannot leak through logging
    or error messages that print the object.
    """

    gemini_api_key: str = field(repr=False)
    tavily_api_key: str | None = field(default=None, repr=False)
    gemini_model: str = "gemini-3.6-flash"
    max_agent_iterations: int = 5
    max_search_calls: int = 2
    max_search_results: int = 5
    max_query_length: int = 200
    max_history_messages: int = 6
    max_output_length: int = 4000
    request_timeout_seconds: float = 30.0
    log_level: str = "INFO"
    data_dir: Path = PROJECT_ROOT / "data"
    log_dir: Path = PROJECT_ROOT / "logs"

    @property
    def search_enabled(self) -> bool:
        """True when a search key is present and searching is allowed."""
        return bool(self.tavily_api_key) and self.max_search_calls > 0

    @property
    def secrets(self) -> list[str]:
        """Secret values that must never appear in logs."""
        return [s for s in (self.gemini_api_key, self.tavily_api_key) if s]

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Settings":
        """Build settings from environment variables.

        All problems are collected and reported together in one ``ConfigError``
        so the user can fix them in a single pass.
        """
        source = os.environ if env is None else env
        errors: list[str] = []

        gemini_key = _get(source, "GEMINI_API_KEY")
        if not gemini_key:
            errors.append(
                "GEMINI_API_KEY is not set. Copy .env.example to .env and add "
                "your Gemini API key (https://aistudio.google.com/apikey)."
            )

        log_level = (_get(source, "LOG_LEVEL") or "INFO").upper()
        if log_level not in VALID_LOG_LEVELS:
            errors.append(f"LOG_LEVEL must be one of {', '.join(VALID_LOG_LEVELS)}.")
            log_level = "INFO"

        settings = cls(
            gemini_api_key=gemini_key or "",
            tavily_api_key=_get(source, "TAVILY_API_KEY"),
            gemini_model=_get(source, "GEMINI_MODEL") or cls.gemini_model,
            max_agent_iterations=_get_int(
                source, "MAX_AGENT_ITERATIONS", cls.max_agent_iterations, 1, 20, errors
            ),
            max_search_calls=_get_int(
                source, "MAX_SEARCH_CALLS", cls.max_search_calls, 0, 10, errors
            ),
            max_search_results=_get_int(
                source, "MAX_SEARCH_RESULTS", cls.max_search_results, 1, 10, errors
            ),
            max_history_messages=_get_int(
                source, "MAX_HISTORY_MESSAGES", cls.max_history_messages, 0, 50, errors
            ),
            max_output_length=_get_int(
                source, "MAX_OUTPUT_LENGTH", cls.max_output_length, 200, 20000, errors
            ),
            request_timeout_seconds=_get_float(
                source, "REQUEST_TIMEOUT_SECONDS", cls.request_timeout_seconds, 1, 300, errors
            ),
            log_level=log_level,
            data_dir=Path(_get(source, "DATA_DIR") or PROJECT_ROOT / "data"),
            log_dir=Path(_get(source, "LOG_DIR") or PROJECT_ROOT / "logs"),
        )
        if errors:
            raise ConfigError("Invalid configuration:\n  - " + "\n  - ".join(errors))
        return settings


def _get(env: Mapping[str, str], name: str) -> str | None:
    value = env.get(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _get_int(
    env: Mapping[str, str],
    name: str,
    default: int,
    minimum: int,
    maximum: int,
    errors: list[str],
) -> int:
    raw = _get(env, name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        errors.append(f"{name} must be a whole number between {minimum} and {maximum}.")
        return default
    if not minimum <= value <= maximum:
        errors.append(f"{name} must be between {minimum} and {maximum} (got {value}).")
        return default
    return value


def _get_float(
    env: Mapping[str, str],
    name: str,
    default: float,
    minimum: float,
    maximum: float,
    errors: list[str],
) -> float:
    raw = _get(env, name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        errors.append(f"{name} must be a number between {minimum} and {maximum}.")
        return default
    if not minimum <= value <= maximum:
        errors.append(f"{name} must be between {minimum} and {maximum} (got {value}).")
        return default
    return value


class RedactingFormatter(logging.Formatter):
    """Formatter that replaces known secret values with ``***``.

    It runs on the final formatted text, so secrets are also removed from
    exception tracebacks.
    """

    def __init__(self, secrets: list[str], fmt: str | None = None) -> None:
        super().__init__(fmt)
        # Longest first so a secret that contains another is fully replaced.
        self._secrets = sorted((s for s in secrets if s), key=len, reverse=True)

    def format(self, record: logging.LogRecord) -> str:
        text = super().format(record)
        for secret in self._secrets:
            text = text.replace(secret, "***")
        return text


def configure_logging(settings: Settings) -> logging.Logger:
    """Configure the ``mra`` logger: rotating file log, secrets redacted.

    Safe to call more than once (previous handlers are replaced). If the log
    file cannot be opened, logging falls back to stderr instead of crashing.
    """
    logger = logging.getLogger(LOGGER_NAME)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    logger.setLevel(settings.log_level)
    logger.propagate = False

    fmt = "%(asctime)s %(levelname)s %(name)s %(message)s"
    handler: logging.Handler
    try:
        settings.log_dir.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(
            settings.log_dir / "agent.log",
            maxBytes=1_000_000,
            backupCount=3,
            encoding="utf-8",
        )
    except OSError as exc:
        handler = logging.StreamHandler(sys.stderr)
        print(f"Warning: cannot write logs to {settings.log_dir} ({exc}); logging to stderr.",
              file=sys.stderr)
    handler.setFormatter(RedactingFormatter(settings.secrets, fmt))
    logger.addHandler(handler)
    return logger
