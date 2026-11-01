"""Tests for configuration loading, validation and secret redaction."""

import logging

import pytest

from config import RedactingFormatter, Settings, configure_logging
from errors import ConfigError

KEY = "fake-test-secret-value-123"


def test_missing_gemini_key_is_a_clear_config_error():
    with pytest.raises(ConfigError) as caught:
        Settings.from_env({})

    assert "GEMINI_API_KEY" in str(caught.value)
    assert ".env.example" in str(caught.value)  # actionable


def test_defaults_when_only_the_required_key_is_given():
    settings = Settings.from_env({"GEMINI_API_KEY": KEY})

    assert settings.max_search_calls == 2
    assert settings.max_agent_iterations == 5
    assert settings.max_history_messages == 6
    assert settings.max_output_length == 4000
    assert settings.request_timeout_seconds == 30.0
    assert settings.gemini_model == "gemini-3.6-flash"
    assert settings.log_level == "INFO"


def test_missing_search_key_means_limited_mode():
    settings = Settings.from_env({"GEMINI_API_KEY": KEY})

    assert settings.tavily_api_key is None
    assert settings.search_enabled is False


def test_search_enabled_requires_key_and_nonzero_limit():
    with_key = {"GEMINI_API_KEY": KEY, "TAVILY_API_KEY": "tvly-x"}

    assert Settings.from_env(with_key).search_enabled is True
    assert Settings.from_env({**with_key, "MAX_SEARCH_CALLS": "0"}).search_enabled is False


def test_blank_values_from_env_example_fall_back_to_defaults():
    env = {name: "" for name in [
        "TAVILY_API_KEY", "GEMINI_MODEL", "MAX_AGENT_ITERATIONS", "MAX_SEARCH_CALLS",
        "MAX_HISTORY_MESSAGES", "MAX_OUTPUT_LENGTH", "REQUEST_TIMEOUT_SECONDS", "LOG_LEVEL",
    ]}
    env["GEMINI_API_KEY"] = KEY

    settings = Settings.from_env(env)

    assert settings.max_search_calls == 2
    assert settings.tavily_api_key is None


def test_overrides_are_applied():
    settings = Settings.from_env({
        "GEMINI_API_KEY": KEY,
        "GEMINI_MODEL": "gemini-x",
        "MAX_SEARCH_CALLS": "1",
        "MAX_AGENT_ITERATIONS": "3",
        "MAX_HISTORY_MESSAGES": "10",
        "MAX_OUTPUT_LENGTH": "1000",
        "REQUEST_TIMEOUT_SECONDS": "12.5",
        "LOG_LEVEL": "debug",
    })

    assert (settings.gemini_model, settings.max_search_calls, settings.max_agent_iterations) == (
        "gemini-x", 1, 3,
    )
    assert settings.max_history_messages == 10
    assert settings.max_output_length == 1000
    assert settings.request_timeout_seconds == 12.5
    assert settings.log_level == "DEBUG"


def test_all_invalid_values_are_reported_together():
    with pytest.raises(ConfigError) as caught:
        Settings.from_env({
            "GEMINI_API_KEY": KEY,
            "MAX_SEARCH_CALLS": "lots",
            "MAX_AGENT_ITERATIONS": "0",
            "REQUEST_TIMEOUT_SECONDS": "-5",
            "LOG_LEVEL": "LOUD",
        })

    message = str(caught.value)
    for name in ("MAX_SEARCH_CALLS", "MAX_AGENT_ITERATIONS", "REQUEST_TIMEOUT_SECONDS", "LOG_LEVEL"):
        assert name in message
    assert KEY not in message


def test_settings_repr_does_not_expose_secrets():
    settings = Settings.from_env({"GEMINI_API_KEY": KEY, "TAVILY_API_KEY": "tvly-secret"})

    assert KEY not in repr(settings)
    assert "tvly-secret" not in repr(settings)
    assert set(settings.secrets) == {KEY, "tvly-secret"}


def test_redacting_formatter_removes_secrets_from_messages_and_tracebacks():
    formatter = RedactingFormatter([KEY], "%(message)s")
    try:
        raise RuntimeError(f"failure with {KEY}")
    except RuntimeError:
        import sys

        record = logging.LogRecord(
            "mra", logging.ERROR, __file__, 1, f"request used {KEY}", None, sys.exc_info()
        )

    output = formatter.format(record)

    assert KEY not in output
    assert "***" in output


def test_configure_logging_writes_redacted_file_and_is_repeatable(tmp_path):
    settings = Settings(gemini_api_key=KEY, log_dir=tmp_path / "logs", log_level="INFO")

    configure_logging(settings)
    logger = configure_logging(settings)  # second call must not duplicate handlers
    logging.getLogger("mra.test").info("hello %s", KEY)
    for handler in logger.handlers:
        handler.flush()

    assert len(logger.handlers) == 1
    content = (tmp_path / "logs" / "agent.log").read_text(encoding="utf-8")
    assert "hello ***" in content
    assert KEY not in content
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
