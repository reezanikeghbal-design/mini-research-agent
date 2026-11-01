"""Tests for the Gemini provider using a fake client (the SDK is not required)."""

import pytest

from errors import (
    ConfigError,
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitError,
    ProviderResponseError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from models import ModelAnswer
from providers.gemini_provider import GeminiProvider


class FakeResponse:
    def __init__(self, text):
        self.text = text


class FakeModels:
    def __init__(self, response=None, error=None):
        self.response, self.error, self.calls = response, error, []

    def generate_content(self, *, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        if self.error:
            raise self.error
        return self.response


class FakeClient:
    def __init__(self, **kwargs):
        self.models = FakeModels(**kwargs)


class FakeAPIError(Exception):
    """Mimics an SDK API error, which carries an HTTP status ``code``."""

    def __init__(self, code, message="details that must not leak"):
        super().__init__(message)
        self.code = code


def provider_with(client) -> GeminiProvider:
    return GeminiProvider(api_key="k", model="gemini-test", client=client)


def test_generate_returns_text_and_sends_expected_request():
    client = FakeClient(response=FakeResponse('{"answer": "a", "source_urls": []}'))

    text = provider_with(client).generate(
        system_instruction="rules", prompt="question", response_schema=ModelAnswer, temperature=0.1
    )

    assert text == '{"answer": "a", "source_urls": []}'
    call = client.models.calls[0]
    assert call["model"] == "gemini-test"
    assert call["contents"] == "question"
    assert call["config"]["system_instruction"] == "rules"
    assert call["config"]["temperature"] == 0.1
    assert call["config"]["response_mime_type"] == "application/json"
    assert call["config"]["response_schema"] is ModelAnswer


def test_plain_text_request_has_no_schema():
    client = FakeClient(response=FakeResponse("hello"))

    provider_with(client).generate(system_instruction="s", prompt="p")

    assert "response_schema" not in client.models.calls[0]["config"]
    assert "response_mime_type" not in client.models.calls[0]["config"]


@pytest.mark.parametrize("text", [None, "", "   "])
def test_empty_or_missing_text_raises_response_error(text):
    with pytest.raises(ProviderResponseError):
        provider_with(FakeClient(response=FakeResponse(text))).generate(
            system_instruction="s", prompt="p"
        )


@pytest.mark.parametrize(
    "error, expected",
    [
        (FakeAPIError(429), ProviderRateLimitError),
        (FakeAPIError(503), ProviderUnavailableError),
        (FakeAPIError(500), ProviderUnavailableError),
        (FakeAPIError(504), ProviderTimeoutError),
        (FakeAPIError(401), ProviderAuthError),
        (FakeAPIError(403), ProviderAuthError),
        (TimeoutError(), ProviderTimeoutError),
        (type("ReadTimeout", (Exception,), {})(), ProviderTimeoutError),
        (type("ConnectError", (Exception,), {})(), ProviderUnavailableError),
    ],
)
def test_sdk_errors_are_translated_to_application_errors(error, expected):
    with pytest.raises(expected) as caught:
        provider_with(FakeClient(error=error)).generate(system_instruction="s", prompt="p")

    assert "must not leak" not in str(caught.value)


def test_unknown_error_becomes_generic_provider_error():
    with pytest.raises(ProviderError) as caught:
        provider_with(FakeClient(error=FakeAPIError(400))).generate(system_instruction="s", prompt="p")

    assert not isinstance(caught.value, (ProviderTimeoutError, ProviderRateLimitError))
    assert "HTTP 400" in str(caught.value)


def test_missing_api_key_is_a_config_error():
    with pytest.raises(ConfigError, match="GEMINI_API_KEY"):
        GeminiProvider(api_key="", model="m")
