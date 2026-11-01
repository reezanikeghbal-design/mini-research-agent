"""Tests for CLI rendering and the interactive loop (scripted input, no real I/O)."""

import pytest

import app
from conftest import answer, decision, make_result
from models import AgentResponse


def test_format_completed_response_shows_answer_sources_and_status():
    source = make_result(1)
    text = app.format_response(
        AgentResponse(
            status="completed",
            conversation_id="c",
            answer="The answer.",
            sources=[source],
            searches_performed=1,
        )
    )

    assert "The answer." in text
    assert f"1. {source.title} - {source.url}" in text
    assert "[status: completed | searches: 1 | limited mode: no]" in text
    assert "PARTIAL" not in text


def test_format_partial_response_is_clearly_marked_with_notes():
    text = app.format_response(
        AgentResponse(
            status="partial", conversation_id="c", answer="Best effort.", warnings=["Web search failed: x"]
        )
    )

    assert "[PARTIAL RESULT]" in text
    assert "Notes:" in text and "Web search failed: x" in text


def test_format_failed_response_shows_error_not_answer():
    text = app.format_response(
        AgentResponse(status="failed", conversation_id="c", error="The model request timed out.")
    )

    assert text.startswith("[FAILED] The model request timed out.")


def test_format_limited_mode_is_visible():
    text = app.format_response(
        AgentResponse(status="completed", conversation_id="c", answer="x", limited_mode=True)
    )

    assert "limited mode: yes" in text


def run_session(agent, lines, conversation_id="conv-a"):
    inputs = iter(lines)
    output: list[str] = []

    def fake_input(prompt):
        try:
            return next(inputs)
        except StopIteration:
            raise EOFError

    app.run_interactive(agent, conversation_id, input_fn=fake_input, output_fn=output.append)
    return "\n".join(output)


def test_interactive_session_answers_then_exits(build_agent):
    agent, provider, _, memory = build_agent([decision("answer"), answer("Hi there.")], [])

    output = run_session(agent, ["", "hello", "/exit"])

    assert "Hi there." in output
    assert "Goodbye." in output
    assert len(memory.get_records("conv-a")) == 1


def test_new_command_starts_a_fresh_conversation(build_agent):
    agent, provider, _, memory = build_agent(
        [decision("answer"), answer("One."), decision("answer"), answer("Two.")], []
    )

    output = run_session(agent, ["first", "/new", "second", "quit"])

    assert "Started a new conversation" in output
    assert len(memory.get_records("conv-a")) == 1  # the second question went elsewhere


def test_help_and_eof_are_handled(build_agent):
    agent, _, _, _ = build_agent([], [])

    output = run_session(agent, ["/help"])  # input then raises EOFError

    assert "/new" in output and "Goodbye." in output


def test_unexpected_bug_does_not_kill_the_session(build_agent, monkeypatch):
    agent, _, _, _ = build_agent([], [])

    def explode(question, conversation_id):
        raise RuntimeError("bug")

    monkeypatch.setattr(agent, "run", explode)

    output = run_session(agent, ["hello", "/exit"])

    assert "unexpected internal error" in output
    assert "Goodbye." in output


def test_main_reports_missing_api_key_without_traceback(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    # load_dotenv() searches next to app.py, not the current directory, so a real
    # .env in the project folder would still be loaded. Disable it for this test.
    monkeypatch.setattr(app, "load_dotenv", lambda *args, **kwargs: False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    code = app.main(["--ask", "hello"])

    err = capsys.readouterr().err
    assert code == 2
    assert "GEMINI_API_KEY" in err
    assert "Traceback" not in err
