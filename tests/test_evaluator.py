"""Unit tests for the deterministic evaluator."""

import json

from conftest import answer, make_result
from evaluator import EvaluationContext, OutputEvaluator, extract_urls, normalize_url

evaluator = OutputEvaluator()
SOURCE = make_result(1)
RESEARCH = EvaluationContext(retrieved_sources=(SOURCE,), max_output_length=500)
NO_RESEARCH = EvaluationContext(max_output_length=500)


def test_valid_answer_with_research_passes_all_checks():
    result = evaluator.evaluate(answer("Automation saves time.", [SOURCE.url]), RESEARCH)

    assert result.passed
    assert result.issues == []
    assert all(result.checks.values())
    assert set(result.checks) == {
        "well_formed", "non_empty", "within_length_limit",
        "no_false_search_claims", "research_sources_present", "urls_from_search_results",
    }
    assert result.parsed_answer.answer == "Automation saves time."


def test_valid_answer_without_research_passes():
    assert evaluator.evaluate(answer("Paris."), NO_RESEARCH).passed


def test_empty_answer_fails():
    result = evaluator.evaluate(answer("   "), NO_RESEARCH)

    assert not result.passed
    assert result.checks["non_empty"] is False
    assert "empty" in result.issues[0].lower()


def test_malformed_output_fails_with_reason():
    for bad in ["", "Just prose, no JSON", "{broken", "[1, 2, 3]"]:
        result = evaluator.evaluate(bad, NO_RESEARCH)
        assert not result.passed
        assert result.checks == {"well_formed": False}
        assert result.parsed_answer is None
        assert "malformed" in result.issues[0].lower()


def test_missing_required_fields_are_named():
    result = evaluator.evaluate(json.dumps({"answer": "x"}), NO_RESEARCH)  # no source_urls
    assert not result.passed
    assert "source_urls" in result.issues[0]

    result = evaluator.evaluate(json.dumps({"source_urls": []}), NO_RESEARCH)  # no answer
    assert "answer" in result.issues[0]


def test_wrong_field_type_fails():
    result = evaluator.evaluate(json.dumps({"answer": 42, "source_urls": "x"}), NO_RESEARCH)
    assert not result.passed and result.checks["well_formed"] is False


def test_markdown_fenced_json_is_accepted():
    fenced = "```json\n" + answer("Fine.") + "\n```"
    assert evaluator.evaluate(fenced, NO_RESEARCH).passed


def test_over_length_answer_fails_length_check_only():
    result = evaluator.evaluate(answer("x" * 501), NO_RESEARCH)

    assert result.checks["within_length_limit"] is False
    assert result.checks["non_empty"] is True
    assert "501" in result.issues[0]


def test_missing_citations_when_research_was_done():
    result = evaluator.evaluate(answer("Claim without sources.", []), RESEARCH)

    assert not result.passed
    assert result.checks["research_sources_present"] is False


def test_citation_in_answer_text_counts():
    text = f"See {SOURCE.url}. for details"
    assert evaluator.evaluate(answer(text, []), RESEARCH).passed


def test_invented_url_in_source_list_fails():
    result = evaluator.evaluate(answer("x", [SOURCE.url, "https://made-up.example/page"]), RESEARCH)

    assert result.checks["urls_from_search_results"] is False
    assert "made-up.example" in result.issues[0]


def test_invented_url_in_answer_text_fails_even_without_research():
    result = evaluator.evaluate(answer("Read https://made-up.example/page for more."), NO_RESEARCH)

    assert result.checks["urls_from_search_results"] is False


def test_url_from_users_own_question_is_allowed():
    context = EvaluationContext(allowed_extra_urls=("https://docs.python.org/3/",))
    result = evaluator.evaluate(answer("The page https://docs.python.org/3 explains it."), context)

    assert result.passed


def test_claiming_a_search_without_results_fails():
    for claim in [
        "I searched the web and found 70% adoption.",
        "According to my web search, it is popular.",
        "Based on the search results, it is popular.",
        "Online search results show growth.",
    ]:
        result = evaluator.evaluate(answer(claim), NO_RESEARCH)
        assert result.checks["no_false_search_claims"] is False, claim


def test_same_wording_is_fine_when_results_really_exist():
    result = evaluator.evaluate(
        answer("Based on the search results, it is popular.", [SOURCE.url]), RESEARCH
    )
    assert result.passed


def test_url_normalization_ignores_case_fragment_and_trailing_slash():
    assert normalize_url("HTTPS://Example.COM/a/#frag") == normalize_url("https://example.com/a")
    assert normalize_url("https://example.com/a?x=1") != normalize_url("https://example.com/a")


def test_extract_urls_strips_trailing_punctuation():
    assert extract_urls("see https://a.com/x, and (https://b.com/y). ok") == [
        "https://a.com/x",
        "https://b.com/y",
    ]
