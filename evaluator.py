"""Deterministic output evaluation.

IMPORTANT LIMITATION: these checks validate *structure and honesty about
sources*, not factual correctness. A well-formed, properly cited answer can
still be wrong. No fact-checking mechanism is implemented.

Evaluation is kept separate from generation: this module never calls an LLM.
"""

import logging
import re
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit

from pydantic import ValidationError

from models import EvaluationResult, ModelAnswer, SearchResult, parse_json_object

logger = logging.getLogger("mra.evaluator")

_URL_PATTERN = re.compile(r"https?://[^\s<>\"'\]\)]+", re.IGNORECASE)

# Heuristic phrases that claim a search/research happened. If nothing was
# retrieved, using them means the answer misrepresents its own provenance.
_SEARCH_CLAIM_PATTERNS = [
    re.compile(r"\b(i|we) (have )?(searched|googled|browsed|looked (it )?up online)\b", re.I),
    re.compile(r"\b(web|internet|online) search (results?|shows?|showed|found|revealed)\b", re.I),
    re.compile(r"\baccording to (the |my |our )?(web |online |internet )?search( results?)?\b", re.I),
    re.compile(r"\bbased on (the |my |our )?(web |online |internet )?search( results?)?\b", re.I),
    re.compile(r"\bthe search results (show|indicate|suggest|say)\b", re.I),
]


@dataclass(frozen=True)
class EvaluationContext:
    """Facts about the run that the evaluator needs.

    Attributes:
        retrieved_sources: results actually returned by a tool in this run.
        allowed_extra_urls: URLs the user supplied in their own question; the
            answer may repeat them without them counting as invented.
        max_output_length: maximum number of characters in ``answer``.
    """

    retrieved_sources: tuple[SearchResult, ...] = ()
    allowed_extra_urls: tuple[str, ...] = ()
    max_output_length: int = 4000


def normalize_url(url: str) -> str:
    """Normalize a URL for comparison (case of scheme/host, fragment, trailing slash)."""
    parts = urlsplit(url.strip())
    return urlunsplit(
        (parts.scheme.lower(), parts.netloc.lower(), parts.path.rstrip("/"), parts.query, "")
    )


def extract_urls(text: str) -> list[str]:
    """Find http(s) URLs in free text, without trailing punctuation."""
    return [match.rstrip(".,;:!?") for match in _URL_PATTERN.findall(text or "")]


class OutputEvaluator:
    """Runs deterministic checks on raw model output."""

    def evaluate(self, raw_output: str, context: EvaluationContext) -> EvaluationResult:
        """Evaluate raw model output.

        Checks:
            well_formed                - valid JSON with the required fields
            non_empty                  - the answer has text
            within_length_limit        - answer length <= configured maximum
            no_false_search_claims     - no "I searched..." wording when nothing
                                         was retrieved (heuristic)
            research_sources_present   - if sources were retrieved, at least
                                         one is cited
            urls_from_search_results   - every URL in the answer / source list
                                         comes from retrieved results (or from
                                         the user's own question)

        Returns:
            An ``EvaluationResult``. ``passed`` is True only if every check
            passed; ``issues`` explains each failure.
        """
        checks: dict[str, bool] = {}
        issues: list[str] = []

        parsed, parse_issue = self._parse(raw_output)
        checks["well_formed"] = parsed is not None
        if parsed is None:
            issues.append(parse_issue or "Output is not well formed.")
            return self._finish(checks, issues, None)

        answer_text = parsed.answer.strip()

        checks["non_empty"] = bool(answer_text)
        if not checks["non_empty"]:
            issues.append("The answer is empty.")

        checks["within_length_limit"] = len(parsed.answer) <= context.max_output_length
        if not checks["within_length_limit"]:
            issues.append(
                f"The answer is {len(parsed.answer)} characters long; "
                f"the limit is {context.max_output_length}."
            )

        retrieved = {normalize_url(s.url) for s in context.retrieved_sources}
        allowed = retrieved | {normalize_url(u) for u in context.allowed_extra_urls}

        checks["no_false_search_claims"] = True
        if not retrieved and any(p.search(answer_text) for p in _SEARCH_CLAIM_PATTERNS):
            checks["no_false_search_claims"] = False
            issues.append(
                "The answer claims that a web search was performed, but no search "
                "results were retrieved."
            )

        cited = {normalize_url(u) for u in parsed.source_urls if u.strip()}
        cited |= {normalize_url(u) for u in extract_urls(parsed.answer)}

        checks["research_sources_present"] = True
        if retrieved and not (cited & retrieved):
            checks["research_sources_present"] = False
            issues.append(
                "Search results were retrieved, but the answer cites none of them "
                "in source_urls."
            )

        invented = sorted(u for u in cited if u not in allowed)
        checks["urls_from_search_results"] = not invented
        if invented:
            issues.append(
                "The answer contains URLs that were not in the search results: "
                + ", ".join(invented[:3])
                + ("..." if len(invented) > 3 else "")
            )

        return self._finish(checks, issues, parsed)

    # --- internals -----------------------------------------------------------
    @staticmethod
    def _parse(raw_output: str) -> tuple[ModelAnswer | None, str | None]:
        try:
            data = parse_json_object(raw_output)
        except ValueError as exc:
            return None, f"Output is malformed: {exc}."
        try:
            return ModelAnswer.model_validate(data), None
        except ValidationError as exc:
            problems = "; ".join(
                f"'{'.'.join(str(p) for p in err['loc'])}' {err['msg'].lower()}"
                for err in exc.errors()
            )
            return None, f"Output does not match the required fields: {problems}."

    @staticmethod
    def _finish(
        checks: dict[str, bool], issues: list[str], parsed: ModelAnswer | None
    ) -> EvaluationResult:
        passed = all(checks.values())
        logger.info(
            "event=evaluation_%s failed_checks=%s",
            "passed" if passed else "failed",
            [name for name, ok in checks.items() if not ok],
        )
        return EvaluationResult(passed=passed, checks=checks, issues=issues, parsed_answer=parsed)
