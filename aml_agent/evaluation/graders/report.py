"""Report-quality graders.

Evaluates the final investigation report — e.g. risk-level accuracy,
structural completeness.

See ``CONTRIBUTING_EVALUATION.md`` for guidance on adding graders.

Metrics
-------
``report_aml_risk_level_accuracy_llm``
    LLM-judged accuracy of the AML risk level assigned by the agent.
    HIGH / MEDIUM are considered AML-risk-positive; LOW / CLEAR are negative.
    The judge reads the expected output context (excluding raw transactions)
    and the agent's report to determine whether the adjudicated risk level
    is accurate.  Returns 1.0 (correct) or 0.0 (incorrect).

``report_completeness``
    Code-based (no LLM) check that all five required sections are present in
    the report markdown.  Returns a float in [0.0, 1.0] equal to the fraction
    of required sections found.  Full score (1.0) only when all five sections
    are present.

    Required sections (matched as Markdown headings, case-insensitive):
      1. Risk Assessment
      2. Internal Knowledge Base Findings
      3. Wire Transactions
      4. External Search Findings
      5. Sources  (Citations)
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from aml_agent.evaluation.types import Evaluation

from .llm_judge import (
    LLMJudgeConfig,
    build_judge_error_evaluation,
    run_llm_judge_structured,
)

logger = logging.getLogger(__name__)

METRIC_NAME = "report_aml_risk_level_accuracy_llm"

# HIGH / MEDIUM → positive; LOW / CLEAR → negative
_POSITIVE_LEVELS = {"HIGH", "MEDIUM"}
_NEGATIVE_LEVELS = {"LOW", "CLEAR"}


# ── Prompt ──────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are an expert AML compliance reviewer.  Your task is to judge whether \
an AML investigation agent assigned the correct risk level to a subject.

Risk-level semantics:
- HIGH or MEDIUM → AML-risk POSITIVE (subject poses meaningful risk).
- LOW or CLEAR   → AML-risk NEGATIVE (subject is low-risk or cleared).

You will receive:
1. **Expected context** — ground-truth information about the subject \
(entity details, expected watchlist matches, expected web-search findings). \
Use this to form your own opinion of what the correct risk level should be.
2. **Agent report** — the investigation report produced by the agent, \
including its risk assessment.

Evaluation rules:
- Consider the subject's context, expected watchlist matches, and expected \
open-search results to decide the appropriate risk polarity \
(POSITIVE = HIGH/MEDIUM vs NEGATIVE = LOW/CLEAR).
- The agent's risk level is ACCURATE if its polarity matches the expected \
polarity.  For example, if the expected context clearly indicates a \
sanctions-listed entity, the agent should assign HIGH or MEDIUM — either \
is correct.  If the context shows no adverse findings, LOW or CLEAR is \
correct.
- If the expected context is ambiguous, give the agent the benefit of the \
doubt as long as its reasoning in the report is sound.

Respond with valid JSON only (no markdown fences).  Schema:
{
  "expected_polarity": "POSITIVE" | "NEGATIVE",
  "agent_risk_level": "<the risk level the agent assigned>",
  "agent_polarity": "POSITIVE" | "NEGATIVE",
  "accurate": true | false,
  "reasoning": "Brief explanation of your judgment."
}
"""

_USER_PROMPT_TEMPLATE = """\
# Expected Context

## Subject Details
{test_case_details}

## Expected Watchlist Matches
{expected_kb_watchlist_matches}

## Expected Open-Source / Web-Search Findings
{expected_open_search_results}

# Agent Report
{report}
"""


def _extract_report(output: Any) -> str:
    """Get the markdown report from agent output."""
    if not isinstance(output, dict):
        return ""
    report = output.get("report_markdown", "") or ""
    if not report and isinstance(output.get("report"), dict):
        report = json.dumps(output["report"], ensure_ascii=False)
    return report


def _extract_agent_risk_level(output: Any) -> str:
    """Get the risk level string from the agent's parsed report."""
    if isinstance(output, dict):
        report_dict = output.get("report")
        if isinstance(report_dict, dict):
            return (report_dict.get("risk_level") or "").upper().strip()
    return ""


def _safe_str(value: Any) -> str:
    """Convert a value to string suitable for prompt injection."""
    if value is None:
        return "(none)"
    if isinstance(value, str):
        return value if value.strip() else "(none)"
    return json.dumps(value, ensure_ascii=False)


def report_aml_risk_level_accuracy_llm_grader(
    input: Any,  # noqa: A002
    output: Any,
    expected_output: Any,
    metadata: dict[str, Any] | None = None,
    **kwargs: Any,
) -> list[Evaluation]:
    """LLM-judged accuracy of the AML risk level in the agent's report.

    Returns
    -------
    list[Evaluation]
        Single evaluation: ``report_aml_risk_level_accuracy_llm`` (1.0 or 0.0).
    """
    del metadata, kwargs

    # --- Extract report & risk level ---
    report = _extract_report(output)
    agent_risk = _extract_agent_risk_level(output)

    if not report.strip():
        return [
            Evaluation(
                name=METRIC_NAME,
                value=0.0,
                comment="No report found in agent output — cannot judge risk level.",
                metadata={"agent_risk_level": agent_risk},
            )
        ]

    if agent_risk not in (_POSITIVE_LEVELS | _NEGATIVE_LEVELS):
        return [
            Evaluation(
                name=METRIC_NAME,
                value=0.0,
                comment=f"Agent risk level '{agent_risk}' is not a recognised value (HIGH/MEDIUM/LOW/CLEAR).",
                metadata={"agent_risk_level": agent_risk},
            )
        ]

    # --- Build expected-context section (exclude transactions) ---
    eo = expected_output if isinstance(expected_output, dict) else {}
    test_case_details = _safe_str(eo.get("test_case_details"))
    expected_kb = _safe_str(eo.get("expected_kb_watchlist_matches"))
    expected_web = _safe_str(eo.get("expected_open_search_results"))

    # --- Call LLM judge ---
    try:
        user_prompt = _USER_PROMPT_TEMPLATE.format(
            test_case_details=test_case_details,
            expected_kb_watchlist_matches=expected_kb,
            expected_open_search_results=expected_web,
            report=report,
        )

        judge_response = run_llm_judge_structured(
            metric_name=METRIC_NAME,
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )

        accurate = bool(judge_response.get("accurate", False))
        reasoning = judge_response.get("reasoning", "")
        expected_polarity = judge_response.get("expected_polarity", "")
        agent_polarity = judge_response.get("agent_polarity", "")

        value = 1.0 if accurate else 0.0
        comment = (
            f"Agent risk={agent_risk} ({agent_polarity}), "
            f"expected polarity={expected_polarity}. "
            f"{'Accurate' if accurate else 'Inaccurate'}. {reasoning}"
        )

        return [
            Evaluation(
                name=METRIC_NAME,
                value=value,
                comment=comment,
                metadata={
                    "agent_risk_level": agent_risk,
                    "agent_polarity": agent_polarity,
                    "expected_polarity": expected_polarity,
                    "accurate": accurate,
                },
            )
        ]

    except Exception as exc:
        logger.exception("LLM judge failed for %s", METRIC_NAME)
        return [build_judge_error_evaluation(metric_name=METRIC_NAME, error=exc)]


# ---------------------------------------------------------------------------
# report_completeness — code-based, no LLM
# ---------------------------------------------------------------------------

COMPLETENESS_METRIC_NAME = "report_completeness"

# Each entry: (canonical label, list of regex patterns that match the heading).
# A section is found when ANY pattern matches a Markdown heading line
# (## ... or # ...) in the report, case-insensitively.
_REQUIRED_SECTIONS: list[tuple[str, list[str]]] = [
    (
        "Risk Assessment",
        [r"risk\s+assessment"],
    ),
    (
        "Internal Knowledge Base Findings",
        [r"internal\s+knowledge\s+base", r"knowledge\s+base\s+findings?"],
    ),
    (
        "Wire Transactions",
        [r"wire\s+transactions?", r"transaction\s+findings?"],
    ),
    (
        "External Search Findings",
        [r"external\s+search", r"web\s+search\s+findings?", r"open.source\s+search"],
    ),
    (
        "Sources / Citations",
        [r"^#{1,3}\s+sources?\b", r"^#{1,3}\s+citations?\b", r"^#{1,3}\s+references?\b"],
    ),
]

# Pre-compile: heading line pattern + per-section keyword patterns
_HEADING_RE = re.compile(r"^#{1,4}\s+(.+)$", re.MULTILINE)


def _section_patterns() -> list[tuple[str, list[re.Pattern[str]]]]:
    compiled = []
    for label, patterns in _REQUIRED_SECTIONS:
        compiled.append(
            (label, [re.compile(p, re.IGNORECASE) for p in patterns])
        )
    return compiled


_SECTION_PATTERNS = _section_patterns()


def _check_sections(report_markdown: str) -> dict[str, bool]:
    """Return a dict mapping each required section label → whether it was found."""
    # Collect all heading text lines
    headings = [m.group(1).strip() for m in _HEADING_RE.finditer(report_markdown)]
    # Also check full heading lines (including ##) for patterns that anchor to ^
    heading_lines = [m.group(0).strip() for m in _HEADING_RE.finditer(report_markdown)]

    found: dict[str, bool] = {}
    for label, patterns in _SECTION_PATTERNS:
        matched = False
        for pat in patterns:
            # Try matching against heading text
            if any(pat.search(h) for h in headings):
                matched = True
                break
            # Try matching against full heading line (e.g. "## Sources")
            if any(pat.search(hl) for hl in heading_lines):
                matched = True
                break
        found[label] = matched
    return found


def report_completeness_grader(
    input: Any,  # noqa: A002
    output: Any,
    expected_output: Any,
    metadata: dict[str, Any] | None = None,
    **kwargs: Any,
) -> list[Evaluation]:
    """Code-based check that all required report sections are present.

    Does **not** call an LLM — uses regex heading detection only.

    Returns
    -------
    list[Evaluation]
        Single evaluation: ``report_completeness`` in [0.0, 1.0].
        Score = (number of sections found) / (total required sections).
        1.0 means all five sections are present.
    """
    del input, expected_output, metadata, kwargs

    report = _extract_report(output)

    if not report.strip():
        return [
            Evaluation(
                name=COMPLETENESS_METRIC_NAME,
                value=0.0,
                comment="No report found in agent output.",
                metadata={"sections_found": {}, "sections_missing": [label for label, _ in _REQUIRED_SECTIONS]},
            )
        ]

    section_results = _check_sections(report)
    n_found = sum(section_results.values())
    n_total = len(section_results)
    score = n_found / n_total

    missing = [label for label, present in section_results.items() if not present]
    present = [label for label, present in section_results.items() if present]

    comment_parts = [f"{n_found}/{n_total} required sections present."]
    if missing:
        comment_parts.append(f"Missing: {', '.join(missing)}.")

    return [
        Evaluation(
            name=COMPLETENESS_METRIC_NAME,
            value=score,
            comment=" ".join(comment_parts),
            metadata={
                "sections_found": present,
                "sections_missing": missing,
                "section_detail": section_results,
            },
        )
    ]


# ---------------------------------------------------------------------------
# report_groundedness — LLM-judged, 4 dimensions
# ---------------------------------------------------------------------------

GROUNDEDNESS_METRIC_NAME = "report_groundedness_llm"
_GROUNDEDNESS_SCORE_DIMS = ["attribution", "faithfulness", "coverage", "hallucination"]

_GROUNDEDNESS_SYSTEM_PROMPT = """\
You are an evaluator for an AML investigation agent.

You are given:
1. Retrieved evidence (KB watchlist, web/opensearch, transaction matches, transaction match details)
2. Generated report

Your task:
Evaluate whether the report is grounded ONLY in the provided evidence.

For each claim in the report:
- Identify supporting evidence (if any)
- Flag hallucinations (claims with no evidence support)

Then score each dimension on a 0-2 scale:

1. **Attribution** (0-2): Are claims in the report backed by cited evidence?
   - 0: No claims cite evidence
   - 1: Some claims cite evidence, others do not
   - 2: All substantive claims properly cite their evidence source

2. **Faithfulness** (0-2): Does the report accurately represent what the evidence says?
   - 0: Report contradicts or significantly misrepresents the evidence
   - 1: Mostly faithful but contains minor unsupported inferences
   - 2: All claims are faithful and accurately reflect the evidence

3. **Coverage** (0-2): Does the report reflect all key evidence found?
   - 0: Report ignores most of the retrieved evidence
   - 1: Report uses some evidence but omits significant findings
   - 2: All key evidence is reflected in the report

4. **Hallucination** (0-2): Is the report free of fabricated content?
   - 0: Contains major fabricated claims not in the evidence
   - 1: Contains minor embellishments or unverifiable details
   - 2: No hallucinated content; everything is grounded in evidence

Return ONLY valid JSON (no markdown fences, no extra text). Use this exact schema:
{
  "attribution": <0-2>,
  "faithfulness": <0-2>,
  "coverage": <0-2>,
  "hallucination": <0-2>,
  "justification": "<2-4 sentences explaining the scores, citing specific examples>"
}
"""

_GROUNDEDNESS_USER_PROMPT_TEMPLATE = """\
# Investigation Subject
{subject}

# Retrieved Evidence

{evidence}

# Generated Report

{report}

Evaluate whether the report is grounded in the provided evidence. Return JSON only.
"""


def _extract_evidence(output: Any) -> str:
    """Extract all retrieved evidence from tool calls into a structured text block."""
    if not isinstance(output, dict):
        return "(no evidence retrieved)"

    sections: list[str] = []

    for tc in output.get("tool_calls", []):
        tool = tc.get("tool", "")
        response = tc.get("response", "")
        args = tc.get("args", {}) or {}

        if tool == "search_knowledgebase":
            sections.append(
                f"### KB Watchlist Search\nQuery: {args.get('keyword', '')}\n\n{response}"
            )
        elif tool == "web_search":
            sections.append(
                f"### Web Search (OpenSearch)\nQuery: {args.get('query', '')}\n\n{response}"
            )
        elif tool == "execute":
            sections.append(
                f"### SQL Query Result\nQuery: {args.get('query', '')}\n\n{response}"
            )
        elif tool == "get_schema_info":
            sections.append(f"### Database Schema\n{response}")

    for sr in output.get("sql_results", []):
        rows_text = json.dumps(sr.get("rows", [])[:20], indent=2)
        sections.append(
            f"### Transaction Matches (structured)\n"
            f"Query: {sr.get('query', '')}\n"
            f"Columns: {sr.get('columns')}\n"
            f"Rows:\n{rows_text}"
        )

    return "\n\n---\n\n".join(sections) if sections else "(no evidence retrieved)"


def report_groundedness_llm_grader(
    input: Any,  # noqa: A002
    output: Any,
    expected_output: Any,
    metadata: dict[str, Any] | None = None,
    **kwargs: Any,
) -> list[Evaluation]:
    """LLM-judged groundedness of the agent's report against retrieved evidence.

    Scores the report on four dimensions (0–2 each):
    - **attribution**: claims backed by cited evidence
    - **faithfulness**: accurate representation of evidence
    - **coverage**: all key evidence reflected
    - **hallucination**: absence of fabricated content

    Returns one ``Evaluation`` per dimension plus a composite score normalised
    to [0.0, 1.0] (``report_groundedness_composite``).
    """
    del input, expected_output, metadata, kwargs

    report = _extract_report(output)
    subject = (output.get("subject", "") or "") if isinstance(output, dict) else ""
    evidence = _extract_evidence(output)

    if not report.strip():
        error_evals = [
            Evaluation(
                name=f"{GROUNDEDNESS_METRIC_NAME}_{dim}",
                value=0.0,
                comment="No report found in agent output.",
            )
            for dim in _GROUNDEDNESS_SCORE_DIMS
        ]
        error_evals.append(
            Evaluation(
                name=f"{GROUNDEDNESS_METRIC_NAME}_composite",
                value=0.0,
                comment="No report found in agent output.",
            )
        )
        return error_evals

    try:
        user_prompt = _GROUNDEDNESS_USER_PROMPT_TEMPLATE.format(
            subject=subject,
            evidence=evidence,
            report=report,
        )
        scores = run_llm_judge_structured(
            metric_name=GROUNDEDNESS_METRIC_NAME,
            system_prompt=_GROUNDEDNESS_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )

        justification = scores.get("justification", "")
        raw_scores = [float(scores[dim]) for dim in _GROUNDEDNESS_SCORE_DIMS if dim in scores]
        # Normalise each dimension from 0-2 scale → 0-1
        evals = [
            Evaluation(
                name=f"{GROUNDEDNESS_METRIC_NAME}_{dim}",
                value=round(float(scores[dim]) / 2.0, 3),
                comment=justification,
                metadata={"subject": subject, "raw_score_0_2": float(scores[dim])},
            )
            for dim in _GROUNDEDNESS_SCORE_DIMS
        ]
        composite = (sum(raw_scores) / (2.0 * len(raw_scores))) if raw_scores else 0.0
        evals.append(
            Evaluation(
                name=f"{GROUNDEDNESS_METRIC_NAME}_composite",
                value=round(composite, 3),
                comment=justification,
                metadata={"subject": subject, "raw_scores": dict(zip(_GROUNDEDNESS_SCORE_DIMS, raw_scores))},
            )
        )
        return evals

    except Exception as exc:
        logger.exception("LLM judge failed for %s", GROUNDEDNESS_METRIC_NAME)
        err_evals = [
            build_judge_error_evaluation(
                metric_name=f"{GROUNDEDNESS_METRIC_NAME}_{dim}", error=exc
            )
            for dim in _GROUNDEDNESS_SCORE_DIMS
        ]
        err_evals.append(
            build_judge_error_evaluation(
                metric_name=f"{GROUNDEDNESS_METRIC_NAME}_composite", error=exc
            )
        )
        return err_evals


__all__ = ["report_aml_risk_level_accuracy_llm_grader", "report_groundedness_llm_grader"]
