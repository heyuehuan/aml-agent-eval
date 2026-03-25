"""Report-quality graders.

Evaluates the final investigation report — e.g. risk-level accuracy.

See ``CONTRIBUTING_EVALUATION.md`` for guidance on adding graders.

Metrics
-------
``report_aml_risk_level_accuracy_llm``
    LLM-judged accuracy of the AML risk level assigned by the agent.
    HIGH / MEDIUM are considered AML-risk-positive; LOW / CLEAR are negative.
    The judge reads the expected output context (excluding raw transactions)
    and the agent's report to determine whether the adjudicated risk level
    is accurate.  Returns 1.0 (correct) or 0.0 (incorrect).
"""

from __future__ import annotations

import json
import logging
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


__all__ = ["report_aml_risk_level_accuracy_llm_grader"]
