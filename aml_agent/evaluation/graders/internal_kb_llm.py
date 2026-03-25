"""LLM-judged internal KB precision based on the agent's final report.

Unlike the deterministic ``internal_kb_cleaness_adj_precision`` which counts
TP/FP from raw tool-call results, this metric uses an LLM to read the
agent's **final report** to classify each KB result into one of four verdicts,
then computes precision as TP / (TP + FP).

Verdict semantics
-----------------
``tp``
    The report positively identifies this KB entity as the same entity as the
    investigation subject (or a genuine watchlist match).
``fp``
    The report claims this entity matches the subject, and the report context
    and entity details make clear it is the same context, while it's in fact not.
``dismissed``
    The report acknowledges this entity was found but explicitly states it is
    NOT the same entity.  Correct analysis — excluded from TP+FP denominator.
``not_mentioned``
    The entity does not appear in the report at all — silently ignored.
    Excluded from TP+FP denominator.

Precision = TP / (TP + FP).  When TP+FP = 0 (no positive identifications
were made), precision is 1.0 (vacuously).  Ground truth is treated as a
hint only — the judge trusts the report's own reasoning and entity details
over potentially incomplete ground-truth lists.

Metric produced
---------------
``internal_kb_cleaness_agent_adj_precision_llm``
    0.0 if ``search_knowledgebase`` was never called.

Environment variables
---------------------
``DEFAULT_EVALUATOR_MODEL``
    Model for the judge.  Defaults to ``gemini-2.5-flash-lite``.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from aml_agent.evaluation.types import Evaluation

from .internal_kb import _kb_was_called
from .llm_judge import (
    LLMJudgeConfig,
    build_judge_error_evaluation,
    run_llm_judge_structured,
)

logger = logging.getLogger(__name__)

METRIC_NAME = "internal_kb_cleaness_agent_adj_precision_llm"


@dataclass
class _KBResult:
    """A single result parsed from a ``search_knowledgebase`` response."""

    entity_id: str
    name: str
    source: str


def _extract_kb_results(tool_calls: list[dict[str, Any]]) -> list[_KBResult]:
    """Parse individual KB results from tool responses."""
    results: list[_KBResult] = []
    for tc in tool_calls:
        if tc.get("tool") != "search_knowledgebase":
            continue
        response = tc.get("response", "")
        if not isinstance(response, str):
            continue

        cur_eid: str | None = None
        cur_name: str | None = None
        cur_src: str | None = None

        for line in response.splitlines():
            eid_m = re.match(r"^\s*Entity ID:\s*(.+)$", line)
            if eid_m:
                if cur_eid is not None and cur_src is not None:
                    results.append(_KBResult(cur_eid, cur_name or "", cur_src))
                cur_eid = eid_m.group(1).strip()
                cur_name = None
                cur_src = None
                continue

            name_m = re.match(r"^\s*Name:\s*(.+)$", line)
            if name_m:
                cur_name = name_m.group(1).strip()

            src_m = re.match(r"^\s*Source:\s*(.+)$", line)
            if src_m:
                cur_src = src_m.group(1).strip()

        if cur_eid is not None and cur_src is not None:
            results.append(_KBResult(cur_eid, cur_name or "", cur_src))

    return results


_SYSTEM_PROMPT = """\
You are an expert AML compliance reviewer assessing the precision of an \
AML agent's knowledge-base (KB) search usage.

Your task: for each KB search result, classify how the agent treated it in \
its final investigation report using exactly one of these four verdicts:

- "tp" (True Positive): The report POSITIVELY identifies this KB entity as \
the SAME entity as the investigation subject or a genuine watchlist match, \
AND the identification is well-reasoned — the entity details (name, type, \
country, aliases, source list) are consistent with the subject. Ground truth \
match lists may be incomplete, so a valid match is still "tp" even if ground \
truth is empty.

- "fp" (False Positive): The report claims this entity is a match, but based \
on the full report context and entity details, it clearly is NOT the same \
entity (e.g. superficially similar name but wrong organisation, wrong \
country, fundamentally different entity type).

- "dismissed": The report explicitly acknowledges this entity was found but \
clearly states it is NOT the same entity as the subject (correct analysis of \
a near-miss). This is good agent behaviour — excluded from precision \
denominator.

- "not_mentioned": The entity does not appear anywhere in the report — the \
agent silently ignored the result. Excluded from precision denominator.

IMPORTANT RULES:
1. Only count an entity in the TP+FP denominator if the agent made an \
explicit POSITIVE claim that it matches. Silence or dismissal = excluded.
2. An entity mentioned briefly in a footnote or reference section WITHOUT \
a positive match claim is "not_mentioned" for precision purposes.
3. If multiple KB results refer to the same underlying real-world entity and \
the agent confirms they are all the same match, each is "tp" individually.
4. Do NOT rely on the ground truth list as the sole authority — assess the \
reasoning in the report and the entity details.

Respond with valid JSON only (no markdown fences). Schema:
{
  "entity_classifications": [
    {
      "entity_id": "<entity_id>",
      "verdict": "tp" | "fp" | "dismissed" | "not_mentioned",
      "reason": "One sentence explaining why."
    }
  ],
  "reasoning": "Overall summary of precision assessment."
}
"""

_USER_PROMPT_TEMPLATE = """\
# Investigation Subject
{subject}

# KB Search Results (from search_knowledgebase tool)
{kb_results}

# Agent Final Report
{report}

Classify each KB result entity using the verdict schema.
"""


def _format_kb_results_for_prompt(results: list[_KBResult]) -> str:
    lines: list[str] = []
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. Entity ID: {r.entity_id} | Name: {r.name} | Source: {r.source}")
    return "\n".join(lines)


def internal_kb_agent_precision_llm_grader(
    input: Any,  # noqa: A002
    output: Any,
    expected_output: Any,
    metadata: dict[str, Any] | None = None,
    **kwargs: Any,
) -> list[Evaluation]:
    """LLM-judged KB precision based on the agent's final report.

    Precision = TP / (TP + FP).  Only KB results the agent explicitly
    claimed as positive matches count toward the denominator.  Entities the
    agent correctly dismissed or silently ignored are excluded.

    Returns
    -------
    list[Evaluation]
        Single evaluation: ``internal_kb_cleaness_agent_adj_precision_llm``.
    """
    del metadata, kwargs

    # --- Extract tool calls ---
    tool_calls: list[dict[str, Any]] = []
    if isinstance(output, dict):
        tool_calls = output.get("tool_calls", [])

    # Short-circuit: no tool calling → 0
    if not _kb_was_called(tool_calls):
        return [
            Evaluation(
                name=METRIC_NAME,
                value=0.0,
                comment="search_knowledgebase was never called.",
                metadata={"executed": False},
            )
        ]

    kb_results = _extract_kb_results(tool_calls)

    # No results returned → no positive identifications possible → precision 1.0
    if not kb_results:
        return [
            Evaluation(
                name=METRIC_NAME,
                value=1.0,
                comment="search_knowledgebase returned no results.",
                metadata={"result_count": 0},
            )
        ]

    # --- Get report and subject ---
    report = ""
    subject = ""
    if isinstance(output, dict):
        report = output.get("report_markdown", "") or output.get("report", "") or ""
    if isinstance(report, dict):
        report = json.dumps(report, ensure_ascii=False)
    if isinstance(input, dict):
        subject = (
            input.get("subject", "")
            or input.get("test_case_info_input", "")
            or input.get("test_case_id", "")
        )

    if not report.strip():
        return [
            Evaluation(
                name=METRIC_NAME,
                value=0.0,
                comment="No report found in output — cannot judge precision.",
                metadata={"result_count": len(kb_results)},
            )
        ]

    # --- Call LLM judge ---
    try:
        user_prompt = _USER_PROMPT_TEMPLATE.format(
            subject=subject or "(see report below)",
            kb_results=_format_kb_results_for_prompt(kb_results),
            report=report,
        )

        judge_response = run_llm_judge_structured(
            metric_name=METRIC_NAME,
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )

        # --- Tally verdicts ---
        classifications: list[dict[str, Any]] = judge_response.get("entity_classifications", [])
        reasoning = judge_response.get("reasoning", "")

        verdict_map: dict[str, str] = {
            c["entity_id"]: c.get("verdict", "not_mentioned")
            for c in classifications
            if "entity_id" in c
        }

        tp_ids: list[str] = []
        fp_ids: list[str] = []
        dismissed_ids: list[str] = []
        not_mentioned_ids: list[str] = []

        for r in kb_results:
            v = verdict_map.get(r.entity_id, "not_mentioned")
            if v == "tp":
                tp_ids.append(r.entity_id)
            elif v == "fp":
                fp_ids.append(r.entity_id)
            elif v == "dismissed":
                dismissed_ids.append(r.entity_id)
            else:
                not_mentioned_ids.append(r.entity_id)

        tp = len(tp_ids)
        fp = len(fp_ids)
        denominator = tp + fp

        # Vacuously perfect when the agent made no positive claims at all
        precision = tp / denominator if denominator > 0 else 1.0

        comment = (
            f"TP={tp}, FP={fp}, dismissed={len(dismissed_ids)}, "
            f"not_mentioned={len(not_mentioned_ids)}. "
            f"Precision={precision:.4f}. {reasoning}"
        )

        return [
            Evaluation(
                name=METRIC_NAME,
                value=round(precision, 4),
                comment=comment,
                metadata={
                    "tp_count": tp,
                    "fp_count": fp,
                    "dismissed_count": len(dismissed_ids),
                    "not_mentioned_count": len(not_mentioned_ids),
                    "tp_entity_ids": sorted(tp_ids),
                    "fp_entity_ids": sorted(fp_ids),
                    "dismissed_entity_ids": sorted(dismissed_ids),
                },
            )
        ]

    except Exception as exc:
        logger.exception("LLM judge failed for %s", METRIC_NAME)
        return [build_judge_error_evaluation(metric_name=METRIC_NAME, error=exc)]


__all__ = ["internal_kb_agent_precision_llm_grader"]
