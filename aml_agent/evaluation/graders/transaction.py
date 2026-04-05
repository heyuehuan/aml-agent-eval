"""Transaction retrieval graders.

Metrics
-------
``sql_result_score_recall``
    Fraction of expected transaction IDs that appear in the agent's SQL
    query results.  Score = |found ∩ expected| / |expected|.
    Returns 1.0 when no transactions are expected (trivially correct).

``sql_result_score_precision``
    Fraction of agent-found transaction IDs that are in the expected set.
    Score = |found ∩ expected| / |found|.
    Returns 1.0 when the agent retrieved nothing (vacuous precision).

``transaction_aggregation_score_llm``
    LLM-as-a-judge that checks whether the agent's report narrative
    accurately describes the expected transaction volume.  Most useful when
    the agent produced no SQL result rows but may have summarised findings
    in prose.

See ``CONTRIBUTING_EVALUATION.md`` for guidance on adding graders.
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_found_transaction_ids(output: Any) -> tuple[set[str], bool]:
    """Extract unique transaction IDs returned across all SQL query results.

    Returns
    -------
    tuple[set[str], bool]
        ``(found_ids, has_unresolvable_results)`` where
        ``has_unresolvable_results`` is True when at least one SQL result has
        rows but no ``transaction_id`` column, meaning those transactions are
        invisible to this grader.
    """
    if not isinstance(output, dict):
        return set(), False
    sql_results = output.get("sql_results") or []
    found: set[str] = set()
    has_unresolvable = False
    for result in sql_results:
        if not isinstance(result, dict):
            continue
        columns = result.get("columns") or []
        rows = result.get("rows") or []
        if "transaction_id" not in columns:
            if rows:  # result has rows but no transaction_id — can't determine IDs
                has_unresolvable = True
            continue
        tid_idx = columns.index("transaction_id")
        for row in rows:
            if isinstance(row, (list, tuple)) and len(row) > tid_idx:
                tid = row[tid_idx]
                if tid:
                    found.add(str(tid))
    return found, has_unresolvable


def _get_expected_ids(expected_output: Any) -> list[str]:
    """Return the list of expected transaction IDs from the test-case row."""
    if not isinstance(expected_output, dict):
        return []
    ids = expected_output.get("expected_transaction_matches") or []
    if isinstance(ids, str):
        try:
            ids = json.loads(ids)
        except (json.JSONDecodeError, TypeError):
            return []
    return [str(i) for i in ids if i]


def _sql_was_called(output: Any) -> bool:
    """Return True if the agent made at least one SQL query against the transactions table."""
    if not isinstance(output, dict):
        return False
    sql_results = output.get("sql_results")
    if isinstance(sql_results, list) and len(sql_results) > 0:
        return True
    # sql_results may be empty even when SQL was executed (e.g. queries that returned
    # no rows, or queries whose results weren't stored there). Fall back to tool_calls.
    tool_calls = output.get("tool_calls")
    if isinstance(tool_calls, list):
        return any(
            isinstance(tc, dict) and tc.get("tool") in ("execute", "sql", "run_sql")
            for tc in tool_calls
        )
    return False


# ---------------------------------------------------------------------------
# sql_result_score_recall
# ---------------------------------------------------------------------------

_RECALL_METRIC = "sql_result_score_recall"


def sql_result_score_recall_grader(
    input: Any,  # noqa: A002
    output: Any,
    expected_output: Any,
    metadata: dict[str, Any] | None = None,
    **kwargs: Any,
) -> list[Evaluation]:
    """Fraction of expected transactions retrieved by the agent's SQL queries.

    Returns
    -------
    list[Evaluation]
        Single evaluation: |found ∩ expected| / |expected|.
        Score is 1.0 when no transactions are expected.
    """
    del input, metadata, kwargs

    expected_ids = set(_get_expected_ids(expected_output))
    found_ids, has_unresolvable = _extract_found_transaction_ids(output)

    if not expected_ids:
        return [
            Evaluation(
                name=_RECALL_METRIC,
                value=1.0,
                comment="No transactions expected — recall is trivially 1.0.",
            )
        ]

    if not _sql_was_called(output):
        return [
            Evaluation(
                name=_RECALL_METRIC,
                value=0.0,
                comment="Agent made no SQL queries — recall is 0.",
            )
        ]

    hits = found_ids & expected_ids
    recall = round(len(hits) / len(expected_ids), 3)
    missed = sorted(expected_ids - found_ids)
    comment = f"{len(hits)}/{len(expected_ids)} expected transactions retrieved."
    if missed:
        comment += f" Missed IDs: {missed}"
    if has_unresolvable:
        comment += " (some SQL results lacked a transaction_id column and could not be counted)"

    return [
        Evaluation(
            name=_RECALL_METRIC,
            value=recall,
            comment=comment,
            metadata={
                "found_ids": sorted(found_ids),
                "expected_ids": sorted(expected_ids),
                "hit_ids": sorted(hits),
                "missed_ids": missed,
                "has_unresolvable_results": has_unresolvable,
            },
        )
    ]


# ---------------------------------------------------------------------------
# sql_result_score_precision
# ---------------------------------------------------------------------------

_PRECISION_METRIC = "sql_result_score_precision"


def sql_result_score_precision_grader(
    input: Any,  # noqa: A002
    output: Any,
    expected_output: Any,
    metadata: dict[str, Any] | None = None,
    **kwargs: Any,
) -> list[Evaluation]:
    """Fraction of the agent's retrieved transactions that are in the expected set.

    Returns
    -------
    list[Evaluation]
        Single evaluation: |found ∩ expected| / |found|.
        Score is 1.0 when the agent retrieved nothing (vacuous precision).
    """
    del input, metadata, kwargs

    expected_ids = set(_get_expected_ids(expected_output))
    found_ids, has_unresolvable = _extract_found_transaction_ids(output)

    if not _sql_was_called(output):
        # No queries made at all — vacuous only when no transactions were expected
        if not expected_ids:
            return [
                Evaluation(
                    name=_PRECISION_METRIC,
                    value=1.0,
                    comment="No transactions expected and no SQL queries made — precision is trivially 1.0.",
                )
            ]
        return [
            Evaluation(
                name=_PRECISION_METRIC,
                value=0.0,
                comment="Agent made no SQL queries — precision is 0.",
            )
        ]

    if not found_ids and not has_unresolvable:
        if not expected_ids:
            return [
                Evaluation(
                    name=_PRECISION_METRIC,
                    value=1.0,
                    comment="Agent ran SQL queries but retrieved no rows — precision is vacuously 1.0.",
                )
            ]
        # Transactions were expected but agent retrieved no transaction rows — 0/0 → 0.
        return [
            Evaluation(
                name=_PRECISION_METRIC,
                value=0.0,
                comment="Agent ran SQL queries but retrieved no transaction rows — treated as precision is 0.",
            )
        ]

    if not found_ids and has_unresolvable:
        return [
            Evaluation(
                name=_PRECISION_METRIC,
                value=0.0,
                comment=(
                    "Agent ran SQL queries but no result included a transaction_id column "
                    "— cannot compute precision."
                ),
                metadata={"has_unresolvable_results": True},
            )
        ]

    hits = found_ids & expected_ids
    precision = round(len(hits) / len(found_ids), 3)
    false_positives = sorted(found_ids - expected_ids)
    comment = f"{len(hits)}/{len(found_ids)} retrieved transactions are in the expected set."
    if false_positives:
        comment += f" Unexpected IDs: {false_positives}"
    if has_unresolvable:
        comment += " (additional SQL results lacked a transaction_id column and are not counted)"

    return [
        Evaluation(
            name=_PRECISION_METRIC,
            value=precision,
            comment=comment,
            metadata={
                "found_ids": sorted(found_ids),
                "expected_ids": sorted(expected_ids),
                "hit_ids": sorted(hits),
                "false_positive_ids": false_positives,
                "has_unresolvable_results": has_unresolvable,
            },
        )
    ]


# ---------------------------------------------------------------------------
# transaction_aggregation_score_llm — LLM-as-a-judge
# ---------------------------------------------------------------------------

_AGGREGATION_METRIC = "transaction_aggregation_score_llm"

_AGGREGATION_SYSTEM_PROMPT = (
    'You are an expert with numbers. Return JSON: {"score": 0 or 1, "reason": "\u2026"}'
)

_AGGREGATION_USER_INSTRUCTIONS = (
    'You are evaluating an aggregated computation.\n'
    'Return JSON: {"score": 0 or 1, "reason": "..."}.\n'
    'score=1 if the observed summary text is correct vs the expected aggregation text,'
    ' score=0 otherwise.'
)

_AGGREGATION_USER_PROMPT_TEMPLATE = (
    "{user_instructions}\n\n"
    "Observed summary text:\n{report_text}\n\n"
    "Expected aggregation text:\n{expected_summary}\n\n"
    "Please compare the observed summary text with the expected aggregation text "
    "and return JSON with score 0 or 1 and reason."
)


def _build_expected_summary(expected_output: Any, output: Any) -> str:
    """Build a plain-text aggregation summary matching the notebook's canonical format.

    Mirrors ``_build_expected_aggregation_text`` from the notebook:
    "A total of N transactions with a cumulative amount of X.XX were identified
    with Y distinct counterparties."

    Amounts and counterparties are derived from the agent's own SQL results
    filtered to the expected transaction IDs, so we compare apples-to-apples
    without re-reading the source CSV.
    """
    expected_ids = set(_get_expected_ids(expected_output))
    n = len(expected_ids)

    if n == 0:
        return (
            "A total of 0 transactions with a cumulative amount of 0.00 "
            "were identified."
        )

    # Pull amount, sender_name, receiver_name for the expected IDs from sql_results.
    total_amount = 0.0
    counterparties: set[str] = set()
    found_count = 0

    sql_results = (output or {}).get("sql_results") or [] if isinstance(output, dict) else []
    for result in sql_results:
        if not isinstance(result, dict):
            continue
        columns = result.get("columns") or []
        rows = result.get("rows") or []
        try:
            tid_idx = columns.index("transaction_id")
        except ValueError:
            continue
        amt_idx = columns.index("amount") if "amount" in columns else -1
        snd_idx = columns.index("sender_name") if "sender_name" in columns else -1
        rcv_idx = columns.index("receiver_name") if "receiver_name" in columns else -1
        for row in rows:
            if not (isinstance(row, (list, tuple)) and len(row) > tid_idx):
                continue
            tid = str(row[tid_idx])
            if tid not in expected_ids:
                continue
            found_count += 1
            if amt_idx >= 0 and len(row) > amt_idx:
                try:
                    total_amount += float(row[amt_idx])
                except (ValueError, TypeError):
                    pass
            if snd_idx >= 0 and len(row) > snd_idx and row[snd_idx]:
                counterparties.add(str(row[snd_idx]))
            if rcv_idx >= 0 and len(row) > rcv_idx and row[rcv_idx]:
                counterparties.add(str(row[rcv_idx]))

    if found_count > 0:
        if counterparties:
            return (
                f"A total of {n} transactions with a cumulative amount of "
                f"{total_amount:,.2f} were identified with "
                f"{len(counterparties)} distinct counterparties."
            )
        return (
            f"A total of {n} transactions with a cumulative amount of "
            f"{total_amount:,.2f} were identified."
        )
    # Fallback when SQL results don’t cover the expected IDs (e.g. agent missed them)
    return f"A total of {n} transactions were expected for this subject."


def transaction_aggregation_score_llm_grader(
    input: Any,  # noqa: A002
    output: Any,
    expected_output: Any,
    metadata: dict[str, Any] | None = None,
    **kwargs: Any,
) -> list[Evaluation]:
    """LLM-judged accuracy of the agent's reported transaction aggregation.

    Checks whether the agent's ``report_markdown`` correctly describes the
    expected transaction volume and key facts.  Particularly useful when the
    agent produced no SQL result rows but may have summarised findings in prose.

    Returns
    -------
    list[Evaluation]
        Single evaluation with value 0 or 1 and a reason string.
    """
    del input, metadata, kwargs

    if not isinstance(output, dict):
        return [
            Evaluation(
                name=_AGGREGATION_METRIC,
                value=0.0,
                comment="No agent output available.",
            )
        ]

    report_text = output.get("report_markdown", "")
    if not report_text:
        # Fallback: reconstruct text from structured report sections
        report = output.get("report") or {}
        if isinstance(report, dict):
            sections = report.get("sections") or []
            parts = [s.get("body", "") for s in sections if isinstance(s, dict)]
            report_text = "\n\n".join(filter(None, parts))

    if not report_text.strip():
        return [
            Evaluation(
                name=_AGGREGATION_METRIC,
                value=0.0,
                comment="No report narrative found in agent output.",
            )
        ]

    expected_summary = _build_expected_summary(expected_output, output)
    user_prompt = _AGGREGATION_USER_PROMPT_TEMPLATE.format(
        user_instructions=_AGGREGATION_USER_INSTRUCTIONS,
        report_text=report_text,
        expected_summary=expected_summary,
    )

    try:
        result = run_llm_judge_structured(
            metric_name=_AGGREGATION_METRIC,
            system_prompt=_AGGREGATION_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            config=LLMJudgeConfig(temperature=0.0),
        )

        raw_score = result.get("score", 0)
        if isinstance(raw_score, (int, float)):
            score = 1 if int(raw_score) == 1 else 0
        else:
            try:
                score = 1 if int(str(raw_score).strip().split(".")[0]) == 1 else 0
            except Exception:
                score = 0

        reason = result.get("reason", "No reason provided.")
        return [Evaluation(name=_AGGREGATION_METRIC, value=score, comment=reason)]

    except Exception as exc:
        logger.exception("LLM judge failed for %s", _AGGREGATION_METRIC)
        return [build_judge_error_evaluation(metric_name=_AGGREGATION_METRIC, error=exc)]


__all__ = [
    "sql_result_score_recall_grader",
    "sql_result_score_precision_grader",
    "transaction_aggregation_score_llm_grader",
]
