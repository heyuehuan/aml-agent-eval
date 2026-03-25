"""Internal knowledge-base graders (item-level, deterministic).

Two adjusted metrics that evaluate the agent's ``search_knowledgebase``
tool-call results against the ground-truth ``expected_kb_watchlist_matches``.

Metrics produced
----------------
``internal_kb_coverage_adj_recall``
    Adjusted recall — fraction of expected watchlist sources found in KB
    tool-call responses.  0.0 if the tool was never called (unconditionally),
    1.0 if the tool was called and no sources were expected.

``internal_kb_cleaness_adj_precision``
    Adjusted precision — TP / (TP + FP) computed over individual KB search
    results.  0.0 if the tool was never called (unconditionally), 1.0 if no
    sources were expected and the search returned empty.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from aml_agent.evaluation.types import Evaluation


@dataclass
class _KBResult:
    """A single result parsed from a ``search_knowledgebase`` response."""

    entity_id: str
    source: str


def _extract_kb_results(tool_calls: list[dict[str, Any]]) -> list[_KBResult]:
    """Parse individual (entity_id, source) pairs from KB tool responses."""

    results: list[_KBResult] = []
    for tc in tool_calls:
        if tc.get("tool") != "search_knowledgebase":
            continue
        response = tc.get("response", "")
        if not isinstance(response, str):
            continue

        cur_eid: str | None = None
        cur_src: str | None = None

        for line in response.splitlines():
            eid_m = re.match(r"^\s*Entity ID:\s*(.+)$", line)
            if eid_m:
                # Flush previous result
                if cur_eid is not None and cur_src is not None:
                    results.append(_KBResult(cur_eid, cur_src))
                cur_eid = eid_m.group(1).strip()
                cur_src = None
                continue

            src_m = re.match(r"^\s*Source:\s*(.+)$", line)
            if src_m:
                cur_src = src_m.group(1).strip()

        # Flush last result in this response
        if cur_eid is not None and cur_src is not None:
            results.append(_KBResult(cur_eid, cur_src))

    return results


def _kb_was_called(tool_calls: list[dict[str, Any]]) -> bool:
    return any(tc.get("tool") == "search_knowledgebase" for tc in tool_calls)


def _source_matches(expected: str, result_source: str) -> bool:
    """Case-insensitive substring match: *expected* appears in *result_source*."""
    return expected.lower() in result_source.lower()


def _result_matches_any_expected(
    result_source: str,
    expected_sources: list[str],
) -> bool:
    return any(_source_matches(exp, result_source) for exp in expected_sources)


def internal_kb_grader(
    input: Any,  # noqa: A002
    output: Any,
    expected_output: Any,
    metadata: dict[str, Any] | None = None,
    **kwargs: Any,
) -> list[Evaluation]:
    """Evaluate internal KB tool-call quality for one test case.

    Returns
    -------
    list[Evaluation]
        ``internal_kb_coverage_adj_recall`` and
        ``internal_kb_cleaness_adj_precision``.
    """
    del input, metadata, kwargs  # unused — part of evaluator interface

    # --- Parse expected sources ---
    expected_sources: list[str] = []
    raw = (
        expected_output.get("expected_kb_watchlist_matches")
        if isinstance(expected_output, dict)
        else None
    )
    if isinstance(raw, list):
        expected_sources = [s for s in raw if isinstance(s, str) and s.strip()]

    # --- Extract tool calls & results ---
    tool_calls: list[dict[str, Any]] = []
    if isinstance(output, dict):
        tool_calls = output.get("tool_calls", [])

    called = _kb_was_called(tool_calls)
    kb_results = _extract_kb_results(tool_calls) if called else []
    found_sources: set[str] = {r.source for r in kb_results}

    # Adjusted Recall
    if not called:
        # "0 if not executed" — regardless of expected content
        recall, recall_cmt = 0.0, "search_knowledgebase was never called."
        recall_meta: dict[str, Any] = {"executed": False}
    elif not expected_sources:
        recall, recall_cmt = 1.0, "No KB matches expected."
        recall_meta = {"expected_count": 0, "found_sources": sorted(found_sources)}
    else:
        matched = [s for s in expected_sources if any(_source_matches(s, f) for f in found_sources)]
        missed = [s for s in expected_sources if s not in matched]
        recall = len(matched) / len(expected_sources)
        recall_cmt = (
            f"{'All' if not missed else f'{len(matched)}/{len(expected_sources)}'} "
            f"expected KB sources {'found' if not missed else 'covered'}."
        )
        recall_meta = {
            "expected_sources": expected_sources,
            "matched_sources": matched,
            "missed_sources": missed,
            "found_sources": sorted(found_sources),
        }

    # Adjusted Precision
    if not called:
        # "0 if not executed" — regardless of expected content
        precision, prec_cmt = 0.0, "search_knowledgebase was never called."
        prec_meta: dict[str, Any] = {"executed": False}
    elif not expected_sources and not kb_results:
        precision, prec_cmt = 1.0, "No KB matches expected and search returned empty."
        prec_meta = {"expected_count": 0, "result_count": 0}
    elif not expected_sources:
        # Expected nothing but got results — all FP
        precision, prec_cmt = 0.0, (
            f"No KB matches expected but {len(kb_results)} result(s) returned (all FP)."
        )
        prec_meta = {"expected_count": 0, "result_count": len(kb_results), "fp_count": len(kb_results)}
    elif not kb_results:
        precision, prec_cmt = 0.0, "search_knowledgebase returned no results."
        prec_meta = {"expected_count": len(expected_sources), "result_count": 0}
    else:
        # Classify each result as TP or FP
        tp_entity_ids: set[str] = set()
        direct_tp: list[_KBResult] = []
        pending: list[_KBResult] = []

        for r in kb_results:
            if _result_matches_any_expected(r.source, expected_sources):
                direct_tp.append(r)
                tp_entity_ids.add(r.entity_id)
            else:
                pending.append(r)

        # Entity-ID tolerance: same entity_id as a TP → still TP
        indirect_tp = [r for r in pending if r.entity_id in tp_entity_ids]
        fp = [r for r in pending if r.entity_id not in tp_entity_ids]

        tp_count = len(direct_tp) + len(indirect_tp)
        fp_count = len(fp)
        precision = tp_count / (tp_count + fp_count)

        prec_cmt = (
            f"TP={tp_count} (direct={len(direct_tp)}, "
            f"entity-id-matched={len(indirect_tp)}), FP={fp_count}."
        )
        prec_meta = {
            "tp_count": tp_count,
            "fp_count": fp_count,
            "direct_tp_count": len(direct_tp),
            "indirect_tp_count": len(indirect_tp),
            "fp_results": [
                {"entity_id": r.entity_id, "source": r.source} for r in fp
            ],
        }

    return [
        Evaluation(
            name="internal_kb_coverage_adj_recall",
            value=recall,
            comment=recall_cmt,
            metadata=recall_meta,
        ),
        Evaluation(
            name="internal_kb_cleaness_adj_precision",
            value=precision,
            comment=prec_cmt,
            metadata=prec_meta,
        ),
    ]


__all__ = ["internal_kb_grader"]
