"""Local experiment runner for offline evaluation.

Evaluates pre-computed agent artifacts against test-case ground truth.
No Langfuse dependency — for Langfuse-managed experiments, see ``langfuse.py``.

Two passes:
1. **Item-level**: each evaluator grades one (input, artifacts, expected) triple.
2. **Run-level**: each evaluator aggregates scores across all items.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

from .progress import track_with_progress
from .types import (
    Evaluation,
    EvaluatorFunction,
    ExperimentResult,
    ItemResult,
    RunEvaluatorFunction,
)

logger = logging.getLogger(__name__)

# JSON-encoded CSV fields that must be parsed
_JSON_FIELDS = {
    "expected_kb_watchlist_matches",
    "expected_open_search_results",
    "expected_transaction_matches",
    "expected_transaction_matches_details",
    "transaction_variation_details",
}


def _parse_json_field(value: str) -> Any:
    """Parse a JSON-encoded CSV field, returning ``None`` on failure."""
    if not value:
        return None
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return value


def load_test_cases(csv_path: Path) -> list[dict[str, Any]]:
    """Load test cases from a CSV file and parse JSON-encoded fields."""
    rows: list[dict[str, Any]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for key in _JSON_FIELDS:
                if key in row:
                    row[key] = _parse_json_field(row[key])
            rows.append(row)
    return rows


def load_artifacts(artifacts_dir: Path, test_case_id: str) -> dict[str, Any] | None:
    """Load an artifacts JSON sidecar for a test case."""
    path = artifacts_dir / f"Test_{test_case_id}.artifacts.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _run_evaluators(
    evaluators: list[EvaluatorFunction],
    tc: dict[str, Any],
    artifacts: dict[str, Any],
) -> list[Evaluation]:
    """Run all item-level evaluators on a single test case."""
    evals: list[Evaluation] = []
    for evaluator in evaluators:
        try:
            result = evaluator(input=tc, output=artifacts, expected_output=tc)
            if isinstance(result, Evaluation):
                result = [result]
            evals.extend(result)
        except Exception:
            logger.exception("Evaluator failed on %s", tc.get("test_case_id"))
    return evals


def run_local_experiment(
    *,
    csv_path: str | Path,
    artifacts_dir: str | Path,
    evaluators: list[EvaluatorFunction] | None = None,
    run_evaluators: list[RunEvaluatorFunction] | None = None,
    filter_ids: list[str] | None = None,
) -> ExperimentResult:
    """Run evaluation locally over pre-computed artifacts with progress bar.

    Parameters
    ----------
    csv_path : str | Path
        Path to the test-cases CSV file.
    artifacts_dir : str | Path
        Directory containing ``Test_<TC-ID>.artifacts.json`` files.
    evaluators : list[EvaluatorFunction] | None
        Item-level evaluator functions.
    run_evaluators : list[RunEvaluatorFunction] | None
        Run-level evaluator functions.
    filter_ids : list[str] | None
        If provided, only evaluate these test-case IDs.

    Returns
    -------
    ExperimentResult
    """
    csv_path = Path(csv_path)
    artifacts_dir = Path(artifacts_dir)
    evaluators = evaluators or []
    run_evaluators = run_evaluators or []

    test_cases = load_test_cases(csv_path)
    if filter_ids:
        test_cases = [tc for tc in test_cases if tc["test_case_id"] in filter_ids]

    result = ExperimentResult()

    # --- Item-level pass with progress bar ---
    for tc in track_with_progress(test_cases, description="Evaluating test cases"):
        tc_id = tc["test_case_id"]
        artifacts = load_artifacts(artifacts_dir, tc_id)
        if artifacts is None:
            logger.warning("No artifacts found for %s — skipping", tc_id)
            continue

        item = ItemResult(
            test_case_id=tc_id,
            input=tc,
            output=artifacts,
            expected_output=tc,
            evaluations=_run_evaluators(evaluators, tc, artifacts),
        )
        result.item_results.append(item)

    # --- Run-level pass ---
    for run_eval in run_evaluators:
        try:
            evals = run_eval(item_results=result.item_results)
            result.run_evaluations.extend(evals)
        except Exception:
            logger.exception("Run evaluator failed")

    return result


__all__ = ["run_local_experiment", "load_test_cases", "load_artifacts"]
