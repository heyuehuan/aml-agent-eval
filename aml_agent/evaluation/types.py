"""Type definitions for the AML evaluation harness.

This module defines protocols and data containers used across the evaluation
framework.  It mirrors the role of
``aieng.agent_evals.evaluation.types`` in the reference implementation but
is **self-contained** — no Langfuse dependency is required for local evaluation.

Evaluation levels
-----------------
Item-level (``EvaluatorFunction``)
    Grades a single test case.  Receives the full artifacts dict (which
    already includes the tool-call audit trail) plus the expected-output dict.
    Both "output quality" checks and "tool-call correctness" checks are
    item-level — there is no separate trace level because our artifacts
    already embed the full trace.
Run-level (``RunEvaluatorFunction``)
    Computes aggregate metrics over all item results (e.g. mean accuracy).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


# ---------------------------------------------------------------------------
# Evaluation — the atomic grading unit
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Evaluation:
    """A single named evaluation score.

    This is the evaluation currency: every grader returns one or more of these.

    Parameters
    ----------
    name : str
        Metric name (e.g. ``"internal_kb_correctness"``).
    value : float | int | bool
        Numeric score.  Booleans are accepted and serialised as 1.0 / 0.0.
    comment : str | None
        Human-readable explanation (optional).
    metadata : dict[str, Any] | None
        Structured detail for debugging / dashboards (optional).
    """

    name: str
    value: float | int | bool
    comment: str | None = None
    metadata: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Evaluator function protocols
# ---------------------------------------------------------------------------

class EvaluatorFunction(Protocol):
    """Item-level evaluator protocol.

    Matches the Langfuse ``EvaluatorFunction`` signature so the same grader
    can be used both locally and in a Langfuse experiment.

    The ``output`` parameter is the full artifacts dict, which contains
    ``tool_calls``, ``sql_results``, ``report_markdown``, ``report``, etc.
    Graders that need to inspect tool-call behaviour simply read from
    ``output["tool_calls"]`` — no separate protocol is needed.

    Parameters
    ----------
    input : Any
        Test-case input payload (the ``test_case_info_input`` or full row dict).
    output : Any
        Agent artifacts dict (``tool_calls``, ``report_markdown``, etc.).
    expected_output : Any
        Ground-truth dict parsed from the test-case CSV row.
    metadata : dict[str, Any] | None
        Optional test-case metadata.
    **kwargs : Any
        Reserved for future use.
    """

    def __call__(
        self,
        input: Any,  # noqa: A002
        output: Any,
        expected_output: Any,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Evaluation | list[Evaluation]: ...


class RunEvaluatorFunction(Protocol):
    """Run-level evaluator protocol.

    Receives all item results from a completed experiment and returns
    aggregate metrics.
    """

    def __call__(
        self,
        *,
        item_results: list[ItemResult],
        **kwargs: Any,
    ) -> list[Evaluation]: ...


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class ItemResult:
    """Result of evaluating one test case.

    Parameters
    ----------
    test_case_id : str
        Identifier from the CSV (e.g. ``"TC-002"``).
    input : dict[str, Any]
        Test-case input fields.
    output : dict[str, Any]
        Full artifacts dict from the agent run.
    expected_output : dict[str, Any]
        Ground-truth fields from the test-case CSV.
    evaluations : list[Evaluation]
        All evaluation scores for this test case.
    """

    test_case_id: str
    input: dict[str, Any]
    output: dict[str, Any]
    expected_output: dict[str, Any]
    evaluations: list[Evaluation] = field(default_factory=list)


@dataclass
class ExperimentResult:
    """Aggregate result for a full evaluation run.

    Parameters
    ----------
    item_results : list[ItemResult]
        Per-test-case evaluation results.
    run_evaluations : list[Evaluation]
        Run-level aggregate metrics.
    """

    item_results: list[ItemResult] = field(default_factory=list)
    run_evaluations: list[Evaluation] = field(default_factory=list)


__all__ = [
    "Evaluation",
    "EvaluatorFunction",
    "RunEvaluatorFunction",
    "ItemResult",
    "ExperimentResult",
]
