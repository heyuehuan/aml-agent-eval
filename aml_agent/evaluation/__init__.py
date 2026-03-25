"""Evaluation harness for the AML investigation agent.

This package provides:

- **Local evaluation** — ``run_local_experiment()`` evaluates pre-computed
  artifacts offline with a progress bar.  No Langfuse needed.
- **Langfuse evaluation** — ``run_langfuse_experiment()`` runs end-to-end
  via Langfuse with dataset upload, parallel execution, and automatic
  score upload.
- Domain-specific **graders** organised by evaluation level.

Grader levels
-------------
- **Item-level** (``EvaluatorFunction``): grades one test case's artifacts.
- **Run-level** (``RunEvaluatorFunction``): aggregates across all items.
"""

from .eval_config import EvalConfig, load_eval_config
from .experiment import run_local_experiment
from .types import (
    Evaluation,
    EvaluatorFunction,
    ExperimentResult,
    ItemResult,
    RunEvaluatorFunction,
)

__all__ = [
    "EvalConfig",
    "load_eval_config",
    "run_local_experiment",
    "Evaluation",
    "EvaluatorFunction",
    "ExperimentResult",
    "ItemResult",
    "RunEvaluatorFunction",
]
