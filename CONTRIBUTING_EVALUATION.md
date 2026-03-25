# Contributing Evaluation Graders

This guide explains how to add new evaluation metrics to the AML agent
evaluation harness in `aml_agent/evaluation/`.

## Architecture Overview

```
aml_agent/evaluation/
├── types.py             # Protocols & data models (Evaluation, EvaluatorFunction, …)
├── experiment.py        # Local experiment runner
├── evaluate.py          # CLI entry point
├── langfuse.py          # Optional Langfuse score upload
└── graders/
    ├── __init__.py      # Re-exports all graders
    ├── llm_judge.py     # LLM-as-judge helpers (run_llm_judge, LLMJudgeConfig)
    ├── internal_kb.py   # ✅ Implemented — KB source coverage
    ├── sql.py           # Stub — SQL quality
    ├── transaction.py   # Stub — Transaction retrieval results
    ├── web_search.py    # Stub — Web-search coverage
    ├── report.py        # Stub — Report quality & completeness
    ├── run.py           # Stub — Run-level aggregates
    └── trace.py         # Stub — Tool-call behavioural checks
```

## Evaluation Levels

**Item-level** (`EvaluatorFunction`) — grades one test case. `output` is the
full artifacts dict containing `tool_calls`, `sql_results`,
`report_markdown`, etc.

**Run-level** (`RunEvaluatorFunction`) — computes aggregate metrics across
all items (e.g. mean recall).

## (a) Code-based Graders

Write a plain function that inspects the artifacts and returns
`Evaluation` objects.

```python
from typing import Any
from aml_agent.evaluation.types import Evaluation

def my_grader(
    input: Any,
    output: Any,
    expected_output: Any,
    metadata: dict[str, Any] | None = None,
    **kwargs: Any,
) -> list[Evaluation]:
    # output["tool_calls"], output["sql_results"], output["report_markdown"], …
    score = ...  # your deterministic logic
    return [Evaluation(name="my_metric", value=score, comment="…")]
```

For run-level aggregation:

```python
from aml_agent.evaluation.types import Evaluation, ItemResult

def my_run_grader(
    *,
    item_results: list[ItemResult],
    **kwargs: Any,
) -> list[Evaluation]:
    values = [r.evaluations["my_metric"].value for r in item_results]
    return [Evaluation(name="my_metric_mean", value=sum(values) / len(values))]
```

## (b) LLM-as-Judge Graders

Use the helpers in `graders/llm_judge.py` to call a Gemini judge and
parse the response.

```python
from typing import Any
from aml_agent.evaluation.types import Evaluation
from aml_agent.evaluation.graders.llm_judge import (
    LLMJudgeConfig,
    run_llm_judge_structured,
    build_judge_error_evaluation,
)

SYSTEM_PROMPT = "You are an evaluator. Return JSON: {\"score\": 0 or 1, \"reason\": \"…\"}"

def my_llm_judge_grader(
    input: Any,
    output: Any,
    expected_output: Any,
    metadata: dict[str, Any] | None = None,
    **kwargs: Any,
) -> list[Evaluation]:
    try:
        result = run_llm_judge_structured(
            metric_name="my_metric",
            system_prompt=SYSTEM_PROMPT,
            user_prompt=f"Output:\n{output.get('report_markdown', '')}",
            config=LLMJudgeConfig(temperature=0.0),
        )
        return [Evaluation(name="my_metric", value=result["score"], comment=result.get("reason"))]
    except Exception as exc:
        return [build_judge_error_evaluation(metric_name="my_metric", error=exc)]
```

`run_llm_judge` returns the raw LLM text; `run_llm_judge_structured`
parses it as JSON. Both log every call to `run_log/llm_as_judge/` for
traceability. `build_judge_error_evaluation` produces a consistent
error `Evaluation` so failed traces are not silently dropped.

## Registering a New Grader

1. **Implement** it in the relevant stub file under `graders/`.
2. **Export** it from `graders/__init__.py`.
3. **Add** it to the `evaluators` (item-level) or `run_evaluators`
   (run-level) list in `evaluate.py`.

## Running Evaluation

```bash
# Local run against pre-computed artifacts
python -m aml_agent.evaluation.evaluate

# Filter to a single test case
python -m aml_agent.evaluation.evaluate --filter TC-002

# Upload scores to Langfuse
python -m aml_agent.evaluation.evaluate --langfuse
```

Langfuse upload requires `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`,
and `LANGFUSE_HOST` environment variables.
