"""Graders for AML investigation agent evaluation.

This subpackage organises evaluators by level:

- **Item-level** (``EvaluatorFunction``): grade one test case.
- **Run-level** (``RunEvaluatorFunction``): aggregate across all items.

See ``CONTRIBUTING_EVALUATION.md`` at the repo root for guidance on adding new
graders.
"""

from .internal_kb import internal_kb_grader
from .internal_kb_llm import internal_kb_agent_precision_llm_grader
from .llm_judge import LLMJudgeConfig, run_llm_judge, run_llm_judge_structured
from .report import report_aml_risk_level_accuracy_llm_grader, report_completeness_grader
from .transaction import (
    sql_result_score_recall_grader,
    sql_result_score_precision_grader,
    transaction_aggregation_score_llm_grader,
)
from .web_search import (
    open_search_urls_reachable_pct_grader,
    open_search_results_relevance_llm_grader,
)

__all__ = [
    "internal_kb_grader",
    "internal_kb_agent_precision_llm_grader",
    "LLMJudgeConfig",
    "run_llm_judge",
    "run_llm_judge_structured",
    "report_aml_risk_level_accuracy_llm_grader",
    "report_completeness_grader",
    "sql_result_score_recall_grader",
    "sql_result_score_precision_grader",
    "transaction_aggregation_score_llm_grader",
    "open_search_urls_reachable_pct_grader",
    "open_search_results_relevance_llm_grader",
]
