#!/usr/bin/env python3
"""Evaluate the AML investigation agent.

Supports three modes:

**Local** (default) — evaluate pre-computed artifacts offline::

    python -m aml_agent.evaluation.evaluate

**Langfuse** — evaluate pre-computed artifacts via Langfuse with dataset
upload, tracing, and automatic score upload::

    python -m aml_agent.evaluation.evaluate --langfuse

**Live** — run the actual agent end-to-end with full Langfuse tracing
(LLM calls, tool invocations, agent steps) and evaluation::

    python -m aml_agent.evaluation.evaluate --live

All modes read from ``eval_config.yaml`` for paths and settings.
CLI flags override config values.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

from aml_agent.evaluation.eval_config import load_eval_config
from aml_agent.evaluation.experiment import run_local_experiment
from aml_agent.evaluation.graders import (
    internal_kb_grader,
    internal_kb_agent_precision_llm_grader,
    report_aml_risk_level_accuracy_llm_grader,
    report_completeness_grader,
    open_search_urls_reachable_pct_grader,
    open_search_results_relevance_llm_grader,
    tool_completeness_grader,
    sql_quality_grader,
    sql_safety_grader
)

from aml_agent.evaluation.types import ExperimentResult


def _print_local_results(result: ExperimentResult) -> None:
    """Pretty-print local evaluation results to stdout."""
    print(f"\n{'='*60}")
    print(f"Evaluation Results — {len(result.item_results)} test cases")
    print(f"{'='*60}\n")

    for item in result.item_results:
        print(f"  {item.test_case_id}:")
        for ev in item.evaluations:
            val = f"{ev.value:.2f}" if isinstance(ev.value, float) else str(ev.value)
            comment = f"  ({ev.comment})" if ev.comment else ""
            print(f"    {ev.name}: {val}{comment}")

        # Show trace metrics if the output has them
        trace_metrics = (item.output or {}).get("trace_metrics")
        if trace_metrics:
            token = trace_metrics.get("token_usage", {})
            print(
                f"    [trace] llm_calls={trace_metrics.get('llm_call_count', '?')}"
                f"  tool_calls={trace_metrics.get('tool_call_count', '?')}"
                f"  tokens={token.get('total_token_count', '?')}"
                f"  elapsed={trace_metrics.get('elapsed_sec', '?')}s"
            )
        print()

    if result.run_evaluations:
        print("Run-level metrics:")
        for ev in result.run_evaluations:
            print(f"  {ev.name}: {ev.value}")
        print()

    # Summary
    kb_scores = [
        ev.value
        for item in result.item_results
        for ev in item.evaluations
        if ev.name == "internal_kb_correctness"
    ]
    if kb_scores:
        avg = sum(kb_scores) / len(kb_scores)
        passed = sum(1 for s in kb_scores if s == 1.0)
        print(f"internal_kb_correctness  avg={avg:.3f}  ({passed}/{len(kb_scores)} passed)")


def _get_tc_id_from_item(item_result) -> str:
    """Extract test_case_id from a Langfuse ExperimentItemResult."""
    raw = item_result.item
    if hasattr(raw, "metadata") and raw.metadata:
        tc = raw.metadata.get("test_case_id")
        if tc:
            return tc
    if hasattr(raw, "input") and not isinstance(raw.input, dict):
        return "?"
    inp = raw.input if hasattr(raw, "input") else raw.get("input", {})
    if isinstance(inp, dict):
        return inp.get("test_case_id", "?")
    return "?"


def _print_langfuse_results(result) -> None:
    """Print summary from a Langfuse ExperimentResult."""
    print(f"\n{'='*60}")
    print(f"Langfuse Experiment: {result.name}")
    print(f"Items: {len(result.item_results)}")
    print(f"{'='*60}\n")

    for item in result.item_results:
        tc_id = _get_tc_id_from_item(item)
        print(f"  {tc_id}:")
        for ev in item.evaluations:
            val = f"{ev.value:.2f}" if isinstance(ev.value, (int, float)) else str(ev.value)
            comment = f"  ({ev.comment})" if ev.comment else ""
            print(f"    {ev.name}: {val}{comment}")
        print()

    if result.run_evaluations:
        print("Run-level metrics:")
        for ev in result.run_evaluations:
            print(f"  {ev.name}: {ev.value}")
        print()

    # Summary
    kb_scores = [
        ev.value
        for item in result.item_results
        for ev in item.evaluations
        if ev.name == "internal_kb_correctness"
    ]
    if kb_scores:
        avg = sum(kb_scores) / len(kb_scores)
        passed = sum(1 for s in kb_scores if s == 1.0)
        print(f"internal_kb_correctness  avg={avg:.3f}  ({passed}/{len(kb_scores)} passed)")

    if result.dataset_run_url:
        print(f"\nLangfuse run: {result.dataset_run_url}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AML Agent Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="By default runs in local mode.  Pass --langfuse for end-to-end Langfuse experiment.",
    )
    parser.add_argument(
        "--langfuse",
        action="store_true",
        help="Run via Langfuse (dataset upload + experiment + score upload)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run the actual agent (not pre-computed artifacts). Implies --langfuse.",
    )
    parser.add_argument(
        "--csv",
        help="Override test-cases CSV path (default: from eval_config.yaml)",
    )
    parser.add_argument(
        "--artifacts",
        help="Override artifacts directory (default: from eval_config.yaml)",
    )
    parser.add_argument(
        "--filter",
        nargs="*",
        metavar="TC-ID",
        help="Evaluate only specific test-case IDs (e.g. TC-002 TC-005)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        metavar="N",
        help="Max parallel agent runs in --live mode (default: live_max_concurrency from config)",
    )
    parser.add_argument(
        "--config",
        help="Path to eval config YAML (default: aml_agent/evaluation/eval_config.yaml)",
    )
    parser.add_argument(
        "--experiment-name",
        help="Override experiment name (Langfuse mode only)",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Write evaluation results to a JSON file (local mode only)",
    )
    parser.add_argument(
        "--llm-eval-off",
        action="store_true",
        help="Disable LLM-as-a-judge evaluators (skips API calls and associated cost/latency).",
    )
    parser.add_argument(
        "--no-local-save",
        action="store_true",
        help="Disable saving HTML and artifact JSON to run_log/agent_output/ (local mode only).",
    )
    args = parser.parse_args()

    # Load .env from the project root so GOOGLE_API_KEY and other secrets are
    # available in all modes (local/langfuse/live) before any grader runs.
    try:
        from dotenv import load_dotenv  # type: ignore[import-untyped]
        _project_root = Path(__file__).resolve().parent.parent.parent
        for _env_file in (_project_root / ".env", Path(".env")):
            if _env_file.exists():
                load_dotenv(_env_file, override=False)
                break
    except ImportError:
        pass

    cfg = load_eval_config(args.config)

    if args.live:
        _run_langfuse_mode(args, cfg, live=True)
    elif args.langfuse:
        _run_langfuse_mode(args, cfg)
    else:
        _run_local_mode(args, cfg)


def _build_evaluator_list(args) -> list:
    """Return the list of evaluators, excluding LLM-judge ones if --llm-eval-off."""
    evaluators = [
        internal_kb_grader,
        report_completeness_grader,
        open_search_urls_reachable_pct_grader,
        tool_completeness_grader,
        sql_safety_grader
    ]
    if not getattr(args, "llm_eval_off", False):
        evaluators.append(sql_quality_grader)
        evaluators.append(internal_kb_agent_precision_llm_grader)
        evaluators.append(report_aml_risk_level_accuracy_llm_grader)
        evaluators.append(open_search_results_relevance_llm_grader)
    return evaluators


def _save_run_outputs(items: list[tuple[str, dict]], run_dt: str) -> None:
    """Save HTML and artifact JSON to run_log/agent_output/<run_dt>/.

    Parameters
    ----------
    items : list of (tc_id, artifacts) tuples
    run_dt : str
        Timestamp string used as the sub-directory name.
    """
    from aml_agent.report_html import render_html

    out_dir = Path("run_log/agent_output") / run_dt
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for tc_id, artifacts in items:
        # Save artifacts JSON
        json_path = out_dir / f"{tc_id}.artifacts.json"
        json_path.write_text(json.dumps(artifacts, indent=2, ensure_ascii=False), encoding="utf-8")

        # Generate and save HTML (requires report_markdown in artifacts)
        report_md = artifacts.get("report_markdown", "")
        if report_md:
            subject = (artifacts.get("report") or {}).get("subject", tc_id)
            html_str = render_html(
                report_md,
                subject=subject,
                sql_results=artifacts.get("sql_results", []),
                version=artifacts.get("version", "unknown"),
            )
            html_path = out_dir / f"{tc_id}.html"
            html_path.write_text(html_str, encoding="utf-8")
        saved += 1

    print(f"\nOutput saved to {out_dir}/ ({saved} item(s))")


def _run_local_mode(args, cfg) -> None:
    """Run evaluation locally with progress bar."""
    csv_path = Path(args.csv) if args.csv else cfg.resolve_csv_path()
    artifacts_dir = Path(args.artifacts) if args.artifacts else cfg.resolve_artifacts_dir()

    if not csv_path.exists():
        print(f"Error: CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)
    if not artifacts_dir.exists():
        print(f"Error: artifacts dir not found: {artifacts_dir}", file=sys.stderr)
        sys.exit(1)

    run_dt = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    result = run_local_experiment(
        csv_path=csv_path,
        artifacts_dir=artifacts_dir,
        evaluators=_build_evaluator_list(args),
        filter_ids=args.filter,
    )

    _print_local_results(result)

    if not getattr(args, "no_local_save", False):
        pairs = [(item.test_case_id, item.output or {}) for item in result.item_results]
        _save_run_outputs(pairs, run_dt)

    if args.output:
        out_data = {
            "item_results": [
                {
                    "test_case_id": item.test_case_id,
                    "evaluations": [
                        {"name": e.name, "value": e.value, "comment": e.comment, "metadata": e.metadata}
                        for e in item.evaluations
                    ],
                    "trace_metrics": (item.output or {}).get("trace_metrics"),
                }
                for item in result.item_results
            ],
            "run_evaluations": [
                {"name": e.name, "value": e.value, "comment": e.comment, "metadata": e.metadata}
                for e in result.run_evaluations
            ],
        }
        out_path = Path(args.output)
        out_path.write_text(json.dumps(out_data, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nResults written to {out_path}")


def _run_langfuse_mode(args, cfg, *, live: bool = False) -> None:
    """Run evaluation via Langfuse end-to-end."""
    from aml_agent.evaluation.langfuse import run_langfuse_experiment

    run_dt = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    task = None
    if live:
        from aml_agent.evaluation.task import AmlAgentTask

        task = AmlAgentTask(
            temperature=cfg.agent.temperature,
            timeout_sec=cfg.agent.timeout_sec,
            enable_tracing=True,
        )

    try:
        result = run_langfuse_experiment(
            evaluators=_build_evaluator_list(args),
            cfg=cfg,
            experiment_name=args.experiment_name,
            filter_ids=args.filter,
            task=task,
            live_concurrency=getattr(args, "concurrency", None),
        )
    finally:
        # Always release tool connections (Weaviate, etc.) after the run.
        if task is not None:
            task.close_sync()

    _print_langfuse_results(result)

    if not getattr(args, "no_local_save", False):
        pairs = []
        for item_result in result.item_results:
            tc_id = _get_tc_id_from_item(item_result)
            artifacts = item_result.output or {}
            if isinstance(artifacts, dict) and tc_id and tc_id != "?":
                pairs.append((tc_id, artifacts))
        if pairs:
            _save_run_outputs(pairs, run_dt)


if __name__ == "__main__":
    main()
