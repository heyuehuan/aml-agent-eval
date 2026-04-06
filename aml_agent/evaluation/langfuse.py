"""Langfuse end-to-end evaluation pipeline.

Provides:
- Dataset upload (idempotent — skips if dataset already exists with items).
- ``run_langfuse_experiment()`` — runs evaluators via ``Langfuse.run_experiment``
  with built-in progress bar, parallelism, and automatic score upload.

Required environment variables: LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY,
LANGFUSE_HOST — loaded automatically from ``.env``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from .eval_config import EvalConfig, load_eval_config
from .experiment import load_artifacts, load_test_cases
from .progress import track_with_progress
from .types import Evaluation as LocalEvaluation, EvaluatorFunction, RunEvaluatorFunction

logger = logging.getLogger(__name__)


def _get_git_version_info() -> dict[str, str]:
    """Return git version metadata for experiment tracking.

    Returns a dict with:
    - ``model_ver_id``: ``<branch>/<sha7>`` (or just ``<sha7>`` on detached HEAD)
    - ``model_ver_ts``: commit datetime in EST as ``YYYYMMMDD-HHMMSS``
      (e.g. ``2026Jan05-143022``)
    """
    import datetime

    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short=7", "HEAD"],
            stderr=subprocess.DEVNULL, text=True, timeout=3,
        ).strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL, text=True, timeout=3,
        ).strip()
        commit_ts_unix = subprocess.check_output(
            ["git", "log", "-1", "--format=%ct"],
            stderr=subprocess.DEVNULL, text=True, timeout=3,
        ).strip()

        ver_id = f"{branch}/{sha}" if branch and branch != "HEAD" else sha

        est = datetime.timezone(datetime.timedelta(hours=-5), name="EST")
        dt = datetime.datetime.fromtimestamp(int(commit_ts_unix), tz=est)
        ver_ts = dt.strftime("%Y%b%d-%H%M%S")  # e.g. 2026Jan05-143022

        return {"model_ver_id": ver_id, "model_ver_ts": ver_ts}
    except Exception:
        return {"model_ver_id": "unknown", "model_ver_ts": "unknown"}


def _wrap_evaluator(evaluator: EvaluatorFunction):
    """Wrap an evaluator so it returns ``langfuse.Evaluation`` objects."""
    from langfuse import Evaluation as LangfuseEvaluation  # type: ignore[import-untyped]

    def wrapped(*, input: Any = None, output: Any = None,
                expected_output: Any = None, metadata: Any = None,
                **kwargs: Any):
        result = evaluator(
            input=input, output=output,
            expected_output=expected_output, metadata=metadata,
            **kwargs,
        )

        def _convert(ev):
            if isinstance(ev, LocalEvaluation):
                return LangfuseEvaluation(
                    name=ev.name,
                    value=ev.value,
                    comment=ev.comment,
                    metadata=ev.metadata,
                )
            return ev  # already a Langfuse Evaluation

        if isinstance(result, list):
            return [_convert(e) for e in result]
        return _convert(result)

    wrapped.__name__ = getattr(evaluator, "__name__", "evaluator")
    return wrapped


def _load_dotenv() -> None:
    """Load ``.env`` from the project root if python-dotenv is available."""
    try:
        from dotenv import load_dotenv  # type: ignore[import-untyped]

        here = Path(__file__).resolve()
        for parent in [here, *here.parents]:
            env_file = parent / ".env"
            if env_file.exists():
                load_dotenv(env_file, override=False)
                break
    except ImportError:
        pass


_langfuse_client = None


def get_langfuse_client():
    """Return a shared ``Langfuse`` client, initialised from env vars."""
    global _langfuse_client
    if _langfuse_client is not None:
        return _langfuse_client

    _load_dotenv()

    try:
        from langfuse import Langfuse  # type: ignore[import-untyped]
    except ImportError:
        print(
            "Error: langfuse package not installed.  "
            "Install it with:  pip install langfuse",
            file=sys.stderr,
        )
        sys.exit(1)

    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY")
    host = os.environ.get("LANGFUSE_HOST")

    missing = [name for name, val in [
        ("LANGFUSE_PUBLIC_KEY", public_key),
        ("LANGFUSE_SECRET_KEY", secret_key),
        ("LANGFUSE_HOST", host),
    ] if not val]
    if missing:
        print(f"Error: missing Langfuse env vars: {', '.join(missing)}", file=sys.stderr)
        sys.exit(1)

    _langfuse_client = Langfuse(public_key=public_key, secret_key=secret_key, host=host)
    return _langfuse_client


def _build_item_id(dataset_name: str, input_payload: Any, expected_output: Any) -> str:
    """Deterministic SHA-256 item ID for deduplication."""
    canonical = json.dumps(
        {"input": input_payload, "expected_output": expected_output},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"{dataset_name}:{digest}"


def ensure_dataset(
    cfg: EvalConfig,
    *,
    force: bool = False,
) -> None:
    """Upload test cases & artifacts as a Langfuse dataset.

    Skips upload if the dataset already exists and has items (unless
    *force* is ``True``).
    """
    client = get_langfuse_client()
    dataset_name = cfg.dataset.name

    # Check if dataset already exists
    if not force:
        try:
            ds = client.get_dataset(dataset_name)
            if ds.items:
                logger.info(
                    "Dataset '%s' already exists with %d items — skipping upload.",
                    dataset_name,
                    len(ds.items),
                )
                print(f"Dataset '{dataset_name}' already exists ({len(ds.items)} items) — skipping upload.")
                return
        except Exception:
            pass  # Dataset doesn't exist yet — proceed with creation

    # Create or update dataset
    client.create_dataset(
        name=dataset_name,
        description=f"AML agent test cases for {cfg.agent.name}",
        metadata={"agent": cfg.agent.name},
    )

    # Load test cases and pair with artifacts
    csv_path = cfg.resolve_csv_path()
    test_cases = load_test_cases(csv_path)

    uploaded = 0
    for tc in track_with_progress(test_cases, description="Uploading dataset"):
        tc_id = tc["test_case_id"]

        # input = the test case prompt + full row; expected_output = ground truth
        input_payload = {
            "test_case_id": tc_id,
            "test_case_info_input": tc.get("test_case_info_input", ""),
            "test_case_details": tc.get("test_case_details", ""),
        }
        expected_output = tc  # full ground-truth row

        item_id = _build_item_id(dataset_name, input_payload, expected_output)

        client.create_dataset_item(
            dataset_name=dataset_name,
            id=item_id,
            input=input_payload,
            expected_output=expected_output,
            metadata={"test_case_id": tc_id},
        )
        uploaded += 1

    client.flush()
    print(f"Uploaded {uploaded} items to dataset '{dataset_name}'.")


def _wrap_live_task(task, total: int):
    """Wrap a live task with a Rich progress bar."""
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )
    task_id = progress.add_task("Running live agent...", total=total)

    # Wire intermediate status logging into the task if supported.
    if hasattr(task, "status_fn"):
        task.status_fn = progress.console.log

    async def tracked(*, item, **kwargs):
        result = await task(item=item, **kwargs)
        progress.advance(task_id)
        return result

    return tracked, progress


def _fetch_and_filter_items(client, cfg: EvalConfig, filter_ids: list[str] | None = None):
    """Fetch DatasetItemClient objects from Langfuse, optionally filtered."""
    dataset = client.get_dataset(cfg.dataset.name)
    items = dataset.items
    if filter_ids:
        filter_set = set(filter_ids)
        items = [
            it for it in items
            if (it.metadata or {}).get("test_case_id") in filter_set
        ]
    return items


def run_langfuse_experiment(
    *,
    evaluators: list[EvaluatorFunction] | None = None,
    run_evaluators: list[RunEvaluatorFunction] | None = None,
    cfg: EvalConfig | None = None,
    experiment_name: str | None = None,
    filter_ids: list[str] | None = None,
    task=None,
    live_concurrency: int | None = None,
):
    """Run evaluation end-to-end via Langfuse ``run_experiment``.

    Supports pre-computed artifact mode (default) or live agent mode
    (pass a ``task`` callable like ``AmlAgentTask()``).
    """
    cfg = cfg or load_eval_config()
    evaluators = evaluators or []
    run_evaluators = run_evaluators or []
    client = get_langfuse_client()

    is_live = task is not None

    ensure_dataset(cfg)
    items = _fetch_and_filter_items(client, cfg, filter_ids)

    concurrency = (
        (live_concurrency or cfg.experiment.live_max_concurrency)
        if is_live
        else cfg.experiment.max_concurrency
    )

    progress = None
    if is_live:
        task, progress = _wrap_live_task(task, len(items))
    else:
        artifacts_dir = cfg.resolve_artifacts_dir()
        artifacts_cache: dict[str, dict[str, Any]] = {}
        for it in items:
            tc_id = (it.metadata or {}).get("test_case_id") or it.input.get("test_case_id")
            if tc_id and tc_id not in artifacts_cache:
                artifacts = load_artifacts(artifacts_dir, tc_id)
                if artifacts is not None:
                    artifacts_cache[tc_id] = artifacts
                else:
                    logger.warning("No artifacts for %s — will skip", tc_id)

        items = [
            it for it in items
            if (
                (it.metadata or {}).get("test_case_id")
                or it.input.get("test_case_id")
            ) in artifacts_cache
        ]

        def task(*, item, **kwargs: Any) -> dict[str, Any]:
            tc_id = (
                getattr(item, "metadata", None) or {}
            ).get("test_case_id") or item.input.get("test_case_id")
            return artifacts_cache[tc_id]

    name = experiment_name or cfg.agent.name
    git_info = _get_git_version_info()
    print(
        f"Experiment: {name!r}  |  "
        f"model_ver_id={git_info['model_ver_id']}  "
        f"model_ver_ts={git_info['model_ver_ts']}"
    )
    if progress is not None:
        progress.start()
    try:
        result = client.run_experiment(
            name=name,
            data=items,
            task=task,
            evaluators=[_wrap_evaluator(e) for e in evaluators],
            run_evaluators=run_evaluators,
            max_concurrency=concurrency,
            metadata={
                "agent": cfg.agent.name,
                **git_info,
            },
        )
    finally:
        if progress is not None:
            progress.stop()

    # Report trace-level scores for live runs
    if is_live:
        _report_experiment_trace_scores(result)

    client.flush()
    return result


def _report_experiment_trace_scores(result) -> None:
    """Extract trace metrics from live-run outputs and upload as Langfuse scores.

    For each experiment item that has ``trace_metrics`` in its output (i.e.
    items processed by ``AmlAgentTask`` with ``CallbackTracer``), report the
    metrics to Langfuse as trace-level scores.
    """
    from aml_agent.evaluation.tracing import report_trace_scores

    for item_result in result.item_results:
        output = getattr(item_result, "output", None)
        if not output or not isinstance(output, dict):
            continue

        trace_metrics = output.get("trace_metrics")
        if not trace_metrics:
            continue

        # Get the trace_id from the item result
        trace_id = getattr(item_result, "trace_id", None)
        if not trace_id:
            continue

        try:
            report_trace_scores(trace_id, trace_metrics)
        except Exception:
            logger.warning(
                "Failed to report trace scores for trace %s", trace_id
            )


__all__ = [
    "get_langfuse_client",
    "ensure_dataset",
    "run_langfuse_experiment",
]
