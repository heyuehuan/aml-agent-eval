"""Reusable LLM-as-a-judge utility for evaluation graders.

Provides a simple, synchronous ``run_llm_judge`` function that calls a
Gemini model to judge agent outputs against a rubric.  Results are logged
to a JSONL file under ``run_log/llm_as_judge/`` for traceability.

Environment variables
---------------------
``DEFAULT_EVALUATOR_MODEL``
    Model name for the judge LLM (from ``.env``).
    Defaults to ``gemini-2.5-flash-lite``.
``GOOGLE_API_KEY`` / ``GEMINI_API_KEY``
    API key for the Gemini API (same as the agent).
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types
from opentelemetry import context as otel_context

from aml_agent.evaluation.types import Evaluation

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

DEFAULT_JUDGE_MODEL = "gemini-2.5-flash-lite"


def _get_judge_model() -> str:
    return os.getenv("DEFAULT_EVALUATOR_MODEL", DEFAULT_JUDGE_MODEL)


def _get_api_key() -> str:
    return os.getenv("GOOGLE_API_KEY", os.getenv("GEMINI_API_KEY", ""))


@dataclass(frozen=True)
class LLMJudgeConfig:
    """Configuration for an LLM judge call."""

    model: str | None = None
    temperature: float = 0.0
    max_output_tokens: int = 2048


_LOG_DIR = _PROJECT_ROOT / "run_log" / "llm_as_judge"


def _log_judge_call(
    *,
    metric_name: str,
    system_prompt: str,
    user_prompt: str,
    llm_output: str,
    model: str,
    elapsed_sec: float,
    run_dir: Path | None = None,
) -> None:
    """Append one JSONL record for a judge call."""
    log_dir = run_dir or _get_or_create_run_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "llm_judge_logs.jsonl"

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metric_name": metric_name,
        "model": model,
        "elapsed_sec": round(elapsed_sec, 3),
        "llm_output": llm_output,
        "user_prompt": user_prompt,
        "system_prompt": system_prompt,
    }
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# Lazily-created run directory (one per process lifetime)
_run_dir_cache: Path | None = None


def _get_or_create_run_dir() -> Path:
    global _run_dir_cache  # noqa: PLW0603
    if _run_dir_cache is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        _run_dir_cache = _LOG_DIR / f"run_{stamp}"
    return _run_dir_cache


def run_llm_judge(
    *,
    metric_name: str,
    system_prompt: str,
    user_prompt: str,
    config: LLMJudgeConfig | None = None,
) -> str:
    """Call an LLM to judge content and return the raw text response.

    Parameters
    ----------
    metric_name : str
        Name of the metric being evaluated (used for logging).
    system_prompt : str
        System-level instructions for the judge.
    user_prompt : str
        User prompt containing the content to judge.
    config : LLMJudgeConfig | None
        Optional config overrides.

    Returns
    -------
    str
        Raw text response from the judge LLM.

    Raises
    ------
    RuntimeError
        If the LLM call fails after retries.
    """
    cfg = config or LLMJudgeConfig()
    model = cfg.model or _get_judge_model()
    api_key = _get_api_key()

    client = genai.Client(api_key=api_key)

    # Detach the active OTel context so the judge LLM call is NOT captured
    # as a child span inside the agent's Langfuse trace.  The agent's span
    # may still be open in the async context when evaluators run, and
    # GoogleADKInstrumentor instruments google.genai calls globally — without
    # this detach, the judge's latency would be attributed to the agent run.
    _otel_token = otel_context.attach(otel_context.Context())
    t0 = time.monotonic()
    try:
        response = client.models.generate_content(
            model=model,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=cfg.temperature,
                max_output_tokens=cfg.max_output_tokens,
                response_mime_type="application/json",
            ),
        )
    finally:
        elapsed = time.monotonic() - t0
        otel_context.detach(_otel_token)

    raw_text = response.text or ""

    _log_judge_call(
        metric_name=metric_name,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        llm_output=raw_text,
        model=model,
        elapsed_sec=elapsed,
    )
    return raw_text


def run_llm_judge_structured(
    *,
    metric_name: str,
    system_prompt: str,
    user_prompt: str,
    config: LLMJudgeConfig | None = None,
) -> dict[str, Any]:
    """Call LLM judge and parse the JSON response.

    Returns
    -------
    dict[str, Any]
        Parsed JSON response from the judge.

    Raises
    ------
    ValueError
        If the response cannot be parsed as JSON.
    """
    raw = run_llm_judge(
        metric_name=metric_name,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        config=config,
    )
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM judge returned invalid JSON: {raw[:200]}") from exc


def build_judge_error_evaluation(
    *,
    metric_name: str,
    error: Exception,
) -> Evaluation:
    """Build an error evaluation when the judge call fails."""
    return Evaluation(
        name=metric_name,
        value=0.0,
        comment=f"LLM judge error: {error}",
        metadata={"error_type": type(error).__name__, "error": str(error)},
    )


__all__ = [
    "LLMJudgeConfig",
    "run_llm_judge",
    "run_llm_judge_structured",
    "build_judge_error_evaluation",
]
