"""Langfuse tracing via OpenTelemetry + OpenInference, and trace score reporting.

Calling ``init_tracing()`` once at startup is sufficient — after that, all
Google ADK ``Runner.run_async``, ``BaseAgent.run_async``, LLM calls, and
tool calls are automatically traced and exported to Langfuse via OTLP.

``report_trace_scores()`` uploads trace-level metrics (token usage, latency,
tool call count) to Langfuse as scores, making them visible in the UI
alongside the auto-captured spans.

No ``@observe`` decorators or manual spans are needed.
"""

from __future__ import annotations

import base64
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_tracing_initialised = False


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


def init_tracing(service_name: str = "aml-agent") -> bool:
    """Initialise Langfuse tracing via OTLP + GoogleADKInstrumentor.

    Safe to call multiple times (idempotent). Returns True on success.
    """
    global _tracing_initialised
    if _tracing_initialised:
        logger.debug("Tracing already initialised")
        return True

    _load_dotenv()

    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY", "")
    host = os.environ.get("LANGFUSE_HOST", "")

    if not all([public_key, secret_key, host]):
        logger.warning(
            "Langfuse env vars not set (LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST). "
            "Tracing disabled."
        )
        return False

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        auth_string = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
        otel_endpoint = f"{host.rstrip('/')}/api/public/otel"

        resource = Resource.create({"service.name": service_name})
        provider = TracerProvider(resource=resource)

        exporter = OTLPSpanExporter(
            endpoint=f"{otel_endpoint}/v1/traces",
            headers={"Authorization": f"Basic {auth_string}"},
        )
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)

        # Auto-instrument Google ADK
        from openinference.instrumentation.google_adk import GoogleADKInstrumentor

        GoogleADKInstrumentor().instrument(tracer_provider=provider)

        _tracing_initialised = True
        logger.info("Langfuse tracing initialised (endpoint: %s)", otel_endpoint)
        return True

    except ImportError as exc:
        logger.warning("Tracing dependencies not installed: %s", exc)
        return False
    except Exception as exc:
        logger.warning("Failed to initialise tracing: %s", exc)
        return False


def is_tracing_enabled() -> bool:
    """Return whether tracing has been initialised."""
    return _tracing_initialised


def report_trace_scores(
    trace_id: str,
    trace_metrics: dict[str, Any],
    *,
    client: Any = None,
) -> None:
    """Upload trace-level metrics to Langfuse as numeric scores."""
    if client is None:
        try:
            from aml_agent.evaluation.langfuse import get_langfuse_client
            client = get_langfuse_client()
        except Exception:
            logger.warning("Cannot report scores — Langfuse client unavailable")
            return

    token_usage = trace_metrics.get("token_usage", {})

    scores = [
        ("llm_call_count", trace_metrics.get("llm_call_count", 0)),
        ("tool_call_count", trace_metrics.get("tool_call_count", 0)),
        ("total_token_count", token_usage.get("total_token_count", 0)),
        ("prompt_token_count", token_usage.get("prompt_token_count", 0)),
        ("candidates_token_count", token_usage.get("candidates_token_count", 0)),
    ]

    elapsed = trace_metrics.get("elapsed_sec")
    if elapsed is not None:
        scores.append(("elapsed_sec", elapsed))

    for name, value in scores:
        try:
            client.create_score(
                trace_id=trace_id,
                name=name,
                value=value,
                data_type="NUMERIC",
            )
        except Exception:
            logger.warning("Failed to report score %s for trace %s", name, trace_id)

    try:
        client.flush()
    except Exception:
        pass
