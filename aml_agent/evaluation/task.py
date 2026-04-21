"""Langfuse-compatible task that runs the AML investigation agent.

This module provides ``AmlAgentTask`` — a callable that implements the
Langfuse ``TaskFunction`` protocol (``__call__(*, item, **kwargs)``).

When used with ``run_experiment``, the agent runs end-to-end for each
dataset item.  Combined with ``init_tracing()``, every LLM call, tool
invocation, and agent step is automatically traced in Langfuse.

Concurrency safety
------------------
``AmlAgentTask`` is designed to be shared across concurrent calls.  Each
``__call__`` creates its own ``CallbackTracer`` and installs it into an
``asyncio.ContextVar`` so that the shared agent callbacks always dispatch to
the tracer that belongs to the *currently executing* call, not a stale one.
"""

from __future__ import annotations

import getpass
import json
import logging
import re
import uuid
import warnings
from contextvars import ContextVar
from typing import Any

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from aml_agent.agent import create_aml_agent
from aml_agent.config import Configs
from aml_agent.runner import _parse_report_sections
from aml_agent.tracing import CallbackTracer, parse_md_table

logger = logging.getLogger(__name__)

# Suppress noisy warnings from upstream libraries
warnings.filterwarnings("ignore", message="Inheritance class AiohttpClientSession", category=DeprecationWarning)
logging.getLogger("google_genai.types").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("google.adk").setLevel(logging.WARNING)
logging.getLogger("google_adk").setLevel(logging.WARNING)
logging.getLogger("aml_agent.tools.web_search").setLevel(logging.WARNING)

# Per-call tracer — each concurrent task sees its own via asyncio ContextVar.
_current_tracer: ContextVar[CallbackTracer | None] = ContextVar(
    "_current_tracer", default=None
)

_TOOL_QUERY_KEY: dict[str, str] = {
    "search_knowledgebase": "keyword",
    "get_entity_by_id": "entity_id",
    "execute": "query",
    "web_search": "query",
    "get_schema_info": "",
}


def _fmt_tool_call(name: str, args: dict) -> str:
    """Return a one-line summary for a tool invocation."""
    key = _TOOL_QUERY_KEY.get(name, "")
    if key and key in args:
        val = str(args[key]).replace("\n", " ")
        if len(val) > 70:
            val = val[:67] + "…"
        return f'{name}("{val}")'
    return name


def _fmt_tool_response(name: str, response: Any) -> str:
    """Return a one-line stats summary for a tool response."""
    if isinstance(response, dict):
        text = str(response.get("result", response))
    else:
        text = str(response)

    if name == "search_knowledgebase":
        n = len(re.findall(r"Result \d+:", text))
        return f"{n} KB result{'s' if n != 1 else ''}"

    if name == "get_entity_by_id":
        if "No entity found" in text:
            return "not found"
        try:
            data = json.loads(text)
            label = data.get("name") or data.get("title") or "found"
            return f"found: {label!r}"
        except Exception:
            return "found"

    if name == "execute":
        if text.startswith("Query Error:"):
            return "error: " + text[12:60].replace("\n", " ")
        m = re.search(r"Truncated at (\d+) rows", text)
        if m:
            return f"{m.group(1)}+ rows (truncated)"
        pipe_rows = [l for l in text.splitlines() if l.startswith("|")]
        n = max(0, len(pipe_rows) - 2)
        return f"{n} record{'s' if n != 1 else ''}"

    if name == "web_search":
        if "[SEARCH_INCONCLUSIVE]" in text:
            return "no grounding sources"
        idx = text.find("CITABLE SOURCES")
        if idx != -1:
            source_lines = [l for l in text[idx:].splitlines() if l.startswith("- ")]
            n = len(source_lines)
            return f"{n} grounding source{'s' if n != 1 else ''}"
        return f"{len(text):,} chars"

    if name == "get_schema_info":
        tables = len(re.findall(r"^Table:", text, re.MULTILINE))
        return f"{tables} table{'s' if tables != 1 else ''} in schema"

    return f"{len(text):,} chars"


def _parse_md_table(text: str) -> tuple[list[str], list[list[str]]] | None:
    """Parse a markdown pipe-table into (columns, rows). Returns None if not a table."""
    lines = [ln for ln in text.strip().splitlines() if ln.strip().startswith("|")]
    if len(lines) < 2:
        return None

    def _cells(line: str) -> list[str]:
        return [c.strip() for c in line.strip().strip("|").split("|")]

    columns = _cells(lines[0])
    rows: list[list[str]] = []
    for line in lines[2:]:  # skip the separator row
        vals = _cells(line)
        if len(vals) == len(columns):
            rows.append(vals)
    return columns, rows


class AmlAgentTask:
    """Langfuse-compatible task wrapper for AML investigations.

    Implements ``__call__(*, item, **kwargs)`` so it can be passed directly
    to ``run_experiment(task=AmlAgentTask(...))``.

    Safe for concurrent use: each call creates its own ``CallbackTracer``
    stored in an ``asyncio.ContextVar``, so concurrent invocations never
    share trace state.

    Parameters
    ----------
    configs : Configs | None
        Agent configuration.  Loaded from env if ``None``.
    temperature : float | None
        LLM sampling temperature.
    timeout_sec : int | None
        Timeout for model calls.
    enable_tracing : bool
        If ``True``, call ``init_tracing()`` before creating the agent.
    """

    def __init__(
        self,
        *,
        configs: Configs | None = None,
        temperature: float | None = None,
        timeout_sec: int | None = None,
        enable_tracing: bool = True,
    ) -> None:
        self.status_fn: Any = None

        if enable_tracing:
            from aml_agent.evaluation.tracing import init_tracing
            init_tracing(service_name="aml-agent")

        # Dispatch callbacks to the tracer active in the current asyncio context.
        def _before_cb(callback_context: Any, llm_request: Any) -> None:
            tracer = _current_tracer.get()
            if tracer is not None:
                return tracer.before_model_callback(callback_context, llm_request)
            return None

        def _after_cb(callback_context: Any, llm_response: Any) -> None:
            tracer = _current_tracer.get()
            if tracer is not None:
                return tracer.after_model_callback(callback_context, llm_response)
            return None

        self._agent = create_aml_agent(
            configs=configs,
            temperature=temperature,
            timeout_sec=timeout_sec,
            before_model_callback=_before_cb,
            after_model_callback=_after_cb,
        )

        # Expose model names sourced directly from the created agent instance
        # so experiment runners (e.g. langfuse.py) can report exactly what
        # models this task is using without any env-var look-ups.
        _na = "unsure or not applicable"
        self.planner_model: str = getattr(self._agent, "model", None) or _na
        self.worker_model: str = _na
        for _t in getattr(self._agent, "tools", []):
            _fn = getattr(_t, "func", None)
            _owner = getattr(_fn, "__self__", None) if _fn else None
            if _owner is not None and hasattr(_owner, "_model_name"):
                self.worker_model = _owner._model_name
                break

        self._runner = Runner(
            app_name="aml_investigation",
            agent=self._agent,
            session_service=InMemorySessionService(),
            auto_create_session=True,
        )

    def _emit(self, msg: str) -> None:
        """Emit an intermediate status line if a status_fn is configured."""
        if self.status_fn is not None:
            try:
                self.status_fn(msg)
            except Exception:
                pass

    async def __call__(self, *, item: Any, **kwargs: Any) -> dict[str, Any] | None:
        """Run the agent on one dataset item.

        Parameters
        ----------
        item
            A Langfuse ``DatasetItemClient`` or a dict-like
            ``LocalExperimentItem``.  Must have an ``input`` field with
            at least ``test_case_info_input``.

        Returns
        -------
        dict | None
            Structured output dict containing the report markdown,
            tool calls, token usage, and trace metrics.
        """
        if isinstance(item, dict):
            item_input = item.get("input", {})
        else:
            item_input = getattr(item, "input", {}) or {}

        tc_id: str = item_input.get("test_case_id", "?")

        subject_prompt = item_input.get("test_case_info_input", "")
        if not subject_prompt:
            subject_prompt = json.dumps(item_input, ensure_ascii=False)

        message = types.Content(
            parts=[types.Part(text=subject_prompt)],
            role="user",
        )

        session_id = str(uuid.uuid4())

        # Fresh tracer bound to this asyncio task context.
        tracer = CallbackTracer()
        token = _current_tracer.set(tracer)

        final_text = ""
        try:
            async for event in self._runner.run_async(
                session_id=session_id,
                user_id=getpass.getuser(),
                new_message=message,
            ):
                if not (event.content and event.content.parts):
                    continue

                for part in event.content.parts:
                    fc = getattr(part, "function_call", None)
                    fr = getattr(part, "function_response", None)
                    if fc:
                        summary = _fmt_tool_call(fc.name, dict(fc.args or {}))
                        self._emit(f"  [dim cyan]{tc_id}[/] [yellow]→[/] {summary}")
                        call_id = getattr(fc, "id", None) or fc.name
                        tracer.record_tool_call(
                            name=fc.name,
                            call_id=call_id,
                            args=dict(fc.args) if fc.args else {},
                        )
                    elif fr:
                        stats = _fmt_tool_response(fr.name, fr.response)
                        self._emit(
                            f"  [dim cyan]{tc_id}[/] [green]←[/] "
                            f"{fr.name}: [dim]{stats}[/]"
                        )
                        call_id = getattr(fr, "id", None) or fr.name
                        resp_text = ""
                        if fr.response:
                            if isinstance(fr.response, dict):
                                resp_text = fr.response.get("result", str(fr.response))
                            else:
                                resp_text = str(fr.response)
                        tracer.record_tool_response(
                            name=fr.name, call_id=call_id, response=resp_text
                        )

                if event.is_final_response() and event.content:
                    final_text = "".join(
                        part.text or "" for part in event.content.parts if part.text
                    )
                    # Strip preamble before formal report heading
                    m = re.search(r"(#\s+AML Investigation Report:)", final_text)
                    if m:
                        final_text = final_text[m.start():]
        finally:
            _current_tracer.reset(token)

        tracer.mark_finished()

        if not final_text:
            logger.warning("No output produced for %s", tc_id)
            self._emit(f"  [dim cyan]{tc_id}[/] [red bold]✗[/] no output produced")
            return None

        # Log final output preview
        words = re.sub(r"\s+", " ", final_text).split()[:10]
        self._emit(
            f"  [dim cyan]{tc_id}[/] [bold green]✓[/] "
            f"{len(final_text):,} chars | [dim]\"{' '.join(words)}…\"[/]"
        )

        # Build rich output with full trace data
        trace_data = tracer.to_dict()

        # Parse SQL results from tool_calls (for HTML transaction table)
        sql_results: list[dict] = []
        for tc_entry in trace_data["tool_calls"]:
            resp_text = tc_entry.get("response", "")
            if (
                tc_entry["tool"] == "execute"
                and resp_text
                and not resp_text.startswith("Query Error")
            ):
                parsed = _parse_md_table(resp_text)
                if parsed and len(parsed[0]) > 2 and parsed[1]:
                    sql_results.append({
                        "query": tc_entry["args"].get("query", ""),
                        "columns": parsed[0],
                        "rows": parsed[1],
                    })

        return {
            "report_markdown": final_text,
            "report": _parse_report_sections(final_text),
            "tool_calls": trace_data["tool_calls"],
            "sql_results": sql_results,
            "session_id": session_id,
            "llm_call_history": trace_data["llm_call_history"],
            "token_usage": trace_data["token_usage"],
            "trace_metrics": trace_data["trace_metrics"],
            "started_at": trace_data["started_at"],
            "finished_at": trace_data["finished_at"],
            "elapsed_sec": trace_data["elapsed_sec"],
        }

    def _close_tools(self) -> None:
        """Close synchronous tool resources (e.g. Weaviate, SQLite connections)."""
        seen: set[int] = set()
        for tool in getattr(self._agent, "tools", []):
            inner = getattr(tool, "func", None)
            owner = getattr(inner, "__self__", None) if inner else None
            if owner is None or id(owner) in seen:
                continue
            seen.add(id(owner))
            if hasattr(owner, "close"):
                try:
                    owner.close()
                except Exception:
                    pass

    async def close(self) -> None:
        """Release agent, runner, and tool resources (async variant)."""
        await self._runner.close()
        self._close_tools()

    def close_sync(self) -> None:
        """Release tool resources synchronously (no event loop needed)."""
        self._close_tools()

