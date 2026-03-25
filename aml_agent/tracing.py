"""Reusable callback-based tracing for LLM calls and tool invocations.

Provides ``CallbackTracer`` which hooks into Google ADK model callbacks
to capture LLM calls, responses, token usage, and tool invocations.
Collected data is available via ``to_dict()`` for artifact JSON or Langfuse.
"""

from __future__ import annotations

import datetime
import logging
from typing import Any

logger = logging.getLogger(__name__)


def serialize_part(part: Any, *, include_thoughts: bool = False) -> dict[str, Any] | None:
    """Serialize a single ``google.genai.types.Part`` to a JSON-friendly dict.

    Parameters
    ----------
    part : Any
        A ``google.genai.types.Part`` object.
    include_thoughts : bool
        If ``True``, include thought parts.  Otherwise they are skipped.
    """
    is_thought = getattr(part, "thought", False)
    if is_thought and not include_thoughts:
        return None

    result: dict[str, Any] | None = None
    if part.text is not None:
        result = {"type": "text", "text": part.text}
    else:
        fc = getattr(part, "function_call", None)
        if fc:
            result = {
                "type": "function_call",
                "name": fc.name,
                "id": getattr(fc, "id", None),
                "args": dict(fc.args) if fc.args else {},
            }
        else:
            fr = getattr(part, "function_response", None)
            if fr:
                resp = fr.response or {}
                resp_val = (
                    resp.get("result", str(resp))
                    if isinstance(resp, dict)
                    else str(resp)
                )
                result = {
                    "type": "function_response",
                    "name": fr.name,
                    "id": getattr(fr, "id", None),
                    "result": resp_val,
                }
    if result is not None and is_thought:
        result["thought"] = True
    return result


def parse_md_table(text: str) -> tuple[list[str], list[list[str]]] | None:
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


def serialize_content(content: Any) -> dict[str, Any] | str | list | None:
    """Serialize a ``types.Content`` (or plain string / list) for JSON."""
    if content is None:
        return None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return [serialize_content(c) for c in content]
    parts = getattr(content, "parts", None)
    if parts is None:
        return str(content)
    serialized_parts = []
    for p in parts:
        s = serialize_part(p, include_thoughts=True)
        if s is not None:
            serialized_parts.append(s)
    return {
        "role": getattr(content, "role", None),
        "parts": serialized_parts,
    }


class CallbackTracer:
    """Records LLM calls and tool invocations during an agent run.

    Safe to reuse across sequential runs by calling ``reset()``.
    """

    def __init__(self, *, capture_tools_info: bool = True) -> None:
        self._capture_tools_info = capture_tools_info
        self.reset()

    def reset(self) -> None:
        """Clear all accumulated trace data."""
        self._system_instruction: dict[str, Any] | str | None = None
        self._llm_calls: list[dict[str, Any]] = []
        self._tool_calls: list[dict[str, Any]] = []
        self._pending_tool_calls: dict[str, dict[str, Any]] = {}
        self._call_counter: int = 0
        self._prev_contents_len: int = 0
        self._started_at: datetime.datetime | None = None
        self._finished_at: datetime.datetime | None = None

    def before_model_callback(self, callback_context: Any, llm_request: Any) -> None:
        """ADK ``before_model_callback`` — records the outgoing LLM request."""
        if self._started_at is None:
            self._started_at = datetime.datetime.now(datetime.timezone.utc)

        # Capture system instruction once
        if (
            self._system_instruction is None
            and llm_request.config
            and llm_request.config.system_instruction
        ):
            self._system_instruction = serialize_content(
                llm_request.config.system_instruction
            )

        # Only serialize the NEW contents added since the last call
        all_contents = llm_request.contents
        new_contents = [
            serialize_content(c) for c in all_contents[self._prev_contents_len :]
        ]
        total_len = len(all_contents)

        # Capture tool declarations on first call (opt-in)
        tools_info = None
        if self._capture_tools_info and self._call_counter == 0 and llm_request.config:
            tool_config = getattr(llm_request.config, "tools", None)
            if tool_config:
                tools_info = []
                for tool_group in tool_config:
                    decls = getattr(tool_group, "function_declarations", None)
                    if decls:
                        for decl in decls:
                            tools_info.append({
                                "name": getattr(decl, "name", None),
                                "description": (
                                    getattr(decl, "description", None) or ""
                                )[:200],
                            })

        entry: dict[str, Any] = {
            "call_index": self._call_counter,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "request": {
                "model": llm_request.model,
                "new_contents": new_contents,
                "total_contents_count": total_len,
            },
            "response": None,
            "usage": None,
        }
        if tools_info is not None:
            entry["available_tools"] = tools_info

        self._llm_calls.append(entry)
        self._prev_contents_len = total_len
        self._call_counter += 1
        return None  # continue with normal LLM call

    def after_model_callback(self, callback_context: Any, llm_response: Any) -> None:
        """ADK ``after_model_callback`` — records the LLM response + usage."""
        resp_data: dict[str, Any] = {}
        if llm_response.content:
            resp_data["content"] = serialize_content(llm_response.content)
        if llm_response.finish_reason:
            resp_data["finish_reason"] = str(llm_response.finish_reason)

        usage_data: dict[str, Any] | None = None
        um = getattr(llm_response, "usage_metadata", None)
        if um:
            usage_data = {
                "prompt_token_count": getattr(um, "prompt_token_count", None),
                "candidates_token_count": getattr(um, "candidates_token_count", None),
                "total_token_count": getattr(um, "total_token_count", None),
                "thoughts_token_count": getattr(um, "thoughts_token_count", None),
                "cached_content_token_count": getattr(
                    um, "cached_content_token_count", None
                ),
            }

        if self._llm_calls:
            self._llm_calls[-1]["response"] = resp_data
            self._llm_calls[-1]["usage"] = usage_data
        return None  # continue with original response

    def record_tool_call(self, name: str, call_id: str, args: dict[str, Any]) -> None:
        self._pending_tool_calls[call_id] = {
            "tool": name,
            "args": args,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "response": None,
        }

    def record_tool_response(
        self, name: str, call_id: str, response: str
    ) -> dict[str, Any]:
        """Record a tool response and pair it with the matching call."""
        if call_id in self._pending_tool_calls:
            entry = self._pending_tool_calls.pop(call_id)
            entry["response"] = response
        else:
            entry = {
                "tool": name,
                "args": {},
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "response": response,
            }
        self._tool_calls.append(entry)
        return entry

    def process_event_parts(self, parts: list[Any]) -> None:
        """Track tool calls/responses from an ADK event's ``content.parts``."""
        for part in parts:
            fc = getattr(part, "function_call", None)
            if fc:
                call_id = getattr(fc, "id", None) or fc.name
                self.record_tool_call(
                    name=fc.name,
                    call_id=call_id,
                    args=dict(fc.args) if fc.args else {},
                )

            fr = getattr(part, "function_response", None)
            if fr:
                call_id = getattr(fr, "id", None) or fr.name
                resp_text = ""
                if fr.response:
                    if isinstance(fr.response, dict):
                        resp_text = fr.response.get("result", str(fr.response))
                    else:
                        resp_text = str(fr.response)
                self.record_tool_response(
                    name=fr.name, call_id=call_id, response=resp_text
                )

    def mark_finished(self) -> None:
        self._finished_at = datetime.datetime.now(datetime.timezone.utc)

    @property
    def llm_calls(self) -> list[dict[str, Any]]:
        return self._llm_calls

    @property
    def tool_calls(self) -> list[dict[str, Any]]:
        return self._tool_calls

    @property
    def system_instruction(self) -> dict[str, Any] | str | None:
        return self._system_instruction

    def get_token_usage(self) -> dict[str, Any]:
        """Compute aggregate token usage across all LLM calls."""
        totals = {
            "total_llm_calls": len(self._llm_calls),
            "prompt_token_count": 0,
            "candidates_token_count": 0,
            "thoughts_token_count": 0,
            "total_token_count": 0,
        }
        for call in self._llm_calls:
            u = call.get("usage")
            if u:
                for key in (
                    "prompt_token_count",
                    "candidates_token_count",
                    "thoughts_token_count",
                    "total_token_count",
                ):
                    totals[key] += u.get(key) or 0
        return totals

    def get_elapsed_sec(self) -> float | None:
        """Elapsed seconds from first LLM call to finish, or None."""
        if self._started_at and self._finished_at:
            return round(
                (self._finished_at - self._started_at).total_seconds(), 3
            )
        return None

    def get_trace_metrics(self) -> dict[str, Any]:
        """High-level trace metrics for dashboards / scoring."""
        unique_tools = sorted({tc["tool"] for tc in self._tool_calls})
        planner_models = sorted({
            c["request"]["model"]
            for c in self._llm_calls
            if c.get("request", {}).get("model")
        })
        return {
            "llm_call_count": len(self._llm_calls),
            "tool_call_count": len(self._tool_calls),
            "unique_tools_used": unique_tools,
            "planner_models_used": planner_models,
            "elapsed_sec": self.get_elapsed_sec(),
            "token_usage": self.get_token_usage(),
        }

    def to_dict(self) -> dict[str, Any]:
        """Export all trace data as a JSON-serialisable dict."""
        return {
            "llm_call_history": {
                "system_instruction": self._system_instruction,
                "calls": self._llm_calls,
            },
            "tool_calls": self._tool_calls,
            "token_usage": self.get_token_usage(),
            "trace_metrics": self.get_trace_metrics(),
            "started_at": (
                self._started_at.isoformat() if self._started_at else None
            ),
            "finished_at": (
                self._finished_at.isoformat() if self._finished_at else None
            ),
            "elapsed_sec": self.get_elapsed_sec(),
        }
