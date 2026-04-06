"""Tool call enforcement for the AML Investigation Agent.

Ensures all required investigation tools are called at least once per session
before the agent writes its final report.

Two complementary mechanisms:
1. **Mid-run injection** (before_model_callback): When the agent has started
   producing turns but has not yet called all required tools, inject a user
   message reminder into the LLM request before the next call.
2. **Premature-finalization intercept** (after_model_callback): If the model
   produces a text-only final response (contains the AML report header) while
   required tools are still uncalled, replace the response with a synthetic
   function call for the first missing tool so the agent continues.

After ``MAX_ENFORCE_TIMES`` combined interventions the agent is allowed to
proceed unchecked to avoid infinite enforcement loops.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from google.adk.models import LlmResponse
from google.genai import types

logger = logging.getLogger(__name__)

# Tools that MUST be called at least once per investigation.
# Ordered by expected invocation sequence (KB → SQL → Web).
REQUIRED_TOOLS_ORDERED: tuple[str, ...] = (
    "search_knowledgebase",
    "execute",
    "web_search",
)
REQUIRED_TOOLS: frozenset[str] = frozenset(REQUIRED_TOOLS_ORDERED)

MAX_ENFORCE_TIMES: int = 3

# Regex that matches the AML report header — indicates the model is finalising.
_REPORT_HEADER_RE = re.compile(r"#\s+AML Investigation Report", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_called_tools(contents: list[Any]) -> set[str]:
    """Return the set of tool names that appear as function_calls in *contents*."""
    called: set[str] = set()
    for content in contents or []:
        for part in getattr(content, "parts", None) or []:
            fc = getattr(part, "function_call", None)
            if fc:
                name = getattr(fc, "name", None)
                if name:
                    called.add(name)
    return called


def _extract_subject_from_contents(contents: list[Any]) -> str:
    """Extract the investigation subject from the first user message."""
    for content in contents or []:
        if getattr(content, "role", "") != "user":
            continue
        for part in getattr(content, "parts", None) or []:
            text = getattr(part, "text", None) or ""
            if "compliance" in text.lower():
                # Runner formats: "Investigate the following subject for AML compliance:\n\n{subject}"
                after_newlines = text.split("\n\n", 1)
                if len(after_newlines) > 1:
                    return after_newlines[1].strip()
            if text.strip():
                return text.strip()
    return ""


def _is_premature_final_response(llm_response: Any) -> bool:
    """Return True when the model produced a text-only response containing the AML report header.

    A "premature" response is one where:
    - There are no function_call parts (the model is not requesting a tool)
    - The text contains the AML report title line
    """
    content = getattr(llm_response, "content", None)
    if not content:
        return False
    parts = getattr(content, "parts", None) or []
    if any(getattr(p, "function_call", None) for p in parts):
        return False
    # Collect all text — skip thought parts since they don't affect finalization
    text = "".join(
        getattr(p, "text", "") or ""
        for p in parts
        if not getattr(p, "thought", False)
    )
    return bool(_REPORT_HEADER_RE.search(text))


def _build_synthetic_function_call(tool_name: str, subject: str) -> dict[str, Any]:
    """Build minimal valid args for a synthetic tool call used to force continuation."""
    clean = subject.strip()[:200]
    if tool_name == "web_search":
        return {
            "query": f'"{clean}" sanctions OR "adverse media" OR fraud OR indictment'
        }
    if tool_name == "search_knowledgebase":
        return {"keyword": clean}
    if tool_name == "execute":
        # A broad name search — better than nothing when the agent skipped SQL entirely.
        # Use the first significant token from the subject as the LIKE pattern.
        token = re.split(r"[\s,]+", clean.upper())[0] if clean else "UNKNOWN"
        return {
            "query": (
                f"SELECT transaction_id, amount, currency, sender_name, receiver_name "
                f"FROM transactions "
                f"WHERE sender_name LIKE '%{token}%' "
                f"   OR receiver_name LIKE '%{token}%' "
                f"   OR memo LIKE '%{token}%' "
                f"LIMIT 20"
            )
        }
    if tool_name == "get_schema_info":
        return {"table_names": None}
    return {}


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class ToolEnforcementCallback:
    """ADK callback pair that enforces minimum tool-call coverage per session.

    Usage::

        enforcer = ToolEnforcementCallback()
        agent = create_aml_agent(
            before_model_callback=enforcer.before_model_callback,
            after_model_callback=enforcer.after_model_callback,
        )
        # After the run:
        stats = enforcer.get_stats(session_id)
    """

    def __init__(self) -> None:
        # Per-session state: keyed by session_id
        self._sessions: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Internal state helpers
    # ------------------------------------------------------------------

    def _state(self, session_id: str) -> dict[str, Any]:
        if session_id not in self._sessions:
            self._sessions[session_id] = {
                "called_tools": set(),
                "subject": "",
                "enforce_count": 0,        # total interventions (before + after)
                "mid_run_reminders": 0,    # interventions via before_model_callback
                "post_intercepts": 0,      # interventions via after_model_callback
            }
        return self._sessions[session_id]

    def _session_id(self, callback_context: Any) -> str:
        try:
            return callback_context.session.id
        except AttributeError:
            return "unknown"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_stats(self, session_id: str) -> dict[str, Any]:
        """Return enforcement statistics for *session_id*.

        Returns an empty dict when the session has never been seen.
        """
        st = self._sessions.get(session_id)
        if st is None:
            return {}
        missing = sorted(REQUIRED_TOOLS - st["called_tools"])
        return {
            "session_id": session_id,
            "called_tools": sorted(st["called_tools"]),
            "missing_tools": missing,
            "all_required_called": not missing,
            "mid_run_reminders": st["mid_run_reminders"],
            "post_intercepts": st["post_intercepts"],
            "total_enforcements": st["enforce_count"],
            "max_enforce_times": MAX_ENFORCE_TIMES,
            "enforcement_limit_reached": st["enforce_count"] >= MAX_ENFORCE_TIMES,
        }

    # ------------------------------------------------------------------
    # ADK callbacks
    # ------------------------------------------------------------------

    def before_model_callback(
        self,
        callback_context: Any,
        llm_request: Any,
    ) -> LlmResponse | None:
        """Inject a mid-run reminder when required tools are still uncalled.

        Modifies ``llm_request.contents`` in-place by appending a user turn
        that names the missing tools and asks the agent to call them now.
        Only fires when at least one model turn already exists in the
        conversation (i.e. the agent has had a chance to call tools).

        Returns ``None`` always — the LLM call proceeds normally.
        """
        session_id = self._session_id(callback_context)
        st = self._state(session_id)

        # Refresh called-tools from the full conversation history
        contents = llm_request.contents or []
        st["called_tools"] = _extract_called_tools(contents)

        # Extract subject on first call
        if not st["subject"]:
            st["subject"] = _extract_subject_from_contents(contents)

        missing = REQUIRED_TOOLS - st["called_tools"]
        if not missing:
            return None  # All required tools already called

        if st["enforce_count"] >= MAX_ENFORCE_TIMES:
            logger.warning(
                "[ToolEnforcement:%s] Limit reached (%d/%d). "
                "Allowing agent to proceed without: %s",
                session_id[:8],
                st["enforce_count"],
                MAX_ENFORCE_TIMES,
                sorted(missing),
            )
            return None

        # Only inject a mid-run reminder after the agent has already produced
        # ≥1 model turn.  On the very first call there is nothing to warn about.
        model_turns = sum(
            1 for c in contents if getattr(c, "role", "") == "model"
        )
        if model_turns == 0:
            return None

        st["enforce_count"] += 1
        st["mid_run_reminders"] += 1

        called_str = ", ".join(sorted(st["called_tools"])) or "none"
        missing_ordered = [t for t in REQUIRED_TOOLS_ORDERED if t in missing]

        reminder = (
            f"[INVESTIGATION TOOL ENFORCEMENT — reminder "
            f"{st['enforce_count']}/{MAX_ENFORCE_TIMES}]\n"
            f"You have NOT yet called all required investigation tools. "
            f"Do NOT write your final report yet.\n\n"
            f"Tools called so far:    {called_str}\n"
            f"Tools still required:   {', '.join(missing_ordered)}\n\n"
            f"Call the next missing tool ({missing_ordered[0]!r}) now before "
            f"continuing.  The subject under investigation is: "
            f"{st['subject'][:150]}"
        )

        logger.info(
            "[ToolEnforcement:%s] Mid-run reminder #%d — "
            "called=%s missing=%s",
            session_id[:8],
            st["enforce_count"],
            sorted(st["called_tools"]),
            missing_ordered,
        )

        # Append in-place so the model sees the reminder on this call
        llm_request.contents.append(
            types.Content(
                role="user",
                parts=[types.Part(text=reminder)],
            )
        )
        return None

    def after_model_callback(
        self,
        callback_context: Any,
        llm_response: Any,
    ) -> LlmResponse | None:
        """Intercept a premature final response and inject a synthetic tool call.

        When the model writes a final report without having called all required
        tools, this method constructs a synthetic ``LlmResponse`` containing a
        ``FunctionCall`` for the first missing tool.  ADK will execute that
        tool call as normal and the agent will continue.

        Returns ``None`` when no intervention is needed (the common case).
        """
        session_id = self._session_id(callback_context)
        st = self._state(session_id)

        missing = REQUIRED_TOOLS - st["called_tools"]
        if not missing:
            return None

        if not _is_premature_final_response(llm_response):
            return None

        if st["enforce_count"] >= MAX_ENFORCE_TIMES:
            logger.warning(
                "[ToolEnforcement:%s] Limit reached (%d/%d). "
                "Accepting premature response. Missing: %s",
                session_id[:8],
                st["enforce_count"],
                MAX_ENFORCE_TIMES,
                sorted(missing),
            )
            return None

        st["enforce_count"] += 1
        st["post_intercepts"] += 1

        # Pick the first missing tool in invocation order
        first_missing = next(
            (t for t in REQUIRED_TOOLS_ORDERED if t in missing), sorted(missing)[0]
        )
        called_str = ", ".join(sorted(st["called_tools"])) or "none"

        logger.warning(
            "[ToolEnforcement:%s] Intercepted premature final response #%d — "
            "called=%s missing=%s  →  injecting call for %r  subject=%r",
            session_id[:8],
            st["enforce_count"],
            sorted(st["called_tools"]),
            sorted(missing),
            first_missing,
            st["subject"][:60],
        )

        args = _build_synthetic_function_call(first_missing, st["subject"])
        return LlmResponse(
            content=types.Content(
                role="model",
                parts=[
                    types.Part(
                        function_call=types.FunctionCall(
                            name=first_missing,
                            args=args,
                        )
                    )
                ],
            )
        )


__all__ = [
    "ToolEnforcementCallback",
    "REQUIRED_TOOLS",
    "REQUIRED_TOOLS_ORDERED",
    "MAX_ENFORCE_TIMES",
]
