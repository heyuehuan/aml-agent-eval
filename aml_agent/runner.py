"""CLI runner for the AML Investigation Agent.

Runs the agent on a subject name and produces a full AML investigation report.
Also captures a structured audit trail of every tool call and response,
saved automatically as a <report>.artifacts.json sidecar.

Usage:
    python -m aml_agent.runner "John Doe"
    python -m aml_agent.runner --subject "John Doe" --output report.md
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import getpass
import json
import logging
import re
import subprocess
import sys
import uuid
import warnings
from pathlib import Path
from typing import Any

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from aml_agent.agent import create_aml_agent
from aml_agent.config import Configs

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _get_git_version() -> str:
    """Return a compact version string: '<branch>/<sha7>' or just '<sha7>'."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short=7", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=3,
        ).strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=3,
        ).strip()
        if branch and branch != "HEAD":
            return f"{branch}/{sha}"
        return sha
    except Exception:
        return "unknown"

# ---------------------------------------------------------------------------
# Suppress noisy warnings from upstream libraries
# ---------------------------------------------------------------------------
# google-genai uses an aiohttp subclass that triggers a DeprecationWarning
warnings.filterwarnings(
    "ignore",
    message="Inheritance class AiohttpClientSession",
    category=DeprecationWarning,
)
# google-genai logs "non-text parts in the response" when a response contains
# function_call parts — this is expected during every agent tool-call cycle.
# Root cause: ADK's _build_response_log() in google_llm.py accesses resp.text
# on every LLM response; when the model returns function_call parts (normal
# during tool use), the .text property in google/genai/types.py:~6817 logs
# a WARNING.  The function_call parts ARE handled by ADK via resp.function_calls
# so this warning is benign.  Logger name uses underscore, not dot.
logging.getLogger("google_genai.types").setLevel(logging.ERROR)
# httpx logs every HTTP request at INFO (including Weaviate's mandatory /v1/meta
# connect call which cannot be skipped — it provides server version + GRPC config).
logging.getLogger("httpx").setLevel(logging.WARNING)


def _parse_md_table(text: str) -> tuple[list[str], list[list[str]]] | None:
    """Parse a markdown pipe-table into (columns, rows). Returns None if not a table."""
    lines = [ln for ln in text.strip().splitlines() if ln.strip().startswith("|")]
    if len(lines) < 2:
        return None

    def _cells(line: str) -> list[str]:
        return [c.strip() for c in line.strip().strip("|").split("|")]

    columns = _cells(lines[0])
    rows: list[list[str]] = []
    for line in lines[2:]:  # skip the separator row (--- | --- | ...)
        vals = _cells(line)
        if len(vals) == len(columns):
            rows.append(vals)
    return columns, rows


def _parse_report_sections(markdown: str) -> dict[str, Any]:
    """Break a markdown report into structured sections + parse citations.

    Returns a dict with:
      - risk_level  : str  (HIGH / MEDIUM / LOW / CLEAR)
      - subject     : str
      - sections    : list[{heading, body}]
      - citations   : list[{num, title, url, excerpt}]
    """
    lines = markdown.strip().splitlines()
    risk_level = "CLEAR"
    subject = ""
    sections: list[dict] = []
    current_heading = ""
    current_body: list[str] = []

    def flush() -> None:
        # Skip the title gap (no heading, no content) and bare risk-assessment lines
        body_text = "\n".join(current_body).strip()
        if not current_heading and not body_text:
            return
        if re.match(r"risk assessment", current_heading, re.IGNORECASE) and not body_text:
            return
        sections.append({"heading": current_heading, "body": body_text})

    for line in lines:
        h1 = re.match(r"^#\s+(.+)$", line)
        if h1:
            flush()
            current_heading = ""
            current_body = []
            m = re.search(r":\s*(.+)$", h1.group(1))
            if m:
                subject = m.group(1).strip()
            continue
        h2 = re.match(r"^##\s+(.+)$", line)
        if h2:
            flush()
            current_heading = h2.group(1)
            current_body = []
            risk_m = re.search(r"Risk Assessment:\s*(HIGH|MEDIUM|LOW|CLEAR)", current_heading, re.IGNORECASE)
            if risk_m:
                risk_level = risk_m.group(1).upper()
            continue
        current_body.append(line)
    flush()

    # Parse citations from the Sources section
    citations: list[dict] = []
    for sec in sections:
        if not re.search(r"source", sec["heading"], re.IGNORECASE):
            continue
        for line in sec["body"].splitlines():
            cite_m = re.match(r"^\[(\d+)\]\s+(.+)$", line.strip())
            if not cite_m:
                continue
            num = int(cite_m.group(1))
            rest = cite_m.group(2).strip()
            parts = [p.strip() for p in rest.split(" | ")]
            entry: dict[str, Any] = {"num": num, "title": parts[0]}
            if len(parts) >= 2:
                url = parts[1]
                entry["url"] = url if url.startswith("http") else None
            if len(parts) >= 3:
                entry["excerpt"] = parts[2]
            citations.append(entry)

    return {
        "risk_level": risk_level,
        "subject": subject,
        "sections": sections,
        "citations": citations,
    }


def _serialize_part(part: Any, *, include_thoughts: bool = False) -> dict[str, Any] | None:
    """Serialize a single Content Part to a JSON-friendly dict.

    Parameters
    ----------
    part : Any
        A google.genai.types.Part object.
    include_thoughts : bool
        If True, include thought parts (marked with ``"thought": true``).
        Otherwise thought parts are skipped entirely.
    """
    is_thought = getattr(part, "thought", False)
    if is_thought and not include_thoughts:
        return None  # skip internal reasoning tokens
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
                resp_val = resp.get("result", str(resp)) if isinstance(resp, dict) else str(resp)
                result = {
                    "type": "function_response",
                    "name": fr.name,
                    "id": getattr(fr, "id", None),
                    "result": resp_val,
                }
    if result is not None and is_thought:
        result["thought"] = True
    return result


def _serialize_content(content: Any) -> dict[str, Any] | str | list | None:
    """Serialize a types.Content (or plain string / list) for JSON output."""
    if content is None:
        return None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return [_serialize_content(c) for c in content]
    parts = getattr(content, "parts", None)
    if parts is None:
        return str(content)
    serialized_parts = []
    for p in parts:
        s = _serialize_part(p, include_thoughts=True)
        if s is not None:
            serialized_parts.append(s)
    return {
        "role": getattr(content, "role", None),
        "parts": serialized_parts,
    }


async def run_investigation(
    subject: str,
    *,
    configs: Configs | None = None,
    temperature: float | None = None,
    timeout_sec: int | None = None,
) -> tuple[str, dict[str, Any]]:
    """Run an AML investigation on a named subject.

    Parameters
    ----------
    subject : str
        The name of the person or entity to investigate.
    configs : Configs | None
        Configuration. If None, loads from environment.
    temperature : float | None
        LLM temperature.
    timeout_sec : int | None
        Timeout for model calls.

    Returns
    -------
    tuple[str, dict]
        The full investigation report text and a structured artifacts dict
        containing the raw tool call/response audit trail.
    """
    # ── Full LLM call trace — captured via ADK before/after model callbacks ──
    # System instruction is the same across all calls; stored once at the top
    # level.  Each call entry records only the NEW contents added since the
    # previous call (the delta), avoiding massive duplication.
    llm_system_instruction: dict[str, Any] | str | None = None
    llm_calls: list[dict[str, Any]] = []
    _call_counter = 0
    _prev_contents_len = 0  # track how many contents the previous call had

    def _before_model_cb(callback_context, llm_request):
        nonlocal _call_counter, _prev_contents_len, llm_system_instruction

        # Capture system instruction once (it's identical for every call)
        if llm_system_instruction is None and llm_request.config and llm_request.config.system_instruction:
            llm_system_instruction = _serialize_content(llm_request.config.system_instruction)

        # Only serialize the NEW contents added since the last call
        all_contents = llm_request.contents
        new_contents = [_serialize_content(c) for c in all_contents[_prev_contents_len:]]
        total_len = len(all_contents)

        llm_calls.append({
            "call_index": _call_counter,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "request": {
                "model": llm_request.model,
                "new_contents": new_contents,
                "total_contents_count": total_len,
            },
            "response": None,  # filled in by after callback
            "usage": None,     # filled in by after callback
        })
        _prev_contents_len = total_len
        _call_counter += 1
        return None  # continue with normal LLM call

    def _after_model_cb(callback_context, llm_response):
        # Attach the response and token usage to the most recent call entry
        resp_data: dict[str, Any] = {}
        if llm_response.content:
            resp_data["content"] = _serialize_content(llm_response.content)
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
                "cached_content_token_count": getattr(um, "cached_content_token_count", None),
            }

        if llm_calls:
            llm_calls[-1]["response"] = resp_data
            llm_calls[-1]["usage"] = usage_data
        return None  # continue with original response

    agent = create_aml_agent(
        configs=configs,
        temperature=temperature,
        timeout_sec=timeout_sec,
        before_model_callback=_before_model_cb,
        after_model_callback=_after_model_cb,
    )

    runner = Runner(
        app_name="aml_investigation",
        agent=agent,
        session_service=InMemorySessionService(),
    )

    session_id = str(uuid.uuid4())
    await runner.session_service.create_session(
        app_name="aml_investigation",
        user_id=getpass.getuser(),
        session_id=session_id,
    )

    message = types.Content(
        parts=[types.Part(text=f"Investigate the following subject for AML compliance:\n\n{subject}")],
        role="user",
    )

    # Audit trail — captured from ADK events
    _started_at = datetime.datetime.now(datetime.timezone.utc)
    _version = _get_git_version()
    artifacts: dict[str, Any] = {
        "subject": subject,
        "session_id": session_id,
        "version": _version,
        "started_at": _started_at.isoformat(),
        "finished_at": None,
        "elapsed_sec": None,
        "tool_calls": [],  # list of {tool, args, response, timestamp} — compact tool audit
        "sql_results": [],  # list of {query, columns, rows} from execute tool calls
    }
    _pending_calls: dict[str, dict] = {}  # call_id → {tool, args, timestamp}

    final_text = ""
    async for event in runner.run_async(
        session_id=session_id,
        user_id=getpass.getuser(),
        new_message=message,
    ):
        if event.content and event.content.parts:
            # ── Maintain compact tool_calls audit trail ──
            for part in event.content.parts:
                if hasattr(part, "function_call") and part.function_call:
                    fc = part.function_call
                    call_id = getattr(fc, "id", None) or fc.name
                    _pending_calls[call_id] = {
                        "tool": fc.name,
                        "args": dict(fc.args) if fc.args else {},
                        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        "response": None,
                    }
                if hasattr(part, "function_response") and part.function_response:
                    fr = part.function_response
                    call_id = getattr(fr, "id", None) or fr.name
                    resp_text = ""
                    if fr.response:
                        if isinstance(fr.response, dict):
                            resp_text = fr.response.get("result", str(fr.response))
                        else:
                            resp_text = str(fr.response)
                    if call_id in _pending_calls:
                        tc_entry = _pending_calls.pop(call_id)
                        tc_entry["response"] = resp_text
                        artifacts["tool_calls"].append(tc_entry)
                        # Accumulate raw SQL rows for the interactive transaction table
                        if tc_entry["tool"] == "execute" and resp_text and not resp_text.startswith("Query Error"):
                            parsed = _parse_md_table(resp_text)
                            if parsed and len(parsed[0]) > 2 and parsed[1]:
                                artifacts["sql_results"].append({
                                    "query": tc_entry["args"].get("query", ""),
                                    "columns": parsed[0],
                                    "rows": parsed[1],
                                })
                    else:
                        # Response without matching call (tool name as id)
                        artifacts["tool_calls"].append({
                            "tool": fr.name,
                            "args": {},
                            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                            "response": resp_text,
                        })

        if event.is_final_response() and event.content:
            final_text = "".join(
                part.text or "" for part in event.content.parts if part.text
            )
            # Strip any conversational preamble before the formal report heading
            match = re.search(r"(#\s+AML Investigation Report:)", final_text)
            if match:
                final_text = final_text[match.start():]

    # ── Structured report breakdown ──
    report_parsed = _parse_report_sections(final_text)

    artifacts["report_markdown"] = final_text
    artifacts["report"] = report_parsed

    # ── Token usage summary ──
    total_usage = {
        "total_llm_calls": len(llm_calls),
        "prompt_token_count": 0,
        "candidates_token_count": 0,
        "thoughts_token_count": 0,
        "total_token_count": 0,
    }
    for call in llm_calls:
        u = call.get("usage")
        if u:
            for key in ("prompt_token_count", "candidates_token_count", "thoughts_token_count", "total_token_count"):
                total_usage[key] += u.get(key) or 0

    _finished_at = datetime.datetime.now(datetime.timezone.utc)
    artifacts["finished_at"] = _finished_at.isoformat()
    artifacts["elapsed_sec"] = round((_finished_at - _started_at).total_seconds(), 3)

    artifacts["token_usage"] = total_usage
    artifacts["llm_call_history"] = {
        "system_instruction": llm_system_instruction,
        "calls": llm_calls,
    }

    await runner.close()

    # Clean up tool resources (Weaviate connection, SQLAlchemy pool)
    for tool in getattr(agent, "tools", []):
        inner = getattr(tool, "func", None)
        owner = getattr(inner, "__self__", None) if inner else None
        if owner and hasattr(owner, "close"):
            try:
                owner.close()
            except Exception:
                pass

    return final_text, artifacts


def main():
    parser = argparse.ArgumentParser(description="AML Investigation Agent")
    parser.add_argument("subject", nargs="?", help="Subject name to investigate")
    parser.add_argument("--subject", "-s", dest="subject_flag", help="Subject name (alternative flag)")
    parser.add_argument("--output", "-o", help="Output file path. Use .html extension for HTML output.")
    parser.add_argument("--html", action="store_true", help="Render output as HTML (auto-detected from .html extension)")
    parser.add_argument("--temperature", "-t", type=float, default=None, help="LLM temperature")
    parser.add_argument("--timeout", type=int, default=None, help="Timeout in seconds")

    args = parser.parse_args()
    subject = args.subject or args.subject_flag

    if not subject:
        parser.error("Subject name is required. Usage: python -m aml_agent.runner 'John Doe'")

    print(f"Starting AML investigation for: {subject}")
    print("-" * 60)

    report, artifacts = asyncio.run(run_investigation(
        subject,
        temperature=args.temperature,
        timeout_sec=args.timeout,
    ))

    output_as_html = args.html or (args.output and args.output.lower().endswith(".html"))

    # Use the subject name the LLM extracted and placed in the report title,
    # falling back to the raw input if parsing finds nothing.
    _title_m = re.search(r"#\s+AML Investigation Report:\s*(.+)", report)
    display_subject = _title_m.group(1).strip() if _title_m else subject

    if output_as_html:
        from aml_agent.report_html import render_html
        content = render_html(
            report,
            subject=display_subject,
            sql_results=artifacts.get("sql_results", []),
            version=artifacts.get("version", "unknown"),
        )
        fmt = "HTML"
    else:
        content = report
        fmt = "markdown"

    if args.output:
        out_path = Path(args.output)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"\nReport saved to: {args.output} ({fmt})")

        # Always save the artifacts sidecar alongside the report
        artifacts_path = out_path.with_suffix(".artifacts.json")
        with open(artifacts_path, "w", encoding="utf-8") as f:
            json.dump(artifacts, f, indent=2, ensure_ascii=False)
        print(f"Artifacts saved to: {artifacts_path}")
    else:
        if output_as_html:
            print("[HTML output to stdout — redirect to a file for best results]")
        print(content)


if __name__ == "__main__":
    main()
