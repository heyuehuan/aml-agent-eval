"""Web-search graders.

Metrics
-------
``open_search_urls_reachable_pct``
    Code-based check that verifies each URL cited in the agent's report
    (from web/open-search results) is reachable via an HTTP HEAD request.
    Returns the fraction of reachable URLs (0.0–1.0, 3 decimal places).

``open_search_results_relevance_llm``
    LLM-as-a-judge metric that evaluates whether the open-search findings
    in the agent's final report are relevant to AML risk assessment and
    enhanced due diligence.  Returns the fraction of relevant information
    points (0.0–1.0, 3 decimal places).

See ``CONTRIBUTING_EVALUATION.md`` for guidance on adding graders.
"""

from __future__ import annotations

import logging
import re
from typing import Any
from urllib.parse import urlparse

import requests

from aml_agent.evaluation.types import Evaluation

from .llm_judge import (
    build_judge_error_evaluation,
    run_llm_judge_structured,
)

logger = logging.getLogger(__name__)


def _web_search_was_called(output: Any) -> bool:
    """Return True if the agent made at least one web_search tool call."""
    if not isinstance(output, dict):
        return False
    return any(
        isinstance(tc, dict) and tc.get("tool") == "web_search"
        for tc in (output.get("tool_calls") or [])
    )


# ---------------------------------------------------------------------------
# open_search_urls_reachable_pct — code-based
# ---------------------------------------------------------------------------

_REACHABLE_METRIC = "open_search_urls_reachable_pct"
_REQUEST_TIMEOUT = 15  # seconds per URL


def _extract_citation_urls(output: Any) -> list[str]:
    """Extract unique, non-empty URLs from the report's citations list."""
    if not isinstance(output, dict):
        return []
    report = output.get("report")
    if not isinstance(report, dict):
        return []
    citations = report.get("citations")
    if not isinstance(citations, list):
        return []

    urls: list[str] = []
    seen: set[str] = set()
    for cite in citations:
        if not isinstance(cite, dict):
            continue
        url = (cite.get("url") or "").strip()
        if not url or url in seen:
            continue
        # Basic validation: must have a scheme and network location
        parsed = urlparse(url)
        if parsed.scheme in ("http", "https") and parsed.netloc:
            seen.add(url)
            urls.append(url)
    return urls


_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# HTTP status codes that prove the server is up but won't serve the content
# to automated clients — the URL itself is reachable.
_REACHABLE_BLOCKED_CODES = {429, 403, 401}


def _is_url_reachable(url: str) -> bool:
    """Return True if the URL responds with a non-error HTTP status.

    429/403/401 are treated as reachable: they prove the server is online and
    the URL resolves — the server is simply rate-limiting or gating access.
    """
    try:
        resp = requests.head(
            url, timeout=_REQUEST_TIMEOUT, allow_redirects=True, headers=_BROWSER_HEADERS
        )
        if resp.status_code < 400 or resp.status_code in _REACHABLE_BLOCKED_CODES:
            return True
        # Some servers reject HEAD; fall back to GET with minimal download.
        with requests.get(
            url, timeout=_REQUEST_TIMEOUT, allow_redirects=True,
            stream=True, headers=_BROWSER_HEADERS
        ) as resp:
            return resp.status_code < 400 or resp.status_code in _REACHABLE_BLOCKED_CODES
    except requests.RequestException:
        return False


def open_search_urls_reachable_pct_grader(
    input: Any,  # noqa: A002
    output: Any,
    expected_output: Any,
    metadata: dict[str, Any] | None = None,
    **kwargs: Any,
) -> list[Evaluation]:
    """Check reachability of every URL cited in the agent's report.

    Returns
    -------
    list[Evaluation]
        Single evaluation with value = reachable / total (3 decimal places).
    """
    del expected_output, metadata, kwargs

    if not _web_search_was_called(output):
        return [
            Evaluation(
                name=_REACHABLE_METRIC,
                value=0.0,
                comment="web_search tool was not called — score is 0.",
            )
        ]

    urls = _extract_citation_urls(output)

    if not urls:
        return [
            Evaluation(
                name=_REACHABLE_METRIC,
                value=1.0,
                comment="No web-search URLs found in citations — nothing to check.",
            )
        ]

    results: dict[str, bool] = {}
    for url in urls:
        results[url] = _is_url_reachable(url)

    reachable = sum(1 for v in results.values() if v)
    total = len(results)
    pct = round(reachable / total, 3)

    unreachable = [u for u, ok in results.items() if not ok]
    comment = f"{reachable}/{total} URLs reachable."
    if unreachable:
        comment += f" Unreachable: {unreachable}"

    return [
        Evaluation(
            name=_REACHABLE_METRIC,
            value=pct,
            comment=comment,
            metadata={"reachable": reachable, "total": total, "details": results},
        )
    ]


# ---------------------------------------------------------------------------
# open_search_results_aml_relevance — LLM-as-a-judge
# ---------------------------------------------------------------------------

_RELEVANCE_METRIC = "open_search_results_relevance_llm"

_RELEVANCE_SYSTEM_PROMPT = """\
You are an expert AML (Anti-Money Laundering) compliance analyst.  Your task \
is to judge whether each information point in the agent's open-search / \
web-search findings is relevant to AML risk assessment or enhanced due \
diligence (EDD).

Relevance criteria — an information point is RELEVANT if it relates to ANY \
of the following:
- Sanctions or watchlist status (OFAC, UN, EU, local sanctions lists)
- Politically Exposed Person (PEP) status
- Adverse media coverage (financial crime, fraud, corruption, terrorism)
- Criminal investigations, indictments, or convictions
- Corporate ownership, beneficial ownership, or shell-company indicators
- Geographic risk factors (high-risk jurisdictions)
- Source-of-wealth or source-of-funds concerns
- Known associates with adverse records
- Regulatory enforcement actions or fines
- Online presence indicators relevant to identity verification or risk \
(e.g. social-media profiles, professional directories, domain registrations, \
digital footprint that corroborates or contradicts the subject's stated identity)
- KYC (Know Your Customer) details that help establish or verify identity, \
occupation, address, date of birth, nationality, or document authenticity
- Any other factor a compliance officer would consider material when \
assessing money-laundering or terrorist-financing risk

An information point is NOT RELEVANT if it is:
- General business news unrelated to compliance risk
- Marketing or promotional content
- Entertainment or lifestyle content
- Unrelated to the investigation subject

Instructions:
1. Enumerate every distinct information point found in the open-search \
results section of the report.
2. For each point, determine whether it is RELEVANT or NOT_RELEVANT.
3. Count the totals and compute the relevance ratio.

Respond with valid JSON only (no markdown fences).  Schema:
{
  "points": [
    {
      "description": "Brief description of the information point",
      "relevant": true | false,
      "reason": "Why this is or is not relevant to AML/EDD"
    }
  ],
  "relevant_count": <int>,
  "total_count": <int>,
  "relevance_ratio": <float rounded to 3 decimal places>
}
"""

_RELEVANCE_USER_PROMPT_TEMPLATE = """\
# Subject Under Investigation
{subject}

# Agent's Open-Search / Web-Search Findings
{web_search_findings}

# Agent's Full Report Citations (Web Sources)
{citations}
"""


def _extract_web_search_section(output: Any) -> str:
    """Extract the open-search / web-search section from the report."""
    if not isinstance(output, dict):
        return ""
    report = output.get("report")
    if not isinstance(report, dict):
        return ""

    # Look for the external/web search section in the parsed report sections
    sections = report.get("sections")
    if isinstance(sections, list):
        for sec in sections:
            if not isinstance(sec, dict):
                continue
            heading = (sec.get("heading") or "").lower()
            if any(kw in heading for kw in ("external", "open", "web", "search", "due diligence")):
                return sec.get("body", "")

    # Fallback: extract from raw markdown
    md = output.get("report_markdown", "")
    if not md:
        return ""
    # Find the section between a matching heading and the next heading.
    # Use [^\n]* to restrict the heading match to a single line (re.DOTALL
    # would let .* cross newlines and match keywords in body text instead).
    pattern = r"(?:^|\n)#+\s+[^\n]*(?:external|open|web|search|due\s+diligence)[^\n]*\n(.*?)(?=\n#+\s+|\Z)"
    match = re.search(pattern, md, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else ""


def _extract_web_citations(output: Any) -> str:
    """Format web-source citations as text for the LLM judge."""
    if not isinstance(output, dict):
        return "(none)"
    report = output.get("report")
    if not isinstance(report, dict):
        return "(none)"
    citations = report.get("citations")
    if not isinstance(citations, list):
        return "(none)"

    web_cites: list[str] = []
    for cite in citations:
        if not isinstance(cite, dict):
            continue
        url = cite.get("url")
        if not url:
            continue  # skip non-web citations (KB, SQL)
        title = cite.get("title", "")
        excerpt = cite.get("excerpt", "")
        web_cites.append(f"- [{cite.get('num', '?')}] {title} | {url} | {excerpt}")

    return "\n".join(web_cites) if web_cites else "(none)"


def _extract_subject(output: Any) -> str:
    """Get the investigation subject from the output."""
    if isinstance(output, dict):
        report = output.get("report")
        if isinstance(report, dict):
            return report.get("subject", "")
        return output.get("subject", "")
    return ""


def open_search_results_relevance_llm_grader(
    input: Any,  # noqa: A002
    output: Any,
    expected_output: Any,
    metadata: dict[str, Any] | None = None,
    **kwargs: Any,
) -> list[Evaluation]:
    """LLM-judged AML relevance of the agent's open-search results.

    Returns
    -------
    list[Evaluation]
        Single evaluation with value = relevant / total (3 decimal places).
    """
    del expected_output, metadata, kwargs

    if not _web_search_was_called(output):
        return [
            Evaluation(
                name=_RELEVANCE_METRIC,
                value=0.0,
                comment="web_search tool was not called — score is 0.",
            )
        ]

    web_section = _extract_web_search_section(output)
    web_citations = _extract_web_citations(output)
    subject = _extract_subject(output)

    if not web_section.strip() and web_citations == "(none)":
        return [
            Evaluation(
                name=_RELEVANCE_METRIC,
                value=1.0,
                comment="No open-search results found in report — nothing to judge.",
            )
        ]

    user_prompt = _RELEVANCE_USER_PROMPT_TEMPLATE.format(
        subject=subject or "(unknown)",
        web_search_findings=web_section or "(not found as a separate section)",
        citations=web_citations,
    )

    try:
        judge_response = run_llm_judge_structured(
            metric_name=_RELEVANCE_METRIC,
            system_prompt=_RELEVANCE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )

        relevant_count = int(judge_response.get("relevant_count", 0))
        total_count = int(judge_response.get("total_count", 0))

        if total_count == 0:
            pct = 1.0
            comment = "LLM judge found no information points to evaluate."
        else:
            # Clamp to [0.0, 1.0] in case the LLM returns inconsistent counts.
            pct = min(1.0, round(relevant_count / total_count, 3))
            comment = f"{relevant_count}/{total_count} information points are AML-relevant."

        points = judge_response.get("points", [])

        return [
            Evaluation(
                name=_RELEVANCE_METRIC,
                value=pct,
                comment=comment,
                metadata={
                    "relevant_count": relevant_count,
                    "total_count": total_count,
                    "points": points,
                },
            )
        ]

    except Exception as exc:
        logger.exception("LLM judge failed for %s", _RELEVANCE_METRIC)
        return [build_judge_error_evaluation(metric_name=_RELEVANCE_METRIC, error=exc)]


__all__ = [
    "open_search_urls_reachable_pct_grader",
    "open_search_results_relevance_llm_grader",
]
