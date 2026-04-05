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

``web_search_query_quality_rule``
    Rule-based check of every ``web_search`` query issued during the agent
    run.  Penalises stop-word-heavy queries, vague terms, overly short or
    long queries, and absent cited sources; rewards named-entity presence
    and government-domain sources.  Scores are averaged across all queries
    and normalised from 0–5 to 0.0–1.0.

``web_search_query_quality_llm``
    LLM-judged AML focus and precision of each ``web_search`` query,
    averaged across all queries and normalised from 1–5 to 0.0–1.0.

``web_search_source_relevancy_llm``
    LLM-judged credibility and AML usefulness of the sources cited by
    each ``web_search`` call, averaged and normalised to 0.0–1.0.

``web_search_recall_llm``
    LLM-judged fraction of ground-truth expected findings
    (``expected_open_search_results``) that the agent's web searches
    collectively covered.  Null when no ground truth is provided.

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
    LLMJudgeConfig,
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
            config=LLMJudgeConfig(max_output_tokens=8192),
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


# ---------------------------------------------------------------------------
# Shared helpers for per-query graders
# ---------------------------------------------------------------------------

_STOP_WORDS = {
    "the", "a", "an", "of", "and", "or", "is", "in", "to", "for",
    "what", "how", "why", "do", "does", "with", "on", "at", "by",
}
_VAGUE_TERMS = {
    "information", "details", "stuff", "things", "data",
    "general", "overview", "about", "related", "various",
}


def _parse_cited_sources(response_text: str) -> list[dict]:
    """Extract {title, url, snippet} dicts from the CITABLE SOURCES block."""
    sources: list[dict] = []
    block_match = re.search(
        r"CITABLE SOURCES.*?(?=\Z)", response_text, re.DOTALL | re.IGNORECASE
    )
    if not block_match:
        return sources
    for line in block_match.group().splitlines():
        line = line.strip().lstrip("- ").strip()
        if "|" not in line:
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 2:
            title = parts[0] if parts[0] not in ("", "&nbsp;") else ""
            url = parts[1]
            snippet = parts[2] if len(parts) > 2 else ""
            if url.startswith("http"):
                sources.append({"title": title, "url": url, "snippet": snippet})
    return sources


def _extract_reasoning_by_query(output: Any) -> dict[str, str]:
    """Map each web_search query to the agent reasoning thought that preceded it."""
    reasoning: dict[str, str] = {}
    if not isinstance(output, dict):
        return reasoning
    llm_section = output.get("llm_call_history", {})
    calls = llm_section.get("calls", []) if isinstance(llm_section, dict) else []
    for call in calls:
        parts = call.get("response", {}).get("content", {}).get("parts", [])
        thought_text: str | None = None
        for part in parts:
            if part.get("thought") is True and part.get("type") == "text":
                thought_text = part.get("text", "").strip()
            elif part.get("type") == "function_call" and part.get("name") == "web_search":
                query = part.get("args", {}).get("query", "")
                if query and thought_text:
                    reasoning[query] = thought_text
                thought_text = None
    return reasoning


def _extract_search_events(output: Any) -> list[dict]:
    """Return one record per web_search tool call from the agent artifacts dict."""
    if not isinstance(output, dict):
        return []
    reasoning_by_query = _extract_reasoning_by_query(output)
    report = output.get("report")
    subject = report.get("subject", "") if isinstance(report, dict) else ""
    events: list[dict] = []
    for call in (output.get("tool_calls") or []):
        if not isinstance(call, dict) or call.get("tool") != "web_search":
            continue
        query = (call.get("args") or {}).get("query", "").strip()
        if not query:
            continue
        response_text = call.get("response", "")
        if not isinstance(response_text, str):
            response_text = ""
        events.append({
            "subject": subject,
            "query": query,
            "response_text": response_text,
            "cited_sources": _parse_cited_sources(response_text),
            "agent_reasoning": reasoning_by_query.get(query, ""),
        })
    return events


def _rule_eval_query(query: str, cited_sources: list[dict]) -> dict:
    """Compute rule-based quality metrics for one web_search query.

    Returns a dict with ``rule_score`` (0.0–5.0) and ``flags``.
    """
    tokens = query.lower().split()
    n = len(tokens)
    stop_ratio = sum(1 for t in tokens if t in _STOP_WORDS) / max(n, 1)
    has_vague = any(t in _VAGUE_TERMS for t in tokens)
    has_entity = bool(re.search(r'(?<=\s)[A-Z][a-zA-Z]+', query))
    too_short = n < 2
    too_long = n > 12
    no_sources = len(cited_sources) == 0
    has_gov_source = any(
        re.search(r'\.(gov|justice\.gov|treasury\.gov|ofac)', s.get("url", ""), re.I)
        for s in cited_sources
    )

    flags: list[str] = []
    if stop_ratio > 0.4:
        flags.append("high_stop_word_ratio")
    if has_vague:
        flags.append("vague_terms")
    if too_short:
        flags.append("search_query_too_short")
    if too_long:
        flags.append("search_query_too_long")
    if no_sources:
        flags.append("no_cited_sources")

    score = 5.0
    score -= 1.25 * (stop_ratio > 0.4)
    score -= 1.00 * has_vague
    score -= 1.25 * too_short
    score -= 0.50 * too_long
    score += 0.50 * has_entity
    score -= 1.00 * no_sources
    score += 0.25 * has_gov_source
    score = round(max(0.0, min(5.0, score)), 3)
    return {"rule_score": score, "flags": flags}


def _format_sources_block(sources: list[dict]) -> str:
    """Format cited sources into a readable block for the LLM judge prompt."""
    if not sources:
        return "No cited sources extracted."
    lines: list[str] = []
    for i, s in enumerate(sources, 1):
        title = s.get("title") or "(no title)"
        lines.append(f"{i}. {title}")
        lines.append(f"   URL: {s.get('url', '')}")
        snippet = s.get("snippet", "")
        if snippet:
            lines.append(f"   {snippet[:200]}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# web_search_query_quality_rule — rule-based
# ---------------------------------------------------------------------------

_RULE_METRIC = "web_search_query_quality_rule"


def web_search_query_quality_rule_grader(
    input: Any,  # noqa: A002
    output: Any,
    expected_output: Any,
    metadata: dict[str, Any] | None = None,
    **kwargs: Any,
) -> list[Evaluation]:
    """Rule-based quality check for every web_search query in the agent run.

    Penalises stop-word-heavy or vague queries, overly short/long queries,
    and absent cited sources; rewards named-entity presence and government-
    domain sources.  Scores are averaged across all queries and normalised
    from 0–5 to 0.0–1.0.

    Returns
    -------
    list[Evaluation]
        Single evaluation named ``web_search_query_quality_rule``.
    """
    del expected_output, metadata, kwargs

    if not _web_search_was_called(output):
        return [
            Evaluation(
                name=_RULE_METRIC,
                value=0.0,
                comment="web_search tool was not called — score is 0.",
            )
        ]

    events = _extract_search_events(output)
    if not events:
        return [
            Evaluation(
                name=_RULE_METRIC,
                value=1.0,
                comment="No web_search queries found to evaluate.",
            )
        ]

    per_query = [_rule_eval_query(ev["query"], ev["cited_sources"]) for ev in events]
    avg_score = sum(r["rule_score"] for r in per_query) / len(per_query)
    value = round(avg_score / 5.0, 3)

    flag_counts: dict[str, int] = {}
    for r in per_query:
        for f in r["flags"]:
            flag_counts[f] = flag_counts.get(f, 0) + 1

    comment = f"{len(events)} queries evaluated. Avg rule score: {avg_score:.3f}/5.0."
    if flag_counts:
        comment += f" Flags: {flag_counts}"

    return [
        Evaluation(
            name=_RULE_METRIC,
            value=value,
            comment=comment,
            metadata={
                "query_count": len(events),
                "avg_raw_score": round(avg_score, 3),
                "flag_counts": flag_counts,
                "per_query": [
                    {"query": ev["query"], **r}
                    for ev, r in zip(events, per_query)
                ],
            },
        )
    ]


# ---------------------------------------------------------------------------
# web_search_query_quality_llm / web_search_source_relevancy_llm /
# web_search_recall_llm — LLM-judged, per-call aggregated
# ---------------------------------------------------------------------------

_QUERY_QUALITY_METRIC = "web_search_query_quality_llm"
_SOURCE_RELEVANCY_METRIC2 = "web_search_source_relevancy_llm"
_RECALL_METRIC = "web_search_recall_llm"

_QUERY_JUDGE_SYSTEM_PROMPT = """\
You are an expert evaluator of agentic web search behaviour for an \
Anti-Money Laundering (AML) due diligence tool.

The agent investigates entities for AML risk. It uses web search to find \
adverse media, sanctions exposure, PEP status, financial crime records etc.

Evaluate three dimensions and return ONLY valid JSON — no markdown fences, \
no extra keys.

1. query_quality (1-5)
   Does the query efficiently and specifically target AML-relevant information?
   Evaluate based on how the query was constructed, not solely on whether
   expected findings were returned — a well-formed query may still miss findings
   due to limitations of the search tool.
   5 = precise, well-formed — a skilled AML analyst would write this
   4 = good, minor improvements possible
   3 = reasonable but too broad or missing a key discriminating term
   2 = somewhat relevant but likely to retrieve noise
   1 = vague, off-topic, or not useful for AML purposes

2. source_relevancy (1-5)
   Do the cited sources appear credible and directly useful for AML due diligence?
   5 = highly relevant, authoritative (gov, regulators, reputable news)
   4 = mostly relevant with minor noise
   3 = mixed quality or only tangentially related
   2 = mostly irrelevant or low-credibility
   1 = no sources, entirely irrelevant, or misleading

3. recall (only if expected findings are provided, otherwise null)
   What fraction of the expected findings does the search response cover,
   even if paraphrased, substring matched, or semantically equivalent?

{
  "query_quality_score": <int 1-5>,
  "query_quality_rationale": "<one concise sentence>",
  "source_relevancy_score": <int 1-5>,
  "source_relevancy_rationale": "<one concise sentence>",
  "recall_score": <float 0.0-1.0, or null if no expected findings provided>,
  "per_finding": [
    {
      "expected": "<the expected finding text>",
      "covered": <true or false>,
      "evidence": "<one short sentence referencing the part of the response that covers it, or 'Not found' if absent>"
    }
  ]
}
"""

_QUERY_JUDGE_USER_TEMPLATE = """\
## Request for investigation
{subject}

## Agent's reasoning before issuing this search
{agent_reasoning}

## Search query issued
{query}

## Cited sources extracted from the search response
{sources_block}

## Web search response
{response_preview}

## Expected findings (ground truth)
These are the facts this search should ideally have surfaced. Use them to \
inform your evaluation of query quality and source relevancy, and assess \
coverage directly in the recall dimension.
{expected_findings_block}
"""


def _run_query_llm_judge(event: dict, expected_findings: list[str]) -> dict | None:
    """Run the combined LLM judge for one search event.

    Returns the parsed judge result dict, or None on unrecoverable failure.
    """
    web_search_response = event["response_text"][:6000].strip()
    if len(event["response_text"]) > 6000:
        web_search_response += "\n... [truncated]"

    reasoning = event["agent_reasoning"][:800].strip()
    if len(event["agent_reasoning"]) > 800:
        reasoning += "\n... [truncated]"

    findings_block = (
        "\n".join(f"{i + 1}. {f}" for i, f in enumerate(expected_findings))
        if expected_findings
        else "No ground truth provided for this subject."
    )

    user_prompt = _QUERY_JUDGE_USER_TEMPLATE.format(
        subject=event["subject"] or "(unknown)",
        agent_reasoning=reasoning or "Not available.",
        query=event["query"],
        sources_block=_format_sources_block(event["cited_sources"]),
        response_preview=web_search_response,
        expected_findings_block=findings_block,
    )

    try:
        result = run_llm_judge_structured(
            metric_name=_QUERY_QUALITY_METRIC,
            system_prompt=_QUERY_JUDGE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            config=LLMJudgeConfig(max_output_tokens=4096),
        )
        if "per_finding" not in result:
            result["per_finding"] = []
        # Enforce null recall when no ground truth was supplied
        if not expected_findings:
            result["recall_score"] = None
            result["per_finding"] = []
        return result
    except Exception as exc:
        logger.warning("LLM judge failed for query %r: %s", event["query"], exc)
        return None


def web_search_query_quality_llm_grader(
    input: Any,  # noqa: A002
    output: Any,
    expected_output: Any,
    metadata: dict[str, Any] | None = None,
    **kwargs: Any,
) -> list[Evaluation]:
    """LLM-judged quality of every web_search query in the agent run.

    Evaluates three dimensions per query and averages across all calls:

    - ``web_search_query_quality_llm``: AML focus and precision (1–5 → 0.0–1.0).
    - ``web_search_source_relevancy_llm``: credibility of returned sources
      (1–5 → 0.0–1.0).
    - ``web_search_recall_llm``: fraction of ground-truth expected findings
      (``expected_open_search_results``) covered (0.0–1.0).  Omitted when no
      ground truth is available. (Score default to 1.0)

    Returns
    -------
    list[Evaluation]
        Two or three Evaluations depending on ground-truth availability.
    """
    del metadata, kwargs

    # Extracted early so every exit path can emit recall when GT is present.
    expected_findings: list[str] = []
    if isinstance(expected_output, dict):
        ef = expected_output.get("expected_open_search_results") or []
        if isinstance(ef, list):
            expected_findings = [str(f) for f in ef if f]

    def _recall_zero(comment: str) -> list[Evaluation]:
        """Return recall=0 when the agent produced no searchable output."""
        if expected_findings:
            return [Evaluation(name=_RECALL_METRIC, value=0.0, comment=comment)]
        return [Evaluation(name=_RECALL_METRIC, value=1.0, comment="not applicable - no ground truth is available")]

    if not _web_search_was_called(output):
        return [
            Evaluation(
                name=_QUERY_QUALITY_METRIC,
                value=0.0,
                comment="web_search tool was not called — score is 0.",
            ),
            Evaluation(
                name=_SOURCE_RELEVANCY_METRIC2,
                value=0.0,
                comment="web_search tool was not called — score is 0.",
            ),
            *_recall_zero("web_search tool was not called — recall is 0."),
        ]

    events = _extract_search_events(output)
    if not events:
        return [
            Evaluation(
                name=_QUERY_QUALITY_METRIC,
                value=1.0,
                comment="No web_search queries found to evaluate.",
            ),
            Evaluation(
                name=_SOURCE_RELEVANCY_METRIC2,
                value=1.0,
                comment="No web_search queries found to evaluate.",
            ),
            *_recall_zero("No web_search queries found — recall is 0."),
        ]

    judge_results: list[dict] = []
    for ev in events:
        result = _run_query_llm_judge(ev, expected_findings)
        if result is not None:
            judge_results.append(result)

    if not judge_results:
        evals = [
            build_judge_error_evaluation(
                metric_name=_QUERY_QUALITY_METRIC,
                error=RuntimeError("All LLM judge calls failed"),
            ),
            build_judge_error_evaluation(
                metric_name=_SOURCE_RELEVANCY_METRIC2,
                error=RuntimeError("All LLM judge calls failed"),
            ),
        ]
        if expected_findings:
            evals.append(
                build_judge_error_evaluation(
                    metric_name=_RECALL_METRIC,
                    error=RuntimeError("All LLM judge calls failed"),
                )
            )
        else:
            evals.append(
                Evaluation(
                    name=_RECALL_METRIC,
                    value=1.0,
                    comment="not applicable - no ground truth is available",
                )
            )
        return evals

    evaluations: list[Evaluation] = []

    # query_quality — 1-5 scale, normalised to 0.0-1.0
    qq_scores = [
        r["query_quality_score"]
        for r in judge_results
        if r.get("query_quality_score") is not None
    ]
    if qq_scores:
        avg_qq = sum(qq_scores) / len(qq_scores)
        evaluations.append(
            Evaluation(
                name=_QUERY_QUALITY_METRIC,
                value=round(avg_qq / 5.0, 3),
                comment=(
                    f"{len(qq_scores)}/{len(events)} queries judged. "
                    f"Avg quality: {avg_qq:.2f}/5."
                ),
                metadata={
                    "avg_raw_score": round(avg_qq, 3),
                    "per_query": [
                        {
                            "query": ev["query"],
                            "score": r.get("query_quality_score"),
                            "rationale": r.get("query_quality_rationale"),
                        }
                        for ev, r in zip(events, judge_results)
                    ],
                },
            )
        )
    else:
        evaluations.append(
            build_judge_error_evaluation(
                metric_name=_QUERY_QUALITY_METRIC,
                error=RuntimeError("No valid query quality scores"),
            )
        )

    # source_relevancy — 1-5 scale, normalised to 0.0-1.0
    sr_scores = [
        r["source_relevancy_score"]
        for r in judge_results
        if r.get("source_relevancy_score") is not None
    ]
    if sr_scores:
        avg_sr = sum(sr_scores) / len(sr_scores)
        evaluations.append(
            Evaluation(
                name=_SOURCE_RELEVANCY_METRIC2,
                value=round(avg_sr / 5.0, 3),
                comment=f"Avg source relevancy: {avg_sr:.2f}/5.",
                metadata={
                    "avg_raw_score": round(avg_sr, 3),
                    "per_query": [
                        {
                            "query": ev["query"],
                            "score": r.get("source_relevancy_score"),
                            "rationale": r.get("source_relevancy_rationale"),
                        }
                        for ev, r in zip(events, judge_results)
                    ],
                },
            )
        )
    else:
        evaluations.append(
            build_judge_error_evaluation(
                metric_name=_SOURCE_RELEVANCY_METRIC2,
                error=RuntimeError("No valid source relevancy scores"),
            )
        )

    # recall — 0.0-1.0, only emitted when ground truth is available
    recall_scores = [
        r["recall_score"]
        for r in judge_results
        if r.get("recall_score") is not None
    ]
    if recall_scores:
        avg_recall = sum(recall_scores) / len(recall_scores)
        all_per_finding = [
            pf for r in judge_results for pf in (r.get("per_finding") or [])
        ]
        evaluations.append(
            Evaluation(
                name=_RECALL_METRIC,
                value=round(avg_recall, 3),
                comment=(
                    f"Avg recall across {len(recall_scores)} "
                    f"queries with ground truth."
                ),
                metadata={
                    "avg_recall": round(avg_recall, 3),
                    "per_finding": all_per_finding,
                },
            )
        )
    elif expected_findings:
        evaluations.append(
            build_judge_error_evaluation(
                metric_name=_RECALL_METRIC,
                error=RuntimeError(
                    "No valid recall scores despite expected findings being provided"
                ),
            )
        )
    else:
        evaluations.append(
            Evaluation(
                name=_RECALL_METRIC,
                value=1.0,
                comment="not applicable - no ground truth is available",
            )
        )

    return evaluations


__all__ = [
    "open_search_urls_reachable_pct_grader",
    "open_search_results_relevance_llm_grader",
    "web_search_query_quality_rule_grader",
    "web_search_query_quality_llm_grader",
]
