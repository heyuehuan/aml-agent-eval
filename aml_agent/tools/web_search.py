"""External web search tool using Gemini Grounded Search.

Uses Google's Gemini API with grounding (Google Search) to perform
real-time due diligence searches for AML investigations.
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.request
import urllib.error

from google import genai
from google.genai import types
from openinference.semconv.trace import (
    DocumentAttributes,
    MessageAttributes,
    SpanAttributes,
    OpenInferenceSpanKindValues,
)
from opentelemetry import trace as _otel_trace

logger = logging.getLogger(__name__)


def _fetch_title_and_url(url: str, timeout: float = 5.0) -> tuple[str, str]:
    """Follow redirects, resolve the final URL, and best-effort fetch the page <title>.

    Makes a single GET request so redirect resolution and title extraction happen
    in one round-trip.  Only reads the first 8 KB of the response body (sufficient
    for the <title> tag in virtually all pages).  Falls back gracefully on any
    network or parse error.

    Parameters
    ----------
    url : str
        Possibly-redirecting source URL (e.g. a Vertex AI grounding redirect).
    timeout : float
        Per-request timeout in seconds.

    Returns
    -------
    tuple[str, str]
        ``(page_title, resolved_url)``.  Either value may be an empty string / the
        original URL if resolution or title extraction failed.
    """
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (AML-Agent/1.0; URL-resolver)"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            resolved_url: str = resp.url
            content_type: str = resp.headers.get("Content-Type", "")
            page_title = ""
            if "html" in content_type:
                raw = resp.read(8192).decode("utf-8", errors="replace")
                m = re.search(r"<title[^>]*>([^<]{1,200})</title>", raw, re.IGNORECASE)
                if m:
                    page_title = re.sub(r"\s+", " ", m.group(1)).strip()
            return page_title, resolved_url
    except Exception as exc:
        logger.debug("_fetch_title_and_url failed for %s: %s", url, exc)
        return "", url


def _is_domain_only(title: str) -> bool:
    """Return True if *title* looks like a bare hostname rather than a page title."""
    # e.g. "chaincatcher.com", "www.reuters.com" — no spaces, has a dot
    return bool(title) and " " not in title.strip() and "." in title


def _clean_excerpt(text: str) -> str:
    """Strip raw markdown artefacts from a grounding excerpt for clean citation display."""
    text = text.strip()
    # Remove leading bullet/list markers (* - •)
    text = re.sub(r"^[*\-•]\s+", "", text)
    # Remove leading **Section name:** markers (e.g. **Negative News:**)
    text = re.sub(r"^\*\*[^*]+\*\*:\s*", "", text)
    # Strip remaining bold/italic markdown
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    # Collapse literal escape sequences and excessive whitespace
    text = text.replace("\\n", " ").replace("\\t", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text[:280]

__all__ = ["WebSearchTool"]

_SEARCH_INSTRUCTION = """\
Conduct enhanced due diligence on "{search_input}". Your goals are to identify and summarize:

1. Detailed biographical information
   - Full legal name, all known aliases, transliterations, and name variants
   - Date and place of birth, nationality, citizenship(s)
   - Passport or national ID numbers if publicly reported
   - Last known addresses or countries of residence
   - Professional background, career history, and current roles

2. Financial crime exposure
   - Any known allegations, ongoing investigations, indictments, or convictions
   - AML / money laundering, fraud, bribery, corruption, tax evasion, insider trading, embezzlement
   - Terrorist financing or proliferation financing links
   - Regulatory fines, enforcement actions, or asset freezes imposed by any authority

3. Corporate ownership and business interests
   - Directorships, shareholdings, beneficial ownership of companies
   - Key subsidiaries, parent companies, joint ventures
   - Shell companies, offshore entities, or trusts linked to the subject
   - Business relationships with sanctioned entities or high-risk jurisdictions

4. Regulatory and person-of-interest status
   - Sanctions listings: OFAC SDN, UN, EU, UK, Canada, or other national sanctions lists
   - Politically Exposed Person (PEP) status — current or former government, judicial, military, or senior party roles; family members or close associates holding such roles
   - Law enforcement interest: Interpol notices, FBI/RCMP most-wanted, arrest warrants, extradition proceedings
   - Export control, debarment, or procurement exclusion lists

5. Related parties
   - Immediate family members and their own public profiles or risk flags
   - Known business partners, co-directors, co-defendants, or co-signatories
   - Advisors, intermediaries, or nominees linked to the subject
   - Organizations or individuals sharing addresses, phone numbers, or corporate registrations

6. Negative news and adverse media
   - Recent controversies, scandals, or reputational incidents from reputable outlets
   - Cross-border enforcement actions or international cooperation requests
   - Civil litigation, arbitration, or bankruptcy proceedings
   - Any reporting that contradicts official statements or raises integrity concerns

The provided name or entity may contain errors — find the most accurate match and correct the input if necessary.
If multiple candidates are found, explain which is the most likely match and justify your reasoning.
Use reputable, up-to-date public sources, and present the results in a structured and concise way.

For EACH finding, clearly state:
- The exact source title (webpage title)
- The direct URL (not a redirect link)
- A concise relevant excerpt

Format your citations clearly so they can be extracted programmatically.
"""


class WebSearchTool:
    """Web search tool using Gemini Grounded Search for AML due diligence.

    Parameters
    ----------
    api_key : str | None
        Google API key. If None, reads from GOOGLE_API_KEY env var.
    model_name : str
        Gemini model to use for grounded search.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = "gemini-2.5-flash",
    ) -> None:
        self._api_key = api_key or os.getenv("GOOGLE_API_KEY", os.getenv("GEMINI_API_KEY", ""))
        self._model_name = model_name
        self._client: genai.Client | None = None

    def _get_client(self) -> genai.Client:
        """Return a reusable genai Client (created once per tool instance)."""
        if self._client is None:
            self._client = genai.Client(api_key=self._api_key)
        return self._client

    def web_search(self, query: str) -> str:
        """Perform a grounded web search for external due diligence on a subject.

        Call this tool up to 3 times with different focused queries targeting
        distinct information needs (e.g. sanctions status, adverse media,
        corporate connections).  Each call should use a different query.

        Pass an enriched query with disambiguation details from the internal
        knowledge base — e.g. aliases, date of birth, nationality, or
        sanctions program:  ``\"Daniel He\" \"He Yi\" OFAC SDN 1965 China``

        NEVER include internal entity IDs (e.g. NK-MQvFt....)
        in the query — they are meaningless to search engines.

        Parameters
        ----------
        query : str
            Focused search string. Use names, aliases, and AML-relevant
            keywords. Do not include internal KB entity IDs.

        Returns
        -------
        str
            Formatted search results with grounding sources.
        """
        if not self._api_key:
            return "Error: GOOGLE_API_KEY environment variable is required for web search."

        # Safety net: strip internal KB entity IDs (e.g. NK-MQvFtRqy8nxGpsmtZLrfG4)
        # before sending to a public search engine.
        clean_query = re.sub(r"\bNK-[A-Za-z0-9]+\b", "", query).strip()
        clean_query = re.sub(r"\s{2,}", " ", clean_query)
        if clean_query != query:
            logger.info("web_search: stripped internal entity IDs from query. original=%r cleaned=%r", query, clean_query)
        query = clean_query
        logger.info("web_search: query=%r  model=%s", query, self._model_name)

        try:
            client = self._get_client()

            result_text, sources = self._single_search(client, query, attempt=1)

            # If grounding returned no sources, retry with a more targeted AML-specific query.
            # Preserve the original query (which may already contain disambiguation details
            # from KB findings) and append standard AML signal keywords.
            if not sources:
                logger.warning(
                    "web_search: NO grounding sources for query %r — retrying with refined query.",
                    query,
                )
                refined = (
                    f'{query} sanctions OR "money laundering" OR fraud OR PEP OR crime'
                )
                result_text, sources = self._single_search(client, refined, attempt=2)

            if sources:
                result_text += (
                    "\n\n---\n"
                    "CITABLE SOURCES (use ONLY these as web citations — do NOT cite the narrative above):\n"
                )
                for i, (title, url, excerpt) in enumerate(sources, 1):
                    logger.info("  [WEB-%d] %s => %s", i, title, url)
                    # Pipe separator avoids trailing-dot URL corruption
                    line = f"- {title} | {url}"
                    if excerpt:
                        line += f" | {excerpt}"
                    result_text += line + "\n"
            else:
                logger.warning(
                    "web_search: NO grounding sources after retry for query %r. "
                    "Model answered from parametric memory; no live URLs available.",
                    query,
                )
                # Append a clear sentinel so the agent doesn't write "no adverse media found".
                # The absence of grounding URLs is a search limitation, NOT a clean bill of health.
                result_text += (
                    "\n\n[SEARCH_INCONCLUSIVE] Google Search grounding returned no verifiable URLs "
                    "for this query after two attempts. This means the search engine could not "
                    "identify the subject with enough confidence to pin results to a specific person "
                    "— it does NOT mean the subject has no adverse history. "
                    "Report this as inconclusive rather than 'no results found'."
                )

            return result_text

        except Exception as e:
            logger.error("web_search: error for query %r: %s", query, e)
            return f"Web search error: {e}"

    def _single_search(
        self,
        client,
        query: str,
        attempt: int = 1,
    ) -> tuple[str, list[tuple[str, str, str]]]:
        """Run one grounded search call and return (result_text, sources).

        Extracted so that ``web_search`` can call it for the retry without
        duplicating the boilerplate.

        Creates an OpenTelemetry child span following the OpenInference
        semantic conventions so that Langfuse renders the correct input
        (query + full prompt), output (response text), grounding sources, and
        token usage.  When no OTel provider is configured (local mode) the
        span is a no-op NonRecordingSpan.
        """
        logger.info("web_search: attempt %d query=%r", attempt, query)
        grounding_tool = types.Tool(google_search=types.GoogleSearch())
        config = types.GenerateContentConfig(tools=[grounding_tool])
        instructions = _SEARCH_INSTRUCTION.format(search_input=query)

        _tracer = _otel_trace.get_tracer(__name__)
        with _tracer.start_as_current_span("web_search.generate_content") as span:
            # --- span kind: LLM (nested inside the ADK-created TOOL span) ---
            span.set_attribute(
                SpanAttributes.OPENINFERENCE_SPAN_KIND,
                OpenInferenceSpanKindValues.LLM.value,
            )

            # --- model identification (both OTel genai + OpenInference) ---
            span.set_attribute("gen_ai.system", "google_genai")
            span.set_attribute("gen_ai.operation.name", "chat")
            span.set_attribute("gen_ai.request.model", self._model_name)
            span.set_attribute(SpanAttributes.LLM_MODEL_NAME, self._model_name)
            span.set_attribute("web_search.attempt", attempt)

            # --- input: concise query value + full prompt as a chat message ---
            span.set_attribute(SpanAttributes.INPUT_VALUE, query)
            span.set_attribute(SpanAttributes.INPUT_MIME_TYPE, "text/plain")
            span.set_attribute(
                f"{SpanAttributes.LLM_INPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_ROLE}",
                "user",
            )
            span.set_attribute(
                f"{SpanAttributes.LLM_INPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_CONTENT}",
                instructions,
            )

            # --- call the model ---
            response = client.models.generate_content(
                model=self._model_name,
                contents=instructions,
                config=config,
            )

            result_text = response.text if response.text else "No results returned."
            sources = self._extract_grounding_sources(response)

            # --- output: raw model response text (mirrors how the ADK LLM spans work) ---
            span.set_attribute(
                f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_ROLE}",
                "model",
            )
            span.set_attribute(
                f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_CONTENT}",
                result_text,
            )
            span.set_attribute(SpanAttributes.OUTPUT_VALUE, result_text)
            span.set_attribute(SpanAttributes.OUTPUT_MIME_TYPE, "text/plain")

            # --- grounding sources as retrieval documents ---
            for i, (title, url, excerpt) in enumerate(sources):
                span.set_attribute(
                    f"{SpanAttributes.RETRIEVAL_DOCUMENTS}.{i}.{DocumentAttributes.DOCUMENT_CONTENT}",
                    excerpt or title,
                )
                span.set_attribute(
                    f"{SpanAttributes.RETRIEVAL_DOCUMENTS}.{i}.{DocumentAttributes.DOCUMENT_METADATA}",
                    json.dumps({"title": title, "url": url}, ensure_ascii=False),
                )

            # --- token usage (OpenInference + OTel genai conventions) ---
            um = getattr(response, "usage_metadata", None)
            if um:
                pt = getattr(um, "prompt_token_count", None)
                ct = getattr(um, "candidates_token_count", None)
                tt = getattr(um, "total_token_count", None)
                if pt is not None:
                    span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_PROMPT, pt)
                    span.set_attribute("gen_ai.usage.input_tokens", pt)
                if ct is not None:
                    span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION, ct)
                    span.set_attribute("gen_ai.usage.output_tokens", ct)
                if tt is not None:
                    span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_TOTAL, tt)
                    span.set_attribute("gen_ai.usage.total_tokens", tt)

        logger.info(
            "web_search: attempt %d response (%d chars). First 200: %s",
            attempt,
            len(result_text),
            result_text[:200].replace("\n", " "),
        )
        logger.info("web_search: attempt %d grounding sources: %d", attempt, len(sources))
        return result_text, sources

    def _extract_grounding_sources(self, response) -> list[tuple[str, str, str]]:
        """Extract grounding sources from a Gemini response.

        Returns a list of ``(title, resolved_url, excerpt)`` tuples — one per
        distinct webpage.  Redirect URLs (e.g. Vertex AI grounding redirects)
        are resolved to their final destination automatically.
        """
        # Build a mapping: chunk_index → (title, raw_url)
        chunk_map: dict[int, tuple[str, str]] = {}
        try:
            if not response.candidates:
                return []

            candidate = response.candidates[0]
            grounding_metadata = getattr(candidate, "grounding_metadata", None)
            if not grounding_metadata:
                logger.warning(
                    "_extract_grounding_sources: no grounding_metadata on candidate. "
                    "finish_reason=%s",
                    getattr(candidate, "finish_reason", "unknown"),
                )
                return []

            chunks = getattr(grounding_metadata, "grounding_chunks", None) or []
            logger.info(
                "_extract_grounding_sources: %d grounding chunk(s) found", len(chunks)
            )
            for idx, chunk in enumerate(chunks):
                web = getattr(chunk, "web", None)
                if web:
                    title = getattr(web, "title", "") or ""
                    uri = getattr(web, "uri", "") or ""
                    if uri:
                        chunk_map[idx] = (title, uri)

            if not chunk_map:
                logger.warning("_extract_grounding_sources: chunks present but none had a web URI")
                return []

            # Build per-chunk excerpt from grounding_supports
            chunk_excerpts: dict[int, list[str]] = {i: [] for i in chunk_map}
            supports = getattr(grounding_metadata, "grounding_supports", None) or []
            for support in supports:
                segment = getattr(support, "segment", None)
                indices = getattr(support, "grounding_chunk_indices", []) or []
                if segment and indices:
                    text = (getattr(segment, "text", "") or "").strip()
                    if len(text) > 10:
                        for idx in indices:
                            if idx in chunk_excerpts:
                                chunk_excerpts[idx].append(text)

            # Resolve URLs, fetch page titles, and assemble results — deduplicate by resolved URL
            seen_urls: set[str] = set()
            results: list[tuple[str, str, str]] = []
            for idx, (api_title, raw_url) in chunk_map.items():
                fetched_title, resolved = _fetch_title_and_url(raw_url)
                if resolved in seen_urls:
                    continue
                seen_urls.add(resolved)
                # Prefer the live page title; fall back to the API title only if it's a real title
                if fetched_title:
                    title = fetched_title
                elif not _is_domain_only(api_title):
                    title = api_title
                else:
                    title = api_title  # domain-only fallback; better than nothing
                excerpts = chunk_excerpts.get(idx, [])
                # Take the longest excerpt, then clean it
                raw_excerpt = max(excerpts, key=len) if excerpts else ""
                excerpt = _clean_excerpt(raw_excerpt)
                results.append((title, resolved, excerpt))

            return results

        except Exception as e:
            logger.warning("_extract_grounding_sources: failed to parse grounding metadata: %s", e)
            return []
