"""Internal knowledge base / watchlist search via Weaviate."""

from __future__ import annotations

import json
import logging
import re
import time

import weaviate
import weaviate.classes.query as wq
from weaviate.classes.init import Auth

from aml_agent.config import WeaviateConfig

logger = logging.getLogger(__name__)

__all__ = ["KBSearchTool"]

# Seconds to wait before each retry attempt (agent may call the tool again
# after the first refusal; these waits apply only to automatic internal retries
# within a single tool call).
_RETRY_DELAYS: tuple[int, ...] = (10, 30, 60)

# Patterns that indicate a transient error from Weaviate or the underlying
# gRPC/HTTP layer that is safe to retry (rate limits, throttling, connection
# hiccups).
_TRANSIENT_PATTERNS: tuple[str, ...] = (
    # Rate-limit / quota
    r"rate.?limit",
    r"too many requests",
    r"429",
    r"resource[_ ]exhausted",
    r"quota",
    r"throttl",
    r"request limit",
    r"rate exceed",
    # Transient connection failures
    r"could not connect",
    r"connection.*(failed|refused|reset|timeout)",
    r"failed to connect",
    r"connection to weaviate failed",
    r"unavailable",
    r"broken pipe",
)
_TRANSIENT_RE = re.compile(
    "|".join(_TRANSIENT_PATTERNS), re.IGNORECASE
)


def _is_transient_error(exc: Exception) -> bool:
    """Return True if *exc* looks like a transient Weaviate error worth retrying."""
    return bool(_TRANSIENT_RE.search(str(exc)))


def _call_with_retry(fn, *, label: str):
    """Call *fn()* and automatically retry on transient Weaviate errors.

    Retries up to ``len(_RETRY_DELAYS)`` times, sleeping ``_RETRY_DELAYS[i]``
    seconds before attempt *i+1*.  Raises the last exception if all attempts
    are exhausted.  Non-transient errors (e.g. bad credentials, schema errors)
    bubble up immediately without any sleep.
    """
    last_exc: Exception | None = None
    for attempt, delay in enumerate((-1,) + _RETRY_DELAYS):  # attempt 0 = first try
        if attempt > 0:
            logger.info(
                "[KB] Transient error for '%s'. Retrying in %ds (attempt %d/%d)…",
                label,
                delay,
                attempt,
                len(_RETRY_DELAYS),
            )
            time.sleep(delay)
        try:
            return fn()
        except Exception as exc:
            if _is_transient_error(exc):
                logger.warning(
                    "[KB] Transient error on attempt %d for '%s': %s",
                    attempt + 1,
                    label,
                    exc,
                )
                last_exc = exc
            else:
                raise  # non-transient errors bubble up immediately

    # All retries exhausted
    raise last_exc  # type: ignore[misc]


class KBSearchTool:
    """Knowledge base search tool backed by a Weaviate vector database.

    Parameters
    ----------
    weaviate_config : WeaviateConfig
        Connection settings for the Weaviate instance.
    num_results : int
        Maximum number of results to return per search.
    snippet_length : int
        Maximum character length of text snippets.
    max_distance : float
        Maximum vector distance (0–2 for cosine).  Results farther than this
        threshold are silently dropped to avoid returning irrelevant neighbours
        when the subject is not in the knowledge base.
    """

    def __init__(
        self,
        weaviate_config: WeaviateConfig,
        num_results: int = 5,
        snippet_length: int = 500,
        max_distance: float = 0.45,
    ) -> None:
        self._cfg = weaviate_config
        self.num_results = num_results
        self.snippet_length = snippet_length
        self.max_distance = max_distance
        self._client: weaviate.WeaviateClient | None = None

    def _connect(self) -> None:
        """Establish a connection to Weaviate Cloud if not already connected."""
        if self._client is not None:
            if self._client.is_connected():
                return
            # Previous connection went stale — close and reconnect.
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None
        if not self._cfg.url:
            raise RuntimeError(
                "WEAVIATE_URL is not set. Add it to your .env file."
            )
        auth = Auth.api_key(self._cfg.api_key) if self._cfg.api_key else None
        self._client = weaviate.connect_to_weaviate_cloud(
            cluster_url=self._cfg.url,
            auth_credentials=auth,
            skip_init_checks=True,
        )
        logger.debug("Connected to Weaviate Cloud at %s", self._cfg.url)

    def _format_result(self, obj: object, index: int, score: float | None) -> str:
        props = obj.properties  # type: ignore[attr-defined]
        doc_id: str = props.get("document_id", "")
        entity_id = (
            doc_id.split("|", 1)[1] if "|" in doc_id else props.get("source_id", doc_id)
        )
        name = props.get("title", "")
        source = props.get("source", "")
        snippet = str(props.get("text", ""))[: self.snippet_length]
        score_line = f"  Score: {score:.4f}\n" if score is not None else ""
        return (
            f"Result {index}:\n"
            f"  Entity ID: {entity_id}\n"
            f"  Name: {name}\n"
            f"  Source: {source}\n"
            f"{score_line}"
            f"  Snippet: {snippet}\n"
        )

    def search_knowledgebase(self, keyword: str) -> str:
        """Search the internal knowledge base for entities matching the keyword.

        Parameters
        ----------
        keyword : str
            The search query (name, alias, or description text).

        Returns
        -------
        str
            Formatted search results with entity details.
        """
        def _run() -> str:
            self._connect()
            assert self._client is not None
            collection = self._client.collections.get(self._cfg.collection_name)

            response = collection.query.near_text(
                query=keyword,
                limit=self.num_results,
                return_metadata=wq.MetadataQuery(distance=True),
            )

            # Filter vector results by distance threshold to avoid returning
            # irrelevant nearest neighbours when the subject is not in the KB.
            if response.objects:
                filtered = []
                for obj in response.objects:
                    dist = getattr(obj.metadata, "distance", None)
                    if dist is not None and dist > self.max_distance:
                        logger.debug(
                            "[KB] Dropping result (distance=%.4f > %.2f): %s",
                            dist,
                            self.max_distance,
                            obj.properties.get("title", ""),
                        )
                        continue
                    filtered.append(obj)
                response.objects = filtered

            use_bm25 = not response.objects
            if use_bm25:
                response = collection.query.bm25(query=keyword, limit=self.num_results)

            if not response.objects:
                return f"No matches found in internal knowledge base for '{keyword}'."

            parts = [f"Knowledge Base Search Results for '{keyword}':\n"]
            for i, obj in enumerate(response.objects, 1):
                if use_bm25:
                    score = None
                else:
                    dist = getattr(obj.metadata, "distance", None)
                    score = (1.0 - dist) if dist is not None else None
                parts.append(self._format_result(obj, i, score))

            return "\n".join(parts)

        try:
            return _call_with_retry(_run, label=keyword)
        except Exception as exc:
            logger.warning("[KB] search unavailable for '%s': %s", keyword, exc)
            return f"Knowledge base search temporarily unavailable: {exc}"

    def get_entity_by_id(self, entity_id: str) -> str:
        """Look up a specific entity by its ID.

        Parameters
        ----------
        entity_id : str
            The entity ID to look up (source_id or the suffix of document_id).

        Returns
        -------
        str
            JSON-formatted entity properties, or a not-found message.
        """
        def _run() -> str:
            self._connect()
            assert self._client is not None
            collection = self._client.collections.get(self._cfg.collection_name)

            # Primary: filter by source_id
            response = collection.query.bm25(
                query=entity_id,
                limit=10,
                filters=wq.Filter.by_property("source_id").equal(entity_id),
            )
            obj = response.objects[0] if response.objects else None

            if obj is None:
                # Fallback: broad search, filter client-side on document_id suffix
                response = collection.query.bm25(query=entity_id, limit=10)
                for candidate in response.objects:
                    props = candidate.properties
                    if (
                        str(props.get("document_id", "")).endswith(f"|{entity_id}")
                        or props.get("source_id") == entity_id
                    ):
                        obj = candidate
                        break

            if obj is not None:
                return json.dumps(obj.properties, indent=2, ensure_ascii=False)

            return f"No entity found with ID '{entity_id}'."

        try:
            return _call_with_retry(_run, label=entity_id)
        except Exception as exc:
            logger.warning("[KB] lookup unavailable for '%s': %s", entity_id, exc)
            return f"Knowledge base lookup temporarily unavailable: {exc}"

    def close(self) -> None:
        """Close the Weaviate connection."""
        if self._client is not None:
            self._client.close()
            self._client = None
