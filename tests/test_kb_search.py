"""Tests for the Knowledge Base search tool (Weaviate backend)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, call, patch

import pytest

from aml_agent.config import WeaviateConfig
from aml_agent.tools.kb_search import KBSearchTool, _is_transient_error


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_obj(
    source_id: str,
    title: str,
    source: str,
    text: str,
    document_id: str | None = None,
    distance: float = 0.15,
) -> MagicMock:
    """Build a mock Weaviate result object."""
    obj = MagicMock()
    obj.properties = {
        "source_id": source_id,
        "title": title,
        "source": source,
        "text": text,
        "document_id": document_id or f"{source}|{source_id}",
    }
    obj.metadata.distance = distance
    return obj


_SAMPLE_OBJECTS = [
    _make_obj(
        "NK-test001", "JOHN DOE", "US OFAC Sanctions",
        "US OFAC Sanctions\nName: JOHN DOE\nType: Person\nNotes: Sanctioned for money laundering.",
        distance=0.05,
    ),
    _make_obj(
        "NK-test002", "JANE SMITH", "US FBI Most Wanted",
        "US FBI Most Wanted\nName: JANE SMITH\nType: Person\nNotes: Wanted for wire fraud and identity theft.",
        distance=0.15,
    ),
    _make_obj(
        "NK-test003", "ACME CORPORATION", "Canada Sanctions List",
        "Canada Sanctions List\nName: ACME CORPORATION\nType: Organization\nNotes: Sanctioned entity.",
        distance=0.20,
    ),
    _make_obj(
        "NK-test004", "BORIS PETROV", "US OFAC Sanctions",
        "US OFAC Sanctions\nName: BORIS PETROV\nType: Person\nNotes: Designated for cyber operations.",
        distance=0.10,
    ),
]


def _near_text_resp(objects: list) -> MagicMock:
    resp = MagicMock()
    resp.objects = objects
    return resp


def _bm25_resp(objects: list) -> MagicMock:
    resp = MagicMock()
    resp.objects = objects
    return resp


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_client():
    client = MagicMock()
    client.is_connected.return_value = True
    return client


@pytest.fixture
def kb_tool(mock_client):
    """KBSearchTool with a patched Weaviate connection, num_results=3."""
    cfg = WeaviateConfig(url="https://test.weaviate.cloud", api_key="dummy")
    with (
        patch("aml_agent.tools.kb_search.weaviate.connect_to_weaviate_cloud", return_value=mock_client),
        patch("aml_agent.tools.kb_search.time.sleep"),
    ):
        tool = KBSearchTool(weaviate_config=cfg, num_results=3)
        yield tool
        tool.close()


def _setup_collection(mock_client, near_text_objs, bm25_objs=None):
    """Attach a mock collection to the client and configure query returns."""
    collection = MagicMock()
    mock_client.collections.get.return_value = collection
    collection.query.near_text.return_value = _near_text_resp(near_text_objs)
    collection.query.bm25.return_value = _bm25_resp(bm25_objs or [])
    return collection


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestKBSearchTool:
    """Tests for KBSearchTool (Weaviate backend)."""

    def test_search_exact_name(self, kb_tool, mock_client):
        _setup_collection(mock_client, [_SAMPLE_OBJECTS[0]])
        result = kb_tool.search_knowledgebase("JOHN DOE")
        assert "NK-test001" in result
        assert "JOHN DOE" in result
        assert "OFAC" in result

    def test_search_no_match(self, kb_tool, mock_client):
        _setup_collection(mock_client, [], bm25_objs=[])
        result = kb_tool.search_knowledgebase("XYZNONEXISTENT12345")
        assert "No matches" in result

    def test_fallback_to_bm25_when_near_text_empty(self, kb_tool, mock_client):
        """Falls back to BM25 when near_text returns no objects."""
        _setup_collection(mock_client, near_text_objs=[], bm25_objs=[_SAMPLE_OBJECTS[1]])
        result = kb_tool.search_knowledgebase("wire fraud")
        assert "JANE SMITH" in result

    def test_search_returns_scores(self, kb_tool, mock_client):
        """near_text results include a Score line computed from distance."""
        _setup_collection(mock_client, [_SAMPLE_OBJECTS[0]])
        result = kb_tool.search_knowledgebase("JOHN DOE")
        assert "Score:" in result

    def test_search_returns_snippets(self, kb_tool, mock_client):
        _setup_collection(mock_client, [_SAMPLE_OBJECTS[0]])
        result = kb_tool.search_knowledgebase("JOHN DOE")
        assert "Snippet:" in result

    def test_bm25_results_have_no_score(self, kb_tool, mock_client):
        """BM25 fallback results do NOT include a Score line."""
        _setup_collection(mock_client, near_text_objs=[], bm25_objs=[_SAMPLE_OBJECTS[0]])
        result = kb_tool.search_knowledgebase("test")
        assert "Score:" not in result

    def test_get_entity_by_id(self, kb_tool, mock_client):
        collection = MagicMock()
        mock_client.collections.get.return_value = collection
        collection.query.bm25.return_value = _bm25_resp([_SAMPLE_OBJECTS[1]])
        result = kb_tool.get_entity_by_id("NK-test002")
        assert "JANE SMITH" in result
        assert "FBI Most Wanted" in result

    def test_get_entity_by_id_not_found(self, kb_tool, mock_client):
        collection = MagicMock()
        mock_client.collections.get.return_value = collection
        collection.query.bm25.return_value = _bm25_resp([])
        result = kb_tool.get_entity_by_id("NK-nonexistent")
        assert "No entity found" in result

    def test_get_entity_by_id_returns_json(self, kb_tool, mock_client):
        """get_entity_by_id returns JSON-parseable output."""
        collection = MagicMock()
        mock_client.collections.get.return_value = collection
        collection.query.bm25.return_value = _bm25_resp([_SAMPLE_OBJECTS[0]])
        result = kb_tool.get_entity_by_id("NK-test001")
        data = json.loads(result)
        assert data["title"] == "JOHN DOE"

    def test_weaviate_error_returns_graceful_message(self, kb_tool, mock_client):
        """Connection/query errors return a descriptive string, not an exception."""
        collection = MagicMock()
        mock_client.collections.get.return_value = collection
        collection.query.near_text.side_effect = Exception("Connection refused")
        result = kb_tool.search_knowledgebase("test")
        assert "unavailable" in result.lower()

    def test_close_disconnects_client(self, mock_client):
        """close() calls client.close() and sets _client to None."""
        cfg = WeaviateConfig(url="https://test.weaviate.cloud", api_key="dummy")
        collection = MagicMock()
        mock_client.collections.get.return_value = collection
        collection.query.near_text.return_value = _near_text_resp([_SAMPLE_OBJECTS[0]])

        with patch("aml_agent.tools.kb_search.weaviate.connect_to_weaviate_cloud", return_value=mock_client):
            tool = KBSearchTool(weaviate_config=cfg)
            tool.search_knowledgebase("test")  # triggers connect
            tool.close()

        mock_client.close.assert_called_once()
        assert tool._client is None


# ---------------------------------------------------------------------------
# Rate-limit detection and retry tests
# ---------------------------------------------------------------------------

class TestRateLimitHelper:
    """Unit tests for _is_transient_error."""

    @pytest.mark.parametrize("msg", [
        # Rate-limit patterns
        "rate limit exceeded",
        "Too Many Requests",
        "HTTP 429",
        "RESOURCE_EXHAUSTED",
        "quota exceeded",
        "throttled",
        "request limit reached",
        "rate exceeds threshold",
        # Connection failure patterns
        "Could not connect to Weaviate:Connection to Weaviate failed. Details: .",
        "connection failed",
        "connection refused",
        "failed to connect",
        "broken pipe",
    ])
    def test_detects_transient_patterns(self, msg):
        assert _is_transient_error(Exception(msg))

    def test_ignores_unrelated_errors(self):
        assert not _is_transient_error(Exception("invalid credentials"))
        assert not _is_transient_error(Exception("schema error: unknown property"))
        assert not _is_transient_error(Exception("object not found"))


class TestRetryBehaviour:
    """Verify that rate-limit errors trigger automatic retries with waits."""

    def test_succeeds_on_second_attempt(self, mock_client):
        """A transient error on the first call is retried and succeeds on the second."""
        cfg = WeaviateConfig(url="https://test.weaviate.cloud", api_key="dummy")
        collection = MagicMock()
        mock_client.collections.get.return_value = collection

        # Use the exact error string seen in production
        rate_err = Exception(
            "Could not connect to Weaviate:Connection to Weaviate failed. Details: ."
        )
        collection.query.near_text.side_effect = [
            rate_err,
            _near_text_resp([_SAMPLE_OBJECTS[0]]),
        ]

        with (
            patch("aml_agent.tools.kb_search.weaviate.connect_to_weaviate_cloud", return_value=mock_client),
            patch("aml_agent.tools.kb_search.time.sleep") as mock_sleep,
        ):
            tool = KBSearchTool(weaviate_config=cfg)
            result = tool.search_knowledgebase("JOHN DOE")

        assert "JOHN DOE" in result
        mock_sleep.assert_called_once_with(10)  # first retry delay

    def test_respects_delay_sequence(self, mock_client):
        """Three consecutive transient failures produce the correct delay sequence."""
        cfg = WeaviateConfig(url="https://test.weaviate.cloud", api_key="dummy")
        collection = MagicMock()
        mock_client.collections.get.return_value = collection

        rate_err = Exception("Could not connect to Weaviate:Connection to Weaviate failed. Details: .")
        collection.query.near_text.side_effect = [
            rate_err,  # attempt 1
            rate_err,  # attempt 2 (after 10s)
            rate_err,  # attempt 3 (after 30s)
            rate_err,  # attempt 4 (after 60s) — exhausts retries
        ]

        with (
            patch("aml_agent.tools.kb_search.weaviate.connect_to_weaviate_cloud", return_value=mock_client),
            patch("aml_agent.tools.kb_search.time.sleep") as mock_sleep,
        ):
            tool = KBSearchTool(weaviate_config=cfg)
            result = tool.search_knowledgebase("test")

        assert "unavailable" in result.lower()
        assert mock_sleep.call_args_list == [call(10), call(30), call(60)]

    def test_non_transient_error_not_retried(self, mock_client):
        """A non-transient error is NOT retried and produces 'unavailable'."""
        cfg = WeaviateConfig(url="https://test.weaviate.cloud", api_key="dummy")
        collection = MagicMock()
        mock_client.collections.get.return_value = collection
        collection.query.near_text.side_effect = Exception("invalid credentials")

        with (
            patch("aml_agent.tools.kb_search.weaviate.connect_to_weaviate_cloud", return_value=mock_client),
            patch("aml_agent.tools.kb_search.time.sleep") as mock_sleep,
        ):
            tool = KBSearchTool(weaviate_config=cfg)
            result = tool.search_knowledgebase("test")

        assert "unavailable" in result.lower()
        mock_sleep.assert_not_called()  # no retry for non-transient errors

