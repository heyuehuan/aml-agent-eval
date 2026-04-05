"""Tests for data parsing and model validation."""

import json
import tempfile
from pathlib import Path

import pytest

from aml_agent.models import (
    Citation,
    InvestigationReport,
    KBMatch,
    SourceType,
    TransactionMatch,
    WebSearchResult,
)


class TestInvestigationReport:
    """Tests for InvestigationReport model."""

    def test_format_report(self):
        report = InvestigationReport(
            subject_name="JOHN DOE",
            summary="Subject has hits in OFAC [1] and adverse media [2].",
            kb_matches=[
                KBMatch(entity_id="NK-001", name="JOHN DOE", source="OFAC Sanctions", score=0.95)
            ],
            transaction_matches=[
                TransactionMatch(
                    transaction_id="tx-001",
                    amount=100000.00,
                    currency="USD",
                    transaction_datetime="2025-01-15T10:00:00",
                    sender_name="JOHN DOE",
                    receiver_name="Shell Corp LLC",
                    match_field="sender",
                )
            ],
            web_findings=[
                WebSearchResult(
                    title="DOJ Press Release",
                    url="https://doj.gov/press/john-doe",
                    excerpt="John Doe indicted for money laundering.",
                )
            ],
            citations=[
                Citation(
                    index=1,
                    source_type=SourceType.INTERNAL_KB,
                    source_name="OFAC Sanctions",
                    entity_id="NK-001",
                ),
                Citation(
                    index=2,
                    source_type=SourceType.WEB_SEARCH,
                    source_name="DOJ",
                    title="DOJ Press Release",
                    url="https://doj.gov/press/john-doe",
                    excerpt="John Doe indicted for money laundering.",
                ),
            ],
            risk_assessment="HIGH",
        )

        formatted = report.format_report()

        assert "# AML Investigation Report: JOHN DOE" in formatted
        assert "## Risk Assessment: HIGH" in formatted
        assert "OFAC Sanctions" in formatted
        assert "tx-001" in formatted
        assert "100,000.00" in formatted
        assert "[1] Internal knowledge base / watchlist: OFAC Sanctions, entity_id: NK-001" in formatted
        assert "[2] DOJ Press Release. https://doj.gov/press/john-doe." in formatted

    def test_clear_report(self):
        """Report for clean subject."""
        report = InvestigationReport(
            subject_name="CLEAN PERSON",
            summary="No adverse findings.",
            risk_assessment="CLEAR",
        )
        formatted = report.format_report()
        assert "CLEAR" in formatted
        assert "No adverse findings" in formatted

    def test_serialization(self):
        """Report can be serialized to/from JSON."""
        report = InvestigationReport(
            subject_name="TEST",
            summary="Test summary",
            risk_assessment="LOW",
        )
        json_str = report.model_dump_json()
        restored = InvestigationReport.model_validate_json(json_str)
        assert restored.subject_name == "TEST"
        assert restored.risk_assessment == "LOW"


class TestDataParsing:
    """Tests for parsing data files."""

    def test_parse_watchlist_json(self):
        """Validate watchlist JSON structure."""
        watchlist_path = Path(__file__).resolve().parent.parent / "data" / "internal_kb_watchlist.json"
        if not watchlist_path.exists():
            pytest.skip("Watchlist data not available")

        with open(watchlist_path, encoding="utf-8") as f:
            entities = json.load(f)

        assert isinstance(entities, list)
        assert len(entities) > 0

        # Validate first entity structure
        e = entities[0]
        assert "entity_id" in e
        assert "name" in e
        assert "source" in e
        assert "aliases" in e
        assert isinstance(e["aliases"], list)

    def test_parse_transactions_csv(self):
        """Validate transactions CSV structure."""
        import csv

        csv_path = Path(__file__).resolve().parent.parent / "data" / "transactions.csv"
        if not csv_path.exists():
            pytest.skip("Transaction data not available")

        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames

        expected_headers = [
            "transaction_id", "amount", "currency", "transaction_datetime",
            "sender_name", "receiver_name",
        ]
        for h in expected_headers:
            assert h in headers, f"Missing header: {h}"
