"""End-to-end test for the AML Investigation Agent.

Runs the full agent pipeline on a sample case and validates
the output structure and citation format.

This test requires:
- GOOGLE_API_KEY environment variable (for Gemini + web search)
- The data files in data/ directory

To run:
    pytest tests/test_e2e.py -v -s --timeout=120
Or skip if no API key:
    pytest tests/test_e2e.py -v -s -k "not e2e"
"""

from __future__ import annotations

import asyncio
import os
import re

import pytest

# Skip all tests if no API key is available
pytestmark = pytest.mark.skipif(
    not os.getenv("GOOGLE_API_KEY") and not os.getenv("GEMINI_API_KEY"),
    reason="GOOGLE_API_KEY or GEMINI_API_KEY not set",
)


@pytest.fixture(scope="module")
def sample_report():
    """Run the agent once and cache the report for all tests in this module."""
    from aml_agent.runner import run_investigation

    # Use a known entity from the FBI Most Wanted list
    subject = "ALIREZA SHAFIE NASAB"
    report, _artifacts = asyncio.run(run_investigation(subject))
    return report


class TestEndToEnd:
    """End-to-end integration tests."""

    def test_report_not_empty(self, sample_report):
        """Agent produces a non-empty report."""
        assert sample_report
        assert len(sample_report) > 100

    def test_report_contains_subject_name(self, sample_report):
        """Report mentions the investigated subject."""
        assert "NASAB" in sample_report.upper() or "ALIREZA" in sample_report.upper()

    def test_report_has_risk_assessment(self, sample_report):
        """Report contains a risk assessment."""
        report_upper = sample_report.upper()
        assert any(
            level in report_upper
            for level in ["HIGH", "MEDIUM", "LOW", "CLEAR"]
        )

    def test_report_has_citations(self, sample_report):
        """Report contains numbered citation markers."""
        # Look for [1], [2], etc.
        citation_pattern = re.compile(r"\[\d+\]")
        matches = citation_pattern.findall(sample_report)
        assert len(matches) > 0, "Report should contain citation markers like [1], [2]"

    def test_report_has_sources_section(self, sample_report):
        """Report contains a Sources section."""
        assert "Sources" in sample_report or "sources" in sample_report.lower()

    def test_report_has_kb_findings(self, sample_report):
        """Report mentions internal KB findings."""
        report_lower = sample_report.lower()
        assert any(
            kw in report_lower
            for kw in ["knowledge base", "watchlist", "sanctions", "fbi", "ofac"]
        ), "Report should mention KB/watchlist findings"

    def test_report_structure(self, sample_report):
        """Report follows expected section structure."""
        # Should have at least some of these sections
        sections_found = 0
        for section in ["Summary", "Risk Assessment", "Transaction", "Sources", "Knowledge Base", "Search"]:
            if section.lower() in sample_report.lower():
                sections_found += 1
        assert sections_found >= 3, f"Report should have structured sections, found {sections_found}"
