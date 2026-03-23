"""Pydantic data models for the AML agent."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class SourceType(str, Enum):
    """Types of sources used in AML investigation."""

    INTERNAL_KB = "internal_kb"
    WEB_SEARCH = "web_search"
    TRANSACTION_DB = "transaction_db"


class Citation(BaseModel):
    """A single citation reference."""

    index: int = Field(description="Citation number")
    source_type: SourceType = Field(description="Type of source")
    source_name: str = Field(description="Name of the source (e.g., 'OFAC Sanctions')")
    entity_id: str | None = Field(default=None, description="Entity ID from KB if applicable")
    url: str | None = Field(default=None, description="Direct URL if web source")
    title: str | None = Field(default=None, description="Webpage title if web source")
    excerpt: str | None = Field(default=None, description="Relevant excerpt")


class KBMatch(BaseModel):
    """A watchlist/knowledge base match result."""

    entity_id: str
    name: str
    source: str
    score: float | None = None
    snippet: str | None = None


class TransactionMatch(BaseModel):
    """A matching transaction from the database."""

    transaction_id: str
    amount: float
    currency: str
    transaction_datetime: str
    sender_name: str
    receiver_name: str
    memo: str | None = None
    match_field: str = Field(description="Which field matched: sender, receiver, or memo")


class WebSearchResult(BaseModel):
    """A web search finding."""

    title: str
    url: str
    excerpt: str
    source_tool: str = "gemini_grounded_search"


class InvestigationReport(BaseModel):
    """Full AML investigation report produced by the agent."""

    subject_name: str = Field(description="Name of the subject being investigated")
    summary: str = Field(description="Executive summary of findings with inline citation markers")
    kb_matches: list[KBMatch] = Field(default_factory=list, description="Internal KB watchlist matches")
    transaction_matches: list[TransactionMatch] = Field(
        default_factory=list, description="Suspicious transaction matches"
    )
    web_findings: list[WebSearchResult] = Field(
        default_factory=list, description="External web search findings"
    )
    citations: list[Citation] = Field(default_factory=list, description="Ordered list of all citations")
    risk_assessment: str = Field(description="Overall risk assessment: HIGH, MEDIUM, LOW, or CLEAR")

    def format_report(self) -> str:
        """Format the report with proper citation rendering."""
        lines = [
            f"# AML Investigation Report: {self.subject_name}",
            "",
            f"## Risk Assessment: {self.risk_assessment}",
            "",
            "## Summary",
            self.summary,
            "",
        ]

        if self.kb_matches:
            lines.append("## Internal Knowledge Base Matches")
            for m in self.kb_matches:
                lines.append(f"- **{m.name}** ({m.source}) [entity_id: {m.entity_id}]")
                if m.snippet:
                    lines.append(f"  _{m.snippet[:200]}_")
            lines.append("")

        if self.transaction_matches:
            lines.append("## Suspicious Transactions")
            lines.append("| ID | Amount | Currency | Date | Sender | Receiver | Match Field |")
            lines.append("|---|---|---|---|---|---|---|")
            for t in self.transaction_matches:
                lines.append(
                    f"| {t.transaction_id[:12]}... | {t.amount:,.2f} | {t.currency} | "
                    f"{t.transaction_datetime} | {t.sender_name} | {t.receiver_name} | {t.match_field} |"
                )
            lines.append("")

        if self.web_findings:
            lines.append("## External Search Findings")
            for w in self.web_findings:
                lines.append(f"- **{w.title}**")
                lines.append(f"  {w.url}")
                lines.append(f"  _{w.excerpt[:200]}_")
            lines.append("")

        if self.citations:
            lines.append("## Sources")
            for c in self.citations:
                if c.source_type == SourceType.INTERNAL_KB:
                    lines.append(f"[{c.index}] Internal knowledge base / watchlist: {c.source_name}, entity_id: {c.entity_id}")
                elif c.source_type == SourceType.WEB_SEARCH:
                    excerpt_part = f" ({c.excerpt[:100]})" if c.excerpt else ""
                    lines.append(f"[{c.index}] {c.title}. {c.url}.{excerpt_part}")
                elif c.source_type == SourceType.TRANSACTION_DB:
                    lines.append(f"[{c.index}] Transaction database query: {c.source_name}")

        return "\n".join(lines)
