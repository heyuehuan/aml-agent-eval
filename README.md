# Real-Time AML Agent (Agentic AI Evaluation Bootcamp)

This repository contains the implementation of a real-time Anti-Money Laundering (AML) agent developed for the Agentic AI Evaluation Bootcamp.

## Overview

The project focuses on building an agentic system capable of analyzing transaction data, detecting suspicious patterns, and supporting AML workflows in real time.

## References

- Vector Institute Reference Implementation:  
  https://github.com/VectorInstitute/eval-agents/tree/main
- AML Agent Dataset:  
  https://github.com/heyuehuan/aml-agent-eval-data

## Implementation

### Architecture

The AML agent is built on [Google ADK](https://github.com/google/adk-python) (`LlmAgent`) backed by Gemini 2.5. It follows a four-step investigation workflow orchestrated entirely by the model, with three specialized tools providing grounded access to structured and unstructured data sources.

```
User Input
    │
    ▼
aml_agent/runner.py  ──  Runner (InMemorySession)
    │
    ▼
aml_agent/agent.py   ──  LlmAgent (Gemini 2.5 Flash)
    │
    ├── aml_agent/tools/sql_database.py   ──  ReadOnlySqlDatabase (SQLite)
    ├── aml_agent/tools/kb_search.py      ──  KBSearchTool (Weaviate Cloud)
    └── aml_agent/tools/web_search.py     ──  WebSearchTool (Gemini Grounded Search)
    │
    ▼
aml_agent/tracing.py      ──  CallbackTracer (LLM + tool call tracing)
    │
    ▼
aml_agent/report_html.py  ──  HTML report renderer
```

### Agent Tools

**`ReadOnlySqlDatabase`** (`tools/sql_database.py`)  
Exposes two functions to the model: `get_schema_info` (returns table/column metadata) and `execute` (runs SQL queries). All queries are validated before execution using [SQLGlot](https://github.com/tobymao/sqlglot) AST parsing — only `SELECT` and `UNION` roots are permitted; any write operation (`INSERT`, `UPDATE`, `DELETE`, `DROP`, `CREATE`, etc.) or multi-statement input is rejected with a security error before it reaches the database. Results are returned as markdown tables, capped at 100 rows.

**`KBSearchTool`** (`tools/kb_search.py`)  
Searches the internal watchlist / sanctions knowledge base stored in [Weaviate Cloud](https://weaviate.io/). Performs a semantic `near_text` query first, falling back to BM25 keyword search if no vector results are found. Results include entity ID, source list name (e.g. OFAC SDN, FBI Most Wanted, UN Sanctions), similarity score, and a text snippet. Also exposes `get_entity_by_id` for direct ID-based lookup. Transient connectivity errors (rate limits, connection resets) are automatically retried with exponential back-off (10 s → 30 s → 60 s).

**`WebSearchTool`** (`tools/web_search.py`)  
Performs real-time due diligence via the Gemini API with Google Search grounding. Constructs a structured prompt covering biographical details, financial crime exposure, sanctions/PEP status, corporate ownership, and adverse media. Grounding chunk metadata is parsed to extract direct page URLs (redirect URLs are resolved to their final destination), page titles (fetched live), and relevant excerpts. If the first search returns no grounding sources, a second attempt is made with appended AML-specific keywords. Internal KB entity IDs are stripped from queries before they are sent to the public search engine.

### Data Layer

| Source                            | Format    | Contents                                                                                             |
| --------------------------------- | --------- | ---------------------------------------------------------------------------------------------------- |
| `data/transactions.csv`           | CSV       | 20,000 synthetic wire transfer transactions (ID, amount, currency, datetime, sender, receiver, memo) |
| `data/internal_kb_watchlist.json` | JSON      | ~20,000 watchlist entities from OFAC, FBI, UN, Canada, and other sanctions lists with aliases        |
| Weaviate Cloud collection         | Vector DB | Same watchlist content chunked and embedded for semantic search                                      |

The SQLite database (`aml_agent/data/aml_transactions.db`) is built on first run from the CSV via `aml_agent/data/build_db.py`. Ground-truth entity IDs from the CSV are intentionally excluded so the agent cannot trivially look them up — it must search by name.

### Prompt Design

The system instruction (`aml_agent/prompts.py`) defines a four-step workflow:

1. **Extract the subject** from free-form input (email, referral note, raw name)
2. **KB search** — up to three progressive queries (full name → surname → key alias), stopping at first hit
3. **SQL analysis** — schema-first, then aggregates, then targeted LIKE-pattern queries across sender, receiver, and memo fields
4. **Web search** — mandatory step; query is enriched with KB-found disambiguation details (aliases, DOB, sanctions programme)

The instruction enforces strict citation formatting: one source per numbered entry, pipe-separated `Title | URL | excerpt` for web results, and a clear distinction between the narrative summary and the citable sources block returned by the web search tool.

### Tracing

Every agent run — whether invoked from the CLI runner or the evaluation framework — captures a full, structured trace via `CallbackTracer` (`aml_agent/tracing.py`).

**Local artifact tracing** (always on):

- **`llm_call_history`** — per-call log of the outgoing request (model, new content delta, tool declarations on first call) and the incoming response (content, finish reason, per-call token counts for prompt/candidates/thoughts/cached)
- **`tool_calls`** — ordered list of every tool invocation with name, args, response text, and timestamp
- **`token_usage`** — aggregate prompt / candidate / thoughts / total token counts across all LLM calls
- **`trace_metrics`** — high-level summary: `llm_call_count`, `tool_call_count`, `unique_tools_used`, `elapsed_sec`

**Langfuse tracing** (optional, `--live` mode):

- `init_tracing()` in `evaluation/tracing.py` installs a `GoogleADKInstrumentor` on top of an OpenTelemetry `TracerProvider` that exports to Langfuse via OTLP — every LLM call and tool call becomes an observation span automatically.
- After each live experiment run, `report_trace_scores()` uploads the `CallbackTracer` metrics (llm_call_count, tool_call_count, token counts, elapsed_sec) as numeric Langfuse scores on the corresponding trace, making them visible in the Langfuse UI alongside the auto-captured spans.

### Output

Running `python -m aml_agent "Subject Name or details" --output report.html` produces two files:

- **`report.html`** — self-contained HTML report with a risk badge (HIGH / MEDIUM / LOW / CLEAR), collapsible sections, an interactive sortable/filterable transaction data table, and hyperlinked citations.
- **`report.artifacts.json`** — full audit trail including: investigation start/finish timestamps and elapsed time, every tool call with arguments, responses, and timestamps, all raw SQL result sets, a structured parse of the markdown report (subject, risk level, sections, citations), per-LLM-call token usage and response content, aggregate token usage, and high-level `trace_metrics`.

### Evaluation Framework

The evaluation harness lives in `aml_agent/evaluation/` and supports three modes:

```bash
# Local — evaluate pre-computed artifact JSON files offline (no API key needed)
python -m aml_agent.evaluation.evaluate

# Langfuse — upload dataset + evaluate pre-computed artifacts via Langfuse SDK
python -m aml_agent.evaluation.evaluate --langfuse

# Live — run the real agent end-to-end with full LLM/tool tracing + Langfuse
python -m aml_agent.evaluation.evaluate --live
```

**Batch runner** — `run_test_cases.py` re-runs all test cases (or a filtered subset) and writes `output/test_cases/test_run/Test_<ID>.html` and `Test_<ID>.artifacts.json` for offline evaluation:

```bash
python run_test_cases.py              # all test cases
python run_test_cases.py TC-001 TC-006  # specific IDs
```

**Graders** (`evaluation/graders/`) operate at two levels:

| Level | Grader        | What it checks                                       |
| ----- | ------------- | ---------------------------------------------------- |
| Item  | `internal_kb` | KB hit correctness, coverage, match count            |
| Item  | `sql`         | SQL query execution and result quality               |
| Item  | `transaction` | Transaction match count and details                  |
| Item  | `web_search`  | Web search grounding and citation quality            |
| Item  | `report`      | Report structure, risk level, citation format        |
| Item  | `trace`       | LLM call count, tool call count from `trace_metrics` |
| Run   | `run`         | Aggregate pass rates across all test cases           |

### Tests

Unit tests cover the read-only SQL enforcement layer (17 tests including all write-operation block cases), the KB search tool with mocked Weaviate responses (16 tests including retry behaviour), data file integrity (2 tests), and a full end-to-end smoke test gated on `GOOGLE_API_KEY` being set.

```bash
# Unit tests (no API key required)
python -m pytest tests/ -v --ignore=tests/test_e2e.py   # 55 tests

# All tests including E2E
GOOGLE_API_KEY=... python -m pytest tests/ -v
```

## Disclaimer

This project is for learning and research purposes only.  
All data used is synthetic and/or publicly available, intended solely for testing and experimentation.  
The content and outputs of this repository do not represent the views, systems, or practices of any team member or their affiliated organizations.

## Team

- Alan
- Arash
- Iris
- Yuehuan
- Zilin
