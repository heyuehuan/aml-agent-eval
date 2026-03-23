"""HTML report renderer for AML investigation reports.

Converts the markdown report produced by the AML agent into a professional,
print-ready compliance report using self-contained HTML/CSS.

Usage:
    from aml_agent.report_html import render_html
    html = render_html(markdown_text, subject="John Doe")
    with open("report.html", "w") as f:
        f.write(html)
"""

from __future__ import annotations

import html
import re
from datetime import datetime, timezone
from zoneinfo import ZoneInfo


# ── Risk level config ──────────────────────────────────────────────────────────

_RISK_COLORS = {
    "HIGH":   {"bg": "#c0392b", "text": "#ffffff", "border": "#922b21"},
    "MEDIUM": {"bg": "#e67e22", "text": "#ffffff", "border": "#b9770e"},
    "LOW":    {"bg": "#27ae60", "text": "#ffffff", "border": "#1e8449"},
    "CLEAR":  {"bg": "#7f8c8d", "text": "#ffffff", "border": "#616a6b"},
}

_RISK_LABELS = {
    "HIGH":   "HIGH RISK",
    "MEDIUM": "MEDIUM RISK",
    "LOW":    "LOW RISK",
    "CLEAR":  "CLEAR",
}


# ── Markdown → HTML parser ─────────────────────────────────────────────────────

def _escape(text: str) -> str:
    return html.escape(text)


def _inline_fmt(text: str) -> str:
    """Apply inline formatting: **bold**, *italic*, `code`, and citation [N]."""
    # Collapse literal \n escape sequences (LLM sometimes writes them in excerpts/titles)
    text = text.replace("\\n", " ").replace("\\t", " ")
    # Collapse runs of whitespace that result from the above
    text = re.sub(r"  +", " ", text)
    # Bold
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    # Italic
    text = re.sub(r"\*(.+?)\*", r"<em>\1</em>", text)
    # Inline code
    text = re.sub(r"`([^`]+)`", r'<code class="inline-code">\1</code>', text)
    # Normalize aggregated citations [1,2] or [1, 2] → [1] [2] before rendering
    def _expand_multi_cite(m: re.Match) -> str:
        nums = re.split(r"[,\s]+", m.group(1).strip())
        return " ".join(f"[{n.strip()}]" for n in nums if n.strip().isdigit())
    text = re.sub(r"\[([\d,\s]+)\]", _expand_multi_cite, text)
    # Citations: [1], [2], etc. → clickable anchor links to source list
    text = re.sub(r"\[(\d+)\]", r'<sup class="cite"><a href="#src-\1">[\1]</a></sup>', text)
    # URLs that remain as plain text → clickable links
    # Stop before trailing punctuation (.,:;!?) or closing paren/bracket
    text = re.sub(
        r'(?<!href=")(https?://[^\s<>"\')]+?)([.,:;!?\)\]]*(?=\s|$|<))',
        lambda m: f'<a href="{m.group(1)}" target="_blank" rel="noopener noreferrer">{m.group(1)}</a>{m.group(2)}',
        text,
    )
    return text


def _parse_markdown(md: str) -> tuple[str, str, str, list[tuple[str, str]]]:
    """Parse the agent's markdown report into structured components.

    Returns
    -------
    (risk_level, subject_name, body_html, sections)
    where sections is a list of (heading, html_content) tuples.
    """
    lines = md.strip().splitlines()
    risk_level = "CLEAR"
    subject_name = ""
    sections: list[tuple[str, str]] = []

    current_heading = ""
    current_body: list[str] = []

    def flush():
        if current_heading or current_body:
            sections.append((current_heading, _render_body(current_body)))

    for line in lines:
        # H1 — report title: "# AML Investigation Report: John Doe"
        h1 = re.match(r"^#\s+(.+)$", line)
        if h1:
            flush()
            current_heading = ""
            current_body = []
            title_text = h1.group(1)
            # Extract subject name after colon
            m = re.search(r":\s*(.+)$", title_text)
            if m:
                subject_name = m.group(1).strip()
            continue

        # H2 — section heading: "## Risk Assessment: HIGH"
        h2 = re.match(r"^##\s+(.+)$", line)
        if h2:
            flush()
            current_heading = h2.group(1)
            current_body = []
            # Extract risk level from heading if present
            risk_m = re.search(r"Risk Assessment:\s*(HIGH|MEDIUM|LOW|CLEAR)", current_heading, re.IGNORECASE)
            if risk_m:
                risk_level = risk_m.group(1).upper()
            continue

        # H3 — sub-section
        h3 = re.match(r"^###\s+(.+)$", line)
        if h3:
            current_body.append(f"<h3>{_escape(h3.group(1))}</h3>")
            continue

        current_body.append(line)

    flush()
    return risk_level, subject_name, "", sections


def _render_body(lines: list[str]) -> str:
    """Convert a list of raw markdown lines into HTML."""
    html_parts: list[str] = []
    in_list = False
    in_ordered_list = False
    i = 0

    while i < len(lines):
        line = lines[i]

        # Unordered list item
        if re.match(r"^[-*]\s+", line):
            if not in_list:
                html_parts.append("<ul>")
                in_list = True
            item = re.sub(r"^[-*]\s+", "", line)
            html_parts.append(f"<li>{_inline_fmt(_escape(item))}</li>")
            i += 1
            continue

        # Citation-style source line: [1] Title | URL | excerpt  OR  [1] Some text
        # The agent writes sources as [N] ... (bracket-number), not numbered list `1.`
        cite_m = re.match(r"^\[(\d+)\]\s+(.+)$", line)
        if cite_m:
            if not in_ordered_list:
                html_parts.append('<ol class="sources-list">')
                in_ordered_list = True
            cite_num = cite_m.group(1)
            item = cite_m.group(2)
            parts = [p.strip() for p in item.split(" | ")]
            if len(parts) >= 2:
                title_part = _inline_fmt(_escape(parts[0]))
                raw_url = parts[1].strip()
                # Skip invalid / placeholder URLs (e.g. "N/A") so they don't pollute source cards
                is_valid_url = raw_url.lower() not in ("n/a", "", "none", "unknown") and raw_url.startswith("http")
                url_part = _inline_fmt(_escape(raw_url)) if is_valid_url else ""
                excerpt_part = _inline_fmt(_escape(parts[2])) if len(parts) > 2 else ""
                inner = '<div class="src-body">'
                inner += f'<span class="src-title">{title_part}</span>'
                if url_part:
                    inner += f'<span class="src-url">{url_part}</span>'
                if excerpt_part:
                    inner += f'<span class="src-excerpt">{excerpt_part}</span>'
                inner += '</div>'
            else:
                inner = _inline_fmt(_escape(item))
            html_parts.append(f'<li id="src-{cite_num}">{inner}</li>')
            i += 1
            continue

        # Ordered list item  (1. text)
        if re.match(r"^\d+\.\s+", line):
            if not in_ordered_list:
                html_parts.append('<ol class="sources-list">')
                in_ordered_list = True
            # Extract source number from "1. " prefix for anchor
            num_m = re.match(r"^(\d+)\.\s+", line)
            cite_num = num_m.group(1) if num_m else "unknown"
            item = re.sub(r"^\d+\.\s+", "", line)
            # Each source gets its own <li> — split on pipe separators used by web_search
            parts = [p.strip() for p in item.split(" | ")]
            if len(parts) >= 2:
                title_part = _inline_fmt(_escape(parts[0]))
                raw_url = parts[1].strip()
                is_valid_url = raw_url.lower() not in ("n/a", "", "none", "unknown") and raw_url.startswith("http")
                url_part = _inline_fmt(_escape(raw_url)) if is_valid_url else ""
                excerpt_part = _inline_fmt(_escape(parts[2])) if len(parts) > 2 else ""
                inner = '<div class="src-body">'
                inner += f'<span class="src-title">{title_part}</span>'
                if url_part:
                    inner += f'<span class="src-url">{url_part}</span>'
                if excerpt_part:
                    inner += f'<span class="src-excerpt">{excerpt_part}</span>'
                inner += '</div>'
            else:
                inner = _inline_fmt(_escape(item))
            html_parts.append(f'<li id="src-{cite_num}">{inner}</li>')
            i += 1
            continue

        # Close any open lists
        if in_list:
            html_parts.append("</ul>")
            in_list = False
        if in_ordered_list:
            html_parts.append("</ol>")
            in_ordered_list = False

        # Horizontal rule
        if re.match(r"^---+$", line.strip()):
            html_parts.append("<hr>")
            i += 1
            continue

        # Empty line → paragraph break
        if not line.strip():
            html_parts.append("")
            i += 1
            continue

        # Regular paragraph line — collect until blank or a new structural element
        para_lines = []
        while i < len(lines) and lines[i].strip() \
                and not re.match(r"^[-*\d]", lines[i]) \
                and not re.match(r"^\[\d+\]", lines[i]) \
                and not re.match(r"^---+$", lines[i].strip()) \
                and not re.match(r"^#{1,3}\s", lines[i]):
            para_lines.append(_escape(lines[i]))
            i += 1
        if para_lines:
            html_parts.append(f"<p>{_inline_fmt(' '.join(para_lines))}</p>")
        else:
            i += 1

    if in_list:
        html_parts.append("</ul>")
    if in_ordered_list:
        html_parts.append("</ol>")

    return "\n".join(html_parts)

def _build_tx_table(sql_results: list[dict]) -> str:
    """Build a sortable, filterable HTML table from accumulated SQL execute results.

    Groups rows by column signature, picks the widest set (most columns),
    deduplicates rows, and returns self-contained HTML.  Returns empty string
    if there is no usable data.
    """
    if not sql_results:
        return ""

    # Group rows by column tuple; deduplicate rows within each group
    by_cols: dict[tuple, list[list[str]]] = {}
    for res in sql_results:
        key = tuple(res["columns"])
        if key not in by_cols:
            by_cols[key] = []
        for row in res["rows"]:
            if row not in by_cols[key]:
                by_cols[key].append(row)

    if not by_cols:
        return ""

    # Prefer column set with most columns; break ties by most rows
    best_key = max(by_cols.keys(), key=lambda k: (len(k), len(by_cols[k])))
    columns = list(best_key)
    rows = by_cols[best_key]
    if not rows:
        return ""

    th_cells = "".join(
        f'<th onclick="txSort({i})">{_escape(col)}<span class="th-sort-icon">⇅</span></th>'
        for i, col in enumerate(columns)
    )
    tbody_rows = "".join(
        "<tr>" + "".join(f'<td title="{_escape(str(v))}">{_escape(v)}</td>' for v in row) + "</tr>"
        for row in rows
    )
    n = len(rows)
    return f"""
<div class="tx-table-wrap">
  <div class="tx-controls">
    <input id="tx-filter" type="text" placeholder="&#128269; Filter transactions…"
           oninput="txFilter()" class="tx-filter-input">
    <span class="tx-row-count" id="tx-row-count">{n} row{'s' if n != 1 else ''}</span>
  </div>
  <div class="tx-scroll">
    <table id="tx-table">
      <thead><tr>{th_cells}</tr></thead>
      <tbody>{tbody_rows}</tbody>
    </table>
  </div>
</div>"""

# ── Section renderers ──────────────────────────────────────────────────────────

_SECTION_ICONS = {
    "summary": "📋",
    "internal knowledge base": "🔍",
    "knowledge base": "🔍",
    "watchlist": "🔍",
    "transaction": "💸",
    "external search": "🌐",
    "web search": "🌐",
    "adverse media": "📰",
    "sources": "📚",
    "risk": "⚠️",
    "conclusion": "✅",
    "notes": "📝",
}


def _section_icon(heading: str) -> str:
    lower = heading.lower()
    for key, icon in _SECTION_ICONS.items():
        if key in lower:
            return icon
    return "📄"


def _render_section(heading: str, body_html: str, index: int) -> str:
    icon = _section_icon(heading)
    section_id = f"section-{index}"
    collapsed = "collapsed" if heading.lower().startswith("source") else ""
    arrow = "▼" if not collapsed else "▶"
    display = "none" if collapsed else "block"

    return f"""
    <div class="section {collapsed}" id="{section_id}">
        <div class="section-header" onclick="toggleSection('{section_id}')">
            <span class="section-icon">{icon}</span>
            <span class="section-title">{_escape(heading)}</span>
            <span class="section-arrow" id="{section_id}-arrow">{arrow}</span>
        </div>
        <div class="section-body" id="{section_id}-body" style="display:{display}">
            {body_html}
        </div>
    </div>"""


# ── CSS ────────────────────────────────────────────────────────────────────────

def _css(risk_color: dict) -> str:
    return f"""
/* === Reset & Base === */
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
html {{ font-size: 15px; scroll-behavior: smooth; }}
body {{
    font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
    background: #f0f2f5;
    color: #2c3e50;
    line-height: 1.65;
}}

/* === Page wrapper === */
.page {{
    max-width: 900px;
    margin: 32px auto 64px;
    background: #ffffff;
    border-radius: 8px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.10);
    overflow: hidden;
}}

/* === Header === */
.header {{
    background: linear-gradient(135deg, #1a2540 0%, #2c3e6e 100%);
    color: #fff;
    padding: 36px 40px 28px;
    position: relative;
}}
.header-logo {{
    font-size: 11px;
    letter-spacing: 2px;
    text-transform: uppercase;
    opacity: 0.70;
    margin-bottom: 10px;
}}
.header h1 {{
    font-size: 22px;
    font-weight: 700;
    letter-spacing: 0.3px;
    margin-bottom: 6px;
}}
.header-subject {{
    font-size: 28px;
    font-weight: 800;
    letter-spacing: -0.5px;
    color: #fff;
    margin-bottom: 20px;
}}
.header-meta {{
    font-size: 12px;
    opacity: 0.65;
    display: flex;
    gap: 24px;
    flex-wrap: wrap;
}}
.header-meta span {{ display: flex; align-items: center; gap: 5px; }}

/* === Risk badge === */
.risk-badge-wrap {{
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 16px;
    flex-wrap: wrap;
    margin-top: 8px;
}}
.risk-badge {{
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: {risk_color['bg']};
    color: {risk_color['text']};
    border: 2px solid {risk_color['border']};
    border-radius: 6px;
    padding: 10px 22px;
    font-size: 16px;
    font-weight: 800;
    letter-spacing: 1px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.18);
    text-transform: uppercase;
}}
.risk-badge .risk-dot {{
    width: 10px; height: 10px;
    border-radius: 50%;
    background: {risk_color['text']};
    opacity: 0.85;
    flex-shrink: 0;
}}

/* === Divider === */
.divider {{
    height: 4px;
    background: linear-gradient(to right, {risk_color['bg']}, {risk_color['border']});
}}

/* === Content area === */
.content {{ padding: 12px 0 24px; }}

/* === Section === */
.section {{
    border-bottom: 1px solid #eaecef;
    transition: background 0.15s;
}}
.section:last-child {{ border-bottom: none; }}

.section-header {{
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 16px 40px;
    cursor: pointer;
    user-select: none;
    transition: background 0.12s;
}}
.section-header:hover {{ background: #f7f8fa; }}

.section-icon {{ font-size: 18px; flex-shrink: 0; }}
.section-title {{
    font-size: 15px;
    font-weight: 700;
    color: #1a2540;
    flex: 1;
    letter-spacing: 0.1px;
}}
.section-arrow {{
    font-size: 12px;
    color: #95a0b0;
    transition: transform 0.2s;
}}

.section-body {{
    padding: 4px 40px 20px 68px;
    font-size: 14px;
    color: #3a4a5c;
}}

/* === Typography === */
.section-body p {{ margin-bottom: 10px; }}
.section-body h3 {{
    font-size: 13px;
    font-weight: 700;
    color: #1a2540;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin: 18px 0 8px;
    padding-bottom: 4px;
    border-bottom: 1px solid #e0e4ea;
}}
.section-body ul {{ margin: 8px 0 10px 18px; }}
.section-body ul li {{ margin-bottom: 4px; }}

.sources-list {{
    padding: 0;
    list-style: none;
    counter-reset: source-counter;
}}
.sources-list li {{
    counter-increment: source-counter;
    display: flex;
    gap: 12px;
    margin-bottom: 12px;
    padding: 10px 14px;
    background: #f7f9fc;
    border-radius: 5px;
    border-left: 3px solid #2c3e6e;
    font-size: 13px;
    line-height: 1.6;
    align-items: flex-start;
}}
.sources-list li::before {{
    content: "[" counter(source-counter) "]";
    font-weight: 700;
    color: #2c3e6e;
    flex-shrink: 0;
    min-width: 28px;
    padding-top: 1px;
}}
.src-title {{
    font-weight: 600;
    color: #1a2540;
    display: block;
}}
.src-url {{
    font-size: 12px;
    color: #2980b9;
    word-break: break-all;
    display: block;
}}
.src-url a {{
    color: #2980b9;
}}
.src-excerpt {{
    font-size: 12px;
    color: #5a6a7c;
    font-style: italic;
    display: block;
    margin-top: 2px;
}}
.src-body {{
    display: flex;
    flex-direction: column;
    gap: 2px;
    flex: 1;
    min-width: 0;
}}

/* === Inline elements === */
code.inline-code {{
    font-family: 'Consolas', 'Fira Code', monospace;
    font-size: 12.5px;
    background: #eef0f4;
    border-radius: 3px;
    padding: 1px 5px;
    color: #c0392b;
}}
sup.cite {{
    font-size: 10px;
    font-weight: 700;
    color: #2980b9;
    vertical-align: super;
    cursor: pointer;
}}
sup.cite a {{
    color: #2980b9;
    text-decoration: none;
    cursor: pointer;
}}
sup.cite a:hover {{
    text-decoration: underline;
    color: #c0392b;
}}
a {{
    color: #2980b9;
    text-decoration: none;
}}
a:hover {{ text-decoration: underline; }}
strong {{ color: #1a2540; }}

/* === Footer === */
.footer {{
    background: #f7f8fa;
    border-top: 1px solid #eaecef;
    padding: 18px 40px;
    font-size: 11.5px;
    color: #8a95a3;
    display: flex;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 8px;
}}

/* === Transaction data table === */
.tx-table-wrap {{
    margin: 14px 0 4px;
}}
.tx-controls {{
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 8px;
}}
.tx-filter-input {{
    flex: 1;
    padding: 7px 12px;
    border: 1px solid #d0d5dd;
    border-radius: 5px;
    font-size: 13px;
    outline: none;
    color: #2c3e50;
    transition: border-color 0.15s;
}}
.tx-filter-input:focus {{
    border-color: #2c3e6e;
    box-shadow: 0 0 0 2px rgba(44,62,110,0.12);
}}
.tx-row-count {{
    font-size: 12px;
    color: #8a95a3;
    white-space: nowrap;
}}
.tx-scroll {{
    overflow-x: auto;
    border: 1px solid #e0e4ea;
    border-radius: 6px;
}}
#tx-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
    min-width: 600px;
}}
#tx-table thead tr {{
    background: #1a2540;
    color: #fff;
    position: sticky;
    top: 0;
    z-index: 1;
}}
#tx-table th {{
    padding: 9px 12px;
    text-align: left;
    font-weight: 600;
    letter-spacing: 0.2px;
    cursor: pointer;
    white-space: nowrap;
    user-select: none;
}}
#tx-table th:hover {{
    background: #2c3e6e;
}}
.th-sort-icon {{
    font-size: 10px;
    opacity: 0.55;
    margin-left: 4px;
}}
#tx-table tbody tr:nth-child(even) {{
    background: #f7f9fc;
}}
#tx-table tbody tr:hover {{
    background: #eef2fa;
}}
#tx-table td {{
    padding: 6px 12px;
    border-bottom: 1px solid #eaecef;
    color: #2c3e50;
    vertical-align: top;
    max-width: 200px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}}
#tx-table td:hover {{
    white-space: normal;
    overflow: visible;
    background: #fffbe6;
    position: relative;
    z-index: 2;
}}

/* === Print styles === */
@media print {{
    body {{ background: #fff; }}
    .page {{ box-shadow: none; margin: 0; border-radius: 0; max-width: 100%; }}
    .section-header {{ cursor: default; }}
    .section-body {{ display: block !important; }}
    .header {{ -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
    .risk-badge {{ -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
    .divider {{ -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
    .sources-list li {{ -webkit-print-color-adjust: exact; print-color-adjust: exact; break-inside: avoid; }}
}}
"""


# ── JS ─────────────────────────────────────────────────────────────────────────

_JS = """
function toggleSection(id) {
    const body = document.getElementById(id + '-body');
    const arrow = document.getElementById(id + '-arrow');
    const section = document.getElementById(id);
    const isHidden = body.style.display === 'none';
    body.style.display = isHidden ? 'block' : 'none';
    arrow.textContent = isHidden ? '▼' : '▶';
    section.classList.toggle('collapsed', !isHidden);
}

// === Transaction table: filter & sort ===
var _txSortCol = -1;
var _txSortAsc = true;

function txFilter() {
    var filter = document.getElementById('tx-filter').value.toLowerCase();
    var tbody = document.querySelector('#tx-table tbody');
    if (!tbody) return;
    var rows = tbody.getElementsByTagName('tr');
    var visible = 0;
    for (var i = 0; i < rows.length; i++) {
        var show = rows[i].textContent.toLowerCase().includes(filter);
        rows[i].style.display = show ? '' : 'none';
        if (show) visible++;
    }
    var countEl = document.getElementById('tx-row-count');
    if (countEl) countEl.textContent = visible + ' row' + (visible !== 1 ? 's' : '') + (filter ? ' (filtered)' : '');
}

function txSort(col) {
    var table = document.getElementById('tx-table');
    if (!table) return;
    var tbody = table.querySelector('tbody');
    var rows = Array.from(tbody.getElementsByTagName('tr'));
    var asc = (_txSortCol === col) ? !_txSortAsc : true;
    _txSortCol = col; _txSortAsc = asc;
    rows.sort(function(a, b) {
        var aVal = a.cells[col] ? a.cells[col].textContent.trim() : '';
        var bVal = b.cells[col] ? b.cells[col].textContent.trim() : '';
        var aNum = parseFloat(aVal.replace(/[,\\s$]/g, ''));
        var bNum = parseFloat(bVal.replace(/[,\\s$]/g, ''));
        if (!isNaN(aNum) && !isNaN(bNum)) return asc ? aNum - bNum : bNum - aNum;
        return asc ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
    });
    table.querySelectorAll('.th-sort-icon').forEach(function(icon, i) {
        icon.textContent = i === col ? (asc ? ' ↑' : ' ↓') : ' ⇅';
    });
    rows.forEach(function(r) { tbody.appendChild(r); });
}
"""


# ── Main public function ───────────────────────────────────────────────────────

def render_html(
    markdown_text: str,
    subject: str = "",
    sql_results: list[dict] | None = None,
    version: str = "",
) -> str:
    """Convert an AML investigation markdown report to a professional HTML document.

    Parameters
    ----------
    markdown_text : str
        The raw markdown string produced by the AML agent.
    subject : str
        The investigated subject's name (used as a fallback in the header).
    sql_results : list[dict] | None
        Structured SQL results captured by the runner (columns + rows).  When
        provided, an interactive sortable/filterable table is injected into the
        Wire Transactions section.
    version : str
        Git version string (``branch/sha7``) stamped into the footer.

    Returns
    -------
    str
        A complete, self-contained HTML document.
    """
    risk_level, parsed_subject, _, sections = _parse_markdown(markdown_text)

    display_subject = parsed_subject or subject or "Unknown Subject"
    risk_color = _RISK_COLORS.get(risk_level, _RISK_COLORS["CLEAR"])
    risk_label = _RISK_LABELS.get(risk_level, risk_level)

    _est = ZoneInfo("America/New_York")
    now = datetime.now(_est)
    tz_label = now.strftime("%Z")  # EDT or EST
    generated_at = now.strftime(f"%Y-%m-%d %H:%M {tz_label}")
    report_date = now.strftime("%B %d, %Y")

    # Build section HTML — skip the bare Risk Assessment line since it's in the header badge
    sections_html = ""
    for i, (heading, body) in enumerate(sections):
        # Skip sections with no heading (stray text between H1 and first H2)
        if not heading.strip():
            continue
        # Skip the risk assessment section — it's displayed in the badge
        if re.match(r"risk assessment", heading, re.IGNORECASE) and not body.strip():
            continue
        # Inject the interactive transaction table into the Wire Transactions section
        if re.search(r"transaction", heading, re.IGNORECASE):
            tx_html = _build_tx_table(sql_results or [])
            if tx_html:
                body = body + tx_html
        sections_html += _render_section(heading, body, i)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Real Time AML Agent Report — {_escape(display_subject)}</title>
<style>
{_css(risk_color)}
</style>
</head>
<body>
<div class="page">

  <div class="header">
    <div class="header-logo">Real Time AML Agent &nbsp;·&nbsp; Report</div>
    <div class="header-subject">{_escape(display_subject)}</div>
    <div class="risk-badge-wrap">
      <h1>Real Time AML Agent Report</h1>
      <div class="risk-badge">
        <span class="risk-dot"></span>
        {_escape(risk_label)}
      </div>
    </div>
    <div class="header-meta">
      <span>📅 {_escape(report_date)}</span>
      <span>⚙️ Generated {_escape(generated_at)}</span>
    </div>
  </div>

  <div class="divider"></div>

  <div class="content">
    {sections_html}
  </div>

  <div class="footer">
    <span>Real Time AML Agent &nbsp;·&nbsp; Confidential &amp; Privileged</span>
    <span>Report generated: {_escape(generated_at)}{(" &nbsp;·&nbsp; version: " + _escape(version)) if version else ""}</span>
  </div>

</div>
<script>{_JS}</script>
</body>
</html>"""
