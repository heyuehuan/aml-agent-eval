#!/usr/bin/env python3
"""Run all test cases and write HTML + artifacts to output/test_cases/test_run/."""

import asyncio
import csv
import json
import re
import sys
from pathlib import Path

from aml_agent.report_html import render_html
from aml_agent.runner import run_investigation

CSV_PATH = Path("data/test_cases/test_cases.csv")
OUT_DIR = Path("output/test_cases/test_run")


async def main(filter_ids: list[str] | None = None):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if not filter_ids or r["test_case_id"] in filter_ids]

    print(f"Running {len(rows)} test case(s) → {OUT_DIR}\n")

    for i, row in enumerate(rows, 1):
        tc_id = row["test_case_id"]
        subject = row["test_case_info_input"]

        print(f"[{i}/{len(rows)}] {tc_id}: {subject[:70].replace(chr(10), ' ')}")

        try:
            report, artifacts = await run_investigation(subject)
        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            continue

        m = re.search(r"#\s+AML Investigation Report:\s*(.+)", report)
        display_subject = m.group(1).strip() if m else subject

        html = render_html(
            report,
            subject=display_subject,
            sql_results=artifacts.get("sql_results", []),
            version=artifacts.get("version", "unknown"),
        )

        stem = f"Test_{tc_id}"
        (OUT_DIR / f"{stem}.html").write_text(html, encoding="utf-8")
        (OUT_DIR / f"{stem}.artifacts.json").write_text(
            json.dumps(artifacts, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"  → {OUT_DIR / stem}.html\n")


if __name__ == "__main__":
    ids = [a for a in sys.argv[1:] if a.startswith("TC-")] or None
    asyncio.run(main(ids))
