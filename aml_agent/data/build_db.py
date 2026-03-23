"""Build the SQLite database from the transactions CSV.

This script creates aml_agent/data/aml_transactions.db with:
- A `transactions` table loaded from data/transactions.csv (text fields only)

Entity IDs are intentionally excluded — they are ground-truth evaluation labels,
not data the agent should see. Watchlist lookup is handled separately by the
KB search tool via internal_kb_watchlist.json.
"""

import csv
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

SCHEMA_DDL = """\
DROP TABLE IF EXISTS "transactions";
CREATE TABLE "transactions" (
    "transaction_id" TEXT PRIMARY KEY,
    "amount" REAL NOT NULL,
    "currency" TEXT NOT NULL,
    "transaction_datetime" TEXT NOT NULL,
    "sender_name" TEXT,
    "sender_address" TEXT,
    "receiver_name" TEXT,
    "receiver_address" TEXT,
    "memo" TEXT
);

CREATE INDEX IF NOT EXISTS idx_transactions_sender ON transactions(sender_name);
CREATE INDEX IF NOT EXISTS idx_transactions_receiver ON transactions(receiver_name);
CREATE INDEX IF NOT EXISTS idx_transactions_datetime ON transactions(transaction_datetime);
"""


def build_database(
    db_path: str | None = None,
    csv_path: str | None = None,
) -> Path:
    """Build the SQLite database.

    Parameters
    ----------
    db_path : str | None
        Output database path. Defaults to aml_agent/data/aml_transactions.db.
    csv_path : str | None
        Path to transactions.csv.

    Returns
    -------
    Path
        Path to the created database file.
    """
    db_path = Path(db_path or PROJECT_ROOT / "aml_agent" / "data" / "aml_transactions.db")
    csv_path = Path(csv_path or PROJECT_ROOT / "data" / "transactions.csv")

    if not csv_path.exists():
        raise FileNotFoundError(f"Transactions CSV not found: {csv_path}")

    # Ensure output directory exists
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove existing DB
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Create schema
    cursor.executescript(SCHEMA_DDL)

    # Load transactions (text fields only; entity IDs are excluded)
    print(f"Loading transactions from {csv_path}...")
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            rows.append((
                row["transaction_id"],
                float(row["amount"]),
                row["currency"],
                row["transaction_datetime"],
                row["sender_name"],
                row.get("sender_address", ""),
                row["receiver_name"],
                row.get("receiver_address", ""),
                row.get("memo", ""),
            ))

    cursor.executemany(
        "INSERT INTO transactions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    print(f"  Loaded {len(rows)} transactions.")

    conn.commit()

    # Verify
    cursor.execute("SELECT COUNT(*) FROM transactions")
    tx_count = cursor.fetchone()[0]

    conn.close()

    print(f"\nDatabase created at: {db_path}")
    print(f"  Transactions: {tx_count}")

    return db_path


if __name__ == "__main__":
    db_out = sys.argv[1] if len(sys.argv) > 1 else None
    build_database(db_path=db_out)
