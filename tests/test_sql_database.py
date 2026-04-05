"""Tests for the ReadOnlySqlDatabase tool."""

import sqlite3
import tempfile
from pathlib import Path

import pytest

from aml_agent.tools.sql_database import ReadOnlySqlDatabase, ReadOnlySqlPolicy


@pytest.fixture
def sample_db():
    """Create a temporary SQLite database with sample data."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE transactions ("
        "  transaction_id TEXT PRIMARY KEY,"
        "  amount REAL,"
        "  currency TEXT,"
        "  sender_name TEXT,"
        "  receiver_name TEXT,"
        "  memo TEXT"
        ")"
    )
    conn.executemany(
        "INSERT INTO transactions VALUES (?, ?, ?, ?, ?, ?)",
        [
            ("tx-001", 10000.00, "USD", "Alice Smith", "Bob Jones", "Payment for services"),
            ("tx-002", 50000.00, "CAD", "Charlie Brown", "Diana Prince", "Wire transfer"),
            ("tx-003", 25000.00, "USD", "Eve Wilson", "Alice Smith", "Invoice settlement"),
            ("tx-004", 75000.00, "EUR", "Frank Miller", "Grace Hopper", "Consulting fee"),
            ("tx-005", 5000.00, "USD", "Alice Smith", "Hank Aaron", "Monthly payment"),
        ],
    )
    conn.commit()
    conn.close()

    yield db_path

    Path(db_path).unlink(missing_ok=True)


@pytest.fixture
def db_tool(sample_db):
    """Create a ReadOnlySqlDatabase instance for the sample database."""
    tool = ReadOnlySqlDatabase(
        connection_uri=f"sqlite:///{sample_db}",
        max_rows=50,
        agent_name="TestAgent",
    )
    yield tool
    tool.close()


class TestReadOnlySqlDatabase:
    """Tests for ReadOnlySqlDatabase."""

    def test_init_valid(self, db_tool):
        """Tool initializes correctly with valid params."""
        assert db_tool.agent_name == "TestAgent"
        assert db_tool.max_rows == 50

    def test_init_invalid_uri(self):
        """Raises ValueError for empty connection URI."""
        with pytest.raises(ValueError, match="connection_uri"):
            ReadOnlySqlDatabase(connection_uri="", agent_name="Test")

    def test_init_invalid_max_rows(self):
        """Raises ValueError for non-positive max_rows."""
        with pytest.raises(ValueError, match="max_rows"):
            ReadOnlySqlDatabase(connection_uri="sqlite:///test.db", max_rows=0)

    def test_init_invalid_agent_name(self):
        """Raises ValueError for empty agent name."""
        with pytest.raises(ValueError, match="agent_name"):
            ReadOnlySqlDatabase(connection_uri="sqlite:///test.db", agent_name="")

    def test_init_invalid_policy_type(self):
        """Raises TypeError for wrong policy type."""
        with pytest.raises(TypeError, match="policy"):
            ReadOnlySqlDatabase(
                connection_uri="sqlite:///test.db",
                policy="not_a_policy",  # type: ignore
            )

    def test_get_schema_info(self, db_tool):
        """get_schema_info returns table and column information."""
        schema = db_tool.get_schema_info()
        assert "transactions" in schema
        assert "transaction_id" in schema
        assert "amount" in schema
        assert "sender_name" in schema

    def test_get_schema_info_filtered(self, db_tool):
        """get_schema_info with specific table names."""
        schema = db_tool.get_schema_info(table_names=["transactions"])
        assert "transactions" in schema

    def test_execute_select(self, db_tool):
        """Basic SELECT query works."""
        result = db_tool.execute("SELECT * FROM transactions WHERE amount > 20000")
        assert "tx-002" in result
        assert "tx-003" in result
        assert "tx-004" in result
        assert "50000" in result

    def test_execute_count(self, db_tool):
        """Aggregate queries work."""
        result = db_tool.execute("SELECT COUNT(*) as cnt FROM transactions")
        assert "5" in result

    def test_execute_like(self, db_tool):
        """LIKE pattern matching works."""
        result = db_tool.execute(
            "SELECT transaction_id FROM transactions WHERE sender_name LIKE '%Alice%'"
        )
        assert "tx-001" in result
        assert "tx-005" in result

    def test_execute_sum(self, db_tool):
        """SUM aggregate works."""
        result = db_tool.execute(
            "SELECT SUM(amount) as total FROM transactions WHERE currency = 'USD'"
        )
        assert "total" in result

    def test_max_rows_limit(self, sample_db):
        """Results are truncated at max_rows."""
        tool = ReadOnlySqlDatabase(
            connection_uri=f"sqlite:///{sample_db}",
            max_rows=2,
            agent_name="TestAgent",
        )
        result = tool.execute("SELECT * FROM transactions")
        assert "Truncated at 2 rows" in result
        tool.close()


class TestReadOnlySafety:
    """Tests for read-only SQL enforcement."""

    def test_block_insert(self, db_tool):
        """INSERT is blocked."""
        result = db_tool.execute(
            "INSERT INTO transactions VALUES ('tx-bad', 100, 'USD', 'Bad', 'Guy', 'Evil')"
        )
        assert "Security Violation" in result or "Query Error" in result

    def test_block_update(self, db_tool):
        """UPDATE is blocked."""
        result = db_tool.execute("UPDATE transactions SET amount = 0 WHERE transaction_id = 'tx-001'")
        assert "Security Violation" in result or "Query Error" in result

    def test_block_delete(self, db_tool):
        """DELETE is blocked."""
        result = db_tool.execute("DELETE FROM transactions")
        assert "Security Violation" in result or "Query Error" in result

    def test_block_drop(self, db_tool):
        """DROP TABLE is blocked."""
        result = db_tool.execute("DROP TABLE transactions")
        assert "Security Violation" in result or "Query Error" in result

    def test_block_create(self, db_tool):
        """CREATE TABLE is blocked."""
        result = db_tool.execute("CREATE TABLE evil (id TEXT)")
        assert "Security Violation" in result or "Query Error" in result

    def test_block_multiple_statements(self, db_tool):
        """Multiple statements are blocked."""
        result = db_tool.execute("SELECT 1; DROP TABLE transactions")
        assert "Security Violation" in result or "Query Error" in result

    def test_block_subquery_write(self, db_tool):
        """Write operations hidden in subqueries are blocked."""
        result = db_tool.execute(
            "SELECT * FROM transactions WHERE transaction_id IN "
            "(SELECT transaction_id FROM transactions); DELETE FROM transactions"
        )
        assert "Security Violation" in result or "Query Error" in result

    def test_allow_union(self, db_tool):
        """UNION queries are allowed."""
        result = db_tool.execute(
            "SELECT transaction_id FROM transactions WHERE amount > 50000 "
            "UNION "
            "SELECT transaction_id FROM transactions WHERE currency = 'EUR'"
        )
        assert "tx-002" in result or "tx-004" in result


class TestReadOnlySqlPolicy:
    """Tests for ReadOnlySqlPolicy configuration."""

    def test_default_policy(self):
        """Default policy has expected settings."""
        policy = ReadOnlySqlPolicy()
        assert "select" in policy.allowed_roots
        assert "union" in policy.allowed_roots
        assert "insert" in policy.forbidden_nodes
        assert "delete" in policy.forbidden_nodes
        assert policy.allow_multiple_statements is False

    def test_custom_policy(self, sample_db):
        """Custom policy is respected."""
        policy = ReadOnlySqlPolicy(allow_multiple_statements=True)
        tool = ReadOnlySqlDatabase(
            connection_uri=f"sqlite:///{sample_db}",
            policy=policy,
            agent_name="TestAgent",
        )
        # Multiple SELECT statements should be allowed
        assert tool._is_safe_readonly_query("SELECT 1; SELECT 2") is True
        tool.close()
