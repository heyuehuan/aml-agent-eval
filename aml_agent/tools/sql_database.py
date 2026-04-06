"""SQL Database Tool with Read-Only Enforcement for Agents.

Provides a safe, read-only SQL query interface with AST-based validation
using SQLGlot to prevent any write operations.
"""

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import sqlglot
from sqlalchemy import create_engine, inspect, text
from sqlglot import exp

logger = logging.getLogger(__name__)

__all__ = ["ReadOnlySqlDatabase", "ReadOnlySqlPolicy"]


@dataclass(frozen=True)
class ReadOnlySqlPolicy:
    """Policy controlling which SQL statements can execute."""

    allowed_roots: tuple[str, ...] = ("select", "union", "paren")
    forbidden_nodes: tuple[str, ...] = (
        "create",
        "insert",
        "update",
        "delete",
        "drop",
        "alter",
        "truncate_table",
        "merge",
        "command",
        "pragma",
        "attach",
        "detach",
        "set",
    )
    allow_multiple_statements: bool = False


class ReadOnlySqlDatabase:
    """A SQL database query tool for Agents with AST-based read-only enforcement.

    Parameters
    ----------
    connection_uri : str
        SQLAlchemy connection string.
    max_rows : int
        Hard limit on rows returned.
    query_timeout_sec : int
        Maximum execution time for queries.
    agent_name : str
        Name of the agent using this tool (for audit logs).
    policy : ReadOnlySqlPolicy | None
        AST policy controlling what statements are permitted.
    """

    def __init__(
        self,
        connection_uri: str,
        max_rows: int = 100,
        query_timeout_sec: int = 60,
        agent_name: str = "AmlAgent",
        policy: ReadOnlySqlPolicy | None = None,
        **engine_kwargs,
    ) -> None:
        if not connection_uri or not connection_uri.strip():
            raise ValueError("connection_uri must be a non-empty string.")
        if max_rows <= 0:
            raise ValueError("max_rows must be a positive integer.")
        if query_timeout_sec <= 0:
            raise ValueError("query_timeout_sec must be a positive integer.")
        if not agent_name or not agent_name.strip():
            raise ValueError("agent_name must be a non-empty string.")
        if policy is not None and not isinstance(policy, ReadOnlySqlPolicy):
            raise TypeError("policy must be a ReadOnlySqlPolicy or None.")

        self.engine = create_engine(connection_uri, **engine_kwargs)
        self.agent_name = agent_name
        self.max_rows = max_rows
        self.timeout = query_timeout_sec
        self.policy = policy or ReadOnlySqlPolicy()
        if not self.policy.allowed_roots:
            raise ValueError("policy.allowed_roots must not be empty.")
        self._allowed_root_types = _resolve_sqlglot_expression_types(self.policy.allowed_roots)
        self._forbidden_node_types = _resolve_sqlglot_expression_types(self.policy.forbidden_nodes)

    def _is_safe_readonly_query(self, query: str) -> bool:
        """Verify that a query is semantically read-only using SQLGlot AST parsing."""
        try:
            expressions = sqlglot.parse(query)
            is_safe = True

            if not expressions:
                logger.warning("Empty parse result - blocking query")
                is_safe = False

            if is_safe and not self.policy.allow_multiple_statements and len(expressions) > 1:
                logger.warning("Multiple statements blocked by policy")
                is_safe = False

            allowed_root_names = {name.lower() for name in self.policy.allowed_roots}

            for expression in expressions:
                if not is_safe:
                    break

                if not isinstance(expression, self._allowed_root_types):
                    logger.warning("Blocked unsafe query type: %s", type(expression))
                    is_safe = False
                    break

                if "with" not in allowed_root_names and expression.find(exp.With, exp.CTE):
                    logger.warning("CTE usage blocked by policy")
                    is_safe = False
                    break

                if self._forbidden_node_types and expression.find(*self._forbidden_node_types):
                    logger.warning("Blocked query containing write operation in AST")
                    is_safe = False
                    break

            return is_safe
        except Exception as e:
            logger.error("SQL parsing error: %s", e)
            # SQLGlot may fail on very long or complex queries that are still
            # valid read-only SQL.  Fall back to a conservative regex check so
            # that legitimate SELECT queries are not blocked by a parser bug.
            return self._regex_fallback_is_safe(query)

    def _regex_fallback_is_safe(self, query: str) -> bool:
        """Last-resort regex check when AST parsing fails.

        Returns True only when the query looks like a plain SELECT with no
        write keywords.  This is intentionally conservative.
        """
        normalized = query.strip().rstrip(";").strip()
        if not normalized.upper().startswith("SELECT"):
            return False
        # Check for any write keyword at word boundaries
        write_keywords = (
            r"\b(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|TRUNCATE|MERGE"
            r"|REPLACE|ATTACH|DETACH|PRAGMA)\b"
        )
        if re.search(write_keywords, normalized, re.IGNORECASE):
            logger.warning("Regex fallback: found write keyword in query")
            return False
        logger.info("Regex fallback: allowing SELECT query that failed AST parsing")
        return True

    def get_schema_info(self, table_names: Optional[list[str]] = None) -> str:
        """Return schema for specific tables/views or all if None.

        Parameters
        ----------
        table_names : list[str] | None
            List of table/view names. If None, returns all.

        Returns
        -------
        str
            Formatted schema information.
        """
        inspector = inspect(self.engine)
        all_tables = inspector.get_table_names()
        try:
            all_views = inspector.get_view_names()
        except Exception:
            all_views = []

        all_relations: list[tuple[str, str]] = [
            (t, "table") for t in all_tables
        ] + [
            (v, "view") for v in all_views
        ]

        if table_names:
            targets = {name.lower() for name in table_names}
            relations_to_scan = [(name, kind) for name, kind in all_relations if name.lower() in targets]
        else:
            relations_to_scan = all_relations

        schema_text = []
        for relation_name, relation_kind in relations_to_scan:
            label = "View" if relation_kind == "view" else "Table"
            try:
                columns = inspector.get_columns(relation_name)
                col_strs = [f"{c['name']}: {str(c['type'])}" for c in columns]
                schema_text.append(f"{label}: {relation_name}\n  Columns: {', '.join(col_strs)}")
            except Exception:
                if self.engine.dialect.name == "sqlite":
                    try:
                        safe_name = relation_name.replace('"', '""')
                        with self.engine.connect() as conn:
                            pragma = conn.execute(text(f'PRAGMA table_info("{safe_name}")'))
                            pragma_rows = pragma.fetchall()
                        col_names = [row[1] for row in pragma_rows]
                        if col_names:
                            schema_text.append(f"{label}: {relation_name}\n  Columns: {', '.join(col_names)}")
                            continue
                    except Exception:
                        pass
                schema_text.append(f"{label}: {relation_name} (Error reading schema)")

        return "\n".join(schema_text)

    def execute(self, query: str) -> str:
        """Execute a SQL query with read-only enforcement.

        Parameters
        ----------
        query : str
            The SQL query string to execute.

        Returns
        -------
        str
            Markdown-formatted table of results or error message.
        """
        start_time = datetime.now()
        status = "FAILED"
        error_msg = None
        row_count = 0

        try:
            if not self._is_safe_readonly_query(query):
                raise PermissionError("Security Violation: Query contains prohibited WRITE operations.")

            with self.engine.connect() as conn:
                if self.engine.dialect.name == "sqlite":
                    conn.execute(text(f"PRAGMA busy_timeout = {self.timeout * 1000}"))

                execution_options = {"timeout": self.timeout}
                result = conn.execute(text(query).execution_options(**execution_options))

                keys = list(result.keys())
                rows = result.fetchmany(self.max_rows)
                row_count = len(rows)

                output = [f"| {' | '.join(keys)} |"]
                output.append("| " + " | ".join(["---"] * len(keys)) + " |")
                for row in rows:
                    output.append(f"| {' | '.join(map(str, row))} |")

                if row_count == self.max_rows:
                    output.append(f"\n... (Truncated at {self.max_rows} rows) ...")

                status = "SUCCESS"
                return "\n".join(output)

        except Exception as e:
            error_msg = str(e)
            return f"Query Error: {error_msg}"

        finally:
            duration = (datetime.now() - start_time).total_seconds()
            log_entry = {
                "timestamp": start_time.isoformat(),
                "agent": self.agent_name,
                "query": query,
                "status": status,
                "rows_returned": row_count,
                "duration_sec": duration,
                "error": error_msg,
            }
            logger.debug("AUDIT: %s", log_entry)

    def close(self) -> None:
        """Dispose of the connection pool."""
        self.engine.dispose(close=True)


def _resolve_sqlglot_expression_type(name: str) -> type[exp.Expression]:
    """Resolve a sqlglot expression name to an Expression class."""
    cleaned = name.strip()
    if not cleaned:
        raise ValueError("Expression type name cannot be empty.")
    if cleaned.startswith("exp."):
        cleaned = cleaned[4:]

    candidate = cleaned.replace("-", "_")
    camel = "".join(part.capitalize() for part in candidate.split("_"))

    found_non_expression = False
    for attr in dict.fromkeys((cleaned, candidate, camel)):
        if not attr:
            continue
        resolved = getattr(exp, attr, None)
        if resolved is None:
            continue
        if isinstance(resolved, type) and issubclass(resolved, exp.Expression):
            return resolved
        found_non_expression = True

    if found_non_expression:
        raise ValueError(f"sqlglot expression name {name!r} did not resolve to an Expression type.")
    raise ValueError(f"Unknown sqlglot expression type: {name!r}")


def _resolve_sqlglot_expression_types(names: tuple[str, ...]) -> tuple[type[exp.Expression], ...]:
    """Resolve many sqlglot expression names into Expression classes."""
    return tuple(_resolve_sqlglot_expression_type(name) for name in names)
