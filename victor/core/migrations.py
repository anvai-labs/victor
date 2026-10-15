# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
# Licensed under the Apache License, Version 2.0.
"""Atomic schema upgrades for the global and project SQLite stores.

Historical import/consolidation jobs are separate from these schema upgrades.
Only trusted, repository-owned SQL belongs in this executor.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from uuid import uuid4

from victor.core import schema
from victor.core.sql_utils import quote_identifier

logger = logging.getLogger(__name__)


def apply_sql(conn: sqlite3.Connection, sql: str) -> None:
    """Execute a SQL block without executescript's implicit transaction commit.

    SQLite's completeness parser handles quoted semicolons and trigger bodies.
    Transaction control and connection PRAGMAs are outside migration SQL's
    contract; only the runner controls the transaction. Exceptions propagate.
    """
    start = 0
    statements = []
    for end, char in enumerate(sql):
        if char == ";" and sqlite3.complete_statement(sql[start : end + 1]):
            statements.append(sql[start : end + 1])
            start = end + 1
    if sql[start:].strip():
        statements.append(sql[start:])
    for statement in statements:
        # Strip leading comments only, not comment-like text within SQL strings.
        head = re.sub(r"\A(?:\s+|--[^\n]*(?:\n|$)|/\*.*?\*/)*", "", statement, flags=re.S)
        if not head.strip("; \n\t\r"):
            continue
        keyword = re.match(r"[A-Za-z]+", head)
        if not keyword or keyword[0].upper() not in {
            "CREATE",
            "ALTER",
            "DROP",
            "INSERT",
            "UPDATE",
            "DELETE",
            "REPLACE",
        }:
            raise sqlite3.OperationalError("Unsupported migration SQL statement")
        conn.execute(statement)


@contextmanager
def _transaction(conn: sqlite3.Connection) -> Iterator[None]:
    # Serialize independent migrators before reading the version. Nested callers
    # keep ownership; releasing a savepoint must never commit their transaction.
    nested = conn.in_transaction
    name = "migration_" + uuid4().hex
    conn.execute(f"SAVEPOINT {name}" if nested else "BEGIN IMMEDIATE")
    try:
        yield
        conn.execute(f"RELEASE SAVEPOINT {name}" if nested else "COMMIT")
    except BaseException:
        if nested:
            conn.execute(f"ROLLBACK TO SAVEPOINT {name}")
            conn.execute(f"RELEASE SAVEPOINT {name}")
        else:
            conn.rollback()
        raise


def _exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT type FROM sqlite_master WHERE name=? AND type IN ('table', 'view', 'index')",
        (table,),
    ).fetchone()
    if row is None:
        return False
    if row[0] != "table":
        raise ValueError(f"Expected table {table}, found {row[0]}")
    return True


def _columns(conn: sqlite3.Connection, table: str) -> list[tuple[object, ...]]:
    return [tuple(row)[1:] for row in conn.execute(f"PRAGMA table_info({quote_identifier(table)})")]


def _merge_metadata(conn: sqlite3.Connection) -> None:
    """Preserve legacy metadata; never guess which conflicting value is right."""
    if not _exists(conn, "_db_metadata"):
        return
    names = {col[0] for col in _columns(conn, "_db_metadata")}
    if not {"key", "value"} <= names:
        raise ValueError("Legacy metadata requires key and value columns")
    seen: set[str] = set()
    for key, value in conn.execute("SELECT key, value FROM _db_metadata"):
        if not isinstance(key, str) or key in seen:
            raise ValueError("Legacy metadata contains invalid or duplicate keys")
        seen.add(key)
        if key == "schema_version":
            continue  # Only the executor publishes schema_version.
        prior = conn.execute("SELECT value FROM sys_metadata WHERE key=?", (key,)).fetchone()
        if prior is not None and prior[0] != value:
            raise ValueError(f"Conflicting legacy metadata key: {key}")
        if prior is None:
            conn.execute(
                "INSERT INTO sys_metadata (key, value, updated_at) VALUES (?, ?, datetime('now'))",
                (key, value),
            )
    # Keep the original legacy table for reconciliation/audit; no data is dropped.


def _reset_empty_delta(conn: sqlite3.Connection) -> None:
    table = schema.Tables.LOCAL_CLASSIFIER_DELTA
    actual = _columns(conn, table)
    with closing(sqlite3.connect(":memory:")) as reference:
        reference.execute(schema.Schema.LOCAL_CLASSIFIER_DELTA)
        expected = _columns(reference, table)
    if actual == expected:
        return
    # Recognize the historical scalar layout, not arbitrary empty user tables.
    legacy_types = {
        "decision_type": "TEXT",
        "feature_hash": "INTEGER",
        "weight": "REAL",
        "samples": "INTEGER",
        "sum_reward": "REAL",
        "updated_at": "TEXT",
    }
    shape = {col[0]: col[1] for col in actual}
    primary_key = {col[0]: col[4] for col in actual if col[4]}
    if shape != legacy_types or primary_key != {"decision_type": 1, "feature_hash": 2}:
        raise ValueError("Unrecognized local_classifier_delta schema; reconcile before migration")
    if conn.execute(f"SELECT 1 FROM {quote_identifier(table)} LIMIT 1").fetchone() is not None:
        raise ValueError("Populated legacy local_classifier_delta requires reconciliation")
    conn.execute(f"DROP TABLE {quote_identifier(table)}")


def _apply_step(conn: sqlite3.Connection, step: schema.MigrationStep) -> None:
    if step.table is not None and not _exists(conn, step.table):
        logger.debug("Skipping migration for absent optional %s table: %s", step.scope, step.table)
        return
    match step.operation:
        case "metadata":
            _merge_metadata(conn)
        case "rename":
            if step.target is None or step.table is None:
                raise ValueError("Rename migration requires source and target")
            if _exists(conn, step.target):
                raise ValueError(f"Conflicting migration tables: {step.table}, {step.target}")
            apply_sql(conn, step.sql)
        case "add_text":
            if step.target is None or step.table is None:
                raise ValueError("Column migration requires table and target")
            existing = next(
                (col for col in _columns(conn, step.table) if col[0] == step.target), None
            )
            if existing is None:
                apply_sql(conn, step.sql)
            elif existing != (step.target, "TEXT", 0, None, 0):
                raise ValueError(f"Incompatible existing column: {step.table}.{step.target}")
        case "reset_empty_delta":
            _reset_empty_delta(conn)
        case "sql":
            apply_sql(conn, step.sql)
        case _:
            raise ValueError(f"Unknown migration operation: {step.operation}")


def migrate(
    conn: sqlite3.Connection, scope: schema.MigrationScope, *, initialize: bool = False
) -> None:
    """Upgrade and publish the version atomically, preserving caller ownership.

    Project initialization runs migrations before current-schema indexes, within
    the same transaction. Global tables retain their lazy initialization policy.
    An unversioned modern store (no legacy sentinel) retains the historical
    current-version adoption policy; this is not exhaustive schema validation.
    """
    metadata = "sys_metadata" if scope == "global" else "_project_metadata"
    with _transaction(conn):
        conn.execute(
            schema.Schema.SYS_METADATA
            if scope == "global"
            else "CREATE TABLE IF NOT EXISTS _project_metadata "
            "(key TEXT PRIMARY KEY, value TEXT, updated_at TEXT)"
        )
        row = conn.execute(f"SELECT value FROM {metadata} WHERE key='schema_version'").fetchone()
        version = schema.CURRENT_SCHEMA_VERSION
        if row is not None:
            version = int(row[0])
        elif _exists(conn, "_db_metadata"):
            legacy = conn.execute(
                "SELECT value FROM _db_metadata WHERE key='schema_version'"
            ).fetchall()
            if len(legacy) > 1:
                raise ValueError("Legacy metadata contains duplicate schema_version keys")
            version = int(legacy[0][0]) if legacy else 0
        if version < 0 or version > schema.CURRENT_SCHEMA_VERSION:
            raise ValueError(f"Unsupported schema version {version}; newer or invalid database")
        for step in schema.get_migration_steps(version, schema.CURRENT_SCHEMA_VERSION):
            if step.scope == scope:
                _apply_step(conn, step)
        if initialize and scope == "project":
            for sql in schema.Schema.get_project_schemas() + schema.Schema.get_project_indexes():
                apply_sql(conn, sql)
        if row is None or version < schema.CURRENT_SCHEMA_VERSION:
            conn.execute(
                f"INSERT INTO {metadata} (key, value, updated_at) "
                "VALUES ('schema_version', ?, datetime('now')) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                (str(schema.CURRENT_SCHEMA_VERSION),),
            )
