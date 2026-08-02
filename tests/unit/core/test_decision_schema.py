# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""FEP-0012 Phase 2b: the decision-learning schema (decision_log,
decision_outcome global; local_classifier_delta project) is registered, migrated
to v8, and yields valid, queryable tables."""

import sqlite3

import pytest

from victor.core.schema import (
    CURRENT_SCHEMA_VERSION,
    Schema,
    Tables,
    get_migration_sql,
)


def test_table_constants_defined():
    assert Tables.DECISION_LOG == "decision_log"
    assert Tables.DECISION_OUTCOME == "decision_outcome"
    assert Tables.LOCAL_CLASSIFIER_DELTA == "local_classifier_delta"


def test_schema_version_bumped_to_9():
    assert CURRENT_SCHEMA_VERSION == 9


def test_global_tables_registered_in_get_all_schemas():
    stmts = Schema.get_all_schemas()
    assert Schema.DECISION_LOG in stmts
    assert Schema.DECISION_OUTCOME in stmts
    # The delta is a PROJECT table — must NOT be in the global list.
    assert Schema.LOCAL_CLASSIFIER_DELTA not in stmts


def test_delta_registered_in_get_project_schemas():
    proj = Schema.get_project_schemas()
    assert Schema.LOCAL_CLASSIFIER_DELTA in proj
    # Global decision tables must NOT be in the project list.
    assert Schema.DECISION_LOG not in proj
    assert Schema.DECISION_OUTCOME not in proj


def test_v8_migration_creates_all_three_tables():
    sqls = get_migration_sql(7, 8)
    # Three single CREATE TABLE statements (indexes are applied separately).
    assert Schema.DECISION_LOG in sqls
    assert Schema.DECISION_OUTCOME in sqls
    assert Schema.LOCAL_CLASSIFIER_DELTA in sqls


def _apply(conn: sqlite3.Connection, statements):
    for s in statements:
        conn.execute(s)
    conn.commit()


def test_global_tables_create_and_query():
    conn = sqlite3.connect(":memory:")
    _apply(conn, [Schema.SYS_METADATA])
    _apply(conn, Schema.get_all_schemas())
    for idx in Schema.get_all_indexes():
        conn.executescript(idx)
    conn.commit()

    # decision_log round-trip with the FEP-0012 correlation fields.
    conn.execute(
        f"""INSERT INTO {Tables.DECISION_LOG}
            (decision_id, decision_type, session_id, turn_id, source,
             confidence, model_version, feature_spec_version, feature_digest,
             context, result)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        (
            "d1",
            "task_type_classification",
            "sess-1",
            "turn-1",
            "local_classifier",
            0.8,
            "edge_v1",
            "1",
            "abc",
            "{}",
            "debug",
        ),
    )
    row = conn.execute(
        f"SELECT session_id, turn_id, model_version FROM {Tables.DECISION_LOG} "
        f"WHERE decision_id='d1'"
    ).fetchone()
    assert row == ("sess-1", "turn-1", "edge_v1")

    # decision_outcome round-trip with attributed_reward.
    conn.execute(
        f"""INSERT INTO {Tables.DECISION_OUTCOME}
            (decision_id, session_id, success, quality_score, attributed_reward,
             credit_method)
            VALUES (?,?,?,?,?,?)""",
        ("d1", "sess-1", 1, 0.9, 0.7, "gae"),
    )
    row = conn.execute(
        f"SELECT attributed_reward FROM {Tables.DECISION_OUTCOME} WHERE decision_id='d1'"
    ).fetchone()
    assert row[0] == pytest.approx(0.7)
    conn.close()


def test_v9_migration_drops_and_recreates_delta_per_label():
    """v8->v9 DROPs the old scalar delta table and recreates it per-label."""
    conn = sqlite3.connect(":memory:")
    # Simulate a v8-shaped project DB with the OLD scalar delta table + a row.
    conn.execute(
        "CREATE TABLE local_classifier_delta ("
        "decision_type TEXT, feature_hash INTEGER, weight REAL, samples INTEGER,"
        "sum_reward REAL, updated_at TEXT, PRIMARY KEY (decision_type, feature_hash))"
    )
    conn.execute("INSERT INTO local_classifier_delta VALUES ('x', 1, 0.5, 1, 0.5, 't')")
    conn.commit()

    # Apply the 8->9 migration.
    _apply(conn, get_migration_sql(8, 9))

    # New per-label columns exist; the old scalar row is gone (DROP+CREATE).
    cols = [r[1] for r in conn.execute("PRAGMA table_info(local_classifier_delta)").fetchall()]
    assert "label" in cols and "feature_spec_version" in cols
    assert conn.execute("SELECT COUNT(*) FROM local_classifier_delta").fetchone()[0] == 0

    # A per-label row inserts cleanly under the new 4-column PK.
    conn.execute(
        f"INSERT INTO {Tables.LOCAL_CLASSIFIER_DELTA}"
        " (decision_type, feature_hash, label, weight, feature_spec_version)"
        " VALUES (?,?,?,?,?)",
        ("task_completion", 7, "pass", 0.3, "1"),
    )
    # Same (type, hash, label, spec) dedups; a different label coexists.
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            f"INSERT INTO {Tables.LOCAL_CLASSIFIER_DELTA}"
            " (decision_type, feature_hash, label, weight, feature_spec_version)"
            " VALUES (?,?,?,?,?)",
            ("task_completion", 7, "pass", 0.9, "1"),
        )
    conn.execute(
        f"INSERT INTO {Tables.LOCAL_CLASSIFIER_DELTA}"
        " (decision_type, feature_hash, label, weight, feature_spec_version)"
        " VALUES (?,?,?,?,?)",
        ("task_completion", 7, "fail", -0.2, "1"),
    )
    assert conn.execute("SELECT COUNT(*) FROM local_classifier_delta").fetchone()[0] == 2
    conn.close()


def test_project_delta_table_creates_and_query():
    conn = sqlite3.connect(":memory:")
    _apply(conn, Schema.get_project_schemas())
    for idx in Schema.get_project_indexes():
        conn.executescript(idx)
    conn.commit()

    conn.execute(
        f"""INSERT INTO {Tables.LOCAL_CLASSIFIER_DELTA}
            (decision_type, feature_hash, label, weight, samples, sum_reward,
             feature_spec_version)
            VALUES (?,?,?,?,?,?,?)""",
        ("task_completion", 12345, "pass", 0.42, 7, 5.5, "1"),
    )
    # Composite PK dedups on (decision_type, feature_hash, label, feature_spec_version).
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            f"""INSERT INTO {Tables.LOCAL_CLASSIFIER_DELTA}
                (decision_type, feature_hash, label, weight, feature_spec_version)
                VALUES (?, ?, ?, ?, ?)""",
            ("task_completion", 12345, "pass", 0.9, "1"),
        )
    # A different label for the same hash coexists (per-label storage).
    conn.execute(
        f"""INSERT INTO {Tables.LOCAL_CLASSIFIER_DELTA}
            (decision_type, feature_hash, label, weight, feature_spec_version)
            VALUES (?, ?, ?, ?, ?)""",
        ("task_completion", 12345, "fail", -0.1, "1"),
    )
    rows = conn.execute(
        f"SELECT label, weight, samples FROM {Tables.LOCAL_CLASSIFIER_DELTA} "
        f"WHERE decision_type='task_completion' ORDER BY label"
    ).fetchall()
    assert rows == [("fail", pytest.approx(-0.1), 0), ("pass", pytest.approx(0.42), 7)]
    conn.close()
