# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""R1 correlation spine — rl_outcome schema migration + outcome correlation helper."""

from __future__ import annotations

from victor.framework.rl.coordinator import RLCoordinator, _correlation_for_outcome


def _rl_outcome_columns(coord) -> set:
    cur = coord.db.cursor()
    cur.execute("PRAGMA table_info(rl_outcome)")
    return {row[1] for row in cur.fetchall()}


def test_rl_outcome_has_correlation_columns(tmp_path):
    db_path = tmp_path / "g.db"
    coord = RLCoordinator(storage_path=tmp_path, db_path=db_path)
    cols = _rl_outcome_columns(coord)
    # session_id pre-existed; turn_id added by the R1 self-heal migration.
    assert "session_id" in cols
    assert "turn_id" in cols


def test_migration_idempotent(tmp_path):
    db_path = tmp_path / "g.db"
    # First coordinator creates + migrates the table.
    coord1 = RLCoordinator(storage_path=tmp_path, db_path=db_path)
    cols1 = _rl_outcome_columns(coord1)
    # Second coordinator on the SAME db must not error (PRAGMA-guarded ALTERs).
    coord2 = RLCoordinator(storage_path=tmp_path, db_path=db_path)
    cols2 = _rl_outcome_columns(coord2)
    assert cols1 == cols2
    assert "turn_id" in cols2
    # turn_id present exactly once (no duplicate-column corruption).
    cur = coord2.db.cursor()
    cur.execute("PRAGMA table_info(rl_outcome)")
    turn_id_count = sum(1 for row in cur.fetchall() if row[1] == "turn_id")
    assert turn_id_count == 1


def test_correlation_for_outcome_from_context():
    from victor.core import context as ctx

    class _O:
        metadata = {}

    s = ctx.set_session_id("sess-ctx")
    t = ctx.set_turn_id("turn-ctx")
    try:
        assert _correlation_for_outcome(_O()) == ("sess-ctx", "turn-ctx")
    finally:
        ctx.session_id.reset(s)
        ctx.turn_id.reset(t)


def test_correlation_for_outcome_metadata_fallback():
    from victor.core import context as ctx

    class _O:
        metadata = {"session_id": "meta-sess", "turn_id": "meta-turn"}

    # No live context -> fall back to outcome.metadata.
    s = ctx.set_session_id("")
    t = ctx.set_turn_id("")
    try:
        assert _correlation_for_outcome(_O()) == ("meta-sess", "meta-turn")
    finally:
        ctx.session_id.reset(s)
        ctx.turn_id.reset(t)


def test_correlation_for_outcome_none_when_absent():
    from victor.core import context as ctx

    class _O:
        metadata = {}

    s = ctx.set_session_id("")
    t = ctx.set_turn_id("")
    try:
        assert _correlation_for_outcome(_O()) == (None, None)
    finally:
        ctx.session_id.reset(s)
        ctx.turn_id.reset(t)
