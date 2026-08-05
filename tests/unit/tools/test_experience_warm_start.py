# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""R3 — ToolExperienceStore as a projection of the durable RL_OUTCOME stream."""

from __future__ import annotations

from victor.tools.experience_store import ExperienceType, ToolExperienceStore


def test_warm_start_rebuilds_stats():
    store = ToolExperienceStore()
    n = store.warm_start_from_outcomes(
        [
            ("read", "search", True, 1.0),
            ("read", "search", False, 0.3),
            ("grep", "search", True, 0.8),
        ]
    )
    assert n == 3
    read = store.get_stats("read")
    assert read.total_uses == 2
    assert read.successes == 1
    assert read.success_rate == 0.5
    assert round(read.avg_reward, 3) == 0.65  # (1.0 + 0.3) / 2
    assert store.get_stats("grep").total_uses == 1


def test_warm_start_skips_blank_tool():
    store = ToolExperienceStore()
    n = store.warm_start_from_outcomes([("", "search", True, 1.0), (None, "x", True, 1.0)])
    assert n == 0


def test_warm_start_idempotent():
    store = ToolExperienceStore()
    assert store.warm_start_from_outcomes([("read", "search", True, 1.0)]) == 1
    # Second call must be a no-op so it cannot double-count alongside the live feed.
    assert store.warm_start_from_outcomes([("read", "search", True, 1.0)]) == 0
    assert store.get_stats("read").total_uses == 1


def test_warm_start_classifies_as_demonstration():
    store = ToolExperienceStore()
    store.warm_start_from_outcomes([("read", "search", True, 1.0)])
    counts = store.get_stats("read").experience_counts
    assert counts.get(ExperienceType.DEMONSTRATION.value) == 1


def test_runtime_warm_start_reads_db(monkeypatch):
    """_warm_start_experience_store parses durable rl_outcome rows into the projection."""
    from victor.agent.services import tool_selection_runtime as tsr

    class _FakeDB:
        def query(self, sql, params):
            # (task_type, success, quality_score, metadata-json)
            return [
                ("search", 1, 0.9, '{"tool_name": "read"}'),
                ("search", 0, None, '{"tool_name": "grep"}'),
                ("search", 1, 0.7, "{}"),  # no tool_name -> skipped
                ("search", 1, 0.5, None),  # null metadata -> skipped
            ]

    monkeypatch.setattr("victor.core.database.get_database", lambda: _FakeDB())

    store = ToolExperienceStore()
    tsr._warm_start_experience_store(store)

    assert store.get_stats("read").total_uses == 1
    assert round(store.get_stats("read").avg_reward, 3) == 0.9  # explicit quality wins
    grep = store.get_stats("grep")
    assert grep.total_uses == 1
    assert grep.successes == 0
    assert round(grep.avg_reward, 3) == 0.3  # no quality + failure -> canonical default


def test_runtime_warm_start_best_effort_on_db_error(monkeypatch):
    from victor.agent.services import tool_selection_runtime as tsr

    def _boom():
        raise RuntimeError("db down")

    monkeypatch.setattr("victor.core.database.get_database", _boom)
    store = ToolExperienceStore()
    # Must not raise; store simply stays empty for the live feed to populate.
    tsr._warm_start_experience_store(store)
    assert store.get_stats("read").total_uses == 0
