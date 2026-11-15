# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""FEP-0046 lazy-guard: unversioned-modern stores never run versioned steps.

Such stores are stamped at CURRENT_SCHEMA_VERSION without replaying steps
(the adoption policy pinned in test_schema_version_migration). The
coordinator's lazy ``_migrate_add_outcome_category`` is therefore the only
protection — without it, every ``_insert_outcome_row`` call would raise
``no such column`` and be silently swallowed by ``record_outcome``.
"""

from pathlib import Path

import pytest

from victor.core.database import get_database, reset_database
from victor.core.schema import Tables
from victor.framework.rl.base import RLOutcome
from victor.framework.rl.coordinator import RLCoordinator

_PRE_FEP_RL_OUTCOME = """
CREATE TABLE IF NOT EXISTS rl_outcome (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    learner_id TEXT NOT NULL,
    provider TEXT,
    model TEXT,
    task_type TEXT,
    vertical TEXT DEFAULT '',
    repo_id TEXT DEFAULT NULL,
    success INTEGER,
    quality_score REAL,
    metadata TEXT,
    created_at TEXT DEFAULT (datetime('now'))
)
"""


@pytest.fixture
def pre_fep_db(tmp_path: Path):
    """An unversioned-modern store whose rl_outcome predates FEP-0046."""
    reset_database()
    db_path = tmp_path / "rl.db"
    get_database(db_path)  # creates sys_metadata; rl_outcome created lazily by the coordinator
    import sqlite3

    conn = sqlite3.connect(db_path)
    # Simulate the adoption-policy trap: drop to the pre-FEP shape WITHOUT a
    # recorded version below 10 (the real-world store was created by older
    # code and stamped current without replaying steps).
    conn.execute("DROP TABLE IF EXISTS rl_outcome")
    conn.execute(_PRE_FEP_RL_OUTCOME)
    conn.execute(
        "INSERT INTO rl_outcome (learner_id, provider, model, task_type, vertical,"
        " success, quality_score, metadata)"
        " VALUES ('grounding_threshold', 'zai', 'glm-5.2', 'code_generation', 'coding',"
        ' 0, 0.5, \'{"event_type": "grounding_check"}\')'
    )
    conn.commit()
    conn.close()
    yield db_path
    reset_database()


def test_lazy_guard_adds_and_backfills_column(pre_fep_db: Path):
    coordinator = RLCoordinator(storage_path=pre_fep_db.parent, db_path=pre_fep_db)

    columns = {row[1] for row in coordinator.db.execute(f"PRAGMA table_info({Tables.RL_OUTCOME})")}
    assert "outcome_category" in columns

    rows = [
        tuple(r)
        for r in coordinator.db.execute(
            f"SELECT learner_id, outcome_category FROM {Tables.RL_OUTCOME}"
        ).fetchall()
    ]
    assert rows == [("grounding_threshold", "learner_internal")]


def test_lazy_guard_lets_writes_succeed(pre_fep_db: Path):
    """The silent-drop failure mode this guard exists to prevent."""
    coordinator = RLCoordinator(storage_path=pre_fep_db.parent, db_path=pre_fep_db)

    outcome = RLOutcome(
        provider="zai",
        model="glm-5.2",
        task_type="analysis",
        success=True,
        quality_score=0.8,
    )
    coordinator.record_outcome("tool_selector", outcome, "coding")

    rows = [
        tuple(r)
        for r in coordinator.db.execute(
            f"SELECT learner_id, outcome_category FROM {Tables.RL_OUTCOME} ORDER BY id"
        ).fetchall()
    ]
    assert rows == [
        ("grounding_threshold", "learner_internal"),
        ("tool_selector", "task"),
    ]


def test_lazy_guard_is_idempotent(pre_fep_db: Path):
    # Coordinator.close() tears down the shared singleton connection; rebuild
    # it the way a fresh process would (same DB file, fresh manager).
    RLCoordinator(storage_path=pre_fep_db.parent, db_path=pre_fep_db)
    reset_database()
    get_database(pre_fep_db)
    second = RLCoordinator(storage_path=pre_fep_db.parent, db_path=pre_fep_db)

    rows = [
        tuple(r)
        for r in second.db.execute(f"SELECT outcome_category FROM {Tables.RL_OUTCOME}").fetchall()
    ]
    assert rows == [("learner_internal",)]  # no re-backfill, no duplicate column error
