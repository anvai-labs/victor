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

"""BaseLearner's unified-table read helpers (shared by the learners)."""

from pathlib import Path

import pytest

from victor.core.database import get_database, reset_database
from victor.core.schema import Tables
from victor.framework.rl.coordinator import RLCoordinator


@pytest.fixture
def learner(tmp_path: Path):
    reset_database()
    db_path = tmp_path / "rl.db"
    get_database(db_path)
    yield RLCoordinator(storage_path=tmp_path, db_path=db_path).get_learner("continuation_patience")
    reset_database()


def _put_stat(learner, task_type, key, value, owner=None):
    learner.db.execute(
        f"INSERT OR REPLACE INTO {Tables.RL_TASK_STAT}"
        " (learner_id, task_type, stat_key, stat_value, sample_count, updated_at)"
        " VALUES (?, ?, ?, ?, 1, 'now')",
        (owner or learner.name, task_type, key, value),
    )


def _put_param(learner, key, value, context, owner=None):
    learner.db.execute(
        f"INSERT OR REPLACE INTO {Tables.RL_PARAM}"
        " (learner_id, param_key, param_value, context, sample_count, updated_at)"
        " VALUES (?, ?, ?, ?, 2, 'now')",
        (owner or learner.name, key, value, context),
    )


def test_read_task_stats_is_scoped_to_learner_and_task_type(learner):
    _put_stat(learner, "ctx", "total_sessions", 4.0)
    _put_stat(learner, "ctx", "false_positives", 1.0)
    _put_stat(learner, "other", "total_sessions", 9.0)
    _put_stat(learner, "ctx", "total_sessions", 99.0, owner="someone_else")
    assert learner._read_task_stats("ctx") == {"total_sessions": 4.0, "false_positives": 1.0}
    assert learner._read_task_stats("missing") == {}


def test_read_param_returns_value_or_none(learner):
    _put_param(learner, "current_patience", 5.0, "ctx")
    _put_param(learner, "current_patience", 7.0, "ctx", owner="someone_else")
    assert learner._read_param("current_patience", "ctx") == 5.0
    assert learner._read_param("current_patience", "elsewhere") is None
    assert learner._read_param("unknown_key", "ctx") is None


def test_read_params_returns_all_rows_of_this_learner(learner):
    _put_param(learner, "a", 1.0, "c1")
    _put_param(learner, "b", 2.0, "c2")
    _put_param(learner, "z", 9.0, "c1", owner="someone_else")
    rows = sorted(learner._read_params(), key=lambda r: r["param_key"])
    assert [(r["param_key"], r["param_value"], r["context"], r["sample_count"]) for r in rows] == [
        ("a", 1.0, "c1", 2),
        ("b", 2.0, "c2", 2),
    ]
