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

"""Each unified-table learner runs exactly its own legacy migration at init.

Pins the contract before the per-learner ``_ensure_tables`` overrides are
replaced by a declarative ``_legacy_migration`` hook on ``BaseLearner``.
"""

import importlib
import sqlite3

import pytest

from victor.framework.rl.base import BaseLearner
from victor.framework.rl.migration import RLTableMigrator

MIGRATING_LEARNERS = [
    ("workflow_execution", "WorkflowExecutionLearner"),
    ("tool_selector", "ToolSelectorLearner"),
    ("cache_eviction", "CacheEvictionLearner"),
    ("model_selector", "ModelSelectorLearner"),
    ("continuation_prompts", "ContinuationPromptLearner"),
    ("semantic_threshold", "SemanticThresholdLearner"),
    ("quality_weights", "QualityWeightLearner"),
    ("continuation_patience", "ContinuationPatienceLearner"),
    ("mode_transition", "ModeTransitionLearner"),
    ("grounding_threshold", "GroundingThresholdLearner"),
    ("context_pruning", "ContextPruningLearner"),
    ("prompt_template", "PromptTemplateLearner"),
]


@pytest.fixture
def calls(monkeypatch):
    seen = []

    def fake_run_if_needed(self, learner_id, migrate_fn):
        seen.append((learner_id, migrate_fn))

    monkeypatch.setattr(RLTableMigrator, "run_if_needed", fake_run_if_needed)
    return seen


@pytest.mark.parametrize("module, cls_name", MIGRATING_LEARNERS)
def test_learner_runs_its_own_migration_once(module, cls_name, calls):
    cls = getattr(importlib.import_module(f"victor.framework.rl.learners.{module}"), cls_name)
    cls(name=module, db_connection=sqlite3.connect(":memory:"))

    assert calls == [(module, getattr(RLTableMigrator, f"migrate_{module}"))]


def test_migration_fn_receives_only_the_connection():
    """The hook must not bind the function as a method (no implicit self)."""
    received = []

    class _Probe(BaseLearner):
        _legacy_migration = staticmethod(lambda *args: received.append(args))

        def record_outcome(self, outcome):  # pragma: no cover - abstract stub
            pass

        def get_recommendation(self, *a, **k):  # pragma: no cover - abstract stub
            return None

    conn = sqlite3.connect(":memory:")
    _Probe(name="probe_learner", db_connection=conn)
    assert received == [(conn,)]


def test_learner_with_no_storage_is_rejected(calls):
    """Keeps the guard the abstract method used to give: a concrete learner
    must choose a migration or own its tables."""

    class _NoStorage(BaseLearner):
        def record_outcome(self, outcome):  # pragma: no cover - abstract stub
            pass

        def get_recommendation(self, *a, **k):  # pragma: no cover - abstract stub
            return None

    with pytest.raises(TypeError, match="_legacy_migration or override _ensure_tables"):
        _NoStorage(name="no_storage", db_connection=sqlite3.connect(":memory:"))
    assert calls == []


def test_overriding_learner_skips_the_migration_hook(calls):
    class _OwnTables(BaseLearner):
        def _ensure_tables(self):
            self.db.execute("CREATE TABLE IF NOT EXISTS rl_own (k TEXT)")

        def record_outcome(self, outcome):  # pragma: no cover - abstract stub
            pass

        def get_recommendation(self, *a, **k):  # pragma: no cover - abstract stub
            return None

    conn = sqlite3.connect(":memory:")
    _OwnTables(name="own_tables", db_connection=conn)
    assert calls == []
    assert conn.execute("SELECT name FROM sqlite_master WHERE name='rl_own'").fetchone()
