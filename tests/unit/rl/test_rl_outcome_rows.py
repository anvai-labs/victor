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

"""Every outcome write path stores the same row.

Three paths insert into rl_outcome: RLCoordinator.record_outcome (direct),
BatchedOutcomeWriter.flush and AsyncWriterQueue._write_batch. They had
drifted: the batch paths dropped the user-feedback columns, and the direct
path dropped turn_id and ignored the live request context for session_id.
"""

from pathlib import Path

import pytest

from victor.core.database import get_database, reset_database
from victor.core.schema import Tables
from victor.framework.rl.base import RLOutcome
from victor.framework.rl.coordinator import AsyncWriterQueue, BatchedOutcomeWriter, RLCoordinator

COLUMNS = (
    "learner_id, provider, model, task_type, vertical, success, quality_score, "
    "feedback_source, user_feedback, helpful, correction, session_id, turn_id, "
    "outcome_category"
)


@pytest.fixture
def coordinator(tmp_path: Path):
    reset_database()
    db_path = tmp_path / "rl.db"
    get_database(db_path)
    yield RLCoordinator(storage_path=tmp_path, db_path=db_path)
    reset_database()


def _outcome() -> RLOutcome:
    return RLOutcome(
        provider="anthropic",
        model="m",
        task_type="analysis",
        success=True,
        quality_score=0.8,
        metadata={
            "feedback_source": "user",
            "user_feedback": "nice",
            "helpful": False,
            "correction": "use X",
            "session_id": "sess-meta",
            "turn_id": "turn-meta",
        },
    )


def _rows(coordinator):
    cursor = coordinator.db.cursor()
    cursor.execute(f"SELECT {COLUMNS} FROM {Tables.RL_OUTCOME}")
    return [tuple(row) for row in cursor.fetchall()]


EXPECTED = (
    "tool_selector",
    "anthropic",
    "m",
    "analysis",
    "coding",
    1,
    0.8,
    "user",
    "nice",
    0,
    "use X",
    "sess-meta",
    "turn-meta",
    "task",  # FEP-0046: no event_type metadata + task-level learner -> 'task'
)


def test_direct_path_row(coordinator):
    coordinator.record_outcome("tool_selector", _outcome(), "coding")
    assert _rows(coordinator) == [EXPECTED]


def test_batched_writer_row(coordinator):
    writer = BatchedOutcomeWriter(coordinator, auto_flush_on_exit=False)
    writer.queue_outcome("tool_selector", _outcome(), "coding")
    writer.flush()
    assert _rows(coordinator) == [EXPECTED]


def test_async_queue_batch_row(coordinator):
    queue = AsyncWriterQueue(coordinator)
    queue._write_batch([("tool_selector", _outcome(), "coding")])
    assert _rows(coordinator) == [EXPECTED]


def test_live_request_context_wins_over_metadata(coordinator, monkeypatch):
    monkeypatch.setattr("victor.core.context.get_session_id", lambda: "sess-live")
    monkeypatch.setattr("victor.core.context.get_turn_id", lambda: "turn-live")
    coordinator.record_outcome("tool_selector", _outcome(), "coding")
    (row,) = _rows(coordinator)
    assert row[-3:-1] == ("sess-live", "turn-live")


def test_helpful_tristate(coordinator):
    for helpful in (True, False, None):
        outcome = _outcome()
        outcome.metadata["helpful"] = helpful
        coordinator.record_outcome("tool_selector", outcome, "coding")
    assert [row[9] for row in _rows(coordinator)] == [1, 0, None]


# --- FEP-0046 outcome_category resolution precedence -------------------------


def _category(coordinator) -> list:
    cursor = coordinator.db.cursor()
    cursor.execute(f"SELECT outcome_category FROM {Tables.RL_OUTCOME}")
    return [row[0] for row in cursor.fetchall()]


def test_category_learner_identity_default_internal(coordinator):
    """Tier 3: learner-internal learner + no event_type -> learner_internal."""
    coordinator.record_outcome("grounding_threshold", _outcome(), "coding")
    assert _category(coordinator) == ["learner_internal"]


def test_category_metadata_event_type_beats_learner_default(coordinator):
    """Tier 2: an internal event_type on a task-level learner wins.

    This is exactly the historical pollution shape: ``tool_selected`` rows
    (placeholder success) recorded under ``tool_selector``.
    """
    outcome = _outcome()
    outcome.metadata["event_type"] = "tool_selected"
    coordinator.record_outcome("tool_selector", outcome, "coding")
    assert _category(coordinator) == ["learner_internal"]


def test_category_task_event_metadata_stays_task(coordinator):
    """Tier 2 (negative): a task-level event_type keeps the row 'task'."""
    outcome = _outcome()
    outcome.metadata["event_type"] = "tool_executed"
    coordinator.record_outcome("tool_selector", outcome, "coding")
    assert _category(coordinator) == ["task"]


def test_category_explicit_value_wins(coordinator):
    """Tier 1: an explicitly set category overrides every inference."""
    outcome = _outcome()
    outcome.outcome_category = "learner_internal"
    coordinator.record_outcome("tool_selector", outcome, "coding")
    assert _category(coordinator) == ["learner_internal"]


def test_category_unknown_event_type_falls_through(coordinator):
    """Tier 2 (negative): an unmapped event_type string falls to learner default."""
    outcome = _outcome()
    outcome.metadata["event_type"] = "some_future_event"
    coordinator.record_outcome("tool_selector", outcome, "coding")
    assert _category(coordinator) == ["task"]
