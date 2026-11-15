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

"""FEP-0046 + column-bug regression tests for RLMetricsExporter.

``get_learner_summary``/``get_system_metrics`` used to filter on a
non-existent ``learner_name`` column (actual: ``learner_id``) with exceptions
swallowed at debug level — per-learner metrics silently reported zeros and
``active_learners`` was always empty. System-wide success now also excludes
learner-internal check rows.
"""

from pathlib import Path

import pytest

from victor.core.database import get_database, reset_database
from victor.framework.rl.base import RLOutcome
from victor.framework.rl.coordinator import RLCoordinator
from victor.framework.rl.metrics import RLMetricsExporter


@pytest.fixture
def coordinator(tmp_path: Path):
    reset_database()
    db_path = tmp_path / "rl.db"
    get_database(db_path)
    yield RLCoordinator(storage_path=tmp_path, db_path=db_path)
    reset_database()


def _task_outcome(success: bool) -> RLOutcome:
    return RLOutcome(
        provider="zai",
        model="glm-5.2",
        task_type="general",
        success=success,
        quality_score=0.8,
        metadata={"event_type": "tool_executed"},
    )


def _internal_outcome(success: bool) -> RLOutcome:
    return RLOutcome(
        provider="zai",
        model="glm-5.2",
        task_type="code_generation",
        success=success,
        quality_score=0.4,
        metadata={"event_type": "grounding_check"},
    )


def test_system_metrics_exclude_learner_internal(coordinator):
    """One task success + one internal failure must read as a 100% system rate.

    Pre-FEP-0046 the internal grounding failure dragged the aggregate down —
    the exact distortion that made code_generation report 0%.
    """
    coordinator.record_outcome("tool_selector", _task_outcome(True), "coding")
    coordinator.record_outcome("grounding_threshold", _internal_outcome(False), "coding")

    exporter = RLMetricsExporter(coordinator=coordinator)
    system = exporter.get_system_metrics()

    assert system.total_outcomes == 1
    assert system.overall_success_rate == 1.0


def test_learner_summary_reads_real_rows(coordinator):
    """Regression pin: queries must filter on learner_id, not learner_name."""
    coordinator.record_outcome("tool_selector", _task_outcome(True), "coding")
    coordinator.record_outcome("tool_selector", _task_outcome(False), "coding")

    exporter = RLMetricsExporter(coordinator=coordinator)
    summary = exporter.get_learner_summary("tool_selector")

    # Per-learner view deliberately includes ALL of the learner's rows
    # (event-driven internal rows included — here both are tool_executed).
    assert summary.outcome_count == 2
    assert summary.success_count == 1
    assert summary.success_rate == 0.5
    assert summary.last_update is not None


def test_active_learners_populated(coordinator):
    """Regression pin: DISTINCT learner_name always returned nothing."""
    coordinator.record_outcome("tool_selector", _task_outcome(True), "coding")

    exporter = RLMetricsExporter(coordinator=coordinator)
    system = exporter.get_system_metrics()

    assert "tool_selector" in system.active_learners
    assert "tool_selector" not in system.inactive_learners
    assert system.total_learners >= 1
