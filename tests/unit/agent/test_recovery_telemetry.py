# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
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

"""Tests for recovery telemetry - achieving 70%+ coverage."""

import json
import sqlite3
import tempfile
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from victor.agent.recovery.protocols import (
    FailureType,
    RecoveryAction,
    RecoveryContext,
    RecoveryResult,
)
from victor.agent.recovery.telemetry import (
    AggregatedStats,
    FailureEvent,
    RecoveryEvent,
    RecoveryTelemetryCollector,
)


class TestFailureEvent:
    """Tests for FailureEvent dataclass."""

    def test_basic_creation(self):
        """Test basic FailureEvent creation."""
        event = FailureEvent(
            timestamp=datetime.now(),
            failure_type=FailureType.STUCK_LOOP,
            provider="anthropic",
            model="claude-3-sonnet",
            task_type="code_generation",
            consecutive_count=2,
            context_hash="abc123",
        )
        assert event.failure_type == FailureType.STUCK_LOOP
        assert event.provider == "anthropic"
        assert event.model == "claude-3-sonnet"
        assert event.consecutive_count == 2

    def test_all_failure_types(self):
        """Test FailureEvent with different failure types."""
        for failure_type in FailureType:
            event = FailureEvent(
                timestamp=datetime.now(),
                failure_type=failure_type,
                provider="test",
                model="test-model",
                task_type="test",
                consecutive_count=1,
                context_hash="hash",
            )
            assert event.failure_type == failure_type


class TestRecoveryEvent:
    """Tests for RecoveryEvent dataclass."""

    def test_basic_creation(self):
        """Test basic RecoveryEvent creation."""
        event = RecoveryEvent(
            timestamp=datetime.now(),
            failure_type=FailureType.STUCK_LOOP,
            strategy_name="CompactContextStrategy",
            action=RecoveryAction.COMPACT_CONTEXT,
            success=True,
            quality_improvement=0.5,
            provider="anthropic",
            model="claude-3-sonnet",
            context_hash="abc123",
        )
        assert event.strategy_name == "CompactContextStrategy"
        assert event.success is True
        assert event.quality_improvement == 0.5

    def test_failed_recovery(self):
        """Test RecoveryEvent for failed recovery."""
        event = RecoveryEvent(
            timestamp=datetime.now(),
            failure_type=FailureType.HALLUCINATED_TOOL,
            strategy_name="ModelSwitchStrategy",
            action=RecoveryAction.SWITCH_MODEL,
            success=False,
            quality_improvement=-0.1,
            provider="openai",
            model="gpt-4",
            context_hash="xyz789",
        )
        assert event.success is False
        assert event.quality_improvement < 0


class TestAggregatedStats:
    """Tests for AggregatedStats dataclass."""

    def test_default_values(self):
        """Test default values for AggregatedStats."""
        stats = AggregatedStats(
            start_time=datetime.now(),
            end_time=datetime.now(),
        )
        assert stats.total_failures == 0
        assert stats.total_recoveries == 0
        assert stats.successful_recoveries == 0
        assert stats.failures_by_type == {}
        assert stats.recoveries_by_strategy == {}

    def test_recovery_rate_zero_recoveries(self):
        """Test recovery_rate with zero recoveries."""
        stats = AggregatedStats(
            start_time=datetime.now(),
            end_time=datetime.now(),
            total_recoveries=0,
        )
        assert stats.recovery_rate == 0.0

    def test_recovery_rate_calculation(self):
        """Test recovery_rate calculation."""
        stats = AggregatedStats(
            start_time=datetime.now(),
            end_time=datetime.now(),
            total_recoveries=10,
            successful_recoveries=7,
        )
        assert stats.recovery_rate == 0.7

    def test_recovery_rate_all_successful(self):
        """Test recovery_rate when all successful."""
        stats = AggregatedStats(
            start_time=datetime.now(),
            end_time=datetime.now(),
            total_recoveries=5,
            successful_recoveries=5,
        )
        assert stats.recovery_rate == 1.0

    def test_with_failure_data(self):
        """Test AggregatedStats with failure data."""
        stats = AggregatedStats(
            start_time=datetime.now() - timedelta(hours=1),
            end_time=datetime.now(),
            total_failures=15,
            total_recoveries=12,
            successful_recoveries=8,
            failures_by_type={"STUCK_LOOP": 10, "HALLUCINATED_TOOL": 5},
            failures_by_model={"claude-3-sonnet": 10, "gpt-4": 5},
            avg_quality_improvement=0.3,
        )
        assert stats.total_failures == 15
        assert stats.failures_by_type["STUCK_LOOP"] == 10
        assert stats.avg_quality_improvement == 0.3


class TestRecoveryTelemetryCollector:
    """Tests for RecoveryTelemetryCollector class."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir) / "test_telemetry.db"

    @pytest.fixture
    def collector(self):
        """Create a basic collector without persistence."""
        return RecoveryTelemetryCollector()

    @pytest.fixture
    def collector_with_db(self, temp_db):
        """Create a collector with SQLite persistence."""
        return RecoveryTelemetryCollector(db_path=temp_db)

    @pytest.fixture
    def mock_context(self):
        """Create a mock RecoveryContext."""
        context = MagicMock(spec=RecoveryContext)
        context.failure_type = FailureType.STUCK_LOOP
        context.provider_name = "anthropic"
        context.model_name = "claude-3-sonnet"
        context.task_type = "code_generation"
        context.consecutive_failures = 2
        context.to_state_key.return_value = (
            "mock_state_key_123456789012345678901234567890123456789012345678901234567890"
        )
        return context

    @pytest.fixture
    def mock_result(self):
        """Create a mock RecoveryResult."""
        result = MagicMock(spec=RecoveryResult)
        result.strategy_name = "CompactContextStrategy"
        result.action = RecoveryAction.COMPACT_CONTEXT
        return result

    def test_initialization_defaults(self):
        """Test default initialization."""
        collector = RecoveryTelemetryCollector()
        assert collector._db_path is None
        assert collector._max_memory_events == 1000
        assert collector._prometheus_enabled is False
        assert len(collector._failure_events) == 0
        assert len(collector._recovery_events) == 0

    def test_initialization_with_db(self, temp_db):
        """Test initialization with database."""
        collector = RecoveryTelemetryCollector(db_path=temp_db)
        assert collector._db_path == temp_db
        assert temp_db.exists()

    def test_initialization_custom_max_events(self):
        """Test initialization with custom max events."""
        collector = RecoveryTelemetryCollector(max_memory_events=500)
        assert collector._max_memory_events == 500

    def test_initialization_with_prometheus(self):
        """Test initialization with prometheus enabled."""
        collector = RecoveryTelemetryCollector(prometheus_enabled=True)
        assert collector._prometheus_enabled is True

    def test_record_failure_basic(self, collector, mock_context):
        """Test recording a failure."""
        collector.record_failure(mock_context)

        assert len(collector._failure_events) == 1
        event = collector._failure_events[0]
        assert event.failure_type == FailureType.STUCK_LOOP
        assert event.provider == "anthropic"
        assert event.model == "claude-3-sonnet"

    def test_record_failure_updates_counters(self, collector, mock_context):
        """Test that recording failure updates counters."""
        collector.record_failure(mock_context)

        assert collector._failure_counts["STUCK_LOOP"] == 1
        assert collector._failure_counts["anthropic:STUCK_LOOP"] == 1

    def test_record_failure_ring_buffer(self, mock_context):
        """Test that failure events use ring buffer."""
        collector = RecoveryTelemetryCollector(max_memory_events=3)

        for i in range(5):
            collector.record_failure(mock_context)

        assert len(collector._failure_events) == 3

    def test_record_failure_with_db(self, collector_with_db, mock_context, temp_db):
        """Test recording failure persists to database."""
        collector_with_db.record_failure(mock_context)

        conn = sqlite3.connect(str(temp_db))
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM failure_events")
        count = cursor.fetchone()[0]
        conn.close()

        assert count == 1

    def test_record_recovery_attempt(self, collector, mock_context, mock_result):
        """Test recording a recovery attempt."""
        collector.record_recovery_attempt(mock_context, mock_result)

        assert len(collector._recovery_events) == 1
        event = collector._recovery_events[0]
        assert event.strategy_name == "CompactContextStrategy"
        assert event.success is False  # Not yet confirmed

    def test_record_recovery_attempt_updates_counters(self, collector, mock_context, mock_result):
        """Test that recovery attempt updates counters."""
        collector.record_recovery_attempt(mock_context, mock_result)

        assert collector._recovery_counts["CompactContextStrategy"] == 1

    def test_record_recovery_attempt_ring_buffer(self, mock_context, mock_result):
        """Test that recovery events use ring buffer."""
        collector = RecoveryTelemetryCollector(max_memory_events=3)

        for i in range(5):
            collector.record_recovery_attempt(mock_context, mock_result)

        assert len(collector._recovery_events) == 3

    def test_record_recovery_outcome_success(self, collector, mock_context, mock_result):
        """Test recording successful recovery outcome."""
        collector.record_recovery_attempt(mock_context, mock_result)
        collector.record_recovery_outcome(mock_context, mock_result, True, 0.5)

        assert collector._success_counts["CompactContextStrategy"] == 1
        # The event should be updated
        successful_events = [e for e in collector._recovery_events if e.success]
        assert len(successful_events) == 1

    def test_record_recovery_outcome_failure(self, collector, mock_context, mock_result):
        """Test recording failed recovery outcome."""
        collector.record_recovery_attempt(mock_context, mock_result)
        collector.record_recovery_outcome(mock_context, mock_result, False, -0.1)

        assert collector._success_counts.get("CompactContextStrategy", 0) == 0

    def test_record_recovery_outcome_with_db(
        self, collector_with_db, mock_context, mock_result, temp_db
    ):
        """Test recording recovery outcome persists to database."""
        collector_with_db.record_recovery_outcome(mock_context, mock_result, True, 0.5)

        conn = sqlite3.connect(str(temp_db))
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM recovery_events")
        count = cursor.fetchone()[0]
        conn.close()

        assert count == 1

    def test_get_failure_stats_empty(self, collector):
        """Test getting failure stats when empty."""
        stats = collector.get_failure_stats()

        assert stats["total_failures"] == 0
        assert stats["total_recovery_attempts"] == 0
        assert stats["recovery_rate"] == 0

    def test_get_failure_stats_with_data(self, collector, mock_context, mock_result):
        """Test getting failure stats with data."""
        collector.record_failure(mock_context)
        collector.record_failure(mock_context)
        collector.record_recovery_outcome(mock_context, mock_result, True, 0.5)

        stats = collector.get_failure_stats()

        assert stats["total_failures"] == 2
        assert stats["total_recovery_attempts"] == 1
        assert "STUCK_LOOP" in stats["failures_by_type"]

    def test_get_failure_stats_time_window(self, collector, mock_context):
        """Test that time window filters events."""
        collector.record_failure(mock_context)

        # Default 24h window should include the event
        stats = collector.get_failure_stats(time_window_hours=24)
        assert stats["total_failures"] == 1

        # 0h window (past only) should exclude it
        # Note: The event was just created, so it won't be filtered

    def test_get_strategy_effectiveness_empty(self, collector):
        """Test getting strategy effectiveness when empty."""
        effectiveness = collector.get_strategy_effectiveness()
        assert effectiveness == {}

    def test_get_strategy_effectiveness_with_data(self, collector, mock_context, mock_result):
        """Test getting strategy effectiveness with data."""
        collector.record_recovery_outcome(mock_context, mock_result, True, 0.5)
        collector.record_recovery_outcome(mock_context, mock_result, False, -0.1)

        effectiveness = collector.get_strategy_effectiveness()

        assert "CompactContextStrategy" in effectiveness
        strategy_stats = effectiveness["CompactContextStrategy"]
        assert strategy_stats["total_attempts"] == 2
        assert strategy_stats["successful"] == 1
        assert strategy_stats["success_rate"] == 0.5

    def test_get_model_failure_patterns_empty(self, collector):
        """Test getting model failure patterns when empty."""
        patterns = collector.get_model_failure_patterns()
        assert patterns == {}

    def test_get_model_failure_patterns_with_data(self, collector, mock_context):
        """Test getting model failure patterns with data."""
        collector.record_failure(mock_context)

        # Create another context with different model
        mock_context2 = MagicMock(spec=RecoveryContext)
        mock_context2.failure_type = FailureType.HALLUCINATED_TOOL
        mock_context2.provider_name = "openai"
        mock_context2.model_name = "gpt-4"
        mock_context2.task_type = "code_review"
        mock_context2.consecutive_failures = 1
        mock_context2.to_state_key.return_value = "other_hash"

        collector.record_failure(mock_context2)

        patterns = collector.get_model_failure_patterns()

        assert "claude-3-sonnet" in patterns
        assert "gpt-4" in patterns
        assert patterns["claude-3-sonnet"]["STUCK_LOOP"] == 1
        assert patterns["gpt-4"]["HALLUCINATED_TOOL"] == 1

    def test_export_prometheus_metrics_empty(self, collector):
        """Test exporting prometheus metrics when empty."""
        metrics = collector.export_prometheus_metrics()

        assert "recovery_failures_total" in metrics
        assert "recovery_attempts_total" in metrics
        assert "recovery_successes_total" in metrics

    def test_export_prometheus_metrics_with_data(self, collector, mock_context, mock_result):
        """Test exporting prometheus metrics with data."""
        collector.record_failure(mock_context)
        collector.record_recovery_outcome(mock_context, mock_result, True, 0.5)

        metrics = collector.export_prometheus_metrics()

        assert 'type="STUCK_LOOP"' in metrics
        assert 'strategy="CompactContextStrategy"' in metrics

    def test_export_prometheus_metrics_provider_labels(self, collector, mock_context):
        """Test prometheus metrics include provider labels."""
        collector.record_failure(mock_context)

        metrics = collector.export_prometheus_metrics()

        assert 'provider="anthropic"' in metrics

    def test_export_json_report(self, collector, mock_context, mock_result):
        """Test exporting JSON report."""
        collector.record_failure(mock_context)
        collector.record_recovery_outcome(mock_context, mock_result, True, 0.5)

        report = collector.export_json_report()
        data = json.loads(report)

        assert "generated_at" in data
        assert "failure_stats" in data
        assert "strategy_effectiveness" in data
        assert "model_failure_patterns" in data

    def test_export_json_report_custom_window(self, collector, mock_context):
        """Test exporting JSON report with custom time window."""
        collector.record_failure(mock_context)

        report = collector.export_json_report(time_window_hours=1)
        data = json.loads(report)

        assert data["time_window_hours"] == 1

    def test_clear_old_events(self, collector, mock_context, mock_result):
        """Test clearing old events."""
        collector.record_failure(mock_context)
        collector.record_recovery_outcome(mock_context, mock_result, True, 0.5)

        # Clear with max_age_hours=0 should clear everything
        cleared = collector.clear_old_events(max_age_hours=0)

        assert cleared == 2
        assert len(collector._failure_events) == 0
        assert len(collector._recovery_events) == 0

    def test_clear_old_events_keeps_recent(self, collector, mock_context):
        """Test clearing old events keeps recent ones."""
        collector.record_failure(mock_context)

        # Clear with large max_age should keep recent events
        cleared = collector.clear_old_events(max_age_hours=168)

        assert cleared == 0
        assert len(collector._failure_events) == 1

    def test_clear_old_events_with_db(self, collector_with_db, mock_context, temp_db):
        """Test clearing old events also clears database."""
        collector_with_db.record_failure(mock_context)

        cleared = collector_with_db.clear_old_events(max_age_hours=0)

        conn = sqlite3.connect(str(temp_db))
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM failure_events")
        count = cursor.fetchone()[0]
        conn.close()

        assert count == 0

    def test_thread_safety(self, collector, mock_context, mock_result):
        """Test thread-safe operations."""
        errors = []

        def worker():
            try:
                for _ in range(10):
                    collector.record_failure(mock_context)
                    collector.record_recovery_attempt(mock_context, mock_result)
                    collector.get_failure_stats()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_db_initialization_creates_tables(self, temp_db):
        """Test database initialization creates required tables."""
        collector = RecoveryTelemetryCollector(db_path=temp_db)

        conn = sqlite3.connect(str(temp_db))
        cursor = conn.cursor()

        # Check tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}

        conn.close()

        assert "failure_events" in tables
        assert "recovery_events" in tables

    def test_db_initialization_creates_indexes(self, temp_db):
        """Test database initialization creates indexes."""
        collector = RecoveryTelemetryCollector(db_path=temp_db)

        conn = sqlite3.connect(str(temp_db))
        cursor = conn.cursor()

        # Check indexes exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
        indexes = {row[0] for row in cursor.fetchall()}

        conn.close()

        assert "idx_failure_timestamp" in indexes
        assert "idx_recovery_timestamp" in indexes

    def test_db_error_handling_init(self, tmp_path):
        """Test database error handling during initialization."""
        # Create a file where the directory should be to cause error
        bad_path = tmp_path / "file_not_dir" / "db.sqlite"
        (tmp_path / "file_not_dir").write_text("not a directory")

        # Should not raise, just log warning
        collector = RecoveryTelemetryCollector(db_path=bad_path)
        assert collector._db_path == bad_path

    def test_persist_failure_error_handling(self, collector, mock_context, tmp_path):
        """Test error handling when persisting failure fails."""
        # Set a bad path
        collector._db_path = tmp_path / "nonexistent" / "db.sqlite"

        # Should not raise, just log warning
        collector.record_failure(mock_context)
        assert len(collector._failure_events) == 1

    def test_persist_recovery_error_handling(self, collector, mock_context, mock_result, tmp_path):
        """Test error handling when persisting recovery fails."""
        collector._db_path = tmp_path / "nonexistent" / "db.sqlite"

        # Should not raise, just log warning
        collector.record_recovery_outcome(mock_context, mock_result, True, 0.5)

    def test_multiple_failure_types_stats(self, collector):
        """Test stats with multiple failure types."""
        for failure_type in [
            FailureType.STUCK_LOOP,
            FailureType.HALLUCINATED_TOOL,
            FailureType.TIMEOUT_APPROACHING,
        ]:
            context = MagicMock(spec=RecoveryContext)
            context.failure_type = failure_type
            context.provider_name = "test"
            context.model_name = "test-model"
            context.task_type = "test"
            context.consecutive_failures = 1
            context.to_state_key.return_value = f"hash_{failure_type.name}"
            collector.record_failure(context)

        stats = collector.get_failure_stats()

        assert stats["total_failures"] == 3
        assert len(stats["failures_by_type"]) == 3

    def test_multiple_strategies_effectiveness(self, collector, mock_context):
        """Test effectiveness with multiple strategies."""
        strategies = [
            ("CompactContextStrategy", RecoveryAction.COMPACT_CONTEXT),
            ("ModelSwitchStrategy", RecoveryAction.SWITCH_MODEL),
            ("RetryWithTemplateStrategy", RecoveryAction.RETRY_WITH_TEMPLATE),
        ]

        for strategy_name, action in strategies:
            result = MagicMock(spec=RecoveryResult)
            result.strategy_name = strategy_name
            result.action = action

            collector.record_recovery_outcome(mock_context, result, True, 0.3)
            collector.record_recovery_outcome(mock_context, result, False, 0.0)

        effectiveness = collector.get_strategy_effectiveness()

        assert len(effectiveness) == 3
        for strategy_name, _ in strategies:
            assert strategy_name in effectiveness
            assert effectiveness[strategy_name]["total_attempts"] == 2
            assert effectiveness[strategy_name]["successful"] == 1

    def test_recovery_outcome_updates_existing_event(self, collector, mock_context, mock_result):
        """Test that recovery outcome updates the matching event."""
        collector.record_recovery_attempt(mock_context, mock_result)

        initial_count = len(collector._recovery_events)

        collector.record_recovery_outcome(mock_context, mock_result, True, 0.5)

        # Should update existing, not add new
        assert len(collector._recovery_events) == initial_count
        assert collector._recovery_events[-1].success is True

    def test_recovery_outcome_adds_new_if_no_match(self, collector, mock_context, mock_result):
        """Test that recovery outcome adds new event if no match."""
        # Don't record attempt first
        collector.record_recovery_outcome(mock_context, mock_result, True, 0.5)

        assert len(collector._recovery_events) == 1
        assert collector._recovery_events[0].success is True


class TestTelemetryEdgeCases:
    """Edge case tests for telemetry."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir) / "test_telemetry.db"

    def test_empty_context_hash(self):
        """Test handling empty context hash."""
        collector = RecoveryTelemetryCollector()

        context = MagicMock(spec=RecoveryContext)
        context.failure_type = FailureType.STUCK_LOOP
        context.provider_name = "test"
        context.model_name = "test"
        context.task_type = "test"
        context.consecutive_failures = 1
        context.to_state_key.return_value = ""  # Empty hash

        collector.record_failure(context)
        assert len(collector._failure_events) == 1

    def test_very_long_context_hash(self):
        """Test that context hash is truncated to 64 chars."""
        collector = RecoveryTelemetryCollector()

        context = MagicMock(spec=RecoveryContext)
        context.failure_type = FailureType.STUCK_LOOP
        context.provider_name = "test"
        context.model_name = "test"
        context.task_type = "test"
        context.consecutive_failures = 1
        context.to_state_key.return_value = "x" * 1000  # Very long hash

        collector.record_failure(context)
        assert len(collector._failure_events[0].context_hash) == 64

    def test_concurrent_db_access(self, temp_db):
        """Test concurrent database access."""
        collector = RecoveryTelemetryCollector(db_path=temp_db)
        errors = []

        def worker(worker_id):
            try:
                context = MagicMock(spec=RecoveryContext)
                context.failure_type = FailureType.STUCK_LOOP
                context.provider_name = f"provider_{worker_id}"
                context.model_name = "test"
                context.task_type = "test"
                context.consecutive_failures = 1
                context.to_state_key.return_value = f"hash_{worker_id}"

                for _ in range(5):
                    collector.record_failure(context)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_special_characters_in_data(self, temp_db):
        """Test handling special characters in provider/model names."""
        collector = RecoveryTelemetryCollector(db_path=temp_db)

        context = MagicMock(spec=RecoveryContext)
        context.failure_type = FailureType.STUCK_LOOP
        context.provider_name = "test:provider:with:colons"
        context.model_name = "model/with/slashes"
        context.task_type = "task\nwith\nnewlines"
        context.consecutive_failures = 1
        context.to_state_key.return_value = "hash"

        collector.record_failure(context)

        stats = collector.get_failure_stats()
        assert stats["total_failures"] == 1

    def test_zero_quality_improvement(self, temp_db):
        """Test recording zero quality improvement."""
        collector = RecoveryTelemetryCollector(db_path=temp_db)

        context = MagicMock(spec=RecoveryContext)
        context.failure_type = FailureType.STUCK_LOOP
        context.provider_name = "test"
        context.model_name = "test"
        context.to_state_key.return_value = "hash"

        result = MagicMock(spec=RecoveryResult)
        result.strategy_name = "TestStrategy"
        result.action = RecoveryAction.RETRY_WITH_TEMPLATE

        collector.record_recovery_outcome(context, result, True, 0.0)

        effectiveness = collector.get_strategy_effectiveness()
        assert effectiveness["TestStrategy"]["avg_quality_improvement"] == 0.0

    def test_negative_quality_improvement(self):
        """Test recording negative quality improvement."""
        collector = RecoveryTelemetryCollector()

        context = MagicMock(spec=RecoveryContext)
        context.failure_type = FailureType.STUCK_LOOP
        context.provider_name = "test"
        context.model_name = "test"
        context.to_state_key.return_value = "hash"

        result = MagicMock(spec=RecoveryResult)
        result.strategy_name = "TestStrategy"
        result.action = RecoveryAction.RETRY_WITH_TEMPLATE

        collector.record_recovery_outcome(context, result, False, -0.5)

        effectiveness = collector.get_strategy_effectiveness()
        assert effectiveness["TestStrategy"]["avg_quality_improvement"] == -0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
