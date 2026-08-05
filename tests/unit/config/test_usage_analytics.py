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

"""Tests for UsageAnalytics."""

import json
import tempfile
import time
from pathlib import Path

import pytest

from victor.agent.usage_analytics import (
    AnalyticsConfig,
    ConversationStats,
    ProviderCallRecord,
    ToolExecutionRecord,
    UsageAnalytics,
    create_usage_analytics,
)


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the UsageAnalytics singleton before and after each test."""
    UsageAnalytics.reset_instance()
    yield
    UsageAnalytics.reset_instance()


class TestAnalyticsConfig:
    """Tests for AnalyticsConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = AnalyticsConfig()

        assert config.max_records_per_tool == 1000
        assert config.max_records_per_provider == 500
        assert config.persistence_interval_seconds == 300
        assert config.cache_dir is None
        assert config.enable_prometheus_export is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = AnalyticsConfig(
            max_records_per_tool=500,
            persistence_interval_seconds=60,
        )

        assert config.max_records_per_tool == 500
        assert config.persistence_interval_seconds == 60


class TestDataClasses:
    """Tests for data classes."""

    def test_tool_execution_record(self):
        """Test ToolExecutionRecord creation."""
        record = ToolExecutionRecord(
            timestamp=time.time(),
            success=True,
            execution_time_ms=150.5,
            error_type=None,
            context_tokens=5000,
        )

        assert record.success is True
        assert record.execution_time_ms == 150.5
        assert record.context_tokens == 5000

    def test_provider_call_record(self):
        """Test ProviderCallRecord creation."""
        record = ProviderCallRecord(
            timestamp=time.time(),
            provider_name="anthropic",
            model="claude-3-opus",
            success=True,
            latency_ms=500.0,
            tokens_in=1000,
            tokens_out=500,
        )

        assert record.provider_name == "anthropic"
        assert record.tokens_in == 1000

    def test_conversation_stats(self):
        """Test ConversationStats creation."""
        stats = ConversationStats(
            start_time=time.time(),
            turn_count=5,
            tool_calls=10,
        )

        assert stats.turn_count == 5
        assert stats.tool_calls == 10


class TestUsageAnalyticsSingleton:
    """Tests for singleton behavior."""

    def test_singleton_same_instance(self):
        """Test that get_instance returns the same instance."""
        instance1 = UsageAnalytics.get_instance()
        instance2 = UsageAnalytics.get_instance()

        assert instance1 is instance2

    def test_reset_creates_new_instance(self):
        """Test that reset creates a new instance."""
        instance1 = UsageAnalytics.get_instance()
        UsageAnalytics.reset_instance()
        instance2 = UsageAnalytics.get_instance()

        assert instance1 is not instance2


class TestToolExecution:
    """Tests for tool execution recording."""

    def test_record_tool_execution(self):
        """Test recording a tool execution."""
        analytics = UsageAnalytics.get_instance()

        analytics.record_tool_execution(
            tool_name="read_file",
            success=True,
            execution_time_ms=150.0,
        )

        insights = analytics.get_tool_insights("read_file")
        assert insights["status"] == "ok"
        assert insights["total_executions"] == 1
        assert insights["success_rate"] == 1.0

    def test_record_multiple_executions(self):
        """Test recording multiple tool executions."""
        analytics = UsageAnalytics.get_instance()

        for _ in range(5):
            analytics.record_tool_execution(
                tool_name="read_file",
                success=True,
                execution_time_ms=100.0,
            )

        insights = analytics.get_tool_insights("read_file")
        assert insights["total_executions"] == 5

    def test_record_failed_execution(self):
        """Test recording a failed execution."""
        analytics = UsageAnalytics.get_instance()

        analytics.record_tool_execution(
            tool_name="read_file",
            success=False,
            execution_time_ms=50.0,
            error_type="FileNotFoundError",
        )

        insights = analytics.get_tool_insights("read_file")
        assert insights["success_rate"] == 0.0
        assert "FileNotFoundError" in insights["error_distribution"]

    def test_success_rate_calculation(self):
        """Test success rate is calculated correctly."""
        analytics = UsageAnalytics.get_instance()

        for i in range(10):
            analytics.record_tool_execution(
                tool_name="test_tool",
                success=i < 8,  # 8 successes, 2 failures
                execution_time_ms=100.0,
            )

        insights = analytics.get_tool_insights("test_tool")
        assert insights["success_rate"] == 0.8

    def test_avg_execution_time(self):
        """Test average execution time calculation."""
        analytics = UsageAnalytics.get_instance()

        times = [100, 200, 300]
        for t in times:
            analytics.record_tool_execution(
                tool_name="test_tool",
                success=True,
                execution_time_ms=float(t),
            )

        insights = analytics.get_tool_insights("test_tool")
        assert insights["avg_execution_ms"] == 200.0  # (100+200+300)/3

    def test_record_limit_enforced(self):
        """Test that record limit is enforced."""
        config = AnalyticsConfig(max_records_per_tool=10)
        analytics = UsageAnalytics(config)

        for _i in range(20):
            analytics.record_tool_execution(
                tool_name="test_tool",
                success=True,
                execution_time_ms=100.0,
            )

        # Should only keep last 10
        assert len(analytics._tool_records["test_tool"]) == 10


class TestProviderCalls:
    """Tests for provider call recording."""

    def test_record_provider_call(self):
        """Test recording a provider call."""
        analytics = UsageAnalytics.get_instance()

        analytics.record_provider_call(
            provider_name="anthropic",
            model="claude-3-opus",
            success=True,
            latency_ms=500.0,
            tokens_in=1000,
            tokens_out=500,
        )

        insights = analytics.get_provider_insights("anthropic")
        assert insights["status"] == "ok"
        assert insights["total_calls"] == 1

    def test_latency_percentiles(self):
        """Test that latency percentiles are calculated."""
        analytics = UsageAnalytics.get_instance()

        # Record 100 calls with increasing latency
        for i in range(100):
            analytics.record_provider_call(
                provider_name="test_provider",
                model="test_model",
                success=True,
                latency_ms=float(i * 10),  # 0, 10, 20, ... 990
            )

        insights = analytics.get_provider_insights("test_provider")
        assert "p50_latency_ms" in insights
        assert "p95_latency_ms" in insights
        assert "p99_latency_ms" in insights


class TestSessions:
    """Tests for session tracking."""

    def test_start_session(self):
        """Test starting a session."""
        analytics = UsageAnalytics.get_instance()

        analytics.start_session()
        assert analytics._current_session is not None
        assert analytics._current_session.turn_count == 0

    def test_record_turn(self):
        """Test recording turns."""
        analytics = UsageAnalytics.get_instance()

        analytics.start_session()
        analytics.record_turn()
        analytics.record_turn()

        assert analytics._current_session.turn_count == 2

    def test_end_session(self):
        """Test ending a session."""
        analytics = UsageAnalytics.get_instance()

        analytics.start_session()
        analytics.record_turn()
        analytics.record_turn()

        stats = analytics.end_session()

        assert stats is not None
        assert stats.turn_count == 2
        assert stats.end_time is not None

    def test_session_history(self):
        """Test that sessions are added to history."""
        analytics = UsageAnalytics.get_instance()

        analytics.start_session()
        analytics.end_session()
        analytics.start_session()
        analytics.end_session()

        assert len(analytics._session_history) == 2

    def test_session_summary(self):
        """Test session summary calculation."""
        analytics = UsageAnalytics.get_instance()

        for _i in range(3):
            analytics.start_session()
            for _ in range(5):
                analytics.record_turn()
            analytics.end_session()

        summary = analytics.get_session_summary()
        assert summary["total_sessions"] == 3
        assert summary["avg_turns_per_session"] == 5.0


class TestInsights:
    """Tests for insights and recommendations."""

    def test_no_data_insight(self):
        """Test insight when no data exists."""
        analytics = UsageAnalytics.get_instance()

        insights = analytics.get_tool_insights("nonexistent_tool")
        assert insights["status"] == "no_data"

    def test_recommendations_for_slow_tool(self):
        """Test recommendations are generated for slow tools."""
        analytics = UsageAnalytics.get_instance()

        analytics.record_tool_execution(
            tool_name="slow_tool",
            success=True,
            execution_time_ms=2000.0,  # Very slow
        )

        insights = analytics.get_tool_insights("slow_tool")
        assert len(insights["recommendations"]) > 0
        assert any("Slow execution" in r for r in insights["recommendations"])

    def test_recommendations_for_unreliable_tool(self):
        """Test recommendations for unreliable tools."""
        analytics = UsageAnalytics.get_instance()

        for i in range(10):
            analytics.record_tool_execution(
                tool_name="unreliable_tool",
                success=i < 5,  # 50% success rate
                execution_time_ms=100.0,
            )

        insights = analytics.get_tool_insights("unreliable_tool")
        assert len(insights["recommendations"]) > 0
        assert any("success rate" in r.lower() for r in insights["recommendations"])

    def test_get_top_tools(self):
        """Test getting top tools by usage."""
        analytics = UsageAnalytics.get_instance()

        for i, tool in enumerate(["tool_a", "tool_b", "tool_c"]):
            for _ in range(i + 1):
                analytics.record_tool_execution(
                    tool_name=tool,
                    success=True,
                    execution_time_ms=100.0,
                )

        top = analytics.get_top_tools(metric="usage", limit=2)
        assert len(top) == 2
        assert top[0][0] == "tool_c"  # Most used

    def test_optimization_recommendations(self):
        """Test overall optimization recommendations."""
        analytics = UsageAnalytics.get_instance()

        # Add an unreliable tool
        for i in range(10):
            analytics.record_tool_execution(
                tool_name="bad_tool",
                success=i < 6,  # 60% success rate
                execution_time_ms=100.0,
            )

        recommendations = analytics.get_optimization_recommendations()
        assert len(recommendations) > 0
        assert recommendations[0]["priority"] == "high"


class TestExport:
    """Tests for metrics export."""

    def test_prometheus_export(self):
        """Test Prometheus metrics export."""
        analytics = UsageAnalytics.get_instance()

        analytics.record_tool_execution(
            tool_name="test_tool",
            success=True,
            execution_time_ms=100.0,
        )

        metrics = analytics.export_prometheus_metrics()
        assert "victor_tool_executions_total" in metrics
        assert 'tool="test_tool"' in metrics

    def test_json_export(self):
        """Test JSON export."""
        analytics = UsageAnalytics.get_instance()

        analytics.record_tool_execution(
            tool_name="test_tool",
            success=True,
            execution_time_ms=100.0,
        )

        json_str = analytics.export_json()
        data = json.loads(json_str)

        assert "exported_at" in data
        assert "tool_aggregates" in data
        assert "test_tool" in data["tool_aggregates"]


class TestPersistence:
    """Tests for persistence."""

    def test_persistence_to_file(self):
        """Test that data is persisted to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AnalyticsConfig(
                cache_dir=Path(tmpdir),
                persistence_interval_seconds=0,  # Immediate persistence
            )
            analytics = UsageAnalytics(config)

            analytics.record_tool_execution(
                tool_name="test_tool",
                success=True,
                execution_time_ms=100.0,
            )
            analytics.flush()

            # Check file exists
            cache_file = Path(tmpdir) / "usage_analytics.pkl"
            assert cache_file.exists()

    def test_load_from_cache(self):
        """Test loading from cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AnalyticsConfig(cache_dir=Path(tmpdir))

            # Create and populate first instance
            UsageAnalytics.reset_instance()
            analytics1 = UsageAnalytics(config)
            analytics1.record_tool_execution(
                tool_name="test_tool",
                success=True,
                execution_time_ms=100.0,
            )
            analytics1.flush()

            # Create second instance (should load from cache)
            UsageAnalytics.reset_instance()
            analytics2 = UsageAnalytics(config)

            # Should have loaded the data
            assert len(analytics2._tool_records) > 0

    def test_clear_data(self):
        """Test clearing all data."""
        analytics = UsageAnalytics.get_instance()

        analytics.record_tool_execution(
            tool_name="test_tool",
            success=True,
            execution_time_ms=100.0,
        )

        analytics.clear()

        assert len(analytics._tool_records) == 0
        assert len(analytics._tool_aggregates) == 0


class TestFactoryFunction:
    """Tests for factory function."""

    def test_create_usage_analytics(self):
        """Test factory function creates instance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            analytics = create_usage_analytics(cache_dir=Path(tmpdir))

            assert analytics is not None
            assert analytics.config.cache_dir == Path(tmpdir)


class TestConcurrency:
    """Tests for thread safety."""

    def test_concurrent_recording(self):
        """Test concurrent tool recording."""
        import threading

        analytics = UsageAnalytics.get_instance()
        errors = []

        def record_tools():
            try:
                for _ in range(100):
                    analytics.record_tool_execution(
                        tool_name="concurrent_tool",
                        success=True,
                        execution_time_ms=100.0,
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=record_tools) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        # Should have 500 total executions (5 threads * 100 each)
        insights = analytics.get_tool_insights("concurrent_tool")
        assert insights["total_executions"] == 500
