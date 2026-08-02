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

"""Unit tests for AggregationService."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from victor.observability.aggregation_service import (
    AggregationService,
    AggregationServiceConfig,
)
from victor.core.events import MessagingEvent


@pytest.fixture
def aggregation_service():
    """Create AggregationService instance for testing."""
    return AggregationService()


@pytest.fixture
def mock_events():
    """Create mock events for testing."""
    events = []
    base_time = datetime.now()

    # Create tool events with different durations and success rates
    for i in range(100):
        event = MagicMock(spec=MessagingEvent)
        event.topic = "tool.end"
        event.datetime = base_time - timedelta(minutes=i)
        event.data = {
            "tool_name": f"tool_{i % 3}",  # 3 different tools
            "success": i % 10 != 0,  # 90% success rate
            "duration_ms": 50 + i * 2,  # Varying durations
        }
        events.append(event)

    return events


class TestAggregationServiceConfig:
    """Tests for AggregationServiceConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = AggregationServiceConfig()
        assert config.time_windows == {"1h": 3600, "24h": 86400, "7d": 604800}
        assert config.cache_ttl_seconds == 300
        assert config.max_events_per_query == 10000

    def test_custom_config(self):
        """Test custom configuration values."""
        config = AggregationServiceConfig(
            time_windows={"30m": 1800},
            cache_ttl_seconds=600,
            max_events_per_query=5000,
        )
        assert config.time_windows == {"30m": 1800}
        assert config.cache_ttl_seconds == 600
        assert config.max_events_per_query == 5000


class TestAggregationService:
    """Tests for AggregationService."""

    def test_initialization(self, aggregation_service):
        """Test service initialization."""
        assert aggregation_service.config is not None
        assert aggregation_service._query_service is not None
        assert aggregation_service._cache == {}

    def test_parse_time_window_predefined(self, aggregation_service):
        """Test parsing predefined time windows."""
        assert aggregation_service._parse_time_window("1h") == timedelta(seconds=3600)
        assert aggregation_service._parse_time_window("24h") == timedelta(seconds=86400)
        assert aggregation_service._parse_time_window("7d") == timedelta(seconds=604800)

    def test_parse_time_window_custom(self, aggregation_service):
        """Test parsing custom time windows."""
        assert aggregation_service._parse_time_window("30m") == timedelta(minutes=30)
        assert aggregation_service._parse_time_window("2h") == timedelta(hours=2)
        assert aggregation_service._parse_time_window("3d") == timedelta(days=3)
        assert aggregation_service._parse_time_window("90s") == timedelta(seconds=90)

    def test_parse_time_window_invalid(self, aggregation_service):
        """Test parsing invalid time window raises ValueError."""
        with pytest.raises(ValueError, match="Invalid time window format"):
            aggregation_service._parse_time_window("invalid")

    def test_get_cache_key(self, aggregation_service):
        """Test cache key generation."""
        key1 = aggregation_service._get_cache_key("test_method", param1="value1", param2="value2")
        key2 = aggregation_service._get_cache_key("test_method", param2="value2", param1="value1")
        key3 = aggregation_service._get_cache_key("test_method", param1="value3")

        # Same parameters should produce same key regardless of order
        assert key1 == key2
        # Different parameters should produce different key
        assert key1 != key3

    @pytest.mark.asyncio
    async def test_get_metrics_history_caching(self, aggregation_service, mock_events):
        """Test metrics history caching behavior."""
        with patch.object(
            aggregation_service._query_service,
            "get_recent_events",
            return_value=mock_events,
        ) as mock_get_events:
            # First call should query events
            result1 = await aggregation_service.get_metrics_history(time_window="1h")
            assert mock_get_events.call_count == 1

            # Second call should use cache
            result2 = await aggregation_service.get_metrics_history(time_window="1h")
            assert mock_get_events.call_count == 1  # No additional calls

            # Results should be identical
            assert result1 == result2

    @pytest.mark.asyncio
    async def test_get_tool_statistics_aggregation(self, aggregation_service, mock_events):
        """Test tool statistics aggregation."""
        with patch.object(
            aggregation_service._query_service,
            "get_recent_events",
            return_value=mock_events,
        ):
            result = await aggregation_service.get_tool_statistics()

            assert "tools" in result
            assert len(result["tools"]) == 3  # 3 different tools

            # Check aggregation
            for tool_stat in result["tools"]:
                assert "tool_name" in tool_stat
                assert "total_calls" in tool_stat
                assert "success_rate" in tool_stat
                assert "avg_duration_ms" in tool_stat
                assert "p50_duration_ms" in tool_stat
                assert "p95_duration_ms" in tool_stat
                assert "p99_duration_ms" in tool_stat

                # Verify success rate is between 0 and 1
                assert 0 <= tool_stat["success_rate"] <= 1

                # Verify percentiles are ordered correctly
                if tool_stat["p50_duration_ms"]:
                    assert (
                        tool_stat["p50_duration_ms"]
                        <= tool_stat["p95_duration_ms"]
                        <= tool_stat["p99_duration_ms"]
                    )

    @pytest.mark.asyncio
    async def test_get_performance_metrics(self, aggregation_service, mock_events):
        """Test performance metrics calculation."""
        with patch.object(
            aggregation_service._query_service,
            "get_recent_events",
            return_value=mock_events,
        ):
            result = await aggregation_service.get_performance_metrics()

            assert "latency_ms" in result
            assert "throughput" in result
            assert "errors" in result

            # Check latency structure
            latency = result["latency_ms"]
            assert "p50" in latency
            assert "p95" in latency
            assert "p99" in latency
            assert "avg" in latency

            # Check throughput structure
            throughput = result["throughput"]
            assert "requests_per_second" in throughput
            assert "tool_calls_per_second" in throughput

            # Check errors structure
            errors = result["errors"]
            assert "total_errors" in errors
            assert "error_rate" in errors

            # Verify percentiles are ordered correctly
            if latency["p50"]:
                assert latency["p50"] <= latency["p95"] <= latency["p99"]

    @pytest.mark.asyncio
    async def test_cache_expiration(self, aggregation_service, mock_events):
        """Test cache expiration after TTL."""
        with patch.object(
            aggregation_service._query_service,
            "get_recent_events",
            return_value=mock_events,
        ) as mock_get_events:
            # Set very short TTL for testing
            aggregation_service.config.cache_ttl_seconds = 0

            # First call
            await aggregation_service.get_metrics_history(time_window="1h")
            assert mock_get_events.call_count == 1

            # Second call should bypass expired cache
            await aggregation_service.get_metrics_history(time_window="1h")
            assert mock_get_events.call_count == 2
