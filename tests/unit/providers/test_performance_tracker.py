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

"""
Unit tests for provider performance tracker.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch

from victor.providers.performance_tracker import (
    ProviderDegradationSnapshot,
    RequestMetric,
    ProviderPerformanceTracker,
)


class TestRequestMetric:
    """Tests for RequestMetric dataclass."""

    def test_metric_creation(self):
        """Test creating a request metric."""
        metric = RequestMetric(
            provider="ollama",
            model="qwen3-coder:30b",
            success=True,
            latency_ms=1250.0,
            timestamp=datetime.now(),
        )

        assert metric.provider == "ollama"
        assert metric.model == "qwen3-coder:30b"
        assert metric.success is True
        assert metric.latency_ms == 1250.0
        assert metric.error_type is None

    def test_metric_with_error(self):
        """Test creating a metric for failed request."""
        metric = RequestMetric(
            provider="anthropic",
            model="claude-3-5-sonnet-20241022",
            success=False,
            latency_ms=5000.0,
            timestamp=datetime.now(),
            error_type="RateLimitError",
        )

        assert metric.success is False
        assert metric.error_type == "RateLimitError"

    def test_metric_normalization(self):
        """Test provider name normalization."""
        metric = RequestMetric(
            provider="OLLAMA",
            model="test",
            success=True,
            latency_ms=1000.0,
            timestamp=datetime.now(),
        )

        # Provider should be normalized by tracker, not metric
        assert metric.provider == "OLLAMA"

    def test_to_dict(self):
        """Test converting metric to dictionary."""
        now = datetime.now()
        metric = RequestMetric(
            provider="ollama",
            model="test",
            success=True,
            latency_ms=1000.0,
            timestamp=now,
        )

        data = metric.to_dict()

        assert data["provider"] == "ollama"
        assert data["model"] == "test"
        assert data["success"] is True
        assert data["latency_ms"] == 1000.0
        assert data["timestamp"] == now.isoformat()


class TestProviderPerformanceTracker:
    """Tests for ProviderPerformanceTracker."""

    @pytest.fixture(autouse=True)
    def _disable_db(self):
        """Patch the global DatabaseManager so tracker stays in-memory for unit tests."""
        with patch(
            "victor.core.database.DatabaseManager",
            side_effect=RuntimeError("no db in unit tests"),
        ):
            yield

    def test_tracker_initialization(self):
        """Test tracker initialization."""
        tracker = ProviderPerformanceTracker(window_size=50)

        assert tracker.window_size == 50
        assert len(tracker.metrics) == 0

    def test_explicit_none_db_disables_persistence(self):
        """db=None should stay purely in-memory and skip DatabaseManager bootstrap."""
        with patch("victor.core.database.DatabaseManager") as mock_db:
            tracker = ProviderPerformanceTracker(db=None)

        assert tracker._db is None
        mock_db.assert_not_called()

    def test_record_request(self):
        """Test recording a request metric."""
        tracker = ProviderPerformanceTracker()
        metric = RequestMetric(
            provider="ollama",
            model="test",
            success=True,
            latency_ms=1000.0,
            timestamp=datetime.now(),
        )

        tracker.record_request(metric)

        assert "ollama" in tracker.metrics
        assert len(tracker.get_metrics("ollama")) == 1

    def test_provider_normalization(self):
        """Test that provider names are normalized to lowercase."""
        tracker = ProviderPerformanceTracker()
        metric = RequestMetric(
            provider="OLLAMA",
            model="test",
            success=True,
            latency_ms=1000.0,
            timestamp=datetime.now(),
        )

        tracker.record_request(metric)

        # Should be accessible with lowercase
        assert "ollama" in tracker.metrics
        assert len(tracker.get_metrics("OLLAMA")) == 1

    def test_window_size_limit(self):
        """Test that metrics are limited to window size."""
        tracker = ProviderPerformanceTracker(window_size=5)

        # Add 10 metrics
        for i in range(10):
            metric = RequestMetric(
                provider="ollama",
                model="test",
                success=True,
                latency_ms=1000.0,
                timestamp=datetime.now(),
            )
            tracker.record_request(metric)

        # Should only keep 5 most recent
        assert len(tracker.get_metrics("ollama")) == 5

    def test_get_success_rate_all_success(self):
        """Test success rate with all successful requests."""
        tracker = ProviderPerformanceTracker()

        for _ in range(10):
            metric = RequestMetric(
                provider="ollama",
                model="test",
                success=True,
                latency_ms=1000.0,
                timestamp=datetime.now(),
            )
            tracker.record_request(metric)

        assert tracker.get_success_rate("ollama") == 1.0

    def test_get_success_rate_mixed(self):
        """Test success rate with mixed results."""
        tracker = ProviderPerformanceTracker()

        # 5 success, 5 failures
        for i in range(5):
            tracker.record_request(
                RequestMetric(
                    provider="ollama",
                    model="test",
                    success=True,
                    latency_ms=1000.0,
                    timestamp=datetime.now(),
                )
            )
            tracker.record_request(
                RequestMetric(
                    provider="ollama",
                    model="test",
                    success=False,
                    latency_ms=1000.0,
                    timestamp=datetime.now(),
                )
            )

        assert tracker.get_success_rate("ollama") == 0.5

    def test_get_success_rate_unknown_provider(self):
        """Test success rate for unknown provider."""
        tracker = ProviderPerformanceTracker()

        # Unknown provider should return neutral score
        assert tracker.get_success_rate("unknown") == 0.5

    def test_get_average_latency(self):
        """Test average latency calculation."""
        tracker = ProviderPerformanceTracker()

        latencies = [1000.0, 2000.0, 3000.0]
        for latency in latencies:
            tracker.record_request(
                RequestMetric(
                    provider="ollama",
                    model="test",
                    success=True,
                    latency_ms=latency,
                    timestamp=datetime.now(),
                )
            )

        assert tracker.get_average_latency("ollama") == 2000.0

    def test_get_average_latency_ignores_failures(self):
        """Test that failed requests are excluded from latency."""
        tracker = ProviderPerformanceTracker()

        # Success: 1000ms, Failure: 5000ms, Success: 2000ms
        tracker.record_request(
            RequestMetric(
                provider="ollama",
                model="test",
                success=True,
                latency_ms=1000.0,
                timestamp=datetime.now(),
            )
        )
        tracker.record_request(
            RequestMetric(
                provider="ollama",
                model="test",
                success=False,
                latency_ms=5000.0,
                timestamp=datetime.now(),
            )
        )
        tracker.record_request(
            RequestMetric(
                provider="ollama",
                model="test",
                success=True,
                latency_ms=2000.0,
                timestamp=datetime.now(),
            )
        )

        # Should only average successful requests
        assert tracker.get_average_latency("ollama") == 1500.0

    def test_get_latency_trend_insufficient_data(self):
        """Test latency trend with insufficient data."""
        tracker = ProviderPerformanceTracker()

        # Add only 5 metrics
        for _ in range(5):
            tracker.record_request(
                RequestMetric(
                    provider="ollama",
                    model="test",
                    success=True,
                    latency_ms=1000.0,
                    timestamp=datetime.now(),
                )
            )

        assert tracker.get_latency_trend("ollama") == "stable"

    def test_get_latency_trend_improving(self):
        """Test improving latency trend."""
        tracker = ProviderPerformanceTracker()
        now = datetime.now()

        # Older: slow, Recent: fast
        for i in range(10):
            latency = 2000.0 if i < 5 else 1000.0
            timestamp = now + timedelta(seconds=i)
            tracker.record_request(
                RequestMetric(
                    provider="ollama",
                    model="test",
                    success=True,
                    latency_ms=latency,
                    timestamp=timestamp,
                )
            )

        assert tracker.get_latency_trend("ollama") == "improving"

    def test_get_latency_trend_degrading(self):
        """Test degrading latency trend."""
        tracker = ProviderPerformanceTracker()
        now = datetime.now()

        # Older: fast, Recent: slow
        for i in range(10):
            latency = 1000.0 if i < 5 else 2000.0
            timestamp = now + timedelta(seconds=i)
            tracker.record_request(
                RequestMetric(
                    provider="ollama",
                    model="test",
                    success=True,
                    latency_ms=latency,
                    timestamp=timestamp,
                )
            )

        assert tracker.get_latency_trend("ollama") == "degrading"

    def test_get_provider_score(self):
        """Test composite provider score calculation."""
        tracker = ProviderPerformanceTracker()

        # Record 10 successful, fast requests
        for _ in range(10):
            tracker.record_request(
                RequestMetric(
                    provider="ollama",
                    model="test",
                    success=True,
                    latency_ms=500.0,
                    timestamp=datetime.now(),
                )
            )

        score = tracker.get_provider_score("ollama")

        # Should be high (good success rate + low latency)
        assert score > 0.7
        assert score <= 1.0

    def test_get_all_scores(self):
        """Test getting scores for all providers."""
        tracker = ProviderPerformanceTracker()

        # Add metrics for multiple providers
        for provider in ["ollama", "anthropic", "openai"]:
            for _ in range(5):
                tracker.record_request(
                    RequestMetric(
                        provider=provider,
                        model="test",
                        success=True,
                        latency_ms=1000.0,
                        timestamp=datetime.now(),
                    )
                )

        scores = tracker.get_all_scores()

        assert len(scores) == 3
        assert "ollama" in scores
        assert "anthropic" in scores
        assert "openai" in scores

    def test_get_best_provider(self):
        """Test selecting best provider from candidates."""
        tracker = ProviderPerformanceTracker()

        # Make ollama perform better
        for _ in range(10):
            tracker.record_request(
                RequestMetric(
                    provider="ollama",
                    model="test",
                    success=True,
                    latency_ms=500.0,
                    timestamp=datetime.now(),
                )
            )
            tracker.record_request(
                RequestMetric(
                    provider="anthropic",
                    model="test",
                    success=True,
                    latency_ms=2000.0,
                    timestamp=datetime.now(),
                )
            )

        best = tracker.get_best_provider(["ollama", "anthropic"])

        assert best == "ollama"

    def test_get_best_provider_empty_candidates(self):
        """Test best provider with empty candidates."""
        tracker = ProviderPerformanceTracker()

        best = tracker.get_best_provider([])

        assert best is None

    def test_reset_provider(self):
        """Test resetting metrics for a provider."""
        tracker = ProviderPerformanceTracker()

        tracker.record_request(
            RequestMetric(
                provider="ollama",
                model="test",
                success=True,
                latency_ms=1000.0,
                timestamp=datetime.now(),
            )
        )

        tracker.reset_provider("ollama")

        assert len(tracker.get_metrics("ollama")) == 0

    def test_reset_all(self):
        """Test resetting all metrics."""
        tracker = ProviderPerformanceTracker()

        for provider in ["ollama", "anthropic"]:
            tracker.record_request(
                RequestMetric(
                    provider=provider,
                    model="test",
                    success=True,
                    latency_ms=1000.0,
                    timestamp=datetime.now(),
                )
            )

        tracker.reset_all()

        assert len(tracker.metrics) == 0

    def test_get_stats(self):
        """Test getting tracker statistics."""
        tracker = ProviderPerformanceTracker(window_size=50)

        # Add metrics
        for _ in range(10):
            tracker.record_request(
                RequestMetric(
                    provider="ollama",
                    model="test",
                    success=True,
                    latency_ms=1000.0,
                    timestamp=datetime.now(),
                )
            )

        stats = tracker.get_stats()

        assert stats["window_size"] == 50
        assert stats["providers_tracked"] == 1
        assert "ollama" in stats["provider_stats"]
        assert stats["provider_stats"]["ollama"]["total_requests"] == 10

    def test_get_degradation_snapshot_detects_persistent_failure_streak(self):
        tracker = ProviderPerformanceTracker()
        now = datetime.now()

        for i in range(2):
            tracker.record_request(
                RequestMetric(
                    provider="ollama",
                    model="test",
                    success=True,
                    latency_ms=800.0,
                    timestamp=now + timedelta(seconds=i),
                )
            )
        for i in range(2, 5):
            tracker.record_request(
                RequestMetric(
                    provider="ollama",
                    model="test",
                    success=False,
                    latency_ms=2400.0,
                    timestamp=now + timedelta(seconds=i),
                    error_type="ProviderError",
                )
            )

        snapshot = tracker.get_degradation_snapshot("ollama")

        assert isinstance(snapshot, ProviderDegradationSnapshot)
        assert snapshot.degraded is True
        assert snapshot.failure_streak == 3
        assert "failure_streak" in snapshot.degradation_reasons
        assert snapshot.recent_error_types == {"ProviderError": 3}

    def test_get_degradation_snapshot_detects_recovery_after_multi_failure_incident(
        self,
    ):
        tracker = ProviderPerformanceTracker()
        now = datetime.now()

        for i in range(3):
            tracker.record_request(
                RequestMetric(
                    provider="ollama",
                    model="test",
                    success=False,
                    latency_ms=2200.0,
                    timestamp=now + timedelta(seconds=i),
                    error_type="ProviderError",
                )
            )
        for i in range(3, 7):
            tracker.record_request(
                RequestMetric(
                    provider="ollama",
                    model="test",
                    success=True,
                    latency_ms=700.0,
                    timestamp=now + timedelta(seconds=i),
                )
            )

        snapshot = tracker.get_degradation_snapshot("ollama")

        assert snapshot.degraded is False
        assert snapshot.recovered_from_recent_incident is True
        assert snapshot.success_streak == 4
        assert snapshot.recent_incident_failure_count == 3
        assert snapshot.time_to_recover_seconds == pytest.approx(3.0)
        assert snapshot.last_failure_at is not None
        assert snapshot.last_recovery_at is not None
