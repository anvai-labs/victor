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

"""Tests for context window monitor - achieving 70%+ coverage."""

import pytest
import time
from unittest.mock import patch

from victor.agent.recovery.context_monitor import (
    ContextHealth,
    CompactionStrategy,
    ContextMetrics,
    CompactionRecommendation,
    ContextWindowMonitor,
)


class TestContextHealth:
    """Tests for ContextHealth enum."""

    def test_context_health_values(self):
        """Test all health status values exist."""
        assert ContextHealth.HEALTHY.name == "HEALTHY"
        assert ContextHealth.WARNING.name == "WARNING"
        assert ContextHealth.CRITICAL.name == "CRITICAL"
        assert ContextHealth.OVERFLOW.name == "OVERFLOW"


class TestCompactionStrategy:
    """Tests for CompactionStrategy enum."""

    def test_compaction_strategy_values(self):
        """Test all compaction strategy values exist."""
        assert CompactionStrategy.NONE.name == "NONE"
        assert CompactionStrategy.TRUNCATE_OLD_MESSAGES.name == "TRUNCATE_OLD_MESSAGES"
        assert CompactionStrategy.SUMMARIZE_TOOL_OUTPUTS.name == "SUMMARIZE_TOOL_OUTPUTS"
        assert CompactionStrategy.SUMMARIZE_CONVERSATION.name == "SUMMARIZE_CONVERSATION"
        assert CompactionStrategy.REMOVE_REDUNDANT.name == "REMOVE_REDUNDANT"
        assert CompactionStrategy.AGGRESSIVE.name == "AGGRESSIVE"


class TestContextMetrics:
    """Tests for ContextMetrics dataclass."""

    def test_default_values(self):
        """Test default values for ContextMetrics."""
        metrics = ContextMetrics()
        assert metrics.total_tokens == 0
        assert metrics.max_tokens == 100000
        assert metrics.message_count == 0
        assert metrics.tool_output_tokens == 0
        assert metrics.system_prompt_tokens == 0
        assert metrics.user_message_tokens == 0
        assert metrics.assistant_message_tokens == 0

    def test_usage_ratio_calculation(self):
        """Test usage ratio calculation."""
        metrics = ContextMetrics(total_tokens=50000, max_tokens=100000)
        assert metrics.usage_ratio == 0.5

    def test_usage_ratio_with_zero_max(self):
        """Test usage ratio with zero max tokens (edge case)."""
        metrics = ContextMetrics(total_tokens=1000, max_tokens=0)
        assert metrics.usage_ratio == 1000.0  # Divides by 1 due to max()

    def test_health_healthy(self):
        """Test healthy status (<50%)."""
        metrics = ContextMetrics(total_tokens=40000, max_tokens=100000)
        assert metrics.health == ContextHealth.HEALTHY

    def test_health_warning(self):
        """Test warning status (50-70%)."""
        metrics = ContextMetrics(total_tokens=60000, max_tokens=100000)
        assert metrics.health == ContextHealth.WARNING

    def test_health_critical(self):
        """Test critical status (70-85%)."""
        metrics = ContextMetrics(total_tokens=75000, max_tokens=100000)
        assert metrics.health == ContextHealth.CRITICAL

    def test_health_overflow(self):
        """Test overflow status (>85%)."""
        metrics = ContextMetrics(total_tokens=90000, max_tokens=100000)
        assert metrics.health == ContextHealth.OVERFLOW

    def test_available_tokens(self):
        """Test available tokens calculation."""
        metrics = ContextMetrics(total_tokens=30000, max_tokens=100000)
        assert metrics.available_tokens == 70000

    def test_available_tokens_when_over_limit(self):
        """Test available tokens when over limit."""
        metrics = ContextMetrics(total_tokens=120000, max_tokens=100000)
        assert metrics.available_tokens == 0


class TestCompactionRecommendation:
    """Tests for CompactionRecommendation dataclass."""

    def test_basic_creation(self):
        """Test basic creation of CompactionRecommendation."""
        rec = CompactionRecommendation(
            strategy=CompactionStrategy.TRUNCATE_OLD_MESSAGES,
            urgency=ContextHealth.WARNING,
            target_reduction_tokens=10000,
            reason="Context at 60% capacity",
        )
        assert rec.strategy == CompactionStrategy.TRUNCATE_OLD_MESSAGES
        assert rec.urgency == ContextHealth.WARNING
        assert rec.target_reduction_tokens == 10000
        assert "60%" in rec.reason

    def test_with_specific_actions(self):
        """Test with specific actions list."""
        actions = ["Remove old messages", "Truncate tool outputs"]
        rec = CompactionRecommendation(
            strategy=CompactionStrategy.AGGRESSIVE,
            urgency=ContextHealth.OVERFLOW,
            target_reduction_tokens=50000,
            reason="Context overflow",
            specific_actions=actions,
        )
        assert len(rec.specific_actions) == 2
        assert "Remove old messages" in rec.specific_actions


class TestContextWindowMonitor:
    """Tests for ContextWindowMonitor class."""

    def test_initialization_defaults(self):
        """Test default initialization."""
        monitor = ContextWindowMonitor()
        assert monitor._max_tokens == 100000 - 4096  # max_context - response_reserve
        assert monitor._response_reserve == 4096

    def test_initialization_custom_values(self):
        """Test initialization with custom values."""
        monitor = ContextWindowMonitor(
            max_context_tokens=50000,
            response_reserve=2000,
            model_name="gpt-4",
        )
        assert monitor._max_tokens == 48000
        assert monitor._response_reserve == 2000
        assert monitor._model_name == "gpt-4"

    def test_initialization_with_custom_thresholds(self):
        """Test initialization with custom thresholds."""
        custom_thresholds = {"warning": 0.4, "critical": 0.6, "overflow": 0.8}
        monitor = ContextWindowMonitor(thresholds=custom_thresholds)
        assert monitor._thresholds["warning"] == 0.4
        assert monitor._thresholds["critical"] == 0.6

    def test_get_token_ratio_claude(self):
        """Test token ratio for Claude model."""
        monitor = ContextWindowMonitor(model_name="claude-3-sonnet")
        assert monitor._get_token_ratio() == 3.5

    def test_get_token_ratio_gpt(self):
        """Test token ratio for GPT model."""
        monitor = ContextWindowMonitor(model_name="gpt-4")
        assert monitor._get_token_ratio() == 4.0

    def test_get_token_ratio_llama(self):
        """Test token ratio for Llama model."""
        monitor = ContextWindowMonitor(model_name="llama-3")
        assert monitor._get_token_ratio() == 3.8

    def test_get_token_ratio_qwen(self):
        """Test token ratio for Qwen model."""
        monitor = ContextWindowMonitor(model_name="qwen2.5")
        assert monitor._get_token_ratio() == 3.5

    def test_get_token_ratio_default(self):
        """Test default token ratio for unknown model."""
        monitor = ContextWindowMonitor(model_name="unknown-model")
        assert monitor._get_token_ratio() == 4.0

    def test_estimate_tokens_empty(self):
        """Test token estimation for empty text."""
        monitor = ContextWindowMonitor()
        assert monitor.estimate_tokens("") == 0

    def test_estimate_tokens_normal_text(self):
        """Test token estimation for normal text."""
        monitor = ContextWindowMonitor()
        text = "Hello world"  # 11 chars
        tokens = monitor.estimate_tokens(text)
        # With default ratio of 4.0, should be ~2.75, so 2
        assert tokens == 2

    def test_update_metrics_empty_messages(self):
        """Test updating metrics with empty messages."""
        monitor = ContextWindowMonitor()
        metrics = monitor.update_metrics([])
        assert metrics.message_count == 0
        assert metrics.total_tokens == 0

    def test_update_metrics_with_system_prompt(self):
        """Test updating metrics with system prompt."""
        monitor = ContextWindowMonitor()
        metrics = monitor.update_metrics([], system_prompt="You are a helpful assistant.")
        assert metrics.system_prompt_tokens > 0

    def test_update_metrics_with_messages(self):
        """Test updating metrics with various messages."""
        monitor = ContextWindowMonitor()
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "tool", "content": "Tool output here"},
        ]
        metrics = monitor.update_metrics(messages)
        assert metrics.message_count == 3
        assert metrics.user_message_tokens > 0
        assert metrics.assistant_message_tokens > 0
        assert metrics.tool_output_tokens > 0

    def test_update_metrics_with_multimodal_content(self):
        """Test updating metrics with multimodal content."""
        monitor = ContextWindowMonitor()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {"type": "image_url", "image_url": {"url": "data:..."}},
                ],
            }
        ]
        metrics = monitor.update_metrics(messages)
        assert metrics.user_message_tokens > 0

    def test_get_health(self):
        """Test getting health status."""
        monitor = ContextWindowMonitor()
        monitor.update_metrics([])
        health = monitor.get_health()
        assert health == ContextHealth.HEALTHY

    def test_get_metrics(self):
        """Test getting metrics."""
        monitor = ContextWindowMonitor()
        monitor.update_metrics([{"role": "user", "content": "Test"}])
        metrics = monitor.get_metrics()
        assert isinstance(metrics, ContextMetrics)
        assert metrics.message_count == 1

    def test_should_compact_healthy(self):
        """Test should_compact returns False when healthy."""
        monitor = ContextWindowMonitor()
        monitor.update_metrics([])
        should_compact, reason = monitor.should_compact()
        assert not should_compact
        assert "healthy" in reason.lower()

    def test_should_compact_overflow(self):
        """Test should_compact returns True on overflow."""
        monitor = ContextWindowMonitor(max_context_tokens=1000, response_reserve=100)
        # Create large messages to trigger overflow
        large_message = "x" * 4000  # Will be ~1000 tokens
        monitor.update_metrics([{"role": "user", "content": large_message}])
        should_compact, reason = monitor.should_compact()
        assert should_compact
        assert "overflow" in reason.lower()

    def test_should_compact_critical(self):
        """Test should_compact returns True on critical."""
        monitor = ContextWindowMonitor(max_context_tokens=1000, response_reserve=100)
        # Create messages to trigger critical (70-85%)
        message = "x" * 3000  # Should be ~750 tokens, ~83% of 900
        monitor.update_metrics([{"role": "user", "content": message}])
        should_compact, reason = monitor.should_compact()
        assert should_compact

    def test_is_growing_fast(self):
        """Test _is_growing_fast detection."""
        monitor = ContextWindowMonitor()
        # Not enough history
        assert not monitor._is_growing_fast()

        # Add history entries to simulate growth
        base_time = int(time.time() * 1000)
        monitor._token_history = [
            (base_time, 1000),
            (base_time + 1000, 1500),
            (base_time + 2000, 2200),  # 120% growth
        ]
        assert monitor._is_growing_fast()

    def test_get_compaction_recommendation_healthy(self):
        """Test getting recommendation when healthy."""
        monitor = ContextWindowMonitor()
        monitor.update_metrics([])
        rec = monitor.get_compaction_recommendation()
        assert rec.strategy == CompactionStrategy.NONE
        assert rec.urgency == ContextHealth.HEALTHY
        assert rec.target_reduction_tokens == 0

    def test_get_compaction_recommendation_warning(self):
        """Test getting recommendation on warning."""
        monitor = ContextWindowMonitor(max_context_tokens=1000, response_reserve=100)
        message = "x" * 2000  # ~500 tokens, ~55% of 900
        monitor.update_metrics([{"role": "user", "content": message}])
        rec = monitor.get_compaction_recommendation()
        assert rec.strategy == CompactionStrategy.TRUNCATE_OLD_MESSAGES
        assert rec.urgency == ContextHealth.WARNING

    def test_get_compaction_recommendation_critical(self):
        """Test getting recommendation on critical."""
        monitor = ContextWindowMonitor(max_context_tokens=1000, response_reserve=100)
        message = "x" * 3000  # ~750 tokens, ~83% of 900
        monitor.update_metrics([{"role": "user", "content": message}])
        rec = monitor.get_compaction_recommendation()
        assert rec.strategy == CompactionStrategy.SUMMARIZE_TOOL_OUTPUTS
        assert rec.urgency == ContextHealth.CRITICAL

    def test_get_compaction_recommendation_overflow(self):
        """Test getting recommendation on overflow."""
        monitor = ContextWindowMonitor(max_context_tokens=1000, response_reserve=100)
        message = "x" * 4000  # ~1000 tokens, overflow
        monitor.update_metrics([{"role": "user", "content": message}])
        rec = monitor.get_compaction_recommendation()
        assert rec.strategy == CompactionStrategy.AGGRESSIVE
        assert rec.urgency == ContextHealth.OVERFLOW
        assert len(rec.specific_actions) > 0

    def test_trigger_compaction_no_callback(self):
        """Test trigger_compaction with no callback."""
        monitor = ContextWindowMonitor()
        result = monitor.trigger_compaction()
        assert result is None

    def test_trigger_compaction_with_callback(self):
        """Test trigger_compaction with callback."""
        monitor = ContextWindowMonitor()

        def mock_callback():
            return 1000  # Freed 1000 tokens

        monitor.register_compaction_callback(mock_callback)

        # Add some metrics first
        monitor.update_metrics([{"role": "user", "content": "test"}])
        freed = monitor.trigger_compaction()

        assert freed == 1000
        assert len(monitor._compaction_history) == 1

    def test_register_compaction_callback(self):
        """Test registering compaction callback."""
        monitor = ContextWindowMonitor()

        def callback():
            return 500

        monitor.register_compaction_callback(callback)
        assert monitor._compaction_callback is not None

    def test_get_token_breakdown(self):
        """Test getting token breakdown."""
        monitor = ContextWindowMonitor()
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]
        monitor.update_metrics(messages, system_prompt="You are helpful.")
        breakdown = monitor.get_token_breakdown()

        assert "total" in breakdown
        assert "max" in breakdown
        assert "available" in breakdown
        assert "usage_ratio" in breakdown
        assert "health" in breakdown
        assert "breakdown" in breakdown
        assert "message_count" in breakdown
        assert breakdown["breakdown"]["system_prompt"] > 0

    def test_get_growth_trend_insufficient_data(self):
        """Test growth trend with insufficient data."""
        monitor = ContextWindowMonitor()
        trend = monitor.get_growth_trend()
        assert trend["trend"] == "insufficient_data"

    def test_get_growth_trend_with_data(self):
        """Test growth trend with data."""
        monitor = ContextWindowMonitor()
        base_time = int(time.time() * 1000)
        monitor._token_history = [
            (base_time, 1000),
            (base_time + 1000, 1010),  # Slow growth
        ]
        trend = monitor.get_growth_trend()
        assert "trend" in trend
        assert "growth_rate_tokens_per_sec" in trend
        assert "samples" in trend

    def test_get_growth_trend_rapid_growth(self):
        """Test detecting rapid growth."""
        monitor = ContextWindowMonitor()
        base_time = int(time.time() * 1000)
        monitor._token_history = [
            (base_time, 1000),
            (base_time + 1000, 2100),  # >100 tokens/sec
        ]
        trend = monitor.get_growth_trend()
        assert trend["trend"] == "rapid_growth"

    def test_get_growth_trend_decreasing(self):
        """Test detecting decreasing trend."""
        monitor = ContextWindowMonitor()
        base_time = int(time.time() * 1000)
        monitor._token_history = [
            (base_time, 2000),
            (base_time + 1000, 1900),  # Decreasing
        ]
        trend = monitor.get_growth_trend()
        assert trend["trend"] in ("decreasing", "stable", "slow_growth")

    def test_predict_overflow_time_not_growing(self):
        """Test predict_overflow_time when not growing."""
        monitor = ContextWindowMonitor()
        result = monitor.predict_overflow_time()
        assert result is None

    def test_predict_overflow_time_growing(self):
        """Test predict_overflow_time when growing."""
        monitor = ContextWindowMonitor(max_context_tokens=10000, response_reserve=1000)
        base_time = int(time.time() * 1000)
        # Set up growth that will overflow
        monitor._token_history = [
            (base_time, 1000),
            (base_time + 1000, 2000),  # 1000 tokens/sec
        ]
        monitor._metrics = ContextMetrics(total_tokens=5000, max_tokens=9000)

        overflow_time = monitor.predict_overflow_time()
        # Should predict when we'll hit overflow threshold
        assert overflow_time is not None or overflow_time == 0.0

    def test_predict_overflow_time_already_overflow(self):
        """Test predict_overflow_time when already at overflow."""
        monitor = ContextWindowMonitor(max_context_tokens=1000, response_reserve=100)
        base_time = int(time.time() * 1000)
        monitor._token_history = [
            (base_time, 800),
            (base_time + 1000, 900),
        ]
        monitor._metrics = ContextMetrics(total_tokens=900, max_tokens=900)

        overflow_time = monitor.predict_overflow_time()
        assert overflow_time == 0.0 or overflow_time is None

    def test_reset(self):
        """Test resetting monitor state."""
        monitor = ContextWindowMonitor()
        monitor.update_metrics([{"role": "user", "content": "Test message"}])
        assert monitor._metrics.message_count > 0
        assert len(monitor._token_history) > 0

        monitor.reset()

        assert monitor._metrics.total_tokens == 0
        assert len(monitor._token_history) == 0

    def test_token_history_limit(self):
        """Test that token history is limited to 100 entries."""
        monitor = ContextWindowMonitor()

        # Add more than 100 updates
        for i in range(110):
            monitor.update_metrics([{"role": "user", "content": f"Message {i}"}])

        assert len(monitor._token_history) <= 100


class TestContextWindowMonitorEdgeCases:
    """Edge case tests for ContextWindowMonitor."""

    def test_empty_content_in_message(self):
        """Test handling empty content in messages."""
        monitor = ContextWindowMonitor()
        messages = [{"role": "user", "content": ""}]
        metrics = monitor.update_metrics(messages)
        assert metrics.user_message_tokens == 0

    def test_missing_role_in_message(self):
        """Test handling missing role in messages."""
        monitor = ContextWindowMonitor()
        messages = [{"content": "Some content"}]
        # Should not raise, handles gracefully
        metrics = monitor.update_metrics(messages)
        assert metrics.message_count == 1

    def test_missing_content_in_message(self):
        """Test handling missing content in messages."""
        monitor = ContextWindowMonitor()
        messages = [{"role": "user"}]
        metrics = monitor.update_metrics(messages)
        assert metrics.message_count == 1

    def test_none_system_prompt(self):
        """Test handling None system prompt."""
        monitor = ContextWindowMonitor()
        metrics = monitor.update_metrics([], system_prompt=None)
        assert metrics.system_prompt_tokens == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
