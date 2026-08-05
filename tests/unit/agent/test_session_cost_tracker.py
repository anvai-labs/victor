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

"""Tests for SessionCostTracker."""

import json
import tempfile
from pathlib import Path

import pytest

from victor.agent.session_cost_tracker import SessionCostTracker, RequestCost


class TestRequestCost:
    """Tests for RequestCost dataclass."""

    def test_to_dict(self):
        """Test RequestCost serialization."""
        request = RequestCost(
            request_id="test-123",
            timestamp=1704067200.0,  # 2024-01-01 00:00:00
            model="claude-3-5-sonnet",
            prompt_tokens=1000,
            completion_tokens=500,
            cache_read_tokens=200,
            cache_write_tokens=100,
            total_tokens=1500,
            input_cost=0.003,
            output_cost=0.0075,
            cache_cost=0.0004,
            total_cost=0.0109,
            duration_seconds=2.5,
            tool_calls=3,
        )

        data = request.to_dict()

        assert data["request_id"] == "test-123"
        assert data["model"] == "claude-3-5-sonnet"
        assert data["tokens"]["prompt"] == 1000
        assert data["tokens"]["completion"] == 500
        assert data["cost"]["total"] == 0.0109
        assert data["metadata"]["tool_calls"] == 3


class TestSessionCostTracker:
    """Tests for SessionCostTracker."""

    def test_initialization(self):
        """Test tracker initialization."""
        tracker = SessionCostTracker(provider="anthropic", model="claude-3-5-sonnet-20241022")

        assert tracker.provider == "anthropic"
        assert tracker.model == "claude-3-5-sonnet-20241022"
        assert len(tracker.session_id) > 0
        assert tracker.total_cost == 0.0
        assert len(tracker.requests) == 0

    def test_record_single_request(self):
        """Test recording a single request."""
        tracker = SessionCostTracker(provider="anthropic", model="claude-3-5-sonnet-20241022")

        request = tracker.record_request(
            prompt_tokens=1000,
            completion_tokens=500,
            cache_read_tokens=200,
        )

        assert len(tracker.requests) == 1
        assert tracker.total_prompt_tokens == 1000
        assert tracker.total_completion_tokens == 500
        assert tracker.total_cache_read_tokens == 200
        assert tracker.total_cost > 0
        assert request.total_cost > 0

    def test_record_multiple_requests(self):
        """Test recording multiple requests."""
        tracker = SessionCostTracker(provider="anthropic", model="claude-3-5-sonnet-20241022")

        tracker.record_request(prompt_tokens=1000, completion_tokens=500)
        tracker.record_request(prompt_tokens=800, completion_tokens=300)
        tracker.record_request(prompt_tokens=1200, completion_tokens=600)

        assert len(tracker.requests) == 3
        assert tracker.total_prompt_tokens == 3000
        assert tracker.total_completion_tokens == 1400
        assert tracker.total_tokens == 4400

    def test_cumulative_cost(self):
        """Test that costs accumulate correctly."""
        tracker = SessionCostTracker(provider="anthropic", model="claude-3-5-sonnet-20241022")

        req1 = tracker.record_request(prompt_tokens=1000, completion_tokens=500)
        cost_after_1 = tracker.total_cost

        req2 = tracker.record_request(prompt_tokens=1000, completion_tokens=500)
        cost_after_2 = tracker.total_cost

        # Costs should accumulate
        assert cost_after_2 == pytest.approx(cost_after_1 * 2, rel=1e-6)
        assert cost_after_2 == pytest.approx(req1.total_cost + req2.total_cost, rel=1e-6)

    def test_format_inline_cost_enabled(self):
        """Test inline cost formatting with cost enabled."""
        tracker = SessionCostTracker(provider="anthropic", model="claude-3-5-sonnet-20241022")
        tracker.record_request(prompt_tokens=1000, completion_tokens=500)

        inline = tracker.format_inline_cost()

        assert inline.startswith("$")
        assert len(inline) > 1

    def test_format_inline_cost_disabled(self):
        """Test inline cost formatting with cost disabled."""
        tracker = SessionCostTracker(provider="ollama", model="llama3")
        tracker.record_request(prompt_tokens=1000, completion_tokens=500)

        inline = tracker.format_inline_cost()

        assert inline == "cost n/a"

    def test_get_summary(self):
        """Test session summary generation."""
        tracker = SessionCostTracker(provider="anthropic", model="claude-3-5-sonnet-20241022")
        tracker.record_request(prompt_tokens=1000, completion_tokens=500)
        tracker.record_request(prompt_tokens=2000, completion_tokens=1000)

        summary = tracker.get_summary()

        assert summary["provider"] == "anthropic"
        assert summary["request_count"] == 2
        assert summary["tokens"]["prompt"] == 3000
        assert summary["tokens"]["completion"] == 1500
        assert summary["cost"]["total"] > 0
        assert "averages" in summary

    def test_get_formatted_summary(self):
        """Test human-readable summary."""
        tracker = SessionCostTracker(provider="anthropic", model="claude-3-5-sonnet-20241022")
        tracker.record_request(prompt_tokens=1000, completion_tokens=500)

        summary = tracker.get_formatted_summary()

        assert "Session Cost Summary" in summary
        assert "anthropic" in summary
        assert "Input:" in summary
        assert "Output:" in summary
        assert "Cost (USD):" in summary

    def test_export_json(self):
        """Test JSON export."""
        tracker = SessionCostTracker(provider="anthropic", model="claude-3-5-sonnet-20241022")
        tracker.record_request(prompt_tokens=1000, completion_tokens=500)
        tracker.record_request(prompt_tokens=800, completion_tokens=400)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            tracker.export_json(path)

            with open(path) as f:
                data = json.load(f)

            assert "summary" in data
            assert "requests" in data
            assert len(data["requests"]) == 2
            assert data["summary"]["request_count"] == 2
        finally:
            path.unlink(missing_ok=True)

    def test_export_csv(self):
        """Test CSV export."""
        tracker = SessionCostTracker(provider="anthropic", model="claude-3-5-sonnet-20241022")
        tracker.record_request(prompt_tokens=1000, completion_tokens=500)
        tracker.record_request(prompt_tokens=800, completion_tokens=400)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            path = Path(f.name)

        try:
            tracker.export_csv(path)

            with open(path) as f:
                lines = f.readlines()

            # Header + 2 data rows
            assert len(lines) == 3
            assert "prompt_tokens" in lines[0]
            assert "total_cost" in lines[0]
        finally:
            path.unlink(missing_ok=True)

    def test_reset(self):
        """Test session reset."""
        tracker = SessionCostTracker(provider="anthropic", model="claude-3-5-sonnet-20241022")
        tracker.record_request(prompt_tokens=1000, completion_tokens=500)
        old_session_id = tracker.session_id

        tracker.reset()

        assert tracker.session_id != old_session_id
        assert len(tracker.requests) == 0
        assert tracker.total_cost == 0.0
        assert tracker.total_prompt_tokens == 0

    def test_free_provider_no_cost(self):
        """Test that free providers don't calculate costs."""
        tracker = SessionCostTracker(provider="ollama", model="llama3")
        tracker.record_request(prompt_tokens=10000, completion_tokens=5000)

        assert tracker.total_cost == 0.0
        assert tracker.total_prompt_tokens == 10000
        assert tracker.total_completion_tokens == 5000

    def test_model_override_in_request(self):
        """Test model override for individual request."""
        tracker = SessionCostTracker(provider="anthropic", model="claude-3-5-sonnet-20241022")

        # Record with different model
        request = tracker.record_request(
            prompt_tokens=1000,
            completion_tokens=500,
            model="claude-3-5-haiku-20241022",
        )

        # Request should have override model
        assert request.model == "claude-3-5-haiku-20241022"
        # But cost calculation uses tracker's capabilities
