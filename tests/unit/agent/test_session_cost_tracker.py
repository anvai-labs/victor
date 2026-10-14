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

    @pytest.mark.parametrize("outcome", ["success", "error", "cancelled", "closed"])
    def test_invocation_without_usage_counts_without_inventing_tokens(self, outcome):
        from unittest.mock import Mock

        tracker = SessionCostTracker()
        tracker._sandhi = Mock()
        request = tracker.begin_request(provider="ollama", model="test")
        assert tracker.get_summary()["request_count"] == 1
        assert request.outcome == "pending"
        tracker.settle_request(request, outcome=outcome, usage=None)
        assert request.outcome == outcome
        assert request.usage_status == "unknown"
        assert request.to_dict()["tokens"] is None
        assert tracker.get_summary()["usage_coverage"]["unknown"] == 1
        tracker._sandhi.record.assert_not_called()

    def test_invocation_settles_once_and_rejects_foreign_record(self):
        tracker = SessionCostTracker()
        request = tracker.begin_request()
        usage = {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14}
        other = SessionCostTracker()
        with pytest.raises(ValueError, match="pending"):
            other.settle_request(request, outcome="success", usage=usage)
        tracker.settle_request(request, outcome="success", usage=usage)
        with pytest.raises(ValueError, match="pending"):
            tracker.settle_request(request, outcome="success", usage=usage)
        assert len(tracker.requests) == 1
        assert tracker.total_tokens == 14

    def test_reset_rejects_pending_invocations_without_mutating_session(self):
        tracker = SessionCostTracker()
        session_id = tracker.session_id
        request = tracker.begin_request()
        with pytest.raises(ValueError, match="pending"):
            tracker.reset()
        assert tracker.session_id == session_id
        assert tracker.requests == [request]
        tracker.settle_request(request, outcome="success", usage=None)
        tracker.reset()
        assert not tracker.requests

    def test_partial_usage_is_visible_without_claiming_complete_metering(self):
        from unittest.mock import Mock

        tracker = SessionCostTracker()
        tracker._sandhi = Mock()
        request = tracker.begin_request()
        tracker.settle_request(
            request, outcome="cancelled", usage={"prompt_tokens": 10}, usage_complete=False
        )
        assert request.usage_status == "partial"
        assert request.to_dict()["tokens"]["prompt"] == 10
        assert tracker.total_prompt_tokens == 10
        tracker._sandhi.record.assert_not_called()

    def test_invocation_captures_attribution_before_context_changes(self):
        from unittest.mock import Mock
        from victor.core.context import bind_attribution

        tracker = SessionCostTracker(subject_id="default")
        tracker._sandhi = Mock()
        with bind_attribution(subject_id="caller", group_id="team"):
            request = tracker.begin_request(provider="ollama", model="test")
        with bind_attribution(subject_id="different"):
            tracker.settle_request(
                request, outcome="success", usage={"prompt_tokens": 1, "completion_tokens": 2}
            )
        call = tracker._sandhi.record.call_args.kwargs
        assert (call["subject_id"], call["group_id"]) == ("caller", "team")
        assert (call["provider"], call["model"]) == ("ollama", "test")

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

    def test_export_csv(self, tmp_path):
        """Exports distinguish missing, partial and explicitly zero usage."""
        import csv

        tracker = SessionCostTracker()
        tracker.record_request(prompt_tokens=0, completion_tokens=0)
        for usage in (None, {"prompt_tokens": 8}):
            request = tracker.begin_request()
            tracker.settle_request(request, outcome="cancelled", usage=usage, usage_complete=False)
        path = tmp_path / "cost.csv"
        tracker.export_csv(path)
        with path.open() as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 3
        assert [row["usage_status"] for row in rows] == ["reported", "unknown", "partial"]
        assert [row["prompt_tokens"] for row in rows] == ["0", "", "8"]
        assert rows[1]["total_cost"] == ""
        assert rows[1]["outcome"] == "cancelled"

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

    def test_model_override_in_request(self, monkeypatch):
        """A different model must not inherit the session model's prices."""
        from types import SimpleNamespace
        from unittest.mock import Mock

        tracker = SessionCostTracker(provider="anthropic", model="claude-3-5-sonnet-20241022")
        lookup = Mock(
            return_value=SimpleNamespace(
                cost_enabled=True,
                calculate_cost=lambda *_args: {
                    "input_cost": 1.0,
                    "output_cost": 2.0,
                    "cache_cost": 3.0,
                },
            )
        )
        monkeypatch.setattr("victor.config.metrics_capabilities.get_metrics_capabilities", lookup)

        # Record with different model
        request = tracker.record_request(
            prompt_tokens=1000,
            completion_tokens=500,
            model="claude-3-5-haiku-20241022",
        )

        # Request should have override model
        assert request.model == "claude-3-5-haiku-20241022"
        lookup.assert_called_once_with("anthropic", "claude-3-5-haiku-20241022")
        assert request.total_cost == tracker.total_cost == 6.0
