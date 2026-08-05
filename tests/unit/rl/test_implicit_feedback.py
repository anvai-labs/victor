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

"""Unit tests for implicit feedback collection.

Tests the ImplicitFeedbackCollector and ImplicitFeedback classes
for deriving reward signals from session behavior.
"""

import time
import pytest
from unittest.mock import MagicMock

from victor.framework.rl.implicit_feedback import (
    ImplicitFeedback,
    ImplicitFeedbackCollector,
    SessionContext,
    ToolExecution,
)


class TestImplicitFeedback:
    """Tests for ImplicitFeedback dataclass."""

    def test_compute_reward_successful_session(self) -> None:
        """Successful session should have positive reward."""
        feedback = ImplicitFeedback(
            tool_success_rate=0.9,
            task_completed=True,
            retry_count=0,
            efficiency_score=0.8,
            grounding_score=0.85,
            quality_score=0.75,
        )

        reward = feedback.compute_reward()

        # Should be positive for successful session
        assert reward > 0.0

    def test_compute_reward_failed_session(self) -> None:
        """Failed session should have negative reward."""
        feedback = ImplicitFeedback(
            tool_success_rate=0.3,
            task_completed=False,
            retry_count=5,
            efficiency_score=0.2,
            grounding_score=0.4,
            quality_score=0.3,
        )

        reward = feedback.compute_reward()

        # Should be negative for failed session
        assert reward < 0.0

    def test_compute_reward_retries_penalize(self) -> None:
        """Retries should decrease reward."""
        feedback_no_retry = ImplicitFeedback(
            tool_success_rate=0.8,
            task_completed=True,
            retry_count=0,
            efficiency_score=0.7,
            grounding_score=0.8,
            quality_score=0.7,
        )

        feedback_with_retry = ImplicitFeedback(
            tool_success_rate=0.8,
            task_completed=True,
            retry_count=3,
            efficiency_score=0.7,
            grounding_score=0.8,
            quality_score=0.7,
        )

        reward_no_retry = feedback_no_retry.compute_reward()
        reward_with_retry = feedback_with_retry.compute_reward()

        assert reward_no_retry > reward_with_retry

    def test_compute_reward_custom_weights(self) -> None:
        """Custom weights should affect reward calculation."""
        feedback = ImplicitFeedback(
            tool_success_rate=0.5,
            task_completed=True,  # 1.0
            efficiency_score=0.5,
            grounding_score=0.5,
            quality_score=0.5,
        )

        # Default weights
        reward_default = feedback.compute_reward()

        # Custom weights emphasizing task completion
        custom_weights = {
            "task_completed": 0.8,
            "tool_success_rate": 0.05,
            "grounding_score": 0.05,
            "efficiency_score": 0.05,
            "quality_score": 0.05,
        }
        reward_custom = feedback.compute_reward(custom_weights)

        # With custom weights emphasizing completion, reward should be higher
        assert reward_custom > reward_default

    def test_reward_range(self) -> None:
        """Reward should be in [-1, 1] range."""
        # Best case
        best = ImplicitFeedback(
            tool_success_rate=1.0,
            task_completed=True,
            retry_count=0,
            efficiency_score=1.0,
            grounding_score=1.0,
            quality_score=1.0,
        )
        assert -1.0 <= best.compute_reward() <= 1.0

        # Worst case
        worst = ImplicitFeedback(
            tool_success_rate=0.0,
            task_completed=False,
            retry_count=10,
            efficiency_score=0.0,
            grounding_score=0.0,
            quality_score=0.0,
        )
        assert -1.0 <= worst.compute_reward() <= 1.0


class TestImplicitFeedbackCollector:
    """Tests for ImplicitFeedbackCollector."""

    @pytest.fixture
    def collector(self) -> ImplicitFeedbackCollector:
        """Fixture for clean collector instance."""
        return ImplicitFeedbackCollector()

    def test_start_session(self, collector: ImplicitFeedbackCollector) -> None:
        """Test session start."""
        session = collector.start_session(
            session_id="test_123",
            task_type="analysis",
            provider="anthropic",
            model="claude-3",
        )

        assert session.session_id == "test_123"
        assert session.task_type == "analysis"
        assert session.provider == "anthropic"
        assert session.model == "claude-3"
        assert session.start_time > 0

    def test_get_session(self, collector: ImplicitFeedbackCollector) -> None:
        """Test getting an active session."""
        collector.start_session("sess_1", "analysis")

        session = collector.get_session("sess_1")
        assert session is not None
        assert session.session_id == "sess_1"

        # Non-existent session
        assert collector.get_session("nonexistent") is None

    def test_record_tool_execution(self, collector: ImplicitFeedbackCollector) -> None:
        """Test recording tool executions."""
        session = collector.start_session("sess_1", "action")

        collector.record_tool_execution(
            session,
            tool_name="code_search",
            success=True,
            execution_time_ms=150.0,
        )
        collector.record_tool_execution(
            session,
            tool_name="read_file",
            success=True,
            execution_time_ms=50.0,
        )
        collector.record_tool_execution(
            session,
            tool_name="write_file",
            success=False,
            execution_time_ms=100.0,
            error_message="Permission denied",
        )

        assert len(session.tool_executions) == 3
        assert session.tool_executions[0].tool_name == "code_search"
        assert session.tool_executions[2].success is False

    def test_record_tool_retry(self, collector: ImplicitFeedbackCollector) -> None:
        """Test recording tool retries."""
        session = collector.start_session("sess_1", "action")

        collector.record_tool_execution(session, "api_call", False, 100.0)
        collector.record_tool_execution(session, "api_call", True, 100.0, is_retry=True)

        assert session.total_retries == 1

    def test_record_grounding_result(self, collector: ImplicitFeedbackCollector) -> None:
        """Test recording grounding results."""
        session = collector.start_session("sess_1", "analysis")

        collector.record_grounding_result(session, 0.85)
        collector.record_grounding_result(session, 0.90)

        assert len(session.grounding_results) == 2
        assert session.grounding_results[0] == 0.85

    def test_record_iteration(self, collector: ImplicitFeedbackCollector) -> None:
        """Test recording iterations."""
        session = collector.start_session("sess_1", "action")

        collector.record_iteration(session)
        collector.record_iteration(session)
        collector.record_iteration(session)

        assert session.iteration_count == 3

    def test_end_session_completed(self, collector: ImplicitFeedbackCollector) -> None:
        """Test ending a completed session."""
        session = collector.start_session("sess_1", "analysis", provider="openai", model="gpt-4")

        # Simulate some activity
        collector.record_tool_execution(session, "search", True, 100.0)
        collector.record_tool_execution(session, "read", True, 50.0)
        collector.record_grounding_result(session, 0.9)
        collector.record_iteration(session)

        feedback = collector.end_session(session, completed=True)

        assert feedback is not None
        assert feedback.task_completed is True
        assert feedback.tool_success_rate == 1.0  # 2/2 successful
        assert feedback.session_id == "sess_1"
        assert feedback.provider == "openai"
        assert feedback.model == "gpt-4"
        assert feedback.grounding_score == 0.9
        assert feedback.tool_count == 2

    def test_end_session_failed(self, collector: ImplicitFeedbackCollector) -> None:
        """Test ending a failed session."""
        session = collector.start_session("sess_1", "action")

        collector.record_tool_execution(session, "write", False, 100.0)
        collector.record_tool_execution(session, "write", False, 100.0, is_retry=True)

        feedback = collector.end_session(session, completed=False)

        assert feedback.task_completed is False
        assert feedback.tool_success_rate == 0.0  # 0/2 successful
        assert feedback.retry_count == 1

    def test_efficiency_score_calculation(self, collector: ImplicitFeedbackCollector) -> None:
        """Test efficiency score based on duration."""
        # Quick session (should be efficient)
        session = collector.start_session("sess_quick", "search")
        session.start_time = time.time() - 5  # 5 seconds ago

        feedback = collector.end_session(session, completed=True)
        quick_efficiency = feedback.efficiency_score

        # Slow session (should be less efficient)
        session2 = collector.start_session("sess_slow", "search")
        session2.start_time = time.time() - 120  # 2 minutes ago

        feedback2 = collector.end_session(session2, completed=True)
        slow_efficiency = feedback2.efficiency_score

        assert quick_efficiency > slow_efficiency

    def test_workflow_pattern_tracking(self, collector: ImplicitFeedbackCollector) -> None:
        """Test workflow pattern tracking."""
        session = collector.start_session("sess_1", "action")

        collector.record_workflow_pattern(session, started=True)
        collector.record_workflow_pattern(session, started=True)
        collector.record_workflow_pattern(session, completed=True)

        assert session.workflow_patterns_started == 2
        assert session.workflow_patterns_completed == 1

    def test_get_recent_feedback(self, collector: ImplicitFeedbackCollector) -> None:
        """Test getting recent feedback records."""
        # Create and end several sessions
        for i in range(5):
            session = collector.start_session(f"sess_{i}", "analysis")
            collector.end_session(session, completed=i % 2 == 0)

        recent = collector.get_recent_feedback(3)

        assert len(recent) == 3
        assert recent[-1].session_id == "sess_4"

    def test_get_aggregate_stats(self, collector: ImplicitFeedbackCollector) -> None:
        """Test aggregate statistics."""
        # Empty stats
        empty_stats = collector.get_aggregate_stats()
        assert empty_stats["total_sessions"] == 0
        assert empty_stats["completion_rate"] == 0.0

        # After some sessions
        for i in range(4):
            session = collector.start_session(f"sess_{i}", "action")
            collector.record_tool_execution(session, "tool", i < 3, 100.0)
            collector.end_session(session, completed=i < 3)

        stats = collector.get_aggregate_stats()

        assert stats["total_sessions"] == 4
        assert stats["completed_sessions"] == 3
        assert stats["completion_rate"] == 0.75
        assert stats["feedback_count"] == 4

    def test_export_for_rl(self, collector: ImplicitFeedbackCollector) -> None:
        """Test exporting feedback for RL consumption."""
        session = collector.start_session(
            "sess_1", "analysis", provider="anthropic", model="claude-3"
        )
        collector.record_tool_execution(session, "search", True, 100.0)
        collector.record_grounding_result(session, 0.85)
        collector.end_session(session, completed=True)

        export = collector.export_for_rl()

        assert len(export) == 1
        assert export[0]["session_id"] == "sess_1"
        assert export[0]["provider"] == "anthropic"
        assert export[0]["task_type"] == "analysis"
        assert export[0]["task_completed"] is True
        assert "reward" in export[0]

    def test_session_removed_after_end(self, collector: ImplicitFeedbackCollector) -> None:
        """Test that sessions are removed from active sessions after ending."""
        session = collector.start_session("sess_1", "analysis")

        assert collector.get_session("sess_1") is not None

        collector.end_session(session, completed=True)

        assert collector.get_session("sess_1") is None

    def test_no_tools_neutral_success_rate(self, collector: ImplicitFeedbackCollector) -> None:
        """Test that sessions with no tools have neutral success rate."""
        session = collector.start_session("sess_1", "analysis")

        feedback = collector.end_session(session, completed=True)

        assert feedback.tool_success_rate == 0.5  # Neutral
