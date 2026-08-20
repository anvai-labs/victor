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

"""Unit tests for FeedbackIntegration.

Tests the integration layer for implicit feedback collection.
"""

import pytest
from unittest.mock import MagicMock, patch

from victor.framework.rl.feedback_integration import (
    FeedbackIntegration,
    get_feedback_integration,
)
from victor.framework.rl.implicit_feedback import SessionContext


@pytest.fixture
def integration():
    """Create FeedbackIntegration instance."""
    # Reset singleton for fresh instance
    FeedbackIntegration._instance = None
    inst = FeedbackIntegration()
    yield inst
    # Clean up
    FeedbackIntegration._instance = None


class TestFeedbackIntegration:
    """Tests for FeedbackIntegration."""

    def test_initialization(self, integration: FeedbackIntegration) -> None:
        """Test integration initialization."""
        assert integration._enabled is True
        assert integration._active_sessions == {}
        assert integration._rl_coordinator is None

    def test_singleton_pattern(self) -> None:
        """Test singleton pattern."""
        FeedbackIntegration._instance = None

        inst1 = FeedbackIntegration.get_instance()
        inst2 = FeedbackIntegration.get_instance()

        assert inst1 is inst2

        FeedbackIntegration._instance = None

    def test_set_enabled(self, integration: FeedbackIntegration) -> None:
        """Test enabling and disabling."""
        integration.set_enabled(False)
        assert integration._enabled is False

        integration.set_enabled(True)
        assert integration._enabled is True

    def test_start_tracking_creates_session(self, integration: FeedbackIntegration) -> None:
        """Test starting tracking creates session."""
        session = integration.start_tracking(
            session_id="test-session-1",
            task_type="analysis",
            provider="anthropic",
            model="claude-3",
        )

        assert session is not None
        assert session.session_id == "test-session-1"
        assert "test-session-1" in integration._active_sessions

    def test_start_tracking_disabled(self, integration: FeedbackIntegration) -> None:
        """Test starting tracking when disabled returns None."""
        integration.set_enabled(False)

        session = integration.start_tracking(
            session_id="test-session-1",
            task_type="analysis",
        )

        assert session is None
        assert "test-session-1" not in integration._active_sessions

    def test_get_session(self, integration: FeedbackIntegration) -> None:
        """Test getting active session."""
        integration.start_tracking(
            session_id="test-session-1",
            task_type="analysis",
        )

        session = integration.get_session("test-session-1")
        assert session is not None
        assert session.session_id == "test-session-1"

    def test_get_session_not_found(self, integration: FeedbackIntegration) -> None:
        """Test getting non-existent session."""
        session = integration.get_session("nonexistent")
        assert session is None

    def test_record_tool(self, integration: FeedbackIntegration) -> None:
        """Test recording tool execution."""
        integration.start_tracking(
            session_id="test-session-1",
            task_type="analysis",
        )

        # Should not raise
        integration.record_tool(
            session_id="test-session-1",
            tool_name="code_search",
            success=True,
            execution_time_ms=150.0,
        )

        session = integration.get_session("test-session-1")
        assert len(session.tool_executions) > 0

    def test_record_tool_disabled(self, integration: FeedbackIntegration) -> None:
        """Test recording tool when disabled."""
        integration.start_tracking(
            session_id="test-session-1",
            task_type="analysis",
        )
        integration.set_enabled(False)

        # Should not record
        initial_count = len(integration.get_session("test-session-1").tool_executions)
        integration.record_tool(
            session_id="test-session-1",
            tool_name="code_search",
            success=True,
            execution_time_ms=150.0,
        )

        # Re-enable to check
        integration.set_enabled(True)
        session = integration.get_session("test-session-1")
        assert len(session.tool_executions) == initial_count

    def test_record_tool_nonexistent_session(self, integration: FeedbackIntegration) -> None:
        """Test recording tool for non-existent session."""
        # Should not raise
        integration.record_tool(
            session_id="nonexistent",
            tool_name="code_search",
            success=True,
            execution_time_ms=150.0,
        )

    def test_record_grounding(self, integration: FeedbackIntegration) -> None:
        """Test recording grounding result."""
        integration.start_tracking(
            session_id="test-session-1",
            task_type="analysis",
        )

        integration.record_grounding("test-session-1", confidence=0.85)

        session = integration.get_session("test-session-1")
        assert len(session.grounding_results) > 0

    def test_record_iteration(self, integration: FeedbackIntegration) -> None:
        """Test recording iteration."""
        integration.start_tracking(
            session_id="test-session-1",
            task_type="analysis",
        )

        integration.record_iteration("test-session-1")

        session = integration.get_session("test-session-1")
        assert session.iteration_count == 1

    def test_record_workflow(self, integration: FeedbackIntegration) -> None:
        """Test recording workflow pattern."""
        integration.start_tracking(
            session_id="test-session-1",
            task_type="analysis",
        )

        integration.record_workflow("test-session-1", started=True)
        integration.record_workflow("test-session-1", completed=True)

        session = integration.get_session("test-session-1")
        assert session.workflow_patterns_started >= 1

    def test_end_tracking_returns_feedback(self, integration: FeedbackIntegration) -> None:
        """Test ending tracking returns feedback."""
        integration.start_tracking(
            session_id="test-session-1",
            task_type="analysis",
            provider="anthropic",
            model="claude-3",
        )

        # Record some activity
        integration.record_tool(
            session_id="test-session-1",
            tool_name="code_search",
            success=True,
            execution_time_ms=150.0,
        )

        feedback = integration.end_tracking("test-session-1", completed=True)

        assert feedback is not None
        assert feedback.task_completed is True
        assert "test-session-1" not in integration._active_sessions

    def test_end_tracking_disabled(self, integration: FeedbackIntegration) -> None:
        """Test ending tracking when disabled."""
        integration.start_tracking(
            session_id="test-session-1",
            task_type="analysis",
        )
        integration.set_enabled(False)

        feedback = integration.end_tracking("test-session-1", completed=True)

        assert feedback is None

    def test_end_tracking_nonexistent_session(self, integration: FeedbackIntegration) -> None:
        """Test ending non-existent session."""
        feedback = integration.end_tracking("nonexistent", completed=True)
        assert feedback is None

    def test_get_stats(self, integration: FeedbackIntegration) -> None:
        """Test getting statistics."""
        integration.start_tracking(
            session_id="test-session-1",
            task_type="analysis",
        )

        stats = integration.get_stats()

        assert "active_sessions" in stats
        assert stats["active_sessions"] == 1
        assert "enabled" in stats
        assert stats["enabled"] is True


class TestFeedbackIntegrationWithCoordinator:
    """Tests for FeedbackIntegration with RL coordinator."""

    def test_distribute_feedback(self, integration: FeedbackIntegration) -> None:
        """Test feedback distribution to coordinator."""
        mock_coordinator = MagicMock()
        integration._rl_coordinator = mock_coordinator

        # Start and record some activity
        integration.start_tracking(
            session_id="test-session-1",
            task_type="analysis",
            provider="anthropic",
            model="claude-3",
        )
        integration.record_tool(
            session_id="test-session-1",
            tool_name="code_search",
            success=True,
            execution_time_ms=150.0,
        )

        # End tracking
        integration.end_tracking("test-session-1", completed=True)

        # Coordinator should have received outcome
        assert mock_coordinator.record_outcome.called

    def test_get_quality_weights_no_coordinator(self, integration: FeedbackIntegration) -> None:
        """Test getting quality weights without coordinator."""
        weights = integration.get_quality_weights("analysis")
        assert weights == {}

    def test_get_quality_weights_with_coordinator(self, integration: FeedbackIntegration) -> None:
        """Test getting quality weights with coordinator."""
        mock_coordinator = MagicMock()
        mock_learner = MagicMock()
        mock_rec = MagicMock()
        mock_rec.is_baseline = False
        mock_rec.value = {"relevance": 1.5, "accuracy": 1.3}
        mock_learner.get_recommendation.return_value = mock_rec
        mock_coordinator.get_learner.return_value = mock_learner
        integration._rl_coordinator = mock_coordinator

        weights = integration.get_quality_weights("analysis")

        assert weights == {"relevance": 1.5, "accuracy": 1.3}


class TestSessionTracking:
    """Tests for session tracking lifecycle."""

    def test_full_session_lifecycle(self, integration: FeedbackIntegration) -> None:
        """Test complete session lifecycle."""
        # Start
        session = integration.start_tracking(
            session_id="lifecycle-test",
            task_type="analysis",
            provider="anthropic",
            model="claude-3",
            max_iterations=30,
        )
        assert session is not None

        # Record activity
        integration.record_iteration("lifecycle-test")
        integration.record_tool("lifecycle-test", "code_search", True, 100.0)
        integration.record_tool("lifecycle-test", "read_file", True, 50.0)
        integration.record_grounding("lifecycle-test", 0.9)
        integration.record_workflow("lifecycle-test", started=True)
        integration.record_workflow("lifecycle-test", completed=True)

        # Check session state
        session = integration.get_session("lifecycle-test")
        assert session.iteration_count == 1
        assert len(session.tool_executions) == 2
        assert len(session.grounding_results) == 1

        # End
        feedback = integration.end_tracking("lifecycle-test", completed=True)

        assert feedback is not None
        assert feedback.task_completed is True
        assert feedback.tool_count == 2

    def test_multiple_concurrent_sessions(self, integration: FeedbackIntegration) -> None:
        """Test multiple concurrent sessions."""
        # Start multiple sessions
        session1 = integration.start_tracking(
            session_id="session-1",
            task_type="analysis",
        )
        session2 = integration.start_tracking(
            session_id="session-2",
            task_type="code_generation",
        )
        session3 = integration.start_tracking(
            session_id="session-3",
            task_type="debugging",
        )

        # Record to different sessions
        integration.record_tool("session-1", "code_search", True, 100.0)
        integration.record_tool("session-2", "write_file", True, 200.0)
        integration.record_tool("session-3", "read_file", True, 50.0)

        # Check isolation
        s1 = integration.get_session("session-1")
        s2 = integration.get_session("session-2")
        s3 = integration.get_session("session-3")

        assert s1.task_type == "analysis"
        assert s2.task_type == "code_generation"
        assert s3.task_type == "debugging"

        # End sessions
        f1 = integration.end_tracking("session-1", completed=True)
        f2 = integration.end_tracking("session-2", completed=False)
        f3 = integration.end_tracking("session-3", completed=True)

        assert f1.task_completed is True
        assert f2.task_completed is False
        assert f3.task_completed is True


class TestGetFeedbackIntegration:
    """Tests for global function."""

    def test_get_feedback_integration(self) -> None:
        """Test getting global instance."""
        FeedbackIntegration._instance = None

        inst = get_feedback_integration()

        assert inst is not None
        assert isinstance(inst, FeedbackIntegration)

        FeedbackIntegration._instance = None

    def test_get_feedback_integration_returns_same_instance(self) -> None:
        """Test global function returns same instance."""
        FeedbackIntegration._instance = None

        inst1 = get_feedback_integration()
        inst2 = get_feedback_integration()

        assert inst1 is inst2

        FeedbackIntegration._instance = None
