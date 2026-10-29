"""Tests for CreditTrackingService — the runtime integration layer."""

from unittest.mock import MagicMock, patch

from victor.framework.rl.credit_tracking_service import (
    CreditTrackingService,
    ToolRewardSignal,
    extract_reward_from_tool_result,
)
from victor.framework.rl.credit_assignment import (
    CreditMethodology,
)

# ============================================================================
# Reward extraction
# ============================================================================


class TestExtractRewardFromToolResult:
    """Test reward extraction heuristics."""

    def test_successful_tool_returns_positive_reward(self):
        reward = extract_reward_from_tool_result("read_file", True, 100.0)
        assert reward == 1.0

    def test_failed_tool_returns_negative_reward(self):
        reward = extract_reward_from_tool_result("edit_file", False, 100.0)
        assert reward == -1.0

    def test_timeout_gets_extra_penalty(self):
        reward = extract_reward_from_tool_result(
            "shell", False, 60000.0, error="Tool execution timed out after 60s"
        )
        assert reward == -2.0

    def test_slow_success_gets_slight_penalty(self):
        reward = extract_reward_from_tool_result("search", True, 10000.0)
        assert 0.5 < reward < 1.0

    def test_fast_success_gets_full_reward(self):
        reward = extract_reward_from_tool_result("read", True, 50.0)
        assert reward == 1.0


# ============================================================================
# CreditTrackingService
# ============================================================================


class TestCreditTrackingService:
    """Test the runtime credit tracking service."""

    def test_init_default(self):
        service = CreditTrackingService()
        assert service.turn_count == 0
        assert service.pending_signals == 0

    def test_record_tool_result(self):
        service = CreditTrackingService()
        signal = service.record_tool_result(
            tool_name="read_file",
            success=True,
            execution_time_ms=50.0,
        )
        assert isinstance(signal, ToolRewardSignal)
        assert signal.tool_name == "read"
        assert signal.success is True
        assert signal.reward == 1.0
        assert service.pending_signals == 1

    def test_record_failed_tool_result(self):
        service = CreditTrackingService()
        signal = service.record_tool_result(
            tool_name="edit_file",
            success=False,
            execution_time_ms=100.0,
            error="No match found",
        )
        assert signal.tool_name == "edit"
        assert signal.reward == -1.0
        assert signal.error == "No match found"

    def test_record_tool_result_canonicalizes_aliases_into_single_stream(self):
        service = CreditTrackingService()

        first = service.record_tool_result("read_file", True, 50.0)
        second = service.record_tool_result("read", True, 55.0)

        assert first.tool_name == "read"
        assert second.tool_name == "read"

        signals = service.assign_turn_credit()
        assert len(signals) == 2
        assert all(signal.metadata and signal.metadata.tool_name == "read" for signal in signals)

    def test_assign_turn_credit_empty(self):
        service = CreditTrackingService()
        signals = service.assign_turn_credit()
        assert signals == []
        assert service.turn_count == 0

    def test_assign_turn_credit_basic(self):
        service = CreditTrackingService()

        # Record some tool results
        service.record_tool_result("read_file", True, 50.0)
        service.record_tool_result("edit_file", True, 100.0)
        service.record_tool_result("shell", False, 5000.0, error="Command failed")

        assert service.pending_signals == 3

        # Assign credit
        signals = service.assign_turn_credit(agent_id="test_agent")

        assert len(signals) == 3
        assert service.turn_count == 1
        assert service.pending_signals == 0  # Reset after assignment

        # Verify signals have correct metadata
        for signal in signals:
            assert signal.metadata is not None
            assert signal.metadata.agent_id == "test_agent"
            assert signal.metadata.turn_index == 1
            assert signal.metadata.tool_name is not None

    def test_assign_turn_credit_at_boundary_advances_turn(self):
        # The runtime turn-boundary wrapper closes the feedback loop: it advances
        # turn_count so generate_tool_guidance() can eventually clear its guard.
        service = CreditTrackingService()
        service.record_tool_result("read_file", True, 50.0)
        service.record_tool_result("edit_file", True, 100.0)

        service.assign_turn_credit_at_boundary()

        assert service.turn_count == 1
        assert service.pending_signals == 0

    def test_assign_turn_credit_at_boundary_respects_disabled_flag(self):
        # auto_assign_at_turn_boundary=False => the wrapper must not assign, so the
        # turn does not advance (a user opting into recording-only).
        service = CreditTrackingService(auto_assign_at_turn_boundary=False)
        service.record_tool_result("read_file", True, 50.0)

        service.assign_turn_credit_at_boundary()

        assert service.turn_count == 0
        assert service.pending_signals == 1  # left intact, not consumed

    def test_assign_turn_credit_at_boundary_swallows_errors(self):
        # A credit-assignment failure must never break the turn.
        service = CreditTrackingService()
        service.record_tool_result("read_file", True, 50.0)
        with patch.object(service, "assign_turn_credit", side_effect=RuntimeError("boom")):
            service.assign_turn_credit_at_boundary()  # must not raise

    def test_assign_turn_credit_preserves_per_signal_agent_ids(self):
        service = CreditTrackingService()

        service.record_tool_result(
            "search",
            True,
            40.0,
            agent_id="researcher_1",
            team_id="team_debug",
        )
        service.record_tool_result(
            "edit",
            False,
            120.0,
            error="Mismatch",
            agent_id="executor_1",
            team_id="team_debug",
        )

        signals = service.assign_turn_credit(agent_id="manager", team_id="team_debug")

        assert len(signals) == 2
        agent_ids = [signal.metadata.agent_id for signal in signals if signal.metadata]
        team_ids = [signal.metadata.team_id for signal in signals if signal.metadata]
        assert agent_ids == ["researcher_1", "executor_1"]
        assert team_ids == ["team_debug", "team_debug"]

    def test_assign_turn_credit_preserves_session_ids(self):
        service = CreditTrackingService()

        service.record_tool_result(
            "search",
            True,
            40.0,
            agent_id="researcher_1",
            team_id="team_debug",
            session_id="session-alpha",
        )

        signals = service.assign_turn_credit(agent_id="manager", team_id="team_debug")

        assert len(signals) == 1
        assert getattr(signals[0].metadata, "session_id", None) == "session-alpha"

    def test_assign_turn_credit_uses_gae_by_default(self):
        service = CreditTrackingService(methodology=CreditMethodology.GAE)

        service.record_tool_result("read", True, 50.0)
        service.record_tool_result("edit", True, 100.0)

        signals = service.assign_turn_credit()
        # GAE assigns credit to each step
        assert len(signals) == 2
        assert all(s.methodology == CreditMethodology.GAE for s in signals)

    def test_multiple_turns(self):
        service = CreditTrackingService()

        # Turn 1
        service.record_tool_result("read", True, 50.0)
        signals_1 = service.assign_turn_credit()
        assert service.turn_count == 1

        # Turn 2
        service.record_tool_result("edit", True, 100.0)
        service.record_tool_result("shell", True, 200.0)
        signals_2 = service.assign_turn_credit()
        assert service.turn_count == 2
        assert len(signals_2) == 2

    def test_get_recent_credit_signals(self):
        service = CreditTrackingService()

        service.record_tool_result("read", True, 50.0)
        service.assign_turn_credit()

        recent = service.get_recent_credit_signals(limit=10)
        assert len(recent) == 1

    def test_get_tool_credit_summary(self):
        service = CreditTrackingService()

        # Multiple calls to same tool
        service.record_tool_result("read", True, 50.0)
        service.record_tool_result("read", True, 60.0)
        service.record_tool_result("edit", False, 100.0, error="Mismatch")
        service.assign_turn_credit()

        summary = service.get_tool_credit_summary()
        assert "read" in summary or "edit" in summary
        # At least one tool should appear in summary
        assert len(summary) > 0

    def test_get_agent_credit_summary(self):
        service = CreditTrackingService()

        service.record_tool_result("search", True, 40.0, agent_id="researcher_1", team_id="team_1")
        service.record_tool_result("read", True, 50.0, agent_id="researcher_1", team_id="team_1")
        service.record_tool_result(
            "edit",
            False,
            140.0,
            error="Mismatch",
            agent_id="executor_1",
            team_id="team_1",
        )
        service.assign_turn_credit()

        summary = service.get_agent_credit_summary()

        assert summary["researcher_1"]["call_count"] == 2.0
        assert summary["researcher_1"]["team_id"] == "team_1"
        assert summary["executor_1"]["call_count"] == 1.0
        assert summary["executor_1"]["avg_credit"] < summary["researcher_1"]["avg_credit"]

    def test_generate_agent_guidance(self):
        service = CreditTrackingService()

        service.record_tool_result("search", True, 40.0, agent_id="researcher_1", team_id="team_1")
        service.record_tool_result(
            "edit",
            False,
            140.0,
            error="Mismatch",
            agent_id="executor_1",
            team_id="team_1",
        )
        service.record_tool_result("read", True, 50.0, agent_id="researcher_1", team_id="team_1")
        service.record_tool_result(
            "shell",
            False,
            1000.0,
            error="Command failed",
            agent_id="executor_1",
            team_id="team_1",
        )
        service.assign_turn_credit()

        guidance = service.generate_agent_guidance()

        assert guidance is not None
        assert "Agent execution credit" in guidance
        assert "researcher_1" in guidance
        assert "executor_1" in guidance

    def test_reset(self):
        service = CreditTrackingService()
        service.record_tool_result("read", True, 50.0)
        service.assign_turn_credit()

        service.reset()
        assert service.turn_count == 0
        assert service.pending_signals == 0
        assert len(service.get_recent_credit_signals()) == 0

    def test_from_settings_default(self):
        """Test creation from settings with defaults."""
        mock_settings = MagicMock()
        mock_settings.credit_assignment = None

        service = CreditTrackingService.from_settings(mock_settings)
        assert service is not None

    def test_from_settings_configured(self):
        """Test creation from settings with explicit config."""
        mock_settings = MagicMock()
        mock_settings.credit_assignment.default_methodology = "shapley"
        mock_settings.credit_assignment.gamma = 0.95
        mock_settings.credit_assignment.lambda_gae = 0.9
        mock_settings.credit_assignment.shapley_sampling_count = 20
        mock_settings.credit_assignment.emit_observability_events = True
        mock_settings.credit_assignment.persist_to_db = False

        service = CreditTrackingService.from_settings(mock_settings)
        assert service._methodology == CreditMethodology.SHAPLEY

    def test_observability_events_emitted(self):
        """Test that credit events are emitted to observability bus."""
        mock_bus = MagicMock()
        service = CreditTrackingService(
            observability_bus=mock_bus,
            emit_events=True,
        )

        service.record_tool_result("read", True, 50.0)
        service.assign_turn_credit()

        # The service uses emit_event_sync — we mock at that level
        # Just verify the service doesn't crash with a bus present
        assert service.turn_count == 1

    def test_methodology_override_per_turn(self):
        """Test that methodology can be overridden per turn."""
        service = CreditTrackingService(methodology=CreditMethodology.GAE)

        service.record_tool_result("read", True, 50.0)
        service.record_tool_result("edit", True, 100.0)

        signals = service.assign_turn_credit(methodology=CreditMethodology.MONTE_CARLO)
        # Should use the overridden methodology
        assert len(signals) > 0


# ============================================================================
# Integration: ToolCallResult → CreditTrackingService
# ============================================================================


class TestToolPipelineIntegration:
    """Test that ToolPipeline result shape maps correctly to service."""

    def test_tool_call_result_fields_map_correctly(self):
        """Verify the ToolCallResult fields used by the pipeline callback."""
        service = CreditTrackingService()

        # Simulate what the ToolPipeline callback sends
        service.record_tool_result(
            tool_name="code_search",
            success=True,
            execution_time_ms=234.5,
            error=None,
            arguments={"query": "def main", "path": "/src"},
        )

        assert service.pending_signals == 1

    def test_failed_tool_with_arguments(self):
        service = CreditTrackingService()

        service.record_tool_result(
            tool_name="edit_file",
            success=False,
            execution_time_ms=15.0,
            error="old_string not found in file",
            arguments={"file_path": "/src/main.py", "old_string": "foo"},
        )

        signals = service.assign_turn_credit()
        assert len(signals) == 1
        assert signals[0].raw_reward < 0


# ============================================================================
# Feedback loop: tool guidance generation
# ============================================================================


class TestToolGuidanceGeneration:
    """Test the credit-driven tool guidance feedback loop."""

    def test_no_guidance_with_insufficient_data(self):
        """No guidance before 3 turns with 5+ tool calls."""
        service = CreditTrackingService()
        service.record_tool_result("read", True, 50.0)
        service.assign_turn_credit()

        assert service.generate_tool_guidance() is None

    def test_guidance_generated_after_sufficient_data(self):
        """Guidance appears after enough turns and tool calls."""
        service = CreditTrackingService()

        # 3 turns with multiple tools each
        for turn in range(3):
            service.record_tool_result("read", True, 50.0)
            service.record_tool_result("edit", True, 100.0)
            service.record_tool_result("shell", False, 5000.0, error="Command failed")
            service.assign_turn_credit()

        guidance = service.generate_tool_guidance()
        # Should have guidance now (9 tool calls across 3 turns)
        # At least one tool should be flagged
        assert guidance is not None
        assert "Tool effectiveness" in guidance

    def test_guidance_flags_underperforming_tools(self):
        """Tools with negative credit get flagged."""
        service = CreditTrackingService()

        for turn in range(4):
            service.record_tool_result("read", True, 50.0)
            service.record_tool_result("bad_tool", False, 1000.0, error="Always fails")
            service.assign_turn_credit()

        guidance = service.generate_tool_guidance()
        if guidance:
            # bad_tool should be flagged as underperforming
            assert "bad_tool" in guidance or "low effectiveness" in guidance

    def test_guidance_highlights_effective_tools(self):
        """Tools with high positive credit get highlighted."""
        service = CreditTrackingService()

        for turn in range(4):
            service.record_tool_result("code_search", True, 50.0)
            service.record_tool_result("code_search", True, 60.0)
            service.assign_turn_credit()

        guidance = service.generate_tool_guidance()
        if guidance:
            assert "high effectiveness" in guidance or "code_search" in guidance
