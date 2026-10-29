"""Tests for evolution audit display and stage prediction integration."""

from unittest.mock import MagicMock


class TestEvolutionAuditReport:
    """Test format_evolution_report for user-visible audit output."""

    def test_function_exists(self):
        from victor.framework.rl.coordinator import RLCoordinator

        assert hasattr(RLCoordinator, "format_evolution_report")

    def test_report_contains_section_name(self):
        from victor.framework.rl.coordinator import RLCoordinator

        report = RLCoordinator.format_evolution_report(
            section="ASI_TOOL_EFFECTIVENESS_GUIDANCE",
            generation=5,
            old_text="old prompt text here for testing",
            new_text="new improved prompt text after evolution",
            alpha=3.0,
            beta_val=1.5,
            sample_count=10,
        )
        assert "ASI_TOOL_EFFECTIVENESS_GUIDANCE" in report
        assert "gen-5" in report or "generation 5" in report.lower()

    def test_report_contains_before_after(self):
        from victor.framework.rl.coordinator import RLCoordinator

        report = RLCoordinator.format_evolution_report(
            section="GROUNDING_RULES",
            generation=2,
            old_text="Always verify file paths",
            new_text="Always verify file paths with ls() before reading",
            alpha=2.0,
            beta_val=1.0,
            sample_count=5,
        )
        assert "Before" in report or "OLD" in report or "before" in report
        assert "After" in report or "NEW" in report or "after" in report

    def test_report_contains_stats(self):
        from victor.framework.rl.coordinator import RLCoordinator

        report = RLCoordinator.format_evolution_report(
            section="COMPLETION_GUIDANCE",
            generation=3,
            old_text="x" * 50,
            new_text="y" * 60,
            alpha=5.0,
            beta_val=2.0,
            sample_count=15,
        )
        # Should show Thompson Sampling stats
        assert "alpha" in report.lower() or "α" in report
        assert "sample" in report.lower() or "15" in report

    def test_report_truncates_long_text(self):
        from victor.framework.rl.coordinator import RLCoordinator

        report = RLCoordinator.format_evolution_report(
            section="SEC",
            generation=1,
            old_text="x" * 2000,
            new_text="y" * 2000,
            alpha=1.0,
            beta_val=1.0,
            sample_count=0,
        )
        # Report should be bounded, not 4000+ chars
        assert len(report) < 2000

    def test_empty_old_text(self):
        from victor.framework.rl.coordinator import RLCoordinator

        report = RLCoordinator.format_evolution_report(
            section="SEC",
            generation=1,
            old_text="",
            new_text="new text",
            alpha=1.0,
            beta_val=1.0,
            sample_count=0,
        )
        assert isinstance(report, str)
        assert len(report) > 0


class TestEvolvReturnsReport:
    """Test that try_evolve_on_session_end returns an audit report."""

    def test_returns_report_on_success(self):
        from victor.framework.rl.coordinator import RLCoordinator

        coord = MagicMock(spec=RLCoordinator)
        coord._evolution_section_idx = 0

        learner = MagicMock()
        learner.EVOLVABLE_SECTIONS = ["SEC_A"]
        rec = MagicMock()
        rec.value = "current text"
        learner.get_recommendation.return_value = rec
        evolved = MagicMock()
        evolved.generation = 3
        evolved.text = "evolved text"
        evolved.alpha = 2.0
        evolved.beta_val = 1.0
        evolved.sample_count = 5
        learner.evolve.return_value = evolved
        coord._learners = {"prompt_optimizer": learner}

        result = RLCoordinator.try_evolve_on_session_end(coord, "ollama", "qwen3")
        # Should return a report dict, not just bool
        assert result is not None
        assert isinstance(result, dict) or result is True


class TestStagePredictionInDecisionFallback:
    """Test that predict_next_stage is used as heuristic for stage detection."""

    def test_prediction_provides_heuristic(self):
        from victor.agent.conversation.state_machine import (
            ConversationStateMachine,
            STAGE_TRANSITION_PROBS,
        )

        sm = MagicMock(spec=ConversationStateMachine)
        sm.state = MagicMock()
        sm.state.stage = MagicMock()
        sm.state.stage.value = "reading"

        stage, conf = ConversationStateMachine.predict_next_stage(sm)
        # Should provide usable heuristic
        assert stage in STAGE_TRANSITION_PROBS.get("reading", {})
        assert 0.0 <= conf <= 1.0
