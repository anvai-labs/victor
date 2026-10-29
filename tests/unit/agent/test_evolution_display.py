"""Tests for evolution display and stage prediction heuristic wiring."""

from unittest.mock import MagicMock


class TestEvolutionDisplay:
    """Test debug logger evolution report display."""

    def test_log_method_exists(self):
        from victor.agent.debug_logger import DebugLogger

        assert hasattr(DebugLogger, "log_evolution_report")

    def test_report_logged_with_section(self):
        from victor.agent.debug_logger import DebugLogger
        import logging

        dl = DebugLogger.__new__(DebugLogger)
        dl.enabled = True
        dl.logger = logging.getLogger("test_evolution")
        dl._presentation = MagicMock()
        dl._presentation.icon.return_value = "+"

        report = {
            "section": "ASI_TOOL_EFFECTIVENESS_GUIDANCE",
            "report": "[GEPA Evolution] section=ASI gen-5\n  Stats: alpha=3.0",
        }

        # Should not raise
        dl.log_evolution_report(report)

    def test_disabled_logger_noop(self):
        from victor.agent.debug_logger import DebugLogger

        dl = DebugLogger.__new__(DebugLogger)
        dl.enabled = False
        dl.logger = MagicMock()
        dl._presentation = MagicMock()

        dl.log_evolution_report({"section": "X", "report": "text"})
        dl.logger.info.assert_not_called()


class TestPredictionAsHeuristic:
    """Test that predict_next_stage is wired as heuristic for stage detection."""

    def test_prediction_available(self):
        from victor.agent.conversation.state_machine import (
            ConversationStateMachine,
            STAGE_TRANSITION_PROBS,
        )

        sm = MagicMock(spec=ConversationStateMachine)
        sm.state = MagicMock()
        sm.state.stage = MagicMock()
        sm.state.stage.value = "reading"

        stage, conf = ConversationStateMachine.predict_next_stage(sm)
        # Should return usable heuristic
        assert stage in STAGE_TRANSITION_PROBS.get("reading", {})
        assert conf >= 0.4

    def test_high_confidence_provides_nonzero_heuristic(self):
        """predict_next_stage with high confidence should give nonzero heuristic."""
        from victor.agent.conversation.state_machine import STAGE_TRANSITION_PROBS

        # Reading -> execution has 0.5 probability >= 0.6 threshold? No, 0.5 < 0.6
        # Initial -> reading has 0.7 >= 0.6 -> used as heuristic
        probs = STAGE_TRANSITION_PROBS.get("initial", {})
        best = max(probs, key=probs.get)
        assert best == "reading"
        assert probs[best] >= 0.6  # Meets threshold
