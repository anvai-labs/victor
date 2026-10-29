"""Tests for session-end GEPA evolution (EvoTest-inspired)."""

from types import SimpleNamespace
from unittest.mock import MagicMock


class TestSessionEndEvolution:
    def test_method_exists(self):
        from victor.framework.rl.coordinator import RLCoordinator

        assert hasattr(RLCoordinator, "try_evolve_on_session_end")

    def test_evolve_called_with_enough_tools(self):
        from victor.framework.rl.coordinator import RLCoordinator

        coord = MagicMock(spec=RLCoordinator)
        coord._evolution_section_idx = 0

        learner = MagicMock()
        learner.EVOLVABLE_SECTIONS = [
            "ASI_TOOL_EFFECTIVENESS_GUIDANCE",
            "GROUNDING_RULES",
        ]
        rec = MagicMock()
        rec.value = "current prompt text"
        learner.get_recommendation.return_value = rec
        evolved = MagicMock()
        evolved.generation = 5
        learner.evolve.return_value = evolved
        coord._learners = {"prompt_optimizer": learner}

        result = RLCoordinator.try_evolve_on_session_end(coord, "ollama", "qwen3")
        assert result  # Truthy (dict or True)
        learner.evolve.assert_called_once()

    def test_no_evolve_without_learner(self):
        from victor.framework.rl.coordinator import RLCoordinator

        coord = MagicMock(spec=RLCoordinator)
        coord._learners = {}
        result = RLCoordinator.try_evolve_on_session_end(coord, "ollama", "qwen3")
        assert result is False

    def test_round_robin_sections(self):
        from victor.framework.rl.coordinator import RLCoordinator

        coord = MagicMock(spec=RLCoordinator)
        coord._evolution_section_idx = 0
        learner = MagicMock()
        learner.EVOLVABLE_SECTIONS = ["SEC_A", "SEC_B", "SEC_C"]
        rec = MagicMock()
        rec.value = "text"
        learner.get_recommendation.return_value = rec
        learner.evolve.return_value = MagicMock(generation=1)
        coord._learners = {"prompt_optimizer": learner}

        # First call: SEC_A
        RLCoordinator.try_evolve_on_session_end(coord, "p", "m")
        call1_section = learner.evolve.call_args[0][0]

        # Update idx for next call
        coord._evolution_section_idx = 1
        learner.evolve.reset_mock()
        learner.get_recommendation.return_value = rec
        RLCoordinator.try_evolve_on_session_end(coord, "p", "m")
        call2_section = learner.evolve.call_args[0][0]

        assert call1_section != call2_section  # Different sections

    def test_exception_doesnt_crash(self):
        from victor.framework.rl.coordinator import RLCoordinator

        coord = MagicMock(spec=RLCoordinator)
        coord._evolution_section_idx = 0
        learner = MagicMock()
        learner.EVOLVABLE_SECTIONS = ["SEC_A"]
        rec = MagicMock()
        rec.value = "text"
        learner.get_recommendation.return_value = rec
        learner.evolve.side_effect = RuntimeError("LLM timeout")
        coord._learners = {"prompt_optimizer": learner}

        # Should not raise
        result = RLCoordinator.try_evolve_on_session_end(coord, "p", "m")
        assert result is False

    def test_registry_backed_section_resolver_is_used_when_available(self):
        from victor.framework.rl.coordinator import RLCoordinator

        coord = MagicMock(spec=RLCoordinator)
        coord._evolution_section_idx = 0

        class _Learner:
            def get_evolvable_sections(self):
                return ["CUSTOM_REVIEW_GUIDANCE"]

            def get_recommendation(self, *_args, **_kwargs):
                return SimpleNamespace(value="review baseline")

            def evolve(self, section, current_text, provider):
                assert section == "CUSTOM_REVIEW_GUIDANCE"
                assert current_text == "review baseline"
                return SimpleNamespace(
                    generation=1,
                    text="review baseline evolved",
                    alpha=1.0,
                    beta_val=1.0,
                    sample_count=1,
                )

        coord._learners = {"prompt_optimizer": _Learner()}

        result = RLCoordinator.try_evolve_on_session_end(coord, "openai", "gpt-5")

        assert result
        assert result["section"] == "CUSTOM_REVIEW_GUIDANCE"
