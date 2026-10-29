"""TDD tests for ScoringWeights sum validator — Wave C.

Verifies: Pydantic migration, sum validator enforces invariant, all builtin
presets pass, frozen property preserved.
"""

from __future__ import annotations

import pytest


class TestScoringWeightsValidator:
    def test_all_builtin_presets_sum_to_one(self):
        from victor.agent.conversation.scoring import (
            CONTROLLER_WEIGHTS,
            DEFAULT_WEIGHTS,
            EXECUTION_WEIGHTS,
            EXPLORATION_WEIGHTS,
            PLANNING_WEIGHTS,
            REVIEW_WEIGHTS,
            STORE_WEIGHTS,
        )

        for name, preset in {
            "STORE_WEIGHTS": STORE_WEIGHTS,
            "CONTROLLER_WEIGHTS": CONTROLLER_WEIGHTS,
            "DEFAULT_WEIGHTS": DEFAULT_WEIGHTS,
            "EXPLORATION_WEIGHTS": EXPLORATION_WEIGHTS,
            "PLANNING_WEIGHTS": PLANNING_WEIGHTS,
            "EXECUTION_WEIGHTS": EXECUTION_WEIGHTS,
            "REVIEW_WEIGHTS": REVIEW_WEIGHTS,
        }.items():
            total = preset.priority + preset.recency + preset.role + preset.length + preset.semantic
            assert (
                abs(total - 1.0) <= 0.01
            ), f"{name} weights sum to {total:.4f}, expected 1.0 ± 0.01"

    def test_valid_custom_weights_accepted(self):
        from victor.agent.conversation.scoring import ScoringWeights

        w = ScoringWeights(priority=0.3, recency=0.3, role=0.2, length=0.1, semantic=0.1)
        assert abs((w.priority + w.recency + w.role + w.length + w.semantic) - 1.0) <= 0.01

    def test_weights_summing_to_wrong_value_raise_error(self):
        from victor.agent.conversation.scoring import ScoringWeights

        with pytest.raises(Exception):  # ValueError or ValidationError
            ScoringWeights(priority=0.5, recency=0.5, role=0.5, length=0.5, semantic=0.5)

    def test_near_one_within_tolerance_accepted(self):
        from victor.agent.conversation.scoring import ScoringWeights

        # 0.999 is within 0.01 of 1.0 — should be accepted
        w = ScoringWeights(priority=0.2, recency=0.399, role=0.2, length=0.1, semantic=0.1)
        total = w.priority + w.recency + w.role + w.length + w.semantic
        assert abs(total - 1.0) <= 0.01

    def test_scoring_weights_still_frozen_after_migration(self):
        from victor.agent.conversation.scoring import ScoringWeights

        w = ScoringWeights()
        with pytest.raises(Exception):  # TypeError or ValidationError from frozen model
            w.priority = 0.99  # type: ignore[misc]

    def test_scoring_weights_importable_as_before(self):
        from victor.agent.conversation.scoring import ScoringWeights

        w = ScoringWeights()
        assert w.priority == pytest.approx(0.2)
        assert w.recency == pytest.approx(0.4)
        assert w.role == pytest.approx(0.2)
        assert w.length == pytest.approx(0.1)
        assert w.semantic == pytest.approx(0.1)

    def test_zero_sum_raises_error(self):
        from victor.agent.conversation.scoring import ScoringWeights

        with pytest.raises(Exception):
            ScoringWeights(priority=0.0, recency=0.0, role=0.0, length=0.0, semantic=0.0)
