# Copyright 2025 Vijaykumar Singh
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for model-specific post-compaction continuation bonuses (P1 feature).

TDD approach: Tests written first, then implementation.
"""

from victor.agent.compaction_continuation_bonus import (
    CompactionContinuationBonus,
    CompactionBonusConfig,
)


class TestCompactionContinuationBonus:
    """Test suite for compaction continuation bonus calculations."""

    def test_no_bonus_when_no_compaction(self):
        """No bonus should be added when compaction hasn't occurred."""
        bonus = CompactionContinuationBonus()
        assert bonus.get_bonus("deepseek", "chat", False) == 0
        assert bonus.get_bonus("openai", "gpt-4o", False) == 0

    def test_deepseek_gets_highest_bonus(self):
        """DeepSeek models should get the highest continuation bonus."""
        bonus = CompactionContinuationBonus()

        # DeepSeek provider
        assert bonus.get_bonus("deepseek", "chat", True) >= 3

        # DeepSeek model
        assert bonus.get_bonus("any_provider", "deepseek-chat", True) >= 3

    def test_gpt_models_get_moderate_bonus(self):
        """GPT models should get moderate continuation bonus."""
        bonus = CompactionContinuationBonus()

        assert bonus.get_bonus("openai", "gpt-4o", True) >= 2
        assert bonus.get_bonus("openai", "gpt-4", True) >= 2
        assert bonus.get_bonus("openai", "gpt-3.5-turbo", True) >= 2

    def test_unknown_models_get_default_bonus(self):
        """Unknown models should get the default bonus."""
        bonus = CompactionContinuationBonus()

        assert bonus.get_bonus("unknown", "unknown-model", True) == 1

    def test_bonus_scales_with_messages_removed(self):
        """Bonus should scale based on severity of compaction."""
        bonus = CompactionContinuationBonus()

        # Small compaction
        small_bonus = bonus.get_bonus("deepseek", "chat", True, messages_removed=10)
        large_bonus = bonus.get_bonus("deepseek", "chat", True, messages_removed=100)

        assert large_bonus > small_bonus

    def test_bonus_has_upper_limit(self):
        """Bonus scaling should have an upper limit."""
        bonus = CompactionContinuationBonus()

        # Very large compaction
        max_bonus = bonus.get_bonus("deepseek", "chat", True, messages_removed=1000)

        # Should be capped at reasonable value (base * 2)
        # With scale formula: min(2.0, 1000/50) = min(2.0, 20) = 2.0
        # So bonus = 3 * (1 + 2.0) = 9, but we need to cap the final result
        assert max_bonus <= 9  # 3 base * (1 + 2.0 max multiplier) = 9

    def test_deepseek_coder_gets_lower_bonus(self):
        """DeepSeek Coder gets slightly lower bonus than DeepSeek Chat."""
        bonus = CompactionContinuationBonus()

        chat_bonus = bonus.get_bonus("deepseek", "chat", True)
        # Use "deepseek-coder" to match the key
        coder_bonus = bonus.get_bonus("deepseek", "deepseek-coder", True)

        assert coder_bonus < chat_bonus

    def test_custom_config_overrides_defaults(self):
        """Custom config should override default bonuses."""
        custom_config = CompactionBonusConfig(
            model_bonuses={"custom-model": 5},
            default_bonus=0,
        )
        bonus = CompactionContinuationBonus(config=custom_config)

        assert bonus.get_bonus("custom", "custom-model", True) == 5
        assert bonus.get_bonus("unknown", "unknown", True) == 0


class TestCompactionBonusIntegration:
    """Integration tests for compaction bonus with continuation strategy."""

    def test_continuation_budget_with_compaction_bonus(self):
        """Test that compaction bonus correctly increases total continuation budget."""
        bonus_calculator = CompactionContinuationBonus()

        # Simulate scenario
        provider = "deepseek"
        model = "chat"
        base_budget = 6  # Default analysis budget
        compaction_occurred = True
        messages_removed = 64

        bonus = bonus_calculator.get_bonus(provider, model, compaction_occurred, messages_removed)
        total_budget = base_budget + bonus

        # DeepSeek with 64 messages removed should get at least +3 bonus
        assert total_budget >= base_budget + 3

    def test_continuation_budget_without_compaction(self):
        """Test that no bonus is added when compaction hasn't occurred."""
        bonus_calculator = CompactionContinuationBonus()

        base_budget = 6
        bonus = bonus_calculator.get_bonus("deepseek", "chat", False)
        total_budget = base_budget + bonus

        assert total_budget == base_budget
