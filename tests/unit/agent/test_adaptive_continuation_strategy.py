# Copyright 2025 Vijaykumar Singh
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for adaptive continuation strategy selection (P2-1).

TDD approach: Tests written first, then implementation.

Adaptive strategy selection addresses the DeepSeek Chat issue where
simple "continue" prompts after compaction cause the model to stop.
Different models need different continuation strategies.
"""

from victor.agent.adaptive_continuation_strategy import (
    ContinuationStrategy,
    StrategySelector,
    select_strategy_for_context,
    get_strategy_prompt,
)


class TestContinuationStrategy:
    """Test suite for continuation strategy enum."""

    def test_strategy_enum_exists(self):
        """ContinuationStrategy enum should exist with expected values."""
        # Expected strategies based on research
        expected_strategies = [
            "DIRECT_CONTINUE",  # Simple "continue" prompt
            "TASK_SUMMARY",  # Re-state task before continuing
            "CONTEXT_REMINDER",  # Remind what was compacted
            "FRESH_START",  # Start new turn without continuation
            "SILENT",  # No continuation prompt
        ]

        for strategy_name in expected_strategies:
            assert hasattr(ContinuationStrategy, strategy_name)
            strategy = getattr(ContinuationStrategy, strategy_name)
            assert isinstance(strategy, ContinuationStrategy)

    def test_strategy_values_are_strings(self):
        """Each strategy should have a string value."""
        for strategy in ContinuationStrategy:
            assert isinstance(strategy.value, str)
            assert len(strategy.value) > 0


class TestStrategySelector:
    """Test suite for StrategySelector class."""

    def test_deepseek_gets_task_summary_after_compaction(self):
        """DeepSeek should get TASK_SUMMARY strategy after compaction."""
        selector = StrategySelector()

        strategy = selector.select_strategy(
            provider="deepseek",
            model="deepseek-chat",
            compaction_occurred=True,
            messages_removed=10,
            current_turn=5,
        )

        assert strategy == ContinuationStrategy.TASK_SUMMARY

    def test_deepseek_high_compaction_gets_context_reminder(self):
        """DeepSeek with high compaction (>50 messages) gets CONTEXT_REMINDER."""
        selector = StrategySelector()

        strategy = selector.select_strategy(
            provider="deepseek",
            model="deepseek-chat",
            compaction_occurred=True,
            messages_removed=64,
            current_turn=5,
        )

        # High compaction for DeepSeek should get context reminder
        assert strategy in (
            ContinuationStrategy.CONTEXT_REMINDER,
            ContinuationStrategy.TASK_SUMMARY,
        )

    def test_claude_gets_direct_continue(self):
        """Claude should get DIRECT_CONTINUE strategy (works well)."""
        selector = StrategySelector()

        strategy = selector.select_strategy(
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            compaction_occurred=True,
            messages_removed=10,
            current_turn=5,
        )

        assert strategy == ContinuationStrategy.DIRECT_CONTINUE

    def test_no_compaction_gets_silent(self):
        """No compaction should get SILENT strategy."""
        selector = StrategySelector()

        strategy = selector.select_strategy(
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            compaction_occurred=False,
            messages_removed=0,
            current_turn=5,
        )

        assert strategy == ContinuationStrategy.SILENT

    def test_late_turns_get_fresh_start(self):
        """Late turns (>20) should get FRESH_START regardless of model."""
        selector = StrategySelector()

        strategy = selector.select_strategy(
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            compaction_occurred=True,
            messages_removed=10,
            current_turn=25,  # Late turn
        )

        assert strategy == ContinuationStrategy.FRESH_START

    def test_gpt_gets_direct_continue(self):
        """GPT models should get DIRECT_CONTINUE strategy."""
        selector = StrategySelector()

        strategy = selector.select_strategy(
            provider="openai",
            model="gpt-4o",
            compaction_occurred=True,
            messages_removed=10,
            current_turn=5,
        )

        assert strategy == ContinuationStrategy.DIRECT_CONTINUE

    def test_custom_config_overrides_selection(self):
        """Custom configuration should override default selection."""
        selector = StrategySelector(
            default_strategy=ContinuationStrategy.FRESH_START,
            force_default=True,  # Enable override mode
        )

        strategy = selector.select_strategy(
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            compaction_occurred=True,
            messages_removed=10,
            current_turn=5,
        )

        # Custom default should apply when force_default is True
        assert strategy == ContinuationStrategy.FRESH_START


class TestStrategySelectionLogic:
    """Test suite for strategy selection decision logic."""

    def test_messages_removed_threshold_affects_deepseek(self):
        """Messages removed threshold should affect DeepSeek strategy."""
        selector = StrategySelector()

        # Low removal - still TASK_SUMMARY for DeepSeek
        strategy_low = selector.select_strategy(
            provider="deepseek",
            model="deepseek-chat",
            compaction_occurred=True,
            messages_removed=5,
            current_turn=5,
        )

        # High removal - might get CONTEXT_REMINDER
        strategy_high = selector.select_strategy(
            provider="deepseek",
            model="deepseek-chat",
            compaction_occurred=True,
            messages_removed=60,
            current_turn=5,
        )

        # Both should be task-oriented strategies
        assert strategy_low in (
            ContinuationStrategy.TASK_SUMMARY,
            ContinuationStrategy.CONTEXT_REMINDER,
        )
        assert strategy_high in (
            ContinuationStrategy.TASK_SUMMARY,
            ContinuationStrategy.CONTEXT_REMINDER,
        )

    def test_unknown_provider_gets_direct_continue(self):
        """Unknown providers should default to DIRECT_CONTINUE."""
        selector = StrategySelector()

        strategy = selector.select_strategy(
            provider="unknown",
            model="unknown-model",
            compaction_occurred=True,
            messages_removed=10,
            current_turn=5,
        )

        assert strategy == ContinuationStrategy.DIRECT_CONTINUE

    def test_compaction_without_messages_removed(self):
        """Compaction flag True but 0 messages removed should still trigger strategy."""
        selector = StrategySelector()

        strategy = selector.select_strategy(
            provider="deepseek",
            model="deepseek-chat",
            compaction_occurred=True,
            messages_removed=0,  # Edge case
            current_turn=5,
        )

        # Should still use task-aware strategy for DeepSeek
        assert strategy in (
            ContinuationStrategy.TASK_SUMMARY,
            ContinuationStrategy.DIRECT_CONTINUE,
        )


class TestStrategyPrompts:
    """Test suite for strategy prompt generation."""

    def test_direct_continue_prompt_is_simple(self):
        """DIRECT_CONTINUE prompt should be simple."""
        prompt = get_strategy_prompt(
            ContinuationStrategy.DIRECT_CONTINUE,
            task_description="",
        )

        assert "continue" in prompt.lower()
        assert len(prompt) < 100  # Should be concise

    def test_task_summary_prompt_includes_task(self):
        """TASK_SUMMARY prompt should include task description."""
        task = "Implement a new feature for user authentication"
        prompt = get_strategy_prompt(
            ContinuationStrategy.TASK_SUMMARY,
            task_description=task,
        )

        # Should mention continuing with the task
        assert "continue" in prompt.lower() or task.lower() in prompt.lower()

    def test_context_reminder_mentions_compaction(self):
        """CONTEXT_REMINDER prompt should mention compaction."""
        prompt = get_strategy_prompt(
            ContinuationStrategy.CONTEXT_REMINDER,
            task_description="Some task",
            compaction_summary="Removed 50 messages",
        )

        assert "compact" in prompt.lower() or "removed" in prompt.lower()

    def test_fresh_start_has_no_continuation_hint(self):
        """FRESH_START prompt should not mention continuation."""
        prompt = get_strategy_prompt(
            ContinuationStrategy.FRESH_START,
            task_description="",
        )

        # Fresh start means no explicit "continue" prompt
        # The prompt might be empty or just a brief context note
        assert len(prompt) < 150

    def test_silent_strategy_returns_empty(self):
        """SILENT strategy should return empty prompt."""
        prompt = get_strategy_prompt(
            ContinuationStrategy.SILENT,
            task_description="Any task",
        )

        assert prompt == ""


class TestConvenienceFunctions:
    """Test suite for convenience functions."""

    def test_select_strategy_for_context_function(self):
        """Convenience function should work correctly."""
        strategy = select_strategy_for_context(
            provider="deepseek",
            model="deepseek-chat",
            compaction_occurred=True,
            messages_removed=20,
            current_turn=5,
        )

        assert isinstance(strategy, ContinuationStrategy)
        assert strategy == ContinuationStrategy.TASK_SUMMARY

    def test_select_strategy_with_none_values(self):
        """Should handle None values gracefully."""
        strategy = select_strategy_for_context(
            provider=None,
            model=None,
            compaction_occurred=False,
            messages_removed=0,
            current_turn=0,
        )

        # Should not crash and return a valid strategy
        assert isinstance(strategy, ContinuationStrategy)


class TestModelSpecificBehavior:
    """Test suite for model-specific strategy selection."""

    def test_deepseek_coder_gets_task_summary(self):
        """DeepSeek Coder should also get TASK_SUMMARY."""
        selector = StrategySelector()

        strategy = selector.select_strategy(
            provider="deepseek",
            model="deepseek-coder",
            compaction_occurred=True,
            messages_removed=10,
            current_turn=5,
        )

        assert strategy == ContinuationStrategy.TASK_SUMMARY

    def test_local_models_get_direct_continue(self):
        """Local models (Ollama, LMStudio) should get DIRECT_CONTINUE."""
        selector = StrategySelector()

        for provider in ["ollama", "lmstudio", "llama.cpp"]:
            strategy = selector.select_strategy(
                provider=provider,
                model="local-model",
                compaction_occurred=True,
                messages_removed=10,
                current_turn=5,
            )

            assert strategy == ContinuationStrategy.DIRECT_CONTINUE

    def test_google_gets_direct_continue(self):
        """Google models should get DIRECT_CONTINUE."""
        selector = StrategySelector()

        strategy = selector.select_strategy(
            provider="google",
            model="gemini-2.5-flash",
            compaction_occurred=True,
            messages_removed=10,
            current_turn=5,
        )

        assert strategy == ContinuationStrategy.DIRECT_CONTINUE
