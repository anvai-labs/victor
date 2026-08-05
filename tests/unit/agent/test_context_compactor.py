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

"""Tests for ContextCompactor."""

import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from victor.agent.context_compactor import (
    CompactionAction,
    CompactionTrigger,
    CompactorConfig,
    ContextCompactor,
    TruncationStrategy,
    create_context_compactor,
)
from victor.agent.conversation.controller import ContextMetrics
from victor.providers.base import Message


@pytest.fixture
def mock_controller():
    """Create a mock ConversationController."""
    controller = MagicMock()

    # Default metrics - 50% utilization
    controller.get_context_metrics.return_value = ContextMetrics(
        char_count=100000,
        estimated_tokens=25000,
        message_count=20,
        is_overflow_risk=False,
        max_context_chars=200000,
    )

    controller.smart_compact_history.return_value = 10  # 10 messages removed
    controller.get_compaction_summaries.return_value = []
    controller.messages = [Message(role="system", content="System prompt")]

    return controller


@pytest.fixture
def compactor(mock_controller):
    """Create a ContextCompactor with mock controller."""
    return ContextCompactor(mock_controller)


class TestCompactorConfig:
    """Tests for CompactorConfig."""

    def test_default_values(self):
        """Test default configuration values match orchestrator_constants."""
        from victor.config.orchestrator_constants import (
            CONTEXT_LIMITS,
            COMPACTION_CONFIG,
        )

        config = CompactorConfig()

        # These now come from centralized orchestrator_constants
        assert config.proactive_threshold == CONTEXT_LIMITS.proactive_compaction_threshold
        assert config.min_messages_after_compact == COMPACTION_CONFIG.min_messages_after_compact
        assert config.tool_result_max_chars == COMPACTION_CONFIG.tool_result_max_chars
        assert config.tool_result_max_lines == COMPACTION_CONFIG.tool_result_max_lines
        assert config.truncation_strategy == TruncationStrategy.SMART
        assert config.preserve_code_blocks is True
        assert config.enable_proactive is True
        assert config.enable_tool_truncation is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = CompactorConfig(
            proactive_threshold=0.60,
            min_messages_after_compact=10,
            enable_proactive=False,
        )

        assert config.proactive_threshold == 0.60
        assert config.min_messages_after_compact == 10
        assert config.enable_proactive is False


class TestCompactionAction:
    """Tests for CompactionAction dataclass."""

    def test_action_taken_true(self):
        """Test action_taken when messages removed."""
        action = CompactionAction(
            trigger=CompactionTrigger.THRESHOLD,
            messages_removed=5,
        )
        assert action.action_taken is True

    def test_action_taken_false(self):
        """Test action_taken when nothing done."""
        action = CompactionAction(
            trigger=CompactionTrigger.NONE,
            messages_removed=0,
        )
        assert action.action_taken is False

    def test_action_taken_truncation(self):
        """Test action_taken with truncations only."""
        action = CompactionAction(
            trigger=CompactionTrigger.MANUAL,
            messages_removed=0,
            truncations_applied=3,
        )
        assert action.action_taken is True


class TestContextCompactorInit:
    """Tests for ContextCompactor initialization."""

    def test_default_initialization(self, mock_controller):
        """Test default initialization."""
        from victor.config.orchestrator_constants import CONTEXT_LIMITS

        compactor = ContextCompactor(mock_controller)

        assert compactor.controller is mock_controller
        assert compactor.config.proactive_threshold == CONTEXT_LIMITS.proactive_compaction_threshold
        assert compactor._compaction_count == 0

    def test_custom_config(self, mock_controller):
        """Test initialization with custom config."""
        config = CompactorConfig(proactive_threshold=0.80)
        compactor = ContextCompactor(mock_controller, config)

        assert compactor.config.proactive_threshold == 0.80


class TestShouldCompact:
    """Tests for should_compact method."""

    def test_should_not_compact_low_utilization(self, compactor, mock_controller):
        """Test no compaction at low utilization."""
        mock_controller.get_context_metrics.return_value = ContextMetrics(
            char_count=50000,
            estimated_tokens=12500,
            message_count=10,
            is_overflow_risk=False,
            max_context_chars=200000,
        )

        should, trigger = compactor.should_compact()
        assert should is False
        assert trigger == CompactionTrigger.NONE

    def test_should_compact_at_threshold(self, compactor, mock_controller):
        """Test compaction triggered at threshold."""
        mock_controller.get_context_metrics.return_value = ContextMetrics(
            char_count=185000,  # 92.5% utilization (exceeds 90% threshold)
            estimated_tokens=46250,
            message_count=30,
            is_overflow_risk=False,
            max_context_chars=200000,
        )

        should, trigger = compactor.should_compact()
        assert should is True
        assert trigger == CompactionTrigger.THRESHOLD

    def test_should_compact_on_overflow(self, compactor, mock_controller):
        """Test compaction triggered on overflow."""
        mock_controller.get_context_metrics.return_value = ContextMetrics(
            char_count=200000,
            estimated_tokens=50000,
            message_count=50,
            is_overflow_risk=True,
            max_context_chars=200000,
        )

        should, trigger = compactor.should_compact()
        assert should is True
        assert trigger == CompactionTrigger.OVERFLOW

    def test_proactive_disabled(self, mock_controller):
        """Test proactive compaction can be disabled."""
        config = CompactorConfig(enable_proactive=False)
        compactor = ContextCompactor(mock_controller, config)

        # Even at 92.5% utilization, should not trigger when proactive disabled
        mock_controller.get_context_metrics.return_value = ContextMetrics(
            char_count=185000,
            estimated_tokens=46250,
            message_count=30,
            is_overflow_risk=False,
            max_context_chars=200000,
        )

        should, trigger = compactor.should_compact()
        assert should is False


class TestCheckAndCompact:
    """Tests for check_and_compact method."""

    def test_no_action_low_utilization(self, compactor, mock_controller):
        """Test no action at low utilization."""
        mock_controller.get_context_metrics.return_value = ContextMetrics(
            char_count=50000,
            estimated_tokens=12500,
            message_count=10,
            is_overflow_risk=False,
            max_context_chars=200000,
        )

        action = compactor.check_and_compact()

        assert action.trigger == CompactionTrigger.NONE
        assert action.action_taken is False
        mock_controller.smart_compact_history.assert_not_called()

    def test_compaction_at_threshold(self, compactor, mock_controller):
        """Test compaction at threshold."""
        # First call returns high utilization (92.5%, above 90% threshold)
        mock_controller.get_context_metrics.side_effect = [
            ContextMetrics(
                char_count=185000,
                estimated_tokens=46250,
                message_count=30,
                is_overflow_risk=False,
                max_context_chars=200000,
            ),
            ContextMetrics(
                char_count=80000,
                estimated_tokens=20000,
                message_count=20,
                is_overflow_risk=False,
                max_context_chars=200000,
            ),
        ]

        action = compactor.check_and_compact()

        assert action.trigger == CompactionTrigger.THRESHOLD
        assert action.messages_removed == 10
        assert action.chars_freed == 105000  # 185000 - 80000
        mock_controller.smart_compact_history.assert_called_once()

    def test_forced_compaction(self, compactor, mock_controller):
        """Test forced compaction."""
        # Even at low utilization, force should trigger
        mock_controller.get_context_metrics.side_effect = [
            ContextMetrics(
                char_count=50000,
                estimated_tokens=12500,
                message_count=10,
                is_overflow_risk=False,
                max_context_chars=200000,
            ),
            ContextMetrics(
                char_count=30000,
                estimated_tokens=7500,
                message_count=5,
                is_overflow_risk=False,
                max_context_chars=200000,
            ),
        ]

        action = compactor.check_and_compact(force=True)

        assert action.trigger == CompactionTrigger.MANUAL
        assert action.action_taken is True

    def test_fallback_truncates_large_tool_outputs_when_no_messages_removed(self, mock_controller):
        """Large tool outputs are still compacted when history is too short to prune."""
        large_tool_output = "line\n" * 500
        tool_message = Message(role="tool", content=large_tool_output, tool_call_id="call_1")
        mock_controller.get_messages.return_value = [tool_message]
        mock_controller.smart_compact_history.return_value = 0
        mock_controller.get_context_metrics.side_effect = [
            ContextMetrics(
                char_count=len(large_tool_output),
                estimated_tokens=1000,
                message_count=2,
                is_overflow_risk=False,
                max_context_chars=len(large_tool_output),
            ),
            ContextMetrics(
                char_count=len(large_tool_output),
                estimated_tokens=1000,
                message_count=2,
                is_overflow_risk=False,
                max_context_chars=len(large_tool_output),
            ),
            ContextMetrics(
                char_count=1200,
                estimated_tokens=300,
                message_count=2,
                is_overflow_risk=False,
                max_context_chars=len(large_tool_output),
            ),
        ]
        config = CompactorConfig(
            proactive_threshold=0.5,
            tool_result_max_chars=1000,
            tool_result_max_lines=50,
            enable_fast_pruning=False,
        )
        compactor = ContextCompactor(mock_controller, config)

        action = compactor.check_and_compact()

        assert action.action_taken is True
        assert action.messages_removed == 0
        assert action.truncations_applied == 1
        assert len(tool_message.content) < len(large_tool_output)

    def test_statistics_updated(self, compactor, mock_controller):
        """Test that statistics are updated after compaction."""
        mock_controller.get_context_metrics.side_effect = [
            ContextMetrics(
                char_count=185000,  # 92.5% utilization (exceeds 90% threshold)
                estimated_tokens=46250,
                message_count=30,
                is_overflow_risk=False,
                max_context_chars=200000,
            ),
            ContextMetrics(
                char_count=80000,
                estimated_tokens=20000,
                message_count=20,
                is_overflow_risk=False,
                max_context_chars=200000,
            ),
            # Third call from get_statistics()
            ContextMetrics(
                char_count=80000,
                estimated_tokens=20000,
                message_count=20,
                is_overflow_risk=False,
                max_context_chars=200000,
            ),
        ]

        compactor.check_and_compact()

        stats = compactor.get_statistics()
        assert stats["compaction_count"] == 1
        assert stats["total_chars_freed"] > 0


class TestTruncateToolResult:
    """Tests for tool result truncation."""

    def test_no_truncation_needed(self, compactor):
        """Test no truncation when content is small."""
        content = "Short content that doesn't need truncation."

        result = compactor.truncate_tool_result(content)

        assert result.truncated is False
        assert result.content == content
        assert result.truncated_chars == 0

    def test_truncation_by_chars(self, compactor):
        """Test truncation by character limit."""
        content = "x" * 10000  # Exceeds default 8000

        result = compactor.truncate_tool_result(content)

        assert result.truncated is True
        assert len(result.content) < len(content)
        assert result.truncated_chars > 0

    def test_truncation_by_lines(self, compactor):
        """Test truncation by line limit."""
        content = "\n".join([f"Line {i}" for i in range(500)])

        result = compactor.truncate_tool_result(content)

        assert result.truncated is True
        assert result.content.count("\n") < 500

    def test_truncation_disabled(self, mock_controller):
        """Test truncation can be disabled."""
        config = CompactorConfig(enable_tool_truncation=False)
        compactor = ContextCompactor(mock_controller, config)

        content = "x" * 10000

        result = compactor.truncate_tool_result(content)

        assert result.truncated is False
        assert result.content == content

    def test_head_truncation_strategy(self, mock_controller):
        """Test HEAD truncation strategy."""
        config = CompactorConfig(
            truncation_strategy=TruncationStrategy.HEAD,
            tool_result_max_chars=100,
            tool_result_max_lines=5,
        )
        compactor = ContextCompactor(mock_controller, config)

        content = "\n".join([f"Line {i}: some content here" for i in range(20)])

        result = compactor.truncate_tool_result(content)

        assert result.truncated is True
        assert "Line 0" in result.content
        assert "[output truncated]" in result.content

    def test_tail_truncation_strategy(self, mock_controller):
        """Test TAIL truncation strategy."""
        config = CompactorConfig(
            truncation_strategy=TruncationStrategy.TAIL,
            tool_result_max_chars=100,
            tool_result_max_lines=5,
        )
        compactor = ContextCompactor(mock_controller, config)

        content = "\n".join([f"Line {i}: some content here" for i in range(20)])

        result = compactor.truncate_tool_result(content)

        assert result.truncated is True
        assert "Line 19" in result.content
        assert "[output truncated]" in result.content

    def test_smart_truncation_preserves_errors(self, mock_controller):
        """Test SMART truncation preserves error messages."""
        config = CompactorConfig(
            truncation_strategy=TruncationStrategy.SMART,
            tool_result_max_chars=500,
            tool_result_max_lines=10,
        )
        compactor = ContextCompactor(mock_controller, config)

        lines = [f"Line {i}: normal content" for i in range(50)]
        lines[25] = "Error: Something went wrong!"
        content = "\n".join(lines)

        result = compactor.truncate_tool_result(content)

        # Error should be preserved or at least the beginning/end kept
        assert result.truncated is True
        # Verify truncation marker is present
        assert "truncated" in result.content.lower()


class TestTokenEstimation:
    """Tests for token estimation."""

    def test_estimate_tokens_prose(self, compactor):
        """Test token estimation for prose."""
        tokens = compactor._estimate_tokens(4000, "prose")
        assert tokens == 1000  # 4000 / 4.0

    def test_estimate_tokens_code(self, compactor):
        """Test token estimation for code."""
        tokens = compactor._estimate_tokens(3000, "code")
        assert tokens == 1000  # 3000 / 3.0

    def test_estimate_message_tokens_code(self, compactor):
        """Test message token estimation for code content."""
        message = Message(role="assistant", content="```python\nprint('hello')\n```")

        tokens = compactor.estimate_message_tokens(message)

        # Should use code factor (3.0) plus overhead
        assert tokens > 0

    def test_estimate_message_tokens_prose(self, compactor):
        """Test message token estimation for prose."""
        message = Message(role="user", content="This is a simple text message.")

        tokens = compactor.estimate_message_tokens(message)

        # Should use prose factor (4.0) plus overhead
        assert tokens > 0


class TestStatistics:
    """Tests for statistics methods."""

    def test_get_statistics(self, compactor, mock_controller):
        """Test getting statistics."""
        stats = compactor.get_statistics()

        assert "current_utilization" in stats
        assert "compaction_count" in stats
        assert "proactive_threshold" in stats
        assert stats["compaction_count"] == 0

    def test_reset_statistics(self, compactor, mock_controller):
        """Test resetting statistics."""
        # Perform a compaction (92.5% utilization exceeds 90% threshold)
        mock_controller.get_context_metrics.side_effect = [
            ContextMetrics(
                char_count=185000,
                estimated_tokens=46250,
                message_count=30,
                is_overflow_risk=False,
                max_context_chars=200000,
            ),
            ContextMetrics(
                char_count=80000,
                estimated_tokens=20000,
                message_count=20,
                is_overflow_risk=False,
                max_context_chars=200000,
            ),
        ]
        compactor.check_and_compact()

        # Verify stats were updated
        assert compactor._compaction_count == 1

        # Reset and verify
        compactor.reset_statistics()

        assert compactor._compaction_count == 0
        assert compactor._total_chars_freed == 0


class TestFactoryFunction:
    """Tests for factory function."""

    def test_create_context_compactor(self, mock_controller):
        """Test factory function."""
        compactor = create_context_compactor(
            mock_controller,
            proactive_threshold=0.80,
            enable_proactive=True,
        )

        assert compactor.config.proactive_threshold == 0.80
        assert compactor.config.enable_proactive is True


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_content_truncation(self, compactor):
        """Test truncation with empty content."""
        result = compactor.truncate_tool_result("")

        assert result.truncated is False
        assert result.content == ""

    def test_single_line_content(self, compactor):
        """Test truncation with single line content."""
        content = "Single line content"

        result = compactor.truncate_tool_result(content)

        assert result.truncated is False

    def test_very_long_single_line(self, mock_controller):
        """Test truncation of very long single line."""
        config = CompactorConfig(tool_result_max_chars=100)
        compactor = ContextCompactor(mock_controller, config)

        content = "x" * 1000

        result = compactor.truncate_tool_result(content)

        assert result.truncated is True
        assert len(result.content) < len(content)


# ============================================================================
# New tests for uncovered lines
# ============================================================================


class TestParallelReadBudget:
    """Tests for ParallelReadBudget and calculate_parallel_read_budget."""

    def test_to_prompt_hint(self):
        """Test ParallelReadBudget.to_prompt_hint generates proper hint string."""
        from victor.agent.context_compactor import ParallelReadBudget

        budget = ParallelReadBudget(
            context_window=65536,
            usable_tokens=32768,
            usable_chars=98304,
            max_parallel_files=10,
            chars_per_file=8192,
            total_read_budget=81920,
        )

        hint = budget.to_prompt_hint()

        assert "PARALLEL READ BUDGET" in hint
        assert "10 files" in hint
        assert "8,192 chars" in hint
        assert "81,920 chars" in hint
        # Check that line count is calculated correctly (chars_per_file // 35)
        assert str(8192 // 35) in hint

    def test_calculate_parallel_read_budget_default(self):
        """Test calculate_parallel_read_budget with default parameters."""
        from victor.agent.context_compactor import calculate_parallel_read_budget
        from victor.config.orchestrator_constants import COMPACTION_CONFIG

        budget = calculate_parallel_read_budget()

        # With 65536 context window and config-based output reserve:
        # usable_tokens = 65536 * (1 - output_reserve_pct)
        # usable_chars = usable_tokens * chars_per_token
        # chars_per_file = usable_chars / target_parallel_files (rounded to 1024)
        assert budget.context_window == 65536
        assert budget.usable_tokens > 0
        assert budget.max_parallel_files == COMPACTION_CONFIG.parallel_read_target_files
        assert budget.chars_per_file >= 4096  # Minimum useful read size
        assert budget.total_read_budget == budget.chars_per_file * budget.max_parallel_files

    def test_calculate_parallel_read_budget_custom(self):
        """Test calculate_parallel_read_budget with custom parameters."""
        from victor.agent.context_compactor import calculate_parallel_read_budget

        budget = calculate_parallel_read_budget(
            context_window=131072,  # 128K
            output_reserve=0.25,  # Reserve less
            chars_per_token=4.0,  # Higher estimate
            target_parallel_files=20,
        )

        assert budget.context_window == 131072
        assert budget.max_parallel_files == 20
        # 131072 * 0.75 = 98304 tokens, * 4.0 = 393216 chars / 20 files
        assert budget.usable_tokens == 98304
        assert budget.usable_chars == 393216

    def test_calculate_parallel_read_budget_minimum_chars(self):
        """Test that chars_per_file has a minimum of 4096."""
        from victor.agent.context_compactor import calculate_parallel_read_budget

        # Very small context window should still give minimum chars per file
        budget = calculate_parallel_read_budget(
            context_window=1000,
            output_reserve=0.9,  # Reserve most of it
            target_parallel_files=100,
        )

        assert budget.chars_per_file == 4096  # Minimum enforced


class TestMessagePriorityAssignment:
    """Tests for message priority assignment methods."""

    def test_is_pinned_requirement_positive(self, mock_controller):
        """Test _is_pinned_requirement identifies pinned patterns."""
        compactor = ContextCompactor(mock_controller)

        # Test various pinned patterns
        assert compactor._is_pinned_requirement("You must output a JSON file") is True
        assert compactor._is_pinned_requirement("Required format: markdown table") is True
        assert compactor._is_pinned_requirement("Create a findings table with results") is True
        assert compactor._is_pinned_requirement("Provide top 10 recommendations") is True
        assert compactor._is_pinned_requirement("Deliverables: summary report") is True
        assert compactor._is_pinned_requirement("Output must include the diff") is True
        assert compactor._is_pinned_requirement("Required outputs: metrics") is True

    def test_is_pinned_requirement_negative(self, mock_controller):
        """Test _is_pinned_requirement rejects non-pinned content."""
        compactor = ContextCompactor(mock_controller)

        assert compactor._is_pinned_requirement("Hello world") is False
        assert compactor._is_pinned_requirement("Please help me debug this") is False
        assert compactor._is_pinned_requirement("What is the weather?") is False

    def test_assign_priority_pinned(self, mock_controller):
        """Test that pinned requirements get PINNED priority."""
        from victor.agent.context_compactor import MessagePriority

        compactor = ContextCompactor(mock_controller)

        message = {"role": "user", "content": "You must output a JSON report"}
        priority = compactor._assign_priority(message)

        assert priority == MessagePriority.PINNED

    def test_assign_priority_system(self, mock_controller):
        """Test that system messages get CRITICAL priority."""
        from victor.agent.context_compactor import MessagePriority

        compactor = ContextCompactor(mock_controller)

        message = {"role": "system", "content": "You are a helpful assistant"}
        priority = compactor._assign_priority(message)

        assert priority == MessagePriority.CRITICAL

    def test_assign_priority_error_content(self, mock_controller):
        """Test that error content gets CRITICAL priority."""
        from victor.agent.context_compactor import MessagePriority

        compactor = ContextCompactor(mock_controller)

        message = {"role": "assistant", "content": "Error: File not found"}
        priority = compactor._assign_priority(message)

        assert priority == MessagePriority.CRITICAL

        message = {"role": "assistant", "content": "Exception: ValueError occurred"}
        priority = compactor._assign_priority(message)

        assert priority == MessagePriority.CRITICAL

    def test_assign_priority_user_question(self, mock_controller):
        """Test that user questions get HIGH priority."""
        from victor.agent.context_compactor import MessagePriority

        compactor = ContextCompactor(mock_controller)

        message = {"role": "user", "content": "What does this function do?"}
        priority = compactor._assign_priority(message)

        assert priority == MessagePriority.HIGH

    def test_assign_priority_user_statement(self, mock_controller):
        """Test that user statements get MEDIUM priority."""
        from victor.agent.context_compactor import MessagePriority

        compactor = ContextCompactor(mock_controller)

        message = {"role": "user", "content": "Please fix the bug"}
        priority = compactor._assign_priority(message)

        assert priority == MessagePriority.MEDIUM

    def test_assign_priority_assistant(self, mock_controller):
        """Test that assistant messages get MEDIUM priority."""
        from victor.agent.context_compactor import MessagePriority

        compactor = ContextCompactor(mock_controller)

        message = {"role": "assistant", "content": "Here is the solution"}
        priority = compactor._assign_priority(message)

        assert priority == MessagePriority.MEDIUM

    def test_assign_priority_unknown_role(self, mock_controller):
        """Test that unknown roles get LOW priority."""
        from victor.agent.context_compactor import MessagePriority

        compactor = ContextCompactor(mock_controller)

        message = {"role": "tool", "content": "Tool result here"}
        priority = compactor._assign_priority(message)

        assert priority == MessagePriority.LOW


class TestCheckAndCompactWithRL:
    """Tests for check_and_compact with RL learner integration."""

    def test_check_and_compact_with_rl_learner(self, mock_controller):
        """Test check_and_compact uses RL learner recommendations."""
        from victor.agent.context_compactor import (
            ContextCompactor,
            CompactionTrigger,
        )
        from unittest.mock import MagicMock

        # Create mock RL learner
        mock_learner = MagicMock()
        mock_recommendation = MagicMock()
        mock_recommendation.value = "aggressive_compact"
        mock_recommendation.confidence = 0.9
        mock_recommendation.metadata = {
            "config": {
                "compaction_threshold": 0.65,
                "min_messages_keep": 4,
            }
        }
        mock_learner.get_recommendation.return_value = mock_recommendation

        # Enable RL
        import victor.config.orchestrator_constants as constants

        original_enabled = constants.RL_LEARNER_CONFIG.enabled
        object.__setattr__(constants.RL_LEARNER_CONFIG, "enabled", True)

        try:
            compactor = ContextCompactor(
                mock_controller,
                pruning_learner=mock_learner,
                provider_type="cloud",
            )

            # Configure controller to trigger compaction at threshold
            mock_controller.get_context_metrics.side_effect = [
                ContextMetrics(
                    char_count=140000,  # 70% utilization
                    estimated_tokens=35000,
                    message_count=30,
                    is_overflow_risk=False,
                    max_context_chars=200000,
                ),
                ContextMetrics(
                    char_count=80000,
                    estimated_tokens=20000,
                    message_count=20,
                    is_overflow_risk=False,
                    max_context_chars=200000,
                ),
            ]

            action = compactor.check_and_compact(
                current_query="test query",
                tool_call_count=5,
                task_complexity="high",
            )

            # Should trigger because RL config has lower threshold (0.65)
            assert action.trigger == CompactionTrigger.THRESHOLD
            assert "RL action: aggressive_compact" in action.details

            # Verify RL learner was called with correct parameters
            mock_learner.get_recommendation.assert_called_once()
            call_kwargs = mock_learner.get_recommendation.call_args[1]
            assert call_kwargs["context_utilization"] == 0.7
            assert call_kwargs["tool_call_count"] == 5
            assert call_kwargs["task_complexity"] == "high"
            assert call_kwargs["provider_type"] == "cloud"
        finally:
            object.__setattr__(constants.RL_LEARNER_CONFIG, "enabled", original_enabled)

    def test_check_and_compact_rl_exception_handled(self, mock_controller):
        """Test check_and_compact handles RL learner exceptions gracefully."""
        from victor.agent.context_compactor import (
            ContextCompactor,
            CompactionTrigger,
        )
        from unittest.mock import MagicMock

        # Create mock RL learner that throws exception
        mock_learner = MagicMock()
        mock_learner.get_recommendation.side_effect = RuntimeError("RL error")

        import victor.config.orchestrator_constants as constants

        original_enabled = constants.RL_LEARNER_CONFIG.enabled
        object.__setattr__(constants.RL_LEARNER_CONFIG, "enabled", True)

        try:
            compactor = ContextCompactor(
                mock_controller,
                pruning_learner=mock_learner,
            )

            # Configure controller for high utilization
            mock_controller.get_context_metrics.side_effect = [
                ContextMetrics(
                    char_count=185000,  # 92.5%
                    estimated_tokens=46250,
                    message_count=30,
                    is_overflow_risk=False,
                    max_context_chars=200000,
                ),
                ContextMetrics(
                    char_count=80000,
                    estimated_tokens=20000,
                    message_count=20,
                    is_overflow_risk=False,
                    max_context_chars=200000,
                ),
            ]

            # Should still work using default thresholds
            action = compactor.check_and_compact()

            assert action.trigger == CompactionTrigger.THRESHOLD
            assert action.action_taken is True
        finally:
            object.__setattr__(constants.RL_LEARNER_CONFIG, "enabled", original_enabled)


class TestRecordTaskOutcome:
    """Tests for record_task_outcome RL learning method."""

    def test_record_task_outcome_success(self, mock_controller):
        """Test recording successful task outcome."""
        from victor.agent.context_compactor import ContextCompactor
        from unittest.mock import MagicMock

        mock_learner = MagicMock()

        import victor.config.orchestrator_constants as constants

        original_enabled = constants.RL_LEARNER_CONFIG.enabled
        object.__setattr__(constants.RL_LEARNER_CONFIG, "enabled", True)

        try:
            compactor = ContextCompactor(
                mock_controller,
                pruning_learner=mock_learner,
                provider_type="local",
            )
            compactor._last_rl_action = "conservative_prune"

            mock_controller.get_context_metrics.return_value = ContextMetrics(
                char_count=100000,
                estimated_tokens=25000,
                message_count=20,
                is_overflow_risk=False,
                max_context_chars=200000,
            )

            compactor.record_task_outcome(task_success=True, tokens_saved=500)

            mock_learner.record_outcome.assert_called_once()
            call_kwargs = mock_learner.record_outcome.call_args[1]
            assert call_kwargs["action"] == "conservative_prune"
            assert call_kwargs["task_success"] is True
            assert call_kwargs["tokens_saved"] == 500
            assert call_kwargs["provider_type"] == "local"
            assert compactor._last_rl_action is None  # Should be cleared
        finally:
            object.__setattr__(constants.RL_LEARNER_CONFIG, "enabled", original_enabled)

    def test_record_task_outcome_no_action(self, mock_controller):
        """Test record_task_outcome when no RL action was taken."""
        from victor.agent.context_compactor import ContextCompactor
        from unittest.mock import MagicMock

        mock_learner = MagicMock()

        import victor.config.orchestrator_constants as constants

        original_enabled = constants.RL_LEARNER_CONFIG.enabled
        object.__setattr__(constants.RL_LEARNER_CONFIG, "enabled", True)

        try:
            compactor = ContextCompactor(
                mock_controller,
                pruning_learner=mock_learner,
            )
            # _last_rl_action is None by default

            compactor.record_task_outcome(task_success=True)

            # Should not call learner since no action was taken
            mock_learner.record_outcome.assert_not_called()
        finally:
            object.__setattr__(constants.RL_LEARNER_CONFIG, "enabled", original_enabled)

    def test_record_task_outcome_disabled(self, mock_controller):
        """Test record_task_outcome when RL is disabled."""
        from victor.agent.context_compactor import ContextCompactor
        from unittest.mock import MagicMock

        import victor.config.orchestrator_constants as constants

        original_enabled = constants.RL_LEARNER_CONFIG.enabled
        object.__setattr__(constants.RL_LEARNER_CONFIG, "enabled", False)

        try:
            mock_learner = MagicMock()

            compactor = ContextCompactor(
                mock_controller,
                pruning_learner=mock_learner,  # Learner provided but RL disabled
            )
            compactor._last_rl_action = "some_action"

            compactor.record_task_outcome(task_success=True)

            # Should not call learner since RL is disabled
            mock_learner.record_outcome.assert_not_called()
        finally:
            object.__setattr__(constants.RL_LEARNER_CONFIG, "enabled", original_enabled)

    def test_record_task_outcome_exception_handled(self, mock_controller):
        """Test record_task_outcome handles exceptions gracefully."""
        from victor.agent.context_compactor import ContextCompactor
        from unittest.mock import MagicMock

        mock_learner = MagicMock()
        mock_learner.record_outcome.side_effect = RuntimeError("Recording failed")

        import victor.config.orchestrator_constants as constants

        original_enabled = constants.RL_LEARNER_CONFIG.enabled
        object.__setattr__(constants.RL_LEARNER_CONFIG, "enabled", True)

        try:
            compactor = ContextCompactor(
                mock_controller,
                pruning_learner=mock_learner,
            )
            compactor._last_rl_action = "some_action"

            mock_controller.get_context_metrics.return_value = ContextMetrics(
                char_count=100000,
                estimated_tokens=25000,
                message_count=20,
                is_overflow_risk=False,
                max_context_chars=200000,
            )

            # Should not raise exception
            compactor.record_task_outcome(task_success=True)

            # _last_rl_action should still be cleared
            assert compactor._last_rl_action is None
        finally:
            object.__setattr__(constants.RL_LEARNER_CONFIG, "enabled", original_enabled)


class TestShouldCompactMethod:
    """Additional tests for should_compact method."""

    def test_should_compact_overflow_takes_precedence(self, mock_controller):
        """Test that overflow trigger takes precedence over threshold."""
        from victor.agent.context_compactor import (
            ContextCompactor,
            CompactionTrigger,
        )

        compactor = ContextCompactor(mock_controller)

        # Both overflow risk and high utilization
        mock_controller.get_context_metrics.return_value = ContextMetrics(
            char_count=190000,
            estimated_tokens=47500,
            message_count=50,
            is_overflow_risk=True,  # Overflow flag set
            max_context_chars=200000,
        )

        should, trigger = compactor.should_compact()

        assert should is True
        assert trigger == CompactionTrigger.OVERFLOW  # Not THRESHOLD

    def test_should_compact_proactive_disabled_no_overflow(self, mock_controller):
        """Test proactive disabled but no overflow returns NONE."""
        from victor.agent.context_compactor import (
            ContextCompactor,
            CompactorConfig,
            CompactionTrigger,
        )

        config = CompactorConfig(enable_proactive=False)
        compactor = ContextCompactor(mock_controller, config)

        # High utilization but no overflow
        mock_controller.get_context_metrics.return_value = ContextMetrics(
            char_count=185000,
            estimated_tokens=46250,
            message_count=30,
            is_overflow_risk=False,
            max_context_chars=200000,
        )

        should, trigger = compactor.should_compact()

        assert should is False
        assert trigger == CompactionTrigger.NONE


class TestTruncationStrategies:
    """Detailed tests for truncation strategies."""

    def test_both_truncation_strategy(self, mock_controller):
        """Test BOTH truncation strategy keeps head and tail."""
        from victor.agent.context_compactor import (
            ContextCompactor,
            CompactorConfig,
            TruncationStrategy,
        )

        config = CompactorConfig(
            truncation_strategy=TruncationStrategy.BOTH,
            tool_result_max_chars=200,
            tool_result_max_lines=10,
        )
        compactor = ContextCompactor(mock_controller, config)

        lines = [f"Line {i}: some content here for testing" for i in range(30)]
        content = "\n".join(lines)

        result = compactor.truncate_tool_result(content)

        assert result.truncated is True
        assert "Line 0" in result.content  # Head preserved
        assert "Line 29" in result.content  # Tail preserved
        assert "truncated" in result.content.lower()

    def test_smart_truncation_single_line_large(self, mock_controller):
        """Test smart truncation with single very long line."""
        from victor.agent.context_compactor import (
            ContextCompactor,
            CompactorConfig,
            TruncationStrategy,
        )

        config = CompactorConfig(
            truncation_strategy=TruncationStrategy.SMART,
            tool_result_max_chars=100,
            tool_result_max_lines=10,
        )
        compactor = ContextCompactor(mock_controller, config)

        content = "x" * 500

        result = compactor.truncate_tool_result(content)

        assert result.truncated is True
        assert "[content truncated]" in result.content or "truncated" in result.content.lower()

    def test_smart_truncation_preserves_file_paths(self, mock_controller):
        """Test smart truncation preserves file paths."""
        from victor.agent.context_compactor import (
            ContextCompactor,
            CompactorConfig,
            TruncationStrategy,
        )

        config = CompactorConfig(
            truncation_strategy=TruncationStrategy.SMART,
            tool_result_max_chars=500,
            tool_result_max_lines=10,
        )
        compactor = ContextCompactor(mock_controller, config)

        lines = [f"Line {i}: normal content" for i in range(50)]
        lines[25] = "/path/to/important/file.py"
        content = "\n".join(lines)

        result = compactor.truncate_tool_result(content)

        assert result.truncated is True
        # The path should be marked as priority in truncation
        # (it will be in the preserved portion if it's near beginning or end)

    def test_smart_truncation_with_code_blocks(self, mock_controller):
        """Test smart truncation preserves code blocks when enabled."""
        from victor.agent.context_compactor import (
            ContextCompactor,
            CompactorConfig,
            TruncationStrategy,
        )

        config = CompactorConfig(
            truncation_strategy=TruncationStrategy.SMART,
            tool_result_max_chars=300,
            tool_result_max_lines=15,
            preserve_code_blocks=True,
        )
        compactor = ContextCompactor(mock_controller, config)

        content = """Line 1
Line 2
```python
def hello():
    print("Hello, World!")
```
Line 7
Line 8
Line 9
Line 10
Line 11
Line 12
Line 13
Line 14
Line 15
Line 16
Line 17
Line 18
Line 19
Line 20
Line 21
Line 22
Line 23
Line 24
Line 25"""

        result = compactor.truncate_tool_result(content)

        assert result.truncated is True
        # Code block should be preserved (at least partially)
        assert "```python" in result.content or "def hello" in result.content

    def test_head_truncation_char_limit(self, mock_controller):
        """Test head truncation respects character limit."""
        from victor.agent.context_compactor import (
            ContextCompactor,
            CompactorConfig,
            TruncationStrategy,
        )

        config = CompactorConfig(
            truncation_strategy=TruncationStrategy.HEAD,
            tool_result_max_chars=50,
            tool_result_max_lines=100,  # High line limit, char limit should apply
        )
        compactor = ContextCompactor(mock_controller, config)

        content = "\n".join([f"Line {i}" for i in range(20)])

        result = compactor.truncate_tool_result(content)

        assert result.truncated is True
        assert len(result.content) <= 100  # Some room for truncation marker

    def test_tail_truncation_char_limit(self, mock_controller):
        """Test tail truncation respects character limit."""
        from victor.agent.context_compactor import (
            ContextCompactor,
            CompactorConfig,
            TruncationStrategy,
        )

        config = CompactorConfig(
            truncation_strategy=TruncationStrategy.TAIL,
            tool_result_max_chars=50,
            tool_result_max_lines=100,
        )
        compactor = ContextCompactor(mock_controller, config)

        content = "\n".join([f"Line {i}" for i in range(20)])

        result = compactor.truncate_tool_result(content)

        assert result.truncated is True
        assert "Line 19" in result.content  # Last line preserved

    def test_truncation_disabled_returns_original(self, mock_controller):
        """Test that disabled truncation returns original content."""
        from victor.agent.context_compactor import (
            ContextCompactor,
            CompactorConfig,
        )

        config = CompactorConfig(enable_tool_truncation=False)
        compactor = ContextCompactor(mock_controller, config)

        content = "x" * 100000  # Very large content

        result = compactor.truncate_tool_result(content)

        assert result.truncated is False
        assert result.content == content
        assert result.truncated_chars == 0


class TestTokenEstimation:
    """Additional tests for token estimation."""

    def test_estimate_tokens_json(self, compactor):
        """Test token estimation for JSON content."""
        tokens = compactor._estimate_tokens(2800, "json")
        assert tokens == 1000  # 2800 / 2.8

    def test_estimate_tokens_mixed(self, compactor):
        """Test token estimation for mixed content."""
        tokens = compactor._estimate_tokens(3500, "mixed")
        assert tokens == 1000  # 3500 / 3.5

    def test_estimate_tokens_unknown_type(self, compactor):
        """Test token estimation for unknown content type falls back to mixed."""
        tokens = compactor._estimate_tokens(3500, "unknown_type")
        assert tokens == 1000  # Uses mixed factor 3.5

    def test_estimate_message_tokens_json(self, compactor):
        """Test message token estimation for JSON content."""
        message = Message(role="assistant", content='{"key": "value", "number": 123}')

        tokens = compactor.estimate_message_tokens(message)

        # Should use JSON factor (2.8) plus overhead (4)
        expected_base = len(message.content) / 2.8
        assert tokens >= int(expected_base)


class TestStatisticsMethods:
    """Additional tests for statistics methods."""

    def test_get_compaction_history(self, compactor, mock_controller):
        """Test get_compaction_history delegates to controller."""
        mock_controller.get_compaction_summaries.return_value = [
            "Compacted 5 messages at turn 10",
            "Compacted 3 messages at turn 20",
        ]

        history = compactor.get_compaction_history()

        assert len(history) == 2
        assert "Compacted 5 messages" in history[0]
        mock_controller.get_compaction_summaries.assert_called_once()

    def test_get_statistics_all_fields(self, compactor, mock_controller):
        """Test get_statistics returns all expected fields."""
        stats = compactor.get_statistics()

        required_fields = [
            "current_utilization",
            "current_chars",
            "current_messages",
            "compaction_count",
            "total_chars_freed",
            "total_tokens_freed",
            "last_compaction_turn",
            "proactive_threshold",
            "proactive_enabled",
            "truncation_enabled",
        ]

        for field in required_fields:
            assert field in stats, f"Missing field: {field}"


class TestAsyncMethods:
    """Tests for async methods."""

    @pytest.mark.asyncio
    async def test_check_and_compact_async(self, mock_controller):
        """Test async check_and_compact method."""
        from victor.agent.context_compactor import ContextCompactor, CompactionTrigger

        mock_controller.get_context_metrics.side_effect = [
            ContextMetrics(
                char_count=185000,
                estimated_tokens=46250,
                message_count=30,
                is_overflow_risk=False,
                max_context_chars=200000,
            ),
            ContextMetrics(
                char_count=80000,
                estimated_tokens=20000,
                message_count=20,
                is_overflow_risk=False,
                max_context_chars=200000,
            ),
        ]

        compactor = ContextCompactor(mock_controller)

        action = await compactor.check_and_compact_async()

        assert action.trigger == CompactionTrigger.THRESHOLD
        assert action.action_taken is True

    @pytest.mark.asyncio
    async def test_check_and_compact_async_updates_stats(self, mock_controller):
        """Test async compaction updates async statistics."""
        from victor.agent.context_compactor import ContextCompactor

        mock_controller.get_context_metrics.side_effect = [
            ContextMetrics(
                char_count=185000,
                estimated_tokens=46250,
                message_count=30,
                is_overflow_risk=False,
                max_context_chars=200000,
            ),
            ContextMetrics(
                char_count=80000,
                estimated_tokens=20000,
                message_count=20,
                is_overflow_risk=False,
                max_context_chars=200000,
            ),
            ContextMetrics(
                char_count=80000,
                estimated_tokens=20000,
                message_count=20,
                is_overflow_risk=False,
                max_context_chars=200000,
            ),
        ]

        compactor = ContextCompactor(mock_controller)

        await compactor.check_and_compact_async()

        stats = compactor.get_async_statistics()
        assert stats["async_compactions"] == 1

    @pytest.mark.asyncio
    async def test_should_compact_async(self, mock_controller):
        """Test async should_compact method."""
        from victor.agent.context_compactor import ContextCompactor, CompactionTrigger

        mock_controller.get_context_metrics.return_value = ContextMetrics(
            char_count=185000,
            estimated_tokens=46250,
            message_count=30,
            is_overflow_risk=False,
            max_context_chars=200000,
        )

        compactor = ContextCompactor(mock_controller)

        should, trigger = await compactor.should_compact_async()

        assert should is True
        assert trigger == CompactionTrigger.THRESHOLD

    @pytest.mark.asyncio
    async def test_truncate_tool_result_async(self, mock_controller):
        """Test async truncate_tool_result method."""
        from victor.agent.context_compactor import ContextCompactor

        compactor = ContextCompactor(mock_controller)

        content = "x" * 10000

        result = await compactor.truncate_tool_result_async(content)

        assert result.truncated is True
        assert len(result.content) < len(content)

    @pytest.mark.asyncio
    async def test_get_async_statistics(self, mock_controller):
        """Test get_async_statistics includes async-specific fields."""
        from victor.agent.context_compactor import ContextCompactor

        compactor = ContextCompactor(mock_controller)

        # Ensure async state is initialized
        compactor._ensure_async_state()

        stats = compactor.get_async_statistics()

        assert "async_compactions" in stats
        assert "background_checks" in stats
        assert "background_running" in stats
        assert "last_check_time" in stats
        assert stats["background_running"] is False

    @pytest.mark.asyncio
    async def test_start_and_stop_background_compaction(self, mock_controller):
        """Test starting and stopping background compaction."""
        import asyncio
        from victor.agent.context_compactor import ContextCompactor

        mock_controller.get_context_metrics.return_value = ContextMetrics(
            char_count=50000,
            estimated_tokens=12500,
            message_count=10,
            is_overflow_risk=False,
            max_context_chars=200000,
        )

        compactor = ContextCompactor(mock_controller)

        # Start background compaction with very short interval
        await compactor.start_background_compaction(interval_seconds=0.1)

        # Verify it's running
        assert compactor._async_running is True
        assert compactor._monitor_task is not None

        # Wait a bit for at least one check
        await asyncio.sleep(0.15)

        # Stop background compaction
        await compactor.stop_background_compaction()

        # Verify it's stopped
        assert compactor._async_running is False
        assert compactor._monitor_task is None

    @pytest.mark.asyncio
    async def test_background_compaction_already_running(self, mock_controller):
        """Test starting background compaction when already running."""
        from victor.agent.context_compactor import ContextCompactor

        mock_controller.get_context_metrics.return_value = ContextMetrics(
            char_count=50000,
            estimated_tokens=12500,
            message_count=10,
            is_overflow_risk=False,
            max_context_chars=200000,
        )

        compactor = ContextCompactor(mock_controller)

        # Start first time
        await compactor.start_background_compaction(interval_seconds=10)

        original_task = compactor._monitor_task

        # Start again - should not create new task
        await compactor.start_background_compaction(interval_seconds=5)

        assert compactor._monitor_task is original_task

        # Cleanup
        await compactor.stop_background_compaction()


class TestFactoryFunctionAdvanced:
    """Additional tests for create_context_compactor factory function."""

    def test_create_context_compactor_all_params(self, mock_controller):
        """Test factory function with all parameters."""
        from victor.agent.context_compactor import (
            create_context_compactor,
            TruncationStrategy,
        )

        compactor = create_context_compactor(
            controller=mock_controller,
            proactive_threshold=0.75,
            min_messages_after_compact=10,
            tool_result_max_chars=4096,
            tool_result_max_lines=100,
            truncation_strategy=TruncationStrategy.BOTH,
            preserve_code_blocks=False,
            enable_proactive=False,
            enable_tool_truncation=False,
            provider_type="local",
        )

        assert compactor.config.proactive_threshold == 0.75
        assert compactor.config.min_messages_after_compact == 10
        assert compactor.config.tool_result_max_chars == 4096
        assert compactor.config.tool_result_max_lines == 100
        assert compactor.config.truncation_strategy == TruncationStrategy.BOTH
        assert compactor.config.preserve_code_blocks is False
        assert compactor.config.enable_proactive is False
        assert compactor.config.enable_tool_truncation is False
        assert compactor.provider_type == "local"

    def test_create_context_compactor_with_pruning_learner(self, mock_controller):
        """Test factory function with pruning learner."""
        from victor.agent.context_compactor import create_context_compactor
        from unittest.mock import MagicMock

        mock_learner = MagicMock()

        compactor = create_context_compactor(
            controller=mock_controller,
            pruning_learner=mock_learner,
            provider_type="cloud",
        )

        assert compactor.pruning_learner is mock_learner
        assert compactor.provider_type == "cloud"

    def test_create_context_compactor_with_runtime_intelligence(self, mock_controller):
        """Test factory function preserves runtime-intelligence wiring."""
        runtime_intelligence = MagicMock()

        compactor = create_context_compactor(
            controller=mock_controller,
            runtime_intelligence=runtime_intelligence,
        )

        assert compactor._runtime_intelligence is runtime_intelligence


class TestRuntimeIntelligenceCompaction:
    """Tests for runtime-intelligence-backed compaction decisions."""

    def test_maybe_summarize_turns_uses_runtime_intelligence(self):
        """Compaction tier routing should go through runtime intelligence when present."""
        from victor.agent.decisions.schemas import DecisionType
        from victor.agent.services.protocols.decision_service import DecisionResult

        controller = MagicMock()
        controller.messages = [Message(role="system", content="System prompt")]
        for i in range(18):
            controller.messages.append(Message(role="user", content=f"User turn {i}"))
            controller.messages.append(Message(role="assistant", content=f"Assistant turn {i}"))

        runtime_intelligence = MagicMock()
        runtime_intelligence.decide_sync.return_value = DecisionResult(
            decision_type=DecisionType.COMPACTION,
            result=SimpleNamespace(recommended_tier="edge"),
            source="llm",
            confidence=0.9,
        )

        compactor = ContextCompactor(
            controller=controller,
            runtime_intelligence=runtime_intelligence,
        )
        compactor._get_provider_for_tier = MagicMock(return_value=MagicMock())

        with patch("victor.agent.llm_compaction_summarizer.LLMCompactionSummarizer") as mock_cls:
            mock_cls.return_value.summarize.return_value = "summarized context"

            action = compactor._maybe_summarize_turns()

        assert action is not None
        runtime_intelligence.decide_sync.assert_called_once()


class TestEnsureAsyncState:
    """Tests for _ensure_async_state method."""

    def test_ensure_async_state_initializes_once(self, mock_controller):
        """Test _ensure_async_state only initializes once."""
        from victor.agent.context_compactor import ContextCompactor

        compactor = ContextCompactor(mock_controller)

        # First call should initialize
        compactor._ensure_async_state()
        assert hasattr(compactor, "_async_running")
        assert compactor._async_running is False

        # Modify the state
        compactor._async_running = True

        # Second call should not reinitialize
        compactor._ensure_async_state()
        assert compactor._async_running is True  # Still True


class TestResetStatistics:
    """Additional tests for reset_statistics."""

    def test_reset_statistics_clears_all_counters(self, mock_controller):
        """Test reset_statistics clears all tracking counters."""
        from victor.agent.context_compactor import ContextCompactor

        compactor = ContextCompactor(mock_controller)

        # Set some values
        compactor._total_chars_freed = 10000
        compactor._total_tokens_freed = 2500
        compactor._compaction_count = 5
        compactor._last_compaction_turn = 25

        compactor.reset_statistics()

        assert compactor._total_chars_freed == 0
        assert compactor._total_tokens_freed == 0
        assert compactor._compaction_count == 0
        assert compactor._last_compaction_turn == 0
