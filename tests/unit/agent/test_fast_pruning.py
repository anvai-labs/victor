# Copyright 2025 Vijaykumar Singh
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for fast pruning before LLM compaction (P1 feature).

TDD approach: Tests written first, then implementation.

Fast pruning (from OpenDev paper) removes old tool results before expensive
LLM-based compaction, replacing them with lightweight [pruned] markers.
This reduces LLM compaction cost by 30-40%.
"""

from victor.agent.fast_pruning import (
    FastPruner,
    FastPruningConfig,
)
from victor.providers.base import Message


class TestFastPruner:
    """Test suite for fast pruning functionality."""

    def test_no_pruning_when_empty_messages(self):
        """Empty message list should return empty list."""
        pruner = FastPruner()
        result = pruner.prune_old_tool_results([], 1)
        assert result == []
        assert pruner.get_pruned_count() == 0

    def test_system_messages_never_pruned(self):
        """System messages should never be pruned by default."""
        pruner = FastPruner()
        messages = [
            Message(role="system", content="System prompt"),
            Message(role="user", content="Hello"),
        ]
        result = pruner.prune_old_tool_results(messages, 1)
        assert len(result) == 2
        assert result[0].role == "system"
        assert result[0].content == "System prompt"
        assert pruner.get_pruned_count() == 0

    def test_user_messages_never_pruned(self):
        """User messages should never be pruned by default (P0: preserve intent)."""
        pruner = FastPruner()
        messages = [
            Message(role="user", content="Original task"),
            Message(role="assistant", content="Response"),
        ]
        result = pruner.prune_old_tool_results(messages, 1)
        assert len(result) == 2
        assert result[0].role == "user"
        assert result[0].content == "Original task"
        assert pruner.get_pruned_count() == 0

    def test_large_tool_results_get_pruned(self):
        """Large tool results (>1000 chars) should be pruned."""
        pruner = FastPruner()
        large_content = "x" * 1500

        messages = [
            Message(role="tool", content=large_content, tool_call_id="tc1"),
        ]

        result = pruner.prune_old_tool_results(messages, 1)
        assert len(result) == 1
        assert "[pruned]" in result[0].content
        assert "1500 chars" in result[0].content
        assert pruner.get_pruned_count() == 1

    def test_small_tool_results_not_pruned(self):
        """Small tool results should not be pruned."""
        pruner = FastPruner()

        messages = [
            Message(role="tool", content="Small result", tool_call_id="tc1"),
        ]

        result = pruner.prune_old_tool_results(messages, 1)
        assert len(result) == 1
        assert result[0].content == "Small result"
        assert pruner.get_pruned_count() == 0

    def test_pruned_marker_contains_original_length(self):
        """Pruned marker should indicate original content length."""
        pruner = FastPruner()

        original_length = 2500
        messages = [
            Message(role="tool", content="x" * original_length, tool_call_id="tc1"),
        ]

        result = pruner.prune_old_tool_results(messages, 1)
        assert f"{original_length} chars" in result[0].content

    def test_pruned_marker_contains_preview(self):
        """Pruned marker should contain preview of original content."""
        pruner = FastPruner()

        messages = [
            Message(role="tool", content="Important data: ABC123", tool_call_id="tc1"),
        ]

        # Make it large enough to be pruned
        messages[0].content = "x" * 1500 + "Important data: ABC123"

        result = pruner.prune_old_tool_results(messages, 1)
        # Should contain first 100 chars
        assert "Important data: ABC123" in result[0].content

    def test_multiple_messages_mixed_pruning(self):
        """Test pruning with mixed message types."""
        pruner = FastPruner()

        messages = [
            Message(role="system", content="System"),
            Message(role="user", content="Task"),
            Message(role="assistant", content="Response"),
            Message(role="tool", content="x" * 1500, tool_call_id="tc1"),  # Large - pruned
            Message(role="tool", content="Small", tool_call_id="tc2"),  # Small - kept
            Message(role="assistant", content="Final"),
        ]

        result = pruner.prune_old_tool_results(messages, 10)

        # System and user preserved
        assert result[0].role == "system"
        assert result[0].content == "System"
        assert result[1].role == "user"
        assert result[1].content == "Task"

        # Large tool result pruned
        assert "[pruned]" in result[3].content

        # Small tool result kept
        assert result[4].content == "Small"

        assert pruner.get_pruned_count() == 1

    def test_custom_config_allows_pruning_system_messages(self):
        """Custom config should allow pruning system messages if enabled."""
        config = FastPruningConfig(prune_system_messages=True)
        pruner = FastPruner(config)

        messages = [
            Message(role="system", content="x" * 1500),  # Large system message
        ]

        result = pruner.prune_old_tool_results(messages, 1)
        # System message is now large enough to be pruned with custom config
        assert "[pruned]" in result[0].content
        assert pruner.get_pruned_count() == 1

    def test_pruned_count_increments(self):
        """Pruned count reflects number pruned in last operation."""
        pruner = FastPruner()

        messages = [
            Message(role="tool", content="x" * 1500, tool_call_id="tc1"),
            Message(role="tool", content="y" * 1500, tool_call_id="tc2"),
        ]

        pruner.prune_old_tool_results(messages, 1)
        assert pruner.get_pruned_count() == 2

        # Next operation resets counter (per-operation count)
        pruner.prune_old_tool_results(messages, 1)
        assert pruner.get_pruned_count() == 2  # Reset for new operation


class TestFastPruningIntegration:
    """Integration tests for fast pruning with compaction."""

    def test_pruning_reduces_content_size(self):
        """Pruning should significantly reduce total content size."""
        pruner = FastPruner()

        large_content = "x" * 5000
        messages = [
            Message(role="user", content="Task"),
            Message(role="tool", content=large_content, tool_call_id="tc1"),
        ]

        original_size = sum(len(m.content) for m in messages)
        result = pruner.prune_old_tool_results(messages, 1)
        pruned_size = sum(len(m.content) for m in result)

        assert pruned_size < original_size
        assert pruned_size < original_size * 0.1  # At least 90% reduction

    def test_pruning_preserves_structure(self):
        """Pruning should preserve message structure and order."""
        pruner = FastPruner()

        messages = [
            Message(role="user", content="Task"),
            Message(
                role="assistant",
                content="Let me check",
                tool_calls=[{"id": "tc1", "name": "read", "arguments": "{}"}],
            ),
            Message(role="tool", content="x" * 1500, tool_call_id="tc1"),
            Message(role="assistant", content="Done"),
        ]

        result = pruner.prune_old_tool_results(messages, 10)

        # Same number of messages
        assert len(result) == len(messages)

        # Roles preserved
        assert result[0].role == "user"
        assert result[1].role == "assistant"
        assert result[2].role == "tool"
        assert result[3].role == "assistant"

        # tool_call_id preserved on pruned message
        assert result[2].tool_call_id == "tc1"
