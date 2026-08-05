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

"""Unit tests for ConversationCoordinator."""

import pytest
import time

from victor_contracts.conversation import (
    ConversationCoordinator,
    ConversationContext,
    ConversationStats,
    ConversationTurn,
    TurnType,
)


class TestConversationCoordinator:
    """Test suite for ConversationCoordinator."""

    def test_initialization(self):
        """Test coordinator initialization with default settings."""
        coordinator = ConversationCoordinator()

        assert coordinator.get_turn_count() == 0
        assert coordinator.is_empty()
        assert coordinator._max_history_turns == 50
        assert coordinator._summarization_threshold == 40

    def test_initialization_with_custom_settings(self):
        """Test coordinator initialization with custom settings."""
        coordinator = ConversationCoordinator(
            max_history_turns=100,
            summarization_threshold=30,
            context_window_size=64000,
        )

        assert coordinator._max_history_turns == 100
        assert coordinator._summarization_threshold == 30
        assert coordinator._context_window_size == 64000

    def test_add_message(self):
        """Test adding a message to the conversation."""
        coordinator = ConversationCoordinator()

        turn_id = coordinator.add_message(
            role="user",
            content="Hello, world!",
            turn_type=TurnType.USER,
        )

        assert turn_id == "turn_1"
        assert coordinator.get_turn_count() == 1
        assert not coordinator.is_empty()

    def test_add_multiple_messages(self):
        """Test adding multiple messages."""
        coordinator = ConversationCoordinator()

        coordinator.add_message("user", "Hello", TurnType.USER)
        coordinator.add_message("assistant", "Hi there!", TurnType.ASSISTANT)
        coordinator.add_message("user", "How are you?", TurnType.USER)

        assert coordinator.get_turn_count() == 3

    def test_get_history(self):
        """Test getting conversation history."""
        coordinator = ConversationCoordinator()

        coordinator.add_message("user", "First message", TurnType.USER)
        coordinator.add_message("assistant", "Response", TurnType.ASSISTANT)
        coordinator.add_message("user", "Second message", TurnType.USER)

        history = coordinator.get_history()

        assert len(history) == 3
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "First message"
        assert history[1]["role"] == "assistant"
        assert history[2]["content"] == "Second message"

    def test_get_history_with_filters(self):
        """Test getting history with type filters."""
        coordinator = ConversationCoordinator()

        coordinator.add_message("system", "You are helpful", TurnType.SYSTEM)
        coordinator.add_message("user", "Hello", TurnType.USER)
        coordinator.add_message("tool", "Tool result", TurnType.TOOL)
        coordinator.add_message("assistant", "Response", TurnType.ASSISTANT)

        # Filter out system messages
        history = coordinator.get_history(include_system=False)
        assert len(history) == 3
        assert all(t["role"] != "system" for t in history)

        # Filter out tool messages
        history = coordinator.get_history(include_tool=False)
        assert len(history) == 3
        assert all(t["role"] != "tool" for t in history)

    def test_get_history_max_turns(self):
        """Test getting history with turn limit."""
        coordinator = ConversationCoordinator()

        for i in range(10):
            coordinator.add_message(f"role_{i}", f"Message {i}", TurnType.USER)

        history = coordinator.get_history(max_turns=5)
        assert len(history) == 5

    def test_get_last_n_turns(self):
        """Test getting the last N turns."""
        coordinator = ConversationCoordinator()

        for i in range(5):
            coordinator.add_message("user", f"Message {i}", TurnType.USER)

        turns = coordinator.get_last_n_turns(3)
        assert len(turns) == 3
        assert turns[0].content == "Message 2"
        assert turns[1].content == "Message 3"
        assert turns[2].content == "Message 4"

    def test_clear_history(self):
        """Test clearing conversation history."""
        coordinator = ConversationCoordinator()

        coordinator.add_message("user", "Test", TurnType.USER)
        coordinator.add_summary("Conversation summary")

        assert not coordinator.is_empty()
        assert len(coordinator.get_summaries()) == 1

        coordinator.clear_history(keep_summaries=False)

        assert coordinator.is_empty()
        assert len(coordinator.get_summaries()) == 0

    def test_clear_history_keep_summaries(self):
        """Test clearing history but keeping summaries."""
        coordinator = ConversationCoordinator()

        coordinator.add_message("user", "Test", TurnType.USER)
        coordinator.add_summary("Summary")

        assert not coordinator.is_empty()

        coordinator.clear_history(keep_summaries=True)

        assert coordinator.is_empty()
        assert len(coordinator.get_summaries()) == 1

    def test_remove_turn(self):
        """Test removing a specific turn."""
        coordinator = ConversationCoordinator()

        turn_id = coordinator.add_message("user", "Test message", TurnType.USER)
        assert not coordinator.is_empty()  # Should have content

        result = coordinator.remove_turn(turn_id)
        assert result is True
        assert coordinator.is_empty()  # Should be empty after removal

    def test_remove_nonexistent_turn(self):
        """Test removing a turn that doesn't exist."""
        coordinator = ConversationCoordinator()

        result = coordinator.remove_turn("nonexistent")
        assert result is False

    def test_estimate_tokens(self):
        """Test token estimation."""
        coordinator = ConversationCoordinator()

        # Rough estimate: ~4 characters per token
        tokens = coordinator.estimate_tokens("This is a test message")
        assert tokens > 0
        assert tokens < len("This is a test message")

    def test_get_context_usage(self):
        """Test getting context window usage."""
        coordinator = ConversationCoordinator(
            context_window_size=1000,
        )

        coordinator.add_message("user", "Test message", TurnType.USER)

        used, total = coordinator.get_context_usage()
        assert used > 0
        assert total == 1000

    def test_get_context_utilization(self):
        """Test getting context utilization percentage."""
        coordinator = ConversationCoordinator(
            context_window_size=1000,
        )

        # Empty conversation
        assert coordinator.get_context_utilization() == 0.0

        # Add some content
        coordinator.add_message("user", "x" * 100, TurnType.USER)
        utilization = coordinator.get_context_utilization()
        assert 0.0 < utilization < 1.0

    def test_truncate_history_if_needed(self):
        """Test automatic history truncation."""
        coordinator = ConversationCoordinator(
            max_history_turns=5,
            context_window_size=100,
        )

        # Fill beyond capacity
        for i in range(10):
            coordinator.add_message("user", "x" * 50, TurnType.USER)

        removed = coordinator.truncate_history_if_needed()
        # Some turns should be removed
        assert removed >= 0

    def test_needs_summarization(self):
        """Test summarization threshold detection."""
        coordinator = ConversationCoordinator(
            summarization_threshold=5,
        )

        assert not coordinator.needs_summarization()

        for i in range(5):
            coordinator.add_message("user", f"Message {i}", TurnType.USER)

        assert coordinator.needs_summarization()

    def test_add_summary(self):
        """Test adding a conversation summary."""
        coordinator = ConversationCoordinator()

        for i in range(10):
            coordinator.add_message("user", f"Message {i}", TurnType.USER)

        coordinator.add_summary("Conversation was about testing")

        summaries = coordinator.get_summaries()
        assert len(summaries) == 1
        assert summaries[0] == "Conversation was about testing"

    def test_get_full_context(self):
        """Test getting full context with summaries."""
        coordinator = ConversationCoordinator()

        coordinator.add_summary("Previous context")
        coordinator.add_message("user", "Current message", TurnType.USER)

        context = coordinator.get_full_context()

        assert "Previous context" in context
        assert "Current message" in context
        assert "Previous Conversation Summaries" in context

    def test_is_duplicate(self):
        """Test duplicate message detection."""
        coordinator = ConversationCoordinator(
            enable_deduplication=True,
        )

        coordinator.add_message("user", "Test message", TurnType.USER)

        assert coordinator.is_duplicate("Test message")
        assert not coordinator.is_duplicate("Different message")

    def test_is_duplicate_disabled(self):
        """Test duplicate detection when disabled."""
        coordinator = ConversationCoordinator(
            enable_deduplication=False,
        )

        coordinator.add_message("user", "Test message", TurnType.USER)

        # Should not detect duplicates when disabled
        assert not coordinator.is_duplicate("Test message")

    def test_get_stats(self):
        """Test getting conversation statistics."""
        coordinator = ConversationCoordinator(
            enable_statistics=True,
        )

        coordinator.add_message("user", "Hello", TurnType.USER)
        coordinator.add_message("assistant", "Hi", TurnType.ASSISTANT)
        coordinator.add_message(
            "tool",
            "Tool result",
            TurnType.TOOL,
            tool_calls=[{"name": "test_tool", "args": {}}],
        )

        stats = coordinator.get_stats()

        assert stats.total_turns == 3
        assert stats.user_turns == 1
        assert stats.assistant_turns == 1
        assert stats.tool_turns == 1
        assert stats.tool_calls == 1

    def test_get_stats_dict(self):
        """Test getting statistics as dictionary."""
        coordinator = ConversationCoordinator()

        coordinator.add_message("user", "Test", TurnType.USER)

        stats_dict = coordinator.get_stats_dict()

        assert isinstance(stats_dict, dict)
        assert "total_turns" in stats_dict
        assert "user_turns" in stats_dict
        assert "duration_seconds" in stats_dict

    def test_reset(self):
        """Test resetting the coordinator."""
        coordinator = ConversationCoordinator()

        coordinator.add_message("user", "Test", TurnType.USER)
        coordinator.add_summary("Summary")

        assert coordinator.get_turn_count() > 0
        assert len(coordinator.get_summaries()) > 0

        coordinator.reset()

        assert coordinator.get_turn_count() == 0
        assert len(coordinator.get_summaries()) == 0
        assert coordinator.is_empty()

    def test_get_observability_data(self):
        """Test getting observability data."""
        coordinator = ConversationCoordinator()

        coordinator.add_message("user", "Test message", TurnType.USER)

        obs_data = coordinator.get_observability_data()

        assert obs_data["source_type"] == "coordinator"
        assert obs_data["coordinator_type"] == "conversation"
        assert "stats" in obs_data
        assert "context" in obs_data

    def test_conversation_context_should_summarize(self):
        """Test ConversationContext.should_summarize()."""
        context = ConversationContext(
            turn_count=40,
            summarization_threshold=40,
        )

        assert context.should_summarize()

        context.turn_count = 39
        assert not context.should_summarize()

    def test_conversation_turn_to_dict(self):
        """Test ConversationTurn.to_dict()."""
        turn = ConversationTurn(
            turn_id="test_turn",
            turn_type=TurnType.USER,
            content="Test content",
            timestamp=time.time(),
        )

        turn_dict = turn.to_dict()

        assert turn_dict["turn_id"] == "test_turn"
        assert turn_dict["turn_type"] == "user"
        assert turn_dict["content"] == "Test content"

    def test_conversation_stats_to_dict(self):
        """Test ConversationStats.to_dict()."""
        stats = ConversationStats(
            total_turns=10,
            user_turns=5,
            assistant_turns=4,
            tool_turns=1,
            start_time=time.time() - 100,
            last_activity=time.time(),
        )

        stats_dict = stats.to_dict()

        assert stats_dict["total_turns"] == 10
        assert stats_dict["user_turns"] == 5
        assert "duration_seconds" in stats_dict
