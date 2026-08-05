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

"""Tests for ConversationController."""

from unittest.mock import MagicMock

from victor.agent.conversation.controller import (
    ConversationController,
    ConversationConfig,
    ContextMetrics,
    CompactionStrategy,
    MessageImportance,
)
from victor.agent.conversation.types import ConversationMessage, MessagePriority
from victor.agent.conversation.state_machine import ConversationStage
from victor.core.completion_markers import SUMMARY_MARKER
from victor.providers.base import Message


class TestContextMetrics:
    """Tests for ContextMetrics dataclass."""

    def test_utilization_zero_max(self):
        """Test utilization with zero max."""
        metrics = ContextMetrics(
            char_count=100,
            estimated_tokens=25,
            message_count=2,
            max_context_chars=0,
        )
        assert metrics.utilization == 0.0

    def test_utilization_normal(self):
        """Test normal utilization calculation."""
        metrics = ContextMetrics(
            char_count=50000,
            estimated_tokens=12500,
            message_count=10,
            max_context_chars=100000,
        )
        assert metrics.utilization == 0.5

    def test_utilization_capped_at_one(self):
        """Test utilization is capped at 1.0."""
        metrics = ContextMetrics(
            char_count=150000,
            estimated_tokens=37500,
            message_count=20,
            max_context_chars=100000,
        )
        assert metrics.utilization == 1.0


class TestConversationConfig:
    """Tests for ConversationConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ConversationConfig()
        assert config.max_context_chars == 800000
        assert config.chars_per_token_estimate == 4
        assert config.enable_stage_tracking is True
        assert config.enable_context_monitoring is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = ConversationConfig(
            max_context_chars=100000,
            chars_per_token_estimate=3,
        )
        assert config.max_context_chars == 100000
        assert config.chars_per_token_estimate == 3


class TestConversationController:
    """Tests for ConversationController class."""

    def test_init_empty(self):
        """Test initialization with no messages."""
        controller = ConversationController()
        assert len(controller.messages) == 0
        assert controller.message_count == 0

    def test_add_user_message(self):
        """Test adding a user message."""
        controller = ConversationController()
        controller.set_system_prompt("You are helpful.")

        msg = controller.add_user_message("Hello!")

        assert msg.role == "user"
        assert msg.content == "Hello!"
        assert controller.message_count == 2  # system + user

    def test_add_assistant_message(self):
        """Test adding an assistant message."""
        controller = ConversationController()

        msg = controller.add_assistant_message("Hi there!")

        assert msg.role == "assistant"
        assert msg.content == "Hi there!"

    def test_add_assistant_with_tool_calls(self):
        """Test adding assistant message with tool calls."""
        controller = ConversationController()
        tool_calls = [{"name": "read_file", "arguments": {"path": "test.py"}}]

        msg = controller.add_assistant_message("Let me read that.", tool_calls=tool_calls)

        assert msg.tool_calls == tool_calls

    def test_add_tool_result(self):
        """Test adding a tool result."""
        controller = ConversationController()

        msg = controller.add_tool_result(
            tool_call_id="123",
            tool_name="read_file",
            result="file contents here",
        )

        assert msg.role == "tool"
        assert msg.content == "file contents here"
        assert msg.name == "read_file"
        assert msg.tool_call_id == "123"

    def test_system_prompt_added_once(self):
        """Test that system prompt is only added once."""
        controller = ConversationController()
        controller.set_system_prompt("You are helpful.")

        controller.add_user_message("Hello")
        controller.add_user_message("World")

        # Should only have 1 system message
        system_messages = [m for m in controller.messages if m.role == "system"]
        assert len(system_messages) == 1

    def test_get_context_metrics(self):
        """Test getting context metrics."""
        controller = ConversationController()
        controller.add_user_message("Hello, this is a test message.")
        controller.add_assistant_message("This is a response.")

        metrics = controller.get_context_metrics()

        assert metrics.char_count > 0
        assert metrics.estimated_tokens > 0
        assert metrics.message_count == 2

    def test_context_overflow_detection(self):
        """Test context overflow detection."""
        config = ConversationConfig(max_context_chars=100)
        controller = ConversationController(config=config)

        # Add a message that exceeds limit
        controller.add_user_message("x" * 150)

        assert controller.check_context_overflow() is True

    def test_reset(self):
        """Test resetting conversation."""
        controller = ConversationController()
        controller.set_system_prompt("System prompt")
        controller.add_user_message("Hello")
        controller.add_assistant_message("Hi")

        controller.reset()

        assert controller.message_count == 0
        assert controller._system_added is False

    def test_compact_history(self):
        """Test compacting history."""
        controller = ConversationController()
        controller.set_system_prompt("System")

        # Add many messages
        for i in range(20):
            controller.add_user_message(f"Message {i}")

        # Compact to keep 5 recent
        removed = controller.compact_history(keep_recent=5)

        assert removed > 0
        # System message + 5 recent
        assert controller.message_count == 6

    def test_get_last_user_message(self):
        """Test getting last user message."""
        controller = ConversationController()
        controller.add_user_message("First")
        controller.add_assistant_message("Response")
        controller.add_user_message("Second")

        assert controller.get_last_user_message() == "Second"

    def test_get_last_assistant_message(self):
        """Test getting last assistant message."""
        controller = ConversationController()
        controller.add_user_message("Question")
        controller.add_assistant_message("Answer 1")
        controller.add_assistant_message("Answer 2")

        assert controller.get_last_assistant_message() == "Answer 2"

    def test_on_context_overflow_callback(self):
        """Test context overflow callback."""
        config = ConversationConfig(max_context_chars=100)
        controller = ConversationController(config=config)

        callback_called = [False]

        def on_overflow(metrics):
            callback_called[0] = True

        controller.on_context_overflow(on_overflow)
        controller.add_user_message("x" * 150)

        assert callback_called[0] is True

    def test_to_dict(self):
        """Test exporting to dictionary."""
        controller = ConversationController()
        controller.add_user_message("Hello")
        controller.add_assistant_message("Hi")

        data = controller.to_dict()

        assert "messages" in data
        assert "stage" in data
        assert "metrics" in data
        assert len(data["messages"]) == 2

    def test_from_messages(self):
        """Test creating from existing messages."""
        messages = [
            Message(role="system", content="You are helpful."),
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi!"),
        ]

        controller = ConversationController.from_messages(messages)

        assert controller.message_count == 3
        assert controller.system_prompt == "You are helpful."
        assert controller._system_added is True

    def test_add_message_backward_compat(self):
        """Test add_message for backward compatibility."""
        controller = ConversationController()

        controller.add_message("user", "Hello")
        controller.add_message("assistant", "Hi")

        assert controller.message_count == 2
        assert controller.messages[0].role == "user"
        assert controller.messages[1].role == "assistant"

    def test_stage_tracking(self):
        """Test that conversation stage is tracked."""
        controller = ConversationController()

        # Initially should be in initial stage
        assert controller.stage == ConversationStage.INITIAL

    def test_get_stage_recommended_tools(self):
        """Test getting stage-recommended tools."""
        controller = ConversationController()

        tools = controller.get_stage_recommended_tools()

        assert isinstance(tools, set)


class TestCompactionStrategy:
    """Tests for CompactionStrategy enum."""

    def test_compaction_strategy_values(self):
        """Test that all strategy values exist."""
        assert CompactionStrategy.SIMPLE.value == "simple"
        assert CompactionStrategy.TIERED.value == "tiered"
        assert CompactionStrategy.SEMANTIC.value == "semantic"
        assert CompactionStrategy.HYBRID.value == "hybrid"


class TestMessageImportance:
    """Tests for MessageImportance dataclass."""

    def test_message_importance_creation(self):
        """Test creating a MessageImportance."""
        msg = Message(role="user", content="Hello")
        importance = MessageImportance(message=msg, index=0, score=5.0, reason="user")

        assert importance.message == msg
        assert importance.index == 0
        assert importance.score == 5.0
        assert importance.reason == "user"


class TestSmartCompaction:
    """Tests for smart context compaction functionality."""

    def test_smart_compact_history_simple_strategy(self):
        """Test smart compaction with SIMPLE strategy."""
        config = ConversationConfig(
            compaction_strategy=CompactionStrategy.SIMPLE,
            compaction_threshold=0.0,  # Phase 4: Set to 0 to bypass threshold check for this test
        )
        controller = ConversationController(config=config)

        # Add many messages
        for i in range(15):
            controller.add_user_message(f"Message {i}")

        removed = controller.smart_compact_history(target_messages=5)

        assert removed > 0
        assert controller.message_count <= 6  # 5 + potential system message

    def test_smart_compact_history_tiered_strategy(self):
        """Test smart compaction with TIERED strategy prioritizes tool results."""
        config = ConversationConfig(compaction_strategy=CompactionStrategy.TIERED)
        controller = ConversationController(config=config)

        # Add a mix of messages
        controller.set_system_prompt("System prompt")
        controller.add_user_message("User message 1")
        controller.add_assistant_message("Assistant response")
        controller.add_tool_result("tool_1", "read_file", "File contents here - important data")
        controller.add_user_message("User message 2")
        controller.add_assistant_message("Another response")
        controller.add_user_message("User message 3")
        controller.add_assistant_message("Final response")

        # Score messages
        scored = controller._score_messages()

        # Tool results should have higher scores than regular assistant messages
        tool_scores = [s for s in scored if s.message.role == "tool"]
        assistant_scores = [s for s in scored if s.message.role == "assistant"]

        assert len(tool_scores) > 0
        # Tool results score higher than assistant messages (boosted by HIGH priority)
        max_tool = max(s.score for s in tool_scores)
        min_assistant = min(s.score for s in assistant_scores)
        assert max_tool > min_assistant

    def test_smart_compact_preserves_system_message(self):
        """Test that system message is always preserved."""
        config = ConversationConfig(compaction_strategy=CompactionStrategy.TIERED)
        controller = ConversationController(config=config)

        controller.set_system_prompt("Always keep this")
        for i in range(20):
            controller.add_user_message(f"Message {i}")

        controller.smart_compact_history(target_messages=5)

        assert controller.messages[0].role == "system"
        assert controller.messages[0].content == "Always keep this"

    def test_score_messages_produces_valid_scores(self):
        """Test that canonical scorer produces positive scores for all messages."""
        config = ConversationConfig(compaction_strategy=CompactionStrategy.TIERED)
        controller = ConversationController(config=config)

        # Add messages
        for i in range(10):
            controller.add_user_message(f"Message {i}")

        scored = controller._score_messages()

        # All non-system scored messages should have positive scores
        non_system = [s for s in scored if s.reason != "system"]
        assert len(non_system) == 10
        assert all(s.score > 0 for s in non_system)

        # System message should have highest score (1000.0)
        system_scores = [s for s in scored if s.reason == "system"]
        if system_scores:
            assert system_scores[0].score == 1000.0

    def test_generate_compaction_summary(self):
        """Test compaction summary generation."""
        controller = ConversationController()

        messages = [
            Message(role="user", content="How do I use the FileReader class?"),
            Message(role="tool", content="def read_file(): pass"),
            Message(role="assistant", content="Here is how to use it."),
        ]

        summary = controller._generate_compaction_summary(messages)

        assert summary != ""
        assert "user" in summary.lower() or "tool" in summary.lower()

    def test_extract_key_topics(self):
        """Test key topic extraction."""
        controller = ConversationController()

        text = "The FileReader class handles file_operations with read_file method"
        topics = controller._extract_key_topics(text)

        assert len(topics) > 0
        assert any("file" in t.lower() for t in topics)

    def test_compaction_summaries_tracking(self):
        """Test that compaction summaries are tracked."""
        config = ConversationConfig(compaction_strategy=CompactionStrategy.TIERED)
        controller = ConversationController(config=config)

        # Add many messages
        controller.set_system_prompt("System")
        for i in range(15):
            controller.add_user_message(f"Message {i}" * 50)  # Substantial content

        # Compact
        controller.smart_compact_history(target_messages=5)

        # Should have generated a summary
        summaries = controller.get_compaction_summaries()
        assert isinstance(summaries, list)

    def test_set_embedding_service(self):
        """Test setting embedding service."""
        controller = ConversationController()

        # Initially None
        assert controller._embedding_service is None

        # Mock embedding service
        class MockEmbeddingService:
            pass

        mock_service = MockEmbeddingService()
        controller.set_embedding_service(mock_service)

        assert controller._embedding_service == mock_service

    def test_config_compaction_defaults(self):
        """Test compaction configuration defaults match orchestrator_constants."""
        from victor.config.orchestrator_constants import COMPACTION_CONFIG

        config = ConversationConfig()

        assert config.compaction_strategy == CompactionStrategy.TIERED
        assert config.min_messages_to_keep == COMPACTION_CONFIG.min_messages_after_compact
        assert config.tool_result_retention_weight == COMPACTION_CONFIG.tool_result_retention_weight
        assert config.recent_message_weight == COMPACTION_CONFIG.recent_message_weight
        assert config.semantic_relevance_threshold == 0.3

    def test_retrieve_relevant_history_prefers_dual_trace_store_when_available(self):
        """Historical retrieval should surface both semantic and execution traces."""
        controller = ConversationController()
        store = MagicMock()

        semantic_message = ConversationMessage(
            id="msg_semantic",
            role="user",
            content="Authentication middleware fails.",
            priority=MessagePriority.MEDIUM,
            token_count=12,
            metadata={"memory_trace_kind": "semantic"},
        )

        execution_message = ConversationMessage(
            id="msg_execution",
            role="tool",
            content='<TOOL_OUTPUT tool="read" path="src/auth.py">config loader</TOOL_OUTPUT>',
            priority=MessagePriority.HIGH,
            token_count=20,
            tool_name="read",
            tool_call_id="call_1",
            metadata={
                "memory_trace_kind": "execution",
                "memory_execution_text": "tool read path src/auth.py config loader",
            },
        )

        store.get_dual_trace_relevant_messages.return_value = {
            "semantic": [(semantic_message, 0.91)],
            "execution": [(execution_message, 0.83)],
        }
        store.get_relevant_summaries.return_value = []
        store.get_message_trace_text.return_value = "tool read path src/auth.py config loader"

        controller.set_conversation_store(store, "session_123")

        result = controller.retrieve_relevant_history("authentication config", limit=3)

        assert len(result) == 2
        assert "Semantic memory" in result[0]
        assert "Execution trace" in result[1]
        store.get_dual_trace_relevant_messages.assert_called_once_with(
            "session_123",
            "authentication config",
            semantic_limit=2,
            execution_limit=1,
            min_similarity=controller.config.semantic_relevance_threshold,
        )

    def test_retrieve_relevant_history_sanitizes_summary_markers(self):
        controller = ConversationController()
        store = MagicMock()
        store.get_dual_trace_relevant_messages.return_value = {
            "semantic": [],
            "execution": [],
        }
        store.get_relevant_summaries.return_value = [
            (f"{SUMMARY_MARKER} Preserve pending architecture findings.", 0.88)
        ]

        controller.set_conversation_store(store, "session_123")

        result = controller.retrieve_relevant_history("architecture findings", limit=2)

        assert result == [
            "[Prior context summary (relevance: 0.88)]: Preserve pending architecture findings."
        ]


class TestPhase4CompactionFrequency:
    """Tests for Phase 4: Fix Compaction Frequency.

    Verifies that compaction respects minimum interval between compactions
    to prevent aggressive context loss.
    """

    def test_should_compact_below_threshold(self):
        """Test that compaction is skipped when utilization is below threshold."""
        controller = ConversationController()
        controller.config.compaction_threshold = 0.85

        # Utilization below threshold
        result = controller._should_compact(0.70)
        assert result is False

    def test_should_compact_above_threshold(self):
        """Test that compaction proceeds when utilization exceeds threshold."""
        import time

        controller = ConversationController()
        controller.config.compaction_threshold = 0.85
        controller._last_compaction_time = 0.0  # Long time ago

        # Utilization above threshold
        result = controller._should_compact(0.90)
        assert result is True

    def test_should_compact_respects_minimum_interval(self):
        """Test that compaction is blocked when minimum interval not elapsed."""
        import time

        controller = ConversationController()
        controller.config.compaction_threshold = 0.85
        controller._last_compaction_time = time.time()  # Just compacted

        # Utilization above threshold but minimum interval not elapsed
        result = controller._should_compact(0.90)
        assert result is False

    def test_should_compact_after_minimum_interval(self):
        """Test that compaction proceeds after minimum interval elapses."""
        import time

        controller = ConversationController()
        controller.config.compaction_threshold = 0.85
        controller._last_compaction_time = time.time() - 70.0  # 70 seconds ago (> 60s minimum)

        # Utilization above threshold and minimum interval elapsed
        result = controller._should_compact(0.90)
        assert result is True

    def test_compaction_interval_constant(self):
        """Test that MIN_COMPACTION_INTERVAL_SECONDS is set correctly."""
        controller = ConversationController()
        assert controller.MIN_COMPACTION_INTERVAL_SECONDS == 60.0

    def test_smart_compact_checks_interval(self):
        """Test that smart_compact_history respects minimum interval."""
        import time

        controller = ConversationController()
        controller.config.compaction_threshold = 0.85
        controller._last_compaction_time = time.time()  # Just compacted

        # Add many messages to trigger compaction
        for i in range(20):
            controller.add_message("user", f"Message {i}")

        # smart_compact_history should return 0 (blocked by minimum interval)
        removed = controller.smart_compact_history(target_messages=5)
        assert removed == 0

    def test_smart_compact_updates_last_compaction_time(self):
        """Test that smart_compact_history updates _last_compaction_time."""
        import time

        controller = ConversationController()
        controller.config.compaction_threshold = 0.0  # Set to 0 to always trigger compaction
        controller._last_compaction_time = 0.0  # Long time ago

        # Add many messages to trigger compaction
        for i in range(20):
            controller.add_message("user", f"Message {i}")

        # Get initial time
        initial_time = controller._last_compaction_time
        assert initial_time == 0.0

        # Compact (should update _last_compaction_time)
        removed = controller.smart_compact_history(target_messages=5)

        # Verify _last_compaction_time was updated
        assert controller._last_compaction_time > initial_time
        assert removed > 0  # Compaction should have occurred

    def test_compaction_threshold_config_default(self):
        """Test that compaction_threshold has correct default value."""
        controller = ConversationController()
        assert controller.config.compaction_threshold == 0.85

    def test_compaction_threshold_config_custom(self):
        """Test that compaction_threshold can be customized."""
        from victor.agent.conversation.controller import ConversationConfig

        config = ConversationConfig(compaction_threshold=0.90)
        controller = ConversationController(config=config)
        assert controller.config.compaction_threshold == 0.90
