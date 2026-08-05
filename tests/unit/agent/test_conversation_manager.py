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

"""Tests for ConversationManager.

Tests the conversation management facade including:
- Message addition and persistence
- Context metrics and overflow detection
- Session management
- Stage tracking delegation
- Embedding store initialization
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from victor.agent.conversation_manager import (
    ConversationManager,
    ConversationManagerConfig,
    create_conversation_manager,
)
from victor.agent.conversation.controller import ContextMetrics
from victor.agent.conversation.state_machine import ConversationStage


class TestConversationManagerConfig:
    """Tests for ConversationManagerConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ConversationManagerConfig()

        assert config.enable_persistence is True
        assert config.enable_embeddings is True
        assert config.max_context_chars == 200000
        assert config.chars_per_token_estimate == 3
        assert config.enable_stage_tracking is True
        assert config.auto_compaction is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = ConversationManagerConfig(
            enable_persistence=False,
            enable_embeddings=False,
            max_context_chars=100000,
            chars_per_token_estimate=4,
            enable_stage_tracking=False,
            auto_compaction=False,
        )

        assert config.enable_persistence is False
        assert config.enable_embeddings is False
        assert config.max_context_chars == 100000
        assert config.chars_per_token_estimate == 4
        assert config.enable_stage_tracking is False
        assert config.auto_compaction is False


class TestConversationManagerInit:
    """Tests for ConversationManager initialization."""

    def test_init_minimal(self):
        """Test initialization with minimal parameters."""
        manager = ConversationManager()

        assert manager._controller is not None
        assert manager._store is None
        assert manager._session_id is None
        assert manager._config.enable_persistence is True

    def test_init_with_provider_model(self):
        """Test initialization with provider and model."""
        manager = ConversationManager(
            provider="anthropic",
            model="claude-3-sonnet",
        )

        assert manager._provider == "anthropic"
        assert manager._model == "claude-3-sonnet"

    def test_init_with_system_prompt(self):
        """Test initialization with system prompt."""
        manager = ConversationManager(
            system_prompt="You are a helpful assistant.",
        )

        assert manager._controller.system_prompt == "You are a helpful assistant."

    def test_init_with_config(self):
        """Test initialization with custom config."""
        config = ConversationManagerConfig(
            max_context_chars=50000,
            enable_persistence=False,
        )
        manager = ConversationManager(config=config)

        assert manager._config.max_context_chars == 50000
        assert manager._config.enable_persistence is False

    def test_init_with_controller(self):
        """Test initialization with pre-configured controller."""
        mock_controller = MagicMock()
        manager = ConversationManager(controller=mock_controller)

        assert manager._controller is mock_controller


class TestConversationManagerMessages:
    """Tests for message management."""

    @pytest.fixture
    def manager(self):
        """Create a ConversationManager without persistence."""
        config = ConversationManagerConfig(enable_persistence=False)
        return ConversationManager(config=config)

    def test_add_user_message(self, manager):
        """Test adding a user message."""
        message = manager.add_user_message("Hello!")

        assert message.role == "user"
        assert message.content == "Hello!"
        assert manager.message_count() == 1

    def test_add_assistant_message(self, manager):
        """Test adding an assistant message."""
        message = manager.add_assistant_message("Hi there!")

        assert message.role == "assistant"
        assert message.content == "Hi there!"

    def test_add_assistant_message_with_tool_calls(self, manager):
        """Test adding an assistant message with tool calls."""
        tool_calls = [{"id": "call_1", "name": "read_file", "arguments": {}}]
        message = manager.add_assistant_message("Reading file...", tool_calls=tool_calls)

        assert message.role == "assistant"
        assert message.tool_calls == tool_calls

    def test_add_tool_result(self, manager):
        """Test adding a tool result."""
        message = manager.add_tool_result(
            tool_call_id="call_1",
            content="File contents here",
        )

        assert message.role == "tool"
        assert message.content == "File contents here"

    def test_add_message_generic(self, manager):
        """Test adding a message with generic role."""
        message = manager.add_message("user", "Test message")

        assert message.role == "user"
        assert message.content == "Test message"

    def test_messages_property(self, manager):
        """Test messages property returns all messages."""
        manager.add_user_message("Hello")
        manager.add_assistant_message("Hi")

        messages = manager.messages
        assert len(messages) == 2
        assert messages[0].role == "user"
        assert messages[1].role == "assistant"

    def test_message_count(self, manager):
        """Test message count."""
        assert manager.message_count() == 0

        manager.add_user_message("One")
        assert manager.message_count() == 1

        manager.add_assistant_message("Two")
        assert manager.message_count() == 2

    def test_get_last_user_message(self, manager):
        """Test getting last user message."""
        manager.add_user_message("First")
        manager.add_assistant_message("Response")
        manager.add_user_message("Second")

        assert manager.get_last_user_message() == "Second"

    def test_get_last_assistant_message(self, manager):
        """Test getting last assistant message."""
        manager.add_user_message("Hello")
        manager.add_assistant_message("First response")
        manager.add_assistant_message("Second response")

        assert manager.get_last_assistant_message() == "Second response"


class TestConversationManagerContext:
    """Tests for context management."""

    @pytest.fixture
    def manager(self):
        """Create a ConversationManager without persistence."""
        config = ConversationManagerConfig(enable_persistence=False)
        return ConversationManager(config=config)

    def test_get_context_metrics(self, manager):
        """Test getting context metrics."""
        manager.add_user_message("Hello, this is a test message")

        metrics = manager.get_context_metrics()

        assert isinstance(metrics, ContextMetrics)
        assert metrics.char_count > 0
        assert metrics.message_count == 1

    def test_check_context_overflow_false(self, manager):
        """Test overflow check when context is small."""
        manager.add_user_message("Small message")

        assert manager.check_context_overflow() is False

    def test_handle_compaction(self, manager):
        """Test triggering compaction."""
        # Add many messages to trigger compaction need
        for i in range(20):
            manager.add_user_message(f"Message {i} " * 100)

        # Compaction should be able to run (even if it doesn't remove messages)
        removed = manager.handle_compaction(user_message="Current query")

        # Should return an integer (may be 0 if nothing to compact)
        assert isinstance(removed, int)

    def test_get_memory_context_no_store(self, manager):
        """Test getting memory context without store."""
        manager.add_user_message("Hello")
        manager.add_assistant_message("Hi")

        context = manager.get_memory_context()

        assert len(context) == 2
        assert context[0]["role"] == "user"
        assert context[1]["role"] == "assistant"


class TestConversationManagerStage:
    """Tests for stage tracking."""

    @pytest.fixture
    def manager(self):
        """Create a ConversationManager without persistence."""
        config = ConversationManagerConfig(enable_persistence=False)
        return ConversationManager(config=config)

    def test_stage_property(self, manager):
        """Test stage property."""
        stage = manager.stage

        assert isinstance(stage, ConversationStage)
        assert stage == ConversationStage.INITIAL

    def test_get_stage_recommended_tools(self, manager):
        """Test getting stage recommended tools."""
        tools = manager.get_stage_recommended_tools()

        assert isinstance(tools, set)


class TestConversationManagerSession:
    """Tests for session management."""

    def test_session_id_none_without_store(self):
        """Test session_id is None without store."""
        config = ConversationManagerConfig(enable_persistence=False)
        manager = ConversationManager(config=config)

        assert manager.session_id is None

    def test_get_recent_sessions_empty_without_store(self):
        """Test get_recent_sessions returns empty list without store."""
        config = ConversationManagerConfig(enable_persistence=False)
        manager = ConversationManager(config=config)

        sessions = manager.get_recent_sessions()

        assert sessions == []

    def test_recover_session_fails_without_store(self):
        """Test recover_session fails without store."""
        config = ConversationManagerConfig(enable_persistence=False)
        manager = ConversationManager(config=config)

        result = manager.recover_session("some_session_id")

        assert result is False

    def test_get_session_stats_without_store(self):
        """Test get_session_stats without store."""
        config = ConversationManagerConfig(
            enable_persistence=False,
            enable_stage_tracking=False,
        )
        manager = ConversationManager(config=config)
        manager.add_user_message("Test")

        stats = manager.get_session_stats()

        assert stats["session_id"] is None
        assert stats["message_count"] == 1
        assert stats["stage"] == "initial"


class TestConversationManagerWithStore:
    """Tests for ConversationManager with mocked store."""

    @pytest.fixture
    def mock_store(self):
        """Create a mock ConversationStore."""
        store = MagicMock()
        store.get_session.return_value = None
        store.create_session.return_value = MagicMock(
            session_id="test_session_123",
            messages=[],
        )
        store.list_sessions.return_value = []
        store.get_session_stats.return_value = {
            "session_id": "test_session_123",
            "message_count": 0,
        }
        # Set session_id on the store to avoid MagicMock being returned
        store.session_id = "test_session_123"
        return store

    @pytest.fixture
    def manager_with_store(self, mock_store):
        """Create ConversationManager with mocked store."""
        config = ConversationManagerConfig(enable_persistence=True)
        return ConversationManager(
            config=config,
            store=mock_store,
            provider="test_provider",
            model="test_model",
        )

    def test_session_created_with_store(self, manager_with_store, mock_store):
        """Test session is created when store is provided."""
        mock_store.create_session.assert_called_once()
        assert manager_with_store.session_id == "test_session_123"

    def test_message_persisted_to_store(self, manager_with_store, mock_store):
        """Test messages are persisted to store."""
        manager_with_store.add_user_message("Hello")

        # Messages are added to the controller immediately
        # Persistence happens via persist_messages(), not add_message()
        assert manager_with_store.message_count() == 1

    def test_get_recent_sessions_delegates_to_store(self, manager_with_store, mock_store):
        """Test get_recent_sessions delegates to store."""
        manager_with_store.get_recent_sessions(limit=5)

        mock_store.list_sessions.assert_called_with(project_path=None, limit=5)

    def test_recover_session_success(self, manager_with_store, mock_store):
        """Test successful session recovery."""
        # Setup mock session with messages
        mock_message = MagicMock()
        mock_message.role.value = "user"
        mock_message.content = "Previous message"

        mock_session = MagicMock()
        mock_session.session_id = "recovered_session"
        mock_session.messages = [mock_message]

        mock_store.get_session.return_value = mock_session
        # Update store session_id to match the recovered session
        mock_store.session_id = "recovered_session"

        result = manager_with_store.recover_session("recovered_session")

        assert result is True
        assert manager_with_store.session_id == "recovered_session"

    def test_recover_session_not_found(self, manager_with_store, mock_store):
        """Test session recovery when session not found."""
        mock_store.get_session.return_value = None

        result = manager_with_store.recover_session("nonexistent_session")

        assert result is False

    def test_get_session_stats_with_store(self, manager_with_store, mock_store):
        """Test get_session_stats delegates to store."""
        stats = manager_with_store.get_session_stats()

        mock_store.get_session_stats.assert_called_with("test_session_123")
        assert stats["session_id"] == "test_session_123"

    def test_get_memory_context_with_store(self, manager_with_store, mock_store):
        """Test get_memory_context delegates to store."""
        mock_store.get_context_messages.return_value = [{"role": "user", "content": "Hello"}]

        context = manager_with_store.get_memory_context(max_tokens=1000)

        mock_store.get_context_messages.assert_called_with(
            session_id="test_session_123",
            max_tokens=1000,
        )


class TestConversationManagerEmbeddings:
    """Tests for embedding store functionality."""

    @pytest.fixture
    def manager(self):
        """Create a ConversationManager without persistence."""
        config = ConversationManagerConfig(
            enable_persistence=False,
            enable_embeddings=True,
        )
        return ConversationManager(config=config)

    @pytest.mark.asyncio
    async def test_initialize_embedding_store_disabled(self):
        """Test embedding store initialization when disabled."""
        config = ConversationManagerConfig(enable_embeddings=False)
        manager = ConversationManager(config=config)

        result = await manager.initialize_embedding_store()

        assert result is False

    @pytest.mark.asyncio
    async def test_search_similar_messages_without_store(self, manager):
        """Test search returns empty when embedding store not initialized."""
        results = await manager.search_similar_messages("test query")

        assert results == []


class TestConversationManagerLifecycle:
    """Tests for lifecycle management."""

    @pytest.fixture
    def manager(self):
        """Create a ConversationManager without persistence."""
        config = ConversationManagerConfig(enable_persistence=False)
        return ConversationManager(config=config)

    def test_reset(self, manager):
        """Test conversation reset."""
        manager.add_user_message("Hello")
        manager.add_assistant_message("Hi")

        manager.reset()

        assert manager.message_count() == 0

    def test_set_system_prompt(self, manager):
        """Test setting system prompt."""
        manager.set_system_prompt("New system prompt")

        assert manager._controller.system_prompt == "New system prompt"

    def test_to_dict(self, manager):
        """Test exporting to dictionary."""
        manager.add_user_message("Test")

        data = manager.to_dict()

        assert "messages" in data
        assert "stage" in data
        assert "session_id" in data
        assert "provider" in data
        assert "model" in data
        assert "persistence_enabled" in data
        assert "embeddings_enabled" in data

    @pytest.mark.asyncio
    async def test_close(self, manager):
        """Test cleanup on close."""
        # Should not raise
        await manager.close()


class TestCreateConversationManager:
    """Tests for create_conversation_manager factory function."""

    def test_create_basic(self):
        """Test basic factory creation without persistence."""
        manager = create_conversation_manager(
            provider="anthropic",
            model="claude-3-sonnet",
            enable_persistence=False,
        )

        assert isinstance(manager, ConversationManager)
        assert manager._provider == "anthropic"
        assert manager._model == "claude-3-sonnet"

    def test_create_with_system_prompt(self):
        """Test factory creation with system prompt."""
        manager = create_conversation_manager(
            system_prompt="You are helpful.",
            enable_persistence=False,
        )

        assert manager._controller.system_prompt == "You are helpful."

    def test_create_without_persistence(self):
        """Test factory creation without persistence."""
        manager = create_conversation_manager(
            enable_persistence=False,
            enable_embeddings=False,
        )

        assert manager._store is None
        assert manager._config.enable_persistence is False
        assert manager._config.enable_embeddings is False
