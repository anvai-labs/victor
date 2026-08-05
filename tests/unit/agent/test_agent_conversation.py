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

"""Tests for agent/message_history module."""

from victor.agent.message_history import MessageHistory


class TestMessageHistory:
    """Tests for MessageHistory class."""

    def test_init_defaults(self):
        """Test MessageHistory initialization with defaults."""
        manager = MessageHistory()
        assert manager.messages == []
        assert manager.system_prompt == ""
        assert manager._max_history == 100000

    def test_init_with_system_prompt(self):
        """Test MessageHistory with system prompt."""
        manager = MessageHistory(system_prompt="You are a helpful assistant")
        assert manager.system_prompt == "You are a helpful assistant"

    def test_init_with_max_history(self):
        """Test MessageHistory with custom max history."""
        manager = MessageHistory(max_history_messages=50)
        assert manager._max_history == 50

    def test_add_message(self):
        """Test adding a message."""
        manager = MessageHistory()
        msg = manager.add_message("user", "Hello")
        assert len(manager.messages) == 1
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_add_user_message(self):
        """Test adding a user message."""
        manager = MessageHistory()
        msg = manager.add_user_message("Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_add_assistant_message(self):
        """Test adding an assistant message."""
        manager = MessageHistory()
        msg = manager.add_assistant_message("Hi there!")
        assert msg.role == "assistant"
        assert msg.content == "Hi there!"

    def test_add_assistant_message_with_tool_calls(self):
        """Test adding assistant message with tool calls."""
        manager = MessageHistory()
        tool_calls = [{"id": "call_1", "function": {"name": "test"}}]
        msg = manager.add_assistant_message("Using tool", tool_calls=tool_calls)
        assert msg.tool_calls == tool_calls

    def test_add_tool_result(self):
        """Test adding a tool result."""
        manager = MessageHistory()
        msg = manager.add_tool_result("call_1", "Tool output", "test_tool")
        assert msg.role == "tool"
        assert msg.content == "Tool output"
        assert msg.tool_call_id == "call_1"
        assert msg.name == "test_tool"

    def test_ensure_system_prompt(self):
        """Test ensure_system_prompt adds system message."""
        manager = MessageHistory(system_prompt="System instruction")
        manager.add_user_message("Hello")
        manager.ensure_system_prompt()

        messages = manager.messages
        assert len(messages) == 2
        assert messages[0].role == "system"
        assert messages[0].content == "System instruction"

    def test_ensure_system_prompt_only_once(self):
        """Test ensure_system_prompt only adds once."""
        manager = MessageHistory(system_prompt="System instruction")
        manager.ensure_system_prompt()
        manager.ensure_system_prompt()

        messages = manager.messages
        assert len(messages) == 1

    def test_get_messages_for_provider(self):
        """Test getting messages for provider."""
        manager = MessageHistory(system_prompt="System")
        manager.add_user_message("Hello")

        messages = manager.get_messages_for_provider()
        assert len(messages) == 2
        assert messages[0].role == "system"
        assert messages[1].role == "user"

    def test_clear(self):
        """Test clearing conversation."""
        manager = MessageHistory(system_prompt="System")
        manager.add_user_message("Hello")
        manager.ensure_system_prompt()

        manager.clear()

        assert len(manager.messages) == 0
        assert manager._system_added is False

    def test_trim_history(self):
        """Test history trimming when over limit."""
        manager = MessageHistory(max_history_messages=5)

        for i in range(10):
            manager.add_user_message(f"Message {i}")

        assert len(manager.messages) == 5

    def test_trim_history_preserves_system(self):
        """Test trim_history preserves system message."""
        manager = MessageHistory(system_prompt="System", max_history_messages=5)
        manager.ensure_system_prompt()

        for i in range(10):
            manager.add_user_message(f"Message {i}")

        messages = manager.messages
        # System message is preserved, so it may be re-added after trimming
        assert messages[0].role == "system"
        # Allow for system message + max_history messages
        assert len(messages) <= 6

    def test_get_last_user_message(self):
        """Test getting last user message."""
        manager = MessageHistory()
        manager.add_user_message("First")
        manager.add_assistant_message("Response")
        manager.add_user_message("Second")

        assert manager.get_last_user_message() == "Second"

    def test_get_last_user_message_empty(self):
        """Test getting last user message when none exists."""
        manager = MessageHistory()
        assert manager.get_last_user_message() is None

    def test_get_last_assistant_message(self):
        """Test getting last assistant message."""
        manager = MessageHistory()
        manager.add_user_message("Hello")
        manager.add_assistant_message("First response")
        manager.add_user_message("Question")
        manager.add_assistant_message("Second response")

        assert manager.get_last_assistant_message() == "Second response"

    def test_get_last_assistant_message_empty(self):
        """Test getting last assistant message when none exists."""
        manager = MessageHistory()
        assert manager.get_last_assistant_message() is None

    def test_message_count(self):
        """Test message count."""
        manager = MessageHistory()
        assert manager.message_count() == 0

        manager.add_user_message("Hello")
        assert manager.message_count() == 1

        manager.add_assistant_message("Hi")
        assert manager.message_count() == 2

    def test_to_dict(self):
        """Test serialization to dict."""
        manager = MessageHistory(system_prompt="System", max_history_messages=50)
        manager.add_user_message("Hello")

        data = manager.to_dict()

        assert data["system_prompt"] == "System"
        assert data["max_history"] == 50
        assert len(data["messages"]) == 1

    def test_from_dict(self):
        """Test deserialization from dict."""
        data = {
            "system_prompt": "System",
            "system_added": True,
            "max_history": 50,
            "messages": [
                {"role": "system", "content": "System"},
                {"role": "user", "content": "Hello"},
            ],
        }

        manager = MessageHistory.from_dict(data)

        assert manager.system_prompt == "System"
        assert manager._system_added is True
        assert manager._max_history == 50
        assert len(manager.messages) == 2

    def test_system_prompt_setter(self):
        """Test setting system prompt."""
        manager = MessageHistory()
        manager.system_prompt = "New system prompt"
        assert manager.system_prompt == "New system prompt"

    def test_messages_returns_copy(self):
        """Test that messages property returns a copy."""
        manager = MessageHistory()
        manager.add_user_message("Hello")

        messages1 = manager.messages
        messages2 = manager.messages

        assert messages1 is not messages2
        assert messages1 == messages2
