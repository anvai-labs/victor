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

"""TDD tests for ChatService._add_tool_result_to_context method.

Tests the correct propagation of tool_call_id per OpenAI API spec.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, call
from victor.agent.services.chat_service import ChatService, ChatServiceConfig
from victor.tools.base import ToolResult


class TestAddToolResultWithContext:
    """Test suite for _add_tool_result_to_context method.

    Verifies OpenAI API compliance:
    - tool_call_id is properly set from tool_call.id
    - Fallback to tool_name when tool_call_id is None
    - 'name' field is set for tool messages
    - Messages are properly added to context
    """

    @pytest.fixture
    def mock_context_service(self):
        """Mock context service."""
        context = MagicMock()
        context.add_message = MagicMock()
        return context

    @pytest.fixture
    def mock_tool_service(self):
        """Mock tool service."""
        tool_service = MagicMock()
        tool_service.execute_tool = AsyncMock()
        return tool_service

    @pytest.fixture
    def mock_provider_service(self):
        """Mock provider service."""
        return MagicMock()

    @pytest.fixture
    def mock_recovery_service(self):
        """Mock recovery service."""
        return MagicMock()

    @pytest.fixture
    def mock_conversation_controller(self):
        """Mock conversation controller."""
        return MagicMock()

    @pytest.fixture
    def mock_streaming_coordinator(self):
        """Mock streaming coordinator."""
        return MagicMock()

    @pytest.fixture
    def chat_service(
        self,
        mock_context_service,
        mock_tool_service,
        mock_provider_service,
        mock_recovery_service,
        mock_conversation_controller,
        mock_streaming_coordinator,
    ):
        """Create ChatService instance with mocked dependencies."""
        config = ChatServiceConfig()
        service = ChatService(
            config=config,
            provider_service=mock_provider_service,
            tool_service=mock_tool_service,
            context_service=mock_context_service,
            recovery_service=mock_recovery_service,
            conversation_controller=mock_conversation_controller,
            streaming_coordinator=mock_streaming_coordinator,
        )
        return service

    def test_add_tool_result_with_tool_call_id(self, chat_service, mock_context_service):
        """Test tool result with proper tool_call_id set.

        Given: A tool result with tool_call_id='call_abc123'
        When: _add_tool_result_to_context is called
        Then: Message is added with tool_call_id='call_abc123' and name='shell'
        """
        # Arrange
        tool_name = "shell"
        tool_call_id = "call_abc123"
        result = ToolResult(output="Command output", error=None, success=True)

        # Act
        chat_service._add_tool_result_to_context(
            tool_name=tool_name,
            result=result,
            tool_call_id=tool_call_id,
        )

        # Assert
        mock_context_service.add_message.assert_called_once()
        call_args = mock_context_service.add_message.call_args
        # add_message is called with keyword arguments, so use kwargs
        message = call_args.kwargs if call_args.kwargs else call_args[0][0]

        assert message["role"] == "tool"
        assert message["content"] == "Command output"
        assert message["tool_call_id"] == "call_abc123", "tool_call_id must match the tool call ID"
        assert message["name"] == "shell", "name field must be set for OpenAI compatibility"

    def test_add_tool_result_without_tool_call_id_fallback_to_tool_name(
        self, chat_service, mock_context_service
    ):
        """Test tool result without tool_call_id falls back to tool_name.

        Given: A tool result with tool_call_id=None
        When: _add_tool_result_to_context is called
        Then: Message is added with tool_call_id='shell' (fallback to tool_name)
        """
        # Arrange
        tool_name = "read"
        tool_call_id = None
        result = ToolResult(output="File content", error=None, success=True)

        # Act
        chat_service._add_tool_result_to_context(
            tool_name=tool_name,
            result=result,
            tool_call_id=tool_call_id,
        )

        # Assert
        mock_context_service.add_message.assert_called_once()
        call_args = mock_context_service.add_message.call_args
        message = call_args.kwargs if call_args.kwargs else call_args[0][0]

        assert message["role"] == "tool"
        assert message["content"] == "File content"
        assert message["tool_call_id"] == "read", "tool_call_id should fallback to tool_name"
        assert message["name"] == "read"

    def test_add_tool_result_with_error(self, chat_service, mock_context_service):
        """Test tool result with error message.

        Given: A tool result with error='File not found'
        When: _add_tool_result_to_context is called
        Then: Message content contains the error
        """
        # Arrange
        tool_name = "read"
        tool_call_id = "call_xyz789"
        result = ToolResult(output=None, error="File not found", success=False)

        # Act
        chat_service._add_tool_result_to_context(
            tool_name=tool_name,
            result=result,
            tool_call_id=tool_call_id,
        )

        # Assert
        mock_context_service.add_message.assert_called_once()
        call_args = mock_context_service.add_message.call_args
        message = call_args.kwargs if call_args.kwargs else call_args[0][0]

        assert message["role"] == "tool"
        assert message["content"] == "File not found"
        assert message["tool_call_id"] == "call_xyz789"
        assert message["name"] == "read"

    def test_add_tool_result_with_empty_output(self, chat_service, mock_context_service):
        """Test tool result with empty output.

        Given: A tool result with output='' (empty string)
        When: _add_tool_result_to_context is called
        Then: Message content is empty string
        """
        # Arrange
        tool_name = "ls"
        tool_call_id = "call_empty"
        result = ToolResult(output="", error=None, success=True)

        # Act
        chat_service._add_tool_result_to_context(
            tool_name=tool_name,
            result=result,
            tool_call_id=tool_call_id,
        )

        # Assert
        mock_context_service.add_message.assert_called_once()
        call_args = mock_context_service.add_message.call_args
        message = call_args.kwargs if call_args.kwargs else call_args[0][0]

        assert message["role"] == "tool"
        assert message["content"] == ""
        assert message["tool_call_id"] == "call_empty"
        assert message["name"] == "ls"

    @pytest.mark.asyncio
    async def test_execute_tool_calls_propagates_tool_call_id(
        self, chat_service, mock_tool_service, mock_context_service
    ):
        """Test that _execute_tool_calls properly extracts and passes tool_call_id.

        Given: A tool_call with id='call_test123' and function.name='shell'
        When: _execute_tool_calls processes the tool_call
        Then: _add_tool_result_to_context is called with tool_call_id='call_test123'
        """

        # Arrange
        class MockFunction:
            def __init__(self):
                self.name = "shell"
                self.arguments = '{"command": "ls"}'

        class MockToolCall:
            def __init__(self):
                self.id = "call_test123"
                self.function = MockFunction()

        tool_call = MockToolCall()
        mock_result = ToolResult(output="file1.txt\nfile2.txt", error=None, success=True)
        mock_tool_service.execute_tool.return_value = mock_result

        # Act
        await chat_service._execute_tool_calls([tool_call])

        # Assert
        mock_tool_service.execute_tool.assert_called_once_with("shell", '{"command": "ls"}')
        mock_context_service.add_message.assert_called_once()
        call_args = mock_context_service.add_message.call_args
        message = call_args.kwargs if call_args.kwargs else call_args[0][0]

        assert (
            message["tool_call_id"] == "call_test123"
        ), "tool_call_id must be propagated from tool_call.id"
        assert message["name"] == "shell"
        assert message["role"] == "tool"

    @pytest.mark.asyncio
    async def test_execute_tool_calls_without_id_attribute(
        self, chat_service, mock_tool_service, mock_context_service
    ):
        """Test graceful handling when tool_call lacks 'id' attribute.

        Given: A tool_call without 'id' attribute
        When: _execute_tool_calls processes the tool_call
        Then: tool_call_id defaults to None, falling back to tool_name
        """

        # Arrange
        class MockFunction:
            def __init__(self):
                self.name = "grep"
                self.arguments = '{"pattern": "test"}'

        class MockToolCall:
            def __init__(self):
                # No 'id' attribute
                self.function = MockFunction()

        tool_call = MockToolCall()
        mock_result = ToolResult(output="match found", error=None, success=True)
        mock_tool_service.execute_tool.return_value = mock_result

        # Act
        await chat_service._execute_tool_calls([tool_call])

        # Assert
        mock_context_service.add_message.assert_called_once()
        call_args = mock_context_service.add_message.call_args
        message = call_args.kwargs if call_args.kwargs else call_args[0][0]

        # Should fallback to tool_name when tool_call_id is None
        assert message["tool_call_id"] == "grep"
        assert message["name"] == "grep"

    def test_multiple_tool_results_with_unique_ids(self, chat_service, mock_context_service):
        """Test multiple tool results each have their own tool_call_id.

        Given: Three different tool calls with unique IDs
        When: Each tool result is added
        Then: Each message has the correct tool_call_id
        """
        # Arrange & Act
        results = [
            (
                "shell",
                "call_1",
                ToolResult(output="ls output", error=None, success=True),
            ),
            (
                "read",
                "call_2",
                ToolResult(output="file content", error=None, success=True),
            ),
            (
                "grep",
                "call_3",
                ToolResult(output="grep result", error=None, success=True),
            ),
        ]

        for tool_name, tool_call_id, result in results:
            chat_service._add_tool_result_to_context(
                tool_name=tool_name,
                result=result,
                tool_call_id=tool_call_id,
            )

        # Assert
        assert mock_context_service.add_message.call_count == 3

        calls = mock_context_service.add_message.call_args_list
        msg0 = calls[0].kwargs if calls[0].kwargs else calls[0][0][0]
        msg1 = calls[1].kwargs if calls[1].kwargs else calls[1][0][0]
        msg2 = calls[2].kwargs if calls[2].kwargs else calls[2][0][0]

        assert msg0["tool_call_id"] == "call_1"
        assert msg0["name"] == "shell"

        assert msg1["tool_call_id"] == "call_2"
        assert msg1["name"] == "read"

        assert msg2["tool_call_id"] == "call_3"
        assert msg2["name"] == "grep"
