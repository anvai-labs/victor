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

"""Tests for service provider nodes (Phase 3 consolidation)."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from victor.framework.agentic_graph.state import (
    create_initial_state,
    AgenticLoopStateModel,
)
from victor.framework.agentic_graph.service_nodes import (
    _get_execution_context,
    _get_service_accessor,
    inject_execution_context,
    chat_service_node,
    tool_service_node,
    context_service_node,
    provider_service_node,
    prompt_service_node,
)


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_get_execution_context_none(self):
        """Test getting ExecutionContext from state without context."""
        state = create_initial_state(query="Test")
        ctx = _get_execution_context(state)
        assert ctx is None

    def test_get_execution_context_from_private_attr(self):
        """Test getting ExecutionContext from private attribute."""
        state = create_initial_state(query="Test")
        mock_ctx = MagicMock()
        state._execution_context_private = mock_ctx

        ctx = _get_execution_context(state)
        assert ctx is mock_ctx

    def test_get_execution_context_from_context_dict(self):
        """Test getting ExecutionContext from context dict."""
        state = create_initial_state(query="Test")
        mock_ctx = MagicMock()
        state = state.model_copy(update={"context": {"_execution_context": mock_ctx}})

        ctx = _get_execution_context(state)
        assert ctx is mock_ctx

    def test_get_service_accessor_none(self):
        """Test getting ServiceAccessor from state without context."""
        state = create_initial_state(query="Test")
        services = _get_service_accessor(state)
        assert services is None

    def test_get_service_accessor_from_context(self):
        """Test getting ServiceAccessor from ExecutionContext."""
        state = create_initial_state(query="Test")
        mock_services = MagicMock()
        mock_ctx = MagicMock()
        mock_ctx.services = mock_services
        state = state.model_copy(update={"context": {"_execution_context": mock_ctx}})

        services = _get_service_accessor(state)
        assert services is mock_services


class TestInjectExecutionContext:
    """Tests for inject_execution_context function."""

    def test_inject_execution_context(self):
        """Test injecting ExecutionContext into state."""
        state = create_initial_state(query="Test")
        mock_ctx = MagicMock()
        mock_ctx.session_id = "test-session"

        result = inject_execution_context(state, mock_ctx)

        assert result._execution_context_private is mock_ctx
        assert result.context.get("_execution_context") is mock_ctx
        assert result.context.get("session_id") == "test-session"


class TestChatServiceNode:
    """Tests for chat_service_node."""

    @pytest.mark.asyncio
    async def test_chat_service_node_no_services(self):
        """Test chat node without available services."""
        state = create_initial_state(query="Hello")
        result = await chat_service_node(state)
        assert result.query == "Hello"

    @pytest.mark.asyncio
    async def test_chat_service_node_with_mock(self):
        """Test chat node with mock service."""
        state = create_initial_state(query="Test message")

        # Create mock services
        mock_services = MagicMock()
        mock_chat = AsyncMock()
        mock_chat.chat = AsyncMock(return_value="Response text")
        mock_services.chat = mock_chat

        # Inject into state
        mock_ctx = MagicMock()
        mock_ctx.services = mock_services
        state = inject_execution_context(state, mock_ctx)

        result = await chat_service_node(state)

        mock_chat.chat.assert_called_once()
        assert result.context.get("last_response") == "Response text"

    @pytest.mark.asyncio
    async def test_chat_service_node_custom_message(self):
        """Test chat node with custom message."""
        state = create_initial_state(query="Default")

        mock_services = MagicMock()
        mock_chat = AsyncMock()
        mock_chat.chat = AsyncMock(return_value="Custom response")
        mock_services.chat = mock_chat

        mock_ctx = MagicMock()
        mock_ctx.services = mock_services
        state = inject_execution_context(state, mock_ctx)

        result = await chat_service_node(state, message="Custom message")

        mock_chat.chat.assert_called_once()
        assert result.context.get("last_response") == "Custom response"

    @pytest.mark.asyncio
    async def test_chat_service_node_empty_query(self):
        """Test chat node skips empty queries."""
        state = AgenticLoopStateModel(query="")

        mock_services = MagicMock()
        mock_chat = AsyncMock()
        mock_services.chat = mock_chat

        mock_ctx = MagicMock()
        mock_ctx.services = mock_services
        state = inject_execution_context(state, mock_ctx)

        result = await chat_service_node(state)

        mock_chat.chat.assert_not_called()


class TestToolServiceNode:
    """Tests for tool_service_node."""

    @pytest.mark.asyncio
    async def test_tool_service_node_no_services(self):
        """Test tool node without available services."""
        state = create_initial_state(query="Run tool")
        result = await tool_service_node(state, tool_name="test_tool")
        assert result.query == "Run tool"

    @pytest.mark.asyncio
    async def test_tool_service_node_with_mock(self):
        """Test tool node with mock service."""
        state = create_initial_state(query="Execute tool")

        mock_services = MagicMock()
        mock_tool = AsyncMock()
        mock_tool.execute_tool = AsyncMock(return_value={"status": "success"})
        mock_services.tool = mock_tool

        mock_ctx = MagicMock()
        mock_ctx.services = mock_services
        state = inject_execution_context(state, mock_ctx)

        result = await tool_service_node(state, tool_name="test_tool", tool_args={"arg1": "value1"})

        mock_tool.execute_tool.assert_called_once()
        tool_results = result.context.get("tool_results", [])
        assert len(tool_results) == 1
        assert tool_results[0]["tool"] == "test_tool"

    @pytest.mark.asyncio
    async def test_tool_service_node_error_handling(self):
        """Test tool node handles errors gracefully."""
        state = create_initial_state(query="Execute tool")

        mock_services = MagicMock()
        mock_tool = AsyncMock()
        mock_tool.execute_tool = AsyncMock(side_effect=Exception("Tool failed"))
        mock_services.tool = mock_tool

        mock_ctx = MagicMock()
        mock_ctx.services = mock_services
        state = inject_execution_context(state, mock_ctx)

        result = await tool_service_node(state, tool_name="failing_tool")

        tool_results = result.context.get("tool_results", [])
        assert len(tool_results) == 1
        assert "error" in tool_results[0]


class TestContextServiceNode:
    """Tests for context_service_node."""

    @pytest.mark.asyncio
    async def test_context_service_node_no_services(self):
        """Test context node without available services."""
        state = create_initial_state(query="Get context")
        result = await context_service_node(state)
        assert result.query == "Get context"

    @pytest.mark.asyncio
    async def test_context_service_node_with_mock(self):
        """Test context node with mock service."""
        state = create_initial_state(query="Search context")

        mock_services = MagicMock()
        mock_context = AsyncMock()
        mock_context.retrieve_context = AsyncMock(return_value=MagicMock(items=["item1", "item2"]))
        mock_services.context = mock_context

        mock_ctx = MagicMock()
        mock_ctx.services = mock_services
        state = inject_execution_context(state, mock_ctx)

        result = await context_service_node(state, max_results=5)

        mock_context.retrieve_context.assert_called_once()
        assert result.context.get("context_items") == ["item1", "item2"]


class TestProviderServiceNode:
    """Tests for provider_service_node."""

    @pytest.mark.asyncio
    async def test_provider_service_node_no_services(self):
        """Test provider node without available services."""
        state = create_initial_state(query="Get provider")
        result = await provider_service_node(state)
        assert result.query == "Get provider"

    @pytest.mark.asyncio
    async def test_provider_service_node_with_mock(self):
        """Test provider node with mock service."""
        state = create_initial_state(query="Check provider")

        mock_services = MagicMock()
        mock_provider = AsyncMock()
        mock_provider.get_provider_info = AsyncMock(
            return_value={"provider": "test-provider", "model": "test-model"}
        )
        mock_services.provider = mock_provider

        mock_ctx = MagicMock()
        mock_ctx.services = mock_services
        state = inject_execution_context(state, mock_ctx)

        result = await provider_service_node(state)

        mock_provider.get_provider_info.assert_called_once()
        assert result.context.get("current_provider") == "test-provider"
        assert result.context.get("current_model") == "test-model"

    @pytest.mark.asyncio
    async def test_provider_service_node_with_overrides(self):
        """Test provider node with provider/model overrides."""
        state = create_initial_state(query="Check provider")

        mock_services = MagicMock()
        mock_provider = AsyncMock()
        mock_provider.get_provider_info = AsyncMock(
            return_value={"provider": "default", "model": "default-model"}
        )
        mock_services.provider = mock_provider

        mock_ctx = MagicMock()
        mock_ctx.services = mock_services
        state = inject_execution_context(state, mock_ctx)

        result = await provider_service_node(
            state, provider_name="override-provider", model_name="override-model"
        )

        assert result.context.get("current_provider") == "override-provider"
        assert result.context.get("current_model") == "override-model"


class TestPromptServiceNode:
    """Tests for prompt_service_node."""

    @pytest.mark.asyncio
    async def test_prompt_service_node_uses_context_prompt_orchestrator(self):
        state = create_initial_state(query="Build prompt")
        mock_prompt_orchestrator = MagicMock()
        mock_prompt_orchestrator.build_system_prompt.return_value = "Prompt text"
        mock_ctx = MagicMock()
        mock_ctx.metadata = {"prompt_orchestrator": mock_prompt_orchestrator}
        state = inject_execution_context(state, mock_ctx)
        state = state.model_copy(
            update={
                "context": {
                    **state.context,
                    "provider": "anthropic",
                    "model": "claude",
                    "task_type": "edit",
                    "base_prompt": "You are an assistant.",
                }
            }
        )

        result = await prompt_service_node(state)

        assert result.context.get("prompt_overlays") == [
            {
                "name": "framework.agentic_graph.system_prompt",
                "content": "Prompt text",
                "placement": "turn_prefix",
            }
        ]
        assert result.context.get("system_prompt_builder_type") == "framework"
        mock_prompt_orchestrator.build_system_prompt.assert_called_once_with(
            builder_type="framework",
            provider="anthropic",
            model="claude",
            task_type="edit",
            base_prompt="You are an assistant.",
        )

    @pytest.mark.asyncio
    async def test_prompt_service_node_activates_and_deactivates_constraints(self):
        state = create_initial_state(query="Build prompt")
        mock_prompt_orchestrator = MagicMock()
        mock_prompt_orchestrator.build_system_prompt.return_value = "Prompt text"
        mock_prompt_orchestrator.activate_constraints.return_value = True
        mock_ctx = MagicMock()
        mock_ctx.metadata = {"prompt_orchestrator": mock_prompt_orchestrator}
        state = inject_execution_context(state, mock_ctx)
        constraints = object()

        result = await prompt_service_node(
            state,
            base_prompt="You are an assistant.",
            constraints=constraints,
            vertical="coding",
        )

        assert result.context.get("constraints_activated") is True
        mock_prompt_orchestrator.activate_constraints.assert_called_once_with(
            constraints,
            "coding",
        )
        mock_prompt_orchestrator.deactivate_constraints.assert_called_once_with()


class TestServiceIntegration:
    """Integration tests for service nodes."""

    @pytest.mark.asyncio
    async def test_chat_to_tool_flow(self):
        """Test flow from chat to tool service node."""
        state = create_initial_state(query="Chat then tool")

        # Mock services
        mock_services = MagicMock()
        mock_chat = AsyncMock()
        mock_chat.chat = AsyncMock(return_value="Use the tool")
        mock_tool = AsyncMock()
        mock_tool.execute_tool = AsyncMock(return_value={"status": "done"})
        mock_services.chat = mock_chat
        mock_services.tool = mock_tool

        mock_ctx = MagicMock()
        mock_ctx.services = mock_services
        state = inject_execution_context(state, mock_ctx)

        # Chat node
        state = await chat_service_node(state)
        assert state.context.get("last_response") == "Use the tool"

        # Tool node
        state = await tool_service_node(state, tool_name="test_tool")
        tool_results = state.context.get("tool_results", [])
        assert len(tool_results) == 1

    @pytest.mark.asyncio
    async def test_all_service_nodes_sequence(self):
        """Test running all service nodes in sequence."""
        state = create_initial_state(query="Full service flow")

        # Mock all services
        mock_services = MagicMock()
        mock_chat = AsyncMock()
        mock_chat.chat = AsyncMock(return_value="Chat response")
        mock_provider = AsyncMock()
        mock_provider.get_provider_info = AsyncMock(
            return_value={"provider": "test", "model": "test-model"}
        )
        mock_context = AsyncMock()
        mock_context.retrieve_context = AsyncMock(return_value=MagicMock(items=["ctx1"]))
        mock_services.chat = mock_chat
        mock_services.provider = mock_provider
        mock_services.context = mock_context

        mock_ctx = MagicMock()
        mock_ctx.services = mock_services
        state = inject_execution_context(state, mock_ctx)

        # Chat
        state = await chat_service_node(state)
        assert "last_response" in state.context

        # Provider
        state = await provider_service_node(state)
        assert "current_provider" in state.context

        # Context
        state = await context_service_node(state)
        assert "context_items" in state.context
