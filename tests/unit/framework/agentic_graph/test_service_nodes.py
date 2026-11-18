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
from unittest.mock import AsyncMock, MagicMock

from victor.framework.agentic_graph.state import (
    create_initial_state,
    AgenticLoopStateModel,
)
from victor.framework.agentic_graph.service_nodes import (
    _get_service_accessor,
    chat_service_node,
    tool_service_node,
    context_service_node,
    provider_service_node,
    prompt_service_node,
)


class TestRuntimeCollaborators:
    "Runtime collaborators are explicit and never become durable state."

    def test_service_accessor_requires_runtime_context(self):
        assert _get_service_accessor(None) is None

    def test_service_accessor_uses_explicit_runtime_context(self):
        mock_services = MagicMock()
        mock_ctx = MagicMock()
        mock_ctx.services = mock_services

        assert _get_service_accessor(mock_ctx) is mock_services

    def test_agentic_state_has_no_runtime_handle_fields(self):
        state = create_initial_state(query="Test")

        assert "_execution_context" not in AgenticLoopStateModel.model_fields
        assert "_execution_context_private" not in AgenticLoopStateModel.model_fields
        assert "_turn_record_channel" not in AgenticLoopStateModel.model_fields
        assert "inject_execution_context" not in globals()


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

        mock_ctx = MagicMock()
        mock_ctx.services = mock_services

        result = await chat_service_node(state, execution_context=mock_ctx)

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

        result = await chat_service_node(
            state, message="Custom message", execution_context=mock_ctx
        )

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

        result = await chat_service_node(state, execution_context=mock_ctx)

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

        result = await tool_service_node(
            state, tool_name="test_tool", tool_args={"arg1": "value1"}, execution_context=mock_ctx
        )

        mock_tool.execute_tool.assert_called_once()
        tool_results = result.context.get("tool_results", [])
        assert len(tool_results) == 1
        assert tool_results[0]["tool"] == "test_tool"

    @pytest.mark.asyncio
    async def test_tool_service_node_copies_nested_tool_results(self):
        existing = {"tool": "previous", "result": "old"}
        state = create_initial_state(
            query="Execute tool",
            context={"tool_results": [existing]},
        )
        mock_tool = AsyncMock()
        mock_tool.execute_tool = AsyncMock(return_value={"status": "success"})
        mock_ctx = MagicMock()
        mock_ctx.services.tool = mock_tool

        result = await tool_service_node(state, tool_name="test_tool", execution_context=mock_ctx)

        assert state.context["tool_results"] == [existing]
        assert len(result.context["tool_results"]) == 2
        assert result.context["tool_results"][0] is existing

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

        result = await tool_service_node(
            state, tool_name="failing_tool", execution_context=mock_ctx
        )

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

        result = await context_service_node(state, max_results=5, execution_context=mock_ctx)

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

        result = await provider_service_node(state, execution_context=mock_ctx)

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

        result = await provider_service_node(
            state,
            provider_name="override-provider",
            model_name="override-model",
            execution_context=mock_ctx,
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
        mock_ctx.prompt_orchestrator = mock_prompt_orchestrator
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

        result = await prompt_service_node(state, execution_context=mock_ctx)

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
        mock_ctx.prompt_orchestrator = mock_prompt_orchestrator
        constraints = object()

        result = await prompt_service_node(
            state,
            base_prompt="You are an assistant.",
            constraints=constraints,
            vertical="coding",
            execution_context=mock_ctx,
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

        # Chat node
        state = await chat_service_node(state, execution_context=mock_ctx)
        assert state.context.get("last_response") == "Use the tool"

        # Tool node
        state = await tool_service_node(state, tool_name="test_tool", execution_context=mock_ctx)
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

        # Chat
        state = await chat_service_node(state, execution_context=mock_ctx)
        assert "last_response" in state.context

        # Provider
        state = await provider_service_node(state, execution_context=mock_ctx)
        assert "current_provider" in state.context

        # Context
        state = await context_service_node(state, execution_context=mock_ctx)
        assert "context_items" in state.context
