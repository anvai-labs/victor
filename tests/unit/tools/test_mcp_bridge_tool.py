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

"""Tests for mcp_bridge_tool module.

Updated for context-based injection pattern (Phase 7: Global Tool State Removal).
"""

import pytest
from unittest.mock import MagicMock, AsyncMock

from victor.tools.mcp_bridge_tool import (
    get_mcp_tool_definitions,
    mcp,
    _prefixed,
    _get_mcp_client,
    _get_mcp_prefix,
)


class TestContextHelpers:
    """Tests for context helper functions."""

    def test_get_mcp_client_from_context(self):
        """Test getting MCP client from context."""
        mock_client = MagicMock()
        context = {"mcp_client": mock_client}
        result = _get_mcp_client(context)
        assert result is mock_client

    def test_get_mcp_client_no_context(self):
        """Test getting MCP client with no context returns None."""
        result = _get_mcp_client(None)
        assert result is None

    def test_get_mcp_client_empty_context(self):
        """Test getting MCP client with empty context returns None."""
        result = _get_mcp_client({})
        assert result is None

    def test_get_mcp_prefix_from_context(self):
        """Test getting MCP prefix from context."""
        context = {"mcp_prefix": "custom"}
        result = _get_mcp_prefix(context)
        assert result == "custom"

    def test_get_mcp_prefix_default(self):
        """Test getting MCP prefix with default."""
        result = _get_mcp_prefix(None)
        assert result == "mcp"

    def test_get_mcp_prefix_empty_context(self):
        """Test getting MCP prefix with empty context returns default."""
        result = _get_mcp_prefix({})
        assert result == "mcp"


class TestPrefixed:
    """Tests for _prefixed function."""

    def test_prefixed_default(self):
        """Test prefixing a name with default prefix."""
        assert _prefixed("tool") == "mcp_tool"

    def test_prefixed_with_context(self):
        """Test prefixing a name with context-based prefix."""
        context = {"mcp_prefix": "test"}
        assert _prefixed("tool", context) == "test_tool"


class TestGetMCPToolDefinitions:
    """Tests for get_mcp_tool_definitions function."""

    def test_no_client_in_context(self):
        """Test when no client is in context."""
        defs = get_mcp_tool_definitions(context=None)
        assert defs == []

    def test_client_with_no_tools(self):
        """Test when client has no tools."""
        mock_client = MagicMock()
        mock_client.tools = None
        context = {"mcp_client": mock_client}
        defs = get_mcp_tool_definitions(context=context)
        assert defs == []

    def test_client_with_tools(self):
        """Test when client has tools."""
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_tool.description = "A test tool"
        mock_tool.parameters = {"param1": {"type": "string"}}

        mock_client = MagicMock()
        mock_client.tools = [mock_tool]

        context = {"mcp_client": mock_client, "mcp_prefix": "mcp"}
        defs = get_mcp_tool_definitions(context=context)
        assert len(defs) == 1
        assert defs[0]["name"] == "mcp_test_tool"
        assert defs[0]["description"] == "A test tool"

    def test_client_with_multiple_tools(self):
        """Test when client has multiple tools."""
        mock_tool1 = MagicMock()
        mock_tool1.name = "tool1"
        mock_tool1.description = "Tool 1"
        mock_tool1.parameters = {}

        mock_tool2 = MagicMock()
        mock_tool2.name = "tool2"
        mock_tool2.description = None  # Test default description
        mock_tool2.parameters = {"arg": {"type": "string"}}

        mock_client = MagicMock()
        mock_client.tools = [mock_tool1, mock_tool2]

        context = {"mcp_client": mock_client}
        defs = get_mcp_tool_definitions(context=context)
        assert len(defs) == 2
        assert defs[0]["name"] == "mcp_tool1"
        assert defs[1]["name"] == "mcp_tool2"
        assert "MCP tool tool2" in defs[1]["description"]


class TestMCPCall:
    """Tests for mcp function."""

    @pytest.mark.asyncio
    async def test_mcp_call_no_client(self):
        """Test mcp when no client is in context."""
        result = await mcp(name="test_tool", context=None)
        assert result["success"] is False
        assert "not configured" in result["error"]

    @pytest.mark.asyncio
    async def test_mcp_call_empty_context(self):
        """Test mcp when context has no client."""
        result = await mcp(name="test_tool", context={})
        assert result["success"] is False
        assert "not configured" in result["error"]

    @pytest.mark.asyncio
    async def test_mcp_call_client_not_initialized(self):
        """Test mcp when client is not initialized."""
        mock_client = MagicMock()
        mock_client.initialized = False
        context = {"mcp_client": mock_client}
        result = await mcp(name="test_tool", context=context)
        assert result["success"] is False
        assert "not initialized" in result["error"]

    @pytest.mark.asyncio
    async def test_mcp_call_success(self):
        """Test successful mcp call."""
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.result = {"data": "test"}

        mock_client = MagicMock()
        mock_client.initialized = True
        mock_client.call_tool = AsyncMock(return_value=mock_result)

        context = {"mcp_client": mock_client, "mcp_prefix": "mcp"}
        result = await mcp(name="mcp_test_tool", arguments={"arg": "value"}, context=context)
        assert result["success"] is True
        assert result["output"] == {"data": "test"}
        mock_client.call_tool.assert_called_once_with("test_tool", arg="value")

    @pytest.mark.asyncio
    async def test_mcp_call_strips_prefix(self):
        """Test mcp strips prefix from tool name."""
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.result = "ok"

        mock_client = MagicMock()
        mock_client.initialized = True
        mock_client.call_tool = AsyncMock(return_value=mock_result)

        context = {"mcp_client": mock_client, "mcp_prefix": "custom"}
        await mcp(name="custom_my_tool", context=context)
        mock_client.call_tool.assert_called_once_with("my_tool")

    @pytest.mark.asyncio
    async def test_mcp_call_no_prefix_match(self):
        """Test mcp passes through name without prefix."""
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.result = "ok"

        mock_client = MagicMock()
        mock_client.initialized = True
        mock_client.call_tool = AsyncMock(return_value=mock_result)

        context = {"mcp_client": mock_client, "mcp_prefix": "mcp"}
        await mcp(name="other_tool", context=context)
        mock_client.call_tool.assert_called_once_with("other_tool")

    @pytest.mark.asyncio
    async def test_mcp_call_failure_result(self):
        """Test mcp when tool returns failure."""
        mock_result = MagicMock()
        mock_result.success = False
        mock_result.error = "Tool failed"

        mock_client = MagicMock()
        mock_client.initialized = True
        mock_client.call_tool = AsyncMock(return_value=mock_result)

        context = {"mcp_client": mock_client}
        result = await mcp(name="test_tool", context=context)
        assert result["success"] is False
        assert result["output"] == "Tool failed"

    @pytest.mark.asyncio
    async def test_mcp_call_exception(self):
        """Test mcp when exception occurs."""
        mock_client = MagicMock()
        mock_client.initialized = True
        mock_client.call_tool = AsyncMock(side_effect=Exception("Test error"))

        context = {"mcp_client": mock_client}
        result = await mcp(name="test_tool", context=context)
        assert result["success"] is False
        assert "Test error" in result["error"]

    @pytest.mark.asyncio
    async def test_mcp_call_no_arguments(self):
        """Test mcp with no arguments."""
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.result = "ok"

        mock_client = MagicMock()
        mock_client.initialized = True
        mock_client.call_tool = AsyncMock(return_value=mock_result)

        context = {"mcp_client": mock_client}
        result = await mcp(name="test_tool", context=context)
        assert result["success"] is True
        mock_client.call_tool.assert_called_once_with("test_tool")
