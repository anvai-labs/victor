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

"""Tests for MCP protocol module."""

from victor.integrations.mcp.protocol import (
    MCPTool,
    MCPToolCallResult,
    MCPResource,
    MCPServerInfo,
    MCPCapabilities,
    MCPParameter,
    MCPParameterType,
    MCPMessageType,
    MCPMessage,
    MCPClientInfo,
    MCPResourceContent,
)


class TestMCPModels:
    """Tests for MCP Pydantic models."""

    def test_mcp_tool(self):
        """Test MCPTool model."""
        tool = MCPTool(
            name="test_tool",
            description="A test tool",
            parameters=[],
        )
        assert tool.name == "test_tool"
        assert tool.description == "A test tool"

    def test_mcp_tool_with_parameters(self):
        """Test MCPTool with parameters."""
        param = MCPParameter(
            name="input",
            type=MCPParameterType.STRING,
            description="Input value",
            required=True,
        )
        tool = MCPTool(
            name="test_tool",
            description="A test tool",
            parameters=[param],
        )
        assert len(tool.parameters) == 1
        assert tool.parameters[0].name == "input"

    def test_mcp_tool_call_result(self):
        """Test MCPToolCallResult model."""
        result = MCPToolCallResult(tool_name="test_tool", success=True, result={"data": "test"})
        assert result.success is True
        assert result.result == {"data": "test"}
        assert result.error is None

    def test_mcp_tool_call_result_error(self):
        """Test MCPToolCallResult with error."""
        result = MCPToolCallResult(
            tool_name="test_tool", success=False, error="Something went wrong"
        )
        assert result.success is False
        assert result.error == "Something went wrong"

    def test_mcp_resource(self):
        """Test MCPResource model."""
        resource = MCPResource(
            uri="file:///test.txt",
            name="test.txt",
            description="A test file",
            mime_type="text/plain",
        )
        assert resource.uri == "file:///test.txt"
        assert resource.name == "test.txt"

    def test_mcp_server_info(self):
        """Test MCPServerInfo model."""
        info = MCPServerInfo(
            name="test_server",
            version="1.0.0",
        )
        assert info.name == "test_server"
        assert info.version == "1.0.0"

    def test_mcp_capabilities(self):
        """Test MCPCapabilities model.

        Per MCP spec, capabilities are represented as objects (not booleans).
        If a capability is supported, include it as {} or with config.
        If not supported, set to None (excluded from serialization).
        """
        caps = MCPCapabilities(
            tools={},  # Empty object = supported
            resources={},  # Empty object = supported
            prompts=None,  # None = not supported (omitted)
        )
        assert caps.tools == {}
        assert caps.resources == {}
        assert caps.prompts is None
        assert caps.supports_tools is True
        assert caps.supports_resources is True
        assert caps.supports_prompts is False

        # Test model_dump excludes None values
        dumped = caps.model_dump()
        assert "tools" in dumped
        assert "resources" in dumped
        assert "prompts" not in dumped  # None values excluded

    def test_mcp_client_info(self):
        """Test MCPClientInfo model."""
        info = MCPClientInfo(
            name="test_client",
            version="1.0.0",
        )
        assert info.name == "test_client"
        assert info.version == "1.0.0"

    def test_mcp_message_type_enum(self):
        """Test MCPMessageType enum."""
        assert MCPMessageType.INITIALIZE == "initialize"
        assert MCPMessageType.LIST_TOOLS == "tools/list"
        assert MCPMessageType.CALL_TOOL == "tools/call"
        assert MCPMessageType.ERROR == "error"

    def test_mcp_parameter_type_enum(self):
        """Test MCPParameterType enum."""
        assert MCPParameterType.STRING == "string"
        assert MCPParameterType.NUMBER == "number"
        assert MCPParameterType.BOOLEAN == "boolean"
        assert MCPParameterType.OBJECT == "object"
        assert MCPParameterType.ARRAY == "array"

    def test_mcp_message(self):
        """Test MCPMessage model."""
        msg = MCPMessage(
            method=MCPMessageType.INITIALIZE,
            params={"key": "value"},
        )
        assert msg.method == MCPMessageType.INITIALIZE
        assert msg.params == {"key": "value"}
        assert msg.jsonrpc == "2.0"

    def test_mcp_resource_content(self):
        """Test MCPResourceContent model."""
        content = MCPResourceContent(
            uri="file:///test.txt",
            content="Hello, World!",
            mime_type="text/plain",
        )
        assert content.uri == "file:///test.txt"
        assert content.content == "Hello, World!"
