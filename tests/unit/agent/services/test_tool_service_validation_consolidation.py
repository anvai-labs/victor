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

"""Tests for ToolService validation method consolidation.

Tests the consolidated validate_tool_calls method that handles both
single and batch validation with a single canonical API.
"""

import pytest
from unittest.mock import Mock, MagicMock


class TestToolServiceValidationConsolidation:
    """Test consolidated validate_tool_calls method."""

    def test_validate_single_tool_call_valid(self):
        """Test validating a single valid tool call."""
        from victor.agent.services.tool_service import (
            ToolService,
            ToolServiceConfig,
            ToolServiceConfig,
        )

        # Create service
        config = ToolServiceConfig()
        tool_selector = Mock()
        tool_executor = Mock()
        tool_registrar = Mock()
        service = ToolService(
            config=config,
            tool_selector=tool_selector,
            tool_executor=tool_executor,
            tool_registrar=tool_registrar,
        )

        # Mock get_available_tools
        service.get_available_tools = Mock(return_value={"code_search", "file_read"})

        # Validate single tool call
        valid, invalid = service.validate_tool_calls(
            {"name": "code_search", "arguments": {"query": "test"}}
        )

        # Verify results
        assert len(valid) == 1
        assert len(invalid) == 0
        assert valid[0]["name"] == "code_search"
        assert valid[0]["arguments"]["query"] == "test"

    def test_validate_single_tool_call_invalid(self):
        """Test validating a single invalid tool call."""
        from victor.agent.services.tool_service import (
            ToolService,
            ToolServiceConfig,
            ToolServiceConfig,
        )

        # Create service
        config = ToolServiceConfig()
        tool_selector = Mock()
        tool_executor = Mock()
        tool_registrar = Mock()
        service = ToolService(
            config=config,
            tool_selector=tool_selector,
            tool_executor=tool_executor,
            tool_registrar=tool_registrar,
        )

        # Mock get_available_tools
        service.get_available_tools = Mock(return_value={"code_search", "file_read"})

        # Validate single invalid tool call
        valid, invalid = service.validate_tool_calls({"name": "invalid_tool", "arguments": {}})

        # Verify results
        assert len(valid) == 0
        assert len(invalid) == 1
        assert invalid[0]["name"] == "invalid_tool"
        assert "_validation_error" in invalid[0]
        assert "not available" in invalid[0]["_validation_error"]

    def test_validate_single_tool_call_missing_name(self):
        """Test validating a tool call without name."""
        from victor.agent.services.tool_service import ToolService, ToolServiceConfig

        # Create service
        config = ToolServiceConfig()
        tool_selector = Mock()
        tool_executor = Mock()
        tool_registrar = Mock()
        service = ToolService(
            config=config,
            tool_selector=tool_selector,
            tool_executor=tool_executor,
            tool_registrar=tool_registrar,
        )

        # Validate tool call without name
        valid, invalid = service.validate_tool_calls({"arguments": {}})

        # Verify results
        assert len(valid) == 0
        assert len(invalid) == 1
        assert "_validation_error" in invalid[0]
        assert "missing 'name' field" in invalid[0]["_validation_error"]

    def test_validate_single_tool_call_invalid_arguments(self):
        """Test validating a tool call with invalid arguments."""
        from victor.agent.services.tool_service import ToolService, ToolServiceConfig

        # Create service
        config = ToolServiceConfig()
        tool_selector = Mock()
        tool_executor = Mock()
        tool_registrar = Mock()
        service = ToolService(
            config=config,
            tool_selector=tool_selector,
            tool_executor=tool_executor,
            tool_registrar=tool_registrar,
        )

        # Mock get_available_tools
        service.get_available_tools = Mock(return_value={"code_search"})

        # Validate tool call with invalid arguments
        valid, invalid = service.validate_tool_calls(
            {"name": "code_search", "arguments": "not_a_dict"}
        )

        # Verify results
        assert len(valid) == 0
        assert len(invalid) == 1
        assert "_validation_error" in invalid[0]
        assert "must be a dictionary" in invalid[0]["_validation_error"]

    def test_validate_multiple_tool_calls_mixed(self):
        """Test validating multiple tool calls with mixed validity."""
        from victor.agent.services.tool_service import ToolService, ToolServiceConfig

        # Create service
        config = ToolServiceConfig()
        tool_selector = Mock()
        tool_executor = Mock()
        tool_registrar = Mock()
        service = ToolService(
            config=config,
            tool_selector=tool_selector,
            tool_executor=tool_executor,
            tool_registrar=tool_registrar,
        )

        # Mock get_available_tools
        service.get_available_tools = Mock(return_value={"code_search", "file_read", "grep"})

        # Validate multiple tool calls
        valid, invalid = service.validate_tool_calls(
            [
                {"name": "code_search", "arguments": {"query": "test"}},
                {"name": "invalid_tool", "arguments": {}},
                {"name": "file_read", "arguments": {"path": "/tmp/test"}},
                {"name": "missing_name_field"},
            ]
        )

        # Verify results
        assert len(valid) == 2
        assert len(invalid) == 2
        assert valid[0]["name"] == "code_search"
        assert valid[1]["name"] == "file_read"
        assert invalid[0]["name"] == "invalid_tool"
        assert (
            invalid[1]["name"] == "missing_name_field"
        )  # Tool exists but name field value is "missing_name_field"
        assert all("_validation_error" in call for call in invalid)

    def test_validate_tool_calls_with_list_all_valid(self):
        """Test validating a list of all valid tool calls."""
        from victor.agent.services.tool_service import ToolService, ToolServiceConfig

        # Create service
        config = ToolServiceConfig()
        tool_selector = Mock()
        tool_executor = Mock()
        tool_registrar = Mock()
        service = ToolService(
            config=config,
            tool_selector=tool_selector,
            tool_executor=tool_executor,
            tool_registrar=tool_registrar,
        )

        # Mock get_available_tools
        service.get_available_tools = Mock(return_value={"code_search", "file_read", "grep"})

        # Validate list of valid tool calls
        valid, invalid = service.validate_tool_calls(
            [
                {"name": "code_search", "arguments": {"query": "test"}},
                {"name": "file_read", "arguments": {"path": "/tmp/test"}},
                {"name": "grep", "arguments": {"pattern": "foo"}},
            ]
        )

        # Verify results
        assert len(valid) == 3
        assert len(invalid) == 0

    def test_validate_tool_calls_with_list_all_invalid(self):
        """Test validating a list of all invalid tool calls."""
        from victor.agent.services.tool_service import ToolService, ToolServiceConfig

        # Create service
        config = ToolServiceConfig()
        tool_selector = Mock()
        tool_executor = Mock()
        tool_registrar = Mock()
        service = ToolService(
            config=config,
            tool_selector=tool_selector,
            tool_executor=tool_executor,
            tool_registrar=tool_registrar,
        )

        # Mock get_available_tools
        service.get_available_tools = Mock(return_value={"code_search"})

        # Validate list of invalid tool calls
        valid, invalid = service.validate_tool_calls(
            [
                {"name": "invalid_tool_1", "arguments": {}},
                {"name": "invalid_tool_2", "arguments": {}},
                {"arguments": {}},  # Missing name
            ]
        )

        # Verify results
        assert len(valid) == 0
        assert len(invalid) == 3
        assert all("_validation_error" in call for call in invalid)

    def test_validate_tool_calls_with_custom_available_tools(self):
        """Test validation with custom available_tools set."""
        from victor.agent.services.tool_service import ToolService, ToolServiceConfig

        # Create service
        config = ToolServiceConfig()
        tool_selector = Mock()
        tool_executor = Mock()
        tool_registrar = Mock()
        service = ToolService(
            config=config,
            tool_selector=tool_selector,
            tool_executor=tool_executor,
            tool_registrar=tool_registrar,
        )

        # Don't mock get_available_tools - use custom set
        custom_tools = {"code_search", "file_read"}

        # Validate with custom available tools
        valid, invalid = service.validate_tool_calls(
            [
                {"name": "code_search", "arguments": {"query": "test"}},
                {"name": "file_read", "arguments": {"path": "/tmp/test"}},
                {"name": "grep", "arguments": {"pattern": "foo"}},  # Not in custom set
            ],
            available_tools=custom_tools,
        )

        # Verify results
        assert len(valid) == 2
        assert len(invalid) == 1
        assert invalid[0]["name"] == "grep"
        assert "grep" in invalid[0]["_validation_error"]

    def test_validate_tool_calls_with_invalid_input_type(self):
        """Test validation with invalid input type."""
        from victor.agent.services.tool_service import ToolService, ToolServiceConfig

        # Create service
        config = ToolServiceConfig()
        tool_selector = Mock()
        tool_executor = Mock()
        tool_registrar = Mock()
        service = ToolService(
            config=config,
            tool_selector=tool_selector,
            tool_executor=tool_executor,
            tool_registrar=tool_registrar,
        )

        # Validate with invalid input type
        valid, invalid = service.validate_tool_calls("invalid")

        # Verify results
        assert len(valid) == 0
        assert len(invalid) == 1
        assert "_validation_error" in invalid[0]
        assert "must be a dict or list" in invalid[0]["_validation_error"]

    def test_validate_tool_calls_with_empty_list(self):
        """Test validation with empty list."""
        from victor.agent.services.tool_service import ToolService, ToolServiceConfig

        # Create service
        config = ToolServiceConfig()
        tool_selector = Mock()
        tool_executor = Mock()
        tool_registrar = Mock()
        service = ToolService(
            config=config,
            tool_selector=tool_selector,
            tool_executor=tool_executor,
            tool_registrar=tool_registrar,
        )

        # Validate empty list
        valid, invalid = service.validate_tool_calls([])

        # Verify results
        assert len(valid) == 0
        assert len(invalid) == 0

    def test_validate_tool_calls_preserves_original_tool_call_data(self):
        """Test that validation preserves original tool call data."""
        from victor.agent.services.tool_service import ToolService, ToolServiceConfig

        # Create service
        config = ToolServiceConfig()
        tool_selector = Mock()
        tool_executor = Mock()
        tool_registrar = Mock()
        service = ToolService(
            config=config,
            tool_selector=tool_selector,
            tool_executor=tool_executor,
            tool_registrar=tool_registrar,
        )

        # Mock get_available_tools
        service.get_available_tools = Mock(return_value={"code_search"})

        # Create tool call with extra metadata
        tool_call = {
            "name": "code_search",
            "arguments": {"query": "test", "case_sensitive": True},
            "id": "call_123",
            "metadata": {"source": "user"},
        }

        # Validate
        valid, invalid = service.validate_tool_calls(tool_call)

        # Verify original data preserved
        assert len(valid) == 1
        assert valid[0]["name"] == "code_search"
        assert valid[0]["arguments"]["query"] == "test"
        assert valid[0]["arguments"]["case_sensitive"] is True
        assert valid[0]["id"] == "call_123"
        assert valid[0]["metadata"]["source"] == "user"

    def test_private_validate_tool_call_method(self):
        """Test that _validate_tool_call is private and works correctly."""
        from victor.agent.services.tool_service import ToolService, ToolServiceConfig

        # Create service
        config = ToolServiceConfig()
        tool_selector = Mock()
        tool_executor = Mock()
        tool_registrar = Mock()
        service = ToolService(
            config=config,
            tool_selector=tool_selector,
            tool_executor=tool_executor,
            tool_registrar=tool_registrar,
        )

        # Mock get_available_tools
        service.get_available_tools = Mock(return_value={"code_search"})

        # Call private method directly
        is_valid, error = service._validate_tool_call(
            {"name": "code_search", "arguments": {"query": "test"}}
        )

        # Verify result
        assert is_valid is True
        assert error is None

        # Test invalid call
        is_valid, error = service._validate_tool_call({"name": "invalid_tool", "arguments": {}})

        # Verify result
        assert is_valid is False
        assert error is not None
        assert "not available" in error

    def test_validate_tool_calls_with_tool_field_fallback(self):
        """Test validation falls back to 'tool' field if 'name' not present."""
        from victor.agent.services.tool_service import ToolService, ToolServiceConfig

        # Create service
        config = ToolServiceConfig()
        tool_selector = Mock()
        tool_executor = Mock()
        tool_registrar = Mock()
        service = ToolService(
            config=config,
            tool_selector=tool_selector,
            tool_executor=tool_executor,
            tool_registrar=tool_registrar,
        )

        # Mock get_available_tools
        service.get_available_tools = Mock(return_value={"code_search"})

        # Validate with 'tool' field instead of 'name'
        valid, invalid = service.validate_tool_calls(
            {"tool": "code_search", "arguments": {"query": "test"}}
        )

        # Verify results - should use 'tool' field
        assert len(valid) == 1
        assert valid[0]["tool"] == "code_search"

    def test_validate_tool_calls_error_messages_are_descriptive(self):
        """Test that validation error messages are descriptive."""
        from victor.agent.services.tool_service import ToolService, ToolServiceConfig

        # Create service
        config = ToolServiceConfig()
        tool_selector = Mock()
        tool_executor = Mock()
        tool_registrar = Mock()
        service = ToolService(
            config=config,
            tool_selector=tool_selector,
            tool_executor=tool_executor,
            tool_registrar=tool_registrar,
        )

        # Mock get_available_tools
        service.get_available_tools = Mock(return_value={"code_search"})

        # Test various error cases
        _, invalid1 = service.validate_tool_calls("not_a_dict")
        assert "must be a dict or list" in invalid1[0]["_validation_error"]

        _, invalid2 = service.validate_tool_calls({"arguments": {}})
        assert "missing 'name' field" in invalid2[0]["_validation_error"]

        _, invalid3 = service.validate_tool_calls({"name": "missing_tool"})
        assert "not available" in invalid3[0]["_validation_error"]

        _, invalid4 = service.validate_tool_calls({"name": "code_search", "arguments": []})
        assert "must be a dictionary" in invalid4[0]["_validation_error"]

    def test_validate_tool_calls_reports_canonical_alias_and_mode_reason(self):
        """Alias-based tool errors should explain the canonical tool and mode limit."""
        from victor.agent.mode_controller import AgentMode, AgentModeController
        from victor.agent.services.tool_service import ToolService, ToolServiceConfig

        service = ToolService(
            config=ToolServiceConfig(),
            tool_selector=Mock(),
            tool_executor=Mock(),
            tool_registrar=Mock(),
        )
        service.get_available_tools = Mock(return_value={"read", "edit", "shell"})
        service._enabled_tools = {"read"}

        controller = AgentModeController()
        controller.switch_mode(AgentMode.PLAN)
        service.set_mode_controller(controller)

        _, invalid = service.validate_tool_calls({"name": "edit_file", "arguments": {}})

        assert "resolves to 'edit'" in invalid[0]["_validation_error"]
        assert "PLAN mode" in invalid[0]["_validation_error"]


class TestToolServiceValidationIntegration:
    """Integration tests for validation with other service methods."""

    def test_validate_and_filter_hallucinated_tools(self):
        """Test validation works with filter_hallucinated_tools."""
        from victor.agent.services.tool_service import ToolService, ToolServiceConfig

        # Create service
        config = ToolServiceConfig()
        tool_selector = Mock()
        tool_executor = Mock()
        tool_registrar = Mock()
        service = ToolService(
            config=config,
            tool_selector=tool_selector,
            tool_executor=tool_executor,
            tool_registrar=tool_registrar,
        )

        # Mock get_available_tools
        service.get_available_tools = Mock(return_value={"code_search", "file_read"})

        # Create tool calls with hallucinated tools
        tool_calls = [
            {"name": "code_search", "arguments": {"query": "test"}},
            {"name": "hallucinated_tool", "arguments": {}},
            {"name": "file_read", "arguments": {"path": "/tmp/test"}},
        ]

        # Validate
        valid, invalid = service.validate_tool_calls(tool_calls)

        # Filter hallucinated
        filtered = service.filter_hallucinated_tools(tool_calls)

        # Verify both methods work correctly
        assert len(valid) == 2
        assert len(invalid) == 1
        assert len(filtered) == 2
        assert invalid[0]["name"] == "hallucinated_tool"

    def test_validate_with_get_available_tools_integration(self):
        """Test validation integrates with get_available_tools."""
        from victor.agent.services.tool_service import ToolService, ToolServiceConfig

        # Create service with real registry mock
        config = ToolServiceConfig()
        tool_selector = Mock()
        tool_executor = Mock()
        tool_registrar = Mock()
        service = ToolService(
            config=config,
            tool_selector=tool_selector,
            tool_executor=tool_executor,
            tool_registrar=tool_registrar,
        )

        # Mock get_available_tools to return a specific set
        service.get_available_tools = Mock(return_value={"code_search", "file_read", "grep"})

        # Validate tool calls
        valid, invalid = service.validate_tool_calls(
            [
                {"name": "code_search", "arguments": {"query": "test"}},
                {"name": "file_read", "arguments": {"path": "/tmp/test"}},
                {"name": "invalid_tool", "arguments": {}},
            ]
        )

        # Verify validation results
        assert len(valid) == 2
        assert len(invalid) == 1


def test_all_tool_calls_are_channel_artifacts():
    """Harmony/role channel tokens leaked as tool names are recognized as artifacts."""
    from victor.agent.services.tool_service import ToolService

    svc = ToolService.__new__(ToolService)
    # gpt-oss harmony leak: the channel role token surfaces as the tool name.
    assert svc._all_tool_calls_are_channel_artifacts([{"name": "assistant", "arguments": {}}])
    assert svc._all_tool_calls_are_channel_artifacts([{"name": "analysis"}, {"name": "commentary"}])
    # A real tool name is never an artifact.
    assert not svc._all_tool_calls_are_channel_artifacts([{"name": "read", "arguments": {}}])
    # Mixed batch with at least one real tool must not be discarded wholesale.
    assert not svc._all_tool_calls_are_channel_artifacts([{"name": "assistant"}, {"name": "read"}])
    assert not svc._all_tool_calls_are_channel_artifacts([])
    assert not svc._all_tool_calls_are_channel_artifacts(None)
