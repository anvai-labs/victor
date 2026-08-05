# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for McpStepHandler — vertical MCP dependency declaration."""

from unittest.mock import MagicMock
from typing import Dict, Any, List, Optional

import pytest


class FakeMcpProvider:
    """Fake vertical implementing McpProvider protocol."""

    @classmethod
    def get_mcp_servers(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "github": {
                "type": "stdio",
                "command": "npx",
                "args": ["github-mcp-server"],
            },
            "sqlite": {
                "type": "stdio",
                "command": "python",
                "args": ["-m", "sqlite_mcp"],
            },
        }

    @classmethod
    def get_mcp_tool_filters(cls) -> Optional[Dict[str, List[str]]]:
        return {"github": ["search", "create_issue"]}


class FakeMcpToolProvider(FakeMcpProvider):
    """Fake vertical implementing both McpProvider and McpToolProvider."""

    @classmethod
    def get_mcp_tool_overrides(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "github_search": {
                "description": "Search GitHub (coding-optimized)",
                "required_permission": "web_access",
            },
        }


class FakeNonMcpVertical:
    """Fake vertical that does NOT implement McpProvider."""

    pass


class FakeContext:
    """Minimal VerticalContext stub."""

    def __init__(self):
        self.capability_configs: Dict[str, Any] = {}

    def set_capability_config(self, name: str, config: Any) -> None:
        self.capability_configs[name] = config

    def get_capability_config(self, name: str, default: Any = None) -> Any:
        return self.capability_configs.get(name, default)


class FakeResult:
    """Minimal IntegrationResult stub."""

    def __init__(self):
        self.infos: list = []
        self.warnings: list = []

    def add_info(self, msg: str) -> None:
        self.infos.append(msg)

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)


class TestMcpStepHandler:
    def _make_handler(self):
        from victor.framework.step_handlers import McpStepHandler

        return McpStepHandler()

    def test_handler_name_and_order(self):
        handler = self._make_handler()
        assert handler.name == "mcp"
        assert handler.order == 12

    def test_applies_mcp_servers_from_provider(self):
        handler = self._make_handler()
        context = FakeContext()
        result = FakeResult()

        handler._do_apply(MagicMock(), FakeMcpProvider, context, result)

        servers = context.capability_configs.get("mcp_servers")
        assert servers is not None
        assert "github" in servers
        assert "sqlite" in servers
        assert servers["github"]["command"] == "npx"

    def test_applies_tool_filters(self):
        handler = self._make_handler()
        context = FakeContext()
        result = FakeResult()

        handler._do_apply(MagicMock(), FakeMcpProvider, context, result)

        filters = context.capability_configs.get("mcp_tool_filters")
        assert filters is not None
        assert filters["github"] == ["search", "create_issue"]

    def test_applies_tool_overrides_from_tool_provider(self):
        handler = self._make_handler()
        context = FakeContext()
        result = FakeResult()

        handler._do_apply(MagicMock(), FakeMcpToolProvider, context, result)

        overrides = context.capability_configs.get("mcp_tool_overrides")
        assert overrides is not None
        assert "github_search" in overrides
        assert overrides["github_search"]["required_permission"] == "web_access"

    def test_skips_non_mcp_vertical(self):
        handler = self._make_handler()
        context = FakeContext()
        result = FakeResult()

        handler._do_apply(MagicMock(), FakeNonMcpVertical, context, result)

        assert "mcp_servers" not in context.capability_configs
        assert len(result.infos) == 0

    def test_result_info_message(self):
        handler = self._make_handler()
        context = FakeContext()
        result = FakeResult()

        handler._do_apply(MagicMock(), FakeMcpProvider, context, result)

        assert len(result.infos) == 1
        assert "github" in result.infos[0]
        assert "sqlite" in result.infos[0]

    def test_empty_mcp_servers_skipped(self):
        class EmptyMcpProvider:
            @classmethod
            def get_mcp_servers(cls):
                return {}

            @classmethod
            def get_mcp_tool_filters(cls):
                return None

        handler = self._make_handler()
        context = FakeContext()
        result = FakeResult()

        handler._do_apply(MagicMock(), EmptyMcpProvider, context, result)

        assert "mcp_servers" not in context.capability_configs


class TestMcpStepHandlerInRegistry:
    def test_mcp_handler_registered_in_default(self):
        from victor.framework.step_handlers import StepHandlerRegistry

        registry = StepHandlerRegistry.default()
        handlers = registry.get_ordered_handlers()
        handler_names = [h.name for h in handlers]
        assert "mcp" in handler_names

    def test_mcp_handler_order_after_tools(self):
        from victor.framework.step_handlers import StepHandlerRegistry

        registry = StepHandlerRegistry.default()
        handlers = registry.get_ordered_handlers()
        handler_names = [h.name for h in handlers]
        tools_idx = handler_names.index("tools")
        mcp_idx = handler_names.index("mcp")
        assert mcp_idx > tools_idx
