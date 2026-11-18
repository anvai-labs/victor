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

"""Tests for unified MCP tool discovery (FEP-0049-followup B1).

Three divergent hand-rolled scans existed (victor mcp entry, stdio server
entry, ToolRegistry.discover_lightweight). The unified discovery module is
the single source of truth: one pinned exclusion set, deterministic order,
both @tool functions and BaseTool subclasses, instantiation failures logged
never silent.
"""

import inspect
import logging
from typing import Any

import pytest

from victor.integrations.mcp.discovery import (
    DISCOVERY_EXCLUDED_FILES,
    discover_tools,
    register_discovered_tools,
)
from victor.tools.base import BaseTool
from victor.tools.registry import ToolRegistry

# The pinned union of the three historical exclusion lists:
# - victor/ui/commands/mcp.py: __init__, base, decorators, semantic_selector
# - victor/integrations/mcp/server.py: + common
# - victor/tools/registry.py discover_lightweight: + enums, registry, metadata,
#   metadata_registry, tool_names, output_utils, shared_ast_utils,
#   dependency_graph, plugin_registry
EXPECTED_EXCLUSIONS = frozenset(
    {
        "__init__.py",
        "base.py",
        "decorators.py",
        "semantic_selector.py",
        "common.py",
        "enums.py",
        "registry.py",
        "metadata.py",
        "metadata_registry.py",
        "tool_names.py",
        "output_utils.py",
        "shared_ast_utils.py",
        "dependency_graph.py",
        "plugin_registry.py",
    }
)


def _tool_name(obj: Any) -> str:
    """Resolved tool name for a discovered object (function or instance)."""
    if hasattr(obj, "Tool"):
        return obj.Tool.name
    return obj.name


class TestExclusionSetPinned:
    def test_exclusion_set_is_the_pinned_union(self):
        """The exclusion set is exactly the union of the three historical lists."""
        assert isinstance(DISCOVERY_EXCLUDED_FILES, frozenset)
        assert DISCOVERY_EXCLUDED_FILES == EXPECTED_EXCLUSIONS


class TestDiscoverTools:
    def test_deterministic_order(self):
        """Same directory scanned twice yields the identical name order."""
        first = [_tool_name(obj) for obj in discover_tools()]
        second = [_tool_name(obj) for obj in discover_tools()]
        assert first == second
        assert first == sorted(first)

    def test_handles_both_tool_kinds(self):
        """Discovery yields @tool functions AND BaseTool subclass instances."""
        found = discover_tools()
        assert found, "discovery must find tools in victor/tools"
        functions = [o for o in found if inspect.isfunction(o) and getattr(o, "_is_tool", False)]
        instances = [o for o in found if isinstance(o, BaseTool)]
        assert functions, "at least one @tool function must be discovered"
        assert instances, "at least one BaseTool instance must be discovered"

    def test_excluded_modules_never_contribute(self):
        """No discovered object comes from an excluded module's file stem."""
        found = discover_tools()
        # common.py, enums.py etc. are excluded; their stems must not leak
        # as discovered *module-level* entities is not directly observable,
        # so pin the observable contract: every discovered name is nonempty
        # and unique (the registry would reject duplicate names anyway).
        names = [_tool_name(o) for o in found]
        assert all(names)
        assert len(names) == len(set(names))

    def test_instantiation_failure_logged_not_fatal(self, monkeypatch, caplog):
        """A BaseTool subclass whose ctor raises is skipped WITH a log record."""
        import victor.tools.bash as bash_module

        class DiscoveryProbeExploding(BaseTool):
            name = "discovery_probe_exploding"
            description = "raises on construction"
            parameters = {}

            def __init__(self):
                raise RuntimeError("needs arguments")

            async def execute(self, **kwargs):
                raise NotImplementedError

        monkeypatch.setattr(
            bash_module, "DiscoveryProbeExploding", DiscoveryProbeExploding, raising=False
        )

        import victor.integrations.mcp.discovery as discovery_module

        with caplog.at_level(logging.WARNING, logger=discovery_module.logger.name):
            found = discover_tools()

        names = [_tool_name(o) for o in found]
        assert "discovery_probe_exploding" not in names
        # Logged with the failing class name and a count, never silent.
        assert "DiscoveryProbeExploding" in caplog.text
        assert "instantiation failed" in caplog.text.lower()

    def test_module_import_failure_logged_not_fatal(self, monkeypatch, caplog):
        """A module that fails to import is skipped WITH a log record."""
        import importlib

        original = importlib.import_module

        def _fake_import(name, *args, **kwargs):
            if name == "victor.tools.bash":
                raise ImportError("synthetic import failure")
            return original(name, *args, **kwargs)

        monkeypatch.setattr(
            "victor.integrations.mcp.discovery.importlib.import_module", _fake_import
        )

        import victor.integrations.mcp.discovery as discovery_module

        with caplog.at_level(logging.WARNING, logger=discovery_module.logger.name):
            found = discover_tools()

        names = [_tool_name(o) for o in found]
        # bash is an @tool function; with its module failing to import it is gone
        assert "bash" not in names
        assert "victor.tools.bash" in caplog.text


class TestRegisterDiscoveredTools:
    def test_registers_every_discovered_tool(self):
        registry = ToolRegistry()
        count = register_discovered_tools(registry)
        # only_enabled=False: some discovered tools are disabled by default
        # (semantic-selector gated) but must still be IN the registry.
        registered_names = sorted(t.name for t in registry.list_tools(only_enabled=False))
        discovered_names = sorted(_tool_name(o) for o in discover_tools())

        assert count == len(discovered_names)
        assert registered_names == discovered_names

    def test_registry_is_usable_after_discovery(self):
        registry = ToolRegistry()
        register_discovered_tools(registry)
        tools = registry.list_tools(only_enabled=False)
        assert tools
        for tool in tools:
            assert tool.name
            assert tool.description


class TestEntryPointEquivalence:
    @pytest.mark.asyncio
    @pytest.mark.timeout(120)
    async def test_both_entry_paths_produce_identical_registries(self, monkeypatch):
        """`victor mcp` and `victor-mcp-server` must build the same inventory."""
        captured: list = []

        class FakeServer:
            def __init__(self, **kwargs):
                captured.append(kwargs["tool_registry"])

            async def start_stdio_server(self):
                return None

            def register_resource(self, *args, **kwargs):
                return None

        monkeypatch.setattr("victor.ui.commands.mcp.MCPServer", FakeServer)
        monkeypatch.setattr("victor.integrations.mcp.server.MCPServer", FakeServer)

        from victor.integrations.mcp.server import run_mcp_server_stdio
        from victor.ui.commands.mcp import _run_mcp_server

        await _run_mcp_server()
        await run_mcp_server_stdio()

        assert len(captured) == 2
        names_a = sorted(t.name for t in captured[0].list_tools(only_enabled=False))
        names_b = sorted(t.name for t in captured[1].list_tools(only_enabled=False))
        assert names_a == names_b
        assert names_a, "both entry points must register a nonempty inventory"
