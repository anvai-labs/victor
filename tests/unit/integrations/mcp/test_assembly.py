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

"""Transport-agnostic MCP server assembly tests (Phase 1 refactor).

The factory must produce identical state to the old inline wiring —
this is the parity guarantee that lets Phase 3 add an HTTP transport
without touching the protocol handlers.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from victor.integrations.mcp.assembly import create_mcp_server


@pytest.fixture
def workspace(tmp_path):
    return tmp_path


class TestFactoryParity:
    def test_factory_returns_assembly(self, workspace):
        assembly = create_mcp_server(workspace)
        assert assembly.server is not None
        assert assembly.tool_count > 0
        assert assembly.exec_context is not None or assembly.admission_refusal is not None

    def test_server_name_and_version(self, workspace):
        assembly = create_mcp_server(workspace)
        assert assembly.server.name == "Victor MCP Server"
        assert assembly.server.version == "1.0.0"

    def test_graph_resource_registered(self, workspace):
        assembly = create_mcp_server(workspace)
        uris = [r.uri for r in assembly.server.resources]
        assert "victor://graph/index-status" in uris

    def test_deterministic_tool_inventory(self, workspace):
        a = create_mcp_server(workspace)
        b = create_mcp_server(workspace)
        names_a = sorted(t.name for t in a.server.tool_registry.list_tools(only_enabled=False))
        names_b = sorted(t.name for t in b.server.tool_registry.list_tools(only_enabled=False))
        assert names_a == names_b


class TestGraphToolFlag:
    def test_no_graph_tools_disables_flag(self, workspace, monkeypatch):
        from victor.core.feature_flags import FeatureFlag, get_feature_flag_manager, reset_feature_flag_manager

        monkeypatch.setenv("VICTOR_USE_GRAPH_QUERY_TOOL", "true")
        reset_feature_flag_manager()
        create_mcp_server(workspace, no_graph_tools=True)
        mgr = get_feature_flag_manager()
        assert not get_feature_flag_manager().is_enabled(FeatureFlag.USE_GRAPH_QUERY_TOOL)
        reset_feature_flag_manager()

    def test_default_enables_graph_tools(self, workspace, monkeypatch):
        from victor.core.feature_flags import FeatureFlag, get_feature_flag_manager, reset_feature_flag_manager

        monkeypatch.delenv("VICTOR_USE_GRAPH_QUERY_TOOL", raising=False)
        reset_feature_flag_manager()
        create_mcp_server(workspace)
        # The flag is enabled by the factory (explicit MCP launch = consent)
        mgr = get_feature_flag_manager()
        assert mgr.is_enabled(FeatureFlag.USE_GRAPH_QUERY_TOOL)
        reset_feature_flag_manager()


class TestAdmission:
    def test_fail_closed_on_bad_settings(self, workspace, monkeypatch):
        import victor.config.settings as settings_mod

        monkeypatch.setattr(
            settings_mod, "get_settings",
            lambda: (_ for _ in ()).throw(RuntimeError("bad yaml")),
        )
        assembly = create_mcp_server(workspace)
        assert assembly.admission_refusal is not None
        assert "fail-closed" in assembly.admission_refusal
        assert assembly.exec_context is None
