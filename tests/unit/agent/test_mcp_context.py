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

"""Tests for the sandboxed MCP execution context (FEP-0049-followup B2)
and the reachable codegraph wiring (B3: flag + index-status resource).

MCP tool calls historically ran with NO sandbox admission (fresh settings
read at spawn, fail-open). Admission now happens at server LAUNCH via
``admit_mcp_execution_context``; the admitted snapshot flows into every
tools/call and an unenforceable required policy refuses execution
(fail-closed) while the server itself still starts.
"""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from victor.integrations.mcp.context import (
    MCPAdmissionError,
    MCPToolContext,
    SANDBOX_ESCAPE_HATCH,
    admit_mcp_execution_context,
)
from victor.integrations.mcp.protocol import MCPMessageType
from victor.integrations.mcp.server import (
    MCPServer,
    register_graph_index_resource,
)
from victor.tools.base import ToolResult
from victor.tools.registry import ToolRegistry
from victor.tools.sandbox.backends import RequiredLaunchDecision


class MockTool:
    """Minimal registered tool used to observe exec-context flow."""

    name = "mock_tool"
    description = "A mock tool for testing"
    parameters = {}

    async def execute(self, **kwargs) -> ToolResult:
        return ToolResult(success=True, output="executed")


def _required_snapshot():
    from victor.config.sandbox_settings import SandboxSettings
    from victor.runtime.policy_snapshot import RunPolicySnapshot

    return RunPolicySnapshot.from_settings(
        SimpleNamespace(sandbox=SandboxSettings(sandbox_policy="required"))
    )


def _install_reloaded_disabled_settings(monkeypatch):
    import victor.config.settings as settings_module
    from victor.config.sandbox_settings import SandboxSettings

    monkeypatch.setattr(
        settings_module,
        "get_settings",
        lambda: SimpleNamespace(sandbox=SandboxSettings(sandbox_policy="disabled")),
    )


def _make_server(exec_context=None, admission_refusal=None):
    registry = ToolRegistry()
    registry.register(MockTool())
    server = MCPServer(
        tool_registry=registry,
        exec_context=exec_context,
        admission_refusal=admission_refusal,
    )
    server.initialized = True
    return server, registry


def _call_tool_message(name="mock_tool", arguments=None):
    return {
        "jsonrpc": "2.0",
        "id": "1",
        "method": MCPMessageType.CALL_TOOL.value,
        "params": {"name": name, "arguments": arguments or {}},
    }


@pytest.fixture(autouse=True)
def _reset_feature_flag_state():
    """Flag wiring tests mutate the process-global manager; restore it."""
    from victor.core.feature_flags import FeatureFlag, get_feature_flag_manager

    yield
    manager = get_feature_flag_manager()
    manager.clear_runtime_override(FeatureFlag.USE_GRAPH_QUERY_TOOL)


class TestAdmissionSuccess:
    def test_success_shape(self, tmp_path):
        ctx = admit_mcp_execution_context(tmp_path)

        assert ctx.workspace_root == tmp_path
        assert ctx.run_policy_snapshot is not None
        assert ctx.run_policy_snapshot.effective_policy in {
            "disabled",
            "preferred",
            "required",
        }
        # Non-required policy never refuses.
        assert ctx.backend_decision.refuse is False
        # The same settings read the admission used is retained for tools.
        assert ctx.settings is not None

    def test_context_is_frozen(self, tmp_path):
        ctx = admit_mcp_execution_context(tmp_path)
        with pytest.raises(Exception):
            ctx.workspace_root = Path("/other")  # type: ignore[misc]


class TestAdmissionFailure:
    def test_settings_read_failure_raises_typed_error(self, monkeypatch, tmp_path):
        """from_settings failure -> MCPAdmissionError carrying the escape hatch.

        (An invalid policy string is IGNORED by the ADR-035 derivation and
        resolves fail-closed; the raising vector is a mistyped field.)
        """
        import victor.config.settings as settings_module

        monkeypatch.setattr(
            settings_module,
            "get_settings",
            lambda: SimpleNamespace(sandbox=SimpleNamespace(sandbox_enabled="not-a-bool")),
        )
        with pytest.raises(MCPAdmissionError) as excinfo:
            admit_mcp_execution_context(tmp_path)
        assert "VICTOR_SANDBOX_POLICY" in str(excinfo.value)

    def test_get_settings_raise_raises_typed_error(self, monkeypatch, tmp_path):
        import victor.config.settings as settings_module

        def _boom():
            raise RuntimeError("settings store unavailable")

        monkeypatch.setattr(settings_module, "get_settings", _boom)
        with pytest.raises(MCPAdmissionError):
            admit_mcp_execution_context(tmp_path)

    def test_never_degrades_open_on_error(self, monkeypatch, tmp_path):
        """Admission NEVER returns a context whose snapshot claims disabled
        on a settings failure — it raises instead (fail-closed)."""
        import victor.config.settings as settings_module

        monkeypatch.setattr(
            settings_module,
            "get_settings",
            lambda: SimpleNamespace(sandbox=SimpleNamespace(sandbox_enabled="not-a-bool")),
        )
        with pytest.raises(MCPAdmissionError):
            admit_mcp_execution_context(tmp_path)


class TestAdmittedSnapshotBeatsReloadedSettings:
    def test_admitted_snapshot_beats_reloaded_disabled(self, monkeypatch, tmp_path):
        """Mirror of tests/unit/tools/sandbox/test_executor_integration.py::test_run_command_uses_admitted_snapshot_over_reloaded_disabled:
        the contextvar-admitted snapshot wins over a fresh (disabled) settings read."""
        from victor.runtime.policy_snapshot import run_policy_context

        _install_reloaded_disabled_settings(monkeypatch)
        required = _required_snapshot()

        with run_policy_context(required):
            ctx = admit_mcp_execution_context(tmp_path)

        assert ctx.run_policy_snapshot is required
        assert ctx.run_policy_snapshot.effective_policy == "required"

    def test_required_with_backend_resolves_active_decision(self, monkeypatch, tmp_path):
        from victor.runtime.policy_snapshot import run_policy_context

        _install_reloaded_disabled_settings(monkeypatch)
        required = _required_snapshot()

        class _FakeBackend:
            def type_name(self):
                return "fake"

            def available(self):
                return True

        monkeypatch.setattr(
            "victor.tools.sandbox.resolve_required_launch",
            lambda settings: RequiredLaunchDecision(required_active=True, backend=_FakeBackend()),
        )
        with run_policy_context(required):
            ctx = admit_mcp_execution_context(tmp_path)

        assert ctx.backend_decision.required_active is True
        assert ctx.backend_decision.refuse is False


class TestDangerouslyUnsandboxed:
    @pytest.mark.asyncio
    async def test_settings_mutated_pre_admission_executes(self, monkeypatch, tmp_path):
        """--dangerously-unsandboxed mutates settings BEFORE admission, so the
        admitted policy is disabled and tool calls execute (no refusal)."""
        from unittest.mock import MagicMock

        from victor.config.sandbox_settings import SandboxSettings

        settings = SimpleNamespace(
            sandbox=SandboxSettings(sandbox_policy="required"),
            tool_settings=MagicMock(),
        )
        import victor.config.settings as settings_module

        monkeypatch.setattr(settings_module, "get_settings", lambda: settings)
        # This is what force_unsandboxed_for_session() does to settings.
        settings.sandbox.sandbox_policy = "disabled"

        ctx = admit_mcp_execution_context(tmp_path)
        assert ctx.run_policy_snapshot.effective_policy == "disabled"
        assert ctx.backend_decision.refuse is False

        server, registry = _make_server(exec_context=ctx)
        with patch.object(
            registry, "execute", return_value=ToolResult(success=True, output="ok")
        ) as mock_exec:
            response = await server.handle_message(_call_tool_message())
        assert response["result"]["isError"] is False
        assert mock_exec.call_count == 1


class TestExecContextInjection:
    @pytest.mark.asyncio
    async def test_no_context_builds_empty_exec_ctx(self):
        """Backward compat: no exec_context -> exactly the historical {} ctx."""
        server, registry = _make_server()
        with patch.object(
            registry, "execute", return_value=ToolResult(success=True, output="ok")
        ) as mock_exec:
            response = await server.handle_message(_call_tool_message())
        assert response["result"]["isError"] is False
        assert mock_exec.call_args.args[1] == {}

    @pytest.mark.asyncio
    async def test_context_flows_into_execution(self, tmp_path):
        snapshot = _required_snapshot()
        sentinel_settings = SimpleNamespace(sandbox="sentinel")
        ctx = MCPToolContext(
            run_policy_snapshot=snapshot,
            workspace_root=tmp_path,
            backend_decision=RequiredLaunchDecision(required_active=True),
            settings=sentinel_settings,
        )
        server, registry = _make_server(exec_context=ctx)
        with patch.object(
            registry, "execute", return_value=ToolResult(success=True, output="ok")
        ) as mock_exec:
            await server.handle_message(_call_tool_message())

        exec_ctx = mock_exec.call_args.args[1]
        assert exec_ctx["run_policy_snapshot"] is snapshot
        assert exec_ctx["workspace_root"] == str(tmp_path)
        assert exec_ctx["settings"] is sentinel_settings


class TestRequiredRefusal:
    @pytest.mark.asyncio
    async def test_unenforceable_required_refuses_without_execution(self):
        """required policy + refusing backend decision -> isError result with
        escape hatch; registry.execute is NEVER called."""
        ctx = MCPToolContext(
            run_policy_snapshot=_required_snapshot(),
            workspace_root=Path("/tmp"),
            backend_decision=RequiredLaunchDecision(
                refuse=True, reason="sandbox policy is required but unenforceable"
            ),
            settings=SimpleNamespace(),
        )
        server, registry = _make_server(exec_context=ctx)
        with patch.object(registry, "execute", side_effect=AssertionError("must never execute")):
            response = await server.handle_message(_call_tool_message())

        assert "result" in response
        assert response["result"]["isError"] is True
        text = response["result"]["content"][0]["text"]
        assert "refused" in text.lower()
        assert "VICTOR_SANDBOX_POLICY" in text

    @pytest.mark.asyncio
    async def test_launch_admission_failure_refuses_every_call(self, capsys):
        """Launch-time admission failure: server starts, every tools/call is
        an isError refusal with the escape hatch, counted onto stderr."""
        refusal = (
            "Sandbox policy could not be established; refusing MCP tool "
            f"execution (fail-closed). {SANDBOX_ESCAPE_HATCH}"
        )
        server, registry = _make_server(admission_refusal=refusal)

        for i in range(2):
            with patch.object(
                registry, "execute", side_effect=AssertionError("must never execute")
            ):
                response = await server.handle_message(_call_tool_message())
            assert response["result"]["isError"] is True
            assert "VICTOR_SANDBOX_POLICY" in response["result"]["content"][0]["text"]

        captured = capsys.readouterr()
        assert "2" in captured.err
        assert "refused" in captured.err.lower()


class TestGraphToolFlagWiring:
    def _flag(self):
        from victor.core.feature_flags import FeatureFlag

        return FeatureFlag.USE_GRAPH_QUERY_TOOL

    def _env_var(self):
        return self._flag().get_env_var_name()

    def test_launch_enable_beats_env_false(self, monkeypatch):
        """Runtime override is TOP precedence: `victor mcp` consent wins even
        over VICTOR_USE_GRAPH_QUERY_TOOL=false (documented: runtime > env)."""
        from victor.core.feature_flags import get_feature_flag_manager
        from victor.integrations.mcp.server import apply_graph_tool_flag

        monkeypatch.setenv(self._env_var(), "false")
        apply_graph_tool_flag(no_graph_tools=False)
        assert get_feature_flag_manager().is_enabled(self._flag()) is True

        # External control resumes once the runtime override is cleared.
        get_feature_flag_manager().clear_runtime_override(self._flag())
        assert get_feature_flag_manager().is_enabled(self._flag()) is False

    def test_no_graph_tools_disables_even_over_env_true(self, monkeypatch):
        """--no-graph-tools must call disable_feature: the runtime override is
        the ONLY mechanism that beats env=true."""
        from victor.core.feature_flags import get_feature_flag_manager
        from victor.integrations.mcp.server import apply_graph_tool_flag

        monkeypatch.setenv(self._env_var(), "true")
        apply_graph_tool_flag(no_graph_tools=True)
        assert get_feature_flag_manager().is_enabled(self._flag()) is False


class TestGraphIndexResource:
    @pytest.mark.asyncio
    async def test_resource_is_listed_and_readable(self, tmp_path):
        server = MCPServer()
        server.initialized = True
        register_graph_index_resource(server, project_path=tmp_path)

        listed = await server.handle_message(
            {
                "jsonrpc": "2.0",
                "id": "1",
                "method": MCPMessageType.LIST_RESOURCES.value,
            }
        )
        assert listed["result"]["resources"][0]["uri"] == "victor://graph/index-status"
        assert listed["result"]["resources"][0]["name"] == "Codegraph index status"

        response = await server.handle_message(
            {
                "jsonrpc": "2.0",
                "id": "2",
                "method": MCPMessageType.READ_RESOURCE.value,
                "params": {"uri": "victor://graph/index-status"},
            }
        )
        assert "result" in response, response
        payload = json.loads(response["result"]["content"])
        assert {"store_path", "nodes", "files_tracked"} <= set(payload.keys())

    @pytest.mark.asyncio
    async def test_store_failure_reports_unavailable(self, tmp_path, monkeypatch):
        def _boom(**kwargs):
            raise RuntimeError("store unavailable")

        monkeypatch.setattr("victor.storage.graph.create_graph_store", _boom)
        server = MCPServer()
        server.initialized = True
        register_graph_index_resource(server, project_path=tmp_path)

        response = await server.handle_message(
            {
                "jsonrpc": "2.0",
                "id": "3",
                "method": MCPMessageType.READ_RESOURCE.value,
                "params": {"uri": "victor://graph/index-status"},
            }
        )
        assert "result" in response, response
        payload = json.loads(response["result"]["content"])
        assert payload["status"] == "unavailable"

    @pytest.mark.asyncio
    async def test_file_resources_still_work(self, tmp_path):
        """The read-handler extension must not break file:// resources."""
        target = tmp_path / "note.txt"
        target.write_text("hello")
        server = MCPServer()
        server.initialized = True
        from victor.integrations.mcp.protocol import MCPResource

        server.register_resource(
            MCPResource(uri=f"file://{target}", name="note", description="file")
        )
        response = await server.handle_message(
            {
                "jsonrpc": "2.0",
                "id": "4",
                "method": MCPMessageType.READ_RESOURCE.value,
                "params": {"uri": f"file://{target}"},
            }
        )
        assert response["result"]["content"] == "hello"


class TestEmptyIndexBoundedLatency:
    @pytest.mark.asyncio
    async def test_graph_query_on_empty_index_returns_not_builds(self, tmp_path):
        """Bounded latency: an empty index yields an actionable result dict —
        the MCP resource/tool path never triggers an index build."""
        from victor.core.feature_flags import FeatureFlag, get_feature_flag_manager
        from victor.integrations.mcp.server import apply_graph_tool_flag
        from victor.tools.graph_query_tool import graph_semantic_search

        apply_graph_tool_flag(no_graph_tools=False)
        assert get_feature_flag_manager().is_enabled(FeatureFlag.USE_GRAPH_QUERY_TOOL)

        result = await graph_semantic_search(
            query="authentication flow", path=str(tmp_path), mode="semantic", max_results=3
        )
        assert isinstance(result, dict)
        assert "results" in result
        assert result["results"] == []
