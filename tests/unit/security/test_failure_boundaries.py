"""Failed guard evaluation must not grant access or report a complete scan."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from victor_contracts.safety.framework import SafetyConfig, SafetyEnforcer, SafetyRule
from victor.agent.factory.tool_builders import ToolBuildersMixin
from victor.framework.session_config import SessionConfig, ToolApprovalConfig
from victor.integrations.api.fastapi_server import CodeSearchRequest
from victor.integrations.api.routes import search_routes, workspace_routes
from victor.integrations.mcp.client import MCPClient
from victor.integrations.mcp.sandbox import SandboxConfig
from victor.tools.filesystem import read


def test_rule_failure_denies_operation_and_invalid_level_is_rejected():
    enforcer = SafetyEnforcer(SafetyConfig())
    rule = SafetyRule("broken", "failed check", MagicMock(side_effect=RuntimeError("failure")))
    enforcer.add_rule(rule)
    allowed, reason = enforcer.check_operation("read")
    assert not allowed and "could not be evaluated" in reason
    with pytest.raises(ValueError):
        SafetyRule("invalid", "invalid level", lambda op: True, level="typo")
    rule.check_fn = lambda op: False
    assert enforcer.check_operation("read") == (True, None)


async def test_read_does_not_open_file_when_workspace_guard_fails(tmp_path, monkeypatch):
    file = tmp_path / "outside.txt"
    file.write_text("private sentinel")
    monkeypatch.delenv("VICTOR_DISABLE_WORKSPACE_GUARD", raising=False)
    with patch("victor.config.settings.get_project_paths", side_effect=RuntimeError("failure")):
        result = await read(path=str(file))
    assert "private sentinel" not in str(result)
    assert "could not be verified" in str(result)


def test_permission_setup_failure_does_not_construct_pipeline():
    builder = ToolBuildersMixin()
    builder._settings = SimpleNamespace(
        permissions=SimpleNamespace(
            permission_mode="read-only", permission_tool_overrides={"shell": "typo"}
        )
    )
    with patch("victor.agent.tool_pipeline.ToolPipeline") as pipeline:
        with pytest.raises(RuntimeError, match="permission policy"):
            builder.create_tool_pipeline(
                MagicMock(), MagicMock(), 10, None, MagicMock(), lambda: None, lambda: None
            )
    pipeline.assert_not_called()


def test_requested_approval_failure_stops_session_configuration():
    settings = SimpleNamespace(governance=SimpleNamespace(enabled=False, ask_on_tools=[]))
    config = SessionConfig(tool_approval=ToolApprovalConfig(enabled=True))
    with patch("victor.core.feature_flags.enable_feature", side_effect=RuntimeError("failure")):
        with pytest.raises(RuntimeError, match="approval could not be enabled"):
            config.apply_to_settings(settings)
    with pytest.raises(RuntimeError, match="requires governance"):
        config.apply_to_settings(SimpleNamespace())


async def test_code_search_keeps_option_shaped_query_as_pattern(tmp_path):
    server = SimpleNamespace(workspace_root=str(tmp_path))
    endpoint = next(
        r.endpoint for r in search_routes.create_router(server).routes if r.path == "/search/code"
    )
    query = "--pre=unexpected-program"
    with patch("subprocess.run", return_value=SimpleNamespace(stdout="", returncode=1)) as run:
        await endpoint(CodeSearchRequest(query=query))
    argv = run.call_args.args[0]
    assert argv[-4:] == ["-e", query, "--", str(tmp_path)]


@pytest.mark.parametrize("failure", ["read", "oversize", "none"])
async def test_fallback_scan_reports_degradation_and_incomplete_files(tmp_path, failure):
    file = tmp_path / "test.py"
    file.write_text("x" * (1_000_001 if failure == "oversize" else 1))
    orchestrator = SimpleNamespace(execute_tool=AsyncMock(side_effect=RuntimeError("unavailable")))
    server = SimpleNamespace(
        workspace_root=str(tmp_path), _get_orchestrator=AsyncMock(return_value=orchestrator)
    )
    endpoint = next(
        r.endpoint
        for r in workspace_routes.create_router(server).routes
        if r.path == "/workspace/security"
    )
    if failure == "read":
        with patch.object(Path, "read_text", side_effect=OSError("unreadable")):
            response = await endpoint()
    else:
        response = await endpoint()
    data = json.loads(response.body)
    assert data["degraded"] is True and data["scanner"] == "fallback_patterns"
    assert data["scan_completed"] is False
    assert data["fallback_completed"] is (failure == "none")
    assert bool(data["errors"]) is (failure == "read")
    assert data["skipped_files"] == (1 if failure == "oversize" else 0)


@pytest.mark.parametrize("reconnect", [False, True])
@pytest.mark.parametrize("fails", [False, True])
async def test_mcp_keeps_sandbox_for_initial_and_reconnected_processes(reconnect, fails):
    client = MCPClient(sandbox_config=SandboxConfig(), health_check_interval=0, reconnect_delay=0)
    client._command = ["server"]
    wrapper = MagicMock()
    process = MagicMock()
    wrapper.start = AsyncMock(
        side_effect=RuntimeError("sandbox unavailable") if fails else None, return_value=process
    )
    wrapper.terminate_all = AsyncMock()
    client.initialize = AsyncMock(return_value=True)
    with (
        patch("victor.integrations.mcp.sandbox.SandboxedProcess", return_value=wrapper),
        patch("subprocess.Popen") as plain,
    ):
        try:
            ok = await client._try_reconnect() if reconnect else await client.connect(["server"])
            assert ok is not fails
            wrapper.start.assert_awaited_once_with(["server"])
            plain.assert_not_called()
            if fails:
                client.initialize.assert_not_awaited()
        finally:
            await client._cleanup_process_async()
    wrapper.terminate_all.assert_awaited_once()


@pytest.mark.requires_victor_coding
def test_coding_safety_provider_propagates_extension_failures():
    from victor_coding.protocols import CodingSafetyProvider

    with patch(
        "victor_coding.protocols.CodingSafetyExtension", side_effect=RuntimeError("init failed")
    ):
        with pytest.raises(RuntimeError, match="init failed"):
            CodingSafetyProvider()
    extension = MagicMock()
    with patch("victor_coding.protocols.CodingSafetyExtension", return_value=extension):
        provider = CodingSafetyProvider()
    for method in ("get_bash_patterns", "get_file_patterns", "get_tool_restrictions"):
        getattr(extension, method).side_effect = RuntimeError("lookup failed")
        with pytest.raises(RuntimeError, match="lookup failed"):
            getattr(provider, method)()


def test_init_does_not_synthesize_when_tool_restriction_fails():
    from victor.ui.commands.init import _run_agentic_synthesis

    registry = SimpleNamespace(disable_tool=MagicMock(side_effect=RuntimeError("cannot restrict")))
    agent = SimpleNamespace(tools=registry, close=AsyncMock())
    bootstrap = SimpleNamespace(
        provider_name="ollama", request_model="model", temperature=None, max_tokens=None
    )
    with (
        patch("victor.framework.agent.Agent.create", AsyncMock(return_value=agent)),
        patch("victor.framework.init_synthesizer.InitSynthesizer") as synth,
    ):
        synth._resolve_provider_bootstrap.return_value = bootstrap
        result = _run_agentic_synthesis(
            provider=None, model=None, graph_context={}, console_=MagicMock()
        )
    assert result == ""
    synth.assert_not_called()
    agent.close.assert_awaited_once()


async def test_kubernetes_detection_reports_unreadable_manifest(tmp_path):
    from victor.iac.scanners import KubernetesScanner

    manifest = tmp_path / "deployment.yaml"
    manifest.write_text("apiVersion: v1\nkind: Pod\n")
    scanner = KubernetesScanner()
    assert await scanner.detect_files(tmp_path) == [manifest]
    with patch.object(Path, "read_text", side_effect=OSError("unreadable")):
        with pytest.raises(ValueError, match="Cannot inspect"):
            await scanner.detect_files(tmp_path)
