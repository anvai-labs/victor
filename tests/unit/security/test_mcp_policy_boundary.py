"""MCP configured-server launches obey the canonical required-policy gate."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import victor.integrations.mcp.client as client_module
from victor.agent.mcp_connector import MCPConnector, MCPConnectorConfig
from victor.config.sandbox_settings import SandboxSettings
from victor.integrations.mcp.client import MCPClient
from victor.integrations.mcp.registry import MCPRegistry, MCPServerConfig
from victor.integrations.mcp.sandbox import SandboxConfig
from victor.runtime.policy_snapshot import RunPolicySnapshot, run_policy_context
from victor.tools.sandbox import backends


class FakeProcess:
    # A negative non-child PID makes owned_process_group reject the fake safely;
    # using a plausible live PID can accidentally adopt an unrelated process group.
    pid = -1

    def __init__(self) -> None:
        import io

        self.stdin = io.StringIO()
        self.stdout = io.StringIO()
        self.stderr = io.StringIO()

    def poll(self) -> int | None:
        return 0

    def wait(self, timeout: float | None = None) -> int:
        return 0


@pytest.fixture
def launched(monkeypatch):
    calls: list[list[str]] = []

    def _popen(command, **_kwargs):
        calls.append(list(command))
        return FakeProcess()

    monkeypatch.setattr(client_module.subprocess, "Popen", _popen)
    return calls


def _install_policy(
    monkeypatch: pytest.MonkeyPatch,
    *,
    policy: str | None,
    backend_available: bool,
) -> RunPolicySnapshot | None:
    sandbox = None
    if policy is not None:
        sandbox = SandboxSettings(sandbox_enabled=True, sandbox_policy=policy)
    settings = SimpleNamespace(sandbox=sandbox)
    monkeypatch.setattr("victor.config.settings.get_settings", lambda: settings)
    monkeypatch.setattr(backends.platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        backends.shutil,
        "which",
        lambda name: "/usr/bin/bwrap" if backend_available and name == "bwrap" else None,
    )
    return RunPolicySnapshot.from_settings(settings) if sandbox is not None else None


async def test_required_available_wraps_configured_mcp_command(monkeypatch, launched):
    _install_policy(monkeypatch, policy="required", backend_available=True)

    process, owner = await MCPClient()._start_process(["configured-mcp", "--stdio"])

    assert process is not None
    assert owner is None
    assert launched[0][0] == "bwrap"
    assert launched[0][-3:] == ["--", "configured-mcp", "--stdio"]


async def test_required_unavailable_refuses_before_process_launch(monkeypatch, launched):
    _install_policy(monkeypatch, policy="required", backend_available=False)

    with pytest.raises(client_module.MCPPolicyRefusal) as refusal:
        await MCPClient()._start_process(["configured-mcp"])

    assert refusal.value.reason == "sandbox_required"
    assert launched == []


async def test_unreadable_policy_refuses_before_process_launch(monkeypatch, launched):
    def _unavailable():
        raise RuntimeError("settings unavailable")

    monkeypatch.setattr("victor.config.settings.get_settings", _unavailable)

    with pytest.raises(client_module.MCPPolicyRefusal) as refusal:
        await MCPClient()._start_process(["configured-mcp"])

    assert refusal.value.reason == "policy_unavailable"
    assert launched == []


async def test_explicit_resource_owner_launch_is_wrapped_when_required(monkeypatch):
    _install_policy(monkeypatch, policy="required", backend_available=True)
    from victor.integrations.mcp.sandbox import SandboxedProcess

    captured: list[list[str]] = []

    async def _start(self, command, *_args, **_kwargs):
        captured.append(list(command))
        return FakeProcess()

    monkeypatch.setattr(SandboxedProcess, "start", _start)

    process, owner = await MCPClient(sandbox_config=SandboxConfig())._start_process(
        ["configured-mcp"]
    )

    assert process is not None
    assert owner is not None
    assert captured[0][0] == "bwrap"
    assert captured[0][-1] == "configured-mcp"


@pytest.mark.parametrize("policy", ["disabled", "preferred"])
async def test_non_required_policy_preserves_raw_command(monkeypatch, launched, policy):
    _install_policy(monkeypatch, policy=policy, backend_available=True)

    await MCPClient()._start_process(["configured-mcp"])

    assert launched == [["configured-mcp"]]


async def test_admitted_snapshot_survives_reload_and_reconnect(monkeypatch, launched):
    admitted = _install_policy(monkeypatch, policy="required", backend_available=True)
    assert admitted is not None
    client = MCPClient(run_policy_snapshot=admitted, reconnect_delay=0)
    client.initialize = AsyncMock(return_value=True)
    _install_policy(monkeypatch, policy="disabled", backend_available=True)
    client._consecutive_failures = 0
    client._command = ["configured-mcp"]

    assert await client._try_reconnect()
    assert launched[0][-1] == "configured-mcp"
    # Detach the fake transport directly: it has no real pipes, process group,
    # or cleanup thread, and exercising that production cleanup path is owned
    # by the existing lifecycle suite rather than this policy boundary.
    transport = client._transport
    assert transport is not None
    client.process = None
    client._transport = None
    client._running = False
    transport.retire()
    transport._close_streams()
    transport.cleanup_done.set()


async def test_admitted_disabled_snapshot_overrides_reload_to_required(monkeypatch, launched):
    admitted = _install_policy(monkeypatch, policy="disabled", backend_available=False)
    assert admitted is not None

    _install_policy(monkeypatch, policy="required", backend_available=False)
    await MCPClient(run_policy_snapshot=admitted)._start_process(["configured-mcp"])

    assert launched == [["configured-mcp"]]


async def test_registry_configured_server_cannot_bypass_policy(monkeypatch, launched):
    _install_policy(monkeypatch, policy="required", backend_available=True)

    async def _initialize(self) -> bool:
        return True

    monkeypatch.setattr(MCPClient, "initialize", _initialize)
    registry = MCPRegistry()
    registry.register_server(MCPServerConfig(name="configured", command=["configured-mcp"]))

    assert await registry.connect("configured")
    assert launched[0][0] == "bwrap"
    assert launched[0][-2:] == ["--", "configured-mcp"]
    client = registry._servers["configured"].client
    assert client is not None
    transport = client._transport
    assert transport is not None
    registry._servers["configured"].client = None
    client.process = None
    client._transport = None
    client._running = False
    transport.retire()
    transport._close_streams()
    transport.cleanup_done.set()


async def test_connector_binds_admitted_policy_to_registry(monkeypatch):
    admitted = _install_policy(monkeypatch, policy="required", backend_available=True)
    assert admitted is not None
    captured: list = []

    def _capture_task(coroutine, _name):
        captured.append(coroutine)
        coroutine.close()
        return None

    connector = MCPConnector(
        registry=MagicMock(),
        settings=SimpleNamespace(mcp_command="configured-mcp", use_mcp_tools=True),
        config=MCPConnectorConfig(enabled=True, auto_discover=False),
        task_callback=_capture_task,
    )

    with run_policy_context(admitted):
        result = await connector.connect()

    assert result.servers_discovered == 1
    assert connector._mcp_registry is not None
    assert connector._mcp_registry._run_policy_snapshot is admitted


async def test_failed_resource_owner_start_still_terminates_owner(monkeypatch):
    _install_policy(monkeypatch, policy="required", backend_available=True)
    import victor.integrations.mcp.sandbox as sandbox_module

    terminated: list[object] = []

    class RecordingOwner:
        def __init__(self, _config: object) -> None:
            pass

        async def start(self, *_args: object, **_kwargs: object) -> FakeProcess:
            raise RuntimeError("startup failed")

        async def terminate_all(self) -> None:
            terminated.append(self)

    monkeypatch.setattr(sandbox_module, "SandboxedProcess", RecordingOwner)

    async def _owner_only_cleanup(self) -> None:
        if self.owner is not None:
            await self.owner.terminate_all()

    monkeypatch.setattr(client_module.StdioTransport, "cleanup", _owner_only_cleanup)

    with pytest.raises(RuntimeError, match="startup failed"):
        await MCPClient(sandbox_config=SandboxConfig())._start_process(["configured-mcp"])

    await asyncio.sleep(0)
    assert len(terminated) == 1
