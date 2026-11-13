"""Workflow process execution respects actual MCP text I/O and cleanup."""

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from victor.workflows.isolation import IsolationConfig
from victor.workflows.sandbox_executor import SandboxedExecutor


async def test_process_workflow_round_trips_text_through_real_backend():
    result = await SandboxedExecutor(docker_available=False)._execute_process(
        [sys.executable, "-c", "import sys; print(sys.stdin.read().upper())"],
        IsolationConfig(sandbox_type="process", network_allowed=True),
        working_dir=None,
        env=None,
        input_data="payload",
    )
    assert result.success, result.error
    assert result.output == "PAYLOAD\n"
    assert result.error == ""


async def test_process_workflow_cannot_pretend_network_denial_is_enforced():
    with patch("victor.integrations.mcp.sandbox.subprocess.Popen") as launch:
        result = await SandboxedExecutor(docker_available=False)._execute_process(
            ["ignored"],
            IsolationConfig(sandbox_type="process", network_allowed=False),
            working_dir=None,
            env=None,
            input_data=None,
        )
    assert not result.success
    assert "cannot enforce" in result.error
    launch.assert_not_called()


@pytest.mark.parametrize(
    "failure", [asyncio.TimeoutError(), RuntimeError("broken pipe"), asyncio.CancelledError()]
)
async def test_workflow_process_is_reaped_after_communication_failure(failure):
    process = MagicMock()
    backend = MagicMock()
    backend.start = AsyncMock(return_value=process)
    backend.communicate = AsyncMock(side_effect=failure)
    backend.terminate = AsyncMock()
    caller_env = {"CUSTOM": "value"}
    with patch("victor.integrations.mcp.sandbox.SandboxedProcess", return_value=backend):
        call = SandboxedExecutor(docker_available=False)._execute_process(
            ["ignored"],
            IsolationConfig(sandbox_type="process", network_allowed=True),
            working_dir=None,
            env=caller_env,
            input_data="text",
        )
        if isinstance(failure, asyncio.CancelledError):
            with pytest.raises(asyncio.CancelledError):
                await call
        else:
            result = await call
            assert not result.success
    backend.communicate.assert_awaited_once_with(process, "text")
    backend.terminate.assert_awaited_once_with(process)
    assert caller_env == {"CUSTOM": "value"}


# ---------------------------------------------------------------------------
# Required sandbox policy convergence (VAS-05i follow-up)
# ---------------------------------------------------------------------------


def _install_required(monkeypatch, backend_available: bool) -> None:
    """Point default settings at a required policy; arm spawn traps."""
    from types import SimpleNamespace

    from victor.config.sandbox_settings import SandboxSettings

    settings = SimpleNamespace(
        sandbox=SandboxSettings(sandbox_enabled=True, sandbox_policy="required")
    )
    monkeypatch.setattr("victor.config.settings.get_settings", lambda: settings)

    import victor.tools.sandbox.backends as sb_mod

    monkeypatch.setattr(sb_mod.platform, "system", lambda: "Linux")
    if backend_available:
        monkeypatch.setattr(sb_mod.shutil, "which", lambda name: "/usr/bin/bwrap")
    else:
        monkeypatch.setattr(sb_mod.shutil, "which", lambda name: None)

    import victor.workflows.sandbox_executor as se

    async def _no_spawn(*args, **kwargs):  # pragma: no cover - trap
        raise AssertionError("child process spawned despite required sandbox policy")

    monkeypatch.setattr(se.asyncio, "create_subprocess_exec", _no_spawn)


async def test_required_policy_unavailable_refuses_inline_with_zero_spawn(monkeypatch):
    """sandbox_type='none' (inline) must refuse under a required policy whose
    backend is unavailable — no child process, structured failure."""
    executor = SandboxedExecutor(docker_available=False)
    _install_required(monkeypatch, backend_available=False)
    result = await executor.execute(
        [sys.executable, "-c", "print('hi')"],
        IsolationConfig(sandbox_type="none"),
    )
    assert not result.success
    assert "required" in result.error.lower()
    assert result.exit_code == -1


async def test_required_policy_available_still_refuses_unenforced_inline(monkeypatch):
    """Even with an enforcing backend present, an explicit sandbox_type='none'
    request is unenforced execution and must refuse under required policy."""
    executor = SandboxedExecutor(docker_available=False)
    _install_required(monkeypatch, backend_available=True)
    result = await executor.execute(
        [sys.executable, "-c", "print('hi')"],
        IsolationConfig(sandbox_type="none"),
    )
    assert not result.success
    assert "sandbox policy is 'required'" in result.error.lower()


async def test_required_policy_blocks_docker_to_process_fallback(monkeypatch):
    """Under required policy, docker-unavailable must not silently fall back
    to the weaker process isolation."""
    executor = SandboxedExecutor(docker_available=False)
    _install_required(monkeypatch, backend_available=False)
    started = []

    class _FakeSandbox:
        def __init__(self, config):
            started.append(self)

        async def start(self, **kwargs):
            started.append("started")
            raise AssertionError("fallback spawn attempted under required policy")

    import victor.integrations.mcp.sandbox as mcp_sandbox

    monkeypatch.setattr(mcp_sandbox, "SandboxedProcess", _FakeSandbox)
    result = await executor.execute(
        [sys.executable, "-c", "print('hi')"],
        IsolationConfig(sandbox_type="docker"),
    )
    assert not result.success
    assert "sandbox policy is 'required'" in result.error.lower()
    assert not started


async def test_preferred_policy_keeps_existing_behavior(monkeypatch):
    """preferred (and legacy bool) keeps today's fail-open behavior: inline
    executes unsandboxed with a degradation warning path, no refusal."""
    from types import SimpleNamespace

    from victor.config.sandbox_settings import SandboxSettings

    settings = SimpleNamespace(
        sandbox=SandboxSettings(sandbox_enabled=True, sandbox_policy="preferred")
    )
    monkeypatch.setattr("victor.config.settings.get_settings", lambda: settings)
    executor = SandboxedExecutor(docker_available=False)
    result = await executor.execute(
        [sys.executable, "-c", "print('ok')"],
        IsolationConfig(sandbox_type="none"),
    )
    assert result.success, result.error
    assert "ok" in result.output


# ---------------------------------------------------------------------------
# HB-03: required policy must apply the enforcing backend at actual launch
# ---------------------------------------------------------------------------


def _install_required_settings(monkeypatch) -> None:
    from types import SimpleNamespace

    from victor.config.sandbox_settings import SandboxSettings

    settings = SimpleNamespace(
        sandbox=SandboxSettings(sandbox_enabled=True, sandbox_policy="required")
    )
    monkeypatch.setattr("victor.config.settings.get_settings", lambda: settings)


def _patch_backend_available(monkeypatch, available: bool) -> None:
    import victor.tools.sandbox.backends as sb_mod

    monkeypatch.setattr(sb_mod.platform, "system", lambda: "Linux")
    if available:
        monkeypatch.setattr(sb_mod.shutil, "which", lambda name: "/usr/bin/bwrap")
    else:
        monkeypatch.setattr(sb_mod.shutil, "which", lambda name: None)


def _patch_popen_trap(monkeypatch, launched: list) -> None:
    import victor.integrations.mcp.sandbox as mcp_sandbox

    def _trap(argv, *args, **kwargs):
        launched.append(list(argv))
        raise RuntimeError("synthetic spawn trap")

    monkeypatch.setattr(mcp_sandbox.subprocess, "Popen", _trap)


async def test_required_process_launch_wraps_argv_with_enforcing_backend(monkeypatch):
    """Under a required policy with an enforcing backend available, the
    workflow process branch must launch through the backend wrapper: the
    spawned argv carries the launcher, not the raw command."""
    _install_required_settings(monkeypatch)
    _patch_backend_available(monkeypatch, available=True)
    launched: list = []
    _patch_popen_trap(monkeypatch, launched)

    executor = SandboxedExecutor(docker_available=False)
    result = await executor.execute(
        ["synthetic-command", "--flag"],
        IsolationConfig(sandbox_type="process", network_allowed=True),
    )
    assert launched, "expected a spawn attempt to inspect"
    assert launched[0][0] == "bwrap", f"raw argv launched: {launched[0]}"
    assert "synthetic-command" in launched[0]
    assert not result.success


async def test_required_resolver_failure_refuses_workflow_process(monkeypatch):
    """A generic resolver exception under required policy refuses the
    workflow process branch; no launch is attempted."""
    import victor.tools.sandbox as vs

    _install_required_settings(monkeypatch)
    _patch_backend_available(monkeypatch, available=True)

    def _boom(settings):
        raise RuntimeError("synthetic resolver failure")

    monkeypatch.setattr(vs, "resolve_sandbox_backend", _boom, raising=False)
    launched: list = []
    _patch_popen_trap(monkeypatch, launched)

    executor = SandboxedExecutor(docker_available=False)
    result = await executor.execute(
        ["synthetic-command"],
        IsolationConfig(sandbox_type="process", network_allowed=True),
    )
    assert launched == [], "no spawn may be attempted on resolver failure"
    assert not result.success
    assert "required" in result.error.lower()


async def test_required_docker_qualified_on_own_capabilities(monkeypatch):
    """Docker under required policy is qualified by its own availability and
    configuration — the platform backend binary is neither necessary nor
    sufficient, so a missing bwrap must not refuse a docker branch."""
    _install_required_settings(monkeypatch)
    _patch_backend_available(monkeypatch, available=False)  # no bwrap

    launched: list = []

    import victor.workflows.sandbox_executor as se

    async def _docker_trap(program, *args, **kwargs):
        launched.append([program, *list(args)])
        raise RuntimeError("synthetic docker trap")

    monkeypatch.setattr(se.asyncio, "create_subprocess_exec", _docker_trap)
    # _execute_docker launches `docker ...` via asyncio.create_subprocess_exec.

    executor = SandboxedExecutor(docker_available=True)
    result = await executor.execute(
        ["synthetic-command"],
        IsolationConfig(sandbox_type="docker", network_allowed=False),
    )
    assert launched, "docker branch should have attempted container execution"
    assert launched[0] and launched[0][0] == "docker", launched[0]
    assert "sandbox policy is 'required'" not in result.error.lower()


# ---------------------------------------------------------------------------
# R2-02: global required restrictions compose as a floor across backends
# ---------------------------------------------------------------------------


def _install_global_sandbox(monkeypatch, policy: str, network_isolation: bool) -> None:
    from types import SimpleNamespace

    from victor.config.sandbox_settings import SandboxSettings

    settings = SimpleNamespace(
        sandbox=SandboxSettings(
            sandbox_enabled=True,
            sandbox_policy=policy,
            sandbox_network_isolation=network_isolation,
        )
    )
    monkeypatch.setattr("victor.config.settings.get_settings", lambda: settings)


def _trap_docker_launch(monkeypatch, launched: list) -> None:
    import victor.workflows.sandbox_executor as se

    async def _trap(program, *args, **kwargs):
        launched.append([str(program), *[str(a) for a in args]])
        raise RuntimeError("synthetic docker launch trap")

    monkeypatch.setattr(se.asyncio, "create_subprocess_exec", _trap)


async def test_required_docker_global_network_floor_beats_node_config(monkeypatch):
    """Global required network isolation is a floor: a workflow Docker
    isolation that allows networking must still launch with --network none."""
    _install_global_sandbox(monkeypatch, policy="required", network_isolation=True)
    launched: list = []
    _trap_docker_launch(monkeypatch, launched)

    executor = SandboxedExecutor(docker_available=True)
    await executor.execute(
        ["synthetic-command"],
        IsolationConfig(sandbox_type="docker", network_allowed=True),
    )
    assert launched, "expected a docker launch attempt to inspect"
    argv = launched[0]
    assert "--network" in argv and "none" in argv, f"loosened network config: {argv}"


async def test_preferred_docker_global_network_floor_applies(monkeypatch):
    """The floor applies whenever the global sandbox is active, not only
    under required policy."""
    _install_global_sandbox(monkeypatch, policy="preferred", network_isolation=True)
    launched: list = []
    _trap_docker_launch(monkeypatch, launched)

    executor = SandboxedExecutor(docker_available=True)
    await executor.execute(
        ["synthetic-command"],
        IsolationConfig(sandbox_type="docker", network_allowed=True),
    )
    assert launched
    argv = launched[0]
    assert "--network" in argv and "none" in argv, f"loosened network config: {argv}"


async def test_disabled_policy_keeps_node_network_config(monkeypatch):
    """No global sandbox configured: node network config is authoritative
    (historical behavior)."""
    from types import SimpleNamespace

    from victor.config.sandbox_settings import SandboxSettings

    settings = SimpleNamespace(sandbox=SandboxSettings(sandbox_enabled=False))
    monkeypatch.setattr("victor.config.settings.get_settings", lambda: settings)
    launched: list = []
    _trap_docker_launch(monkeypatch, launched)

    executor = SandboxedExecutor(docker_available=True)
    await executor.execute(
        ["synthetic-command"],
        IsolationConfig(sandbox_type="docker", network_allowed=True),
    )
    assert launched
    argv = launched[0]
    assert (
        "--network" not in argv
        or "none" not in argv[argv.index("--network") + 1 : argv.index("--network") + 2]
    )
