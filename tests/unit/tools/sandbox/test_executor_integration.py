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

"""Integration tests: sandbox wiring in subprocess_executor (no real isolation).

These use a recording fake backend that rewrites the command to a harmless
``echo`` so we verify the wrap path end-to-end without invoking bwrap/seatbelt.
"""

import os
from pathlib import Path
from typing import List, Optional

import pytest

from victor.tools.sandbox import NoneSandbox
from victor.tools.sandbox.backends import SandboxBackend
from victor.tools.subprocess_executor import (
    CommandErrorType,
    run_command,
    run_command_async,
)


class _RecordingBackend(SandboxBackend):
    """Records wrap_argv calls and rewrites the command to ``echo WRAPPED``."""

    type_name = "fake"

    def __init__(self) -> None:
        self.calls: List[tuple] = []

    def available(self) -> bool:
        return True

    def wrap_argv(self, argv: List[str], cwd: Optional[Path] = None) -> List[str]:
        self.calls.append((list(argv), cwd))
        return ["echo", "WRAPPED"]


# -- sync run_command -------------------------------------------------------


def test_run_command_wraps_shell_string_when_sandboxed():
    fake = _RecordingBackend()
    result = run_command("echo hi", shell=True, sandbox=fake)
    assert result.success
    assert result.stdout.strip() == "WRAPPED"
    # The original shell command was handed to the backend as `sh -c`.
    assert fake.calls[0][0] == ["/bin/sh", "-c", "echo hi"]


def test_run_command_wraps_list_args_when_sandboxed():
    fake = _RecordingBackend()
    result = run_command(["echo", "hi"], sandbox=fake)
    assert result.success
    assert result.stdout.strip() == "WRAPPED"
    assert fake.calls[0][0] == ["echo", "hi"]


def test_run_command_passthrough_with_none_sandbox():
    # Explicit NoneSandbox => no wrapping, original command runs.
    result = run_command("echo hi", shell=True, sandbox=NoneSandbox())
    assert result.success
    assert result.stdout.strip() == "hi"


def test_run_command_default_is_unsandboxed():
    # Default (sandbox=None) resolves from settings, which are off by default.
    result = run_command("echo hi", shell=True)
    assert result.success
    assert result.stdout.strip() == "hi"


# -- async run_command_async ------------------------------------------------


async def test_run_command_async_wraps_when_sandboxed():
    fake = _RecordingBackend()
    result = await run_command_async("echo hi", sandbox=fake)
    assert result.success
    assert result.stdout.strip() == "WRAPPED"
    assert fake.calls[0][0] == ["/bin/sh", "-c", "echo hi"]


async def test_run_command_async_passthrough_with_none_sandbox():
    result = await run_command_async("echo hi", sandbox=NoneSandbox())
    assert result.success
    assert result.stdout.strip() == "hi"


# -- guarded real-sandbox smoke test ----------------------------------------


@pytest.mark.skipif(
    os.environ.get("VICTOR_SANDBOX_E2E") != "1",
    reason="opt-in real sandbox e2e (set VICTOR_SANDBOX_E2E=1)",
)
async def test_e2e_real_platform_sandbox(tmp_path):
    from victor.tools.sandbox import resolve_sandbox_backend

    class _S:
        sandbox_enabled = True
        sandbox_filesystem_mode = "workspace-only"
        sandbox_namespace_restrictions = True
        sandbox_network_isolation = False
        sandbox_allowed_mounts: list = []

    backend = resolve_sandbox_backend(_S())
    if backend.type_name == "none":
        pytest.skip("no platform sandbox backend available")
    result = await run_command_async("echo hello", sandbox=backend, working_dir=str(tmp_path))
    assert result.success, result.stderr
    assert "hello" in result.stdout


# -- required-policy enforcement (fail closed, zero execution) ---------------


def _install_required_unavailable(monkeypatch):
    """Point default settings at a required policy whose backend is missing,
    and arm execution traps that fail the test if any child is spawned."""
    from types import SimpleNamespace

    from victor.config.sandbox_settings import SandboxSettings

    import subprocess

    settings = SimpleNamespace(
        sandbox=SandboxSettings(sandbox_enabled=True, sandbox_policy="required")
    )
    monkeypatch.setattr("victor.config.settings.get_settings", lambda: settings)
    import victor.tools.sandbox.backends as sb_mod

    monkeypatch.setattr(sb_mod.platform, "system", lambda: "Linux")
    monkeypatch.setattr(sb_mod.shutil, "which", lambda name: None)

    def _no_execution(*args, **kwargs):  # pragma: no cover - trap
        raise AssertionError("child process spawned despite required sandbox policy")

    monkeypatch.setattr(subprocess, "run", _no_execution)

    async def _no_execution_async(*args, **kwargs):  # pragma: no cover - trap
        raise AssertionError("child process spawned despite required sandbox policy")

    import victor.tools.subprocess_executor as se

    monkeypatch.setattr(se, "run_managed_process", _no_execution_async)


def test_run_command_required_policy_blocks_execution(monkeypatch):
    _install_required_unavailable(monkeypatch)
    result = run_command("echo hi", shell=True)
    assert result.success is False
    assert result.error_type == CommandErrorType.SANDBOX_REQUIRED
    assert result.stdout == "" and result.stderr == ""
    assert "required" in (result.error_message or "").lower()


async def test_run_command_async_required_policy_blocks_execution(monkeypatch):
    _install_required_unavailable(monkeypatch)
    result = await run_command_async("echo hi")
    assert result.success is False
    assert result.error_type == CommandErrorType.SANDBOX_REQUIRED
    assert result.stdout == "" and result.stderr == ""


def test_run_command_disabled_policy_still_executes(monkeypatch):
    # Legacy default: sandbox off, no policy — behavior unchanged.
    from types import SimpleNamespace

    from victor.config.sandbox_settings import SandboxSettings

    settings = SimpleNamespace(sandbox=SandboxSettings(sandbox_enabled=False))
    monkeypatch.setattr("victor.config.settings.get_settings", lambda: settings)
    result = run_command("echo hi", shell=True)
    assert result.success
    assert result.stdout.strip() == "hi"


def test_run_command_explicit_backend_bypasses_policy_resolution():
    # An explicit backend is the caller's explicit choice; resolution policy
    # applies only to the settings-driven default path.
    fake = _RecordingBackend()
    result = run_command("echo hi", shell=True, sandbox=fake)
    assert result.success


async def test_production_shell_tool_required_policy_blocks_execution(monkeypatch, tmp_path):
    """The production shell tool (victor.tools.bash.shell) resolves the same
    default sandbox; a required-unavailable policy must refuse it with zero
    process spawns and a structured failure."""
    _install_required_unavailable(monkeypatch)
    from victor.tools.bash import shell

    result = await shell("echo hi", cwd=str(tmp_path), action="exec")
    assert result["success"] is False
    assert "required" in result["error"].lower()
    assert result["stdout"] == "" and result["stderr"] == ""


# ---------------------------------------------------------------------------
# HB-02: required-policy resolver failures must refuse, never degrade
# ---------------------------------------------------------------------------


def _install_required_settings(monkeypatch) -> None:
    """Load a required policy without availability patching (bwrap presence
    on the host must not matter for resolver-failure tests)."""
    from types import SimpleNamespace

    from victor.config.sandbox_settings import SandboxSettings

    settings = SimpleNamespace(
        sandbox=SandboxSettings(sandbox_enabled=True, sandbox_policy="required")
    )
    monkeypatch.setattr("victor.config.settings.get_settings", lambda: settings)


def test_resolver_exception_under_required_raises(monkeypatch):
    """A generic resolver exception under a required policy must surface as a
    typed refusal, never silently degrade to NoneSandbox."""
    import victor.tools.sandbox.backends as sb_mod

    _install_required_settings(monkeypatch)
    monkeypatch.setattr(sb_mod, "resolve_sandbox_backend", _raise_runtime_error, raising=False)
    from victor.tools.subprocess_executor import _resolve_default_sandbox

    with pytest.raises(sb_mod.SandboxRequiredUnavailableError) as excinfo:
        _resolve_default_sandbox()
    assert excinfo.value.reason == "resolver_error"


def _raise_runtime_error(settings):  # pragma: no cover - helper
    raise RuntimeError("synthetic resolver failure")


def test_run_command_required_resolver_failure_refuses_execution(monkeypatch):
    """run_command under required policy with a broken resolver returns the
    structured refusal and never spawns (sync, shell path)."""
    import subprocess

    import victor.tools.sandbox.backends as sb_mod

    _install_required_settings(monkeypatch)
    monkeypatch.setattr(sb_mod, "resolve_sandbox_backend", _raise_runtime_error, raising=False)

    def _no_spawn(*args, **kwargs):  # pragma: no cover - trap
        raise AssertionError("child spawned despite required-policy resolver failure")

    monkeypatch.setattr(subprocess, "run", _no_spawn)
    result = run_command("echo hi", shell=True)
    assert result.success is False
    assert result.error_type == CommandErrorType.SANDBOX_REQUIRED
    assert "resolver" in result.error_message.lower()


async def test_run_command_async_required_resolver_failure_refuses_execution(
    monkeypatch,
):
    import victor.tools.sandbox.backends as sb_mod

    _install_required_settings(monkeypatch)
    monkeypatch.setattr(sb_mod, "resolve_sandbox_backend", _raise_runtime_error, raising=False)

    import victor.tools.subprocess_executor as se

    async def _no_spawn(*args, **kwargs):  # pragma: no cover - trap
        raise AssertionError("child spawned despite required-policy resolver failure")

    monkeypatch.setattr(se, "run_managed_process", _no_spawn)
    result = await run_command_async("echo hi")
    assert result.success is False
    assert result.error_type == CommandErrorType.SANDBOX_REQUIRED


def test_explicit_none_sandbox_cannot_weaken_required_policy(monkeypatch):
    """A caller-supplied NoneSandbox must not bypass a required policy: the
    execution is refused instead of running unwrapped."""
    import subprocess

    _install_required_settings(monkeypatch)
    import victor.tools.sandbox.backends as sb_mod

    monkeypatch.setattr(sb_mod.platform, "system", lambda: "Linux")
    monkeypatch.setattr(sb_mod.shutil, "which", lambda name: "/usr/bin/bwrap")

    def _no_spawn(*args, **kwargs):  # pragma: no cover - trap
        raise AssertionError("unwrapped child spawned under required policy")

    monkeypatch.setattr(subprocess, "run", _no_spawn)
    result = run_command("echo hi", shell=True, sandbox=NoneSandbox())
    assert result.success is False
    assert result.error_type == CommandErrorType.SANDBOX_REQUIRED


def test_policy_unreadable_degrades_open_only_without_required(monkeypatch):
    """If settings cannot be loaded the policy is unknowable — historical
    degradation applies (documented; distinct from a known-required
    resolution failure)."""
    import victor.config.settings as cs
    import victor.tools.subprocess_executor as se

    def _boom():
        raise RuntimeError("settings unavailable")

    monkeypatch.setattr(cs, "get_settings", _boom)
    assert se._resolve_default_sandbox().type_name == "none"
