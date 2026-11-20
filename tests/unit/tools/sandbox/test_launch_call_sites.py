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

"""Every wrap-then-spawn site launches through the shared seam.

The Codex finding-8 residual: launchers outside the canonical executor
wrapped the backend argv WITHOUT forwarding the backend-owned descriptors
(bwrap's seccomp filter fd) — under the fail-closed required posture each
ran bwrap without its filter and leaked one parent descriptor per launch.
Each site now goes through :func:`prepare_launch` +
:func:`run_launch` / :func:`run_launch_async`; these tests pin the
descriptor forwarding per site, the raw-argv parity when no enforcing
backend applies, the descriptor drain, parent fd-table stability across
repeated launches, and the structural invariant that no call site outside
the sandbox seam touches ``wrap_argv`` at all.
"""

from __future__ import annotations

import asyncio
import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import victor.tools.sandbox as sandbox_pkg
import victor.tools.subprocess_executor as subprocess_executor_mod
from victor.agent.auto_commit import AutoCommitter
from victor.agent.snapshot_store import FileSnapshotStore
from victor.integrations.mcp import client as mcp_client_mod
from victor.integrations.mcp.client import MCPClient
from victor.processing.merge.manager import MergeManager
from victor.tools.bash import shell
from victor.tools.filesystem import _discover_read_roots
from victor.tools.lsp_write_enhancer import LSPWriteEnhancer
from victor.tools.sandbox import NoneSandbox, RequiredLaunchDecision
from victor.tools.sandbox import launch as launch_mod
from victor.tools.sandbox.backends import BubblewrapSandbox
from victor.tools.verification.temporal_analyzer import GitHistoryAnalyzer
from victor.ui.slash.commands import navigation as navigation_mod
from victor.workflows.isolation import IsolationConfig
from victor.workflows.sandbox_executor import SandboxedExecutor


def _bwrap_available() -> bool:
    return platform.system() == "Linux" and shutil.which("bwrap") is not None


requires_bwrap = pytest.mark.skipif(
    not _bwrap_available(), reason="bwrap not available on this host"
)


class _PipeBackend(NoneSandbox):
    """Backend that opens one pipe per wrap (the bwrap seccomp-fd shape).

    ``wrap_argv`` prefixes a NON-executable token so the spawn itself fails
    loudly (FileNotFoundError) — exactly the negative-control shape of
    ``test_launch_helper.py`` — and leaves the read end pending until
    ``take_pass_fds`` drains it.
    """

    type_name = "pipe"

    def __init__(self) -> None:
        self._pending: List[int] = []
        self.wrapped: List[List[str]] = []

    def wrap_argv(self, argv, cwd=None):  # type: ignore[override]
        r, w = os.pipe()
        os.close(w)
        self._pending.append(r)
        wrapped = ["wrapped-victor-probe"] + list(argv)
        self.wrapped.append(wrapped)
        return wrapped

    def take_pass_fds(self):
        fds, self._pending = self._pending, []
        return fds


class _PipePassthroughBackend(_PipeBackend):
    """Same descriptor ownership, but the argv stays executable."""

    def wrap_argv(self, argv, cwd=None):  # type: ignore[override]
        r, w = os.pipe()
        os.close(w)
        self._pending.append(r)
        return list(argv)


def _required(backend: Any) -> RequiredLaunchDecision:
    return RequiredLaunchDecision(required_active=True, backend=backend)


def _not_required() -> RequiredLaunchDecision:
    return RequiredLaunchDecision()


def _patch_required_launch(monkeypatch: pytest.MonkeyPatch, decision: RequiredLaunchDecision):
    """Sites import ``resolve_required_launch`` at call time from the package."""
    monkeypatch.setattr(sandbox_pkg, "resolve_required_launch", lambda _settings: decision)


def _capture_subprocess_run(monkeypatch: pytest.MonkeyPatch) -> List[Dict[str, Any]]:
    """Record ``(argv, pass_fds)`` of every subprocess.run through the seam."""
    calls: List[Dict[str, Any]] = []
    real_run = subprocess.run

    def fake_run(argv, **kwargs):
        calls.append({"argv": list(argv), "pass_fds": kwargs.get("pass_fds", ())})
        return real_run(argv, **kwargs)

    monkeypatch.setattr(subprocess, "run", fake_run)
    return calls


def _assert_closed(fds) -> None:
    for fd in fds:
        with pytest.raises(OSError):
            os.fstat(fd)


class _SnapshotStub:
    """Minimal admitted-policy stand-in for site constructors."""

    def to_sandbox_settings(self) -> Any:
        return object()


# ── site 1: bash tool ────────────────────────────────────────────────────────


async def test_bash_tool_wraps_and_forwards_pass_fds(monkeypatch):
    backend = _PipeBackend()
    monkeypatch.setattr(subprocess_executor_mod, "_resolve_default_sandbox", lambda **_: backend)
    captured: Dict[str, Any] = {}

    async def fake_managed_process(**kwargs):
        captured["argv"] = kwargs.get("argv")
        captured["pass_fds"] = kwargs.get("pass_fds")
        return b"", b"", 0, False, False

    with patch(
        "victor.tools.subprocess_executor.run_managed_process",
        new=AsyncMock(side_effect=fake_managed_process),
    ):
        result = await shell(cmd="echo hi", readonly=False)

    assert result["success"] is True
    assert captured["argv"] == ["wrapped-victor-probe", "/bin/sh", "-c", "echo hi"]
    assert captured["pass_fds"], "the seccomp filter descriptor must reach the spawn"
    _assert_closed(captured["pass_fds"])
    assert backend.take_pass_fds() == []


async def test_bash_tool_without_enforcing_backend_runs_raw_shell(monkeypatch):
    monkeypatch.setattr(
        subprocess_executor_mod, "_resolve_default_sandbox", lambda **_: NoneSandbox()
    )
    captured: Dict[str, Any] = {}

    async def fake_managed_process(**kwargs):
        captured["command"] = kwargs.get("command")
        captured["argv"] = kwargs.get("argv")
        captured["pass_fds"] = kwargs.get("pass_fds")
        return b"hi\n", b"", 0, False, False

    with patch(
        "victor.tools.subprocess_executor.run_managed_process",
        new=AsyncMock(side_effect=fake_managed_process),
    ):
        result = await shell(cmd="echo hi", readonly=False)

    assert result["success"] is True
    assert captured["command"] == "echo hi" and captured["argv"] is None
    assert captured["pass_fds"] == ()


# ── site 2: workflow sandbox executor (process branch) ───────────────────────


def _fake_sandboxed_process(monkeypatch, started: Dict[str, Any]) -> None:
    import victor.integrations.mcp.sandbox as mcp_sandbox_mod

    process = MagicMock(returncode=0)

    class _FakeOwner:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        async def start(self, command, cwd=None, env=None, pass_fds=()):
            started["command"] = list(command)
            started["pass_fds"] = pass_fds
            return process

        async def communicate(self, _process, _input):
            return "", ""

        async def terminate(self, _process):
            return None

    monkeypatch.setattr(mcp_sandbox_mod, "SandboxedProcess", _FakeOwner)


async def test_workflow_process_launch_wraps_and_forwards_pass_fds(monkeypatch):
    backend = _PipePassthroughBackend()
    started: Dict[str, Any] = {}
    _fake_sandboxed_process(monkeypatch, started)

    executor = SandboxedExecutor()
    result = await executor._execute_process(
        command=["/bin/echo", "hi"],
        isolation=IsolationConfig(sandbox_type="process"),
        working_dir=None,
        env=None,
        input_data=None,
        backend=backend,
    )

    assert result.success is True
    assert started["command"] == ["/bin/echo", "hi"]
    assert started["pass_fds"], "the seccomp filter descriptor must reach the spawn"
    _assert_closed(started["pass_fds"])
    assert backend.take_pass_fds() == []


async def test_workflow_process_launch_without_backend_keeps_raw_argv(monkeypatch):
    started: Dict[str, Any] = {}
    _fake_sandboxed_process(monkeypatch, started)

    executor = SandboxedExecutor()
    result = await executor._execute_process(
        command=["/bin/echo", "hi"],
        isolation=IsolationConfig(sandbox_type="process"),
        working_dir=None,
        env=None,
        input_data=None,
        backend=None,
    )

    assert result.success is True
    assert started["command"] == ["/bin/echo", "hi"]
    assert started["pass_fds"] == ()


# ── site 3: MCP stdio client ─────────────────────────────────────────────────


def _mcp_client() -> MCPClient:
    return MCPClient(name="probe", version="0.0.0")


async def test_mcp_stdio_popen_wraps_and_forwards_pass_fds(monkeypatch):
    backend = _PipePassthroughBackend()
    monkeypatch.setattr(
        mcp_client_mod, "_resolve_mcp_launch_policy", lambda _snapshot=None: _required(backend)
    )
    captured: Dict[str, Any] = {}

    def fake_popen(argv, **kwargs):
        captured["argv"] = list(argv)
        captured["pass_fds"] = kwargs.get("pass_fds", ())
        return MagicMock()

    monkeypatch.setattr(mcp_client_mod.subprocess, "Popen", fake_popen)

    client = _mcp_client()
    process, owner = await client._start_process(["python", "-c", "pass"])

    assert owner is None and process is not None
    assert captured["argv"] == ["python", "-c", "pass"]
    assert captured["pass_fds"], "the seccomp filter descriptor must reach the spawn"
    _assert_closed(captured["pass_fds"])
    assert backend.take_pass_fds() == []


async def test_mcp_stdio_popen_without_policy_keeps_raw_argv(monkeypatch):
    monkeypatch.setattr(
        mcp_client_mod, "_resolve_mcp_launch_policy", lambda _snapshot=None: _not_required()
    )
    captured: Dict[str, Any] = {}

    def fake_popen(argv, **kwargs):
        captured["argv"] = list(argv)
        captured["pass_fds"] = kwargs.get("pass_fds", ())
        return MagicMock()

    monkeypatch.setattr(mcp_client_mod.subprocess, "Popen", fake_popen)

    client = _mcp_client()
    _, owner = await client._start_process(["python", "-c", "pass"])

    assert owner is None
    assert captured["argv"] == ["python", "-c", "pass"]
    assert captured["pass_fds"] == ()


# ── site 4: filesystem write enhancer (LSP formatter subprocess) ─────────────


class _FakeFormatter:
    name = "stubfmt"

    command = ["/bin/echo"]


class _FakePlugin:
    def get_formatter(self, _project_root: Path) -> _FakeFormatter:
        return _FakeFormatter()


class _FakeRegistry:
    def detect_language(self, _file_path: Path) -> str:
        return "py-stub"

    def get(self, _language: str) -> _FakePlugin:
        return _FakePlugin()


async def _format_once(monkeypatch, workspace_root: Path, decision: RequiredLaunchDecision):
    _patch_required_launch(monkeypatch, decision)
    captured: Dict[str, Any] = {}
    real_exec = asyncio.create_subprocess_exec

    async def fake_exec(*argv, **kwargs):
        captured["argv"] = list(argv)
        captured["pass_fds"] = kwargs.get("pass_fds", ())
        return await real_exec(*argv, **kwargs)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    enhancer = LSPWriteEnhancer(
        workspace_root=str(workspace_root), language_registry=_FakeRegistry()
    )
    formatted, formatter_name = await enhancer.format_with_formatter(
        str(workspace_root / "probe.py"), "probe = 1"
    )
    return captured, formatted, formatter_name


async def test_lsp_formatter_launch_wraps_and_forwards_pass_fds(monkeypatch, tmp_path):
    captured, formatted, formatter_name = await _format_once(
        monkeypatch, tmp_path, _required(_PipePassthroughBackend())
    )

    assert formatter_name == "stubfmt"
    assert formatted == "probe = 1"
    assert len(captured["argv"]) == 2
    assert captured["argv"][0] == "/bin/echo"
    assert captured["argv"][1].endswith(".py")
    assert captured["pass_fds"], "the seccomp filter descriptor must reach the spawn"
    _assert_closed(captured["pass_fds"])


async def test_lsp_formatter_launch_without_policy_keeps_raw_argv(monkeypatch, tmp_path):
    captured, _formatted, formatter_name = await _format_once(
        monkeypatch, tmp_path, _not_required()
    )

    assert formatter_name == "stubfmt"
    assert captured["pass_fds"] == ()


# ── site 5: merge manager ────────────────────────────────────────────────────


def test_merge_manager_git_wraps_and_forwards_pass_fds(monkeypatch, tmp_path):
    backend = _PipePassthroughBackend()
    calls = _capture_subprocess_run(monkeypatch)

    manager = MergeManager(root_path=tmp_path)
    manager._run_git(["status"], decision=_required(backend), capture_output=True, text=True)

    assert calls[0]["argv"][0] == "git"
    assert calls[0]["pass_fds"], "the seccomp filter descriptor must reach the spawn"
    assert backend.take_pass_fds() == []


def test_merge_manager_git_without_policy_keeps_raw_argv(monkeypatch, tmp_path):
    calls = _capture_subprocess_run(monkeypatch)

    manager = MergeManager(root_path=tmp_path)
    manager._run_git(["status"], decision=_not_required(), capture_output=True, text=True)

    assert calls[0]["argv"] == ["git", "status"]
    assert calls[0]["pass_fds"] == ()


def test_merge_manager_repeated_failed_launches_keep_fd_count_stable(monkeypatch, tmp_path):
    """The leak shape: N failing wrapped launches must not grow the fd table."""

    def _open_fd_count() -> int:
        return len(os.listdir("/proc/self/fd"))

    manager = MergeManager(root_path=tmp_path)
    baseline = _open_fd_count()
    for _ in range(8):
        with pytest.raises(FileNotFoundError):
            manager._run_git(["status"], decision=_required(_PipeBackend()))
    assert _open_fd_count() == baseline


# ── site 6: temporal analyzer (git history) ──────────────────────────────────


def test_temporal_analyzer_git_wraps_and_forwards_pass_fds(monkeypatch, tmp_path):
    backend = _PipePassthroughBackend()
    _patch_required_launch(monkeypatch, _required(backend))
    calls = _capture_subprocess_run(monkeypatch)

    analyzer = GitHistoryAnalyzer(tmp_path, run_policy_snapshot=_SnapshotStub())
    analyzer._run_git(["log", "--format=%ct", "--", "probe.py"])

    assert calls[0]["argv"][0] == "git"
    assert calls[0]["pass_fds"], "the seccomp filter descriptor must reach the spawn"
    assert backend.take_pass_fds() == []


def test_temporal_analyzer_git_without_policy_keeps_raw_argv(monkeypatch, tmp_path):
    _patch_required_launch(monkeypatch, _not_required())
    calls = _capture_subprocess_run(monkeypatch)

    analyzer = GitHistoryAnalyzer(tmp_path, run_policy_snapshot=_SnapshotStub())
    analyzer._run_git(["log", "--format=%ct", "--", "probe.py"])

    assert calls[0]["argv"][0] == "git"
    assert calls[0]["pass_fds"] == ()


# ── site 7: auto-commit ──────────────────────────────────────────────────────


def test_auto_committer_git_wraps_and_forwards_pass_fds(monkeypatch, tmp_path):
    backend = _PipePassthroughBackend()
    calls = _capture_subprocess_run(monkeypatch)

    committer = AutoCommitter(workspace_root=tmp_path)
    committer._run_git("status", check=False, decision=_required(backend))

    assert calls[0]["argv"][0] == "git"
    assert calls[0]["pass_fds"], "the seccomp filter descriptor must reach the spawn"
    assert backend.take_pass_fds() == []


def test_auto_committer_git_without_policy_keeps_raw_argv(monkeypatch, tmp_path):
    calls = _capture_subprocess_run(monkeypatch)

    committer = AutoCommitter(workspace_root=tmp_path)
    committer._run_git("status", check=False, decision=_not_required())

    assert calls[0]["argv"] == ["git", "status"]
    assert calls[0]["pass_fds"] == ()


# ── site 8: snapshot store ───────────────────────────────────────────────────


def test_snapshot_store_git_wraps_and_forwards_pass_fds(monkeypatch, tmp_path):
    backend = _PipePassthroughBackend()
    calls = _capture_subprocess_run(monkeypatch)

    store = FileSnapshotStore(workspace_root=tmp_path)
    store._run_git(["rev-parse", "HEAD"], decision=_required(backend))

    assert calls[0]["argv"][0] == "git"
    assert calls[0]["pass_fds"], "the seccomp filter descriptor must reach the spawn"
    assert backend.take_pass_fds() == []


def test_snapshot_store_git_without_policy_keeps_raw_argv(monkeypatch, tmp_path):
    calls = _capture_subprocess_run(monkeypatch)

    store = FileSnapshotStore(workspace_root=tmp_path)
    store._run_git(["rev-parse", "HEAD"], decision=_not_required())

    assert calls[0]["argv"] == ["git", "rev-parse", "HEAD"]
    assert calls[0]["pass_fds"] == ()


# ── site 9: navigation slash command (UI commit) ─────────────────────────────


def test_navigation_commit_git_wraps_and_forwards_pass_fds(monkeypatch, tmp_path):
    backend = _PipePassthroughBackend()
    calls = _capture_subprocess_run(monkeypatch)

    navigation_mod._run_git(
        ["git", "status", "--porcelain"],
        decision=_required(backend),
        workspace_root=tmp_path,
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    assert calls[0]["argv"][0] == "git"
    assert calls[0]["pass_fds"], "the seccomp filter descriptor must reach the spawn"
    assert backend.take_pass_fds() == []


def test_navigation_commit_git_without_policy_keeps_raw_argv(monkeypatch, tmp_path):
    calls = _capture_subprocess_run(monkeypatch)

    navigation_mod._run_git(
        ["git", "status", "--porcelain"],
        decision=_not_required(),
        workspace_root=tmp_path,
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    assert calls[0]["argv"] == ["git", "status", "--porcelain"]
    assert calls[0]["pass_fds"] == ()


# ── site 10: filesystem read-root discovery (same defect class, found in sweep) ─


def test_read_root_discovery_wraps_and_forwards_pass_fds(monkeypatch, tmp_path):
    backend = _PipePassthroughBackend()
    _patch_required_launch(monkeypatch, _required(backend))
    calls = _capture_subprocess_run(monkeypatch)

    _discover_read_roots(tmp_path, run_policy_snapshot=_SnapshotStub())

    assert calls[0]["argv"][0] == "git"
    assert calls[0]["pass_fds"], "the seccomp filter descriptor must reach the spawn"
    assert backend.take_pass_fds() == []


def test_read_root_discovery_without_policy_keeps_raw_argv(monkeypatch, tmp_path):
    _patch_required_launch(monkeypatch, _not_required())
    calls = _capture_subprocess_run(monkeypatch)

    _discover_read_roots(tmp_path, run_policy_snapshot=_SnapshotStub())

    assert calls[0]["argv"][0] == "git"
    assert calls[0]["pass_fds"] == ()


# ── live qualification through a real site ───────────────────────────────────


@requires_bwrap
def test_live_snapshot_store_git_runs_sandboxed_with_filter_attached(tmp_path):
    """Real bwrap through a converted site: the wrapped git child starts and
    exits cleanly — only possible when the seccomp descriptor is inherited
    (bwrap 0.9 refuses outright without it)."""
    decision = RequiredLaunchDecision(required_active=True, backend=BubblewrapSandbox())
    store = FileSnapshotStore(workspace_root=tmp_path)
    result = store._run_git(["--version"], decision=decision)
    assert result.returncode == 0, result.stderr
    assert "git version" in result.stdout


# ── structural sweep: no call site outside the seam wraps argv ───────────────


def test_no_call_site_wraps_argv_outside_the_shared_seam():
    """Every argv wrap must live in ``victor/tools/sandbox/`` (the seam).

    ``prepare_launch`` is the only sanctioned ``wrap_argv`` consumer: it
    pairs the wrapped argv with the backend's owned descriptors and drains
    them, so a wrap anywhere else is by construction a wrap-then-spawn site
    that leaks the seccomp filter fd and weakens isolation.
    """
    seam_dir = Path(launch_mod.__file__).parent
    package_root = seam_dir.parent.parent
    assert package_root.name == "victor"

    offenders = []
    for path in sorted(package_root.rglob("*.py")):
        rel = path.relative_to(package_root).as_posix()
        if rel.startswith("tools/sandbox/"):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if ".wrap_argv(" in text or "take_pass_fds(" in text:
            offenders.append(rel)
    assert offenders == [], (
        "call sites bypassing prepare_launch/run_launch (wrap-then-spawn): " f"{offenders}"
    )
