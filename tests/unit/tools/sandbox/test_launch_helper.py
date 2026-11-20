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

"""The shared sandboxed-launch seam (Codex finding 8).

The direct terminal/git/search launchers used to forward the backend's
wrapped argv WITHOUT the descriptors the wrapper opens (bwrap's seccomp
filter fd): the child ran without its filter (or refused to start) and the
parent leaked one open descriptor per launch. Every spawn site now returns
argv AND owned descriptors together through :func:`prepare_launch` and
spawns through :func:`run_launch` / :func:`run_launch_async`, which close
the owned descriptors on every exit.
"""

from __future__ import annotations

import asyncio
import os
import platform
import shutil
import subprocess

import pytest

from victor.tools.sandbox import (
    BubblewrapSandbox,
    NoneSandbox,
    close_pass_fds,
    prepare_launch,
    run_launch,
    run_launch_async,
)
from victor.tools.sandbox import launch as launch_mod


def _bwrap_available() -> bool:
    return platform.system() == "Linux" and shutil.which("bwrap") is not None


requires_bwrap = pytest.mark.skipif(
    not _bwrap_available(), reason="bwrap not available on this host"
)


class _PipeBackend(NoneSandbox):
    """Backend that opens one pipe per wrap (the bwrap seccomp-fd shape).

    ``wrap_argv`` leaves the read end pending, exactly like
    ``BubblewrapSandbox``; ``take_pass_fds`` returns and drains it.
    """

    type_name = "pipe"

    def __init__(self) -> None:
        self._pending: list[int] = []
        self.wrapped: list[list[str]] = []

    def wrap_argv(self, argv, cwd=None):  # type: ignore[override]
        r, w = os.pipe()
        os.close(w)
        self._pending.append(r)
        self.wrapped.append(["wrapped"] + list(argv))
        return ["wrapped"] + list(argv)

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


def _open_fd_count() -> int:
    return len(os.listdir("/proc/self/fd"))


# ── argv and owned descriptors travel together ───────────────────────────────


def test_prepare_launch_pairs_argv_with_descriptors():
    backend = _PipeBackend()
    launch = prepare_launch(backend, ["echo", "hi"], cwd="/tmp")

    assert launch.argv == ["wrapped", "echo", "hi"]
    assert len(launch.pass_fds) == 1
    # The drain happened with the wrap: the backend holds nothing back.
    assert backend.take_pass_fds() == []
    close_pass_fds(launch.pass_fds)


def test_prepare_launch_without_backend_is_raw():
    assert prepare_launch(None, ["echo", "hi"]).argv == ["echo", "hi"]
    assert prepare_launch(None, ["echo", "hi"]).pass_fds == ()
    noop = prepare_launch(NoneSandbox(), ["echo", "hi"])
    assert noop.argv == ["echo", "hi"] and noop.pass_fds == ()


def test_prepare_launch_enforcing_backend_rejects_shell_string():
    with pytest.raises(TypeError):
        prepare_launch(_PipeBackend(), "echo hi")


@requires_bwrap
def test_prepare_launch_bwrap_opens_seccomp_descriptor():
    sandbox = BubblewrapSandbox()
    launch = prepare_launch(sandbox, ["/bin/echo", "hi"], cwd=None)

    assert launch.argv[0] == "bwrap"
    assert "--seccomp" in launch.argv
    assert launch.pass_fds, "bwrap wrap must own the seccomp filter descriptor"
    assert sandbox.take_pass_fds() == []
    close_pass_fds(launch.pass_fds)


# ── spawn + cleanup on all exits ─────────────────────────────────────────────


def test_run_launch_forwards_pass_fds(monkeypatch):
    backend = _PipeBackend()
    launch = prepare_launch(backend, ["/bin/echo", "hi"])
    captured = {}
    real_run = launch_mod.subprocess.run

    def fake_run(argv, **kwargs):
        captured["pass_fds"] = kwargs.get("pass_fds")
        return real_run(["/bin/echo", "hi"], capture_output=True)

    monkeypatch.setattr(launch_mod.subprocess, "run", fake_run)
    result = run_launch(launch, capture_output=True)

    assert result.returncode == 0
    assert captured["pass_fds"] == launch.pass_fds
    # Cleanup: the owned descriptors are closed once the spawn ends.
    for fd in launch.pass_fds:
        with pytest.raises(OSError):
            os.fstat(fd)


def test_run_launch_closes_descriptors_when_spawn_fails():
    """Negative: a failing spawn must still close the owned descriptors."""
    backend = _PipeBackend()
    launch = prepare_launch(backend, ["/does/not/exist-victor-probe"])
    fds = launch.pass_fds
    assert fds

    with pytest.raises(FileNotFoundError):
        run_launch(launch)

    for fd in fds:
        with pytest.raises(OSError):
            os.fstat(fd)


def test_repeated_failed_launches_keep_parent_descriptor_count_stable():
    """The leak shape: N failing launches must not grow the parent fd table."""
    backend = _PipeBackend()
    baseline = _open_fd_count()
    for _ in range(8):
        launch = prepare_launch(backend, ["/does/not/exist-victor-probe"])
        with pytest.raises(FileNotFoundError):
            run_launch(launch)
    assert _open_fd_count() == baseline


def test_repeated_admission_refusals_keep_parent_descriptor_count_stable(monkeypatch):
    """Admission failure = refuse BEFORE any wrap: zero descriptors opened."""
    from victor.config.sandbox_settings import SandboxSettings
    from victor.tools.sandbox import resolve_required_launch
    from victor.tools.sandbox import backends as sb_mod

    monkeypatch.setattr(sb_mod.platform, "system", lambda: "Linux")
    monkeypatch.setattr(sb_mod.shutil, "which", lambda name: None)

    settings = SandboxSettings(sandbox_enabled=True, sandbox_policy="required")
    baseline = _open_fd_count()
    for _ in range(8):
        decision = resolve_required_launch(settings)
        assert decision.refuse, "unavailable backend under required must refuse"
    assert _open_fd_count() == baseline


async def test_run_launch_async_forwards_pass_fds_and_closes():
    backend = _PipePassthroughBackend()
    launch = prepare_launch(backend, ["/bin/echo", "hi"])
    proc = await run_launch_async(launch, stdout=asyncio.subprocess.PIPE)
    stdout, _ = await proc.communicate()

    assert proc.returncode == 0
    assert b"hi" in stdout
    for fd in launch.pass_fds:
        with pytest.raises(OSError):
            os.fstat(fd)


async def test_run_launch_async_closes_descriptors_when_spawn_fails():
    launch = prepare_launch(_PipeBackend(), ["/does/not/exist-victor-probe"])
    fds = launch.pass_fds
    with pytest.raises(FileNotFoundError):
        await run_launch_async(launch)
    for fd in fds:
        with pytest.raises(OSError):
            os.fstat(fd)


def test_repeated_async_admission_failures_keep_descriptor_count_stable():
    """Async route shape: failing async spawns keep the parent table stable."""

    async def _scenario():
        baseline = _open_fd_count()
        for _ in range(8):
            launch = prepare_launch(_PipeBackend(), ["/does/not/exist-victor-probe"])
            with pytest.raises(FileNotFoundError):
                await run_launch_async(launch)
        assert _open_fd_count() == baseline

    asyncio.run(_scenario())


# ── live qualification: harmless subprocess, filter actually attached ────────


@requires_bwrap
def test_live_harmless_subprocess_runs_sandboxed_through_helper():
    """A plain command through the shared helper stays green under bwrap."""
    launch = prepare_launch(BubblewrapSandbox(), ["/bin/echo", "LAUNCH-OK"], cwd=None)
    result = run_launch(launch, capture_output=True, timeout=15)
    assert result.returncode == 0, result.stderr.decode("utf-8", "replace")
    assert b"LAUNCH-OK" in result.stdout


@requires_bwrap
def test_live_seccomp_filter_is_actually_attached():
    """The unshare probe must be DENIED: proves the descriptor was forwarded
    and the filter loaded in the kernel for the wrapped child."""
    probe = (
        "import ctypes,sys\n"
        "libc = ctypes.CDLL(None, use_errno=True)\n"
        "rc = libc.unshare(0x00020000)\n"  # CLONE_NEWUSER
        "sys.exit(0 if rc == 0 else 7)\n"
    )
    launch = prepare_launch(BubblewrapSandbox(), ["/usr/bin/python3", "-c", probe], cwd=None)
    result = run_launch(launch, capture_output=True, timeout=15)
    assert result.returncode == 7, (
        f"unshare succeeded inside the sandbox (rc={result.returncode}); "
        "the seccomp filter did not apply"
    )


@requires_bwrap
def test_live_argv_without_descriptors_is_rejected():
    """Negative control: the OLD route behavior (argv only, descriptors
    withheld) fails loudly — forwarding the descriptors is load-bearing."""
    sandbox = BubblewrapSandbox()
    argv = sandbox.wrap_argv(["/bin/echo", "hi"], cwd=None)
    # Deliberately NOT forwarding sandbox.take_pass_fds().
    result = subprocess.run(argv, capture_output=True, timeout=15)  # nosec B603
    fds = sandbox.take_pass_fds()
    close_pass_fds(fds)
    assert result.returncode != 0, (
        "bwrap accepted its seccomp descriptor without inheritance; "
        "this control no longer exercises the finding"
    )
