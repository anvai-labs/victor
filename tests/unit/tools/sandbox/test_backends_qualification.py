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

"""FOU-S1a executor plumbing + live bwrap qualification suite.

The live tests are the ADR-035 RED/GREEN evidence: under the unhardened
backend, an egress attempt inside bwrap succeeds (RED); after hardening it
is denied (GREEN), while normal tool execution keeps working.
"""

import os
import platform
import shutil

import pytest

from victor.tools.sandbox import BubblewrapSandbox
from victor.tools.subprocess_executor import run_managed_process


def _bwrap_available() -> bool:
    return platform.system() == "Linux" and shutil.which("bwrap") is not None


requires_bwrap = pytest.mark.skipif(
    not _bwrap_available(), reason="bwrap not available on this host"
)


# -- pass_fds plumbing -------------------------------------------------------


@pytest.mark.asyncio
async def test_run_managed_process_accepts_pass_fds(tmp_path):
    """The managed runner must support inheriting extra fds across exec."""
    argv = ["/bin/echo", "ok"]
    stdout, stderr, code, timed_out, capped = await run_managed_process(
        argv=argv, cwd=tmp_path, timeout=10, pass_fds=()
    )
    assert code == 0
    assert b"ok" in stdout
    assert not timed_out


def test_sync_run_command_threads_pass_fds(monkeypatch, tmp_path):
    """The sync path must forward backend pass fds to subprocess.run."""
    from victor.tools import subprocess_executor as se

    captured = {}

    real_run = se.subprocess.run

    def fake_run(args, **kwargs):
        captured["pass_fds"] = kwargs.get("pass_fds")
        return real_run(
            ["/bin/echo", "ok"],
            capture_output=True,
            text=True,
            cwd=str(tmp_path),
        )

    monkeypatch.setattr(se.subprocess, "run", fake_run)
    sandbox = BubblewrapSandbox()
    se.run_command(["echo", "ok"], working_dir=str(tmp_path), timeout=10, sandbox=sandbox)
    # The executor forwarded the backend's seccomp fd and drained it.
    assert captured.get("pass_fds") is not None
    assert sandbox.take_pass_fds() == []


def test_sync_shell_mode_threads_pass_fds(monkeypatch, tmp_path):
    """shell=True must still deliver the seccomp fd (regression: the shell
    branch returned no fds while wrap_argv had already created one)."""
    from victor.tools import subprocess_executor as se

    captured = {}
    real_run = se.subprocess.run

    def fake_run(args, **kwargs):
        captured["pass_fds"] = kwargs.get("pass_fds")
        return real_run(
            ["/bin/echo", "ok"],
            capture_output=True,
            text=True,
            cwd=str(tmp_path),
        )

    monkeypatch.setattr(se.subprocess, "run", fake_run)
    sandbox = BubblewrapSandbox()
    se.run_command("echo ok", shell=True, working_dir=str(tmp_path), timeout=10, sandbox=sandbox)
    assert captured.get("pass_fds"), "shell mode dropped the seccomp fd"
    assert sandbox.take_pass_fds() == []


# -- live bwrap qualification -------------------------------------------------


@requires_bwrap
def test_live_egress_denied_under_default_isolation():
    """GREEN: network access inside the wrapped command must fail."""
    sandbox = BubblewrapSandbox()
    assert sandbox.available()
    probe = (
        "import socket,sys\n"
        "try:\n"
        "    socket.create_connection(('1.1.1.1', 53), timeout=3)\n"
        "    sys.exit(0)\n"
        "except OSError:\n"
        "    sys.exit(7)\n"
    )
    argv = sandbox.wrap_argv(["/usr/bin/python3", "-c", probe], cwd=None)
    proc = os.spawnvpe(os.P_WAIT, argv[0], argv, os.environ)
    assert proc == 7, "egress succeeded inside the sandbox; isolation did not apply"


@requires_bwrap
def test_live_plain_command_still_works():
    """GREEN: hardened wrapping must not break normal tool execution."""
    sandbox = BubblewrapSandbox()
    argv = sandbox.wrap_argv(["/bin/echo", "SOAK-OK"], cwd=None)
    proc = os.spawnvpe(os.P_WAIT, argv[0], argv, os.environ)
    assert proc == 0


@requires_bwrap
def test_live_unshare_denied_by_seccomp():
    """GREEN: the seccomp filter must deny unshare inside the sandbox."""
    sandbox = BubblewrapSandbox()
    probe = (
        "import ctypes,sys\n"
        "libc = ctypes.CDLL(None, use_errno=True)\n"
        "rc = libc.unshare(0x00020000)\n"  # CLONE_NEWUSER
        "sys.exit(0 if rc == 0 else 7)\n"
    )
    argv = sandbox.wrap_argv(["/usr/bin/python3", "-c", probe], cwd=None)
    proc = os.spawnvpe(os.P_WAIT, argv[0], argv, os.environ)
    assert proc == 7, "unshare succeeded inside the sandbox; seccomp did not apply"


@requires_bwrap
def test_live_full_executor_run_wrapped():
    """The canonical executor path stays green end-to-end under hardening."""
    import asyncio

    async def _run():
        stdout, stderr, code, timed_out, _ = await run_managed_process(
            argv=["/bin/echo", "ok"], cwd=None, timeout=15
        )
        return stdout, code, timed_out

    stdout, code, timed_out = asyncio.run(_run())
    assert code == 0 and b"ok" in stdout and not timed_out
