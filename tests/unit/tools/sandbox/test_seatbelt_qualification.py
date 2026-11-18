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

"""ADR-035 S1b qualification battery for the deny-default Seatbelt profile.

SKIPPED on non-macOS hosts (the CI matrix has no macOS runner today).
Run on any macOS machine with:

    pytest tests/unit/tools/sandbox/test_seatbelt_qualification.py -v

Each test drives a REAL ``sandbox-exec`` process against the production
profile from :class:`SeatbeltSandbox` — this is the execution evidence the
S1b gate asks for before the posture flip is considered qualified on
macOS (profile-construction tests already run everywhere).
"""

import os
import platform
import subprocess

import pytest

from victor.tools.sandbox import SeatbeltSandbox

pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin", reason="Seatbelt qualification requires macOS"
)


def _run(argv: list[str], timeout: int = 15) -> subprocess.CompletedProcess:
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)


def _wrapped(sandbox: SeatbeltSandbox, payload: list[str], cwd: str | None = None):
    return sandbox.wrap_argv(payload, cwd=cwd)


class TestSeatbeltQualification:
    """Execution battery: every assertion is the ADR-035 S1a profile
    contract observed through a real macOS sandbox process."""

    def test_plain_command_runs_inside_sandbox(self, tmp_path):
        """Baseline: echo executes under the deny-default profile."""
        argv = _wrapped(SeatbeltSandbox(), ["/bin/echo", "QUALIFY-OK"], cwd=str(tmp_path))
        proc = _run(argv)
        assert proc.returncode == 0 and "QUALIFY-OK" in proc.stdout

    def test_workspace_write_allowed(self, tmp_path):
        sandbox = SeatbeltSandbox()
        argv = _wrapped(
            sandbox,
            ["/usr/bin/touch", str(tmp_path / "workspace-write.ok")],
            cwd=str(tmp_path),
        )
        proc = _run(argv)
        assert proc.returncode == 0, proc.stderr
        assert (tmp_path / "workspace-write.ok").exists()

    def test_system_write_denied(self, tmp_path):
        """Deny-default: writes outside workspace+tmp must fail."""
        sandbox = SeatbeltSandbox()
        argv = _wrapped(
            sandbox,
            ["/usr/bin/touch", "/etc/victor-seatbelt-qualify.probe"],
            cwd=str(tmp_path),
        )
        proc = _run(argv)
        assert proc.returncode != 0, "system-area write was NOT denied"
        assert not os.path.exists("/etc/victor-seatbelt-qualify.probe")

    def test_broad_read_allowed(self, tmp_path):
        sandbox = SeatbeltSandbox()
        argv = _wrapped(sandbox, ["/bin/cat", "/etc/hosts"], cwd=str(tmp_path))
        proc = _run(argv)
        assert proc.returncode == 0, proc.stderr

    def test_network_denied_by_default(self, tmp_path):
        """Egress deny is the default (network_isolation=True)."""
        sandbox = SeatbeltSandbox()
        probe = (
            "import socket,sys\n"
            "try:\n"
            "    socket.create_connection(('127.0.0.1', 1), timeout=2)\n"
            "    sys.exit(0)\n"
            "except OSError:\n"
            "    sys.exit(7)\n"
        )
        script = tmp_path / "probe.py"
        script.write_text(probe)
        argv = _wrapped(sandbox, ["/usr/bin/python3", "-c", probe], cwd=str(tmp_path))
        proc = _run(argv)
        # Exit 7 = connect raised OSError (denied or unreachable-by-denial);
        # exit 0 = the connection attempt SUCCEEDED, which must not happen.
        assert proc.returncode == 7, (
            f"egress attempt did not fail under the default profile "
            f"(rc={proc.returncode}, stderr={proc.stderr[:200]})"
        )

    def test_network_allowed_when_isolation_off(self, tmp_path):
        sandbox = SeatbeltSandbox(network_isolation=False)
        probe = (
            "import socket,sys\n"
            "s = socket.socket()\n"
            "s.settimeout(2)\n"
            "try:\n"
            "    s.connect(('127.0.0.1', 65534))\n"
            "except OSError:\n"
            "    pass\n"
            "sys.exit(0)\n"
        )
        argv = _wrapped(sandbox, ["/usr/bin/python3", "-c", probe], cwd=str(tmp_path))
        proc = _run(argv)
        assert proc.returncode == 0, (
            f"socket() creation itself was denied with isolation off "
            f"(rc={proc.returncode}, stderr={proc.stderr[:200]})"
        )

    def test_process_fork_allowed(self, tmp_path):
        sandbox = SeatbeltSandbox()
        argv = _wrapped(
            sandbox,
            ["/bin/sh", "-c", "/bin/echo forked-from-child"],
            cwd=str(tmp_path),
        )
        proc = _run(argv)
        assert proc.returncode == 0 and "forked-from-child" in proc.stdout
