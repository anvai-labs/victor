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
from victor.tools.subprocess_executor import run_command, run_command_async


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
