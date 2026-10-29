# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
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

"""Real-Docker lifecycle of the code-execution sandbox (skipped without Docker).

Pins the guarantees that stop sandbox containers from accumulating: removal
on dispose and on garbage collection, self-expiry after the idle TTL even
when the owner is SIGKILLed, and a reaper that never touches a live owner.
"""

import gc
import os
import signal
import subprocess
import sys
import time

import pytest

cet = pytest.importorskip("victor_coding.tools.code_executor_tool")


def _client():
    try:
        import docker

        client = docker.from_env(timeout=5)
        client.ping()
        return client
    except Exception:
        return None


pytestmark = pytest.mark.skipif(_client() is None, reason="Docker daemon not available")


def _exists(container_id: str) -> bool:
    import docker

    try:
        _client().containers.get(container_id)
        return True
    except docker.errors.NotFound:
        return False


def _wait_gone(container_id: str, seconds: float) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if not _exists(container_id):
            return True
        time.sleep(0.5)
    return False


_CHILD = """
import sys, time
from victor_coding.tools.code_executor_tool import CodeSandbox
s = CodeSandbox(idle_ttl_seconds=int(sys.argv[1]))
s.execute("print(1)")
print(s.container.id, flush=True)
time.sleep(600)
"""


def _spawn_owner(ttl: int) -> tuple:
    proc = subprocess.Popen(
        [sys.executable, "-c", _CHILD, str(ttl)],
        stdout=subprocess.PIPE,
        text=True,
        env=dict(os.environ),
    )
    return proc, proc.stdout.readline().strip()


def test_execute_then_dispose_removes_the_container():
    sandbox = cet.CodeSandbox()
    try:
        assert sandbox.execute("print(6 * 7)")["stdout"].strip() == "42"
        container_id = sandbox.container.id
    finally:
        sandbox.dispose()
    assert _wait_gone(container_id, 10)


def test_garbage_collected_sandbox_removes_its_container():
    sandbox = cet.CodeSandbox()
    sandbox.execute("print(1)")
    container_id = sandbox.container.id
    del sandbox
    gc.collect()
    assert _wait_gone(container_id, 10)


def test_sigkilled_owner_container_expires_on_its_own():
    proc, container_id = _spawn_owner(ttl=4)
    try:
        assert _exists(container_id)
        proc.send_signal(signal.SIGKILL)
        proc.wait()
        # nothing on the host cleans up: the idle watchdog + auto_remove must
        assert _wait_gone(container_id, 4 + 20)
    finally:
        proc.kill()


def test_reaper_spares_a_live_owner_and_removes_a_dead_one():
    live, live_id = _spawn_owner(ttl=600)
    dead, dead_id = _spawn_owner(ttl=600)
    try:
        dead.send_signal(signal.SIGKILL)
        dead.wait()
        reaped = cet.reap_orphaned_sandboxes()
        assert dead_id[:12] in [r[:12] for r in reaped]
        assert _exists(live_id) and not _exists(dead_id)
    finally:
        for proc in (live, dead):
            proc.kill()
            proc.wait()
        cet.reap_orphaned_sandboxes()  # the SIGKILLed live owner's container too


def test_image_has_the_watchdog_toolchain():
    sandbox = cet.CodeSandbox()
    try:
        result = sandbox.execute(
            "import shutil; print(all(shutil.which(t) for t in ('sh', 'stat', 'date', 'timeout')))"
        )
        assert result["stdout"].strip() == "True"
        assert sandbox.execute("import time; time.sleep(10)", timeout=1)["exit_code"] == 137
    finally:
        sandbox.dispose()
