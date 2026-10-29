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

"""Tests for the code executor sandbox (Docker-based code execution).

Pins the leak-proof lifecycle: lazy acquisition (no Docker contact until
first use), ownership (stop/dispose/finalizer remove the container),
self-expiry (idle watchdog + auto_remove), transparent re-acquisition after
expiry, and the owner-aware reaper's decisions. Docker is always mocked here;
tests/integration/infra/test_docker_availability.py exercises a real daemon.
"""

from __future__ import annotations

import asyncio
import gc
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import victor_coding.tools.code_executor_tool as cet
from victor_coding.tools.code_executor_tool import (
    CodeSandbox,
    ResourceError,
    find_orphaned_sandboxes,
    owner_alive,
    process_start_marker,
    reap_orphaned_sandboxes,
    watchdog_command,
)

MODULE = "victor_coding.tools.code_executor_tool"


# ---------------------------------------------------------------------------
# Degraded mode: docker package absent or daemon unreachable
# ---------------------------------------------------------------------------


class TestDegradedMode:
    def test_without_docker_package(self):
        with patch(f"{MODULE}.DOCKER_AVAILABLE", False):
            sandbox = CodeSandbox()
            assert sandbox.docker_available is False
            assert sandbox.docker_client is None
        assert sandbox.container is None

    def test_without_docker_package_require_docker_raises(self):
        with patch(f"{MODULE}.DOCKER_AVAILABLE", False):
            with pytest.raises(ResourceError, match="Docker"):
                CodeSandbox(require_docker=True)

    def test_connection_failure_degrades_on_first_use(self):
        from docker.errors import DockerException

        with (
            patch(f"{MODULE}.DOCKER_AVAILABLE", True),
            patch(f"{MODULE}.docker") as mock_docker,
        ):
            mock_docker.from_env.side_effect = DockerException("not running")
            sandbox = CodeSandbox()
            mock_docker.from_env.assert_not_called()  # lazy: no probe at construction
            result = sandbox.execute("print(1)")
            assert sandbox.docker_available is False
        assert result["exit_code"] == 1
        assert "not available" in result["stderr"].lower()

    def test_connection_failure_require_docker_raises_eagerly(self):
        from docker.errors import DockerException

        with (
            patch(f"{MODULE}.DOCKER_AVAILABLE", True),
            patch(f"{MODULE}.docker") as mock_docker,
        ):
            mock_docker.from_env.side_effect = DockerException("daemon down")
            with pytest.raises(ResourceError, match="Docker"):
                CodeSandbox(require_docker=True)

    def test_degraded_lifecycle_is_inert(self):
        with patch(f"{MODULE}.DOCKER_AVAILABLE", False):
            with CodeSandbox() as sandbox:
                sandbox.put_files(["/nonexistent"])  # no-op, no raise
                assert sandbox.get_file("/app/x") == b""
            sandbox.dispose()
        assert sandbox.container is None


# ---------------------------------------------------------------------------
# Mocked Docker
# ---------------------------------------------------------------------------


def _mock_container(container_id="c0ffee"):
    container = MagicMock()
    container.id = container_id
    container.short_id = container_id[:6]
    container.status = "running"
    container.exec_run.return_value = MagicMock(exit_code=0, output=(b"hello world\n", b""))
    return container


@pytest.fixture
def docker_env():
    """Mocked docker module whose client hands out a fresh container per run()."""
    with (
        patch(f"{MODULE}.DOCKER_AVAILABLE", True),
        patch(f"{MODULE}.docker") as mock_docker,
        patch(f"{MODULE}._reap_once"),
    ):
        client = mock_docker.from_env.return_value
        created = []

        def _run(*_args, **_kwargs):
            created.append(_mock_container(f"c{len(created)}ffee"))
            return created[-1]

        client.containers.run.side_effect = _run
        yield SimpleNamespace(docker=mock_docker, client=client, created=created)


class TestLazyAcquisition:
    def test_construction_makes_no_docker_contact(self, docker_env):
        CodeSandbox()
        docker_env.docker.from_env.assert_not_called()
        docker_env.client.containers.run.assert_not_called()

    def test_first_execute_acquires_one_container(self, docker_env):
        sandbox = CodeSandbox()
        assert sandbox.execute("print('hi')") == {
            "exit_code": 0,
            "stdout": "hello world\n",
            "stderr": "",
        }
        sandbox.execute("print('again')")
        assert docker_env.client.containers.run.call_count == 1

    def test_connect_probe_is_bounded(self, docker_env, monkeypatch):
        monkeypatch.setenv("VICTOR_CODE_EXEC_START_TIMEOUT", "2.5")
        CodeSandbox().execute("print(1)")
        docker_env.docker.from_env.assert_called_once_with(timeout=2.5)

    def test_start_is_explicit_and_idempotent(self, docker_env):
        sandbox = CodeSandbox()
        sandbox.start()
        sandbox.start()
        assert docker_env.client.containers.run.call_count == 1

    def test_image_pulled_only_when_missing(self, docker_env):
        from docker.errors import ImageNotFound

        CodeSandbox().start()
        docker_env.client.images.pull.assert_not_called()
        docker_env.client.images.get.side_effect = ImageNotFound("missing")
        CodeSandbox(docker_image="python:3.12-slim").start()
        docker_env.client.images.pull.assert_called_once_with("python:3.12-slim")

    def test_container_is_self_expiring_and_owner_labelled(self, docker_env):
        CodeSandbox(idle_ttl_seconds=120).start()
        kwargs = docker_env.client.containers.run.call_args.kwargs
        assert kwargs["auto_remove"] is True and kwargs["init"] is True
        assert kwargs["command"] == watchdog_command(120)
        labels = kwargs["labels"]
        assert labels[cet.SANDBOX_CONTAINER_LABEL] == cet.SANDBOX_CONTAINER_VALUE
        assert labels[cet.OWNER_PID_LABEL] == str(os.getpid())
        assert labels[cet.OWNER_HOST_LABEL] == cet.host_identity()
        assert labels[cet.IDLE_TTL_LABEL] == "120"
        assert "sleep infinity" not in " ".join(kwargs["command"])


class TestOwnership:
    def test_stop_removes_and_next_use_reacquires(self, docker_env):
        sandbox = CodeSandbox()
        sandbox.start()
        first = docker_env.created[0]
        sandbox.stop()
        first.remove.assert_called_once_with(force=True)
        assert sandbox.container is None
        sandbox.execute("print(1)")
        assert len(docker_env.created) == 2

    def test_dispose_is_stop(self, docker_env):
        sandbox = CodeSandbox()
        sandbox.start()
        sandbox.dispose()
        docker_env.created[0].remove.assert_called_once_with(force=True)

    def test_stop_is_idempotent(self, docker_env):
        sandbox = CodeSandbox()
        sandbox.start()
        sandbox.stop()
        sandbox.stop()
        docker_env.created[0].remove.assert_called_once_with(force=True)

    def test_garbage_collection_removes_container(self, docker_env):
        sandbox = CodeSandbox()
        sandbox.start()
        container = docker_env.created[0]
        del sandbox
        gc.collect()
        container.remove.assert_called_once_with(force=True)

    def test_context_managers_release(self, docker_env):
        with CodeSandbox() as sandbox:
            assert sandbox.container is docker_env.created[0]
        assert sandbox.container is None

        async def _run():
            async with CodeSandbox() as s:
                assert s.container is not None
            return s

        assert asyncio.run(_run()).container is None

    def test_expired_container_is_reacquired_transparently(self, docker_env):
        from docker.errors import NotFound

        sandbox = CodeSandbox()
        sandbox.start()
        docker_env.created[0].reload.side_effect = NotFound("auto-removed after idle TTL")
        assert sandbox.execute("print(1)")["exit_code"] == 0
        assert len(docker_env.created) == 2
        docker_env.created[0].remove.assert_not_called()  # already gone; finalizer detached

    def test_failed_start_degrades_and_retries_next_use(self, docker_env):
        docker_env.client.containers.run.side_effect = [RuntimeError("no space"), _mock_container()]
        sandbox = CodeSandbox()
        result = sandbox.execute("print(1)")
        assert result["exit_code"] == 1 and "no space" in result["stderr"]
        assert sandbox.execute("print(1)")["exit_code"] == 0


class TestExecution:
    def test_code_is_passed_as_argument_never_interpolated(self, docker_env):
        code = """print("it's $HOME; `id`")"""
        sandbox = CodeSandbox()
        sandbox.execute(code, timeout=7)
        argv = docker_env.created[0].exec_run.call_args.args[0]
        assert argv[:2] == ["sh", "-c"]
        assert argv[3] == code  # $0, untouched by the shell
        # The script is a fixed template: nothing derived from the code enters it.
        assert argv[2] == f'touch {cet.LEASE_PATH}; exec timeout -s KILL 7 python -c "$0"'

    def test_timeout_kill_is_reported(self, docker_env):
        sandbox = CodeSandbox()
        sandbox.start()
        docker_env.created[0].exec_run.return_value = MagicMock(exit_code=137, output=(None, None))
        result = sandbox.execute("while True: pass", timeout=3)
        assert result["exit_code"] == 137
        assert "exceeded 3s" in result["stderr"]

    def test_stderr_path(self, docker_env):
        sandbox = CodeSandbox()
        sandbox.start()
        docker_env.created[0].exec_run.return_value = MagicMock(
            exit_code=1, output=(b"", b"ValueError: bad")
        )
        result = sandbox.execute("raise ValueError('bad')")
        assert result == {"exit_code": 1, "stdout": "", "stderr": "ValueError: bad"}

    def test_put_files_uploads_and_refreshes_lease(self, docker_env, tmp_path):
        src = tmp_path / "hello.py"
        src.write_text("print('hello')")
        sandbox = CodeSandbox()
        sandbox.put_files([str(src)])
        container = docker_env.created[0]
        container.put_archive.assert_called_once()
        container.exec_run.assert_called_with(["touch", cet.LEASE_PATH])

    def test_put_files_missing_local_file(self, docker_env):
        with pytest.raises(FileNotFoundError):
            CodeSandbox().put_files(["/definitely/not/here.py"])

    def test_get_file_returns_bytes(self, docker_env):
        import io
        import tarfile

        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w") as tar:
            payload = b"file contents"
            info = tarfile.TarInfo("out.txt")
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
        sandbox = CodeSandbox()
        sandbox.start()
        docker_env.created[0].get_archive.return_value = (
            io.BytesIO(buffer.getvalue()),
            {"name": "out.txt"},
        )
        assert sandbox.get_file("/app/out.txt") == b"file contents"


# ---------------------------------------------------------------------------
# Owner checks and the reaper
# ---------------------------------------------------------------------------


def _labelled(cid, *, host=None, pid=None, start="", legacy_cmd=False, age_s=0.0):
    container = MagicMock()
    container.short_id = cid
    labels = {cet.SANDBOX_CONTAINER_LABEL: cet.SANDBOX_CONTAINER_VALUE}
    if host is not None:
        labels.update({cet.OWNER_HOST_LABEL: host, cet.OWNER_PID_LABEL: str(pid)})
        labels[cet.OWNER_START_LABEL] = start
    container.labels = labels
    created = __import__("datetime").datetime.fromtimestamp(
        __import__("time").time() - age_s, __import__("datetime").timezone.utc
    )
    container.attrs = {
        "Created": created.strftime("%Y-%m-%dT%H:%M:%S.%f") + "123Z",
        "Config": {"Cmd": ["sleep", "infinity"] if legacy_cmd else ["sh", "-c", "watchdog"]},
    }
    return container


class TestReaper:
    def _client(self, *containers):
        client = MagicMock()
        client.containers.list.return_value = list(containers)
        return client

    def test_decisions(self):
        me = cet.host_identity()
        pid = os.getpid()
        live = _labelled("live", host=me, pid=pid, start=process_start_marker(pid))
        dead = _labelled("dead", host=me, pid=999999)
        reused = _labelled("reused", host=me, pid=pid, start="older-process")
        foreign = _labelled("foreign", host="another-boot-id", pid=999999)
        legacy_old = _labelled("legacy_old", legacy_cmd=True, age_s=7200)
        legacy_young = _labelled("legacy_young", legacy_cmd=True, age_s=60)
        unowned_cmd = _labelled("unowned", age_s=7200)  # labelled, no owner, not sleep-infinity
        client = self._client(live, dead, reused, foreign, legacy_old, legacy_young, unowned_cmd)
        with patch(f"{MODULE}.owner_alive", wraps=owner_alive):
            found = find_orphaned_sandboxes(client, legacy_min_age_seconds=3600)
        assert sorted(c.short_id for c in found) == ["dead", "legacy_old", "reused"]

    def test_filter_is_exact_code_executor_label(self):
        client = self._client()
        find_orphaned_sandboxes(client)
        assert client.containers.list.call_args.kwargs == {
            "all": True,
            "filters": {"label": "victor.sandbox=code-executor"},
        }

    def test_reap_removes_only_orphans_and_dry_run_removes_nothing(self):
        me = cet.host_identity()
        live = _labelled("live", host=me, pid=os.getpid(), start=process_start_marker(os.getpid()))
        dead = _labelled("dead", host=me, pid=999999)
        assert reap_orphaned_sandboxes(self._client(live, dead), dry_run=True) == ["dead"]
        dead.remove.assert_not_called()
        assert reap_orphaned_sandboxes(self._client(live, dead)) == ["dead"]
        dead.remove.assert_called_once_with(force=True)
        live.remove.assert_not_called()

    def test_legacy_can_be_excluded(self):
        legacy = _labelled("legacy", legacy_cmd=True, age_s=99999)
        assert find_orphaned_sandboxes(self._client(legacy), include_legacy=False) == []

    def test_unreachable_daemon_reaps_nothing(self):
        client = MagicMock()
        client.containers.list.side_effect = RuntimeError("daemon gone")
        assert reap_orphaned_sandboxes(client) == []

    def test_runs_once_per_process(self, monkeypatch):
        calls = []
        monkeypatch.setattr(cet, "_reaped_this_process", False)
        monkeypatch.setattr(cet, "reap_orphaned_sandboxes", lambda *a, **k: calls.append(1) or [])
        cet._reap_once(MagicMock())
        cet._reap_once(MagicMock())
        assert calls == [1]

    def test_watchdog_interval_scales_with_short_ttl(self):
        script = watchdog_command(4)[2]
        assert "sleep 1;" in script and "-gt 4 ]" in script
        assert "sleep 30;" in watchdog_command(900)[2]


# ---------------------------------------------------------------------------
# Async helpers and the @tool entry point (unchanged contract)
# ---------------------------------------------------------------------------


class TestAsyncHelpers:
    def test_execute_code_formats_output(self):
        from victor_coding.tools.code_executor_tool import _execute_code

        sandbox = MagicMock()
        sandbox.execute.return_value = {
            "exit_code": 0,
            "stdout": "42\n",
            "stderr": "",
        }
        result = asyncio.run(_execute_code(sandbox, "print(42)"))
        assert "Exit Code: 0" in result
        assert "42" in result

    def test_upload_files_success(self):
        from victor_coding.tools.code_executor_tool import _upload_files

        sandbox = MagicMock()
        result = asyncio.run(_upload_files(sandbox, ["/a.py", "/b.py"]))
        assert "2 files" in result

    def test_upload_files_error(self):
        from victor_coding.tools.code_executor_tool import _upload_files

        sandbox = MagicMock()
        sandbox.put_files.side_effect = RuntimeError("disk full")
        result = asyncio.run(_upload_files(sandbox, ["/a.py"]))
        assert "disk full" in result


# ---------------------------------------------------------------------------
# The @tool decorated sandbox entry point
# ---------------------------------------------------------------------------


class TestSandboxTool:
    def _make_ctx(self, sandbox_instance):
        return {"code_manager": sandbox_instance}

    def test_execute_operation(self):
        from victor_coding.tools.code_executor_tool import sandbox as sandbox_fn

        sandbox_mock = MagicMock()
        sandbox_mock.execute.return_value = {
            "exit_code": 0,
            "stdout": "ok",
            "stderr": "",
        }
        result = asyncio.run(
            sandbox_fn(
                operation="execute",
                code="print(1)",
                context=self._make_ctx(sandbox_mock),
            )
        )
        assert "ok" in result

    def test_upload_operation(self):
        from victor_coding.tools.code_executor_tool import sandbox as sandbox_fn

        sandbox_mock = MagicMock()
        result = asyncio.run(
            sandbox_fn(
                operation="upload",
                file_paths=["/a.py"],
                context=self._make_ctx(sandbox_mock),
            )
        )
        assert "uploaded" in result.lower()

    def test_unknown_operation(self):
        from victor_coding.tools.code_executor_tool import sandbox as sandbox_fn

        result = asyncio.run(
            sandbox_fn(operation="frobnicate", context=self._make_ctx(MagicMock()))
        )
        assert "Unknown operation" in result

    def test_no_context(self):
        from victor_coding.tools.code_executor_tool import sandbox as sandbox_fn

        result = asyncio.run(sandbox_fn(operation="execute", context=None))
        assert "Context not provided" in result
