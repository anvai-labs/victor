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

"""Core container-sandbox lifecycle quality bar (FOU-S2, ADR-035 item 5).

Promoted from the victor-coding vertical's ``CodeSandbox``: lazy acquisition
(no Docker contact until first use), ownership (stop/dispose/finalizer remove
the container), self-expiry (idle watchdog + ``auto_remove``), network
disabled by default, resource caps on the container-run invocation, and the
owner-aware reaper. Docker is always mocked here; real-daemon lifecycle lives
in tests/integration/infra/ (skipif no Docker).
"""

from __future__ import annotations

import asyncio
import gc
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import victor.tools.sandbox.container_sandbox as cs
from victor.tools.sandbox.container_sandbox import (
    ContainerSandbox,
    ContainerSandboxUnavailableError,
    find_orphaned_sandboxes,
    process_start_marker,
    reap_orphaned_sandboxes,
    resolve_container_sandbox,
    watchdog_command,
)

MODULE = "victor.tools.sandbox.container_sandbox"


# ---------------------------------------------------------------------------
# Degraded mode: docker package absent or daemon unreachable
# ---------------------------------------------------------------------------


class TestDegradedMode:
    def test_without_docker_package(self):
        with patch(f"{MODULE}.DOCKER_AVAILABLE", False):
            sandbox = ContainerSandbox()
            assert sandbox.docker_available is False
            assert sandbox.docker_client is None
        assert sandbox.container is None

    def test_without_docker_package_require_docker_raises_typed_reason(self):
        with patch(f"{MODULE}.DOCKER_AVAILABLE", False):
            with pytest.raises(ContainerSandboxUnavailableError) as excinfo:
                ContainerSandbox(require_docker=True)
        assert excinfo.value.reason == "binary_missing"
        assert "Docker" in str(excinfo.value)

    def test_connection_failure_degrades_on_first_use(self):
        from docker.errors import DockerException

        with (
            patch(f"{MODULE}.DOCKER_AVAILABLE", True),
            patch(f"{MODULE}.docker") as mock_docker,
        ):
            mock_docker.from_env.side_effect = DockerException("not running")
            sandbox = ContainerSandbox()
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
            with pytest.raises(ContainerSandboxUnavailableError, match="Docker"):
                ContainerSandbox(require_docker=True)

    def test_degraded_lifecycle_is_inert(self):
        with patch(f"{MODULE}.DOCKER_AVAILABLE", False):
            with ContainerSandbox() as sandbox:
                # Local-path validation happens BEFORE any Docker contact.
                with pytest.raises(FileNotFoundError):
                    sandbox.put_files(["/nonexistent"])
                # Downloads refuse typed, never a silent empty-bytes success.
                with pytest.raises(ContainerSandboxUnavailableError) as excinfo:
                    sandbox.get_file("/app/x")
                assert excinfo.value.reason == "binary_missing"
            sandbox.dispose()
        assert sandbox.container is None

    def test_upload_without_daemon_is_a_typed_failure_not_success(self, tmp_path):
        """Codex finding 18 (negative): the historical silent return let
        upload_files format a no-op as success. Without a daemon the upload
        must fail with a typed reason."""
        src = tmp_path / "real.py"
        src.write_text("x = 1")
        with patch(f"{MODULE}.DOCKER_AVAILABLE", False):
            sandbox = ContainerSandbox()
            with pytest.raises(ContainerSandboxUnavailableError) as excinfo:
                sandbox.put_files([str(src)])
        assert excinfo.value.reason == "binary_missing"
        assert sandbox.container is None
        # The async helper surfaces the failure as an error, never as
        # "Successfully uploaded".
        result = asyncio.run(cs.upload_files(sandbox, [str(src)]))
        assert "Successfully uploaded" not in result
        assert "Error uploading files" in result

    def test_get_file_without_docker_is_a_typed_failure_not_success(self):
        """get_file residual of Codex finding 18 (negative): the historical
        silent ``b""`` return let callers read "no Docker" as an empty-but-
        successful download. Without the docker package the download must
        refuse with a typed reason."""
        with patch(f"{MODULE}.DOCKER_AVAILABLE", False):
            sandbox = ContainerSandbox()
            with pytest.raises(ContainerSandboxUnavailableError) as excinfo:
                sandbox.get_file("/app/out.txt")
        assert excinfo.value.reason == "binary_missing"
        assert sandbox.container is None

    def test_get_file_refusal_is_handled_as_a_structured_error_not_a_crash(self):
        """Caller contract: a get_file refusal surfaces as a structured error
        result carrying the typed reason — never as an empty-but-successful
        payload and never as an unhandled crash."""
        with patch(f"{MODULE}.DOCKER_AVAILABLE", False):
            sandbox = ContainerSandbox()

            def download(remote_path: str) -> dict:
                try:
                    data = sandbox.get_file(remote_path)
                except ContainerSandboxUnavailableError as exc:
                    return {"ok": False, "reason": exc.reason, "error": str(exc)}
                return {"ok": True, "data": data}

            result = download("/app/out.txt")
        assert result["ok"] is False
        assert result["reason"] == "binary_missing"
        assert "unavailable" in result["error"].lower()
        assert "data" not in result


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
        ContainerSandbox()
        docker_env.docker.from_env.assert_not_called()
        docker_env.client.containers.run.assert_not_called()

    def test_first_execute_acquires_one_container(self, docker_env):
        sandbox = ContainerSandbox()
        assert sandbox.execute("print('hi')") == {
            "exit_code": 0,
            "stdout": "hello world\n",
            "stderr": "",
        }
        sandbox.execute("print('again')")
        assert docker_env.client.containers.run.call_count == 1

    def test_first_put_files_acquires_one_container(self, docker_env, tmp_path):
        src = tmp_path / "m.py"
        src.write_text("x = 1")
        sandbox = ContainerSandbox()
        docker_env.docker.from_env.assert_not_called()
        sandbox.put_files([str(src)])
        assert docker_env.client.containers.run.call_count == 1

    def test_connect_probe_is_bounded(self, docker_env, monkeypatch):
        monkeypatch.setenv("VICTOR_CODE_EXEC_START_TIMEOUT", "2.5")
        ContainerSandbox().execute("print(1)")
        docker_env.docker.from_env.assert_called_once_with(timeout=2.5)

    def test_start_is_explicit_and_idempotent(self, docker_env):
        sandbox = ContainerSandbox()
        sandbox.start()
        sandbox.start()
        assert docker_env.client.containers.run.call_count == 1

    def test_image_pulled_only_when_missing(self, docker_env):
        from docker.errors import ImageNotFound

        ContainerSandbox().start()
        docker_env.client.images.pull.assert_not_called()
        docker_env.client.images.get.side_effect = ImageNotFound("missing")
        ContainerSandbox(docker_image="python:3.12-slim").start()
        docker_env.client.images.pull.assert_called_once_with("python:3.12-slim")


# ---------------------------------------------------------------------------
# Self-expiry, network default and resource caps (docker-run invocation level)
# ---------------------------------------------------------------------------


class TestRunInvocationQualityBar:
    def test_container_is_self_expiring_and_owner_labelled(self, docker_env):
        ContainerSandbox(idle_ttl_seconds=120).start()
        kwargs = docker_env.client.containers.run.call_args.kwargs
        assert kwargs["auto_remove"] is True and kwargs["init"] is True
        assert kwargs["command"] == watchdog_command(120)
        labels = kwargs["labels"]
        assert labels[cs.SANDBOX_CONTAINER_LABEL] == cs.SANDBOX_CONTAINER_VALUE
        assert labels[cs.OWNER_PID_LABEL] == str(os.getpid())
        assert labels[cs.OWNER_HOST_LABEL] == cs.host_identity()
        assert labels[cs.IDLE_TTL_LABEL] == "120"
        assert "sleep infinity" not in " ".join(kwargs["command"])

    def test_network_is_disabled_by_default(self, docker_env):
        ContainerSandbox().start()
        assert docker_env.client.containers.run.call_args.kwargs["network_disabled"] is True

    def test_network_can_be_explicitly_enabled(self, docker_env):
        ContainerSandbox(network_disabled=False).start()
        assert docker_env.client.containers.run.call_args.kwargs["network_disabled"] is False

    def test_resource_caps_reach_the_run_invocation(self, docker_env):
        ContainerSandbox(memory_limit="256m", cpu_shares=128).start()
        kwargs = docker_env.client.containers.run.call_args.kwargs
        assert kwargs["mem_limit"] == "256m"
        assert kwargs["cpu_shares"] == 128

    def test_cpu_shares_alone_is_a_weight_never_a_quota(self, docker_env):
        """Codex finding 18: cpu_shares is a relative scheduler weight. With
        no explicit quota configured, NO hard CPU limit is promised — no
        quota must reach the run invocation."""
        ContainerSandbox().start()
        assert "cpu_quota" not in docker_env.client.containers.run.call_args.kwargs

    def test_explicit_cpu_quota_reaches_the_run_invocation(self, docker_env):
        """Codex finding 18 (positive): a promised hard CPU limit is enforced
        with an explicit CFS quota on the container-run invocation."""
        ContainerSandbox(cpu_quota=50000).start()  # 0.5 CPU
        kwargs = docker_env.client.containers.run.call_args.kwargs
        assert kwargs["cpu_quota"] == 50000

    def test_cap_defaults_match_the_settings_defaults(self, docker_env):
        ContainerSandbox().start()
        kwargs = docker_env.client.containers.run.call_args.kwargs
        assert kwargs["mem_limit"] == "512m"
        assert kwargs["cpu_shares"] == 256

    def test_watchdog_interval_scales_with_short_ttl(self):
        script = watchdog_command(4)[2]
        assert "sleep 1;" in script and "-gt 4 ]" in script
        assert "sleep 30;" in watchdog_command(900)[2]


# ---------------------------------------------------------------------------
# Ownership: the #1563 zero-leak guarantees
# ---------------------------------------------------------------------------


class TestOwnership:
    def test_stop_removes_and_next_use_reacquires(self, docker_env):
        sandbox = ContainerSandbox()
        sandbox.start()
        first = docker_env.created[0]
        sandbox.stop()
        first.remove.assert_called_once_with(force=True)
        assert sandbox.container is None
        sandbox.execute("print(1)")
        assert len(docker_env.created) == 2

    def test_dispose_is_stop(self, docker_env):
        sandbox = ContainerSandbox()
        sandbox.start()
        sandbox.dispose()
        docker_env.created[0].remove.assert_called_once_with(force=True)

    def test_stop_is_idempotent(self, docker_env):
        sandbox = ContainerSandbox()
        sandbox.start()
        sandbox.stop()
        sandbox.stop()
        docker_env.created[0].remove.assert_called_once_with(force=True)

    def test_abandoned_sandbox_is_removed_by_the_finalizer(self, docker_env):
        """The #1563 guarantee: no stop() call, container still removed."""
        sandbox = ContainerSandbox()
        sandbox.start()
        container = docker_env.created[0]
        del sandbox
        gc.collect()
        container.remove.assert_called_once_with(force=True)

    def test_context_managers_release(self, docker_env):
        with ContainerSandbox() as sandbox:
            assert sandbox.container is docker_env.created[0]
        assert sandbox.container is None

        async def _run():
            async with ContainerSandbox() as s:
                assert s.container is not None
            return s

        assert asyncio.run(_run()).container is None

    def test_expired_container_is_reacquired_transparently(self, docker_env):
        from docker.errors import NotFound

        sandbox = ContainerSandbox()
        sandbox.start()
        docker_env.created[0].reload.side_effect = NotFound("auto-removed after idle TTL")
        assert sandbox.execute("print(1)")["exit_code"] == 0
        assert len(docker_env.created) == 2
        docker_env.created[0].remove.assert_not_called()  # already gone; finalizer detached

    def test_transient_reload_error_retains_ownership(self, docker_env):
        """Codex finding 18: an ambiguous daemon error (transient reload
        failure) must NOT detach the finalizer or spawn a replacement — the
        container may still be alive, and dropping ownership would leak it
        until TTL."""

        def _transient_error(*_args):  # fresh instance per call: no lingering
            raise RuntimeError("connection reset by peer")  # traceback handles

        sandbox = ContainerSandbox()
        sandbox.start()
        container = docker_env.created[0]
        container.reload.side_effect = _transient_error

        result = sandbox.execute("print(1)")
        assert result["exit_code"] == 1  # degraded, retryable — no execution
        container.remove.assert_not_called()
        assert sandbox.container is container  # ownership retained
        assert len(docker_env.created) == 1  # no duplicate replacement spawned

        # The zero-leak guarantee survives the ambiguous error: the finalizer
        # was never detached (peek returns the registered (obj, func, args,
        # kwargs) tuple while attached) and it still targets THIS container,
        # so an explicit stop removes the possibly-live container.
        # (GC-based firing is pinned by
        # test_abandoned_sandbox_is_removed_by_the_finalizer on the clean
        # path; it is environment-sensitive after a caught exception.)
        registry = sandbox._finalizer.peek()
        assert registry is not None
        assert registry[0] is sandbox
        assert registry[2] == (container,)
        sandbox.stop()
        container.remove.assert_called_once_with(force=True)

    def test_failed_start_degrades_and_retries_next_use(self, docker_env):
        docker_env.client.containers.run.side_effect = [RuntimeError("no space"), _mock_container()]
        sandbox = ContainerSandbox()
        result = sandbox.execute("print(1)")
        assert result["exit_code"] == 1 and "no space" in result["stderr"]
        assert sandbox.execute("print(1)")["exit_code"] == 0


# ---------------------------------------------------------------------------
# Execution surface
# ---------------------------------------------------------------------------


class TestExecution:
    def test_code_is_passed_as_argument_never_interpolated(self, docker_env):
        code = """print("it's $HOME; `id`")"""
        sandbox = ContainerSandbox()
        sandbox.execute(code, timeout=7)
        argv = docker_env.created[0].exec_run.call_args.args[0]
        assert argv[:2] == ["sh", "-c"]
        assert argv[3] == code  # $0, untouched by the shell
        # The script is a fixed template: nothing derived from the code enters it.
        assert argv[2] == f'touch {cs.LEASE_PATH}; exec timeout -s KILL 7 python -c "$0"'

    def test_timeout_kill_is_reported(self, docker_env):
        sandbox = ContainerSandbox()
        sandbox.start()
        docker_env.created[0].exec_run.return_value = MagicMock(exit_code=137, output=(None, None))
        result = sandbox.execute("while True: pass", timeout=3)
        assert result["exit_code"] == 137
        assert "exceeded 3s" in result["stderr"]

    def test_stderr_path(self, docker_env):
        sandbox = ContainerSandbox()
        sandbox.start()
        docker_env.created[0].exec_run.return_value = MagicMock(
            exit_code=1, output=(b"", b"ValueError: bad")
        )
        result = sandbox.execute("raise ValueError('bad')")
        assert result == {"exit_code": 1, "stdout": "", "stderr": "ValueError: bad"}

    def test_put_files_uploads_and_refreshes_lease(self, docker_env, tmp_path):
        src = tmp_path / "hello.py"
        src.write_text("print('hello')")
        sandbox = ContainerSandbox()
        sandbox.put_files([str(src)])
        container = docker_env.created[0]
        container.put_archive.assert_called_once()
        container.exec_run.assert_called_with(["touch", cs.LEASE_PATH])

    def test_put_files_missing_local_file(self, docker_env):
        with pytest.raises(FileNotFoundError):
            ContainerSandbox().put_files(["/definitely/not/here.py"])

    def test_get_file_returns_bytes(self, docker_env):
        import io
        import tarfile

        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w") as tar:
            payload = b"file contents"
            info = tarfile.TarInfo("out.txt")
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
        sandbox = ContainerSandbox()
        sandbox.start()
        docker_env.created[0].get_archive.return_value = (
            io.BytesIO(buffer.getvalue()),
            {"name": "out.txt"},
        )
        assert sandbox.get_file("/app/out.txt") == b"file contents"


# ---------------------------------------------------------------------------
# Capability gating (FOU-S2): Docker unavailable → backend not offered;
# required → typed refusal.
# ---------------------------------------------------------------------------


class _ExecutorSettings:
    """Settings-shaped object with the code_executor_* knobs."""

    def __init__(self, **kw):
        self.subprocess = SimpleNamespace(
            code_executor_image=kw.get("code_executor_image", "python:3.11-slim"),
            code_executor_network_disabled=kw.get("code_executor_network_disabled", True),
            code_executor_memory_limit=kw.get("code_executor_memory_limit", "512m"),
            code_executor_cpu_shares=kw.get("code_executor_cpu_shares", 256),
            code_executor_cpu_quota=kw.get("code_executor_cpu_quota", None),
            code_executor_idle_ttl_seconds=kw.get("code_executor_idle_ttl_seconds", 900),
        )


class TestCapabilityGating:
    def test_no_docker_package_backend_is_not_offered(self):
        with patch(f"{MODULE}.DOCKER_AVAILABLE", False):
            assert resolve_container_sandbox() is None

    def test_required_with_no_docker_package_refuses_binary_missing(self):
        with patch(f"{MODULE}.DOCKER_AVAILABLE", False):
            with pytest.raises(ContainerSandboxUnavailableError) as excinfo:
                resolve_container_sandbox(required=True)
        assert excinfo.value.reason == "binary_missing"
        assert "Docker" in str(excinfo.value)

    def test_required_with_unreachable_daemon_refuses_daemon_unreachable(self):
        with (
            patch(f"{MODULE}.DOCKER_AVAILABLE", True),
            patch(f"{MODULE}.docker") as mock_docker,
        ):
            from docker.errors import DockerException

            mock_docker.from_env.side_effect = DockerException("daemon down")
            with pytest.raises(ContainerSandboxUnavailableError) as excinfo:
                resolve_container_sandbox(required=True)
        assert excinfo.value.reason == "daemon_unreachable"

    def test_not_required_does_not_probe_the_daemon(self):
        """Non-required resolution stays lazy: no Docker contact, backend offered."""
        with (
            patch(f"{MODULE}.DOCKER_AVAILABLE", True),
            patch(f"{MODULE}.docker") as mock_docker,
        ):
            sandbox = resolve_container_sandbox()
            mock_docker.from_env.assert_not_called()
        assert isinstance(sandbox, ContainerSandbox)
        assert sandbox.container is None

    def test_settings_reach_the_resolved_sandbox(self, docker_env):
        sandbox = resolve_container_sandbox(
            _ExecutorSettings(
                code_executor_image="python:3.12-slim",
                code_executor_network_disabled=False,
                code_executor_memory_limit="1g",
                code_executor_cpu_shares=512,
                code_executor_idle_ttl_seconds=60,
            )
        )
        assert sandbox.docker_image == "python:3.12-slim"
        assert sandbox.network_disabled is False
        assert sandbox.memory_limit == "1g"
        assert sandbox.cpu_shares == 512
        assert sandbox.idle_ttl_seconds == 60

    def test_cpu_quota_setting_reaches_the_resolved_sandbox(self, docker_env):
        """Codex finding 18: the explicit hard-CPU-limit knob threads through
        the settings cascade to the sandbox (and then to the run invocation)."""
        sandbox = resolve_container_sandbox(_ExecutorSettings(code_executor_cpu_quota=100000))
        assert sandbox.cpu_quota == 100000
        sandbox.start()
        assert docker_env.client.containers.run.call_args.kwargs["cpu_quota"] == 100000

    def test_no_daemon_upload_is_a_typed_failure(self, tmp_path):
        """Codex finding 18 (negative): with the docker package present but
        the daemon unreachable, upload must fail with a typed
        daemon_unreachable reason — never a silent no-op success."""
        from docker.errors import DockerException

        src = tmp_path / "m.py"
        src.write_text("x = 1")
        with (
            patch(f"{MODULE}.DOCKER_AVAILABLE", True),
            patch(f"{MODULE}.docker") as mock_docker,
        ):
            mock_docker.from_env.side_effect = DockerException("daemon down mid-session")
            sandbox = ContainerSandbox()
            with pytest.raises(ContainerSandboxUnavailableError) as excinfo:
                sandbox.put_files([str(src)])
        assert excinfo.value.reason == "daemon_unreachable"

    def test_no_daemon_get_file_is_a_typed_failure(self):
        """get_file residual of Codex finding 18 (negative): with the docker
        package present but the daemon unreachable, download must fail with a
        typed daemon_unreachable reason — never a silent ``b""`` success."""
        from docker.errors import DockerException

        with (
            patch(f"{MODULE}.DOCKER_AVAILABLE", True),
            patch(f"{MODULE}.docker") as mock_docker,
        ):
            mock_docker.from_env.side_effect = DockerException("daemon down mid-session")
            sandbox = ContainerSandbox()
            with pytest.raises(ContainerSandboxUnavailableError) as excinfo:
                sandbox.get_file("/app/out.txt")
        assert excinfo.value.reason == "daemon_unreachable"

    def test_daemon_available_false_without_docker_package(self):
        with patch(f"{MODULE}.DOCKER_AVAILABLE", False):
            assert cs.daemon_available() is False


# ---------------------------------------------------------------------------
# Owner checks and the reaper
# ---------------------------------------------------------------------------


def _labelled(cid, *, host=None, pid=None, start="", legacy_cmd=False, age_s=0.0):
    container = MagicMock()
    container.short_id = cid
    labels = {cs.SANDBOX_CONTAINER_LABEL: cs.SANDBOX_CONTAINER_VALUE}
    if host is not None:
        labels.update({cs.OWNER_HOST_LABEL: host, cs.OWNER_PID_LABEL: str(pid)})
        labels[cs.OWNER_START_LABEL] = start
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
        me = cs.host_identity()
        pid = os.getpid()
        live = _labelled("live", host=me, pid=pid, start=process_start_marker(pid))
        dead = _labelled("dead", host=me, pid=999999)
        reused = _labelled("reused", host=me, pid=pid, start="older-process")
        foreign = _labelled("foreign", host="another-boot-id", pid=999999)
        legacy_old = _labelled("legacy_old", legacy_cmd=True, age_s=7200)
        legacy_young = _labelled("legacy_young", legacy_cmd=True, age_s=60)
        unowned_cmd = _labelled("unowned", age_s=7200)  # labelled, no owner, not sleep-infinity
        client = self._client(live, dead, reused, foreign, legacy_old, legacy_young, unowned_cmd)
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
        me = cs.host_identity()
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
        monkeypatch.setattr(cs, "_reaped_this_process", False)
        monkeypatch.setattr(cs, "reap_orphaned_sandboxes", lambda *a, **k: calls.append(1) or [])
        cs._reap_once(MagicMock())
        cs._reap_once(MagicMock())
        assert calls == [1]


# ---------------------------------------------------------------------------
# Async helpers mirroring the vertical's tool plumbing contract
# ---------------------------------------------------------------------------


class TestAsyncHelpers:
    def test_execute_code_formats_output(self):
        from victor.tools.sandbox.container_sandbox import execute_code

        sandbox = MagicMock()
        sandbox.execute.return_value = {"exit_code": 0, "stdout": "42\n", "stderr": ""}
        result = asyncio.run(execute_code(sandbox, "print(42)"))
        assert "Exit Code: 0" in result
        assert "42\n" in result

    def test_upload_files_success(self):
        from victor.tools.sandbox.container_sandbox import upload_files

        sandbox = MagicMock()
        result = asyncio.run(upload_files(sandbox, ["/a.py", "/b.py"]))
        assert "2 files" in result

    def test_upload_files_error(self):
        from victor.tools.sandbox.container_sandbox import upload_files

        sandbox = MagicMock()
        sandbox.put_files.side_effect = RuntimeError("disk full")
        result = asyncio.run(upload_files(sandbox, ["/a.py"]))
        assert "disk full" in result
