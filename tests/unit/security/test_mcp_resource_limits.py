"""The MCP process backend must not claim policies it cannot enforce."""

import asyncio
import os
import subprocess
import sys
import time
from unittest.mock import MagicMock, patch

import pytest

from victor.integrations.mcp import sandbox


@pytest.mark.parametrize(
    "options",
    [
        {"allow_network": False},
        {"allowed_hosts": ["example.com"]},
        {"allowed_paths": ["/workspace"]},
        {"read_only_paths": ["/data"]},
        {"use_namespace": True},
        {"use_seccomp": True},
    ],
)
async def test_unsupported_isolation_rejects_before_process_or_temp_creation(options):
    handler = sandbox.SandboxedProcess(sandbox.SandboxConfig(**options))
    with (
        patch.object(sandbox.subprocess, "Popen") as launch,
        patch.object(sandbox.tempfile, "mkdtemp") as temp,
    ):
        with pytest.raises(ValueError, match="cannot enforce"):
            await handler.start(["ignored"])
    launch.assert_not_called()
    temp.assert_not_called()


@pytest.mark.parametrize(
    "options",
    [
        {"max_memory_mb": 0},
        {"max_cpu_seconds": -1},
        {"max_file_descriptors": True},
        {"max_processes": 1.5},
        {"timeout_seconds": float("nan")},
        {"timeout_seconds": float("inf")},
        {"graceful_shutdown_seconds": 0},
    ],
)
async def test_invalid_limits_reject_before_launch(options):
    with patch.object(sandbox.subprocess, "Popen") as launch:
        with pytest.raises(ValueError):
            await sandbox.SandboxedProcess(sandbox.SandboxConfig(**options)).start(["ignored"])
    launch.assert_not_called()


async def test_unavailable_resource_backend_cannot_fall_back_to_unrestricted_process():
    with (
        patch.object(sandbox, "resource", None),
        patch.object(sandbox.subprocess, "Popen") as launch,
    ):
        with pytest.raises(RuntimeError, match="POSIX"):
            await sandbox.SandboxedProcess().start(["ignored"])
    launch.assert_not_called()


def test_resource_limit_failure_propagates_out_of_child_setup():
    with patch.object(sandbox.resource, "setrlimit", side_effect=PermissionError("denied")):
        with pytest.raises(PermissionError):
            sandbox._set_resource_limits(sandbox.SandboxConfig())


def test_every_resource_limit_is_applied():
    with patch.object(sandbox.resource, "setrlimit") as set_limit:
        sandbox._set_resource_limits(sandbox.SandboxConfig())
    assert {c.args[0] for c in set_limit.call_args_list} == {
        sandbox.resource.RLIMIT_AS,
        sandbox.resource.RLIMIT_CPU,
        sandbox.resource.RLIMIT_NOFILE,
        sandbox.resource.RLIMIT_NPROC,
    }


def test_root_demotion_clears_supplementary_groups_before_ids():
    import pwd

    calls = MagicMock()
    with (
        patch.object(sandbox.os, "geteuid", side_effect=[0, 65534]),
        patch.object(pwd, "getpwnam", return_value=MagicMock(pw_gid=65534, pw_uid=65534)),
        patch.object(sandbox.os, "setgroups", calls.groups),
        patch.object(sandbox.os, "setgid", calls.gid),
        patch.object(sandbox.os, "setuid", calls.uid),
    ):
        sandbox._drop_privileges()
    assert [c[0] for c in calls.mock_calls] == ["groups", "gid", "uid"]
    calls.groups.assert_called_once_with([])


def test_root_demotion_failure_aborts_instead_of_launching_privileged():
    with (
        patch.object(sandbox.os, "geteuid", return_value=0),
        patch.object(sandbox.os, "setgroups", side_effect=PermissionError("denied")),
    ):
        with pytest.raises(PermissionError):
            sandbox._drop_privileges()


async def test_supported_resource_limits_reach_real_child():
    handler = sandbox.SandboxedProcess(sandbox.SandboxConfig(drop_capabilities=False))
    process = await handler.start(
        [
            sys.executable,
            "-c",
            "import resource; print(resource.getrlimit(resource.RLIMIT_NOFILE)[0])",
        ]
    )
    try:
        stdout, stderr = await handler.communicate(process)
        assert process.returncode == 0, stderr
        assert stdout.strip() == "256"
    finally:
        await handler.terminate(process)


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX process groups")
async def test_terminate_kills_descendants_in_owned_process_group(tmp_path):
    descendant_pid_path = tmp_path / "descendant.pid"
    descendant = f"""import os,pathlib,signal,time
pathlib.Path({str(descendant_pid_path)!r}).write_text(str(os.getpid()))
signal.signal(signal.SIGTERM, signal.SIG_IGN)
time.sleep(30)
"""
    parent = f"""import subprocess,sys,time
subprocess.Popen([sys.executable,'-c',{descendant!r}])
time.sleep(30)
"""
    handler = sandbox.SandboxedProcess(
        sandbox.SandboxConfig(
            drop_capabilities=False,
            graceful_shutdown_seconds=0.1,
            max_processes=4096,
        )
    )
    process = await handler.start([sys.executable, "-c", parent])
    try:
        for _ in range(200):
            if descendant_pid_path.exists():
                break
            await asyncio.sleep(0.01)
        assert descendant_pid_path.exists()
        descendant_pid = int(descendant_pid_path.read_text())

        await handler.terminate(process)

        assert process.poll() is not None
        descendant_stopped = False
        for _ in range(100):
            try:
                os.kill(descendant_pid, 0)
            except ProcessLookupError:
                descendant_stopped = True
                break
            proc_stat = f"/proc/{descendant_pid}/stat"
            if os.path.exists(proc_stat):
                with open(proc_stat) as stat_file:
                    if stat_file.read().split()[2] == "Z":
                        descendant_stopped = True
                        break
            await asyncio.sleep(0.01)
        assert descendant_stopped
    finally:
        if process.returncode is None:
            await handler.terminate(process)


def test_process_tree_signal_rejects_callers_process_group():
    process = MagicMock()
    process.pid = os.getpgrp()
    process.poll.return_value = None

    with patch.object(os, "killpg") as kill_group:
        sandbox.signal_process_tree(process, process_group=os.getpgrp())

    kill_group.assert_not_called()
    process.terminate.assert_called_once_with()


def test_owned_process_group_rejects_mismatched_kernel_group():
    process = MagicMock()
    process.pid = 12345
    process.returncode = None

    with patch.object(os, "getpgid", return_value=54321):
        assert sandbox.owned_process_group(process) is None


def test_reaped_process_invalidates_group_before_signaling():
    process = MagicMock()
    process.pid = 12345
    process.returncode = 0
    group = sandbox.OwnedProcessGroup(leader_pid=12345, pgid=12345)

    with patch.object(os, "killpg") as kill_group:
        sandbox.signal_process_tree(process, force=True, process_group=group)

    assert not group.active
    kill_group.assert_not_called()
    process.kill.assert_called_once_with()


async def test_force_wait_does_not_block_event_loop():
    handler = sandbox.SandboxedProcess(
        sandbox.SandboxConfig(drop_capabilities=False, graceful_shutdown_seconds=0.01)
    )
    process = MagicMock()
    process.pid = 12345
    process.returncode = None

    def blocking_wait(timeout):
        time.sleep(0.15)
        return 0

    process.wait.side_effect = blocking_wait
    handler._processes[process.pid] = process
    heartbeat = asyncio.Event()
    asyncio.get_running_loop().call_later(0.03, heartbeat.set)

    await asyncio.gather(handler.terminate(process), asyncio.wait_for(heartbeat.wait(), 0.1))

    assert heartbeat.is_set()


async def test_uncertain_group_exit_force_kills_and_reaps_direct_child():
    handler = sandbox.SandboxedProcess(
        sandbox.SandboxConfig(drop_capabilities=False, graceful_shutdown_seconds=0.01)
    )
    process = MagicMock()
    process.pid = 12345
    process.returncode = None
    process.wait.side_effect = [subprocess.TimeoutExpired("wait", 1), 0]
    group = sandbox.OwnedProcessGroup(leader_pid=12345, pgid=12345)
    handler._processes[process.pid] = process
    handler._process_groups[process.pid] = group

    with (
        patch.object(sandbox.os, "killpg"),
        patch.object(sandbox, "wait_for_exit_without_reaping", return_value=None),
    ):
        await handler.terminate(process)

    process.kill.assert_called_once_with()
    assert process.wait.call_count == 2
    assert not group.active
    assert process.pid not in handler._processes
