"""The MCP process backend must not claim policies it cannot enforce."""

from unittest.mock import MagicMock, patch
import sys

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
