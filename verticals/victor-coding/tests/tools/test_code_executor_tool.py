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

"""Tests for the code executor sandbox tool (Docker-based code execution).

Covers both degraded (no Docker) and mocked-Docker paths: init, lifecycle,
execute, file transfer, cleanup, and the @tool decorated entry point.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from victor_coding.tools.code_executor_tool import (
    CodeSandbox,
    ResourceError,
)

# ---------------------------------------------------------------------------
# Degraded mode (no Docker)
# ---------------------------------------------------------------------------


class TestDegradedMode:
    """Docker package absent or daemon unreachable: degraded but functional."""

    def test_init_without_docker_package(self):
        with patch("victor_coding.tools.code_executor_tool.DOCKER_AVAILABLE", False):
            sandbox = CodeSandbox()
        assert sandbox.docker_available is False
        assert sandbox.docker_client is None
        assert sandbox.container is None

    def test_init_without_docker_package_require_docker_raises(self):
        with patch("victor_coding.tools.code_executor_tool.DOCKER_AVAILABLE", False):
            with pytest.raises(ResourceError, match="Docker"):
                CodeSandbox(require_docker=True)

    def test_init_docker_connection_failure_degrades(self):
        from docker.errors import DockerException

        with (
            patch("victor_coding.tools.code_executor_tool.DOCKER_AVAILABLE", True),
            patch("victor_coding.tools.code_executor_tool.docker") as mock_docker,
        ):
            mock_docker.from_env.side_effect = DockerException("not running")
            sandbox = CodeSandbox()
        assert sandbox.docker_available is False
        assert sandbox.docker_client is None

    def test_init_docker_connection_failure_require_docker_raises(self):
        from docker.errors import DockerException

        with (
            patch("victor_coding.tools.code_executor_tool.DOCKER_AVAILABLE", True),
            patch("victor_coding.tools.code_executor_tool.docker") as mock_docker,
        ):
            mock_docker.from_env.side_effect = DockerException("daemon down")
            with pytest.raises(ResourceError, match="Docker"):
                CodeSandbox(require_docker=True)

    def test_start_noop_when_degraded(self):
        with patch("victor_coding.tools.code_executor_tool.DOCKER_AVAILABLE", False):
            sandbox = CodeSandbox()
        sandbox.start()
        assert sandbox.container is None

    def test_stop_noop_when_degraded(self):
        with patch("victor_coding.tools.code_executor_tool.DOCKER_AVAILABLE", False):
            sandbox = CodeSandbox()
        sandbox.stop()
        assert sandbox.container is None

    def test_execute_returns_error_when_degraded(self):
        with patch("victor_coding.tools.code_executor_tool.DOCKER_AVAILABLE", False):
            sandbox = CodeSandbox()
        result = sandbox.execute("print('hello')")
        assert result["exit_code"] == 1
        assert "not available" in result["stderr"].lower()

    def test_context_manager_degraded(self):
        with patch("victor_coding.tools.code_executor_tool.DOCKER_AVAILABLE", False):
            with CodeSandbox() as sandbox:
                assert sandbox.docker_available is False
        assert sandbox.container is None


# ---------------------------------------------------------------------------
# With mocked Docker
# ---------------------------------------------------------------------------


def _mock_container(**overrides):
    container = MagicMock()
    container.short_id = "abc123"
    container.exec_run.return_value = MagicMock(
        exit_code=0,
        output=(b"hello world\n", b""),
    )
    for key, value in overrides.items():
        setattr(container, key, value)
    return container


@pytest.fixture
def docker_env():
    """Patch the docker module with a fully mocked environment."""
    with (
        patch("victor_coding.tools.code_executor_tool.DOCKER_AVAILABLE", True),
        patch("victor_coding.tools.code_executor_tool.docker") as mock_docker,
    ):
        container = _mock_container()
        mock_docker.from_env.return_value.images.pull.return_value = None
        mock_docker.from_env.return_value.containers.run.return_value = container
        yield mock_docker, container


class TestWithMockedDocker:
    def test_init_connects(self, docker_env):
        mock_docker, _container = docker_env
        sandbox = CodeSandbox()
        assert sandbox.docker_available is True
        assert sandbox.docker_client is not None

    def test_start_pulls_image_and_runs_container(self, docker_env):
        mock_docker, container = docker_env
        sandbox = CodeSandbox()
        sandbox.start()
        assert sandbox.container is container
        mock_docker.from_env.return_value.images.pull.assert_called_once_with("python:3.11-slim")

    def test_start_idempotent(self, docker_env):
        mock_docker, container = docker_env
        sandbox = CodeSandbox()
        sandbox.start()
        sandbox.start()
        assert mock_docker.from_env.return_value.containers.run.call_count == 1

    def test_stop_removes_container(self, docker_env):
        mock_docker, container = docker_env
        sandbox = CodeSandbox()
        sandbox.start()
        sandbox.stop()
        container.remove.assert_called_once_with(force=True)
        assert sandbox.container is None

    def test_execute_returns_demuxed_result(self, docker_env):
        mock_docker, container = docker_env
        sandbox = CodeSandbox()
        sandbox.start()
        result = sandbox.execute("print('hi')")
        assert result == {"exit_code": 0, "stdout": "hello world\n", "stderr": ""}

    def test_execute_stderr_path(self, docker_env):
        mock_docker, container = docker_env
        container.exec_run.return_value = MagicMock(
            exit_code=1,
            output=(b"", b"ValueError: bad"),
        )
        sandbox = CodeSandbox()
        sandbox.start()
        result = sandbox.execute("raise ValueError('bad')")
        assert result["exit_code"] == 1
        assert result["stdout"] == ""
        assert "ValueError" in result["stderr"]

    def test_context_manager_starts_and_stops(self, docker_env):
        mock_docker, container = docker_env
        sandbox = CodeSandbox()
        sandbox.start()
        with sandbox:
            assert sandbox.container is container
        assert sandbox.container is None

    def test_async_context_manager(self, docker_env):
        mock_docker, container = docker_env

        async def _run():
            async with CodeSandbox() as sandbox:
                assert sandbox.container is container
            assert sandbox.container is None

        asyncio.run(_run())

    def test_put_files_calls_container(self, docker_env, tmp_path):
        mock_docker, container = docker_env
        src = tmp_path / "hello.py"
        src.write_text("print('hello')")
        sandbox = CodeSandbox()
        sandbox.start()
        sandbox.put_files([str(src)])
        container.put_archive.assert_called_once()

    def test_get_file_returns_bytes(self, docker_env):
        import io
        import tarfile as tarfile_mod

        mock_docker, container = docker_env
        # Build a real in-memory tar so the extraction path succeeds.
        tar_buffer = io.BytesIO()
        with tarfile_mod.open(fileobj=tar_buffer, mode="w") as tar:
            payload = b"file contents"
            info = tarfile_mod.TarInfo("out.txt")
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
        bits = io.BytesIO(tar_buffer.getvalue())
        container.get_archive.return_value = (bits, {"name": "out.txt"})

        sandbox = CodeSandbox()
        sandbox.start()
        data = sandbox.get_file("/app/out.txt")
        assert data == b"file contents"


# ---------------------------------------------------------------------------
# Async helpers
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


# ---------------------------------------------------------------------------
# Signal handlers and atexit
# ---------------------------------------------------------------------------


class TestCleanupRegistration:
    def test_register_atexit_is_idempotent(self):
        from victor_coding.tools.code_executor_tool import _register_atexit_cleanup

        # Calling twice must not raise
        _register_atexit_cleanup()
        _register_atexit_cleanup()

    def test_register_signal_handlers_is_idempotent(self):
        from victor_coding.tools.code_executor_tool import _register_signal_handlers

        _register_signal_handlers()
        _register_signal_handlers()
