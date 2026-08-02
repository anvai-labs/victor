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

"""Sandboxed execution for workflow compute nodes.

Provides Docker and subprocess isolation for compute node execution
based on IsolationConfig settings. Integrates with the existing
sandbox infrastructure in victor.integrations.mcp.sandbox.

Execution Modes:
- none: Direct inline execution (fastest)
- process: Subprocess with resource limits (rlimit)
- docker: Full container isolation (safest)

Example:
    from victor.workflows.sandbox_executor import SandboxedExecutor
    from victor.workflows.isolation import IsolationMapper, IsolationConfig

    executor = SandboxedExecutor()

    # Execute with isolation
    result = await executor.execute(
        command=["python", "compute_stats.py"],
        isolation=IsolationConfig(sandbox_type="process"),
        context={"data_path": "/tmp/data.csv"},
    )
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from victor.workflows.isolation import IsolationConfig, ResourceLimits

logger = logging.getLogger(__name__)


@dataclass
class SandboxExecutionResult:
    """Result of sandboxed execution.

    Attributes:
        success: Whether execution succeeded
        output: Standard output from command
        error: Error message if failed
        exit_code: Process exit code
        duration_seconds: Execution duration
        sandbox_type: Type of sandbox used
    """

    success: bool
    output: str = ""
    error: str = ""
    exit_code: int = 0
    duration_seconds: float = 0.0
    sandbox_type: str = "none"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "exit_code": self.exit_code,
            "duration_seconds": self.duration_seconds,
            "sandbox_type": self.sandbox_type,
        }


# Label applied to every victor-spawned container for identification + cleanup
# (e.g. `docker rm -f $(docker ps -aq --filter label=victor.sandbox=...)`).
SANDBOX_CONTAINER_LABEL = "victor.sandbox"


def build_docker_run_flags(
    *,
    working_dir: Optional[str],
    env: Optional[Dict[str, str]],
    isolation: IsolationConfig,
    limits: Optional[ResourceLimits] = None,
    label_value: str = "code-executor",
    workspace_mount: str = "/workspace",
) -> List[str]:
    """Build the common docker flags shared by ``docker run`` and ``docker create``.

    Returns the flag list that sits between ``["docker", "<run|create>", "--name", N]``
    and ``[image] + command``: the sandbox label, resource limits, network policy,
    filesystem read-only flag, working-dir + extra-volume mounts, and env vars.

    Factored out of ``SandboxedExecutor._execute_docker`` so the evaluation
    container path (`victor.evaluation.container_eval.EvalContainer`) reuses the
    exact same flag assembly instead of duplicating it.

    Args:
        working_dir: Host path to bind-mount at ``workspace_mount`` (the cwd).
        env: Per-call env vars (merged on top of ``isolation.environment``).
        isolation: Isolation config (network/fs/volumes/base env).
        limits: Resource limits; defaults to ``isolation.resource_limits``.
        label_value: Value for the ``victor.sandbox`` label (e.g. ``code-executor``
            for workflow sandboxes, ``eval`` for eval containers).
        workspace_mount: Container path to mount ``working_dir`` at.
    """
    resolved_limits = limits or isolation.resource_limits or ResourceLimits()
    flags: List[str] = []
    # Label for identification + cleanup.
    flags.extend(["--label", f"{SANDBOX_CONTAINER_LABEL}={label_value}"])
    # Resource limits.
    flags.extend(
        [
            "--memory",
            f"{resolved_limits.max_memory_mb}m",
            "--cpus",
            "1.0",
            "--pids-limit",
            str(resolved_limits.max_processes),
        ]
    )
    # Network policy.
    if not isolation.network_allowed:
        flags.extend(["--network", "none"])
    # Filesystem policy.
    if isolation.filesystem_readonly:
        flags.append("--read-only")
    # Working-directory mount.
    if working_dir:
        mount_mode = "ro" if isolation.filesystem_readonly else "rw"
        flags.extend(["-v", f"{working_dir}:{workspace_mount}:{mount_mode}"])
        flags.extend(["-w", workspace_mount])
    # Additional volume mounts.
    for host_path, container_path in isolation.docker_volumes.items():
        flags.extend(["-v", f"{host_path}:{container_path}"])
    # Environment variables (per-call then isolation base).
    for key, value in (env or {}).items():
        flags.extend(["-e", f"{key}={value}"])
    for key, value in isolation.environment.items():
        flags.extend(["-e", f"{key}={value}"])
    return flags


class SandboxedExecutor:
    """Execute commands with configurable isolation.

    Supports three isolation levels:
    - none: Direct execution (no isolation)
    - process: Subprocess with rlimit resource controls
    - docker: Full container isolation

    Example:
        executor = SandboxedExecutor()

        result = await executor.execute(
            command=["python", "script.py"],
            isolation=IsolationConfig(sandbox_type="docker"),
        )
    """

    def __init__(
        self,
        default_docker_image: str = "python:3.11-slim",
        docker_available: Optional[bool] = None,
    ):
        """Initialize executor.

        Args:
            default_docker_image: Default Docker image for container execution
            docker_available: Override Docker availability check
        """
        self._default_image = default_docker_image
        self._docker_available = docker_available
        if docker_available is None:
            self._docker_available = self._check_docker()

    def _check_docker(self) -> bool:
        """Check if Docker is available."""
        try:
            result = subprocess.run(
                ["docker", "version", "--format", "{{.Server.Version}}"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.returncode == 0
        except Exception:
            return False

    async def execute(
        self,
        command: List[str],
        isolation: IsolationConfig,
        working_dir: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        input_data: Optional[str] = None,
    ) -> SandboxExecutionResult:
        """Execute command with specified isolation.

        Args:
            command: Command and arguments to execute
            isolation: Isolation configuration
            working_dir: Working directory for execution
            env: Environment variables
            input_data: Data to pipe to stdin

        Returns:
            SandboxExecutionResult with output and status
        """
        import time

        start_time = time.time()

        if isolation.sandbox_type == "docker":
            if self._docker_available:
                result = await self._execute_docker(
                    command, isolation, working_dir, env, input_data
                )
            else:
                logger.warning("Docker not available, falling back to process isolation")
                result = await self._execute_process(
                    command, isolation, working_dir, env, input_data
                )
        elif isolation.sandbox_type == "process":
            result = await self._execute_process(command, isolation, working_dir, env, input_data)
        else:
            result = await self._execute_inline(command, isolation, working_dir, env, input_data)

        result.duration_seconds = time.time() - start_time
        return result

    async def _execute_inline(
        self,
        command: List[str],
        isolation: IsolationConfig,
        working_dir: Optional[str],
        env: Optional[Dict[str, str]],
        input_data: Optional[str],
    ) -> SandboxExecutionResult:
        """Execute command inline without isolation.

        Args:
            command: Command to execute
            isolation: Isolation config (for timeout)
            working_dir: Working directory
            env: Environment variables
            input_data: Stdin data

        Returns:
            SandboxExecutionResult
        """
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdin=asyncio.subprocess.PIPE if input_data else None,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=working_dir,
                env={**os.environ, **(env or {})},
            )

            timeout = (
                isolation.resource_limits.timeout_seconds if isolation.resource_limits else 60.0
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(input_data.encode() if input_data else None),
                timeout=timeout,
            )

            return SandboxExecutionResult(
                success=process.returncode == 0,
                output=stdout.decode("utf-8", errors="replace"),
                error=(stderr.decode("utf-8", errors="replace") if process.returncode != 0 else ""),
                exit_code=process.returncode or 0,
                sandbox_type="none",
            )

        except asyncio.TimeoutError:
            return SandboxExecutionResult(
                success=False,
                error=f"Command timed out after {timeout}s",
                sandbox_type="none",
            )
        except Exception as e:
            return SandboxExecutionResult(
                success=False,
                error=str(e),
                sandbox_type="none",
            )

    async def _execute_process(
        self,
        command: List[str],
        isolation: IsolationConfig,
        working_dir: Optional[str],
        env: Optional[Dict[str, str]],
        input_data: Optional[str],
    ) -> SandboxExecutionResult:
        """Execute command in sandboxed subprocess with rlimit.

        Args:
            command: Command to execute
            isolation: Isolation config with resource limits
            working_dir: Working directory
            env: Environment variables
            input_data: Stdin data

        Returns:
            SandboxExecutionResult
        """
        from victor.integrations.mcp.sandbox import SandboxConfig, SandboxedProcess

        # Convert isolation config to sandbox config
        limits = isolation.resource_limits or ResourceLimits()
        sandbox_config = SandboxConfig(
            max_memory_mb=limits.max_memory_mb,
            max_cpu_seconds=limits.max_cpu_seconds,
            max_file_descriptors=limits.max_file_descriptors,
            max_processes=limits.max_processes,
            timeout_seconds=limits.timeout_seconds,
            allow_network=isolation.network_allowed,
        )

        sandbox = SandboxedProcess(sandbox_config)

        try:
            # Prepare environment
            exec_env = env or {}
            if not isolation.network_allowed:
                exec_env["VICTOR_NETWORK_DISABLED"] = "1"

            process = await sandbox.start(
                command=command,
                cwd=working_dir or isolation.working_directory,
                env=exec_env,
            )

            # Communicate with process
            stdout, stderr = await asyncio.wait_for(
                asyncio.to_thread(
                    process.communicate,
                    input_data.encode() if input_data else None,
                ),
                timeout=limits.timeout_seconds,
            )

            await sandbox.terminate(process)

            return SandboxExecutionResult(
                success=process.returncode == 0,
                output=stdout.decode("utf-8", errors="replace") if stdout else "",
                error=(
                    stderr.decode("utf-8", errors="replace")
                    if stderr and process.returncode != 0
                    else ""
                ),
                exit_code=process.returncode or 0,
                sandbox_type="process",
            )

        except asyncio.TimeoutError:
            return SandboxExecutionResult(
                success=False,
                error=f"Command timed out after {limits.timeout_seconds}s",
                sandbox_type="process",
            )
        except Exception as e:
            logger.error(f"Process sandbox execution failed: {e}")
            return SandboxExecutionResult(
                success=False,
                error=str(e),
                sandbox_type="process",
            )

    async def _execute_docker(
        self,
        command: List[str],
        isolation: IsolationConfig,
        working_dir: Optional[str],
        env: Optional[Dict[str, str]],
        input_data: Optional[str],
    ) -> SandboxExecutionResult:
        """Execute command in Docker container.

        Args:
            command: Command to execute
            isolation: Isolation config with Docker settings
            working_dir: Working directory to mount
            env: Environment variables
            input_data: Stdin data

        Returns:
            SandboxExecutionResult
        """
        limits = isolation.resource_limits or ResourceLimits()
        image = isolation.docker_image or self._default_image

        # Generate a unique container name for cleanup tracking
        import uuid

        container_name = f"victor-sandbox-{uuid.uuid4().hex[:12]}"

        # Build docker run command. Flags (label/limits/network/mounts/env) are
        # assembled by the shared build_docker_run_flags helper — reused by the
        # eval container path so the two never drift.
        docker_cmd = ["docker", "run", "--rm", "--name", container_name]
        docker_cmd.extend(
            build_docker_run_flags(
                working_dir=working_dir,
                env=env,
                isolation=isolation,
                limits=limits,
                label_value="code-executor",
            )
        )
        docker_cmd.append(image)
        docker_cmd.extend(command)

        process = None
        try:
            process = await asyncio.create_subprocess_exec(
                *docker_cmd,
                stdin=asyncio.subprocess.PIPE if input_data else None,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(input_data.encode() if input_data else None),
                timeout=limits.timeout_seconds + 30,  # Extra time for container startup
            )

            return SandboxExecutionResult(
                success=process.returncode == 0,
                output=stdout.decode("utf-8", errors="replace"),
                error=(stderr.decode("utf-8", errors="replace") if process.returncode != 0 else ""),
                exit_code=process.returncode or 0,
                sandbox_type="docker",
            )

        except asyncio.TimeoutError:
            # On timeout, we need to force-kill the container
            logger.warning(f"Docker execution timed out, killing container {container_name}")
            await self._force_kill_container(container_name)
            return SandboxExecutionResult(
                success=False,
                error=f"Docker execution timed out after {limits.timeout_seconds}s",
                sandbox_type="docker",
            )
        except Exception as e:
            logger.error(f"Docker execution failed: {e}")
            # Also try to clean up on any exception
            await self._force_kill_container(container_name)
            return SandboxExecutionResult(
                success=False,
                error=str(e),
                sandbox_type="docker",
            )
        finally:
            # Ensure process is terminated if still running
            if process and process.returncode is None:
                try:
                    process.kill()
                    await process.wait()
                except Exception:
                    pass

    async def _force_kill_container(self, container_name: str) -> None:
        """Force-kill a Docker container by name.

        Args:
            container_name: Name of the container to kill
        """
        try:
            # First try to stop gracefully with short timeout
            stop_process = await asyncio.create_subprocess_exec(
                "docker",
                "stop",
                "-t",
                "2",
                container_name,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(stop_process.wait(), timeout=5)
        except (asyncio.TimeoutError, Exception):
            pass

        try:
            # Force remove if still exists
            rm_process = await asyncio.create_subprocess_exec(
                "docker",
                "rm",
                "-f",
                container_name,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(rm_process.wait(), timeout=5)
        except (asyncio.TimeoutError, Exception) as e:
            logger.debug(f"Could not remove container {container_name}: {e}")


# Global executor instance
_default_executor: Optional[SandboxedExecutor] = None


def get_sandboxed_executor() -> SandboxedExecutor:
    """Get or create the default sandboxed executor."""
    global _default_executor
    if _default_executor is None:
        _default_executor = SandboxedExecutor()
    return _default_executor


__all__ = [
    "SandboxExecutionResult",
    "SandboxedExecutor",
    "get_sandboxed_executor",
]
