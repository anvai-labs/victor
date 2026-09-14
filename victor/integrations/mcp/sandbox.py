# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""MCP subprocess resource limits with explicit capability checks.

This backend applies POSIX rlimits and optional root user demotion. It is not
an OS isolation boundary. Filesystem, network, namespace and seccomp policies
require an external sandbox/container; unsupported policies fail before launch.
"""

from __future__ import annotations

import asyncio
import logging
import os
import math
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

try:
    import resource
except ImportError:  # Windows can import MCP; resource-limited launch is unsupported.
    resource = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


@dataclass
class SandboxConfig:
    """Configuration for subprocess sandboxing."""

    # Resource limits
    max_memory_mb: int = 512  # Maximum memory (MB)
    max_cpu_seconds: int = 300  # Maximum CPU time (seconds)
    max_file_descriptors: int = 256  # Maximum open files
    max_processes: int = 32  # Maximum child processes

    # Execution limits
    timeout_seconds: float = 60.0  # Hard timeout for operations
    graceful_shutdown_seconds: float = 5.0  # Time to wait for graceful exit

    # Filesystem restrictions
    allowed_paths: List[str] = field(default_factory=list)  # Writable paths
    read_only_paths: List[str] = field(default_factory=list)  # Read-only paths
    temp_dir: Optional[str] = None  # Custom temp directory

    # Network restrictions (Linux only)
    allow_network: bool = True  # Allow network access
    allowed_hosts: List[str] = field(default_factory=list)  # Allowed hosts

    # Isolation level
    use_namespace: bool = False  # Requires an external isolation backend
    use_seccomp: bool = False  # Requires an external isolation backend
    drop_capabilities: bool = True  # Demote root to nobody; not full capability isolation


def _validate_sandbox_config(config: SandboxConfig) -> None:
    """Reject promises this backend cannot enforce before creating a child."""
    if resource is None or sys.platform not in {"linux", "darwin"}:
        raise RuntimeError("MCP resource limits require a supported POSIX platform")
    if (
        config.allowed_paths
        or config.read_only_paths
        or not config.allow_network
        or config.allowed_hosts
        or config.use_namespace
        or config.use_seccomp
    ):
        raise ValueError(
            "MCP process backend cannot enforce filesystem, network, namespace or seccomp "
            "isolation; use an external sandbox/container for these policies"
        )
    for name in ("max_memory_mb", "max_cpu_seconds", "max_file_descriptors", "max_processes"):
        value = getattr(config, name)
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    for name in ("timeout_seconds", "graceful_shutdown_seconds"):
        value = getattr(config, name)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")


def _set_resource_limits(config: SandboxConfig) -> None:
    """Apply every configured rlimit; a failure aborts child exec."""
    if resource is None:
        raise RuntimeError("POSIX resource limits are unavailable")
    mem_bytes = config.max_memory_mb * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
    resource.setrlimit(resource.RLIMIT_CPU, (config.max_cpu_seconds, config.max_cpu_seconds + 10))
    resource.setrlimit(
        resource.RLIMIT_NOFILE, (config.max_file_descriptors, config.max_file_descriptors)
    )
    resource.setrlimit(resource.RLIMIT_NPROC, (config.max_processes, config.max_processes))


def _drop_privileges() -> None:
    """Demote root and clear supplementary groups; never continue after failure."""
    if os.geteuid() == 0:
        import pwd

        nobody = pwd.getpwnam("nobody")
        os.setgroups([])
        os.setgid(nobody.pw_gid)
        os.setuid(nobody.pw_uid)
        if os.geteuid() == 0:
            raise RuntimeError("MCP child remained root after privilege demotion")


def _setup_sandbox_env(config: SandboxConfig) -> Dict[str, str]:
    """Set up sandboxed environment variables.

    Args:
        config: Sandbox configuration

    Returns:
        Environment dictionary for subprocess
    """
    env = os.environ.copy()

    # Clear sensitive environment variables — uses consolidated list.
    from victor.security.env_filtering import SENSITIVE_ENV_VARS

    for var in SENSITIVE_ENV_VARS:
        env.pop(var, None)

    # Set custom temp directory if specified
    if config.temp_dir:
        env["TMPDIR"] = config.temp_dir
        env["TEMP"] = config.temp_dir
        env["TMP"] = config.temp_dir

    # Mark as sandboxed
    env["VICTOR_SANDBOXED"] = "1"

    return env


class SandboxedProcess:
    """Subprocess wrapper with resource limits; external isolation is separate.

    Example:
        config = SandboxConfig(max_memory_mb=256, timeout_seconds=30)
        sandbox = SandboxedProcess(config)

        proc = await sandbox.start(["python", "mcp_server.py"])
        output = await sandbox.communicate(proc, "input data")
        await sandbox.terminate(proc)
    """

    def __init__(self, config: Optional[SandboxConfig] = None):
        """Initialize sandboxed process handler.

        Args:
            config: Sandbox configuration
        """
        self.config = config or SandboxConfig()
        self._processes: Dict[int, subprocess.Popen] = {}

    def _create_preexec_fn(self) -> Callable[[], None]:
        """Create preexec function for subprocess.

        Returns:
            Function to run before exec in child process
        """
        config = self.config

        def preexec():
            # Set resource limits
            _set_resource_limits(config)

            # Drop privileges if root
            if config.drop_capabilities:
                _drop_privileges()

        return preexec

    async def start(
        self,
        command: List[str],
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> subprocess.Popen:
        """Start a sandboxed subprocess.

        Args:
            command: Command to execute
            cwd: Working directory
            env: Additional environment variables

        Returns:
            Subprocess handle
        """
        _validate_sandbox_config(self.config)
        # Prepare environment
        sandbox_env = _setup_sandbox_env(self.config)
        if env:
            sandbox_env.update(env)

        # Create temp directory if needed
        temp_dir = None
        if self.config.temp_dir is None:
            temp_dir = tempfile.mkdtemp(prefix="mcp_sandbox_")
            sandbox_env["TMPDIR"] = temp_dir

        try:
            process = await self._start_basic(command, cwd, sandbox_env)

            self._processes[process.pid] = process
            logger.info(f"Started sandboxed process {process.pid}: {command[0]}")

            return process

        except Exception as e:
            logger.error(f"Failed to start sandboxed process: {e}")
            # Clean up temp directory
            if temp_dir:
                try:
                    os.rmdir(temp_dir)
                except Exception:
                    pass
            raise

    async def _start_basic(
        self,
        command: List[str],
        cwd: Optional[str],
        env: Dict[str, str],
    ) -> subprocess.Popen:
        """Start process with basic resource limits.

        Args:
            command: Command to execute
            cwd: Working directory
            env: Environment variables

        Returns:
            Subprocess handle
        """
        return subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd,
            env=env,
            text=True,
            bufsize=1,
            preexec_fn=self._create_preexec_fn(),
            start_new_session=True,
        )

    async def communicate(
        self,
        process: subprocess.Popen,
        input_data: Optional[str] = None,
    ) -> tuple[str, str]:
        """Communicate with sandboxed process.

        Args:
            process: Subprocess handle
            input_data: Data to send to stdin

        Returns:
            Tuple of (stdout, stderr)
        """
        try:
            stdout, stderr = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: process.communicate(input_data),
                ),
                timeout=self.config.timeout_seconds,
            )
            return stdout or "", stderr or ""

        except asyncio.TimeoutError:
            logger.warning(f"Process {process.pid} timed out")
            await self.terminate(process)
            raise

    async def terminate(self, process: subprocess.Popen) -> None:
        """Terminate sandboxed process gracefully.

        Args:
            process: Subprocess handle
        """
        if process.poll() is not None:
            # Already terminated
            self._processes.pop(process.pid, None)
            return

        try:
            # Send SIGTERM first
            process.terminate()

            try:
                await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: process.wait(timeout=self.config.graceful_shutdown_seconds),
                    ),
                    timeout=self.config.graceful_shutdown_seconds + 1,
                )
            except (asyncio.TimeoutError, subprocess.TimeoutExpired):
                # Force kill
                logger.warning(f"Force killing process {process.pid}")
                process.kill()
                process.wait(timeout=1)

        except Exception as e:
            logger.error(f"Error terminating process {process.pid}: {e}")
            try:
                process.kill()
            except Exception:
                pass

        finally:
            self._processes.pop(process.pid, None)

    async def terminate_all(self) -> None:
        """Terminate all sandboxed processes."""
        pids = list(self._processes.keys())
        for pid in pids:
            process = self._processes.get(pid)
            if process:
                await self.terminate(process)

    def get_stats(self) -> Dict[str, Any]:
        """Get sandbox statistics.

        Returns:
            Dictionary with sandbox stats
        """
        active = [pid for pid, p in self._processes.items() if p.poll() is None]
        return {
            "active_processes": len(active),
            "pids": active,
            "config": {
                "max_memory_mb": self.config.max_memory_mb,
                "max_cpu_seconds": self.config.max_cpu_seconds,
                "timeout_seconds": self.config.timeout_seconds,
                "allow_network": self.config.allow_network,
            },
        }


# Factory function for creating sandboxed MCP client
def create_sandboxed_mcp_client(
    config: Optional[SandboxConfig] = None,
) -> "Any":  # Returns MCPClient subclass (locally defined)
    """Create an MCP client with sandboxing.

    Args:
        config: Sandbox configuration

    Returns:
        SandboxedMCPClient instance
    """
    from victor.integrations.mcp.client import MCPClient

    class SandboxedMCPClient(MCPClient):
        """MCP client with sandboxed subprocess."""

        def __init__(self, sandbox_config: Optional[SandboxConfig] = None, **kwargs):
            super().__init__(**kwargs)
            self._sandbox = SandboxedProcess(sandbox_config)

        async def connect(self, command: List[str]) -> bool:
            """Connect using sandboxed process."""
            self._command = command

            try:
                self.process = await self._sandbox.start(command)
                success = await self.initialize()

                if success:
                    self._running = True
                    return True

                await self._sandbox.terminate(self.process)
                return False

            except Exception as e:
                logger.error(f"Sandboxed connection failed: {e}")
                return False

        def disconnect(self, reason: Optional[str] = None) -> None:
            """Disconnect and cleanup sandbox."""
            self._running = False

            if self.process:
                asyncio.create_task(self._sandbox.terminate(self.process))
                self.process = None
                self.initialized = False

    return SandboxedMCPClient(config)
