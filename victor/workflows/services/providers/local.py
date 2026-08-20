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

"""Local process service provider for development tools.

Runs services as OS subprocesses with proper lifecycle management.
Useful for local development databases, mock servers, etc.

Example:
    provider = LocalProcessProvider()

    config = ServiceConfig(
        name="mock-api",
        provider="local",
        command=["python", "-m", "http.server", "8080"],
        health_check=HealthCheckConfig.for_http(8080),
    )

    handle = await provider.start(config)
    # ... use service ...
    await provider.stop(handle)
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from victor.workflows.services.definition import (
    HealthCheckConfig,
    ServiceConfig,
    ServiceHandle,
    ServiceStartError,
    ServiceState,
)
from victor.workflows.services.providers.base import BaseServiceProvider

logger = logging.getLogger(__name__)


class LocalProcessProvider(BaseServiceProvider):
    """Service provider using local OS processes.

    Starts services as subprocesses with proper signal handling
    and resource limits (on Unix systems).

    Attributes:
        working_dir: Default working directory for processes
        env: Additional environment variables
    """

    def __init__(
        self,
        working_dir: Optional[Path] = None,
        env: Optional[Dict[str, str]] = None,
    ):
        """Initialize local process provider.

        Args:
            working_dir: Default working directory
            env: Additional environment variables
        """
        self._working_dir = working_dir or Path.cwd()
        self._env = env or {}
        self._processes: Dict[str, asyncio.subprocess.Process] = {}

    async def _do_start(self, config: ServiceConfig) -> ServiceHandle:
        """Start a local process."""
        if not config.command:
            raise ServiceStartError(config.name, "No command specified")

        handle = ServiceHandle.create(config)

        # Build environment
        env = os.environ.copy()
        env.update(self._env)
        env.update(config.environment)

        # Determine working directory
        working_dir = config.working_dir or str(self._working_dir)

        try:
            # Start subprocess
            process = await asyncio.create_subprocess_exec(
                *config.command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=working_dir,
                env=env,
                start_new_session=True,  # Create new process group
            )

            handle.process_id = process.pid
            handle.state = ServiceState.STARTING
            handle.metadata["working_dir"] = working_dir
            handle.metadata["command"] = config.command

            self._processes[handle.service_id] = process

            # Set port mappings (local process uses ports directly)
            for pm in config.ports:
                handle.ports[pm.container_port] = pm.container_port

            logger.info(
                f"Started process PID={process.pid} for '{config.name}' "
                f"(cmd={' '.join(config.command[:3])}...)"
            )

            return handle

        except Exception as e:
            handle.state = ServiceState.FAILED
            handle.error = str(e)
            raise ServiceStartError(config.name, str(e))

    async def _do_stop(self, handle: ServiceHandle, grace_period: float) -> None:
        """Stop a local process gracefully."""
        process = self._processes.get(handle.service_id)
        if not process:
            return

        try:
            # Send SIGTERM for graceful shutdown
            if sys.platform != "win32":
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            else:
                process.terminate()

            # Wait for graceful exit
            try:
                await asyncio.wait_for(process.wait(), timeout=grace_period)
                logger.info(f"Process PID={handle.process_id} terminated gracefully")
            except asyncio.TimeoutError:
                # Force kill
                logger.warning(f"Process PID={handle.process_id} didn't exit, sending SIGKILL")
                if sys.platform != "win32":
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                else:
                    process.kill()
                await process.wait()

        finally:
            del self._processes[handle.service_id]

    async def _do_cleanup(self, handle: ServiceHandle) -> None:
        """Force kill process."""
        process = self._processes.get(handle.service_id)
        if not process:
            return

        try:
            if sys.platform != "win32":
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            else:
                process.kill()
            await process.wait()
        except Exception as e:
            logger.warning(f"Failed to kill process {handle.process_id}: {e}")
        finally:
            self._processes.pop(handle.service_id, None)

    async def get_logs(self, handle: ServiceHandle, tail: int = 100) -> str:
        """Get process stdout/stderr."""
        process = self._processes.get(handle.service_id)
        if not process:
            return "[Process not found]"

        # For running processes, we can't easily get historical logs
        # This would require redirecting to a file
        return f"[Process PID={handle.process_id} running - logs not buffered]"

    async def _run_command_in_service(
        self,
        handle: ServiceHandle,
        command: str,
    ) -> Tuple[int, str]:
        """Run a command (for local process, just run in same env)."""
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=handle.metadata.get("working_dir"),
            )

            stdout, stderr = await process.communicate()
            output = (stdout + stderr).decode("utf-8", errors="replace")

            return process.returncode or 0, output

        except Exception as e:
            return 1, str(e)

    def is_running(self, handle: ServiceHandle) -> bool:
        """Check if process is still running."""
        process = self._processes.get(handle.service_id)
        if not process:
            return False
        return process.returncode is None


class PostgresLocalProvider(LocalProcessProvider):
    """Specialized provider for local PostgreSQL (via pg_ctl)."""

    def __init__(self, pg_data_dir: Optional[Path] = None):
        super().__init__()
        self._pg_data_dir = pg_data_dir or Path.home() / ".victor" / "postgres_data"

    async def _do_start(self, config: ServiceConfig) -> ServiceHandle:
        """Start PostgreSQL using pg_ctl."""
        # Ensure data directory exists and is initialized
        if not self._pg_data_dir.exists():
            logger.info(f"Initializing PostgreSQL data directory: {self._pg_data_dir}")
            await self._init_db()

        # Start postgres
        handle = ServiceHandle.create(config)
        port = config.ports[0].container_port if config.ports else 5432

        cmd = [
            "pg_ctl",
            "start",
            "-D",
            str(self._pg_data_dir),
            "-o",
            f"-p {port}",
            "-l",
            str(self._pg_data_dir / "server.log"),
        ]

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await process.wait()

            if process.returncode != 0:
                stderr = await process.stderr.read()
                raise ServiceStartError(config.name, f"pg_ctl failed: {stderr.decode()}")

            handle.state = ServiceState.STARTING
            handle.ports[5432] = port
            handle.metadata["data_dir"] = str(self._pg_data_dir)

            return handle

        except FileNotFoundError:
            raise ServiceStartError(
                config.name,
                "PostgreSQL not found. Install with: brew install postgresql",
            )

    async def _init_db(self) -> None:
        """Initialize PostgreSQL data directory."""
        self._pg_data_dir.mkdir(parents=True, exist_ok=True)

        process = await asyncio.create_subprocess_exec(
            "initdb",
            "-D",
            str(self._pg_data_dir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await process.wait()

    async def _do_stop(self, handle: ServiceHandle, grace_period: float) -> None:
        """Stop PostgreSQL using pg_ctl."""
        data_dir = handle.metadata.get("data_dir", str(self._pg_data_dir))

        process = await asyncio.create_subprocess_exec(
            "pg_ctl",
            "stop",
            "-D",
            data_dir,
            "-m",
            "fast",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            await asyncio.wait_for(process.wait(), timeout=grace_period)
        except asyncio.TimeoutError:
            # Force stop
            await asyncio.create_subprocess_exec(
                "pg_ctl", "stop", "-D", data_dir, "-m", "immediate"
            )

    async def _do_cleanup(self, handle: ServiceHandle) -> None:
        """Force stop PostgreSQL."""
        data_dir = handle.metadata.get("data_dir", str(self._pg_data_dir))
        await asyncio.create_subprocess_exec("pg_ctl", "stop", "-D", data_dir, "-m", "immediate")

    async def get_logs(self, handle: ServiceHandle, tail: int = 100) -> str:
        """Get PostgreSQL server logs."""
        log_file = self._pg_data_dir / "server.log"
        if log_file.exists():
            lines = log_file.read_text().splitlines()
            return "\n".join(lines[-tail:])
        return "[No log file found]"


class RedisLocalProvider(LocalProcessProvider):
    """Specialized provider for local Redis (via redis-server)."""

    async def _do_start(self, config: ServiceConfig) -> ServiceHandle:
        """Start Redis server."""
        port = config.ports[0].container_port if config.ports else 6379

        # Build command
        cmd = ["redis-server", "--port", str(port)]

        # Add password if configured
        if config.environment.get("REDIS_PASSWORD"):
            cmd.extend(["--requirepass", config.environment["REDIS_PASSWORD"]])

        # Override config command
        original_cmd = config.command
        config.command = cmd

        try:
            return await super()._do_start(config)
        finally:
            config.command = original_cmd
