from __future__ import annotations

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

"""
A tool for executing Python code in a secure, stateful Docker container.

Features:
- Proper context manager support for automatic cleanup (sync and async)
- Container labeling for cleanup identification
- atexit handler for cleanup on Python exit
- Signal handlers for cleanup on SIGINT/SIGTERM
- Container reuse within session
"""

import atexit
import io
import logging
import signal
import tarfile
import os
from pathlib import Path
from typing import List, Optional
import weakref

logger = logging.getLogger(__name__)


# Contextual error types (local definitions — boundary-compliant)
class ResourceError(RuntimeError):
    def __init__(self, resource_type, error=None, suggestion=None):
        message = f"Insufficient {resource_type}"
        if error:
            message += f": {error}"
        if suggestion:
            message += f"\n\n💡 Suggestion: {suggestion}"
        # For backward compatibility with tests
        if resource_type == "Docker" and "not running" in str(error).lower():
            message = "Docker is not running or not installed. This feature requires Docker."
        super().__init__(message)


class ToolExecutionError(RuntimeError):
    def __init__(self, tool_name, operation, suggestion=None, **kwargs):
        message = "Execution session not started. Call start() first."
        if suggestion:
            message += f"\n\n💡 Suggestion: {suggestion}"
        super().__init__(message)


def create_contextual_error(msg, **kwargs):
    return RuntimeError(msg)


# Optional docker import
try:
    import docker
    from docker.models.containers import Container
    from docker.errors import DockerException

    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False
    docker = None  # type: ignore
    Container = None  # type: ignore
    DockerException = Exception  # Fallback

from victor.tools.base import AccessMode, DangerLevel, Priority
from victor.tools.decorators import tool

# Container label for identifying Victor sandbox containers
SANDBOX_CONTAINER_LABEL = "victor.sandbox"
SANDBOX_CONTAINER_VALUE = "code-executor"

# Global registry of active sandbox instances for cleanup
_active_sandboxes: weakref.WeakSet = weakref.WeakSet()
_cleanup_registered = False
_signal_handlers_registered = False
_original_sigint_handler = None
_original_sigterm_handler = None


def cleanup_all_sandboxes() -> int:
    """Clean up all active sandbox containers.

    Returns:
        Number of containers cleaned up
    """
    cleaned = 0
    for sandbox in list(_active_sandboxes):
        try:
            sandbox.stop()
            cleaned += 1
        except Exception as e:
            logger.warning(f"Failed to cleanup sandbox: {e}")
    return cleaned


def startup_cleanup(include_unlabeled: bool = True) -> int:
    """Clean up any orphaned containers from previous sessions at startup.

    This should be called during Victor initialization to ensure no
    stale containers are left running from crashed or interrupted sessions.

    Args:
        include_unlabeled: If True, also removes unlabeled legacy containers
            matching the sandbox profile (python-slim with sleep infinity).
            Default True for thorough cleanup.

    Returns:
        Number of containers cleaned up

    Example:
        # In application startup code:
        from victor.tools.code_executor_tool import startup_cleanup
        cleaned = startup_cleanup()
        if cleaned > 0:
            logger.info(f"Cleaned up {cleaned} orphaned sandbox containers")
    """
    cleaned = cleanup_orphaned_containers(include_unlabeled=include_unlabeled)
    if cleaned > 0:
        logger.info(f"Startup cleanup: removed {cleaned} orphaned sandbox container(s)")
    return cleaned


def cleanup_orphaned_containers(include_unlabeled: bool = False) -> int:
    """Clean up any orphaned Victor sandbox containers.

    This finds and removes any containers labeled as Victor sandboxes
    that may have been left behind from previous sessions.

    Args:
        include_unlabeled: If True, also removes python:*-slim containers
            running 'sleep infinity' that lack labels. Use for cleaning up
            legacy containers from before labeling was implemented.

    Returns:
        Number of containers cleaned up
    """
    if not DOCKER_AVAILABLE:
        return 0

    try:
        client = docker.from_env()
        cleaned = 0

        # First, clean up labeled containers
        containers = client.containers.list(
            all=True,
            filters={"label": f"{SANDBOX_CONTAINER_LABEL}={SANDBOX_CONTAINER_VALUE}"},
        )
        for container in containers:
            try:
                logger.info(f"Removing orphaned sandbox container: {container.short_id}")
                container.remove(force=True)
                cleaned += 1
            except Exception as e:
                logger.warning(f"Failed to remove container {container.short_id}: {e}")

        # Optionally clean up unlabeled legacy containers
        if include_unlabeled:
            # Find containers with 'sleep infinity' command on python images
            # Exclude kubernetes infrastructure containers (not Docker's random names like kind_hugle)
            k8s_name_patterns = (
                "k8s_",  # Kubernetes pods
                "kube-",  # Kubernetes components
                "kind-",  # kind infrastructure (hyphen, not underscore)
                "minikube",
                "desktop-",  # Docker Desktop k8s
            )
            all_containers = client.containers.list(all=True)
            for container in all_containers:
                try:
                    # Skip kubernetes infrastructure containers
                    container_name = container.name.lower()
                    if any(container_name.startswith(p) for p in k8s_name_patterns):
                        continue

                    # Check if it's a python-slim based sandbox container
                    image_tags = container.image.tags
                    is_python_slim = (
                        any("python" in tag and "slim" in tag for tag in image_tags)
                        if image_tags
                        else False
                    )

                    # Also check for untagged python images (dangling)
                    if not image_tags:
                        # Container might be from old python image
                        # Check if command is 'sleep infinity'
                        cmd = container.attrs.get("Config", {}).get("Cmd", [])
                        if cmd and "sleep" in str(cmd) and "infinity" in str(cmd):
                            is_python_slim = True

                    # If it's a python-slim container with sleep infinity command
                    if is_python_slim:
                        cmd = container.attrs.get("Config", {}).get("Cmd", [])
                        if cmd and "sleep" in str(cmd) and "infinity" in str(cmd):
                            logger.info(
                                f"Removing unlabeled sandbox container: {container.short_id}"
                            )
                            container.remove(force=True)
                            cleaned += 1
                except Exception as e:
                    logger.debug(f"Skipping container check: {e}")

        return cleaned
    except Exception as e:
        logger.warning(f"Failed to cleanup orphaned containers: {e}")
        return 0


def _signal_cleanup_handler(signum, frame):
    """Signal handler that cleans up containers before exiting.

    This ensures containers are cleaned up even when the process
    is terminated via SIGINT (Ctrl+C) or SIGTERM.
    """
    global _original_sigint_handler, _original_sigterm_handler

    logger.info(f"Received signal {signum}, cleaning up sandbox containers...")
    cleaned = cleanup_all_sandboxes()
    if cleaned > 0:
        logger.info(f"Cleaned up {cleaned} sandbox container(s)")

    # Call the original handler if it exists
    if signum == signal.SIGINT and _original_sigint_handler:
        if callable(_original_sigint_handler):
            _original_sigint_handler(signum, frame)
        elif _original_sigint_handler == signal.SIG_DFL:
            # Re-raise KeyboardInterrupt for default behavior
            raise KeyboardInterrupt
    elif signum == signal.SIGTERM and _original_sigterm_handler:
        if callable(_original_sigterm_handler):
            _original_sigterm_handler(signum, frame)
        elif _original_sigterm_handler == signal.SIG_DFL:
            # Exit with the signal for default behavior
            import sys

            sys.exit(128 + signum)


def _register_signal_handlers():
    """Register signal handlers for cleanup on SIGINT/SIGTERM."""
    global _signal_handlers_registered, _original_sigint_handler, _original_sigterm_handler

    if _signal_handlers_registered:
        return

    try:
        # Only register in main thread
        import threading

        if threading.current_thread() is not threading.main_thread():
            logger.debug("Not registering signal handlers (not main thread)")
            return

        # Save original handlers
        _original_sigint_handler = signal.getsignal(signal.SIGINT)
        _original_sigterm_handler = signal.getsignal(signal.SIGTERM)

        # Install our handlers
        signal.signal(signal.SIGINT, _signal_cleanup_handler)
        signal.signal(signal.SIGTERM, _signal_cleanup_handler)

        _signal_handlers_registered = True
        logger.debug("Signal handlers registered for sandbox cleanup")
    except (ValueError, OSError) as e:
        # Can fail in some environments (e.g., non-main thread, Windows service)
        logger.debug(f"Could not register signal handlers: {e}")


def _register_atexit_cleanup():
    """Register atexit handler for cleanup (called once)."""
    global _cleanup_registered
    if not _cleanup_registered:
        atexit.register(cleanup_all_sandboxes)
        _cleanup_registered = True
        # Also register signal handlers for graceful termination
        _register_signal_handlers()


class CodeSandbox:
    """Manages a persistent, isolated Docker container for stateful code execution."""

    def __init__(
        self,
        docker_image: str = "python:3.11-slim",
        require_docker: bool = False,
        network_disabled: bool = True,
        memory_limit: str | None = os.getenv("VICTOR_CODE_EXECUTOR_MEM", "512m"),
        cpu_shares: int | None = None,
    ):
        self.docker_image = docker_image
        self.container: Container | None = None
        self.working_dir = "/app"
        self.docker_available = False
        self.docker_client = None
        self.network_disabled = network_disabled
        self.memory_limit = memory_limit
        try:
            self.cpu_shares = (
                int(os.getenv("VICTOR_CODE_EXECUTOR_CPU_SHARES", "256"))
                if cpu_shares is None
                else cpu_shares
            )
        except ValueError:
            self.cpu_shares = None

        if not DOCKER_AVAILABLE:
            if require_docker:
                raise ResourceError(
                    resource_type="Docker",
                    suggestion="Install Docker package: pip install docker\n"
                    "Then install Docker Desktop: https://www.docker.com/products/docker-desktop",
                )
            # Docker package not available - continue without it
            return

        try:
            self.docker_client = docker.from_env()
            self.docker_available = True
        except DockerException:
            if require_docker:
                # Maintain backward compatibility with existing tests
                # while providing helpful context
                raise ResourceError(
                    resource_type="Docker",
                    error=RuntimeError("Docker is not running or not installed"),
                    suggestion="Start Docker Desktop or ensure Docker daemon is running.\n"
                    "Verify: docker ps\n"
                    "Install: https://www.docker.com/products/docker-desktop",
                )
            # Docker not available, but not required - continue without it
            self.docker_client = None

    def __enter__(self) -> "CodeSandbox":
        """Context manager entry - start the container."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - stop and cleanup the container."""
        self.stop()

    async def __aenter__(self) -> "CodeSandbox":
        """Async context manager entry - start the container."""
        self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit - stop and cleanup the container."""
        self.stop()

    def start(self) -> None:
        """Starts the persistent Docker container.

        If Docker fails to start, the code executor will continue in
        degraded mode (no container execution available).
        """
        if not self.docker_available:
            # Docker not available - skip container startup
            return

        if self.container:
            return  # Already started

        # Register for atexit cleanup
        _register_atexit_cleanup()
        _active_sandboxes.add(self)

        try:
            self.docker_client.images.pull(self.docker_image)
            self.container = self.docker_client.containers.run(
                self.docker_image,
                command="sleep infinity",  # Keep the container running
                detach=True,
                working_dir=self.working_dir,
                network_disabled=self.network_disabled,
                mem_limit=self.memory_limit,
                cpu_shares=self.cpu_shares,
                labels={SANDBOX_CONTAINER_LABEL: SANDBOX_CONTAINER_VALUE},
            )
            logger.debug(f"Started sandbox container: {self.container.short_id}")
        except Exception as e:
            # Log the error but don't crash Victor
            # Docker container execution will be unavailable
            logger.warning(
                f"Docker container startup failed: {e}. "
                "Code execution in containers will be unavailable."
            )
            self.container = None
            self.docker_available = False  # Mark Docker as unavailable
            # Remove from active set since startup failed
            _active_sandboxes.discard(self)

    def stop(self) -> None:
        """Stops and removes the Docker container."""
        if not self.docker_available:
            # Docker not available - nothing to stop
            return

        if self.container:
            container_id = self.container.short_id
            try:
                self.container.remove(force=True)
                logger.debug(f"Stopped sandbox container: {container_id}")
            except Exception as e:
                logger.debug(f"Container {container_id} cleanup: {e}")
            self.container = None

        # Remove from active set (safe even if not present)
        _active_sandboxes.discard(self)

    def execute(self, code: str, timeout: int = 60) -> dict:
        """Executes a block of Python code inside the running container."""
        if not self.docker_available:
            return {
                "exit_code": 1,
                "stdout": "",
                "stderr": "Docker is not available. Code execution in containers is disabled.",
            }

        if not self.container:
            raise ToolExecutionError(
                tool_name="code_executor",
                operation="execute_code",
                suggestion="Call sandbox.start() before using execute().\n"
                "Example:\n"
                "  sandbox = CodeSandbox()\n"
                "  sandbox.start()\n"
                "  result = sandbox.execute(code)",
            )

        exec_result = self.container.exec_run(
            ["python", "-c", code],
            demux=True,  # Separate stdout and stderr
        )

        stdout = exec_result.output[0].decode("utf-8") if exec_result.output[0] else ""
        stderr = exec_result.output[1].decode("utf-8") if exec_result.output[1] else ""

        return {
            "exit_code": exec_result.exit_code,
            "stdout": stdout,
            "stderr": stderr,
        }

    def put_files(self, file_paths: List[str]):
        """Copies files from the local filesystem into the container's working dir."""
        if not self.docker_available:
            # Docker not available - skip file operations
            return

        if not self.container:
            raise ToolExecutionError(
                tool_name="code_executor",
                operation="execute_code",
                suggestion="Call sandbox.start() before using execute().\n"
                "Example:\n"
                "  sandbox = CodeSandbox()\n"
                "  sandbox.start()\n"
                "  result = sandbox.execute(code)",
            )

        # Create a tarball in memory
        tar_stream = io.BytesIO()
        with tarfile.open(fileobj=tar_stream, mode="w") as tar:
            for file_path in file_paths:
                p = Path(file_path)
                if not p.exists():
                    raise FileNotFoundError(f"Local file not found: {file_path}")
                tar.add(str(p), arcname=p.name)

        tar_stream.seek(0)
        self.container.put_archive(self.working_dir, tar_stream)

    def get_file(self, remote_path: str) -> bytes:
        """Retrieves a single file from the container."""
        if not self.docker_available:
            # Docker not available - return empty bytes
            return b""

        if not self.container:
            raise ToolExecutionError(
                tool_name="code_executor",
                operation="execute_code",
                suggestion="Call sandbox.start() before using execute().\n"
                "Example:\n"
                "  sandbox = CodeSandbox()\n"
                "  sandbox.start()\n"
                "  result = sandbox.execute(code)",
            )

        bits, _ = self.container.get_archive(remote_path)

        # Read the tar archive from the bits
        with io.BytesIO(b"".join(bits)) as tar_stream:
            with tarfile.open(fileobj=tar_stream, mode="r") as tar:
                # Assuming the tar contains one file
                member = tar.getmembers()[0]
                file_obj = tar.extractfile(member)
                if file_obj:
                    return file_obj.read()
        raise FileNotFoundError(f"File not found in container: {remote_path}")


async def _execute_code(sandbox_instance: CodeSandbox, code: str) -> str:
    """Internal: Execute Python code in sandbox.

    Offloads the synchronous execute() call to a thread pool to avoid
    blocking the event loop during Docker container execution.
    """
    import asyncio

    result = await asyncio.to_thread(sandbox_instance.execute, code)
    output = f"Exit Code: {result['exit_code']}\n"
    if result["stdout"]:
        output += f"--- STDOUT ---\n{result['stdout']}\n"
    if result["stderr"]:
        output += f"--- STDERR ---\n{result['stderr']}\n"
    return output


async def _upload_files(sandbox_instance: CodeSandbox, file_paths: List[str]) -> str:
    """Internal: Upload files to sandbox.

    Offloads the synchronous put_files() call to a thread pool to avoid
    blocking the event loop during Docker file operations.
    """
    import asyncio

    try:
        await asyncio.to_thread(sandbox_instance.put_files, file_paths)
        return f"Successfully uploaded {len(file_paths)} files to the sandbox."
    except Exception as e:
        return f"Error uploading files: {e}"


@tool(
    category="execution",
    priority=Priority.MEDIUM,  # Task-specific code execution
    access_mode=AccessMode.EXECUTE,  # Runs code in container
    danger_level=DangerLevel.HIGH,  # Arbitrary code execution
    keywords=["sandbox", "execute", "python", "container", "docker", "upload", "code"],
)
async def sandbox(
    operation: str,
    code: str = "",
    file_paths: List[str] = None,
    context: dict = None,
) -> str:
    """Unified sandbox operations for code execution in isolated Docker container.

    Operations:
    - "execute": Run Python code in the sandbox
    - "upload": Upload local files to the sandbox

    Args:
        operation: Operation to perform - "execute" or "upload"
        code: Python code to execute (for "execute" operation)
        file_paths: List of local file paths to upload (for "upload" operation)
        context: Tool context provided by orchestrator

    Returns:
        Operation result string
    """
    if context is None:
        return "Error: Context not provided."

    sandbox_instance: CodeSandbox = context.get("code_manager")
    if not sandbox_instance:
        return "Error: CodeSandbox not found in context."

    op = operation.lower().strip()

    if op == "execute":
        if not code:
            return "Error: 'code' parameter required for execute operation."
        return await _execute_code(sandbox_instance, code)

    elif op == "upload":
        if not file_paths:
            return "Error: 'file_paths' parameter required for upload operation."
        return await _upload_files(sandbox_instance, file_paths)

    else:
        return f"Error: Unknown operation '{operation}'. Use 'execute' or 'upload'."
