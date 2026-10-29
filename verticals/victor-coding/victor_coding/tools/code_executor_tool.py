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

"""Docker sandbox for executing Python code (the ``sandbox`` tool).

The lifecycle is designed so sandbox containers cannot accumulate:

- **Lazy**: constructing a :class:`CodeSandbox` makes no Docker contact and
  starts nothing. A container is acquired on first ``execute``/``put_files``
  (or an explicit ``start``), so bootstrap, tests and sessions that never run
  code never touch Docker.
- **Owned**: an acquired container is removed by ``stop``/``dispose`` and by a
  ``weakref.finalize`` that fires when the sandbox is garbage-collected or the
  interpreter exits.
- **Self-expiring**: the container runs an idle watchdog instead of
  ``sleep infinity`` and is started with ``auto_remove``. It deletes itself
  after ``idle_ttl_seconds`` without use, even if its owner was SIGKILLed.
  An expired container is transparently re-acquired on next use (uploaded
  files do not survive expiry).
- **Reapable**: containers carry owner labels (host boot id, pid, process
  start time). :func:`reap_orphaned_sandboxes` removes only containers whose
  owner on this host is dead, never a live sandbox of another process.
"""

import io
import logging
import os
import socket
import tarfile
import threading
import time
import weakref
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    from docker.errors import DockerException, NotFound
    from docker.models.containers import Container

    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False
    docker = None  # type: ignore
    Container = None  # type: ignore
    DockerException = Exception  # Fallback
    NotFound = Exception  # Fallback

from victor.tools.base import AccessMode, DangerLevel, Priority
from victor.tools.decorators import tool

# Labels identifying Victor sandbox containers and their owner
SANDBOX_CONTAINER_LABEL = "victor.sandbox"
SANDBOX_CONTAINER_VALUE = "code-executor"
OWNER_HOST_LABEL = "victor.owner.host"
OWNER_PID_LABEL = "victor.owner.pid"
OWNER_START_LABEL = "victor.owner.start"
IDLE_TTL_LABEL = "victor.idle_ttl"

DEFAULT_IMAGE = "python:3.11-slim"
DEFAULT_IDLE_TTL_SECONDS = 900
# Pre-redesign containers ran `sleep infinity` without owner labels; the
# automatic reaper only removes ones older than this (an old-version session
# may still be using a younger one). `victor tools cleanup-containers
# --include-unlabeled` removes them regardless of age.
LEGACY_AUTO_REAP_MIN_AGE_SECONDS = 24 * 3600
LEASE_PATH = "/tmp/.victor_lease"
_API_TIMEOUT_SECONDS = 120
_UNSET: Any = object()


def _connect_timeout() -> float:
    """Budget for the Docker version probe (Docker Desktop closed = 60s hang)."""
    try:
        return float(os.getenv("VICTOR_CODE_EXEC_START_TIMEOUT", "5.0"))
    except ValueError:
        return 5.0


def host_identity() -> str:
    """Identity of this kernel boot; owners on other hosts are never reaped."""
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        return socket.gethostname()


def process_start_marker(pid: int) -> str:
    """Kernel start time of ``pid`` (distinguishes a reused PID); "" if unknown."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return ""
    # comm (field 2) may contain spaces and parens: split after the last ')'.
    fields = stat.rsplit(")", 1)[1].split()
    return fields[19] if len(fields) > 19 else ""  # field 22: starttime


def owner_alive(pid: int, start: str) -> bool:
    """True unless ``pid`` is provably gone or now belongs to another process."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass  # exists, owned by another user
    except OSError:
        return True  # cannot tell: never reap on doubt
    current = process_start_marker(pid)
    return not (start and current and current != start)


def watchdog_command(idle_ttl_seconds: int) -> List[str]:
    """Container command: exit once the lease is older than the idle TTL."""
    ttl = max(1, int(idle_ttl_seconds))
    interval = max(1, min(30, ttl // 4))
    script = (
        f"touch {LEASE_PATH}; "
        f"while sleep {interval}; do "
        f"[ $(( $(date +%s) - $(stat -c %Y {LEASE_PATH} 2>/dev/null || echo 0) )) "
        f"-gt {ttl} ] && exit 0; "
        "done"
    )
    return ["sh", "-c", script]


def _remove_container(container: Any) -> None:
    """Finalizer target: remove ``container``; never raises (TTL covers misses)."""
    try:
        container.remove(force=True)
    except Exception:  # noqa: BLE001 - already gone / daemon unreachable
        pass


def _age_seconds(container: Any) -> float:
    created = (getattr(container, "attrs", None) or {}).get("Created", "")
    try:
        # Docker reports nanoseconds; fromisoformat accepts at most microseconds.
        head, _, frac = created.rstrip("Z").partition(".")
        stamp = datetime.fromisoformat(f"{head}.{(frac + '000000')[:6]}+00:00")
    except ValueError:
        return 0.0
    return time.time() - stamp.timestamp()


def _is_legacy_sandbox(container: Any) -> bool:
    cmd = " ".join(((getattr(container, "attrs", None) or {}).get("Config") or {}).get("Cmd") or [])
    return "sleep" in cmd and "infinity" in cmd


def _docker_client_or_none(client: Any) -> Any:
    if client is not None:
        return client
    if not DOCKER_AVAILABLE:
        return None
    try:
        return docker.from_env(timeout=_connect_timeout())
    except Exception:  # noqa: BLE001 - no daemon, nothing to inspect
        return None


def find_orphaned_sandboxes(
    client: Any = None,
    *,
    include_legacy: bool = True,
    legacy_min_age_seconds: float = 0.0,
) -> List[Any]:
    """Sandbox containers whose owner is gone (the reaper's decision, no side effects).

    A container qualifies only when its owner is on this host and is dead (or
    its PID now belongs to another process), or when it is a pre-redesign
    ``sleep infinity`` container without owner labels that is at least
    ``legacy_min_age_seconds`` old. Containers owned by another host are left
    to their own idle TTL; a live owner's container never qualifies.
    """
    client = _docker_client_or_none(client)
    if client is None:
        return []
    try:
        containers = client.containers.list(
            all=True, filters={"label": f"{SANDBOX_CONTAINER_LABEL}={SANDBOX_CONTAINER_VALUE}"}
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("Sandbox reaper could not list containers: %s", exc)
        return []

    me = host_identity()
    orphans: List[Any] = []
    for container in containers:
        labels = getattr(container, "labels", None) or {}
        host = labels.get(OWNER_HOST_LABEL)
        if host is None:
            if not include_legacy or not _is_legacy_sandbox(container):
                continue
            if _age_seconds(container) < legacy_min_age_seconds:
                continue
        elif host != me:
            continue
        else:
            try:
                pid = int(labels.get(OWNER_PID_LABEL, "0"))
            except ValueError:
                pid = 0
            if owner_alive(pid, labels.get(OWNER_START_LABEL, "")):
                continue
        orphans.append(container)
    return orphans


def reap_orphaned_sandboxes(
    client: Any = None,
    *,
    dry_run: bool = False,
    include_legacy: bool = True,
    legacy_min_age_seconds: float = 0.0,
) -> List[str]:
    """Remove the containers :func:`find_orphaned_sandboxes` selects; return short ids."""
    orphans = find_orphaned_sandboxes(
        client, include_legacy=include_legacy, legacy_min_age_seconds=legacy_min_age_seconds
    )
    if not dry_run:
        for container in orphans:
            _remove_container(container)
    return [container.short_id for container in orphans]


_reap_lock = threading.Lock()
_reaped_this_process = False


def _reap_once(client: Any) -> None:
    """Reap orphans once per process, on first container acquisition."""
    global _reaped_this_process
    with _reap_lock:
        if _reaped_this_process:
            return
        _reaped_this_process = True
    reaped = reap_orphaned_sandboxes(
        client, legacy_min_age_seconds=LEGACY_AUTO_REAP_MIN_AGE_SECONDS
    )
    if reaped:
        logger.info("Removed %d orphaned sandbox container(s)", len(reaped))


class CodeSandbox:
    """Lazily acquired, owned, self-expiring Docker container for code execution."""

    def __init__(
        self,
        docker_image: str = DEFAULT_IMAGE,
        require_docker: bool = False,
        network_disabled: bool = True,
        memory_limit: Optional[str] = _UNSET,
        cpu_shares: Optional[int] = None,
        idle_ttl_seconds: int = DEFAULT_IDLE_TTL_SECONDS,
    ):
        self.docker_image = docker_image
        self.container: Optional[Container] = None
        self.working_dir = "/app"
        self.network_disabled = network_disabled
        self.memory_limit = (
            os.getenv("VICTOR_CODE_EXECUTOR_MEM", "512m")
            if memory_limit is _UNSET
            else memory_limit
        )
        try:
            self.cpu_shares = (
                int(os.getenv("VICTOR_CODE_EXECUTOR_CPU_SHARES", "256"))
                if cpu_shares is None
                else cpu_shares
            )
        except ValueError:
            self.cpu_shares = None
        self.idle_ttl_seconds = max(1, int(idle_ttl_seconds))
        self._lock = threading.RLock()
        self._client: Any = None
        self._probed = False
        self._finalizer: Optional[weakref.finalize] = None

        if not DOCKER_AVAILABLE:
            self._probed = True
            if require_docker:
                raise ResourceError(
                    resource_type="Docker",
                    suggestion="Install Docker package: pip install docker\n"
                    "Then install Docker Desktop: https://www.docker.com/products/docker-desktop",
                )
            return
        if require_docker and self.docker_client is None:
            raise ResourceError(
                resource_type="Docker",
                error=RuntimeError("Docker is not running or not installed"),
                suggestion="Start Docker Desktop or ensure Docker daemon is running.\n"
                "Verify: docker ps\n"
                "Install: https://www.docker.com/products/docker-desktop",
            )

    # -- Docker connection (lazy, bounded) ---------------------------------

    @property
    def docker_client(self) -> Any:
        """Connected Docker client, probed on first access; None if unavailable."""
        with self._lock:
            if not self._probed:
                self._probed = True
                try:
                    client = docker.from_env(timeout=_connect_timeout())
                    client.api.timeout = _API_TIMEOUT_SECONDS
                    self._client = client
                except Exception as exc:  # noqa: BLE001 - degraded mode covers all
                    logger.warning(
                        "Code execution sandbox unavailable (%s). Start Docker, or tune "
                        "the connect budget via VICTOR_CODE_EXEC_START_TIMEOUT.",
                        exc,
                    )
            return self._client

    @property
    def docker_available(self) -> bool:
        return self.docker_client is not None

    # -- Container lifecycle -------------------------------------------------

    def _labels(self) -> Dict[str, str]:
        pid = os.getpid()
        return {
            SANDBOX_CONTAINER_LABEL: SANDBOX_CONTAINER_VALUE,
            OWNER_HOST_LABEL: host_identity(),
            OWNER_PID_LABEL: str(pid),
            OWNER_START_LABEL: process_start_marker(pid),
            IDLE_TTL_LABEL: str(self.idle_ttl_seconds),
        }

    def _ensure_image(self, client: Any) -> None:
        try:
            client.images.get(self.docker_image)
        except NotFound:
            client.images.pull(self.docker_image)

    def _forget_container(self) -> None:
        if self._finalizer is not None:
            self._finalizer.detach()
        self._finalizer = None
        self.container = None

    def _ensure_container(self) -> Optional[Any]:
        """Return a live container, acquiring one if needed; None if Docker is unavailable."""
        with self._lock:
            client = self.docker_client
            if client is None:
                return None
            if self.container is not None:
                try:
                    self.container.reload()
                    if self.container.status in ("created", "running", "restarting"):
                        return self.container
                except Exception:  # noqa: BLE001 - expired and auto-removed
                    pass
                self._forget_container()
            _reap_once(client)
            self._ensure_image(client)
            container = client.containers.run(
                self.docker_image,
                command=watchdog_command(self.idle_ttl_seconds),
                detach=True,
                auto_remove=True,
                init=True,
                working_dir=self.working_dir,
                network_disabled=self.network_disabled,
                mem_limit=self.memory_limit,
                cpu_shares=self.cpu_shares,
                labels=self._labels(),
            )
            self.container = container
            self._finalizer = weakref.finalize(self, _remove_container, container)
            logger.debug("Acquired sandbox container: %s", container.short_id)
            return container

    def start(self) -> None:
        """Acquire the container now (idempotent). Normally implicit on first use."""
        try:
            self._ensure_container()
        except Exception as exc:  # noqa: BLE001 - degraded, retried on next use
            logger.warning("Sandbox container could not be started: %s", exc)

    def stop(self) -> None:
        """Remove the container (idempotent). The next use acquires a new one."""
        with self._lock:
            container, finalizer = self.container, self._finalizer
            self.container, self._finalizer = None, None
        if finalizer is not None:
            finalizer()  # runs _remove_container exactly once
        elif container is not None:
            _remove_container(container)

    dispose = stop

    def __enter__(self) -> "CodeSandbox":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()

    async def __aenter__(self) -> "CodeSandbox":
        self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()

    # -- Operations ----------------------------------------------------------

    def execute(self, code: str, timeout: int = 60) -> dict:
        """Run ``code`` with ``python -c`` in the sandbox, killed after ``timeout`` seconds."""
        try:
            container = self._ensure_container()
        except Exception as exc:  # noqa: BLE001
            return {"exit_code": 1, "stdout": "", "stderr": f"Sandbox could not start: {exc}"}
        if container is None:
            return {
                "exit_code": 1,
                "stdout": "",
                "stderr": "Docker is not available. Code execution in containers is disabled.",
            }

        # The code is passed as $0, never interpolated into the shell script.
        script = f'touch {LEASE_PATH}; exec timeout -s KILL {int(timeout)} python -c "$0"'
        exec_result = container.exec_run(["sh", "-c", script, code], demux=True)

        out, err = exec_result.output or (None, None)
        stdout = out.decode("utf-8") if out else ""
        stderr = err.decode("utf-8") if err else ""
        if exec_result.exit_code == 137:
            stderr += f"\n[sandbox] killed: exceeded {int(timeout)}s or the memory limit"
        return {"exit_code": exec_result.exit_code, "stdout": stdout, "stderr": stderr}

    def put_files(self, file_paths: List[str]) -> None:
        """Copy local files into the sandbox working directory."""
        container = self._ensure_container()
        if container is None:
            return

        tar_stream = io.BytesIO()
        with tarfile.open(fileobj=tar_stream, mode="w") as tar:
            for file_path in file_paths:
                p = Path(file_path)
                if not p.exists():
                    raise FileNotFoundError(f"Local file not found: {file_path}")
                tar.add(str(p), arcname=p.name)

        tar_stream.seek(0)
        container.put_archive(self.working_dir, tar_stream)
        container.exec_run(["touch", LEASE_PATH])

    def get_file(self, remote_path: str) -> bytes:
        """Retrieve a single file from the sandbox."""
        container = self._ensure_container()
        if container is None:
            return b""

        bits, _ = container.get_archive(remote_path)
        with io.BytesIO(b"".join(bits)) as tar_stream:
            with tarfile.open(fileobj=tar_stream, mode="r") as tar:
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

    The sandbox container is created on first use and removed after a period
    of inactivity; files uploaded before it expires are not kept.

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
