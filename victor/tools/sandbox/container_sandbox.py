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

"""Docker container sandbox for code execution (FOU-S2, ADR-035 item 5).

The victor-coding vertical's ``CodeSandbox`` lifecycle quality bar, promoted
into core so Docker-available installs get container isolation for code
execution without the vertical. The OS-level namespace backends
(:mod:`victor.tools.sandbox.backends`) remain the fallback for installs
without Docker.

Unlike :class:`~victor.tools.sandbox.backends.SandboxBackend` (which wraps a
host subprocess argv in a launcher), a container sandbox *is* the execution
host: code runs inside a Docker container. The lifecycle is designed so
sandbox containers cannot accumulate:

- **Lazy**: constructing a :class:`ContainerSandbox` makes no Docker contact
  and starts nothing. A container is acquired on first
  ``execute``/``put_files`` (or an explicit ``start``), so bootstrap, tests
  and sessions that never run code never touch Docker.
- **Owned**: an acquired container is removed by ``stop``/``dispose`` and by
  a ``weakref.finalize`` that fires when the sandbox is garbage-collected or
  the interpreter exits.
- **Self-expiring**: the container runs an idle watchdog instead of
  ``sleep infinity`` and is started with ``auto_remove``. It deletes itself
  after ``idle_ttl_seconds`` without use, even if its owner was SIGKILLed.
  An expired container is transparently re-acquired on next use (uploaded
  files do not survive expiry).
- **Network-disabled by default**: containers are created with
  ``network_disabled=True``; egress is an explicit opt-in.
- **Capped**: ``mem_limit`` and the CPU controls ride the container-run
  invocation itself, so the caps hold regardless of what the executed code
  does. ``cpu_shares`` is a RELATIVE scheduler weight, not a ceiling; when a
  hard CPU limit is intended, set ``cpu_quota`` (microseconds per 100ms
  period — ``50000`` caps the container at 0.5 CPU), which Docker enforces
  via CFS bandwidth control.
- **Reapable**: containers carry owner labels (host boot id, pid, process
  start time). :func:`reap_orphaned_sandboxes` removes only containers whose
  owner on this host is dead, never a live sandbox of another process.

Capability gating mirrors :func:`~victor.tools.sandbox.backends.resolve_sandbox_backend`:
:func:`resolve_container_sandbox` returns the sandbox when Docker is
available, ``None`` (backend simply not offered) when it is not, and a typed
:class:`ContainerSandboxUnavailableError` refusal under ``required=True``.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import tarfile
import threading
import weakref
from pathlib import Path
from typing import Any, Dict, List, Optional

from victor.tools.sandbox_ownership import (
    OWNER_HOST_LABEL,
    OWNER_PID_LABEL,
    OWNER_START_LABEL,
    created_age_seconds,
    host_identity,
    owner_alive,
    owner_gone,
    owner_labels,
    process_start_marker,
)

logger = logging.getLogger(__name__)

# Optional docker import — the container backend is simply not offered when
# the package is missing (mirrors backends.py's binary-missing degradation).
try:
    import docker
    from docker.errors import DockerException, NotFound

    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False
    docker = None  # type: ignore
    DockerException = Exception  # type: ignore[misc,assignment]
    NotFound = Exception  # type: ignore[misc,assignment]

# Labels identifying Victor sandbox containers and their owner
SANDBOX_CONTAINER_LABEL = "victor.sandbox"
SANDBOX_CONTAINER_VALUE = "code-executor"
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


class ContainerSandboxUnavailableError(RuntimeError):
    """Docker cannot provide the container sandbox.

    Raised by :class:`ContainerSandbox` (``require_docker=True``) and
    :func:`resolve_container_sandbox` (``required=True``). ``reason`` is a
    stable machine-readable code — ``binary_missing`` (docker package or
    binary absent) or ``daemon_unreachable`` (probe failed) — mirroring the
    typed reasons of :class:`~victor.tools.sandbox.backends.SandboxRequiredUnavailableError`.
    Callers must treat this as a refusal (zero container execution), never as
    permission to run code unsandboxed.
    """

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


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
    return created_age_seconds((getattr(container, "attrs", None) or {}).get("Created", ""))


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

    orphans: List[Any] = []
    for container in containers:
        gone = owner_gone(getattr(container, "labels", None) or {})
        if gone is None:
            if not include_legacy or not _is_legacy_sandbox(container):
                continue
            if _age_seconds(container) < legacy_min_age_seconds:
                continue
        elif not gone:
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


class ContainerSandbox:
    """Lazily acquired, owned, self-expiring Docker container for code execution."""

    def __init__(
        self,
        docker_image: str = DEFAULT_IMAGE,
        require_docker: bool = False,
        network_disabled: bool = True,
        memory_limit: Optional[str] = _UNSET,
        cpu_shares: Optional[int] = None,
        cpu_quota: Optional[int] = None,
        idle_ttl_seconds: int = DEFAULT_IDLE_TTL_SECONDS,
    ):
        self.docker_image = docker_image
        self.container: Optional[Any] = None
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
        # Explicit hard CPU ceiling (CFS quota, microseconds per 100ms
        # period). None = no quota — shares alone are only a weight, never a
        # cap, so a hard limit must be configured explicitly.
        try:
            self.cpu_quota = (
                int(os.getenv("VICTOR_CODE_EXECUTOR_CPU_QUOTA", "0")) or None
                if cpu_quota is None
                else cpu_quota
            )
        except ValueError:
            self.cpu_quota = None
        self.idle_ttl_seconds = max(1, int(idle_ttl_seconds))
        self._lock = threading.RLock()
        self._client: Any = None
        self._probed = False
        self._finalizer: Optional[weakref.finalize] = None

        if not DOCKER_AVAILABLE:
            self._probed = True
            if require_docker:
                raise ContainerSandboxUnavailableError(
                    "binary_missing",
                    "Docker is not installed or importable. This feature requires Docker "
                    "(pip install docker) and a running daemon.",
                )
            return
        if require_docker and self.docker_client is None:
            raise ContainerSandboxUnavailableError(
                "daemon_unreachable",
                "Docker is not running or not installed. "
                "Start Docker Desktop or ensure the Docker daemon is running (verify: docker ps).",
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
        return {
            SANDBOX_CONTAINER_LABEL: SANDBOX_CONTAINER_VALUE,
            **owner_labels(),
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
                except NotFound:
                    # CONFIRMED deletion (idle TTL + auto_remove): the finalizer
                    # would target a gone container — detach and fall through to
                    # the replacement acquisition below.
                    self._forget_container()
                except Exception as exc:  # noqa: BLE001 - ambiguous daemon error
                    # Codex finding 18: a transient/unreachable daemon is NOT
                    # evidence the container died. Detaching the finalizer here
                    # would leave a live container unowned until its TTL (a
                    # leak); replacing could also duplicate sandboxes. Keep
                    # ownership and surface the error — the next use retries.
                    logger.warning(
                        "Sandbox container state unknown after daemon error (%s); "
                        "retaining ownership and retrying on next use.",
                        exc,
                    )
                    raise
                else:
                    if self.container.status in ("created", "running", "restarting"):
                        return self.container
                    self._forget_container()  # reload confirmed dead/exited: replace
            _reap_once(client)
            self._ensure_image(client)
            run_kwargs: Dict[str, Any] = {
                "command": watchdog_command(self.idle_ttl_seconds),
                "detach": True,
                "auto_remove": True,
                "init": True,
                "working_dir": self.working_dir,
                "network_disabled": self.network_disabled,
                "mem_limit": self.memory_limit,
                "cpu_shares": self.cpu_shares,
                "labels": self._labels(),
            }
            if self.cpu_quota is not None:
                # Hard CPU ceiling: CFS quota over the default 100000us period
                # (docker-py leaves cpu_period at the kernel default when unset).
                run_kwargs["cpu_quota"] = int(self.cpu_quota)
            container = client.containers.run(self.docker_image, **run_kwargs)
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

    def __enter__(self) -> "ContainerSandbox":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()

    async def __aenter__(self) -> "ContainerSandbox":
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
        """Copy local files into the sandbox working directory.

        Local paths are validated and the tar stream built BEFORE any Docker
        contact. Raises:

        * :class:`FileNotFoundError` — a local path does not exist.
        * :class:`ContainerSandboxUnavailableError` — no container can be
          acquired (typed upload failure, Codex finding 18). The historical
          silent ``return`` let callers proceed believing files existed
          inside the container; the async :func:`upload_files` helper then
          formatted the no-op as success. Fail-closed instead.
        """
        tar_stream = io.BytesIO()
        with tarfile.open(fileobj=tar_stream, mode="w") as tar:
            for file_path in file_paths:
                p = Path(file_path)
                if not p.exists():
                    raise FileNotFoundError(f"Local file not found: {file_path}")
                tar.add(str(p), arcname=p.name)

        container = self._ensure_container()
        if container is None:
            raise ContainerSandboxUnavailableError(
                "daemon_unreachable" if DOCKER_AVAILABLE else "binary_missing",
                "Cannot upload files: the Docker container sandbox is unavailable; "
                "refusing the upload instead of reporting success (fail-closed). "
                "Verify: docker ps.",
            )
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


# -- Capability gating (FOU-S2) -----------------------------------------------

_warned_categories: set = set()


def _warn_once(category: str, message: str) -> None:
    """Emit ``message`` once per process for ``category``."""
    if category not in _warned_categories:
        logger.warning(message)
        _warned_categories.add(category)


def daemon_available() -> bool:
    """Whether a Docker daemon is reachable, under the bounded connect probe."""
    if not DOCKER_AVAILABLE:
        return False
    try:
        client = docker.from_env(timeout=_connect_timeout())
        client.ping()
        return True
    except Exception:  # noqa: BLE001 - any probe failure is "not offered"
        return False


def sandbox_kwargs_from_settings(settings: Any) -> Dict[str, Any]:
    """Container-run configuration from the ``code_executor_*`` settings knobs.

    Accepts a ``Settings``-like object (reads the ``subprocess`` group with a
    flat-attribute fallback, matching the historical settings cascade) or
    None for the built-in defaults.
    """
    sub = getattr(settings, "subprocess", None)

    def _setting(name: str, default: Any) -> Any:
        return getattr(sub, name, getattr(settings, name, default))

    return {
        "docker_image": _setting("code_executor_image", DEFAULT_IMAGE),
        "network_disabled": bool(_setting("code_executor_network_disabled", True)),
        "memory_limit": _setting("code_executor_memory_limit", "512m"),
        "cpu_shares": _setting("code_executor_cpu_shares", 256),
        "cpu_quota": _setting("code_executor_cpu_quota", None),
        "idle_ttl_seconds": int(
            _setting("code_executor_idle_ttl_seconds", DEFAULT_IDLE_TTL_SECONDS)
        ),
    }


def resolve_container_sandbox(settings: Any = None, *, required: bool = False):
    """Select a container sandbox from settings (FOU-S2 capability gate).

    Mirrors :func:`~victor.tools.sandbox.backends.resolve_sandbox_backend`:

    * Docker package/daemon available — a :class:`ContainerSandbox`
      configured from the ``code_executor_*`` settings. Without
      ``required=True`` the daemon is NOT probed here (resolution stays
      lazy); first use degrades gracefully if the daemon dies meanwhile.
    * Docker package missing — ``None``: the container backend is simply not
      offered; callers fall back to the OS sandbox. With ``required=True``
      this refuses with reason ``binary_missing``.
    * ``required=True`` and the daemon is unreachable — typed refusal with
      reason ``daemon_unreachable``. Callers must refuse container execution,
      never run it unsandboxed.
    """
    if not DOCKER_AVAILABLE:
        if required:
            raise ContainerSandboxUnavailableError(
                "binary_missing",
                "Container sandbox policy is 'required' but the Docker package is not "
                "installed; refusing code execution (fail-closed).",
            )
        _warn_once(
            "container_unavailable",
            "Docker is not installed: the container code-execution sandbox is not "
            "offered this session (code execution will degrade).",
        )
        return None
    if required and not daemon_available():
        raise ContainerSandboxUnavailableError(
            "daemon_unreachable",
            "Container sandbox policy is 'required' but the Docker daemon is not "
            "reachable; refusing code execution (fail-closed). Verify: docker ps.",
        )
    return ContainerSandbox(**sandbox_kwargs_from_settings(settings))


async def execute_code(sandbox: ContainerSandbox, code: str) -> str:
    """Execute Python code in the sandbox, offloading the blocking call to a thread.

    Shared formatting contract with the coding vertical's ``sandbox`` tool.
    """
    result = await asyncio.to_thread(sandbox.execute, code)
    output = f"Exit Code: {result['exit_code']}\n"
    if result["stdout"]:
        output += f"--- STDOUT ---\n{result['stdout']}\n"
    if result["stderr"]:
        output += f"--- STDERR ---\n{result['stderr']}\n"
    return output


async def upload_files(sandbox: ContainerSandbox, file_paths: List[str]) -> str:
    """Upload files to the sandbox, offloading the blocking call to a thread."""
    try:
        await asyncio.to_thread(sandbox.put_files, file_paths)
        return f"Successfully uploaded {len(file_paths)} files to the sandbox."
    except Exception as e:
        return f"Error uploading files: {e}"


__all__ = [
    "DOCKER_AVAILABLE",
    "ContainerSandbox",
    "ContainerSandboxUnavailableError",
    "DEFAULT_IDLE_TTL_SECONDS",
    "DEFAULT_IMAGE",
    "IDLE_TTL_LABEL",
    "LEGACY_AUTO_REAP_MIN_AGE_SECONDS",
    "LEASE_PATH",
    "OWNER_HOST_LABEL",
    "OWNER_PID_LABEL",
    "OWNER_START_LABEL",
    "SANDBOX_CONTAINER_LABEL",
    "SANDBOX_CONTAINER_VALUE",
    "created_age_seconds",
    "daemon_available",
    "execute_code",
    "find_orphaned_sandboxes",
    "host_identity",
    "owner_alive",
    "owner_gone",
    "owner_labels",
    "process_start_marker",
    "reap_orphaned_sandboxes",
    "resolve_container_sandbox",
    "sandbox_kwargs_from_settings",
    "upload_files",
    "watchdog_command",
]
