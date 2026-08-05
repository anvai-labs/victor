# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License").
"""Containerized evaluation backend — generic container lifecycle + image resolution.

Each benchmark task runs in a Docker container with the correct runtime
(language + version + deps) so the host needs nothing but Docker and there is
no host-venv pollution. For SWE-bench the official per-instance images
(``sweb.eval.<repo>__<instance>``) supply the correct Python + compiled
C-extensions + pinned deps — the dataset carries no ``python_version``, so the
image is the source of truth.

This module is intentionally **generic** (container lifecycle + image
resolution only). Benchmark-specific eval logic (git reset/apply, test-patch,
site-packages patching, FAIL_TO_PASS selection) lives in the benchmark runner
(``victor.evaluation.benchmarks.swe_bench._run_tests_in_container``), which
drives an :class:`EvalContainer` through ``start``/``exec``/``stop``.

If Docker (or the image) is unavailable, callers fall back to the existing
host path — see :class:`DockerUnavailable`.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from typing import Optional

from victor.workflows.isolation import IsolationConfig, ResourceLimits
from victor.workflows.sandbox_executor import (
    SANDBOX_CONTAINER_LABEL,
    build_docker_run_flags,
)

logger = logging.getLogger(__name__)

# Container label value for eval containers (the label key is
# ``victor.sandbox``; value distinguishes eval from workflow sandboxes).
EVAL_CONTAINER_LABEL_VALUE = "eval"


async def cleanup_stale_sandbox_containers() -> int:
    """Remove orphaned victor sandbox containers from prior crashed runs.

    If a benchmark process dies (SIGHUP, crash, OOM, manual kill) mid-task, the
    ``finally``-block teardown doesn't run — ``EvalContainer.stop()`` for the
    eval sandbox, and likewise the code-executor sandbox's own cleanup — so that
    container stays alive. Over many crashed runs these accumulate and can
    exhaust Docker resources (disk, memory, the daemon's container limit).

    Call this at benchmark START to sweep every orphan under the
    ``victor.sandbox`` label, both ``=eval`` and ``=code-executor`` values: any
    sandbox container still alive at a fresh run start is by definition an
    orphan (a live run holds its own container reference). Filtering by the
    label KEY — not a single value — is what lets one sweep reclaim both
    subsystems' leaks (an eval-only filter passed a 3,456-container / ~14 GB
    code-executor pileup by).

    Returns the number of containers removed.
    """
    try:
        rc, out, _ = await _run_cmd(
            [
                "docker",
                "ps",
                "-aq",
                "--filter",
                f"label={SANDBOX_CONTAINER_LABEL}",
            ],
            timeout=30,
        )
        if rc != 0:
            return 0
        ids = [line.strip() for line in out.decode().splitlines() if line.strip()]
        if not ids:
            return 0
        rc, _, _ = await _run_cmd(["docker", "rm", "-f", *ids], timeout=120)
        if rc == 0:
            logger.info("Cleaned up %d stale sandbox container(s) at startup", len(ids))
            return len(ids)
    except Exception as exc:
        logger.debug("Stale container cleanup failed (non-fatal): %s", exc)
    return 0


# Where the task's cached repo is mounted inside the container.
EVAL_WORKSPACE_MOUNT = "/workspace"


class DockerUnavailable(RuntimeError):
    """Raised when Docker isn't usable (daemon down / CLI missing / pull failed).

    Callers catch this to fall back to the host eval path so a missing Docker
    runtime never hard-fails a benchmark run (the darwin dev case).
    """


@dataclass
class RuntimeSpec:
    """Resolved runtime for a containerized task."""

    language: str
    base_image: str
    version: Optional[str] = None
    # Commands to set up the env inside the container (run via exec before
    # tests). Populated by benchmark runners from env_setup.py in Phase 2.
    setup_commands: list[str] = field(default_factory=list)
    test_command: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)


# Static language → base-image map. Version comes from task.language_version or
# config.runtime_version; the default per-language tag is used when unset.
_LANGUAGE_BASE_IMAGE = {
    "python": lambda v: f"python:{v or '3.11'}-slim",
    "javascript": lambda v: f"node:{v or '20'}-slim",
    "typescript": lambda v: f"node:{v or '20'}-slim",
    "go": lambda v: f"golang:{v or '1.22'}-bookworm",
    "rust": lambda v: f"rust:{v or '1.75'}-slim-bookworm",
    "java": lambda v: f"maven:{v or '3.9'}-eclipse-temurin-21",
}


def resolve_runtime(task, config) -> RuntimeSpec:
    """Resolve a task's runtime to a base image + version.

    Precedence: ``task.docker_image`` → ``config.docker_image_override`` → the
    static language+version map. For SWE-bench the official per-instance image
    pins the runtime, so callers use :func:`resolve_swebench_image` instead;
    this resolver is authoritative for non-SWE-bench polyglot tasks.
    """
    language = (getattr(task, "language", None) or "python").lower()
    version = getattr(task, "language_version", None) or getattr(config, "runtime_version", None)

    if getattr(task, "docker_image", None):
        image = task.docker_image
    elif getattr(config, "docker_image_override", None):
        image = config.docker_image_override
    else:
        builder = _LANGUAGE_BASE_IMAGE.get(language, _LANGUAGE_BASE_IMAGE["python"])
        image = builder(version)
    return RuntimeSpec(language=language, base_image=image, version=version)


def resolve_swebench_image(task, config) -> str:
    """Resolve the official SWE-bench per-instance image for a task.

    The SWE-bench dataset has no ``python_version`` column; the official images
    bake in the correct Python + compiled C-extensions + pinned deps. The
    published scheme (confirmed against Docker Hub) is::

        docker.io/swebench/sweb.eval.<arch>.<instance_id with __ -> _1776_>

    where ``1776`` is a **constant** separator SWE-bench uses in place of the
    ``__`` in an instance id — NOT a per-repo version. Verified against Docker
    Hub across every major SWE-bench repo (astropy, django, sympy, matplotlib,
    scikit-learn, pytest-dev, psf/requests, pylint-dev), including the
    ``org != repo`` cases (``psf__requests-2317`` → ``psf_1776_requests-2317``).
    So the exact official name is fully deterministic from the instance id::

        astropy__astropy-12907  ->  sweb.eval.x86_64.astropy_1776_astropy-12907
        django__django-10924    ->  sweb.eval.x86_64.django_1776_django-10924

    Precedence still favors an explicit ``--docker-image`` / ``task.docker_image``.
    A wrong name is safe: ``docker pull`` fails → :class:`DockerUnavailable` →
    the caller falls back to the host eval path.
    """
    registry = getattr(config, "swebench_image_registry", None) or "docker.io/swebench"
    instance = (getattr(task, "task_id", "") or "").replace("__", "_1776_").lower() or "unknown"
    return f"{registry}/sweb.eval.x86_64.{instance}"


async def resolve_swebench_image_exact(task, config) -> str:
    """Resolve the EXACT official SWE-bench image for a task.

    The official image name is fully deterministic from the instance id
    (``__`` → the constant ``_1776_`` separator — see
    :func:`resolve_swebench_image`), so this just applies precedence and returns
    that name; no network lookup is needed. (Historically this did a Docker Hub
    *search* to recover a presumed per-repo version segment, but that segment is
    a constant — and the search failed for high-instance repos like django,
    whose images fall off the first page of results, silently degrading every
    non-cached instance to the deps-less host eval path.)

    Precedence: ``task.docker_image`` → ``config.docker_image_override`` →
    deterministic official name. A wrong name is still safe: ``docker pull``
    fails → :class:`DockerUnavailable` → host fallback.

    Async is retained for call-site compatibility (callers ``await`` this).
    """
    if getattr(task, "docker_image", None):
        return task.docker_image
    if getattr(config, "docker_image_override", None):
        return config.docker_image_override
    return resolve_swebench_image(task, config)


async def _run_cmd(cmd: list[str], timeout: float) -> tuple[int, bytes, bytes]:
    """Run a command, return (returncode, stdout, stderr). Raises TimeoutError."""
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise
    return proc.returncode or 0, out, err


async def check_docker_available() -> None:
    """Raise :class:`DockerUnavailable` if the Docker daemon / CLI isn't usable."""
    try:
        rc, _, err = await _run_cmd(
            ["docker", "version", "--format", "{{.Server.Version}}"], timeout=20
        )
    except (FileNotFoundError, asyncio.TimeoutError) as exc:
        raise DockerUnavailable(f"docker CLI unavailable: {exc}") from exc
    if rc != 0:
        raise DockerUnavailable(
            f"docker daemon not reachable (is Docker Desktop running?): {err.decode()[:200]}"
        )


async def docker_pull(image: str, timeout: float = 1800.0, platform: Optional[str] = None) -> None:
    """Pull an image; raise :class:`DockerUnavailable` on failure.

    Large official SWE-bench images (3-8 GB) take minutes on first pull; the
    default timeout is generous. A missing image (pull fails) is treated as
    unavailable so callers fall back to the host path rather than failing hard.

    ``platform`` (e.g. ``linux/amd64``) forces a specific arch — required when
    the host arch differs from the image's (e.g. Apple Silicon running the
    x86_64-only SWE-bench images via Rosetta/QEMU emulation).
    """
    cmd = ["docker", "pull"]
    if platform:
        cmd += ["--platform", platform]
    cmd.append(image)
    try:
        rc, out, err = await _run_cmd(cmd, timeout=timeout)
    except asyncio.TimeoutError as exc:
        raise DockerUnavailable(f"docker pull timed out for {image}: {exc}") from exc
    except FileNotFoundError as exc:
        raise DockerUnavailable(f"docker CLI unavailable: {exc}") from exc
    if rc != 0:
        raise DockerUnavailable(f"docker pull failed for {image}: {err.decode()[:300]}")


class EvalContainer:
    """Persistent Docker container lifecycle (create → start → exec×N → rm).

    Unlike :class:`victor.workflows.sandbox_executor.SandboxedExecutor` (which
    does one ``docker run --rm`` per command), an ``EvalContainer`` stays alive
    across multiple ``exec`` calls so a multi-step eval (setup → apply patch →
    run tests) reuses one container/installed env.

    All docker invocations go through :meth:`_run` so tests can monkeypatch a
    single seam instead of spawning real containers.
    """

    def __init__(
        self,
        *,
        image: str,
        workspace_host_path: str,
        network_allowed: bool = False,
        resource_limits: Optional[ResourceLimits] = None,
        label_value: str = EVAL_CONTAINER_LABEL_VALUE,
        env: Optional[dict[str, str]] = None,
        platform: str = "linux/amd64",
    ) -> None:
        self.image = image
        self.workspace = workspace_host_path
        self.network_allowed = network_allowed
        self.resource_limits = resource_limits or ResourceLimits()
        self.label_value = label_value
        self.env = env or {}
        # SWE-bench official images are x86_64-only; on Apple Silicon (aarch64)
        # they must run via Rosetta/QEMU emulation, so we pin linux/amd64. Pass
        # platform="" or "linux/arm64" for arm64-native polyglot images.
        self.platform = platform
        self.name = f"victor-eval-{uuid.uuid4().hex[:12]}"
        self.container_id: Optional[str] = None
        self._started = False

    async def _run(self, cmd: list[str], timeout: float) -> tuple[int, bytes, bytes]:
        return await _run_cmd(cmd, timeout)

    async def start(self, pull_timeout: float = 1800.0) -> None:
        """Pull (best-effort), then ``docker create`` + ``docker start``.

        Raises :class:`DockerUnavailable` if Docker or the image can't be used.
        """
        await check_docker_available()
        # Best-effort pull — if the image is missing and source=official, treat
        # as unavailable so the caller falls back to the host path. Pin the
        # platform so x86_64-only SWE-bench images pull+run on Apple Silicon.
        await docker_pull(self.image, timeout=pull_timeout, platform=self.platform or None)

        isolation = IsolationConfig(
            sandbox_type="docker",
            network_allowed=self.network_allowed,
            resource_limits=self.resource_limits,
        )
        create_cmd = ["docker", "create"]
        if self.platform:
            create_cmd += ["--platform", self.platform]
        create_cmd += ["--name", self.name]
        create_cmd += build_docker_run_flags(
            working_dir=self.workspace,
            env=self.env,
            isolation=isolation,
            limits=self.resource_limits,
            label_value=self.label_value,
            workspace_mount=EVAL_WORKSPACE_MOUNT,
        )
        create_cmd += [self.image, "sleep", "infinity"]

        rc, out, err = await self._run(create_cmd, timeout=120)
        if rc != 0:
            raise DockerUnavailable(f"docker create failed for {self.image}: {err.decode()[:300]}")
        # `docker create` prints the container id on stdout.
        self.container_id = out.decode().strip() or None

        rc, _, err = await self._run(["docker", "start", self.name], timeout=120)
        if rc != 0:
            await self.stop()
            raise DockerUnavailable(f"docker start failed: {err.decode()[:300]}")
        self._started = True
        logger.info("EvalContainer started: name=%s image=%s", self.name, self.image)

    async def exec(
        self,
        command: list[str],
        *,
        timeout: float = 600.0,
        cwd: str = EVAL_WORKSPACE_MOUNT,
        env: Optional[dict[str, str]] = None,
    ) -> tuple[int, str, str]:
        """Run a command inside the running container. Returns (rc, stdout, stderr).

        Commands run via ``bash -lc`` (a login shell). The SWE-bench images use
        a conda ``testbed`` env that's only on PATH under a login shell — a bare
        ``docker exec ... python`` resolves to the BASE conda env (no
        pytest/astropy) → "No module named pytest" → 0 tests collected.
        """
        if not self._started:
            raise RuntimeError("EvalContainer.exec before start()")
        import shlex

        shell_cmd = shlex.join(list(command))
        cmd = ["docker", "exec"]
        if cwd:
            cmd += ["-w", cwd]
        for key, value in (env or {}).items():
            cmd += ["-e", f"{key}={value}"]
        cmd += [self.name, "bash", "-lc", shell_cmd]
        rc, out, err = await self._run(cmd, timeout=timeout)
        return (
            rc,
            out.decode("utf-8", errors="replace"),
            err.decode("utf-8", errors="replace"),
        )

    async def stop(self) -> None:
        """Force-remove the container. Never raises (cleanup is best-effort)."""
        if self.name:
            try:
                await self._run(["docker", "rm", "-f", self.name], timeout=60)
            except Exception as exc:  # noqa: BLE001 — cleanup must not raise
                logger.debug("EvalContainer.stop(%s) swallowed: %s", self.name, exc)
        self._started = False
        self.container_id = None
