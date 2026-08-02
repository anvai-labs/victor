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

import httpx

from victor.workflows.isolation import IsolationConfig, ResourceLimits
from victor.workflows.sandbox_executor import (
    SANDBOX_CONTAINER_LABEL,
    build_docker_run_flags,
)

logger = logging.getLogger(__name__)

# Container label value for eval containers (the label key is
# ``victor.sandbox``; value distinguishes eval from workflow sandboxes).
EVAL_CONTAINER_LABEL_VALUE = "eval"


async def cleanup_stale_eval_containers() -> int:
    """Remove orphaned victor-eval containers from prior crashed runs.

    If a benchmark process dies (SIGHUP, crash, OOM) mid-task, the
    ``EvalContainer.stop()`` in the ``finally`` block doesn't run → that
    task's container stays. Over multiple runs these accumulate and can
    exhaust Docker resources (disk, memory, container-limit). Call this at
    benchmark START to sweep orphans labelled ``victor.sandbox=eval``.

    Returns the number of containers removed.
    """
    try:
        rc, out, _ = await _run_cmd(
            [
                "docker",
                "ps",
                "-aq",
                "--filter",
                f"label={SANDBOX_CONTAINER_LABEL}={EVAL_CONTAINER_LABEL_VALUE}",
            ],
            timeout=30,
        )
        if rc != 0:
            return 0
        ids = [line.strip() for line in out.decode().splitlines() if line.strip()]
        if not ids:
            return 0
        rc, _, _ = await _run_cmd(["docker", "rm", "-f", *ids], timeout=60)
        if rc == 0:
            logger.info("Cleaned up %d stale eval container(s) at startup", len(ids))
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

        docker.io/swebench/sweb.eval.<arch>.<repo>_<version>_<instance>

    e.g. ``swebench/sweb.eval.x86_64.sympy_1776_sympy-20590``. The exact tag
    includes a ``<version>`` segment that only the ``swebench`` package's
    ``make_image_name`` computes from the dataset row, so this resolver is
    **best-effort**: it produces the right structure but may miss the version
    segment. Two reliable paths:
      * pass the exact image via ``--docker-image`` / ``task.docker_image``
        (highest precedence — use this for a known-correct name), or
      * ``pip install swebench`` and the harness can name images authoritatively.

    A wrong name is safe: ``docker pull`` fails → :class:`DockerUnavailable` →
    the caller falls back to the host eval path.
    """
    registry = getattr(config, "swebench_image_registry", None) or "docker.io/swebench"
    instance = (getattr(task, "task_id", "") or "").replace("__", "_").lower() or "unknown"
    return f"{registry}/sweb.eval.x86_64.{instance}"


# Per-instance exact-image cache (short-circuits the cheap match scan).
_INSTANCE_IMAGE_CACHE: dict[str, str] = {}

# Per-REPO index of Docker Hub candidate image names. The search
# ``sweb.eval.x86_64.<repo>`` is the expensive part (one API call) and its
# result is shared across ALL instances of that repo, so we cache the
# candidate list per repo and match each instance against it. This matters
# for django: the query returns 100+ results and SWE-bench images fall off
# page 1, so the match fails — without this index that failing lookup
# repeated once per django task (≈44 redundant API calls). A repo whose
# lookup fails (or yields no candidates) is cached as an empty list so every
# subsequent instance skips straight to the heuristic/host fallback.
_REPO_IMAGE_INDEX: dict[str, list[str]] = {}


async def _resolve_repo_image_candidates(repo: str) -> list[str]:
    """Return (cached) Docker Hub repo-name candidates for a SWE-bench repo.

    The first caller for a given repo performs the search; all later callers
    reuse the cached list. Network failure / empty results are cached as ``[]``
    so the lookup is never retried within a process.
    """
    if repo in _REPO_IMAGE_INDEX:
        return _REPO_IMAGE_INDEX[repo]
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.get(
                "https://hub.docker.com/v2/search/repositories/",
                params={"query": f"sweb.eval.x86_64.{repo}", "page_size": 100},
            )
            resp.raise_for_status()
            candidates = [
                r.get("repo_name", "") for r in resp.json().get("results", []) if r.get("repo_name")
            ]
    except Exception as exc:  # noqa: BLE001 — network failure → empty → heuristic
        logger.debug("Docker Hub lookup failed for repo %s: %s", repo, exc)
        candidates = []
    _REPO_IMAGE_INDEX[repo] = candidates
    return candidates


async def resolve_swebench_image_exact(task, config) -> str:
    """Resolve the EXACT official SWE-bench image via a Docker Hub lookup.

    The official name embeds a per-repo version segment (e.g. ``1776`` in
    ``sweb.eval.x86_64.astropy_1776_astropy-12907``) that only the swebench
    package's versioning map computes — not derivable from the dataset fields
    victor has. Rather than take a hard dep on swebench, look the exact repo up
    on Docker Hub (one search per REPO — see :data:`_REPO_IMAGE_INDEX` — then a
    cheap per-instance match against the cached candidate list). Falls back to
    the heuristic :func:`resolve_swebench_image` on any failure (which then
    degrades gracefully via host fallback on a pull failure).

    Precedence: ``task.docker_image`` → ``config.docker_image_override`` → Docker
    Hub lookup → heuristic.
    """
    if getattr(task, "docker_image", None):
        return task.docker_image
    if getattr(config, "docker_image_override", None):
        return config.docker_image_override

    instance_id = (getattr(task, "task_id", "") or "").lower()
    if "__" not in instance_id:
        return resolve_swebench_image(task, config)
    if instance_id in _INSTANCE_IMAGE_CACHE:
        return _INSTANCE_IMAGE_CACHE[instance_id]

    repo, _, issue = instance_id.partition("__")
    candidates = await _resolve_repo_image_candidates(repo)

    # The exact repo ends with the instance's issue id, e.g.
    # swebench/sweb.eval.x86_64.astropy_1776_astropy-12907  (issue = astropy-12907).
    match = next((rn for rn in candidates if rn.endswith(issue)), None)
    if not match:
        logger.warning(
            "No Docker Hub image match for %s among %d candidate(s) for repo "
            "%s; using heuristic name",
            instance_id,
            len(candidates),
            repo,
        )
        return resolve_swebench_image(task, config)
    image = f"docker.io/{match}"
    _INSTANCE_IMAGE_CACHE[instance_id] = image
    logger.info("Resolved SWE-bench image for %s via Docker Hub: %s", instance_id, image)
    return image


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
