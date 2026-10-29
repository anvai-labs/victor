# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License").

"""Tests for the containerized eval backend (victor.evaluation.container_eval).

No real Docker is used — the ``_run_cmd`` subprocess seam is monkeypatched to
assert the docker create/start/exec/rm command sequence and the image resolvers.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import pytest

from victor.evaluation import container_eval as ce
from victor.evaluation.container_eval import (
    LEGACY_EVAL_MIN_AGE_SECONDS,
    DockerUnavailable,
    EvalContainer,
    cleanup_stale_sandbox_containers,
    resolve_runtime,
    resolve_swebench_image,
)
from victor.tools.sandbox_ownership import OWNER_HOST_LABEL, OWNER_PID_LABEL, owner_labels

# ---- lightweight task/config stand-ins (duck-typed to what the resolvers read) ----


class _Task:
    def __init__(self, **kw):
        defaults = {
            "task_id": "astropy__astropy-12907",
            "language": "python",
            "language_version": None,
            "docker_image": None,
            "repo": "https://github.com/astropy/astropy.git",
        }
        for k, v in defaults.items():
            setattr(self, k, kw.get(k, v))


class _Config:
    def __init__(self, **kw):
        defaults = {
            "swebench_image_registry": "docker.io/swebench",
            "runtime_version": None,
            "docker_image_override": None,
        }
        for k, v in defaults.items():
            setattr(self, k, kw.get(k, v))


# ---- image resolvers ----


def test_resolve_swebench_image_deterministic_1776_form():
    # Official Docker Hub scheme uses the CONSTANT `_1776_` separator in place
    # of `__` (verified across every major SWE-bench repo), so the per-instance
    # image name is exact — no version lookup needed.
    img = resolve_swebench_image(_Task(), _Config())
    assert img == "docker.io/swebench/sweb.eval.x86_64.astropy_1776_astropy-12907"


def test_resolve_swebench_image_django_form():
    # Regression: high-instance repos like django must resolve to the real
    # `<repo>_1776_<repo>-<n>` image. The old `<repo>_<repo>-<n>` form 404s and
    # silently degraded every non-cached instance to the deps-less host path.
    img = resolve_swebench_image(_Task(task_id="django__django-10924"), _Config())
    assert img == "docker.io/swebench/sweb.eval.x86_64.django_1776_django-10924"


def test_resolve_swebench_image_org_ne_repo_form():
    # `org != repo` instances (e.g. psf/requests) keep the org in the left slot.
    img = resolve_swebench_image(_Task(task_id="psf__requests-2317"), _Config())
    assert img == "docker.io/swebench/sweb.eval.x86_64.psf_1776_requests-2317"


def test_resolve_swebench_image_custom_registry():
    img = resolve_swebench_image(_Task(), _Config(swebench_image_registry="ghcr.io/myorg"))
    assert img.startswith("ghcr.io/myorg/sweb.eval.x86_64.")


# ---- exact resolver (deterministic, no network) ----


async def test_resolve_swebench_image_exact_is_deterministic():
    # The exact resolver returns the deterministic official name with no network
    # lookup — both a cached-style (astropy) and a high-instance (django) repo.
    img = await ce.resolve_swebench_image_exact(_Task(), _Config())
    assert img == "docker.io/swebench/sweb.eval.x86_64.astropy_1776_astropy-12907"
    dj = await ce.resolve_swebench_image_exact(_Task(task_id="django__django-10924"), _Config())
    assert dj == "docker.io/swebench/sweb.eval.x86_64.django_1776_django-10924"


async def test_resolve_swebench_image_exact_override_wins():
    img = await ce.resolve_swebench_image_exact(_Task(docker_image="my/exact:1"), _Config())
    assert img == "my/exact:1"
    # config-level override also wins over the computed name.
    img2 = await ce.resolve_swebench_image_exact(_Task(), _Config(docker_image_override="cfg/i:2"))
    assert img2 == "cfg/i:2"


def test_resolve_runtime_language_version_map():
    assert (
        resolve_runtime(_Task(language="python", language_version="3.9"), _Config()).base_image
        == "python:3.9-slim"
    )
    assert (
        resolve_runtime(_Task(language="javascript", language_version="20"), _Config()).base_image
        == "node:20-slim"
    )
    assert (
        resolve_runtime(_Task(language="go", language_version="1.22"), _Config()).base_image
        == "golang:1.22-bookworm"
    )
    # default version when unset
    assert (
        resolve_runtime(_Task(language="python", language_version=None), _Config()).base_image
        == "python:3.11-slim"
    )
    assert (
        resolve_runtime(_Task(language="javascript", language_version=None), _Config()).base_image
        == "node:24-slim"
    )
    assert (
        resolve_runtime(_Task(language="typescript", language_version=None), _Config()).base_image
        == "node:24-slim"
    )


def test_resolve_runtime_precedence_override_then_task_image():
    # task.docker_image wins over everything
    spec = resolve_runtime(
        _Task(docker_image="myrepo/custom:1"), _Config(docker_image_override="node:20")
    )
    assert spec.base_image == "myrepo/custom:1"
    # config.docker_image_override wins over the language map
    spec = resolve_runtime(_Task(docker_image=None), _Config(docker_image_override="node:20"))
    assert spec.base_image == "node:20"


# ---- backend selector (lives on the SWE-bench runner) ----


def test_select_backend_defaults_local_docker_opt_in():
    from victor.evaluation.benchmarks.swe_bench import SWEBenchRunner
    from victor.evaluation.protocol import EvaluationConfig

    def cfg(**kw):
        return EvaluationConfig(benchmark="swe_bench", model="x", **kw)

    assert SWEBenchRunner._select_backend(cfg()) == "local"
    assert SWEBenchRunner._select_backend(cfg(eval_backend="docker")) == "docker"
    assert SWEBenchRunner._select_backend(cfg(use_docker=True)) == "docker"  # legacy alias
    assert SWEBenchRunner._select_backend(cfg(eval_backend="local", use_docker=True)) == "docker"


# ---- EvalContainer lifecycle (mocked _run_cmd) ----


def _fake_run_factory(issued: list, *, docker_ok: bool = True, exec_rc: int = 0):
    """Return a fake _run_cmd that records commands + returns canned responses."""

    async def _fake(cmd, timeout):
        issued.append(list(cmd))
        sub = cmd[1] if len(cmd) > 1 else ""
        if not docker_ok and sub == "version":
            return 1, b"", b"Cannot connect to the Docker daemon"
        if sub == "version":
            return 0, b"24.0.7\n", b""
        if sub == "pull":
            return 0, b"", b""
        if sub == "create":
            return 0, b"deadbeefcafe\n", b""
        if sub == "start":
            return 0, b"", b""
        if sub == "exec":
            return exec_rc, b"test output\n", b""
        if sub == "rm":
            return 0, b"", b""
        return 0, b"", b""

    return _fake


async def test_eval_container_start_exec_stop_issues_correct_docker_cmds(monkeypatch):
    issued: list = []
    monkeypatch.setattr(ce, "_run_cmd", _fake_run_factory(issued))

    c = EvalContainer(
        image="sweb.eval.astropy:astropy__astropy-12907",
        workspace_host_path="/tmp/repo",
    )
    await c.start()
    assert c.container_id == "deadbeefcafe"
    assert c._started is True

    rc, out, err = await c.exec(["python", "-m", "pytest", "-x"])
    assert rc == 0 and "test output" in out

    await c.stop()
    assert c._started is False

    # Verify the docker subcommand sequence.
    subs = [cmd[1] for cmd in issued]
    assert subs == ["version", "pull", "create", "start", "exec", "rm"]
    # The create command pins platform (x86_64 SWE-bench images on Apple
    # Silicon), mounts the workspace, labels, and sleeps infinity.
    create_cmd = next(cmd for cmd in issued if cmd[1] == "create")
    assert "--platform" in create_cmd
    assert "linux/amd64" in create_cmd
    assert "--name" in create_cmd
    assert any("victor.sandbox=eval" in flag for flag in create_cmd)
    assert any("/tmp/repo:/workspace:rw" == flag for flag in create_cmd)
    # Owner labels let a concurrent run's sweep tell this live container apart
    # from a crashed run's orphan.
    for key, value in owner_labels().items():
        assert f"{key}={value}" in create_cmd
    assert create_cmd[-3:] == [
        "sweb.eval.astropy:astropy__astropy-12907",
        "sleep",
        "infinity",
    ]
    # The pull also pins the platform (so the right arch variant is fetched).
    pull_cmd = next(cmd for cmd in issued if cmd[1] == "pull")
    assert "--platform" in pull_cmd and "linux/amd64" in pull_cmd
    # The exec targets the container by name with cwd=/workspace, and runs via
    # bash -lc (login shell) so the image's conda env activates.
    exec_cmd = next(cmd for cmd in issued if cmd[1] == "exec")
    assert c.name in exec_cmd
    assert "-w" in exec_cmd and "/workspace" in exec_cmd
    assert "bash" in exec_cmd and "-lc" in exec_cmd
    assert "python" in exec_cmd[-1]  # the joined shell command


async def test_eval_container_start_raises_docker_unavailable_when_daemon_down(
    monkeypatch,
):
    issued: list = []
    monkeypatch.setattr(ce, "_run_cmd", _fake_run_factory(issued, docker_ok=False))

    c = EvalContainer(image="x", workspace_host_path="/tmp/repo")
    with pytest.raises(DockerUnavailable):
        await c.start()
    # No container was created, so nothing to clean up; stop() is a no-op-safe.
    await c.stop()


async def test_eval_container_stop_never_raises(monkeypatch):
    """Cleanup must be best-effort even if docker rm fails."""

    async def _boom(cmd, timeout):
        raise RuntimeError("docker exploded")

    monkeypatch.setattr(ce, "_run_cmd", _boom)
    c = EvalContainer(image="x", workspace_host_path="/tmp/repo")
    c.name = "victor-eval-test"
    await c.stop()  # must not raise


# ---- stale-sandbox sweep ----


def _inspect_entry(cid: str, labels: dict, age_s: float = 0.0) -> dict:
    created = datetime.fromtimestamp(time.time() - age_s, timezone.utc)
    return {
        "Id": cid + "0" * 52,
        "Created": created.strftime("%Y-%m-%dT%H:%M:%S.%f") + "123Z",
        "Config": {"Labels": {"victor.sandbox": "eval", **labels}},
    }


def _fake_docker(issued: list, entries: list):
    async def _fake(cmd, timeout):
        issued.append(list(cmd))
        if cmd[1] == "ps":
            return 0, "\n".join(e["Id"][:12] for e in entries).encode(), b""
        if cmd[1] == "inspect":
            return 0, json.dumps(entries).encode(), b""
        return 0, b"", b""

    return _fake


async def test_cleanup_sweeps_eval_and_reaps_code_executor_orphans(monkeypatch):
    """Both subsystems' leaks are reclaimed (a 3,456-container code-executor
    pileup once went uncollected), each by a sweep that only removes what it
    can prove is orphaned: eval containers whose owner is gone, code-executor
    sandboxes through their owner-aware reaper."""
    issued: list = []
    me = owner_labels()
    dead = {**me, OWNER_PID_LABEL: "999999"}
    entries = [
        _inspect_entry("live00000000", me),  # this process: a concurrent run's container
        _inspect_entry("dead00000000", dead),
        _inspect_entry("foreign00000", {OWNER_HOST_LABEL: "other-boot", OWNER_PID_LABEL: "1"}),
        _inspect_entry("legacyold000", {}, age_s=LEGACY_EVAL_MIN_AGE_SECONDS + 60),
        _inspect_entry("legacynew000", {}, age_s=60),
    ]
    monkeypatch.setattr(ce, "_run_cmd", _fake_docker(issued, entries))
    monkeypatch.setattr(
        "victor.tools.sandbox_ownership.owner_alive", lambda pid, start: pid != 999999
    )
    _patch_code_exec_reaper(monkeypatch, lambda: ["c1", "c2"])  # owner-dead code-executor
    removed = await cleanup_stale_sandbox_containers()

    assert removed == 2 + 2
    ps = next(c for c in issued if c[1] == "ps")
    assert "label=victor.sandbox=eval" in ps
    rm = next(c for c in issued if c[1] == "rm")
    assert rm == ["docker", "rm", "-f", "dead00000000", "legacyold000"]


async def test_cleanup_never_removes_a_live_runs_eval_container(monkeypatch):
    """The old sweep removed every eval container, killing concurrent runs mid-task."""
    issued: list = []
    monkeypatch.setattr(
        ce, "_run_cmd", _fake_docker(issued, [_inspect_entry("live00000000", owner_labels())])
    )
    _patch_code_exec_reaper(monkeypatch, lambda: [])
    assert await cleanup_stale_sandbox_containers() == 0
    assert not any(c[1] == "rm" for c in issued)


async def test_cleanup_inspect_failure_removes_nothing(monkeypatch):
    issued: list = []

    async def _fake(cmd, timeout):
        issued.append(list(cmd))
        if cmd[1] == "ps":
            return 0, b"aaa\n", b""
        return 1, b"", b"boom"

    monkeypatch.setattr(ce, "_run_cmd", _fake)
    _patch_code_exec_reaper(monkeypatch, lambda: [])
    assert await cleanup_stale_sandbox_containers() == 0
    assert not any(c[1] == "rm" for c in issued)


async def test_cleanup_never_sweeps_code_executor_by_label_key(monkeypatch):
    """A blanket ``label=victor.sandbox`` sweep removed live chat-session sandboxes."""
    issued: list = []

    async def _fake(cmd, timeout):
        issued.append(list(cmd))
        return 0, b"", b""

    monkeypatch.setattr(ce, "_run_cmd", _fake)
    _patch_code_exec_reaper(monkeypatch, lambda: [])
    await cleanup_stale_sandbox_containers()
    filters = [c[c.index("--filter") + 1] for c in issued if "--filter" in c]
    assert filters == ["label=victor.sandbox=eval"]


async def test_cleanup_tolerates_missing_coding_vertical(monkeypatch):
    async def _fake(cmd, timeout):
        return 0, b"", b""

    monkeypatch.setattr(ce, "_run_cmd", _fake)

    def _missing():
        raise ImportError("victor-coding not installed")

    monkeypatch.setattr("victor.core.utils.capability_loader.load_code_executor_module", _missing)
    assert await cleanup_stale_sandbox_containers() == 0


def _patch_code_exec_reaper(monkeypatch, reaper):
    from types import SimpleNamespace

    monkeypatch.setattr(
        "victor.core.utils.capability_loader.load_code_executor_module",
        lambda: SimpleNamespace(reap_orphaned_sandboxes=reaper),
    )


async def test_cleanup_no_orphans_is_noop(monkeypatch):
    issued: list = []

    async def _fake(cmd, timeout):
        issued.append(list(cmd))
        return 0, b"", b""  # ps returns nothing

    monkeypatch.setattr(ce, "_run_cmd", _fake)
    _patch_code_exec_reaper(monkeypatch, lambda: [])
    assert await cleanup_stale_sandbox_containers() == 0
    assert not any(c[1] == "rm" for c in issued)  # never attempts a remove
