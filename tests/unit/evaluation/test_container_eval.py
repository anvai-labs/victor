# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License").

"""Tests for the containerized eval backend (victor.evaluation.container_eval).

No real Docker is used — the ``_run_cmd`` subprocess seam is monkeypatched to
assert the docker create/start/exec/rm command sequence and the image resolvers.
"""

from __future__ import annotations

from typing import Optional

import pytest

from victor.evaluation import container_eval as ce
from victor.evaluation.container_eval import (
    DockerUnavailable,
    EvalContainer,
    resolve_runtime,
    resolve_swebench_image,
)

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


def test_resolve_swebench_image_standard_form():
    img = resolve_swebench_image(_Task(), _Config())
    # Official Docker Hub scheme: swebench/sweb.eval.<arch>.<instance>; the
    # resolver is best-effort (exact <version> segment needs the swebench pkg).
    assert img == "docker.io/swebench/sweb.eval.x86_64.astropy_astropy-12907"


def test_resolve_swebench_image_custom_registry():
    img = resolve_swebench_image(_Task(), _Config(swebench_image_registry="ghcr.io/myorg"))
    assert img.startswith("ghcr.io/myorg/sweb.eval.x86_64.")


# ---- exact resolver (Docker Hub lookup) ----


class _FakeDockerHubResp:
    def __init__(self, repos):
        self._repos = repos

    def raise_for_status(self):
        pass

    def json(self):
        return {"results": [{"repo_name": r} for r in self._repos]}


def _fake_dockerhub_client(repos, *, boom=False):
    class _Client:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            pass

        async def get(self, url, params=None, **k):
            if boom:
                raise RuntimeError("network down")
            return _FakeDockerHubResp(repos)

    return _Client


async def test_resolve_swebench_image_exact_dockerhub_lookup(monkeypatch):
    ce._INSTANCE_IMAGE_CACHE.clear()
    ce._REPO_IMAGE_INDEX.clear()
    repos = [
        "swebench/sweb.eval.x86_64.astropy_1776_astropy-14182",
        "swebench/sweb.eval.x86_64.astropy_1776_astropy-12907",
        "swebench/sweb.eval.x86_64.astropy_1776_astropy-7166",
    ]
    monkeypatch.setattr(ce.httpx, "AsyncClient", _fake_dockerhub_client(repos))
    img = await ce.resolve_swebench_image_exact(_Task(), _Config())
    # Picks the repo ending with the instance's issue id, prefixed docker.io.
    assert img == "docker.io/swebench/sweb.eval.x86_64.astropy_1776_astropy-12907"
    assert "astropy__astropy-12907" in ce._INSTANCE_IMAGE_CACHE  # cached


async def test_resolve_swebench_image_exact_falls_back_on_network_failure(monkeypatch):
    ce._INSTANCE_IMAGE_CACHE.clear()
    ce._REPO_IMAGE_INDEX.clear()
    monkeypatch.setattr(ce.httpx, "AsyncClient", _fake_dockerhub_client([], boom=True))
    img = await ce.resolve_swebench_image_exact(_Task(), _Config())
    # Falls back to the heuristic resolver (graceful).
    assert img == "docker.io/swebench/sweb.eval.x86_64.astropy_astropy-12907"


async def test_repo_image_index_dedupes_lookups_per_repo(monkeypatch):
    """The Docker Hub search fires ONCE per repo, not once per instance.

    A repo whose lookup fails is cached empty so every subsequent instance of
    that repo skips straight to the heuristic — this is the django fix (≈44
    redundant failing calls reduced to 1).
    """
    ce._INSTANCE_IMAGE_CACHE.clear()
    ce._REPO_IMAGE_INDEX.clear()
    calls = {"n": 0}

    class _CountingClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            pass

        async def get(self, url, params=None, **k):
            calls["n"] += 1
            return _FakeDockerHubResp([])  # no matching images → heuristic

    monkeypatch.setattr(ce.httpx, "AsyncClient", _CountingClient)
    # Three instances of the same repo → the lookup should fire only once.
    for tid in (
        "astropy__astropy-12907",
        "astropy__astropy-14182",
        "astropy__astropy-7166",
    ):
        await ce.resolve_swebench_image_exact(_Task(task_id=tid), _Config())
    assert calls["n"] == 1
    # The empty result is cached, so a network failure isn't retried per task.
    assert ce._REPO_IMAGE_INDEX["astropy"] == []


async def test_resolve_swebench_image_exact_override_wins():
    img = await ce.resolve_swebench_image_exact(_Task(docker_image="my/exact:1"), _Config())
    assert img == "my/exact:1"


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
