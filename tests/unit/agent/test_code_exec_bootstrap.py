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

"""Session bootstrap must never touch Docker.

``_create_code_execution_manager`` builds the core container-sandbox handle
(FOU-S2: ``victor.tools.sandbox.container_sandbox.ContainerSandbox``, no
victor-coding dependency), which acquires its container lazily on first use.
Bootstrap therefore starts no containers (eager start leaked one per
container bootstrap: 2,242 in one test run) and cannot stall on an
unresponsive Docker daemon (the version probe's 60s hang moved to the first
sandbox use, under its own budget).
"""

import time
from types import SimpleNamespace
from typing import Any

import pytest

from victor.agent.service_provider import OrchestratorServiceProvider
from victor.tools.sandbox.container_sandbox import (
    ContainerSandbox,
    ContainerSandboxUnavailableError,
)

CORE = "victor.tools.sandbox.container_sandbox"


def _make_provider(settings: Any = None) -> OrchestratorServiceProvider:
    provider = OrchestratorServiceProvider.__new__(OrchestratorServiceProvider)
    if settings is not None:
        provider._settings = settings
    return provider


class _RecordingSandbox:
    kwargs: dict = {}

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.started = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None: ...


def test_bootstrap_builds_handle_without_starting_it(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(f"{CORE}.ContainerSandbox", _RecordingSandbox)
    manager = _make_provider()._create_code_execution_manager()
    assert isinstance(manager, _RecordingSandbox)
    assert manager.started is False  # no container until first use


def test_settings_reach_the_sandbox(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(f"{CORE}.ContainerSandbox", _RecordingSandbox)
    sub = SimpleNamespace(
        code_executor_image="python:3.12-slim",
        code_executor_network_disabled=False,
        code_executor_memory_limit="1g",
        code_executor_cpu_shares=512,
        code_executor_cpu_quota=50000,
        code_executor_idle_ttl_seconds=60,
    )
    manager = _make_provider(SimpleNamespace(subprocess=sub))._create_code_execution_manager()
    assert manager.kwargs == {
        "docker_image": "python:3.12-slim",
        "network_disabled": False,
        "memory_limit": "1g",
        "cpu_shares": 512,
        "cpu_quota": 50000,
        "idle_ttl_seconds": 60,
    }


def test_unresponsive_docker_cannot_stall_bootstrap(monkeypatch: pytest.MonkeyPatch) -> None:
    """The real core sandbox, with docker.from_env hanging like a closed Docker Desktop."""

    def _hang(**_kwargs: Any) -> Any:
        time.sleep(30)

    monkeypatch.setattr(f"{CORE}.DOCKER_AVAILABLE", True)
    monkeypatch.setattr(f"{CORE}.docker", SimpleNamespace(from_env=_hang))
    t0 = time.monotonic()
    manager = _make_provider()._create_code_execution_manager()
    assert time.monotonic() - t0 < 1.0
    assert isinstance(manager, ContainerSandbox)


def test_no_vertical_required_core_is_importable() -> None:
    """FOU-S2: the sandbox handle no longer depends on the coding vertical."""
    try:
        import victor_coding  # noqa: F401
    except ImportError:
        pass  # the point: construction below must succeed either way
    manager = _make_provider()._create_code_execution_manager()
    assert isinstance(manager, ContainerSandbox)
    assert manager.container is None  # lazy: Docker untouched at bootstrap


def test_no_docker_package_handle_is_inert(monkeypatch: pytest.MonkeyPatch) -> None:
    """Docker absent: the backend is simply not offered; bootstrap stays silent."""
    monkeypatch.setattr(f"{CORE}.DOCKER_AVAILABLE", False)
    with pytest.raises(ContainerSandboxUnavailableError) as excinfo:
        ContainerSandbox(require_docker=True)
    assert excinfo.value.reason == "binary_missing"
    manager = _make_provider()._create_code_execution_manager()
    assert isinstance(manager, ContainerSandbox)  # lazy handle still built, inert


def test_failing_construction_degrades_loudly(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    class ExplodingSandbox:
        def __init__(self, **_kwargs: Any) -> None:
            raise RuntimeError("no container runtime")

    monkeypatch.setattr(f"{CORE}.ContainerSandbox", ExplodingSandbox)
    with caplog.at_level("WARNING", logger="victor.agent.service_provider"):
        manager = _make_provider()._create_code_execution_manager()
    assert manager.stop() is None
    assert manager.dispose() is None
    with pytest.raises(RuntimeError, match="unavailable"):
        manager.execute("print('hi')")  # type: ignore[attr-defined]
    assert any("failed:" in r.getMessage() for r in caplog.records)
