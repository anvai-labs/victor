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

``_create_code_execution_manager`` builds the coding vertical's CodeSandbox
handle, which acquires its container lazily on first use. Bootstrap therefore
starts no containers (eager start leaked one per container bootstrap: 2,242 in
one test run) and cannot stall on an unresponsive Docker daemon (the version
probe's 60s hang moved to the first sandbox use, under its own budget).
"""

import time
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from victor.agent.service_provider import OrchestratorServiceProvider


def _make_provider(settings: Any = None) -> OrchestratorServiceProvider:
    provider = OrchestratorServiceProvider.__new__(OrchestratorServiceProvider)
    if settings is not None:
        provider._settings = settings
    return provider


def _patch_loader(monkeypatch: pytest.MonkeyPatch, sandbox: Any) -> None:
    module = ModuleType("fake_code_executor_tool")
    module.CodeSandbox = sandbox  # type: ignore[attr-defined]
    monkeypatch.setattr(
        "victor.core.utils.capability_loader.load_code_executor_module", lambda: module
    )


class _RecordingSandbox:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.started = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None: ...


def test_bootstrap_builds_handle_without_starting_it(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_loader(monkeypatch, _RecordingSandbox)
    manager = _make_provider()._create_code_execution_manager()
    assert isinstance(manager, _RecordingSandbox)
    assert manager.started is False  # no container until first use


def test_settings_reach_the_sandbox(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_loader(monkeypatch, _RecordingSandbox)
    sub = SimpleNamespace(
        code_executor_image="python:3.12-slim",
        code_executor_network_disabled=False,
        code_executor_memory_limit="1g",
        code_executor_cpu_shares=512,
        code_executor_idle_ttl_seconds=60,
    )
    manager = _make_provider(SimpleNamespace(subprocess=sub))._create_code_execution_manager()
    assert manager.kwargs == {
        "docker_image": "python:3.12-slim",
        "network_disabled": False,
        "memory_limit": "1g",
        "cpu_shares": 512,
        "idle_ttl_seconds": 60,
    }


def test_unresponsive_docker_cannot_stall_bootstrap(monkeypatch: pytest.MonkeyPatch) -> None:
    """The real CodeSandbox, with docker.from_env hanging like a closed Docker Desktop."""
    # The sharded unit battery installs core only; the vertical's own CI covers it.
    cet = pytest.importorskip("victor_coding.tools.code_executor_tool")

    def _hang(**_kwargs: Any) -> Any:
        time.sleep(30)

    monkeypatch.setattr(cet, "DOCKER_AVAILABLE", True)
    monkeypatch.setattr(cet, "docker", SimpleNamespace(from_env=_hang))
    monkeypatch.setattr(
        "victor.core.utils.capability_loader.load_code_executor_module", lambda: cet
    )
    t0 = time.monotonic()
    manager = _make_provider()._create_code_execution_manager()
    assert time.monotonic() - t0 < 1.0
    assert isinstance(manager, cet.CodeSandbox)


def test_missing_vertical_stays_silent(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def _boom() -> Any:
        raise ImportError("victor-coding not installed")

    monkeypatch.setattr("victor.core.utils.capability_loader.load_code_executor_module", _boom)
    with caplog.at_level("WARNING", logger="victor.agent.service_provider"):
        manager = _make_provider()._create_code_execution_manager()
    assert manager.start() is None
    assert manager.dispose() is None
    with pytest.raises(RuntimeError, match="victor-coding"):
        manager.execute("print('hi')")  # type: ignore[attr-defined]
    assert not [
        r for r in caplog.records if r.levelno >= 30
    ], "missing vertical must stay the long-standing silent case"


def test_failing_vertical_degrades_loudly(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    class ExplodingSandbox:
        def __init__(self, **_kwargs: Any) -> None:
            raise RuntimeError("no container runtime")

    _patch_loader(monkeypatch, ExplodingSandbox)
    with caplog.at_level("WARNING", logger="victor.agent.service_provider"):
        manager = _make_provider()._create_code_execution_manager()
    assert manager.stop() is None
    assert any("failed:" in r.getMessage() for r in caplog.records)
