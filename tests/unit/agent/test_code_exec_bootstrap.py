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

"""Session bootstrap must never block on Docker availability.

The coding vertical's CodeSandbox constructor contacts the Docker daemon
(``docker.from_env`` → version probe, requests-timeout default 60s). With
Docker Desktop down, every ``victor chat`` bootstrap stalled a flat minute.
``ServiceProvider._create_code_execution_manager`` now bounds the load+start
and degrades loudly; these tests pin that contract.
"""

import threading
import time
from types import ModuleType
from typing import Any

import pytest

from victor.agent.service_provider import OrchestratorServiceProvider


def _make_provider() -> OrchestratorServiceProvider:
    # The factory method under test reads no instance state.
    return OrchestratorServiceProvider.__new__(OrchestratorServiceProvider)


def _vertical_module(sandbox: Any) -> ModuleType:
    module = ModuleType("fake_code_executor_tool")
    module.CodeSandbox = sandbox  # type: ignore[attr-defined]
    return module


def _patch_loader(monkeypatch: pytest.MonkeyPatch, sandbox: Any) -> None:
    monkeypatch.setattr(
        "victor.core.utils.capability_loader.load_code_executor_module",
        lambda: _vertical_module(sandbox),
    )


def test_healthy_vertical_loads_and_starts(monkeypatch: pytest.MonkeyPatch) -> None:
    started = []

    class Sandbox:
        def __init__(self) -> None:
            self.started = False

        def start(self) -> None:
            self.started = True
            started.append(self)

        def stop(self) -> None: ...

    _patch_loader(monkeypatch, Sandbox)
    manager = _make_provider()._create_code_execution_manager()
    assert isinstance(manager, Sandbox)
    assert manager.started is True
    assert started == [manager]


def test_hanging_vertical_degrades_within_budget(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    class HangingSandbox:
        def __init__(self) -> None:
            # Stands in for docker.from_env()'s 60s daemon probe.
            self._block = threading.Event()

        def start(self) -> None:
            self._block.wait(timeout=10)

        def stop(self) -> None: ...

    _patch_loader(monkeypatch, HangingSandbox)
    monkeypatch.setenv("VICTOR_CODE_EXEC_START_TIMEOUT", "0.2")

    t0 = time.monotonic()
    with caplog.at_level("WARNING", logger="victor.agent.service_provider"):
        manager = _make_provider()._create_code_execution_manager()
    elapsed = time.monotonic() - t0

    assert elapsed < 2.0, f"bootstrap still blocked {elapsed:.1f}s on the sandbox"
    assert manager.start() is None  # degraded manager is inert
    with pytest.raises(RuntimeError, match="VICTOR_CODE_EXEC_START_TIMEOUT"):
        manager.execute_python("print('hi')")  # type: ignore[attr-defined]
    assert any("timed out after 0s" in r.message for r in caplog.records) or any(
        "timed out" in r.getMessage() for r in caplog.records
    )


def test_missing_vertical_stays_silent(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def _boom() -> Any:
        raise ImportError("victor-coding not installed")

    monkeypatch.setattr("victor.core.utils.capability_loader.load_code_executor_module", _boom)
    with caplog.at_level("WARNING", logger="victor.agent.service_provider"):
        manager = _make_provider()._create_code_execution_manager()
    assert manager.start() is None
    assert not [
        r for r in caplog.records if r.levelno >= 30
    ], "missing vertical must stay the long-standing silent case"


def test_failing_vertical_degrades_loudly(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    class ExplodingSandbox:
        def __init__(self) -> None:
            raise RuntimeError("no container runtime")

        def start(self) -> None: ...

        def stop(self) -> None: ...

    _patch_loader(monkeypatch, ExplodingSandbox)
    with caplog.at_level("WARNING", logger="victor.agent.service_provider"):
        manager = _make_provider()._create_code_execution_manager()
    assert manager.stop() is None
    assert any("failed:" in r.getMessage() for r in caplog.records)
