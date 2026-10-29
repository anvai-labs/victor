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

"""CodeExecutionManagerProtocol: the sandbox contract the runtime depends on."""

import pytest

from victor.agent.protocols.infrastructure_protocols import CodeExecutionManagerProtocol


class _Full:
    def start(self) -> None: ...

    def stop(self) -> None: ...

    def dispose(self) -> None: ...


class _NoDispose:
    def start(self) -> None: ...

    def stop(self) -> None: ...


def test_dispose_is_part_of_the_contract():
    assert isinstance(_Full(), CodeExecutionManagerProtocol)
    # without dispose() a container reset could not release the sandbox
    assert not isinstance(_NoDispose(), CodeExecutionManagerProtocol)


def test_real_sandbox_satisfies_the_protocol():
    cet = pytest.importorskip("victor_coding.tools.code_executor_tool")
    assert isinstance(cet.CodeSandbox(), CodeExecutionManagerProtocol)


def test_degraded_manager_satisfies_the_protocol(monkeypatch):
    from victor.agent.service_provider import OrchestratorServiceProvider

    def _missing():
        raise ImportError("victor-coding not installed")

    monkeypatch.setattr("victor.core.utils.capability_loader.load_code_executor_module", _missing)
    provider = OrchestratorServiceProvider.__new__(OrchestratorServiceProvider)
    manager = provider._create_code_execution_manager()
    assert isinstance(manager, CodeExecutionManagerProtocol)
