"""TDD tests for typed-contract protocols in service layer.

Wave 1: verifies that OrchestratorRuntimeProtocol, ParallelExplorationProtocol,
StatePassedExplorationProtocol, and the ISP role-protocols (CapabilityDiscovery,
CapabilityRead, CapabilityMutation) are correctly wired and satisfied by the
concrete implementations they describe.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.mark.timeout(260)
def test_tool_execution_collaborators_are_statically_checked(tmp_path):
    """The real constructor rejects missing dependencies and incompatible contracts."""
    import subprocess
    import sys

    fixture = tmp_path / "tool_contract_probe.py"
    valid = """from victor.agent.services.tool_execution_runtime import ToolExecutionRuntime
from victor.agent.services.orchestrator_protocol_adapter import OrchestratorProtocolAdapter
from victor.agent.services.tool_execution_contracts import ToolBatchExecution, ToolResultPublication

def bind(adapter: OrchestratorProtocolAdapter) -> ToolExecutionRuntime:
    execution: ToolBatchExecution = adapter
    publication: ToolResultPublication = adapter
    return ToolExecutionRuntime(adapter, execution=execution, publication=publication)
"""

    def check(source):
        fixture.write_text(source)
        return subprocess.run(
            [sys.executable, "-m", "mypy", "--strict", "--follow-imports=silent", str(fixture)],
            text=True,
            capture_output=True,
            timeout=120,
        )

    positive = check(valid)
    assert positive.returncode == 0, positive.stdout + positive.stderr
    negative = check(valid + """
def invalid(adapter: OrchestratorProtocolAdapter) -> None:
    ToolExecutionRuntime(adapter)
    ToolExecutionRuntime(adapter, execution=object(), publication=object())
""")
    assert negative.returncode == 1, negative.stdout + negative.stderr
    assert 'Missing named argument "execution"' in negative.stdout
    assert 'Missing named argument "publication"' in negative.stdout
    assert 'Argument "execution"' in negative.stdout and '"ToolBatchExecution"' in negative.stdout
    assert (
        'Argument "publication"' in negative.stdout and '"ToolResultPublication"' in negative.stdout
    )


class TestISPRoleProtocols:
    """ISP role-protocols defined in protocols.py must be satisfied by OrchestratorCapabilityMixin."""

    def test_isp_discovery_protocol_satisfied_by_mixin(self):
        from victor.agent.capability_registry import OrchestratorCapabilityMixin
        from victor.framework.protocols import CapabilityDiscoveryProtocol

        mixin = OrchestratorCapabilityMixin()
        assert isinstance(
            mixin, CapabilityDiscoveryProtocol
        ), "OrchestratorCapabilityMixin must satisfy CapabilityDiscoveryProtocol"

    def test_isp_read_protocol_satisfied_by_mixin(self):
        from victor.agent.capability_registry import OrchestratorCapabilityMixin
        from victor.framework.protocols import CapabilityReadProtocol

        mixin = OrchestratorCapabilityMixin()
        assert isinstance(
            mixin, CapabilityReadProtocol
        ), "OrchestratorCapabilityMixin must satisfy CapabilityReadProtocol"

    def test_isp_mutation_protocol_satisfied_by_mixin(self):
        from victor.agent.capability_registry import OrchestratorCapabilityMixin
        from victor.framework.protocols import CapabilityMutationProtocol

        mixin = OrchestratorCapabilityMixin()
        assert isinstance(
            mixin, CapabilityMutationProtocol
        ), "OrchestratorCapabilityMixin must satisfy CapabilityMutationProtocol"

    def test_isp_role_protocols_are_narrower_than_full_registry_protocol(self):
        """Each role-protocol should declare fewer methods than the full CapabilityRegistryProtocol."""
        from victor.framework.protocols import (
            CapabilityDiscoveryProtocol,
            CapabilityMutationProtocol,
            CapabilityReadProtocol,
            CapabilityRegistryProtocol,
        )

        full_methods = {m for m in dir(CapabilityRegistryProtocol) if not m.startswith("_")}
        discovery_methods = {m for m in dir(CapabilityDiscoveryProtocol) if not m.startswith("_")}
        read_methods = {m for m in dir(CapabilityReadProtocol) if not m.startswith("_")}
        mutation_methods = {m for m in dir(CapabilityMutationProtocol) if not m.startswith("_")}

        assert len(discovery_methods) < len(full_methods)
        assert len(read_methods) < len(full_methods)
        assert len(mutation_methods) < len(full_methods)


class TestOrchestratorRuntimeProtocol:
    """OrchestratorRuntimeProtocol must be importable from chat_runtime and be @runtime_checkable."""

    def test_protocol_is_importable(self):
        from victor.agent.services.protocols.chat_runtime import (
            OrchestratorRuntimeProtocol,
        )

        assert OrchestratorRuntimeProtocol is not None

    def test_protocol_is_runtime_checkable(self):
        from victor.agent.services.protocols.chat_runtime import (
            OrchestratorRuntimeProtocol,
        )

        # A mock with get_messages() should pass isinstance check
        mock = MagicMock()
        mock.get_messages = AsyncMock(return_value=[])
        # runtime_checkable protocols only check callable methods, not attributes
        assert isinstance(mock, OrchestratorRuntimeProtocol)

    def test_protocol_exported_in_all(self):
        import victor.agent.services.protocols.chat_runtime as m

        assert "OrchestratorRuntimeProtocol" in m.__all__

    def test_resolve_orchestrator_return_annotation_is_protocol(self):
        """_resolve_orchestrator() should declare OrchestratorRuntimeProtocol return type."""
        import inspect

        from victor.agent.services.turn_execution_runtime import TurnExecutor
        from victor.agent.services.protocols.chat_runtime import (
            OrchestratorRuntimeProtocol,
        )

        hints = {}
        try:
            hints = TurnExecutor._resolve_orchestrator.__annotations__
        except AttributeError:
            hints = {}

        return_hint = hints.get("return")
        # Accept Optional[OrchestratorRuntimeProtocol] or OrchestratorRuntimeProtocol
        # The annotation should reference the protocol, not plain Any
        assert (
            return_hint is not None
        ), "_resolve_orchestrator() should have a return type annotation"
        annotation_str = str(return_hint)
        # Must NOT be bare 'Any' or missing
        assert (
            "Any" not in annotation_str or "Optional" in annotation_str
        ), f"Return type should use OrchestratorRuntimeProtocol, got: {annotation_str}"


class TestExplorationProtocols:
    """ParallelExplorationProtocol and StatePassedExplorationProtocol must be importable."""

    def test_parallel_exploration_protocol_importable(self):
        from victor.agent.services.protocols.chat_runtime import (
            ParallelExplorationProtocol,
        )

        assert ParallelExplorationProtocol is not None

    def test_state_passed_exploration_protocol_importable(self):
        from victor.agent.services.protocols.chat_runtime import (
            StatePassedExplorationProtocol,
        )

        assert StatePassedExplorationProtocol is not None

    def test_exploration_coordinator_satisfies_parallel_protocol(self):
        from victor.agent.services.exploration_runtime import ExplorationCoordinator
        from victor.agent.services.protocols.chat_runtime import (
            ParallelExplorationProtocol,
        )

        coordinator = ExplorationCoordinator()
        assert isinstance(coordinator, ParallelExplorationProtocol)

    def test_state_passed_coordinator_satisfies_state_passed_protocol(self):
        from victor.agent.coordinators.exploration_state_passed import (
            ExplorationStatePassedCoordinator,
        )
        from victor.agent.services.protocols.chat_runtime import (
            StatePassedExplorationProtocol,
        )

        coordinator = ExplorationStatePassedCoordinator()
        assert isinstance(coordinator, StatePassedExplorationProtocol)

    def test_resolve_parallel_explorer_return_annotation_is_typed(self):
        """_resolve_parallel_explorer() should NOT return tuple[Any, bool]."""
        from victor.agent.services.turn_execution_runtime import TurnExecutor

        hints = getattr(TurnExecutor._resolve_parallel_explorer, "__annotations__", {})
        return_hint = hints.get("return")
        assert (
            return_hint is not None
        ), "_resolve_parallel_explorer() should have a return type annotation"
        annotation_str = str(return_hint)
        # The return type should not be a bare Any tuple
        assert (
            annotation_str != "tuple[typing.Any, bool]"
        ), "Return type should use typed protocols instead of Any"

    def test_protocols_exported_in_all(self):
        import victor.agent.services.protocols.chat_runtime as m

        assert "ParallelExplorationProtocol" in m.__all__
        assert "StatePassedExplorationProtocol" in m.__all__
