# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
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

"""Compatibility helpers for deprecated workflow compiler shims.

These helpers keep older compiler surfaces wired into the canonical
``victor.workflows.executors.factory.NodeExecutorFactory`` implementation
without each compatibility layer re-implementing the same orchestration and
tool-registry context plumbing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

from victor.workflows.executors.factory import (
    NodeExecutorFactory as SharedNodeExecutorFactory,
)

if TYPE_CHECKING:
    from victor.agent.orchestrator import AgentOrchestrator
    from victor.tools.registry import ToolRegistry
    from victor.workflows.definition import WorkflowNode


class WorkflowNodeExecutionError(RuntimeError):
    """A workflow node signalled failure via the ``_error`` state convention.

    Raised by the compiler-facing executor wrapper so the StateGraph engine's
    native failure path (``NodeExecutionResult.fail`` → run outcome
    ``success=False``) engages, matching the BFS walker's semantics where a
    node that wrote ``_error`` failed the workflow.
    """

    def __init__(self, message: str, result_state: Dict[str, Any]) -> None:
        super().__init__(message)
        self.result_state = result_state


class CompilerOrchestratorPool:
    """Profile-aware orchestrator lookup used by compatibility contexts."""

    def __init__(
        self,
        orchestrator: Optional["AgentOrchestrator"] = None,
        orchestrators: Optional[Dict[str, "AgentOrchestrator"]] = None,
    ) -> None:
        self._orchestrator = orchestrator
        self._orchestrators = orchestrators or {}

    def get_orchestrator(self, profile: Optional[str] = None) -> Optional["AgentOrchestrator"]:
        if profile and profile in self._orchestrators:
            return self._orchestrators[profile]
        return self._orchestrator

    def get_default_orchestrator(self) -> Optional["AgentOrchestrator"]:
        return self._orchestrator


class CompilerExecutionContext:
    """Minimal execution context expected by registered workflow executors."""

    def __init__(
        self,
        orchestrator: Optional["AgentOrchestrator"] = None,
        orchestrators: Optional[Dict[str, "AgentOrchestrator"]] = None,
        tool_registry: Optional["ToolRegistry"] = None,
    ) -> None:
        self.orchestrator = orchestrator
        self.orchestrator_pool = CompilerOrchestratorPool(
            orchestrator=orchestrator,
            orchestrators=orchestrators,
        )
        self.tool_registry = tool_registry
        self.services = None


class CompatibilityNodeExecutorFactory:
    """Compatibility adapter over the canonical workflow node executor factory."""

    def __init__(
        self,
        *,
        orchestrator: Optional["AgentOrchestrator"] = None,
        orchestrators: Optional[Dict[str, "AgentOrchestrator"]] = None,
        tool_registry: Optional["ToolRegistry"] = None,
    ) -> None:
        self.orchestrator = orchestrator
        self.orchestrators = orchestrators or {}
        self.tool_registry = tool_registry
        self._delegate = SharedNodeExecutorFactory()
        self._compat_context = CompilerExecutionContext(
            orchestrator=orchestrator,
            orchestrators=self.orchestrators,
            tool_registry=tool_registry,
        )

    def _resolve_execution_context(self) -> CompilerExecutionContext:
        return self._compat_context

    def register_executor_type(
        self,
        node_type: str,
        executor_class: Any,
        *,
        replace: bool = False,
    ) -> None:
        self._delegate.register_executor_type(node_type, executor_class, replace=replace)

    def create_executor(
        self,
        node: "WorkflowNode",
    ) -> Callable[[Dict[str, Any]], Any]:
        original_resolver = self._delegate._resolve_execution_context
        self._delegate._resolve_execution_context = self._resolve_execution_context
        try:
            executor = self._delegate.create_executor(node)
        finally:
            self._delegate._resolve_execution_context = original_resolver

        async def execute_with_failure_propagation(
            state: Dict[str, Any],
        ) -> Dict[str, Any]:
            """Run the node and surface its ``_error`` failure convention.

            Workflow node executors signal failure by writing ``state["_error"]``
            and returning normally (a convention inherited from the BFS walker,
            which inspected that key after every node). The StateGraph engine
            instead fails a run when a node callable *raises*. Without this
            translation, a failed transform/parallel/team/HITL node is silently
            reported as a successful workflow — the divergence ADR-030's parity
            gate pinned (engine parity requires failure propagation).

            Raises:
                WorkflowNodeExecutionError: if the node wrote ``_error``.
            """
            result_state = await executor(state)
            if isinstance(result_state, dict) and result_state.get("_error") is not None:
                raise WorkflowNodeExecutionError(
                    str(result_state["_error"]),
                    result_state,
                )
            return result_state

        return execute_with_failure_propagation

    def supports_node_type(self, node_type: str) -> bool:
        return self._delegate.supports_node_type(node_type)


__all__ = [
    "CompilerExecutionContext",
    "CompilerOrchestratorPool",
    "CompatibilityNodeExecutorFactory",
    "WorkflowNodeExecutionError",
]
