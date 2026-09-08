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

"""Condition node executor.

Executes condition nodes by evaluating branching logic.
This is a stub that delegates to legacy implementation during migration.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from victor.workflows.definition import ConditionNode
    from victor.workflows.runtime_types import WorkflowState

logger = logging.getLogger(__name__)


class ConditionNodeExecutor:
    """Executor for condition nodes.

    Responsibility (SRP):
    - Evaluate condition functions
    - Route execution based on condition result
    - Handle missing/invalid conditions
    - Return branch identifier

    Non-responsibility:
    - Workflow compilation (handled by WorkflowCompiler)
    - Edge traversal (handled by StateGraph)
    """

    def __init__(self, context: Any = None):
        """Initialize the executor.

        Args:
            context: ExecutionContext with services, settings
        """
        self._context = context

    async def execute(self, node: "ConditionNode", state: "WorkflowState") -> "WorkflowState":
        """Execute a condition node.

        Args:
            node: Condition node definition
            state: Current workflow state

        Returns:
            Updated workflow state with branch result

        The executor evaluates once and records the branch. The compiler router
        consumes that result so conditions with effects are not called twice.
        """
        from victor.workflows.runtime_types import GraphNodeResult

        import time

        state = dict(state)
        start_time = time.time()
        try:
            branch = node.condition(state)
            if branch not in node.branches:
                if "default" in node.branches:
                    branch = "default"
                else:
                    raise ValueError(
                        f"Condition node '{node.id}' returned unknown branch {branch!r}"
                    )
            result = GraphNodeResult(
                node_id=node.id,
                success=True,
                output={"branch": branch, "next_node": node.branches[branch]},
                duration_seconds=time.time() - start_time,
            )
        except Exception as exc:
            error = f"Condition evaluation failed for node '{node.id}': {exc}"
            state["_error"] = error
            result = GraphNodeResult(
                node_id=node.id,
                success=False,
                error=error,
                duration_seconds=time.time() - start_time,
            )
        state.setdefault("_node_results", {})[node.id] = result
        return state

    def supports_node_type(self, node_type: str) -> bool:
        """Check if this executor supports the given node type."""
        return node_type == "condition"


__all__ = ["ConditionNodeExecutor"]
