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

        Note:
            Condition nodes are actually handled as conditional edges during
            graph compilation via ConditionEvaluator.create_router(). This
            executor is a passthrough since routing is already resolved.
        """
        from victor.workflows.runtime_types import GraphNodeResult

        logger.debug(f"Condition node {node.id} is passthrough (routing handled by StateGraph)")

        # Make mutable copy of state
        state = dict(state)

        # Track that we passed through this condition node
        if "_node_results" not in state:
            state["_node_results"] = {}

        state["_node_results"][node.id] = GraphNodeResult(
            node_id=node.id,
            success=True,
            output={
                "passthrough": True,
                "branches": (list(node.branches.keys()) if hasattr(node, "branches") else []),
            },
        )

        return state

    def supports_node_type(self, node_type: str) -> bool:
        """Check if this executor supports the given node type."""
        return node_type == "condition"


__all__ = ["ConditionNodeExecutor"]
