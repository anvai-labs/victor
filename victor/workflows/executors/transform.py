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

"""Transform node executor.

Executes transform nodes by applying state transformations.
This is a stub that delegates to legacy implementation during migration.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from victor.workflows.definition import TransformNode
    from victor.workflows.runtime_types import WorkflowState

logger = logging.getLogger(__name__)


class TransformNodeExecutor:
    """Executor for transform nodes.

    Responsibility (SRP):
    - Apply state transformations
    - Map input keys to output keys
    - Transform data types
    - Merge results into state

    Non-responsibility:
    - Workflow compilation (handled by WorkflowCompiler)
    - Validation of transformations (handled by WorkflowValidator)
    """

    def __init__(self, context: Any = None):
        """Initialize the executor.

        Args:
            context: ExecutionContext with services, settings
        """
        self._context = context

    async def execute(self, node: "TransformNode", state: "WorkflowState") -> "WorkflowState":
        """Execute a transform node.

        Args:
            node: Transform node definition
            state: Current workflow state

        Returns:
            Updated workflow state

        Raises:
            Exception: If transform function fails
        """
        import time
        from victor.workflows.runtime_types import GraphNodeResult

        logger.info(f"Executing transform node: {node.id}")
        start_time = time.time()

        # Make mutable copy of state
        state = dict(state)

        try:
            # Step 1: Execute transform function
            transformed = node.transform(state)

            # Step 2: Merge transformed data back into state
            for key, value in transformed.items():
                state[key] = value

            # Step 3: Update node results for observability
            if "_node_results" not in state:
                state["_node_results"] = {}

            state["_node_results"][node.id] = GraphNodeResult(
                node_id=node.id,
                success=True,
                output={"transformed_keys": list(transformed.keys())},
                duration_seconds=time.time() - start_time,
            )

            logger.info(f"Transform node {node.id} completed successfully")
            return state

        except Exception as e:
            logger.error(f"Transform node '{node.id}' failed: {e}", exc_info=True)
            state["_error"] = f"Transform node '{node.id}' failed: {e}"

            if "_node_results" not in state:
                state["_node_results"] = {}

            state["_node_results"][node.id] = GraphNodeResult(
                node_id=node.id,
                success=False,
                error=str(e),
                duration_seconds=time.time() - start_time,
            )

            return state

    def supports_node_type(self, node_type: str) -> bool:
        """Check if this executor supports the given node type."""
        return node_type == "transform"


__all__ = ["TransformNodeExecutor"]
