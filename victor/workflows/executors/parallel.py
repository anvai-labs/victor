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

"""Parallel node executor.

Executes parallel nodes by running child nodes concurrently.
This is a stub that delegates to legacy implementation during migration.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from victor.workflows.definition import ParallelNode
    from victor.workflows.runtime_types import WorkflowState

logger = logging.getLogger(__name__)


class ParallelNodeExecutor:
    """Executor for parallel nodes.

    Responsibility (SRP):
    - Execute child nodes concurrently
    - Join results based on strategy (all, any, first, majority)
    - Handle parallel errors
    - Aggregate outputs

    Non-responsibility:
    - Workflow compilation (handled by WorkflowCompiler)
    - Child node execution (handled by child executors)
    """

    def __init__(self, context: Any = None):
        """Initialize the executor.

        Args:
            context: ExecutionContext with services, settings
        """
        self._context = context

    async def execute(self, node: "ParallelNode", state: "WorkflowState") -> "WorkflowState":
        """Execute a parallel node.

        Args:
            node: Parallel node definition
            state: Current workflow state

        Returns:
            Updated workflow state with aggregated results

        Note:
            Parallel results should already be populated by child nodes.
            This node serves as a join point and applies the join strategy.
        """
        import time
        from victor.workflows.runtime_types import GraphNodeResult

        logger.info(f"Executing parallel node: {node.id}")
        start_time = time.time()

        # Make mutable copy of state
        state = dict(state)

        try:
            # Step 1: Get parallel results (populated by child nodes)
            parallel_results = state.get("_parallel_results", {})

            # Step 2: Apply join strategy
            join_strategy = getattr(node, "join_strategy", "all")

            if join_strategy == "all":
                # All must succeed
                all_success = all(
                    r.get("success", True) if isinstance(r, dict) else True
                    for r in parallel_results.values()
                )
                if not all_success:
                    state["_error"] = "Not all parallel nodes succeeded"

            elif join_strategy == "any":
                # At least one must succeed
                any_success = any(
                    r.get("success", True) if isinstance(r, dict) else True
                    for r in parallel_results.values()
                )
                if not any_success:
                    state["_error"] = "No parallel nodes succeeded"

            elif join_strategy == "first":
                # First success wins (results should be ordered)
                first_success = None
                for result in parallel_results.values():
                    if isinstance(result, dict) and result.get("success", True):
                        first_success = result
                        break
                if first_success:
                    state["_parallel_first"] = first_success
                else:
                    state["_error"] = "No parallel node succeeded"

            # "merge" strategy just combines all results (no validation)

            # Step 3: Update node results for observability
            if "_node_results" not in state:
                state["_node_results"] = {}

            state["_node_results"][node.id] = GraphNodeResult(
                node_id=node.id,
                success="_error" not in state,
                output={
                    "parallel_nodes": getattr(node, "parallel_nodes", []),
                    "join_strategy": join_strategy,
                    "results_count": len(parallel_results),
                },
                duration_seconds=time.time() - start_time,
            )

            logger.info(f"Parallel node {node.id} completed with strategy: {join_strategy}")
            return state

        except Exception as e:
            logger.error(f"Parallel node '{node.id}' failed: {e}", exc_info=True)
            state["_error"] = f"Parallel node '{node.id}' failed: {e}"

            if "_node_results" not in state:
                state["_node_results"] = {}

            state["_node_results"][node.id] = GraphNodeResult(
                node_id=node.id,
                success=False,
                error=str(e),
                duration_seconds=time.time() - start_time,
            )

            raise

    def supports_node_type(self, node_type: str) -> bool:
        """Check if this executor supports the given node type."""
        return node_type == "parallel"


__all__ = ["ParallelNodeExecutor"]
