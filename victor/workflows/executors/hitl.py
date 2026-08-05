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

"""HITL node executor.

Compiled graphs pause before HITL nodes via interrupt_before. This executor
keeps the resumed-node path native and records a consistent GraphNodeResult.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from victor.workflows.hitl import HITLNode
    from victor.workflows.runtime_types import WorkflowState

logger = logging.getLogger(__name__)


class HITLNodeExecutor:
    """Executor for human-in-the-loop workflow nodes."""

    def __init__(self, context: Any = None):
        self._context = context

    async def execute(self, node: "HITLNode", state: "WorkflowState") -> "WorkflowState":
        """Record HITL completion metadata for resumed compiled workflows."""
        from victor.workflows.runtime_types import GraphNodeResult

        current_state = dict(state)
        start_time = time.time()
        response = current_state.get("_hitl_response")

        current_state["_hitl_pending"] = False
        current_state["_hitl_response"] = response

        success = True
        error = None
        if (
            isinstance(response, dict)
            and response.get("approved") is False
            and node.fallback.value == "abort"
        ):
            error = f"HITL node '{node.id}' rejected: {response.get('reason', 'rejected')}"
            current_state["_error"] = error
            success = False

        if "_node_results" not in current_state:
            current_state["_node_results"] = {}
        current_state["_node_results"][node.id] = GraphNodeResult(
            node_id=node.id,
            success=success,
            output={
                "hitl_type": node.hitl_type.value,
                "prompt": node.prompt,
                "context_keys": list(node.context_keys),
                "fallback": node.fallback.value,
                "response": response,
            },
            error=error,
            duration_seconds=time.time() - start_time,
        )
        return current_state

    def supports_node_type(self, node_type: str) -> bool:
        """Return whether this executor supports the given workflow node type."""
        return node_type == "hitl"
