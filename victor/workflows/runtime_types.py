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

"""Shared runtime state types for compiled workflow execution."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional

from victor.workflows.models import WorkflowStateModel

# Type alias for backward compatibility
WorkflowState = WorkflowStateModel


@dataclass
class GraphNodeResult:
    """Result from executing a workflow node in the compiled graph runtime."""

    node_id: str
    success: bool
    output: Any = None
    error: Optional[str] = None
    duration_seconds: float = 0.0
    tool_calls_used: int = 0


def create_initial_workflow_state(
    *,
    current_node: str = "",
    workflow_id: Optional[str] = None,
    workflow_name: str = "",
    initial_state: Optional[Dict[str, Any]] = None,
) -> WorkflowStateModel:
    """Create the canonical initial state for compiled workflow execution."""
    # Build kwargs, only including workflow_id if provided
    kwargs = {
        "workflow_name": workflow_name,
        "current_node": current_node,
        "data": {},  # Initialize with empty data dict
    }
    if workflow_id is not None:
        kwargs["workflow_id"] = workflow_id

    state = WorkflowStateModel(**kwargs)

    if initial_state:
        # Separate user data from system fields
        system_keys = {
            "_workflow_id",
            "_workflow_name",
            "_current_node",
            "_node_results",
            "_error",
            "_iteration",
            "_parallel_results",
            "_hitl_pending",
            "_hitl_response",
        }

        # User data goes into the data field
        user_data = {k: v for k, v in initial_state.items() if k not in system_keys}
        state.data.update(user_data)

        # System fields update the model directly
        if "_node_results" in initial_state:
            state.node_results = initial_state["_node_results"]
        if "_error" in initial_state:
            state.error = initial_state["_error"]
        if "_iteration" in initial_state:
            state.iteration = initial_state["_iteration"]
        if "_parallel_results" in initial_state:
            state.parallel_results = initial_state["_parallel_results"]
        if "_hitl_pending" in initial_state:
            state.hitl_pending = initial_state["_hitl_pending"]
        if "_hitl_response" in initial_state:
            state.hitl_response = initial_state["_hitl_response"]

    return state


__all__ = [
    "create_initial_workflow_state",
    "GraphNodeResult",
    "WorkflowState",
]
