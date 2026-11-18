# Copyright 2025 Vijaykumar Singh <anvaiops.com>
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

"""Input-state translation at the workflow compatibility boundary."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

from victor.workflows.runtime_types import create_initial_workflow_state

if TYPE_CHECKING:
    from victor.framework.graph import CheckpointerProtocol
    from victor.workflows.definition import WorkflowDefinition


async def create_invocation_state(
    workflow: "WorkflowDefinition",
    initial_context: Optional[Dict[str, Any]],
    thread_id: Optional[str],
    checkpointer: Optional["CheckpointerProtocol"],
) -> Any:
    """Create graph input while preserving omitted-input checkpoint semantics."""
    if (
        initial_context is None
        and thread_id is not None
        and checkpointer is not None
        and await checkpointer.load(thread_id) is not None
    ):
        return None

    return create_initial_workflow_state(
        workflow_id=thread_id,
        current_node=workflow.start_node or "",
        initial_state=initial_context,
    )
