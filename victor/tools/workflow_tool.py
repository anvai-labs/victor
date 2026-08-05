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


from typing import Any, Dict

from victor.tools.base import AccessMode, DangerLevel, Priority
from victor.tools.decorators import tool
from victor.workflows.base import WorkflowRegistry


@tool(
    category="workflow",
    priority=Priority.LOW,  # Specialized automation tool
    access_mode=AccessMode.MIXED,  # Depends on workflow actions
    danger_level=DangerLevel.MEDIUM,  # Workflows can modify files
    keywords=["workflow", "automation", "task", "sequence"],
)
async def workflow(
    workflow_name: str, context: dict, workflow_args: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Runs a pre-defined, multi-step workflow to automate a complex task.

    Args:
        workflow_name: The name of the workflow to run (e.g., 'python_feature').
        context: The tool context provided by the orchestrator.
        workflow_args: A dictionary of arguments required by the specific workflow.

    Returns:
        A dictionary containing the results of the workflow execution.
    """
    workflow_registry: WorkflowRegistry = context.get("workflow_registry")
    if not workflow_registry:
        return {"error": "WorkflowRegistry not found in context."}

    workflow = workflow_registry.get(workflow_name)
    if not workflow:
        return {"error": f"Workflow '{workflow_name}' not found."}

    try:
        # Pass the full context down to the workflow
        result = await workflow.run(context, **workflow_args)
        return result
    except Exception as e:
        return {
            "error": f"An unexpected error occurred while running workflow '{workflow_name}': {e}"
        }
