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


from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseWorkflow(ABC):
    """
    Abstract base class for all workflows.
    A workflow is a sequence of steps that use tools to achieve a higher-level goal.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """The unique name of the workflow."""
        raise NotImplementedError

    @property
    @abstractmethod
    def description(self) -> str:
        """A short description of what the workflow does."""
        raise NotImplementedError

    @abstractmethod
    async def run(self, context: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        """
        Executes the workflow.

        Args:
            context: The context containing shared resources like the ToolRegistry.
            **kwargs: Arguments specific to the workflow.

        Returns:
            A dictionary containing the results of the workflow execution.
        """
        raise NotImplementedError


class WorkflowRegistry:
    """A registry for discovering and managing available workflows."""

    def __init__(self) -> None:
        self._workflows: Dict[str, BaseWorkflow] = {}

    def register(self, workflow: BaseWorkflow) -> None:
        """Registers a workflow instance."""
        if workflow.name in self._workflows:
            raise ValueError(f"Workflow '{workflow.name}' is already registered.")
        self._workflows[workflow.name] = workflow

    def get(self, name: str) -> Optional[BaseWorkflow]:
        """Retrieves a workflow by its name."""
        return self._workflows.get(name)

    def list_workflows(self) -> list[BaseWorkflow]:
        """Returns a list of all registered workflows."""
        return list(self._workflows.values())
