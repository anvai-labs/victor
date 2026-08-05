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

"""Unit tests for workflows base module."""

import pytest
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock

from victor.workflows.base import BaseWorkflow, WorkflowRegistry
from victor.tools.base import ToolResult
from victor.tools.registry import ToolRegistry


class TestWorkflowBase(BaseWorkflow):
    """Concrete implementation for testing."""

    @property
    def name(self) -> str:
        return "test_workflow"

    @property
    def description(self) -> str:
        return "A test workflow"

    async def run(self, context: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        return {"success": True, "message": "Test completed"}


class TestWorkflowRegistry:
    """Tests for WorkflowRegistry class."""

    def test_register_workflow(self):
        """Test registering a workflow."""
        registry = WorkflowRegistry()
        workflow = TestWorkflowBase()
        registry.register(workflow)
        assert registry.get("test_workflow") is workflow

    def test_register_duplicate_raises(self):
        """Test registering duplicate workflow raises ValueError."""
        registry = WorkflowRegistry()
        workflow1 = TestWorkflowBase()
        workflow2 = TestWorkflowBase()
        registry.register(workflow1)
        with pytest.raises(ValueError, match="already registered"):
            registry.register(workflow2)

    def test_get_unknown_returns_none(self):
        """Test getting unknown workflow returns None."""
        registry = WorkflowRegistry()
        assert registry.get("nonexistent") is None

    def test_list_workflows_empty(self):
        """Test listing workflows when empty."""
        registry = WorkflowRegistry()
        assert registry.list_workflows() == []

    def test_list_workflows_multiple(self):
        """Test listing multiple workflows."""
        registry = WorkflowRegistry()

        class AnotherWorkflow(BaseWorkflow):
            @property
            def name(self) -> str:
                return "another"

            @property
            def description(self) -> str:
                return "Another workflow"

            async def run(self, context: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
                return {}

        workflow1 = TestWorkflowBase()
        workflow2 = AnotherWorkflow()
        registry.register(workflow1)
        registry.register(workflow2)

        workflows = registry.list_workflows()
        assert len(workflows) == 2
        assert workflow1 in workflows
        assert workflow2 in workflows
