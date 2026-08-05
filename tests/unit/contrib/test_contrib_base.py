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

"""Tests for victor.contrib packages."""

import pytest

from victor.contrib.safety import (
    BaseSafetyExtension,
    SafetyContext,
    VerticalSafetyMixin,
)
from victor.contrib.conversation import (
    BaseConversationManager,
    VerticalConversationContext,
)
from victor.contrib.mode_config import BaseModeConfigProvider, ModeHelperMixin
from victor.contrib.workflows import BaseWorkflowProvider, WorkflowLoaderMixin
from victor.contrib.testing import VerticalTestCase, MockProviderMixin


class TestSafetyContrib:
    """Test safety contrib package."""

    def test_safety_context_creation(self):
        """Test SafetyContext can be created."""
        ctx = SafetyContext(vertical_name="test")
        assert ctx.vertical_name == "test"
        assert ctx.operations == []
        assert ctx.metadata == {}

    def test_safety_context_track_operation(self):
        """Test SafetyContext tracks operations."""
        from victor_contracts.safety import SafetyCheckResult

        ctx = SafetyContext(vertical_name="test")
        result = SafetyCheckResult(is_safe=True, action=None)
        ctx.track_operation("test_tool", ["arg1", "arg2"], result)

        assert len(ctx.operations) == 1
        assert ctx.operations[0].tool_name == "test_tool"

    def test_safety_context_to_dict(self):
        """Test SafetyContext serialization."""
        ctx = SafetyContext(vertical_name="test")
        data = ctx.to_dict()

        assert data["vertical_name"] == "test"
        assert "operation_count" in data

    def test_vertical_safety_mixin_create_rule(self):
        """Test VerticalSafetyMixin creates rules."""
        mixin = VerticalSafetyMixin()
        rule = mixin.create_dangerous_command_rule("test_rule", r"test.*cmd", "Test command")

        assert rule.rule_id == "test_rule"
        assert rule.pattern == r"test.*cmd"
        assert rule.severity == 8


class TestConversationContrib:
    """Test conversation contrib package."""

    def test_vertical_conversation_context_creation(self):
        """Test VerticalConversationContext can be created."""
        ctx = VerticalConversationContext(vertical_name="test", domain="testing")
        assert ctx.vertical_name == "test"
        assert ctx.domain == "testing"

    def test_vertical_conversation_context_task_management(self):
        """Test VerticalConversationContext manages tasks."""
        from victor.contrib.conversation.vertical_context import TaskContext

        ctx = VerticalConversationContext(vertical_name="test", domain="testing")
        task = TaskContext(task_id="task1", task_type="test_task")

        ctx.add_task(task)
        assert "task1" in ctx.tasks
        assert ctx.get_task("task1") == task

    def test_vertical_conversation_context_active_tasks(self):
        """Test VerticalConversationContext filters active tasks."""
        from victor.contrib.conversation.vertical_context import TaskContext

        ctx = VerticalConversationContext(vertical_name="test", domain="testing")

        active_task = TaskContext(task_id="active", task_type="test", status="pending")
        completed_task = TaskContext(task_id="completed", task_type="test", status="completed")

        ctx.add_task(active_task)
        ctx.add_task(completed_task)

        active_tasks = ctx.get_active_tasks()
        assert len(active_tasks) == 1
        assert active_tasks[0].task_id == "active"


class TestModeConfigContrib:
    """Test mode config contrib package."""

    def test_mode_helper_mixin_creates_quick_mode(self):
        """Test ModeHelperMixin creates quick mode."""
        from victor.core.mode_config import ModeDefinition

        mixin = ModeHelperMixin()
        mode = mixin.create_quick_mode()

        assert isinstance(mode, ModeDefinition)
        assert mode.name == "quick"
        assert mode.tool_budget == 5

    def test_mode_helper_mixin_creates_standard_mode(self):
        """Test ModeHelperMixin creates standard mode."""
        from victor.core.mode_config import ModeDefinition

        mixin = ModeHelperMixin()
        mode = mixin.create_standard_mode()

        assert isinstance(mode, ModeDefinition)
        assert mode.name == "standard"
        assert mode.tool_budget == 15

    def test_mode_helper_mixin_creates_thorough_mode(self):
        """Test ModeHelperMixin creates thorough mode."""
        from victor.core.mode_config import ModeDefinition

        mixin = ModeHelperMixin()
        mode = mixin.create_thorough_mode()

        assert isinstance(mode, ModeDefinition)
        assert mode.name == "thorough"
        assert mode.tool_budget == 30


class TestWorkflowsContrib:
    """Test workflows contrib package."""

    def test_workflow_loader_mixin_validate_structure(self):
        """Test WorkflowLoaderMixin validates workflow structure."""
        mixin = WorkflowLoaderMixin()

        valid_workflow = {"nodes": [{"id": "test", "type": "agent"}]}
        is_valid, errors = mixin.validate_workflow_structure(valid_workflow)

        assert is_valid is True
        assert len(errors) == 0

    def test_workflow_loader_mixin_invalid_structure(self):
        """Test WorkflowLoaderMixin detects invalid structure."""
        mixin = WorkflowLoaderMixin()

        invalid_workflow = {"no_nodes": True}
        is_valid, errors = mixin.validate_workflow_structure(invalid_workflow)

        assert is_valid is False
        assert len(errors) > 0

    def test_workflow_loader_mixin_create_agent_node(self):
        """Test WorkflowLoaderMixin creates agent nodes."""
        mixin = WorkflowLoaderMixin()
        node = mixin.create_agent_node("test", "planner", "Test prompt")

        assert node["id"] == "test"
        assert node["type"] == "agent"
        assert node["agent_type"] == "planner"

    def test_workflow_loader_mixin_create_compute_node(self):
        """Test WorkflowLoaderMixin creates compute nodes."""
        mixin = WorkflowLoaderMixin()
        node = mixin.create_compute_node("test", ["read", "write"])

        assert node["id"] == "test"
        assert node["type"] == "compute"
        assert node["tools"] == ["read", "write"]


class TestTestingContrib:
    """Test testing contrib package."""

    def test_mock_provider_mixin_creates_provider(self):
        """Test MockProviderMixin creates mock providers."""
        mixin = MockProviderMixin()
        provider = mixin.create_mock_provider("Test response")

        assert provider is not None
        assert provider.provider_name == "test"

    def test_mock_provider_mixin_sequence(self):
        """Test MockProviderMixin creates sequence providers."""
        mixin = MockProviderMixin()
        provider = mixin.create_mock_provider_with_sequence(["Response 1", "Response 2"])

        assert provider is not None
        # Test that it returns responses in sequence
        response1 = provider.chat()
        response2 = provider.chat()
        assert response1.content == "Response 1"
        assert response2.content == "Response 2"

    def test_mock_tool_mixin(self):
        """Test MockToolMixin creates mock tools."""
        from victor.contrib.testing import MockToolMixin

        mixin = MockToolMixin()
        tool = mixin.create_mock_tool("test_tool", return_value="result")

        assert tool.name == "test_tool"
        assert tool() == "result"
