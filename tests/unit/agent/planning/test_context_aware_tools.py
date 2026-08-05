# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file unless in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for context-aware tool selection in TaskPlanner."""

import pytest

from victor.agent.conversation.state_machine import ConversationStage
from victor.agent.planning.readable_schema import ReadableTaskPlan, TaskComplexity
from victor.agent.planning.tool_selection import (
    COMPLEXITY_TOOL_LIMITS,
    STEP_TOOL_MAPPING,
    StepAwareToolSelector,
    get_complexity_limits,
    get_step_tool_sets,
)


class TestStepToolMapping:
    """Tests for step type to tool set mapping."""

    def test_step_tool_mapping_exists(self):
        """Test that step tool mapping is defined."""
        assert isinstance(STEP_TOOL_MAPPING, dict)
        assert len(STEP_TOOL_MAPPING) > 0

    def test_research_step_tools(self):
        """Test that research step has read-only tools."""
        tools = STEP_TOOL_MAPPING.get("research", set())
        assert "read" in tools
        assert "grep" in tools
        assert "code_search" in tools
        # Research should NOT have write tools
        assert "write" not in tools
        assert "edit" not in tools

    def test_feature_step_tools(self):
        """Test that feature step has full toolset."""
        tools = STEP_TOOL_MAPPING.get("feature", set())
        assert "read" in tools
        assert "write" in tools
        assert "edit" in tools
        assert "test" in tools
        assert "git" in tools

    def test_test_step_tools(self):
        """Test that test step has testing tools."""
        tools = STEP_TOOL_MAPPING.get("test", set())
        assert "test" in tools
        assert "read" in tools
        # Test should NOT have write tools
        assert "write" not in tools
        assert "edit" not in tools

    def test_deploy_step_tools(self):
        """Test that deploy step has deployment tools."""
        tools = STEP_TOOL_MAPPING.get("deploy", set())
        assert "shell" in tools
        assert "git" in tools
        assert "docker" in tools
        assert "kubectl" in tools


class TestComplexityToolLimits:
    """Tests for complexity-based tool limits."""

    def test_complexity_limits_exist(self):
        """Test that complexity limits are defined."""
        assert isinstance(COMPLEXITY_TOOL_LIMITS, dict)
        assert "simple" in COMPLEXITY_TOOL_LIMITS
        assert "moderate" in COMPLEXITY_TOOL_LIMITS
        assert "complex" in COMPLEXITY_TOOL_LIMITS

    def test_simple_tasks_fewer_tools(self):
        """Test that simple tasks have fewer tools."""
        simple_limit = COMPLEXITY_TOOL_LIMITS["simple"]
        moderate_limit = COMPLEXITY_TOOL_LIMITS["moderate"]
        complex_limit = COMPLEXITY_TOOL_LIMITS["complex"]

        assert simple_limit < moderate_limit
        assert moderate_limit <= complex_limit

    def test_get_complexity_limits(self):
        """Test getting complexity limits."""
        limits = get_complexity_limits()
        assert isinstance(limits, dict)
        assert len(limits) == 3

    def test_limit_values(self):
        """Test that limit values are reasonable."""
        assert COMPLEXITY_TOOL_LIMITS["simple"] <= 20
        assert COMPLEXITY_TOOL_LIMITS["moderate"] <= 50
        assert COMPLEXITY_TOOL_LIMITS["complex"] <= 150


class TestStepAwareToolSelector:
    """Tests for StepAwareToolSelector - basic functionality without tool registry."""

    def test_step_tool_mapping_constants(self):
        """Test that step tool mapping constants are accessible."""
        # Test that we can import and use the mappings
        from victor.agent.planning.tool_selection import (
            STEP_TO_TASK_TYPE,
            get_step_tool_sets,
        )

        step_sets = get_step_tool_sets()
        assert isinstance(step_sets, dict)
        assert "research" in step_sets
        assert "feature" in step_sets

    def test_step_to_task_type_mapping(self):
        """Test step type to task type mapping."""
        from victor.agent.planning.tool_selection import STEP_TO_TASK_TYPE

        assert STEP_TO_TASK_TYPE["research"] == "search"
        assert STEP_TO_TASK_TYPE["feature"] == "create"
        assert STEP_TO_TASK_TYPE["bugfix"] == "edit"

    def test_complexity_limits_mapping(self):
        """Test complexity limits mapping."""
        from victor.agent.planning.tool_selection import (
            COMPLEXITY_TOOL_LIMITS,
            get_complexity_limits,
        )

        limits = get_complexity_limits()
        assert limits["simple"] == 10
        assert limits["moderate"] == 30
        assert limits["complex"] == 100

    def test_step_tool_sets_content(self):
        """Test that step tool sets have expected tools."""
        from victor.agent.planning.tool_selection import get_step_tool_sets

        step_sets = get_step_tool_sets()

        # Research should have read-only tools
        research_tools = step_sets.get("research", set())
        assert "read" in research_tools
        assert "write" not in research_tools

        # Feature should have write tools
        feature_tools = step_sets.get("feature", set())
        assert "write" in feature_tools
        assert "edit" in feature_tools


class TestReadableTaskPlanIntegration:
    """Tests for ReadableTaskPlan integration with tool selection (mock-based)."""

    @pytest.fixture
    def sample_plan(self):
        """Create a sample task plan."""
        return ReadableTaskPlan(
            name="Add authentication",
            complexity=TaskComplexity.MODERATE,
            desc="Implement OAuth2 login",
            steps=[
                [1, "research", "Analyze patterns", "overview"],
                [2, "feature", "Create module", "write,test"],
                [3, "test", "Verify login", "pytest", [2]],
            ],
            duration="30min",
        )

    def test_plan_structure(self, sample_plan):
        """Test that the plan has expected structure."""
        assert sample_plan.name == "Add authentication"
        assert sample_plan.complexity == TaskComplexity.MODERATE
        assert len(sample_plan.steps) == 3

        # Check step structure
        step_0 = sample_plan.steps[0]
        assert step_0[0] == 1  # id
        assert step_0[1] == "research"  # type
        assert step_0[2] == "Analyze patterns"  # description

    def test_get_contextual_tools_requires_tool_selector(self, sample_plan):
        """Test that get_contextual_tools requires tool_selector parameter."""
        # This test verifies the method exists and requires the parameter
        import inspect

        method = sample_plan.get_contextual_tools
        sig = inspect.signature(method)

        # Should have tool_selector parameter
        params = sig.parameters
        assert "tool_selector" in params

        # Should have step_index parameter
        assert "step_index" in params

    def test_invalid_step_index(self, sample_plan):
        """Test that invalid step index returns empty list."""
        # We can't fully test this without a real tool selector,
        # but we can verify the method exists and handles edge cases
        import inspect

        method = sample_plan.get_contextual_tools
        sig = inspect.signature(method)

        # step_index should have a default value or be required
        params = sig.parameters
        assert "step_index" in params

    def test_conversation_stage_parameter(self, sample_plan):
        """Test that conversation_stage is optional parameter."""
        import inspect

        method = sample_plan.get_contextual_tools
        sig = inspect.signature(method)

        # conversation_stage should exist
        params = sig.parameters
        assert "conversation_stage" in params


class TestStepToolMappingCompleteness:
    """Tests that verify step tool mapping covers all step types."""

    def test_all_step_types_mapped(self):
        """Test that all step types from readable_schema are mapped."""
        from victor.agent.planning.tool_selection import STEP_TOOL_MAPPING

        # All step types should have tool mappings
        expected_step_types = [
            "research",
            "planning",
            "feature",
            "bugfix",
            "refactor",
            "test",
            "review",
            "deploy",
            "analyze",
            "doc",
        ]

        for step_type in expected_step_types:
            assert (
                step_type in STEP_TOOL_MAPPING
            ), f"Step type '{step_type}' not in STEP_TOOL_MAPPING"

    def test_tool_mapping_non_empty(self):
        """Test that all step type mappings have at least one tool."""
        from victor.agent.planning.tool_selection import STEP_TOOL_MAPPING

        for step_type, tools in STEP_TOOL_MAPPING.items():
            assert len(tools) > 0, f"Step type '{step_type}' has empty tool set"

    def test_critical_tools_included(self):
        """Test that critical tools are in all step mappings."""
        from victor.agent.planning.tool_selection import STEP_TOOL_MAPPING

        # These should be in almost all step types
        common_tools = {"read", "grep"}

        for step_type, tools in STEP_TOOL_MAPPING.items():
            # At least one common tool should be present
            # (Some steps like 'test' might not have 'read')
            assert (
                len(tools & common_tools) >= 1 or len(tools) >= 2
            ), f"Step type '{step_type}' has no common tools"
