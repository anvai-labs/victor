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

"""Tests for the unified TaskTypeRegistry."""

import pytest

from victor.framework.task_types import (
    TaskCategory,
    TaskTypeDefinition,
    TaskTypeRegistry,
    get_task_budget,
    get_task_hint,
    get_task_type_registry,
    register_vertical_task_type,
    register_devops_task_types,
    register_data_analysis_task_types,
    register_coding_task_types,
    register_research_task_types,
    setup_vertical_task_types,
)


class TestTaskTypeDefinition:
    """Tests for TaskTypeDefinition dataclass."""

    def test_creation_with_defaults(self):
        """Test creating a definition with default values."""
        defn = TaskTypeDefinition(
            name="test_task",
            category=TaskCategory.ANALYSIS,
        )
        assert defn.name == "test_task"
        assert defn.category == TaskCategory.ANALYSIS
        assert defn.hint == ""
        assert defn.tool_budget == 10
        assert defn.priority_tools == []
        assert defn.max_iterations == 30
        assert defn.aliases == set()
        assert defn.vertical is None
        assert defn.needs_tools is True

    def test_creation_with_all_fields(self):
        """Test creating a definition with all fields specified."""
        defn = TaskTypeDefinition(
            name="CUSTOM_TASK",  # Should be normalized to lowercase
            category=TaskCategory.MODIFICATION,
            hint="[CUSTOM] A custom task type",
            tool_budget=25,
            priority_tools=["read_file", "edit_files"],
            max_iterations=15,
            aliases=["custom", "my_task"],
            vertical="my_vertical",
            needs_tools=False,
            force_action_after_read=True,
            stage_tools={"initial": ["list_directory"]},
            force_action_hints={"max_iterations": "Complete the task"},
            exploration_multiplier=1.5,
        )
        assert defn.name == "custom_task"  # Normalized
        assert defn.category == TaskCategory.MODIFICATION
        assert defn.hint == "[CUSTOM] A custom task type"
        assert defn.tool_budget == 25
        assert defn.priority_tools == ["read", "edit"]
        assert defn.max_iterations == 15
        assert defn.aliases == {"custom", "my_task"}
        assert defn.vertical == "my_vertical"
        assert defn.needs_tools is False
        assert defn.force_action_after_read is True
        assert defn.stage_tools == {"initial": ["ls"]}
        assert defn.force_action_hints == {"max_iterations": "Complete the task"}
        assert defn.exploration_multiplier == 1.5

    def test_aliases_converted_to_set(self):
        """Test that aliases list is converted to set."""
        defn = TaskTypeDefinition(
            name="test",
            category=TaskCategory.ANALYSIS,
            aliases=["alias1", "alias2", "alias1"],  # Duplicate
        )
        assert isinstance(defn.aliases, set)
        assert defn.aliases == {"alias1", "alias2"}


class TestTaskTypeRegistry:
    """Tests for TaskTypeRegistry singleton."""

    @pytest.fixture(autouse=True)
    def reset_registry(self):
        """Reset the registry before each test."""
        TaskTypeRegistry.reset_instance()
        yield
        TaskTypeRegistry.reset_instance()

    def test_singleton_pattern(self):
        """Test that get_instance returns the same instance."""
        registry1 = TaskTypeRegistry.get_instance()
        registry2 = TaskTypeRegistry.get_instance()
        assert registry1 is registry2

    def test_reset_instance(self):
        """Test that reset_instance creates a new instance."""
        registry1 = TaskTypeRegistry.get_instance()
        TaskTypeRegistry.reset_instance()
        registry2 = TaskTypeRegistry.get_instance()
        # After reset, should get a new instance
        assert registry1 is not registry2

    def test_default_types_registered(self):
        """Test that default task types are registered."""
        registry = TaskTypeRegistry.get_instance()

        # Check core types exist
        assert registry.get("edit") is not None
        assert registry.get("create") is not None
        assert registry.get("search") is not None
        assert registry.get("analyze") is not None
        assert registry.get("general") is not None
        assert registry.get("design") is not None
        assert registry.get("action") is not None
        assert registry.get("research") is not None

    def test_register_custom_type(self):
        """Test registering a custom task type."""
        registry = TaskTypeRegistry.get_instance()

        defn = TaskTypeDefinition(
            name="my_custom_type",
            category=TaskCategory.EXECUTION,
            hint="[CUSTOM] My custom type",
            tool_budget=20,
        )
        registry.register(defn)

        result = registry.get("my_custom_type")
        assert result is not None
        assert result.name == "my_custom_type"
        assert result.hint == "[CUSTOM] My custom type"
        assert result.tool_budget == 20

    def test_alias_resolution(self):
        """Test that aliases resolve to canonical names."""
        registry = TaskTypeRegistry.get_instance()

        # "fix" is an alias for "edit"
        edit_defn = registry.get("edit")
        fix_defn = registry.get("fix")

        assert edit_defn is not None
        assert fix_defn is not None
        assert edit_defn.name == fix_defn.name == "edit"

    def test_resolve_alias_function(self):
        """Test the resolve_alias method."""
        registry = TaskTypeRegistry.get_instance()

        assert registry.resolve_alias("fix") == "edit"
        assert registry.resolve_alias("modify") == "edit"
        assert registry.resolve_alias("generate") == "create_simple"
        assert registry.resolve_alias("unknown") == "unknown"

    def test_get_hint(self):
        """Test getting hints for task types."""
        registry = TaskTypeRegistry.get_instance()

        hint = registry.get_hint("edit")
        assert "[EDIT]" in hint

        hint = registry.get_hint("search")
        assert "[SEARCH]" in hint

        # Unknown type returns empty string
        hint = registry.get_hint("nonexistent")
        assert hint == ""

    def test_get_tool_budget(self):
        """Test getting tool budgets for task types."""
        registry = TaskTypeRegistry.get_instance()

        budget = registry.get_tool_budget("edit")
        assert budget == 25

        budget = registry.get_tool_budget("simple")
        assert budget == 2

        budget = registry.get_tool_budget("analyze")
        assert budget == 40

        # Unknown type returns default of 10
        budget = registry.get_tool_budget("nonexistent")
        assert budget == 10

    def test_get_max_iterations(self):
        """Test getting max iterations for task types."""
        registry = TaskTypeRegistry.get_instance()

        iterations = registry.get_max_iterations("edit")
        assert iterations == 10

        iterations = registry.get_max_iterations("analyze")
        assert iterations == 20

        # Unknown type returns default of 30
        iterations = registry.get_max_iterations("nonexistent")
        assert iterations == 30

    def test_get_priority_tools(self):
        """Test getting priority tools for task types."""
        registry = TaskTypeRegistry.get_instance()

        tools = registry.get_priority_tools("edit")
        assert "read" in tools
        assert "edit" in tools

        tools = registry.get_priority_tools("search")
        assert "code_search" in tools

        # Unknown type returns empty list
        tools = registry.get_priority_tools("nonexistent")
        assert tools == []

    def test_get_category(self):
        """Test getting category for task types."""
        registry = TaskTypeRegistry.get_instance()

        category = registry.get_category("edit")
        assert category == TaskCategory.MODIFICATION

        category = registry.get_category("search")
        assert category == TaskCategory.ANALYSIS

        category = registry.get_category("action")
        assert category == TaskCategory.EXECUTION

        category = registry.get_category("general")
        assert category == TaskCategory.CONVERSATION

        # Unknown type returns None
        category = registry.get_category("nonexistent")
        assert category is None

    def test_list_types(self):
        """Test listing all registered types."""
        registry = TaskTypeRegistry.get_instance()

        types = registry.list_types()
        assert "edit" in types
        assert "create" in types
        assert "search" in types
        assert len(types) >= 10  # At least core types

    def test_list_verticals_empty_initially(self):
        """Test listing verticals is empty before registration."""
        registry = TaskTypeRegistry.get_instance()

        verticals = registry.list_verticals()
        assert verticals == []


class TestVerticalRegistration:
    """Tests for vertical-specific task type registration."""

    @pytest.fixture(autouse=True)
    def reset_registry(self):
        """Reset the registry before each test."""
        TaskTypeRegistry.reset_instance()
        yield
        TaskTypeRegistry.reset_instance()

    def test_register_for_vertical(self):
        """Test registering a task type for a specific vertical."""
        registry = TaskTypeRegistry.get_instance()

        defn = TaskTypeDefinition(
            name="infrastructure",
            category=TaskCategory.MODIFICATION,
            hint="[INFRA] Infrastructure task",
            tool_budget=30,
        )
        registry.register_for_vertical("devops", defn)

        # Should be available with vertical specified
        result = registry.get("infrastructure", vertical="devops")
        assert result is not None
        assert result.name == "infrastructure"
        assert result.vertical == "devops"

        # Not available without vertical
        result = registry.get("infrastructure")
        assert result is None

    def test_vertical_override(self):
        """Test that vertical-specific definitions override core ones."""
        registry = TaskTypeRegistry.get_instance()

        # Get core "edit" definition
        core_edit = registry.get("edit")
        assert core_edit is not None
        core_budget = core_edit.tool_budget

        # Register vertical-specific "edit"
        registry.register_for_vertical(
            "devops",
            TaskTypeDefinition(
                name="edit",
                category=TaskCategory.MODIFICATION,
                hint="[DEVOPS EDIT] Special edit for DevOps",
                tool_budget=50,
            ),
        )

        # With vertical, should get override
        devops_edit = registry.get("edit", vertical="devops")
        assert devops_edit is not None
        assert devops_edit.tool_budget == 50
        assert "DEVOPS" in devops_edit.hint

        # Without vertical, should get core
        core_again = registry.get("edit")
        assert core_again is not None
        assert core_again.tool_budget == core_budget

    def test_list_types_with_vertical(self):
        """Test listing types includes vertical-specific ones."""
        registry = TaskTypeRegistry.get_instance()

        registry.register_for_vertical(
            "devops",
            TaskTypeDefinition(
                name="kubernetes",
                category=TaskCategory.MODIFICATION,
                hint="[K8S] Kubernetes",
                tool_budget=30,
            ),
        )

        # Without vertical, kubernetes not listed
        types = registry.list_types()
        assert "kubernetes" not in types

        # With vertical, kubernetes is listed
        types = registry.list_types(vertical="devops")
        assert "kubernetes" in types

    def test_list_verticals(self):
        """Test listing registered verticals."""
        registry = TaskTypeRegistry.get_instance()

        registry.register_for_vertical(
            "devops",
            TaskTypeDefinition(
                name="test1",
                category=TaskCategory.MODIFICATION,
                hint="",
                tool_budget=10,
            ),
        )
        registry.register_for_vertical(
            "data_analysis",
            TaskTypeDefinition(
                name="test2", category=TaskCategory.ANALYSIS, hint="", tool_budget=10
            ),
        )

        verticals = registry.list_verticals()
        assert "devops" in verticals
        assert "data_analysis" in verticals

    def test_register_vertical_task_type_function(self):
        """Test the convenience function for vertical registration."""
        register_vertical_task_type(
            vertical="test_vertical",
            name="test_task",
            category=TaskCategory.EXECUTION,
            hint="[TEST] Test task",
            tool_budget=15,
            priority_tools=["execute_bash"],
        )

        registry = TaskTypeRegistry.get_instance()
        result = registry.get("test_task", vertical="test_vertical")

        assert result is not None
        assert result.name == "test_task"
        assert result.hint == "[TEST] Test task"
        assert result.tool_budget == 15
        assert result.priority_tools == ["shell"]


class TestVerticalRegistrationHooks:
    """Tests for vertical registration hook functions."""

    @pytest.fixture(autouse=True)
    def reset_registry(self):
        """Reset the registry before each test."""
        TaskTypeRegistry.reset_instance()
        yield
        TaskTypeRegistry.reset_instance()

    def test_register_devops_task_types(self):
        """Test DevOps task type registration."""
        registry = TaskTypeRegistry.get_instance()
        register_devops_task_types(registry)

        # Check DevOps-specific types
        assert registry.get("infrastructure", vertical="devops") is not None
        assert registry.get("kubernetes", vertical="devops") is not None
        assert registry.get("terraform", vertical="devops") is not None
        assert registry.get("dockerfile", vertical="devops") is not None
        assert registry.get("ci_cd", vertical="devops") is not None

        # Check DevOps override for "edit"
        devops_edit = registry.get("edit", vertical="devops")
        assert devops_edit is not None
        assert "DEVOPS" in devops_edit.hint

    def test_register_data_analysis_task_types(self):
        """Test Data Analysis task type registration."""
        registry = TaskTypeRegistry.get_instance()
        register_data_analysis_task_types(registry)

        # Check Data Analysis-specific types
        assert registry.get("data_profiling", vertical="data_analysis") is not None
        assert registry.get("statistical_analysis", vertical="data_analysis") is not None
        assert registry.get("regression", vertical="data_analysis") is not None
        assert registry.get("clustering", vertical="data_analysis") is not None
        assert registry.get("visualization", vertical="data_analysis") is not None

        # Check Data Analysis override for "analyze"
        da_analyze = registry.get("analyze", vertical="data_analysis")
        assert da_analyze is not None
        assert da_analyze.tool_budget == 50  # Higher budget for data analysis

    def test_register_coding_task_types(self):
        """Test Coding task type registration."""
        registry = TaskTypeRegistry.get_instance()
        register_coding_task_types(registry)

        # Check Coding-specific types
        assert registry.get("code_generation", vertical="coding") is not None

        # Check Coding overrides
        coding_refactor = registry.get("refactor", vertical="coding")
        assert coding_refactor is not None
        assert coding_refactor.tool_budget == 35

    def test_register_research_task_types(self):
        """Test Research task type registration."""
        registry = TaskTypeRegistry.get_instance()
        register_research_task_types(registry)

        # Check Research-specific types
        assert registry.get("general_query", vertical="research") is not None

        # Check Research override
        research = registry.get("research", vertical="research")
        assert research is not None
        assert research.tool_budget == 40

    def test_setup_vertical_task_types(self):
        """Test the setup function registers all verticals."""
        setup_vertical_task_types()

        registry = TaskTypeRegistry.get_instance()
        verticals = registry.list_verticals()

        assert "devops" in verticals
        assert "data_analysis" in verticals
        assert "coding" in verticals
        assert "research" in verticals


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    @pytest.fixture(autouse=True)
    def reset_registry(self):
        """Reset the registry before each test."""
        TaskTypeRegistry.reset_instance()
        yield
        TaskTypeRegistry.reset_instance()

    def test_get_task_type_registry(self):
        """Test get_task_type_registry returns the singleton."""
        registry = get_task_type_registry()
        assert registry is TaskTypeRegistry.get_instance()

    def test_get_task_hint_function(self):
        """Test the convenience function for getting hints."""
        hint = get_task_hint("edit")
        assert "[EDIT]" in hint

        # With vertical
        setup_vertical_task_types()
        hint = get_task_hint("edit", vertical="devops")
        assert "DEVOPS" in hint

    def test_get_task_budget_function(self):
        """Test the convenience function for getting budgets."""
        budget = get_task_budget("edit")
        assert budget == 25

        # With vertical
        setup_vertical_task_types()
        budget = get_task_budget("analyze", vertical="data_analysis")
        assert budget == 50


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.fixture(autouse=True)
    def reset_registry(self):
        """Reset the registry before each test."""
        TaskTypeRegistry.reset_instance()
        yield
        TaskTypeRegistry.reset_instance()

    def test_case_insensitivity(self):
        """Test that lookups are case-insensitive."""
        registry = TaskTypeRegistry.get_instance()

        assert registry.get("EDIT") is not None
        assert registry.get("Edit") is not None
        assert registry.get("eDiT") is not None

        # All should resolve to the same definition
        assert registry.get("EDIT").name == "edit"

    def test_vertical_case_insensitivity(self):
        """Test that vertical names are case-insensitive."""
        registry = TaskTypeRegistry.get_instance()

        registry.register_for_vertical(
            "DevOps",
            TaskTypeDefinition(
                name="test",
                category=TaskCategory.MODIFICATION,
                hint="Test",
                tool_budget=10,
            ),
        )

        assert registry.get("test", vertical="devops") is not None
        assert registry.get("test", vertical="DEVOPS") is not None
        assert registry.get("test", vertical="DevOps") is not None

    def test_nonexistent_type_returns_none(self):
        """Test that looking up nonexistent type returns None."""
        registry = TaskTypeRegistry.get_instance()

        assert registry.get("totally_nonexistent_type") is None

    def test_nonexistent_vertical_falls_back_to_core(self):
        """Test that nonexistent vertical falls back to core definition."""
        registry = TaskTypeRegistry.get_instance()

        # "edit" exists in core
        result = registry.get("edit", vertical="nonexistent_vertical")
        assert result is not None
        assert result.name == "edit"

    def test_registration_hook(self):
        """Test that registration hooks are called."""
        TaskTypeRegistry.reset_instance()

        hook_called = []

        def my_hook(registry):
            hook_called.append(True)
            registry.register(
                TaskTypeDefinition(
                    name="from_hook",
                    category=TaskCategory.ANALYSIS,
                    hint="From hook",
                    tool_budget=10,
                )
            )

        # Add hook before get_instance
        new_registry = TaskTypeRegistry()
        new_registry.add_registration_hook(my_hook)
        new_registry._register_defaults()

        assert len(hook_called) == 1
        assert new_registry.get("from_hook") is not None
