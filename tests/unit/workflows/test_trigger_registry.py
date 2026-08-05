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

"""Comprehensive unit tests for WorkflowTriggerRegistry.

Tests cover:
- WorkflowTrigger dataclass behavior
- Pattern matching with regex
- Task type matching
- WorkflowTriggerRegistry singleton behavior
- Registration methods (register, register_from_vertical)
- Lookup methods (find_workflow, find_by_task_type, get_triggers_for_vertical)
- Listing methods (list_verticals, list_task_types)
- Clear and reset behavior
- Thread safety
- Module-level convenience functions
"""

import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pytest

from victor.workflows.trigger_registry import (
    WorkflowTrigger,
    WorkflowTriggerRegistry,
    get_trigger_registry,
    register_trigger,
    find_workflow_for_query,
    _registry_instance,
)

# ============ Fixtures ============


@pytest.fixture
def fresh_registry():
    """Create a fresh WorkflowTriggerRegistry instance for testing."""
    return WorkflowTriggerRegistry()


@pytest.fixture
def reset_global_registry():
    """Reset the global registry singleton before and after test."""
    import victor.workflows.trigger_registry as module

    # Store original state
    original_instance = module._registry_instance

    # Reset for test
    module._registry_instance = None

    yield

    # Restore original state
    module._registry_instance = original_instance


@pytest.fixture
def sample_trigger():
    """Create a sample WorkflowTrigger for testing."""
    return WorkflowTrigger(
        pattern=r"implement\s+.+feature",
        workflow_name="feature_implementation",
        vertical="coding",
        task_types=["feature", "implement"],
        priority=5,
        description="Implements a new feature",
    )


@pytest.fixture
def multiple_triggers():
    """Create multiple triggers across different verticals."""
    return [
        WorkflowTrigger(
            pattern=r"review\s+code",
            workflow_name="code_review",
            vertical="coding",
            task_types=["review", "code"],
            priority=3,
        ),
        WorkflowTrigger(
            pattern=r"deploy\s+to\s+production",
            workflow_name="production_deploy",
            vertical="devops",
            task_types=["deploy", "production"],
            priority=10,
        ),
        WorkflowTrigger(
            pattern=r"search\s+documents",
            workflow_name="document_search",
            vertical="rag",
            task_types=["search", "retrieve"],
            priority=1,
        ),
        WorkflowTrigger(
            pattern=r"analyze\s+data",
            workflow_name="data_analysis",
            vertical="dataanalysis",
            task_types=["analyze", "statistics"],
            priority=2,
        ),
    ]


# ============ WorkflowTrigger Tests ============


class TestWorkflowTrigger:
    """Tests for the WorkflowTrigger dataclass."""

    def test_trigger_creation_basic(self):
        """Test basic trigger creation with required fields."""
        trigger = WorkflowTrigger(
            pattern=r"test\s+pattern",
            workflow_name="test_workflow",
            vertical="coding",
        )

        assert trigger.pattern == r"test\s+pattern"
        assert trigger.workflow_name == "test_workflow"
        assert trigger.vertical == "coding"
        assert trigger.task_types == []
        assert trigger.priority == 0
        assert trigger.description == ""

    def test_trigger_creation_full(self, sample_trigger):
        """Test trigger creation with all fields."""
        assert sample_trigger.pattern == r"implement\s+.+feature"
        assert sample_trigger.workflow_name == "feature_implementation"
        assert sample_trigger.vertical == "coding"
        assert sample_trigger.task_types == ["feature", "implement"]
        assert sample_trigger.priority == 5
        assert sample_trigger.description == "Implements a new feature"

    def test_pattern_compilation(self):
        """Test that valid patterns are compiled on init."""
        trigger = WorkflowTrigger(
            pattern=r"test\s+\d+",
            workflow_name="test",
            vertical="coding",
        )

        assert trigger._compiled_pattern is not None
        assert trigger._compiled_pattern.pattern == r"test\s+\d+"

    def test_invalid_pattern_compilation(self):
        """Test that invalid patterns result in None compiled pattern."""
        # Invalid regex with unbalanced brackets
        trigger = WorkflowTrigger(
            pattern=r"test[invalid",
            workflow_name="test",
            vertical="coding",
        )

        assert trigger._compiled_pattern is None

    def test_empty_pattern(self):
        """Test trigger with empty pattern."""
        trigger = WorkflowTrigger(
            pattern="",
            workflow_name="test",
            vertical="coding",
        )

        assert trigger._compiled_pattern is None
        assert not trigger.matches_query("anything")

    def test_matches_query_positive(self, sample_trigger):
        """Test query matching with matching input."""
        assert sample_trigger.matches_query("implement new feature")
        assert sample_trigger.matches_query("implement the feature")
        assert sample_trigger.matches_query("please implement cool feature now")

    def test_matches_query_negative(self, sample_trigger):
        """Test query matching with non-matching input."""
        assert not sample_trigger.matches_query("review code")
        assert not sample_trigger.matches_query("implement")  # missing 'feature'
        assert not sample_trigger.matches_query("feature implementation")

    def test_matches_query_case_insensitive(self, sample_trigger):
        """Test that query matching is case insensitive."""
        assert sample_trigger.matches_query("IMPLEMENT NEW FEATURE")
        assert sample_trigger.matches_query("Implement New Feature")
        assert sample_trigger.matches_query("iMpLeMeNt NeW fEaTuRe")

    def test_matches_query_with_invalid_pattern(self):
        """Test query matching when pattern is invalid."""
        trigger = WorkflowTrigger(
            pattern=r"[invalid",
            workflow_name="test",
            vertical="coding",
        )

        assert not trigger.matches_query("anything")

    def test_matches_task_type_positive(self, sample_trigger):
        """Test task type matching with matching types."""
        assert sample_trigger.matches_task_type("feature")
        assert sample_trigger.matches_task_type("implement")

    def test_matches_task_type_negative(self, sample_trigger):
        """Test task type matching with non-matching types."""
        assert not sample_trigger.matches_task_type("review")
        assert not sample_trigger.matches_task_type("deploy")

    def test_matches_task_type_case_insensitive(self, sample_trigger):
        """Test that task type matching is case insensitive."""
        assert sample_trigger.matches_task_type("FEATURE")
        assert sample_trigger.matches_task_type("Feature")
        assert sample_trigger.matches_task_type("IMPLEMENT")

    def test_matches_task_type_empty_list(self):
        """Test task type matching when task_types is empty."""
        trigger = WorkflowTrigger(
            pattern=r"test",
            workflow_name="test",
            vertical="coding",
            task_types=[],
        )

        assert not trigger.matches_task_type("anything")


# ============ WorkflowTriggerRegistry Tests ============


class TestWorkflowTriggerRegistry:
    """Tests for WorkflowTriggerRegistry class."""

    def test_registry_initialization(self, fresh_registry):
        """Test that registry initializes with empty state."""
        assert fresh_registry._triggers == []
        assert fresh_registry._by_vertical == {}
        assert fresh_registry._by_task_type == {}

    def test_register_single_trigger(self, fresh_registry, sample_trigger):
        """Test registering a single trigger."""
        fresh_registry.register(sample_trigger)

        assert len(fresh_registry._triggers) == 1
        assert fresh_registry._triggers[0] == sample_trigger

    def test_register_indexes_by_vertical(self, fresh_registry, sample_trigger):
        """Test that register indexes trigger by vertical."""
        fresh_registry.register(sample_trigger)

        assert "coding" in fresh_registry._by_vertical
        assert sample_trigger in fresh_registry._by_vertical["coding"]

    def test_register_indexes_by_task_type(self, fresh_registry, sample_trigger):
        """Test that register indexes trigger by task types."""
        fresh_registry.register(sample_trigger)

        assert "feature" in fresh_registry._by_task_type
        assert "implement" in fresh_registry._by_task_type
        assert sample_trigger in fresh_registry._by_task_type["feature"]
        assert sample_trigger in fresh_registry._by_task_type["implement"]

    def test_register_multiple_triggers(self, fresh_registry, multiple_triggers):
        """Test registering multiple triggers."""
        for trigger in multiple_triggers:
            fresh_registry.register(trigger)

        assert len(fresh_registry._triggers) == 4
        assert len(fresh_registry._by_vertical) == 4

    def test_register_from_vertical(self, fresh_registry):
        """Test register_from_vertical method."""
        triggers = [
            (r"test\s+pattern1", "workflow1"),
            (r"test\s+pattern2", "workflow2"),
            (r"test\s+pattern3", "workflow3"),
        ]

        count = fresh_registry.register_from_vertical("testing", triggers)

        assert count == 3
        assert len(fresh_registry._triggers) == 3
        assert "testing" in fresh_registry._by_vertical
        assert len(fresh_registry._by_vertical["testing"]) == 3

    def test_register_from_vertical_empty(self, fresh_registry):
        """Test register_from_vertical with empty list."""
        count = fresh_registry.register_from_vertical("testing", [])

        assert count == 0
        assert len(fresh_registry._triggers) == 0

    def test_register_from_vertical_preserves_vertical(self, fresh_registry):
        """Test that register_from_vertical sets correct vertical on triggers."""
        triggers = [(r"pattern", "workflow")]

        fresh_registry.register_from_vertical("my_vertical", triggers)

        assert fresh_registry._triggers[0].vertical == "my_vertical"


# ============ Find Workflow Tests ============


class TestFindWorkflow:
    """Tests for find_workflow method."""

    def test_find_workflow_basic(self, fresh_registry, sample_trigger):
        """Test basic workflow finding."""
        fresh_registry.register(sample_trigger)

        result = fresh_registry.find_workflow("implement new feature")

        assert result is not None
        assert result == ("feature_implementation", "coding")

    def test_find_workflow_no_match(self, fresh_registry, sample_trigger):
        """Test workflow finding with no matching query."""
        fresh_registry.register(sample_trigger)

        result = fresh_registry.find_workflow("review code changes")

        assert result is None

    def test_find_workflow_empty_registry(self, fresh_registry):
        """Test workflow finding with empty registry."""
        result = fresh_registry.find_workflow("any query")

        assert result is None

    def test_find_workflow_priority_ordering(self, fresh_registry):
        """Test that higher priority triggers are matched first."""
        low_priority = WorkflowTrigger(
            pattern=r"test\s+something",
            workflow_name="low_priority_workflow",
            vertical="coding",
            priority=1,
        )
        high_priority = WorkflowTrigger(
            pattern=r"test\s+something",
            workflow_name="high_priority_workflow",
            vertical="coding",
            priority=10,
        )

        fresh_registry.register(low_priority)
        fresh_registry.register(high_priority)

        result = fresh_registry.find_workflow("test something here")

        assert result == ("high_priority_workflow", "coding")

    def test_find_workflow_preferred_vertical(self, fresh_registry):
        """Test workflow finding with preferred vertical."""
        coding_trigger = WorkflowTrigger(
            pattern=r"analyze\s+code",
            workflow_name="code_analysis",
            vertical="coding",
            priority=5,
        )
        devops_trigger = WorkflowTrigger(
            pattern=r"analyze\s+code",
            workflow_name="devops_analysis",
            vertical="devops",
            priority=5,
        )

        fresh_registry.register(coding_trigger)
        fresh_registry.register(devops_trigger)

        # Without preference - returns first registered
        result1 = fresh_registry.find_workflow("analyze code")
        assert result1 is not None

        # With preference - returns preferred vertical
        result2 = fresh_registry.find_workflow("analyze code", preferred_vertical="devops")
        assert result2 == ("devops_analysis", "devops")

    def test_find_workflow_priority_beats_preference(self, fresh_registry):
        """Test that higher priority beats preferred vertical."""
        low_prio_preferred = WorkflowTrigger(
            pattern=r"run\s+tests",
            workflow_name="preferred_workflow",
            vertical="preferred",
            priority=1,
        )
        high_prio_other = WorkflowTrigger(
            pattern=r"run\s+tests",
            workflow_name="high_priority_workflow",
            vertical="other",
            priority=10,
        )

        fresh_registry.register(low_prio_preferred)
        fresh_registry.register(high_prio_other)

        result = fresh_registry.find_workflow("run tests", preferred_vertical="preferred")

        assert result == ("high_priority_workflow", "other")


# ============ Find By Task Type Tests ============


class TestFindByTaskType:
    """Tests for find_by_task_type method."""

    def test_find_by_task_type_basic(self, fresh_registry, sample_trigger):
        """Test basic task type lookup."""
        fresh_registry.register(sample_trigger)

        result = fresh_registry.find_by_task_type("feature")

        assert result is not None
        assert result == ("feature_implementation", "coding")

    def test_find_by_task_type_case_insensitive(self, fresh_registry, sample_trigger):
        """Test that task type lookup is case insensitive."""
        fresh_registry.register(sample_trigger)

        result = fresh_registry.find_by_task_type("FEATURE")

        assert result == ("feature_implementation", "coding")

    def test_find_by_task_type_no_match(self, fresh_registry, sample_trigger):
        """Test task type lookup with no match."""
        fresh_registry.register(sample_trigger)

        result = fresh_registry.find_by_task_type("nonexistent")

        assert result is None

    def test_find_by_task_type_empty_registry(self, fresh_registry):
        """Test task type lookup with empty registry."""
        result = fresh_registry.find_by_task_type("anything")

        assert result is None

    def test_find_by_task_type_priority_ordering(self, fresh_registry):
        """Test that task type lookup respects priority."""
        low_priority = WorkflowTrigger(
            pattern=r"test",
            workflow_name="low_priority",
            vertical="coding",
            task_types=["deploy"],
            priority=1,
        )
        high_priority = WorkflowTrigger(
            pattern=r"test",
            workflow_name="high_priority",
            vertical="coding",
            task_types=["deploy"],
            priority=10,
        )

        fresh_registry.register(low_priority)
        fresh_registry.register(high_priority)

        result = fresh_registry.find_by_task_type("deploy")

        assert result == ("high_priority", "coding")

    def test_find_by_task_type_preferred_vertical(self, fresh_registry):
        """Test task type lookup with preferred vertical."""
        coding_trigger = WorkflowTrigger(
            pattern=r"test",
            workflow_name="coding_workflow",
            vertical="coding",
            task_types=["analyze"],
            priority=5,
        )
        rag_trigger = WorkflowTrigger(
            pattern=r"test",
            workflow_name="rag_workflow",
            vertical="rag",
            task_types=["analyze"],
            priority=5,
        )

        fresh_registry.register(coding_trigger)
        fresh_registry.register(rag_trigger)

        result = fresh_registry.find_by_task_type("analyze", preferred_vertical="rag")

        assert result == ("rag_workflow", "rag")


# ============ Get Triggers Tests ============


class TestGetTriggers:
    """Tests for get_triggers_for_vertical method."""

    def test_get_triggers_for_vertical(self, fresh_registry, multiple_triggers):
        """Test getting triggers for a specific vertical."""
        for trigger in multiple_triggers:
            fresh_registry.register(trigger)

        coding_triggers = fresh_registry.get_triggers_for_vertical("coding")

        assert len(coding_triggers) == 1
        assert coding_triggers[0].workflow_name == "code_review"

    def test_get_triggers_for_nonexistent_vertical(self, fresh_registry, sample_trigger):
        """Test getting triggers for non-existent vertical."""
        fresh_registry.register(sample_trigger)

        result = fresh_registry.get_triggers_for_vertical("nonexistent")

        assert result == []

    def test_get_triggers_returns_copy(self, fresh_registry, sample_trigger):
        """Test that get_triggers_for_vertical returns a copy."""
        fresh_registry.register(sample_trigger)

        triggers = fresh_registry.get_triggers_for_vertical("coding")
        triggers.append("modified")

        # Original should be unchanged
        assert len(fresh_registry._by_vertical["coding"]) == 1


# ============ Listing Methods Tests ============


class TestListingMethods:
    """Tests for list_verticals and list_task_types methods."""

    def test_list_verticals(self, fresh_registry, multiple_triggers):
        """Test listing all verticals."""
        for trigger in multiple_triggers:
            fresh_registry.register(trigger)

        verticals = fresh_registry.list_verticals()

        assert len(verticals) == 4
        assert "coding" in verticals
        assert "devops" in verticals
        assert "rag" in verticals
        assert "dataanalysis" in verticals

    def test_list_verticals_empty(self, fresh_registry):
        """Test listing verticals with empty registry."""
        verticals = fresh_registry.list_verticals()

        assert verticals == []

    def test_list_task_types(self, fresh_registry, multiple_triggers):
        """Test listing all task types."""
        for trigger in multiple_triggers:
            fresh_registry.register(trigger)

        task_types = fresh_registry.list_task_types()

        assert "review" in task_types
        assert "code" in task_types
        assert "deploy" in task_types
        assert "production" in task_types
        assert "search" in task_types
        assert "retrieve" in task_types

    def test_list_task_types_empty(self, fresh_registry):
        """Test listing task types with empty registry."""
        task_types = fresh_registry.list_task_types()

        assert task_types == []


# ============ Clear Method Tests ============


class TestClearMethod:
    """Tests for clear method."""

    def test_clear_removes_all_triggers(self, fresh_registry, multiple_triggers):
        """Test that clear removes all triggers."""
        for trigger in multiple_triggers:
            fresh_registry.register(trigger)

        fresh_registry.clear()

        assert fresh_registry._triggers == []
        assert fresh_registry._by_vertical == {}
        assert fresh_registry._by_task_type == {}

    def test_clear_empty_registry(self, fresh_registry):
        """Test clearing an already empty registry."""
        fresh_registry.clear()

        assert fresh_registry._triggers == []


# ============ To Auto Workflows Tests ============


class TestToAutoWorkflows:
    """Tests for to_auto_workflows method."""

    def test_to_auto_workflows_basic(self, fresh_registry):
        """Test converting triggers to auto_workflows format."""
        triggers = [
            (r"pattern1", "workflow1"),
            (r"pattern2", "workflow2"),
        ]
        fresh_registry.register_from_vertical("coding", triggers)

        result = fresh_registry.to_auto_workflows("coding")

        assert len(result) == 2
        assert (r"pattern1", "workflow1") in result
        assert (r"pattern2", "workflow2") in result

    def test_to_auto_workflows_nonexistent_vertical(self, fresh_registry, sample_trigger):
        """Test to_auto_workflows for non-existent vertical."""
        fresh_registry.register(sample_trigger)

        result = fresh_registry.to_auto_workflows("nonexistent")

        assert result == []

    def test_to_auto_workflows_filters_empty_patterns(self, fresh_registry):
        """Test that to_auto_workflows filters out empty patterns."""
        trigger_with_pattern = WorkflowTrigger(
            pattern=r"test",
            workflow_name="with_pattern",
            vertical="coding",
        )
        trigger_without_pattern = WorkflowTrigger(
            pattern="",
            workflow_name="without_pattern",
            vertical="coding",
        )

        fresh_registry.register(trigger_with_pattern)
        fresh_registry.register(trigger_without_pattern)

        result = fresh_registry.to_auto_workflows("coding")

        assert len(result) == 1
        assert result[0] == (r"test", "with_pattern")


# ============ Singleton Tests ============


class TestSingletonBehavior:
    """Tests for singleton behavior of get_trigger_registry."""

    def test_get_trigger_registry_returns_same_instance(self, reset_global_registry):
        """Test that get_trigger_registry returns the same instance."""
        registry1 = get_trigger_registry()
        registry2 = get_trigger_registry()

        assert registry1 is registry2

    def test_get_trigger_registry_creates_instance(self, reset_global_registry):
        """Test that get_trigger_registry creates instance when None."""
        registry = get_trigger_registry()

        assert isinstance(registry, WorkflowTriggerRegistry)


# ============ Module-Level Functions Tests ============


class TestModuleLevelFunctions:
    """Tests for module-level convenience functions."""

    def test_register_trigger_function(self, reset_global_registry):
        """Test register_trigger convenience function."""
        trigger = WorkflowTrigger(
            pattern=r"test",
            workflow_name="test_workflow",
            vertical="testing",
        )

        register_trigger(trigger)

        registry = get_trigger_registry()
        assert len(registry._triggers) == 1
        assert registry._triggers[0] == trigger

    def test_find_workflow_for_query_function(self, reset_global_registry):
        """Test find_workflow_for_query convenience function."""
        trigger = WorkflowTrigger(
            pattern=r"search\s+query",
            workflow_name="search_workflow",
            vertical="research",
        )
        register_trigger(trigger)

        result = find_workflow_for_query("search query here")

        assert result == ("search_workflow", "research")

    def test_find_workflow_for_query_with_preference(self, reset_global_registry):
        """Test find_workflow_for_query with preferred vertical."""
        trigger1 = WorkflowTrigger(
            pattern=r"find\s+data",
            workflow_name="workflow1",
            vertical="rag",
            priority=5,
        )
        trigger2 = WorkflowTrigger(
            pattern=r"find\s+data",
            workflow_name="workflow2",
            vertical="research",
            priority=5,
        )

        register_trigger(trigger1)
        register_trigger(trigger2)

        result = find_workflow_for_query("find data", preferred_vertical="research")

        assert result == ("workflow2", "research")

    def test_find_workflow_for_query_no_match(self, reset_global_registry):
        """Test find_workflow_for_query with no matching query."""
        trigger = WorkflowTrigger(
            pattern=r"specific\s+pattern",
            workflow_name="workflow",
            vertical="testing",
        )
        register_trigger(trigger)

        result = find_workflow_for_query("unrelated query")

        assert result is None


# ============ Thread Safety Tests ============


class TestThreadSafety:
    """Tests for thread safety of the registry."""

    def test_concurrent_registration(self, fresh_registry):
        """Test that concurrent registrations don't cause issues."""
        triggers = []
        for i in range(100):
            triggers.append(
                WorkflowTrigger(
                    pattern=rf"pattern{i}",
                    workflow_name=f"workflow{i}",
                    vertical=f"vertical{i % 5}",
                    task_types=[f"type{i}"],
                )
            )

        def register_trigger(trigger):
            fresh_registry.register(trigger)

        with ThreadPoolExecutor(max_workers=10) as executor:
            executor.map(register_trigger, triggers)

        assert len(fresh_registry._triggers) == 100

    def test_concurrent_reads_and_writes(self, fresh_registry):
        """Test concurrent reads and writes."""
        # Pre-populate with some triggers
        for i in range(10):
            fresh_registry.register(
                WorkflowTrigger(
                    pattern=rf"existing{i}",
                    workflow_name=f"existing_workflow{i}",
                    vertical="existing",
                )
            )

        results = []
        errors = []

        def reader():
            try:
                for _ in range(50):
                    fresh_registry.find_workflow(f"existing{5}")
                    fresh_registry.list_verticals()
                    fresh_registry.get_triggers_for_vertical("existing")
                results.append("read_success")
            except Exception as e:
                errors.append(str(e))

        def writer():
            try:
                for i in range(50):
                    fresh_registry.register(
                        WorkflowTrigger(
                            pattern=rf"new{i}",
                            workflow_name=f"new_workflow{i}",
                            vertical="new",
                        )
                    )
                results.append("write_success")
            except Exception as e:
                errors.append(str(e))

        threads = []
        for _ in range(5):
            threads.append(threading.Thread(target=reader))
            threads.append(threading.Thread(target=writer))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 10  # 5 readers + 5 writers


# ============ Edge Cases Tests ============


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_register_trigger_with_none_task_types(self, fresh_registry):
        """Test registration when task_types defaults to empty list."""
        trigger = WorkflowTrigger(
            pattern=r"test",
            workflow_name="workflow",
            vertical="coding",
        )

        fresh_registry.register(trigger)

        # Should not crash and task_types should be empty
        assert trigger.task_types == []
        assert len(fresh_registry._triggers) == 1

    def test_find_workflow_with_empty_query(self, fresh_registry, sample_trigger):
        """Test finding workflow with empty query string."""
        fresh_registry.register(sample_trigger)

        result = fresh_registry.find_workflow("")

        assert result is None

    def test_special_regex_characters_in_pattern(self, fresh_registry):
        """Test patterns with special regex characters."""
        trigger = WorkflowTrigger(
            pattern=r"test\[bracket\]",
            workflow_name="bracket_workflow",
            vertical="coding",
        )

        fresh_registry.register(trigger)

        result = fresh_registry.find_workflow("test[bracket]")

        assert result == ("bracket_workflow", "coding")

    def test_unicode_in_patterns(self, fresh_registry):
        """Test unicode characters in patterns and queries."""
        trigger = WorkflowTrigger(
            pattern=r"recherche\s+donn.es",
            workflow_name="unicode_workflow",
            vertical="research",
        )

        fresh_registry.register(trigger)

        result = fresh_registry.find_workflow("recherche donnees")

        assert result == ("unicode_workflow", "research")

    def test_very_long_pattern(self, fresh_registry):
        """Test with very long regex pattern."""
        long_pattern = r"test" + r"\s+word" * 50
        trigger = WorkflowTrigger(
            pattern=long_pattern,
            workflow_name="long_pattern_workflow",
            vertical="coding",
        )

        fresh_registry.register(trigger)

        long_query = "test" + " word" * 50
        result = fresh_registry.find_workflow(long_query)

        assert result == ("long_pattern_workflow", "coding")

    def test_duplicate_task_types(self, fresh_registry):
        """Test trigger with duplicate task types."""
        trigger = WorkflowTrigger(
            pattern=r"test",
            workflow_name="workflow",
            vertical="coding",
            task_types=["type1", "type1", "type2"],
        )

        fresh_registry.register(trigger)

        # Should register both (duplicate and unique)
        assert "type1" in fresh_registry._by_task_type
        assert "type2" in fresh_registry._by_task_type

    def test_find_workflow_with_whitespace_query(self, fresh_registry, sample_trigger):
        """Test finding workflow with whitespace-only query."""
        fresh_registry.register(sample_trigger)

        result = fresh_registry.find_workflow("   \t\n  ")

        assert result is None

    def test_clear_then_register(self, fresh_registry, sample_trigger):
        """Test that registry works after clear."""
        fresh_registry.register(sample_trigger)
        fresh_registry.clear()

        new_trigger = WorkflowTrigger(
            pattern=r"new\s+pattern",
            workflow_name="new_workflow",
            vertical="new",
        )
        fresh_registry.register(new_trigger)

        assert len(fresh_registry._triggers) == 1
        assert fresh_registry._triggers[0] == new_trigger

    def test_multiple_verticals_same_workflow_name(self, fresh_registry):
        """Test different verticals with same workflow name."""
        trigger1 = WorkflowTrigger(
            pattern=r"analyze",
            workflow_name="analyze_workflow",
            vertical="coding",
        )
        trigger2 = WorkflowTrigger(
            pattern=r"analyze\s+data",
            workflow_name="analyze_workflow",
            vertical="dataanalysis",
        )

        fresh_registry.register(trigger1)
        fresh_registry.register(trigger2)

        # Both should be registered
        assert len(fresh_registry._triggers) == 2
        assert len(fresh_registry.list_verticals()) == 2
