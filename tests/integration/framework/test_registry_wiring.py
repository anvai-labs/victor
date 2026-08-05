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

"""Integration tests for registry wiring in FrameworkStepHandler.

Tests the integration between verticals and global registries:
- WorkflowTriggerRegistry population from verticals
- TeamSpecRegistry population from verticals
- ToolGraphRegistry population from verticals
- HandlerRegistry population from verticals
"""

import pytest
from unittest.mock import MagicMock, patch
from typing import Dict, Any, Optional, List

from victor.core.verticals.base import VerticalBase
from victor.core.verticals.import_resolver import import_module_with_fallback
from victor.agent.vertical_context import VerticalContext


def _load_vertical_attr(module_path: str, attr_name: str):
    """Resolve a vertical attribute and skip test when unavailable."""
    module, _resolved = import_module_with_fallback(module_path)
    if module is None or not hasattr(module, attr_name):
        pytest.skip(f"Vertical module or attribute unavailable: {module_path}:{attr_name}")
    return getattr(module, attr_name)


class MockVertical(VerticalBase):
    """Mock vertical for testing registry wiring."""

    name = "mock_vertical"

    @classmethod
    def get_tools(cls) -> List[str]:
        return ["read", "write"]

    @classmethod
    def get_system_prompt(cls) -> str:
        return "Mock vertical prompt"

    @classmethod
    def get_handlers(cls) -> Dict[str, Any]:
        return {
            "mock_handler_1": MagicMock(),
            "mock_handler_2": MagicMock(),
        }


class MockWorkflowProvider:
    """Mock workflow provider for testing."""

    def get_workflows(self) -> Dict[str, Any]:
        return {"mock_workflow": MagicMock()}

    def get_auto_workflows(self) -> List[tuple]:
        return [
            (r"mock.*pattern", "mock_workflow"),
            (r"test.*trigger", "test_workflow"),
        ]


class MockTeamSpecProvider:
    """Mock team spec provider for testing."""

    def get_team_specs(self) -> Dict[str, Any]:
        return {
            "team_1": MagicMock(),
            "team_2": MagicMock(),
        }


@pytest.fixture
def reset_handler_registry():
    """Reset handler registry before and after test."""
    from victor.framework.handler_registry import HandlerRegistry

    HandlerRegistry.reset_instance()
    yield
    HandlerRegistry.reset_instance()


@pytest.fixture
def reset_trigger_registry():
    """Reset trigger registry before and after test."""
    try:
        from victor.workflows.trigger_registry import WorkflowTriggerRegistry

        # Reset if exists
        if hasattr(WorkflowTriggerRegistry, "_instance"):
            WorkflowTriggerRegistry._instance = None
    except ImportError:
        pass
    yield
    try:
        from victor.workflows.trigger_registry import WorkflowTriggerRegistry

        if hasattr(WorkflowTriggerRegistry, "_instance"):
            WorkflowTriggerRegistry._instance = None
    except ImportError:
        pass


@pytest.fixture
def reset_team_registry():
    """Reset team registry before and after test."""
    try:
        from victor.framework.team_registry import TeamSpecRegistry

        if hasattr(TeamSpecRegistry, "_instance"):
            TeamSpecRegistry._instance = None
    except ImportError:
        pass
    yield
    try:
        from victor.framework.team_registry import TeamSpecRegistry

        if hasattr(TeamSpecRegistry, "_instance"):
            TeamSpecRegistry._instance = None
    except ImportError:
        pass


class TestHandlerRegistryWiring:
    """Tests for handler registry wiring."""

    def test_handler_registry_import(self, reset_handler_registry):
        """Test handler registry can be imported."""
        from victor.framework.handler_registry import get_handler_registry

        registry = get_handler_registry()
        assert registry is not None

    def test_handlers_registered_from_vertical(self, reset_handler_registry):
        """Test handlers are registered from vertical get_handlers()."""
        from victor.framework.handler_registry import get_handler_registry

        registry = get_handler_registry()
        handlers = MockVertical.get_handlers()

        # Simulate what apply_handlers does - register handlers by vertical
        registry.register_vertical("mock_vertical", handlers, category="test")

        # Verify registration
        retrieved_handlers = registry.get_vertical_handlers("mock_vertical")
        assert "mock_handler_1" in retrieved_handlers
        assert "mock_handler_2" in retrieved_handlers

    def test_handlers_listed_by_vertical(self, reset_handler_registry):
        """Test handlers can be listed by vertical."""
        from victor.framework.handler_registry import get_handler_registry

        registry = get_handler_registry()

        # Register handlers for multiple verticals
        registry.register_vertical(
            "coding",
            {
                "coding_h1": MagicMock(),
                "coding_h2": MagicMock(),
            },
        )
        registry.register_vertical(
            "devops",
            {
                "devops_h1": MagicMock(),
            },
        )

        coding_handlers = registry.get_vertical_handlers("coding")
        assert len(coding_handlers) == 2
        assert "coding_h1" in coding_handlers
        assert "coding_h2" in coding_handlers
        assert "devops_h1" not in coding_handlers


class TestWorkflowTriggerRegistryWiring:
    """Tests for workflow trigger registry wiring."""

    def test_trigger_registry_exists(self, reset_trigger_registry):
        """Test trigger registry module exists."""
        try:
            from victor.workflows.trigger_registry import get_trigger_registry

            registry = get_trigger_registry()
            assert registry is not None
        except ImportError:
            pytest.skip("Trigger registry not available")

    def test_triggers_registered_from_provider(self, reset_trigger_registry):
        """Test workflow triggers are registered from provider."""
        try:
            from victor.workflows.trigger_registry import get_trigger_registry
        except ImportError:
            pytest.skip("Trigger registry not available")

        registry = get_trigger_registry()
        provider = MockWorkflowProvider()

        auto_workflows = provider.get_auto_workflows()
        registry.register_from_vertical("mock_vertical", auto_workflows)

        # Verify triggers were registered
        triggers = registry.get_triggers_for_vertical("mock_vertical")
        assert len(triggers) >= 2


class TestTeamSpecRegistryWiring:
    """Tests for team spec registry wiring."""

    def test_team_registry_exists(self, reset_team_registry):
        """Test team registry module exists."""
        try:
            from victor.framework.team_registry import get_team_registry

            registry = get_team_registry()
            assert registry is not None
        except ImportError:
            pytest.skip("Team registry not available")

    def test_teams_registered_from_vertical(self, reset_team_registry):
        """Test team specs are registered from vertical."""
        try:
            from victor.framework.team_registry import get_team_registry
        except ImportError:
            pytest.skip("Team registry not available")

        registry = get_team_registry()
        provider = MockTeamSpecProvider()

        team_specs = provider.get_team_specs()
        registry.register_from_vertical("mock_vertical", team_specs, replace=True)

        # Verify teams were registered
        teams = registry.find_by_vertical("mock_vertical")
        assert len(teams) >= 2


class TestVerticalBaseHandlers:
    """Tests for VerticalBase get_handlers() method."""

    def test_vertical_base_has_get_handlers(self):
        """Test VerticalBase has get_handlers method."""
        assert hasattr(VerticalBase, "get_handlers")

    def test_vertical_base_get_handlers_returns_dict(self):
        """Test VerticalBase.get_handlers returns empty dict by default."""
        result = VerticalBase.get_handlers()
        assert isinstance(result, dict)
        assert len(result) == 0

    def test_mock_vertical_get_handlers_returns_handlers(self):
        """Test MockVertical.get_handlers returns handlers."""
        result = MockVertical.get_handlers()
        assert isinstance(result, dict)
        assert len(result) == 2
        assert "mock_handler_1" in result
        assert "mock_handler_2" in result


class TestVerticalBaseToolGraph:
    """Tests for VerticalBase get_tool_graph() method."""

    def test_vertical_base_has_get_tool_graph(self):
        """Test VerticalBase has get_tool_graph method."""
        assert hasattr(VerticalBase, "get_tool_graph")

    def test_vertical_base_get_tool_graph_returns_none(self):
        """Test VerticalBase.get_tool_graph returns None by default."""
        result = VerticalBase.get_tool_graph()
        assert result is None


class TestCodingVerticalHandlers:
    """Tests for CodingAssistant handlers integration."""

    def test_coding_vertical_has_get_handlers(self):
        """Test CodingAssistant has get_handlers method."""
        CodingAssistant = _load_vertical_attr("victor.coding", "CodingAssistant")

        if not hasattr(CodingAssistant, "get_handlers"):
            pytest.skip("External victor-coding CodingAssistant lacks get_handlers")
        assert hasattr(CodingAssistant, "get_handlers")

    def test_coding_vertical_get_handlers_returns_handlers(self):
        """Test CodingAssistant.get_handlers returns handlers."""
        CodingAssistant = _load_vertical_attr("victor.coding", "CodingAssistant")

        if not hasattr(CodingAssistant, "get_handlers"):
            pytest.skip("External victor-coding CodingAssistant lacks get_handlers")
        result = CodingAssistant.get_handlers()
        assert isinstance(result, dict)
        # External victor-coding returns SDK default (empty dict);
        # internal contrib coding may have handlers
        # Just verify the method works and returns a dict
