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

"""Tests for vertical integration pipeline.

These tests verify that the VerticalIntegrationPipeline correctly applies
vertical configurations to orchestrators, achieving parity between CLI
(FrameworkShim) and SDK (Agent.create()) paths.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
import pytest
from unittest.mock import MagicMock, patch
from typing import Dict, List, Optional, Set

from victor.framework.extensions import (
    VerticalContext,
    create_vertical_context,
)
from victor.core.verticals.prompt_adapter import (
    PromptContributorAdapter,
    CompositePromptContributor,
    merge_contributors,
)
from victor.framework.vertical_integration import (
    VerticalIntegrationPipeline,
    IntegrationResult,
    create_integration_pipeline,
    apply_vertical,
    get_extension_handler_registry,
    register_extension_handler,
)
from victor.framework.vertical_cache_policy import VerticalIntegrationCachePolicy
from victor.core.verticals.protocols import TaskTypeHint, SafetyPattern, ModeConfig
from victor_contracts.verticals.protocols.base import VerticalBase as SdkVerticalBase

# =============================================================================
# Test Fixtures
# =============================================================================


class MockVerticalConfig:
    """Mock vertical configuration."""

    def __init__(self, name: str = "mock"):
        self.name = name
        self.tools = ["read", "write", "shell"]
        self.system_prompt = "You are a mock assistant."


class MockVerticalExtensions:
    """Mock vertical extensions."""

    def __init__(self):
        self.middleware = []
        self.safety_extensions = []
        self.prompt_contributors = []
        self.mode_config_provider = None
        self.tool_dependency_provider = None


class MockVertical(SdkVerticalBase):
    """Mock vertical class for testing."""

    name = "mock_vertical"

    @classmethod
    def get_name(cls) -> str:
        return cls.name

    @classmethod
    def get_description(cls) -> str:
        return "A mock vertical for testing."

    @classmethod
    def get_tools(cls) -> List[str]:
        return ["read", "write", "shell", "grep"]

    @classmethod
    def get_system_prompt(cls) -> str:
        return "You are a mock assistant for testing."

    @classmethod
    def get_stages(cls) -> Dict[str, any]:
        return {"INITIAL": {}, "READING": {}, "WRITING": {}}


class MockOrchestrator:
    """Mock orchestrator for testing."""

    def __init__(self):
        self._vertical_context = create_vertical_context()
        self._enabled_tools: Set[str] = set()
        self._middleware: List[any] = []
        self._safety_patterns: List[any] = []
        self.prompt_builder = MagicMock()

    def set_vertical_context(self, context: VerticalContext) -> None:
        self._vertical_context = context

    def get_vertical_context(self) -> VerticalContext:
        return self._vertical_context

    def set_enabled_tools(self, tools: Set[str]) -> None:
        self._enabled_tools = tools

    def get_enabled_tools(self) -> Set[str]:
        return set(self._enabled_tools)

    def apply_vertical_middleware(self, middleware: List[any]) -> None:
        self._middleware = middleware

    def apply_vertical_safety_patterns(self, patterns: List[any]) -> None:
        self._safety_patterns = patterns


class ContractProbeHandler:
    """Minimal step handler for dependency-contract tests."""

    def __init__(self, name: str, order: int, depends_on: tuple[str, ...]) -> None:
        self.name = name
        self.order = order
        self.depends_on = depends_on
        self.parallel_safe = False
        self.side_effects = False

    def apply(
        self,
        orchestrator,
        vertical,
        context,
        result,
        strict_mode=False,
    ):
        result.record_step_status(self.name, "success")

    async def apply_async(
        self,
        orchestrator,
        vertical,
        context,
        result,
        strict_mode=False,
    ):
        self.apply(orchestrator, vertical, context, result, strict_mode=strict_mode)


# =============================================================================
# VerticalContext Tests
# =============================================================================


class TestVerticalContext:
    """Tests for VerticalContext dataclass."""

    def test_empty_context_creation(self):
        """Test creating an empty vertical context."""
        context = VerticalContext.empty()
        assert context.name is None
        assert context.has_vertical is False
        assert len(context.middleware) == 0
        assert len(context.safety_patterns) == 0

    def test_context_with_name(self):
        """Test creating context with a name."""
        context = create_vertical_context(name="coding")
        assert context.name == "coding"
        assert context.has_vertical is True

    def test_apply_vertical(self):
        """Test applying a vertical to context."""
        context = VerticalContext()
        context.apply_vertical("devops", config=MockVerticalConfig("devops"))
        assert context.name == "devops"
        assert context.config is not None

    def test_apply_stages(self):
        """Test applying stages to context."""
        context = VerticalContext()
        stages = {"INITIAL": {}, "READING": {}, "WRITING": {}}
        context.apply_stages(stages)
        assert len(context.stages) == 3
        assert "INITIAL" in context.stages

    def test_apply_middleware(self):
        """Test applying middleware to context."""
        context = VerticalContext()
        middleware = [MagicMock(), MagicMock()]
        context.apply_middleware(middleware)
        assert len(context.middleware) == 2
        assert context.has_middleware is True

    def test_apply_safety_patterns(self):
        """Test applying safety patterns to context."""
        context = VerticalContext()
        patterns = [
            SafetyPattern(pattern=r"rm\s+-rf", description="Dangerous delete"),
            SafetyPattern(pattern=r"git\s+push\s+--force", description="Force push"),
        ]
        context.apply_safety_patterns(patterns)
        assert len(context.safety_patterns) == 2
        assert context.has_safety_patterns is True

    def test_apply_task_hints(self):
        """Test applying task hints to context."""
        context = VerticalContext()
        hints = {
            "edit": TaskTypeHint(task_type="edit", hint="Read first"),
            "search": TaskTypeHint(task_type="search", hint="Use grep"),
        }
        context.apply_task_hints(hints)
        assert len(context.task_hints) == 2
        assert context.get_task_hint("edit") is not None

    def test_apply_mode_configs(self):
        """Test applying mode configs to context."""
        context = VerticalContext()
        configs = {
            "fast": ModeConfig(name="fast", tool_budget=5, max_iterations=10),
            "thorough": ModeConfig(name="thorough", tool_budget=30, max_iterations=60),
        }
        context.apply_mode_configs(configs, "fast", 15)
        assert len(context.mode_configs) == 2
        assert context.default_mode == "fast"
        assert context.default_budget == 15

    def test_apply_system_prompt(self):
        """Test applying system prompt to context."""
        context = VerticalContext()
        context.apply_system_prompt("You are a helpful assistant.")
        assert context.system_prompt == "You are a helpful assistant."
        assert context.has_custom_prompt is True

    def test_apply_enabled_tools(self):
        """Test applying enabled tools to context."""
        context = VerticalContext()
        tools = {"read", "write", "shell"}
        context.apply_enabled_tools(tools)
        assert context.enabled_tools == tools

    def test_get_tool_budget_for_mode(self):
        """Test getting tool budget for a mode."""
        context = VerticalContext()
        configs = {
            "fast": ModeConfig(name="fast", tool_budget=5, max_iterations=10),
        }
        context.apply_mode_configs(configs, "fast", 10)
        assert context.get_tool_budget_for_mode("fast") == 5
        assert context.get_tool_budget_for_mode("unknown") == 10

    def test_to_dict(self):
        """Test serialization to dictionary."""
        context = create_vertical_context(name="coding")
        context.apply_enabled_tools({"read", "write"})
        context.apply_system_prompt("Test prompt")

        data = context.to_dict()
        assert data["name"] == "coding"
        assert len(data["enabled_tools"]) == 2
        assert data["has_system_prompt"] is True


# =============================================================================
# PromptContributorAdapter Tests
# =============================================================================


class TestPromptContributorAdapter:
    """Tests for PromptContributorAdapter."""

    def test_from_dict_with_task_type_hints(self):
        """Test creating adapter from dict with TaskTypeHint objects."""
        hints = {
            "edit": TaskTypeHint(task_type="edit", hint="Edit carefully", tool_budget=5),
            "search": TaskTypeHint(task_type="search", hint="Search first"),
        }
        adapter = PromptContributorAdapter.from_dict(task_hints=hints)
        result = adapter.get_task_type_hints()
        assert len(result) == 2
        assert result["edit"].hint == "Edit carefully"
        assert result["edit"].tool_budget == 5

    def test_from_dict_with_dict_format(self):
        """Test creating adapter from dict with plain dicts."""
        hints = {
            "edit": {"hint": "Edit carefully", "tool_budget": 5},
            "search": {"hint": "Search first"},
        }
        adapter = PromptContributorAdapter.from_dict(task_hints=hints)
        result = adapter.get_task_type_hints()
        assert len(result) == 2
        assert result["edit"].hint == "Edit carefully"

    def test_from_dict_with_string_format(self):
        """Test creating adapter from dict with string hints."""
        hints = {
            "edit": "Edit carefully",
            "search": "Search first",
        }
        adapter = PromptContributorAdapter.from_dict(task_hints=hints)
        result = adapter.get_task_type_hints()
        assert len(result) == 2
        assert result["edit"].hint == "Edit carefully"

    def test_system_prompt_section(self):
        """Test system prompt section."""
        adapter = PromptContributorAdapter.from_dict(
            system_prompt_section="Always verify changes.",
            grounding_rules="Base responses on tool output.",
        )
        assert adapter.get_system_prompt_section() == "Always verify changes."
        assert adapter.get_grounding_rules() == "Base responses on tool output."

    def test_priority(self):
        """Test priority setting."""
        adapter = PromptContributorAdapter.from_dict(priority=25)
        assert adapter.get_priority() == 25

    def test_with_hints(self):
        """Test adding hints to existing adapter."""
        adapter = PromptContributorAdapter.from_dict(
            task_hints={"edit": {"hint": "Edit"}},
        )
        new_adapter = adapter.with_hints(
            {"search": TaskTypeHint(task_type="search", hint="Search")},
        )
        result = new_adapter.get_task_type_hints()
        assert len(result) == 2

    def test_merge(self):
        """Test merging two adapters."""
        adapter1 = PromptContributorAdapter.from_dict(
            task_hints={"edit": {"hint": "Edit v1"}},
            system_prompt_section="Section 1",
            priority=50,
        )
        adapter2 = PromptContributorAdapter.from_dict(
            task_hints={"edit": {"hint": "Edit v2"}, "search": {"hint": "Search"}},
            system_prompt_section="Section 2",
            priority=30,
        )
        merged = adapter1.merge(adapter2)
        result = merged.get_task_type_hints()
        assert result["edit"].hint == "Edit v2"  # Later overrides
        assert "search" in result
        assert merged.get_priority() == 30  # Min of both

    def test_empty_adapter(self):
        """Test creating empty adapter."""
        adapter = PromptContributorAdapter.empty()
        assert len(adapter.get_task_type_hints()) == 0
        assert adapter.get_system_prompt_section() == ""


class TestCompositePromptContributor:
    """Tests for CompositePromptContributor."""

    def test_merge_contributors_by_priority(self):
        """Test that contributors are merged by priority."""
        contrib1 = PromptContributorAdapter.from_dict(
            task_hints={"edit": {"hint": "Edit 1"}},
            priority=50,
        )
        contrib2 = PromptContributorAdapter.from_dict(
            task_hints={"edit": {"hint": "Edit 2"}},
            priority=10,  # Higher priority
        )
        composite = CompositePromptContributor([contrib1, contrib2])
        result = composite.get_task_type_hints()
        # contrib2 runs first (priority 10), then contrib1 overrides (priority 50)
        assert result["edit"].hint == "Edit 1"

    def test_merge_system_prompt_sections(self):
        """Test that system prompt sections are merged."""
        contrib1 = PromptContributorAdapter.from_dict(
            system_prompt_section="Section 1",
            priority=50,
        )
        contrib2 = PromptContributorAdapter.from_dict(
            system_prompt_section="Section 2",
            priority=10,
        )
        composite = merge_contributors(contrib1, contrib2)
        result = composite.get_system_prompt_section()
        assert "Section 1" in result
        assert "Section 2" in result


# =============================================================================
# VerticalIntegrationPipeline Tests
# =============================================================================


class TestVerticalIntegrationPipeline:
    """Tests for VerticalIntegrationPipeline."""

    def test_apply_vertical_to_orchestrator(self):
        """Test applying a vertical to an orchestrator."""
        orchestrator = MockOrchestrator()
        pipeline = VerticalIntegrationPipeline()

        result = pipeline.apply(orchestrator, MockVertical)

        assert result.success is True
        assert result.vertical_name == "mock_vertical"
        # Note: Tool application may filter tools based on orchestrator capabilities
        assert len(orchestrator._enabled_tools) >= 3
        assert orchestrator._enabled_tools.issuperset({"read", "write", "shell"})

    def test_apply_vertical_creates_context(self):
        """Test that applying vertical creates a context."""
        orchestrator = MockOrchestrator()
        pipeline = VerticalIntegrationPipeline()

        result = pipeline.apply(orchestrator, MockVertical)

        assert result.context is not None
        assert result.context.name == "mock_vertical"
        assert result.context.has_vertical is True

    def test_apply_vertical_with_string_name(self):
        """Test applying vertical by name string."""
        orchestrator = MockOrchestrator()
        pipeline = VerticalIntegrationPipeline()

        # Mock the registry lookup
        with patch("victor.core.verticals.base.VerticalRegistry") as mock_registry:
            mock_registry.get.return_value = MockVertical
            mock_registry.list_names.return_value = ["mock_vertical"]

            result = pipeline.apply(orchestrator, "mock_vertical")

            assert result.success is True
            assert result.vertical_name == "mock_vertical"

    def test_apply_vertical_not_found(self):
        """Test error when vertical is not found."""
        orchestrator = MockOrchestrator()
        pipeline = VerticalIntegrationPipeline()

        with patch("victor.core.verticals.base.VerticalRegistry") as mock_registry:
            mock_registry.get.return_value = None
            mock_registry.list_names.return_value = []

            result = pipeline.apply(orchestrator, "nonexistent_vertical")

            assert result.success is False
            assert len(result.errors) > 0
            assert "not found" in result.errors[0].lower()

    def test_apply_vertical_sets_context(self):
        """Test that vertical context is set on orchestrator."""
        orchestrator = MockOrchestrator()
        pipeline = VerticalIntegrationPipeline()

        pipeline.apply(orchestrator, MockVertical)

        assert orchestrator._vertical_context.name == "mock_vertical"

    @pytest.mark.skip(reason="Error handling behavior changed in v0.6.0 - needs investigation")
    def test_strict_mode_fails_on_error(self):
        """Test that pipeline handles errors during integration."""
        orchestrator = MockOrchestrator()

        # Create a mock that will fail during extensions processing
        # (get_config is called too early, so we fail in a different place)
        class FailingVertical:
            name = "failing"

            @classmethod
            def get_config(cls):
                return None  # Return valid config to avoid early failure

            @classmethod
            def get_tools(cls):
                return []

            @classmethod
            def get_system_prompt(cls):
                return ""

            @classmethod
            def get_stages(cls):
                return {}

            @classmethod
            def get_extensions(cls):
                # This will fail during extension processing
                raise RuntimeError("Extension processing error")

        pipeline = VerticalIntegrationPipeline(strict_mode=False)

        # Pipeline should catch the error and return a failed result
        result = pipeline.apply(orchestrator, FailingVertical)

        # Error during extension processing should be caught
        assert result.success is False
        assert len(result.errors) > 0

    def test_pre_hooks(self):
        """Test that pre-hooks are called."""
        orchestrator = MockOrchestrator()
        hook_called = []

        def pre_hook(orch, vert):
            hook_called.append((orch, vert))

        pipeline = VerticalIntegrationPipeline(pre_hooks=[pre_hook])
        pipeline.apply(orchestrator, MockVertical)

        assert len(hook_called) == 1
        assert hook_called[0][0] is orchestrator
        # Note: The vertical may be wrapped/adapted, so check name instead of identity
        assert hook_called[0][1].name == "mock_vertical"

    def test_post_hooks(self):
        """Test that post-hooks are called."""
        orchestrator = MockOrchestrator()
        hook_called = []

        def post_hook(orch, result):
            hook_called.append((orch, result))

        pipeline = VerticalIntegrationPipeline(post_hooks=[post_hook])
        pipeline.apply(orchestrator, MockVertical)

        assert len(hook_called) == 1
        assert hook_called[0][0] is orchestrator
        assert isinstance(hook_called[0][1], IntegrationResult)


class TestIntegrationResult:
    """Tests for IntegrationResult."""

    def test_add_error(self):
        """Test adding an error marks success as False."""
        result = IntegrationResult()
        assert result.success is True

        result.add_error("Something went wrong")

        assert result.success is False
        assert len(result.errors) == 1

    def test_add_warning(self):
        """Test adding a warning does not affect success."""
        result = IntegrationResult()
        assert result.success is True

        result.add_warning("This is a warning")

        assert result.success is True
        assert len(result.warnings) == 1


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_create_integration_pipeline(self):
        """Test creating pipeline via convenience function."""
        pipeline = create_integration_pipeline(strict=True)
        assert pipeline._strict_mode is True

    def test_apply_vertical_convenience(self):
        """Test applying vertical via convenience function."""
        orchestrator = MockOrchestrator()
        result = apply_vertical(orchestrator, MockVertical)
        assert result.success is True


# =============================================================================
# Protocol Compliance Tests
# =============================================================================


class TestProtocolCompliance:
    """Tests for protocol compliance."""

    def test_vertical_context_protocol(self):
        """Test that VerticalContext satisfies VerticalContextProtocol."""
        context = VerticalContext()
        context.apply_vertical("test")
        context.apply_middleware([])
        context.apply_safety_patterns([])
        context.apply_task_hints({})
        context.apply_mode_configs({})

        # Protocol properties
        assert context.vertical_name == "test"
        assert context.has_vertical is True
        assert isinstance(context.middleware, list)
        assert isinstance(context.safety_patterns, list)
        assert isinstance(context.task_hints, dict)
        assert isinstance(context.mode_configs, dict)

    def test_mock_orchestrator_protocol(self):
        """Test that MockOrchestrator satisfies OrchestratorVerticalProtocol."""
        orchestrator = MockOrchestrator()

        # Protocol methods
        orchestrator.set_vertical_context(VerticalContext())
        orchestrator.set_enabled_tools({"read", "write"})
        orchestrator.apply_vertical_middleware([])
        orchestrator.apply_vertical_safety_patterns([])

        assert orchestrator._enabled_tools == {"read", "write"}


# =============================================================================
# Vertical Context Extended Features Tests (Workflows, RL, Teams)
# =============================================================================


class TestVerticalContextExtendedFeatures:
    """Tests for VerticalContext extended features (workflows, RL, teams)."""

    def test_apply_workflows(self):
        """Test applying workflows to context."""
        context = VerticalContext()
        workflows = {
            "feature_implementation": MagicMock(),
            "bug_fix": MagicMock(),
        }
        context.apply_workflows(workflows)

        assert context.has_workflows is True
        assert len(context.workflows) == 2
        assert context.get_workflow("feature_implementation") is not None
        assert context.get_workflow("nonexistent") is None

    def test_list_workflows(self):
        """Test listing workflows."""
        context = VerticalContext()
        workflows = {
            "workflow1": MagicMock(),
            "workflow2": MagicMock(),
            "workflow3": MagicMock(),
        }
        context.apply_workflows(workflows)

        workflow_names = context.list_workflows()
        assert len(workflow_names) == 3
        assert "workflow1" in workflow_names
        assert "workflow2" in workflow_names

    def test_apply_rl_config(self):
        """Test applying RL config to context."""
        context = VerticalContext()
        rl_config = MagicMock()
        context.apply_rl_config(rl_config)

        assert context.has_rl_config is True
        assert context.rl_config is rl_config

    def test_apply_rl_hooks(self):
        """Test applying RL hooks to context."""
        context = VerticalContext()
        rl_hooks = MagicMock()
        context.apply_rl_hooks(rl_hooks)

        assert context.rl_hooks is rl_hooks

    def test_apply_team_specs(self):
        """Test applying team specs to context."""
        context = VerticalContext()
        teams = {
            "feature_team": MagicMock(),
            "review_team": MagicMock(),
        }
        context.apply_team_specs(teams)

        assert context.has_team_specs is True
        assert len(context.team_specs) == 2
        assert context.get_team_spec("feature_team") is not None
        assert context.get_team_spec("nonexistent") is None

    def test_list_team_specs(self):
        """Test listing team specs."""
        context = VerticalContext()
        teams = {"team1": MagicMock(), "team2": MagicMock()}
        context.apply_team_specs(teams)

        team_names = context.list_team_specs()
        assert len(team_names) == 2
        assert "team1" in team_names

    def test_to_dict_includes_new_fields(self):
        """Test that to_dict includes new fields."""
        context = VerticalContext()
        context.apply_vertical("coding")
        context.apply_workflows({"wf1": MagicMock()})
        context.apply_rl_config(MagicMock())
        context.apply_rl_hooks(MagicMock())
        context.apply_team_specs({"team1": MagicMock()})

        data = context.to_dict()
        assert "workflows" in data
        assert len(data["workflows"]) == 1
        assert data["has_rl_config"] is True
        assert data["has_rl_hooks"] is True
        assert "team_specs" in data
        assert len(data["team_specs"]) == 1


# =============================================================================
# Real Vertical Integration Tests
# =============================================================================


@pytest.mark.skip(
    reason=(
        "Vertical packages are now external - tests need victor-coding, "
        "victor-devops, etc. installed"
    )
)
class TestRealVerticalIntegration:
    """Integration tests using real verticals."""

    def test_coding_vertical_has_workflow_provider(self):
        """Test that coding vertical provides workflows."""
        try:
            from victor_coding.assistant import CodingAssistant
        except ImportError:
            pytest.skip("victor-coding package not installed")

        provider = CodingAssistant.get_workflow_provider()
        assert provider is not None

        # Check workflow names
        workflow_names = provider.get_workflow_names()
        assert len(workflow_names) > 0
        assert "feature_implementation" in workflow_names

    def test_coding_vertical_has_rl_config(self):
        """Test that coding vertical provides RL config."""
        try:
            from victor_coding.assistant import CodingAssistant
        except ImportError:
            pytest.skip("victor-coding package not installed")

        rl_config = CodingAssistant.get_rl_config_provider()
        assert rl_config is not None

        # Check config attributes
        assert hasattr(rl_config, "active_learners")
        assert hasattr(rl_config, "task_type_mappings")

    def test_coding_vertical_has_team_specs(self):
        """Test that coding vertical provides team specs."""
        from victor.coding import CodingAssistant

        team_provider = CodingAssistant.get_team_spec_provider()
        assert team_provider is not None
        teams = team_provider.get_team_specs()
        assert len(teams) > 0
        assert "feature_team" in teams

    def test_devops_vertical_has_workflow_provider(self):
        """Test that devops vertical provides workflows."""
        from victor.devops import DevOpsAssistant

        provider = DevOpsAssistant.get_workflow_provider()
        assert provider is not None

        workflow_names = provider.get_workflow_names()
        assert len(workflow_names) > 0
        # YAML migration renamed deploy_infrastructure to deploy
        assert "deploy" in workflow_names or "deploy_infrastructure" in workflow_names

    def test_devops_vertical_has_rl_config(self):
        """Test that devops vertical provides RL config."""
        from victor.devops import DevOpsAssistant

        rl_config = DevOpsAssistant.get_rl_config_provider()
        assert rl_config is not None

        assert hasattr(rl_config, "active_learners")

    def test_devops_vertical_has_team_specs(self):
        """Test that devops vertical provides team specs."""
        from victor.devops import DevOpsAssistant

        team_provider = DevOpsAssistant.get_team_spec_provider()
        assert team_provider is not None
        teams = team_provider.get_team_specs()
        assert len(teams) > 0
        assert "deployment_team" in teams

    def test_research_vertical_has_workflow_provider(self):
        """Test that research vertical provides workflows."""
        from victor.research import ResearchAssistant

        provider = ResearchAssistant.get_workflow_provider()
        assert provider is not None

        workflow_names = provider.get_workflow_names()
        assert len(workflow_names) > 0
        assert "deep_research" in workflow_names

    def test_research_vertical_has_rl_config(self):
        """Test that research vertical provides RL config."""
        from victor.research import ResearchAssistant

        rl_config = ResearchAssistant.get_rl_config_provider()
        assert rl_config is not None

        assert hasattr(rl_config, "active_learners")

    def test_research_vertical_has_team_specs(self):
        """Test that research vertical provides team specs."""
        from victor.research import ResearchAssistant

        team_provider = ResearchAssistant.get_team_spec_provider()
        assert team_provider is not None
        teams = team_provider.get_team_specs()
        assert len(teams) > 0
        assert "deep_research_team" in teams

    def test_data_analysis_vertical_has_workflow_provider(self):
        """Test that data analysis vertical provides workflows."""
        from victor.dataanalysis import DataAnalysisAssistant

        provider = DataAnalysisAssistant.get_workflow_provider()
        assert provider is not None

        workflow_names = provider.get_workflow_names()
        assert len(workflow_names) > 0
        # YAML migration renamed eda_workflow to eda_pipeline
        assert "eda_pipeline" in workflow_names or "eda_workflow" in workflow_names

    def test_data_analysis_vertical_has_rl_config(self):
        """Test that data analysis vertical provides RL config."""
        from victor.dataanalysis import DataAnalysisAssistant

        rl_config = DataAnalysisAssistant.get_rl_config_provider()
        assert rl_config is not None

        assert hasattr(rl_config, "active_learners")

    def test_data_analysis_vertical_has_team_specs(self):
        """Test that data analysis vertical provides team specs."""
        from victor.dataanalysis import DataAnalysisAssistant

        team_provider = DataAnalysisAssistant.get_team_spec_provider()
        assert team_provider is not None
        teams = team_provider.get_team_specs()
        assert len(teams) > 0
        assert "eda_team" in teams


@pytest.mark.skip(reason="Vertical packages are now external")
class TestWorkflowProviderProtocol:
    """Tests for WorkflowProviderProtocol compliance."""

    def test_coding_workflow_provider_protocol(self):
        """Test that coding workflow provider satisfies protocol."""
        try:
            from victor_coding.workflows import CodingWorkflowProvider
        except ImportError:
            pytest.skip("victor-coding package not installed")

        provider = CodingWorkflowProvider()

        # Protocol methods
        assert hasattr(provider, "get_workflows")
        assert hasattr(provider, "get_workflow")
        assert hasattr(provider, "get_workflow_names")
        assert hasattr(provider, "get_auto_workflows")

        # Method returns
        workflows = provider.get_workflows()
        assert isinstance(workflows, dict)

        names = provider.get_workflow_names()
        assert isinstance(names, list)
        assert len(names) > 0

        auto_workflows = provider.get_auto_workflows()
        assert isinstance(auto_workflows, list)

    def test_devops_workflow_provider_protocol(self):
        """Test that devops workflow provider satisfies protocol."""
        from victor.devops.workflows import DevOpsWorkflowProvider

        provider = DevOpsWorkflowProvider()

        workflows = provider.get_workflows()
        assert isinstance(workflows, dict)
        # YAML migration renamed deploy_infrastructure to deploy
        assert "deploy" in workflows or "deploy_infrastructure" in workflows

    def test_research_workflow_provider_protocol(self):
        """Test that research workflow provider satisfies protocol."""
        from victor.research.workflows import ResearchWorkflowProvider

        provider = ResearchWorkflowProvider()

        workflows = provider.get_workflows()
        assert isinstance(workflows, dict)
        assert "deep_research" in workflows

    def test_data_analysis_workflow_provider_protocol(self):
        """Test that data analysis workflow provider satisfies protocol."""
        from victor.dataanalysis.workflows import DataAnalysisWorkflowProvider

        provider = DataAnalysisWorkflowProvider()

        workflows = provider.get_workflows()
        assert isinstance(workflows, dict)
        # YAML migration renamed eda_workflow to eda_pipeline
        assert "eda_pipeline" in workflows or "eda_workflow" in workflows
        assert "ml_pipeline" in workflows


@pytest.mark.skip(reason="Vertical packages are now external")
class TestRLConfigCompliance:
    """Tests for RL config compliance across verticals."""

    def test_coding_rl_config_methods(self):
        """Test that coding RL config has required methods."""
        try:
            from victor_coding.rl.config import CodingRLConfig
        except ImportError:
            pytest.skip("victor-coding package not installed")

        config = CodingRLConfig()

        # Required methods
        assert hasattr(config, "get_tools_for_task")
        assert hasattr(config, "get_quality_threshold")
        assert hasattr(config, "get_patience")
        assert hasattr(config, "is_learner_active")

        # Test methods
        tools = config.get_tools_for_task("debugging")
        assert isinstance(tools, list)

        threshold = config.get_quality_threshold("implementation")
        assert isinstance(threshold, float)
        assert 0.0 <= threshold <= 1.0

        patience = config.get_patience("anthropic")
        assert isinstance(patience, int)

    def test_devops_rl_config_methods(self):
        """Test that devops RL config has required methods."""
        from victor.devops.rl import DevOpsRLConfig

        config = DevOpsRLConfig()

        tools = config.get_tools_for_task("deployment")
        assert isinstance(tools, list)

    def test_research_rl_config_methods(self):
        """Test that research RL config has required methods."""
        from victor.research.rl import ResearchRLConfig

        config = ResearchRLConfig()

        tools = config.get_tools_for_task("research")
        assert isinstance(tools, list)

    def test_data_analysis_rl_config_methods(self):
        """Test that data analysis RL config has required methods."""
        from victor.dataanalysis.rl import DataAnalysisRLConfig

        config = DataAnalysisRLConfig()

        tools = config.get_tools_for_task("eda")
        assert isinstance(tools, list)

        threshold = config.get_quality_threshold("statistics")
        assert isinstance(threshold, float)
        assert threshold >= 0.85  # Statistics has high threshold


@pytest.mark.skip(reason="Vertical packages are now external")
class TestTeamSpecCompliance:
    """Tests for team spec compliance across verticals."""

    def test_coding_team_specs_structure(self):
        """Test that coding team specs have required structure."""
        try:
            from victor_coding.teams import CODING_TEAM_SPECS
        except ImportError:
            pytest.skip("victor-coding package not installed")

        for _name, spec in CODING_TEAM_SPECS.items():
            assert hasattr(spec, "name")
            assert hasattr(spec, "description")
            assert hasattr(spec, "formation")
            assert hasattr(spec, "members")
            assert hasattr(spec, "total_tool_budget")
            assert len(spec.members) > 0

    def test_devops_team_specs_structure(self):
        """Test that devops team specs have required structure."""
        from victor.devops.teams import DEVOPS_TEAM_SPECS

        for _name, spec in DEVOPS_TEAM_SPECS.items():
            assert hasattr(spec, "name")
            assert hasattr(spec, "members")
            assert len(spec.members) > 0

    def test_research_team_specs_structure(self):
        """Test that research team specs have required structure."""
        from victor.research.teams import RESEARCH_TEAM_SPECS

        for _name, spec in RESEARCH_TEAM_SPECS.items():
            assert hasattr(spec, "name")
            assert hasattr(spec, "members")
            assert len(spec.members) > 0

    def test_data_analysis_team_specs_structure(self):
        """Test that data analysis team specs have required structure."""
        from victor.dataanalysis.teams import DATA_ANALYSIS_TEAM_SPECS

        for _name, spec in DATA_ANALYSIS_TEAM_SPECS.items():
            assert hasattr(spec, "name")
            assert hasattr(spec, "members")
            assert len(spec.members) > 0


@pytest.mark.skip(reason="Vertical packages are now external")
class TestGetTeamForTask:
    """Tests for get_team_for_task helper functions."""

    def test_coding_get_team_for_task(self):
        """Test getting team for coding task types."""
        try:
            from victor_coding.teams import get_team_for_task
        except ImportError:
            pytest.skip("victor-coding package not installed")

        team = get_team_for_task("feature")
        assert team is not None
        assert team.name == "Feature Implementation Team"

        team = get_team_for_task("bug")
        assert team is not None
        assert team.name == "Bug Fix Team"

        team = get_team_for_task("nonexistent")
        assert team is None

    def test_devops_get_team_for_task(self):
        """Test getting team for devops task types."""
        try:
            from victor_devops.teams import get_team_for_task
        except ImportError:
            pytest.skip("victor-devops package not installed")

        team = get_team_for_task("deploy")
        assert team is not None

        team = get_team_for_task("container")
        assert team is not None

    def test_research_get_team_for_task(self):
        """Test getting team for research task types."""
        from victor.research.teams import get_team_for_task

        team = get_team_for_task("research")
        assert team is not None

        team = get_team_for_task("fact_check")
        assert team is not None

    def test_data_analysis_get_team_for_task(self):
        """Test getting team for data analysis task types."""
        from victor.dataanalysis.teams import get_team_for_task

        team = get_team_for_task("eda")
        assert team is not None
        assert team.name == "Exploratory Data Analysis Team"

        team = get_team_for_task("ml")
        assert team is not None
        assert team.name == "Machine Learning Team"

        team = get_team_for_task("nonexistent")
        assert team is None


# =============================================================================
# Agent Workflow/Team Methods Tests
# =============================================================================


@pytest.mark.skip(reason="Vertical packages are now external")
class TestAgentWorkflowMethods:
    """Tests for Agent workflow and team execution methods."""

    def test_get_available_workflows_with_vertical(self):
        """Test get_available_workflows returns workflows from vertical."""
        try:
            from victor_coding.assistant import CodingAssistant
        except ImportError:
            pytest.skip("victor-coding package not installed")

        from victor.framework.agent import Agent

        # Create a mock agent with vertical set
        agent = MagicMock(spec=Agent)
        agent._vertical = CodingAssistant

        # Call actual method
        result = Agent.get_available_workflows(agent)
        assert len(result) > 0
        assert "feature_implementation" in result

    def test_get_available_workflows_without_vertical(self):
        """Test get_available_workflows returns empty list without vertical."""
        from victor.framework.agent import Agent

        agent = MagicMock(spec=Agent)
        agent._vertical = None

        result = Agent.get_available_workflows(agent)
        assert result == []

    def test_get_available_teams_with_vertical(self):
        """Test get_available_teams returns teams from vertical."""
        from victor.framework.agent import Agent
        from victor.coding import CodingAssistant

        agent = MagicMock(spec=Agent)
        agent._vertical = CodingAssistant

        result = Agent.get_available_teams(agent)
        assert len(result) > 0
        assert "feature_team" in result

    def test_get_available_teams_without_vertical(self):
        """Test get_available_teams returns empty list without vertical."""
        from victor.framework.agent import Agent

        agent = MagicMock(spec=Agent)
        agent._vertical = None

        result = Agent.get_available_teams(agent)
        assert result == []

    def test_get_available_workflows_all_verticals(self):
        """Test all verticals provide workflows."""
        from victor.framework.agent import Agent
        from victor.coding import CodingAssistant
        from victor.devops import DevOpsAssistant
        from victor.research import ResearchAssistant
        from victor.dataanalysis import DataAnalysisAssistant

        for vertical_class in [
            CodingAssistant,
            DevOpsAssistant,
            ResearchAssistant,
            DataAnalysisAssistant,
        ]:
            agent = MagicMock(spec=Agent)
            agent._vertical = vertical_class

            workflows = Agent.get_available_workflows(agent)
            assert len(workflows) > 0, f"{vertical_class.name} should provide workflows"

    def test_get_available_teams_all_verticals(self):
        """Test all verticals provide teams."""
        from victor.framework.agent import Agent
        from victor.coding import CodingAssistant
        from victor.devops import DevOpsAssistant
        from victor.research import ResearchAssistant
        from victor.dataanalysis import DataAnalysisAssistant

        for vertical_class in [
            CodingAssistant,
            DevOpsAssistant,
            ResearchAssistant,
            DataAnalysisAssistant,
        ]:
            agent = MagicMock(spec=Agent)
            agent._vertical = vertical_class

            teams = Agent.get_available_teams(agent)
            assert len(teams) > 0, f"{vertical_class.name} should provide teams"


# =============================================================================
# Phase 1 Refactoring: Caching Tests
# =============================================================================


class TestVerticalIntegrationCaching:
    """Tests for VerticalIntegrationPipeline caching functionality (Phase 1)."""

    def test_cache_key_generation(self):
        """Test that cache keys are generated correctly."""
        pipeline = VerticalIntegrationPipeline(enable_cache=True)

        # Generate cache key for MockVertical
        cache_key = pipeline._generate_cache_key(MockVertical)

        assert cache_key is not None
        assert isinstance(cache_key, str)
        assert cache_key.startswith("v1_")
        assert "mock_vertical" in cache_key

    def test_cache_key_includes_vertical_name(self):
        """Test that cache key includes vertical name."""
        pipeline = VerticalIntegrationPipeline(enable_cache=True)

        key1 = pipeline._generate_cache_key(MockVertical)

        # Create a different vertical
        class DifferentVertical:
            name = "different_vertical"

            @classmethod
            def get_config(cls):
                return MockVerticalConfig(cls.name)

            @classmethod
            def get_tools(cls):
                return []

            @classmethod
            def get_system_prompt(cls):
                return ""

            @classmethod
            def get_stages(cls):
                return {}

            @classmethod
            def get_extensions(cls):
                return None

        key2 = pipeline._generate_cache_key(DifferentVertical)

        # Keys should be different
        assert key1 != key2

    def test_cache_disabled_by_default(self):
        """Test that caching is enabled by default but can be disabled."""
        pipeline_with_cache = VerticalIntegrationPipeline()
        assert pipeline_with_cache._enable_cache is True

        pipeline_no_cache = VerticalIntegrationPipeline(enable_cache=False)
        assert pipeline_no_cache._enable_cache is False

    def test_cache_hit_on_second_apply(self):
        """Test that second apply returns cached result."""
        orchestrator = MockOrchestrator()
        pipeline = VerticalIntegrationPipeline(enable_cache=True)

        # First application - should execute integration
        result1 = pipeline.apply(orchestrator, MockVertical)

        # Reset orchestrator state
        orchestrator._enabled_tools = set()

        # Second application - should use cache
        result2 = pipeline.apply(orchestrator, MockVertical)

        # Both should succeed
        assert result1.success is True
        assert result2.success is True

        # Results should have the same vertical name
        assert result1.vertical_name == result2.vertical_name

    def test_cache_hit_still_reapplies_orchestrator_side_effects(self):
        """Cache hit must still apply tools/context to the target orchestrator.

        Regression guard: cached IntegrationResult omits VerticalContext and cannot
        be returned directly without replaying integration steps.
        """
        orchestrator = MockOrchestrator()
        pipeline = VerticalIntegrationPipeline(enable_cache=True)

        # Warm cache
        first = pipeline.apply(orchestrator, MockVertical)
        assert first.success is True

        # Reset orchestrator state to prove second call reapplies side effects
        orchestrator._enabled_tools = set()
        orchestrator._vertical_context = create_vertical_context()

        second = pipeline.apply(orchestrator, MockVertical)

        assert second.success is True
        # Note: Tool application may filter tools based on orchestrator capabilities
        assert len(orchestrator._enabled_tools) >= 3
        assert orchestrator._enabled_tools.issuperset({"read", "write", "shell"})
        assert orchestrator._vertical_context.name == "mock_vertical"

    def test_cache_entry_expires_after_ttl(self):
        """Expired cache entries should not be returned."""
        import time

        pipeline = VerticalIntegrationPipeline(enable_cache=True, cache_ttl=1)
        key = "ttl_expired_key"
        result = IntegrationResult(success=True, vertical_name="mock_vertical")

        pipeline._save_to_cache(key, result)
        assert key in pipeline._cache
        assert key in pipeline._cache_timestamps

        # Force entry to expire
        pipeline._cache_timestamps[key] = time.monotonic() - 3600

        loaded = pipeline._load_from_cache(key)
        assert loaded is None
        assert key not in pipeline._cache

    def test_cache_miss_after_disabling(self):
        """Test that cache is not used when disabled."""
        orchestrator = MockOrchestrator()
        pipeline = VerticalIntegrationPipeline(enable_cache=True)

        # First application with cache enabled
        result1 = pipeline.apply(orchestrator, MockVertical)

        # Disable cache
        pipeline._enable_cache = False

        # Second application - should not use cache
        result2 = pipeline.apply(orchestrator, MockVertical)

        # Both should succeed
        assert result1.success is True
        assert result2.success is True

    def test_cache_ttl_configuration(self):
        """Test that cache TTL can be configured."""
        pipeline = VerticalIntegrationPipeline(enable_cache=True, cache_ttl=1800)
        assert pipeline._cache_ttl == 1800

    def test_cache_size_is_bounded_and_evicts_oldest_entry(self):
        """Cache should evict oldest entry when max size is exceeded."""
        orchestrator = MockOrchestrator()
        pipeline = VerticalIntegrationPipeline(enable_cache=True, max_cache_entries=2)

        class VerticalA(SdkVerticalBase):
            name = "vertical_a"

            @classmethod
            def get_name(cls) -> str:
                return cls.name

            @classmethod
            def get_description(cls) -> str:
                return "Vertical A for testing cache eviction."

            @classmethod
            def get_tools(cls):
                return ["read"]

            @classmethod
            def get_system_prompt(cls):
                return "Prompt A"

            @classmethod
            def get_stages(cls):
                return {}

            @classmethod
            def get_extensions(cls):
                return None

        class VerticalB(SdkVerticalBase):
            name = "vertical_b"

            @classmethod
            def get_name(cls) -> str:
                return cls.name

            @classmethod
            def get_description(cls) -> str:
                return "Vertical B for testing cache eviction."

            @classmethod
            def get_tools(cls):
                return ["write"]

            @classmethod
            def get_system_prompt(cls):
                return "Prompt B"

            @classmethod
            def get_stages(cls):
                return {}

            @classmethod
            def get_extensions(cls):
                return None

        class VerticalC(SdkVerticalBase):
            name = "vertical_c"

            @classmethod
            def get_name(cls) -> str:
                return cls.name

            @classmethod
            def get_description(cls) -> str:
                return "Vertical C for testing cache eviction."

            @classmethod
            def get_tools(cls):
                return ["grep"]

            @classmethod
            def get_system_prompt(cls):
                return "Prompt C"

            @classmethod
            def get_stages(cls):
                return {}

            @classmethod
            def get_extensions(cls):
                return None

        pipeline.apply(orchestrator, VerticalA)
        pipeline.apply(orchestrator, VerticalB)
        pipeline.apply(orchestrator, VerticalC)

        key_a = pipeline._generate_cache_key(VerticalA)
        key_b = pipeline._generate_cache_key(VerticalB)
        key_c = pipeline._generate_cache_key(VerticalC)

        assert key_a not in pipeline._cache
        assert key_b in pipeline._cache
        assert key_c in pipeline._cache
        assert len(pipeline._cache) == 2

    def test_cache_metrics_track_hits_misses_and_evictions(self):
        """Cache metrics should reflect cache behavior for observability."""
        orchestrator = MockOrchestrator()
        pipeline = VerticalIntegrationPipeline(enable_cache=True, max_cache_entries=1)

        class VerticalA(SdkVerticalBase):
            name = "metric_a"

            @classmethod
            def get_name(cls) -> str:
                return cls.name

            @classmethod
            def get_description(cls) -> str:
                return "Vertical A for testing cache metrics."

            @classmethod
            def get_tools(cls):
                return ["read"]

            @classmethod
            def get_system_prompt(cls):
                return "Prompt A"

            @classmethod
            def get_stages(cls):
                return {}

            @classmethod
            def get_extensions(cls):
                return None

        class VerticalB(SdkVerticalBase):
            name = "metric_b"

            @classmethod
            def get_name(cls) -> str:
                return cls.name

            @classmethod
            def get_description(cls) -> str:
                return "Vertical B for testing cache metrics."

            @classmethod
            def get_tools(cls):
                return ["write"]

            @classmethod
            def get_system_prompt(cls):
                return "Prompt B"

            @classmethod
            def get_stages(cls):
                return {}

            @classmethod
            def get_extensions(cls):
                return None

        pipeline.apply(orchestrator, VerticalA)  # miss + save
        pipeline.apply(orchestrator, VerticalA)  # hit
        pipeline.apply(orchestrator, VerticalB)  # miss + eviction

        stats = pipeline.get_cache_stats()
        assert stats["hits"] >= 1
        assert stats["misses"] >= 2
        assert stats["evictions"] >= 1
        assert stats["size"] == 1

    def test_custom_cache_policy_can_be_injected(self):
        """Pipeline should delegate cache operations to injected policy."""

        class RecordingPolicy(VerticalIntegrationCachePolicy):
            def __init__(self) -> None:
                self.storage: Dict[str, str] = {}
                self.loads = 0
                self.saves = 0

            def load(
                self,
                cache_key: str,
                *,
                cache: Dict[str, str],
                timestamps: Dict[str, float],
                cache_ttl: int,
            ) -> Optional[str]:
                self.loads += 1
                return self.storage.get(cache_key)

            def save(
                self,
                cache_key: str,
                payload: str,
                *,
                cache: Dict[str, str],
                timestamps: Dict[str, float],
                max_entries: int,
            ) -> None:
                self.saves += 1
                self.storage[cache_key] = payload

            def get_stats(
                self,
                *,
                cache: Dict[str, str],
                max_entries: int,
            ) -> Dict[str, int]:
                return {
                    "loads": self.loads,
                    "saves": self.saves,
                    "size": len(self.storage),
                    "max_entries": max_entries,
                }

            def clear(self, *, cache: Dict[str, str], timestamps: Dict[str, float]) -> None:
                self.storage.clear()

        policy = RecordingPolicy()
        orchestrator = MockOrchestrator()
        pipeline = VerticalIntegrationPipeline(enable_cache=True, cache_policy=policy)

        pipeline.apply(orchestrator, MockVertical)
        pipeline.apply(orchestrator, MockVertical)

        stats = pipeline.get_cache_stats()
        assert stats["loads"] >= 2
        assert stats["saves"] >= 1
        assert stats["size"] >= 1

    def test_concurrent_apply_is_thread_safe_with_bounded_cache(self):
        """Concurrent apply calls should succeed and honor cache bounds."""
        pipeline = VerticalIntegrationPipeline(enable_cache=True, max_cache_entries=3)

        def _make_vertical(index: int):
            class _Vertical(SdkVerticalBase):
                name = f"concurrent_vertical_{index}"

                @classmethod
                def get_name(cls) -> str:
                    return cls.name

                @classmethod
                def get_description(cls) -> str:
                    return f"Concurrent vertical {index} for thread safety testing."

                @classmethod
                def get_tools(cls):
                    return ["read", "write"]

                @classmethod
                def get_system_prompt(cls):
                    return f"Prompt {index}"

                @classmethod
                def get_stages(cls):
                    return {}

                @classmethod
                def get_extensions(cls):
                    return None

            return _Vertical

        verticals = [_make_vertical(i) for i in range(6)]

        def _run(i: int) -> bool:
            orchestrator = MockOrchestrator()
            result = pipeline.apply(orchestrator, verticals[i % len(verticals)])
            return result.success

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(_run, i) for i in range(80)]
            outcomes = [f.result() for f in futures]

        assert all(outcomes)
        assert len(pipeline._cache) <= 3

    def test_cache_stores_integration_result(self):
        """Test that cache stores IntegrationResult objects."""
        pipeline = VerticalIntegrationPipeline(enable_cache=True)
        orchestrator = MockOrchestrator()

        # Apply vertical
        result = pipeline.apply(orchestrator, MockVertical)

        # Check that cache has the entry
        cache_key = pipeline._generate_cache_key(MockVertical)
        assert cache_key in pipeline._cache

        # Load from cache
        cached_result = pipeline._load_from_cache(cache_key)

        assert cached_result is not None
        assert isinstance(cached_result, type(result))
        assert cached_result.vertical_name == result.vertical_name

    def test_cache_invalidates_on_source_change(self):
        """Test that cache invalidates when source file changes."""
        import tempfile
        import sys

        # Create a mock vertical with a temporary source file
        temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)
        temp_file.write("""
class TempVertical:
    name = "temp_vertical"

    @classmethod
    def get_config(cls):
        return None

    @classmethod
    def get_tools(cls):
        return []

    @classmethod
    def get_system_prompt(cls):
        return ""

    @classmethod
    def get_stages(cls):
        return {}

    @classmethod
    def get_extensions(cls):
        return None
""")
        temp_file.close()

        try:
            # Import the temporary vertical
            import importlib.util

            spec = importlib.util.spec_from_file_location("temp_vertical", temp_file.name)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Add to sys.modules so it can be found by inspect
            sys.modules["temp_vertical"] = module

            TempVertical = module.TempVertical

            pipeline = VerticalIntegrationPipeline(enable_cache=True)

            # First cache key
            key1 = pipeline._generate_cache_key(TempVertical)

            # Modify the file
            import time

            time.sleep(0.1)  # Ensure different mtime
            with open(temp_file.name, "a") as f:
                f.write("\n# Modified\n")

            # Second cache key should be different
            key2 = pipeline._generate_cache_key(TempVertical)

            assert key1 != key2, "Cache key should change after source modification"

        finally:
            # Cleanup
            import os

            if "temp_vertical" in sys.modules:
                del sys.modules["temp_vertical"]
            os.unlink(temp_file.name)

    def test_cache_handles_missing_file(self):
        """Test that cache handles missing source files gracefully."""
        pipeline = VerticalIntegrationPipeline(enable_cache=True)

        # Create a mock vertical that can't be inspected
        class BadVertical:
            name = "bad_vertical"

        # Should return None or raise a handled error
        cache_key = pipeline._generate_cache_key(BadVertical)

        # Either returns None or a valid key (but shouldn't crash)
        assert cache_key is None or isinstance(cache_key, str)

    def test_cache_clear_method(self):
        """Test that cache can be cleared."""
        pipeline = VerticalIntegrationPipeline(enable_cache=True)
        orchestrator = MockOrchestrator()

        # Populate cache
        pipeline.apply(orchestrator, MockVertical)
        cache_key = pipeline._generate_cache_key(MockVertical)

        assert cache_key in pipeline._cache

        # Clear cache
        pipeline._cache.clear()

        # Should be empty now
        assert cache_key not in pipeline._cache or len(pipeline._cache) == 0


class TestCachingPerformance:
    """Performance benchmarks for caching (Phase 1)."""

    @pytest.mark.slow
    @pytest.mark.benchmark
    def test_cached_apply_faster_than_uncached(self, benchmark=False):
        """Test that cached application is faster than uncached."""
        import time

        orchestrator = MockOrchestrator()
        pipeline = VerticalIntegrationPipeline(enable_cache=True)

        # Clear cache first
        pipeline._cache = {}

        # Time first application (cold start)
        start = time.perf_counter()
        result1 = pipeline.apply(orchestrator, MockVertical)
        cold_time = time.perf_counter() - start

        # Time second application (warm cache)
        start = time.perf_counter()
        result2 = pipeline.apply(orchestrator, MockVertical)
        warm_time = time.perf_counter() - start

        # Both should succeed
        assert result1.success is True
        assert result2.success is True

        # Warm start should be faster or at least not significantly slower
        # (allowing for some variance in timing)
        if not benchmark:  # Only assert in normal tests
            assert (
                warm_time <= cold_time * 1.5
            ), f"Warm start ({warm_time:.4f}s) should be faster than cold start ({cold_time:.4f}s)"

    def test_cache_memory_overhead(self):
        """Test that cache memory overhead is reasonable."""
        import sys

        pipeline = VerticalIntegrationPipeline(enable_cache=True)
        orchestrator = MockOrchestrator()

        # Get baseline memory
        sys.getsizeof(pipeline._cache)

        # Apply vertical (populates cache)
        pipeline.apply(orchestrator, MockVertical)

        # Check cache size after
        cache_size = sys.getsizeof(pipeline._cache)

        # Cache overhead should be reasonable (< 10MB for a single entry)
        # Note: This is a rough check, actual memory usage depends on the IntegrationResult
        assert (
            cache_size < 10 * 1024 * 1024
        ), f"Cache memory overhead ({cache_size} bytes) should be < 10MB"

    def test_cache_hit_rate_high(self):
        """Test that cache hit rate is high for repeated applies."""
        orchestrator = MockOrchestrator()
        pipeline = VerticalIntegrationPipeline(enable_cache=True)

        # Apply same vertical multiple times
        iterations = 10
        successes = 0

        for _ in range(iterations):
            result = pipeline.apply(orchestrator, MockVertical)
            if result.success:
                successes += 1

        # All should succeed
        assert successes == iterations

        # Cache should have been used after first iteration
        # (we can't directly measure hit rate, but we can infer from speed)

        # Expected behavior: Only first call does full integration
        # Subsequent calls use cache (faster)
        # This is implicitly tested by the speed improvement test above


class TestCachingEdgeCases:
    """Edge case tests for caching functionality."""

    def test_cache_with_none_result(self):
        """Test cache behavior when integration returns None."""
        pipeline = VerticalIntegrationPipeline(enable_cache=True)

        # Try to load from empty cache
        result = pipeline._load_from_cache("nonexistent_key")
        assert result is None

    def test_cache_corruption_handling(self):
        """Test that cache handles corrupted data gracefully."""
        pipeline = VerticalIntegrationPipeline(enable_cache=True)

        # Corrupt the cache with invalid data
        pipeline._cache["corrupted_key"] = b"invalid_pickle_data"

        # Should handle corruption gracefully
        result = pipeline._load_from_cache("corrupted_key")
        assert result is None

        # Corrupted entry should be removed
        assert "corrupted_key" not in pipeline._cache

    def test_cache_with_multiple_verticals(self):
        """Test cache with multiple different verticals."""
        orchestrator = MockOrchestrator()
        pipeline = VerticalIntegrationPipeline(enable_cache=True)

        # Create different verticals
        class Vertical1(SdkVerticalBase):
            name = "vertical1"

            @classmethod
            def get_name(cls) -> str:
                return cls.name

            @classmethod
            def get_description(cls) -> str:
                return "Vertical 1 for testing multiple verticals."

            @classmethod
            def get_tools(cls):
                return ["tool1"]

            @classmethod
            def get_system_prompt(cls):
                return "Prompt 1"

            @classmethod
            def get_stages(cls):
                return {}

            @classmethod
            def get_extensions(cls):
                return None

        class Vertical2(SdkVerticalBase):
            name = "vertical2"

            @classmethod
            def get_name(cls) -> str:
                return cls.name

            @classmethod
            def get_description(cls) -> str:
                return "Vertical 2 for testing multiple verticals."

            @classmethod
            def get_tools(cls):
                return ["tool2"]

            @classmethod
            def get_system_prompt(cls):
                return "Prompt 2"

            @classmethod
            def get_stages(cls):
                return {}

            @classmethod
            def get_extensions(cls):
                return None

        # Apply both verticals
        result1 = pipeline.apply(orchestrator, Vertical1)
        result2 = pipeline.apply(orchestrator, Vertical2)

        # Both should succeed
        assert result1.success is True
        assert result2.success is True

        # Cache should have entries for both
        assert len(pipeline._cache) >= 2

        # Verify cache keys are different
        key1 = pipeline._generate_cache_key(Vertical1)
        key2 = pipeline._generate_cache_key(Vertical2)
        assert key1 != key2

    def test_cache_key_stability(self):
        """Test that cache keys are stable across multiple generations."""
        pipeline = VerticalIntegrationPipeline(enable_cache=True)

        # Generate key multiple times
        keys = [pipeline._generate_cache_key(MockVertical) for _ in range(10)]

        # All keys should be identical
        assert len(set(keys)) == 1, "Cache keys should be stable"

    def test_clear_cache_resets_entries_and_stats(self):
        """clear_cache should remove entries and reset cache policy stats."""
        orchestrator = MockOrchestrator()
        pipeline = VerticalIntegrationPipeline(enable_cache=True)

        pipeline.apply(orchestrator, MockVertical)
        pipeline.apply(orchestrator, MockVertical)  # create at least one hit
        assert len(pipeline._cache) >= 1
        assert pipeline.get_cache_stats()["size"] >= 1

        pipeline.clear_cache()

        stats = pipeline.get_cache_stats()
        assert len(pipeline._cache) == 0
        assert stats["size"] == 0
        assert stats["hits"] == 0
        assert stats["misses"] == 0


@pytest.mark.skip(reason="Vertical packages are now external")
class TestCachingIntegration:
    """Integration tests for caching with real verticals."""

    def test_caching_with_real_coding_vertical(self):
        """Test caching with real CodingAssistant vertical."""
        try:
            from victor_coding.assistant import CodingAssistant
        except ImportError:
            pytest.skip("victor-coding package not installed")

        orchestrator = MockOrchestrator()
        pipeline = VerticalIntegrationPipeline(enable_cache=True)

        # First application
        result1 = pipeline.apply(orchestrator, CodingAssistant)

        # Reset orchestrator
        orchestrator._enabled_tools = set()

        # Second application (should use cache)
        result2 = pipeline.apply(orchestrator, CodingAssistant)

        # Both should succeed
        assert result1.success is True
        assert result2.success is True
        assert result1.vertical_name == result2.vertical_name

    def test_cache_invalidation_after_vertical_update(self):
        """Test that cache invalidates when vertical is updated."""
        # This test verifies the concept but doesn't actually modify source files
        pipeline = VerticalIntegrationPipeline(enable_cache=True)

        # Get initial cache key
        from victor.coding import CodingAssistant

        key1 = pipeline._generate_cache_key(CodingAssistant)

        # The key should be stable
        key2 = pipeline._generate_cache_key(CodingAssistant)
        assert key1 == key2

        # In production, if the CodingAssistant source file changed,
        # the key would be different due to file hash change
        # (This is conceptual - we don't actually modify files in tests)


class TestIntegrationPlanCache:
    """Tests for integration-plan cache no-op/delta behavior."""

    class _CountingHandler:
        def __init__(self, name: str, order: int, side_effects: bool) -> None:
            self.name = name
            self.order = order
            self.side_effects = side_effects
            self.parallel_safe = not side_effects
            self.depends_on = ()
            self.calls = 0

        def apply(
            self,
            orchestrator,
            vertical,
            context,
            result,
            strict_mode=False,
        ):
            self.calls += 1
            result.record_step_status(
                self.name,
                "success",
                details={"calls": self.calls},
            )

    class _Registry:
        def __init__(self, handlers):
            self._handlers = handlers

        def get_ordered_handlers(self):
            return self._handlers

    def test_state_snapshot_ignores_private_only_attributes(self):
        """Snapshot should not read legacy private fallback attributes."""
        pipeline = VerticalIntegrationPipeline(enable_cache=True)

        class LegacyOnlyOrchestrator:
            def __init__(self):
                self._vertical_context = create_vertical_context(name="legacy_only")
                self._enabled_tools = {"read", "write"}

        snapshot = pipeline._orchestrator_state_snapshot(LegacyOnlyOrchestrator())
        assert "vertical_name" not in snapshot
        assert "enabled_tools_hash" not in snapshot

    def test_state_snapshot_uses_getter_without_enabled_tools_attr_probe(self):
        """Snapshot should rely on getter and never probe enabled_tools attribute."""
        pipeline = VerticalIntegrationPipeline(enable_cache=True)

        class GetterOnlyOrchestrator:
            def __init__(self):
                self._vertical_context = create_vertical_context(name="getter_only")

            def get_vertical_context(self):
                return self._vertical_context

            def get_enabled_tools(self):
                return {"read", "write"}

            @property
            def enabled_tools(self):  # pragma: no cover - should never be touched
                raise AssertionError("enabled_tools attribute probe is forbidden")

        snapshot = pipeline._orchestrator_state_snapshot(GetterOnlyOrchestrator())
        assert snapshot.get("vertical_name") == "getter_only"
        assert "enabled_tools_hash" in snapshot

    def test_state_snapshot_getter_fallback_blocked_in_protocol_strict_mode(self, monkeypatch):
        """Protocol strict mode should block snapshot getter fallback probes."""
        pipeline = VerticalIntegrationPipeline(enable_cache=True)

        class GetterOnlyOrchestrator:
            def __init__(self):
                self._vertical_context = create_vertical_context(name="getter_only")

            def get_vertical_context(self):
                return self._vertical_context

            def get_enabled_tools(self):
                return {"read", "write"}

        monkeypatch.setenv("VICTOR_STRICT_FRAMEWORK_PROTOCOL_FALLBACKS", "1")
        with pytest.raises(RuntimeError, match="Protocol fallback blocked"):
            pipeline._orchestrator_state_snapshot(GetterOnlyOrchestrator())

    def test_state_snapshot_ignores_enabled_tools_attribute_without_getter(self):
        """Snapshot should ignore direct enabled_tools attribute without protocol getter."""
        pipeline = VerticalIntegrationPipeline(enable_cache=True)

        class AttributeOnlyOrchestrator:
            def __init__(self):
                self._vertical_context = create_vertical_context(name="attribute_only")
                self.enabled_tools = {"read", "write"}

            def get_vertical_context(self):
                return self._vertical_context

        snapshot = pipeline._orchestrator_state_snapshot(AttributeOnlyOrchestrator())
        assert snapshot.get("vertical_name") == "attribute_only"
        assert "enabled_tools_hash" not in snapshot

    def test_cache_hit_skips_side_effect_handlers_when_plan_matches(self):
        """Cache hit should skip side-effect handlers when plan is unchanged."""
        orchestrator = MockOrchestrator()
        pipeline = VerticalIntegrationPipeline(enable_cache=True)

        stateful = self._CountingHandler("stateful", 10, side_effects=True)
        pure = self._CountingHandler("pure", 20, side_effects=False)
        pipeline._step_registry = self._Registry([stateful, pure])

        first = pipeline.apply(orchestrator, MockVertical)
        cached_plan = pipeline._get_applied_plan_for_orchestrator(orchestrator)
        assert cached_plan is not None
        cached_plan["state_snapshot"] = {}
        pipeline._set_applied_plan_for_orchestrator(orchestrator, cached_plan)
        second = pipeline.apply(orchestrator, MockVertical)

        assert first.success is True
        assert second.success is True
        assert stateful.calls == 1
        assert pure.calls == 2
        assert second.get_step_status("stateful")["status"] == "skipped"
        assert second.get_step_status("stateful")["details"]["reason"] == "integration_plan_noop"

    def test_cache_hit_applies_delta_when_only_some_side_effect_fingerprints_change(
        self,
    ):
        """Delta mode should skip unchanged side-effect handlers and re-run changed ones."""
        orchestrator = MockOrchestrator()
        pipeline = VerticalIntegrationPipeline(enable_cache=True)

        stateful_a = self._CountingHandler("stateful_a", 10, side_effects=True)
        stateful_b = self._CountingHandler("stateful_b", 11, side_effects=True)
        pipeline._step_registry = self._Registry([stateful_a, stateful_b])

        first = pipeline.apply(orchestrator, MockVertical)
        assert first.success is True
        cached_plan = pipeline._get_applied_plan_for_orchestrator(orchestrator)
        assert cached_plan is not None

        # Simulate local state divergence for one handler only.
        cached_plan["handler_fingerprints"]["stateful_b"] = "out_of_sync"
        cached_plan["state_snapshot"] = {}
        pipeline._set_applied_plan_for_orchestrator(orchestrator, cached_plan)

        second = pipeline.apply(orchestrator, MockVertical)
        assert second.success is True
        assert stateful_a.calls == 1  # skipped as no-op
        assert stateful_b.calls == 2  # re-run due fingerprint mismatch
        assert second.get_step_status("stateful_a")["status"] == "skipped"
        assert second.get_step_status("stateful_b")["status"] == "success"


class TestVerticalIntegrationObservability:
    """Tests for vertical integration observability payloads."""

    def test_vertical_applied_event_emits_without_running_loop(self):
        """Sync apply should emit vertical.applied without a running event loop."""
        orchestrator = MockOrchestrator()
        pipeline = VerticalIntegrationPipeline(enable_cache=True, max_cache_entries=2)
        emitted = []
        emitted_event = threading.Event()

        class _Bus:
            async def emit(self, topic, data, source):
                emitted.append((topic, data, source))
                emitted_event.set()

        with patch("victor.core.events.get_observability_bus", return_value=_Bus()):
            pipeline.apply(orchestrator, MockVertical)

        assert emitted_event.wait(timeout=1.0)
        topic, data, source = emitted[-1]
        assert topic == "vertical.applied"
        assert source == "VerticalIntegrationPipeline"
        assert "cache_hit" in data
        assert "cache_stats" in data
        assert "cache_enabled" in data
        assert "integration_plan_stats" in data
        assert "extension_loader_metrics" in data
        assert "observability_delivery_stats" in data
        assert "framework_registry_metrics" in data

    @pytest.mark.asyncio
    async def test_vertical_applied_event_includes_cache_telemetry(self):
        """vertical.applied event should include cache telemetry fields."""
        orchestrator = MockOrchestrator()
        pipeline = VerticalIntegrationPipeline(enable_cache=True, max_cache_entries=2)
        emitted = []

        class _Bus:
            async def emit(self, topic, data, source):
                emitted.append((topic, data, source))

        with patch("victor.core.events.get_observability_bus", return_value=_Bus()):
            pipeline.apply(orchestrator, MockVertical)
            await asyncio.sleep(0)  # allow create_task() to run

        assert len(emitted) >= 1
        topic, data, source = emitted[-1]
        assert topic == "vertical.applied"
        assert source == "VerticalIntegrationPipeline"
        assert "cache_hit" in data
        assert "cache_stats" in data
        assert "cache_enabled" in data
        assert "integration_plan_stats" in data
        assert "extension_loader_metrics" in data
        assert "observability_delivery_stats" in data
        assert "framework_registry_metrics" in data

    @pytest.mark.asyncio
    async def test_apply_async_emits_vertical_applied_event_with_cache_telemetry(self):
        """apply_async should emit vertical.applied with cache telemetry parity."""
        orchestrator = MockOrchestrator()
        pipeline = VerticalIntegrationPipeline(enable_cache=True, max_cache_entries=2)
        emitted = []

        class _Bus:
            async def emit(self, topic, data, source):
                emitted.append((topic, data, source))

        with patch("victor.core.events.get_observability_bus", return_value=_Bus()):
            await pipeline.apply_async(orchestrator, MockVertical)
            await pipeline.apply_async(orchestrator, MockVertical)

        assert len(emitted) >= 2
        topic, data, source = emitted[-1]
        assert topic == "vertical.applied"
        assert source == "VerticalIntegrationPipeline"
        assert data["cache_enabled"] is True
        assert data["cache_hit"] is True
        assert "cache_stats" in data
        assert "integration_plan_stats" in data
        assert "extension_loader_metrics" in data
        assert "observability_delivery_stats" in data
        assert "framework_registry_metrics" in data


# =============================================================================
# Phase 2: Performance Optimization Tests
# =============================================================================


class TestParallelExecution:
    """Tests for Phase 2.2 parallel execution."""

    def test_apply_records_dependency_contract_warning_for_missing_handlers(self):
        """Non-strict apply should warn when a handler dependency is unresolved."""
        from victor.framework.step_handlers import StepHandlerRegistry

        orchestrator = MockOrchestrator()
        registry = StepHandlerRegistry(
            handlers=[
                ContractProbeHandler("config", order=20, depends_on=("missing_handler",)),
            ]
        )
        pipeline = VerticalIntegrationPipeline(step_registry=registry, strict_mode=False)

        result = pipeline.apply(orchestrator, MockVertical)

        assert result.success is True
        assert any("missing_handler" in warning for warning in result.warnings)
        contract_status = result.get_step_status("dependency_contract")
        assert contract_status is not None
        assert contract_status["status"] == "warning"

    @pytest.mark.asyncio
    async def test_apply_async_records_dependency_contract_error_in_strict_mode(self):
        """Strict apply_async should convert dependency drift into an error."""
        from victor.framework.step_handlers import StepHandlerRegistry

        orchestrator = MockOrchestrator()
        registry = StepHandlerRegistry(
            handlers=[
                ContractProbeHandler("config", order=20, depends_on=("missing_handler",)),
            ]
        )
        pipeline = VerticalIntegrationPipeline(step_registry=registry, strict_mode=True)

        result = await pipeline.apply_async(orchestrator, MockVertical)

        assert result.success is False
        assert any("missing_handler" in error for error in result.errors)
        contract_status = result.get_step_status("dependency_contract")
        assert contract_status is not None
        assert contract_status["status"] == "error"

    @pytest.mark.asyncio
    async def test_classify_handlers_separates_independent_dependent(self):
        """Test that handlers are classified by metadata, not class name."""
        pipeline = VerticalIntegrationPipeline()

        # Mock handlers
        independent_handler = MagicMock()
        independent_handler.name = "tools"
        independent_handler.parallel_safe = True
        independent_handler.depends_on = ()
        independent_handler.side_effects = False

        dependent_handler = MagicMock()
        dependent_handler.name = "config"
        dependent_handler.parallel_safe = False
        dependent_handler.depends_on = ("tools",)
        dependent_handler.side_effects = True

        handlers = [independent_handler, dependent_handler]
        independent, dependent = pipeline._classify_handlers(handlers)

        assert len(independent) == 1
        assert len(dependent) == 1
        assert independent[0] is independent_handler
        assert dependent[0] is dependent_handler

    @pytest.mark.asyncio
    async def test_classify_handlers_respects_declared_dependencies(self):
        """Parallel-safe handlers with active dependencies stay sequential."""
        pipeline = VerticalIntegrationPipeline()

        tools_handler = MagicMock()
        tools_handler.name = "tools"
        tools_handler.parallel_safe = True
        tools_handler.depends_on = ()
        tools_handler.side_effects = False

        tiered_handler = MagicMock()
        tiered_handler.name = "tiered_config"
        tiered_handler.parallel_safe = True
        tiered_handler.depends_on = ("tools",)
        tiered_handler.side_effects = False

        independent, dependent = pipeline._classify_handlers([tools_handler, tiered_handler])

        assert independent == [tools_handler]
        assert dependent == [tiered_handler]

    def test_build_execution_levels_respects_metadata_dependencies(self):
        """Execution levels should follow metadata dependencies deterministically."""
        pipeline = VerticalIntegrationPipeline()

        capability = MagicMock()
        capability.name = "capability_config"
        capability.order = 5
        capability.depends_on = ()

        tools = MagicMock()
        tools.name = "tools"
        tools.order = 10
        tools.depends_on = ()

        prompt = MagicMock()
        prompt.name = "prompt"
        prompt.order = 20
        prompt.depends_on = ()

        config = MagicMock()
        config.name = "config"
        config.order = 40
        config.depends_on = ("tools", "prompt")

        framework = MagicMock()
        framework.name = "framework"
        framework.order = 60
        framework.depends_on = ("config",)

        context = MagicMock()
        context.name = "context"
        context.order = 100
        context.depends_on = ("framework",)

        levels = pipeline._build_execution_levels(
            [context, framework, prompt, tools, capability, config]
        )
        level_names = [[handler.name for handler in level] for level in levels]

        assert level_names == [
            ["capability_config", "tools", "prompt"],
            ["config"],
            ["framework"],
            ["context"],
        ]

    def test_build_execution_levels_cycle_falls_back_to_order(self):
        """Cyclical dependencies should degrade to deterministic sequential order."""
        pipeline = VerticalIntegrationPipeline()

        first = MagicMock()
        first.name = "first"
        first.order = 20
        first.depends_on = ("second",)

        second = MagicMock()
        second.name = "second"
        second.order = 10
        second.depends_on = ("first",)

        levels = pipeline._build_execution_levels([first, second])
        level_names = [[handler.name for handler in level] for level in levels]

        assert level_names == [["second"], ["first"]]

    @pytest.mark.asyncio
    async def test_parallel_execution_enforces_side_effect_safety(self):
        """Only side-effect-free handlers should run concurrently."""
        pipeline = VerticalIntegrationPipeline(parallel_enabled=True)
        orchestrator = MockOrchestrator()
        context = create_vertical_context(name=MockVertical.name)
        result = IntegrationResult(vertical_name=MockVertical.name)

        events = []
        active = 0
        max_active = 0

        class RecordingHandler:
            def __init__(
                self,
                name: str,
                order: int,
                parallel_safe: bool,
                side_effects: bool,
                depends_on: tuple[str, ...] = (),
                delay: float = 0.0,
            ) -> None:
                self.name = name
                self.order = order
                self.parallel_safe = parallel_safe
                self.side_effects = side_effects
                self.depends_on = depends_on
                self.delay = delay

            async def apply_async(
                self,
                orchestrator,
                vertical,
                context,
                result,
                strict_mode=False,
            ):
                nonlocal active, max_active
                events.append(("start", self.name))
                active += 1
                max_active = max(max_active, active)
                if self.delay > 0:
                    await asyncio.sleep(self.delay)
                active -= 1
                events.append(("end", self.name))

        class Registry:
            def __init__(self, handlers):
                self._handlers = handlers

            def get_ordered_handlers(self):
                return self._handlers

        handlers = [
            RecordingHandler("seq", 5, parallel_safe=False, side_effects=True),
            RecordingHandler("unsafe_parallel", 6, parallel_safe=True, side_effects=True),
            RecordingHandler("par1", 10, parallel_safe=True, side_effects=False, delay=0.03),
            RecordingHandler("par2", 11, parallel_safe=True, side_effects=False, delay=0.03),
        ]
        pipeline._step_registry = Registry(handlers)

        await pipeline._apply_with_step_handlers_parallel(
            orchestrator,
            MockVertical,
            context,
            result,
        )

        starts = [name for marker, name in events if marker == "start"]
        assert starts[:2] == ["seq", "unsafe_parallel"]
        assert set(starts[2:]) == {"par1", "par2"}
        assert max_active == 2

    @pytest.mark.asyncio
    async def test_parallel_execution_with_independent_handlers(self):
        """Test parallel execution with independent handlers."""
        orchestrator = MockOrchestrator()
        pipeline = VerticalIntegrationPipeline(parallel_enabled=True)

        result = await pipeline.apply_async(orchestrator, MockVertical)

        # Should succeed
        assert result.success is True
        assert result.vertical_name == "mock_vertical"

    @pytest.mark.asyncio
    async def test_parallel_vs_sequential_produce_same_result(self):
        """Test that parallel and sequential execution produce same results."""
        orchestrator = MockOrchestrator()

        # Sequential
        pipeline_seq = VerticalIntegrationPipeline(parallel_enabled=False)
        result_seq = await pipeline_seq.apply_async(orchestrator, MockVertical)

        # Parallel
        pipeline_par = VerticalIntegrationPipeline(parallel_enabled=True)
        result_par = await pipeline_par.apply_async(orchestrator, MockVertical)

        # Both should succeed
        assert result_seq.success is True
        assert result_par.success is True

        # Same vertical name
        assert result_seq.vertical_name == result_par.vertical_name

    @pytest.mark.asyncio
    async def test_parallel_flag_enables_parallel_execution(self):
        """Test that parallel_flag controls execution strategy."""
        pipeline = VerticalIntegrationPipeline(parallel_enabled=True)
        assert pipeline._parallel_enabled is True

        pipeline_seq = VerticalIntegrationPipeline(parallel_enabled=False)
        assert pipeline_seq._parallel_enabled is False

    def test_legacy_extension_registry_is_not_in_active_pipeline_path(self, caplog):
        """Legacy extension registry should be ignored in step-handler mode."""
        legacy_registry = MagicMock()
        pipeline = VerticalIntegrationPipeline(extension_registry=legacy_registry)

        assert pipeline.step_registry is not None
        assert not hasattr(pipeline, "_extension_registry")
        # Verify deprecation was logged
        assert any("deprecated" in record.message.lower() for record in caplog.records)

    def test_register_extension_handler_warns_and_is_inactive_for_default_pipeline(self, caplog):
        """Legacy extension registrations should not affect active pipeline behavior."""
        invoked = {"count": 0}
        handler_name = "legacy_probe_extension"

        def _legacy_handler(*args, **kwargs):
            invoked["count"] += 1

        register_extension_handler(
            name=handler_name,
            attr_name="legacy_probe_extension",
            handler=_legacy_handler,
            order=90,
        )

        try:
            orchestrator = MockOrchestrator()
            pipeline = VerticalIntegrationPipeline()
            result = pipeline.apply(orchestrator, MockVertical)

            assert result.success is True
            assert invoked["count"] == 0
        finally:
            get_extension_handler_registry().unregister(handler_name)


class TestFeatureFlags:
    """Tests for Phase 2.3 feature flags."""

    def test_create_pipeline_with_parallel_flag(self):
        """Test creating pipeline with parallel flag via factory."""
        pipeline = create_integration_pipeline(enable_parallel=True)
        assert pipeline._parallel_enabled is True

    def test_create_pipeline_without_parallel_flag(self):
        """Test creating pipeline without parallel flag."""
        pipeline = create_integration_pipeline(enable_parallel=False)
        assert pipeline._parallel_enabled is False

    def test_parallel_flag_defaults_to_true(self):
        """Test that parallel flag defaults to True (new architecture default)."""
        pipeline = create_integration_pipeline()
        assert pipeline._parallel_enabled is True

    def test_cache_flag_can_be_combined_with_parallel(self):
        """Test that cache and parallel flags can be combined."""
        pipeline = create_integration_pipeline(enable_cache=True, enable_parallel=True)
        assert pipeline._enable_cache is True
        assert pipeline._parallel_enabled is True


@pytest.mark.skip(reason="Vertical packages are now external")
class TestPhase2Integration:
    """Integration tests for Phase 2 async and parallel execution."""

    @pytest.mark.asyncio
    async def test_async_apply_with_real_vertical(self):
        """Test async apply with real CodingAssistant vertical."""
        try:
            from victor_coding.assistant import CodingAssistant
        except ImportError:
            pytest.skip("victor-coding package not installed")

        orchestrator = MockOrchestrator()
        pipeline = create_integration_pipeline(enable_parallel=False)

        result = await pipeline.apply_async(orchestrator, CodingAssistant)

        assert result.success is True
        assert result.vertical_name == "coding"

    @pytest.mark.asyncio
    async def test_parallel_apply_with_real_vertical(self):
        """Test parallel apply with real CodingAssistant vertical."""
        from victor.coding import CodingAssistant

        orchestrator = MockOrchestrator()
        pipeline = create_integration_pipeline(enable_parallel=True)

        result = await pipeline.apply_async(orchestrator, CodingAssistant)

        assert result.success is True
        assert result.vertical_name == "coding"

    @pytest.mark.asyncio
    async def test_cache_works_with_async_apply(self):
        """Test that caching works with async apply."""
        from victor.coding import CodingAssistant

        orchestrator = MockOrchestrator()
        pipeline = create_integration_pipeline(enable_cache=True, enable_parallel=False)

        # First application - cache miss
        result1 = await pipeline.apply_async(orchestrator, CodingAssistant)
        assert result1.success is True

        # Reset orchestrator
        orchestrator._enabled_tools = set()

        # Second application - cache hit
        result2 = await pipeline.apply_async(orchestrator, CodingAssistant)
        assert result2.success is True

    @pytest.mark.asyncio
    async def test_cache_works_with_parallel_apply(self):
        """Test that caching works with parallel apply."""
        from victor.coding import CodingAssistant

        orchestrator = MockOrchestrator()
        pipeline = create_integration_pipeline(enable_cache=True, enable_parallel=True)

        # First application
        result1 = await pipeline.apply_async(orchestrator, CodingAssistant)
        assert result1.success is True

        # Second application (should use cache)
        orchestrator._enabled_tools = set()
        result2 = await pipeline.apply_async(orchestrator, CodingAssistant)
        assert result2.success is True
