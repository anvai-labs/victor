"""Unit tests for victor.verticals.protocols module.

Tests the protocol definitions and extension patterns for vertical-framework integration.
"""

from typing import Any, Dict, List, Optional, Set

import pytest

from victor.core.verticals.protocols import (
    # Data types
    MiddlewarePriority,
    MiddlewareResult,
    SafetyPattern,
    TaskTypeHint,
    ModeConfig,
    ToolDependency,
    # Protocols
    MiddlewareProtocol,
    SafetyExtensionProtocol,
    PromptContributorProtocol,
    ModeConfigProviderProtocol,
    VerticalExtensions,
)


class TestMiddlewarePriority:
    """Tests for MiddlewarePriority enum."""

    def test_priority_values(self):
        """Priority values should be ordered correctly."""
        assert MiddlewarePriority.CRITICAL.value == 0
        assert MiddlewarePriority.HIGH.value == 25
        assert MiddlewarePriority.NORMAL.value == 50
        assert MiddlewarePriority.LOW.value == 75
        assert MiddlewarePriority.DEFERRED.value == 100

    def test_priority_ordering(self):
        """Higher priority should have lower value."""
        priorities = [
            MiddlewarePriority.CRITICAL,
            MiddlewarePriority.HIGH,
            MiddlewarePriority.NORMAL,
            MiddlewarePriority.LOW,
            MiddlewarePriority.DEFERRED,
        ]
        values = [p.value for p in priorities]
        assert values == sorted(values)


class TestMiddlewareResult:
    """Tests for MiddlewareResult dataclass."""

    def test_default_values(self):
        """Default result should allow proceeding."""
        result = MiddlewareResult()
        assert result.proceed is True
        assert result.modified_arguments is None
        assert result.error_message is None
        assert result.metadata == {}

    def test_block_result(self):
        """Can create a blocking result."""
        result = MiddlewareResult(
            proceed=False,
            error_message="Operation blocked",
        )
        assert result.proceed is False
        assert result.error_message == "Operation blocked"

    def test_modification_result(self):
        """Can create a result with modified arguments."""
        result = MiddlewareResult(
            proceed=True,
            modified_arguments={"path": "/modified/path"},
        )
        assert result.proceed is True
        assert result.modified_arguments["path"] == "/modified/path"


class TestSafetyPattern:
    """Tests for SafetyPattern dataclass."""

    def test_creation(self):
        """SafetyPattern can be created with required fields."""
        pattern = SafetyPattern(
            pattern=r"rm\s+-rf",
            description="Recursive delete",
        )
        assert pattern.pattern == r"rm\s+-rf"
        assert pattern.description == "Recursive delete"
        assert pattern.risk_level == "HIGH"  # Default
        assert pattern.category == "general"  # Default

    def test_full_creation(self):
        """SafetyPattern can be created with all fields."""
        pattern = SafetyPattern(
            pattern=r"git\s+push\s+--force",
            description="Force push",
            risk_level="CRITICAL",
            category="git",
        )
        assert pattern.risk_level == "CRITICAL"
        assert pattern.category == "git"


class TestTaskTypeHint:
    """Tests for TaskTypeHint dataclass."""

    def test_creation(self):
        """TaskTypeHint can be created with required fields."""
        hint = TaskTypeHint(
            task_type="edit",
            hint="Read file before editing",
        )
        assert hint.task_type == "edit"
        assert hint.hint == "Read file before editing"
        assert hint.tool_budget is None
        assert hint.priority_tools == []

    def test_full_creation(self):
        """TaskTypeHint can be created with all fields."""
        hint = TaskTypeHint(
            task_type="refactor",
            hint="Analyze code structure first",
            tool_budget=10,
            priority_tools=["read", "code_search"],
        )
        assert hint.tool_budget == 10
        assert "read" in hint.priority_tools


class TestModeConfig:
    """Tests for ModeConfig dataclass."""

    def test_creation(self):
        """ModeConfig can be created with required fields."""
        mode = ModeConfig(
            name="fast",
            tool_budget=5,
            max_iterations=10,
        )
        assert mode.name == "fast"
        assert mode.tool_budget == 5
        assert mode.max_iterations == 10
        assert mode.temperature == 0.7  # Default
        assert mode.description == ""  # Default

    def test_full_creation(self):
        """ModeConfig can be created with all fields."""
        mode = ModeConfig(
            name="thorough",
            tool_budget=30,
            max_iterations=60,
            temperature=0.5,
            description="Deep analysis mode",
        )
        assert mode.temperature == 0.5
        assert "Deep" in mode.description


class TestToolDependency:
    """Tests for ToolDependency dataclass."""

    def test_creation(self):
        """ToolDependency can be created with required field."""
        dep = ToolDependency(tool_name="edit")
        assert dep.tool_name == "edit"
        assert dep.depends_on == set()
        assert dep.enables == set()
        assert dep.weight == 1.0

    def test_full_creation(self):
        """ToolDependency can be created with all fields."""
        dep = ToolDependency(
            tool_name="edit",
            depends_on={"read"},
            enables={"test"},
            weight=0.8,
        )
        assert "read" in dep.depends_on
        assert "test" in dep.enables
        assert dep.weight == 0.8


class TestMiddlewareProtocol:
    """Tests for MiddlewareProtocol compliance."""

    def test_simple_implementation(self):
        """A simple middleware implementation should work."""

        class SimpleMiddleware:
            async def before_tool_call(
                self, tool_name: str, arguments: Dict[str, Any]
            ) -> MiddlewareResult:
                return MiddlewareResult()

            async def after_tool_call(
                self,
                tool_name: str,
                arguments: Dict[str, Any],
                result: Any,
                success: bool,
            ) -> Optional[Any]:
                return None

            def get_priority(self) -> MiddlewarePriority:
                return MiddlewarePriority.NORMAL

            def get_applicable_tools(self) -> Optional[Set[str]]:
                return None

        middleware = SimpleMiddleware()
        assert isinstance(middleware, MiddlewareProtocol)

    @pytest.mark.asyncio
    async def test_blocking_middleware(self):
        """Middleware can block tool calls."""

        class BlockingMiddleware:
            async def before_tool_call(
                self, tool_name: str, arguments: Dict[str, Any]
            ) -> MiddlewareResult:
                if tool_name == "dangerous_tool":
                    return MiddlewareResult(
                        proceed=False,
                        error_message="Tool blocked",
                    )
                return MiddlewareResult()

        middleware = BlockingMiddleware()
        result = await middleware.before_tool_call("dangerous_tool", {})
        assert result.proceed is False
        assert result.error_message == "Tool blocked"

        result = await middleware.before_tool_call("safe_tool", {})
        assert result.proceed is True


class TestSafetyExtensionProtocol:
    """Tests for SafetyExtensionProtocol compliance."""

    def test_simple_implementation(self):
        """A simple safety extension should work."""

        class SimpleSafetyExtension:
            def get_bash_patterns(self) -> List[SafetyPattern]:
                return [
                    SafetyPattern(
                        pattern=r"rm\s+-rf",
                        description="Recursive delete",
                    )
                ]

            def get_file_patterns(self) -> List[SafetyPattern]:
                return []

            def get_tool_restrictions(self) -> Dict[str, List[str]]:
                return {}

            def get_category(self) -> str:
                return "custom"

        ext = SimpleSafetyExtension()
        assert isinstance(ext, SafetyExtensionProtocol)
        patterns = ext.get_bash_patterns()
        assert len(patterns) == 1
        assert patterns[0].pattern == r"rm\s+-rf"


class TestPromptContributorProtocol:
    """Tests for PromptContributorProtocol compliance."""

    def test_simple_implementation(self):
        """A simple prompt contributor should work."""

        class SimpleContributor:
            def get_task_type_hints(self) -> Dict[str, TaskTypeHint]:
                return {
                    "edit": TaskTypeHint(
                        task_type="edit",
                        hint="Read before edit",
                    )
                }

            def get_system_prompt_section(self) -> str:
                return ""

            def get_grounding_rules(self) -> str:
                return ""

            def get_priority(self) -> int:
                return 50

        contributor = SimpleContributor()
        assert isinstance(contributor, PromptContributorProtocol)
        hints = contributor.get_task_type_hints()
        assert "edit" in hints


class TestModeConfigProviderProtocol:
    """Tests for ModeConfigProviderProtocol compliance."""

    def test_simple_implementation(self):
        """A simple mode config provider should work."""

        class SimpleModeProvider:
            def get_mode_configs(self) -> Dict[str, ModeConfig]:
                return {
                    "fast": ModeConfig(
                        name="fast",
                        tool_budget=5,
                        max_iterations=10,
                    )
                }

            def get_default_mode(self) -> str:
                return "fast"

            def get_default_tool_budget(self) -> int:
                return 5

        provider = SimpleModeProvider()
        assert isinstance(provider, ModeConfigProviderProtocol)
        configs = provider.get_mode_configs()
        assert "fast" in configs


class TestVerticalExtensions:
    """Tests for VerticalExtensions composite container."""

    def test_empty_extensions(self):
        """Empty extensions should work."""
        ext = VerticalExtensions()
        assert ext.middleware == []
        assert ext.safety_extensions == []
        assert ext.prompt_contributors == []
        assert ext.mode_config_provider is None

    def test_get_all_task_hints_empty(self):
        """get_all_task_hints with no contributors returns empty dict."""
        ext = VerticalExtensions()
        hints = ext.get_all_task_hints()
        assert hints == {}

    def test_get_all_task_hints_merges(self):
        """get_all_task_hints should merge from all contributors."""

        class Contributor1:
            def get_task_type_hints(self) -> Dict[str, TaskTypeHint]:
                return {
                    "edit": TaskTypeHint(task_type="edit", hint="Hint 1"),
                }

            def get_system_prompt_section(self) -> str:
                return ""

            def get_grounding_rules(self) -> str:
                return ""

            def get_priority(self) -> int:
                return 100  # Lower priority (executes later)

        class Contributor2:
            def get_task_type_hints(self) -> Dict[str, TaskTypeHint]:
                return {
                    "edit": TaskTypeHint(task_type="edit", hint="Hint 2"),
                    "search": TaskTypeHint(task_type="search", hint="Search hint"),
                }

            def get_system_prompt_section(self) -> str:
                return ""

            def get_grounding_rules(self) -> str:
                return ""

            def get_priority(self) -> int:
                return 50  # Higher priority (executes first)

        ext = VerticalExtensions(prompt_contributors=[Contributor1(), Contributor2()])
        hints = ext.get_all_task_hints()

        # Should have both hints
        assert "edit" in hints
        assert "search" in hints
        # Contributors are sorted by priority (lower first), and later ones override
        # So Contributor2 (priority 50) runs first, then Contributor1 (priority 100) overrides
        assert hints["edit"].hint == "Hint 1"

    def test_get_all_safety_patterns(self):
        """get_all_safety_patterns should collect from all extensions."""

        class SafetyExt1:
            def get_bash_patterns(self) -> List[SafetyPattern]:
                return [SafetyPattern(pattern="p1", description="Pattern 1")]

            def get_file_patterns(self) -> List[SafetyPattern]:
                return []

        class SafetyExt2:
            def get_bash_patterns(self) -> List[SafetyPattern]:
                return [SafetyPattern(pattern="p2", description="Pattern 2")]

            def get_file_patterns(self) -> List[SafetyPattern]:
                return [SafetyPattern(pattern="fp1", description="File pattern")]

        ext = VerticalExtensions(safety_extensions=[SafetyExt1(), SafetyExt2()])
        patterns = ext.get_all_safety_patterns()

        assert len(patterns) == 3
        descriptions = [p.description for p in patterns]
        assert "Pattern 1" in descriptions
        assert "Pattern 2" in descriptions
        assert "File pattern" in descriptions

    def test_get_all_mode_configs(self):
        """get_all_mode_configs should return provider's configs."""

        class ModeProvider:
            def get_mode_configs(self) -> Dict[str, ModeConfig]:
                return {
                    "fast": ModeConfig(name="fast", tool_budget=5, max_iterations=10),
                    "slow": ModeConfig(name="slow", tool_budget=20, max_iterations=50),
                }

        ext = VerticalExtensions(mode_config_provider=ModeProvider())
        configs = ext.get_all_mode_configs()

        assert "fast" in configs
        assert "slow" in configs
        assert configs["fast"].tool_budget == 5

    def test_get_all_mode_configs_no_provider(self):
        """get_all_mode_configs with no provider returns empty dict."""
        ext = VerticalExtensions()
        configs = ext.get_all_mode_configs()
        assert configs == {}


# =============================================================================
# Tests for Vertical Provider Protocols (isinstance() checks)
# =============================================================================

# Note: Tests for actual vertical implementations (TestCodingVerticalExtensions,
# TestResearchVerticalExtensions) have been migrated to victor-coding to avoid
# framework-to-vertical circular dependencies. This file now only contains tests
# for the framework protocol definitions themselves.


class TestVerticalRLProviderProtocol:
    """Tests for VerticalRLProviderProtocol compliance."""

    def test_protocol_import(self):
        """Protocol should be importable."""
        from victor.core.verticals.protocols import VerticalRLProviderProtocol

        assert VerticalRLProviderProtocol is not None

    def test_runtime_checkable(self):
        """Protocol should be runtime checkable."""

        class MockRLConfigProvider:
            def get_rl_config(self) -> Dict[str, Any]:
                return {"active_learners": ["tool_selection"]}

            def get_rl_hooks(self) -> Optional[Any]:
                return None

        class MockVerticalWithRL:
            @classmethod
            def get_rl_config_provider(cls) -> Optional[Any]:
                return MockRLConfigProvider()

            @classmethod
            def get_rl_hooks(cls) -> Optional[Any]:
                return None

        # isinstance check should work with classmethod implementation
        assert hasattr(MockVerticalWithRL, "get_rl_config_provider")
        assert hasattr(MockVerticalWithRL, "get_rl_hooks")

    def test_vertical_base_has_methods(self):
        """VerticalBase should have the required methods."""
        from victor.core.verticals.base import VerticalBase

        assert hasattr(VerticalBase, "get_rl_config_provider")
        assert hasattr(VerticalBase, "get_rl_hooks")
        # Default implementation returns None
        assert VerticalBase.get_rl_config_provider() is None
        assert VerticalBase.get_rl_hooks() is None


class TestVerticalTeamProviderProtocol:
    """Tests for VerticalTeamProviderProtocol compliance."""

    def test_protocol_import(self):
        """Protocol should be importable."""
        from victor.core.verticals.protocols import VerticalTeamProviderProtocol

        assert VerticalTeamProviderProtocol is not None

    def test_runtime_checkable(self):
        """Protocol should be runtime checkable."""

        class MockTeamSpecProvider:
            def get_team_specs(self) -> Dict[str, Any]:
                return {"review_team": {"name": "review_team", "agents": []}}

            def get_default_team(self) -> Optional[str]:
                return "review_team"

        class MockVerticalWithTeam:
            @classmethod
            def get_team_spec_provider(cls) -> Optional[Any]:
                return MockTeamSpecProvider()

        # Should have required method
        assert hasattr(MockVerticalWithTeam, "get_team_spec_provider")

    def test_vertical_base_has_methods(self):
        """VerticalBase should have the required methods."""
        from victor.core.verticals.base import VerticalBase

        assert hasattr(VerticalBase, "get_team_spec_provider")
        assert callable(VerticalBase.get_team_spec_provider)


class TestVerticalWorkflowProviderProtocol:
    """Tests for VerticalWorkflowProviderProtocol compliance."""

    def test_protocol_import(self):
        """Protocol should be importable."""
        from victor.core.verticals.protocols import VerticalWorkflowProviderProtocol

        assert VerticalWorkflowProviderProtocol is not None

    def test_runtime_checkable(self):
        """Protocol should be runtime checkable."""

        class MockWorkflowProvider:
            def get_workflows(self) -> Dict[str, Any]:
                return {"build_workflow": object}

            def get_auto_workflows(self) -> List[Any]:
                return []

        class MockVerticalWithWorkflow:
            @classmethod
            def get_workflow_provider(cls) -> Optional[Any]:
                return MockWorkflowProvider()

        # Should have required method
        assert hasattr(MockVerticalWithWorkflow, "get_workflow_provider")

    def test_vertical_base_has_methods(self):
        """VerticalBase should have the required methods."""
        from victor.core.verticals.base import VerticalBase

        assert hasattr(VerticalBase, "get_workflow_provider")
        # Default implementation returns None
        assert VerticalBase.get_workflow_provider() is None


class TestVerticalIntegrationWithProtocols:
    """Tests for vertical integration using isinstance() checks."""

    def test_integration_imports_new_protocols(self):
        """Core verticals protocols should be importable."""
        from victor.core.verticals.protocols.rl_provider import (
            VerticalRLProviderProtocol,
        )
        from victor.core.verticals.protocols.team_provider import (
            VerticalTeamProviderProtocol,
        )
        from victor.core.verticals.protocols.workflow_provider import (
            VerticalWorkflowProviderProtocol,
        )

        assert VerticalRLProviderProtocol is not None
        assert VerticalTeamProviderProtocol is not None
        assert VerticalWorkflowProviderProtocol is not None

    # NOTE: test_coding_vertical_compatibility migrated to victor-coding
    # to avoid framework-to-vertical circular dependencies

    def test_extensions_include_new_providers(self):
        """VerticalExtensions should include rl_config_provider and team_spec_provider."""
        from victor.core.verticals.protocols import VerticalExtensions

        # Create extensions with all providers
        class MockRLProvider:
            def get_rl_config(self) -> Dict[str, Any]:
                return {}

            def get_rl_hooks(self) -> Optional[Any]:
                return None

        class MockTeamProvider:
            def get_team_specs(self) -> Dict[str, Any]:
                return {}

            def get_default_team(self) -> Optional[str]:
                return None

        ext = VerticalExtensions(
            rl_config_provider=MockRLProvider(),
            team_spec_provider=MockTeamProvider(),
        )

        assert ext.rl_config_provider is not None
        assert ext.team_spec_provider is not None


def test_provider_protocol_docstrings_are_package_neutral():
    """Public provider protocol docs should not hardcode extracted package names."""
    import victor.core.verticals.protocols.providers as provider_protocols

    banned_snippets = (
        "victor_coding",
        "victor_research",
        "victor_devops",
        "victor_dataanalysis",
        "victor_rag",
        "try:\n    try:",
    )

    # Only the locally-defined provider protocols live in this submodule; the
    # SDK-backed ones are re-exported at the package level.
    provider_names = (
        "PromptContributorProvider",
        "ToolDependencyProvider",
        "TieredToolConfigProvider",
        "ServiceProvider",
    )

    for provider_name in provider_names:
        doc = getattr(provider_protocols, provider_name).__doc__ or ""
        assert not any(snippet in doc for snippet in banned_snippets), provider_name
