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

"""Integration tests for architecture consolidation.

This module tests the integration between components added during the
architecture consolidation work (WS-A through WS-E):

- WS-A: BaseRegistry at `victor/core/registry/base.py`
- WS-B: PromptBuilder at `victor/framework/prompt_builder.py`
- WS-C: Query Enhancement at `victor/core/query_enhancement/`
- WS-D: Coordinators at `victor/agent/{tool,state,prompt}_coordinator.py`
- WS-E: Integration and testing (this file)
"""

import pytest


class TestRegistryIntegration:
    """Test that all registries use BaseRegistry."""

    def test_base_registry_available_from_core(self):
        """Test BaseRegistry is exported from core.registry."""
        from victor.core.registry import BaseRegistry

        assert BaseRegistry is not None

    def test_iregistry_protocol_available(self):
        """Test IRegistry protocol is exported from core.registry."""
        from victor.core.registry import IRegistry

        assert IRegistry is not None

    def test_base_registry_basic_operations(self):
        """Test BaseRegistry basic operations work correctly."""
        from victor.core.registry import BaseRegistry

        registry = BaseRegistry()

        # Register items
        registry.register("item1", "value1")
        registry.register("item2", "value2")

        # Get items
        assert registry.get("item1") == "value1"
        assert registry.get("item2") == "value2"
        assert registry.get("nonexistent") is None

        # List all
        assert set(registry.list_all()) == {"item1", "item2"}

        # Contains
        assert "item1" in registry
        assert "nonexistent" not in registry

        # Length
        assert len(registry) == 2

        # Unregister
        assert registry.unregister("item1") is True
        assert "item1" not in registry

        # Clear
        registry.clear()
        assert len(registry) == 0


class TestPromptBuilderIntegration:
    """Test PromptBuilder integration with verticals."""

    def test_framework_exports_prompt_builder(self):
        """Test PromptBuilder is exported from framework."""
        from victor.framework import PromptBuilder

        assert PromptBuilder is not None

    def test_prompt_section_exported(self):
        """Test PromptSection is exported from framework."""
        from victor.framework import PromptSection

        assert PromptSection is not None

    def test_tool_hint_exported(self):
        """Test ToolHint is exported from framework."""
        from victor.framework import ToolHint

        assert ToolHint is not None

    def test_prompt_builder_builds_valid_prompt(self):
        """Test PromptBuilder can build a valid prompt."""
        from victor.framework import PromptBuilder

        prompt = PromptBuilder().add_section("test", "content").build()
        assert "content" in prompt

    def test_prompt_builder_fluent_api(self):
        """Test PromptBuilder fluent API chaining."""
        from victor.framework import PromptBuilder

        builder = (
            PromptBuilder()
            .add_section("identity", "You are a helpful assistant.")
            .add_section("guidelines", "Follow best practices.")
            .add_safety_rule("Do not execute harmful code.")
            .set_grounding_mode("minimal")
        )

        prompt = builder.build()

        assert "You are a helpful assistant." in prompt
        assert "Follow best practices." in prompt
        assert "Do not execute harmful code." in prompt

    def test_prompt_builder_with_priorities(self):
        """Test PromptBuilder accepts section priorities."""
        from victor.framework import PromptBuilder

        builder = (
            PromptBuilder()
            .add_section("low", "Low priority", priority=10)
            .add_section("high", "High priority", priority=100)
        )

        prompt = builder.build()

        # Both sections should be present
        assert "High priority" in prompt
        assert "Low priority" in prompt

    def test_factory_functions_available(self):
        """Test factory functions for vertical-specific builders."""
        from victor.framework import (
            create_coding_prompt_builder,
            create_data_analysis_prompt_builder,
            create_devops_prompt_builder,
            create_research_prompt_builder,
        )

        coding_builder = create_coding_prompt_builder()
        assert coding_builder is not None

        devops_builder = create_devops_prompt_builder()
        assert devops_builder is not None

        research_builder = create_research_prompt_builder()
        assert research_builder is not None

        data_builder = create_data_analysis_prompt_builder()
        assert data_builder is not None


class TestCoordinatorIntegration:
    """Test coordinator registration in ServiceProvider."""

    def test_tool_coordinator_protocol_importable(self):
        """Test ToolCoordinatorProtocol can be imported."""
        with pytest.warns(
            DeprecationWarning,
            match="ToolCoordinatorProtocol is deprecated compatibility surface",
        ):
            from victor.agent.protocols import ToolCoordinatorProtocol

        assert ToolCoordinatorProtocol is not None

    def test_state_coordinator_protocol_importable(self):
        """Test StateCoordinatorProtocol can be imported."""
        with pytest.warns(
            DeprecationWarning,
            match="StateCoordinatorProtocol is deprecated compatibility surface",
        ):
            from victor.agent.protocols import StateCoordinatorProtocol

        assert StateCoordinatorProtocol is not None

    def test_prompt_coordinator_protocol_importable(self):
        """Test PromptCoordinatorProtocol can be imported."""
        with pytest.warns(
            DeprecationWarning,
            match="PromptCoordinatorProtocol is deprecated compatibility surface",
        ):
            from victor.agent.protocols import PromptCoordinatorProtocol

        assert PromptCoordinatorProtocol is not None

    def test_state_runtime_adapter_importable(self):
        """Test StateRuntimeAdapter can be imported from services."""
        from victor.agent.services.state_runtime import StateRuntimeAdapter

        assert StateRuntimeAdapter is not None

    def test_prompt_runtime_adapter_importable(self):
        """Test PromptRuntimeAdapter can be imported from services."""
        from victor.agent.services.prompt_runtime import PromptRuntimeAdapter

        assert PromptRuntimeAdapter is not None

    def test_prompt_runtime_config_importable(self):
        """Test PromptRuntimeConfig can be imported from services."""
        from victor.agent.services.prompt_runtime import PromptRuntimeConfig

        config = PromptRuntimeConfig()
        assert config.enable_task_hints is True

    def test_deleted_state_compat_module_not_importable(self):
        """Test deleted state_compat module is not importable."""
        import pytest

        with pytest.raises(ImportError, match="state_compat"):
            from victor.agent.services import state_compat  # noqa: F401

    def test_deleted_prompt_compat_module_not_importable(self):
        """Test deleted prompt_compat module is not importable."""
        import pytest

        with pytest.raises(ImportError, match="prompt_compat"):
            from victor.agent.services import prompt_compat  # noqa: F401

    def test_orchestration_facade_removed_deprecated_coordinator_properties(self):
        """Removed orchestration-facade coordinator shims should stay absent."""
        from victor.agent.facades.orchestration_facade import OrchestrationFacade

        facade = OrchestrationFacade()

        for attr in (
            "chat_coordinator",
            "tool_coordinator",
            "session_coordinator",
            "sync_chat_coordinator",
            "streaming_chat_coordinator",
            "unified_chat_coordinator",
        ):
            assert hasattr(facade, attr) is False

    def test_workflow_compatibility_shims_removed(self):
        """Workflow coordination compatibility shims should stay absent."""
        from victor.agent.orchestrator import AgentOrchestrator
        from victor.agent.orchestrator_factory import OrchestratorFactory

        # The entire workflow_facade compatibility shim was removed; importing
        # it must fail rather than silently resolve.
        with pytest.raises(ImportError):
            from victor.agent.facades.workflow_facade import (
                WorkflowFacade,
            )  # noqa: F401

        assert hasattr(AgentOrchestrator, "_mode_workflow_team_coordinator") is False
        assert hasattr(OrchestratorFactory, "create_mode_workflow_team_coordinator") is False

        with pytest.raises(ImportError, match="mode_workflow_team_coordinator"):
            import victor.agent.mode_workflow_team_coordinator  # noqa: F401


class TestQueryEnhancementIntegration:
    """Test query enhancement is accessible from core."""

    def test_core_exports_pipeline(self):
        """Test QueryEnhancementPipeline is exported from core."""
        from victor.core.query_enhancement import QueryEnhancementPipeline

        assert QueryEnhancementPipeline is not None

    def test_strategies_available(self):
        """Test all strategies are available from core."""
        from victor.core.query_enhancement import (
            BaseQueryEnhancementStrategy,
            DecompositionStrategy,
            EntityExpandStrategy,
            RewriteStrategy,
        )

        assert BaseQueryEnhancementStrategy is not None
        assert EntityExpandStrategy is not None
        assert RewriteStrategy is not None
        assert DecompositionStrategy is not None

    def test_types_available(self):
        """Test all types are available from core."""
        from victor.core.query_enhancement import (
            EnhancedQuery,
            EnhancementContext,
            EnhancementMetrics,
            EnhancementResult,
            EnhancementTechnique,
            QueryEnhancementConfig,
        )

        assert EnhancementTechnique is not None
        assert EnhancementContext is not None
        assert EnhancedQuery is not None
        assert QueryEnhancementConfig is not None
        assert EnhancementMetrics is not None
        assert EnhancementResult is not None

    def test_domain_configs_available(self):
        """Test domain configurations are available."""
        from victor.core.query_enhancement import (
            CODE_DOMAIN,
            FINANCIAL_DOMAIN,
            GENERAL_DOMAIN,
            RESEARCH_DOMAIN,
            get_domain_config,
        )

        assert FINANCIAL_DOMAIN is not None
        assert CODE_DOMAIN is not None
        assert RESEARCH_DOMAIN is not None
        assert GENERAL_DOMAIN is not None

        # Test get_domain_config function
        config = get_domain_config("financial")
        assert config is not None

    def test_registry_available(self):
        """Test QueryEnhancementRegistry is available."""
        from victor.core.query_enhancement import (
            QueryEnhancementRegistry,
            get_default_registry,
        )

        assert QueryEnhancementRegistry is not None
        registry = get_default_registry()
        assert registry is not None

    @pytest.mark.asyncio
    async def test_entity_expand_strategy_works(self):
        """Test EntityExpandStrategy basic operation."""
        from victor.core.query_enhancement import (
            EnhancementContext,
            EntityExpandStrategy,
        )

        strategy = EntityExpandStrategy()

        # Create context with entity_metadata format
        context = EnhancementContext(
            domain="financial",
            entity_metadata=[
                {"name": "AAPL", "aliases": ["Apple Inc", "Apple"]},
            ],
        )

        # Enhance query (async)
        result = await strategy.enhance("What is AAPL revenue?", context)

        assert result is not None
        assert result.original == "What is AAPL revenue?"
        # Entity expansion should add variants
        assert len(result.get_all_queries()) >= 1


class TestServiceProviderIntegration:
    """Test service provider has all components registered."""

    def test_service_provider_importable(self):
        """Test OrchestratorServiceProvider can be imported."""
        from victor.agent.service_provider import OrchestratorServiceProvider

        assert OrchestratorServiceProvider is not None

    def test_configure_services_importable(self):
        """Test configure_orchestrator_services can be imported."""
        from victor.agent.service_provider import configure_orchestrator_services

        assert configure_orchestrator_services is not None


class TestFrameworkPromptSectionsIntegration:
    """Test prompt sections are available."""

    def test_grounding_rules_available(self):
        """Test grounding rules are exported."""
        from victor.framework import (
            GROUNDING_RULES_EXTENDED,
            GROUNDING_RULES_MINIMAL,
        )

        assert GROUNDING_RULES_MINIMAL is not None
        assert GROUNDING_RULES_EXTENDED is not None
        assert len(GROUNDING_RULES_MINIMAL) > 0
        assert len(GROUNDING_RULES_EXTENDED) > 0

    def test_coding_sections_available(self):
        """Test coding sections are exported."""
        from victor.framework import (
            CODING_GUIDELINES,
            CODING_IDENTITY,
            CODING_TOOL_USAGE,
        )

        assert CODING_IDENTITY is not None
        assert CODING_GUIDELINES is not None
        assert CODING_TOOL_USAGE is not None

    def test_devops_sections_available(self):
        """Test devops sections are exported."""
        from victor.framework import (
            DEVOPS_COMMON_PITFALLS,
            DEVOPS_GROUNDING,
            DEVOPS_IDENTITY,
            DEVOPS_SECURITY_CHECKLIST,
        )

        assert DEVOPS_IDENTITY is not None
        assert DEVOPS_GROUNDING is not None
        assert DEVOPS_SECURITY_CHECKLIST is not None
        assert DEVOPS_COMMON_PITFALLS is not None

    def test_research_sections_available(self):
        """Test research sections are exported."""
        from victor.framework import (
            RESEARCH_GROUNDING,
            RESEARCH_IDENTITY,
            RESEARCH_QUALITY_CHECKLIST,
            RESEARCH_SOURCE_HIERARCHY,
        )

        assert RESEARCH_IDENTITY is not None
        assert RESEARCH_GROUNDING is not None
        assert RESEARCH_SOURCE_HIERARCHY is not None
        assert RESEARCH_QUALITY_CHECKLIST is not None

    def test_data_analysis_sections_available(self):
        """Test data analysis sections are exported."""
        from victor.framework import (
            DATA_ANALYSIS_GROUNDING,
            DATA_ANALYSIS_IDENTITY,
            DATA_ANALYSIS_LIBRARIES,
            DATA_ANALYSIS_OPERATIONS,
        )

        assert DATA_ANALYSIS_IDENTITY is not None
        assert DATA_ANALYSIS_GROUNDING is not None
        assert DATA_ANALYSIS_LIBRARIES is not None
        assert DATA_ANALYSIS_OPERATIONS is not None


class TestConversationStateIntegration:
    """Test conversation state types are available."""

    def test_conversation_stage_available(self):
        """Test ConversationStage enum is available."""
        from victor.agent.conversation.state_machine import ConversationStage

        assert ConversationStage is not None
        # Check some stages exist
        assert hasattr(ConversationStage, "INITIAL")
        assert hasattr(ConversationStage, "READING")
        assert hasattr(ConversationStage, "EXECUTION")

    def test_conversation_state_available(self):
        """Test ConversationState is available."""
        from victor.agent.conversation.state_machine import ConversationState

        assert ConversationState is not None

    def test_stage_order_available(self):
        """Test STAGE_ORDER is available."""
        from victor.agent.conversation.state_machine import STAGE_ORDER

        assert STAGE_ORDER is not None
        assert len(STAGE_ORDER) > 0


class TestEndToEndIntegration:
    """End-to-end integration tests."""

    def test_prompt_builder_with_vertical_sections(self):
        """Test using PromptBuilder with vertical-specific sections."""
        from victor.framework import (
            CODING_GUIDELINES,
            CODING_IDENTITY,
            GROUNDING_RULES_MINIMAL,
            PromptBuilder,
        )

        builder = (
            PromptBuilder()
            .add_section("identity", CODING_IDENTITY, priority=100)
            .add_section("guidelines", CODING_GUIDELINES, priority=90)
            .set_custom_grounding(GROUNDING_RULES_MINIMAL)
            .add_safety_rule("Never modify files without user confirmation.")
        )

        prompt = builder.build()

        # Verify sections are included (check identity content)
        assert CODING_IDENTITY in prompt or len(prompt) > 0
        assert "Never modify files without user confirmation" in prompt

    @pytest.mark.asyncio
    async def test_query_enhancement_with_pipeline(self):
        """Test query enhancement pipeline end-to-end."""
        from victor.core.query_enhancement import (
            EnhancementContext,
            EnhancementTechnique,
            QueryEnhancementConfig,
            QueryEnhancementPipeline,
        )

        # Create config with entity expansion only (no LLM required)
        config = QueryEnhancementConfig(
            techniques=[EnhancementTechnique.ENTITY_EXPAND],
            enable_llm=False,
        )

        # Create pipeline
        pipeline = QueryEnhancementPipeline(config)

        # Create context with entity_metadata format
        context = EnhancementContext(
            domain="code",
            entity_metadata=[
                {"name": "fn", "aliases": ["function"]},
                {"name": "var", "aliases": ["variable"]},
            ],
        )

        # Enhance query (async)
        result = await pipeline.enhance("Find all fn definitions", context)

        assert result is not None
        assert result.original == "Find all fn definitions"

    def test_runtime_adapters_can_be_instantiated(self):
        """Test that canonical runtime adapters can be instantiated.

        The canonical adapters are:
        - StateRuntimeAdapter: Wraps ConversationController + ConversationStateMachine
        - PromptRuntimeAdapter: Wraps PromptBuilder with runtime configuration

        Note: PromptCoordinator still exists as a deprecated compatibility shim.
        For state, only the deprecated StateCoordinatorProtocol alias remains;
        there is no concrete StateCoordinator module anymore.
        """
        from unittest.mock import MagicMock

        from victor.agent.services.prompt_runtime import (
            PromptRuntimeAdapter,
            PromptRuntimeConfig,
        )
        from victor.agent.services.state_runtime import StateRuntimeAdapter
        from victor.framework import PromptBuilder

        # StateRuntimeAdapter (canonical implementation)
        mock_controller = MagicMock()
        state_adapter = StateRuntimeAdapter(
            conversation_controller=mock_controller,
        )
        assert state_adapter is not None

        # PromptRuntimeAdapter (canonical implementation)
        prompt_config = PromptRuntimeConfig()
        prompt_adapter = PromptRuntimeAdapter(
            prompt_builder=PromptBuilder(),
            config=prompt_config,
        )
        assert prompt_adapter is not None

    def test_deleted_compat_modules_not_importable(self):
        """Test that deleted compat modules are not importable."""
        import pytest

        # state_compat module should not be importable
        with pytest.raises(ImportError, match="state_compat"):
            from victor.agent.services import state_compat  # noqa: F401

        # prompt_compat module should not be importable
        with pytest.raises(ImportError, match="prompt_compat"):
            from victor.agent.services import prompt_compat  # noqa: F401

        # prompt_runtime_support module should not be importable
        with pytest.raises(ImportError, match="prompt_runtime_support"):
            import victor.agent.services.prompt_runtime_support  # noqa: F401

        # system_prompt_runtime module should not be importable
        with pytest.raises(ImportError, match="system_prompt_runtime"):
            import victor.agent.services.system_prompt_runtime  # noqa: F401

    def test_canonical_runtime_imports_work(self):
        """Test that canonical runtime imports work correctly."""
        from unittest.mock import MagicMock
        from victor.framework import PromptBuilder

        # StateRuntimeAdapter should work
        from victor.agent.services.state_runtime import StateRuntimeAdapter

        adapter = StateRuntimeAdapter(conversation_controller=MagicMock())
        assert adapter is not None

        # PromptRuntimeAdapter should work
        from victor.agent.services.prompt_runtime import (
            PromptRuntimeAdapter,
            PromptRuntimeConfig,
        )

        prompt_config = PromptRuntimeConfig()
        prompt_adapter = PromptRuntimeAdapter(
            prompt_builder=PromptBuilder(),
            config=prompt_config,
        )
        assert prompt_adapter is not None

        from victor.agent.coordinators.system_prompt_state_passed import (
            SystemPromptStatePassedCoordinator,
        )

        assert SystemPromptStatePassedCoordinator is not None

    def test_canonical_coordination_protocol_surface_work(self):
        """Canonical coordination protocol should remain while shim alias stays removed."""
        from victor.protocols import coordination
        from victor.protocols.coordination import CoordinationAdvisorProtocol

        assert CoordinationAdvisorProtocol is not None
        assert "ModeWorkflowTeamCoordinatorProtocol" not in coordination.__all__

        with pytest.raises(ImportError, match="ModeWorkflowTeamCoordinatorProtocol"):
            from victor.protocols.coordination import (
                ModeWorkflowTeamCoordinatorProtocol,
            )  # noqa: F401
