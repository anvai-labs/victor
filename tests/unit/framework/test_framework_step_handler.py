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

"""Unit tests for FrameworkStepHandler apply methods."""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from typing import Dict, Any, List, Optional
from types import SimpleNamespace

from victor.core.verticals.base import VerticalBase
from victor.core.verticals.import_resolver import import_module_with_fallback
from victor.agent.vertical_context import VerticalContext


def _load_vertical_attr(module_path: str, attr_name: str):
    """Load vertical attribute using external-first import fallbacks."""
    module, _resolved = import_module_with_fallback(module_path)
    if module is None or not hasattr(module, attr_name):
        raise ImportError(f"Unable to resolve {module_path}:{attr_name}")
    return getattr(module, attr_name)


class MockVerticalWithHandlers(VerticalBase):
    """Mock vertical that provides handlers."""

    name = "test_vertical"

    @classmethod
    def get_tools(cls) -> List[str]:
        return ["read"]

    @classmethod
    def get_system_prompt(cls) -> str:
        return "Test"

    @classmethod
    def get_handlers(cls) -> Dict[str, Any]:
        return {
            "test_handler": MagicMock(),
            "another_handler": MagicMock(),
        }


class MockVerticalWithToolGraph(VerticalBase):
    """Mock vertical that provides tool graph."""

    name = "test_graph_vertical"

    @classmethod
    def get_tools(cls) -> List[str]:
        return ["read"]

    @classmethod
    def get_system_prompt(cls) -> str:
        return "Test"

    @classmethod
    def get_tool_graph(cls) -> Any:
        return MagicMock(name="MockToolGraph")


class MockVerticalWithWorkflows(VerticalBase):
    """Mock vertical that provides workflows with auto triggers."""

    name = "test_workflow_vertical"

    @classmethod
    def get_tools(cls) -> List[str]:
        return ["read"]

    @classmethod
    def get_system_prompt(cls) -> str:
        return "Test"

    @classmethod
    def get_workflow_provider(cls) -> Any:
        provider = MagicMock()
        provider.get_workflows.return_value = {"workflow1": MagicMock()}
        provider.get_auto_workflows.return_value = [
            (r"test.*pattern", "workflow1"),
        ]
        return provider


class MockVerticalWithTeams(VerticalBase):
    """Mock vertical that provides team specs."""

    name = "test_team_vertical"

    @classmethod
    def get_tools(cls) -> List[str]:
        return ["read"]

    @classmethod
    def get_system_prompt(cls) -> str:
        return "Test"

    @classmethod
    def get_team_specs(cls) -> Dict[str, Any]:
        return {
            "team1": MagicMock(),
            "team2": MagicMock(),
        }


class MockVerticalWithServiceProvider(VerticalBase):
    """Mock vertical that exposes a service provider extension only."""

    name = "test_service_vertical"

    @classmethod
    def get_tools(cls) -> List[str]:
        return ["read"]

    @classmethod
    def get_system_prompt(cls) -> str:
        return "Test"

    @classmethod
    def get_extensions(cls) -> Any:
        service_provider = MagicMock()
        service_provider.get_required_services.return_value = ["svc1"]
        service_provider.get_optional_services.return_value = ["svc2"]
        return SimpleNamespace(
            service_provider=service_provider,
            middleware=None,
            safety_extensions=None,
            prompt_contributors=None,
            mode_config_provider=None,
            tool_dependency_provider=None,
            enrichment_strategy=None,
            tool_selection_strategy=None,
        )


class MockVerticalWithCapabilityProvider(VerticalBase):
    """Mock vertical that provides dynamic capabilities."""

    name = "test_cap_provider_vertical"

    @classmethod
    def get_tools(cls) -> List[str]:
        return ["read"]

    @classmethod
    def get_system_prompt(cls) -> str:
        return "Test"

    @classmethod
    def get_capability_provider(cls) -> Any:
        cap = SimpleNamespace(
            name="custom_capability",
            handler=lambda *_args, **_kwargs: None,
            capability_type=None,
            version="1.0",
        )
        provider = MagicMock()
        provider.get_capabilities.return_value = [cap]
        return provider


class MockVerticalWithCapabilityConfigs(VerticalBase):
    """Mock vertical that exposes centralized capability config defaults."""

    name = "test_cap_config_vertical"

    @classmethod
    def get_tools(cls) -> List[str]:
        return ["read"]

    @classmethod
    def get_system_prompt(cls) -> str:
        return "Test"

    @classmethod
    def get_capability_configs(cls) -> Dict[str, Any]:
        return {
            "source_verification_config": {"min_credibility": 0.8},
            "citation_config": {"default_style": "apa"},
        }


@pytest.fixture
def mock_orchestrator():
    """Create a mock orchestrator."""
    orchestrator = MagicMock()
    orchestrator.set_team_specs = MagicMock()
    return orchestrator


@pytest.fixture
def mock_context():
    """Create a mock vertical context."""
    context = MagicMock(spec=VerticalContext)
    context.apply_workflows = MagicMock()
    context.apply_team_specs = MagicMock()
    return context


@pytest.fixture
def mock_result():
    """Create a mock integration result."""
    result = MagicMock()
    result.add_info = MagicMock()
    result.add_warning = MagicMock()
    result.workflows_count = 0
    result.team_specs_count = 0
    return result


@pytest.fixture
def reset_registries():
    """Reset all registries before and after test."""
    from victor.framework.handler_registry import HandlerRegistry

    if hasattr(HandlerRegistry, "reset_instance"):
        HandlerRegistry.reset_instance()
    else:
        HandlerRegistry._instance = None
    yield
    if hasattr(HandlerRegistry, "reset_instance"):
        HandlerRegistry.reset_instance()
    else:
        HandlerRegistry._instance = None


class TestApplyHandlers:
    """Tests for apply_handlers method."""

    def test_apply_handlers_registers_to_registry(
        self, mock_orchestrator, mock_context, mock_result, reset_registries
    ):
        """Test apply_handlers registers handlers to HandlerRegistry."""
        from victor.framework.step_handlers import FrameworkStepHandler
        from victor.framework.handler_registry import get_handler_registry

        handler = FrameworkStepHandler()
        handler.apply_handlers(
            mock_orchestrator,
            MockVerticalWithHandlers,
            mock_context,
            mock_result,
        )

        registry = get_handler_registry()
        assert registry.get_handler("test_vertical", "test_handler") is not None
        assert registry.get_handler("test_vertical", "another_handler") is not None
        listed = registry.list_handlers("test_vertical")
        assert "test_handler" in listed.get("test_vertical", [])
        assert "another_handler" in listed.get("test_vertical", [])

    def test_apply_handlers_adds_info_to_result(
        self, mock_orchestrator, mock_context, mock_result, reset_registries
    ):
        """Test apply_handlers adds info message to result."""
        from victor.framework.step_handlers import FrameworkStepHandler

        handler = FrameworkStepHandler()
        handler.apply_handlers(
            mock_orchestrator,
            MockVerticalWithHandlers,
            mock_context,
            mock_result,
        )

        mock_result.add_info.assert_called()
        call_args = str(mock_result.add_info.call_args)
        assert "handler" in call_args.lower()

    def test_apply_handlers_skips_vertical_without_handlers(
        self, mock_orchestrator, mock_context, mock_result, reset_registries
    ):
        """Test apply_handlers skips vertical without get_handlers."""
        from victor.framework.step_handlers import FrameworkStepHandler

        # Create a minimal vertical without get_handlers
        class MinimalVertical(VerticalBase):
            name = "minimal"

            @classmethod
            def get_tools(cls):
                return ["read"]

            @classmethod
            def get_system_prompt(cls):
                return "Test"

        handler = FrameworkStepHandler()
        # Should not raise
        handler.apply_handlers(
            mock_orchestrator,
            MinimalVertical,
            mock_context,
            mock_result,
        )


class TestApplyToolGraphs:
    """Tests for apply_tool_graphs method."""

    def test_apply_tool_graphs_registers_to_registry(
        self, mock_orchestrator, mock_context, mock_result
    ):
        """Test apply_tool_graphs registers graph to ToolGraphRegistry."""
        from victor.framework.step_handlers import FrameworkStepHandler

        handler = FrameworkStepHandler()

        with patch("victor.tools.tool_graph.ToolGraphRegistry") as mock_registry_class:
            mock_registry = MagicMock()
            mock_registry_class.get_instance.return_value = mock_registry

            handler.apply_tool_graphs(
                mock_orchestrator,
                MockVerticalWithToolGraph,
                mock_context,
                mock_result,
            )

            mock_registry.register_graph.assert_called_once()
            call_args = mock_registry.register_graph.call_args
            assert call_args[0][0] == "test_graph_vertical"

    def test_apply_tool_graphs_skips_vertical_without_graph(
        self, mock_orchestrator, mock_context, mock_result
    ):
        """Test apply_tool_graphs skips vertical without tool graph."""
        from victor.framework.step_handlers import FrameworkStepHandler

        # VerticalBase returns None for get_tool_graph
        handler = FrameworkStepHandler()

        with patch("victor.tools.tool_graph.ToolGraphRegistry") as mock_registry_class:
            mock_registry = MagicMock()
            mock_registry_class.get_instance.return_value = mock_registry

            handler.apply_tool_graphs(
                mock_orchestrator,
                VerticalBase,  # Use base class which returns None
                mock_context,
                mock_result,
            )

            # Should not register since get_tool_graph returns None
            mock_registry.register_graph.assert_not_called()


class TestApplyWorkflowsWithTriggers:
    """Tests for workflow trigger registration in apply_workflows."""

    def test_apply_workflows_registers_triggers(self, mock_orchestrator, mock_context, mock_result):
        """Test apply_workflows registers triggers with WorkflowTriggerRegistry."""
        from victor.framework.step_handlers import FrameworkStepHandler

        handler = FrameworkStepHandler()

        # Use create=True since get_workflow_registry may not exist in the module
        with patch(
            "victor.workflows.registry.get_workflow_registry",
            create=True,
        ) as mock_wf_registry:
            mock_wf_registry.return_value = MagicMock()

            with patch(
                "victor.workflows.trigger_registry.get_trigger_registry",
                create=True,
            ) as mock_trigger:
                mock_trigger_registry = MagicMock()
                mock_trigger.return_value = mock_trigger_registry

                handler.apply_workflows(
                    mock_orchestrator,
                    MockVerticalWithWorkflows,
                    mock_context,
                    mock_result,
                )

                # Verify triggers were registered
                mock_trigger_registry.register_from_vertical.assert_called_once()
                call_args = mock_trigger_registry.register_from_vertical.call_args
                assert call_args[0][0] == "test_workflow_vertical"

    def test_apply_workflows_uses_framework_registry_service(
        self, mock_orchestrator, mock_context, mock_result
    ):
        """Workflow registration should route through framework registry service."""
        from victor.framework.step_handlers import FrameworkStepHandler

        handler = FrameworkStepHandler()
        mock_service = MagicMock()

        with patch(
            "victor.framework.step_handlers.resolve_framework_integration_registry_service",
            return_value=mock_service,
        ):
            handler.apply_workflows(
                mock_orchestrator,
                MockVerticalWithWorkflows,
                mock_context,
                mock_result,
            )

        mock_service.register_workflows.assert_called_once()
        mock_service.register_workflow_triggers.assert_called_once()
        wf_kwargs = mock_service.register_workflows.call_args.kwargs
        trigger_kwargs = mock_service.register_workflow_triggers.call_args.kwargs
        assert "registration_version" in wf_kwargs
        assert wf_kwargs["registration_version"] is None
        assert "registration_version" in trigger_kwargs
        assert trigger_kwargs["registration_version"] is None

    def test_apply_workflows_passes_vertical_version_to_registry_service(
        self, mock_orchestrator, mock_context, mock_result
    ):
        """Workflow registration should include explicit vertical version token."""
        from victor.framework.step_handlers import FrameworkStepHandler

        class VersionedWorkflowVertical(MockVerticalWithWorkflows):
            version = "2.4.0"

        handler = FrameworkStepHandler()
        mock_service = MagicMock()

        with patch(
            "victor.framework.step_handlers.resolve_framework_integration_registry_service",
            return_value=mock_service,
        ):
            registration_version = handler._resolve_registration_version(
                VersionedWorkflowVertical,
                mock_context,
            )
            handler.apply_workflows(
                mock_orchestrator,
                VersionedWorkflowVertical,
                mock_context,
                mock_result,
                registration_version=registration_version,
            )

        wf_kwargs = mock_service.register_workflows.call_args.kwargs
        trigger_kwargs = mock_service.register_workflow_triggers.call_args.kwargs
        assert wf_kwargs["registration_version"] == "2.4.0"
        assert trigger_kwargs["registration_version"] == "2.4.0"


class TestApplyTeamSpecsWithRegistry:
    """Tests for team spec registry registration in apply_team_specs."""

    @pytest.mark.skip(
        reason="Vertical discovery system tries to import victor.test_team_vertical which doesn't exist"
    )
    def test_apply_team_specs_registers_to_global_registry(
        self, mock_orchestrator, mock_context, mock_result
    ):
        """Test apply_team_specs registers to TeamSpecRegistry."""
        from victor.framework.step_handlers import FrameworkStepHandler

        handler = FrameworkStepHandler()

        # Use create=True since get_team_registry may not exist
        with patch(
            "victor.framework.team_registry.get_team_registry",
            create=True,
        ) as mock_team:
            mock_team_registry = MagicMock()
            mock_team.return_value = mock_team_registry

            handler.apply_team_specs(
                mock_orchestrator,
                MockVerticalWithTeams,
                mock_context,
                mock_result,
            )

            # Verify teams were registered with global registry
            mock_team_registry.register_from_vertical.assert_called_once()
            call_args = mock_team_registry.register_from_vertical.call_args
            assert call_args[0][0] == "test_team_vertical"


class TestServiceProviderRegistrationPorts:
    """Tests for service-provider registration through public orchestrator ports."""

    def test_extensions_step_handler_uses_activation_helper(self):
        """Service provider registration should route through activate_vertical_services."""
        from victor.framework.step_handlers import ExtensionsStepHandler

        handler = ExtensionsStepHandler()
        orchestrator = MagicMock()
        container = MagicMock()
        settings = MagicMock()
        context = MagicMock(spec=VerticalContext)
        context.vertical_name = "test_service_vertical"
        result = MagicMock()
        result.add_info = MagicMock()
        result.add_warning = MagicMock()

        orchestrator.get_service_container.return_value = container
        orchestrator.settings = settings

        activation = SimpleNamespace(services_registered=True)
        with patch(
            "victor.core.verticals.vertical_loader.activate_vertical_services",
            return_value=activation,
        ) as mock_activate:
            handler._do_apply(orchestrator, MockVerticalWithServiceProvider, context, result)

        mock_activate.assert_called_once_with(container, settings, "test_service_vertical")

    def test_extensions_step_handler_warns_without_public_container_port(self):
        """Service provider registration should fail fast without container access port."""
        from victor.framework.step_handlers import ExtensionsStepHandler

        handler = ExtensionsStepHandler()
        context = MagicMock(spec=VerticalContext)
        context.vertical_name = "test_service_vertical"
        result = MagicMock()
        result.add_info = MagicMock()
        result.add_warning = MagicMock()

        class NoContainerOrchestrator:
            settings = MagicMock()

        handler._do_apply(
            NoContainerOrchestrator(), MockVerticalWithServiceProvider, context, result
        )
        assert result.add_warning.call_count >= 1


class TestCapabilityProviderPorts:
    """Tests for capability provider wiring through loader ports."""

    def test_apply_capability_provider_uses_public_loader_port(self):
        """FrameworkStepHandler should use get_or_create_capability_loader when available."""
        from victor.framework.step_handlers import FrameworkStepHandler

        handler = FrameworkStepHandler()
        orchestrator = MagicMock()
        loader = MagicMock()
        context = MagicMock(spec=VerticalContext)
        result = MagicMock()

        orchestrator.get_or_create_capability_loader.return_value = loader

        handler.apply_capability_provider(
            orchestrator,
            MockVerticalWithCapabilityProvider,
            context,
            result,
        )

        loader.register_capability.assert_called_once()
        loader.apply_to.assert_called_once_with(orchestrator)

    def test_apply_capability_provider_warns_without_loader_port(self):
        """FrameworkStepHandler should warn if capability-loader port is missing."""
        from victor.framework.step_handlers import FrameworkStepHandler

        handler = FrameworkStepHandler()
        context = MagicMock(spec=VerticalContext)
        result = MagicMock()
        result.add_warning = MagicMock()

        class NoLoaderPortOrchestrator:
            pass

        handler.apply_capability_provider(
            NoLoaderPortOrchestrator(),
            MockVerticalWithCapabilityProvider,
            context,
            result,
        )

        result.add_warning.assert_any_call(
            "Cannot wire capability provider: orchestrator lacks capability-loader port"
        )


class TestCapabilityConfigPersistence:
    """Tests for capability-config persistence in framework service."""

    def test_capability_config_step_persists_to_framework_service(self):
        """CapabilityConfigStepHandler should persist defaults in CapabilityConfigService."""
        from victor.framework.capability_config_service import CapabilityConfigService
        from victor.framework.step_handlers import CapabilityConfigStepHandler

        class StubContainer:
            def __init__(self) -> None:
                self._services = {}

            def get_optional(self, service_type):
                return self._services.get(service_type)

            def register_instance(self, service_type, instance):
                self._services[service_type] = instance

        class StubOrchestrator:
            def __init__(self):
                self._container = StubContainer()

            def get_service_container(self):
                return self._container

        handler = CapabilityConfigStepHandler()
        orchestrator = StubOrchestrator()
        context = MagicMock(spec=VerticalContext)
        context.apply_capability_configs = MagicMock()
        result = MagicMock()
        result.add_info = MagicMock()

        handler._do_apply(orchestrator, MockVerticalWithCapabilityConfigs, context, result)

        service = orchestrator.get_service_container().get_optional(CapabilityConfigService)
        assert service is not None
        assert service.get_config("source_verification_config") == {"min_credibility": 0.8}
        assert service.get_config("citation_config") == {"default_style": "apa"}

    def test_end_to_end_defaults_flow_service_to_runtime_getter(self):
        """Defaults should flow from step handler into runtime capability getter via service."""
        from victor.framework.step_handlers import CapabilityConfigStepHandler

        try:
            get_source_verification = _load_vertical_attr(
                "victor.research.capabilities",
                "get_source_verification",
            )
        except ImportError:
            pytest.skip("victor-research package not installed")

        class StubContainer:
            def __init__(self) -> None:
                self._services = {}

            def get_optional(self, service_type):
                return self._services.get(service_type)

            def register_instance(self, service_type, instance):
                self._services[service_type] = instance

        class StubOrchestrator:
            def __init__(self):
                self._container = StubContainer()

            def get_service_container(self):
                return self._container

        handler = CapabilityConfigStepHandler()
        orchestrator = StubOrchestrator()
        context = MagicMock(spec=VerticalContext)
        context.apply_capability_configs = MagicMock()
        result = MagicMock()
        result.add_info = MagicMock()

        handler._do_apply(orchestrator, MockVerticalWithCapabilityConfigs, context, result)

        source_verification = get_source_verification(orchestrator)
        assert source_verification == {"min_credibility": 0.8}

    def test_capability_config_step_persists_by_scope_key(self):
        """Capability config persistence should isolate writes by orchestrator scope key."""
        from victor.framework.capability_config_service import CapabilityConfigService
        from victor.framework.step_handlers import CapabilityConfigStepHandler

        class ScopeA(VerticalBase):
            name = "scope_a_vertical"

            @classmethod
            def get_tools(cls):
                return ["read"]

            @classmethod
            def get_system_prompt(cls):
                return "Test"

            @classmethod
            def get_capability_configs(cls):
                return {"source_verification_config": {"min_credibility": 0.9}}

        class ScopeB(VerticalBase):
            name = "scope_b_vertical"

            @classmethod
            def get_tools(cls):
                return ["read"]

            @classmethod
            def get_system_prompt(cls):
                return "Test"

            @classmethod
            def get_capability_configs(cls):
                return {"source_verification_config": {"min_credibility": 0.6}}

        class StubContainer:
            def __init__(self, service):
                self._services = {CapabilityConfigService: service}

            def get_optional(self, service_type):
                return self._services.get(service_type)

            def register_instance(self, service_type, instance):
                self._services[service_type] = instance

        class ScopedOrchestrator:
            def __init__(self, service, scope_key):
                self._container = StubContainer(service)
                self._scope_key = scope_key

            def get_service_container(self):
                return self._container

            def get_capability_config_scope_key(self):
                return self._scope_key

        service = CapabilityConfigService()
        handler = CapabilityConfigStepHandler()
        context = MagicMock(spec=VerticalContext)
        context.apply_capability_configs = MagicMock()
        result = MagicMock()
        result.add_info = MagicMock()

        orchestrator_a = ScopedOrchestrator(service, "session-a")
        orchestrator_b = ScopedOrchestrator(service, "session-b")

        handler._do_apply(orchestrator_a, ScopeA, context, result)
        handler._do_apply(orchestrator_b, ScopeB, context, result)

        assert service.get_config(
            "source_verification_config",
            scope_key="session-a",
        ) == {"min_credibility": 0.9}
        assert service.get_config(
            "source_verification_config",
            scope_key="session-b",
        ) == {"min_credibility": 0.6}


class TestStepHandlerRegistryContract:
    """Contract tests for default step-handler registry composition."""

    def test_default_registry_execution_order_contract(self):
        """Default registry should expose deterministic, documented handler order."""
        from victor.framework.step_handlers import StepHandlerRegistry

        registry = StepHandlerRegistry.default()
        ordered_names = [handler.name for handler in registry.get_ordered_handlers()]

        assert ordered_names == [
            "compatibility",
            "capability_config",
            "tools",
            "mcp",
            "tiered_config",
            "prompt",
            "config",
            "extensions",
            "framework",
            "context",
        ]

    def test_default_registry_has_no_unresolved_dependencies(self):
        """All declared default-handler dependencies should resolve within the default registry."""
        from victor.framework.step_handlers import StepHandlerRegistry

        registry = StepHandlerRegistry.default()
        handlers = registry.get_ordered_handlers()
        available = {handler.name for handler in handlers}

        unresolved = {}
        for handler in handlers:
            deps = tuple(getattr(handler, "depends_on", ()) or ())
            missing = [dep for dep in deps if dep not in available]
            if missing:
                unresolved[handler.name] = missing

        assert unresolved == {}
