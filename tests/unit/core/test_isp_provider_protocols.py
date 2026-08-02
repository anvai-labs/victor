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

"""Tests for ISP-Compliant Vertical Provider Protocols.

These tests verify that all provider protocols in providers.py are properly
defined, @runtime_checkable, and can be used for isinstance() checks.
"""

import pytest
from typing import Any, Dict, List, Optional


class TestMiddlewareProvider:
    """Tests for MiddlewareProvider protocol."""

    def test_middleware_provider_is_runtime_checkable(self):
        """MiddlewareProvider must be @runtime_checkable."""
        from victor.core.verticals.protocols.providers import MiddlewareProvider

        class DummyClass:
            pass

        try:
            isinstance(DummyClass(), MiddlewareProvider)
        except TypeError:
            pytest.fail("MiddlewareProvider must be decorated with @runtime_checkable")

    def test_middleware_provider_has_get_middleware(self):
        """MiddlewareProvider must define get_middleware method."""
        from victor.core.verticals.protocols.providers import MiddlewareProvider

        assert hasattr(
            MiddlewareProvider, "get_middleware"
        ), "MiddlewareProvider must define get_middleware method"

    def test_middleware_provider_isinstance_check(self):
        """Verify isinstance() works with MiddlewareProvider."""
        from victor.core.verticals.protocols.providers import MiddlewareProvider

        class ValidProvider:
            @classmethod
            def get_middleware(cls) -> List[Any]:
                return []

        class InvalidProvider:
            pass

        assert isinstance(
            ValidProvider, MiddlewareProvider
        ), "Valid implementation should pass isinstance check"
        assert not isinstance(
            InvalidProvider, MiddlewareProvider
        ), "Invalid implementation should fail isinstance check"


class TestSafetyProvider:
    """Tests for SafetyProvider protocol."""

    def test_safety_provider_is_runtime_checkable(self):
        """SafetyProvider must be @runtime_checkable."""
        from victor.core.verticals.protocols.providers import SafetyProvider

        class DummyClass:
            pass

        try:
            isinstance(DummyClass(), SafetyProvider)
        except TypeError:
            pytest.fail("SafetyProvider must be decorated with @runtime_checkable")

    def test_safety_provider_has_get_safety_extension(self):
        """SafetyProvider must define get_safety_extension method."""
        from victor.core.verticals.protocols.providers import SafetyProvider

        assert hasattr(
            SafetyProvider, "get_safety_extension"
        ), "SafetyProvider must define get_safety_extension method"

    def test_safety_provider_isinstance_check(self):
        """Verify isinstance() works with SafetyProvider."""
        from victor.core.verticals.protocols.providers import SafetyProvider

        class ValidProvider:
            @classmethod
            def get_safety_extension(cls) -> Optional[Any]:
                return None

        class InvalidProvider:
            pass

        assert isinstance(
            ValidProvider, SafetyProvider
        ), "Valid implementation should pass isinstance check"
        assert not isinstance(
            InvalidProvider, SafetyProvider
        ), "Invalid implementation should fail isinstance check"


class TestWorkflowProvider:
    """Tests for WorkflowProvider protocol."""

    def test_workflow_provider_is_runtime_checkable(self):
        """WorkflowProvider must be @runtime_checkable."""
        from victor.core.verticals.protocols.providers import WorkflowProvider

        class DummyClass:
            pass

        try:
            isinstance(DummyClass(), WorkflowProvider)
        except TypeError:
            pytest.fail("WorkflowProvider must be decorated with @runtime_checkable")

    def test_workflow_provider_has_required_methods(self):
        """WorkflowProvider must define required methods."""
        from victor.core.verticals.protocols.providers import WorkflowProvider

        assert hasattr(
            WorkflowProvider, "get_workflow_provider"
        ), "WorkflowProvider must define get_workflow_provider method"
        assert hasattr(
            WorkflowProvider, "get_workflows"
        ), "WorkflowProvider must define get_workflows method"

    def test_workflow_provider_isinstance_check(self):
        """Verify isinstance() works with WorkflowProvider."""
        from victor.core.verticals.protocols.providers import WorkflowProvider

        class ValidProvider:
            @classmethod
            def get_workflow_provider(cls) -> Optional[Any]:
                return None

            @classmethod
            def get_workflows(cls) -> Dict[str, Any]:
                return {}

        class InvalidProvider:
            pass

        assert isinstance(
            ValidProvider, WorkflowProvider
        ), "Valid implementation should pass isinstance check"
        assert not isinstance(
            InvalidProvider, WorkflowProvider
        ), "Invalid implementation should fail isinstance check"


class TestTeamProvider:
    """Tests for TeamProvider protocol."""

    def test_team_provider_is_runtime_checkable(self):
        """TeamProvider must be @runtime_checkable."""
        from victor.core.verticals.protocols.providers import TeamProvider

        class DummyClass:
            pass

        try:
            isinstance(DummyClass(), TeamProvider)
        except TypeError:
            pytest.fail("TeamProvider must be decorated with @runtime_checkable")

    def test_team_provider_has_required_methods(self):
        """TeamProvider must define required methods."""
        from victor.core.verticals.protocols.providers import TeamProvider

        assert hasattr(
            TeamProvider, "get_team_spec_provider"
        ), "TeamProvider must define get_team_spec_provider method"
        assert hasattr(
            TeamProvider, "get_team_specs"
        ), "TeamProvider must define get_team_specs method"

    def test_team_provider_isinstance_check(self):
        """Verify isinstance() works with TeamProvider."""
        from victor.core.verticals.protocols.providers import TeamProvider

        class ValidProvider:
            @classmethod
            def get_team_spec_provider(cls) -> Optional[Any]:
                return None

            @classmethod
            def get_team_specs(cls) -> Dict[str, Any]:
                return {}

        class InvalidProvider:
            pass

        assert isinstance(
            ValidProvider, TeamProvider
        ), "Valid implementation should pass isinstance check"
        assert not isinstance(
            InvalidProvider, TeamProvider
        ), "Invalid implementation should fail isinstance check"


class TestRLProvider:
    """Tests for RLProvider protocol."""

    def test_rl_provider_is_runtime_checkable(self):
        """RLProvider must be @runtime_checkable."""
        from victor.core.verticals.protocols.providers import RLProvider

        class DummyClass:
            pass

        try:
            isinstance(DummyClass(), RLProvider)
        except TypeError:
            pytest.fail("RLProvider must be decorated with @runtime_checkable")

    def test_rl_provider_has_required_methods(self):
        """RLProvider must define required methods."""
        from victor.core.verticals.protocols.providers import RLProvider

        assert hasattr(
            RLProvider, "get_rl_config_provider"
        ), "RLProvider must define get_rl_config_provider method"
        assert hasattr(RLProvider, "get_rl_hooks"), "RLProvider must define get_rl_hooks method"

    def test_rl_provider_isinstance_check(self):
        """Verify isinstance() works with RLProvider."""
        from victor.core.verticals.protocols.providers import RLProvider

        class ValidProvider:
            @classmethod
            def get_rl_config_provider(cls) -> Optional[Any]:
                return None

            @classmethod
            def get_rl_hooks(cls) -> List[Any]:
                return []

        class InvalidProvider:
            pass

        assert isinstance(
            ValidProvider, RLProvider
        ), "Valid implementation should pass isinstance check"
        assert not isinstance(
            InvalidProvider, RLProvider
        ), "Invalid implementation should fail isinstance check"


class TestEnrichmentProvider:
    """Tests for EnrichmentProvider protocol."""

    def test_enrichment_provider_is_runtime_checkable(self):
        """EnrichmentProvider must be @runtime_checkable."""
        from victor.core.verticals.protocols.providers import EnrichmentProvider

        class DummyClass:
            pass

        try:
            isinstance(DummyClass(), EnrichmentProvider)
        except TypeError:
            pytest.fail("EnrichmentProvider must be decorated with @runtime_checkable")

    def test_enrichment_provider_has_get_enrichment_strategy(self):
        """EnrichmentProvider must define get_enrichment_strategy method."""
        from victor.core.verticals.protocols.providers import EnrichmentProvider

        assert hasattr(
            EnrichmentProvider, "get_enrichment_strategy"
        ), "EnrichmentProvider must define get_enrichment_strategy method"

    def test_enrichment_provider_isinstance_check(self):
        """Verify isinstance() works with EnrichmentProvider."""
        from victor.core.verticals.protocols.providers import EnrichmentProvider

        class ValidProvider:
            @classmethod
            def get_enrichment_strategy(cls) -> Optional[Any]:
                return None

        class InvalidProvider:
            pass

        assert isinstance(
            ValidProvider, EnrichmentProvider
        ), "Valid implementation should pass isinstance check"
        assert not isinstance(
            InvalidProvider, EnrichmentProvider
        ), "Invalid implementation should fail isinstance check"


class TestToolProvider:
    """Tests for ToolProvider protocol."""

    def test_tool_provider_is_runtime_checkable(self):
        """ToolProvider must be @runtime_checkable."""
        from victor.core.verticals.protocols.providers import ToolProvider

        class DummyClass:
            pass

        try:
            isinstance(DummyClass(), ToolProvider)
        except TypeError:
            pytest.fail("ToolProvider must be decorated with @runtime_checkable")

    def test_tool_provider_has_required_methods(self):
        """ToolProvider must define required methods."""
        from victor.core.verticals.protocols.providers import ToolProvider

        assert hasattr(ToolProvider, "get_tools"), "ToolProvider must define get_tools method"
        assert hasattr(
            ToolProvider, "get_tool_graph"
        ), "ToolProvider must define get_tool_graph method"

    def test_tool_provider_isinstance_check(self):
        """Verify isinstance() works with ToolProvider."""
        from victor.core.verticals.protocols.providers import ToolProvider

        class ValidProvider:
            @classmethod
            def get_tools(cls) -> List[str]:
                return ["read", "write"]

            @classmethod
            def get_tool_graph(cls) -> Optional[Any]:
                return None

        class InvalidProvider:
            pass

        assert isinstance(
            ValidProvider, ToolProvider
        ), "Valid implementation should pass isinstance check"
        assert not isinstance(
            InvalidProvider, ToolProvider
        ), "Invalid implementation should fail isinstance check"


class TestHandlerProvider:
    """Tests for HandlerProvider protocol."""

    def test_handler_provider_is_runtime_checkable(self):
        """HandlerProvider must be @runtime_checkable."""
        from victor.core.verticals.protocols.providers import HandlerProvider

        class DummyClass:
            pass

        try:
            isinstance(DummyClass(), HandlerProvider)
        except TypeError:
            pytest.fail("HandlerProvider must be decorated with @runtime_checkable")

    def test_handler_provider_has_get_handlers(self):
        """HandlerProvider must define get_handlers method."""
        from victor.core.verticals.protocols.providers import HandlerProvider

        assert hasattr(
            HandlerProvider, "get_handlers"
        ), "HandlerProvider must define get_handlers method"

    def test_handler_provider_isinstance_check(self):
        """Verify isinstance() works with HandlerProvider."""
        from victor.core.verticals.protocols.providers import HandlerProvider

        class ValidProvider:
            @classmethod
            def get_handlers(cls) -> Dict[str, Any]:
                return {}

        class InvalidProvider:
            pass

        assert isinstance(
            ValidProvider, HandlerProvider
        ), "Valid implementation should pass isinstance check"
        assert not isinstance(
            InvalidProvider, HandlerProvider
        ), "Invalid implementation should fail isinstance check"


class TestCapabilityProvider:
    """Tests for CapabilityProvider protocol."""

    def test_capability_provider_is_runtime_checkable(self):
        """CapabilityProvider must be @runtime_checkable."""
        from victor.core.verticals.protocols.providers import CapabilityProvider

        class DummyClass:
            pass

        try:
            isinstance(DummyClass(), CapabilityProvider)
        except TypeError:
            pytest.fail("CapabilityProvider must be decorated with @runtime_checkable")

    def test_capability_provider_has_get_capability_provider(self):
        """CapabilityProvider must define get_capability_provider method."""
        from victor.core.verticals.protocols.providers import CapabilityProvider

        assert hasattr(
            CapabilityProvider, "get_capability_provider"
        ), "CapabilityProvider must define get_capability_provider method"

    def test_capability_provider_isinstance_check(self):
        """Verify isinstance() works with CapabilityProvider."""
        from victor.core.verticals.protocols.providers import CapabilityProvider

        class ValidProvider:
            @classmethod
            def get_capability_provider(cls) -> Optional[Any]:
                return None

        class InvalidProvider:
            pass

        assert isinstance(
            ValidProvider, CapabilityProvider
        ), "Valid implementation should pass isinstance check"
        assert not isinstance(
            InvalidProvider, CapabilityProvider
        ), "Invalid implementation should fail isinstance check"


class TestModeConfigProvider:
    """Tests for ModeConfigProvider protocol."""

    def test_mode_config_provider_is_runtime_checkable(self):
        """ModeConfigProvider must be @runtime_checkable."""
        from victor.core.verticals.protocols.providers import ModeConfigProvider

        class DummyClass:
            pass

        try:
            isinstance(DummyClass(), ModeConfigProvider)
        except TypeError:
            pytest.fail("ModeConfigProvider must be decorated with @runtime_checkable")

    def test_mode_config_provider_has_required_methods(self):
        """ModeConfigProvider must define required methods."""
        from victor.core.verticals.protocols.providers import ModeConfigProvider

        assert hasattr(
            ModeConfigProvider, "get_mode_config_provider"
        ), "ModeConfigProvider must define get_mode_config_provider method"
        assert hasattr(
            ModeConfigProvider, "get_mode_config"
        ), "ModeConfigProvider must define get_mode_config method"

    def test_mode_config_provider_isinstance_check(self):
        """Verify isinstance() works with ModeConfigProvider."""
        from victor.core.verticals.protocols.providers import ModeConfigProvider

        class ValidProvider:
            @classmethod
            def get_mode_config_provider(cls) -> Optional[Any]:
                return None

            @classmethod
            def get_mode_config(cls) -> Dict[str, Any]:
                return {}

        class InvalidProvider:
            pass

        assert isinstance(
            ValidProvider, ModeConfigProvider
        ), "Valid implementation should pass isinstance check"
        assert not isinstance(
            InvalidProvider, ModeConfigProvider
        ), "Invalid implementation should fail isinstance check"


class TestPromptContributorProvider:
    """Tests for PromptContributorProvider protocol."""

    def test_prompt_contributor_provider_is_runtime_checkable(self):
        """PromptContributorProvider must be @runtime_checkable."""
        from victor.core.verticals.protocols.providers import PromptContributorProvider

        class DummyClass:
            pass

        try:
            isinstance(DummyClass(), PromptContributorProvider)
        except TypeError:
            pytest.fail("PromptContributorProvider must be decorated with @runtime_checkable")

    def test_prompt_contributor_provider_has_required_methods(self):
        """PromptContributorProvider must define required methods."""
        from victor.core.verticals.protocols.providers import PromptContributorProvider

        assert hasattr(
            PromptContributorProvider, "get_prompt_contributor"
        ), "PromptContributorProvider must define get_prompt_contributor method"
        assert hasattr(
            PromptContributorProvider, "get_task_type_hints"
        ), "PromptContributorProvider must define get_task_type_hints method"

    def test_prompt_contributor_provider_isinstance_check(self):
        """Verify isinstance() works with PromptContributorProvider."""
        from victor.core.verticals.protocols.providers import PromptContributorProvider

        class ValidProvider:
            @classmethod
            def get_prompt_contributor(cls) -> Optional[Any]:
                return None

            @classmethod
            def get_task_type_hints(cls) -> Dict[str, Any]:
                return {}

        class InvalidProvider:
            pass

        assert isinstance(
            ValidProvider, PromptContributorProvider
        ), "Valid implementation should pass isinstance check"
        assert not isinstance(
            InvalidProvider, PromptContributorProvider
        ), "Invalid implementation should fail isinstance check"


class TestToolDependencyProvider:
    """Tests for ToolDependencyProvider protocol."""

    def test_tool_dependency_provider_is_runtime_checkable(self):
        """ToolDependencyProvider must be @runtime_checkable."""
        from victor.core.verticals.protocols.providers import ToolDependencyProvider

        class DummyClass:
            pass

        try:
            isinstance(DummyClass(), ToolDependencyProvider)
        except TypeError:
            pytest.fail("ToolDependencyProvider must be decorated with @runtime_checkable")

    def test_tool_dependency_provider_has_required_method(self):
        """ToolDependencyProvider must define get_tool_dependency_provider method."""
        from victor.core.verticals.protocols.providers import ToolDependencyProvider

        assert hasattr(
            ToolDependencyProvider, "get_tool_dependency_provider"
        ), "ToolDependencyProvider must define get_tool_dependency_provider method"

    def test_tool_dependency_provider_isinstance_check(self):
        """Verify isinstance() works with ToolDependencyProvider."""
        from victor.core.verticals.protocols.providers import ToolDependencyProvider

        class ValidProvider:
            @classmethod
            def get_tool_dependency_provider(cls) -> Optional[Any]:
                return None

        class InvalidProvider:
            pass

        assert isinstance(
            ValidProvider, ToolDependencyProvider
        ), "Valid implementation should pass isinstance check"
        assert not isinstance(
            InvalidProvider, ToolDependencyProvider
        ), "Invalid implementation should fail isinstance check"


class TestTieredToolConfigProvider:
    """Tests for TieredToolConfigProvider protocol."""

    def test_tiered_tool_config_provider_is_runtime_checkable(self):
        """TieredToolConfigProvider must be @runtime_checkable."""
        from victor.core.verticals.protocols.providers import TieredToolConfigProvider

        class DummyClass:
            pass

        try:
            isinstance(DummyClass(), TieredToolConfigProvider)
        except TypeError:
            pytest.fail("TieredToolConfigProvider must be decorated with @runtime_checkable")

    def test_tiered_tool_config_provider_has_required_method(self):
        """TieredToolConfigProvider must define get_tiered_tool_config method."""
        from victor.core.verticals.protocols.providers import TieredToolConfigProvider

        assert hasattr(
            TieredToolConfigProvider, "get_tiered_tool_config"
        ), "TieredToolConfigProvider must define get_tiered_tool_config method"

    def test_tiered_tool_config_provider_isinstance_check(self):
        """Verify isinstance() works with TieredToolConfigProvider."""
        from victor.core.verticals.protocols.providers import TieredToolConfigProvider

        class ValidProvider:
            @classmethod
            def get_tiered_tool_config(cls) -> Optional[Any]:
                return None

        class InvalidProvider:
            pass

        assert isinstance(
            ValidProvider, TieredToolConfigProvider
        ), "Valid implementation should pass isinstance check"
        assert not isinstance(
            InvalidProvider, TieredToolConfigProvider
        ), "Invalid implementation should fail isinstance check"


class TestServiceProvider:
    """Tests for ServiceProvider protocol."""

    def test_service_provider_is_runtime_checkable(self):
        """ServiceProvider must be @runtime_checkable."""
        from victor.core.verticals.protocols.providers import ServiceProvider

        class DummyClass:
            pass

        try:
            isinstance(DummyClass(), ServiceProvider)
        except TypeError:
            pytest.fail("ServiceProvider must be decorated with @runtime_checkable")

    def test_service_provider_has_required_method(self):
        """ServiceProvider must define get_service_provider method."""
        from victor.core.verticals.protocols.providers import ServiceProvider

        assert hasattr(
            ServiceProvider, "get_service_provider"
        ), "ServiceProvider must define get_service_provider method"

    def test_service_provider_isinstance_check(self):
        """Verify isinstance() works with ServiceProvider."""
        from victor.core.verticals.protocols.providers import ServiceProvider

        class ValidProvider:
            @classmethod
            def get_service_provider(cls) -> Optional[Any]:
                return None

        class InvalidProvider:
            pass

        assert isinstance(
            ValidProvider, ServiceProvider
        ), "Valid implementation should pass isinstance check"
        assert not isinstance(
            InvalidProvider, ServiceProvider
        ), "Invalid implementation should fail isinstance check"


class TestProvidersExported:
    """Tests for protocol exports in __all__."""

    def test_all_providers_exported_from_providers_module(self):
        """All provider protocols must be exported in providers.__all__."""
        from victor.core.verticals.protocols import providers

        expected_exports = [
            "MiddlewareProvider",
            "SafetyProvider",
            "WorkflowProvider",
            "TeamProvider",
            "RLProvider",
            "EnrichmentProvider",
            "ToolProvider",
            "HandlerProvider",
            "CapabilityProvider",
            "ModeConfigProvider",
            "PromptContributorProvider",
            "ToolDependencyProvider",
            "TieredToolConfigProvider",
            "ServiceProvider",
        ]

        for export in expected_exports:
            assert export in providers.__all__, f"{export} must be in providers.__all__"

    def test_all_providers_exported_from_protocols_package(self):
        """All provider protocols must be exported from protocols package."""
        from victor.core.verticals import protocols

        expected_exports = [
            "MiddlewareProvider",
            "SafetyProvider",
            "WorkflowProvider",
            "TeamProvider",
            "RLProvider",
            "EnrichmentProvider",
            "ToolProvider",
            "HandlerProvider",
            "CapabilityProvider",
            "ModeConfigProvider",
            "PromptContributorProvider",
            "ToolDependencyProvider",
            "TieredToolConfigProvider",
            "ServiceProvider",
        ]

        for export in expected_exports:
            assert export in protocols.__all__, f"{export} must be in protocols.__all__"

    def test_all_providers_exported_from_verticals_package(self):
        """All provider protocols must be exported from verticals package."""
        from victor.core import verticals

        expected_exports = [
            "MiddlewareProvider",
            "SafetyProvider",
            "WorkflowProvider",
            "TeamProvider",
            "RLProvider",
            "EnrichmentProvider",
            "ToolProvider",
            "HandlerProvider",
            "CapabilityProvider",
            "ModeConfigProvider",
            "PromptContributorProvider",
            "ToolDependencyProvider",
            "TieredToolConfigProvider",
            "ServiceProvider",
        ]

        for export in expected_exports:
            assert export in verticals.__all__, f"{export} must be in verticals.__all__"


class TestCompositeProviderUsage:
    """Tests for using multiple provider protocols together."""

    def test_vertical_can_implement_multiple_providers(self):
        """A vertical can implement multiple provider protocols."""
        from victor.core.verticals.protocols.providers import (
            MiddlewareProvider,
            SafetyProvider,
            ToolProvider,
            HandlerProvider,
        )

        class MultiCapabilityVertical:
            """A vertical implementing multiple provider protocols."""

            @classmethod
            def get_middleware(cls) -> List[Any]:
                return []

            @classmethod
            def get_safety_extension(cls) -> Optional[Any]:
                return None

            @classmethod
            def get_tools(cls) -> List[str]:
                return ["read", "write"]

            @classmethod
            def get_tool_graph(cls) -> Optional[Any]:
                return None

            @classmethod
            def get_handlers(cls) -> Dict[str, Any]:
                return {}

        # Verify all protocols are satisfied
        assert isinstance(MultiCapabilityVertical, MiddlewareProvider)
        assert isinstance(MultiCapabilityVertical, SafetyProvider)
        assert isinstance(MultiCapabilityVertical, ToolProvider)
        assert isinstance(MultiCapabilityVertical, HandlerProvider)

    def test_partial_implementation_only_matches_implemented(self):
        """A vertical implementing only some protocols only matches those."""
        from victor.core.verticals.protocols.providers import (
            MiddlewareProvider,
            SafetyProvider,
            ToolProvider,
        )

        class PartialVertical:
            """A vertical implementing only MiddlewareProvider."""

            @classmethod
            def get_middleware(cls) -> List[Any]:
                return []

        # Should match MiddlewareProvider
        assert isinstance(PartialVertical, MiddlewareProvider)

        # Should NOT match SafetyProvider or ToolProvider
        assert not isinstance(PartialVertical, SafetyProvider)
        assert not isinstance(PartialVertical, ToolProvider)


class TestISPCompliance:
    """Tests verifying ISP compliance of the provider protocols."""

    def test_protocols_are_single_responsibility(self):
        """Each protocol should focus on a single responsibility."""
        from victor.core.verticals.protocols import providers

        # Count methods per protocol (excluding __class__, etc.)
        protocol_methods = {}
        for name in providers.__all__:
            protocol = getattr(providers, name)
            methods = [
                m
                for m in dir(protocol)
                if not m.startswith("_") and callable(getattr(protocol, m, None))
            ]
            protocol_methods[name] = methods

        # Each protocol should have a small, focused interface (1-3 methods typically)
        for name, methods in protocol_methods.items():
            # Allow up to 3 methods per protocol for ISP compliance
            assert len(methods) <= 3, (
                f"{name} has {len(methods)} methods ({methods}), "
                f"which may violate ISP. Consider splitting."
            )

    def test_protocols_can_be_imported_individually(self):
        """Each protocol can be imported directly from its module."""
        # This ensures protocols are properly segregated
        from victor.core.verticals.protocols.providers import MiddlewareProvider
        from victor.core.verticals.protocols.providers import SafetyProvider
        from victor.core.verticals.protocols.providers import WorkflowProvider
        from victor.core.verticals.protocols.providers import TeamProvider
        from victor.core.verticals.protocols.providers import RLProvider
        from victor.core.verticals.protocols.providers import EnrichmentProvider
        from victor.core.verticals.protocols.providers import ToolProvider
        from victor.core.verticals.protocols.providers import HandlerProvider
        from victor.core.verticals.protocols.providers import CapabilityProvider
        from victor.core.verticals.protocols.providers import ModeConfigProvider
        from victor.core.verticals.protocols.providers import PromptContributorProvider
        from victor.core.verticals.protocols.providers import ToolDependencyProvider
        from victor.core.verticals.protocols.providers import TieredToolConfigProvider
        from victor.core.verticals.protocols.providers import ServiceProvider

        # All imports should succeed
        assert MiddlewareProvider is not None
        assert SafetyProvider is not None
        assert WorkflowProvider is not None
        assert TeamProvider is not None
        assert RLProvider is not None
        assert EnrichmentProvider is not None
        assert ToolProvider is not None
        assert HandlerProvider is not None
        assert CapabilityProvider is not None
        assert ModeConfigProvider is not None
        assert PromptContributorProvider is not None
        assert ToolDependencyProvider is not None
        assert TieredToolConfigProvider is not None
        assert ServiceProvider is not None
