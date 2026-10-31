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

``victor.core.verticals.protocols.providers`` now only defines the provider
protocols whose hook shape is still core-local. The remaining ISP providers
are canonically defined in the SDK and re-exported by the
``victor.core.verticals.protocols`` package; the package-level tests below pin
that public surface.
"""

import pytest
from typing import Any, Dict, Optional


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


# The remaining ISP providers are SDK-backed re-exports; every one of them must
# stay importable from the public package path.
_SDK_BACKED_PROVIDERS = [
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
]

_LOCAL_PROVIDERS = [
    "PromptContributorProvider",
    "ToolDependencyProvider",
    "TieredToolConfigProvider",
    "ServiceProvider",
]


class TestProvidersExported:
    """Tests for protocol exports in __all__."""

    def test_local_providers_exported_from_providers_module(self):
        """Locally-defined provider protocols must be in providers.__all__."""
        from victor.core.verticals.protocols import providers

        for export in _LOCAL_PROVIDERS:
            assert export in providers.__all__, f"{export} must be in providers.__all__"

        # The SDK-backed forks were deleted from this submodule; importing them
        # here must now fail loudly instead of silently binding a second,
        # divergent definition of the same name.
        for stale in _SDK_BACKED_PROVIDERS:
            assert stale not in providers.__all__, (
                f"{stale} is SDK-backed and must not be re-declared in the " "providers submodule"
            )

    def test_all_providers_exported_from_protocols_package(self):
        """All provider protocols must be exported from protocols package."""
        from victor.core.verticals import protocols

        for export in _SDK_BACKED_PROVIDERS + _LOCAL_PROVIDERS:
            assert export in protocols.__all__, f"{export} must be in protocols.__all__"

    def test_all_providers_exported_from_verticals_package(self):
        """All provider protocols must be exported from verticals package."""
        from victor.core import verticals

        for export in _SDK_BACKED_PROVIDERS + _LOCAL_PROVIDERS:
            assert export in verticals.__all__, f"{export} must be in verticals.__all__"

    def test_sdk_backed_providers_are_the_sdk_objects(self):
        """Package-level provider names must resolve to the SDK definitions.

        Guards against re-introducing a local fork under the same name.
        """
        from victor.core.verticals import protocols
        from victor_contracts.verticals import protocols as sdk_protocols

        for name in _SDK_BACKED_PROVIDERS:
            assert getattr(protocols, name) is getattr(sdk_protocols, name), name


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
        from victor.core.verticals.protocols.providers import (
            PromptContributorProvider,
            ToolDependencyProvider,
            TieredToolConfigProvider,
            ServiceProvider,
        )

        # All imports should succeed
        assert PromptContributorProvider is not None
        assert ToolDependencyProvider is not None
        assert TieredToolConfigProvider is not None
        assert ServiceProvider is not None

    def test_deleted_forks_are_not_importable_from_submodule(self):
        """Negative test: the removed local forks must stay gone.

        If one of these names resolves from the submodule again, the package
        ``__init__`` would still shadow it — reintroducing the silent
        "same name, two objects" divergence this sweep removed.
        """
        import importlib

        providers = importlib.import_module("victor.core.verticals.protocols.providers")

        for name in _SDK_BACKED_PROVIDERS:
            assert not hasattr(providers, name), (
                f"{name} is SDK-backed and must only be imported from the "
                "victor.core.verticals.protocols package"
            )
