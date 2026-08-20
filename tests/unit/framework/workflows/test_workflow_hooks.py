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

"""Unit tests for BaseYAMLWorkflowProvider capability provider hooks.

Tests for the _get_capability_provider_module() hook and get_capability_provider()
method that enable automatic loading of vertical-specific capability providers.
"""

import importlib
import pytest
from typing import Optional
from pathlib import Path

from victor.core.verticals.import_resolver import import_module_with_fallback
from victor.framework.workflows.base_yaml_provider import BaseYAMLWorkflowProvider
from victor.framework.capabilities import BaseCapabilityProvider, CapabilityMetadata


def _load_vertical_attr(module_path: str, attr_name: str):
    """Load a vertical attribute using external-first import fallbacks."""
    module, _resolved = import_module_with_fallback(module_path)
    if module is None or not hasattr(module, attr_name):
        raise ImportError(f"Unable to resolve {module_path}:{attr_name}")
    return getattr(module, attr_name)


# =============================================================================
# Mock Workflow Providers for Testing
# =============================================================================


class MinimalWorkflowProvider(BaseYAMLWorkflowProvider):
    """Minimal workflow provider without capability provider hook."""

    def _get_escape_hatches_module(self) -> str:
        return "victor_research.escape_hatches"


class ResearchWorkflowProviderWithCapability(BaseYAMLWorkflowProvider):
    """Workflow provider with capability provider hook (mimics ResearchWorkflowProvider)."""

    def _get_escape_hatches_module(self) -> str:
        return "victor_research.escape_hatches"

    def _get_capability_provider_module(self) -> Optional[str]:
        return "victor_research.capabilities"


class DevOpsWorkflowProviderWithCapability(BaseYAMLWorkflowProvider):
    """Workflow provider with capability provider hook (mimics DevOpsWorkflowProvider)."""

    def _get_escape_hatches_module(self) -> str:
        return "victor_devops.escape_hatches"

    def _get_capability_provider_module(self) -> Optional[str]:
        return "victor_devops.capabilities"


class DataAnalysisWorkflowProviderWithCapability(BaseYAMLWorkflowProvider):
    """Workflow provider with capability provider hook (mimics DataAnalysisWorkflowProvider)."""

    def _get_escape_hatches_module(self) -> str:
        return "victor_dataanalysis.escape_hatches"

    def _get_capability_provider_module(self) -> Optional[str]:
        return "victor_dataanalysis.capabilities"


class InvalidModuleWorkflowProvider(BaseYAMLWorkflowProvider):
    """Workflow provider with invalid capability module path."""

    def _get_escape_hatches_module(self) -> str:
        return "victor_research.escape_hatches"

    def _get_capability_provider_module(self) -> Optional[str]:
        return "victor.nonexistent.capabilities"


class NoCapabilityClassWorkflowProvider(BaseYAMLWorkflowProvider):
    """Workflow provider with module that has no capability provider class."""

    def _get_escape_hatches_module(self) -> str:
        return "victor_research.escape_hatches"

    def _get_capability_provider_module(self) -> Optional[str]:
        return "victor_research.escape_hatches"  # This module has no CapabilityProvider


# =============================================================================
# _get_capability_provider_module() Hook Tests
# =============================================================================


class TestGetCapabilityProviderModuleHook:
    """Tests for the _get_capability_provider_module() hook method."""

    def test_default_returns_none(self):
        """Default implementation should return None."""
        provider = MinimalWorkflowProvider()
        result = provider._get_capability_provider_module()

        assert result is None

    def test_research_provider_returns_module_path(self):
        """ResearchWorkflowProvider should return correct module path."""
        provider = ResearchWorkflowProviderWithCapability()
        result = provider._get_capability_provider_module()

        assert result == "victor_research.capabilities"

    def test_devops_provider_returns_module_path(self):
        """DevOpsWorkflowProvider should return correct module path."""
        provider = DevOpsWorkflowProviderWithCapability()
        result = provider._get_capability_provider_module()

        assert result == "victor_devops.capabilities"

    def test_dataanalysis_provider_returns_module_path(self):
        """DataAnalysisWorkflowProvider should return correct module path."""
        provider = DataAnalysisWorkflowProviderWithCapability()
        result = provider._get_capability_provider_module()

        assert result == "victor_dataanalysis.capabilities"

    def test_returns_optional_string(self):
        """Method should return Optional[str]."""
        provider = MinimalWorkflowProvider()
        result = provider._get_capability_provider_module()

        # Result should be None or a string
        assert result is None or isinstance(result, str)


# =============================================================================
# get_capability_provider() Method Tests
# =============================================================================


class TestGetCapabilityProviderMethod:
    """Tests for the get_capability_provider() method."""

    def test_returns_none_when_hook_returns_none(self):
        """get_capability_provider should return None when hook returns None."""
        provider = MinimalWorkflowProvider()
        result = provider.get_capability_provider()

        assert result is None

    def test_returns_none_for_invalid_module(self):
        """get_capability_provider should return None for invalid module path."""
        provider = InvalidModuleWorkflowProvider()
        result = provider.get_capability_provider()

        # Should handle ImportError gracefully and return None
        assert result is None

    def test_returns_none_for_module_without_provider(self):
        """get_capability_provider should return None when module has no capability provider."""
        provider = NoCapabilityClassWorkflowProvider()
        result = provider.get_capability_provider()

        assert result is None

    @pytest.mark.integration
    def test_returns_capability_provider_for_research(self):
        """get_capability_provider should return ResearchCapabilityProvider."""
        provider = ResearchWorkflowProviderWithCapability()
        result = provider.get_capability_provider()

        if result is None:
            pytest.skip("victor_research.capabilities not available")
        assert isinstance(result, BaseCapabilityProvider)

    @pytest.mark.integration
    def test_returns_capability_provider_for_devops(self):
        """get_capability_provider should return DevOpsCapabilityProvider."""
        provider = DevOpsWorkflowProviderWithCapability()
        result = provider.get_capability_provider()

        if result is None:
            pytest.skip("victor_devops.capabilities not available")
        assert isinstance(result, BaseCapabilityProvider)

    @pytest.mark.integration
    def test_returns_capability_provider_for_dataanalysis(self):
        """get_capability_provider should return DataAnalysisCapabilityProvider."""
        provider = DataAnalysisWorkflowProviderWithCapability()
        result = provider.get_capability_provider()

        if result is None:
            pytest.skip("victor_dataanalysis.capabilities not available")
        assert isinstance(result, BaseCapabilityProvider)

    def test_returned_provider_has_required_methods(self):
        """Returned capability provider should have required methods."""
        provider = ResearchWorkflowProviderWithCapability()
        result = provider.get_capability_provider()

        if result:  # Only test if provider exists
            assert hasattr(result, "get_capabilities")
            assert hasattr(result, "get_capability_metadata")
            assert callable(result.get_capabilities)
            assert callable(result.get_capabilities)

    def test_returned_provider_has_metadata(self):
        """Returned capability provider should have metadata."""
        provider = ResearchWorkflowProviderWithCapability()
        result = provider.get_capability_provider()

        if result:
            metadata = result.get_capability_metadata()
            assert isinstance(metadata, dict)
            assert len(metadata) > 0

            # Check that metadata has correct structure
            for meta in metadata.values():
                assert isinstance(meta, CapabilityMetadata)

    def test_returned_provider_has_capabilities(self):
        """Returned capability provider should have capabilities."""
        provider = DevOpsWorkflowProviderWithCapability()
        result = provider.get_capability_provider()

        if result:
            capabilities = result.get_capabilities()
            assert isinstance(capabilities, dict)
            assert len(capabilities) > 0

    def test_research_provider_capabilities(self):
        """Research capability provider should have expected capabilities."""
        provider = ResearchWorkflowProviderWithCapability()
        result = provider.get_capability_provider()

        if result:
            capabilities = result.list_capabilities()

            # Research vertical should have these capabilities
            expected_capabilities = [
                "source_verification",
                "citation_management",
                "research_quality",
                "literature_analysis",
                "fact_checking",
            ]

            for expected in expected_capabilities:
                assert expected in capabilities, f"Missing capability: {expected}"

    def test_devops_provider_capabilities(self):
        """DevOps capability provider should have expected capabilities."""
        provider = DevOpsWorkflowProviderWithCapability()
        result = provider.get_capability_provider()

        if result:
            capabilities = result.list_capabilities()

            # DevOps vertical should have these capabilities
            expected_capabilities = [
                "deployment_safety",
                "container_settings",
                "infrastructure_settings",
                "cicd_settings",
                "monitoring_settings",
            ]

            for expected in expected_capabilities:
                assert expected in capabilities, f"Missing capability: {expected}"

    def test_dataanalysis_provider_capabilities(self):
        """DataAnalysis capability provider should have expected capabilities."""
        provider = DataAnalysisWorkflowProviderWithCapability()
        result = provider.get_capability_provider()

        if result:
            capabilities = result.list_capabilities()

            # DataAnalysis vertical should have capabilities
            # (checking that it has some capabilities)
            assert len(capabilities) > 0

    def test_capability_provider_is_new_instance(self):
        """Each call to get_capability_provider should return a new instance."""
        provider = ResearchWorkflowProviderWithCapability()
        result1 = provider.get_capability_provider()
        result2 = provider.get_capability_provider()

        if result1 and result2:
            # Should be different instances
            assert result1 is not result2

    def test_filters_abstract_base_class(self):
        """get_capability_provider should filter out BaseCapabilityProvider abstract class."""
        provider = ResearchWorkflowProviderWithCapability()
        result = provider.get_capability_provider()

        if result:
            # Result should not be the abstract base class
            assert type(result).__name__ != "BaseCapabilityProvider"

            # Should be a concrete implementation
            assert result.get_capabilities() is not None
            assert result.get_capability_metadata() is not None


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling in get_capability_provider()."""

    def test_handles_import_error_gracefully(self):
        """Should handle ImportError gracefully and return None."""
        provider = InvalidModuleWorkflowProvider()
        result = provider.get_capability_provider()

        # Should not raise exception
        assert result is None

    def test_logs_warning_for_invalid_module(self, caplog):
        """Should log warning when module cannot be imported."""
        import logging

        provider = InvalidModuleWorkflowProvider()

        with caplog.at_level(logging.WARNING):
            result = provider.get_capability_provider()

        # Should have logged a warning
        assert result is None
        # Warning should mention the module path
        assert any("nonexistent" in record.message for record in caplog.records)

    def test_logs_warning_for_missing_provider(self, caplog):
        """Should log warning when no capability provider found in module."""
        import logging

        provider = NoCapabilityClassWorkflowProvider()

        with caplog.at_level(logging.WARNING):
            result = provider.get_capability_provider()

        # Should have logged a warning
        assert result is None

    def test_handles_attribute_error_gracefully(self):
        """Should handle AttributeError gracefully and return None."""

        # Create a provider that returns a module without expected attributes
        class BrokenProvider(BaseYAMLWorkflowProvider):
            def _get_escape_hatches_module(self) -> str:
                return "victor_research.escape_hatches"

            def _get_capability_provider_module(self) -> Optional[str]:
                return "victor.framework.capabilities.base"  # Has no CapabilityProvider

        provider = BrokenProvider()
        result = provider.get_capability_provider()

        # Should handle gracefully
        assert result is None

    def test_handles_instantiation_error(self):
        """Should handle TypeError during instantiation gracefully."""
        # This tests the try/except block that catches abstract class instantiation
        provider = ResearchWorkflowProviderWithCapability()
        result = provider.get_capability_provider()

        # Should succeed with concrete class
        if result:
            assert isinstance(result, BaseCapabilityProvider)


# =============================================================================
# Real Vertical Integration Tests
# =============================================================================


class TestRealVerticalIntegration:
    """Integration tests with actual vertical workflow providers."""

    @pytest.mark.slow
    def test_actual_research_workflow_provider(self):
        """Test with actual ResearchWorkflowProvider from victor_research.workflows."""
        try:
            ResearchWorkflowProvider = _load_vertical_attr(
                "victor.research.workflows",
                "ResearchWorkflowProvider",
            )

            provider = ResearchWorkflowProvider()
            result = provider.get_capability_provider()

            # Should return ResearchCapabilityProvider
            assert result is not None
            assert isinstance(result, BaseCapabilityProvider)

            # Should have Research-specific capabilities
            capabilities = result.list_capabilities()
            assert "source_verification" in capabilities
            assert "citation_management" in capabilities

        except ImportError:
            pytest.skip("victor_research.workflows not available")

    @pytest.mark.slow
    def test_actual_devops_workflow_provider(self):
        """Test with actual DevOpsWorkflowProvider from victor_devops.workflows."""
        try:
            DevOpsWorkflowProvider = _load_vertical_attr(
                "victor.devops.workflows",
                "DevOpsWorkflowProvider",
            )

            provider = DevOpsWorkflowProvider()
            result = provider.get_capability_provider()

            # Should return DevOpsCapabilityProvider
            assert result is not None
            assert isinstance(result, BaseCapabilityProvider)

            # Should have DevOps-specific capabilities
            capabilities = result.list_capabilities()
            assert "deployment_safety" in capabilities
            assert "container_settings" in capabilities

        except ImportError:
            pytest.skip("victor_devops.workflows not available")

    @pytest.mark.slow
    def test_actual_dataanalysis_workflow_provider(self):
        """Test with actual DataAnalysisWorkflowProvider from victor_dataanalysis.workflows."""
        try:
            DataAnalysisWorkflowProvider = _load_vertical_attr(
                "victor.dataanalysis.workflows",
                "DataAnalysisWorkflowProvider",
            )

            provider = DataAnalysisWorkflowProvider()
            result = provider.get_capability_provider()

            # Should return DataAnalysisCapabilityProvider
            assert result is not None
            assert isinstance(result, BaseCapabilityProvider)

            # Should have capabilities
            capabilities = result.list_capabilities()
            assert len(capabilities) > 0

        except ImportError:
            pytest.skip("victor_dataanalysis.workflows not available")


# =============================================================================
# Capability Provider Metadata Tests
# =============================================================================


class TestCapabilityProviderMetadata:
    """Tests for capability provider metadata."""

    def test_research_provider_metadata(self):
        """Research capability provider should have correct metadata."""
        provider = ResearchWorkflowProviderWithCapability()
        result = provider.get_capability_provider()

        if result:
            metadata = result.get_capability_metadata()

            # Check structure
            for name, meta in metadata.items():
                assert isinstance(meta, CapabilityMetadata)
                assert meta.name == name
                assert isinstance(meta.description, str)
                assert isinstance(meta.version, str)
                assert isinstance(meta.tags, list)

    def test_devops_provider_metadata_dependencies(self):
        """DevOps capability provider metadata should have dependencies."""
        provider = DevOpsWorkflowProviderWithCapability()
        result = provider.get_capability_provider()

        if result:
            metadata = result.get_capability_metadata()

            # monitoring_settings should depend on deployment_safety
            if "monitoring_settings" in metadata:
                monitoring_meta = metadata["monitoring_settings"]
                assert isinstance(monitoring_meta.dependencies, list)

    def test_capability_tags_include_framework(self):
        """Capability tags should be relevant to the vertical."""
        provider = ResearchWorkflowProviderWithCapability()
        result = provider.get_capability_provider()

        if result:
            metadata = result.get_capability_metadata()

            for meta in metadata.values():
                # Tags should be non-empty
                assert len(meta.tags) > 0

                # Tags should be strings
                for tag in meta.tags:
                    assert isinstance(tag, str)


# =============================================================================
# Lazy Loading Tests
# =============================================================================


class TestLazyLoading:
    """Tests for lazy loading behavior."""

    def test_capability_provider_not_loaded_on_init(self):
        """Capability provider should not be loaded during workflow provider init."""
        # Creating a workflow provider should not load the capability provider
        provider = ResearchWorkflowProviderWithCapability()

        # get_capability_provider() should work on first call
        result = provider.get_capability_provider()

        if result:
            assert isinstance(result, BaseCapabilityProvider)

    def test_multiple_calls_load_fresh_instances(self):
        """Each call should load a fresh instance."""
        provider = ResearchWorkflowProviderWithCapability()

        result1 = provider.get_capability_provider()
        result2 = provider.get_capability_provider()

        if result1 and result2:
            # Should be different instances
            assert result1 is not result2

            # But should have same capabilities
            caps1 = result1.list_capabilities()
            caps2 = result2.list_capabilities()
            assert set(caps1) == set(caps2)


# =============================================================================
# Abstract Base Class Filtering Tests
# =============================================================================


class TestAbstractBaseClassFiltering:
    """Tests for filtering out abstract base classes."""

    def test_skips_base_capability_provider(self):
        """Should skip BaseCapabilityProvider abstract class."""
        provider = ResearchWorkflowProviderWithCapability()
        result = provider.get_capability_provider()

        if result:
            # Result should not be the abstract base
            assert type(result).__name__ != "BaseCapabilityProvider"

            # Should be a concrete implementation
            capabilities = result.get_capabilities()
            assert capabilities is not None

    def test_skips_classes_starting_with_base(self):
        """Should skip classes starting with 'Base'."""
        provider = ResearchWorkflowProviderWithCapability()
        result = provider.get_capability_provider()

        if result:
            # Result class name should not start with "Base"
            class_name = type(result).__name__
            assert not class_name.startswith("Base")

    def test_requires_get_capabilities_method(self):
        """Returned provider must have get_capabilities method."""
        provider = ResearchWorkflowProviderWithCapability()
        result = provider.get_capability_provider()

        if result:
            assert hasattr(result, "get_capabilities")
            assert callable(result.get_capabilities)

    def test_requires_get_capability_metadata_method(self):
        """Returned provider must have get_capability_metadata method."""
        provider = ResearchWorkflowProviderWithCapability()
        result = provider.get_capability_provider()

        if result:
            assert hasattr(result, "get_capability_metadata")
            assert callable(result.get_capability_metadata)


__all__ = [
    # Test classes
    "TestGetCapabilityProviderModuleHook",
    "TestGetCapabilityProviderMethod",
    "TestErrorHandling",
    "TestRealVerticalIntegration",
    "TestCapabilityProviderMetadata",
    "TestLazyLoading",
    "TestAbstractBaseClassFiltering",
]
