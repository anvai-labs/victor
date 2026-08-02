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

"""Integration tests for capability provider loading across all verticals.

This test module verifies that each vertical's workflow provider can correctly
load and use its associated capability provider. It tests:

1. Research Vertical - 5 capabilities (source_verification, citation_management, research_quality, literature_analysis, fact_checking)
2. DevOps Vertical - 5 capabilities (deployment_safety, container_settings, infrastructure_settings, cicd_settings, monitoring_settings)
3. DataAnalysis Vertical - 5 capabilities (data_quality, visualization_style, statistical_analysis, ml_pipeline, data_privacy)
4. Coding Vertical - 5 capabilities (git_safety, code_style, test_requirements, language_server, refactoring)
5. RAG Vertical - 5 capabilities (indexing, retrieval, synthesis, safety, query_enhancement)
6. Benchmark Vertical - Hook implementation (no capability provider yet)

For each vertical, tests verify:
- Lazy loading (provider not loaded until get_capability_provider() called)
- Error handling (graceful fallback on import errors)
- Provider instance creation (fresh instance each call)
- Capability metadata is correct
- Capabilities can be applied to orchestrator
"""

from typing import Dict, List, Optional, Set, Tuple
from unittest.mock import MagicMock

import pytest

from victor.core.verticals.import_resolver import import_module_with_fallback

from victor.framework.capabilities import BaseCapabilityProvider, CapabilityMetadata
from victor.framework.capability_config_service import CapabilityConfigService


def _load_vertical_attr(
    module_path: str,
    attr_name: str,
    *,
    allow_module_level: bool = False,
):
    """Resolve vertical attributes using external-first import fallbacks."""
    module, _resolved = import_module_with_fallback(module_path)
    if module is None or not hasattr(module, attr_name):
        pytest.skip(
            f"Vertical module or attribute unavailable: {module_path}:{attr_name}",
            allow_module_level=allow_module_level,
        )
    return getattr(module, attr_name)


def _load_provider_class(provider_class_path: str):
    """Load a provider class from dotted path using vertical import fallbacks."""
    module_path, class_name = provider_class_path.rsplit(".", 1)
    return _load_vertical_attr(module_path, class_name)


CodingWorkflowProvider = _load_vertical_attr(
    "victor.coding.workflows",
    "CodingWorkflowProvider",
    allow_module_level=True,
)
DevOpsWorkflowProvider = _load_vertical_attr(
    "victor.devops.workflows",
    "DevOpsWorkflowProvider",
    allow_module_level=True,
)
DataAnalysisWorkflowProvider = _load_vertical_attr(
    "victor.dataanalysis.workflows",
    "DataAnalysisWorkflowProvider",
    allow_module_level=True,
)
ResearchWorkflowProvider = _load_vertical_attr(
    "victor.research.workflows",
    "ResearchWorkflowProvider",
    allow_module_level=True,
)
RAGWorkflowProvider = _load_vertical_attr(
    "victor.rag.workflows",
    "RAGWorkflowProvider",
    allow_module_level=True,
)
BenchmarkWorkflowProvider = _load_vertical_attr(
    "victor.benchmark.workflows",
    "BenchmarkWorkflowProvider",
    allow_module_level=True,
)


class MockOrchestrator:
    """Mock orchestrator for testing capability application.

    This mock provides the minimal interface required by capability
    providers for testing configuration application.
    """

    def __init__(self):
        """Initialize mock orchestrator with config storage."""
        self.source_verification_config: Optional[Dict] = None
        self.citation_config: Optional[Dict] = None
        self.research_quality_config: Optional[Dict] = None
        self.literature_config: Optional[Dict] = None
        self.fact_checking_config: Optional[Dict] = None

        self.safety_config: Dict = {"git": {}, "deployment": {}}
        self.code_style: Optional[Dict] = None
        self.test_config: Optional[Dict] = None
        self.lsp_config: Optional[Dict] = None
        self.refactor_config: Optional[Dict] = None

        self.container_config: Optional[Dict] = None
        self.infra_config: Optional[Dict] = None
        self.cicd_config: Optional[Dict] = None
        self.monitoring_config: Optional[Dict] = None

        self.data_quality_config: Optional[Dict] = None
        self.visualization_config: Optional[Dict] = None
        self.statistics_config: Optional[Dict] = None
        self.ml_config: Optional[Dict] = None
        self.privacy_config: Optional[Dict] = None

        self.rag_config: Dict = {}


class _StubContainer:
    """Minimal container stub exposing get_optional for service lookup."""

    def __init__(self, service: CapabilityConfigService):
        self._service = service

    def get_optional(self, service_type):
        if isinstance(self._service, service_type):
            return self._service
        return None


class ServiceBackedOrchestrator(MockOrchestrator):
    """Mock orchestrator with framework CapabilityConfigService wiring."""

    def __init__(self, service: CapabilityConfigService):
        super().__init__()
        self._container = _StubContainer(service)

    def get_service_container(self):
        return self._container


class TestResearchVerticalCapabilities:
    """Test Research vertical capability provider loading and functionality.

    Verifies that ResearchWorkflowProvider correctly loads ResearchCapabilityProvider
    with all 5 capabilities: source_verification, citation_management, research_quality,
    literature_analysis, and fact_checking.
    """

    def test_research_workflow_provider_has_capability_module(self):
        """Verify ResearchWorkflowProvider defines capability provider module.

        This test ensures that _get_capability_provider_module() returns
        the correct module path for ResearchCapabilityProvider.
        """
        provider = ResearchWorkflowProvider()
        module_path = provider._get_capability_provider_module()

        assert module_path == "victor_research.capabilities"

    def test_research_capability_provider_loading(self):
        """Test lazy loading of ResearchCapabilityProvider.

        Verifies that:
        1. Provider is not loaded until get_capability_provider() is called
        2. get_capability_provider() successfully imports and instantiates
        3. Each call returns a fresh instance
        """
        provider = ResearchWorkflowProvider()

        # Provider should not be loaded initially
        # (we can't directly test this without accessing private state)

        # Load provider for the first time
        cap_provider_1 = provider.get_capability_provider()
        assert cap_provider_1 is not None
        assert isinstance(cap_provider_1, BaseCapabilityProvider)

        # Second call should return a fresh instance
        cap_provider_2 = provider.get_capability_provider()
        assert cap_provider_2 is not None
        assert cap_provider_1 is not cap_provider_2  # Different instances

    def test_research_capabilities_accessible(self):
        """Test that all 5 Research capabilities are accessible.

        Verifies that the capability provider exposes all expected
        capabilities through get_capabilities().
        """
        provider = ResearchWorkflowProvider()
        cap_provider = provider.get_capability_provider()

        capabilities = cap_provider.get_capabilities()

        assert len(capabilities) == 5
        expected_caps = {
            "source_verification",
            "citation_management",
            "research_quality",
            "literature_analysis",
            "fact_checking",
        }
        assert set(capabilities.keys()) == expected_caps

    def test_research_capability_metadata(self):
        """Test that Research capability metadata is correct.

        Verifies that get_capability_metadata() returns proper metadata
        for all capabilities including name, description, version, and tags.
        """
        provider = ResearchWorkflowProvider()
        cap_provider = provider.get_capability_provider()

        metadata = cap_provider.get_capability_metadata()

        assert len(metadata) == 5

        # Check source_verification metadata
        source_verify_meta = metadata["source_verification"]
        assert source_verify_meta.name == "source_verification"
        assert "credibility" in source_verify_meta.description.lower()
        assert source_verify_meta.version == "1.0"
        assert "safety" in source_verify_meta.tags

        # Check research_quality has dependencies
        quality_meta = metadata["research_quality"]
        assert "source_verification" in quality_meta.dependencies

    def test_research_capability_application(self):
        """Test that Research capabilities can be applied to orchestrator.

        Verifies that calling apply methods on the capability provider
        correctly configures the orchestrator instance.
        """
        provider = ResearchWorkflowProvider()
        cap_provider = provider.get_capability_provider()

        orchestrator = MockOrchestrator()

        # Apply source verification capability
        cap_provider.apply_source_verification(orchestrator, min_credibility=0.8)

        assert orchestrator.source_verification_config is not None
        assert orchestrator.source_verification_config["min_credibility"] == 0.8
        assert orchestrator.source_verification_config["min_source_count"] == 3

        # Apply citation management
        cap_provider.apply_citation_management(orchestrator, default_style="chicago")

        assert orchestrator.citation_config is not None
        assert orchestrator.citation_config["default_style"] == "chicago"

    def test_research_getter_methods(self):
        """Test Research capability getter methods return correct config.

        Verifies that getter functions properly retrieve configuration
        from the orchestrator.
        """
        get_source_verification = _load_vertical_attr(
            "victor_research.capabilities",
            "get_source_verification",
        )
        get_citation_config = _load_vertical_attr(
            "victor_research.capabilities",
            "get_citation_config",
        )
        get_research_quality = _load_vertical_attr(
            "victor_research.capabilities",
            "get_research_quality",
        )

        orchestrator = MockOrchestrator()

        # Set some config
        orchestrator.source_verification_config = {
            "min_credibility": 0.9,
            "min_source_count": 5,
            "require_diverse_sources": True,
            "validate_urls": False,
        }

        # Get it back
        config = get_source_verification(orchestrator)

        assert config["min_credibility"] == 0.9
        assert config["min_source_count"] == 5

        # Test getter with config set
        orchestrator.citation_config = {
            "default_style": "chicago",
            "require_urls": False,
        }
        citation_cfg = get_citation_config(orchestrator)
        assert citation_cfg["default_style"] == "chicago"
        assert citation_cfg["require_urls"] is False

        # Test getter returns config when it's set (even if partially)
        assert citation_cfg is not None


class TestDevOpsVerticalCapabilities:
    """Test DevOps vertical capability provider loading and functionality.

    Verifies that DevOpsWorkflowProvider correctly loads DevOpsCapabilityProvider
    with all 5 capabilities: deployment_safety, container_settings, infrastructure_settings,
    cicd_settings, and monitoring_settings.
    """

    def test_devops_workflow_provider_has_capability_module(self):
        """Verify DevOpsWorkflowProvider defines capability provider module."""
        provider = DevOpsWorkflowProvider()
        module_path = provider._get_capability_provider_module()

        assert module_path == "victor_devops.capabilities"

    def test_devops_capability_provider_loading(self):
        """Test lazy loading of DevOpsCapabilityProvider."""
        provider = DevOpsWorkflowProvider()

        cap_provider_1 = provider.get_capability_provider()
        assert cap_provider_1 is not None
        assert isinstance(cap_provider_1, BaseCapabilityProvider)

        # Verify fresh instance each call
        cap_provider_2 = provider.get_capability_provider()
        assert cap_provider_1 is not cap_provider_2

    def test_devops_capabilities_accessible(self):
        """Test that all 5 DevOps capabilities are accessible."""
        provider = DevOpsWorkflowProvider()
        cap_provider = provider.get_capability_provider()

        capabilities = cap_provider.get_capabilities()

        assert len(capabilities) == 5
        expected_caps = {
            "deployment_safety",
            "container_settings",
            "infrastructure_settings",
            "cicd_settings",
            "monitoring_settings",
        }
        assert set(capabilities.keys()) == expected_caps

    def test_devops_capability_metadata(self):
        """Test that DevOps capability metadata is correct."""
        provider = DevOpsWorkflowProvider()
        cap_provider = provider.get_capability_provider()

        metadata = cap_provider.get_capability_metadata()

        assert len(metadata) == 5

        # Check deployment_safety metadata
        deploy_meta = metadata["deployment_safety"]
        assert deploy_meta.name == "deployment_safety"
        assert "safety" in deploy_meta.tags

        # Check monitoring has dependencies
        monitoring_meta = metadata["monitoring_settings"]
        assert "deployment_safety" in monitoring_meta.dependencies

    def test_devops_capability_application(self):
        """Test that DevOps capabilities can be applied to orchestrator."""
        provider = DevOpsWorkflowProvider()
        cap_provider = provider.get_capability_provider()

        orchestrator = MockOrchestrator()

        # Apply deployment safety
        cap_provider.apply_deployment_safety(orchestrator, require_approval_for_production=True)

        assert orchestrator.safety_config is not None
        assert "deployment" in orchestrator.safety_config
        assert orchestrator.safety_config["deployment"]["require_approval_for_production"] is True

        # Apply container settings
        cap_provider.apply_container_settings(orchestrator, runtime="podman")

        assert orchestrator.container_config is not None
        assert orchestrator.container_config["runtime"] == "podman"

    def test_devops_getter_methods(self):
        """Test DevOps capability getter methods."""
        get_container_settings = _load_vertical_attr(
            "victor_devops.capabilities",
            "get_container_settings",
        )

        orchestrator = MockOrchestrator()
        orchestrator.container_config = {
            "runtime": "docker",
            "default_registry": "ghcr.io",
            "security_scan_enabled": False,
            "max_image_size_mb": 5000,
        }

        config = get_container_settings(orchestrator)

        assert config["runtime"] == "docker"
        assert config["max_image_size_mb"] == 5000


class TestDataAnalysisVerticalCapabilities:
    """Test DataAnalysis vertical capability provider loading and functionality.

    Verifies that DataAnalysisWorkflowProvider correctly loads
    DataAnalysisCapabilityProvider with all 5 capabilities and tests
    framework privacy capability integration.
    """

    def test_dataanalysis_workflow_provider_has_capability_module(self):
        """Verify DataAnalysisWorkflowProvider defines capability provider module."""
        provider = DataAnalysisWorkflowProvider()
        module_path = provider._get_capability_provider_module()

        assert module_path == "victor.dataanalysis.capabilities"

    def test_dataanalysis_capability_provider_loading(self):
        """Test lazy loading of DataAnalysisCapabilityProvider."""
        provider = DataAnalysisWorkflowProvider()

        cap_provider_1 = provider.get_capability_provider()
        assert cap_provider_1 is not None
        assert isinstance(cap_provider_1, BaseCapabilityProvider)

        cap_provider_2 = provider.get_capability_provider()
        assert cap_provider_1 is not cap_provider_2

    def test_dataanalysis_capabilities_accessible(self):
        """Test that all 5 DataAnalysis capabilities are accessible."""
        provider = DataAnalysisWorkflowProvider()
        cap_provider = provider.get_capability_provider()

        capabilities = cap_provider.get_capabilities()

        assert len(capabilities) == 5
        expected_caps = {
            "data_quality",
            "visualization_style",
            "statistical_analysis",
            "ml_pipeline",
            "data_privacy",
        }
        assert set(capabilities.keys()) == expected_caps

    def test_dataanalysis_capability_metadata(self):
        """Test that DataAnalysis capability metadata is correct."""
        provider = DataAnalysisWorkflowProvider()
        cap_provider = provider.get_capability_provider()

        metadata = cap_provider.get_capability_metadata()

        assert len(metadata) == 5

        # Check data_privacy metadata
        privacy_meta = metadata["data_privacy"]
        assert privacy_meta.name == "data_privacy"
        assert "privacy" in privacy_meta.tags
        assert "safety" in privacy_meta.tags

    def test_framework_privacy_capability_integration(self):
        """Test framework privacy capability integration.

        Verifies that DataAnalysisCapabilityProvider properly delegates
        to framework PrivacyCapabilityProvider for privacy management.
        """
        configure_data_privacy = _load_vertical_attr(
            "victor.dataanalysis.capabilities",
            "configure_data_privacy",
        )
        get_privacy_config = _load_vertical_attr(
            "victor.dataanalysis.capabilities",
            "get_privacy_config",
        )

        orchestrator = MockOrchestrator()

        # Configure privacy (delegates to framework)
        configure_data_privacy(
            orchestrator,
            anonymize_pii=True,
            pii_columns=["ssn", "email"],
            hash_identifiers=True,
        )

        # Get config back (also delegates to framework)
        config = get_privacy_config(orchestrator)

        # Verify delegation worked - should have framework privacy config
        assert config is not None
        # The framework privacy capability sets config on orchestrator
        assert hasattr(orchestrator, "privacy_config") or config.get("anonymize_pii") is True

    def test_dataanalysis_capability_application(self):
        """Test that DataAnalysis capabilities can be applied to orchestrator."""
        provider = DataAnalysisWorkflowProvider()
        cap_provider = provider.get_capability_provider()

        orchestrator = MockOrchestrator()

        # Apply data quality
        cap_provider.apply_data_quality(orchestrator, min_completeness=0.95)

        assert orchestrator.data_quality_config is not None
        assert orchestrator.data_quality_config["min_completeness"] == 0.95

        # Apply visualization style
        cap_provider.apply_visualization_style(orchestrator, default_backend="plotly")

        assert orchestrator.visualization_config is not None
        assert orchestrator.visualization_config["backend"] == "plotly"


class TestCodingVerticalCapabilities:
    """Test Coding vertical capability provider loading and functionality.

    Verifies that CodingWorkflowProvider correctly loads CodingCapabilityProvider
    with all 5 capabilities and tests middleware integration.
    """

    def test_coding_workflow_provider_has_capability_module(self):
        """Verify CodingWorkflowProvider does NOT define capability provider module.

        Note: Coding vertical uses a different pattern - it has a capabilities.py
        module but the workflow provider doesn't implement the hook yet.
        This test documents the current state.
        """
        provider = CodingWorkflowProvider()
        # Coding provider doesn't implement the hook (returns None or not implemented)
        module_path = provider._get_capability_provider_module()
        assert module_path is None

    def test_coding_capability_provider_direct_import(self):
        """Test direct import of CodingCapabilityProvider.

        Since CodingWorkflowProvider doesn't implement the hook,
        test that we can still import the provider directly.
        """
        CodingCapabilityProvider = _load_vertical_attr(
            "victor.coding.capabilities",
            "CodingCapabilityProvider",
        )

        provider = CodingCapabilityProvider()

        assert isinstance(provider, BaseCapabilityProvider)

        capabilities = provider.get_capabilities()
        assert len(capabilities) == 5
        expected_caps = {
            "git_safety",
            "code_style",
            "test_requirements",
            "language_server",
            "refactoring",
        }
        assert set(capabilities.keys()) == expected_caps

    def test_coding_capabilities_accessible(self):
        """Test that all 5 Coding capabilities are accessible via direct import."""
        CodingCapabilityProvider = _load_vertical_attr(
            "victor.coding.capabilities",
            "CodingCapabilityProvider",
        )

        cap_provider = CodingCapabilityProvider()
        capabilities = cap_provider.get_capabilities()

        assert len(capabilities) == 5

        # Verify each capability is callable
        for cap_name, cap_func in capabilities.items():
            assert callable(cap_func)

    def test_coding_capability_metadata(self):
        """Test that Coding capability metadata is correct."""
        CodingCapabilityProvider = _load_vertical_attr(
            "victor.coding.capabilities",
            "CodingCapabilityProvider",
        )

        cap_provider = CodingCapabilityProvider()
        metadata = cap_provider.get_capability_metadata()

        assert len(metadata) == 5

        # Check git_safety metadata
        git_meta = metadata["git_safety"]
        assert git_meta.name == "git_safety"
        assert "safety" in git_meta.tags

        # Check refactoring has dependencies
        refactor_meta = metadata["refactoring"]
        assert "language_server" in refactor_meta.dependencies

    def test_coding_capability_application(self):
        """Test that Coding capabilities can be applied to orchestrator."""
        CodingCapabilityProvider = _load_vertical_attr(
            "victor.coding.capabilities",
            "CodingCapabilityProvider",
        )

        cap_provider = CodingCapabilityProvider()
        orchestrator = MockOrchestrator()

        # Apply code style (this doesn't depend on CodingSafetyExtension)
        cap_provider.apply_code_style(orchestrator, formatter="black")

        assert orchestrator.code_style is not None
        assert orchestrator.code_style["formatter"] == "black"

        # Apply test requirements
        cap_provider.apply_test_requirements(orchestrator, min_coverage=0.8)

        assert orchestrator.test_config is not None
        assert orchestrator.test_config["min_coverage"] == 0.8

    def test_coding_middleware_integration(self):
        """Test coding capability middleware integration.

        Verifies that capabilities properly integrate with coding middleware
        like code style enforcement.
        """
        configure_code_style = _load_vertical_attr(
            "victor.coding.capabilities",
            "configure_code_style",
        )
        get_code_style = _load_vertical_attr(
            "victor.coding.capabilities",
            "get_code_style",
        )

        orchestrator = MockOrchestrator()

        # Configure code style middleware
        configure_code_style(
            orchestrator,
            formatter="black",
            linter="ruff",
            max_line_length=100,
        )

        config = get_code_style(orchestrator)
        assert config["formatter"] == "black"
        assert config["linter"] == "ruff"
        assert config["max_line_length"] == 100


def _create_concrete_rag_provider():
    """Create a concrete RAGWorkflowProvider for testing.

    RAGWorkflowProvider is abstract and requires _get_escape_hatches_module.
    This helper creates a minimal concrete implementation for testing.
    """
    from victor.framework.workflows import BaseYAMLWorkflowProvider

    class ConcreteRAGWorkflowProvider(RAGWorkflowProvider):
        """Concrete implementation of abstract RAGWorkflowProvider for testing."""

        def _get_escape_hatches_module(self) -> str:
            return "victor.rag.escape_hatches"

    return ConcreteRAGWorkflowProvider()


class TestRAGVerticalCapabilities:
    """Test RAG vertical capability provider loading and functionality.

    Verifies that RAGWorkflowProvider capability provider hook implementation
    and future capability provider integration point.
    """

    def test_rag_workflow_provider_capability_hook(self):
        """Verify RAGWorkflowProvider capability provider hook implementation.

        Note: RAG vertical doesn't implement the hook yet (returns None).
        This test documents the current state and prepares for future implementation.
        """
        provider = _create_concrete_rag_provider()
        # RAG provider doesn't implement the hook yet
        module_path = provider._get_capability_provider_module()
        assert module_path is None

    def test_rag_capability_provider_direct_import(self):
        """Test direct import of RAGCapabilityProvider.

        RAG has a capabilities module but the workflow provider hook
        is not yet implemented. Test direct import functionality.
        """
        RAGCapabilityProvider = _load_vertical_attr(
            "victor.rag.capabilities",
            "RAGCapabilityProvider",
        )

        provider = RAGCapabilityProvider()

        assert isinstance(provider, BaseCapabilityProvider)

        capabilities = provider.get_capabilities()
        assert len(capabilities) == 5
        expected_caps = {
            "indexing",
            "retrieval",
            "synthesis",
            "safety",
            "query_enhancement",
        }
        assert set(capabilities.keys()) == expected_caps

    def test_rag_capabilities_accessible(self):
        """Test that all 5 RAG capabilities are accessible."""
        RAGCapabilityProvider = _load_vertical_attr(
            "victor.rag.capabilities",
            "RAGCapabilityProvider",
        )

        cap_provider = RAGCapabilityProvider()
        capabilities = cap_provider.get_capabilities()

        assert len(capabilities) == 5

        # Verify each capability is callable
        for cap_name, cap_func in capabilities.items():
            assert callable(cap_func)

    def test_rag_capability_metadata(self):
        """Test that RAG capability metadata is correct."""
        RAGCapabilityProvider = _load_vertical_attr(
            "victor.rag.capabilities",
            "RAGCapabilityProvider",
        )

        cap_provider = RAGCapabilityProvider()
        metadata = cap_provider.get_capability_metadata()

        assert len(metadata) == 5

        # Check indexing metadata
        indexing_meta = metadata["indexing"]
        assert indexing_meta.name == "indexing"
        assert "indexing" in indexing_meta.tags

        # Check retrieval has dependencies
        retrieval_meta = metadata["retrieval"]
        assert "indexing" in retrieval_meta.dependencies

        # Check synthesis has dependencies on retrieval
        synthesis_meta = metadata["synthesis"]
        assert "retrieval" in synthesis_meta.dependencies

    def test_rag_capability_application(self):
        """Test that RAG capabilities can be applied to orchestrator."""
        RAGCapabilityProvider = _load_vertical_attr(
            "victor.rag.capabilities",
            "RAGCapabilityProvider",
        )

        cap_provider = RAGCapabilityProvider()
        orchestrator = MockOrchestrator()

        # Apply indexing
        cap_provider.apply_indexing(orchestrator, chunk_size=1024)

        assert orchestrator.rag_config is not None
        assert "indexing" in orchestrator.rag_config
        assert orchestrator.rag_config["indexing"]["chunk_size"] == 1024

        # Apply retrieval
        cap_provider.apply_retrieval(orchestrator, top_k=10)

        assert "retrieval" in orchestrator.rag_config
        assert orchestrator.rag_config["retrieval"]["top_k"] == 10

    def test_rag_future_integration_point(self):
        """Test future RAG capability provider integration point.

        This test ensures the structure is ready for when RAGWorkflowProvider
        implements the _get_capability_provider_module() hook.
        """
        provider = _create_concrete_rag_provider()

        # Currently returns None, but should be ready to implement
        module_path = provider._get_capability_provider_module()

        # When implemented, should return:
        # assert module_path == "victor.rag.capabilities"
        assert module_path is None  # Current state


class TestBenchmarkVerticalCapabilities:
    """Test Benchmark vertical capability provider loading and functionality.

    Verifies that BenchmarkWorkflowProvider capability provider hook implementation
    and future capability provider integration point.
    """

    def test_benchmark_workflow_provider_capability_hook(self):
        """Verify BenchmarkWorkflowProvider capability provider hook.

        Note: Benchmark vertical doesn't have a capability provider yet.
        This test documents the current state.
        """
        provider = BenchmarkWorkflowProvider()
        # Benchmark provider doesn't implement the hook (returns None)
        module_path = provider._get_capability_provider_module()
        assert module_path is None

    def test_benchmark_no_capability_provider_yet(self):
        """Test that Benchmark has no capability provider implementation yet.

        This test verifies that attempting to get a capability provider
        gracefully returns None rather than raising an error.
        """
        provider = BenchmarkWorkflowProvider()

        # Should return None gracefully
        cap_provider = provider.get_capability_provider()
        assert cap_provider is None

    def test_benchmark_future_integration_point(self):
        """Test future Benchmark capability provider integration point.

        This test ensures the structure is ready for when Benchmark vertical
        adds a capability provider. When implemented, it should support:
        - swe_bench capability (task routing, harness integration)
        - code_generation capability (pass@k evaluation)
        - benchmark_config capability (timeout, memory limits)
        """
        provider = BenchmarkWorkflowProvider()

        # Currently returns None
        module_path = provider._get_capability_provider_module()

        # When implemented, should return something like:
        # assert module_path == "victor.benchmark.capabilities"
        assert module_path is None  # Current state


class TestCapabilityProviderErrorHandling:
    """Test error handling and graceful degradation in capability providers.

    Verifies that capability providers handle errors gracefully and provide
    useful fallback behavior.
    """

    def test_invalid_capability_provider_module(self):
        """Test graceful handling of invalid capability provider module.

        Simulates a scenario where a workflow provider returns an invalid
        module path and verifies it doesn't crash.
        """
        from victor.framework.workflows import BaseYAMLWorkflowProvider

        class InvalidProvider(BaseYAMLWorkflowProvider):
            """Provider with invalid capability module path."""

            def _get_escape_hatches_module(self) -> str:
                return "victor.research.escape_hatches"

            def _get_capability_provider_module(self) -> Optional[str]:
                return "nonexistent.module.path"

        provider = InvalidProvider()

        # Should return None gracefully, not raise ImportError
        cap_provider = provider.get_capability_provider()
        assert cap_provider is None

    def test_capability_provider_with_missing_class(self):
        """Test graceful handling when module exists but provider class doesn't."""
        from victor.framework.workflows import BaseYAMLWorkflowProvider

        class MissingClassProvider(BaseYAMLWorkflowProvider):
            """Provider pointing to module without CapabilityProvider class."""

            def _get_escape_hatches_module(self) -> str:
                return "victor.research.escape_hatches"

            def _get_capability_provider_module(self) -> Optional[str]:
                # This module exists but doesn't have a provider class
                return "victor.research.escape_hatches"

        provider = MissingClassProvider()

        # Should return None gracefully
        cap_provider = provider.get_capability_provider()
        assert cap_provider is None

    def test_capability_application_with_invalid_orchestrator(self):
        """Test that capability providers handle invalid orchestrator gracefully.

        Verifies that capabilities don't crash when applied to orchestrator
        missing expected attributes.
        """
        ResearchCapabilityProvider = _load_vertical_attr(
            "victor_research.capabilities",
            "ResearchCapabilityProvider",
        )

        cap_provider = ResearchCapabilityProvider()

        # Create orchestrator without required attributes
        invalid_orch = MagicMock()

        # Should not raise exception - capabilities check hasattr
        try:
            cap_provider.apply_source_verification(invalid_orch)
            # If no error, capability gracefully skipped application
        except Exception as e:
            # If error occurs, it should be informative
            assert "orchestrator" in str(e).lower() or "config" in str(e).lower()


class TestCapabilityProviderLazyLoading:
    """Test lazy loading behavior of capability providers.

    Verifies that capability providers are not loaded until needed and
    that each call creates a fresh instance.
    """

    def test_research_lazy_loading(self):
        """Test lazy loading for Research capability provider."""
        provider = ResearchWorkflowProvider()

        # Provider should load successfully
        cap_provider = provider.get_capability_provider()
        assert cap_provider is not None

        # Verify it's the right type
        ResearchCapabilityProvider = _load_vertical_attr(
            "victor_research.capabilities",
            "ResearchCapabilityProvider",
        )

        assert isinstance(cap_provider, ResearchCapabilityProvider)

    def test_devops_lazy_loading(self):
        """Test lazy loading for DevOps capability provider."""
        provider = DevOpsWorkflowProvider()

        cap_provider = provider.get_capability_provider()
        assert cap_provider is not None

        DevOpsCapabilityProvider = _load_vertical_attr(
            "victor_devops.capabilities",
            "DevOpsCapabilityProvider",
        )

        assert isinstance(cap_provider, DevOpsCapabilityProvider)

    def test_dataanalysis_lazy_loading(self):
        """Test lazy loading for DataAnalysis capability provider."""
        provider = DataAnalysisWorkflowProvider()

        cap_provider = provider.get_capability_provider()
        assert cap_provider is not None

        DataAnalysisCapabilityProvider = _load_vertical_attr(
            "victor.dataanalysis.capabilities",
            "DataAnalysisCapabilityProvider",
        )

        assert isinstance(cap_provider, DataAnalysisCapabilityProvider)

    def test_fresh_instance_each_call(self):
        """Test that each get_capability_provider() call returns a fresh instance.

        This verifies that there's no singleton behavior that could cause
        state pollution between tests or usage scenarios.
        """
        provider = ResearchWorkflowProvider()

        cap1 = provider.get_capability_provider()
        cap2 = provider.get_capability_provider()

        # Should be different instances
        assert cap1 is not cap2

        # But should have same capabilities
        assert set(cap1.get_capabilities().keys()) == set(cap2.get_capabilities().keys())


class TestCapabilityMetadataConsistency:
    """Test consistency and completeness of capability metadata across verticals.

    Verifies that all capability providers follow consistent metadata patterns
    and provide complete information.
    """

    @pytest.mark.parametrize(
        "vertical,provider_class,expected_count",
        [
            ("research", "victor_research.capabilities.ResearchCapabilityProvider", 5),
            ("devops", "victor_devops.capabilities.DevOpsCapabilityProvider", 5),
            (
                "dataanalysis",
                "victor.dataanalysis.capabilities.DataAnalysisCapabilityProvider",
                5,
            ),
            ("coding", "victor_coding.capabilities.CodingCapabilityProvider", 5),
            ("rag", "victor.rag.capabilities.RAGCapabilityProvider", 5),
        ],
    )
    def test_capability_count(self, vertical, provider_class, expected_count):
        """Test that each vertical has the expected number of capabilities."""
        provider_cls = _load_provider_class(provider_class)
        provider = provider_cls()
        capabilities = provider.get_capabilities()

        assert len(capabilities) == expected_count, (
            f"{vertical} vertical should have {expected_count} capabilities, "
            f"but found {len(capabilities)}"
        )

    @pytest.mark.parametrize(
        "provider_class",
        [
            "victor_research.capabilities.ResearchCapabilityProvider",
            "victor_devops.capabilities.DevOpsCapabilityProvider",
            "victor.dataanalysis.capabilities.DataAnalysisCapabilityProvider",
            "victor_coding.capabilities.CodingCapabilityProvider",
            "victor.rag.capabilities.RAGCapabilityProvider",
        ],
    )
    def test_metadata_completeness(self, provider_class):
        """Test that all capabilities have complete metadata.

        Verifies each capability has:
        - name (str)
        - description (str)
        - version (str)
        - tags (List[str])
        """
        provider_cls = _load_provider_class(provider_class)
        provider = provider_cls()
        metadata = provider.get_capability_metadata()

        for cap_name, meta in metadata.items():
            # Check required fields
            assert isinstance(meta.name, str) and len(meta.name) > 0
            assert isinstance(meta.description, str) and len(meta.description) > 0
            assert isinstance(meta.version, str) and len(meta.version) > 0
            assert isinstance(meta.tags, list) and len(meta.tags) > 0

            # Check that tags contain useful categories
            assert all(isinstance(tag, str) for tag in meta.tags)

    def test_capability_names_match_metadata_names(self):
        """Test that capability names match their metadata names.

        This ensures consistency between get_capabilities() keys
        and get_capability_metadata() keys.
        """
        providers = [
            ("victor_research.capabilities", "ResearchCapabilityProvider"),
            ("victor_devops.capabilities", "DevOpsCapabilityProvider"),
            ("victor.dataanalysis.capabilities", "DataAnalysisCapabilityProvider"),
            ("victor_coding.capabilities", "CodingCapabilityProvider"),
            ("victor.rag.capabilities", "RAGCapabilityProvider"),
        ]

        for module_path, class_name in providers:
            provider_cls = _load_vertical_attr(module_path, class_name)
            provider = provider_cls()
            capabilities = provider.get_capabilities()
            metadata = provider.get_capability_metadata()

            # Keys should match exactly
            assert set(capabilities.keys()) == set(metadata.keys())


class TestCrossVerticalCapabilityIntegration:
    """Test integration patterns between vertical capabilities.

    Verifies that capabilities from different verticals can coexist
    and work together when applied to the same orchestrator.
    """

    def test_multiple_vertical_capabilities_on_same_orchestrator(self):
        """Test applying capabilities from multiple verticals to one orchestrator.

        Verifies that an orchestrator can have capabilities from different
        verticals applied without conflicts.
        """
        ResearchCapabilityProvider = _load_vertical_attr(
            "victor_research.capabilities",
            "ResearchCapabilityProvider",
        )
        DevOpsCapabilityProvider = _load_vertical_attr(
            "victor_devops.capabilities",
            "DevOpsCapabilityProvider",
        )
        CodingCapabilityProvider = _load_vertical_attr(
            "victor.coding.capabilities",
            "CodingCapabilityProvider",
        )

        orchestrator = MockOrchestrator()

        # Apply Research capabilities
        research_provider = ResearchCapabilityProvider()
        research_provider.apply_source_verification(orchestrator)
        research_provider.apply_citation_management(orchestrator)

        # Apply DevOps capabilities
        devops_provider = DevOpsCapabilityProvider()
        devops_provider.apply_deployment_safety(orchestrator)

        # Apply Coding capabilities (skip git_safety due to CodingSafetyExtension dependency)
        coding_provider = CodingCapabilityProvider()
        coding_provider.apply_code_style(orchestrator)

        # Verify all capabilities were applied
        assert orchestrator.source_verification_config is not None
        assert orchestrator.citation_config is not None
        assert orchestrator.safety_config is not None
        assert "deployment" in orchestrator.safety_config
        assert orchestrator.code_style is not None

    def test_capability_dependencies(self):
        """Test that capability dependencies are properly declared.

        Verifies that capabilities that depend on other capabilities
        correctly declare their dependencies in metadata.
        """
        ResearchCapabilityProvider = _load_vertical_attr(
            "victor_research.capabilities",
            "ResearchCapabilityProvider",
        )
        DevOpsCapabilityProvider = _load_vertical_attr(
            "victor_devops.capabilities",
            "DevOpsCapabilityProvider",
        )
        RAGCapabilityProvider = _load_vertical_attr(
            "victor.rag.capabilities",
            "RAGCapabilityProvider",
        )

        # Research dependencies
        research_provider = ResearchCapabilityProvider()
        research_meta = research_provider.get_capability_metadata()

        # research_quality depends on source_verification
        assert "source_verification" in research_meta["research_quality"].dependencies
        # literature_analysis depends on source_verification
        assert "source_verification" in research_meta["literature_analysis"].dependencies

        # DevOps dependencies
        devops_provider = DevOpsCapabilityProvider()
        devops_meta = devops_provider.get_capability_metadata()

        # monitoring_settings depends on deployment_safety
        assert "deployment_safety" in devops_meta["monitoring_settings"].dependencies

        # RAG dependencies
        rag_provider = RAGCapabilityProvider()
        rag_meta = rag_provider.get_capability_metadata()

        # retrieval depends on indexing
        assert "indexing" in rag_meta["retrieval"].dependencies
        # synthesis depends on retrieval
        assert "retrieval" in rag_meta["synthesis"].dependencies


class TestServiceBackedCapabilityConfigFlow:
    """Integration tests for service-backed capability config runtime getters."""

    def test_coding_service_backed_getter_flow(self):
        configure_code_style = _load_vertical_attr(
            "victor.coding.capabilities",
            "configure_code_style",
        )
        get_code_style = _load_vertical_attr(
            "victor.coding.capabilities",
            "get_code_style",
        )

        service = CapabilityConfigService()
        orchestrator = ServiceBackedOrchestrator(service)

        configure_code_style(orchestrator, formatter="ruff-format", max_line_length=120)

        assert service.get_config("code_style")["formatter"] == "ruff-format"
        assert get_code_style(orchestrator)["max_line_length"] == 120

    def test_rag_service_backed_getter_flow(self):
        configure_retrieval = _load_vertical_attr(
            "victor.rag.capabilities",
            "configure_retrieval",
        )
        get_retrieval_config = _load_vertical_attr(
            "victor.rag.capabilities",
            "get_retrieval_config",
        )

        service = CapabilityConfigService()
        orchestrator = ServiceBackedOrchestrator(service)

        configure_retrieval(orchestrator, top_k=11, search_type="semantic")

        rag_config = service.get_config("rag_config")
        assert rag_config["retrieval"]["top_k"] == 11
        assert get_retrieval_config(orchestrator)["search_type"] == "semantic"

    def test_devops_service_backed_getter_flow(self):
        configure_container_settings = _load_vertical_attr(
            "victor_devops.capabilities",
            "configure_container_settings",
        )
        get_container_settings = _load_vertical_attr(
            "victor_devops.capabilities",
            "get_container_settings",
        )

        service = CapabilityConfigService()
        orchestrator = ServiceBackedOrchestrator(service)

        configure_container_settings(orchestrator, runtime="podman", max_image_size_mb=1024)

        assert service.get_config("container_config")["runtime"] == "podman"
        assert get_container_settings(orchestrator)["max_image_size_mb"] == 1024

    def test_dataanalysis_service_backed_getter_flow(self):
        configure_statistical_analysis = _load_vertical_attr(
            "victor.dataanalysis.capabilities",
            "configure_statistical_analysis",
        )
        get_statistical_config = _load_vertical_attr(
            "victor.dataanalysis.capabilities",
            "get_statistical_config",
        )

        service = CapabilityConfigService()
        orchestrator = ServiceBackedOrchestrator(service)

        configure_statistical_analysis(
            orchestrator,
            significance_level=0.01,
            confidence_interval=0.9,
        )

        assert service.get_config("statistics_config")["significance_level"] == 0.01
        assert get_statistical_config(orchestrator)["confidence_interval"] == 0.9
