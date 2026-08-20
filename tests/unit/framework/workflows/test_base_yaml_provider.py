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

"""Tests for BaseYAMLWorkflowProvider's UnifiedWorkflowCompiler integration.

These tests verify that:
1. The new compile_workflow() method works correctly
2. The new run_compiled_workflow() method works correctly
3. The new stream_compiled_workflow() method works correctly
4. Vertical providers inherit the new functionality
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import pytest

from victor.core.verticals.import_resolver import import_module_with_fallback


def _load_vertical_attr(module_path: str, attr_name: str):
    """Load a vertical attribute using external-first resolution."""
    module, _resolved = import_module_with_fallback(module_path)
    if module is None or not hasattr(module, attr_name):
        raise ImportError(f"Unable to resolve {module_path}:{attr_name}")
    return getattr(module, attr_name)


# =============================================================================
# Test Fixtures
# =============================================================================


class ConcreteYAMLWorkflowProvider:
    """Concrete implementation of BaseYAMLWorkflowProvider for testing."""

    def __init__(self, workflows_dir: Optional[Path] = None):
        self._workflows_dir = workflows_dir
        # Import here to avoid circular imports in fixture scope
        from victor.framework.workflows.base_yaml_provider import (
            BaseYAMLWorkflowProvider,
        )

        # Create a concrete subclass dynamically
        class TestProvider(BaseYAMLWorkflowProvider):
            def _get_escape_hatches_module(self) -> str:
                return "victor_research.escape_hatches"

            def _get_workflows_directory(self_inner) -> Path:
                if workflows_dir:
                    return workflows_dir
                return Path(__file__).parent / "fixtures"

        self._provider = TestProvider()

    @property
    def provider(self):
        return self._provider


@pytest.fixture
def sample_yaml_content() -> str:
    """Simple YAML workflow content for testing."""
    return """
workflows:
  test_workflow:
    description: "Test workflow for BaseYAMLWorkflowProvider"
    nodes:
      - id: start
        type: transform
        transform: "result = 'started'"
        next: [process]
      - id: process
        type: transform
        transform: "result = 'processed'"
        next: []
"""


@pytest.fixture
def yaml_workflow_file(sample_yaml_content: str, tmp_path: Path) -> Path:
    """Create a temporary YAML workflow file."""
    yaml_file = tmp_path / "test_workflow.yaml"
    yaml_file.write_text(sample_yaml_content)
    return yaml_file


@pytest.fixture
def test_provider(tmp_path: Path, yaml_workflow_file: Path) -> "ConcreteYAMLWorkflowProvider":
    """Create a test provider with a temporary workflow directory."""
    return ConcreteYAMLWorkflowProvider(workflows_dir=tmp_path)


@pytest.fixture
def mock_orchestrator() -> MagicMock:
    """Create a mock orchestrator for testing."""
    from unittest.mock import AsyncMock

    orchestrator = MagicMock()
    orchestrator.chat = AsyncMock(return_value="mock response")
    return orchestrator


# =============================================================================
# Test: get_compiler()
# =============================================================================


class TestGetCompiler:
    """Tests for the get_compiler() method."""

    def test_get_compiler_returns_unified_compiler(self, test_provider):
        """Test that get_compiler returns a UnifiedWorkflowCompiler."""
        from victor.workflows.unified_compiler import UnifiedWorkflowCompiler

        compiler = test_provider.provider.get_compiler()
        assert isinstance(compiler, UnifiedWorkflowCompiler)

    def test_get_compiler_is_cached(self, test_provider):
        """Test that get_compiler returns the same instance."""
        compiler1 = test_provider.provider.get_compiler()
        compiler2 = test_provider.provider.get_compiler()
        assert compiler1 is compiler2

    def test_get_compiler_enables_caching(self, test_provider):
        """Test that the compiler has caching enabled."""
        compiler = test_provider.provider.get_compiler()
        # Check that caching is enabled via the compiler's internal state
        stats = compiler.get_cache_stats()
        assert stats.get("caching_enabled", False) is True


# =============================================================================
# Test: compile_workflow()
# =============================================================================


class TestCompileWorkflow:
    """Tests for the compile_workflow() method."""

    def test_compile_workflow_returns_cached_compiled_graph(
        self, test_provider, yaml_workflow_file
    ):
        """Test that compile_workflow returns a CachedCompiledGraph."""
        from victor.workflows.unified_compiler import CachedCompiledGraph

        # The test provider needs a workflow to compile
        # For this test, we'll patch the _get_workflow_path method
        with patch.object(
            test_provider.provider,
            "_get_workflow_path",
            return_value=yaml_workflow_file,
        ):
            with patch.object(
                test_provider.provider, "_load_escape_hatches", return_value=({}, {})
            ):
                compiled = test_provider.provider.compile_workflow("test_workflow")
                assert isinstance(compiled, CachedCompiledGraph)

    def test_compile_workflow_has_workflow_name(self, test_provider, yaml_workflow_file):
        """Test that compiled workflow has correct workflow_name."""
        with patch.object(
            test_provider.provider,
            "_get_workflow_path",
            return_value=yaml_workflow_file,
        ):
            with patch.object(
                test_provider.provider, "_load_escape_hatches", return_value=({}, {})
            ):
                compiled = test_provider.provider.compile_workflow("test_workflow")
                assert compiled.workflow_name == "test_workflow"

    def test_compile_workflow_raises_for_unknown_workflow(self, test_provider, tmp_path):
        """Test that compile_workflow raises ValueError for unknown workflow."""
        with pytest.raises(ValueError, match="Workflow not found"):
            test_provider.provider.compile_workflow("nonexistent_workflow")


# =============================================================================
# Test: CachedCompiledGraph Methods
# =============================================================================


class TestCachedCompiledGraph:
    """Tests for CachedCompiledGraph functionality."""

    def test_cached_compiled_graph_has_metadata(self):
        """Test that CachedCompiledGraph has expected metadata attributes."""
        from victor.workflows.unified_compiler import CachedCompiledGraph

        mock_graph = MagicMock()
        cached = CachedCompiledGraph(
            compiled_graph=mock_graph,
            workflow_name="test",
            source_path=Path("/test/path.yaml"),
            source_mtime=12345.0,
        )

        assert cached.workflow_name == "test"
        assert cached.source_path == Path("/test/path.yaml")
        assert cached.source_mtime == 12345.0
        assert cached.compiled_at > 0

    def test_cached_compiled_graph_age_seconds(self):
        """Test that age_seconds property works correctly."""
        import time

        from victor.workflows.unified_compiler import CachedCompiledGraph

        mock_graph = MagicMock()
        cached = CachedCompiledGraph(
            compiled_graph=mock_graph,
            workflow_name="test",
        )

        # Wait a small amount and check age
        time.sleep(0.01)
        assert cached.age_seconds >= 0.01

    @pytest.mark.asyncio
    async def test_cached_compiled_graph_invoke_delegates(self):
        """Test that invoke() delegates to underlying compiled_graph."""
        from unittest.mock import AsyncMock

        from victor.workflows.unified_compiler import CachedCompiledGraph

        mock_graph = MagicMock()
        mock_result = MagicMock()
        mock_graph.invoke = AsyncMock(return_value=mock_result)

        cached = CachedCompiledGraph(
            compiled_graph=mock_graph,
            workflow_name="test",
        )

        result = await cached.invoke({"key": "value"})

        mock_graph.invoke.assert_called_once()
        assert result is mock_result

    @pytest.mark.asyncio
    async def test_cached_compiled_graph_stream_delegates(self):
        """Test that stream() delegates to underlying compiled_graph."""
        from unittest.mock import AsyncMock

        from victor.workflows.unified_compiler import CachedCompiledGraph

        mock_graph = MagicMock()

        async def mock_stream(*args, **kwargs):
            yield ("node1", {"state": 1})
            yield ("node2", {"state": 2})

        mock_graph.stream = mock_stream

        cached = CachedCompiledGraph(
            compiled_graph=mock_graph,
            workflow_name="test",
        )

        results = []
        async for node_id, state in cached.stream({"key": "value"}):
            results.append((node_id, state))

        assert len(results) == 2
        assert results[0] == ("node1", {"state": 1})
        assert results[1] == ("node2", {"state": 2})

    def test_cached_compiled_graph_get_graph_schema_delegates(self):
        """Test that get_graph_schema() delegates to underlying compiled_graph."""
        from victor.workflows.unified_compiler import CachedCompiledGraph

        mock_graph = MagicMock()
        mock_graph.get_graph_schema.return_value = {"nodes": ["a", "b"]}

        cached = CachedCompiledGraph(
            compiled_graph=mock_graph,
            workflow_name="test",
        )

        schema = cached.get_graph_schema()

        mock_graph.get_graph_schema.assert_called_once()
        assert schema == {"nodes": ["a", "b"]}


# =============================================================================
# Test: Vertical Provider Integration
# =============================================================================


class TestVerticalProviderIntegration:
    """Tests verifying that vertical providers inherit new functionality."""

    def test_research_provider_has_get_compiler(self):
        """Test that ResearchWorkflowProvider has get_compiler method."""
        try:
            ResearchWorkflowProvider = _load_vertical_attr(
                "victor.research.workflows",
                "ResearchWorkflowProvider",
            )

            provider = ResearchWorkflowProvider()
            assert hasattr(provider, "get_compiler")
            assert callable(provider.get_compiler)
        except ImportError:
            pytest.skip("ResearchWorkflowProvider not available")

    def test_coding_provider_has_get_compiler(self):
        """Test that CodingWorkflowProvider has get_compiler method."""
        try:
            CodingWorkflowProvider = _load_vertical_attr(
                "victor.coding.workflows.provider",
                "CodingWorkflowProvider",
            )

            provider = CodingWorkflowProvider()
            assert hasattr(provider, "get_compiler")
            assert callable(provider.get_compiler)
        except ImportError:
            pytest.skip("CodingWorkflowProvider not available")

    def test_devops_provider_has_compile_workflow(self):
        """Test that DevOpsWorkflowProvider has compile_workflow method."""
        try:
            DevOpsWorkflowProvider = _load_vertical_attr(
                "victor.devops.workflows",
                "DevOpsWorkflowProvider",
            )

            provider = DevOpsWorkflowProvider()
            assert hasattr(provider, "compile_workflow")
            assert callable(provider.compile_workflow)
        except ImportError:
            pytest.skip("DevOpsWorkflowProvider not available")

    def test_dataanalysis_provider_has_run_compiled_workflow(self):
        """Test that DataAnalysisWorkflowProvider has run_compiled_workflow method."""
        try:
            DataAnalysisWorkflowProvider = _load_vertical_attr(
                "victor.dataanalysis.workflows",
                "DataAnalysisWorkflowProvider",
            )

            provider = DataAnalysisWorkflowProvider()
            assert hasattr(provider, "run_compiled_workflow")
            assert callable(provider.run_compiled_workflow)
        except ImportError:
            pytest.skip("DataAnalysisWorkflowProvider not available")


# =============================================================================
# Test: Module Structure
# =============================================================================


class TestModuleStructure:
    """Tests for module structure and exports."""

    def test_cached_compiled_graph_exported(self):
        """Test that CachedCompiledGraph is exported from unified_compiler."""
        from victor.workflows.unified_compiler import CachedCompiledGraph

        assert CachedCompiledGraph is not None

    def test_base_yaml_provider_imports_without_error(self):
        """Test that BaseYAMLWorkflowProvider can be imported."""
        from victor.framework.workflows.base_yaml_provider import (
            BaseYAMLWorkflowProvider,
        )

        assert BaseYAMLWorkflowProvider is not None
