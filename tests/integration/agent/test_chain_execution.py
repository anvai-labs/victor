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

"""Integration tests for chain execution system.

Tests the integration between:
- ChainRegistry and @chain decorator for registration
- LCEL composition primitives (Runnable, RunnableSequence, etc.)
- YAML workflow chain references (handler: chain:vertical:name)
- End-to-end chain execution through the workflow executor
"""

from __future__ import annotations

import asyncio
import pytest
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

from victor.framework.chain_registry import (
    ChainRegistry,
    ChainMetadata,
    get_chain_registry,
    register_chain,
    get_chain,
    create_chain,
    chain,
    reset_chain_registry,
)
from victor.tools.composition import (
    Runnable,
    RunnableConfig,
    RunnableSequence,
    RunnableParallel,
    RunnableBranch,
    RunnableLambda,
    RunnablePassthrough,
    as_runnable,
    chain as chain_helper,
    parallel,
    branch,
    extract_output,
    map_keys,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def reset_registry_fixture():
    """Reset the chain registry before and after each test."""
    reset_chain_registry()
    yield
    reset_chain_registry()


@pytest.fixture
def mock_tool():
    """Create a mock tool that behaves like a Victor tool."""
    from victor.tools.base import BaseTool, ToolResult

    class MockTool(BaseTool):
        name = "mock_tool"
        description = "A mock tool for testing"
        parameters = {"type": "object", "properties": {"input": {"type": "string"}}}

        async def execute(self, ctx: Dict[str, Any], **kwargs) -> ToolResult:
            return ToolResult(
                success=True,
                output={"processed": kwargs.get("input", "default")},
            )

    return MockTool()


# =============================================================================
# Integration: @chain Decorator Registration
# =============================================================================


@pytest.mark.integration
class TestChainDecoratorRegistration:
    """Integration tests for @chain decorator registration."""

    def test_chain_decorator_registers_factory_in_global_registry(self):
        """@chain decorator should register factory in global registry."""

        @chain("coding:read_and_analyze", description="Read and analyze code files")
        def read_and_analyze_chain():
            return RunnableLambda(lambda x: {"analyzed": x.get("content", "")})

        registry = get_chain_registry()

        # Verify registration
        assert registry.has("read_and_analyze", vertical="coding")

        # Verify metadata
        metadata = registry.get_metadata("read_and_analyze", vertical="coding")
        assert metadata is not None
        assert metadata.description == "Read and analyze code files"
        assert metadata.is_factory is True

    def test_chain_decorator_creates_working_runnable(self):
        """@chain decorated factory should create working Runnable."""

        @chain("testing:uppercase", description="Uppercase transformer")
        def uppercase_chain():
            return RunnableLambda(lambda x: {"result": x.get("text", "").upper()})

        # Create chain from registry
        runnable = create_chain("testing:uppercase")

        assert runnable is not None
        assert isinstance(runnable, Runnable)

    @pytest.mark.asyncio
    async def test_chain_decorator_factory_creates_executable_chain(self):
        """Chain created from decorated factory should execute correctly."""

        @chain("testing:double", description="Double a number")
        def double_chain():
            return RunnableLambda(lambda x: {"result": x.get("value", 0) * 2})

        runnable = create_chain("testing:double")
        result = await runnable.invoke({"value": 21})

        assert result["result"] == 42

    def test_multiple_chains_same_vertical(self):
        """Multiple chains in the same vertical should coexist."""

        @chain("devops:deploy", description="Deploy application")
        def deploy_chain():
            return RunnableLambda(lambda x: {"status": "deployed"})

        @chain("devops:rollback", description="Rollback deployment")
        def rollback_chain():
            return RunnableLambda(lambda x: {"status": "rolled_back"})

        registry = get_chain_registry()
        devops_factories = registry.list_factories(vertical="devops")

        assert len(devops_factories) == 2
        assert "devops:deploy" in devops_factories
        assert "devops:rollback" in devops_factories

    def test_chains_different_verticals(self):
        """Chains in different verticals should be properly namespaced."""

        @chain("coding:format", description="Format code")
        def coding_format():
            return RunnableLambda(lambda x: {"formatted": True})

        @chain("research:format", description="Format citations")
        def research_format():
            return RunnableLambda(lambda x: {"citations_formatted": True})

        registry = get_chain_registry()

        # Both should exist independently
        assert registry.has("format", vertical="coding")
        assert registry.has("format", vertical="research")

        # Can create both
        coding_chain = create_chain("format", vertical="coding")
        research_chain = create_chain("format", vertical="research")

        assert coding_chain is not None
        assert research_chain is not None


# =============================================================================
# Integration: Chain Retrieval
# =============================================================================


@pytest.mark.integration
class TestChainRetrieval:
    """Integration tests for chain retrieval patterns."""

    def test_get_chain_by_name_only(self):
        """Can retrieve chains registered without vertical."""
        mock_chain = RunnableLambda(lambda x: x)
        register_chain("global_transform", mock_chain, description="Global transform")

        retrieved = get_chain("global_transform")
        assert retrieved is mock_chain

    def test_get_chain_by_name_and_vertical(self):
        """Can retrieve chains with vertical namespace."""
        mock_chain = RunnableLambda(lambda x: x)
        register_chain(
            "process",
            mock_chain,
            vertical="dataanalysis",
            description="Process data",
        )

        retrieved = get_chain("process", vertical="dataanalysis")
        assert retrieved is mock_chain

    def test_get_chain_returns_none_for_missing(self):
        """get_chain returns None for non-existent chains."""
        result = get_chain("nonexistent")
        assert result is None

        result_with_vertical = get_chain("nonexistent", vertical="coding")
        assert result_with_vertical is None

    def test_create_chain_from_factory(self):
        """create_chain creates fresh instances from factory."""
        call_count = [0]

        @chain("testing:counter")
        def counter_factory():
            call_count[0] += 1
            return RunnableLambda(lambda x: {"count": call_count[0]})

        # Each create should invoke factory
        chain1 = create_chain("testing:counter")
        chain2 = create_chain("testing:counter")

        assert call_count[0] == 2

    def test_create_chain_with_colon_format(self):
        """create_chain supports 'vertical:name' format."""

        @chain("rag:search")
        def search_factory():
            return RunnableLambda(lambda x: {"results": []})

        # Both formats should work
        chain1 = create_chain("rag:search")
        chain2 = create_chain("search", vertical="rag")

        assert chain1 is not None
        assert chain2 is not None


# =============================================================================
# Integration: Chain Factory Patterns
# =============================================================================


@pytest.mark.integration
class TestChainFactoryPatterns:
    """Integration tests for chain factory creation patterns."""

    @pytest.mark.asyncio
    async def test_factory_creates_sequence_chain(self):
        """Factory can create RunnableSequence chains."""

        @chain("testing:pipeline", description="Pipeline chain")
        def pipeline_factory():
            step1 = RunnableLambda(lambda x: {**x, "step1": True})
            step2 = RunnableLambda(lambda x: {**x, "step2": True})
            step3 = RunnableLambda(lambda x: {**x, "step3": True})
            return step1 | step2 | step3

        pipeline = create_chain("testing:pipeline")
        result = await pipeline.invoke({"initial": "data"})

        assert result["initial"] == "data"
        assert result["step1"] is True
        assert result["step2"] is True
        assert result["step3"] is True

    @pytest.mark.asyncio
    async def test_factory_creates_parallel_chain(self):
        """Factory can create RunnableParallel chains."""

        @chain("testing:parallel_analysis", description="Parallel analysis")
        def parallel_factory():
            return RunnableParallel(
                syntax=RunnableLambda(lambda x: {"syntax_score": 0.9}),
                security=RunnableLambda(lambda x: {"security_score": 0.8}),
                performance=RunnableLambda(lambda x: {"performance_score": 0.7}),
            )

        parallel_chain = create_chain("testing:parallel_analysis")
        result = await parallel_chain.invoke({"code": "print('hello')"})

        assert "syntax" in result
        assert "security" in result
        assert "performance" in result
        assert result["syntax"]["syntax_score"] == 0.9

    @pytest.mark.asyncio
    async def test_factory_creates_branch_chain(self):
        """Factory can create RunnableBranch chains."""

        @chain("testing:router", description="Language router")
        def router_factory():
            return RunnableBranch(
                (
                    lambda x: x.get("lang") == "python",
                    RunnableLambda(lambda x: {"linter": "pylint"}),
                ),
                (
                    lambda x: x.get("lang") == "javascript",
                    RunnableLambda(lambda x: {"linter": "eslint"}),
                ),
                default=RunnableLambda(lambda x: {"linter": "generic"}),
            )

        router = create_chain("testing:router")

        python_result = await router.invoke({"lang": "python"})
        assert python_result["linter"] == "pylint"

        js_result = await router.invoke({"lang": "javascript"})
        assert js_result["linter"] == "eslint"

        other_result = await router.invoke({"lang": "rust"})
        assert other_result["linter"] == "generic"

    @pytest.mark.asyncio
    async def test_factory_creates_complex_composed_chain(self):
        """Factory can create complex composed chains with multiple patterns."""

        @chain("coding:full_review", description="Complete code review pipeline")
        def full_review_factory():
            # Initial validation
            validate = RunnableLambda(lambda x: {**x, "validated": True})

            # Parallel analysis
            analyze = RunnableParallel(
                style=RunnableLambda(lambda x: {"issues": 0}),
                complexity=RunnableLambda(lambda x: {"score": 5}),
            )

            # Format results
            format_result = RunnableLambda(
                lambda x: {
                    "review_complete": True,
                    "style_issues": x["style"]["issues"],
                    "complexity_score": x["complexity"]["score"],
                }
            )

            return validate | analyze | format_result

        review_chain = create_chain("coding:full_review")
        result = await review_chain.invoke({"code": "def foo(): pass"})

        assert result["review_complete"] is True
        assert result["style_issues"] == 0
        assert result["complexity_score"] == 5


# =============================================================================
# Integration: YAML Workflow Chain References
# =============================================================================


@pytest.mark.integration
class TestYAMLWorkflowChainReferences:
    """Integration tests for YAML workflows referencing chains.

    These tests verify the chain handler pattern used by workflows when
    referencing chains via 'handler: chain:vertical:name' syntax.
    """

    @pytest.mark.asyncio
    async def test_chain_handler_prefix_parsing(self):
        """Chain handler prefix is correctly recognized and parsed."""
        from victor.workflows.compute_registry import CHAIN_HANDLER_PREFIX

        # Register a chain
        @chain("coding:analyze_imports")
        def analyze_imports_factory():
            return RunnableLambda(lambda x: {"imports": ["os", "sys"], "code": x.get("code", "")})

        # The handler string for YAML workflow would be: "chain:coding:analyze_imports"
        handler = f"{CHAIN_HANDLER_PREFIX}coding:analyze_imports"
        assert handler == "chain:coding:analyze_imports"
        assert handler.startswith(CHAIN_HANDLER_PREFIX)

        # Extract chain name from handler
        chain_name = handler[len(CHAIN_HANDLER_PREFIX) :]
        assert chain_name == "coding:analyze_imports"

        # Verify the chain can be created
        runnable = create_chain(chain_name)
        assert runnable is not None

        # Execute it
        result = await runnable.invoke({"code": "import os"})
        assert result["imports"] == ["os", "sys"]

    @pytest.mark.asyncio
    async def test_chain_registry_integration_for_workflow_handler(self):
        """Chain registry correctly resolves chains for workflow handlers."""

        # Register chains that would be referenced from YAML workflows
        @chain("dataanalysis:preprocess")
        def preprocess_factory():
            return RunnableLambda(lambda x: {"preprocessed": True, "rows": len(x.get("data", []))})

        @chain("dataanalysis:analyze")
        def analyze_factory():
            return RunnableLambda(
                lambda x: {
                    "analyzed": True,
                    "result": f"Analyzed {x.get('rows', 0)} rows",
                }
            )

        # Simulate what the workflow executor does when resolving chain handlers
        registry = get_chain_registry()

        # Resolve using the vertical:name format
        preprocess_chain = registry.create("preprocess", vertical="dataanalysis")
        analyze_chain = registry.create("analyze", vertical="dataanalysis")

        assert preprocess_chain is not None
        assert analyze_chain is not None

        # Execute in sequence like a workflow would
        step1_result = await preprocess_chain.invoke({"data": [1, 2, 3, 4, 5]})
        step2_result = await analyze_chain.invoke(step1_result)

        assert step1_result["preprocessed"] is True
        assert step1_result["rows"] == 5
        assert step2_result["analyzed"] is True
        assert "5 rows" in step2_result["result"]

    @pytest.mark.asyncio
    async def test_chain_not_found_for_workflow_handler(self):
        """Non-existent chain returns None from registry."""
        registry = get_chain_registry()

        # Try to create a non-existent chain
        missing_chain = registry.create("nonexistent", vertical="missing")
        assert missing_chain is None

        # Verify has() returns False
        assert registry.has("nonexistent", vertical="missing") is False

    @pytest.mark.asyncio
    async def test_workflow_style_chain_composition(self):
        """Chains can be composed in workflow-like patterns."""

        # Register chains that simulate a workflow pipeline
        @chain("research:gather")
        def gather_factory():
            return RunnableLambda(lambda x: {**x, "sources": ["source1", "source2", "source3"]})

        @chain("research:filter")
        def filter_factory():
            return RunnableLambda(
                lambda x: {
                    **x,
                    "filtered_sources": [s for s in x.get("sources", []) if "1" in s or "2" in s],
                }
            )

        @chain("research:summarize")
        def summarize_factory():
            return RunnableLambda(
                lambda x: {
                    "summary": f"Found {len(x.get('filtered_sources', []))} relevant sources",
                    "sources": x.get("filtered_sources", []),
                }
            )

        # Create a workflow-like chain by composing registered chains
        gather = create_chain("research:gather")
        filter_chain = create_chain("research:filter")
        summarize = create_chain("research:summarize")

        # Compose them as a workflow would
        workflow_chain = gather | filter_chain | summarize

        result = await workflow_chain.invoke({"query": "test"})

        assert result["summary"] == "Found 2 relevant sources"
        assert result["sources"] == ["source1", "source2"]


# =============================================================================
# Integration: End-to-End Chain Execution
# =============================================================================


@pytest.mark.integration
class TestEndToEndChainExecution:
    """End-to-end integration tests for chain execution."""

    @pytest.mark.asyncio
    async def test_simple_chain_execution_end_to_end(self):
        """Simple chain executes end-to-end correctly."""

        @chain("e2e:simple", description="Simple end-to-end chain")
        def simple_chain():
            return RunnableLambda(lambda x: {"greeting": f"Hello, {x.get('name', 'World')}!"})

        # Create and execute
        runnable = create_chain("e2e:simple")
        result = await runnable.invoke({"name": "Victor"})

        assert result["greeting"] == "Hello, Victor!"

    @pytest.mark.asyncio
    async def test_chained_transformations_end_to_end(self):
        """Multi-step chain transformations work end-to-end."""

        @chain("e2e:multi_step", description="Multi-step processing")
        def multi_step_chain():
            normalize = RunnableLambda(lambda x: {"text": x.get("raw", "").strip().lower()})
            tokenize = RunnableLambda(lambda x: {"tokens": x.get("text", "").split()})
            count = RunnableLambda(lambda x: {"word_count": len(x.get("tokens", []))})
            return normalize | tokenize | count

        runnable = create_chain("e2e:multi_step")
        result = await runnable.invoke({"raw": "  Hello World From Victor  "})

        assert result["word_count"] == 4

    @pytest.mark.asyncio
    async def test_parallel_execution_with_aggregation_end_to_end(self):
        """Parallel chains aggregate results correctly end-to-end."""

        @chain("e2e:parallel_metrics", description="Parallel metrics gathering")
        def parallel_metrics_chain():
            return RunnableParallel(
                lines=RunnableLambda(lambda x: len(x.get("code", "").split("\n"))),
                chars=RunnableLambda(lambda x: len(x.get("code", ""))),
                words=RunnableLambda(lambda x: len(x.get("code", "").split())),
            )

        runnable = create_chain("e2e:parallel_metrics")
        code = "def foo():\n    return 42"
        result = await runnable.invoke({"code": code})

        assert result["lines"] == 2
        assert result["chars"] == len(code)
        # "def foo(): return 42" splits into 4 words: ["def", "foo():", "return", "42"]
        assert result["words"] == 4

    @pytest.mark.asyncio
    async def test_conditional_routing_end_to_end(self):
        """Conditional routing works correctly end-to-end."""

        @chain("e2e:file_processor", description="Process files by type")
        def file_processor_chain():
            return RunnableBranch(
                (
                    lambda x: x.get("ext") == ".py",
                    RunnableLambda(lambda x: {"processor": "python", "syntax": "valid"}),
                ),
                (
                    lambda x: x.get("ext") == ".js",
                    RunnableLambda(lambda x: {"processor": "javascript", "syntax": "valid"}),
                ),
                (
                    lambda x: x.get("ext") == ".md",
                    RunnableLambda(lambda x: {"processor": "markdown", "rendered": True}),
                ),
                default=RunnableLambda(lambda x: {"processor": "text", "raw": True}),
            )

        runnable = create_chain("e2e:file_processor")

        py_result = await runnable.invoke({"ext": ".py", "content": "print()"})
        assert py_result["processor"] == "python"

        md_result = await runnable.invoke({"ext": ".md", "content": "# Title"})
        assert md_result["processor"] == "markdown"
        assert md_result["rendered"] is True

        txt_result = await runnable.invoke({"ext": ".txt", "content": "plain text"})
        assert txt_result["processor"] == "text"


# =============================================================================
# Integration: Realistic Chain Examples
# =============================================================================


@pytest.mark.integration
class TestRealisticChainExamples:
    """Integration tests with realistic chain composition patterns."""

    @pytest.mark.asyncio
    async def test_code_review_pipeline_chain(self):
        """Realistic code review pipeline chain."""

        @chain(
            "coding:review_pipeline",
            description="Complete code review pipeline",
            tags=["review", "quality"],
        )
        def code_review_pipeline():
            # Parse input
            parse = RunnableLambda(
                lambda x: {
                    "code": x.get("code", ""),
                    "filename": x.get("filename", "unknown.py"),
                    "language": x.get("language", "python"),
                }
            )

            # Run parallel checks
            checks = RunnableParallel(
                syntax=RunnableLambda(lambda x: {"passed": True, "errors": []}),
                style=RunnableLambda(
                    lambda x: {
                        "score": 8.5,
                        "issues": [{"line": 10, "msg": "Line too long"}],
                    }
                ),
                security=RunnableLambda(
                    lambda x: {
                        "vulnerabilities": [],
                        "risk_level": "low",
                    }
                ),
                complexity=RunnableLambda(
                    lambda x: {
                        "cyclomatic": 3,
                        "maintainability_index": 75,
                    }
                ),
            )

            # Aggregate results
            aggregate = RunnableLambda(
                lambda x: {
                    "review_complete": True,
                    "syntax_valid": x["syntax"]["passed"],
                    "style_score": x["style"]["score"],
                    "security_risk": x["security"]["risk_level"],
                    "complexity_ok": x["complexity"]["cyclomatic"] < 10,
                    "issues": x["style"]["issues"],
                }
            )

            return parse | checks | aggregate

        runnable = create_chain("coding:review_pipeline")
        result = await runnable.invoke(
            {
                "code": "def foo():\n    pass",
                "filename": "main.py",
                "language": "python",
            }
        )

        assert result["review_complete"] is True
        assert result["syntax_valid"] is True
        assert result["style_score"] == 8.5
        assert result["security_risk"] == "low"
        assert result["complexity_ok"] is True
        assert len(result["issues"]) == 1

    @pytest.mark.asyncio
    async def test_document_processing_chain(self):
        """Realistic document processing chain for RAG vertical."""

        @chain(
            "rag:document_processor",
            description="Process documents for RAG ingestion",
            tags=["ingestion", "rag"],
        )
        def document_processor():
            # Extract metadata
            extract_meta = RunnableLambda(
                lambda x: {
                    **x,
                    "metadata": {
                        "title": x.get("content", "")[:50],
                        "word_count": len(x.get("content", "").split()),
                        "source": x.get("source", "unknown"),
                    },
                }
            )

            # Chunk content
            chunk = RunnableLambda(
                lambda x: {
                    **x,
                    "chunks": [
                        {"text": x["content"][i : i + 500], "index": i // 500}
                        for i in range(0, len(x.get("content", "")), 500)
                    ],
                }
            )

            # Generate embeddings (mocked)
            embed = RunnableLambda(
                lambda x: {
                    **x,
                    "embeddings": [
                        {"chunk_index": c["index"], "vector": [0.1] * 384}
                        for c in x.get("chunks", [])
                    ],
                }
            )

            return extract_meta | chunk | embed

        runnable = create_chain("rag:document_processor")
        result = await runnable.invoke(
            {
                "content": "This is a test document. " * 100,
                "source": "test.txt",
            }
        )

        assert "metadata" in result
        assert result["metadata"]["source"] == "test.txt"
        assert len(result["chunks"]) > 0
        assert len(result["embeddings"]) == len(result["chunks"])

    @pytest.mark.asyncio
    async def test_research_synthesis_chain(self):
        """Realistic research synthesis chain."""

        @chain(
            "research:synthesize",
            description="Synthesize research findings",
            tags=["research", "synthesis"],
        )
        def synthesis_chain():
            # Collect sources
            collect = RunnableLambda(
                lambda x: {
                    **x,
                    "sources": x.get("sources", []),
                    "query": x.get("query", ""),
                }
            )

            # Extract key points (parallel per source type)
            extract = RunnableParallel(
                academic=RunnableLambda(
                    lambda x: {
                        "points": [f"Academic finding {i}" for i in range(3)],
                        "citations": 5,
                    }
                ),
                web=RunnableLambda(
                    lambda x: {
                        "points": [f"Web finding {i}" for i in range(2)],
                        "urls": 10,
                    }
                ),
            )

            # Synthesize
            synthesize = RunnableLambda(
                lambda x: {
                    "synthesis": "Combined findings from all sources",
                    "total_academic_points": len(x["academic"]["points"]),
                    "total_web_points": len(x["web"]["points"]),
                    "total_citations": x["academic"]["citations"],
                    "total_urls": x["web"]["urls"],
                }
            )

            return collect | extract | synthesize

        runnable = create_chain("research:synthesize")
        result = await runnable.invoke(
            {
                "query": "machine learning trends",
                "sources": ["arxiv", "google"],
            }
        )

        assert result["total_academic_points"] == 3
        assert result["total_web_points"] == 2
        assert result["total_citations"] == 5


# =============================================================================
# Integration: Registry Metadata and Discovery
# =============================================================================


@pytest.mark.integration
class TestRegistryMetadataAndDiscovery:
    """Integration tests for chain registry metadata and discovery."""

    def test_discover_chains_by_vertical(self):
        """Can discover all chains in a vertical."""

        @chain("coding:lint")
        def lint_chain():
            return RunnableLambda(lambda x: x)

        @chain("coding:format")
        def format_chain():
            return RunnableLambda(lambda x: x)

        @chain("devops:deploy")
        def deploy_chain():
            return RunnableLambda(lambda x: x)

        registry = get_chain_registry()
        coding_chains = registry.list_factories(vertical="coding")

        assert len(coding_chains) == 2
        assert "coding:lint" in coding_chains
        assert "coding:format" in coding_chains
        assert "devops:deploy" not in coding_chains

    def test_discover_chains_by_tags(self):
        """Can discover chains by tags via metadata."""

        @chain("testing:a", tags=["quality", "testing"])
        def chain_a():
            return RunnableLambda(lambda x: x)

        @chain("testing:b", tags=["testing"])
        def chain_b():
            return RunnableLambda(lambda x: x)

        @chain("testing:c", tags=["quality", "production"])
        def chain_c():
            return RunnableLambda(lambda x: x)

        registry = get_chain_registry()

        # For factories, we check metadata directly since find_by_tag
        # only works for directly registered chains, not factories

        # Check metadata tags for chain a
        metadata_a = registry.get_metadata("a", vertical="testing")
        assert metadata_a is not None
        assert "quality" in metadata_a.tags
        assert "testing" in metadata_a.tags

        # Check metadata tags for chain b
        metadata_b = registry.get_metadata("b", vertical="testing")
        assert metadata_b is not None
        assert "testing" in metadata_b.tags
        assert "quality" not in metadata_b.tags

        # Check metadata tags for chain c
        metadata_c = registry.get_metadata("c", vertical="testing")
        assert metadata_c is not None
        assert "quality" in metadata_c.tags
        assert "production" in metadata_c.tags

        # We can filter factories by tags manually via metadata
        all_metadata = registry.list_metadata(vertical="testing")
        quality_factories = [m for m in all_metadata if "quality" in m.tags]
        assert len(quality_factories) == 2  # a and c have "quality" tag

    def test_registry_serialization(self):
        """Registry can be serialized for inspection."""

        @chain("serialization:test1", description="Test chain 1", version="2.0.0")
        def test1():
            return RunnableLambda(lambda x: x)

        @chain("serialization:test2", description="Test chain 2", tags=["important"])
        def test2():
            return RunnableLambda(lambda x: x)

        registry = get_chain_registry()
        serialized = registry.to_dict()

        assert "serialization:test1" in serialized
        assert serialized["serialization:test1"]["description"] == "Test chain 1"
        assert serialized["serialization:test1"]["version"] == "2.0.0"
        assert serialized["serialization:test1"]["is_factory"] is True

        assert "serialization:test2" in serialized
        assert "important" in serialized["serialization:test2"]["tags"]


# =============================================================================
# Integration: Thread Safety Under Concurrent Access
# =============================================================================


@pytest.mark.integration
class TestConcurrentChainAccess:
    """Integration tests for thread-safe concurrent chain access."""

    @pytest.mark.asyncio
    async def test_concurrent_chain_creation(self):
        """Chains can be created concurrently without race conditions."""

        @chain("concurrent:counter")
        def counter_factory():
            return RunnableLambda(lambda x: {"value": x.get("n", 0) + 1})

        async def create_and_invoke(n: int):
            chain_instance = create_chain("concurrent:counter")
            return await chain_instance.invoke({"n": n})

        # Run many concurrent creations and invocations
        tasks = [create_and_invoke(i) for i in range(50)]
        results = await asyncio.gather(*tasks)

        # All should complete successfully
        assert len(results) == 50
        assert all(r["value"] == i + 1 for i, r in enumerate(results))

    @pytest.mark.asyncio
    async def test_concurrent_registration_and_retrieval(self):
        """Registration and retrieval can happen concurrently."""
        import threading

        errors = []
        registered_names = []

        def register_chains(prefix: str, count: int):
            try:
                for i in range(count):
                    name = f"concurrent:{prefix}_{i}"

                    @chain(name, replace=True)
                    def factory(idx=i):
                        return RunnableLambda(lambda x, n=idx: {"index": n})

                    registered_names.append(name)
            except Exception as e:
                errors.append(e)

        # Register from multiple threads
        threads = [
            threading.Thread(target=register_chains, args=(f"thread{t}", 20)) for t in range(3)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

        # Should have registered 60 chains
        registry = get_chain_registry()
        concurrent_factories = [
            name for name in registry.list_factories() if name.startswith("concurrent:")
        ]
        assert len(concurrent_factories) == 60


# =============================================================================
# Integration: Error Handling
# =============================================================================


@pytest.mark.integration
class TestChainErrorHandling:
    """Integration tests for chain error handling."""

    @pytest.mark.asyncio
    async def test_chain_execution_error_propagates(self):
        """Errors in chain execution propagate correctly."""

        @chain("errors:failing")
        def failing_chain():
            def fail(x):
                raise ValueError("Intentional failure")

            return RunnableLambda(fail)

        runnable = create_chain("errors:failing")

        with pytest.raises(ValueError, match="Intentional failure"):
            await runnable.invoke({})

    def test_factory_error_raises_runtime_error(self):
        """Errors in factory execution raise RuntimeError."""

        @chain("errors:bad_factory")
        def bad_factory():
            raise RuntimeError("Factory construction failed")

        with pytest.raises(RuntimeError, match="Factory execution failed"):
            create_chain("errors:bad_factory")

    @pytest.mark.asyncio
    async def test_partial_failure_in_parallel_chain(self):
        """Partial failures in parallel chains are handled gracefully."""

        @chain("errors:partial_parallel")
        def partial_parallel_factory():
            return RunnableParallel(
                success=RunnableLambda(lambda x: {"status": "ok"}),
                failure=RunnableLambda(lambda x: (_ for _ in ()).throw(ValueError("Oops"))),
            )

        runnable = create_chain("errors:partial_parallel")
        result = await runnable.invoke({})

        # Success branch should complete
        assert result["success"]["status"] == "ok"
        # Failure branch should have error info
        assert "error" in result["failure"]


# =============================================================================
# Integration: Chain Binding and Configuration
# =============================================================================


@pytest.mark.integration
class TestChainBindingAndConfiguration:
    """Integration tests for chain binding and configuration."""

    @pytest.mark.asyncio
    async def test_chain_with_bound_arguments(self):
        """Chains can be created with bound arguments."""

        @chain("binding:configurable")
        def configurable_factory():
            def process(x):
                return {
                    "result": x.get("value", 0) * x.get("multiplier", 1),
                    "offset": x.get("offset", 0),
                }

            return RunnableLambda(process)

        runnable = create_chain("binding:configurable")

        # Bind multiplier
        bound = runnable.bind(multiplier=10)
        result = await bound.invoke({"value": 5})

        assert result["result"] == 50

    @pytest.mark.asyncio
    async def test_chain_with_config(self):
        """Chains can be executed with RunnableConfig."""

        @chain("config:tracked")
        def tracked_factory():
            return RunnableLambda(lambda x: {"processed": True})

        runnable = create_chain("config:tracked")
        config = RunnableConfig(
            tags=["test", "integration"],
            metadata={"run_id": "test-123"},
        )

        result = await runnable.invoke({"data": "test"}, config)
        assert result["processed"] is True


# =============================================================================
# Integration: Workflow Chain Handler Resolution
# =============================================================================


@pytest.mark.integration
class TestWorkflowChainHandlerResolution:
    """Integration tests for chain handler resolution in workflows."""

    @pytest.mark.asyncio
    async def test_chain_handler_resolution_with_vertical(self):
        """Chain handlers with vertical namespace are resolved correctly."""

        # Register a chain
        @chain("workflow_test:analyze")
        def analyze_chain():
            return RunnableLambda(lambda x: {"analyzed": True, "input": x.get("data")})

        # Simulate workflow executor resolution
        from victor.workflows.executors.compute import ComputeNodeExecutor

        executor = ComputeNodeExecutor()

        # Test resolution with vertical
        handler = executor._resolve_chain_handler("chain:workflow_test:analyze")
        assert handler is not None

        # Create mock node and context
        from victor.workflows.definition import ComputeNode
        from victor.workflows.context import WorkflowContext

        node = ComputeNode(
            id="test_node",
            name="Test Node",
            handler="chain:workflow_test:analyze",
            input_mapping={"data": "$ctx.input_data"},
            output_key="result",
        )

        context = WorkflowContext({"input_data": "test data"})

        # Execute the handler
        result = await handler(node, context, None)

        assert result.status.value == "completed"
        assert result.output["analyzed"] is True
        assert result.output["input"] == "test data"
        assert context.get("result") == result.output

    @pytest.mark.asyncio
    async def test_chain_handler_resolution_without_vertical(self):
        """Chain handlers without vertical namespace are resolved correctly."""

        # Register a chain without vertical
        @chain("simple_process")
        def simple_chain():
            return RunnableLambda(lambda x: {"processed": x.get("value") * 2})

        # Simulate workflow executor resolution
        from victor.workflows.executors.compute import ComputeNodeExecutor

        executor = ComputeNodeExecutor()

        # Test resolution without vertical
        handler = executor._resolve_chain_handler("chain:simple_process")
        assert handler is not None

        # Create mock node and context
        from victor.workflows.definition import ComputeNode
        from victor.workflows.context import WorkflowContext

        node = ComputeNode(
            id="test_node",
            name="Test Node",
            handler="chain:simple_process",
        )

        context = WorkflowContext({"value": 21})

        # Execute the handler
        result = await handler(node, context, None)

        assert result.status.value == "completed"
        assert result.output["processed"] == 42

    @pytest.mark.asyncio
    async def test_chain_handler_not_found(self):
        """Non-existent chain handlers return None gracefully."""

        from victor.workflows.executors.compute import ComputeNodeExecutor

        executor = ComputeNodeExecutor()

        # Test with non-existent chain
        handler = executor._resolve_chain_handler("chain:nonexistent:missing")
        assert handler is None

    @pytest.mark.asyncio
    async def test_chain_handler_timeout(self):
        """Chain handlers respect timeout configuration."""

        import asyncio

        @chain("timeout_test:slow")
        def slow_chain():
            async def slow_fn(x):
                await asyncio.sleep(5)
                return {"done": True}

            return RunnableLambda(slow_fn)

        from victor.workflows.executors.compute import ComputeNodeExecutor
        from victor.workflows.definition import ComputeNode
        from victor.workflows.context import WorkflowContext

        executor = ComputeNodeExecutor()
        handler = executor._resolve_chain_handler("chain:timeout_test:slow")

        from victor.workflows.definition import TaskConstraints

        node = ComputeNode(
            id="timeout_node",
            name="Timeout Node",
            handler="chain:timeout_test:slow",
            constraints=TaskConstraints(_timeout=1.0),  # 1 second timeout
        )

        context = WorkflowContext({})

        # Execute should timeout
        result = await handler(node, context, None)

        assert result.status.value == "failed"
        assert "timed out" in result.error.lower()

    @pytest.mark.asyncio
    async def test_chain_handler_with_input_output_mapping(self):
        """Chain handlers properly map input and output using input_mapping."""

        @chain("mapping_test:transform")
        def transform_chain():
            return RunnableLambda(
                lambda x: {
                    "result": f"Processed: {x.get('content')}",
                    "length": len(x.get("content", "")),
                }
            )

        from victor.workflows.executors.compute import ComputeNodeExecutor
        from victor.workflows.definition import ComputeNode
        from victor.workflows.context import WorkflowContext

        executor = ComputeNodeExecutor()
        handler = executor._resolve_chain_handler("chain:mapping_test:transform")

        node = ComputeNode(
            id="mapping_node",
            name="Mapping Node",
            handler="chain:mapping_test:transform",
            input_mapping={"content": "$ctx.input_text"},
            output_key="transformed",
        )

        context = WorkflowContext({"input_text": "Hello, World!"})

        result = await handler(node, context, None)

        assert result.status.value == "completed"
        assert result.output["result"] == "Processed: Hello, World!"
        assert result.output["length"] == 13
        assert context.get("transformed") == result.output

    @pytest.mark.asyncio
    async def test_chain_handler_error_propagation(self):
        """Chain handlers propagate exceptions as failed results."""

        @chain("error_test:failing")
        def failing_chain():
            def fail_fn(x):
                raise ValueError("Simulated chain failure")

            return RunnableLambda(fail_fn)

        from victor.workflows.executors.compute import ComputeNodeExecutor
        from victor.workflows.definition import ComputeNode
        from victor.workflows.context import WorkflowContext

        executor = ComputeNodeExecutor()
        handler = executor._resolve_chain_handler("chain:error_test:failing")

        node = ComputeNode(
            id="error_node",
            name="Error Node",
            handler="chain:error_test:failing",
        )

        context = WorkflowContext({})

        result = await handler(node, context, None)

        assert result.status.value == "failed"
        assert "execution failed" in result.error.lower()
        assert "Simulated chain failure" in result.error

    @pytest.mark.asyncio
    async def test_get_compute_handler_with_chain_prefix(self):
        """_get_compute_handler correctly routes chain: prefixed handlers."""

        @chain("routing_test:target")
        def target_chain():
            return RunnableLambda(lambda x: {"routed": True})

        from victor.workflows.executors.compute import ComputeNodeExecutor

        executor = ComputeNodeExecutor()

        # Chain handler should be resolved
        chain_handler = executor._get_compute_handler("chain:routing_test:target")
        assert chain_handler is not None

        # Regular handler should use compute_registry
        regular_handler = executor._get_compute_handler("some_regular_handler")
        # Will be None since handler not registered, but different code path
        assert regular_handler is None
