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

"""Integration tests for Graph RAG pipeline."""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

import pytest

from victor.core.graph_rag import (
    GraphIndexingPipeline,
    GraphIndexConfig,
    GraphIndexStats,
    MultiHopRetriever,
    RetrievalConfig,
    GraphAwarePromptBuilder,
    PromptConfig,
)
from victor.storage.graph import create_graph_store, GraphNode, GraphEdge
from victor.storage.graph.edge_types import EdgeType


@pytest.mark.integration
@pytest.mark.asyncio
async def test_graph_indexing_pipeline_full():
    """Test full graph indexing pipeline with CCG."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create test Python files
        test_file = tmpdir / "test_module.py"
        test_file.write_text("""
def calculate_sum(a: int, b: int) -> int:
    '''Calculate the sum of two integers.'''
    if a > 0:
        return a + b
    else:
        return b

class Calculator:
    def __init__(self):
        self.value = 0

    def add(self, x: int) -> None:
        self.value += x

    def multiply(self, x: int) -> None:
        self.value *= x
""")

        # Create graph store
        graph_store = create_graph_store(name="sqlite", project_path=tmpdir)
        await graph_store.initialize()
        await graph_store.delete_by_repo()

        # Configure indexing
        config = GraphIndexConfig(
            root_path=tmpdir,
            enable_ccg=True,
            enable_embeddings=False,  # Skip embeddings for speed
        )

        # Build index
        pipeline = GraphIndexingPipeline(graph_store, config)
        stats = await pipeline.index_repository()

        # Verify results
        assert isinstance(stats, GraphIndexStats)
        assert stats.files_processed >= 1
        assert stats.nodes_created > 0
        assert stats.edges_created > 0

        # Verify we can query the graph
        all_nodes = await graph_store.get_all_nodes()
        assert len(all_nodes) > 0

        # Check for expected symbols
        function_nodes = [n for n in all_nodes if n.type == "function"]
        assert len(function_nodes) > 0

        await graph_store.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_graph_indexing_pipeline_reuses_unchanged_files_incrementally():
    """Incremental indexing should touch only changed/deleted source files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        changed_file = tmpdir / "changed.py"
        unchanged_file = tmpdir / "stable.py"
        ignored_file = tmpdir / "README.md"

        changed_file.write_text("""
def old_name():
    return 1
""".strip())
        unchanged_file.write_text("""
def stable_name():
    return 2
""".strip())
        ignored_file.write_text("# Not indexable by the graph pipeline")

        graph_store = create_graph_store(name="sqlite", project_path=tmpdir)
        await graph_store.initialize()
        await graph_store.delete_by_repo()

        config = GraphIndexConfig(
            root_path=tmpdir,
            enable_ccg=False,
            enable_embeddings=False,
            enable_subgraph_cache=False,
        )
        pipeline = GraphIndexingPipeline(graph_store, config)

        initial_stats = await pipeline.index_repository()
        assert initial_stats.files_processed == 2
        assert initial_stats.files_deleted == 0
        assert initial_stats.files_unchanged == 0

        os.utime(changed_file, (time.time() + 2, time.time() + 2))
        changed_file.write_text("""
def new_name():
    return 3
""".strip())

        incremental_stats = await pipeline.index_repository()
        assert incremental_stats.files_processed == 1
        assert incremental_stats.files_deleted == 0
        assert incremental_stats.files_unchanged == 1

        changed_nodes = await graph_store.get_nodes_by_file(str(changed_file))
        unchanged_nodes = await graph_store.get_nodes_by_file(str(unchanged_file))
        # A synthetic module node (named after the file stem) is prepended per file
        # so IMPORTS and top-level CONTAINS edges have an anchor (commit 35bdcc57a).
        assert {node.name for node in changed_nodes} == {"changed", "new_name"}
        assert {node.name for node in unchanged_nodes} == {"stable", "stable_name"}

        unchanged_file.unlink()
        deleted_stats = await pipeline.index_repository()
        assert deleted_stats.files_processed == 0
        assert deleted_stats.files_deleted == 1
        assert deleted_stats.files_unchanged == 1
        assert await graph_store.get_nodes_by_file(str(unchanged_file)) == []

        await graph_store.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_multi_hop_retrieval():
    """Test multi-hop retrieval with graph traversal."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create test files with dependencies
        (tmpdir / "auth.py").write_text("""
def validate_user(user_id: str) -> bool:
    return user_id is not None

def authentication(credentials: dict) -> bool:
    user_id = credentials.get('user_id')
    return validate_user(user_id)
""")

        (tmpdir / "api.py").write_text("""
from auth import authentication

def login(request):
    return authentication(request.json)

def logout(session):
    session.clear()
""")

        # Create graph store and index
        graph_store = create_graph_store(name="sqlite", project_path=tmpdir)
        await graph_store.initialize()
        await graph_store.delete_by_repo()

        config = GraphIndexConfig(
            root_path=tmpdir,
            enable_ccg=True,
            enable_embeddings=False,
        )

        pipeline = GraphIndexingPipeline(graph_store, config)
        await pipeline.index_repository()

        # Test multi-hop retrieval
        retrieval_config = RetrievalConfig(
            seed_count=5,
            max_hops=2,
            top_k=10,
        )

        retriever = MultiHopRetriever(graph_store, retrieval_config)
        result = await retriever.retrieve("authentication", retrieval_config)

        # Verify retrieval
        assert result is not None
        assert len(result.nodes) > 0
        assert result.execution_time_ms >= 0
        assert result.metadata.get("hop_count", 0) <= 2

        await graph_store.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_graph_aware_prompt_builder():
    """Test graph-aware prompt building."""
    from victor.storage.graph.protocol import GraphNode

    # Create mock nodes
    nodes = [
        GraphNode(
            node_id="func_1",
            type="function",
            name="authenticate",
            file="auth.py",
            line=10,
            end_line=15,
            lang="python",
            signature="def authenticate(credentials: dict) -> bool",
            docstring="Authenticate user with credentials",
        ),
        GraphNode(
            node_id="func_2",
            type="function",
            name="validate_user",
            file="auth.py",
            line=5,
            end_line=7,
            lang="python",
            signature="def validate_user(user_id: str) -> bool",
            docstring="Validate user ID",
        ),
    ]

    # Create mock retrieval result
    from victor.core.graph_rag.retrieval import RetrievalResult

    retrieval_result = RetrievalResult(
        nodes=nodes,
        edges=[],
        subgraphs=[],  # Empty subgraphs for this test
        query="How does authentication work?",
        seed_nodes=["func_1"],
        hop_distances={"func_1": 0, "func_2": 1},
        scores={"func_1": 1.0, "func_2": 0.8},
        execution_time_ms=10.5,
    )

    # Test different prompt formats
    from victor.core.graph_rag.config import PromptConfig

    config = PromptConfig(format_style="hierarchical")
    builder = GraphAwarePromptBuilder(config)
    prompt_hierarchical = builder.build_prompt(
        query="How does authentication work?",
        retrieval_result=retrieval_result,
    )
    assert "authenticate" in prompt_hierarchical.lower()  # Function name is "authenticate"

    config = PromptConfig(format_style="flat")
    builder = GraphAwarePromptBuilder(config)
    prompt_flat = builder.build_prompt(
        query="Explain the authentication function",
        retrieval_result=retrieval_result,
    )
    assert "authenticate" in prompt_flat.lower()

    # Test format_context
    context = builder.format_context(retrieval_result, max_tokens=1000)
    assert context.token_estimate > 0 or len(context.text) > 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_ccg_construction():
    """Test CCG construction for a Python file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create test file with control flow
        test_file = tmpdir / "control_flow.py"
        test_file.write_text("""
def process_items(items: list) -> list:
    '''Process items with various control flow.'''
    results = []

    for item in items:
        if item > 0:
            results.append(item * 2)
        elif item < 0:
            results.append(item * -1)
        else:
            results.append(0)

    try:
        results.sort()
    except ValueError:
        pass

    return results
""")

        # Build CCG
        from victor.core.indexing.ccg_builder import CodeContextGraphBuilder

        graph_store = create_graph_store(name="sqlite", project_path=tmpdir)
        await graph_store.initialize()

        builder = CodeContextGraphBuilder(graph_store, language="python")
        nodes, edges = await builder.build_ccg_for_file(test_file)

        # Verify CCG construction
        assert len(nodes) > 0

        # Check for CFG edges
        cfg_edges = [e for e in edges if EdgeType.is_cfg_edge(e.type)]
        assert len(cfg_edges) > 0, "Should have CFG edges for control flow"

        # Check for statement nodes
        statement_nodes = [n for n in nodes if n.type == "statement"]
        assert len(statement_nodes) > 0, "Should have statement nodes"

        await graph_store.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_impact_analysis():
    """Test impact analysis using graph traversal."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create dependency chain in a single file
        (tmpdir / "chain.py").write_text("""
def base_function():
    return "base"

def middle_function():
    result = base_function()
    return result.upper()

def top_function():
    return middle_function() + "!"
""")

        # Index the codebase
        graph_store = create_graph_store(name="sqlite", project_path=tmpdir)
        await graph_store.initialize()
        await graph_store.delete_by_repo()

        config = GraphIndexConfig(
            root_path=tmpdir,
            enable_ccg=False,
            enable_embeddings=False,
        )

        pipeline = GraphIndexingPipeline(graph_store, config)
        await pipeline.index_repository()

        # Find base_function node
        all_nodes = await graph_store.get_all_nodes()
        base_func = next((n for n in all_nodes if n.name == "base_function"), None)
        assert base_func is not None

        # Forward impact analysis (what calls base_function)
        # Use "in" direction to find incoming edges (callers)
        forward_edges = await graph_store.get_neighbors(
            base_func.node_id,
            direction="in",  # Incoming edges (what calls this function)
            max_depth=2,
        )

        impacted = set()
        for edge in forward_edges:
            # For incoming edges, src is the caller
            impacted.add(edge.src)

        # Should find middle_function which calls base_function
        assert len(impacted) > 0, "Should find functions that depend on base_function"

        await graph_store.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_graph_query_semantic_search():
    """Test semantic search through graph."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create test files
        (tmpdir / "math_utils.py").write_text("""
def add(a, b):
    return a + b

def multiply(a, b):
    return a * b

def divide(a, b):
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b
""")

        (tmpdir / "string_utils.py").write_text("""
def capitalize(s):
    return s.upper()

def lowercase(s):
    return s.lower()
""")

        # Index
        graph_store = create_graph_store(name="sqlite", project_path=tmpdir)
        await graph_store.initialize()
        await graph_store.delete_by_repo()

        config = GraphIndexConfig(
            root_path=tmpdir,
            enable_ccg=False,
            enable_embeddings=False,
        )

        pipeline = GraphIndexingPipeline(graph_store, config)
        await pipeline.index_repository()

        # Search for math-related symbols
        results = await graph_store.search_symbols("divide", limit=5)
        assert len(results) > 0

        # Should find divide function
        divide_found = any(n.name == "divide" for n in results)
        assert divide_found, "Should find divide function for 'divide' query"

        await graph_store.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_requirement_graph_mapping():
    """Test requirement-to-code mapping."""
    from victor.core.graph_rag.requirement_graph import RequirementGraphBuilder

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create test code
        (tmpdir / "user_auth.py").write_text("""
def login(username, password):
    '''Authenticate user with username and password.'''
    if validate_credentials(username, password):
        return create_session(username)
    return None

def logout(session_id):
    '''End user session.'''
    invalidate_session(session_id)

def authentication(username, password):
    '''Handle authentication with username and password.'''
    return login(username, password)
""")

        # Index
        graph_store = create_graph_store(name="sqlite", project_path=tmpdir)
        await graph_store.initialize()
        await graph_store.delete_by_repo()

        config = GraphIndexConfig(
            root_path=tmpdir,
            enable_ccg=False,
            enable_embeddings=False,
        )

        pipeline = GraphIndexingPipeline(graph_store, config)
        await pipeline.index_repository()

        # Map requirement to code
        req_builder = RequirementGraphBuilder(graph_store)
        mapping = await req_builder.map_requirement(
            requirement="authentication",  # Match function name
        )

        # Should find login-related functions
        # Note: Without embeddings, mapping relies on keyword matching
        # The function named "authentication" should be found
        assert len(mapping.mapped_symbols) > 0, "Should find at least one mapped symbol"

        # Get node names by fetching each node individually
        symbol_names = []
        for symbol_id in mapping.mapped_symbols:
            node = await graph_store.get_node_by_id(symbol_id)
            if node:
                symbol_names.append(node.name)

        # Should find the authentication function
        assert any(
            "authentication" in name.lower() for name in symbol_names
        ), f"Expected 'authentication' in {symbol_names}"

        await graph_store.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_subgraph_caching():
    """Test subgraph caching for performance."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create test file
        (tmpdir / "cache_test.py").write_text("""
def function_a():
    return function_b()

def function_b():
    return function_c()

def function_c():
    return 42
""")

        # Index
        graph_store = create_graph_store(name="sqlite", project_path=tmpdir)
        await graph_store.initialize()
        await graph_store.delete_by_repo()

        config = GraphIndexConfig(
            root_path=tmpdir,
            enable_ccg=True,
            enable_embeddings=False,
        )

        pipeline = GraphIndexingPipeline(graph_store, config)
        await pipeline.index_repository()

        # Get all nodes
        all_nodes = await graph_store.get_all_nodes()
        if all_nodes:
            anchor = all_nodes[0]

            # First retrieval (cold)
            result1 = await graph_store.get_neighbors(
                anchor.node_id,
                max_depth=2,
            )

            # Second retrieval (warm)
            result2 = await graph_store.get_neighbors(
                anchor.node_id,
                max_depth=2,
            )

            # Results should be consistent
            assert len(result1) == len(result2)

        await graph_store.close()
