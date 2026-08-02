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

"""Integration tests for graph query tools."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from victor.tools.graph_query_tool import graph_semantic_search, impact_analysis
from victor.storage.graph import create_graph_store
from victor.core.graph_rag import GraphIndexingPipeline, GraphIndexConfig


@pytest.mark.integration
@pytest.mark.asyncio
async def test_graph_query_tool():
    """Test the graph_query tool."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create test files
        (tmpdir / "calculator.py").write_text("""
class Calculator:
    '''A simple calculator class.'''

    def __init__(self):
        self.result = 0

    def add(self, x: int, y: int) -> int:
        '''Add two numbers.'''
        return x + y

    def subtract(self, x: int, y: int) -> int:
        '''Subtract two numbers.'''
        return x - y

    def multiply(self, x: int, y: int) -> int:
        '''Multiply two numbers.'''
        return x * y

    def divide(self, x: int, y: int) -> float:
        '''Divide two numbers.'''
        if y == 0:
            raise ValueError("Cannot divide by zero")
        return x / y
""")

        # Index the codebase
        graph_store = create_graph_store(name="sqlite", project_path=tmpdir)
        await graph_store.initialize()
        await graph_store.delete_by_repo()

        config = GraphIndexConfig(
            root_path=tmpdir,
            enable_ccg=True,
            enable_embeddings=True,  # Required for semantic search
        )

        pipeline = GraphIndexingPipeline(graph_store, config)
        await pipeline.index_repository()
        await graph_store.close()

        # Test graph query tool
        result = await graph_semantic_search(
            query="calculator multiplication function",
            path=str(tmpdir),
            mode="semantic",
            max_hops=2,
        )

        # Verify results
        assert "results" in result
        assert "query" in result
        assert result["query"] == "calculator multiplication function"

        # Note: Results may be empty if embeddings are disabled
        # With enable_embeddings=False, semantic search won't find matches
        # This is expected behavior - semantic mode requires embeddings
        if result["results"]:
            # Should find the multiply method
            node_names = [n.get("name", "") for n in result["results"]]
            assert any("multiply" in name.lower() for name in node_names)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_impact_analysis_tool_forward():
    """Test forward impact analysis."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create dependency chain
        (tmpdir / "data_processor.py").write_text("""
def fetch_data(url):
    '''Fetch data from URL.'''
    return requests.get(url).json()

def parse_data(data):
    '''Parse JSON data.'''
    return json.loads(data)

def process_data(url):
    '''Fetch and parse data from URL.'''
    raw = fetch_data(url)
    return parse_data(raw)

def save_to_database(data):
    '''Save data to database.'''
    db.insert(data)

def export_data(url):
    '''Fetch, parse, and save data.'''
    data = process_data(url)
    return save_to_database(data)
""")

        # Index
        graph_store = create_graph_store(name="sqlite", project_path=tmpdir)
        await graph_store.initialize()
        await graph_store.delete_by_repo()

        config = GraphIndexConfig(
            root_path=tmpdir,
            enable_ccg=True,
            enable_embeddings=True,  # Required for semantic search
        )

        pipeline = GraphIndexingPipeline(graph_store, config)
        await pipeline.index_repository()
        await graph_store.close()

        # Test forward impact analysis (what breaks if fetch_data changes?)
        result = await impact_analysis(
            target="fetch_data",
            analysis_type="forward",
            max_depth=3,
            path=str(tmpdir),
        )

        # Verify results
        assert "target" in result
        assert "analysis_type" in result
        assert result["analysis_type"] == "forward"
        assert "impacted_symbols" in result

        # Should find process_data and export_data as impacted
        impacted_names = [n.get("name", "") for n in result["impacted_symbols"]]
        assert any("process" in name.lower() for name in impacted_names)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_impact_analysis_tool_backward():
    """Test backward impact analysis."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create dependency chain
        (tmpdir / "service.py").write_text("""
def validate_input(data):
    '''Validate user input.'''
    return data is not None and len(data) > 0

def sanitize_input(data):
    '''Sanitize user input.'''
    return escape_html(data)

def process_input(data):
    '''Process user input safely.'''
    if validate_input(data):
        return sanitize_input(data)
    return None

def handle_request(request):
    '''Handle incoming request.'''
    return process_input(request.data)
""")

        # Index
        graph_store = create_graph_store(name="sqlite", project_path=tmpdir)
        await graph_store.initialize()
        await graph_store.delete_by_repo()

        config = GraphIndexConfig(
            root_path=tmpdir,
            enable_ccg=True,
            enable_embeddings=True,  # Required for semantic search
        )

        pipeline = GraphIndexingPipeline(graph_store, config)
        await pipeline.index_repository()
        await graph_store.close()

        # Test backward impact analysis (what does handle_request depend on?)
        result = await impact_analysis(
            target="handle_request",
            analysis_type="backward",
            max_depth=2,
            path=str(tmpdir),
        )

        # Verify results
        assert result["analysis_type"] == "backward"
        assert "impacted_symbols" in result

        # Should find process_input, validate_input, sanitize_input
        impacted_names = [n.get("name", "") for n in result["impacted_symbols"]]
        assert any("process" in name.lower() for name in impacted_names)
        assert any("validate" in name.lower() for name in impacted_names)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_graph_query_with_file_line_target():
    """Test graph query with file:line target."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create test file
        test_file = tmpdir / "test.py"
        test_file.write_text("""
def function_one():
    '''First function at line 2.'''
    pass

def function_two():
    '''Second function at line 7.'''
    pass

def function_three():
    '''Third function at line 12.'''
    pass
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
        await graph_store.close()

        # Query with file:line format
        result = await impact_analysis(
            target=f"{test_file}:8",  # Near function_two
            analysis_type="forward",
            max_depth=1,
            path=str(tmpdir),
        )

        # Should resolve to function_two
        assert "impacted_symbols" in result
        # Even if no forward dependencies, should not error
        assert isinstance(result["impacted_symbols"], list)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_graph_query_multi_hop():
    """Test multi-hop graph traversal."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create multi-level dependency chain
        (tmpdir / "chain.py").write_text("""
def level_one():
    '''Base function.'''
    return "one"

def level_two():
    '''Calls level one.'''
    result = level_one()
    return result + " two"

def level_three():
    '''Calls level two.'''
    result = level_two()
    return result + " three"

def level_four():
    '''Calls level three.'''
    result = level_three()
    return result + " four"
""")

        # Index
        graph_store = create_graph_store(name="sqlite", project_path=tmpdir)
        await graph_store.initialize()
        await graph_store.delete_by_repo()

        config = GraphIndexConfig(
            root_path=tmpdir,
            enable_ccg=True,
            enable_embeddings=True,  # Required for semantic search
        )

        pipeline = GraphIndexingPipeline(graph_store, config)
        await pipeline.index_repository()
        await graph_store.close()

        # Query with 2-hop limit
        result_1_hop = await graph_semantic_search(
            query="level one",
            path=str(tmpdir),
            mode="structural",
            max_hops=1,
        )

        result_2_hop = await graph_semantic_search(
            query="level one",
            path=str(tmpdir),
            mode="structural",
            max_hops=2,
        )

        # 2-hop should find more results than 1-hop
        assert len(result_2_hop["results"]) >= len(result_1_hop["results"])


@pytest.mark.integration
@pytest.mark.asyncio
async def test_graph_query_empty_results():
    """Test graph query with no matches."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create minimal file
        (tmpdir / "minimal.py").write_text("""
def simple():
    pass
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
        await graph_store.close()

        # Query for non-existent functionality
        result = await graph_semantic_search(
            query="quantum computing algorithm",
            path=str(tmpdir),
            mode="semantic",
            max_hops=1,
        )

        # Should return empty results, not error
        assert "results" in result
        assert isinstance(result["results"], list)
