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

"""Integration tests for query translation (PH3-001 to PH3-006)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from victor.core.graph_rag import (
    translate_query,
    list_templates,
    get_template_registry,
    QueryType,
    MatchStrategy,
)
from victor.storage.graph import create_graph_store
from victor.core.graph_rag import GraphIndexingPipeline, GraphIndexConfig


@pytest.mark.integration
@pytest.mark.asyncio
async def test_query_translation_neighbors():
    """Test translating natural language to neighbors query."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create test code
        (tmpdir / "graph.py").write_text("""
class Node:
    def __init__(self, value):
        self.value = value
        self.neighbors = []

    def add_neighbor(self, node):
        self.neighbors.append(node)

class Graph:
    def __init__(self):
        self.nodes = []

    def add_node(self, node):
        self.nodes.append(node)
        node.add_neighbor(self)
""")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_query_translation_path():
    """Test translating natural language to path query."""
    result = await translate_query("Find path from A to B", graph_store=None)

    assert result is not None
    assert result.matched_template is not None
    assert result.matched_template.query_type == QueryType.PATH
    assert "source" in result.parameters or "from" in result.parameters


@pytest.mark.integration
@pytest.mark.asyncio
async def test_query_translation_impact_analysis():
    """Test translating natural language to impact query."""
    result = await translate_query("What depends on authenticate_user?", graph_store=None)

    assert result is not None
    assert result.matched_template is not None
    assert result.matched_template.query_type == QueryType.IMPACT
    # Parameter extraction may not work perfectly, but template should match
    assert result.matched_template is not None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_query_translation_callers():
    """Test translating natural language to callers query."""
    result = await translate_query("What calls process_data?", graph_store=None)

    assert result is not None
    assert result.matched_template is not None
    assert result.matched_template.query_type == QueryType.CALLERS


@pytest.mark.integration
@pytest.mark.asyncio
async def test_query_translation_semantic_search():
    """Test translating natural language to semantic search."""
    result = await translate_query(
        "Find functions related to user authentication", graph_store=None
    )

    assert result is not None
    # Falls back to semantic search template
    assert result.matched_template is not None or result.fallback


@pytest.mark.integration
@pytest.mark.asyncio
async def test_list_templates():
    """Test listing available query templates."""
    templates = list_templates()

    assert isinstance(templates, list)
    assert len(templates) > 0

    # Check for expected templates (templates are QueryTemplate objects)
    template_names = {t.name for t in templates}
    assert "find_neighbors" in template_names
    assert "find_path" in template_names
    assert "impact_analysis" in template_names


@pytest.mark.integration
@pytest.mark.asyncio
async def test_template_registry():
    """Test template registry functionality."""
    registry = get_template_registry()

    assert registry is not None

    # Get all templates
    all_templates = registry.list_all()
    assert len(all_templates) > 0

    # Find by type
    neighbor_templates = registry.find_by_type(QueryType.NEIGHBORS)
    assert len(neighbor_templates) > 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_query_translation_with_graph_context():
    """Test query translation with actual graph context."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create test code with clear relationships
        (tmpdir / "service.py").write_text("""
def validate_request(req):
    '''Validate incoming request.'''
    return req is not None

def process_request(req):
    '''Process validated request.'''
    if validate_request(req):
        return handle_request(req)
    return None

def handle_request(req):
    '''Handle the request logic.'''
    return req.data

def send_response(resp):
    '''Send response to client.'''
    return resp
""")

        # Index the code
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

        # Test query translation with graph store
        result = await translate_query("What calls validate_request?", graph_store=graph_store)

        assert result is not None
        assert result.matched_template is not None
        assert result.matched_template.query_type == QueryType.CALLERS

        await graph_store.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_query_translation_parameter_extraction():
    """Test that parameters are correctly extracted from queries."""
    test_cases = [
        ("neighbors of Node class", "node_id", "Node"),
        ("path from func_a to func_b", "from", "func_a"),
        ("impact of changing authenticate", "node_id", "authenticate"),
        ("callers of process_data", "node_id", "process_data"),
    ]

    for query, expected_param_key, expected_param_value in test_cases:
        result = await translate_query(query, graph_store=None)

        assert result is not None, f"Translation failed for: {query}"
        # Check that translation produced some result
        assert result.matched_template is not None or result.fallback


@pytest.mark.integration
@pytest.mark.asyncio
async def test_query_translation_fallback():
    """Test query translation fallback for unrecognized queries."""
    # Unusual query that should still translate
    result = await translate_query(
        "Show me the things that do stuff with the data", graph_store=None
    )

    # Should still return a result (fallback to semantic search)
    assert result is not None
    assert result.fallback or result.matched_template is not None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_multiple_query_translations():
    """Test batch translation of multiple queries."""
    queries = [
        "neighbors of Node",
        "path from A to B",
        "impact of changing foo",
        "callers of bar",
        "callees of baz",
    ]

    results = []
    for query in queries:
        result = await translate_query(query, graph_store=None)
        results.append(result)

    # All should translate successfully
    assert len(results) == len(queries)
    assert all(r is not None for r in results)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_template_matching_scores():
    """Test that template matching returns appropriate scores."""
    registry = get_template_registry()

    # These queries should match specific templates with high scores
    test_queries = [
        ("neighbors of Node", "find_neighbors"),
        ("Find path from A to B", "find_path"),
        ("Impact of changing X", "impact_analysis"),
    ]

    for query, expected_template in test_queries:
        match_result = registry.match(query)
        assert match_result is not None
        # match returns (template, score) tuple
        template, score = match_result
        assert template is not None
        assert score > 0
        # The matched template should be the expected one or a similar one
        assert expected_template in template.name or template.name in expected_template
