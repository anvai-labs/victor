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

"""Focused tests for search query routing on AgentOrchestrator."""

from victor.agent.orchestrator import AgentOrchestrator
from victor.agent.search_router import SearchRouter


def _make_orchestrator() -> AgentOrchestrator:
    orchestrator = object.__new__(AgentOrchestrator)
    orchestrator.search_router = SearchRouter()
    return orchestrator


def test_route_search_query_includes_bug_mode_arguments() -> None:
    """Bug/regression queries should carry tool arguments for code_search."""
    orchestrator = _make_orchestrator()

    result = orchestrator.route_search_query("json parsing crash on empty payload")

    assert result["recommended_tool"] == "code_search"
    assert result["recommended_args"] == {"mode": "bugs"}
    assert result["search_type"] == "semantic"


def test_get_recommended_search_tool_uses_bug_route_tool_name() -> None:
    """Bug/regression routes should still recommend code_search as the tool name."""
    orchestrator = _make_orchestrator()

    result = orchestrator.get_recommended_search_tool("find similar regressions in auth")

    assert result == "code_search"


def test_route_search_query_includes_localize_mode_arguments() -> None:
    """Issue localization queries should route to code_search with localize mode."""
    orchestrator = _make_orchestrator()

    result = orchestrator.route_search_query(
        "which files should I edit to add a logger parameter to BaseRepository"
    )

    assert result["recommended_tool"] == "code_search"
    assert result["recommended_args"] == {"mode": "localize"}
    assert result["search_type"] == "semantic"


def test_route_search_query_includes_impact_mode_arguments() -> None:
    """Change-impact queries should route to code_search with impact mode."""
    orchestrator = _make_orchestrator()

    result = orchestrator.route_search_query("what breaks if I change BaseRepository.save")

    assert result["recommended_tool"] == "code_search"
    assert result["recommended_args"] == {"mode": "impact"}
    assert result["search_type"] == "semantic"


def test_route_search_query_includes_graph_arguments() -> None:
    """Call-graph queries should recommend the graph tool with traversal args."""
    orchestrator = _make_orchestrator()

    result = orchestrator.route_search_query("who calls parse_json")

    assert result["recommended_tool"] == "graph"
    assert result["recommended_args"] == {
        "mode": "callers",
        "node": "parse_json",
        "depth": 2,
    }
    assert result["search_type"] == "semantic"


def test_route_search_query_includes_graph_neighbors_arguments() -> None:
    """Neighbor queries should recommend the graph tool with neighbor args."""
    orchestrator = _make_orchestrator()

    result = orchestrator.route_search_query("show neighbors of BaseProvider")

    assert result["recommended_tool"] == "graph"
    assert result["recommended_args"] == {
        "mode": "neighbors",
        "node": "BaseProvider",
        "depth": 1,
    }
    assert result["search_type"] == "semantic"


def test_route_search_query_includes_graph_path_arguments() -> None:
    """Dependency-path queries should recommend the graph tool with path args."""
    orchestrator = _make_orchestrator()

    result = orchestrator.route_search_query("find dependency path between Parser and Provider")

    assert result["recommended_tool"] == "graph"
    assert result["recommended_args"] == {
        "mode": "path",
        "source": "Parser",
        "target": "Provider",
    }
    assert result["search_type"] == "semantic"


def test_route_search_query_includes_graph_module_pagerank_arguments() -> None:
    """Architecture-hotspot queries should recommend module pagerank graph args."""
    orchestrator = _make_orchestrator()

    result = orchestrator.route_search_query("show the top 4 most important modules")

    assert result["recommended_tool"] == "graph"
    assert result["recommended_args"] == {
        "mode": "module_pagerank",
        "top_k": 4,
        "only_runtime": True,
        "include_callsites": True,
        "max_callsites": 3,
    }
    assert result["search_type"] == "semantic"


def test_route_search_query_includes_graph_file_dependency_arguments() -> None:
    """File-dependency queries should recommend the graph tool with file args."""
    orchestrator = _make_orchestrator()

    result = orchestrator.route_search_query(
        "show file dependencies for victor/agent/orchestrator.py"
    )

    assert result["recommended_tool"] == "graph"
    assert result["recommended_args"] == {
        "mode": "file_deps",
        "file": "victor/agent/orchestrator.py",
    }
    assert result["search_type"] == "semantic"


def test_route_search_query_includes_graph_module_dependents_arguments() -> None:
    """Reverse module-dependency queries should recommend graph neighbor args."""
    orchestrator = _make_orchestrator()

    result = orchestrator.route_search_query("what modules depend on auth")

    assert result["recommended_tool"] == "graph"
    assert result["recommended_args"] == {
        "mode": "neighbors",
        "node": "auth",
        "depth": 1,
        "direction": "in",
        "modules_only": True,
    }
    assert result["search_type"] == "semantic"


def test_route_search_query_includes_graph_module_dependencies_arguments() -> None:
    """Forward module-dependency queries should recommend graph neighbor args."""
    orchestrator = _make_orchestrator()

    result = orchestrator.route_search_query("what modules does auth depend on")

    assert result["recommended_tool"] == "graph"
    assert result["recommended_args"] == {
        "mode": "neighbors",
        "node": "auth",
        "depth": 1,
        "direction": "out",
        "modules_only": True,
    }
    assert result["search_type"] == "semantic"


def test_route_search_query_includes_graph_downstream_dependency_arguments() -> None:
    """Deep downstream dependency queries should preserve requested depth."""
    orchestrator = _make_orchestrator()

    result = orchestrator.route_search_query("show downstream dependencies of auth to depth 3")

    assert result["recommended_tool"] == "graph"
    assert result["recommended_args"] == {
        "mode": "neighbors",
        "node": "auth",
        "depth": 3,
        "direction": "out",
        "modules_only": True,
    }
    assert result["search_type"] == "semantic"


def test_route_search_query_includes_scoped_architecture_summary_arguments() -> None:
    """Scoped architecture summary queries should route to structured graph neighbors."""
    orchestrator = _make_orchestrator()

    result = orchestrator.route_search_query("summarize architecture of auth")

    assert result["recommended_tool"] == "graph"
    assert result["recommended_args"] == {
        "mode": "neighbors",
        "node": "auth",
        "depth": 2,
        "direction": "both",
        "structured": True,
        "include_modules": True,
        "include_symbols": True,
        "include_calls": True,
        "include_refs": True,
    }
    assert result["search_type"] == "semantic"


def test_route_search_query_includes_graph_file_dependents_arguments() -> None:
    """Reverse file-dependency queries should route to graph file dependencies."""
    orchestrator = _make_orchestrator()

    result = orchestrator.route_search_query("what files depend on victor/agent/orchestrator.py")

    assert result["recommended_tool"] == "graph"
    assert result["recommended_args"] == {
        "mode": "file_deps",
        "file": "victor/agent/orchestrator.py",
        "direction": "in",
    }
    assert result["search_type"] == "semantic"


def test_route_search_query_includes_graph_forward_file_dependencies_arguments() -> None:
    """Forward file-dependency queries should route to graph file dependencies."""
    orchestrator = _make_orchestrator()

    result = orchestrator.route_search_query(
        "what files does victor/agent/orchestrator.py depend on"
    )

    assert result["recommended_tool"] == "graph"
    assert result["recommended_args"] == {
        "mode": "file_deps",
        "file": "victor/agent/orchestrator.py",
        "direction": "out",
    }
    assert result["search_type"] == "semantic"


def test_route_search_query_includes_file_architecture_summary_arguments() -> None:
    """File-scoped architecture queries should route to structured file graph output."""
    orchestrator = _make_orchestrator()

    result = orchestrator.route_search_query(
        "summarize architecture around victor/agent/orchestrator.py"
    )

    assert result["recommended_tool"] == "graph"
    assert result["recommended_args"] == {
        "mode": "file_deps",
        "file": "victor/agent/orchestrator.py",
        "direction": "both",
        "structured": True,
        "include_modules": True,
        "include_symbols": True,
        "include_calls": True,
        "include_refs": True,
    }
    assert result["search_type"] == "semantic"


def test_route_search_query_includes_graph_pagerank_arguments() -> None:
    """Centrality queries should recommend the graph tool with pagerank args."""
    orchestrator = _make_orchestrator()

    result = orchestrator.route_search_query("show the top 7 most central symbols in this repo")

    assert result["recommended_tool"] == "graph"
    assert result["recommended_args"] == {"mode": "pagerank", "top_k": 7}
    assert result["search_type"] == "semantic"


def test_get_recommended_search_tool_uses_graph_route_tool_name() -> None:
    """Graph traversal routes should surface graph as the recommended tool."""
    orchestrator = _make_orchestrator()

    result = orchestrator.get_recommended_search_tool("trace execution from main")

    assert result == "graph"


def test_get_recommended_search_tool_uses_localize_route_tool_name() -> None:
    """Issue localization routes should still recommend code_search."""
    orchestrator = _make_orchestrator()

    result = orchestrator.get_recommended_search_tool(
        "localize the issue for repetitive output to counter BREACH attacks"
    )

    assert result == "code_search"


def test_get_recommended_search_tool_uses_impact_route_tool_name() -> None:
    """Impact-analysis routes should still recommend code_search."""
    orchestrator = _make_orchestrator()

    result = orchestrator.get_recommended_search_tool(
        "show me the blast radius if I modify BaseRepository.save"
    )

    assert result == "code_search"


def test_get_recommended_search_tool_uses_graph_pagerank_tool_name() -> None:
    """Centrality questions should recommend the graph tool."""
    orchestrator = _make_orchestrator()

    result = orchestrator.get_recommended_search_tool("what are the most important modules here")

    assert result == "graph"


def test_get_recommended_search_tool_uses_graph_path_tool_name() -> None:
    """Dependency-path questions should recommend the graph tool."""
    orchestrator = _make_orchestrator()

    result = orchestrator.get_recommended_search_tool("show the path from Parser to Provider")

    assert result == "graph"


def test_get_recommended_search_tool_uses_graph_file_dependency_tool_name() -> None:
    """File-dependency questions should recommend the graph tool."""
    orchestrator = _make_orchestrator()

    result = orchestrator.get_recommended_search_tool(
        "what files does victor/agent/orchestrator.py depend on"
    )

    assert result == "graph"


def test_get_recommended_search_tool_uses_graph_architecture_summary_tool_name() -> None:
    """Architecture summary questions should recommend the graph tool."""
    orchestrator = _make_orchestrator()

    result = orchestrator.get_recommended_search_tool("summarize the architecture")

    assert result == "graph"


def test_get_recommended_search_tool_uses_scoped_architecture_summary_tool_name() -> None:
    """Scoped architecture summary questions should recommend the graph tool."""
    orchestrator = _make_orchestrator()

    result = orchestrator.get_recommended_search_tool("explain architecture around auth")

    assert result == "graph"


def test_get_recommended_search_tool_uses_file_architecture_summary_tool_name() -> None:
    """File-scoped architecture questions should recommend the graph tool."""
    orchestrator = _make_orchestrator()

    result = orchestrator.get_recommended_search_tool(
        "explain architecture around victor/agent/orchestrator.py"
    )

    assert result == "graph"
