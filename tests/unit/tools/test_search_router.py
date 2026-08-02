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

"""Tests for SearchRouter - Gap 3 implementation."""

import pytest


class TestKeywordRouting:
    """Tests for KEYWORD routing."""

    def test_quoted_string_keyword(self):
        """Test that quoted strings route to KEYWORD."""
        from victor.agent.search_router import SearchRouter, SearchType

        router = SearchRouter()
        result = router.route('Find "class BaseTool" in the codebase')

        assert result.search_type == SearchType.KEYWORD
        assert result.confidence == 1.0
        assert "class BaseTool" in result.transformed_query

    def test_single_quoted_string(self):
        """Test that single-quoted strings route to KEYWORD."""
        from victor.agent.search_router import SearchRouter, SearchType

        router = SearchRouter()
        result = router.route("Search for 'def execute'")

        assert result.search_type == SearchType.KEYWORD
        assert result.confidence == 1.0

    def test_class_name_keyword(self):
        """Test that class names route to KEYWORD."""
        from victor.agent.search_router import SearchRouter, SearchType

        router = SearchRouter()
        result = router.route("class AgentOrchestrator")

        assert result.search_type == SearchType.KEYWORD
        assert "class_name" in result.matched_patterns

    def test_function_def_keyword(self):
        """Test that function definitions route to KEYWORD."""
        from victor.agent.search_router import SearchRouter, SearchType

        router = SearchRouter()
        result = router.route("def execute_tool")

        assert result.search_type == SearchType.KEYWORD
        assert "function_def" in result.matched_patterns

    def test_import_statement_keyword(self):
        """Test that import statements route to KEYWORD."""
        from victor.agent.search_router import SearchRouter, SearchType

        router = SearchRouter()
        result = router.route("import victor.agent.orchestrator")

        assert result.search_type == SearchType.KEYWORD
        assert "import_statement" in result.matched_patterns

    def test_error_class_keyword(self):
        """Test that error classes route to KEYWORD."""
        from victor.agent.search_router import SearchRouter, SearchType

        router = SearchRouter()
        result = router.route("ToolExecutionError")

        assert result.search_type == SearchType.KEYWORD
        assert "error_class" in result.matched_patterns

    def test_file_extension_keyword(self):
        """Test that file patterns route to KEYWORD."""
        from victor.agent.search_router import SearchRouter, SearchType

        router = SearchRouter()
        result = router.route("orchestrator.py")

        assert result.search_type == SearchType.KEYWORD
        assert "file_extension" in result.matched_patterns

    def test_decorator_keyword(self):
        """Test that decorators route to KEYWORD."""
        from victor.agent.search_router import SearchRouter, SearchType

        router = SearchRouter()
        result = router.route("@property")

        assert result.search_type == SearchType.KEYWORD
        assert "decorator" in result.matched_patterns


class TestSemanticRouting:
    """Tests for SEMANTIC routing."""

    def test_how_question_semantic(self):
        """Test that 'how does' routes to SEMANTIC."""
        from victor.agent.search_router import SearchRouter, SearchType

        router = SearchRouter()
        result = router.route("how does error handling work in this project")

        assert result.search_type == SearchType.SEMANTIC
        assert "how_question" in result.matched_patterns

    def test_why_question_semantic(self):
        """Test that 'why does' routes to SEMANTIC."""
        from victor.agent.search_router import SearchRouter, SearchType

        router = SearchRouter()
        result = router.route("why does the orchestrator use streaming")

        assert result.search_type == SearchType.SEMANTIC
        assert "why_question" in result.matched_patterns

    def test_explain_semantic(self):
        """Test that 'explain' routes to SEMANTIC."""
        from victor.agent.search_router import SearchRouter, SearchType

        router = SearchRouter()
        result = router.route("explain the tool selection mechanism")

        assert result.search_type == SearchType.SEMANTIC
        assert "explain" in result.matched_patterns

    def test_patterns_semantic(self):
        """Test that 'patterns' routes to SEMANTIC."""
        from victor.agent.search_router import SearchRouter, SearchType

        router = SearchRouter()
        result = router.route("find error handling patterns")

        assert result.search_type == SearchType.SEMANTIC
        assert "patterns" in result.matched_patterns

    def test_architecture_semantic(self):
        """Test that 'architecture' routes to SEMANTIC."""
        from victor.agent.search_router import SearchRouter, SearchType

        router = SearchRouter()
        result = router.route("understand the architecture")

        assert result.search_type == SearchType.SEMANTIC
        assert "architecture" in result.matched_patterns

    def test_best_practices_semantic(self):
        """Test that 'best practices' routes to SEMANTIC."""
        from victor.agent.search_router import SearchRouter, SearchType

        router = SearchRouter()
        result = router.route("find best practices for error handling")

        assert result.search_type == SearchType.SEMANTIC
        assert "best_practices" in result.matched_patterns


class TestHybridRouting:
    """Tests for HYBRID routing."""

    def test_mixed_signals_hybrid(self):
        """Test that mixed signals can route to HYBRID."""
        from victor.agent.search_router import SearchRouter, SearchType

        router = SearchRouter()
        # Query with both keyword (class name) and semantic (how) signals
        result = router.route("how does class BaseTool work")

        # Should be hybrid or one of keyword/semantic
        assert result.search_type in (
            SearchType.HYBRID,
            SearchType.KEYWORD,
            SearchType.SEMANTIC,
        )


class TestDefaultRouting:
    """Tests for default routing behavior."""

    def test_ambiguous_query_defaults_keyword(self):
        """Test that ambiguous queries default to KEYWORD."""
        from victor.agent.search_router import SearchRouter, SearchType

        router = SearchRouter()
        result = router.route("xyzzy foobar baz")

        # Should default to keyword for precision
        assert result.search_type == SearchType.KEYWORD
        assert result.confidence < 0.5

    def test_empty_query(self):
        """Test empty query handling."""
        from victor.agent.search_router import SearchRouter, SearchType

        router = SearchRouter()
        result = router.route("")

        assert result.search_type == SearchType.KEYWORD
        assert result.confidence < 0.5


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_route_query(self):
        """Test route_query convenience function."""
        from victor.agent.search_router import route_query, SearchType

        result = route_query("class BaseTool")
        assert result.search_type == SearchType.KEYWORD

    def test_suggest_search_tool_keyword(self):
        """Test suggest_search_tool returns code_search for keyword."""
        from victor.agent.search_router import suggest_search_tool

        tool = suggest_search_tool("class BaseTool")
        assert tool == "code_search"

    def test_suggest_search_tool_semantic(self):
        """Test suggest_search_tool returns semantic_code_search."""
        from victor.agent.search_router import suggest_search_tool

        tool = suggest_search_tool("how does the caching mechanism work")
        assert tool == "semantic_code_search"

    def test_suggest_search_tool_bug_similarity(self):
        """Bug/regression queries should keep using code_search with bug mode."""
        from victor.agent.search_router import suggest_search_tool

        tool = suggest_search_tool("find similar crashes in the auth flow")
        assert tool == "code_search"

    def test_suggest_search_tool_graph_callers(self):
        """Call-graph questions should recommend the graph tool."""
        from victor.agent.search_router import suggest_search_tool

        tool = suggest_search_tool("who calls parse_json")
        assert tool == "graph"

    def test_suggest_search_tool_graph_pagerank(self):
        """Centrality questions should recommend the graph tool."""
        from victor.agent.search_router import suggest_search_tool

        tool = suggest_search_tool("show the most central symbols in this repo")
        assert tool == "graph"

    def test_suggest_search_tool_graph_path(self):
        """Dependency-path questions should recommend the graph tool."""
        from victor.agent.search_router import suggest_search_tool

        tool = suggest_search_tool("find dependency path between Parser and Provider")
        assert tool == "graph"

    def test_suggest_search_tool_graph_file_dependencies(self):
        """File-dependency questions should recommend the graph tool."""
        from victor.agent.search_router import suggest_search_tool

        tool = suggest_search_tool("show file dependencies for victor/agent/orchestrator.py")
        assert tool == "graph"

    def test_suggest_search_tool_graph_architecture_summary(self):
        """Architecture summary questions should recommend the graph tool."""
        from victor.agent.search_router import suggest_search_tool

        tool = suggest_search_tool("summarize the architecture")
        assert tool == "graph"

    def test_suggest_search_tool_graph_scoped_architecture_summary(self):
        """Scoped architecture summary questions should recommend the graph tool."""
        from victor.agent.search_router import suggest_search_tool

        tool = suggest_search_tool("summarize architecture of auth")
        assert tool == "graph"

    def test_suggest_search_tool_graph_file_architecture_summary(self):
        """File-scoped architecture questions should recommend the graph tool."""
        from victor.agent.search_router import suggest_search_tool

        tool = suggest_search_tool("summarize architecture around victor/agent/orchestrator.py")
        assert tool == "graph"

    def test_is_keyword_query(self):
        """Test is_keyword_query function."""
        from victor.agent.search_router import is_keyword_query

        assert is_keyword_query("class BaseTool")
        assert not is_keyword_query("how does error handling work")

    def test_is_semantic_query(self):
        """Test is_semantic_query function."""
        from victor.agent.search_router import is_semantic_query

        assert is_semantic_query("explain the architecture")
        assert not is_semantic_query("class BaseTool")


class TestCustomRouter:
    """Tests for custom router support."""

    def test_custom_router_takes_precedence(self):
        """Test that custom routers are tried first."""
        from victor.agent.search_router import SearchRouter, SearchRoute, SearchType

        def custom_router(query: str):
            if "URGENT" in query.upper():
                return SearchRoute(
                    search_type=SearchType.KEYWORD,
                    confidence=1.0,
                    reason="Urgent query",
                    transformed_query=query,
                    matched_patterns=["urgent_override"],
                )
            return None

        router = SearchRouter(custom_routers=[custom_router])
        result = router.route("URGENT: find all errors")

        assert result.confidence == 1.0
        assert "urgent_override" in result.matched_patterns


class TestRealWorldQueries:
    """Tests with real-world example queries."""

    def test_find_class_inherits(self):
        """Test 'find classes that inherit' routes correctly."""
        from victor.agent.search_router import SearchRouter, SearchType

        router = SearchRouter()
        result = router.route("find all classes that inherit from BaseTool")

        # "class" is keyword signal, but "find all" and conceptual query makes it semantic
        assert result.search_type in (
            SearchType.KEYWORD,
            SearchType.SEMANTIC,
            SearchType.HYBRID,
        )

    def test_specific_function_search(self):
        """Test specific function search routes to KEYWORD."""
        from victor.agent.search_router import SearchRouter, SearchType

        router = SearchRouter()
        result = router.route("def _execute_tool_call")

        assert result.search_type == SearchType.KEYWORD

    def test_error_handling_conceptual(self):
        """Test conceptual error handling query routes to SEMANTIC."""
        from victor.agent.search_router import SearchRouter, SearchType

        router = SearchRouter()
        result = router.route("how is error handling implemented across providers")

        assert result.search_type == SearchType.SEMANTIC

    def test_bug_similarity_query_sets_tool_arguments(self):
        """Bug-style queries should route to code_search with mode='bugs'."""
        from victor.agent.search_router import SearchRouter, SearchType

        router = SearchRouter()
        result = router.route("json parsing crash on empty payload")

        assert result.search_type == SearchType.SEMANTIC
        assert result.tool_name == "code_search"
        assert result.tool_arguments == {"mode": "bugs"}
        assert "crash" in result.matched_patterns

    def test_callers_query_routes_to_graph_tool(self):
        """Caller questions should route to graph(mode='callers')."""
        from victor.agent.search_router import SearchRouter, SearchType

        router = SearchRouter()
        result = router.route("who calls parse_json?")

        assert result.search_type == SearchType.SEMANTIC
        assert result.tool_name == "graph"
        assert result.tool_arguments == {
            "mode": "callers",
            "node": "parse_json",
            "depth": 2,
        }
        assert result.transformed_query == "parse_json"
        assert "who_calls" in result.matched_patterns

    def test_callees_query_routes_to_graph_tool(self):
        """Forward call questions should route to graph(mode='callees')."""
        from victor.agent.search_router import SearchRouter

        router = SearchRouter()
        result = router.route("what does main call")

        assert result.tool_name == "graph"
        assert result.tool_arguments == {"mode": "callees", "node": "main", "depth": 2}

    def test_trace_query_routes_to_graph_tool(self):
        """Execution-path questions should route to graph(mode='trace')."""
        from victor.agent.search_router import SearchRouter

        router = SearchRouter()
        result = router.route('trace execution from "main"')

        assert result.tool_name == "graph"
        assert result.tool_arguments == {"mode": "trace", "node": "main", "depth": 3}

    def test_neighbors_query_routes_to_graph_tool(self):
        """Neighbor questions should route to graph(mode='neighbors')."""
        from victor.agent.search_router import SearchRouter

        router = SearchRouter()
        result = router.route("show neighbors of BaseProvider")

        assert result.tool_name == "graph"
        assert result.tool_arguments == {
            "mode": "neighbors",
            "node": "BaseProvider",
            "depth": 1,
        }
        assert result.transformed_query == "BaseProvider"
        assert "neighbor_of" in result.matched_patterns

    def test_path_query_routes_to_graph_tool(self):
        """Dependency-path questions should route to graph(mode='path')."""
        from victor.agent.search_router import SearchRouter

        router = SearchRouter()
        result = router.route("find dependency path between Parser and Provider")

        assert result.tool_name == "graph"
        assert result.tool_arguments == {
            "mode": "path",
            "source": "Parser",
            "target": "Provider",
        }
        assert result.transformed_query is None
        assert "path_between" in result.matched_patterns

    def test_module_pagerank_query_routes_to_graph_tool(self):
        """Architecture-hotspot questions should route to graph(mode='module_pagerank')."""
        from victor.agent.search_router import SearchRouter

        router = SearchRouter()
        result = router.route("show the top 4 most important modules")

        assert result.tool_name == "graph"
        assert result.tool_arguments == {
            "mode": "module_pagerank",
            "top_k": 4,
            "only_runtime": True,
            "include_callsites": True,
            "max_callsites": 3,
        }
        assert "important_modules" in result.matched_patterns

    def test_file_dependency_query_routes_to_graph_tool(self):
        """File-dependency questions should route to graph(mode='file_deps')."""
        from victor.agent.search_router import SearchRouter

        router = SearchRouter()
        result = router.route("show file dependencies for victor/agent/orchestrator.py")

        assert result.tool_name == "graph"
        assert result.tool_arguments == {
            "mode": "file_deps",
            "file": "victor/agent/orchestrator.py",
        }
        assert "file_dependencies_for" in result.matched_patterns

    def test_module_dependents_query_routes_to_graph_tool(self):
        """Reverse module-dependency questions should route to graph neighbors."""
        from victor.agent.search_router import SearchRouter

        router = SearchRouter()
        result = router.route("what modules depend on auth")

        assert result.tool_name == "graph"
        assert result.tool_arguments == {
            "mode": "neighbors",
            "node": "auth",
            "depth": 1,
            "direction": "in",
            "modules_only": True,
        }
        assert "module_dependents" in result.matched_patterns

    def test_module_dependencies_query_routes_to_graph_tool(self):
        """Forward module-dependency questions should route to graph neighbors."""
        from victor.agent.search_router import SearchRouter

        router = SearchRouter()
        result = router.route("what modules does auth depend on")

        assert result.tool_name == "graph"
        assert result.tool_arguments == {
            "mode": "neighbors",
            "node": "auth",
            "depth": 1,
            "direction": "out",
            "modules_only": True,
        }
        assert "module_dependencies" in result.matched_patterns

    def test_architecture_summary_query_routes_to_graph_tool(self):
        """Architecture summary questions should route to structured module graph ranking."""
        from victor.agent.search_router import SearchRouter

        router = SearchRouter()
        result = router.route("summarize the architecture")

        assert result.tool_name == "graph"
        assert result.tool_arguments == {
            "mode": "module_pagerank",
            "top_k": 5,
            "only_runtime": True,
            "include_callsites": True,
            "max_callsites": 3,
            "structured": True,
            "include_modules": True,
        }
        assert "architecture_summary" in result.matched_patterns

    def test_downstream_dependency_query_routes_to_graph_tool(self):
        """Deep downstream dependency questions should route to graph neighbors."""
        from victor.agent.search_router import SearchRouter

        router = SearchRouter()
        result = router.route("show downstream dependencies of auth to depth 3")

        assert result.tool_name == "graph"
        assert result.tool_arguments == {
            "mode": "neighbors",
            "node": "auth",
            "depth": 3,
            "direction": "out",
            "modules_only": True,
        }
        assert "downstream_dependencies" in result.matched_patterns

    def test_scoped_architecture_summary_query_routes_to_graph_tool(self):
        """Scoped architecture questions should route to structured graph neighbors."""
        from victor.agent.search_router import SearchRouter

        router = SearchRouter()
        result = router.route("summarize architecture of auth")

        assert result.tool_name == "graph"
        assert result.tool_arguments == {
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
        assert "scoped_architecture_summary" in result.matched_patterns

    def test_file_dependents_query_routes_to_graph_tool(self):
        """Reverse file-dependency questions should route to graph file dependencies."""
        from victor.agent.search_router import SearchRouter

        router = SearchRouter()
        result = router.route("what files depend on victor/agent/orchestrator.py")

        assert result.tool_name == "graph"
        assert result.tool_arguments == {
            "mode": "file_deps",
            "file": "victor/agent/orchestrator.py",
            "direction": "in",
        }
        assert "file_dependents" in result.matched_patterns

    def test_forward_file_dependencies_query_routes_to_graph_tool(self):
        """Forward file-dependency questions should route to graph file dependencies."""
        from victor.agent.search_router import SearchRouter

        router = SearchRouter()
        result = router.route("what files does victor/agent/orchestrator.py depend on")

        assert result.tool_name == "graph"
        assert result.tool_arguments == {
            "mode": "file_deps",
            "file": "victor/agent/orchestrator.py",
            "direction": "out",
        }
        assert "file_depends_on" in result.matched_patterns

    def test_file_architecture_summary_query_routes_to_graph_tool(self):
        """File-scoped architecture questions should route to structured file graph output."""
        from victor.agent.search_router import SearchRouter

        router = SearchRouter()
        result = router.route("summarize architecture around victor/agent/orchestrator.py")

        assert result.tool_name == "graph"
        assert result.tool_arguments == {
            "mode": "file_deps",
            "file": "victor/agent/orchestrator.py",
            "direction": "both",
            "structured": True,
            "include_modules": True,
            "include_symbols": True,
            "include_calls": True,
            "include_refs": True,
        }
        assert "file_architecture_summary" in result.matched_patterns

    def test_pagerank_query_routes_to_graph_tool(self):
        """Centrality questions should route to graph(mode='pagerank')."""
        from victor.agent.search_router import SearchRouter

        router = SearchRouter()
        result = router.route("show the top 7 most central symbols in this repo")

        assert result.tool_name == "graph"
        assert result.tool_arguments == {"mode": "pagerank", "top_k": 7}
        assert "central_symbols" in result.matched_patterns

    def test_import_exact_match(self):
        """Test exact import routes to KEYWORD."""
        from victor.agent.search_router import SearchRouter, SearchType

        router = SearchRouter()
        result = router.route("from victor.tools.base import BaseTool")

        assert result.search_type == SearchType.KEYWORD


class TestQuotedStrings:
    """Tests for quoted string handling."""

    def test_double_quoted_extraction(self):
        """Test double-quoted string extraction."""
        from victor.agent.search_router import SearchRouter

        router = SearchRouter()
        result = router.route('search for "ToolRegistry"')

        assert result.transformed_query == "ToolRegistry"

    def test_single_quoted_extraction(self):
        """Test single-quoted string extraction."""
        from victor.agent.search_router import SearchRouter

        router = SearchRouter()
        result = router.route("find 'execute_bash'")

        assert result.transformed_query == "execute_bash"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
