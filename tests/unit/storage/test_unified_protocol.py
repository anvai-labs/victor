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

"""Tests for unified storage protocol and data types."""

import pytest
from victor.storage.unified.protocol import (
    UnifiedId,
    SymbolType,
    UnifiedSymbol,
    UnifiedEdge,
    SearchMode,
    SearchParams,
    UnifiedSearchResult,
)


class TestUnifiedId:
    """Tests for UnifiedId dataclass."""

    def test_create_symbol_id(self):
        """Test creating a symbol unified ID."""
        uid = UnifiedId.for_symbol("victor/tools/graph_tool.py", "find_symbols")
        assert uid.type == "symbol"
        assert uid.path == "victor/tools/graph_tool.py"
        assert uid.name == "find_symbols"

    def test_create_file_id(self):
        """Test creating a file unified ID."""
        uid = UnifiedId.for_file("victor/tools/graph_tool.py")
        assert uid.type == "file"
        assert uid.path == "victor/tools/graph_tool.py"
        assert uid.name == ""

    def test_create_class_id(self):
        """Test creating a class unified ID."""
        uid = UnifiedId.for_class("victor/agent/orchestrator.py", "AgentOrchestrator")
        assert uid.type == "class"
        assert uid.path == "victor/agent/orchestrator.py"
        assert uid.name == "AgentOrchestrator"

    def test_create_function_id(self):
        """Test creating a function unified ID."""
        uid = UnifiedId.for_function("victor/tools/bash.py", "execute")
        assert uid.type == "function"
        assert uid.path == "victor/tools/bash.py"
        assert uid.name == "execute"

    def test_str_representation_with_name(self):
        """Test string representation of ID with name."""
        uid = UnifiedId.for_symbol("path/to/file.py", "my_func")
        assert str(uid) == "symbol:path/to/file.py:my_func"

    def test_str_representation_without_name(self):
        """Test string representation of ID without name."""
        uid = UnifiedId.for_file("path/to/file.py")
        assert str(uid) == "file:path/to/file.py"

    def test_from_string_with_name(self):
        """Test parsing unified ID with name from string."""
        uid = UnifiedId.from_string("symbol:victor/tools.py:find")
        assert uid.type == "symbol"
        assert uid.path == "victor/tools.py"
        assert uid.name == "find"

    def test_from_string_without_name(self):
        """Test parsing unified ID without name from string."""
        uid = UnifiedId.from_string("file:victor/tools.py")
        assert uid.type == "file"
        assert uid.path == "victor/tools.py"
        assert uid.name == ""

    def test_from_string_invalid_format(self):
        """Test that invalid format raises ValueError."""
        with pytest.raises(ValueError, match="Invalid unified ID format"):
            UnifiedId.from_string("invalid")

    def test_frozen_immutable(self):
        """Test that UnifiedId is immutable (frozen)."""
        uid = UnifiedId.for_symbol("file.py", "func")
        with pytest.raises(AttributeError):
            uid.type = "class"

    def test_equality(self):
        """Test equality comparison."""
        uid1 = UnifiedId.for_symbol("file.py", "func")
        uid2 = UnifiedId.for_symbol("file.py", "func")
        uid3 = UnifiedId.for_symbol("file.py", "other")
        assert uid1 == uid2
        assert uid1 != uid3

    def test_hashable(self):
        """Test that UnifiedId is hashable (can be used in sets/dicts)."""
        uid1 = UnifiedId.for_symbol("file.py", "func")
        uid2 = UnifiedId.for_symbol("file.py", "func")
        uid3 = UnifiedId.for_symbol("file.py", "other")

        uid_set = {uid1, uid2, uid3}
        assert len(uid_set) == 2  # uid1 and uid2 are equal


class TestUnifiedSymbol:
    """Tests for UnifiedSymbol dataclass."""

    def test_create_minimal(self):
        """Test creating minimal symbol."""
        symbol = UnifiedSymbol(
            unified_id="symbol:file.py:func",
            name="func",
            type="function",
            file_path="file.py",
        )
        assert symbol.unified_id == "symbol:file.py:func"
        assert symbol.name == "func"
        assert symbol.type == "function"
        assert symbol.file_path == "file.py"

    def test_create_full(self):
        """Test creating symbol with all fields."""
        symbol = UnifiedSymbol(
            unified_id="symbol:file.py:MyClass",
            name="MyClass",
            type="class",
            file_path="file.py",
            line=10,
            end_line=50,
            lang="python",
            signature="class MyClass(Base):",
            docstring="A test class.",
            parent_id=None,
            callers=["symbol:other.py:use_class"],
            callees=["symbol:base.py:Base"],
            semantic_score=0.95,
            graph_score=0.8,
            combined_score=0.87,
            metadata={"test": True},
        )
        assert symbol.line == 10
        assert symbol.end_line == 50
        assert symbol.lang == "python"
        assert symbol.semantic_score == 0.95
        assert len(symbol.callers) == 1
        assert len(symbol.callees) == 1

    def test_default_lists(self):
        """Test that list fields default to empty."""
        symbol = UnifiedSymbol(
            unified_id="test",
            name="test",
            type="function",
            file_path="test.py",
        )
        assert symbol.callers == []
        assert symbol.callees == []
        assert symbol.inherits == []
        assert symbol.implementors == []
        assert symbol.metadata == {}


class TestUnifiedEdge:
    """Tests for UnifiedEdge dataclass."""

    def test_create_edge(self):
        """Test creating an edge."""
        edge = UnifiedEdge(
            src_id="symbol:a.py:func_a",
            dst_id="symbol:b.py:func_b",
            type="CALLS",
        )
        assert edge.src_id == "symbol:a.py:func_a"
        assert edge.dst_id == "symbol:b.py:func_b"
        assert edge.type == "CALLS"

    def test_create_edge_with_weight(self):
        """Test creating edge with weight."""
        edge = UnifiedEdge(
            src_id="src",
            dst_id="dst",
            type="INHERITS",
            weight=0.5,
            metadata={"depth": 1},
        )
        assert edge.weight == 0.5
        assert edge.metadata["depth"] == 1


class TestSearchParams:
    """Tests for SearchParams dataclass."""

    def test_default_params(self):
        """Test default search parameters."""
        params = SearchParams(query="find authentication")
        assert params.query == "find authentication"
        assert params.mode == SearchMode.HYBRID
        assert params.limit == 20
        assert params.semantic_weight == 0.7
        assert params.graph_weight == 0.3
        assert params.similarity_threshold == 0.25

    def test_custom_params(self):
        """Test custom search parameters."""
        params = SearchParams(
            query="test",
            mode=SearchMode.SEMANTIC,
            limit=10,
            file_patterns=["*.py"],
            symbol_types=["function", "class"],
            semantic_weight=0.9,
            graph_weight=0.1,
            include_neighbors=True,
            max_depth=2,
        )
        assert params.mode == SearchMode.SEMANTIC
        assert params.limit == 10
        assert params.file_patterns == ["*.py"]
        assert params.symbol_types == ["function", "class"]
        assert params.include_neighbors is True
        assert params.max_depth == 2


class TestUnifiedSearchResult:
    """Tests for UnifiedSearchResult dataclass."""

    def test_create_result(self):
        """Test creating a search result."""
        symbol = UnifiedSymbol(
            unified_id="symbol:file.py:func",
            name="func",
            type="function",
            file_path="file.py",
        )
        result = UnifiedSearchResult(
            symbol=symbol,
            score=0.85,
            match_type="hybrid",
            semantic_score=0.9,
            keyword_score=0.8,
            graph_score=0.75,
            matched_content="def func():",
        )
        assert result.symbol.name == "func"
        assert result.score == 0.85
        assert result.match_type == "hybrid"
        assert result.semantic_score == 0.9


class TestSearchMode:
    """Tests for SearchMode enum."""

    def test_search_modes(self):
        """Test all search modes exist."""
        assert SearchMode.KEYWORD.value == "keyword"
        assert SearchMode.SEMANTIC.value == "semantic"
        assert SearchMode.HYBRID.value == "hybrid"
        assert SearchMode.GRAPH.value == "graph"

    def test_search_mode_is_string(self):
        """Test that search mode is a string enum."""
        assert isinstance(SearchMode.HYBRID.value, str)
        assert str(SearchMode.HYBRID) == "SearchMode.HYBRID"


class TestSymbolType:
    """Tests for SymbolType enum."""

    def test_symbol_types(self):
        """Test all symbol types exist."""
        assert SymbolType.FILE.value == "file"
        assert SymbolType.MODULE.value == "module"
        assert SymbolType.CLASS.value == "class"
        assert SymbolType.FUNCTION.value == "function"
        assert SymbolType.METHOD.value == "method"
        assert SymbolType.VARIABLE.value == "variable"
        assert SymbolType.CONSTANT.value == "constant"
        assert SymbolType.IMPORT.value == "import"
