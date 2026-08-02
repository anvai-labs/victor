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

"""Unit tests for CCG builder."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from victor.core.indexing.ccg_builder import (
    CodeContextGraphBuilder,
    StatementType,
    SUPPORTED_CCG_LANGUAGES,
    BasicBlock,
    VariableInfo,
)


class TestStatementType:
    """Tests for StatementType enum."""

    def test_statement_types_exist(self) -> None:
        """Test that all statement types are defined."""
        assert StatementType.CONDITION == "condition"
        assert StatementType.LOOP == "loop"
        assert StatementType.TRY == "try"
        assert StatementType.CATCH == "catch"
        assert StatementType.FINALLY == "finally"
        assert StatementType.SWITCH == "switch"
        assert StatementType.CASE == "case"
        assert StatementType.DEFAULT == "default"
        assert StatementType.ASSIGNMENT == "assignment"
        assert StatementType.CALL == "call"
        assert StatementType.RETURN == "return"
        assert StatementType.YIELD == "yield"
        assert StatementType.AWAIT == "await"
        assert StatementType.THROW == "throw"
        assert StatementType.FUNCTION_DEF == "function_def"
        assert StatementType.CLASS_DEF == "class_def"
        assert StatementType.VARIABLE_DEF == "variable_def"
        assert StatementType.BLOCK == "block"
        assert StatementType.EXPRESSION == "expression"
        assert StatementType.UNKNOWN == "unknown"


class TestBasicBlock:
    """Tests for BasicBlock dataclass."""

    def test_create_basic_block(self) -> None:
        """Test creating a BasicBlock."""
        block = BasicBlock(
            block_id="block_1",
            entry_line=10,
            exit_line=20,
            statements=["stmt1", "stmt2"],
            successors=["block_2"],
            predecessors=["block_0"],
        )
        assert block.block_id == "block_1"
        assert block.entry_line == 10
        assert block.exit_line == 20
        assert block.statements == ["stmt1", "stmt2"]
        assert block.successors == ["block_2"]
        assert block.predecessors == ["block_0"]
        assert block.is_loop_body is False
        assert block.is_conditional is False

    def test_basic_block_defaults(self) -> None:
        """Test BasicBlock default values."""
        block = BasicBlock(
            block_id="block_2",
            entry_line=30,
            exit_line=40,
        )
        assert block.statements == []
        assert block.successors == []
        assert block.predecessors == []
        assert block.is_loop_body is False
        assert block.is_conditional is False
        assert block.loop_type is None
        assert block.condition is None


class TestVariableInfo:
    """Tests for VariableInfo dataclass."""

    def test_create_variable_info(self) -> None:
        """Test creating a VariableInfo."""
        var = VariableInfo(
            name="x",
            defining_node="node_1",
            scope="scope_123",
            type="int",
        )
        assert var.name == "x"
        assert var.defining_node == "node_1"
        assert var.scope == "scope_123"
        assert var.type == "int"
        assert var.use_sites == []

    def test_variable_info_with_use_sites(self) -> None:
        """Test VariableInfo with use sites."""
        var = VariableInfo(
            name="y",
            defining_node="node_2",
            use_sites=["node_3", "node_4"],
        )
        assert var.use_sites == ["node_3", "node_4"]


class TestCodeContextGraphBuilder:
    """Tests for CodeContextGraphBuilder."""

    @pytest.fixture
    def mock_graph_store(self) -> AsyncMock:
        """Create a mock graph store."""
        store = AsyncMock()
        store.upsert_nodes = AsyncMock()
        store.upsert_edges = AsyncMock()
        return store

    @pytest.fixture
    def builder(self, mock_graph_store: AsyncMock) -> CodeContextGraphBuilder:
        """Create a CCG builder instance."""
        return CodeContextGraphBuilder(mock_graph_store, language="python")

    def test_initialization(self, builder: CodeContextGraphBuilder) -> None:
        """Test builder initialization."""
        assert builder.language == "python"
        assert builder.graph_store is not None

    def test_initialization_with_language(self) -> None:
        """Test builder initialization with specific language."""
        store = AsyncMock()
        builder = CodeContextGraphBuilder(store, language="javascript")
        assert builder.language == "javascript"

    def test_supported_ccg_languages(self) -> None:
        """Test that supported languages are defined."""
        assert "python" in SUPPORTED_CCG_LANGUAGES
        assert "javascript" in SUPPORTED_CCG_LANGUAGES
        assert "typescript" in SUPPORTED_CCG_LANGUAGES
        assert "go" in SUPPORTED_CCG_LANGUAGES
        assert "rust" in SUPPORTED_CCG_LANGUAGES

    def test_detect_language_python(self, builder: CodeContextGraphBuilder) -> None:
        """Test language detection for Python."""
        assert builder._detect_language(Path("test.py")) == "python"
        assert builder._detect_language(Path("test.py")) == "python"

    def test_detect_language_javascript(self, builder: CodeContextGraphBuilder) -> None:
        """Test language detection for JavaScript."""
        assert builder._detect_language(Path("test.js")) == "javascript"
        assert builder._detect_language(Path("test.jsx")) == "javascript"

    def test_detect_language_typescript(self, builder: CodeContextGraphBuilder) -> None:
        """Test language detection for TypeScript."""
        assert builder._detect_language(Path("test.ts")) == "typescript"
        assert builder._detect_language(Path("test.tsx")) == "typescript"

    def test_detect_language_go(self, builder: CodeContextGraphBuilder) -> None:
        """Test language detection for Go."""
        assert builder._detect_language(Path("test.go")) == "go"

    def test_detect_language_rust(self, builder: CodeContextGraphBuilder) -> None:
        """Test language detection for Rust."""
        assert builder._detect_language(Path("test.rs")) == "rust"

    def test_detect_language_java(self, builder: CodeContextGraphBuilder) -> None:
        """Test language detection for Java."""
        assert builder._detect_language(Path("test.java")) == "java"

    def test_detect_language_cpp(self, builder: CodeContextGraphBuilder) -> None:
        """Test language detection for C++."""
        assert builder._detect_language(Path("test.cpp")) == "cpp"
        assert builder._detect_language(Path("test.cc")) == "cpp"
        assert builder._detect_language(Path("test.cxx")) == "cpp"
        assert builder._detect_language(Path("test.hpp")) == "cpp"

    def test_detect_language_c(self, builder: CodeContextGraphBuilder) -> None:
        """Test language detection for C."""
        assert builder._detect_language(Path("test.c")) == "c"
        assert builder._detect_language(Path("test.h")) == "c"

    def test_detect_language_unknown(self, builder: CodeContextGraphBuilder) -> None:
        """Test language detection for unknown extensions."""
        assert builder._detect_language(Path("test.unknown")) == "unknown"
        assert builder._detect_language(Path("test")) == "unknown"

    def test_generate_statement_id(self, builder: CodeContextGraphBuilder) -> None:
        """Test statement ID generation is deterministic."""
        file_path = "/path/to/file.py"
        start_line = 10
        end_line = 20
        node_type = "function_definition"

        id1 = builder._generate_statement_id(file_path, start_line, end_line, node_type)
        id2 = builder._generate_statement_id(file_path, start_line, end_line, node_type)

        assert id1 == id2  # Same inputs should produce same ID
        assert len(id1) == 16  # SHA256 truncated to 16 chars

    def test_generate_statement_id_unique(self, builder: CodeContextGraphBuilder) -> None:
        """Test that different inputs produce different IDs."""
        id1 = builder._generate_statement_id("/path/to/file.py", 10, 20, "function_definition")
        id2 = builder._generate_statement_id("/path/to/file.py", 11, 20, "function_definition")
        id3 = builder._generate_statement_id("/path/to/other.py", 10, 20, "function_definition")

        assert id1 != id2  # Different line
        assert id1 != id3  # Different file

    def test_classify_statement_python(self, builder: CodeContextGraphBuilder) -> None:
        """Test statement classification for Python."""
        assert builder._classify_statement("if_statement") == StatementType.CONDITION
        assert builder._classify_statement("for_statement") == StatementType.LOOP
        assert builder._classify_statement("while_statement") == StatementType.LOOP
        assert builder._classify_statement("try_statement") == StatementType.TRY
        assert builder._classify_statement("except_clause") == StatementType.CATCH
        assert builder._classify_statement("finally_clause") == StatementType.FINALLY
        assert builder._classify_statement("assignment") == StatementType.ASSIGNMENT
        assert builder._classify_statement("call") == StatementType.CALL
        assert builder._classify_statement("return_statement") == StatementType.RETURN
        assert builder._classify_statement("yield_statement") == StatementType.YIELD
        assert builder._classify_statement("function_definition") == StatementType.FUNCTION_DEF
        assert builder._classify_statement("class_definition") == StatementType.CLASS_DEF

    def test_classify_statement_unknown(self, builder: CodeContextGraphBuilder) -> None:
        """Test statement classification for unknown types."""
        assert builder._classify_statement("unknown_type") == StatementType.UNKNOWN
        assert builder._classify_statement("random_stuff") == StatementType.UNKNOWN

    @pytest.mark.asyncio
    async def test_build_ccg_for_file_unsupported_language(
        self, builder: CodeContextGraphBuilder
    ) -> None:
        """Test CCG building for unsupported language returns empty."""
        # Create a temporary file
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".unknown", delete=False) as f:
            f.write("test content")
            temp_path = Path(f.name)

        try:
            nodes, edges = await builder.build_ccg_for_file(temp_path)
            assert nodes == []
            assert edges == []
        finally:
            temp_path.unlink()

    @pytest.mark.asyncio
    async def test_build_ccg_for_file_missing_tree_sitter(
        self, mock_graph_store: AsyncMock
    ) -> None:
        """Test CCG building gracefully handles missing Tree-sitter."""
        import tempfile

        builder = CodeContextGraphBuilder(mock_graph_store, language="python")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("def foo(): pass")
            temp_path = Path(f.name)

        try:
            # Mock the internal parse method to raise ImportError
            # This simulates tree-sitter not being available
            original_parse = getattr(builder, "_parse_with_tree_sitter", None)

            async def mock_parse(*args, **kwargs):
                raise ImportError("tree_sitter_languages not available")

            if original_parse:
                builder._parse_with_tree_sitter = mock_parse

            nodes, edges = await builder.build_ccg_for_file(temp_path)

            # Should return empty results gracefully when tree-sitter is unavailable
            assert isinstance(nodes, list)
            assert isinstance(edges, list)
        finally:
            temp_path.unlink()

    @pytest.mark.asyncio
    async def test_build_ccg_for_file_uses_per_file_language(
        self, mock_graph_store: AsyncMock, tmp_path: Path
    ) -> None:
        """Language override should drive parsing/classification for mixed-language batches."""
        builder = CodeContextGraphBuilder(mock_graph_store, language="python")
        file_path = tmp_path / "main.rs"
        file_path.write_text("fn main() {}\n", encoding="utf-8")
        observed: list[tuple[str, str | None]] = []

        def fake_parse(source_code: str, path: Path, language: str | None = None):
            observed.append((builder.language, language))
            return object()

        async def fake_build_cfg(ast_root, path: Path, source_code: str):
            assert builder.language == "rust"
            return [], []

        async def fake_build_cdg(nodes, edges):
            assert builder.language == "rust"
            return []

        async def fake_build_ddg(ast_root, nodes, path: Path, source_code: str):
            assert builder.language == "rust"
            return []

        builder._parse_source = fake_parse
        builder._build_cfg = fake_build_cfg
        builder._build_cdg = fake_build_cdg
        builder._build_ddg = fake_build_ddg
        builder._resolve_enhanced_builder = MagicMock(return_value=None)

        nodes, edges = await builder.build_ccg_for_file(file_path, "rust")

        assert nodes == []
        assert edges == []
        assert observed == [("rust", "rust")]
        assert builder.language == "python"

    def test_parse_source_requests_parser_for_explicit_language(
        self, mock_graph_store: AsyncMock, tmp_path: Path
    ) -> None:
        """_parse_source should not fall back to the builder's default language."""
        builder = CodeContextGraphBuilder(mock_graph_store, language="python")
        file_path = tmp_path / "main.rs"
        parsed_root = object()
        parser = MagicMock()
        parser.parse.return_value = MagicMock(root_node=parsed_root)
        builder._get_tree_sitter_parser = MagicMock(return_value=parser)

        root = builder._parse_source("fn main() {}", file_path, "rust")

        assert root is parsed_root
        builder._get_tree_sitter_parser.assert_called_once_with("rust")

    def test_is_keyword_python(self, builder: CodeContextGraphBuilder) -> None:
        """Test keyword detection for Python."""
        assert builder._is_keyword("if") is True
        assert builder._is_keyword("else") is True
        assert builder._is_keyword("for") is True
        assert builder._is_keyword("while") is True
        assert builder._is_keyword("def") is True
        assert builder._is_keyword("class") is True
        assert builder._is_keyword("return") is True
        assert builder._is_keyword("None") is True
        assert builder._is_keyword("True") is True
        assert builder._is_keyword("False") is True
        assert builder._is_keyword("and") is True
        assert builder._is_keyword("or") is True
        assert builder._is_keyword("not") is True
        assert builder._is_keyword("in") is True
        assert builder._is_keyword("is") is True
        assert builder._is_keyword("lambda") is True

        # Not keywords
        assert builder._is_keyword("my_function") is False
        assert builder._is_keyword("variable_name") is False
        assert builder._is_keyword("ClassName") is False

    def test_decay_score(self, builder: CodeContextGraphBuilder) -> None:
        """Test distance decay scoring."""
        # No decay at distance 0
        assert builder._decay_score(1.0, 0, 3) == 1.0

        # Decay increases with distance
        score_1 = builder._decay_score(1.0, 1, 3)
        score_2 = builder._decay_score(1.0, 2, 3)
        score_3 = builder._decay_score(1.0, 3, 3)

        assert score_1 < 1.0
        assert score_2 < score_1
        assert score_3 < score_2

    def test_should_connect_cfg(self, builder: CodeContextGraphBuilder) -> None:
        """Test CFG connection logic."""
        from victor.storage.graph.protocol import GraphNode

        node1 = GraphNode(node_id="n1", type="stmt", name="stmt1", file="file.py", line=10)
        node2 = GraphNode(node_id="n2", type="stmt", name="stmt2", file="file.py", line=11)
        node3 = GraphNode(node_id="n3", type="stmt", name="stmt3", file="file.py", line=50)

        assert builder._should_connect_cfg(node1, node2) is True
        assert builder._should_connect_cfg(node1, node3) is False

    @pytest.mark.asyncio
    async def test_enhanced_builder_delegation(self, mock_graph_store: AsyncMock) -> None:
        """Test that enhanced builder from capability registry is used."""
        from victor.core.capability_registry import CapabilityRegistry, CapabilityStatus
        from victor.framework.vertical_protocols import CCGBuilderProtocol
        from victor.storage.graph.protocol import GraphNode, GraphEdge
        import tempfile

        # Create a mock enhanced builder
        class EnhancedBuilder:
            def __init__(self):
                self.called = False

            def supports_language(self, lang: str) -> bool:
                return lang == "python"

            async def build_ccg_for_file(self, file_path, language=None):
                self.called = True
                return [
                    GraphNode(
                        node_id="enhanced",
                        type="statement",
                        name="enhanced",
                        file=str(file_path),
                        line=1,
                    )
                ], []

        # Reset the registry to ensure clean state
        CapabilityRegistry.reset()

        # Create and register the enhanced builder
        enhanced = EnhancedBuilder()
        registry = CapabilityRegistry.get_instance()
        registry.register(CCGBuilderProtocol, enhanced, CapabilityStatus.ENHANCED)

        # Create builder - should use enhanced
        builder = CodeContextGraphBuilder(mock_graph_store, language="python")

        # Verify enhanced builder is detected
        assert builder._enhanced_builder is not None
        assert builder._enhanced_builder.supports_language("python")

        # Create a temp file and test delegation
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("def foo(): pass")
            temp_path = Path(f.name)

        try:
            nodes, edges = await builder.build_ccg_for_file(temp_path)

            # Verify the enhanced builder was called
            assert enhanced.called is True
            # Verify we got the enhanced node
            assert len(nodes) == 1
            assert nodes[0].node_id == "enhanced"
        finally:
            temp_path.unlink()
            # Clean up registry
            CapabilityRegistry.reset()

    @pytest.mark.asyncio
    async def test_enhanced_builder_resolves_per_file_language(
        self, mock_graph_store: AsyncMock, tmp_path: Path
    ) -> None:
        """A builder constructed for Python should still delegate Rust files."""
        from victor.core.capability_registry import CapabilityRegistry, CapabilityStatus
        from victor.framework.vertical_protocols import CCGBuilderProtocol
        from victor.storage.graph.protocol import GraphNode

        class RustEnhancedBuilder:
            def __init__(self):
                self.languages: list[str | None] = []

            def supports_language(self, lang: str) -> bool:
                return lang == "rust"

            async def build_ccg_for_file(self, file_path, language=None):
                self.languages.append(language)
                return [
                    GraphNode(
                        node_id="rust-enhanced",
                        type="statement",
                        name="rust-enhanced",
                        file=str(file_path),
                        line=1,
                        lang=language,
                    )
                ], []

        CapabilityRegistry.reset()
        enhanced = RustEnhancedBuilder()
        registry = CapabilityRegistry.get_instance()
        registry.register(CCGBuilderProtocol, enhanced, CapabilityStatus.ENHANCED)

        builder = CodeContextGraphBuilder(mock_graph_store, language="python")
        file_path = tmp_path / "main.rs"
        file_path.write_text("fn main() {}\n", encoding="utf-8")

        try:
            nodes, edges = await builder.build_ccg_for_file(file_path, "rust")

            assert [node.node_id for node in nodes] == ["rust-enhanced"]
            assert edges == []
            assert enhanced.languages == ["rust"]
        finally:
            CapabilityRegistry.reset()

    @pytest.mark.asyncio
    async def test_enhanced_builder_fallback_on_error(self, mock_graph_store: AsyncMock) -> None:
        """Test that builder falls back to built-in if enhanced builder fails."""
        from victor.core.capability_registry import CapabilityRegistry, CapabilityStatus
        from victor.framework.vertical_protocols import CCGBuilderProtocol
        import tempfile

        # Create a failing enhanced builder
        class FailingBuilder:
            def supports_language(self, lang: str) -> bool:
                return True

            async def build_ccg_for_file(self, file_path, language=None):
                raise RuntimeError("Enhanced builder failed!")

        # Reset and register the failing builder
        CapabilityRegistry.reset()

        failing = FailingBuilder()
        registry = CapabilityRegistry.get_instance()
        registry.register(CCGBuilderProtocol, failing, CapabilityStatus.ENHANCED)

        # Create builder
        builder = CodeContextGraphBuilder(mock_graph_store, language="python")

        # Verify enhanced builder is detected
        assert builder._enhanced_builder is not None

        # Create a temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("def foo(): pass")
            temp_path = Path(f.name)

        try:
            # Should fall back to built-in and not raise
            nodes, edges = await builder.build_ccg_for_file(temp_path)

            # Should return results from built-in (may be empty due to tree-sitter)
            assert isinstance(nodes, list)
            assert isinstance(edges, list)
        finally:
            temp_path.unlink()
            # Clean up registry
            CapabilityRegistry.reset()

    def test_supports_language_method(self, builder: CodeContextGraphBuilder) -> None:
        """Test that supports_language method works correctly."""
        # Test supported languages
        assert builder.supports_language("python") is True
        assert builder.supports_language("Python") is True
        assert builder.supports_language("PYTHON") is True
        assert builder.supports_language("javascript") is True
        assert builder.supports_language("typescript") is True
        assert builder.supports_language("go") is True
        assert builder.supports_language("rust") is True

        # Test unsupported languages
        assert builder.supports_language("ruby") is False
        assert builder.supports_language("php") is False
        assert builder.supports_language("unknown") is False


class TestVariableInfo:
    """Tests for VariableInfo dataclass."""

    def test_variable_info_creation(self) -> None:
        """Test creating VariableInfo."""
        var = VariableInfo(
            name="test_var",
            defining_node="node_123",
            scope="scope_456",
            type="str",
        )
        assert var.name == "test_var"
        assert var.defining_node == "node_123"
        assert var.scope == "scope_456"
        assert var.type == "str"
        assert var.use_sites == []

    def test_variable_info_with_use_sites(self) -> None:
        """Test VariableInfo with use sites."""
        var = VariableInfo(
            name="x",
            defining_node="def_node",
            use_sites=["use1", "use2", "use3"],
        )
        assert len(var.use_sites) == 3
        assert "use1" in var.use_sites
