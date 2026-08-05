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

"""Tests for native protocol definitions and data types."""

import pytest

from victor.native.protocols import (
    Symbol,
    SymbolType,
    ChunkInfo,
    CoercedValue,
    CoercedType,
    NativeAcceleratorProtocol,
    SymbolExtractorProtocol,
    ArgumentNormalizerProtocol,
    SimilarityComputerProtocol,
    TextChunkerProtocol,
)


class TestSymbol:
    """Tests for the Symbol dataclass."""

    def test_create_function_symbol(self):
        """Test creating a function symbol."""
        symbol = Symbol(
            name="my_function",
            type=SymbolType.FUNCTION,
            line=10,
            end_line=20,
            signature="def my_function(x: int) -> int",
            docstring="A sample function.",
        )
        assert symbol.name == "my_function"
        assert symbol.type == SymbolType.FUNCTION
        assert symbol.line == 10
        assert symbol.end_line == 20
        assert symbol.visibility == "public"

    def test_create_private_symbol(self):
        """Test creating a private symbol."""
        symbol = Symbol(
            name="_private",
            type=SymbolType.FUNCTION,
            line=1,
            end_line=5,
            visibility="private",
        )
        assert symbol.visibility == "private"

    def test_symbol_is_frozen(self):
        """Test that Symbol is immutable."""
        symbol = Symbol(name="test", type=SymbolType.FUNCTION, line=1, end_line=1)
        with pytest.raises(AttributeError):
            symbol.name = "changed"

    def test_symbol_hash(self):
        """Test that Symbol is hashable."""
        symbol1 = Symbol(name="func", type=SymbolType.FUNCTION, line=1, end_line=5)
        symbol2 = Symbol(name="func", type=SymbolType.FUNCTION, line=1, end_line=5)

        # Same name/type/line should have same hash
        assert hash(symbol1) == hash(symbol2)

        # Can be used in sets
        symbol_set = {symbol1, symbol2}
        assert len(symbol_set) == 1

    def test_symbol_with_decorators(self):
        """Test symbol with decorators."""
        symbol = Symbol(
            name="decorated_func",
            type=SymbolType.FUNCTION,
            line=1,
            end_line=10,
            decorators=("staticmethod", "lru_cache"),
        )
        assert len(symbol.decorators) == 2
        assert "staticmethod" in symbol.decorators

    def test_symbol_with_parent(self):
        """Test method symbol with parent class."""
        symbol = Symbol(
            name="method",
            type=SymbolType.METHOD,
            line=5,
            end_line=10,
            parent="MyClass",
        )
        assert symbol.parent == "MyClass"
        assert symbol.type == SymbolType.METHOD


class TestChunkInfo:
    """Tests for the ChunkInfo dataclass."""

    def test_create_chunk(self):
        """Test creating a chunk info."""
        chunk = ChunkInfo(
            text="Hello, world!",
            start_line=1,
            end_line=1,
            start_offset=0,
            end_offset=13,
        )
        assert chunk.text == "Hello, world!"
        assert chunk.start_line == 1
        assert chunk.end_line == 1
        assert chunk.overlap_prev == 0

    def test_chunk_with_overlap(self):
        """Test chunk with overlap."""
        chunk = ChunkInfo(
            text="overlapping text",
            start_line=5,
            end_line=10,
            start_offset=100,
            end_offset=200,
            overlap_prev=50,
        )
        assert chunk.overlap_prev == 50

    def test_chunk_with_metadata(self):
        """Test chunk with metadata."""
        chunk = ChunkInfo(
            text="text",
            start_line=1,
            end_line=1,
            start_offset=0,
            end_offset=4,
            metadata={"chunk_index": 0, "is_code": True},
        )
        assert chunk.metadata["chunk_index"] == 0
        assert chunk.metadata["is_code"] is True


class TestCoercedValue:
    """Tests for the CoercedValue dataclass."""

    def test_coerce_to_int(self):
        """Test integer coercion."""
        result = CoercedValue(
            value=42,
            original="42",
            coerced_type=CoercedType.INT,
        )
        assert result.value == 42
        assert result.coerced_type == CoercedType.INT
        assert result.confidence == 1.0

    def test_coerce_to_float(self):
        """Test float coercion."""
        result = CoercedValue(
            value=3.14,
            original="3.14",
            coerced_type=CoercedType.FLOAT,
        )
        assert result.value == 3.14
        assert result.coerced_type == CoercedType.FLOAT

    def test_coerce_to_bool(self):
        """Test boolean coercion."""
        result = CoercedValue(
            value=True,
            original="true",
            coerced_type=CoercedType.BOOL,
        )
        assert result.value is True
        assert result.coerced_type == CoercedType.BOOL

    def test_coerce_to_null(self):
        """Test null coercion."""
        result = CoercedValue(
            value=None,
            original="null",
            coerced_type=CoercedType.NULL,
        )
        assert result.value is None
        assert result.coerced_type == CoercedType.NULL


class TestSymbolType:
    """Tests for SymbolType enum."""

    def test_symbol_types_exist(self):
        """Test all expected symbol types exist."""
        assert SymbolType.FILE.value == "file"
        assert SymbolType.MODULE.value == "module"
        assert SymbolType.CLASS.value == "class"
        assert SymbolType.FUNCTION.value == "function"
        assert SymbolType.METHOD.value == "method"
        assert SymbolType.VARIABLE.value == "variable"
        assert SymbolType.CONSTANT.value == "constant"
        assert SymbolType.IMPORT.value == "import"


class TestCoercedType:
    """Tests for CoercedType enum."""

    def test_coerced_types_exist(self):
        """Test all expected coerced types exist."""
        assert CoercedType.STRING.value == "string"
        assert CoercedType.INT.value == "int"
        assert CoercedType.FLOAT.value == "float"
        assert CoercedType.BOOL.value == "bool"
        assert CoercedType.NULL.value == "null"
        assert CoercedType.LIST.value == "list"
        assert CoercedType.DICT.value == "dict"


class TestProtocolsAreRuntimeCheckable:
    """Test that protocols can be used with isinstance()."""

    def test_native_accelerator_protocol_checkable(self):
        """Test NativeAcceleratorProtocol is runtime checkable."""
        from victor.native.python.symbol_extractor import PythonSymbolExtractor

        extractor = PythonSymbolExtractor()
        assert isinstance(extractor, NativeAcceleratorProtocol)

    def test_symbol_extractor_protocol_checkable(self):
        """Test SymbolExtractorProtocol is runtime checkable."""
        from victor.native.python.symbol_extractor import PythonSymbolExtractor

        extractor = PythonSymbolExtractor()
        assert isinstance(extractor, SymbolExtractorProtocol)

    def test_argument_normalizer_protocol_checkable(self):
        """Test ArgumentNormalizerProtocol is runtime checkable."""
        from victor.native.python.arg_normalizer import PythonArgumentNormalizer

        normalizer = PythonArgumentNormalizer()
        assert isinstance(normalizer, ArgumentNormalizerProtocol)

    def test_similarity_computer_protocol_checkable(self):
        """Test SimilarityComputerProtocol is runtime checkable."""
        from victor.native.python.similarity import PythonSimilarityComputer

        computer = PythonSimilarityComputer()
        assert isinstance(computer, SimilarityComputerProtocol)

    def test_text_chunker_protocol_checkable(self):
        """Test TextChunkerProtocol is runtime checkable."""
        from victor.native.python.chunker import PythonTextChunker

        chunker = PythonTextChunker()
        assert isinstance(chunker, TextChunkerProtocol)
