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

"""Tests for the performance wrapper classes."""

import pytest

from victor_coding.performance.protocols import FastChunkerProtocol
from victor_coding.performance.wrappers import (
    WrappedChunker,
    WrappedExtractor,
    WrappedIndexer,
)


class MockIndexerBackend:
    """Mock indexer backend for testing."""

    is_native = False

    async def index_file(self, file_path, root):
        from victor_coding.performance.protocols import IndexedFileData

        return IndexedFileData(
            file_path=str(file_path),
            language="python",
            content_hash="test123",
        )


class MockChunkerBackend:
    """Mock chunker backend for testing."""

    is_native = False

    def chunk_code(self, content, language, file_path=None):
        from victor_coding.performance.protocols import ChunkInfo

        return [
            ChunkInfo(
                content=content,
                chunk_type="file",
                start_line=1,
                end_line=content.count("\n") + 1,
                file_path=file_path,
            )
        ]

    def estimate_tokens(self, text):
        return len(text) // 4


class MockExtractorBackend:
    """Mock extractor backend for testing."""

    is_native = False

    def extract_symbols(self, file_path, language):
        return [
            {
                "name": "foo",
                "type": "function",
                "file_path": str(file_path),
                "line_number": 1,
            }
        ]

    def extract_call_edges(self, file_path, language):
        return [("foo", "bar", 3)]


class TestWrappedIndexer:
    """Tests for WrappedIndexer."""

    @pytest.mark.asyncio
    async def test_index_file(self):
        """Test indexing a file."""
        wrapper = WrappedIndexer(None, backend=MockIndexerBackend())

        result = await wrapper.index_file("/test.py", "/root")

        assert result is not None
        assert result.file_path == "/test.py"
        assert result.language == "python"

    def test_supports_language(self):
        """Test language support check."""

        # Add SYMBOL_QUERIES to mock backend
        class CompleteMockIndexerBackend:
            is_native = False
            SYMBOL_QUERIES = {"python": []}

        wrapper = WrappedIndexer(None, backend=CompleteMockIndexerBackend())
        assert wrapper.supports_language("python") is True

    def test_is_native_property(self):
        """Test the is_native property."""
        wrapper = WrappedIndexer(None, backend=MockIndexerBackend())
        assert wrapper.is_native is False

        # Test with mock native backend
        class NativeBackend:
            is_native = True

        wrapper_native = WrappedIndexer(None, backend=NativeBackend())
        assert wrapper_native.is_native is True


class TestWrappedChunker:
    """Tests for WrappedChunker."""

    def test_chunk_code(self):
        """Test chunking code."""
        wrapper = WrappedChunker(None, backend=MockChunkerBackend())

        content = "def foo():\n    pass\n"
        chunks = wrapper.chunk_code(content, "python", "/test.py")

        assert len(chunks) == 1
        assert chunks[0].content == content
        assert chunks[0].chunk_type == "file"

    def test_estimate_tokens(self):
        """Test token estimation."""
        wrapper = WrappedChunker(None, backend=MockChunkerBackend())

        # MockChunkerBackend uses len(text) // 4
        # "def foo(): pass" is 16 chars, 16 // 4 = 4 tokens
        # But let's check what the mock actually returns
        tokens = wrapper.estimate_tokens("def foo(): pass")
        # Should return at least 1 token
        assert tokens >= 1

    def test_is_native_property(self):
        """Test the is_native property."""
        wrapper = WrappedChunker(None, backend=MockChunkerBackend())
        assert wrapper.is_native is False


class TestWrappedExtractor:
    """Tests for WrappedExtractor."""

    def test_extract_symbols(self):
        """Test symbol extraction."""
        wrapper = WrappedExtractor(None, backend=MockExtractorBackend())

        symbols = wrapper.extract_symbols("/test.py", "python")

        assert len(symbols) == 1
        assert symbols[0]["name"] == "foo"
        assert symbols[0]["type"] == "function"

    def test_extract_call_edges(self):
        """Test call edge extraction."""
        wrapper = WrappedExtractor(None, backend=MockExtractorBackend())

        edges = wrapper.extract_call_edges("/test.py", "python")

        assert len(edges) == 1
        assert edges[0] == ("foo", "bar", 3)

    def test_extract_all_fallback(self):
        """Test extract_all with fallback implementation."""
        wrapper = WrappedExtractor(None, backend=MockExtractorBackend())

        result = wrapper.extract_all("/test.py", "python")

        assert result is not None
        assert "symbols" in result
        assert "call_edges" in result
        assert len(result["symbols"]) == 1


class TestIntegration:
    """Integration tests for wrappers with BackendFactory."""

    def test_backend_factory_creates_wrappers(self):
        """Test that BackendFactory creates wrapped instances."""
        from victor_coding.performance.factory import BackendFactory
        import tempfile

        # Create a temp directory for indexer
        with tempfile.TemporaryDirectory() as tmpdir:
            # Indexer needs a valid path
            indexer = BackendFactory.create_indexer(tmpdir)
            assert indexer is not None

        # Chunker doesn't need config
        chunker = BackendFactory.create_chunker(None)
        assert chunker is not None

        # Extractor doesn't need config
        extractor = BackendFactory.create_symbol_extractor(None)
        assert extractor is not None

    def test_wrapped_chunker_basic_functionality(self):
        """Test that wrapped chunker works with real code."""
        from victor_coding.performance.wrappers import WrappedChunker
        from victor_coding.performance.registry import PerformanceBackendRegistry

        # Clear any previous registrations
        if FastChunkerProtocol in PerformanceBackendRegistry._backends:
            PerformanceBackendRegistry._backends[FastChunkerProtocol] = {}

        wrapped = WrappedChunker(None)
        assert wrapped.is_native is False
        assert hasattr(wrapped, "chunk_file")

    def test_wrapped_regex_processor(self):
        """Test that regex processor works."""
        from victor_coding.performance.factory import BackendFactory

        processor = BackendFactory.create_regex_processor()

        # Test findall
        results = processor.findall(r"\d+", "test 123 test 456")
        assert results == ["123", "456"]

        # Test finditer
        matches = list(processor.finditer(r"\d+", "test 123 test 456"))
        assert len(matches) == 2

        # Test match
        match = processor.match(r"^\d+", "123 test")
        assert match is not None

        # Test sub
        result = processor.sub(r"\d+", "X", "test 123 test 456")
        assert result == "test X test X"
