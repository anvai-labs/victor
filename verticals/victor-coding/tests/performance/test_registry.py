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

"""Tests for the performance backend registry."""

from victor_coding.performance.protocols import (
    FastChunkerProtocol,
    FastRegexProcessorProtocol,
    IndexedFileData,
)
from victor_coding.performance.registry import (
    BackendConfig,
    PerformanceBackendRegistry,
)


# MockIndexer for tests that don't use FastChunkerProtocol
class MockIndexer:
    """Mock indexer for testing."""

    priority = 50

    def __init__(self, config=None):
        self.config = config

    async def index_file(self, file_path, root):
        return IndexedFileData(
            file_path=file_path,
            language="python",
            content_hash="test",
        )

    async def index_batch(self, file_paths, root):
        return []

    def supports_language(self, language):
        return True

    def get_supported_languages(self):
        return ["python"]


class MockNativeIndexer:
    """Mock native indexer for testing."""

    priority = 80
    is_native = True

    def __init__(self, config=None):
        self.config = config

    async def index_file(self, file_path, root):
        return IndexedFileData(
            file_path=file_path,
            language="python",
            content_hash="test",
        )

    async def index_batch(self, file_paths, root):
        return []

    def supports_language(self, language):
        return True

    def get_supported_languages(self):
        return ["python", "javascript"]


class TestPerformanceBackendRegistry:
    """Tests for PerformanceBackendRegistry."""

    def setup_method(self):
        """Clear registry before each test."""
        # Clear backends for the protocols we're testing
        for protocol in [
            FastChunkerProtocol,
            FastRegexProcessorProtocol,
        ]:
            if protocol in PerformanceBackendRegistry._backends:
                PerformanceBackendRegistry._backends[protocol] = {}

    def test_register_backend(self):
        """Test registering a backend."""
        PerformanceBackendRegistry.register(
            FastChunkerProtocol,
            "mock_indexer",
            MockIndexer,
            priority=50,
        )

        assert PerformanceBackendRegistry.is_registered(FastChunkerProtocol, "mock_indexer")

    def test_register_backend_with_factory(self):
        """Test registering a backend with a factory function."""

        def create_indexer(config):
            return MockIndexer(config)

        PerformanceBackendRegistry.register_factory(
            FastChunkerProtocol,
            "factory_indexer",
            create_indexer,
            priority=60,
        )

        assert PerformanceBackendRegistry.is_registered(FastChunkerProtocol, "factory_indexer")

    def test_get_backend(self):
        """Test getting a specific backend."""
        PerformanceBackendRegistry.register(
            FastChunkerProtocol,
            "mock_indexer",
            MockIndexer,
            priority=50,
        )

        backend = PerformanceBackendRegistry.get(FastChunkerProtocol, "mock_indexer")
        assert backend is not None
        assert backend == MockIndexer

    def test_get_backend_not_found(self):
        """Test getting a non-existent backend."""
        backend = PerformanceBackendRegistry.get(FastChunkerProtocol, "nonexistent", default=None)
        assert backend is None

    def test_create_uses_highest_priority(self):
        """Test that create() uses the highest priority backend."""
        PerformanceBackendRegistry.register(
            FastChunkerProtocol,
            "low_priority",
            MockIndexer,
            priority=30,
        )
        PerformanceBackendRegistry.register(
            FastChunkerProtocol,
            "high_priority",
            MockNativeIndexer,
            priority=80,
        )

        backend = PerformanceBackendRegistry.create(FastChunkerProtocol, None)
        assert backend is not None
        # Should be the high priority one
        assert type(backend).__name__ == "MockNativeIndexer"

    def test_list_backends(self):
        """Test listing backends."""
        PerformanceBackendRegistry.register(
            FastChunkerProtocol,
            "indexer1",
            MockIndexer,
            priority=50,
        )
        PerformanceBackendRegistry.register(
            FastChunkerProtocol,
            "indexer2",
            MockNativeIndexer,
            priority=80,
        )

        backends = PerformanceBackendRegistry.list_backends(FastChunkerProtocol)
        assert len(backends) == 2

        names = PerformanceBackendRegistry.list_backend_names(FastChunkerProtocol)
        assert set(names) == {"indexer1", "indexer2"}

    def test_unregister_backend(self):
        """Test unregistering a backend."""
        PerformanceBackendRegistry.register(
            FastChunkerProtocol,
            "temp_indexer",
            MockIndexer,
            priority=50,
        )

        assert PerformanceBackendRegistry.is_registered(FastChunkerProtocol, "temp_indexer")

        result = PerformanceBackendRegistry.unregister(FastChunkerProtocol, "temp_indexer")
        assert result is True

        assert not PerformanceBackendRegistry.is_registered(FastChunkerProtocol, "temp_indexer")

    def test_get_native_available(self):
        """Test checking for native backends."""
        # No native backends initially
        assert not PerformanceBackendRegistry.get_native_available(FastChunkerProtocol)

        # Register a native backend
        PerformanceBackendRegistry.register(
            FastChunkerProtocol,
            "native_indexer",
            MockNativeIndexer,
            priority=80,
        )

        # Now native should be available
        assert PerformanceBackendRegistry.get_native_available(FastChunkerProtocol)

    def test_backend_config_filtering(self):
        """Test that BackendConfig filters backends correctly."""
        PerformanceBackendRegistry.register(
            FastChunkerProtocol,
            "low_priority",
            MockIndexer,
            priority=30,
        )
        PerformanceBackendRegistry.register(
            FastChunkerProtocol,
            "high_priority",
            MockNativeIndexer,
            priority=80,
        )

        # Request with min_priority
        config = BackendConfig(min_priority=50)
        backend = PerformanceBackendRegistry.create(
            FastChunkerProtocol, None, backend_config=config
        )
        assert backend is not None
        assert type(backend).__name__ == "MockNativeIndexer"


class TestBackendFactory:
    """Tests for BackendFactory."""

    def test_create_indexer_with_fallback(self):
        """Test that factory falls back to Python when native unavailable."""
        from victor_coding.performance.factory import BackendFactory
        import tempfile

        # Should not raise even if native is unavailable
        # CodebaseIndex needs a valid path
        with tempfile.TemporaryDirectory() as tmpdir:
            indexer = BackendFactory.create_indexer(tmpdir)
            assert indexer is not None

    def test_has_native_methods(self):
        """Test the has_native_* check methods."""
        from victor_coding.performance.factory import BackendFactory

        # These should not raise and return bool
        assert isinstance(BackendFactory.has_native_indexer(), bool)
        assert isinstance(BackendFactory.has_native_chunker(), bool)
        assert isinstance(BackendFactory.has_native_regex(), bool)
        assert isinstance(BackendFactory.has_native_extractor(), bool)

    def test_create_regex_processor(self):
        """Test creating a regex processor."""
        from victor_coding.performance.factory import BackendFactory

        processor = BackendFactory.create_regex_processor()
        assert processor is not None

        # Should have findall method
        assert hasattr(processor, "findall")

        # Test basic functionality

        pattern = r"\d+"
        text = "test 123 test 456"

        # Test via our processor
        results = processor.findall(pattern, text)
        assert results == ["123", "456"]


class TestProtocols:
    """Tests for protocol definitions."""

    def test_indexed_file_data(self):
        """Test IndexedFileData dataclass."""
        data = IndexedFileData(
            file_path="/test.py",
            language="python",
            content_hash="abc123",
        )

        assert data.file_path == "/test.py"
        assert data.language == "python"
        assert data.content_hash == "abc123"
        assert data.symbols == []
        assert data.call_edges == []

    def test_chunk_info(self):
        """Test ChunkInfo dataclass."""
        from victor_coding.performance.protocols import ChunkInfo

        chunk = ChunkInfo(
            content="def foo(): pass",
            chunk_type="function",
            start_line=1,
            end_line=1,
            symbol_name="foo",
            token_count=3,
        )

        assert chunk.content == "def foo(): pass"
        assert chunk.chunk_type == "function"
        assert chunk.symbol_name == "foo"
        assert chunk.token_count == 3

    def test_backend_capabilities(self):
        """Test BackendCapabilities."""
        from victor_coding.performance.protocols import (
            get_backend_capabilities,
        )

        class MockBackend:
            supports_parallel = True
            is_native = True
            memory_overhead_mb = 100

        caps = get_backend_capabilities(MockBackend())
        assert caps.supports_parallel is True
        assert caps.is_native is True
        assert caps.memory_overhead_mb == 100

        # Test with backend that doesn't have attributes
        class MinimalBackend:
            pass

        caps = get_backend_capabilities(MinimalBackend())
        assert caps.supports_parallel is False
        assert caps.is_native is False
        assert caps.memory_overhead_mb == 0
