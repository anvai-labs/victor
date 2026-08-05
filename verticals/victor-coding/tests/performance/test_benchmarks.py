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

"""Performance benchmarks for native vs Python implementations.

These tests compare the performance of native (Rust) and Python implementations.
They are marked with pytest.mark.benchmark and can be run with pytest-benchmark.
"""

import time
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from victor_coding.performance import BackendFactory

# Sample code for testing
SAMPLE_PYTHON_CODE = """
import asyncio
from typing import List, Optional, Dict
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class User:
    '''Represents a user in the system.'''
    id: int
    name: str
    email: str
    is_active: bool = True


class UserService:
    '''Service for managing users.'''

    def __init__(self):
        self._users: Dict[int, User] = {}
        self._next_id = 1

    def create_user(self, name: str, email: str) -> User:
        '''Create a new user.'''
        user = User(
            id=self._next_id,
            name=name,
            email=email
        )
        self._users[user.id] = user
        self._next_id += 1
        return user

    def get_user(self, user_id: int) -> Optional[User]:
        '''Get a user by ID.'''
        return self._users.get(user_id)

    async def send_email(self, user: User, message: str) -> bool:
        '''Send an email to a user asynchronously.'''
        await asyncio.sleep(0.1)
        print(f"Sending to {user.email}: {message}")
        return True


class UserRepository(ABC):
    '''Abstract base class for user repositories.'''

    @abstractmethod
    def save(self, user: User) -> bool:
        '''Save a user to storage.'''
        pass

    @abstractmethod
    def find_by_email(self, email: str) -> Optional[User]:
        '''Find a user by email address.'''
        pass


class SQLUserRepository(UserRepository):
    '''SQL implementation of user repository.'''

    def __init__(self, connection_string: str):
        self.connection_string = connection_string

    def save(self, user: User) -> bool:
        '''Save user to database.'''
        # Simulate database save
        return True

    def find_by_email(self, email: str) -> Optional[User]:
        '''Find user by email.'''
        # Simulate database query
        return None
"""


@pytest.mark.benchmark
class TestRegexPerformance:
    """Benchmarks for regex processing."""

    def test_python_regex_findall(self, benchmark):
        """Benchmark Python regex findall."""
        import re

        pattern = r"class\s+(\w+).*?:"
        result = benchmark(re.findall, pattern, SAMPLE_PYTHON_CODE)
        assert len(result) >= 3

    def test_wrapper_regex_findall(self, benchmark):
        """Benchmark wrapped regex processor."""
        processor = BackendFactory.create_regex_processor()
        result = benchmark(processor.findall, r"class\s+(\w+).*?:", SAMPLE_PYTHON_CODE)
        assert len(result) >= 3

    def test_complex_regex(self, benchmark):
        """Benchmark complex regex pattern."""
        processor = BackendFactory.create_regex_processor()
        # Complex pattern matching imports with inline multiline flag
        pattern = r"(?m)^from\s+([a-zA-Z0-9_.]+)\s+import|^\s+import\s+([a-zA-Z0-9_.]+)"
        result = benchmark(processor.findall, pattern, SAMPLE_PYTHON_CODE)
        # Sample code has 4 imports: asyncio, typing, dataclasses, abc
        assert len(result) >= 4


@pytest.mark.benchmark
class TestChunkingPerformance:
    """Benchmarks for code chunking."""

    def test_chunking_large_file(self, benchmark):
        """Benchmark chunking a large Python file."""
        with TemporaryDirectory() as tmpdir:
            chunker = BackendFactory.create_chunker(None, prefer_native=False)

            # Create a large code sample and write to file
            large_code = SAMPLE_PYTHON_CODE * 100
            test_file = Path(tmpdir) / "large_test.py"
            test_file.write_text(large_code)

            def chunk_file():
                return chunker.chunk_file(test_file, "large_test.py")

            chunks = benchmark(chunk_file)
            assert len(chunks) > 0

    def test_token_estimation(self, benchmark):
        """Benchmark token estimation using the wrapper."""
        from victor_coding.performance.wrappers import WrappedChunker

        chunker = WrappedChunker(None)

        large_code = SAMPLE_PYTHON_CODE * 1000
        tokens = benchmark(chunker.estimate_tokens, large_code)
        assert tokens > 0


@pytest.mark.benchmark
class TestSymbolExtractionPerformance:
    """Benchmarks for symbol extraction."""

    def test_symbol_extraction(self, benchmark):
        """Benchmark symbol extraction using tree-sitter."""
        try:
            import tree_sitter_python as tsp_python
            from tree_sitter import Parser, Language
        except ImportError:
            pytest.skip("tree-sitter not available")

        # Newer tree-sitter API: wrap PyCapsule in Language, then pass to Parser
        lang = Language(tsp_python.language())
        parser = Parser(lang)

        def extract_symbols():
            tree = parser.parse(bytes(SAMPLE_PYTHON_CODE, "utf8"))
            # Just benchmark the parsing - tree is not None if successful
            return tree is not None

        result = benchmark(extract_symbols)
        assert result is True


@pytest.mark.benchmark
class TestIntegrationPerformance:
    """Integration benchmarks for complete workflows."""

    @pytest.mark.asyncio
    async def test_full_indexing_workflow(self, benchmark):
        """Benchmark complete file indexing workflow."""
        with TemporaryDirectory() as tmpdir:
            # Create multiple test files
            tmpdir_path = Path(tmpdir)

            for i in range(10):
                test_file = tmpdir_path / f"module_{i}.py"
                test_file.write_text(SAMPLE_PYTHON_CODE)

            async def index_all():
                indexer = BackendFactory.create_indexer(str(tmpdir_path))
                await indexer.index_codebase()
                # Return the indexer to verify it completed
                return indexer

            # Use benchmark with async function directly
            indexer = await benchmark(index_all)
            assert indexer is not None

    def test_chunk_and_index_workflow(self, benchmark):
        """Benchmark chunking followed by symbol extraction."""
        chunker = BackendFactory.create_chunker(None, prefer_native=False)
        extractor = BackendFactory.create_symbol_extractor(None)

        with TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.py"
            test_file.write_text(SAMPLE_PYTHON_CODE)

            def process_file():
                # Chunk the file
                chunks = chunker.chunk_file(test_file, "test.py")

                # Extract symbols
                symbols = extractor.extract_symbols(str(test_file), "python")

                return {
                    "chunk_count": len(chunks),
                    "symbol_count": len(symbols),
                }

            result = benchmark(process_file)
            assert result["chunk_count"] > 0
            # Note: symbol extraction may return 0 if the file is not indexed


@pytest.mark.benchmark
class TestMemoryUsage:
    """Benchmarks for memory efficiency."""

    def test_large_file_memory(self, benchmark):
        """Test memory usage when processing large files."""
        with TemporaryDirectory() as tmpdir:
            chunker = BackendFactory.create_chunker(None, prefer_native=False)

            # Very large code sample - write to file
            large_code = SAMPLE_PYTHON_CODE * 10000
            test_file = Path(tmpdir) / "large_test.py"
            test_file.write_text(large_code)

            def chunk_file():
                return chunker.chunk_file(test_file, "large_test.py")

            chunks = benchmark(chunk_file)
            # Should complete without running out of memory
            assert len(chunks) > 0


@pytest.mark.benchmark
class TestParallelProcessing:
    """Benchmarks for parallel processing capabilities."""

    @pytest.mark.asyncio
    async def test_parallel_file_indexing(self, benchmark):
        """Benchmark parallel file indexing."""
        with TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create 100 test files
            for i in range(100):
                test_file = tmpdir_path / f"module_{i}.py"
                test_file.write_text(SAMPLE_PYTHON_CODE)

            async def index_parallel():
                indexer = BackendFactory.create_indexer(str(tmpdir_path))
                await indexer.index_codebase()
                return indexer

            # Don't use asyncio.run() inside a benchmark when already in async context
            indexer = await benchmark(index_parallel)
            assert indexer is not None


class TestPerformanceComparison:
    """Direct comparison tests (not benchmarks)."""

    def test_native_vs_python_check(self):
        """Check which backends are available."""
        has_native = BackendFactory.has_native_regex()

        if has_native:
            print("Native regex backend is available")
        else:
            print("Using Python regex backend (native not available)")

    def test_backend_availability_report(self):
        """Generate a report of available backends."""
        report = {
            "native_indexer": BackendFactory.has_native_indexer(),
            "native_chunker": BackendFactory.has_native_chunker(),
            "native_regex": BackendFactory.has_native_regex(),
            "native_extractor": BackendFactory.has_native_extractor(),
        }

        print("\nBackend Availability Report:")
        for backend, available in report.items():
            status = "Available" if available else "Not available"
            print(f"  {backend}: {status}")

        return report


@pytest.fixture
def benchmark_data(request):
    """Fixture to provide benchmark data."""
    return {
        "small_code": SAMPLE_PYTHON_CODE,
        "medium_code": SAMPLE_PYTHON_CODE * 100,
        "large_code": SAMPLE_PYTHON_CODE * 1000,
    }


@pytest.mark.parametrize("code_size", ["small", "medium", "large"])
def test_chunking_scaling(code_size, benchmark_data):
    """Test chunking performance at different scales."""
    with TemporaryDirectory() as tmpdir:
        chunker = BackendFactory.create_chunker(None, prefer_native=False)
        code_key = f"{code_size}_code"
        code = benchmark_data[code_key]

        # Write code to file since chunker expects file paths
        test_file = Path(tmpdir) / "test.py"
        test_file.write_text(code)

        start = time.time()
        chunks = chunker.chunk_file(test_file, "test.py")
        elapsed = time.time() - start

        print(f"\n{code_size} code chunking took {elapsed:.4f}s")
        print(f"  Code size: {len(code)} chars")
        print(f"  Chunks produced: {len(chunks)}")
        print(f"  Throughput: {len(code) / elapsed:.0f} chars/sec")

        assert len(chunks) > 0
