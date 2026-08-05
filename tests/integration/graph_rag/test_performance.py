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

"""Performance benchmarks for Graph RAG pipeline."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import pytest

from victor.storage.graph import create_graph_store
from victor.core.graph_rag import (
    GraphIndexingPipeline,
    GraphIndexConfig,
    MultiHopRetriever,
    RetrievalConfig,
)
from victor.core.indexing.ccg_builder import CodeContextGraphBuilder
from victor.processing.graph_profiler import (
    GraphProfiler,
    profile_graph_operation,
)
from victor.processing.graph_optimizations import (
    GraphOptimizer,
    optimize_batch_size,
)


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.asyncio
async def test_indexing_performance():
    """Benchmark graph indexing performance."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create multiple test files
        for i in range(10):
            (tmpdir / f"module_{i}.py").write_text(f"""
class Class{i}:
    '''Test class {i}.'''

    def __init__(self):
        self.value = {i}

    def method_one(self, x):
        '''First method.'''
        return x + {i}

    def method_two(self, x, y):
        '''Second method.'''
        if x > 0:
            return self.method_one(x) + y
        else:
            return y

    def method_three(self, data):
        '''Third method with loop.'''
        results = []
        for item in data:
            if item > 0:
                results.append(item * 2)
        return results
""")

        # Benchmark indexing
        graph_store = create_graph_store(name="sqlite", project_path=tmpdir)
        await graph_store.initialize()
        await graph_store.delete_by_repo()

        config = GraphIndexConfig(
            root_path=tmpdir,
            enable_ccg=True,
            enable_embeddings=False,
        )

        start_time = time.time()
        pipeline = GraphIndexingPipeline(graph_store, config)
        stats = await pipeline.index_repository()
        elapsed_ms = (time.time() - start_time) * 1000

        # Verify indexing completed
        assert stats.files_processed >= 10
        assert stats.nodes_created > 0
        assert stats.edges_created > 0

        # Performance assertion: should complete in reasonable time
        # (Adjust threshold based on system performance)
        assert elapsed_ms < 30000, f"Indexing took {elapsed_ms}ms, expected < 30000ms"

        print(f"Indexed {stats.files_processed} files in {elapsed_ms:.1f}ms")
        print(f"Created {stats.nodes_created} nodes, {stats.edges_created} edges")

        await graph_store.close()


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.asyncio
async def test_multi_hop_retrieval_performance():
    """Benchmark multi-hop retrieval performance."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create a dependency chain
        (tmpdir / "chain.py").write_text("""
def level_00():
    return 0

def level_01():
    return level_00()

def level_02():
    return level_01()

def level_03():
    return level_02()

def level_04():
    return level_03()

def level_05():
    return level_04()

def level_06():
    return level_05()

def level_07():
    return level_06()

def level_08():
    return level_07()

def level_09():
    return level_08()

def level_10():
    return level_09()
""")

        # Index
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

        # Benchmark retrieval with different hop counts
        for max_hops in [1, 2, 3, 5]:
            retrieval_config = RetrievalConfig(
                seed_count=5,
                max_hops=max_hops,
                top_k=20,
            )

            retriever = MultiHopRetriever(graph_store, retrieval_config)

            start_time = time.time()
            result = await retriever.retrieve("level_00", retrieval_config)
            elapsed_ms = (time.time() - start_time) * 1000

            # Performance assertion
            assert elapsed_ms < 1000, f"{max_hops}-hop retrieval took {elapsed_ms}ms"

            print(f"{max_hops}-hop retrieval: {elapsed_ms:.1f}ms, {len(result.nodes)} nodes")

        await graph_store.close()


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.asyncio
async def test_ccg_construction_performance():
    """Benchmark CCG construction performance."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create file with complex control flow
        test_file = tmpdir / "complex.py"
        test_file.write_text("""
def complex_function(data, options):
    '''Function with complex control flow.'''
    results = []

    # Multiple nested conditions
    for item in data:
        if item > 0:
            if item % 2 == 0:
                for i in range(item):
                    if i > 5:
                        results.append(i * 2)
                    else:
                        results.append(i)
            else:
                try:
                    results.append(process_item(item))
                except ValueError:
                    results.append(0)
                except Exception as e:
                    results.append(-1)
        elif item < 0:
            match item:
                case -1:
                    results.append(1)
                case -2:
                    results.append(2)
                case _:
                    results.append(0)
        else:
            with create_context():
                results.append(default_value())

    # Final processing
    if results:
        if len(results) > 100:
            return results[:100]
        else:
            return results
    else:
        return []
""")

        # Benchmark CCG construction
        graph_store = create_graph_store(name="sqlite", project_path=tmpdir)
        await graph_store.initialize()

        builder = CodeContextGraphBuilder(graph_store, language="python")

        start_time = time.time()
        nodes, edges = await builder.build_ccg_for_file(test_file)
        elapsed_ms = (time.time() - start_time) * 1000

        # Verify CCG construction
        assert len(nodes) > 0
        assert len(edges) > 0

        # Performance assertion: should complete in reasonable time
        assert elapsed_ms < 5000, f"CCG construction took {elapsed_ms}ms"

        print(f"CCG construction: {elapsed_ms:.1f}ms, {len(nodes)} nodes, {len(edges)} edges")

        await graph_store.close()


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.asyncio
async def test_graph_profiler_integration():
    """Test graph profiler with real operations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create test code
        (tmpdir / "test.py").write_text("""
def foo():
    return bar()

def bar():
    return baz()

def baz():
    return 42
""")

        # Index
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

        # Use profiler to measure retrieval
        profiler = GraphProfiler()

        with profiler.profile_operation("test_retrieval"):
            retrieval_config = RetrievalConfig(
                seed_count=5,
                max_hops=2,
                top_k=10,
            )

            retriever = MultiHopRetriever(graph_store, retrieval_config)
            result = await retriever.retrieve("foo", retrieval_config)

        # Get metrics
        metrics = profiler.get_metrics("test_retrieval")

        assert metrics is not None
        assert metrics.call_count == 1
        assert metrics.total_time_ms > 0

        print(f"Retrieval: {metrics.avg_time_ms:.1f}ms avg")

        await graph_store.close()


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.asyncio
async def test_batch_size_optimization():
    """Test dynamic batch size optimization."""
    # Simulate different performance scenarios
    scenarios = [
        (100, 50, 10, 100),  # Fast operation, small batch -> increase
        (100, 500, 10, 50),  # Slow operation, large batch -> decrease
        (100, 100, 10, 100),  # Medium performance -> keep same
    ]

    for current_size, avg_time, node_count, expected_range in scenarios:
        optimized = optimize_batch_size(
            operation_type="test",
            current_batch_size=current_size,
            avg_time_ms=avg_time,
            node_count=node_count,
        )

        # Should move in expected direction
        if avg_time < 1:
            assert optimized >= current_size, "Should increase batch for fast operations"
        elif avg_time > 100:
            assert optimized <= current_size, "Should decrease batch for slow operations"

        print(f"Batch optimization: {current_size} -> {optimized} (time: {avg_time}ms)")


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.asyncio
async def test_graph_optimizer_analysis():
    """Test graph optimizer with profile data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create test code
        (tmpdir / "optimize_test.py").write_text("""
class MyClass:
    def method_a(self):
        return self.method_b()

    def method_b(self):
        return self.method_c()

    def method_c(self):
        return 42
""")

        # Index
        graph_store = create_graph_store(name="sqlite", project_path=tmpdir)
        await graph_store.initialize()
        await graph_store.delete_by_repo()

        config = GraphIndexConfig(
            root_path=tmpdir,
            enable_ccg=True,
            enable_embeddings=False,
        )

        pipeline = GraphIndexingPipeline(graph_store, config)
        stats = await pipeline.index_repository()

        # Create optimizer
        optimizer = GraphOptimizer()

        # Analyze and get recommendations
        hints = await optimizer.analyze_graph(
            graph_store=graph_store,
            profile_data={"nodes": stats.nodes_created, "edges": stats.edges_created},
        )

        assert hints is not None
        assert hasattr(hints, "batch_size_recommendations")

        print(f"Optimizer hints: {hints}")

        await graph_store.close()


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.asyncio
async def test_concurrent_retrieval_performance():
    """Test concurrent retrieval performance."""
    import asyncio

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create multiple files
        for i in range(5):
            (tmpdir / f"file_{i}.py").write_text(f"""
def func_{i}():
    return {i}

def caller_{i}():
    return func_{i}()
""")

        # Index
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

        retrieval_config = RetrievalConfig(
            seed_count=5,
            max_hops=2,
            top_k=10,
        )

        retriever = MultiHopRetriever(graph_store, retrieval_config)

        # Benchmark sequential vs concurrent
        queries = [f"func_{i}" for i in range(5)]

        # Sequential
        start_time = time.time()
        sequential_results = []
        for query in queries:
            result = await retriever.retrieve(query, retrieval_config)
            sequential_results.append(result)
        sequential_time = (time.time() - start_time) * 1000

        # Concurrent (using gather)
        start_time = time.time()
        concurrent_tasks = [retriever.retrieve(q, retrieval_config) for q in queries]
        concurrent_results = await asyncio.gather(*concurrent_tasks)
        concurrent_time = (time.time() - start_time) * 1000

        print(f"Sequential retrieval: {sequential_time:.1f}ms")
        print(f"Concurrent retrieval: {concurrent_time:.1f}ms")
        print(f"Speedup: {sequential_time / concurrent_time:.2f}x")

        # Concurrent should be faster or similar
        assert concurrent_time <= sequential_time * 1.2

        await graph_store.close()


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.asyncio
async def test_cache_effectiveness():
    """Test query cache effectiveness."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        (tmpdir / "cache_test.py").write_text("""
def cached_function():
    return expensive_operation()

def expensive_operation():
    return 42
""")

        # Index
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

        retrieval_config = RetrievalConfig(
            seed_count=5,
            max_hops=2,
            top_k=10,
        )

        retriever = MultiHopRetriever(graph_store, retrieval_config)

        # First retrieval (cold cache)
        start_time = time.time()
        result1 = await retriever.retrieve("cached_function", retrieval_config)
        cold_time = (time.time() - start_time) * 1000

        # Second retrieval (warm cache)
        start_time = time.time()
        result2 = await retriever.retrieve("cached_function", retrieval_config)
        warm_time = (time.time() - start_time) * 1000

        print(f"Cold retrieval: {cold_time:.1f}ms")
        print(f"Warm retrieval: {warm_time:.1f}ms")

        # Results should be identical
        assert len(result1.nodes) == len(result2.nodes)

        await graph_store.close()
