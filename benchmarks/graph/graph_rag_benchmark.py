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

"""Performance benchmarks for Graph RAG operations.

This module provides benchmarks for the new graph-based code intelligence
features to ensure they meet the performance targets specified in the
implementation plan.

Performance Targets:
- CCG building: <100ms per file
- Multi-hop retrieval: <500ms for 2 hops
- Graph query execution: <200ms
"""

from __future__ import annotations

import asyncio
import random
import statistics
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

import pytest

from victor.core.indexing.ccg_builder import CodeContextGraphBuilder
from victor.core.graph_rag import MultiHopRetriever, RetrievalConfig
from victor.storage.graph import create_graph_store, GraphNode, GraphEdge
from victor.storage.graph.edge_types import EdgeType


@dataclass
class BenchmarkResult:
    """Result from a benchmark run.

    Attributes:
        name: Benchmark name
        iterations: Number of iterations run
        total_time_seconds: Total execution time
        avg_time_ms: Average time per iteration in milliseconds
        min_time_ms: Minimum time across all iterations
        max_time_ms: Maximum time across all iterations
        median_time_ms: Median time across all iterations
        p95_time_ms: 95th percentile time
        target_ms: Target time in milliseconds
        passed: Whether the benchmark meets the target
    """

    name: str
    iterations: int
    total_time_seconds: float
    avg_time_ms: float
    min_time_ms: float
    max_time_ms: float
    median_time_ms: float
    p95_time_ms: float
    target_ms: float
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "name": self.name,
            "iterations": self.iterations,
            "total_time_seconds": self.total_time_seconds,
            "avg_time_ms": self.avg_time_ms,
            "min_time_ms": self.min_time_ms,
            "max_time_ms": self.max_time_ms,
            "median_time_ms": self.median_time_ms,
            "p95_time_ms": self.p95_time_ms,
            "target_ms": self.target_ms,
            "passed": self.passed,
        }


class GraphRAGBenchmark:
    """Benchmark runner for Graph RAG operations.

    This class runs performance benchmarks for:
    1. CCG building per file
    2. Multi-hop retrieval
    3. Graph query execution
    4. Subgraph caching
    """

    def __init__(self, project_path: Optional[Path] = None) -> None:
        """Initialize the benchmark runner.

        Args:
            project_path: Path to test project (creates temp if None)
        """
        self.project_path = project_path or Path(tempfile.mkdtemp())
        self.graph_store = create_graph_store("sqlite", self.project_path)

    async def setup(self) -> None:
        """Set up the test environment."""
        await self.graph_store.initialize()
        await self.graph_store.delete_by_repo()

    async def teardown(self) -> None:
        """Clean up the test environment."""
        await self.graph_store.close()

    async def benchmark_ccg_building(
        self,
        iterations: int = 10,
        file_size_lines: int = 100,
    ) -> BenchmarkResult:
        """Benchmark CCG building performance.

        Target: <100ms per file

        Args:
            iterations: Number of benchmark iterations
            file_size_lines: Size of test file in lines

        Returns:
            BenchmarkResult with metrics
        """
        # Create a test Python file
        test_code = self._generate_test_code(file_size_lines)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(test_code)
            test_file = Path(f.name)

        try:
            builder = CodeContextGraphBuilder(self.graph_store, language="python")

            times_ms = []
            for _ in range(iterations):
                start = time.perf_counter()
                nodes, edges = await builder.build_ccg_for_file(test_file)
                end = time.perf_counter()

                elapsed_ms = (end - start) * 1000
                times_ms.append(elapsed_ms)

            # Calculate statistics
            return self._calculate_result(
                name="CCG Building",
                times_ms=times_ms,
                target_ms=100.0,
            )
        finally:
            test_file.unlink(missing_ok=True)

    async def benchmark_multi_hop_retrieval(
        self,
        iterations: int = 10,
        num_nodes: int = 100,
        max_hops: int = 2,
    ) -> BenchmarkResult:
        """Benchmark multi-hop retrieval performance.

        Target: <500ms for 2 hops

        Args:
            iterations: Number of benchmark iterations
            num_nodes: Number of nodes in test graph
            max_hops: Maximum hops for retrieval

        Returns:
            BenchmarkResult with metrics
        """
        # Create test graph
        await self._create_test_graph(num_nodes)

        config = RetrievalConfig(
            seed_count=5,
            max_hops=max_hops,
            top_k=10,
        )
        retriever = MultiHopRetriever(self.graph_store, config)

        times_ms = []
        for _ in range(iterations):
            start = time.perf_counter()
            await retriever.retrieve("test query", config)
            end = time.perf_counter()

            elapsed_ms = (end - start) * 1000
            times_ms.append(elapsed_ms)

        return self._calculate_result(
            name="Multi-hop Retrieval",
            times_ms=times_ms,
            target_ms=500.0,
        )

    async def benchmark_graph_query(
        self,
        iterations: int = 10,
        num_nodes: int = 100,
    ) -> BenchmarkResult:
        """Benchmark graph query execution performance.

        Target: <200ms

        Args:
            iterations: Number of benchmark iterations
            num_nodes: Number of nodes in test graph

        Returns:
            BenchmarkResult with metrics
        """
        # Create test graph
        await self._create_test_graph(num_nodes)

        times_ms = []
        for _ in range(iterations):
            start = time.perf_counter()
            await self.graph_store.search_symbols("test query", limit=10)
            end = time.perf_counter()

            elapsed_ms = (end - start) * 1000
            times_ms.append(elapsed_ms)

        return self._calculate_result(
            name="Graph Query Execution",
            times_ms=times_ms,
            target_ms=200.0,
        )

    async def benchmark_subgraph_caching(
        self,
        iterations: int = 10,
        num_nodes: int = 100,
    ) -> BenchmarkResult:
        """Benchmark subgraph caching performance.

        Target: <50ms for cached subgraph retrieval

        Args:
            iterations: Number of benchmark iterations
            num_nodes: Number of nodes in test graph

        Returns:
            BenchmarkResult with metrics
        """
        # Create test graph with subgraphs
        await self._create_test_graph_with_subgraphs(num_nodes)

        # First retrieval (cache miss)
        start = time.perf_counter()
        await self.graph_store.get_neighbors(
            list(await self._get_all_node_ids())[0],
            max_depth=2,
        )
        end = time.perf_counter()
        _cache_miss_ms = (end - start) * 1000

        # Subsequent retrievals (cache hit)
        times_ms = []
        for _ in range(iterations):
            start = time.perf_counter()
            await self.graph_store.get_neighbors(
                list(await self._get_all_node_ids())[0],
                max_depth=2,
            )
            end = time.perf_counter()

            elapsed_ms = (end - start) * 1000
            times_ms.append(elapsed_ms)

        return self._calculate_result(
            name="Subgraph Caching",
            times_ms=times_ms,
            target_ms=50.0,
        )

    def _calculate_result(
        self,
        name: str,
        times_ms: List[float],
        target_ms: float,
    ) -> BenchmarkResult:
        """Calculate benchmark statistics from timing data.

        Args:
            name: Benchmark name
            times_ms: List of timing results in milliseconds
            target_ms: Target time in milliseconds

        Returns:
            BenchmarkResult with calculated statistics
        """
        iterations = len(times_ms)
        total_time = sum(times_ms) / 1000
        avg = statistics.mean(times_ms)
        median = statistics.median(times_ms)
        p95 = times_ms[int(len(times_ms) * 0.95)] if times_ms else avg
        passed = median < target_ms

        return BenchmarkResult(
            name=name,
            iterations=iterations,
            total_time_seconds=total_time,
            avg_time_ms=avg,
            min_time_ms=min(times_ms),
            max_time_ms=max(times_ms),
            median_time_ms=median,
            p95_time_ms=p95,
            target_ms=target_ms,
            passed=passed,
        )

    def _generate_test_code(self, lines: int) -> str:
        """Generate test Python code with various constructs.

        Args:
            lines: Number of lines to generate

        Returns:
            Generated Python code
        """
        statements = [
            "def function_{i}(x, y):",
            "    if x > 0:",
            "        return x + y",
            "    else:",
            "        return x - y",
            "",
            "class Class_{i}:",
            "    def method_{i}(self):",
            "        pass",
            "",
            "for i in range(10):",
            "    print(i)",
            "",
            "try:",
            "    result = risky_operation()",
            "except ValueError:",
            "    pass",
            "",
            "with open('file.txt') as f:",
            "    data = f.read()",
            "",
            "while condition:",
            "    process()",
            "    if done:",
            "        break",
        ]

        code = []
        line_count = 0
        while line_count < lines:
            for stmt in statements:
                code.append(stmt)
                line_count += len(stmt.split("\n"))
                if line_count >= lines:
                    break

        return "\n".join(code[:lines])

    async def _create_test_graph(self, num_nodes: int) -> None:
        """Create a test graph with the specified number of nodes.

        Args:
            num_nodes: Number of nodes to create
        """
        nodes = []
        edges = []

        for i in range(num_nodes):
            node = GraphNode(
                node_id=f"node_{i}",
                type="function" if i % 3 == 0 else "class",
                name=f"symbol_{i}",
                file=f"file_{i // 10}.py",
                line=i * 10,
            )
            nodes.append(node)

            # Create edges
            if i > 0:
                edges.append(
                    GraphEdge(
                        src=f"node_{i-1}",
                        dst=f"node_{i}",
                        type=EdgeType.CALLS if i % 2 == 0 else EdgeType.REFERENCES,
                    )
                )

        await self.graph_store.upsert_nodes(nodes)
        await self.graph_store.upsert_edges(edges)

    async def _create_test_graph_with_subgraphs(self, num_nodes: int) -> None:
        """Create a test graph with subgraphs.

        Args:
            num_nodes: Number of nodes to create
        """
        await self._create_test_graph(num_nodes)

        # Create subgraphs
        for i in range(min(5, num_nodes // 20)):
            from victor.storage.graph.protocol import Subgraph

            _subgraph = Subgraph(
                subgraph_id=f"sub_{i}",
                anchor_node_id=f"node_{i}",
                radius=2,
                edge_types=[EdgeType.CALLS, EdgeType.REFERENCES],
                node_ids=[f"node_{j}" for j in range(i, min(i + 10, num_nodes))],
                edges=[],
                node_count=10,
            )
            # Note: Subgraph caching is not yet implemented
            # await self.graph_store.cache_subgraph(subgraph)

    async def _get_all_node_ids(self) -> List[str]:
        """Get all node IDs from the graph store.

        Returns:
            List of node IDs
        """
        nodes = await self.graph_store.get_all_nodes()
        return [n.node_id for n in nodes]


# ============================================================================
# Pytest Benchmarks
# ============================================================================


@pytest.mark.benchmark(group="graph_rag")
@pytest.mark.asyncio
async def test_ccg_building_benchmark():
    """Benchmark CCG building performance."""
    benchmark = GraphRAGBenchmark()

    await benchmark.setup()

    try:
        result = await benchmark.benchmark_ccg_building(iterations=20)

        print("\n=== CCG Building Benchmark ===")
        print(f"Iterations: {result.iterations}")
        print(f"Avg time: {result.avg_time_ms:.2f}ms")
        print(f"Median time: {result.median_time_ms:.2f}ms")
        print(f"P95 time: {result.p95_time_ms:.2f}ms")
        print(f"Target: <{result.target_ms}ms")
        print(f"Passed: {result.passed}")

        assert (
            result.passed
        ), f"CCG building too slow: {result.median_time_ms:.2f}ms > {result.target_ms}ms"
    finally:
        await benchmark.teardown()


@pytest.mark.benchmark(group="graph_rag")
@pytest.mark.asyncio
async def test_multi_hop_retrieval_benchmark():
    """Benchmark multi-hop retrieval performance."""
    benchmark = GraphRAGBenchmark()

    await benchmark.setup()

    try:
        result = await benchmark.benchmark_multi_hop_retrieval(iterations=20)

        print("\n=== Multi-hop Retrieval Benchmark ===")
        print(f"Iterations: {result.iterations}")
        print(f"Avg time: {result.avg_time_ms:.2f}ms")
        print(f"Median time: {result.median_time_ms:.2f}ms")
        print(f"P95 time: {result.p95_time_ms:.2f}ms")
        print(f"Target: <{result.target_ms}ms")
        print(f"Passed: {result.passed}")

        assert (
            result.passed
        ), f"Multi-hop retrieval too slow: {result.median_time_ms:.2f}ms > {result.target_ms}ms"
    finally:
        await benchmark.teardown()


@pytest.mark.benchmark(group="graph_rag")
@pytest.mark.asyncio
async def test_graph_query_benchmark():
    """Benchmark graph query execution performance."""
    benchmark = GraphRAGBenchmark()

    await benchmark.setup()

    try:
        result = await benchmark.benchmark_graph_query(iterations=50)

        print("\n=== Graph Query Benchmark ===")
        print(f"Iterations: {result.iterations}")
        print(f"Avg time: {result.avg_time_ms:.2f}ms")
        print(f"Median time: {result.median_time_ms:.2f}ms")
        print(f"P95 time: {result.p95_time_ms:.2f}ms")
        print(f"Target: <{result.target_ms}ms")
        print(f"Passed: {result.passed}")

        assert (
            result.passed
        ), f"Graph query too slow: {result.median_time_ms:.2f}ms > {result.target_ms}ms"
    finally:
        await benchmark.teardown()


# ============================================================================
# Manual Benchmark Runner
# ============================================================================


async def run_all_benchmarks() -> None:
    """Run all benchmarks and print results."""
    benchmark = GraphRAGBenchmark()

    await benchmark.setup()

    try:
        results = [
            await benchmark.benchmark_ccg_building(iterations=20),
            await benchmark.benchmark_multi_hop_retrieval(iterations=20),
            await benchmark.benchmark_graph_query(iterations=50),
            await benchmark.benchmark_subgraph_caching(iterations=20),
        ]

        print("\n" + "=" * 60)
        print("GRAPH RAG BENCHMARK RESULTS")
        print("=" * 60)

        for result in results:
            status = "[PASS]" if result.passed else "[FAIL]"
            print(f"\n{status} {result.name}")
            print(f"  Target:   <{result.target_ms}ms")
            print(f"  Median:   {result.median_time_ms:.2f}ms")
            print(f"  P95:       {result.p95_time_ms:.2f}ms")
            print(f"  Avg:       {result.avg_time_ms:.2f}ms")

        # Summary
        passed = sum(1 for r in results if r.passed)
        total = len(results)
        print(f"\nSummary: {passed}/{total} benchmarks passed")

    finally:
        await benchmark.teardown()


if __name__ == "__main__":
    asyncio.run(run_all_benchmarks())
