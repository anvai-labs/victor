#!/usr/bin/env python
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

"""Performance benchmarks for SOLID refactoring.

This script compares performance between the old and new service-oriented
architectures to ensure no regression and measure improvements.

Run with:
    python benchmarks/performance/solid_refactoring_benchmark.py

Or with pytest:
    pytest benchmarks/performance/test_performance.py -v
"""

from __future__ import annotations

import asyncio
import statistics
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# Optional imports for benchmarking
try:
    import memory_profiler

    HAS_MEMORY_PROFILER = True
except ImportError:
    HAS_MEMORY_PROFILER = False

from rich.console import Console
from rich.table import Table

console = Console()


class BenchmarkResult:
    """Result of a single benchmark run."""

    def __init__(
        self,
        name: str,
        duration_ms: float,
        operations: int,
        memory_mb: Optional[float] = None,
    ):
        self.name = name
        self.duration_ms = duration_ms
        self.operations = operations
        self.memory_mb = memory_mb

    @property
    def ops_per_second(self) -> float:
        """Operations per second."""
        if self.duration_ms == 0:
            return float("inf")
        return (self.operations / self.duration_ms) * 1000

    @property
    def avg_latency_ms(self) -> float:
        """Average latency per operation in milliseconds."""
        if self.operations == 0:
            return 0
        return self.duration_ms / self.operations


class BenchmarkSuite:
    """Suite for running performance benchmarks."""

    def __init__(self, name: str):
        self.name = name
        self.results: List[BenchmarkResult] = []

    def add_result(self, result: BenchmarkResult) -> None:
        """Add a benchmark result."""
        self.results.append(result)

    def print_summary(self) -> None:
        """Print benchmark summary table."""
        table = Table(title=self.name)
        table.add_column("Benchmark", style="cyan")
        table.add_column("Ops/sec", style="green")
        table.add_column("Avg Latency (ms)", style="yellow")
        table.add_column("Duration (ms)", style="blue")
        if HAS_MEMORY_PROFILER:
            table.add_column("Memory (MB)", style="magenta")

        for result in self.results:
            row = [
                result.name,
                f"{result.ops_per_second:,.0f}",
                f"{result.avg_latency_ms:.4f}",
                f"{result.duration_ms:,.2f}",
            ]
            if HAS_MEMORY_PROFILER and result.memory_mb is not None:
                row.append(f"{result.memory_mb:.2f}")
            table.add_row(*row)

        console.print(table)

    def compare(self, other: "BenchmarkSuite") -> Dict[str, float]:
        """Compare this suite with another and return improvement percentages.

        Positive values mean this suite is better (faster, less memory).
        """
        improvements: Dict[str, float] = {}

        # Match results by name
        my_results = {r.name: r for r in self.results}
        other_results = {r.name: r for r in other.results}

        for name, my_result in my_results.items():
            if name not in other_results:
                continue

            other_result = other_results[name]

            # Compare ops per second (higher is better)
            ops_improvement = ((my_result.ops_per_second / other_result.ops_per_second) - 1) * 100
            improvements[name] = ops_improvement

        return improvements


async def run_async_benchmark(
    name: str,
    operation: Callable,
    iterations: int = 100,
    warmup_iterations: int = 10,
) -> BenchmarkResult:
    """Run an async benchmark.

    Args:
        name: Benchmark name
        operation: Async callable to benchmark
        iterations: Number of iterations
        warmup_iterations: Number of warmup iterations

    Returns:
        BenchmarkResult with timing data
    """
    # Warmup
    for _ in range(warmup_iterations):
        await operation()

    # Benchmark
    start_time = time.perf_counter()
    for _ in range(iterations):
        await operation()
    end_time = time.perf_counter()

    duration_ms = (end_time - start_time) * 1000

    return BenchmarkResult(
        name=name,
        duration_ms=duration_ms,
        operations=iterations,
    )


def run_sync_benchmark(
    name: str,
    operation: Callable,
    iterations: int = 100,
    warmup_iterations: int = 10,
) -> BenchmarkResult:
    """Run a synchronous benchmark.

    Args:
        name: Benchmark name
        operation: Callable to benchmark
        iterations: Number of iterations
        warmup_iterations: Number of warmup iterations

    Returns:
        BenchmarkResult with timing data
    """
    # Warmup
    for _ in range(warmup_iterations):
        operation()

    # Benchmark
    start_time = time.perf_counter()
    for _ in range(iterations):
        operation()
    end_time = time.perf_counter()

    duration_ms = (end_time - start_time) * 1000

    return BenchmarkResult(
        name=name,
        duration_ms=duration_ms,
        operations=iterations,
    )


async def benchmark_tool_registration() -> BenchmarkSuite:
    """Benchmark tool registration performance.

    Compares old ToolRegistry vs new strategy-based registration.
    """
    suite = BenchmarkSuite("Tool Registration Performance")

    # Old registration (direct)
    def register_old_style():
        from victor.tools.registry import ToolRegistry
        from victor.tools.base import BaseTool

        registry = ToolRegistry()

        class TestTool(BaseTool):
            @property
            def name(self):
                return "test_tool"

            @property
            def description(self):
                return "Test tool"

            @property
            def parameters(self):
                return {"type": "object"}

            async def execute(self, _exec_ctx, **kwargs):
                from victor.tools.base import ToolResult

                return ToolResult(success=True, output="test")

        for i in range(10):
            tool = TestTool()
            registry.register(tool)

    # New registration (strategy-based)
    async def register_new_style():
        from victor.tools.registry import ToolRegistry
        from victor.tools.base import BaseTool
        from victor.core.feature_flags import get_feature_flag_manager, FeatureFlag

        # Enable strategy flag
        manager = get_feature_flag_manager()
        manager.enable(FeatureFlag.USE_STRATEGY_BASED_TOOL_REGISTRATION)

        registry = ToolRegistry()

        class TestTool(BaseTool):
            @property
            def name(self):
                return "test_tool"

            @property
            def description(self):
                return "Test tool"

            @property
            def parameters(self):
                return {"type": "object"}

            async def execute(self, _exec_ctx, **kwargs):
                from victor.tools.base import ToolResult

                return ToolResult(success=True, output="test")

        for i in range(10):
            tool = TestTool()
            registry.register(tool)

    # Run benchmarks
    result_old = run_sync_benchmark(
        "Old Registration (Direct)",
        register_old_style,
        iterations=100,
    )
    suite.add_result(result_old)

    result_new = run_sync_benchmark(
        "New Registration (Strategy)",
        lambda: asyncio.run(register_new_style()),
        iterations=100,
    )
    suite.add_result(result_new)

    return suite


async def benchmark_service_creation() -> BenchmarkSuite:
    """Benchmark service creation performance.

    Compares creating services directly vs via DI container.
    """
    suite = BenchmarkSuite("Service Creation Performance")

    # Direct creation
    def create_direct():
        from victor.agent.services.context_service import (
            ContextService,
            ContextServiceConfig,
        )

        for _ in range(10):
            service = ContextService(config=ContextServiceConfig())
            assert service is not None

    # DI container creation
    def create_via_container():
        from victor.core.container import ServiceContainer
        from victor.agent.services.protocols.context_service import (
            ContextServiceProtocol,
        )
        from victor.agent.services.context_service import (
            ContextService,
            ContextServiceConfig,
        )

        container = ServiceContainer()
        container.register(
            ContextServiceProtocol,
            lambda c: ContextService(config=ContextServiceConfig()),
        )

        for _ in range(10):
            service = container.get(ContextServiceProtocol)
            assert service is not None

    # Run benchmarks
    result_direct = run_sync_benchmark(
        "Direct Creation",
        create_direct,
        iterations=100,
    )
    suite.add_result(result_direct)

    result_container = run_sync_benchmark(
        "DI Container Creation",
        create_via_container,
        iterations=100,
    )
    suite.add_result(result_container)

    return suite


async def benchmark_vertical_creation() -> BenchmarkSuite:
    """Benchmark vertical creation performance.

    Compares inheritance-based vs composition-based vertical creation.
    """
    suite = BenchmarkSuite("Vertical Creation Performance")

    # Inheritance-based
    def create_inheritance():
        from victor.core.verticals.base import VerticalBase

        for _ in range(10):

            class InheritanceVertical(VerticalBase):
                @classmethod
                def get_tools(cls):
                    return ["read", "write"]

                @classmethod
                def get_system_prompt(cls):
                    return "You are an assistant"

            vertical = InheritanceVertical()
            assert vertical is not None

    # Composition-based
    def create_composition():
        from victor.core.verticals.base import VerticalBase

        for _ in range(10):
            ComposedVertical = (
                VerticalBase.compose()
                .with_metadata("test", "Test", "1.0.0")
                .with_tools(["read", "write"])
                .with_system_prompt("You are an assistant")
                .build()
            )
            assert ComposedVertical is not None

    # Run benchmarks
    result_inheritance = run_sync_benchmark(
        "Inheritance-Based",
        create_inheritance,
        iterations=100,
    )
    suite.add_result(result_inheritance)

    result_composition = run_sync_benchmark(
        "Composition-Based",
        create_composition,
        iterations=100,
    )
    suite.add_result(result_composition)

    return suite


async def run_all_benchmarks() -> None:
    """Run all performance benchmarks."""
    console.print("\n[bold cyan]Running SOLID Refactoring Performance Benchmarks[/bold cyan]\n")

    # Tool Registration
    console.print("[yellow]Tool Registration Benchmarks[/yellow]")
    tool_suite = await benchmark_tool_registration()
    tool_suite.print_summary()

    # Service Creation
    console.print("\n[yellow]Service Creation Benchmarks[/yellow]")
    service_suite = await benchmark_service_creation()
    service_suite.print_summary()

    # Vertical Creation
    console.print("\n[yellow]Vertical Creation Benchmarks[/yellow]")
    vertical_suite = await benchmark_vertical_creation()
    vertical_suite.print_summary()

    # Print comparison
    console.print("\n[bold cyan]Performance Comparison[/bold cyan]\n")
    print_comparison(tool_suite, service_suite, vertical_suite)


def print_comparison(
    tool_suite: BenchmarkSuite,
    service_suite: BenchmarkSuite,
    vertical_suite: BenchmarkSuite,
) -> None:
    """Print performance improvement summary."""
    table = Table(title="Performance Improvements (New vs Old)")
    table.add_column("Benchmark", style="cyan")
    table.add_column("Improvement", style="green")
    table.add_column("Status", style="yellow")

    # Extract comparisons
    improvements: List[tuple[str, float, str]] = []

    # Tool registration
    if tool_suite.results and len(tool_suite.results) >= 2:
        new_ops = tool_suite.results[1].ops_per_second
        old_ops = tool_suite.results[0].ops_per_second
        improvement = ((new_ops / old_ops) - 1) * 100
        status = (
            "✅ Improved"
            if improvement > 0
            else "⚠️  Regressed" if improvement < -5 else "➡️  Neutral"
        )
        improvements.append(("Tool Registration", improvement, status))

    # Service creation
    if service_suite.results and len(service_suite.results) >= 2:
        new_ops = service_suite.results[1].ops_per_second
        old_ops = service_suite.results[0].ops_per_second
        improvement = ((new_ops / old_ops) - 1) * 100
        status = (
            "✅ Improved"
            if improvement > 0
            else "⚠️  Regressed" if improvement < -5 else "➡️  Neutral"
        )
        improvements.append(("Service Creation", improvement, status))

    # Vertical creation
    if vertical_suite.results and len(vertical_suite.results) >= 2:
        new_ops = vertical_suite.results[1].ops_per_second
        old_ops = vertical_suite.results[0].ops_per_second
        improvement = ((new_ops / old_ops) - 1) * 100
        status = (
            "✅ Improved"
            if improvement > 0
            else "⚠️  Regressed" if improvement < -5 else "➡️  Neutral"
        )
        improvements.append(("Vertical Creation", improvement, status))

    for name, improvement, status in improvements:
        color = "green" if improvement > 0 else "red" if improvement < -5 else "white"
        table.add_row(name, f"{improvement:+.1f}%", f"[{color}]{status}[/{color}]")

    console.print(table)

    # Overall verdict
    avg_improvement = statistics.mean([imp for _, imp, _ in improvements])
    console.print(f"\n[bold]Average Performance Change:[/bold] {avg_improvement:+.1f}%\n")

    if avg_improvement > 5:
        console.print("[green]✓ New architecture shows performance improvement[/green]")
    elif avg_improvement < -5:
        console.print(
            "[red]✗ New architecture shows performance regression - investigation needed[/red]"
        )
    else:
        console.print("[yellow]→ Performance is comparable (±5%)[/yellow]")


def main():
    """Main entry point for benchmarks."""
    asyncio.run(run_all_benchmarks())


if __name__ == "__main__":
    main()
