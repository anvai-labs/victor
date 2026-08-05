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

"""Performance profiler manager.

Provides high-level API for profiling operations.
"""

import json
import logging
import statistics
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Optional

from victor.observability.profiler.profilers import (
    BaseProfiler,
    CPUProfiler,
    LineProfiler,
    MemoryProfiler,
    TimingProfiler,
)
from victor.observability.profiler.protocol import (
    Benchmark,
    BenchmarkSuite,
    ProfileConfig,
    ProfileResult,
    ProfilerType,
)

logger = logging.getLogger(__name__)


class ProfilerManager:
    """High-level manager for profiling operations.

    Orchestrates different profilers and provides unified API.
    """

    def __init__(
        self,
        config: Optional[ProfileConfig] = None,
    ):
        """Initialize the manager.

        Args:
            config: Default profiling configuration
        """
        self.config = config or ProfileConfig()

        # Initialize profilers
        self._profilers: dict[ProfilerType, BaseProfiler] = {
            ProfilerType.CPU: CPUProfiler(self.config),
            ProfilerType.MEMORY: MemoryProfiler(self.config),
            ProfilerType.LINE: LineProfiler(self.config),
        }

        self._timing_profiler = TimingProfiler()

    def get_profiler(self, profiler_type: ProfilerType) -> BaseProfiler:
        """Get a specific profiler.

        Args:
            profiler_type: Type of profiler

        Returns:
            Profiler instance
        """
        return self._profilers.get(profiler_type, self._profilers[ProfilerType.CPU])

    @contextmanager
    def profile(
        self,
        profiler_type: ProfilerType = ProfilerType.CPU,
        config: Optional[ProfileConfig] = None,
    ):
        """Context manager for profiling code blocks.

        Args:
            profiler_type: Type of profiling to perform
            config: Optional custom configuration

        Yields:
            Function to get result after block completes

        Usage:
            with manager.profile(ProfilerType.CPU) as get_result:
                do_something()
            result = get_result()
        """
        profiler = self._profilers.get(profiler_type)
        if not profiler:
            profiler = CPUProfiler(config or self.config)

        profiler.start()
        result_holder = [None]
        try:
            yield lambda: result_holder[0]
        finally:
            result_holder[0] = profiler.stop()

    def profile_function(
        self,
        func: Callable,
        *args,
        profiler_type: ProfilerType = ProfilerType.CPU,
        **kwargs,
    ) -> tuple[Any, ProfileResult]:
        """Profile a function execution.

        Args:
            func: Function to profile
            *args: Positional arguments
            profiler_type: Type of profiling
            **kwargs: Keyword arguments

        Returns:
            Tuple of (function result, profile result)
        """
        profiler = self._profilers.get(profiler_type)
        if isinstance(profiler, (CPUProfiler, MemoryProfiler)):
            return profiler.profile_function(func, *args, **kwargs)

        # Fallback for other profilers
        profiler.start()
        try:
            result = func(*args, **kwargs)
        finally:
            profile_result = profiler.stop()

        return result, profile_result

    def benchmark(
        self,
        func: Callable,
        *args,
        iterations: int = 100,
        warmup: int = 10,
        name: Optional[str] = None,
        **kwargs,
    ) -> Benchmark:
        """Run a benchmark on a function.

        Args:
            func: Function to benchmark
            *args: Positional arguments
            iterations: Number of iterations
            warmup: Warmup iterations (not counted)
            name: Benchmark name
            **kwargs: Keyword arguments

        Returns:
            Benchmark result
        """
        name = name or func.__name__

        # Warmup
        for _ in range(warmup):
            func(*args, **kwargs)

        # Measure
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        return Benchmark(
            name=name,
            iterations=iterations,
            total_time=sum(times),
            min_time=min(times),
            max_time=max(times),
            mean_time=statistics.mean(times),
            std_dev=statistics.stdev(times) if len(times) > 1 else 0.0,
        )

    def compare_functions(
        self,
        functions: list[tuple[str, Callable]],
        *args,
        iterations: int = 100,
        **kwargs,
    ) -> BenchmarkSuite:
        """Compare multiple functions.

        Args:
            functions: List of (name, function) tuples
            *args: Arguments to pass
            iterations: Iterations per function
            **kwargs: Keyword arguments

        Returns:
            BenchmarkSuite with results
        """
        suite = BenchmarkSuite(
            name="function_comparison",
            timestamp=time.time(),
        )

        for name, func in functions:
            benchmark = self.benchmark(
                func,
                *args,
                iterations=iterations,
                name=name,
                **kwargs,
            )
            suite.benchmarks.append(benchmark)

        suite.total_time = sum(b.total_time for b in suite.benchmarks)
        return suite

    @contextmanager
    def measure(self, name: str):
        """Context manager for simple timing measurements.

        Args:
            name: Name for the measurement

        Usage:
            with manager.measure("my_operation"):
                do_something()
        """
        with self._timing_profiler.measure(name):
            yield

    def get_timing_stats(self, name: Optional[str] = None) -> dict:
        """Get timing statistics.

        Args:
            name: Specific measurement name (or None for all)

        Returns:
            Statistics dictionary
        """
        if name:
            return self._timing_profiler.get_stats(name)
        return self._timing_profiler.get_all_stats()

    def reset_timing_stats(self) -> None:
        """Reset timing statistics."""
        self._timing_profiler.reset()

    def format_report(
        self,
        result: ProfileResult,
        format: str = "text",
    ) -> str:
        """Format profile result as report.

        Args:
            result: Profile result
            format: Output format (text, json, markdown)

        Returns:
            Formatted report
        """
        if format == "json":
            return self._format_json(result)
        elif format == "markdown":
            return self._format_markdown(result)
        else:
            return self._format_text(result)

    def _format_text(self, result: ProfileResult) -> str:
        """Format as plain text."""
        lines = []
        lines.append("=" * 60)
        lines.append(f"PROFILE REPORT - {result.profiler_type.value.upper()}")
        lines.append("=" * 60)
        lines.append(f"Duration: {result.duration:.3f}s")

        if result.peak_memory > 0:
            lines.append(f"Peak Memory: {result.peak_memory / 1024 / 1024:.2f} MB")

        # Top functions
        if result.function_stats:
            lines.append("")
            lines.append("Top Functions by Cumulative Time:")
            lines.append("-" * 60)
            lines.append(f"{'Function':<40} {'Calls':>8} {'Time (s)':>10}")
            lines.append("-" * 60)

            for func in result.get_top_functions(15):
                name = func.name[:38] if len(func.name) > 38 else func.name
                lines.append(f"{name:<40} {func.call_count:>8} {func.cumulative_time:>10.4f}")

        # Hotspots
        if result.hotspots:
            lines.append("")
            lines.append("Performance Hotspots:")
            lines.append("-" * 60)

            for hotspot in result.hotspots[:5]:
                lines.append(f"  {hotspot.function_name}")
                lines.append(f"    {hotspot.percent_of_total:.1f}% of total time")
                if hotspot.suggestion:
                    lines.append(f"    Suggestion: {hotspot.suggestion}")

        # Memory allocations
        if result.memory_allocations:
            lines.append("")
            lines.append("Top Memory Allocations:")
            lines.append("-" * 60)

            for alloc in result.memory_allocations[:5]:
                size_kb = alloc.total_size / 1024
                lines.append(f"  {size_kb:.2f} KB ({alloc.count} allocations)")
                if alloc.traceback:
                    lines.append(f"    at: {alloc.traceback[0]}")

        lines.append("=" * 60)
        return "\n".join(lines)

    def _format_markdown(self, result: ProfileResult) -> str:
        """Format as Markdown."""
        lines = []
        lines.append(f"# Profile Report - {result.profiler_type.value.upper()}")
        lines.append("")
        lines.append(f"**Duration:** {result.duration:.3f}s")

        if result.peak_memory > 0:
            lines.append(f"**Peak Memory:** {result.peak_memory / 1024 / 1024:.2f} MB")

        lines.append("")

        # Top functions
        if result.function_stats:
            lines.append("## Top Functions")
            lines.append("")
            lines.append("| Function | Calls | Time (s) |")
            lines.append("|----------|-------|----------|")

            for func in result.get_top_functions(10):
                lines.append(f"| `{func.name}` | {func.call_count} | {func.cumulative_time:.4f} |")

            lines.append("")

        # Hotspots
        if result.hotspots:
            lines.append("## Performance Hotspots")
            lines.append("")

            for i, hotspot in enumerate(result.hotspots[:5], 1):
                lines.append(f"### {i}. {hotspot.function_name}")
                lines.append(f"- **Percentage:** {hotspot.percent_of_total:.1f}%")
                if hotspot.suggestion:
                    lines.append(f"- **Suggestion:** {hotspot.suggestion}")
                lines.append("")

        return "\n".join(lines)

    def _format_json(self, result: ProfileResult) -> str:
        """Format as JSON."""
        data = {
            "profiler_type": result.profiler_type.value,
            "duration": result.duration,
            "total_time": result.total_time,
            "peak_memory": result.peak_memory,
            "function_stats": [
                {
                    "name": f.name,
                    "module": f.module,
                    "calls": f.call_count,
                    "total_time": f.total_time,
                    "cumulative_time": f.cumulative_time,
                }
                for f in result.function_stats[:50]
            ],
            "hotspots": [
                {
                    "function": h.function_name,
                    "percent": h.percent_of_total,
                    "value": h.value,
                    "suggestion": h.suggestion,
                }
                for h in result.hotspots
            ],
        }
        return json.dumps(data, indent=2)

    def save_result(self, result: ProfileResult, path: Path) -> None:
        """Save profile result to file.

        Args:
            result: Profile result to save
            path: Output path
        """
        # Determine format from extension
        ext = path.suffix.lower()
        if ext == ".json":
            content = self.format_report(result, "json")
        elif ext == ".md":
            content = self.format_report(result, "markdown")
        else:
            content = self.format_report(result, "text")

        path.write_text(content)
        logger.info(f"Saved profile result to {path}")


# Global manager singleton
_profiler_manager: Optional[ProfilerManager] = None


def get_profiler_manager(
    config: Optional[ProfileConfig] = None,
) -> ProfilerManager:
    """Get the global profiler manager.

    Args:
        config: Profiling configuration

    Returns:
        ProfilerManager instance
    """
    global _profiler_manager
    if _profiler_manager is None:
        _profiler_manager = ProfilerManager(config=config)
    return _profiler_manager


def reset_profiler_manager() -> None:
    """Reset the global manager."""
    global _profiler_manager
    _profiler_manager = None
