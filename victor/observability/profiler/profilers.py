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

"""Profiler implementations for CPU and memory profiling.

Provides various profiling strategies for performance analysis.
"""

import cProfile
import gc
import logging
import pstats
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Optional

from victor.observability.profiler.protocol import (
    CallGraphNode,
    FileProfile,
    FunctionStats,
    HotSpot,
    LineStats,
    MemoryAllocation,
    MemorySnapshot,
    MetricType,
    ProfileConfig,
    ProfileResult,
    ProfilerType,
)

logger = logging.getLogger(__name__)


class BaseProfiler(ABC):
    """Abstract base for profilers.

    Implements Strategy pattern for different profiling strategies.
    """

    @property
    @abstractmethod
    def profiler_type(self) -> ProfilerType:
        """Get the profiler type."""
        pass

    @abstractmethod
    def start(self) -> None:
        """Start profiling."""
        pass

    @abstractmethod
    def stop(self) -> ProfileResult:
        """Stop profiling and return results."""
        pass

    @contextmanager
    def profile(self):
        """Context manager for profiling.

        Yields:
            ProfileResult after block execution
        """
        self.start()
        result = None
        try:
            yield lambda: result
        finally:
            result = self.stop()


class CPUProfiler(BaseProfiler):
    """CPU time profiler using cProfile."""

    def __init__(self, config: Optional[ProfileConfig] = None):
        """Initialize the CPU profiler.

        Args:
            config: Profiling configuration
        """
        self.config = config or ProfileConfig()
        self._profiler: Optional[cProfile.Profile] = None
        self._start_time: float = 0.0

    @property
    def profiler_type(self) -> ProfilerType:
        return ProfilerType.CPU

    def start(self) -> None:
        """Start CPU profiling."""
        self._profiler = cProfile.Profile()
        self._start_time = time.time()
        self._profiler.enable()

    def stop(self) -> ProfileResult:
        """Stop CPU profiling and return results."""
        if self._profiler is None:
            return ProfileResult(profiler_type=ProfilerType.CPU)

        self._profiler.disable()
        end_time = time.time()

        # Get stats
        stats = pstats.Stats(self._profiler)
        stats.sort_stats("cumulative")

        # Extract function statistics
        function_stats = self._extract_function_stats(stats)

        # Detect hotspots
        hotspots = self._detect_hotspots(function_stats)

        # Build call graph
        call_graph = self._build_call_graph(stats) if self.config.generate_flame_graph else None

        result = ProfileResult(
            profiler_type=ProfilerType.CPU,
            function_stats=function_stats,
            hotspots=hotspots,
            call_graph=call_graph,
            total_time=end_time - self._start_time,
            start_time=self._start_time,
            end_time=end_time,
        )

        self._profiler = None
        return result

    def _extract_function_stats(self, stats: pstats.Stats) -> list[FunctionStats]:
        """Extract function statistics from pstats."""
        function_stats = []
        total_time = 0.0

        for (filename, line_number, func_name), (
            primitive_calls,
            total_calls,
            total_time_func,
            cumulative_time,
            callers,
        ) in stats.stats.items():
            # Filter based on config
            if not self.config.include_builtins and func_name.startswith("<"):
                continue
            if not self.config.include_stdlib:
                if "lib/python" in filename or "site-packages" in filename:
                    continue

            # Get caller information
            caller_names = []
            for caller_key in callers:
                if len(caller_key) >= 3:
                    caller_names.append(caller_key[2])

            func_stat = FunctionStats(
                name=func_name,
                module=self._extract_module(filename),
                filename=filename,
                line_number=line_number,
                total_time=total_time_func,
                cumulative_time=cumulative_time,
                call_count=total_calls,
                recursive_calls=total_calls - primitive_calls,
                callers=caller_names,
            )
            function_stats.append(func_stat)
            total_time += total_time_func

        return function_stats

    def _detect_hotspots(self, function_stats: list[FunctionStats]) -> list[HotSpot]:
        """Detect performance hotspots."""
        if not function_stats:
            return []

        hotspots = []
        total_time = sum(f.total_time for f in function_stats)

        # Sort by cumulative time
        sorted_stats = sorted(
            function_stats,
            key=lambda f: f.cumulative_time,
            reverse=True,
        )

        for stat in sorted_stats[:10]:  # Top 10 hotspots
            percent = (stat.cumulative_time / total_time * 100) if total_time > 0 else 0

            suggestion = ""
            if stat.call_count > 1000:
                suggestion = "Consider caching or reducing call frequency"
            elif stat.cumulative_time > total_time * 0.2:
                suggestion = "Major time consumer - consider optimization"

            hotspots.append(
                HotSpot(
                    function_name=stat.name,
                    file_path=stat.filename,
                    line_number=stat.line_number,
                    metric_type=MetricType.TIME,
                    value=stat.cumulative_time,
                    percent_of_total=percent,
                    suggestion=suggestion,
                )
            )

        return hotspots

    def _build_call_graph(self, stats: pstats.Stats) -> Optional[CallGraphNode]:
        """Build a call graph from profiler stats."""
        # Create root node
        root = CallGraphNode(
            function_name="<root>",
            module="",
        )

        # Build nodes from stats
        nodes: dict[tuple, CallGraphNode] = {}

        for key, (_, total_calls, total_time, _, callers) in stats.stats.items():
            filename, line_number, func_name = key
            node = CallGraphNode(
                function_name=func_name,
                module=self._extract_module(filename),
                total_time=total_time,
                call_count=total_calls,
            )
            nodes[key] = node

            # If no callers, it's a root function
            if not callers:
                root.children.append(node)

        # Link callers to callees
        for key, (_, _, _, _, callers) in stats.stats.items():
            node = nodes.get(key)
            if not node:
                continue

            for caller_key in callers:
                caller_node = nodes.get(caller_key)
                if caller_node:
                    caller_node.children.append(node)

        # Calculate total time
        root.total_time = sum(c.total_time for c in root.children)

        return root

    def _extract_module(self, filename: str) -> str:
        """Extract module name from filename."""
        path = Path(filename)
        if path.suffix == ".py":
            return path.stem
        return filename

    def profile_function(
        self,
        func: Callable,
        *args,
        **kwargs,
    ) -> tuple[Any, ProfileResult]:
        """Profile a single function call.

        Args:
            func: Function to profile
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Tuple of (function result, profile result)
        """
        self.start()
        try:
            result = func(*args, **kwargs)
        finally:
            profile_result = self.stop()

        return result, profile_result


class MemoryProfiler(BaseProfiler):
    """Memory profiler for tracking allocations."""

    def __init__(self, config: Optional[ProfileConfig] = None):
        """Initialize the memory profiler.

        Args:
            config: Profiling configuration
        """
        self.config = config or ProfileConfig()
        self._snapshots: list[MemorySnapshot] = []
        self._start_time: float = 0.0
        self._tracking: bool = False
        self._tracemalloc_started: bool = False  # Track if tracemalloc was started
        self._tracemalloc_available: bool = False

        try:
            import tracemalloc  # noqa: F401 - Checking availability

            self._tracemalloc_available = True
        except ImportError:
            logger.warning("tracemalloc not available")

    def __del__(self):
        """Cleanup tracemalloc if profiler is garbage collected.

        This ensures tracemalloc is stopped even if stop() is not called explicitly,
        preventing resource leaks and 'too many files open' errors.
        """
        if self._tracemalloc_started:
            try:
                import tracemalloc

                tracemalloc.stop()
                self._tracemalloc_started = False
            except Exception:
                pass  # Ignore errors during cleanup

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures cleanup."""
        self.stop()
        return False

    @property
    def profiler_type(self) -> ProfilerType:
        return ProfilerType.MEMORY

    def start(self) -> None:
        """Start memory profiling."""
        self._start_time = time.time()
        self._snapshots = []
        self._tracking = True

        # Take initial snapshot
        self._take_snapshot()

        # Start tracemalloc if available
        if self._tracemalloc_available and not self._tracemalloc_started:
            import tracemalloc

            tracemalloc.start()
            self._tracemalloc_started = True

    def stop(self) -> ProfileResult:
        """Stop memory profiling and return results."""
        self._tracking = False

        # Take final snapshot
        self._take_snapshot()

        # Get tracemalloc data if available
        allocations = []
        peak_memory = 0

        if self._tracemalloc_available and self._tracemalloc_started:
            import tracemalloc

            try:
                snapshot = tracemalloc.take_snapshot()
                stats = snapshot.statistics("lineno")

                for stat in stats[:50]:  # Top 50 allocations
                    allocations.append(
                        MemoryAllocation(
                            size=stat.size,
                            traceback=[str(frame) for frame in stat.traceback],
                            count=stat.count,
                            total_size=stat.size,
                        )
                    )

                current, peak = tracemalloc.get_traced_memory()
                peak_memory = peak
                tracemalloc.stop()
                self._tracemalloc_started = False
            except Exception as e:
                logger.warning(f"Error stopping tracemalloc: {e}")
                self._tracemalloc_started = False

        # Detect memory hotspots
        hotspots = self._detect_memory_hotspots(allocations)

        return ProfileResult(
            profiler_type=ProfilerType.MEMORY,
            memory_snapshots=self._snapshots,
            memory_allocations=allocations,
            hotspots=hotspots,
            peak_memory=peak_memory,
            total_time=time.time() - self._start_time,
            start_time=self._start_time,
            end_time=time.time(),
        )

    def _take_snapshot(self) -> None:
        """Take a memory snapshot."""
        import resource

        # Get memory info using resource module
        usage = resource.getrusage(resource.RUSAGE_SELF)

        snapshot = MemorySnapshot(
            timestamp=time.time(),
            rss=usage.ru_maxrss * 1024,  # Convert to bytes
            vms=0,  # Not directly available
            shared=usage.ru_ixrss * 1024,
        )

        # Try to get more detailed info with psutil if available
        try:
            import psutil

            process = psutil.Process()
            mem_info = process.memory_info()
            snapshot.rss = mem_info.rss
            snapshot.vms = mem_info.vms
        except ImportError:
            pass

        self._snapshots.append(snapshot)

    def _detect_memory_hotspots(
        self,
        allocations: list[MemoryAllocation],
    ) -> list[HotSpot]:
        """Detect memory allocation hotspots."""
        hotspots = []
        total_memory = sum(a.total_size for a in allocations)

        for alloc in allocations[:10]:  # Top 10
            percent = (alloc.total_size / total_memory * 100) if total_memory > 0 else 0

            # Extract location from traceback
            location = alloc.traceback[0] if alloc.traceback else "unknown"

            hotspots.append(
                HotSpot(
                    function_name=location,
                    file_path="",
                    line_number=0,
                    metric_type=MetricType.MEMORY,
                    value=float(alloc.total_size),
                    percent_of_total=percent,
                    suggestion=f"Allocated {alloc.total_size:,} bytes in {alloc.count} calls",
                )
            )

        return hotspots

    def profile_function(
        self,
        func: Callable,
        *args,
        **kwargs,
    ) -> tuple[Any, ProfileResult]:
        """Profile a single function's memory usage.

        Args:
            func: Function to profile
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Tuple of (function result, profile result)
        """
        # Force garbage collection before
        gc.collect()

        self.start()
        try:
            result = func(*args, **kwargs)
        finally:
            # Force garbage collection after
            gc.collect()
            profile_result = self.stop()

        return result, profile_result


class LineProfiler(BaseProfiler):
    """Line-by-line profiler."""

    def __init__(self, config: Optional[ProfileConfig] = None):
        """Initialize the line profiler.

        Args:
            config: Profiling configuration
        """
        self.config = config or ProfileConfig()
        self._functions_to_profile: list[Callable] = []
        self._start_time: float = 0.0
        self._line_profiler = None

    @property
    def profiler_type(self) -> ProfilerType:
        return ProfilerType.LINE

    def add_function(self, func: Callable) -> None:
        """Add a function to profile line-by-line.

        Args:
            func: Function to profile
        """
        self._functions_to_profile.append(func)

    def start(self) -> None:
        """Start line profiling."""
        self._start_time = time.time()

        try:
            from line_profiler import LineProfiler as LP

            self._line_profiler = LP()
            for func in self._functions_to_profile:
                self._line_profiler.add_function(func)
            self._line_profiler.enable_by_count()
        except ImportError:
            logger.warning("line_profiler not available, using fallback")
            self._line_profiler = None

    def stop(self) -> ProfileResult:
        """Stop line profiling and return results."""
        file_profiles = []

        if self._line_profiler is not None:
            self._line_profiler.disable_by_count()

            # Extract line statistics
            for func, timings in self._line_profiler.get_stats().timings.items():
                filename, start_line, func_name = func

                line_stats = []
                for line_no, hits, time_ns in timings:
                    line_stats.append(
                        LineStats(
                            line_number=line_no,
                            content="",  # Would need to read source file
                            hits=hits,
                            time=time_ns / 1e9,  # Convert to seconds
                        )
                    )

                file_profiles.append(
                    FileProfile(
                        file_path=Path(filename),
                        line_stats=line_stats,
                        total_time=sum(ls.time for ls in line_stats),
                    )
                )

        return ProfileResult(
            profiler_type=ProfilerType.LINE,
            file_profiles=file_profiles,
            total_time=time.time() - self._start_time,
            start_time=self._start_time,
            end_time=time.time(),
        )


class TimingProfiler:
    """Simple timing profiler for micro-benchmarks."""

    def __init__(self):
        """Initialize the timing profiler."""
        self._times: dict[str, list[float]] = {}

    @contextmanager
    def measure(self, name: str):
        """Context manager for measuring a code block.

        Args:
            name: Name for this measurement

        Usage:
            with profiler.measure("my_operation"):
                do_something()
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            if name not in self._times:
                self._times[name] = []
            self._times[name].append(elapsed)

    def get_stats(self, name: str) -> dict[str, float]:
        """Get statistics for a named measurement.

        Args:
            name: Measurement name

        Returns:
            Statistics dictionary
        """
        times = self._times.get(name, [])
        if not times:
            return {}

        import statistics

        return {
            "count": len(times),
            "total": sum(times),
            "min": min(times),
            "max": max(times),
            "mean": statistics.mean(times),
            "std_dev": statistics.stdev(times) if len(times) > 1 else 0.0,
        }

    def get_all_stats(self) -> dict[str, dict[str, float]]:
        """Get statistics for all measurements."""
        return {name: self.get_stats(name) for name in self._times}

    def reset(self) -> None:
        """Reset all measurements."""
        self._times.clear()
