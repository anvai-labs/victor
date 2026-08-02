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

"""Performance profiling module.

This module provides performance profiling capabilities for
CPU, memory, and timing analysis.

Example usage:
    from victor.observability.profiler import get_profiler_manager, ProfilerType

    # Get manager
    manager = get_profiler_manager()

    # Profile a function
    def my_function():
        result = sum(range(100000))
        return result

    result, profile = manager.profile_function(my_function)
    print(manager.format_report(profile))

    # Profile a code block
    with manager.profile(ProfilerType.CPU) as get_result:
        data = [i**2 for i in range(10000)]
    result = get_result()

    # Benchmark a function
    benchmark = manager.benchmark(my_function, iterations=1000)
    print(f"Mean time: {benchmark.mean_time*1000:.2f}ms")

    # Compare functions
    suite = manager.compare_functions([
        ("builtin_sum", lambda: sum(range(1000))),
        ("manual_sum", lambda: reduce(lambda a,b: a+b, range(1000))),
    ])

    # Simple timing measurements
    with manager.measure("my_operation"):
        expensive_operation()
    stats = manager.get_timing_stats("my_operation")
"""

from victor.observability.profiler.protocol import (
    Benchmark,
    BenchmarkSuite,
    CallGraphNode,
    FileProfile,
    FlameGraphData,
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
from victor.observability.profiler.profilers import (
    BaseProfiler,
    CPUProfiler,
    LineProfiler,
    MemoryProfiler,
    TimingProfiler,
)
from victor.observability.profiler.manager import (
    ProfilerManager,
    get_profiler_manager,
    reset_profiler_manager,
)

__all__ = [
    # Protocol types
    "Benchmark",
    "BenchmarkSuite",
    "CallGraphNode",
    "FileProfile",
    "FlameGraphData",
    "FunctionStats",
    "HotSpot",
    "LineStats",
    "MemoryAllocation",
    "MemorySnapshot",
    "MetricType",
    "ProfileConfig",
    "ProfileResult",
    "ProfilerType",
    # Profilers
    "BaseProfiler",
    "CPUProfiler",
    "LineProfiler",
    "MemoryProfiler",
    "TimingProfiler",
    # Manager
    "ProfilerManager",
    "get_profiler_manager",
    "reset_profiler_manager",
]
