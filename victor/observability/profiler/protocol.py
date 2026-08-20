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

"""Performance profiling protocol types.

Defines data structures for performance profiling and analysis.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional


class ProfilerType(Enum):
    """Types of profilers."""

    CPU = "cpu"  # CPU time profiling
    MEMORY = "memory"  # Memory allocation profiling
    LINE = "line"  # Line-by-line profiling
    TRACE = "trace"  # Execution tracing
    ASYNC = "async"  # Async/await profiling


class MetricType(Enum):
    """Types of performance metrics."""

    TIME = "time"  # Execution time
    CALLS = "calls"  # Number of calls
    MEMORY = "memory"  # Memory usage
    ALLOCATIONS = "allocations"  # Memory allocations
    CPU_PERCENT = "cpu_percent"  # CPU percentage


@dataclass
class FunctionStats:
    """Statistics for a single function."""

    name: str
    module: str
    filename: str
    line_number: int = 0
    total_time: float = 0.0  # Total time in function
    cumulative_time: float = 0.0  # Time including subcalls
    call_count: int = 0
    recursive_calls: int = 0
    inline_time: float = 0.0  # Time excluding subcalls
    callers: list[str] = field(default_factory=list)
    callees: list[str] = field(default_factory=list)
    memory_allocated: int = 0  # Bytes
    peak_memory: int = 0  # Peak memory in bytes

    @property
    def time_per_call(self) -> float:
        """Average time per call."""
        return self.total_time / self.call_count if self.call_count > 0 else 0.0

    @property
    def percent_of_total(self) -> float:
        """Placeholder - calculated during analysis."""
        return 0.0


@dataclass
class LineStats:
    """Statistics for a single line of code."""

    line_number: int
    content: str
    hits: int = 0
    time: float = 0.0
    memory_increment: int = 0  # Bytes allocated


@dataclass
class FileProfile:
    """Profile data for a single file."""

    file_path: Path
    line_stats: list[LineStats] = field(default_factory=list)
    total_time: float = 0.0
    total_memory: int = 0


@dataclass
class MemorySnapshot:
    """A snapshot of memory usage."""

    timestamp: float
    rss: int  # Resident Set Size in bytes
    vms: int  # Virtual Memory Size
    shared: int  # Shared memory
    heap_allocated: int = 0
    heap_used: int = 0


@dataclass
class MemoryAllocation:
    """A memory allocation event."""

    size: int  # Bytes allocated
    traceback: list[str]
    count: int = 1  # Number of allocations
    total_size: int = 0


@dataclass
class HotSpot:
    """A performance hotspot."""

    function_name: str
    file_path: str
    line_number: int
    metric_type: MetricType
    value: float
    percent_of_total: float
    suggestion: str = ""


@dataclass
class CallGraphNode:
    """Node in a call graph."""

    function_name: str
    module: str
    total_time: float = 0.0
    call_count: int = 0
    children: list["CallGraphNode"] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.function_name,
            "module": self.module,
            "time": self.total_time,
            "calls": self.call_count,
            "children": [c.to_dict() for c in self.children],
        }


@dataclass
class FlameGraphData:
    """Data for flame graph visualization."""

    root: CallGraphNode
    total_time: float = 0.0
    sample_count: int = 0


@dataclass
class ProfileResult:
    """Result of a profiling session."""

    profiler_type: ProfilerType
    function_stats: list[FunctionStats] = field(default_factory=list)
    file_profiles: list[FileProfile] = field(default_factory=list)
    memory_snapshots: list[MemorySnapshot] = field(default_factory=list)
    memory_allocations: list[MemoryAllocation] = field(default_factory=list)
    hotspots: list[HotSpot] = field(default_factory=list)
    call_graph: Optional[CallGraphNode] = None
    total_time: float = 0.0
    peak_memory: int = 0
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def duration(self) -> float:
        """Profiling duration in seconds."""
        return self.end_time - self.start_time

    def get_top_functions(self, n: int = 10, by: str = "cumulative") -> list[FunctionStats]:
        """Get top N functions by metric.

        Args:
            n: Number of functions to return
            by: Metric to sort by (cumulative, total, calls, memory)

        Returns:
            Top N functions
        """
        key_map = {
            "cumulative": lambda f: f.cumulative_time,
            "total": lambda f: f.total_time,
            "calls": lambda f: f.call_count,
            "memory": lambda f: f.memory_allocated,
        }
        key = key_map.get(by, key_map["cumulative"])
        return sorted(self.function_stats, key=key, reverse=True)[:n]


@dataclass
class ProfileConfig:
    """Configuration for profiling."""

    profiler_type: ProfilerType = ProfilerType.CPU
    include_builtins: bool = False
    include_stdlib: bool = False
    sample_interval: float = 0.001  # Seconds
    memory_tracking: bool = True
    line_profiling: bool = False
    generate_flame_graph: bool = False
    output_dir: Optional[Path] = None
    max_depth: int = 50  # Maximum call stack depth


@dataclass
class Benchmark:
    """A benchmark result."""

    name: str
    iterations: int
    total_time: float
    min_time: float
    max_time: float
    mean_time: float
    std_dev: float
    memory_delta: int = 0  # Change in memory

    @property
    def ops_per_second(self) -> float:
        """Operations per second."""
        return self.iterations / self.total_time if self.total_time > 0 else 0.0


@dataclass
class BenchmarkSuite:
    """A collection of benchmarks."""

    name: str
    benchmarks: list[Benchmark] = field(default_factory=list)
    total_time: float = 0.0
    timestamp: float = 0.0
