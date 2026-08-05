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

"""Tests for performance profiling protocol types and data structures."""

from pathlib import Path

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

# =============================================================================
# ENUM TESTS
# =============================================================================


class TestProfilerType:
    """Tests for ProfilerType enum."""

    def test_cpu_profiler(self):
        """Test CPU profiler type."""
        assert ProfilerType.CPU.value == "cpu"

    def test_memory_profiler(self):
        """Test memory profiler type."""
        assert ProfilerType.MEMORY.value == "memory"

    def test_line_profiler(self):
        """Test line profiler type."""
        assert ProfilerType.LINE.value == "line"

    def test_trace_profiler(self):
        """Test trace profiler type."""
        assert ProfilerType.TRACE.value == "trace"

    def test_async_profiler(self):
        """Test async profiler type."""
        assert ProfilerType.ASYNC.value == "async"


class TestMetricType:
    """Tests for MetricType enum."""

    def test_time_metric(self):
        """Test time metric type."""
        assert MetricType.TIME.value == "time"

    def test_calls_metric(self):
        """Test calls metric type."""
        assert MetricType.CALLS.value == "calls"

    def test_memory_metric(self):
        """Test memory metric type."""
        assert MetricType.MEMORY.value == "memory"

    def test_allocations_metric(self):
        """Test allocations metric type."""
        assert MetricType.ALLOCATIONS.value == "allocations"

    def test_cpu_percent_metric(self):
        """Test cpu_percent metric type."""
        assert MetricType.CPU_PERCENT.value == "cpu_percent"


# =============================================================================
# FUNCTION STATS TESTS
# =============================================================================


class TestFunctionStats:
    """Tests for FunctionStats dataclass."""

    def test_creation_minimal(self):
        """Test minimal function stats creation."""
        stats = FunctionStats(
            name="process_data",
            module="mymodule",
            filename="mymodule.py",
        )
        assert stats.name == "process_data"
        assert stats.module == "mymodule"
        assert stats.line_number == 0
        assert stats.total_time == 0.0
        assert stats.call_count == 0

    def test_creation_full(self):
        """Test full function stats creation."""
        stats = FunctionStats(
            name="process_data",
            module="mymodule",
            filename="mymodule.py",
            line_number=42,
            total_time=1.5,
            cumulative_time=2.0,
            call_count=100,
            recursive_calls=5,
            inline_time=0.5,
            callers=["main", "run"],
            callees=["helper", "util"],
            memory_allocated=1024,
            peak_memory=2048,
        )
        assert stats.total_time == 1.5
        assert stats.call_count == 100
        assert "main" in stats.callers
        assert "helper" in stats.callees

    def test_time_per_call_with_calls(self):
        """Test time_per_call with calls."""
        stats = FunctionStats(
            name="func",
            module="mod",
            filename="mod.py",
            total_time=10.0,
            call_count=5,
        )
        assert stats.time_per_call == 2.0

    def test_time_per_call_zero_calls(self):
        """Test time_per_call with zero calls."""
        stats = FunctionStats(
            name="func",
            module="mod",
            filename="mod.py",
            total_time=10.0,
            call_count=0,
        )
        assert stats.time_per_call == 0.0

    def test_percent_of_total_default(self):
        """Test percent_of_total default."""
        stats = FunctionStats(name="func", module="mod", filename="mod.py")
        assert stats.percent_of_total == 0.0


# =============================================================================
# LINE STATS TESTS
# =============================================================================


class TestLineStats:
    """Tests for LineStats dataclass."""

    def test_creation_minimal(self):
        """Test minimal line stats creation."""
        stats = LineStats(line_number=10, content="x = 42")
        assert stats.line_number == 10
        assert stats.content == "x = 42"
        assert stats.hits == 0
        assert stats.time == 0.0

    def test_creation_full(self):
        """Test full line stats creation."""
        stats = LineStats(
            line_number=10,
            content="result = process(data)",
            hits=1000,
            time=0.5,
            memory_increment=512,
        )
        assert stats.hits == 1000
        assert stats.time == 0.5
        assert stats.memory_increment == 512


# =============================================================================
# FILE PROFILE TESTS
# =============================================================================


class TestFileProfile:
    """Tests for FileProfile dataclass."""

    def test_creation_minimal(self):
        """Test minimal file profile creation."""
        profile = FileProfile(file_path=Path("test.py"))
        assert profile.file_path == Path("test.py")
        assert profile.line_stats == []
        assert profile.total_time == 0.0

    def test_creation_full(self):
        """Test full file profile creation."""
        line_stats = [
            LineStats(10, "x = 1", 100, 0.1),
            LineStats(11, "y = 2", 200, 0.2),
        ]
        profile = FileProfile(
            file_path=Path("test.py"),
            line_stats=line_stats,
            total_time=0.3,
            total_memory=1024,
        )
        assert len(profile.line_stats) == 2
        assert profile.total_memory == 1024


# =============================================================================
# MEMORY SNAPSHOT TESTS
# =============================================================================


class TestMemorySnapshot:
    """Tests for MemorySnapshot dataclass."""

    def test_creation_minimal(self):
        """Test minimal memory snapshot creation."""
        snapshot = MemorySnapshot(
            timestamp=1234567890.0,
            rss=100 * 1024 * 1024,  # 100 MB
            vms=200 * 1024 * 1024,
            shared=10 * 1024 * 1024,
        )
        assert snapshot.timestamp == 1234567890.0
        assert snapshot.rss == 100 * 1024 * 1024
        assert snapshot.heap_allocated == 0

    def test_creation_full(self):
        """Test full memory snapshot creation."""
        snapshot = MemorySnapshot(
            timestamp=1234567890.0,
            rss=100 * 1024 * 1024,
            vms=200 * 1024 * 1024,
            shared=10 * 1024 * 1024,
            heap_allocated=50 * 1024 * 1024,
            heap_used=40 * 1024 * 1024,
        )
        assert snapshot.heap_allocated == 50 * 1024 * 1024
        assert snapshot.heap_used == 40 * 1024 * 1024


# =============================================================================
# MEMORY ALLOCATION TESTS
# =============================================================================


class TestMemoryAllocation:
    """Tests for MemoryAllocation dataclass."""

    def test_creation_minimal(self):
        """Test minimal memory allocation creation."""
        alloc = MemoryAllocation(
            size=1024,
            traceback=["main:10", "process:25"],
        )
        assert alloc.size == 1024
        assert len(alloc.traceback) == 2
        assert alloc.count == 1

    def test_creation_full(self):
        """Test full memory allocation creation."""
        alloc = MemoryAllocation(
            size=1024,
            traceback=["main:10"],
            count=100,
            total_size=102400,
        )
        assert alloc.count == 100
        assert alloc.total_size == 102400


# =============================================================================
# HOTSPOT TESTS
# =============================================================================


class TestHotSpot:
    """Tests for HotSpot dataclass."""

    def test_creation_minimal(self):
        """Test minimal hotspot creation."""
        hotspot = HotSpot(
            function_name="slow_function",
            file_path="module.py",
            line_number=42,
            metric_type=MetricType.TIME,
            value=5.0,
            percent_of_total=50.0,
        )
        assert hotspot.function_name == "slow_function"
        assert hotspot.metric_type == MetricType.TIME
        assert hotspot.percent_of_total == 50.0
        assert hotspot.suggestion == ""

    def test_creation_with_suggestion(self):
        """Test hotspot with suggestion."""
        hotspot = HotSpot(
            function_name="slow_function",
            file_path="module.py",
            line_number=42,
            metric_type=MetricType.MEMORY,
            value=1024 * 1024,
            percent_of_total=30.0,
            suggestion="Consider caching results",
        )
        assert hotspot.suggestion == "Consider caching results"


# =============================================================================
# CALL GRAPH NODE TESTS
# =============================================================================


class TestCallGraphNode:
    """Tests for CallGraphNode dataclass."""

    def test_creation_minimal(self):
        """Test minimal call graph node creation."""
        node = CallGraphNode(
            function_name="main",
            module="app",
        )
        assert node.function_name == "main"
        assert node.module == "app"
        assert node.children == []

    def test_creation_with_children(self):
        """Test call graph node with children."""
        child1 = CallGraphNode("helper1", "utils", 0.1, 5)
        child2 = CallGraphNode("helper2", "utils", 0.2, 10)
        node = CallGraphNode(
            function_name="main",
            module="app",
            total_time=0.5,
            call_count=1,
            children=[child1, child2],
        )
        assert len(node.children) == 2

    def test_to_dict_simple(self):
        """Test to_dict with no children."""
        node = CallGraphNode(
            function_name="main",
            module="app",
            total_time=1.5,
            call_count=10,
        )
        d = node.to_dict()
        assert d["name"] == "main"
        assert d["module"] == "app"
        assert d["time"] == 1.5
        assert d["calls"] == 10
        assert d["children"] == []

    def test_to_dict_with_children(self):
        """Test to_dict with children."""
        child = CallGraphNode("helper", "utils", 0.5, 5)
        node = CallGraphNode(
            function_name="main",
            module="app",
            total_time=1.0,
            call_count=1,
            children=[child],
        )
        d = node.to_dict()
        assert len(d["children"]) == 1
        assert d["children"][0]["name"] == "helper"


# =============================================================================
# FLAME GRAPH DATA TESTS
# =============================================================================


class TestFlameGraphData:
    """Tests for FlameGraphData dataclass."""

    def test_creation(self):
        """Test flame graph data creation."""
        root = CallGraphNode("main", "app", 10.0, 1)
        fg = FlameGraphData(
            root=root,
            total_time=10.0,
            sample_count=1000,
        )
        assert fg.root.function_name == "main"
        assert fg.total_time == 10.0
        assert fg.sample_count == 1000


# =============================================================================
# PROFILE RESULT TESTS
# =============================================================================


class TestProfileResult:
    """Tests for ProfileResult dataclass."""

    def test_creation_minimal(self):
        """Test minimal profile result creation."""
        result = ProfileResult(profiler_type=ProfilerType.CPU)
        assert result.profiler_type == ProfilerType.CPU
        assert result.function_stats == []
        assert result.hotspots == []

    def test_creation_full(self):
        """Test full profile result creation."""
        func_stats = [
            FunctionStats("func1", "mod", "mod.py", 10, cumulative_time=2.0, call_count=50),
            FunctionStats("func2", "mod", "mod.py", 20, cumulative_time=1.0, call_count=100),
        ]
        result = ProfileResult(
            profiler_type=ProfilerType.CPU,
            function_stats=func_stats,
            total_time=3.0,
            start_time=1000.0,
            end_time=1003.0,
            peak_memory=1024 * 1024,
        )
        assert len(result.function_stats) == 2
        assert result.peak_memory == 1024 * 1024

    def test_duration(self):
        """Test duration property."""
        result = ProfileResult(
            profiler_type=ProfilerType.CPU,
            start_time=100.0,
            end_time=105.5,
        )
        assert result.duration == 5.5

    def test_get_top_functions_by_cumulative(self):
        """Test get_top_functions by cumulative time."""
        func_stats = [
            FunctionStats("func1", "mod", "mod.py", cumulative_time=1.0),
            FunctionStats("func2", "mod", "mod.py", cumulative_time=3.0),
            FunctionStats("func3", "mod", "mod.py", cumulative_time=2.0),
        ]
        result = ProfileResult(
            profiler_type=ProfilerType.CPU,
            function_stats=func_stats,
        )
        top = result.get_top_functions(2, by="cumulative")
        assert len(top) == 2
        assert top[0].name == "func2"
        assert top[1].name == "func3"

    def test_get_top_functions_by_total(self):
        """Test get_top_functions by total time."""
        func_stats = [
            FunctionStats("func1", "mod", "mod.py", total_time=5.0),
            FunctionStats("func2", "mod", "mod.py", total_time=1.0),
        ]
        result = ProfileResult(
            profiler_type=ProfilerType.CPU,
            function_stats=func_stats,
        )
        top = result.get_top_functions(1, by="total")
        assert top[0].name == "func1"

    def test_get_top_functions_by_calls(self):
        """Test get_top_functions by call count."""
        func_stats = [
            FunctionStats("func1", "mod", "mod.py", call_count=10),
            FunctionStats("func2", "mod", "mod.py", call_count=100),
        ]
        result = ProfileResult(
            profiler_type=ProfilerType.CPU,
            function_stats=func_stats,
        )
        top = result.get_top_functions(1, by="calls")
        assert top[0].name == "func2"

    def test_get_top_functions_by_memory(self):
        """Test get_top_functions by memory."""
        func_stats = [
            FunctionStats("func1", "mod", "mod.py", memory_allocated=1024),
            FunctionStats("func2", "mod", "mod.py", memory_allocated=2048),
        ]
        result = ProfileResult(
            profiler_type=ProfilerType.CPU,
            function_stats=func_stats,
        )
        top = result.get_top_functions(1, by="memory")
        assert top[0].name == "func2"

    def test_get_top_functions_invalid_sort(self):
        """Test get_top_functions with invalid sort key."""
        func_stats = [
            FunctionStats("func1", "mod", "mod.py", cumulative_time=1.0),
        ]
        result = ProfileResult(
            profiler_type=ProfilerType.CPU,
            function_stats=func_stats,
        )
        # Should fall back to cumulative
        top = result.get_top_functions(1, by="invalid")
        assert top[0].name == "func1"


# =============================================================================
# PROFILE CONFIG TESTS
# =============================================================================


class TestProfileConfig:
    """Tests for ProfileConfig dataclass."""

    def test_default_config(self):
        """Test default config values."""
        config = ProfileConfig()
        assert config.profiler_type == ProfilerType.CPU
        assert config.include_builtins is False
        assert config.include_stdlib is False
        assert config.sample_interval == 0.001
        assert config.memory_tracking is True
        assert config.line_profiling is False
        assert config.generate_flame_graph is False
        assert config.max_depth == 50

    def test_custom_config(self):
        """Test custom config."""
        config = ProfileConfig(
            profiler_type=ProfilerType.MEMORY,
            include_builtins=True,
            include_stdlib=True,
            sample_interval=0.01,
            line_profiling=True,
            generate_flame_graph=True,
            output_dir=Path("profiles"),
            max_depth=100,
        )
        assert config.profiler_type == ProfilerType.MEMORY
        assert config.include_builtins is True
        assert config.line_profiling is True
        assert config.output_dir == Path("profiles")


# =============================================================================
# BENCHMARK TESTS
# =============================================================================


class TestBenchmark:
    """Tests for Benchmark dataclass."""

    def test_creation(self):
        """Test benchmark creation."""
        bench = Benchmark(
            name="test_operation",
            iterations=1000,
            total_time=1.5,
            min_time=0.001,
            max_time=0.005,
            mean_time=0.0015,
            std_dev=0.0002,
        )
        assert bench.name == "test_operation"
        assert bench.iterations == 1000
        assert bench.memory_delta == 0

    def test_creation_with_memory(self):
        """Test benchmark with memory delta."""
        bench = Benchmark(
            name="test_operation",
            iterations=100,
            total_time=1.0,
            min_time=0.009,
            max_time=0.011,
            mean_time=0.01,
            std_dev=0.001,
            memory_delta=1024,
        )
        assert bench.memory_delta == 1024

    def test_ops_per_second(self):
        """Test ops_per_second property."""
        bench = Benchmark(
            name="test",
            iterations=1000,
            total_time=2.0,
            min_time=0.001,
            max_time=0.003,
            mean_time=0.002,
            std_dev=0.0005,
        )
        assert bench.ops_per_second == 500.0

    def test_ops_per_second_zero_time(self):
        """Test ops_per_second with zero time."""
        bench = Benchmark(
            name="test",
            iterations=1000,
            total_time=0.0,
            min_time=0.0,
            max_time=0.0,
            mean_time=0.0,
            std_dev=0.0,
        )
        assert bench.ops_per_second == 0.0


# =============================================================================
# BENCHMARK SUITE TESTS
# =============================================================================


class TestBenchmarkSuite:
    """Tests for BenchmarkSuite dataclass."""

    def test_creation_empty(self):
        """Test empty benchmark suite creation."""
        suite = BenchmarkSuite(name="performance_tests")
        assert suite.name == "performance_tests"
        assert suite.benchmarks == []
        assert suite.total_time == 0.0

    def test_creation_with_benchmarks(self):
        """Test benchmark suite with benchmarks."""
        bench1 = Benchmark("test1", 100, 1.0, 0.009, 0.011, 0.01, 0.001)
        bench2 = Benchmark("test2", 200, 2.0, 0.009, 0.011, 0.01, 0.001)
        suite = BenchmarkSuite(
            name="performance_tests",
            benchmarks=[bench1, bench2],
            total_time=3.0,
            timestamp=1234567890.0,
        )
        assert len(suite.benchmarks) == 2
        assert suite.total_time == 3.0
        assert suite.timestamp == 1234567890.0
