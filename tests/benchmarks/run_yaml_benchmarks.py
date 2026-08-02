# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,# software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""YAML workflow benchmark suite runner.

This script runs comprehensive benchmarks for the YAML workflow system
and generates a detailed performance report.
"""

import gc
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any, Dict, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from victor.workflows.yaml_loader import YAMLWorkflowLoader

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    print("ERROR: YAML workflow loader not available")
    print("This is expected - the refactor/framework-driven-cleanup branch")
    print("has not been merged yet.")
    sys.exit(1)


class WorkflowBenchmarkSuite:
    """Comprehensive benchmark suite for YAML workflows."""

    def __init__(self) -> None:
        self.loader = YAMLWorkflowLoader(enable_cache=False)
        self.results: List[Dict[str, Any]] = []

    def generate_workflow(self, node_count: int, with_conditions: bool = False) -> str:
        """Generate a workflow YAML for testing.

        Args:
            node_count: Number of nodes
            with_conditions: Whether to include conditional branching

        Returns:
            YAML workflow definition
        """
        nodes_yaml = []

        for i in range(node_count):
            is_last = i == node_count - 1
            is_middle = (i == node_count // 2) and with_conditions

            if is_middle:
                # Add conditional node in the middle
                node_yaml = f"""
      - id: node{i}
        type: condition
        condition: "score > 0.5"
        branches:
          true: node{i + 1}
          false: node{i + 1}
"""
            else:
                node_yaml = f"""
      - id: node{i}
        type: agent
        role: worker
        goal: "Process step {i}"
        tool_budget: 10
"""

            if not is_last:
                node_yaml += f"        next: [node{i + 1}]"

            nodes_yaml.append(node_yaml)

        return f"""
workflows:
  test_workflow:
    description: "Benchmark workflow with {node_count} nodes"
    metadata:
      version: "1.0"
      benchmark: true
    nodes:
{''.join(nodes_yaml)}
"""

    def benchmark_compilation(self, yaml_content: str, name: str) -> Dict[str, Any]:
        """Benchmark workflow compilation.

        Args:
            yaml_content: YAML workflow definition
            name: Test name

        Returns:
            Benchmark results dictionary
        """
        try:
            workflow_def = self.loader.load(yaml_content)

            gc.collect()
            elapsed_runs = []
            for _ in range(7):
                start_time = time.perf_counter()
                workflow_def = self.loader.load(yaml_content)
                elapsed_runs.append((time.perf_counter() - start_time) * 1000)

            gc.collect()
            tracemalloc.start()
            workflow_def = self.loader.load(yaml_content)
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            node_count = len(workflow_def.workflows) if hasattr(workflow_def, "workflows") else 1

            return {
                "name": name,
                "success": True,
                "time_ms": min(elapsed_runs),
                "memory_peak_mb": peak / 1024 / 1024,
                "memory_current_mb": current / 1024 / 1024,
                "node_count": node_count,
                "error": None,
            }
        except Exception as e:
            if tracemalloc.is_tracing():
                tracemalloc.stop()
            return {
                "name": name,
                "success": False,
                "time_ms": 0,
                "memory_peak_mb": 0,
                "memory_current_mb": 0,
                "node_count": 0,
                "error": str(e),
            }

    def run_compilation_benchmarks(self) -> List[Dict[str, Any]]:
        """Run compilation benchmarks across workflow sizes."""
        print("\n" + "=" * 80)
        print("YAML WORKFLOW COMPILATION BENCHMARKS")
        print("=" * 80)

        test_cases = [
            ("Tiny (5 nodes)", 5, False),
            ("Small (10 nodes)", 10, False),
            ("Medium (20 nodes)", 20, False),
            ("Large (50 nodes)", 50, False),
            ("XLarge (100 nodes)", 100, False),
            ("With Conditions (20 nodes)", 20, True),
        ]

        results = []

        for name, node_count, with_conditions in test_cases:
            yaml_content = self.generate_workflow(node_count, with_conditions)
            result = self.benchmark_compilation(yaml_content, name)
            results.append(result)

            status = "✓" if result["success"] else "✗"
            print(f"\n{status} {name}")
            print(f"  Nodes: {result['node_count']}")
            if result["success"]:
                print(f"  Time:   {result['time_ms']:6.2f}ms")
                print(f"  Memory: {result['memory_peak_mb']:5.2f}MB peak")
            else:
                print(f"  Error:  {result['error']}")

        return results

    def analyze_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze benchmark results.

        Args:
            results: List of benchmark results

        Returns:
            Analysis summary
        """
        successful = [r for r in results if r["success"]]
        failed = [r for r in results if not r["success"]]

        if not successful:
            return {
                "total": len(results),
                "successful": 0,
                "failed": len(failed),
                "errors": [r["error"] for r in failed],
            }

        times = [r["time_ms"] for r in successful]
        memories = [r["memory_peak_mb"] for r in successful]

        return {
            "total": len(results),
            "successful": len(successful),
            "failed": len(failed),
            "time_min_ms": min(times),
            "time_max_ms": max(times),
            "time_avg_ms": sum(times) / len(times),
            "memory_min_mb": min(memories),
            "memory_max_mb": max(memories),
            "memory_avg_mb": sum(memories) / len(memories),
        }

    def print_summary(self, analysis: Dict[str, Any]) -> None:
        """Print benchmark summary.

        Args:
            analysis: Analysis results
        """
        print("\n" + "=" * 80)
        print("BENCHMARK SUMMARY")
        print("=" * 80)

        print(f"\nTotal Tests:     {analysis['total']}")
        print(f"Successful:     {analysis['successful']}")
        print(f"Failed:         {analysis['failed']}")

        if analysis["successful"] > 0:
            print("\nTime Performance:")
            print(f"  Min:  {analysis['time_min_ms']:6.2f}ms")
            print(f"  Max:  {analysis['time_max_ms']:6.2f}ms")
            print(f"  Avg:  {analysis['time_avg_ms']:6.2f}ms")

            print("\nMemory Performance:")
            print(f"  Min:  {analysis['memory_min_mb']:5.2f}MB")
            print(f"  Max:  {analysis['memory_max_mb']:5.2f}MB")
            print(f"  Avg:  {analysis['memory_avg_mb']:5.2f}MB")

        print("\n" + "=" * 80)
        print("SUCCESS CRITERIA")
        print("=" * 80)

        if analysis["successful"] > 0:
            # Check success criteria
            simple_fast = analysis["time_min_ms"] < 100
            large_ok = analysis["time_max_ms"] < 2000
            memory_ok = analysis["memory_max_mb"] < 50

            print(f"\n✓ Simple workflow < 100ms:    {simple_fast}")
            print(f"✓ Large workflow < 2000ms:      {large_ok}")
            print(f"✓ Memory < 50MB:                 {memory_ok}")

            all_pass = simple_fast and large_ok and memory_ok

            if all_pass:
                print("\n" + "🎉 " * 20)
                print("ALL BENCHMARKS PASSED!")
                print("🎉 " * 20)
            else:
                print("\n⚠️  SOME BENCHMARKS DID NOT MEET CRITERIA")

        if analysis["failed"] > 0:
            print("\nErrors:")
            for error in analysis.get("errors", []):
                print(f"  - {error}")

    def run(self) -> int:
        """Run complete benchmark suite."""
        print("\n🚀 Starting YAML Workflow Benchmark Suite...")
        print(f"   YAML Workflows Available: {YAML_AVAILABLE}")

        results = self.run_compilation_benchmarks()
        analysis = self.analyze_results(results)
        self.print_summary(analysis)

        # Return exit code based on success
        if analysis["failed"] > 0:
            return 1

        # Check if criteria met
        if analysis["successful"] > 0:
            if analysis["time_max_ms"] > 2000 or analysis["memory_max_mb"] > 50:
                return 1

        return 0


def main() -> int:
    """Main entry point."""
    suite = WorkflowBenchmarkSuite()
    return suite.run()


if __name__ == "__main__":
    sys.exit(main())
