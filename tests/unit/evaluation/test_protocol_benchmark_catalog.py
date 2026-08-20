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

"""Tests for benchmark protocol catalog helpers."""

from victor.evaluation.protocol import (
    BenchmarkType,
    get_benchmark_catalog,
    get_benchmark_metadata,
    is_browser_task_benchmark,
    is_external_agentic_benchmark,
    requires_local_manifest_benchmark,
    normalize_benchmark_name,
)


class TestBenchmarkCatalog:
    """Tests for benchmark metadata catalog."""

    def test_external_agentic_benchmarks_present(self):
        """External agentic benchmarks should be visible in the shared catalog."""
        catalog = {metadata.name: metadata for metadata in get_benchmark_catalog()}

        assert catalog["clawbench"].type == BenchmarkType.CLAW_BENCH
        assert catalog["dr3-eval"].type == BenchmarkType.DR3_EVAL
        assert catalog["guide"].type == BenchmarkType.GUIDE
        assert catalog["vlaa-gui"].type == BenchmarkType.VLAA_GUI
        assert catalog["dr3-eval"].runner_status == "implemented"
        assert catalog["guide"].runner_status == "implemented"
        assert catalog["clawbench"].runner_status == "implemented"

    def test_benchmark_alias_resolution(self):
        """CLI aliases should resolve to the same benchmark metadata."""
        assert get_benchmark_metadata("human_eval").type == BenchmarkType.HUMAN_EVAL
        assert get_benchmark_metadata("claw-bench").type == BenchmarkType.CLAW_BENCH
        assert get_benchmark_metadata("dr3_eval").type == BenchmarkType.DR3_EVAL
        assert get_benchmark_metadata(BenchmarkType.GUIDE).name == "guide"

    def test_external_agentic_detection(self):
        """Perception-heavy agentic benchmarks should be flagged explicitly."""
        assert normalize_benchmark_name("VLAA_GUI") == "vlaa-gui"
        assert is_external_agentic_benchmark(BenchmarkType.CLAW_BENCH) is True
        assert is_external_agentic_benchmark("guide") is True
        assert is_external_agentic_benchmark(BenchmarkType.SWE_BENCH) is False
        assert is_browser_task_benchmark(BenchmarkType.CLAW_BENCH) is True
        assert is_browser_task_benchmark("guide") is True
        assert is_browser_task_benchmark(BenchmarkType.DR3_EVAL) is False
        assert requires_local_manifest_benchmark(BenchmarkType.CLAW_BENCH) is True
        assert requires_local_manifest_benchmark("dr3-eval") is True
        assert requires_local_manifest_benchmark(BenchmarkType.SWE_BENCH) is False
