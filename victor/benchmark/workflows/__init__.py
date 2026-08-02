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

"""Benchmark vertical workflows.

This package provides workflow definitions for AI coding benchmarks:
- SWE-bench evaluation (bug fixing, issue resolution)
- Code generation (HumanEval, MBPP style tasks)
- Agentic benchmarks (LiveCodeBench, BigCodeBench, AIDER polyglot)
- Pass@k evaluation (multi-attempt generation)

Uses YAML-first architecture with Python escape hatches for complex conditions
and transforms that cannot be expressed in YAML.

Example:
    provider = BenchmarkWorkflowProvider()

    # Compile and execute (recommended - uses UnifiedWorkflowCompiler with caching)
    result = await provider.run_compiled_workflow("swe_bench", {"task_id": "test-1"})

    # Stream execution with real-time progress
    async for node_id, state in provider.stream_compiled_workflow("swe_bench", context):
        print(f"Completed: {node_id}")

Available workflows (all YAML-defined):
- swe_bench: SWE-bench style issue resolution workflow
- swe_bench_lite: Simplified SWE-bench workflow
- code_generation: HumanEval/MBPP function generation workflow
- live_code_bench: Live code execution benchmark
- big_code_bench: Large-scale code understanding benchmark
- aider_polyglot: Multi-language code modification benchmark
- passk_generation: Multi-attempt pass@k evaluation
- passk_high: High-k generation (pass@10, pass@100)
- passk_refined: Iterative refinement for single best solution
"""

from typing import List, Optional, Tuple

from victor.framework.workflows import BaseYAMLWorkflowProvider


class BenchmarkWorkflowProvider(BaseYAMLWorkflowProvider):
    """Provides benchmark-specific workflows.

    Uses YAML-first architecture with Python escape hatches for complex
    conditions and transforms that cannot be expressed in YAML.

    Inherits from BaseYAMLWorkflowProvider which provides:
    - YAML workflow loading with two-level caching
    - UnifiedWorkflowCompiler integration for consistent execution
    - Checkpointing support for resumable benchmark runs
    - Auto-workflow triggers via class attributes

    Example:
        provider = BenchmarkWorkflowProvider()

        # List available workflows
        print(provider.get_workflow_names())

        # Execute with caching (recommended)
        result = await provider.run_compiled_workflow("swe_bench", {"task_id": "1"})

        # Stream with real-time progress
        async for node_id, state in provider.stream_compiled_workflow("swe_bench", {}):
            print(f"Completed: {node_id}")
    """

    # Auto-workflow triggers for benchmark tasks
    AUTO_WORKFLOW_PATTERNS = [
        # SWE-bench patterns
        (r"swe.*bench", "swe_bench"),
        (r"fix\s+issue", "swe_bench"),
        (r"resolve\s+bug", "swe_bench"),
        (r"patch\s+fix", "swe_bench"),
        # Code generation patterns
        (r"humaneval", "code_generation"),
        (r"mbpp", "code_generation"),
        (r"generate\s+function", "code_generation"),
        (r"implement\s+function", "code_generation"),
        (r"write\s+code\s+for", "code_generation"),
        # Agentic benchmark patterns
        (r"live.*code.*bench", "live_code_bench"),
        (r"big.*code.*bench", "big_code_bench"),
        (r"aider", "aider_polyglot"),
        (r"polyglot", "aider_polyglot"),
        # Pass@k patterns
        (r"pass@\d+", "passk_generation"),
        (r"pass\s*at\s*k", "passk_generation"),
        (r"multi.*attempt", "passk_generation"),
    ]

    # Task type to workflow mappings
    TASK_TYPE_MAPPINGS = {
        # SWE-bench style
        "swe_bench": "swe_bench",
        "swe-bench": "swe_bench",
        "swe_bench_lite": "swe_bench_lite",
        "bug_fix": "swe_bench",
        "issue": "swe_bench",
        "patch": "swe_bench",
        # Code generation style
        "code_generation": "code_generation",
        "humaneval": "code_generation",
        "mbpp": "code_generation",
        "function": "code_generation",
        # Agentic benchmarks
        "live_code_bench": "live_code_bench",
        "livecodebench": "live_code_bench",
        "big_code_bench": "big_code_bench",
        "bigcodebench": "big_code_bench",
        "aider": "aider_polyglot",
        "aider_polyglot": "aider_polyglot",
        "polyglot": "aider_polyglot",
        # Pass@k evaluation
        "passk": "passk_generation",
        "pass_at_k": "passk_generation",
        "passk_generation": "passk_generation",
        "passk_high": "passk_high",
        "passk_refined": "passk_refined",
    }

    def _get_escape_hatches_module(self) -> str:
        """Return the module path for benchmark escape hatches.

        Returns:
            Module path string for CONDITIONS and TRANSFORMS dictionaries
        """
        return "victor.benchmark.escape_hatches"


# Note: Handlers are now registered via BenchmarkVertical.get_handlers()
# instead of import-side-effect registration. This follows SRP by making
# registration explicit through the vertical protocol.

__all__ = [
    # YAML-first workflow provider
    "BenchmarkWorkflowProvider",
]
