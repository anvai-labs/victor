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

"""Benchmark Vertical - High-level API for AI coding evaluations.

This vertical provides a framework-aligned approach to running benchmarks,
using the same patterns as other verticals (Coding, Research, DevOps).

Usage:
    from victor import Agent
    from victor.benchmark import BenchmarkVertical

    # Create agent with benchmark vertical
    agent = await Agent.create(
        provider="anthropic",
        model="claude-sonnet-4-20250514",
        vertical=BenchmarkVertical,
    )

    # Run a benchmark task
    result = await agent.run(task_prompt)

    # Or use the high-level benchmark API
    from victor.benchmark import BenchmarkAgent

    bench_agent = await BenchmarkAgent.create(provider="anthropic")
    result = await bench_agent.execute_task(benchmark_task)
"""

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import typer

from victor_contracts import PluginContext, VictorPlugin

from victor.benchmark.assistant import BenchmarkVertical
from victor.benchmark.agent import (
    BenchmarkAgent,
    BenchmarkAgentConfig,
    ExecutionTrace,
    create_benchmark_agent,
)
from victor.benchmark.task_bridge import (
    benchmark_task_to_framework_task,
    framework_result_to_benchmark_result,
    build_benchmark_prompt,
)
from victor.benchmark.harness_integration import (
    create_agent_callback,
    HighLevelEvaluationRunner,
)
from victor.benchmark.workflows import BenchmarkWorkflowProvider
from victor.benchmark.capabilities import (
    BenchmarkCapabilityProvider,
    get_benchmark_capabilities,
    create_benchmark_capability_loader,
    get_capability_configs,
)
from victor.benchmark.prompts import (
    BenchmarkPromptContributor,
    get_task_type_hint,
)


class BenchmarkPlugin(VictorPlugin):
    """Victor Plugin for Benchmark vertical."""

    @property
    def name(self) -> str:
        return "benchmark"

    def register(self, context: PluginContext) -> None:
        """Register benchmark vertical."""
        context.register_vertical(BenchmarkVertical)

    def get_cli_app(self) -> Optional["typer.Typer"]:
        return None

    def on_activate(self) -> None:
        pass

    def on_deactivate(self) -> None:
        pass

    def health_check(self) -> bool:
        return True


plugin = BenchmarkPlugin()

__all__ = [
    # Plugin
    "BenchmarkPlugin",
    "plugin",
    # Vertical
    "BenchmarkVertical",
    # Agent API
    "BenchmarkAgent",
    "BenchmarkAgentConfig",
    "ExecutionTrace",
    "create_benchmark_agent",
    # Task bridge
    "benchmark_task_to_framework_task",
    "framework_result_to_benchmark_result",
    "build_benchmark_prompt",
    # Harness integration
    "create_agent_callback",
    "HighLevelEvaluationRunner",
    # Workflow provider
    "BenchmarkWorkflowProvider",
    # Capabilities
    "BenchmarkCapabilityProvider",
    "get_benchmark_capabilities",
    "create_benchmark_capability_loader",
    "get_capability_configs",
    # Prompts
    "BenchmarkPromptContributor",
    "get_task_type_hint",
]
