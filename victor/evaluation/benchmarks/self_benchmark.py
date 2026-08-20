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

"""Self-benchmark runner for Victor.

Runs Victor against standard benchmarks (SWE-bench, HumanEval, etc.)
and produces comparison reports against published competitor results.

This module bridges the evaluation orchestrator pipeline with the
framework comparison reporting system.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from victor.evaluation.protocol import (
    BenchmarkType,
    EvaluationConfig,
    EvaluationResult,
    TaskResult,
    TaskStatus,
)
from victor.evaluation.result_correlation import CorrelationReport
from victor.evaluation.benchmarks.framework_comparison import (
    ComparisonReport,
    create_comparison_report,
    save_comparison_report_bundle,
)

logger = logging.getLogger(__name__)


@dataclass
class SelfBenchmarkConfig:
    """Configuration for the self-benchmark runner.

    Attributes:
        benchmark_types: Which benchmarks to run
        model: Model to use for evaluation
        provider: Provider to use
        max_tasks: Maximum number of tasks (subset for quick runs)
        output_dir: Directory for benchmark results
        include_published: Include published competitor results in report
        parallel: Number of parallel tasks
        timeout_per_task: Timeout per task in seconds
    """

    benchmark_types: list[BenchmarkType] = field(default_factory=lambda: [BenchmarkType.SWE_BENCH])
    model: str = "claude-3-sonnet"
    provider: str = "anthropic"
    max_tasks: int = 10
    output_dir: Path = field(default_factory=lambda: Path("./benchmark_results"))
    include_published: bool = True
    parallel: int = 1
    timeout_per_task: int = 1800


class SelfBenchmarkRunner:
    """Run Victor against standard benchmarks and compare with published results.

    Bridges the EvaluationOrchestrator (which runs tasks and produces
    CorrelationReports) with the framework comparison system (which
    produces ComparisonReports against published competitor data).

    Example:
        config = SelfBenchmarkConfig(max_tasks=5)
        runner = SelfBenchmarkRunner(config)
        report = await runner.run()
        print(report.to_markdown())
    """

    def __init__(self, config: SelfBenchmarkConfig) -> None:
        self.config = config

    async def run(self) -> ComparisonReport:
        """Run benchmark suite and produce comparison report.

        Returns:
            ComparisonReport with Victor results and published competitor data
        """
        from victor.evaluation.evaluation_orchestrator import (
            EvaluationOrchestrator,
            OrchestratorConfig,
        )

        benchmark_type = self.config.benchmark_types[0]

        orchestrator_config = OrchestratorConfig(
            max_tasks=self.config.max_tasks,
            max_parallel=self.config.parallel,
            task_timeout=self.config.timeout_per_task,
            output_dir=self.config.output_dir,
        )

        orchestrator = EvaluationOrchestrator(orchestrator_config)
        correlation_report = await orchestrator.run_evaluation()

        eval_result = self._correlation_to_eval_result(correlation_report, self.config)

        report = create_comparison_report(
            benchmark=benchmark_type,
            victor_result=eval_result,
            include_published=self.config.include_published,
        )

        self._save_report(report)
        return report

    async def run_quick(self) -> ComparisonReport:
        """Quick benchmark (5 tasks) for development iteration.

        Returns:
            ComparisonReport from a small task subset
        """
        quick_config = SelfBenchmarkConfig(
            benchmark_types=self.config.benchmark_types,
            model=self.config.model,
            provider=self.config.provider,
            max_tasks=5,
            output_dir=self.config.output_dir,
            include_published=self.config.include_published,
            parallel=self.config.parallel,
            timeout_per_task=self.config.timeout_per_task,
        )
        quick_runner = SelfBenchmarkRunner(quick_config)
        return await quick_runner.run()

    def _correlation_to_eval_result(
        self, report: CorrelationReport, config: SelfBenchmarkConfig
    ) -> EvaluationResult:
        """Convert CorrelationReport to EvaluationResult for comparison.

        Maps the SWE-bench correlation data to the generic evaluation
        result format used by the comparison pipeline.

        Args:
            report: CorrelationReport from EvaluationOrchestrator
            config: Self-benchmark configuration

        Returns:
            EvaluationResult compatible with create_comparison_report()
        """
        eval_config = EvaluationConfig(
            benchmark=config.benchmark_types[0],
            model=config.model,
            provider=config.provider,
            max_tasks=config.max_tasks,
            timeout_per_task=config.timeout_per_task,
            parallel_tasks=config.parallel,
        )

        task_results = []
        for score in report.scores:
            status = TaskStatus.PASSED if score.resolved else TaskStatus.FAILED
            if score.partial and not score.resolved:
                status = TaskStatus.FAILED

            task_results.append(
                TaskResult(
                    task_id=score.instance_id,
                    status=status,
                    tests_passed=score.tests_fixed,
                    tests_total=score.total_fail_to_pass + score.total_pass_to_pass,
                    completion_score=score.overall_score,
                )
            )

        return EvaluationResult(
            config=eval_config,
            task_results=task_results,
            start_time=report.timestamp,
            end_time=datetime.now(),
        )

    def _save_report(self, report: ComparisonReport) -> None:
        """Save comparison report to output directory."""
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        save_comparison_report_bundle(report, self.config.output_dir, primary_format="markdown")

        logger.info(f"Benchmark report saved to {self.config.output_dir}")
