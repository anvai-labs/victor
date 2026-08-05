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

"""Pass@k evaluator for code generation benchmarks.

Implements the Pass@k metric as described in the Codex paper:
"Evaluating Large Language Models Trained on Code"
https://arxiv.org/abs/2107.03374

Pass@k estimates the probability that at least one of k generated
samples passes all tests, using the unbiased estimator:
    pass@k = 1 - C(n-c, k) / C(n, k)

where n = total samples, c = correct samples
"""

import asyncio
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from victor.evaluation.protocol import (
    BenchmarkTask,
    EvaluationConfig,
)

logger = logging.getLogger(__name__)


def combinations(n: int, k: int) -> float:
    """Calculate n choose k (binomial coefficient).

    Uses logarithms for numerical stability with large numbers.
    """
    if k < 0 or k > n:
        return 0.0
    if k == 0 or k == n:
        return 1.0

    # Use log to avoid overflow
    result = 0.0
    for i in range(min(k, n - k)):
        result += math.log(n - i) - math.log(i + 1)
    return math.exp(result)


def pass_at_k(n: int, c: int, k: int) -> float:
    """Calculate pass@k using the unbiased estimator.

    Args:
        n: Total number of samples generated
        c: Number of correct (passing) samples
        k: The k in pass@k (e.g., k=1, k=10, k=100)

    Returns:
        Estimated pass@k probability

    Example:
        >>> pass_at_k(n=100, c=50, k=10)
        0.9999...  # Very likely to get at least 1 correct in 10 tries
    """
    if n <= 0 or k <= 0:
        return 0.0
    if c >= n:
        return 1.0
    if c <= 0:
        return 0.0
    if k > n:
        k = n

    # pass@k = 1 - C(n-c, k) / C(n, k)
    # Numerator: ways to choose k failures from (n-c) failures
    # Denominator: ways to choose k samples from n total
    numerator = combinations(n - c, k)
    denominator = combinations(n, k)

    if denominator == 0:
        return 1.0 if c > 0 else 0.0

    return 1.0 - (numerator / denominator)


@dataclass
class PassAtKResult:
    """Result of Pass@k evaluation for a single task."""

    task_id: str
    total_samples: int
    correct_samples: int
    k_values: list[int]
    pass_at_k_scores: dict[int, float]  # k -> score
    sample_results: list[bool] = field(default_factory=list)  # Per-sample pass/fail

    @property
    def pass_rate(self) -> float:
        """Simple pass rate (c/n)."""
        if self.total_samples == 0:
            return 0.0
        return self.correct_samples / self.total_samples


@dataclass
class AggregatePassAtKResult:
    """Aggregated Pass@k results across multiple tasks."""

    total_tasks: int
    k_values: list[int]
    mean_pass_at_k: dict[int, float]
    task_results: list[PassAtKResult] = field(default_factory=list)

    def get_summary(self) -> dict:
        """Get summary statistics."""
        return {
            "total_tasks": self.total_tasks,
            "k_values": self.k_values,
            "mean_pass_at_k": self.mean_pass_at_k,
            "mean_pass_rate": sum(r.pass_rate for r in self.task_results)
            / max(1, self.total_tasks),
        }


class PassAtKEvaluator:
    """Evaluator for Pass@k metric.

    Generates multiple samples for each task and evaluates
    the Pass@k metric at different k values.

    Example:
        evaluator = PassAtKEvaluator(k_values=[1, 5, 10, 100])
        result = await evaluator.evaluate_task(
            task=task,
            generate_fn=agent_generate,
            evaluate_fn=run_tests,
            n_samples=100,
        )
        print(f"Pass@1: {result.pass_at_k_scores[1]:.2%}")
    """

    def __init__(
        self,
        k_values: Optional[list[int]] = None,
        default_n_samples: int = 100,
        temperature: float = 0.8,
        concurrent_samples: int = 10,
    ):
        """Initialize the evaluator.

        Args:
            k_values: k values to compute (default: [1, 5, 10, 100])
            default_n_samples: Default number of samples per task
            temperature: Sampling temperature for diversity
            concurrent_samples: Max concurrent sample generation
        """
        self.k_values = k_values or [1, 5, 10, 100]
        self.default_n_samples = default_n_samples
        self.temperature = temperature
        self.concurrent_samples = concurrent_samples

    async def evaluate_task(
        self,
        task: BenchmarkTask,
        generate_fn: Callable[[BenchmarkTask, float], Any],  # -> code string
        evaluate_fn: Callable[[BenchmarkTask, str], Any],  # -> bool (pass/fail)
        n_samples: Optional[int] = None,
        config: Optional[EvaluationConfig] = None,
    ) -> PassAtKResult:
        """Evaluate a single task with Pass@k.

        Args:
            task: The benchmark task
            generate_fn: Async function to generate code samples
            evaluate_fn: Async function to evaluate a sample (returns bool)
            n_samples: Number of samples to generate
            config: Optional evaluation config

        Returns:
            PassAtKResult with scores for all k values
        """
        n = n_samples or self.default_n_samples
        sample_results: list[bool] = []

        # Generate samples with concurrency limit
        semaphore = asyncio.Semaphore(self.concurrent_samples)

        async def generate_and_evaluate(idx: int) -> bool:
            async with semaphore:
                try:
                    # Generate with temperature for diversity
                    sample = await generate_fn(task, self.temperature)
                    # Evaluate
                    passed = await evaluate_fn(task, sample)
                    return passed
                except Exception as e:
                    logger.warning(f"Sample {idx} failed: {e}")
                    return False

        # Run all samples
        logger.info(f"Generating {n} samples for task {task.task_id}")
        results = await asyncio.gather(*[generate_and_evaluate(i) for i in range(n)])
        sample_results = list(results)

        # Count correct samples
        correct = sum(1 for r in sample_results if r)

        # Calculate pass@k for each k
        pass_at_k_scores = {}
        for k in self.k_values:
            if k <= n:
                pass_at_k_scores[k] = pass_at_k(n, correct, k)

        return PassAtKResult(
            task_id=task.task_id,
            total_samples=n,
            correct_samples=correct,
            k_values=self.k_values,
            pass_at_k_scores=pass_at_k_scores,
            sample_results=sample_results,
        )

    async def evaluate_tasks(
        self,
        tasks: list[BenchmarkTask],
        generate_fn: Callable[[BenchmarkTask, float], Any],
        evaluate_fn: Callable[[BenchmarkTask, str], Any],
        n_samples: Optional[int] = None,
        config: Optional[EvaluationConfig] = None,
    ) -> AggregatePassAtKResult:
        """Evaluate multiple tasks and aggregate results.

        Args:
            tasks: List of benchmark tasks
            generate_fn: Async function to generate code samples
            evaluate_fn: Async function to evaluate samples
            n_samples: Samples per task
            config: Optional config

        Returns:
            AggregatePassAtKResult with mean pass@k scores
        """
        task_results = []

        for i, task in enumerate(tasks):
            logger.info(f"Evaluating task {i + 1}/{len(tasks)}: {task.task_id}")
            result = await self.evaluate_task(
                task=task,
                generate_fn=generate_fn,
                evaluate_fn=evaluate_fn,
                n_samples=n_samples,
                config=config,
            )
            task_results.append(result)

        # Aggregate pass@k across tasks
        mean_pass_at_k = {}
        for k in self.k_values:
            scores = [r.pass_at_k_scores.get(k, 0.0) for r in task_results]
            mean_pass_at_k[k] = sum(scores) / len(scores) if scores else 0.0

        return AggregatePassAtKResult(
            total_tasks=len(tasks),
            k_values=self.k_values,
            mean_pass_at_k=mean_pass_at_k,
            task_results=task_results,
        )


class GreedyVsSamplingComparison:
    """Compare greedy decoding vs sampling strategies.

    Useful for understanding the trade-off between:
    - Greedy (temperature=0): Most likely output, pass@1
    - Sampling (temperature>0): Diverse outputs, higher pass@k for k>1
    """

    def __init__(self, evaluator: Optional[PassAtKEvaluator] = None):
        """Initialize comparator.

        Args:
            evaluator: PassAtKEvaluator instance
        """
        self.evaluator = evaluator or PassAtKEvaluator()

    async def compare(
        self,
        task: BenchmarkTask,
        generate_fn: Callable[[BenchmarkTask, float], Any],
        evaluate_fn: Callable[[BenchmarkTask, str], Any],
        n_samples: int = 100,
    ) -> dict:
        """Compare greedy vs sampling for a task.

        Args:
            task: Benchmark task
            generate_fn: Generation function
            evaluate_fn: Evaluation function
            n_samples: Number of samples for sampling mode

        Returns:
            Comparison results
        """
        # Greedy (temperature=0, single sample)
        try:
            greedy_sample = await generate_fn(task, 0.0)
            greedy_passed = await evaluate_fn(task, greedy_sample)
        except Exception as e:
            logger.warning(f"Greedy generation failed: {e}")
            greedy_passed = False

        # Sampling
        sampling_result = await self.evaluator.evaluate_task(
            task=task,
            generate_fn=generate_fn,
            evaluate_fn=evaluate_fn,
            n_samples=n_samples,
        )

        return {
            "task_id": task.task_id,
            "greedy_passed": greedy_passed,
            "sampling_pass@1": sampling_result.pass_at_k_scores.get(1, 0.0),
            "sampling_pass@10": sampling_result.pass_at_k_scores.get(10, 0.0),
            "sampling_pass@100": sampling_result.pass_at_k_scores.get(100, 0.0),
            "total_samples": sampling_result.total_samples,
            "correct_samples": sampling_result.correct_samples,
        }


def estimate_required_samples(
    target_pass_rate: float,
    k: int,
    confidence: float = 0.95,
) -> int:
    """Estimate samples needed to achieve target pass@k.

    Useful for planning evaluation runs.

    Args:
        target_pass_rate: Target pass@k probability
        k: The k in pass@k
        confidence: Confidence level

    Returns:
        Estimated number of samples needed
    """
    # Binary search for n
    for n in range(k, 10000):
        # Assume c samples are correct
        # We want pass@k >= target with c/n = some pass rate
        # Iterate over possible c values
        for c in range(0, n + 1):
            if pass_at_k(n, c, k) >= target_pass_rate:
                return n
    return 10000


def generate_report(result: AggregatePassAtKResult) -> str:
    """Generate a formatted Pass@k report.

    Args:
        result: Aggregate Pass@k result

    Returns:
        Formatted report string
    """
    lines = []
    lines.append("=" * 60)
    lines.append("PASS@K EVALUATION REPORT")
    lines.append("=" * 60)
    lines.append("")

    lines.append(f"Total Tasks: {result.total_tasks}")
    lines.append("")

    lines.append("Pass@k Results:")
    lines.append("-" * 40)
    for k in sorted(result.mean_pass_at_k.keys()):
        score = result.mean_pass_at_k[k]
        lines.append(f"  Pass@{k:3d}: {score:.2%}")
    lines.append("")

    # Per-task breakdown
    lines.append("Per-Task Results:")
    lines.append("-" * 40)
    for task_result in result.task_results[:10]:  # Show first 10
        lines.append(
            f"  {task_result.task_id}: "
            f"{task_result.correct_samples}/{task_result.total_samples} correct "
            f"({task_result.pass_rate:.1%})"
        )

    if len(result.task_results) > 10:
        lines.append(f"  ... and {len(result.task_results) - 10} more tasks")

    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)
