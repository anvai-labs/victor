#!/usr/bin/env python3
"""Run full HumanEval benchmark with Victor agent.

This script uses the code generation harness to run the complete
HumanEval benchmark using Victor's LLM providers.

BENCHMARK MODE GUIDE:
HumanEval/MBPP benchmarks test pure code generation - use provider mode only.
This tests raw LLM capability without tools which aren't needed for code gen.

  - provider: Raw LLM capability (DEFAULT - this is what you want for HumanEval)

NOTE: For agentic benchmarks (SWE-bench, Aider Polyglot) that require tools,
file editing, and multi-turn interactions, use the AgenticBenchmarkRunner
from victor.evaluation.agentic_harness instead.

Usage:
    # Run with default (Ollama) profile
    python scripts/run_full_benchmark.py --profile default --tasks 10

    # Run with claude-haiku profile, parallelism=4
    python scripts/run_full_benchmark.py --profile claude-haiku --parallel 4 --tasks 10

    # Run full benchmark (164 tasks)
    python scripts/run_full_benchmark.py --profile claude-haiku --parallel 4

    # Target specific Ollama host directly (bypasses tiered URL selection)
    python scripts/run_full_benchmark.py --profile default --base-url http://192.168.1.20:11434 --model gpt-oss:latest

    # Run two benchmarks in parallel on different hosts (in separate terminals)
    # Terminal 1: gpt-oss on 192.168.1.20
    python scripts/run_full_benchmark.py --profile default --base-url http://192.168.1.20:11434 --model gpt-oss:latest --output /tmp/bench_gpt_oss.json

    # Terminal 2: qwen3-coder on 192.168.1.73
    python scripts/run_full_benchmark.py --profile default --base-url http://192.168.1.73:11434 --model qwen3-coder:30b --output /tmp/bench_qwen3.json
"""

import argparse
import asyncio
import tempfile
import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from victor.config.settings import load_settings
from victor.evaluation import (
    BenchmarkType,
    EvaluationConfig,
    HumanEvalRunner,
    TaskStatus,
    get_harness,
    pass_at_k,
    # Code generation harness (provider-only for HumanEval)
    CodeGenerationBenchmark,
    create_code_gen_runner,
)
from victor.evaluation.protocol import BenchmarkTask
from victor.providers.base import Message
from victor.providers.registry import ProviderRegistry

# Set log level to WARN for cleaner output
logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def print_banner(profile: str, num_tasks: int, parallel: int):
    """Print benchmark banner."""
    print()
    print("=" * 70)
    print("  VICTOR AI CODING ASSISTANT - HumanEval Benchmark")
    print("=" * 70)
    print()
    print(f"Profile:     {profile}")
    print(f"Tasks:       {num_tasks if num_tasks else 'ALL (164)'}")
    print(f"Parallelism: {parallel}")
    print(f"Timestamp:   {datetime.now().isoformat()}")
    print()


def create_agent_callback(
    profile: str, timeout: int = 120, base_url: str = None, model_override: str = None
):
    """Create an agent callback that uses Victor's providers.

    Returns a callback function compatible with EvaluationHarness.run_evaluation().

    Args:
        profile: Profile name from profiles.yaml
        timeout: Request timeout in seconds
        base_url: Override base URL (useful for targeting specific Ollama hosts)
        model_override: Override model name from profile
    """
    # Load settings and profile
    settings = load_settings()
    profiles = settings.load_profiles()

    if profile not in profiles:
        raise ValueError(f"Profile '{profile}' not found. Available: {list(profiles.keys())}")

    profile_config = profiles[profile]
    model_name = model_override or profile_config.model
    print(f"Provider:    {profile_config.provider}/{model_name}")

    # Get provider settings
    provider_settings = settings.get_provider_settings(profile_config.provider)

    # Override base_url if specified (for direct host targeting)
    if base_url:
        provider_settings["base_url"] = base_url
        print(f"Base URL:    {base_url} (override)")

    provider = ProviderRegistry.create(
        profile_config.provider,
        model=model_name,
        **provider_settings,
    )

    async def agent_callback(task: BenchmarkTask) -> str:
        """Generate code for a single task using the LLM."""
        prompt = f"""You are an expert Python programmer. Complete the following function.
Your response must contain ONLY the complete Python code, starting with any necessary imports.
Do not include any explanations, markdown formatting, or code blocks - just the raw Python code.

{task.prompt}

Complete the function above. Write the implementation that passes all the test cases.
"""

        messages = [Message(role="user", content=prompt)]

        try:
            response = await asyncio.wait_for(
                provider.chat(
                    messages=messages,
                    model=model_name,
                    temperature=0.2,
                    max_tokens=1024,
                ),
                timeout=timeout,
            )

            generated_code = response.content.strip()

            # Clean up markdown code blocks if present
            if "```python" in generated_code:
                match = re.search(r"```python\n(.*?)```", generated_code, re.DOTALL)
                if match:
                    generated_code = match.group(1).strip()
            elif "```" in generated_code:
                match = re.search(r"```\n(.*?)```", generated_code, re.DOTALL)
                if match:
                    generated_code = match.group(1).strip()

            return generated_code

        except asyncio.TimeoutError:
            return task.prompt  # Return prompt only (will fail tests)
        except Exception as e:
            logger.warning(f"Error generating code for {task.task_id}: {e}")
            return task.prompt

    return agent_callback, provider, model_name


def create_retry_callback(provider, model_name: str, timeout: int = 120):
    """Create a retry callback for self-correction.

    This callback is called when the generated code fails tests and
    self-correction is enabled. It receives feedback about what went
    wrong and generates a corrected solution.

    Args:
        provider: The LLM provider instance
        model_name: Model name to use
        timeout: Request timeout in seconds

    Returns:
        Async callback function for retry with feedback
    """

    async def retry_callback(task: BenchmarkTask, previous_code: str, feedback_prompt: str) -> str:
        """Generate corrected code based on feedback."""
        messages = [Message(role="user", content=feedback_prompt)]

        try:
            response = await asyncio.wait_for(
                provider.chat(
                    messages=messages,
                    model=model_name,
                    temperature=0.3,  # Slightly higher temp for exploration
                    max_tokens=1024,
                ),
                timeout=timeout,
            )

            generated_code = response.content.strip()

            # Clean up markdown code blocks if present
            if "```python" in generated_code:
                match = re.search(r"```python\n(.*?)```", generated_code, re.DOTALL)
                if match:
                    generated_code = match.group(1).strip()
            elif "```" in generated_code:
                match = re.search(r"```\n(.*?)```", generated_code, re.DOTALL)
                if match:
                    generated_code = match.group(1).strip()

            return generated_code

        except asyncio.TimeoutError:
            return previous_code  # Return previous code on timeout
        except Exception as e:
            logger.warning(f"Error in retry for {task.task_id}: {e}")
            return previous_code

    return retry_callback


async def run_code_gen_benchmark(
    profile: str,
    num_tasks: int,
    output_file: str = None,
    base_url: str = None,
    model_override: str = None,
):
    """Run HumanEval code generation benchmark (provider-only).

    Args:
        profile: Profile name from profiles.yaml
        num_tasks: Number of tasks to run (None for all)
        output_file: Output JSON file path
        base_url: Override base URL
        model_override: Override model name
    """
    print()
    print("=" * 70)
    print("  VICTOR AI CODING ASSISTANT - Code Generation Benchmark")
    print("=" * 70)
    print()
    print(f"Profile:     {profile}")
    print("Mode:        provider (code generation)")
    print(f"Tasks:       {num_tasks if num_tasks else 'ALL (164)'}")
    print(f"Timestamp:   {datetime.now().isoformat()}")
    print()

    # Load tasks using HumanEvalRunner
    harness = get_harness()
    harness.register_runner(HumanEvalRunner())

    config = EvaluationConfig(
        benchmark=BenchmarkType.HUMAN_EVAL,
        model=model_override or profile,
        max_tasks=num_tasks,
        timeout_per_task=120,
    )

    runner = harness.get_runner(config.benchmark)
    tasks = await runner.load_tasks(config)
    print(f"Loaded {len(tasks)} tasks")
    print()

    # Create test callback for validating generated code
    async def test_callback(task: BenchmarkTask, code: str) -> tuple[bool, int, int]:
        """Test generated code against task tests."""
        try:
            result = await runner.run_task(task, code, config)
            passed = result.status == TaskStatus.PASSED
            return passed, result.tests_passed or 0, result.tests_total or 1
        except Exception as e:
            logger.warning(f"Test error for {task.task_id}: {e}")
            return False, 0, 1

    # Create code generation runner
    print("Initializing provider runner...")
    code_gen_runner = create_code_gen_runner(
        profile=profile,
        base_url=base_url,
        model_override=model_override,
    )
    benchmark = CodeGenerationBenchmark(code_gen_runner)

    print()
    print("-" * 70)
    print("Running benchmark (progress shown in real-time)...")
    print("-" * 70)
    print(flush=True)

    # Progress callback
    def print_progress(task_idx, total_tasks, result):
        status = "PASS" if result.success else "FAIL"
        print(f"[provider] [{task_idx+1}/{total_tasks}] {result.task_id}")
        print(f"    Status: {status}, Turns: 1, Tools: 0, Time: {result.duration_seconds:.1f}s")
        if result.error_message:
            print(f"    Error: {result.error_message[:60]}")
        print(flush=True)

    # Run benchmark
    metrics = await benchmark.run_benchmark(
        tasks=tasks,
        test_callback=test_callback,
        config=config,
        progress_callback=print_progress,
    )

    # Print summary
    print()
    print("=" * 70)
    print("CODE GENERATION BENCHMARK RESULTS")
    print("=" * 70)
    print()

    print("LEVEL: PROVIDER")
    print("-" * 40)
    print(f"  Pass Rate:      {metrics.pass_rate:.1%} ({metrics.passed}/{metrics.total_tasks})")
    print(f"  Errors:         {metrics.errors}")
    print(f"  Timeouts:       {metrics.timeouts}")
    print(f"  Avg Tokens:     {metrics.avg_tokens:.0f}")
    print(f"  Avg Time:       {metrics.avg_time:.1f}s")
    print()

    print("=" * 70)

    # Save results
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "profile": profile,
        "mode": "provider",
        "levels": {"provider": metrics.to_dict()},
    }

    if output_file:
        output_path = Path(output_file)
    else:
        output_path = Path(tempfile.mkdtemp(prefix="victor-codegen-")) / "results.json"

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")

    # Also save report
    report = benchmark.generate_report()
    report_path = output_path.with_suffix(".txt")
    report_path.write_text(report)
    print(f"Report saved to: {report_path}")

    return metrics


async def run_benchmark(
    profile: str,
    num_tasks: int,
    parallel: int,
    output_file: str = None,
    base_url: str = None,
    model_override: str = None,
    self_correct: bool = False,
    self_correct_iterations: int = 3,
):
    """Run the benchmark using the existing evaluation harness.

    Args:
        profile: Profile name from profiles.yaml
        num_tasks: Number of tasks to run (None for all)
        parallel: Parallelism level
        output_file: Output JSON file path
        base_url: Override base URL (for targeting specific Ollama hosts)
        model_override: Override model name from profile
        self_correct: Enable self-correction loop
        self_correct_iterations: Max self-correction iterations
    """
    print_banner(profile, num_tasks, parallel)

    if self_correct:
        print(f"Self-Correction: ENABLED (max {self_correct_iterations} iterations)")
        print()

    # Register runner with harness
    harness = get_harness()
    harness.register_runner(HumanEvalRunner())

    # Configure evaluation
    config = EvaluationConfig(
        benchmark=BenchmarkType.HUMAN_EVAL,
        model=model_override or profile,
        max_tasks=num_tasks,
        parallel_tasks=parallel,
        timeout_per_task=120,
        enable_self_correction=self_correct,
        self_correction_max_iterations=self_correct_iterations,
        auto_fix_imports=True,
    )

    # Create agent callback
    print("Initializing provider...")
    agent_callback, provider, model_name = create_agent_callback(
        profile, base_url=base_url, model_override=model_override
    )

    # Create retry callback for self-correction
    retry_callback = None
    if self_correct:
        retry_callback = create_retry_callback(provider, model_name)

    print()

    # Run evaluation using the harness
    print("-" * 70)
    print("Running benchmark (progress shown in real-time)...")
    print("-" * 70)
    print(flush=True)

    # Progress callback to print results as they complete
    def print_progress(task_idx: int, total_tasks: int, task_result):
        status = "PASS" if task_result.is_success else "FAIL"
        quality = task_result.code_quality.get_overall_score() if task_result.code_quality else 0
        duration = task_result.duration_seconds or 0

        print(f"[{task_idx+1}/{total_tasks}] {task_result.task_id}")
        print(f"    Status: {status}")
        print(f"    Quality: {quality:.1f}/100")
        print(f"    Time: {duration:.1f}s")
        if task_result.error_message:
            print(f"    Error: {task_result.error_message[:80]}")
        print(flush=True)

    result = await harness.run_evaluation(
        config,
        agent_callback,
        progress_callback=print_progress,
        retry_callback=retry_callback,
    )

    # Get metrics from harness
    metrics = result.get_metrics()

    # Calculate Pass@k
    n = metrics["total_tasks"]
    c = metrics["passed"]
    pass_at_1 = pass_at_k(n, c, 1)
    pass_at_5 = pass_at_k(n, c, 5) if n >= 5 else None
    pass_at_10 = pass_at_k(n, c, 10) if n >= 10 else None

    # Print summary
    print()
    print("=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)
    print()
    print(f"Profile:          {profile}")
    print(f"Total Tasks:      {metrics['total_tasks']}")
    print(f"Passed:           {metrics['passed']}")
    print(f"Failed:           {metrics['failed']}")
    print(f"Errors:           {metrics['errors']}")
    print(f"Timeouts:         {metrics['timeouts']}")
    print(f"Pass Rate:        {metrics['pass_rate']:.1%}")
    print()
    print("Pass@k Metrics:")
    print(f"  Pass@1:         {pass_at_1:.2%}")
    if pass_at_5:
        print(f"  Pass@5:         {pass_at_5:.2%}")
    if pass_at_10:
        print(f"  Pass@10:        {pass_at_10:.2%}")
    print()
    print(f"Total Duration:   {metrics['duration_seconds']:.1f}s")
    print(f"Avg per Task:     {metrics['duration_seconds']/metrics['total_tasks']:.1f}s")
    print()
    print("=" * 70)

    # Save results
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "profile": profile,
        "metrics": metrics,
        "pass_at_k": {
            "pass_at_1": pass_at_1,
            "pass_at_5": pass_at_5,
            "pass_at_10": pass_at_10,
        },
    }

    if output_file:
        output_path = Path(output_file)
    else:
        output_path = Path(tempfile.mkdtemp(prefix="victor-benchmark-")) / "results.json"

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")

    # Also generate markdown report
    report = harness.generate_report(result, format="markdown")
    report_path = output_path.with_suffix(".md")
    report_path.write_text(report)
    print(f"Report saved to: {report_path}")

    return result


async def main():
    parser = argparse.ArgumentParser(description="Run HumanEval benchmark with Victor agent")
    parser.add_argument(
        "--profile",
        type=str,
        default="default",
        help="Profile to use (default, claude-haiku, etc.)",
    )
    parser.add_argument(
        "--tasks",
        type=int,
        default=None,
        help="Number of tasks to run (default: all 164)",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Parallelism level (default: 1)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for results",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Override base URL (e.g., http://192.168.1.20:11434 for specific Ollama host)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override model name from profile (e.g., gpt-oss:latest)",
    )
    parser.add_argument(
        "--self-correct",
        action="store_true",
        help="Enable self-correction loop (validate→test→feedback→retry)",
    )
    parser.add_argument(
        "--self-correct-iterations",
        type=int,
        default=3,
        help="Max iterations for self-correction (default: 3)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["provider", "orchestrator", "cli", "compare"],
        default=None,
        help="Benchmark mode: 'provider' only (orchestrator/cli/compare deprecated for HumanEval)",
    )

    args = parser.parse_args()

    # Check for deprecated modes
    if args.mode in ["orchestrator", "cli", "compare"]:
        print()
        print("=" * 70)
        print("  DEPRECATED MODE")
        print("=" * 70)
        print()
        print(f"  The '{args.mode}' mode has been removed from HumanEval benchmarks.")
        print()
        print("  HumanEval tests pure code generation - tools add overhead without value.")
        print()
        print("  Options:")
        print("    1. Remove --mode or use --mode provider for HumanEval")
        print("    2. For agentic tasks requiring tools, use AgenticBenchmarkRunner")
        print("       from victor.evaluation.agentic_harness")
        print()
        print("=" * 70)
        sys.exit(1)

    # Route to appropriate benchmark based on mode
    if args.mode == "provider":
        # Explicit provider mode - use code generation benchmark
        await run_code_gen_benchmark(
            profile=args.profile,
            num_tasks=args.tasks,
            output_file=args.output,
            base_url=args.base_url,
            model_override=args.model,
        )
    else:
        # Standard benchmark (provider-level with existing harness)
        await run_benchmark(
            profile=args.profile,
            num_tasks=args.tasks,
            parallel=args.parallel,
            output_file=args.output,
            base_url=args.base_url,
            model_override=args.model,
            self_correct=args.self_correct,
            self_correct_iterations=args.self_correct_iterations,
        )


if __name__ == "__main__":
    asyncio.run(main())
