# Copyright 2025 Vijaykumar Singh
# SPDX-License-Identifier: Apache-2.0
"""Code generation benchmark harness for testing raw LLM capabilities.

This harness evaluates pure code generation capabilities using benchmarks like:
- HumanEval (164 function-level Python tasks)
- MBPP (Mostly Basic Programming Problems)
- BigCodeBench (complex function-level tasks)

These benchmarks test RAW LLM capability without tools or agent overhead.
For agentic tasks (file editing, tool usage), use agentic_harness.py instead.

Why provider-only for HumanEval?
================================
HumanEval tasks are single-shot code generation problems. Adding tools/agents:
1. Introduces unnecessary complexity
2. Causes tool hallucination (models try to read/write files that don't exist)
3. Adds latency without improving accuracy
4. Doesn't test what HumanEval was designed to measure

Metrics tracked:
- Pass@1, Pass@5, Pass@10 rates
- Token efficiency (input/output tokens)
- Time per task
- Code quality scores
"""

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable, Optional

from victor.evaluation.protocol import (
    BenchmarkTask,
    EvaluationConfig,
)

logger = logging.getLogger(__name__)


@dataclass
class CodeGenMetrics:
    """Metrics for code generation benchmarks."""

    # Pass metrics
    passed: int = 0
    failed: int = 0
    errors: int = 0
    timeouts: int = 0

    # Efficiency metrics
    total_tokens_input: int = 0
    total_tokens_output: int = 0
    total_time_seconds: float = 0.0

    # Per-task details
    task_metrics: list[dict[str, Any]] = field(default_factory=list)

    @property
    def total_tasks(self) -> int:
        """Total tasks attempted."""
        return self.passed + self.failed + self.errors + self.timeouts

    @property
    def pass_rate(self) -> float:
        """Pass rate as a decimal."""
        if self.total_tasks == 0:
            return 0.0
        return self.passed / self.total_tasks

    @property
    def avg_tokens(self) -> float:
        """Average total tokens per task."""
        if self.total_tasks == 0:
            return 0.0
        return (self.total_tokens_input + self.total_tokens_output) / self.total_tasks

    @property
    def avg_time(self) -> float:
        """Average time per task in seconds."""
        if self.total_tasks == 0:
            return 0.0
        return self.total_time_seconds / self.total_tasks

    def to_dict(self) -> dict[str, Any]:
        """Export metrics as dictionary."""
        return {
            "summary": {
                "total_tasks": self.total_tasks,
                "passed": self.passed,
                "failed": self.failed,
                "errors": self.errors,
                "timeouts": self.timeouts,
                "pass_rate": round(self.pass_rate, 4),
            },
            "efficiency": {
                "total_tokens_input": self.total_tokens_input,
                "total_tokens_output": self.total_tokens_output,
                "avg_tokens": round(self.avg_tokens, 2),
                "total_time_seconds": round(self.total_time_seconds, 2),
                "avg_time_seconds": round(self.avg_time, 2),
            },
            "tasks": self.task_metrics,
        }


@dataclass
class CodeGenResult:
    """Result from generating code for a single task."""

    task_id: str
    success: bool
    generated_code: str = ""
    error_message: str = ""

    # Token metrics
    tokens_input: int = 0
    tokens_output: int = 0
    duration_seconds: float = 0.0

    # Test results
    tests_passed: int = 0
    tests_total: int = 0


class CodeGenerationRunner:
    """Runner for code generation benchmarks (HumanEval, MBPP, etc.).

    Uses direct provider calls (no tools, no agent loop) to test
    pure LLM code generation capability.
    """

    def __init__(
        self,
        provider: Any,
        model_name: str,
        timeout: int = 120,
    ):
        """Initialize the runner.

        Args:
            provider: LLM provider instance (must implement chat method)
            model_name: Model name to use for generation
            timeout: Request timeout in seconds
        """
        self._provider = provider
        self._model_name = model_name
        self._timeout = timeout

    async def generate_code(
        self,
        task: BenchmarkTask,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> CodeGenResult:
        """Generate code for a single task.

        Args:
            task: The benchmark task
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            CodeGenResult with generated code and metrics
        """
        from victor.providers.base import Message

        start_time = time.time()

        # Build the prompt - clean, focused on code generation
        prompt = f"""You are an expert Python programmer. Complete the following function.
Your response must contain ONLY the complete Python code, starting with any necessary imports.
Do not include any explanations, markdown formatting, or code blocks - just the raw Python code.

{task.prompt}

Complete the function above. Write the implementation that passes all the test cases.
"""

        messages = [Message(role="user", content=prompt)]

        try:
            response = await asyncio.wait_for(
                self._provider.chat(
                    messages=messages,
                    model=self._model_name,
                    temperature=temperature,
                    max_tokens=max_tokens,
                ),
                timeout=self._timeout,
            )

            generated_code = response.content.strip()

            # Clean up markdown code blocks if present
            generated_code = self._clean_code(generated_code)

            duration = time.time() - start_time

            return CodeGenResult(
                task_id=task.task_id,
                success=True,  # Will be validated by test callback
                generated_code=generated_code,
                tokens_input=getattr(response, "input_tokens", 0),
                tokens_output=getattr(response, "output_tokens", 0),
                duration_seconds=duration,
            )

        except asyncio.TimeoutError:
            return CodeGenResult(
                task_id=task.task_id,
                success=False,
                error_message="Timeout",
                duration_seconds=time.time() - start_time,
            )
        except Exception as e:
            return CodeGenResult(
                task_id=task.task_id,
                success=False,
                error_message=str(e),
                duration_seconds=time.time() - start_time,
            )

    def _clean_code(self, code: str) -> str:
        """Clean up markdown code blocks from response."""
        if "```python" in code:
            match = re.search(r"```python\n(.*?)```", code, re.DOTALL)
            if match:
                return match.group(1).strip()
        elif "```" in code:
            match = re.search(r"```\n(.*?)```", code, re.DOTALL)
            if match:
                return match.group(1).strip()
        return code


class CodeGenerationBenchmark:
    """Orchestrates code generation benchmarks.

    This is a simplified benchmark runner specifically for HumanEval-style
    code generation tasks. It uses direct provider calls without any
    tool or agent infrastructure.
    """

    def __init__(self, runner: CodeGenerationRunner):
        """Initialize the benchmark.

        Args:
            runner: CodeGenerationRunner instance
        """
        self._runner = runner
        self._metrics = CodeGenMetrics()

    async def run_benchmark(
        self,
        tasks: list[BenchmarkTask],
        test_callback: Callable[[BenchmarkTask, str], Awaitable[tuple[bool, int, int]]],
        config: EvaluationConfig,
        progress_callback: Optional[Callable[[int, int, CodeGenResult], None]] = None,
    ) -> CodeGenMetrics:
        """Run the code generation benchmark.

        Args:
            tasks: List of benchmark tasks
            test_callback: Async callback to test generated code.
                          Signature: (task, code) -> (passed, tests_passed, tests_total)
            config: Evaluation configuration
            progress_callback: Optional progress callback.
                              Signature: (task_idx, total, result) -> None

        Returns:
            CodeGenMetrics with benchmark results
        """
        self._metrics = CodeGenMetrics()

        logger.info(f"Running code generation benchmark with {len(tasks)} tasks...")

        for i, task in enumerate(tasks):
            try:
                # Generate code
                result = await self._runner.generate_code(task)

                # Test the generated code
                if result.generated_code:
                    passed, tests_passed, tests_total = await test_callback(
                        task, result.generated_code
                    )
                    result.success = passed
                    result.tests_passed = tests_passed
                    result.tests_total = tests_total

                # Update metrics
                if result.success:
                    self._metrics.passed += 1
                elif result.error_message == "Timeout":
                    self._metrics.timeouts += 1
                elif result.error_message:
                    self._metrics.errors += 1
                else:
                    self._metrics.failed += 1

                self._metrics.total_tokens_input += result.tokens_input
                self._metrics.total_tokens_output += result.tokens_output
                self._metrics.total_time_seconds += result.duration_seconds

                self._metrics.task_metrics.append(
                    {
                        "task_id": result.task_id,
                        "success": result.success,
                        "tokens": result.tokens_input + result.tokens_output,
                        "duration": round(result.duration_seconds, 2),
                        "tests_passed": result.tests_passed,
                        "tests_total": result.tests_total,
                    }
                )

                # Progress callback
                if progress_callback:
                    progress_callback(i, len(tasks), result)

            except Exception as e:
                logger.error(f"Error running task {task.task_id}: {e}")
                self._metrics.errors += 1

        return self._metrics

    def generate_report(self) -> str:
        """Generate human-readable benchmark report."""
        lines = []
        lines.append("=" * 70)
        lines.append("       CODE GENERATION BENCHMARK REPORT")
        lines.append("=" * 70)
        lines.append("")

        lines.append("SUMMARY")
        lines.append("-" * 40)
        lines.append(f"  Total Tasks:    {self._metrics.total_tasks}")
        lines.append(f"  Pass Rate:      {self._metrics.pass_rate:.1%}")
        lines.append(f"  Passed:         {self._metrics.passed}")
        lines.append(f"  Failed:         {self._metrics.failed}")
        lines.append(f"  Errors:         {self._metrics.errors}")
        lines.append(f"  Timeouts:       {self._metrics.timeouts}")
        lines.append("")

        lines.append("EFFICIENCY")
        lines.append("-" * 40)
        lines.append(f"  Avg Tokens:     {self._metrics.avg_tokens:.0f}")
        lines.append(f"  Avg Time:       {self._metrics.avg_time:.1f}s")
        lines.append(f"  Total Time:     {self._metrics.total_time_seconds:.1f}s")
        lines.append("")

        lines.append("=" * 70)
        return "\n".join(lines)


def create_code_gen_runner(
    profile: str,
    base_url: Optional[str] = None,
    model_override: Optional[str] = None,
    timeout: int = 120,
) -> CodeGenerationRunner:
    """Create a code generation runner from profile.

    Args:
        profile: Profile name from profiles.yaml
        base_url: Override base URL
        model_override: Override model name
        timeout: Request timeout

    Returns:
        CodeGenerationRunner instance
    """
    from victor.config.settings import load_settings
    from victor.providers.registry import ProviderRegistry

    settings = load_settings()
    profiles = settings.load_profiles()

    if profile not in profiles:
        raise ValueError(f"Profile '{profile}' not found")

    profile_config = profiles[profile]
    model_name = model_override or profile_config.model
    # Thread the profile's credential-identity account so the eval harness
    # resolves the profile's OWN key/endpoint (account-scoped), matching the
    # chat/orchestrator path. Mirrors #703's account threading.
    provider_settings = settings.get_provider_settings(
        profile_config.provider, account_name=getattr(profile_config, "account", None)
    )

    if base_url:
        provider_settings["base_url"] = base_url

    provider = ProviderRegistry.create(
        profile_config.provider,
        model=model_name,
        **provider_settings,
    )

    return CodeGenerationRunner(provider, model_name, timeout)
