"""Real-run benchmark runner for live ChatService sessions against benchmark corpora.

Conforms to BenchmarkRunner protocol so EvaluationHarness can execute it without
modification. Unlike fixture-based runners this runner drives actual agent sessions and
produces FrameworkResult artifacts that can be fed into save_stable_run_publication_bundle().

Example::

    config = RealRunConfig(
        framework=Framework.VICTOR,
        model="claude-opus-4-7",
        benchmark=BenchmarkType.ISSUE_FIX,
        max_tasks=10,
        output_dir=Path("/tmp/bench"),
    )
    runner = RealRunBenchmarkRunner(config)
    eval_result, framework_result = await runner.execute_real_run(eval_config)
"""

from __future__ import annotations

import asyncio
import dataclasses
import datetime
import json
import logging
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

logger = logging.getLogger(__name__)


def _resolve_chat_service() -> Any:
    """Return the canonical ChatService from the container.

    Raises rather than returning ``None``: every caller here treats absence as
    fatal, because a run that never reached ChatService is not a measurement of
    it.
    """
    if get_container is None or ChatServiceProtocol is None:
        raise RuntimeError("ChatService dependencies are unavailable in this install")

    chat_service = get_container().get_optional(ChatServiceProtocol)
    if chat_service is None:
        # Deliberately not bootstrapped here: this module's contract is to drive
        # ChatService via the container and never import AgentOrchestrator.
        # Registering the service graph is the caller's job (see
        # _ensure_canonical_services in the benchmark CLI).
        raise RuntimeError(
            "ChatService is not registered; the caller must build the service graph first"
        )
    return chat_service


try:
    from victor.core import get_container
    from victor.agent.services.protocols import ChatServiceProtocol
    from victor.evaluation.benchmarks.framework_comparison import (
        FrameworkResult,
        compute_metrics_from_result,
        save_stable_run_publication_bundle,
    )
    from victor.evaluation.harness import EvaluationHarness
except Exception:  # pragma: no cover - import isolation for minimal evaluation installs
    get_container = None  # type: ignore[assignment]
    ChatServiceProtocol = None  # type: ignore[assignment]
    FrameworkResult = None  # type: ignore[assignment]
    compute_metrics_from_result = None  # type: ignore[assignment]
    save_stable_run_publication_bundle = None  # type: ignore[assignment]
    EvaluationHarness = None  # type: ignore[assignment]


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class RealRunConfig:
    """Runtime parameters for a live benchmark execution."""

    framework: Any  # Framework enum value; imported lazily to avoid circular deps
    model: str
    benchmark: Any  # BenchmarkType enum value
    max_tasks: Optional[int] = None
    timeout_per_task: int = 300
    parallel_tasks: int = 1
    output_dir: Optional[Path] = None


# =============================================================================
# Runner
# =============================================================================


class RealRunBenchmarkRunner:
    """Drives a live ChatService session against a benchmark corpus.

    Design contract:
    - Does NOT import AgentOrchestrator; uses ChatService via DI container.
    - Does NOT modify EvaluationHarness, BenchmarkRunner protocol, or
      BaseBenchmarkRunner.
    - Falls back gracefully when ChatService or EvaluationHarness are unavailable
      (import-time isolation).
    """

    def __init__(self, config: RealRunConfig) -> None:
        self._config = config

    async def execute_real_run(
        self,
        eval_config: Any,
        *,
        resume: bool = False,
        benchmark_runner: Any = None,
    ) -> tuple[Any, Any]:
        """Run the benchmark and return (EvaluationResult, FrameworkResult).

        Args:
            eval_config: EvaluationConfig instance.
            resume: Forward to EvaluationHarness to resume from checkpoint.
            benchmark_runner: Optional concrete benchmark runner to register before execution.

        Returns:
            Tuple of (EvaluationResult, FrameworkResult).
        """
        if (
            EvaluationHarness is None
            or FrameworkResult is None
            or compute_metrics_from_result is None
        ):
            raise RuntimeError("Real-run benchmark dependencies are unavailable")

        harness = EvaluationHarness()
        if benchmark_runner is not None:
            harness.register_runner(benchmark_runner)
        agent_callback = self._make_agent_callback()

        eval_result = await harness.run_evaluation(
            eval_config,
            agent_callback=agent_callback,
            resume=resume,
        )

        metrics = compute_metrics_from_result(eval_result)
        framework_result = FrameworkResult(
            framework=self._config.framework,
            benchmark=self._config.benchmark,
            model=self._config.model,
            metrics=metrics,
            config={
                "source": "real_run",
                "timeout_per_task": self._config.timeout_per_task,
                "parallel_tasks": self._config.parallel_tasks,
                "max_tasks": self._config.max_tasks,
            },
            task_results=[],
        )

        if self._config.output_dir is not None:
            self._maybe_save_bundle(eval_result, framework_result)

        return eval_result, framework_result

    def _make_agent_callback(self) -> Callable[[Any], Awaitable[str]]:
        """Return an async callable that submits a BenchmarkTask to a live ChatService session."""
        timeout = self._config.timeout_per_task

        async def _run_task(task: Any) -> str:
            try:
                chat_service = _resolve_chat_service()
            except Exception as exc:
                # Returning "" here used to look like a task the agent simply
                # failed. It is not: no agent ran at all, and the harness went on
                # to write a full artifact reporting 0 tool calls, 0 tokens and a
                # test failure caused by the empty answer. Whole benchmark runs
                # were recorded that way — indistinguishable from genuine poor
                # performance unless you noticed the token count. The point of
                # this runner is to measure the canonical ChatService path, so if
                # that path is unavailable the run is meaningless and must stop.
                raise RuntimeError(
                    "RealRunBenchmarkRunner requires the canonical ChatService, which could "
                    f"not be resolved ({exc}). Refusing to record a run that never invoked "
                    "an agent. Use `victor benchmark run` for the adapter path."
                ) from exc

            prompt = getattr(task, "prompt", None) or str(task)
            try:
                return await asyncio.wait_for(
                    self._invoke_chat_service(chat_service, prompt),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "RealRunBenchmarkRunner: task timed out after %ss (task_id=%s)",
                    timeout,
                    getattr(task, "task_id", "?"),
                )
                return ""
            except Exception as exc:
                logger.warning("RealRunBenchmarkRunner: task failed: %s", exc)
                return ""

        return _run_task

    async def _invoke_chat_service(self, chat_service: Any, prompt: str) -> str:
        """Submit a single prompt to ChatService and collect the text response."""
        chunks: list[str] = []
        try:
            async for event in chat_service.stream_response(prompt):
                content = getattr(event, "content", None) or getattr(event, "text", None)
                if content:
                    chunks.append(str(content))
        except Exception:
            # Fallback: try synchronous-style chat if stream not available
            result = await chat_service.chat(prompt)
            return str(result)
        return "".join(chunks)

    def _maybe_save_bundle(self, eval_result: Any, framework_result: Any) -> None:
        """Attempt to persist the framework result as a publication bundle."""
        try:
            output_dir = self._config.output_dir
            assert output_dir is not None
            if save_stable_run_publication_bundle is None:
                raise RuntimeError("stable-run publication bundler is unavailable")
            output_dir.mkdir(parents=True, exist_ok=True)

            # Serialize an evaluation-shaped artifact; the stable-run loader derives
            # task-level KPIs from this shape.
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False, dir=output_dir
            ) as fh:
                json.dump(
                    self._to_saved_result_artifact(eval_result, framework_result),
                    fh,
                    default=self._json_default,
                    indent=2,
                )
                tmp_path = Path(fh.name)

            save_stable_run_publication_bundle(
                output_path=output_dir,
                result_paths=[tmp_path],
                benchmark=getattr(self._config.benchmark, "value", str(self._config.benchmark)),
            )
            tmp_path.unlink(missing_ok=True)
            logger.info("RealRunBenchmarkRunner: publication bundle saved to %s", output_dir)
        except Exception as exc:
            logger.warning("RealRunBenchmarkRunner: bundle save failed: %s", exc)

    def _to_saved_result_artifact(self, eval_result: Any, framework_result: Any) -> dict[str, Any]:
        """Return the saved-result JSON shape consumed by stable-run publication."""
        config = getattr(eval_result, "config", None)
        benchmark_value = getattr(getattr(config, "benchmark", None), "value", None)
        if not isinstance(benchmark_value, str):
            benchmark_value = getattr(self._config.benchmark, "value", None)
        model_value = getattr(config, "model", None)
        model = model_value if isinstance(model_value, str) else self._config.model
        provider_value = getattr(config, "provider", None)
        provider = provider_value if isinstance(provider_value, str) else None
        metrics = (
            eval_result.get_metrics()
            if hasattr(eval_result, "get_metrics")
            else dataclasses.asdict(getattr(framework_result, "metrics", {}))
        )
        artifact_config: dict[str, Any]
        try:
            to_artifact_config = getattr(config, "to_artifact_config", None)
            candidate_config = to_artifact_config() if callable(to_artifact_config) else None
        except Exception:
            candidate_config = None
        if isinstance(candidate_config, dict):
            artifact_config = candidate_config
        else:
            artifact_config = {
                "benchmark": benchmark_value or getattr(self._config.benchmark, "value", None),
                "model": model,
                "provider": provider,
                "max_tasks": self._config.max_tasks,
                "timeout_per_task": self._config.timeout_per_task,
                "parallel_tasks": self._config.parallel_tasks,
            }
        artifact_config["source"] = "real_run"

        return {
            "benchmark": benchmark_value or getattr(self._config.benchmark, "value", None),
            "model": model,
            "provider": provider,
            "timestamp": datetime.datetime.now().isoformat(),
            "config": artifact_config,
            "metrics": metrics,
            "task_results": [
                self._task_result_to_artifact(task)
                for task in list(getattr(eval_result, "task_results", []) or [])
            ],
        }

    def _task_result_to_artifact(self, task_result: Any) -> dict[str, Any]:
        """Serialize a TaskResult-like object into benchmark artifact fields."""
        status = getattr(task_result, "status", None)
        failure_category = getattr(task_result, "failure_category", None)
        return {
            "task_id": getattr(task_result, "task_id", None),
            "status": getattr(status, "value", status),
            # Correlation spine — joins this task's decisions to its outcome.
            # The full execution trace lives in the per-run eval_manifest_*.jsonl.
            "session_id": getattr(task_result, "session_id", ""),
            "tests_passed": getattr(task_result, "tests_passed", 0),
            "tests_total": getattr(task_result, "tests_total", 0),
            "duration": getattr(task_result, "duration_seconds", 0.0),
            "duration_seconds": getattr(task_result, "duration_seconds", 0.0),
            "tokens_used": getattr(task_result, "tokens_used", 0),
            "tokens_input": getattr(task_result, "tokens_input", 0),
            "tokens_output": getattr(task_result, "tokens_output", 0),
            "cached_tokens": getattr(task_result, "cached_tokens", 0),
            "reasoning_tokens": getattr(task_result, "reasoning_tokens", 0),
            "cost_usd_micros": getattr(task_result, "cost_usd_micros", 0),
            "tool_calls": getattr(task_result, "tool_calls", 0),
            "turns": getattr(task_result, "turns", 0),
            "code_search_calls": getattr(task_result, "code_search_calls", 0),
            "graph_calls": getattr(task_result, "graph_calls", 0),
            "completion_score": getattr(task_result, "completion_score", 0.0),
            "failure_category": getattr(failure_category, "value", failure_category),
            "failure_details": dict(getattr(task_result, "failure_details", {}) or {}),
            "metadata": dict(getattr(task_result, "metadata", {}) or {}),
        }

    @staticmethod
    def _json_default(obj: Any) -> Any:
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return dataclasses.asdict(obj)
        if hasattr(obj, "value"):
            return obj.value
        if isinstance(obj, (datetime.datetime, datetime.date)):
            return obj.isoformat()
        return str(obj)
