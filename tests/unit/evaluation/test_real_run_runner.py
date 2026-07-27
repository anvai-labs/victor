"""Unit tests for RealRunBenchmarkRunner (Item 1)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from victor.evaluation.real_run_runner import RealRunBenchmarkRunner, RealRunConfig


def _config(**kw) -> RealRunConfig:
    defaults = {
        "framework": MagicMock(value="victor"),
        "model": "m",
        "benchmark": MagicMock(value="issue_fix"),
    }
    defaults.update(kw)
    return RealRunConfig(**defaults)


def _mock_task(task_id: str = "t1", prompt: str = "fix the bug") -> MagicMock:
    t = MagicMock()
    t.task_id = task_id
    t.prompt = prompt
    return t


# ---------------------------------------------------------------------------
# execute_real_run
# ---------------------------------------------------------------------------


class TestExecuteRealRun:
    async def test_execute_real_run_calls_harness_with_agent_callback(self):
        mock_eval_result = MagicMock()
        mock_eval_result.total_tasks = 1
        mock_eval_result.pass_rate = 1.0
        mock_eval_result.duration_seconds = 1.0
        mock_eval_result.task_results = []
        mock_eval_result.get_metrics.return_value = {}

        mock_harness = MagicMock()
        mock_harness.register_runner = MagicMock()
        mock_harness.run_evaluation = AsyncMock(return_value=mock_eval_result)

        mock_metrics = MagicMock()
        mock_metrics.pass_rate = 1.0

        config = _config()
        runner = RealRunBenchmarkRunner(config)

        with (
            patch(
                "victor.evaluation.real_run_runner.EvaluationHarness",
                return_value=mock_harness,
            ),
            patch(
                "victor.evaluation.real_run_runner.compute_metrics_from_result",
                return_value=mock_metrics,
            ),
            patch(
                "victor.evaluation.benchmarks.framework_comparison.FrameworkResult"
            ) as mock_fr_cls,
        ):
            mock_fr_cls.return_value = MagicMock()
            benchmark_runner = MagicMock()
            eval_result, framework_result = await runner.execute_real_run(
                MagicMock(),
                benchmark_runner=benchmark_runner,
            )

        mock_harness.register_runner.assert_called_once_with(benchmark_runner)
        mock_harness.run_evaluation.assert_awaited_once()
        assert eval_result is mock_eval_result

    async def test_framework_result_metrics_match_task_outcomes(self):
        mock_eval_result = MagicMock()
        mock_eval_result.total_tasks = 2
        mock_eval_result.pass_rate = 0.5
        mock_eval_result.duration_seconds = 10.0
        mock_eval_result.task_results = []
        mock_eval_result.get_metrics.return_value = {}

        mock_harness = MagicMock()
        mock_harness.run_evaluation = AsyncMock(return_value=mock_eval_result)

        from victor.evaluation.benchmarks.framework_comparison import ComparisonMetrics

        real_metrics = ComparisonMetrics(pass_rate=0.5)

        config = _config()
        runner = RealRunBenchmarkRunner(config)

        with (
            patch(
                "victor.evaluation.real_run_runner.EvaluationHarness",
                return_value=mock_harness,
            ),
            patch(
                "victor.evaluation.real_run_runner.compute_metrics_from_result",
                return_value=real_metrics,
            ),
        ):
            _, framework_result = await runner.execute_real_run(MagicMock())

        assert framework_result.metrics.pass_rate == pytest.approx(0.5)

    async def test_execute_real_run_without_runner_preserves_existing_harness_registry(
        self,
    ):
        mock_eval_result = MagicMock()
        mock_eval_result.task_results = []
        mock_harness = MagicMock()
        mock_harness.register_runner = MagicMock()
        mock_harness.run_evaluation = AsyncMock(return_value=mock_eval_result)

        config = _config()
        runner = RealRunBenchmarkRunner(config)

        with (
            patch(
                "victor.evaluation.real_run_runner.EvaluationHarness",
                return_value=mock_harness,
            ),
            patch(
                "victor.evaluation.real_run_runner.compute_metrics_from_result",
                return_value=MagicMock(),
            ),
        ):
            await runner.execute_real_run(MagicMock())

        mock_harness.register_runner.assert_not_called()


# ---------------------------------------------------------------------------
# agent callback
# ---------------------------------------------------------------------------


class TestAgentCallback:
    async def test_agent_callback_calls_chat_service(self):
        config = _config()
        runner = RealRunBenchmarkRunner(config)
        callback = runner._make_agent_callback()

        mock_chat = MagicMock()

        async def _fake_stream(prompt):
            event = MagicMock()
            event.content = "hello"
            yield event

        mock_chat.stream_response = _fake_stream

        with patch("victor.evaluation.real_run_runner.get_container") as mock_get:
            container = MagicMock()
            container.get_optional.return_value = mock_chat
            mock_get.return_value = container
            result = await callback(_mock_task())

        assert "hello" in result

    async def test_agent_callback_raises_when_chat_service_is_unavailable(self):
        """A run that never reached ChatService is not a measurement of it.

        This used to return "" and let the harness record a complete artifact
        showing 0 tool calls, 0 tokens and a test failure caused by the empty
        answer — indistinguishable from genuine poor performance. Whole
        benchmark runs were written that way.
        """
        config = _config()
        runner = RealRunBenchmarkRunner(config)
        callback = runner._make_agent_callback()

        with patch(
            "victor.evaluation.real_run_runner.get_container",
            side_effect=RuntimeError("no container"),
        ):
            with pytest.raises(RuntimeError, match="canonical ChatService"):
                await callback(_mock_task())

    async def test_agent_callback_raises_when_chat_service_is_unregistered(self):
        """Present container, absent service — still fatal, not a silent empty run."""
        config = _config()
        runner = RealRunBenchmarkRunner(config)
        callback = runner._make_agent_callback()

        with patch("victor.evaluation.real_run_runner.get_container") as mock_get:
            container = MagicMock()
            container.get_optional.return_value = None
            mock_get.return_value = container
            with pytest.raises(RuntimeError, match="canonical ChatService"):
                await callback(_mock_task())


# ---------------------------------------------------------------------------
# output_dir triggers bundle save
# ---------------------------------------------------------------------------


class TestOutputDir:
    async def test_output_dir_triggers_publication_bundle(self, tmp_path):
        mock_eval_result = MagicMock()
        mock_eval_result.total_tasks = 0
        mock_eval_result.pass_rate = 0.0
        mock_eval_result.duration_seconds = 0.0
        mock_eval_result.task_results = []
        mock_eval_result.get_metrics.return_value = {}

        mock_harness = MagicMock()
        mock_harness.run_evaluation = AsyncMock(return_value=mock_eval_result)

        from victor.evaluation.benchmarks.framework_comparison import ComparisonMetrics

        config = _config(output_dir=tmp_path)
        runner = RealRunBenchmarkRunner(config)

        save_calls = []

        with (
            patch(
                "victor.evaluation.real_run_runner.EvaluationHarness",
                return_value=mock_harness,
            ),
            patch(
                "victor.evaluation.real_run_runner.compute_metrics_from_result",
                return_value=ComparisonMetrics(),
            ),
            patch(
                "victor.evaluation.real_run_runner.save_stable_run_publication_bundle",
                side_effect=lambda **kw: save_calls.append(kw),
            ),
        ):
            await runner.execute_real_run(MagicMock())

        assert len(save_calls) >= 1
        assert save_calls[0]["output_path"] == tmp_path

    def test_saved_artifact_includes_task_results_for_publication_readiness(self):
        from victor.evaluation.benchmarks.framework_comparison import ComparisonMetrics
        from victor.evaluation.protocol import (
            BenchmarkType,
            EvaluationConfig,
            EvaluationResult,
            TaskResult,
            TaskStatus,
        )

        eval_config = EvaluationConfig(benchmark=BenchmarkType.HUMAN_EVAL, model="m")
        task_result = TaskResult(
            task_id="task-1",
            status=TaskStatus.PASSED,
            tests_passed=1,
            tests_total=1,
            tokens_used=42,
            tool_calls=3,
        )
        eval_result = EvaluationResult(config=eval_config, task_results=[task_result])
        framework_result = MagicMock(metrics=ComparisonMetrics(pass_rate=1.0))

        artifact = RealRunBenchmarkRunner(_config())._to_saved_result_artifact(
            eval_result,
            framework_result,
        )

        assert artifact["benchmark"] == "human_eval"
        assert artifact["config"]["source"] == "real_run"
        assert artifact["metrics"]["total_tasks"] == 1
        assert artifact["task_results"][0]["task_id"] == "task-1"
        assert artifact["task_results"][0]["tokens_used"] == 42
