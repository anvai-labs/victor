"""Tests for the benchmark real-run command seam."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestRunRealBenchmarkAsync:
    @pytest.mark.asyncio
    async def test_registers_resolved_runner_and_preserves_publication_config(
        self,
        tmp_path: Path,
    ):
        from victor.evaluation.protocol import BenchmarkType, EvaluationConfig
        from victor.ui.commands.benchmark import _run_real_benchmark_async

        config = EvaluationConfig(
            benchmark=BenchmarkType.HUMAN_EVAL,
            model="test-model",
            max_tasks=2,
            timeout_per_task=123,
            parallel_tasks=3,
        )
        benchmark_runner = MagicMock()
        eval_result = MagicMock()
        framework_result = MagicMock()
        real_runner = MagicMock()
        real_runner.execute_real_run = AsyncMock(return_value=(eval_result, framework_result))
        manifest_path = tmp_path / "eval_manifest_realrun.jsonl"

        with (
            patch(
                "victor.evaluation.real_run_runner.RealRunBenchmarkRunner",
                return_value=real_runner,
            ) as runner_cls,
            patch(
                "victor.evaluation.manifest.emit_execution_manifest",
                return_value=manifest_path,
            ) as emit_manifest,
        ):
            result = await _run_real_benchmark_async(
                runner=benchmark_runner,
                config=config,
                output_dir=tmp_path,
                resume=True,
                profile="default",
            )

        real_config = runner_cls.call_args.args[0]
        assert real_config.model == "test-model"
        assert real_config.benchmark == BenchmarkType.HUMAN_EVAL
        assert real_config.max_tasks == 2
        assert real_config.timeout_per_task == 123
        assert real_config.parallel_tasks == 3
        assert real_config.output_dir == tmp_path
        real_runner.execute_real_run.assert_awaited_once_with(
            config,
            resume=True,
            benchmark_runner=benchmark_runner,
        )
        emit_manifest.assert_called_once_with(eval_result)
        assert result == (eval_result, framework_result)

    @pytest.mark.asyncio
    async def test_manifest_failure_does_not_discard_real_run_result(self):
        from victor.evaluation.protocol import BenchmarkType, EvaluationConfig
        from victor.ui.commands.benchmark import _run_real_benchmark_async

        config = EvaluationConfig(
            benchmark=BenchmarkType.HUMAN_EVAL,
            model="test-model",
        )
        eval_result = MagicMock()
        framework_result = MagicMock()
        real_runner = MagicMock()
        real_runner.execute_real_run = AsyncMock(return_value=(eval_result, framework_result))

        with (
            patch(
                "victor.evaluation.real_run_runner.RealRunBenchmarkRunner",
                return_value=real_runner,
            ),
            patch(
                "victor.evaluation.manifest.emit_execution_manifest",
                side_effect=OSError("artifact directory is read-only"),
            ),
        ):
            result = await _run_real_benchmark_async(
                runner=MagicMock(),
                config=config,
                output_dir=None,
                resume=False,
                profile="default",
            )

        assert result == (eval_result, framework_result)
