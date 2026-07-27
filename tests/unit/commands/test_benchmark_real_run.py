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
        real_runner = MagicMock()
        real_runner.execute_real_run = AsyncMock(return_value=(MagicMock(), MagicMock()))

        with patch(
            "victor.evaluation.real_run_runner.RealRunBenchmarkRunner",
            return_value=real_runner,
        ) as runner_cls:
            await _run_real_benchmark_async(
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
