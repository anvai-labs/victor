"""Tests for the Victor self-benchmark runner."""

import json

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from victor.evaluation.benchmarks.framework_comparison import (
    ComparisonReport,
    Framework,
    create_quick_comparison,
)
from victor.evaluation.benchmarks.self_benchmark import (
    SelfBenchmarkConfig,
    SelfBenchmarkRunner,
)
from victor.evaluation.protocol import BenchmarkType, TaskStatus
from victor.evaluation.result_correlation import CorrelationReport, SWEBenchScore


class TestSelfBenchmarkConfig:
    """Tests for SelfBenchmarkConfig defaults."""

    def test_config_defaults(self):
        """Config should have sensible defaults."""
        config = SelfBenchmarkConfig()
        assert config.benchmark_types == [BenchmarkType.SWE_BENCH]
        assert config.model == "claude-3-sonnet"
        assert config.provider == "anthropic"
        assert config.max_tasks == 10
        assert config.include_published is True
        assert config.parallel == 1
        assert config.timeout_per_task == 1800

    def test_config_custom_values(self):
        """Config should accept custom values."""
        config = SelfBenchmarkConfig(
            model="gpt-4",
            provider="openai",
            max_tasks=50,
            parallel=4,
        )
        assert config.model == "gpt-4"
        assert config.provider == "openai"
        assert config.max_tasks == 50
        assert config.parallel == 4


class TestCorrelationToEvalResult:
    """Tests for _correlation_to_eval_result conversion."""

    def test_conversion_produces_valid_eval_result(self):
        """Conversion should produce a valid EvaluationResult."""
        config = SelfBenchmarkConfig(max_tasks=3)
        runner = SelfBenchmarkRunner(config)

        report = CorrelationReport(
            total_instances=3,
            resolved_count=2,
            scores=[
                SWEBenchScore(
                    instance_id="test-1",
                    resolved=True,
                    tests_fixed=3,
                    total_fail_to_pass=3,
                    total_pass_to_pass=5,
                    overall_score=1.0,
                ),
                SWEBenchScore(
                    instance_id="test-2",
                    resolved=True,
                    tests_fixed=2,
                    total_fail_to_pass=2,
                    total_pass_to_pass=4,
                    overall_score=1.0,
                ),
                SWEBenchScore(
                    instance_id="test-3",
                    resolved=False,
                    partial=True,
                    tests_fixed=0,
                    total_fail_to_pass=1,
                    total_pass_to_pass=3,
                    overall_score=0.3,
                ),
            ],
        )

        result = runner._correlation_to_eval_result(report, config)

        assert result.total_tasks == 3
        assert result.passed_tasks == 2
        assert result.config.model == "claude-3-sonnet"
        assert result.config.benchmark == BenchmarkType.SWE_BENCH

    def test_conversion_maps_resolved_to_passed(self):
        """Resolved scores should map to PASSED status."""
        config = SelfBenchmarkConfig()
        runner = SelfBenchmarkRunner(config)

        report = CorrelationReport(
            total_instances=1,
            resolved_count=1,
            scores=[
                SWEBenchScore(instance_id="t-1", resolved=True, overall_score=1.0),
            ],
        )

        result = runner._correlation_to_eval_result(report, config)
        assert result.task_results[0].status == TaskStatus.PASSED

    def test_conversion_maps_unresolved_to_failed(self):
        """Unresolved scores should map to FAILED status."""
        config = SelfBenchmarkConfig()
        runner = SelfBenchmarkRunner(config)

        report = CorrelationReport(
            total_instances=1,
            resolved_count=0,
            scores=[
                SWEBenchScore(instance_id="t-1", resolved=False, overall_score=0.0),
            ],
        )

        result = runner._correlation_to_eval_result(report, config)
        assert result.task_results[0].status == TaskStatus.FAILED


class TestSelfBenchmarkRunner:
    """Tests for the SelfBenchmarkRunner."""

    @pytest.mark.asyncio
    async def test_run_produces_comparison_report(self):
        """Run should produce a ComparisonReport."""
        config = SelfBenchmarkConfig(max_tasks=3, output_dir=Path("/tmp/test_benchmark"))
        runner = SelfBenchmarkRunner(config)

        mock_correlation = CorrelationReport(
            total_instances=3,
            resolved_count=1,
            scores=[
                SWEBenchScore(instance_id="t-1", resolved=True, overall_score=1.0),
                SWEBenchScore(instance_id="t-2", resolved=False, overall_score=0.0),
                SWEBenchScore(instance_id="t-3", resolved=False, overall_score=0.2),
            ],
        )

        with patch("victor.evaluation.evaluation_orchestrator.EvaluationOrchestrator") as MockOrch:
            mock_instance = MagicMock()
            mock_instance.run_evaluation = AsyncMock(return_value=mock_correlation)
            MockOrch.return_value = mock_instance

            with patch.object(runner, "_save_report"):
                report = await runner.run()

        assert isinstance(report, ComparisonReport)
        assert report.benchmark == BenchmarkType.SWE_BENCH
        assert len(report.results) >= 1  # At least Victor's result

        # Victor should be in the results
        victor_results = [r for r in report.results if r.framework == Framework.VICTOR]
        assert len(victor_results) == 1

    @pytest.mark.asyncio
    async def test_run_quick_uses_subset(self):
        """run_quick should use max_tasks=5."""
        config = SelfBenchmarkConfig(max_tasks=50)
        runner = SelfBenchmarkRunner(config)

        mock_correlation = CorrelationReport(
            total_instances=5,
            resolved_count=2,
            scores=[SWEBenchScore(instance_id=f"t-{i}", resolved=(i < 2)) for i in range(5)],
        )

        with patch("victor.evaluation.evaluation_orchestrator.EvaluationOrchestrator") as MockOrch:
            mock_instance = MagicMock()
            mock_instance.run_evaluation = AsyncMock(return_value=mock_correlation)
            MockOrch.return_value = mock_instance

            with patch(
                "victor.evaluation.benchmarks.self_benchmark.SelfBenchmarkRunner._save_report"
            ):
                report = await runner.run_quick()

        # The OrchestratorConfig should have been created with max_tasks=5
        call_args = MockOrch.call_args
        orch_config = call_args[0][0]
        assert orch_config.max_tasks == 5

    @pytest.mark.asyncio
    async def test_report_includes_published_results(self):
        """Report should include published competitor data."""
        config = SelfBenchmarkConfig(include_published=True)
        runner = SelfBenchmarkRunner(config)

        mock_correlation = CorrelationReport(
            total_instances=1,
            resolved_count=1,
            scores=[SWEBenchScore(instance_id="t-1", resolved=True)],
        )

        with patch("victor.evaluation.evaluation_orchestrator.EvaluationOrchestrator") as MockOrch:
            mock_instance = MagicMock()
            mock_instance.run_evaluation = AsyncMock(return_value=mock_correlation)
            MockOrch.return_value = mock_instance

            with patch.object(runner, "_save_report"):
                report = await runner.run()

        # Should have Victor + published results
        frameworks = {r.framework for r in report.results}
        assert Framework.VICTOR in frameworks
        # SWE_BENCH has published results for CLAUDE_CODE and AIDER
        assert len(report.results) > 1


class TestReportMarkdownOutput:
    """Tests for markdown report generation."""

    def test_report_markdown_output(self):
        """Markdown generation should work end-to-end."""
        report = create_quick_comparison(
            benchmark=BenchmarkType.SWE_BENCH,
            victor_pass_rate=0.35,
            include_published=True,
        )

        md = report.to_markdown()
        assert "Framework Comparison" in md
        assert "victor" in md
        assert "Leaderboard" in md

    def test_report_json_output(self):
        """JSON export should work."""
        report = create_quick_comparison(
            benchmark=BenchmarkType.SWE_BENCH,
            victor_pass_rate=0.35,
        )

        json_str = report.to_json()
        import json

        data = json.loads(json_str)
        assert data["benchmark"] == "swe_bench"
        assert len(data["results"]) >= 1

    def test_save_report_writes_summary_bundle(self, tmp_path):
        """Saving a report should emit a compact summary artifact for automation."""
        config = SelfBenchmarkConfig(output_dir=tmp_path, include_published=False)
        runner = SelfBenchmarkRunner(config)
        report = create_quick_comparison(
            benchmark=BenchmarkType.SWE_BENCH,
            victor_pass_rate=0.72,
            include_published=False,
        )
        victor_result = report.results[0]
        victor_result.metrics.accepted_patch_rate = 0.6
        victor_result.metrics.tokens_to_merge = 120.0
        victor_result.metrics.avg_time_to_first_edit_seconds = 1.75
        victor_result.metrics.code_intelligence_task_coverage = 0.8

        runner._save_report(report)

        summary = json.loads((tmp_path / "comparison_report_summary.json").read_text())
        assert summary["benchmark"] == "swe_bench"
        assert summary["winner"] == "victor"
        assert summary["framework_count"] == 1
        assert summary["results"][0]["framework"] == "victor"
        assert summary["results"][0]["accepted_patch_rate"] == pytest.approx(0.6)
        assert summary["results"][0]["tokens_to_merge"] == pytest.approx(120.0)
        assert summary["results"][0]["avg_time_to_first_edit_seconds"] == pytest.approx(1.75)


class TestCreateQuickComparison:
    """Tests for the create_quick_comparison helper."""

    def test_quick_comparison_with_defaults(self):
        """Quick comparison with defaults should work."""
        report = create_quick_comparison()
        assert isinstance(report, ComparisonReport)
        assert report.benchmark == BenchmarkType.SWE_BENCH

    def test_quick_comparison_victor_pass_rate(self):
        """Victor pass rate should be reflected in the report."""
        report = create_quick_comparison(victor_pass_rate=0.42)
        victor_result = [r for r in report.results if r.framework == Framework.VICTOR][0]
        assert victor_result.metrics.pass_rate == 0.42

    def test_quick_comparison_without_published(self):
        """Report without published data should only have Victor."""
        report = create_quick_comparison(include_published=False)
        assert len(report.results) == 1
        assert report.results[0].framework == Framework.VICTOR
