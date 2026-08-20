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

"""Tests for framework comparison module."""

import hashlib
import json
from pathlib import Path

import pytest
from victor.evaluation.benchmarks.framework_comparison import (
    DEFAULT_FIXTURE_SET_ROOT,
    Framework,
    FrameworkCapabilities,
    FixtureBenchmarkDescriptor,
    FixtureSetVerificationResult,
    FrameworkResult,
    ComparisonMetrics,
    ComparisonReport,
    FRAMEWORK_CAPABILITIES,
    PUBLISHED_RESULTS,
    compute_metrics_from_result,
    compute_metrics_from_saved_result,
    create_comparison_report,
    create_comparison_report_from_fixture_manifest,
    create_comparison_report_from_saved_result,
    create_comparison_report_from_saved_results,
    create_quick_comparison,
    build_fixture_benchmark_catalog,
    canonicalize_fixture_benchmark_name,
    discover_fixture_benchmarks,
    discover_fixture_sets,
    fixture_benchmark_matches,
    get_published_result,
    load_framework_result_from_file,
    resolve_fixture_benchmark_publication_manifests,
    resolve_fixture_sets_for_benchmark,
    resolve_fixture_set_names,
    save_fixture_benchmark_catalog,
    save_fixture_benchmark_publication_bundle,
    save_comparison_report_bundle,
    save_stable_run_publication_bundle,
    verify_fixture_sets,
)
from victor.evaluation.protocol import (
    BenchmarkType,
    EvaluationConfig,
    EvaluationResult,
    TaskResult,
    TaskStatus,
)


class TestFramework:
    """Tests for Framework enum."""

    def test_all_frameworks_defined(self):
        """Test that all expected frameworks are defined."""
        expected = {
            "victor",
            "aider",
            "claude_code",
            "cursor",
            "github_copilot",
            "cody",
            "continue",
            "tabby",
            "codegpt",
            "custom",
        }
        actual = {f.value for f in Framework}
        assert expected == actual

    def test_framework_values(self):
        """Test framework enum values."""
        assert Framework.VICTOR.value == "victor"
        assert Framework.AIDER.value == "aider"
        assert Framework.CLAUDE_CODE.value == "claude_code"


class TestFrameworkCapabilities:
    """Tests for FrameworkCapabilities dataclass."""

    def test_required_values(self):
        """Test that required capability values are set."""
        caps = FrameworkCapabilities(name="Test", framework=Framework.CUSTOM)
        assert caps.name == "Test"
        assert caps.framework == Framework.CUSTOM
        assert caps.code_generation is True
        assert caps.code_editing is True
        assert caps.multi_file_editing is False
        assert caps.autonomous_mode is False
        assert caps.open_source is False

    def test_victor_capabilities(self):
        """Test Victor's capabilities are defined correctly."""
        victor_caps = FRAMEWORK_CAPABILITIES[Framework.VICTOR]
        assert victor_caps.code_generation is True
        assert victor_caps.code_editing is True
        assert victor_caps.multi_file_editing is True
        assert victor_caps.tool_use is True
        assert victor_caps.autonomous_mode is True
        assert victor_caps.planning is True
        assert victor_caps.local_models is True
        assert victor_caps.air_gapped is True
        assert victor_caps.mcp_support is True
        assert victor_caps.open_source is True

    def test_all_frameworks_have_capabilities(self):
        """Test that all frameworks have defined capabilities."""
        for framework in [
            Framework.VICTOR,
            Framework.AIDER,
            Framework.CLAUDE_CODE,
            Framework.CURSOR,
        ]:
            assert framework in FRAMEWORK_CAPABILITIES


class TestComparisonMetrics:
    """Tests for ComparisonMetrics dataclass."""

    def test_default_values(self):
        """Test default metric values."""
        metrics = ComparisonMetrics()
        assert metrics.pass_rate == 0.0
        assert metrics.avg_latency_ms == 0.0
        assert metrics.tokens_per_task == 0.0
        assert metrics.cost_per_task == 0.0
        assert metrics.code_quality_score == 0.0
        assert metrics.test_pass_rate == 0.0
        assert metrics.error_rate == 0.0
        assert metrics.timeout_rate == 0.0
        assert metrics.accepted_patch_rate == 0.0
        assert metrics.tokens_to_merge == 0.0
        assert metrics.cost_per_accepted_patch_usd == 0.0
        assert metrics.avg_time_to_first_edit_seconds == 0.0
        assert metrics.code_intelligence_task_coverage == 0.0


class TestFrameworkResult:
    """Tests for FrameworkResult dataclass."""

    def test_creation(self):
        """Test creating a framework result."""
        result = FrameworkResult(
            framework=Framework.VICTOR,
            benchmark=BenchmarkType.SWE_BENCH,
            model="claude-3-sonnet",
            metrics=ComparisonMetrics(pass_rate=0.45),
        )
        assert result.framework == Framework.VICTOR
        assert result.benchmark == BenchmarkType.SWE_BENCH
        assert result.model == "claude-3-sonnet"
        assert result.metrics.pass_rate == 0.45


class TestPublishedResults:
    """Tests for published benchmark results."""

    def test_swe_bench_results_exist(self):
        """Test that SWE-bench results exist."""
        assert BenchmarkType.SWE_BENCH in PUBLISHED_RESULTS
        results = PUBLISHED_RESULTS[BenchmarkType.SWE_BENCH]
        assert len(results) > 0

    def test_results_have_required_fields(self):
        """Test that results have required fields."""
        for bench_type, results in PUBLISHED_RESULTS.items():
            for framework, data in results.items():
                assert "model" in data
                assert "pass_rate" in data
                assert isinstance(data["pass_rate"], (int, float))
                assert 0 <= data["pass_rate"] <= 1

    def test_get_published_result(self):
        """Test getting a published result."""
        result = get_published_result(BenchmarkType.SWE_BENCH, Framework.CLAUDE_CODE)
        if result:
            assert result["model"] is not None
            assert "pass_rate" in result

    def test_get_published_result_missing(self):
        """Test getting a non-existent result."""
        result = get_published_result(BenchmarkType.SWE_BENCH, Framework.CUSTOM)
        assert result is None


class TestComputeMetrics:
    """Tests for compute_metrics_from_result function."""

    def test_compute_metrics_empty(self):
        """Test computing metrics from empty result."""
        config = EvaluationConfig(
            benchmark=BenchmarkType.HUMAN_EVAL,
            model="test-model",
        )
        result = EvaluationResult(config=config)
        metrics = compute_metrics_from_result(result)
        assert metrics.pass_rate == 0.0

    def test_compute_metrics_with_results(self):
        """Test computing metrics from results."""
        config = EvaluationConfig(
            benchmark=BenchmarkType.HUMAN_EVAL,
            model="test-model",
        )
        task_results = [
            TaskResult(
                task_id="1",
                status=TaskStatus.PASSED,
                tests_passed=5,
                tests_total=5,
                duration_seconds=1.0,
                tokens_used=100,
                generated_patch="diff --git a/a.py b/a.py",
                code_search_calls=1,
                metadata={
                    "accepted_patch": True,
                    "time_to_first_edit_seconds": 1.5,
                },
            ),
            TaskResult(
                task_id="2",
                status=TaskStatus.PASSED,
                tests_passed=3,
                tests_total=3,
                duration_seconds=2.0,
                tokens_used=150,
                graph_calls=1,
                metadata={"time_to_first_edit_seconds": 2.5},
            ),
            TaskResult(
                task_id="3",
                status=TaskStatus.FAILED,
                tests_passed=1,
                tests_total=5,
                duration_seconds=1.5,
                tokens_used=120,
            ),
        ]
        result = EvaluationResult(config=config, task_results=task_results)
        metrics = compute_metrics_from_result(result)
        assert metrics.pass_rate == result.pass_rate
        assert metrics.accepted_patch_rate == pytest.approx(1 / 3)
        assert metrics.tokens_to_merge == pytest.approx(100.0)
        assert metrics.avg_time_to_first_edit_seconds == pytest.approx(2.0)
        assert metrics.code_intelligence_task_coverage == pytest.approx(2 / 3)
        assert metrics.code_intelligence_pass_rate == pytest.approx(1.0)
        assert metrics.non_code_intelligence_pass_rate == pytest.approx(0.0)


class TestComparisonReport:
    """Tests for ComparisonReport dataclass."""

    def test_create_report(self):
        """Test creating a comparison report."""
        report = ComparisonReport(
            benchmark=BenchmarkType.SWE_BENCH,
            results=[
                FrameworkResult(
                    framework=Framework.VICTOR,
                    benchmark=BenchmarkType.SWE_BENCH,
                    model="claude-3-sonnet",
                    metrics=ComparisonMetrics(pass_rate=0.45),
                ),
            ],
        )
        assert report.benchmark == BenchmarkType.SWE_BENCH
        assert len(report.results) == 1

    def test_leaderboard(self):
        """Test getting leaderboard from report."""
        report = ComparisonReport(
            benchmark=BenchmarkType.SWE_BENCH,
            results=[
                FrameworkResult(
                    framework=Framework.VICTOR,
                    benchmark=BenchmarkType.SWE_BENCH,
                    model="claude-3-sonnet",
                    metrics=ComparisonMetrics(pass_rate=0.45),
                ),
                FrameworkResult(
                    framework=Framework.AIDER,
                    benchmark=BenchmarkType.SWE_BENCH,
                    model="claude-3-opus",
                    metrics=ComparisonMetrics(pass_rate=0.268),
                ),
            ],
        )
        leaderboard = report.get_leaderboard()
        assert len(leaderboard) == 2
        assert leaderboard[0][0] == Framework.VICTOR  # Higher pass rate first
        assert leaderboard[1][0] == Framework.AIDER

    def test_get_winner(self):
        """Test getting winner from report."""
        report = ComparisonReport(
            benchmark=BenchmarkType.SWE_BENCH,
            results=[
                FrameworkResult(
                    framework=Framework.VICTOR,
                    benchmark=BenchmarkType.SWE_BENCH,
                    model="claude-3-sonnet",
                    metrics=ComparisonMetrics(pass_rate=0.45),
                ),
                FrameworkResult(
                    framework=Framework.AIDER,
                    benchmark=BenchmarkType.SWE_BENCH,
                    model="claude-3-opus",
                    metrics=ComparisonMetrics(pass_rate=0.268),
                ),
            ],
        )
        winner = report.get_winner()
        assert winner == Framework.VICTOR

    def test_to_markdown(self):
        """Test generating markdown from report."""
        report = ComparisonReport(
            benchmark=BenchmarkType.SWE_BENCH,
            results=[
                FrameworkResult(
                    framework=Framework.VICTOR,
                    benchmark=BenchmarkType.SWE_BENCH,
                    model="claude-3-sonnet",
                    metrics=ComparisonMetrics(
                        pass_rate=0.45,
                        accepted_patch_rate=0.4,
                        tokens_to_merge=180.0,
                        avg_time_to_first_edit_seconds=2.25,
                        code_intelligence_task_coverage=0.75,
                    ),
                ),
            ],
        )
        markdown = report.to_markdown()
        assert "Framework Comparison" in markdown
        assert "swe_bench" in markdown
        assert "victor" in markdown
        assert "Accepted Patch Rate" in markdown
        assert "Time to First Edit" in markdown
        assert "Code-Intelligence Coverage" in markdown

    def test_to_json(self):
        """Test exporting report as JSON."""
        import json

        report = ComparisonReport(
            benchmark=BenchmarkType.SWE_BENCH,
            results=[
                FrameworkResult(
                    framework=Framework.VICTOR,
                    benchmark=BenchmarkType.SWE_BENCH,
                    model="claude-3-sonnet",
                    metrics=ComparisonMetrics(pass_rate=0.45),
                ),
            ],
        )
        json_str = report.to_json()
        data = json.loads(json_str)
        assert data["benchmark"] == "swe_bench"
        assert len(data["results"]) == 1


class TestCreateComparisonReport:
    """Tests for create_comparison_report function."""

    def test_create_report_with_victor_result(self):
        """Test creating report with Victor results."""
        config = EvaluationConfig(
            benchmark=BenchmarkType.SWE_BENCH,
            model="claude-3-sonnet",
        )
        victor_result = EvaluationResult(config=config)
        report = create_comparison_report(BenchmarkType.SWE_BENCH, victor_result)
        assert report.benchmark == BenchmarkType.SWE_BENCH
        # Should have Victor result plus published results
        assert len(report.results) >= 1
        assert any(r.framework == Framework.VICTOR for r in report.results)

    def test_create_report_without_published(self):
        """Test creating report without published results."""
        config = EvaluationConfig(
            benchmark=BenchmarkType.SWE_BENCH,
            model="claude-3-sonnet",
        )
        victor_result = EvaluationResult(config=config)
        report = create_comparison_report(
            BenchmarkType.SWE_BENCH, victor_result, include_published=False
        )
        assert len(report.results) == 1
        assert report.results[0].framework == Framework.VICTOR


class TestSavedResultIngestion:
    """Tests for loading saved benchmark result artifacts."""

    def test_load_framework_result_from_run_output(self, tmp_path):
        """CLI run output should round-trip into a FrameworkResult."""
        path = tmp_path / "guide_result.json"
        path.write_text(
            json.dumps(
                {
                    "benchmark": "guide",
                    "model": "test-model",
                    "timestamp": "2026-04-25T12:00:00",
                    "dataset_metadata": {"source_name": "GUIDE Consortium"},
                    "metrics": {
                        "total_tasks": 2,
                        "passed": 1,
                        "failed": 1,
                        "errors": 0,
                        "timeouts": 0,
                        "pass_rate": 0.5,
                        "duration_seconds": 10.0,
                        "total_tokens": 200,
                        "total_tool_calls": 6,
                    },
                    "task_results": [
                        {
                            "task_id": "guide-1",
                            "status": "passed",
                            "tests_passed": 1,
                            "tests_total": 1,
                            "duration": 4.0,
                            "tool_calls": 4,
                        },
                        {
                            "task_id": "guide-2",
                            "status": "failed",
                            "tests_passed": 0,
                            "tests_total": 1,
                            "duration": 6.0,
                            "tool_calls": 2,
                        },
                    ],
                }
            )
        )

        result = load_framework_result_from_file(path)

        assert result.framework == Framework.VICTOR
        assert result.benchmark == BenchmarkType.GUIDE
        assert result.metrics.pass_rate == 0.5
        assert result.metrics.tokens_per_task == 100.0
        assert result.metrics.tool_calls_per_task == 3.0
        assert result.config["source"] == "GUIDE Consortium"

    def test_load_framework_result_from_harness_output(self, tmp_path):
        """Harness-saved result JSON should also be ingestible."""
        path = tmp_path / "swe_result.json"
        path.write_text(
            json.dumps(
                {
                    "config": {
                        "benchmark": "swe_bench",
                        "model": "claude-3-sonnet",
                        "dataset_metadata": {"source_name": "Harness Fixture"},
                        "prompt_candidate_hash": "cand-123",
                        "prompt_section_name": "GROUNDING_RULES",
                    },
                    "summary": {
                        "total_tasks": 1,
                        "passed": 1,
                        "failed": 0,
                        "errors": 0,
                        "timeouts": 0,
                        "pass_rate": 1.0,
                        "duration_seconds": 2.5,
                        "total_tokens": 50,
                        "total_tool_calls": 1,
                        "accepted_patch_rate": 1.0,
                        "avg_tokens_to_merge": 50.0,
                        "cost_per_accepted_patch_usd": 0.0015,
                        "avg_time_to_first_edit_seconds": 1.25,
                        "code_intelligence_task_coverage": 1.0,
                        "code_intelligence_pass_rate": 1.0,
                        "non_code_intelligence_pass_rate": 0.0,
                    },
                    "tasks": [
                        {
                            "task_id": "swe-1",
                            "status": "passed",
                            "tests_passed": 2,
                            "tests_total": 2,
                            "duration_seconds": 2.5,
                            "tool_calls": 1,
                            "completion_score": 0.8,
                            "code_quality": {"overall_score": 75.0},
                        }
                    ],
                }
            )
        )

        result = load_framework_result_from_file(path)

        assert result.benchmark == BenchmarkType.SWE_BENCH
        assert result.metrics.test_pass_rate == 1.0
        assert result.metrics.partial_completion == 0.8
        assert result.metrics.code_quality_score == 75.0
        assert result.metrics.accepted_patch_rate == 1.0
        assert result.metrics.tokens_to_merge == 50.0
        assert result.metrics.avg_time_to_first_edit_seconds == 1.25
        assert result.metrics.code_intelligence_task_coverage == 1.0
        assert result.config["prompt_candidate_hash"] == "cand-123"
        assert result.config["prompt_section_name"] == "GROUNDING_RULES"

    def test_create_report_from_saved_result_includes_published(self, tmp_path):
        """Saved Victor artifacts should plug into published comparisons."""
        path = tmp_path / "swe_result.json"
        path.write_text(
            json.dumps(
                {
                    "benchmark": "swe-bench",
                    "model": "claude-3-sonnet",
                    "metrics": {
                        "total_tasks": 1,
                        "passed": 1,
                        "failed": 0,
                        "errors": 0,
                        "timeouts": 0,
                        "pass_rate": 1.0,
                    },
                    "task_results": [{"task_id": "swe-1", "status": "passed"}],
                }
            )
        )

        report = create_comparison_report_from_saved_result(path)

        assert any(result.framework == Framework.VICTOR for result in report.results)
        assert any(result.framework == Framework.AIDER for result in report.results)

    def test_create_report_from_multiple_saved_results(self, tmp_path):
        """Multiple local artifacts should compare side by side in one report."""
        first = tmp_path / "guide_result_a.json"
        second = tmp_path / "guide_result_b.json"
        first.write_text(
            json.dumps(
                {
                    "benchmark": "guide",
                    "model": "model-a",
                    "dataset_metadata": {"source_name": "GUIDE Run A"},
                    "metrics": {
                        "total_tasks": 1,
                        "passed": 1,
                        "failed": 0,
                        "errors": 0,
                        "timeouts": 0,
                        "pass_rate": 1.0,
                    },
                    "task_results": [{"task_id": "guide-1", "status": "passed"}],
                }
            )
        )
        second.write_text(
            json.dumps(
                {
                    "benchmark": "guide",
                    "model": "model-b",
                    "dataset_metadata": {"source_name": "GUIDE Run B"},
                    "metrics": {
                        "total_tasks": 1,
                        "passed": 0,
                        "failed": 1,
                        "errors": 0,
                        "timeouts": 0,
                        "pass_rate": 0.0,
                    },
                    "task_results": [{"task_id": "guide-2", "status": "failed"}],
                }
            )
        )

        report = create_comparison_report_from_saved_results(
            [first, second],
            include_published=False,
        )

        assert report.benchmark == BenchmarkType.GUIDE
        assert len(report.results) == 2
        assert [result.model for result in report.results] == ["model-a", "model-b"]
        assert [result.config["source"] for result in report.results] == [
            "GUIDE Run A",
            "GUIDE Run B",
        ]

    def test_save_comparison_report_bundle_writes_primary_and_sidecars(self, tmp_path):
        """Comparison bundle saves should emit markdown, json, and summary artifacts."""
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

        written = save_comparison_report_bundle(
            report,
            tmp_path / "custom_report.json",
            primary_format="json",
        )

        assert written["primary"] == tmp_path / "custom_report.json"
        assert written["markdown"] == tmp_path / "custom_report.md"
        assert written["json"] == tmp_path / "custom_report.json"
        assert written["summary"] == tmp_path / "custom_report_summary.json"
        assert written["summary"].exists()

        summary = json.loads(written["summary"].read_text())
        assert summary["benchmark"] == "swe_bench"
        assert summary["winner"] == "victor"
        assert summary["results"][0]["tokens_to_merge"] == pytest.approx(120.0)

    def test_saved_result_comparison_surfaces_workspace_policy_effects(self, tmp_path):
        """Comparison summaries should preserve workspace-policy effects across saved runs."""
        path = tmp_path / "guide_workspace_result.json"
        path.write_text(
            json.dumps(
                {
                    "benchmark": "guide",
                    "model": "workspace-model",
                    "dataset_metadata": {"source_name": "GUIDE Workspace Run"},
                    "metrics": {
                        "total_tasks": 2,
                        "passed": 1,
                        "failed": 1,
                        "pass_rate": 0.5,
                    },
                    "task_results": [
                        {
                            "task_id": "guide-workspace-1",
                            "status": "passed",
                            "team_feedback_summary": {
                                "has_workspace_isolation_policy": True,
                                "workspace_policy_mode": "delegate",
                                "workspace_policy_materialize_worktrees": True,
                                "workspace_policy_dry_run_worktrees": False,
                                "workspace_policy_auto_merge_worktrees": True,
                                "has_workspace_isolation_diagnostics": True,
                                "workspace_diagnostic_count": 1,
                            },
                        },
                        {"task_id": "guide-workspace-2", "status": "failed"},
                    ],
                }
            )
        )

        report = create_comparison_report_from_saved_result(path, include_published=False)
        written = save_comparison_report_bundle(
            report,
            tmp_path / "guide_workspace_compare.json",
            primary_format="json",
        )

        summary = json.loads(written["summary"].read_text())
        metrics = json.loads(written["json"].read_text())["results"][0]["metrics"]
        markdown = written["markdown"].read_text()

        result_summary = summary["results"][0]
        assert result_summary["workspace_policy_task_coverage"] == pytest.approx(0.5)
        assert result_summary["workspace_policy_pass_rate"] == pytest.approx(1.0)
        assert result_summary["non_workspace_policy_pass_rate"] == pytest.approx(0.0)
        assert result_summary["workspace_policy_pass_delta"] == pytest.approx(1.0)
        assert result_summary["workspace_policy_materialize_rate"] == pytest.approx(0.5)
        assert result_summary["workspace_policy_auto_merge_rate"] == pytest.approx(0.5)
        assert result_summary["workspace_diagnostic_task_coverage"] == pytest.approx(0.5)
        assert metrics["workspace_policy_task_coverage"] == pytest.approx(0.5)
        assert metrics["workspace_policy_pass_delta"] == pytest.approx(1.0)
        assert "- **Workspace-Policy Coverage**: 50.0%" in markdown
        assert "- **Workspace-Policy Pass Delta**: +100.0%" in markdown

    def test_save_comparison_report_bundle_writes_fixture_manifest(self, tmp_path):
        """Comparison bundle saves should emit a fixture manifest for local artifacts."""
        first = tmp_path / "guide_result_a.json"
        second = tmp_path / "guide_result_b.json"
        first.write_text(
            json.dumps(
                {
                    "benchmark": "guide",
                    "model": "model-a",
                    "dataset_metadata": {"source_name": "GUIDE Run A"},
                    "prompt_candidate_hash": "cand-a",
                    "section_name": "GROUNDING_RULES",
                    "metrics": {
                        "total_tasks": 1,
                        "passed": 1,
                        "failed": 0,
                        "errors": 0,
                        "timeouts": 0,
                        "pass_rate": 1.0,
                    },
                    "task_results": [{"task_id": "guide-1", "status": "passed"}],
                }
            )
        )
        second.write_text(
            json.dumps(
                {
                    "benchmark": "guide",
                    "model": "model-b",
                    "dataset_metadata": {"source_name": "GUIDE Run B"},
                    "prompt_candidate_hash": "cand-b",
                    "section_name": "COMPLETION_GUIDANCE",
                    "metrics": {
                        "total_tasks": 1,
                        "passed": 0,
                        "failed": 1,
                        "errors": 0,
                        "timeouts": 0,
                        "pass_rate": 0.0,
                    },
                    "task_results": [{"task_id": "guide-2", "status": "failed"}],
                }
            )
        )

        report = create_comparison_report_from_saved_results(
            [first, second],
            include_published=False,
        )
        written = save_comparison_report_bundle(
            report,
            tmp_path / "guide_compare.json",
            primary_format="json",
        )

        fixture_manifest = json.loads(written["fixtures"].read_text())
        assert fixture_manifest["benchmark"] == "guide"
        assert fixture_manifest["artifact_count"] == 2
        assert [artifact["model"] for artifact in fixture_manifest["artifacts"]] == [
            "model-a",
            "model-b",
        ]
        assert fixture_manifest["artifacts"][0]["prompt_candidate_hash"] == "cand-a"
        assert fixture_manifest["artifacts"][1]["section_name"] == "COMPLETION_GUIDANCE"
        assert written["fixture_dir"].is_dir()
        copied_names = sorted(path.name for path in written["fixture_dir"].iterdir())
        assert copied_names == ["01_victor_model-a.json", "02_victor_model-b.json"]
        assert fixture_manifest["artifacts"][0]["bundled_artifact_path"] == (
            "guide_compare_fixtures/01_victor_model-a.json"
        )
        first_sha = hashlib.sha256(first.read_bytes()).hexdigest()
        second_sha = hashlib.sha256(second.read_bytes()).hexdigest()
        assert fixture_manifest["checksum_algorithm"] == "sha256"
        assert fixture_manifest["artifacts"][0]["artifact_sha256"] == first_sha
        assert fixture_manifest["artifacts"][0]["artifact_size_bytes"] == len(first.read_bytes())
        assert fixture_manifest["artifacts"][0]["bundled_artifact_sha256"] == first_sha
        assert fixture_manifest["artifacts"][0]["bundled_artifact_size_bytes"] == len(
            first.read_bytes()
        )
        assert fixture_manifest["artifacts"][1]["artifact_sha256"] == second_sha
        assert fixture_manifest["artifacts"][1]["bundled_artifact_sha256"] == second_sha

    def test_create_report_from_fixture_manifest_bundle(self, tmp_path):
        """Saved fixture manifests should round-trip into comparison reports."""
        first = tmp_path / "guide_result_a.json"
        second = tmp_path / "guide_result_b.json"
        first.write_text(
            json.dumps(
                {
                    "benchmark": "guide",
                    "model": "model-a",
                    "dataset_metadata": {"source_name": "GUIDE Run A"},
                    "metrics": {"total_tasks": 1, "passed": 1, "pass_rate": 1.0},
                    "task_results": [{"task_id": "guide-1", "status": "passed"}],
                }
            )
        )
        second.write_text(
            json.dumps(
                {
                    "benchmark": "guide",
                    "model": "model-b",
                    "dataset_metadata": {"source_name": "GUIDE Run B"},
                    "metrics": {"total_tasks": 1, "failed": 1, "pass_rate": 0.0},
                    "task_results": [{"task_id": "guide-2", "status": "failed"}],
                }
            )
        )

        report = create_comparison_report_from_saved_results(
            [first, second],
            include_published=False,
        )
        written = save_comparison_report_bundle(
            report,
            tmp_path / "guide_compare.json",
            primary_format="json",
        )

        reloaded = create_comparison_report_from_fixture_manifest(
            written["fixtures"],
            include_published=False,
        )

        assert reloaded.benchmark == BenchmarkType.GUIDE
        assert [result.model for result in reloaded.results] == ["model-a", "model-b"]
        assert [result.config["source"] for result in reloaded.results] == [
            "GUIDE Run A",
            "GUIDE Run B",
        ]

    def test_fixture_manifest_rejects_checksum_drift(self, tmp_path):
        """Fixture manifest loading should fail when bundled artifact bytes drift."""
        saved_result = tmp_path / "guide_result.json"
        saved_result.write_text(
            json.dumps(
                {
                    "benchmark": "guide",
                    "model": "model-a",
                    "dataset_metadata": {"source_name": "GUIDE Run A"},
                    "metrics": {"total_tasks": 1, "passed": 1, "pass_rate": 1.0},
                    "task_results": [{"task_id": "guide-1", "status": "passed"}],
                }
            )
        )

        report = create_comparison_report_from_saved_result(
            saved_result,
            include_published=False,
        )
        written = save_comparison_report_bundle(
            report,
            tmp_path / "guide_compare.json",
            primary_format="json",
        )
        bundled_path = written["fixture_dir"] / "01_victor_model-a.json"
        bundled_path.write_text('{"corrupted": true}')

        with pytest.raises(ValueError, match="integrity mismatch"):
            create_comparison_report_from_fixture_manifest(
                written["fixtures"],
                include_published=False,
            )

    def test_create_report_from_checked_in_fixture_set_directory(self):
        """Checked-in fixture set directories should load as stable comparison inputs."""
        fixture_dir = Path("tests/fixtures/benchmarks/guide_fixture_set")

        report = create_comparison_report_from_saved_results(
            [fixture_dir],
            include_published=False,
        )

        assert report.benchmark == BenchmarkType.GUIDE
        assert [result.model for result in report.results] == [
            "fixture-model-a",
            "fixture-model-b",
        ]
        assert [result.config["source"] for result in report.results] == [
            "GUIDE Fixture A",
            "GUIDE Fixture B",
        ]

    def test_discover_fixture_sets_lists_checked_in_examples(self):
        """Fixture-set discovery should enumerate checked-in benchmark examples."""
        descriptors = discover_fixture_sets(Path("tests/fixtures/benchmarks"))
        by_name = {descriptor.name: descriptor for descriptor in descriptors}

        assert "dr3_eval_fixture_set" in by_name
        assert by_name["dr3_eval_fixture_set"].benchmark == "dr3-eval"
        assert by_name["dr3_eval_fixture_set"].artifact_count == 1
        assert by_name["dr3_eval_fixture_set"].models == ("fixture-model-dr3",)
        assert "guide_fixture_set" in by_name
        assert "guide_regression_fixture_set" in by_name
        assert by_name["guide_fixture_set"].benchmark == "guide"
        assert by_name["guide_fixture_set"].artifact_count == 2
        assert by_name["guide_fixture_set"].models == (
            "fixture-model-a",
            "fixture-model-b",
        )
        assert by_name["guide_regression_fixture_set"].benchmark == "guide"
        assert by_name["guide_regression_fixture_set"].artifact_count == 1
        assert by_name["guide_regression_fixture_set"].models == ("fixture-model-c",)
        assert "swe_bench_fixture_set" in by_name
        assert by_name["swe_bench_fixture_set"].benchmark == "swe_bench"
        assert by_name["swe_bench_fixture_set"].artifact_count == 1
        assert by_name["swe_bench_fixture_set"].models == ("fixture-model-swe",)
        assert "humaneval_fixture_set" in by_name
        assert by_name["humaneval_fixture_set"].benchmark == "humaneval"
        assert by_name["humaneval_fixture_set"].artifact_count == 1
        assert by_name["humaneval_fixture_set"].models == ("fixture-model-he",)
        assert "clawbench_fixture_set" in by_name
        assert by_name["clawbench_fixture_set"].benchmark == "clawbench"
        assert by_name["clawbench_fixture_set"].artifact_count == 1
        assert by_name["clawbench_fixture_set"].models == ("fixture-model-claw",)
        assert "mbpp_fixture_set" in by_name
        assert by_name["mbpp_fixture_set"].benchmark == "mbpp"
        assert by_name["mbpp_fixture_set"].artifact_count == 1
        assert by_name["mbpp_fixture_set"].models == ("fixture-model-mbpp",)
        assert "mbpp_test_fixture_set" in by_name
        assert by_name["mbpp_test_fixture_set"].benchmark == "mbpp-test"
        assert by_name["mbpp_test_fixture_set"].artifact_count == 1
        assert by_name["mbpp_test_fixture_set"].models == ("fixture-model-mbpp-test",)
        assert "swe_bench_lite_fixture_set" in by_name
        assert by_name["swe_bench_lite_fixture_set"].benchmark == "swe-bench-lite"
        assert by_name["swe_bench_lite_fixture_set"].artifact_count == 1
        assert by_name["swe_bench_lite_fixture_set"].models == ("fixture-model-swe-lite",)
        assert "vlaa_gui_fixture_set" in by_name
        assert by_name["vlaa_gui_fixture_set"].benchmark == "vlaa-gui"
        assert by_name["vlaa_gui_fixture_set"].artifact_count == 1
        assert by_name["vlaa_gui_fixture_set"].models == ("fixture-model-vlaa",)

    def test_resolve_fixture_set_names_returns_checked_in_manifest_paths(self):
        """Fixture-set names should resolve to their checked-in manifest paths."""
        manifest_paths = resolve_fixture_set_names(
            ["guide_fixture_set", "swe_bench_fixture_set"],
            root=DEFAULT_FIXTURE_SET_ROOT,
        )

        assert manifest_paths == [
            Path("tests/fixtures/benchmarks/guide_fixture_set/comparison_report_fixtures.json"),
            Path(
                "tests/fixtures/benchmarks/swe_bench_fixture_set/" "comparison_report_fixtures.json"
            ),
        ]

    def test_resolve_fixture_set_names_rejects_unknown_names(self):
        """Fixture-set name resolution should fail with available-name guidance."""
        with pytest.raises(ValueError, match="Unknown fixture set name"):
            resolve_fixture_set_names(
                ["missing_fixture_set"],
                root=DEFAULT_FIXTURE_SET_ROOT,
            )

    def test_resolve_fixture_sets_for_benchmark_returns_all_matching_manifests(self):
        """Benchmark-scoped fixture resolution should return all checked-in matches."""
        manifest_paths = resolve_fixture_sets_for_benchmark(
            "guide",
            root=DEFAULT_FIXTURE_SET_ROOT,
        )

        assert manifest_paths == [
            Path("tests/fixtures/benchmarks/guide_fixture_set/comparison_report_fixtures.json"),
            Path(
                "tests/fixtures/benchmarks/guide_regression_fixture_set/"
                "comparison_report_fixtures.json"
            ),
        ]

    def test_resolve_fixture_sets_for_benchmark_accepts_alias_backed_names(self):
        """Fixture benchmark resolution should normalize CLI aliases and raw manifest names."""
        humaneval_paths = resolve_fixture_sets_for_benchmark(
            "human-eval",
            root=DEFAULT_FIXTURE_SET_ROOT,
        )
        clawbench_paths = resolve_fixture_sets_for_benchmark(
            "claw-bench",
            root=DEFAULT_FIXTURE_SET_ROOT,
        )

        assert humaneval_paths == [
            Path(
                "tests/fixtures/benchmarks/humaneval_fixture_set/" "comparison_report_fixtures.json"
            )
        ]
        assert clawbench_paths == [
            Path(
                "tests/fixtures/benchmarks/clawbench_fixture_set/" "comparison_report_fixtures.json"
            )
        ]

    def test_verify_fixture_sets_validates_checked_in_guide_examples(self):
        """Fixture verification should validate all checked-in guide fixture sets."""
        results = verify_fixture_sets(root=DEFAULT_FIXTURE_SET_ROOT, benchmark="guide")

        assert results == [
            FixtureSetVerificationResult(
                name="guide_fixture_set",
                benchmark="guide",
                manifest_path=Path(
                    "tests/fixtures/benchmarks/guide_fixture_set/comparison_report_fixtures.json"
                ),
                artifact_count=2,
                verified_artifact_count=2,
            ),
            FixtureSetVerificationResult(
                name="guide_regression_fixture_set",
                benchmark="guide",
                manifest_path=Path(
                    "tests/fixtures/benchmarks/guide_regression_fixture_set/"
                    "comparison_report_fixtures.json"
                ),
                artifact_count=1,
                verified_artifact_count=1,
            ),
        ]

    def test_discover_fixture_benchmarks_groups_checked_in_examples(self):
        """Fixture benchmarks should group sets into stable benchmark corpora."""
        descriptors = discover_fixture_benchmarks(DEFAULT_FIXTURE_SET_ROOT)

        assert descriptors == [
            FixtureBenchmarkDescriptor(
                benchmark="clawbench",
                fixture_set_count=1,
                artifact_count=1,
                models=("fixture-model-claw",),
                fixture_set_names=("clawbench_fixture_set",),
            ),
            FixtureBenchmarkDescriptor(
                benchmark="dr3-eval",
                fixture_set_count=1,
                artifact_count=1,
                models=("fixture-model-dr3",),
                fixture_set_names=("dr3_eval_fixture_set",),
            ),
            FixtureBenchmarkDescriptor(
                benchmark="guide",
                fixture_set_count=2,
                artifact_count=3,
                models=("fixture-model-a", "fixture-model-b", "fixture-model-c"),
                fixture_set_names=("guide_fixture_set", "guide_regression_fixture_set"),
            ),
            FixtureBenchmarkDescriptor(
                benchmark="humaneval",
                fixture_set_count=1,
                artifact_count=1,
                models=("fixture-model-he",),
                fixture_set_names=("humaneval_fixture_set",),
            ),
            FixtureBenchmarkDescriptor(
                benchmark="mbpp",
                fixture_set_count=1,
                artifact_count=1,
                models=("fixture-model-mbpp",),
                fixture_set_names=("mbpp_fixture_set",),
            ),
            FixtureBenchmarkDescriptor(
                benchmark="mbpp-test",
                fixture_set_count=1,
                artifact_count=1,
                models=("fixture-model-mbpp-test",),
                fixture_set_names=("mbpp_test_fixture_set",),
            ),
            FixtureBenchmarkDescriptor(
                benchmark="swe-bench-lite",
                fixture_set_count=1,
                artifact_count=1,
                models=("fixture-model-swe-lite",),
                fixture_set_names=("swe_bench_lite_fixture_set",),
            ),
            FixtureBenchmarkDescriptor(
                benchmark="swe_bench",
                fixture_set_count=1,
                artifact_count=1,
                models=("fixture-model-swe",),
                fixture_set_names=("swe_bench_fixture_set",),
            ),
            FixtureBenchmarkDescriptor(
                benchmark="vlaa-gui",
                fixture_set_count=1,
                artifact_count=1,
                models=("fixture-model-vlaa",),
                fixture_set_names=("vlaa_gui_fixture_set",),
            ),
        ]

    def test_fixture_benchmark_normalization_handles_aliases_and_raw_names(self):
        """Fixture benchmark helpers should normalize metadata aliases and raw manifest names."""
        assert canonicalize_fixture_benchmark_name("human_eval") == "humaneval"
        assert canonicalize_fixture_benchmark_name("claw-bench") == "clawbench"
        assert canonicalize_fixture_benchmark_name("swe_bench") == "swe-bench"
        assert fixture_benchmark_matches("humaneval", "human-eval") is True
        assert fixture_benchmark_matches("clawbench", "claw_bench") is True
        assert fixture_benchmark_matches("mbpp", "guide") is False

    def test_build_fixture_benchmark_catalog_includes_verification_summary(self):
        """Fixture benchmark catalogs should include aggregate and verification counts."""
        catalog = build_fixture_benchmark_catalog(
            root=DEFAULT_FIXTURE_SET_ROOT,
            benchmark="guide",
            verify=True,
        )

        assert catalog["root"] == "tests/fixtures/benchmarks"
        assert catalog["benchmark_count"] == 1
        assert catalog["fixture_set_count"] == 2
        assert catalog["artifact_count"] == 3
        assert catalog["verified"] is True
        assert catalog["verified_benchmark_count"] == 1
        assert catalog["catalog_benchmark_count"] == 1
        assert catalog["covered_catalog_benchmark_count"] == 1
        assert catalog["catalog_benchmark_coverage_rate"] == 1.0
        assert catalog["has_full_catalog_coverage"] is True
        assert catalog["missing_catalog_benchmarks"] == []
        assert catalog["benchmarks"] == [
            {
                "benchmark": "guide",
                "catalog_name": "guide",
                "benchmark_source_name": "Research",
                "description": "External grounded UI agent benchmark coverage.",
                "evaluation_mode": "agentic-perception",
                "runner_status": "implemented",
                "languages": ["gui", "web"],
                "categories": ["agentic", "perception", "grounding"],
                "fixture_set_count": 2,
                "artifact_count": 3,
                "models": ["fixture-model-a", "fixture-model-b", "fixture-model-c"],
                "fixture_set_names": [
                    "guide_fixture_set",
                    "guide_regression_fixture_set",
                ],
                "fixture_sources": [
                    "GUIDE Fixture A",
                    "GUIDE Fixture B",
                    "GUIDE Fixture C",
                ],
                "fixture_manifest_paths": [
                    "tests/fixtures/benchmarks/guide_fixture_set/comparison_report_fixtures.json",
                    "tests/fixtures/benchmarks/guide_regression_fixture_set/comparison_report_fixtures.json",
                ],
                "verified_fixture_set_count": 2,
                "verified_artifact_count": 3,
            }
        ]

    def test_save_fixture_benchmark_catalog_writes_json(self, tmp_path):
        """Fixture benchmark catalogs should save as machine-readable JSON."""
        output = save_fixture_benchmark_catalog(
            output_path=tmp_path / "fixture_benchmark_catalog.json",
            root=DEFAULT_FIXTURE_SET_ROOT,
            benchmark="humaneval",
            verify=True,
        )

        assert output == tmp_path / "fixture_benchmark_catalog.json"
        saved = json.loads(output.read_text())
        assert saved["benchmark_count"] == 1
        assert saved["fixture_set_count"] == 1
        assert saved["artifact_count"] == 1
        assert saved["verified"] is True
        assert saved["verified_benchmark_count"] == 1
        assert saved["catalog_benchmark_count"] == 1
        assert saved["covered_catalog_benchmark_count"] == 1
        assert saved["catalog_benchmark_coverage_rate"] == 1.0
        assert saved["has_full_catalog_coverage"] is True
        assert saved["missing_catalog_benchmarks"] == []
        assert saved["benchmarks"][0]["benchmark"] == "humaneval"
        assert saved["benchmarks"][0]["catalog_name"] == "humaneval"
        assert saved["benchmarks"][0]["benchmark_source_name"] == "OpenAI"
        assert saved["benchmarks"][0]["evaluation_mode"] == "code-generation"
        assert saved["benchmarks"][0]["languages"] == ["python"]
        assert saved["benchmarks"][0]["categories"] == ["code-generation"]
        assert saved["benchmarks"][0]["fixture_sources"] == ["HumanEval Fixture A"]
        assert saved["benchmarks"][0]["verified_artifact_count"] == 1

    def test_save_fixture_benchmark_publication_bundle_writes_combined_manifest(self, tmp_path):
        """Publication bundles should export direct-load benchmark manifests plus copied sets."""
        publication = save_fixture_benchmark_publication_bundle(
            output_path=tmp_path / "published_fixtures",
            root=DEFAULT_FIXTURE_SET_ROOT,
            benchmark="guide",
            verify=True,
        )

        assert publication["root"] == tmp_path / "published_fixtures"
        catalog_path = publication["catalog"]
        manifest_path = publication["benchmark_manifests"]["guide"]
        assert catalog_path == (
            tmp_path / "published_fixtures" / "fixture_benchmark_publication_catalog.json"
        )
        assert manifest_path == (
            tmp_path
            / "published_fixtures"
            / "guide_fixture_bundle"
            / "comparison_report_fixtures.json"
        )
        assert manifest_path.is_file()

        catalog = json.loads(catalog_path.read_text())
        assert catalog["benchmark_count"] == 1
        assert catalog["verified"] is True
        assert catalog["publication_bundle_root"] == str(tmp_path / "published_fixtures")
        assert catalog["benchmarks"][0]["published_bundle_dir"] == "guide_fixture_bundle"
        assert (
            catalog["benchmarks"][0]["published_manifest_path"]
            == "guide_fixture_bundle/comparison_report_fixtures.json"
        )
        assert catalog["benchmarks"][0]["published_fixture_set_manifest_paths"] == [
            "guide_fixture_bundle/fixture_sets/guide_fixture_set/comparison_report_fixtures.json",
            (
                "guide_fixture_bundle/fixture_sets/guide_regression_fixture_set/"
                "comparison_report_fixtures.json"
            ),
        ]

        manifest = json.loads(manifest_path.read_text())
        assert manifest["benchmark"] == "guide"
        assert manifest["artifact_count"] == 3
        assert manifest["fixture_set_count"] == 2
        assert manifest["fixture_set_names"] == [
            "guide_fixture_set",
            "guide_regression_fixture_set",
        ]
        assert manifest["fixture_sources"] == [
            "GUIDE Fixture A",
            "GUIDE Fixture B",
            "GUIDE Fixture C",
        ]
        stable_summary_path = (
            tmp_path / "published_fixtures" / "guide_fixture_bundle" / "stable_run_summary.json"
        )
        assert stable_summary_path.is_file()
        stable_summary = json.loads(stable_summary_path.read_text())
        assert stable_summary["benchmark"] == "guide"
        assert stable_summary["stable_run_artifact_count"] == 3
        assert stable_summary["best_result"]["model"] == "fixture-model-a"
        assert stable_summary["best_result"]["pass_rate"] == 1.0
        assert stable_summary["required_public_kpis"] == {
            "issue_fix_success_rate": None,
            "review_bug_catch_rate": None,
            "tokens_to_merge": 0.0,
            "time_to_first_edit_seconds": 0.0,
            "cost_per_accepted_patch_usd": 0.0,
        }
        assert stable_summary["kpi_availability"] == {
            "issue_fix_success_rate": False,
            "review_bug_catch_rate": False,
            "tokens_to_merge": False,
            "time_to_first_edit_seconds": False,
            "cost_per_accepted_patch_usd": False,
        }
        assert (
            catalog["benchmarks"][0]["stable_run_summary_path"]
            == "guide_fixture_bundle/stable_run_summary.json"
        )
        assert catalog["benchmarks"][0]["stable_run_summary"]["best_pass_rate"] == 1.0
        assert all(
            artifact["bundled_artifact_path"].startswith("fixture_sets/")
            for artifact in manifest["artifacts"]
        )
        assert all(
            "source_fixture_set" in artifact and "published_fixture_set_manifest_path" in artifact
            for artifact in manifest["artifacts"]
        )

        report = create_comparison_report_from_fixture_manifest(
            manifest_path,
            include_published=False,
        )
        assert report.benchmark == BenchmarkType.GUIDE
        assert [result.model for result in report.results] == [
            "fixture-model-a",
            "fixture-model-b",
            "fixture-model-c",
        ]

        bundle_report = create_comparison_report_from_saved_results(
            [manifest_path.parent],
            include_published=False,
        )
        assert bundle_report.benchmark == BenchmarkType.GUIDE
        assert [result.model for result in bundle_report.results] == [
            "fixture-model-a",
            "fixture-model-b",
            "fixture-model-c",
        ]

    def test_publication_stable_run_summary_reports_issue_fix_kpis(self, tmp_path):
        """SWE-style publication summaries should expose issue-fix success KPIs."""
        publication = save_fixture_benchmark_publication_bundle(
            output_path=tmp_path / "published_fixtures",
            root=DEFAULT_FIXTURE_SET_ROOT,
            benchmark="swe-bench",
            verify=True,
        )

        summary_path = publication["root"] / "swe-bench_fixture_bundle" / "stable_run_summary.json"
        summary = json.loads(summary_path.read_text())

        assert summary["benchmark"] == "swe_bench"
        assert summary["required_public_kpis"]["issue_fix_success_rate"] == 1.0
        assert summary["kpi_availability"]["issue_fix_success_rate"] is True
        assert "tokens_to_merge" in summary["required_public_kpis"]
        assert "time_to_first_edit_seconds" in summary["required_public_kpis"]
        assert "cost_per_accepted_patch_usd" in summary["required_public_kpis"]

    def test_save_stable_run_publication_bundle_marks_real_run_provenance(self, tmp_path):
        """Stable publication bundles should publish real saved runs separately from fixtures."""
        saved_result = tmp_path / "swe_real_run.json"
        saved_result.write_text(
            json.dumps(
                {
                    "benchmark": "swe-bench",
                    "model": "real-run-model",
                    "dataset_metadata": {"source_name": "SWE Real Run"},
                    "metrics": {
                        "total_tasks": 1,
                        "passed": 1,
                        "pass_rate": 1.0,
                        "accepted_patch_rate": 1.0,
                        "avg_tokens_to_merge": 4321.0,
                        "avg_time_to_first_edit_seconds": 2.5,
                        "cost_per_accepted_patch_usd": 0.42,
                    },
                    "task_results": [{"task_id": "swe-real-1", "status": "passed"}],
                }
            )
        )

        publication = save_stable_run_publication_bundle(
            output_path=tmp_path / "published_real_runs",
            result_paths=[saved_result],
        )

        catalog_path = publication["catalog"]
        manifest_path = publication["benchmark_manifests"]["swe_bench"]
        stable_summary_path = (
            tmp_path
            / "published_real_runs"
            / "swe-bench_stable_run_bundle"
            / "stable_run_summary.json"
        )

        assert catalog_path == (
            tmp_path / "published_real_runs" / "stable_run_publication_catalog.json"
        )
        assert manifest_path.is_file()
        assert stable_summary_path.is_file()

        catalog = json.loads(catalog_path.read_text())
        assert catalog["publication_kind"] == "stable_run"
        assert catalog["artifact_provenance"] == "real_run"
        assert catalog["benchmark_count"] == 1
        assert catalog["benchmarks"][0]["benchmark"] == "swe_bench"
        assert catalog["benchmarks"][0]["published_bundle_dir"] == "swe-bench_stable_run_bundle"
        assert (
            catalog["benchmarks"][0]["published_manifest_path"]
            == "swe-bench_stable_run_bundle/comparison_report_fixtures.json"
        )

        manifest = json.loads(manifest_path.read_text())
        assert manifest["publication_kind"] == "stable_run"
        assert manifest["artifact_provenance"] == "real_run"
        assert manifest["artifact_count"] == 1
        assert manifest["artifacts"][0]["source"] == "SWE Real Run"

        stable_summary = json.loads(stable_summary_path.read_text())
        assert stable_summary["publication_kind"] == "stable_run"
        assert stable_summary["artifact_provenance"] == "real_run"
        assert stable_summary["corpus_readiness"] == {
            "publishable": True,
            "artifact_count": 1,
            "task_count": 1,
            "source_count": 1,
            "model_count": 1,
            "required_public_kpi_complete": True,
            "missing_reasons": [],
        }
        assert stable_summary["required_public_kpis"] == {
            "issue_fix_success_rate": 1.0,
            "review_bug_catch_rate": None,
            "tokens_to_merge": 4321.0,
            "time_to_first_edit_seconds": 2.5,
            "cost_per_accepted_patch_usd": 0.42,
        }
        assert stable_summary["required_public_kpi_complete"] is True
        assert stable_summary["missing_public_kpis"] == []
        assert (
            catalog["benchmarks"][0]["stable_run_summary"]["required_public_kpi_complete"] is True
        )
        assert (
            catalog["benchmarks"][0]["stable_run_summary"]["corpus_readiness"]["publishable"]
            is True
        )

    def test_stable_run_publication_reports_missing_public_kpis(self, tmp_path):
        """Real-run summaries should make missing public benchmark KPIs explicit."""
        saved_result = tmp_path / "swe_real_run_missing_kpis.json"
        saved_result.write_text(
            json.dumps(
                {
                    "benchmark": "swe-bench",
                    "model": "real-run-model",
                    "dataset_metadata": {"source_name": "SWE Real Run"},
                    "metrics": {
                        "total_tasks": 1,
                        "passed": 1,
                        "pass_rate": 1.0,
                        "avg_tokens_to_merge": 4321.0,
                    },
                    "task_results": [{"task_id": "swe-real-1", "status": "passed"}],
                }
            )
        )

        publication = save_stable_run_publication_bundle(
            output_path=tmp_path / "published_real_runs",
            result_paths=[saved_result],
        )

        catalog = json.loads(publication["catalog"].read_text())
        summary_path = publication["root"] / catalog["benchmarks"][0]["stable_run_summary_path"]
        summary = json.loads(summary_path.read_text())

        assert summary["required_public_kpi_complete"] is False
        assert summary["missing_public_kpis"] == [
            "time_to_first_edit_seconds",
            "cost_per_accepted_patch_usd",
        ]
        assert summary["corpus_readiness"] == {
            "publishable": False,
            "artifact_count": 1,
            "task_count": 1,
            "source_count": 1,
            "model_count": 1,
            "required_public_kpi_complete": False,
            "missing_reasons": ["missing_public_kpis"],
        }
        assert (
            catalog["benchmarks"][0]["stable_run_summary"]["required_public_kpi_complete"] is False
        )
        assert catalog["benchmarks"][0]["stable_run_summary"]["missing_public_kpis"] == [
            "time_to_first_edit_seconds",
            "cost_per_accepted_patch_usd",
        ]

    def test_stable_run_publication_rejects_fixture_artifacts_by_default(self, tmp_path):
        """Stable real-run publication must not silently relabel fixture corpora."""
        fixture_result = (
            DEFAULT_FIXTURE_SET_ROOT
            / "swe_bench_fixture_set"
            / "comparison_report_fixtures"
            / "01_victor_fixture-model-swe.json"
        )

        with pytest.raises(ValueError, match="non-fixture saved artifacts"):
            save_stable_run_publication_bundle(
                output_path=tmp_path / "published_real_runs",
                result_paths=[fixture_result],
            )

    def test_resolve_fixture_benchmark_publication_manifests_accepts_root_and_catalog(
        self, tmp_path
    ):
        """Published fixture bundles should resolve through either the bundle root or catalog."""
        publication = save_fixture_benchmark_publication_bundle(
            output_path=tmp_path / "published_fixtures",
            root=DEFAULT_FIXTURE_SET_ROOT,
            benchmark="guide",
            verify=True,
        )

        expected_manifest = publication["benchmark_manifests"]["guide"]

        root_paths = resolve_fixture_benchmark_publication_manifests(
            root=publication["root"],
            benchmark="guide",
        )
        catalog_paths = resolve_fixture_benchmark_publication_manifests(
            root=publication["catalog"],
            benchmark="guide",
        )

        assert root_paths == [expected_manifest]
        assert catalog_paths == [expected_manifest]

    def test_create_report_from_fixture_benchmark_publication_bundle_root(self, tmp_path):
        """Publication bundle roots and catalogs should round-trip into comparison reports."""
        publication = save_fixture_benchmark_publication_bundle(
            output_path=tmp_path / "published_fixtures",
            root=DEFAULT_FIXTURE_SET_ROOT,
            benchmark="guide",
            verify=True,
        )

        for publication_input in (publication["root"], publication["catalog"]):
            report = create_comparison_report_from_saved_results(
                [publication_input],
                include_published=False,
            )

            assert report.benchmark == BenchmarkType.GUIDE
            assert [result.model for result in report.results] == [
                "fixture-model-a",
                "fixture-model-b",
                "fixture-model-c",
            ]

    def test_build_fixture_benchmark_catalog_reports_full_catalog_coverage(self):
        """Unfiltered catalogs should report complete fixture coverage for the benchmark catalog."""
        catalog = build_fixture_benchmark_catalog(
            root=DEFAULT_FIXTURE_SET_ROOT,
            verify=False,
        )

        assert catalog["catalog_benchmark_count"] == 9
        assert catalog["covered_catalog_benchmark_count"] == 9
        assert catalog["catalog_benchmark_coverage_rate"] == 1.0
        assert catalog["has_full_catalog_coverage"] is True
        assert catalog["missing_catalog_benchmarks"] == []
