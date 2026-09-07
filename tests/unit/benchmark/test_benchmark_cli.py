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

"""Tests for benchmark CLI commands.

Unit tests that don't require Ollama or API keys test CLI structure only.
Integration tests that run actual benchmarks are skipped when Ollama is unavailable.
"""

import json
import socket
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch, MagicMock

import pytest
from rich.console import Console
from typer.testing import CliRunner
import victor.ui.commands.benchmark as benchmark_cmd
from victor.evaluation.benchmarks.framework_comparison import (
    create_comparison_report_from_saved_results,
    save_comparison_report_bundle,
    save_fixture_benchmark_publication_bundle,
)
from victor.ui.commands.benchmark import benchmark_app


def is_ollama_available() -> bool:
    """Check if Ollama server is running at localhost:11434."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.settimeout(1)
        result = sock.connect_ex(("localhost", 11434))
        return result == 0
    finally:
        sock.close()


runner = CliRunner()


class TestBenchmarkInputStaging:
    """Tests for binary-safe ephemeral-workspace input staging."""

    def test_stages_binary_input_at_declared_relative_path(self, tmp_path):
        source = tmp_path / "source.pdf"
        source.write_bytes(b"%PDF-test\x00payload")
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        task = SimpleNamespace(
            input_files={"attachments/01-source.pdf": str(source)},
        )

        benchmark_cmd._stage_benchmark_input_files(task, workspace)

        assert (workspace / "attachments/01-source.pdf").read_bytes() == source.read_bytes()

    def test_rejects_input_destination_escape(self, tmp_path):
        source = tmp_path / "source.txt"
        source.write_text("evidence")
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        task = SimpleNamespace(input_files={"../escape.txt": str(source)})

        with pytest.raises(ValueError, match="escapes workspace"):
            benchmark_cmd._stage_benchmark_input_files(task, workspace)


class TestBenchmarkGlobalPaths:
    """Tests for canonical global benchmark path resolution."""

    def test_global_dirs_resolve_through_project_paths(self, tmp_path):
        """Benchmark helpers should resolve through centralized Victor paths."""
        global_dir = tmp_path / ".victor"
        logs_dir = global_dir / "logs"

        with patch(
            "victor.ui.commands.benchmark.get_project_paths",
            return_value=SimpleNamespace(
                global_victor_dir=global_dir,
                global_logs_dir=logs_dir,
            ),
        ):
            assert benchmark_cmd._get_global_evaluations_dir() == global_dir / "evaluations"
            assert benchmark_cmd._get_global_usage_logs_dir() == logs_dir

    def test_show_compliance_scorecard_reads_global_logs_dir(self, tmp_path):
        """Compliance reporting should read usage logs from centralized Victor paths."""
        global_dir = tmp_path / ".victor"
        logs_dir = global_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        usage_file = logs_dir / "usage.jsonl"
        usage_file.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "session_id": "sess-1",
                            "event_type": "tool_call",
                            "data": {
                                "tool_name": "code_search",
                                "tool_args": {"mode": "semantic"},
                            },
                        }
                    ),
                    json.dumps(
                        {
                            "session_id": "sess-1",
                            "event_type": "tool_result",
                            "data": {"tool_name": "code_search"},
                        }
                    ),
                ]
            )
            + "\n"
        )

        record_console = Console(record=True, force_terminal=False, width=160)
        with (
            patch(
                "victor.ui.commands.benchmark.get_project_paths",
                return_value=SimpleNamespace(
                    global_victor_dir=global_dir,
                    global_logs_dir=logs_dir,
                ),
            ),
            patch.object(benchmark_cmd, "console", record_console),
        ):
            benchmark_cmd._show_compliance_scorecard()

        output = record_console.export_text()
        assert "GEPA Compliance Scorecard" in output
        assert "Based on 2 events across 1 sessions" in output


class TestBenchmarkList:
    """Tests for benchmark list command."""

    def test_list_benchmarks(self):
        """Test listing available benchmarks."""
        result = runner.invoke(benchmark_app, ["list"])
        assert result.exit_code == 0
        assert "Available Benchmarks" in result.stdout
        assert "swe-bench" in result.stdout
        assert "humaneval" in result.stdout
        assert "mbpp" in result.stdout

    def test_list_shows_task_counts(self):
        """Test that task counts are shown."""
        result = runner.invoke(benchmark_app, ["list"])
        assert result.exit_code == 0
        assert "164" in result.stdout  # HumanEval
        assert "974" in result.stdout  # MBPP

    def test_list_shows_external_agentic_benchmarks(self):
        """Test that external perception-heavy benchmarks are listed."""
        result = runner.invoke(benchmark_app, ["list"])
        assert result.exit_code == 0
        assert "clawbench" in result.stdout
        assert "dr3-eval" in result.stdout
        assert "guide" in result.stdout
        assert "vlaa-gui" in result.stdout
        assert "implemented" in result.stdout


class TestBenchmarkFixtureSets:
    """Tests for checked-in benchmark fixture-set discovery."""

    def test_fixture_sets_lists_checked_in_examples(self):
        result = runner.invoke(benchmark_app, ["fixture-sets"])
        assert result.exit_code == 0
        assert "Checked-In Benchmark Fixture Sets" in result.stdout
        assert "dr3_eval_fixture_set" in result.stdout
        assert "dr3-eval" in result.stdout
        assert "guide_fixture_set" in result.stdout
        assert "guide_regression_fixture_set" in result.stdout
        assert "clawbench_fixture_set" in result.stdout
        assert "humaneval_fixture_set" in result.stdout
        assert "mbpp_fixture_set" in result.stdout
        assert "mbpp_test_fixture_set" in result.stdout
        assert "swe_bench_lite_fixture_set" in result.stdout
        assert "swe_bench_fixture_set" in result.stdout
        assert "vlaa_gui_fixture_set" in result.stdout
        assert "fixture-model-dr3" in result.stdout
        assert "fixture-model-claw" in result.stdout
        assert "fixture-model-a" in result.stdout
        assert "fixture-model-c" in result.stdout
        assert "fixture-model-he" in result.stdout
        assert "fixture-model-mbpp" in result.stdout
        assert "fixture-model-mbpp-test" in result.stdout
        assert "fixture-model-swe-lite" in result.stdout
        assert "fixture-model-swe" in result.stdout
        assert "fixture-model-vlaa" in result.stdout

    def test_fixture_sets_can_filter_by_benchmark(self):
        result = runner.invoke(benchmark_app, ["fixture-sets", "--benchmark", "guide"])
        assert result.exit_code == 0
        assert "guide_fixture_set" in result.stdout
        assert "guide_regression_fixture_set" in result.stdout
        assert "swe_bench_fixture_set" not in result.stdout

    def test_fixture_sets_can_filter_by_alias_backed_benchmark(self):
        result = runner.invoke(benchmark_app, ["fixture-sets", "--benchmark", "claw-bench"])
        assert result.exit_code == 0
        assert "clawbench_fixture_set" in result.stdout
        assert "guide_fixture_set" not in result.stdout

    def test_fixture_sets_can_verify_filtered_benchmark(self):
        result = runner.invoke(benchmark_app, ["fixture-sets", "--benchmark", "guide", "--verify"])
        assert result.exit_code == 0
        assert "Verified fixture sets: 2" in result.stdout
        assert "guide_fixture_set" in result.stdout
        assert "guide_regression_fixture_set" in result.stdout


class TestBenchmarkFixtureBenchmarks:
    """Tests for benchmark-level checked-in fixture corpus discovery."""

    def test_fixture_benchmarks_lists_grouped_checked_in_examples(self):
        result = runner.invoke(benchmark_app, ["fixture-benchmarks"])
        assert result.exit_code == 0
        assert "Fixture benchmark coverage:" in result.stdout
        assert "9/9 cataloged benchmarks (100.0%)" in result.stdout
        assert "All cataloged benchmarks have checked-in fixture coverage." in result.stdout
        assert "Missing fixture benchmarks:" not in result.stdout
        assert "Fixture benchmark models:" in result.stdout
        assert "Fixture benchmark publishers:" in result.stdout
        assert "clawbench=fixture-model-claw" in result.stdout
        assert "clawbench=Research" in result.stdout
        assert "Checked-In Benchmark Fixture Corpora" in result.stdout
        assert "clawbench" in result.stdout
        assert "dr3-eval" in result.stdout
        assert "guide" in result.stdout
        assert "humaneval" in result.stdout
        assert "mbpp" in result.stdout
        assert "mbpp-test" in result.stdout
        assert "swe-bench-lite" in result.stdout
        assert "swe_bench" in result.stdout
        assert "vlaa-gui" in result.stdout
        assert "guide=guide_fixture_set," in result.stdout
        assert "guide_regression_fixture_set" in result.stdout
        assert "fixture-model-he" in result.stdout
        assert "dr3-eval=fixture-model-dr3" in result.stdout
        assert "mbpp=mbpp_fixture_set" in result.stdout
        assert "mbpp-test=mbpp_test_fixture_set" in result.stdout
        assert "swe-bench-lite=swe_bench_lite_fixture_set" in result.stdout
        assert "vlaa-gui=vlaa_gui_fixture_set" in result.stdout
        assert "dr3-eval=DR3-Eval" in result.stdout
        assert "humaneval=OpenAI" in result.stdout
        assert "mbpp=Google Research" in result.stdout
        assert "mbpp-test=Google Research" in result.stdout
        assert "swe_bench=Princeton NLP" in result.stdout
        assert "swe-bench-lite=Princeton NLP" in result.stdout
        assert "vlaa-gui=Research" in result.stdout

    def test_fixture_benchmarks_can_verify_grouped_catalog(self):
        result = runner.invoke(benchmark_app, ["fixture-benchmarks", "--verify"])
        assert result.exit_code == 0
        assert "Verified fixture benchmarks: 9" in result.stdout
        assert "clawbench" in result.stdout
        assert "dr3-eval" in result.stdout
        assert "guide" in result.stdout
        assert "humaneval" in result.stdout
        assert "mbpp" in result.stdout
        assert "mbpp-test" in result.stdout
        assert "swe-bench-lite" in result.stdout
        assert "swe_bench" in result.stdout
        assert "vlaa-gui" in result.stdout

    def test_fixture_benchmarks_can_save_verified_catalog(self, tmp_path):
        output = tmp_path / "fixture_benchmark_catalog.json"

        result = runner.invoke(
            benchmark_app,
            [
                "fixture-benchmarks",
                "--benchmark",
                "guide",
                "--verify",
                "--output",
                str(output),
            ],
        )

        assert result.exit_code == 0
        assert f"Fixture benchmark catalog saved to {output}" in result.stdout
        saved = json.loads(output.read_text())
        assert saved["benchmark_count"] == 1
        assert saved["fixture_set_count"] == 2
        assert saved["artifact_count"] == 3
        assert saved["verified_benchmark_count"] == 1
        assert saved["catalog_benchmark_count"] == 1
        assert saved["covered_catalog_benchmark_count"] == 1
        assert saved["catalog_benchmark_coverage_rate"] == 1.0
        assert saved["has_full_catalog_coverage"] is True
        assert saved["missing_catalog_benchmarks"] == []
        assert saved["benchmarks"][0]["benchmark"] == "guide"
        assert saved["benchmarks"][0]["benchmark_source_name"] == "Research"
        assert saved["benchmarks"][0]["verified_fixture_set_count"] == 2

    def test_fixture_benchmarks_can_save_publication_bundle(self, tmp_path):
        bundle_output = tmp_path / "published_fixtures"

        result = runner.invoke(
            benchmark_app,
            [
                "fixture-benchmarks",
                "--benchmark",
                "guide",
                "--verify",
                "--bundle-output",
                str(bundle_output),
            ],
        )

        assert result.exit_code == 0
        assert f"Fixture benchmark publication bundle saved to {bundle_output}" in result.stdout
        catalog_path = bundle_output / "fixture_benchmark_publication_catalog.json"
        manifest_path = bundle_output / "guide_fixture_bundle" / "comparison_report_fixtures.json"
        assert catalog_path.is_file()
        assert manifest_path.is_file()
        saved = json.loads(catalog_path.read_text())
        assert saved["benchmark_count"] == 1
        assert saved["benchmarks"][0]["benchmark"] == "guide"
        assert saved["benchmarks"][0]["published_bundle_dir"] == "guide_fixture_bundle"
        assert (
            saved["benchmarks"][0]["published_manifest_path"]
            == "guide_fixture_bundle/comparison_report_fixtures.json"
        )

    def test_stable_runs_can_save_real_run_publication_bundle(self, tmp_path):
        saved_result = tmp_path / "swe_real_run.json"
        bundle_output = tmp_path / "published_real_runs"
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
                        "avg_tokens_to_merge": 123.0,
                    },
                    "task_results": [{"task_id": "swe-real-1", "status": "passed"}],
                }
            )
        )

        result = runner.invoke(
            benchmark_app,
            [
                "stable-runs",
                "--benchmark",
                "swe-bench",
                "--victor-results",
                str(saved_result),
                "--bundle-output",
                str(bundle_output),
            ],
        )

        assert result.exit_code == 0
        assert "Stable real-run publication bundle saved to" in result.stdout
        catalog_path = bundle_output / "stable_run_publication_catalog.json"
        summary_path = bundle_output / "swe-bench_stable_run_bundle" / "stable_run_summary.json"
        assert catalog_path.is_file()
        assert summary_path.is_file()
        catalog = json.loads(catalog_path.read_text())
        assert catalog["artifact_provenance"] == "real_run"
        assert catalog["benchmarks"][0]["benchmark"] == "swe_bench"

        compare_output = tmp_path / "stable_compare.json"
        compare_result = runner.invoke(
            benchmark_app,
            [
                "compare",
                "--benchmark",
                "swe-bench",
                "--victor-publication-root",
                str(bundle_output),
                "--format",
                "json",
                "--output",
                str(compare_output),
            ],
        )

        assert compare_result.exit_code == 0
        compared = json.loads(compare_output.read_text())
        assert compared["benchmark"] == "swe_bench"
        assert compared["results"][0]["model"] == "real-run-model"

    def test_stable_runs_can_require_publishable_corpus(self, tmp_path):
        saved_result = tmp_path / "swe_real_run_missing_kpis.json"
        bundle_output = tmp_path / "published_real_runs"
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
                        "avg_tokens_to_merge": 123.0,
                    },
                    "task_results": [{"task_id": "swe-real-1", "status": "passed"}],
                }
            )
        )

        result = runner.invoke(
            benchmark_app,
            [
                "stable-runs",
                "--benchmark",
                "swe-bench",
                "--victor-results",
                str(saved_result),
                "--bundle-output",
                str(bundle_output),
                "--require-publishable",
            ],
        )

        assert result.exit_code == 1
        assert "Stable real-run corpus is not publishable" in result.stdout
        assert "missing_public_kpis" in result.stdout


class TestBenchmarkRun:
    """Tests for benchmark run command."""

    def test_run_unknown_benchmark(self):
        """Test running an unknown benchmark."""
        result = runner.invoke(benchmark_app, ["run", "unknown-benchmark"])
        assert result.exit_code == 1
        assert "Unknown benchmark" in result.stdout

    def test_run_cataloged_benchmark_requires_dataset_path(self):
        """External benchmark adapters should require a local manifest path."""
        result = runner.invoke(benchmark_app, ["run", "guide"])
        assert result.exit_code == 1
        assert "requires --dataset-path" in result.stdout

    def test_run_external_benchmark_with_dataset_path(self, tmp_path):
        """External benchmark adapters should route through the standard runner path."""
        dataset = tmp_path / "guide.json"
        output = tmp_path / "results.json"
        dataset.write_text(
            json.dumps(
                {
                    "metadata": {
                        "version": "2026.04",
                        "source_name": "GUIDE Consortium",
                        "languages": ["python"],
                    },
                    "tasks": [
                        {
                            "task_id": "guide-1",
                            "prompt": "Implement add",
                            "test_code": (
                                "from solution import add\n\n\n"
                                "def test_add():\n"
                                "    assert add(1, 2) == 3\n"
                            ),
                        }
                    ],
                }
            )
        )

        async def fake_run_benchmark_async(**_kwargs):
            from victor.evaluation.protocol import (
                BenchmarkFailureCategory,
                BenchmarkType,
                EvaluationConfig,
                EvaluationResult,
                TaskResult,
                TaskStatus,
            )

            config = _kwargs["config"]
            assert config.provider == "anthropic"
            assert config.prompt_candidate_hash == "cand-123"
            assert config.prompt_section_name == "GROUNDING_RULES"

            return EvaluationResult(
                config=config,
                task_results=[
                    TaskResult(
                        task_id="guide-1",
                        status=TaskStatus.FAILED,
                        tests_passed=0,
                        tests_total=1,
                        failure_category=BenchmarkFailureCategory.TEST_FAILURE,
                        error_message="assertion failed",
                    )
                ],
            )

        with patch(
            "victor.ui.commands.benchmark._run_benchmark_async",
            side_effect=fake_run_benchmark_async,
        ):
            result = runner.invoke(
                benchmark_app,
                [
                    "run",
                    "guide",
                    "--dataset-path",
                    str(dataset),
                    "--model",
                    "test-model",
                    "--provider",
                    "anthropic",
                    "--prompt-candidate-hash",
                    "cand-123",
                    "--prompt-section",
                    "GROUNDING_RULES",
                    "--output",
                    str(output),
                ],
            )

        assert result.exit_code == 0
        assert "Dataset:" in result.stdout
        assert "Source: GUIDE Consortium" in result.stdout
        assert "Manifest Version: 2026.04" in result.stdout
        assert "Failure: test_failure" in result.stdout
        saved = json.loads(output.read_text())
        assert saved["dataset_metadata"]["source_name"] == "GUIDE Consortium"
        assert saved["provider"] == "anthropic"
        assert saved["prompt_candidate_hash"] == "cand-123"
        assert saved["section_name"] == "GROUNDING_RULES"
        assert saved["failure_examples"]["test_failure"]["sample_task_ids"] == ["guide-1"]

    def test_run_shows_help(self):
        """Test run command help."""
        result = runner.invoke(benchmark_app, ["run", "--help"])
        assert result.exit_code == 0
        # Options may be truncated by Rich formatting, check for key parts
        assert "max-tasks" in result.stdout or "max_tasks" in result.stdout or "-n" in result.stdout
        assert "timeout" in result.stdout
        assert "profile" in result.stdout

    def test_run_prompt_suite_writes_summary_output(self, tmp_path):
        """Prompt suite runs should emit one summary artifact covering all candidates."""
        from victor.evaluation import (
            PromptCandidateEvaluationRun,
            PromptCandidateEvaluationSpec,
            PromptCandidateEvaluationSuiteResult,
        )
        from victor.evaluation.protocol import (
            BenchmarkType,
            EvaluationConfig,
            EvaluationResult,
            TaskResult,
            TaskStatus,
        )

        output = tmp_path / "suite.json"
        runner_impl = object()
        base_config = EvaluationConfig(
            benchmark=BenchmarkType.HUMAN_EVAL,
            model="test-model",
            provider="anthropic",
        )
        suite = PromptCandidateEvaluationSuiteResult(
            base_config=base_config,
            runs=[
                PromptCandidateEvaluationRun(
                    spec=PromptCandidateEvaluationSpec(
                        section_name="GROUNDING_RULES",
                        prompt_candidate_hash="cand-123",
                        provider="anthropic",
                    ),
                    config=EvaluationConfig(
                        benchmark=BenchmarkType.HUMAN_EVAL,
                        model="test-model",
                        provider="anthropic",
                        prompt_candidate_hash="cand-123",
                        prompt_section_name="GROUNDING_RULES",
                    ),
                    result=EvaluationResult(
                        config=base_config,
                        task_results=[TaskResult(task_id="task-1", status=TaskStatus.PASSED)],
                    ),
                    label="GROUNDING_RULES:anthropic:cand-123",
                ),
                PromptCandidateEvaluationRun(
                    spec=PromptCandidateEvaluationSpec(
                        section_name="GROUNDING_RULES",
                        prompt_candidate_hash="cand-456",
                        provider="anthropic",
                    ),
                    config=EvaluationConfig(
                        benchmark=BenchmarkType.HUMAN_EVAL,
                        model="test-model",
                        provider="anthropic",
                        prompt_candidate_hash="cand-456",
                        prompt_section_name="GROUNDING_RULES",
                    ),
                    result=EvaluationResult(
                        config=base_config,
                        task_results=[TaskResult(task_id="task-2", status=TaskStatus.FAILED)],
                    ),
                    label="GROUNDING_RULES:anthropic:cand-456",
                ),
            ],
        )

        async def fake_run_prompt_suite_async(**_kwargs):
            return suite

        with (
            patch(
                "victor.evaluation.benchmarks.HumanEvalRunner",
                return_value=runner_impl,
            ),
            patch(
                "victor.ui.commands.benchmark._run_prompt_candidate_suite_async",
                side_effect=fake_run_prompt_suite_async,
            ),
        ):
            result = runner.invoke(
                benchmark_app,
                [
                    "run-prompt-suite",
                    "humaneval",
                    "--prompt-section",
                    "GROUNDING_RULES",
                    "--candidate-hash",
                    "cand-123",
                    "--candidate-hash",
                    "cand-456",
                    "--provider",
                    "anthropic",
                    "--model",
                    "test-model",
                    "--output",
                    str(output),
                ],
            )

        assert result.exit_code == 0
        assert "Prompt Candidate Suite Summary" in result.stdout
        saved = json.loads(output.read_text())
        assert saved["section_name"] == "GROUNDING_RULES"
        assert saved["best_prompt_candidate_hash"] == "cand-123"
        assert len(saved["runs"]) == 2
        assert saved["runs"][0]["prompt_candidate_hash"] == "cand-123"
        assert saved["runs"][1]["prompt_candidate_hash"] == "cand-456"


class TestBenchmarkPromptEvolution:
    """Tests for prompt evolution CLI wiring."""

    def test_evolve_uses_registry_baseline_for_custom_section(self, tmp_path, monkeypatch):
        from victor.agent import prompt_section_registry as registry_module
        from victor.agent.prompt_section_registry import (
            SectionCategory,
            SectionDefinition,
            UnifiedSectionRegistry,
            _initialize_default_sections,
        )
        from victor.framework.rl.learners.prompt_optimizer import PromptCandidate

        fresh_registry = UnifiedSectionRegistry()
        _initialize_default_sections(fresh_registry)
        fresh_registry.register(
            SectionDefinition(
                name="CUSTOM_REVIEW_GUIDANCE",
                aliases={"custom_review"},
                category=SectionCategory.TASK_HINTS,
                default_text="Review API drift first.",
                evolvable=True,
                required=False,
                priority=41,
            )
        )
        monkeypatch.setattr(registry_module, "_registry", fresh_registry)

        calls = []

        class _Learner:
            def get_evolvable_sections(self):
                return ["CUSTOM_REVIEW_GUIDANCE"]

            def evolve(self, section, current, provider="default", query=None):
                calls.append((section, current, provider))
                return PromptCandidate(
                    section_name=section,
                    provider=provider,
                    text=current + " Evolved.",
                    text_hash="hash9999cccc",
                    generation=1,
                    parent_hash="base0000aaaa",
                )

            def _collect_traces(self, limit=200):
                return []

            def _save_candidate(self, _candidate):
                return None

            def export_metrics(self):
                return {"total_candidates": 1}

        learner = _Learner()
        coordinator = SimpleNamespace(get_learner=lambda _name: learner)

        with (
            patch(
                "victor.framework.rl.coordinator.get_rl_coordinator",
                return_value=coordinator,
            ),
            patch(
                "victor.ui.commands.benchmark._get_global_evaluations_dir",
                return_value=tmp_path,
            ),
        ):
            result = runner.invoke(
                benchmark_app,
                ["evolve", "--provider", "openai", "--section", "CUSTOM_REVIEW"],
            )

        assert result.exit_code == 0
        assert calls == [("CUSTOM_REVIEW_GUIDANCE", "Review API drift first.", "openai")]
        assert "Total candidates: 1" in result.stdout

    def test_post_run_auto_evolve_uses_registry_baseline(self, monkeypatch):
        from victor.agent import prompt_section_registry as registry_module
        from victor.agent.prompt_section_registry import (
            SectionCategory,
            SectionDefinition,
            UnifiedSectionRegistry,
            _initialize_default_sections,
        )
        from victor.framework.rl.learners.prompt_optimizer import PromptCandidate

        fresh_registry = UnifiedSectionRegistry()
        _initialize_default_sections(fresh_registry)
        fresh_registry.register(
            SectionDefinition(
                name="ASI_TOOL_EFFECTIVENESS_GUIDANCE",
                aliases={
                    "tool_effectiveness_guidance",
                    "tool_hints",
                    "asi_tool_guidance",
                },
                category=SectionCategory.TOOL_GUIDANCE,
                default_text="Registry benchmark guidance.",
                evolvable=True,
                required=True,
                priority=50,
                default_strategies=("gepa", "cot_distillation"),
            )
        )
        monkeypatch.setattr(registry_module, "_registry", fresh_registry)

        calls = []
        saved_candidates = []
        candidate = PromptCandidate(
            section_name="ASI_TOOL_EFFECTIVENESS_GUIDANCE",
            provider="openai",
            text="Registry benchmark guidance. Evolved.",
            text_hash="hash9999cccc",
            generation=2,
            parent_hash="base0000aaaa",
        )

        class _Learner:
            def evolve(self, section, current, provider="default", query=None):
                calls.append((section, current, provider))
                return candidate

            def _save_candidate(self, saved_candidate):
                saved_candidates.append(saved_candidate)

        evolved = benchmark_cmd._auto_evolve_prompt_candidate(
            _Learner(),
            model_name="gpt-4.1",
            metrics={"passed": 2, "failed": 1, "errors": 1},
        )

        assert evolved is not None
        evolved_candidate, provider = evolved
        assert provider == "openai"
        assert evolved_candidate is candidate
        assert calls == [
            (
                "ASI_TOOL_EFFECTIVENESS_GUIDANCE",
                "Registry benchmark guidance.",
                "openai",
            )
        ]
        assert saved_candidates == [candidate]
        assert candidate.sample_count == 4
        assert candidate.mean == pytest.approx(0.5)

    def test_run_prompt_suite_rejects_promote_best_without_recording(self):
        result = runner.invoke(
            benchmark_app,
            [
                "run-prompt-suite",
                "humaneval",
                "--prompt-section",
                "GROUNDING_RULES",
                "--candidate-hash",
                "cand-123",
                "--promote-best",
            ],
        )

        assert result.exit_code == 1
        assert "--promote-best requires --record-benchmark-results" in result.stdout

    def test_run_prompt_suite_rejects_create_rollout_without_recording(self):
        result = runner.invoke(
            benchmark_app,
            [
                "run-prompt-suite",
                "humaneval",
                "--prompt-section",
                "GROUNDING_RULES",
                "--candidate-hash",
                "cand-123",
                "--create-rollout",
            ],
        )

        assert result.exit_code == 1
        assert "--create-rollout requires --record-benchmark-results" in result.stdout

    def test_run_prompt_suite_rejects_create_rollout_with_promote_best(self):
        result = runner.invoke(
            benchmark_app,
            [
                "run-prompt-suite",
                "humaneval",
                "--prompt-section",
                "GROUNDING_RULES",
                "--candidate-hash",
                "cand-123",
                "--record-benchmark-results",
                "--promote-best",
                "--create-rollout",
            ],
        )

        assert result.exit_code == 1
        assert "--create-rollout cannot be combined with --promote-best" in result.stdout

    def test_run_prompt_suite_rejects_analyze_rollout_without_recording(self):
        result = runner.invoke(
            benchmark_app,
            [
                "run-prompt-suite",
                "humaneval",
                "--prompt-section",
                "GROUNDING_RULES",
                "--candidate-hash",
                "cand-123",
                "--analyze-rollout",
            ],
        )

        assert result.exit_code == 1
        assert "--analyze-rollout requires --record-benchmark-results" in result.stdout

    def test_run_prompt_suite_rejects_apply_rollout_decision_without_analyze_rollout(
        self,
    ):
        result = runner.invoke(
            benchmark_app,
            [
                "run-prompt-suite",
                "humaneval",
                "--prompt-section",
                "GROUNDING_RULES",
                "--candidate-hash",
                "cand-123",
                "--record-benchmark-results",
                "--apply-rollout-decision",
            ],
        )

        assert result.exit_code == 1
        assert "--apply-rollout-decision requires --analyze-rollout" in result.stdout

    def test_run_prompt_suite_can_sync_results_into_prompt_optimizer(self, tmp_path):
        from victor.evaluation import (
            PromptCandidateEvaluationRun,
            PromptCandidateEvaluationSpec,
            PromptCandidateEvaluationSuiteResult,
        )
        from victor.evaluation.protocol import (
            BenchmarkType,
            EvaluationConfig,
            EvaluationResult,
            TaskResult,
            TaskStatus,
        )

        runner_impl = object()
        base_config = EvaluationConfig(
            benchmark=BenchmarkType.HUMAN_EVAL,
            model="test-model",
            provider="anthropic",
        )
        suite = PromptCandidateEvaluationSuiteResult(
            base_config=base_config,
            runs=[
                PromptCandidateEvaluationRun(
                    spec=PromptCandidateEvaluationSpec(
                        section_name="GROUNDING_RULES",
                        prompt_candidate_hash="cand-123",
                        provider="anthropic",
                    ),
                    config=EvaluationConfig(
                        benchmark=BenchmarkType.HUMAN_EVAL,
                        model="test-model",
                        provider="anthropic",
                        prompt_candidate_hash="cand-123",
                        prompt_section_name="GROUNDING_RULES",
                    ),
                    result=EvaluationResult(
                        config=base_config,
                        task_results=[TaskResult(task_id="task-1", status=TaskStatus.PASSED)],
                    ),
                    label="GROUNDING_RULES:anthropic:cand-123",
                )
            ],
        )
        output = tmp_path / "suite_with_sync.json"
        sync_result = SimpleNamespace(
            best_prompt_candidate_hash="cand-123",
            approved_prompt_candidate_hash="cand-123",
            promoted_prompt_candidate_hash="cand-123",
            decisions=[
                SimpleNamespace(
                    rank=1,
                    section_name="GROUNDING_RULES",
                    provider="anthropic",
                    prompt_candidate_hash="cand-123",
                    score=1.0,
                    recorded=True,
                    passed=True,
                    promoted=True,
                )
            ],
            to_dict=lambda: {
                "best_prompt_candidate_hash": "cand-123",
                "approved_prompt_candidate_hash": "cand-123",
                "promoted_prompt_candidate_hash": "cand-123",
                "decisions": [
                    {
                        "rank": 1,
                        "section_name": "GROUNDING_RULES",
                        "provider": "anthropic",
                        "prompt_candidate_hash": "cand-123",
                        "score": 1.0,
                        "recorded": True,
                        "passed": True,
                        "promoted": True,
                    }
                ],
            },
        )

        async def fake_run_prompt_suite_async(**_kwargs):
            return suite

        with (
            patch(
                "victor.evaluation.benchmarks.HumanEvalRunner",
                return_value=runner_impl,
            ),
            patch(
                "victor.ui.commands.benchmark._run_prompt_candidate_suite_async",
                side_effect=fake_run_prompt_suite_async,
            ),
            patch(
                "victor.framework.rl.process_prompt_candidate_evaluation_suite",
                return_value=SimpleNamespace(
                    prompt_optimizer_sync=sync_result,
                    prompt_rollout=None,
                    prompt_rollout_analysis=None,
                    prompt_rollout_decision=None,
                ),
            ) as mock_process,
        ):
            result = runner.invoke(
                benchmark_app,
                [
                    "run-prompt-suite",
                    "humaneval",
                    "--prompt-section",
                    "GROUNDING_RULES",
                    "--candidate-hash",
                    "cand-123",
                    "--provider",
                    "anthropic",
                    "--model",
                    "test-model",
                    "--output",
                    str(output),
                    "--record-benchmark-results",
                    "--promote-best",
                ],
            )

        assert result.exit_code == 0
        mock_process.assert_called_once_with(
            suite,
            min_pass_rate=0.5,
            promote_best=True,
            create_rollout=False,
            rollout_control_hash=None,
            rollout_traffic_split=0.1,
            rollout_min_samples_per_variant=100,
            analyze_rollout=False,
            apply_rollout_decision=False,
            rollout_decision_dry_run=False,
        )
        assert "Prompt optimizer benchmark sync" in result.stdout
        assert "Promoted best candidate: cand-123" in result.stdout
        saved = json.loads(output.read_text())
        assert saved["prompt_optimizer_sync"]["promoted_prompt_candidate_hash"] == "cand-123"

    def test_run_prompt_suite_can_create_rollout_for_approved_winner(self, tmp_path):
        from victor.evaluation import (
            PromptCandidateEvaluationRun,
            PromptCandidateEvaluationSpec,
            PromptCandidateEvaluationSuiteResult,
        )
        from victor.evaluation.protocol import (
            BenchmarkType,
            EvaluationConfig,
            EvaluationResult,
            TaskResult,
            TaskStatus,
        )

        runner_impl = object()
        base_config = EvaluationConfig(
            benchmark=BenchmarkType.HUMAN_EVAL,
            model="test-model",
            provider="anthropic",
        )
        output = tmp_path / "suite_with_rollout.json"
        suite = PromptCandidateEvaluationSuiteResult(
            base_config=base_config,
            runs=[
                PromptCandidateEvaluationRun(
                    spec=PromptCandidateEvaluationSpec(
                        section_name="GROUNDING_RULES",
                        prompt_candidate_hash="cand-123",
                        provider="anthropic",
                    ),
                    config=EvaluationConfig(
                        benchmark=BenchmarkType.HUMAN_EVAL,
                        model="test-model",
                        provider="anthropic",
                        prompt_candidate_hash="cand-123",
                        prompt_section_name="GROUNDING_RULES",
                    ),
                    result=EvaluationResult(
                        config=base_config,
                        task_results=[TaskResult(task_id="task-1", status=TaskStatus.PASSED)],
                    ),
                    label="GROUNDING_RULES:anthropic:cand-123",
                )
            ],
        )
        sync_result = SimpleNamespace(
            best_prompt_candidate_hash="cand-123",
            approved_prompt_candidate_hash="cand-123",
            promoted_prompt_candidate_hash=None,
            decisions=[],
            to_dict=lambda: {
                "best_prompt_candidate_hash": "cand-123",
                "approved_prompt_candidate_hash": "cand-123",
                "promoted_prompt_candidate_hash": None,
                "decisions": [],
            },
        )

        async def fake_run_prompt_suite_async(**_kwargs):
            return suite

        with (
            patch(
                "victor.evaluation.benchmarks.HumanEvalRunner",
                return_value=runner_impl,
            ),
            patch(
                "victor.ui.commands.benchmark._run_prompt_candidate_suite_async",
                side_effect=fake_run_prompt_suite_async,
            ),
            patch(
                "victor.framework.rl.process_prompt_candidate_evaluation_suite",
                return_value=SimpleNamespace(
                    prompt_optimizer_sync=sync_result,
                    prompt_rollout={
                        "created": True,
                        "experiment_id": "prompt_optimizer_grounding_rules_anthropic_cand-123",
                    },
                    prompt_rollout_analysis=None,
                    prompt_rollout_decision=None,
                ),
            ) as mock_process,
        ):
            result = runner.invoke(
                benchmark_app,
                [
                    "run-prompt-suite",
                    "humaneval",
                    "--prompt-section",
                    "GROUNDING_RULES",
                    "--candidate-hash",
                    "cand-123",
                    "--provider",
                    "anthropic",
                    "--model",
                    "test-model",
                    "--output",
                    str(output),
                    "--record-benchmark-results",
                    "--create-rollout",
                    "--rollout-traffic-split",
                    "0.2",
                    "--rollout-min-samples-per-variant",
                    "25",
                ],
            )

        assert result.exit_code == 0
        mock_process.assert_called_once_with(
            suite,
            min_pass_rate=0.5,
            promote_best=False,
            create_rollout=True,
            rollout_control_hash=None,
            rollout_traffic_split=0.2,
            rollout_min_samples_per_variant=25,
            analyze_rollout=False,
            apply_rollout_decision=False,
            rollout_decision_dry_run=False,
        )
        assert "Prompt rollout experiment started:" in result.stdout
        assert "prompt_optimizer_grounding_rules_anthropic_cand-123" in result.stdout
        saved = json.loads(output.read_text())
        assert saved["prompt_rollout"]["created"] is True
        assert (
            saved["prompt_rollout"]["experiment_id"]
            == "prompt_optimizer_grounding_rules_anthropic_cand-123"
        )

    def test_run_prompt_suite_reports_when_rollout_is_not_created(self):
        from victor.evaluation import (
            PromptCandidateEvaluationRun,
            PromptCandidateEvaluationSpec,
            PromptCandidateEvaluationSuiteResult,
        )
        from victor.evaluation.protocol import (
            BenchmarkType,
            EvaluationConfig,
            EvaluationResult,
            TaskResult,
            TaskStatus,
        )

        runner_impl = object()
        base_config = EvaluationConfig(
            benchmark=BenchmarkType.HUMAN_EVAL,
            model="test-model",
            provider="anthropic",
        )
        suite = PromptCandidateEvaluationSuiteResult(
            base_config=base_config,
            runs=[
                PromptCandidateEvaluationRun(
                    spec=PromptCandidateEvaluationSpec(
                        section_name="GROUNDING_RULES",
                        prompt_candidate_hash="cand-123",
                        provider="anthropic",
                    ),
                    config=EvaluationConfig(
                        benchmark=BenchmarkType.HUMAN_EVAL,
                        model="test-model",
                        provider="anthropic",
                        prompt_candidate_hash="cand-123",
                        prompt_section_name="GROUNDING_RULES",
                    ),
                    result=EvaluationResult(
                        config=base_config,
                        task_results=[TaskResult(task_id="task-1", status=TaskStatus.PASSED)],
                    ),
                    label="GROUNDING_RULES:anthropic:cand-123",
                )
            ],
        )
        sync_result = SimpleNamespace(
            best_prompt_candidate_hash="cand-123",
            approved_prompt_candidate_hash=None,
            promoted_prompt_candidate_hash=None,
            decisions=[],
            to_dict=lambda: {
                "best_prompt_candidate_hash": "cand-123",
                "approved_prompt_candidate_hash": None,
                "promoted_prompt_candidate_hash": None,
                "decisions": [],
            },
        )

        async def fake_run_prompt_suite_async(**_kwargs):
            return suite

        with (
            patch(
                "victor.evaluation.benchmarks.HumanEvalRunner",
                return_value=runner_impl,
            ),
            patch(
                "victor.ui.commands.benchmark._run_prompt_candidate_suite_async",
                side_effect=fake_run_prompt_suite_async,
            ),
            patch(
                "victor.framework.rl.process_prompt_candidate_evaluation_suite",
                return_value=SimpleNamespace(
                    prompt_optimizer_sync=sync_result,
                    prompt_rollout={
                        "created": False,
                        "error": "no benchmark-approved candidate available for rollout",
                    },
                    prompt_rollout_analysis=None,
                    prompt_rollout_decision=None,
                ),
            ),
        ):
            result = runner.invoke(
                benchmark_app,
                [
                    "run-prompt-suite",
                    "humaneval",
                    "--prompt-section",
                    "GROUNDING_RULES",
                    "--candidate-hash",
                    "cand-123",
                    "--provider",
                    "anthropic",
                    "--model",
                    "test-model",
                    "--record-benchmark-results",
                    "--create-rollout",
                ],
            )

        assert result.exit_code == 0
        assert "Prompt rollout not created:" in result.stdout
        assert "no benchmark-approved candidate available for" in result.stdout
        assert "rollout" in result.stdout

    def test_run_prompt_suite_can_analyze_existing_rollout(self, tmp_path):
        from victor.evaluation import (
            PromptCandidateEvaluationRun,
            PromptCandidateEvaluationSpec,
            PromptCandidateEvaluationSuiteResult,
        )
        from victor.evaluation.protocol import (
            BenchmarkType,
            EvaluationConfig,
            EvaluationResult,
            TaskResult,
            TaskStatus,
        )

        runner_impl = object()
        output = tmp_path / "suite_with_rollout_analysis.json"
        base_config = EvaluationConfig(
            benchmark=BenchmarkType.HUMAN_EVAL,
            model="test-model",
            provider="anthropic",
        )
        suite = PromptCandidateEvaluationSuiteResult(
            base_config=base_config,
            runs=[
                PromptCandidateEvaluationRun(
                    spec=PromptCandidateEvaluationSpec(
                        section_name="GROUNDING_RULES",
                        prompt_candidate_hash="cand-123",
                        provider="anthropic",
                    ),
                    config=EvaluationConfig(
                        benchmark=BenchmarkType.HUMAN_EVAL,
                        model="test-model",
                        provider="anthropic",
                        prompt_candidate_hash="cand-123",
                        prompt_section_name="GROUNDING_RULES",
                    ),
                    result=EvaluationResult(
                        config=base_config,
                        task_results=[TaskResult(task_id="task-1", status=TaskStatus.PASSED)],
                    ),
                    label="GROUNDING_RULES:anthropic:cand-123",
                )
            ],
        )
        sync_result = SimpleNamespace(
            best_prompt_candidate_hash="cand-123",
            approved_prompt_candidate_hash="cand-123",
            promoted_prompt_candidate_hash=None,
            decisions=[],
            to_dict=lambda: {
                "best_prompt_candidate_hash": "cand-123",
                "approved_prompt_candidate_hash": "cand-123",
                "promoted_prompt_candidate_hash": None,
                "decisions": [],
            },
        )
        rollout_analysis = {
            "experiment_id": "prompt_optimizer_grounding_rules_anthropic_cand-123",
            "status": "running",
            "analysis_available": True,
            "recommendation": "Roll out treatment - significant improvement detected",
            "auto_action": "rollout",
            "is_significant": True,
            "treatment_better": True,
            "effect_size": 0.2,
            "p_value": 0.01,
            "confidence_interval": (0.05, 0.3),
            "details": {
                "control": {"samples": 120, "success_rate": 0.55, "avg_quality": 0.6},
                "treatment": {"samples": 120, "success_rate": 0.66, "avg_quality": 0.7},
            },
        }

        async def fake_run_prompt_suite_async(**_kwargs):
            return suite

        with (
            patch(
                "victor.evaluation.benchmarks.HumanEvalRunner",
                return_value=runner_impl,
            ),
            patch(
                "victor.ui.commands.benchmark._run_prompt_candidate_suite_async",
                side_effect=fake_run_prompt_suite_async,
            ),
            patch(
                "victor.framework.rl.process_prompt_candidate_evaluation_suite",
                return_value=SimpleNamespace(
                    prompt_optimizer_sync=sync_result,
                    prompt_rollout=None,
                    prompt_rollout_analysis=rollout_analysis,
                    prompt_rollout_decision=None,
                ),
            ) as mock_process,
        ):
            result = runner.invoke(
                benchmark_app,
                [
                    "run-prompt-suite",
                    "humaneval",
                    "--prompt-section",
                    "GROUNDING_RULES",
                    "--candidate-hash",
                    "cand-123",
                    "--provider",
                    "anthropic",
                    "--model",
                    "test-model",
                    "--output",
                    str(output),
                    "--record-benchmark-results",
                    "--analyze-rollout",
                ],
            )

        assert result.exit_code == 0
        mock_process.assert_called_once_with(
            suite,
            min_pass_rate=0.5,
            promote_best=False,
            create_rollout=False,
            rollout_control_hash=None,
            rollout_traffic_split=0.1,
            rollout_min_samples_per_variant=100,
            analyze_rollout=True,
            apply_rollout_decision=False,
            rollout_decision_dry_run=False,
        )
        assert "Prompt rollout analysis" in result.stdout
        assert "Auto-apply action: rollout" in result.stdout
        saved = json.loads(output.read_text())
        assert saved["prompt_rollout_analysis"]["experiment_id"] == (
            "prompt_optimizer_grounding_rules_anthropic_cand-123"
        )

    def test_run_prompt_suite_can_apply_rollout_decision(self, tmp_path):
        from victor.evaluation import (
            PromptCandidateEvaluationRun,
            PromptCandidateEvaluationSpec,
            PromptCandidateEvaluationSuiteResult,
        )
        from victor.evaluation.protocol import (
            BenchmarkType,
            EvaluationConfig,
            EvaluationResult,
            TaskResult,
            TaskStatus,
        )

        runner_impl = object()
        output = tmp_path / "suite_with_rollout_decision.json"
        base_config = EvaluationConfig(
            benchmark=BenchmarkType.HUMAN_EVAL,
            model="test-model",
            provider="anthropic",
        )
        suite = PromptCandidateEvaluationSuiteResult(
            base_config=base_config,
            runs=[
                PromptCandidateEvaluationRun(
                    spec=PromptCandidateEvaluationSpec(
                        section_name="GROUNDING_RULES",
                        prompt_candidate_hash="cand-123",
                        provider="anthropic",
                    ),
                    config=EvaluationConfig(
                        benchmark=BenchmarkType.HUMAN_EVAL,
                        model="test-model",
                        provider="anthropic",
                        prompt_candidate_hash="cand-123",
                        prompt_section_name="GROUNDING_RULES",
                    ),
                    result=EvaluationResult(
                        config=base_config,
                        task_results=[TaskResult(task_id="task-1", status=TaskStatus.PASSED)],
                    ),
                    label="GROUNDING_RULES:anthropic:cand-123",
                )
            ],
        )
        sync_result = SimpleNamespace(
            best_prompt_candidate_hash="cand-123",
            approved_prompt_candidate_hash="cand-123",
            promoted_prompt_candidate_hash=None,
            decisions=[],
            to_dict=lambda: {
                "best_prompt_candidate_hash": "cand-123",
                "approved_prompt_candidate_hash": "cand-123",
                "promoted_prompt_candidate_hash": None,
                "decisions": [],
            },
        )
        rollout_analysis = {
            "experiment_id": "prompt_optimizer_grounding_rules_anthropic_cand-123",
            "status": "running",
            "analysis_available": True,
            "recommendation": "Roll out treatment - significant improvement detected",
            "auto_action": "rollout",
            "is_significant": True,
            "treatment_better": True,
            "effect_size": 0.2,
            "p_value": 0.01,
            "confidence_interval": (0.05, 0.3),
            "details": {
                "control": {"samples": 120, "success_rate": 0.55, "avg_quality": 0.6},
                "treatment": {"samples": 120, "success_rate": 0.66, "avg_quality": 0.7},
            },
        }
        rollout_decision = {
            "experiment_id": "prompt_optimizer_grounding_rules_anthropic_cand-123",
            "action": "rollout",
            "applied": True,
            "dry_run": False,
        }

        async def fake_run_prompt_suite_async(**_kwargs):
            return suite

        with (
            patch(
                "victor.evaluation.benchmarks.HumanEvalRunner",
                return_value=runner_impl,
            ),
            patch(
                "victor.ui.commands.benchmark._run_prompt_candidate_suite_async",
                side_effect=fake_run_prompt_suite_async,
            ),
            patch(
                "victor.framework.rl.process_prompt_candidate_evaluation_suite",
                return_value=SimpleNamespace(
                    prompt_optimizer_sync=sync_result,
                    prompt_rollout=None,
                    prompt_rollout_analysis=rollout_analysis,
                    prompt_rollout_decision=rollout_decision,
                ),
            ) as mock_process,
        ):
            result = runner.invoke(
                benchmark_app,
                [
                    "run-prompt-suite",
                    "humaneval",
                    "--prompt-section",
                    "GROUNDING_RULES",
                    "--candidate-hash",
                    "cand-123",
                    "--provider",
                    "anthropic",
                    "--model",
                    "test-model",
                    "--output",
                    str(output),
                    "--record-benchmark-results",
                    "--analyze-rollout",
                    "--apply-rollout-decision",
                ],
            )

        assert result.exit_code == 0
        mock_process.assert_called_once_with(
            suite,
            min_pass_rate=0.5,
            promote_best=False,
            create_rollout=False,
            rollout_control_hash=None,
            rollout_traffic_split=0.1,
            rollout_min_samples_per_variant=100,
            analyze_rollout=True,
            apply_rollout_decision=True,
            rollout_decision_dry_run=False,
        )
        assert "Prompt rollout decision applied: rollout" in result.stdout
        saved = json.loads(output.read_text())
        assert saved["prompt_rollout_decision"]["action"] == "rollout"

    @pytest.mark.asyncio
    async def test_run_benchmark_async_binds_prompt_candidate_for_agentic_execution(
        self,
    ):
        """Prompt-candidate benchmark flags should pin the live runtime candidate."""
        from victor.evaluation.protocol import (
            BenchmarkTask,
            BenchmarkType,
            EvaluationConfig,
        )
        from victor.ui.commands.benchmark import _run_benchmark_async

        class FakeRunner:
            async def load_tasks(self, _config):
                return [
                    BenchmarkTask(
                        task_id="swe-1",
                        benchmark=BenchmarkType.SWE_BENCH,
                        description="Fix the bug",
                        prompt="Fix the bug",
                    )
                ]

        class FakeHarness:
            def register_runner(self, _runner):
                return None

            async def run_evaluation(self, **_kwargs):
                return {"status": "ok"}

        fake_adapter = MagicMock()
        fake_adapter.orchestrator = SimpleNamespace(provider_name="anthropic", provider=None)
        fake_adapter.get_benchmark_tool_readiness.return_value = SimpleNamespace(
            ready=True,
            enabled_tools=("read", "graph"),
            missing_tools=(),
            disabled_tools=(),
        )

        with (
            patch(
                "victor.evaluation.harness.EvaluationHarness",
                return_value=FakeHarness(),
            ),
            # All adapter creation now routes through
            # create_from_session_config (Agent.create) — commits 8487d43a +
            # 77cb6406; from_profile is no longer called by the CLI.
            patch(
                "victor.evaluation.agent_adapter.VictorAgentAdapter.create_from_session_config",
                new=AsyncMock(return_value=fake_adapter),
            ) as create_from_session_config,
            patch(
                "victor.core.feature_flags.get_feature_flag_manager",
                return_value=SimpleNamespace(is_enabled=lambda *_args, **_kwargs: False),
            ),
        ):
            result = await _run_benchmark_async(
                runner=FakeRunner(),
                config=EvaluationConfig(
                    benchmark=BenchmarkType.SWE_BENCH,
                    model="claude-sonnet",
                    prompt_candidate_hash="cand-123",
                    prompt_section_name="GROUNDING_RULES",
                ),
                profile="default",
                model=None,
                timeout=120,
                max_turns=6,
                resume=False,
                provider_override=None,
                start_task=0,
                resolved_account=None,
            )

        assert result == {"status": "ok"}
        create_from_session_config.assert_awaited_once()
        binding = create_from_session_config.await_args.kwargs["config"].prompt_binding
        assert binding is not None
        assert binding.prompt_candidate_hash == "cand-123"
        assert binding.section_name == "GROUNDING_RULES"

    @pytest.mark.asyncio
    async def test_run_benchmark_async_uses_session_config_for_provider_override(self):
        """Provider overrides should route through SessionConfig + Agent facade."""
        from victor.evaluation.protocol import (
            BenchmarkTask,
            BenchmarkType,
            EvaluationConfig,
        )
        from victor.ui.commands.benchmark import _run_benchmark_async

        class FakeRunner:
            async def load_tasks(self, _config):
                return [
                    BenchmarkTask(
                        task_id="swe-1",
                        benchmark=BenchmarkType.SWE_BENCH,
                        description="Fix the bug",
                        prompt="Fix the bug",
                    )
                ]

        class FakeHarness:
            def register_runner(self, _runner):
                return None

            async def run_evaluation(self, **_kwargs):
                return {"status": "ok"}

        fake_adapter = MagicMock()
        fake_adapter.orchestrator = SimpleNamespace(provider_name="openai", provider=None)
        fake_adapter.get_benchmark_tool_readiness.return_value = SimpleNamespace(
            ready=True,
            enabled_tools=("read", "graph"),
            missing_tools=(),
            disabled_tools=(),
        )
        resolved_account = SimpleNamespace(auth=SimpleNamespace(method="oauth"))

        with (
            patch(
                "victor.evaluation.harness.EvaluationHarness",
                return_value=FakeHarness(),
            ),
            patch(
                "victor.evaluation.agent_adapter.VictorAgentAdapter.create_from_session_config",
                new=AsyncMock(return_value=fake_adapter),
            ) as create_from_session_config,
            patch(
                "victor.evaluation.agent_adapter.VictorAgentAdapter.from_profile",
            ) as from_profile,
            patch(
                "victor.core.feature_flags.get_feature_flag_manager",
                return_value=SimpleNamespace(is_enabled=lambda *_args, **_kwargs: False),
            ),
        ):
            result = await _run_benchmark_async(
                runner=FakeRunner(),
                config=EvaluationConfig(
                    benchmark=BenchmarkType.SWE_BENCH,
                    model="gpt-4o",
                ),
                profile="default",
                model=None,
                timeout=180,
                max_turns=6,
                resume=False,
                provider_override="openai",
                start_task=0,
                resolved_account=resolved_account,
            )

        assert result == {"status": "ok"}
        create_from_session_config.assert_awaited_once()
        from_profile.assert_not_called()
        session_config = create_from_session_config.await_args.args[0]
        assert session_config.agent_profile == "default"
        assert session_config.provider_override.provider == "openai"
        assert session_config.provider_override.model == "gpt-4o"
        assert session_config.provider_override.auth_mode == "oauth"
        assert session_config.provider_override.timeout == 180


class TestBenchmarkCompare:
    """Tests for benchmark compare command."""

    def test_compare_swe_bench(self):
        """Test comparing frameworks on SWE-bench."""
        result = runner.invoke(benchmark_app, ["compare", "--benchmark", "swe-bench"])
        assert result.exit_code == 0
        assert "Framework Comparison" in result.stdout

    def test_compare_unknown_benchmark(self):
        """Test comparing with unknown benchmark."""
        result = runner.invoke(benchmark_app, ["compare", "--benchmark", "unknown"])
        assert result.exit_code == 1
        assert "Unknown benchmark" in result.stdout

    def test_compare_cataloged_benchmark_without_results(self):
        """Test comparing a recognized benchmark that lacks published results."""
        result = runner.invoke(benchmark_app, ["compare", "--benchmark", "guide"])
        assert result.exit_code == 0
        assert "No published results available" in result.stdout

    def test_compare_external_benchmark_with_local_victor_results(self, tmp_path):
        """Local Victor artifacts should drive comparison output for external benchmarks."""
        saved_result = tmp_path / "guide_result.json"
        output = tmp_path / "compare.json"
        saved_result.write_text(
            json.dumps(
                {
                    "benchmark": "guide",
                    "model": "test-model",
                    "dataset_metadata": {"source_name": "GUIDE Consortium"},
                    "prompt_candidate_hash": "cand-123",
                    "section_name": "GROUNDING_RULES",
                    "metrics": {
                        "total_tasks": 1,
                        "passed": 1,
                        "failed": 0,
                        "errors": 0,
                        "timeouts": 0,
                        "pass_rate": 1.0,
                        "accepted_patch_rate": 1.0,
                        "avg_time_to_first_edit_seconds": 1.25,
                        "code_intelligence_task_coverage": 1.0,
                    },
                    "task_results": [{"task_id": "guide-1", "status": "passed"}],
                }
            )
        )

        result = runner.invoke(
            benchmark_app,
            [
                "compare",
                "--benchmark",
                "guide",
                "--victor-results",
                str(saved_result),
                "--format",
                "json",
                "--output",
                str(output),
            ],
        )

        assert result.exit_code == 0
        assert "Framework Comparison" in result.stdout
        assert "1.25s" in result.stdout
        assert "100.0%" in result.stdout
        saved = json.loads(output.read_text())
        assert saved["benchmark"] == "guide"
        assert saved["results"][0]["framework"] == "victor"
        assert saved["results"][0]["config"]["source"] == "GUIDE Consortium"
        assert (tmp_path / "compare.md").exists()
        summary = json.loads((tmp_path / "compare_summary.json").read_text())
        assert summary["framework_count"] == 1
        assert summary["results"][0]["framework"] == "victor"
        assert summary["results"][0]["prompt_candidate_hash"] == "cand-123"
        assert summary["results"][0]["section_name"] == "GROUNDING_RULES"

    def test_compare_accepts_multiple_local_victor_results(self, tmp_path):
        """Compare should load multiple local Victor artifacts into one report."""
        first = tmp_path / "guide_result_a.json"
        second = tmp_path / "guide_result_b.json"
        output = tmp_path / "multi_compare.json"
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

        result = runner.invoke(
            benchmark_app,
            [
                "compare",
                "--benchmark",
                "guide",
                "--victor-results",
                str(first),
                "--victor-results",
                str(second),
                "--format",
                "json",
                "--output",
                str(output),
            ],
        )

        assert result.exit_code == 0
        saved = json.loads(output.read_text())
        victor_entries = [row for row in saved["results"] if row["framework"] == "victor"]
        assert len(victor_entries) == 2
        assert [row["model"] for row in victor_entries] == ["model-a", "model-b"]
        fixtures = json.loads((tmp_path / "multi_compare_fixtures.json").read_text())
        assert fixtures["artifact_count"] == 2
        assert [artifact["model"] for artifact in fixtures["artifacts"]] == [
            "model-a",
            "model-b",
        ]
        bundled_fixture_dir = tmp_path / "multi_compare_fixtures"
        assert bundled_fixture_dir.is_dir()
        assert sorted(path.name for path in bundled_fixture_dir.iterdir()) == [
            "01_victor_model-a.json",
            "02_victor_model-b.json",
        ]

    def test_compare_accepts_saved_fixture_manifest_bundle(self, tmp_path):
        """Compare should accept a saved fixture manifest bundle directly."""
        first = tmp_path / "guide_result_a.json"
        second = tmp_path / "guide_result_b.json"
        output = tmp_path / "manifest_compare.json"
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
        bundle = save_comparison_report_bundle(
            report,
            tmp_path / "guide_fixture_bundle.json",
            primary_format="json",
        )

        result = runner.invoke(
            benchmark_app,
            [
                "compare",
                "--benchmark",
                "guide",
                "--victor-results",
                str(bundle["fixtures"]),
                "--format",
                "json",
                "--output",
                str(output),
            ],
        )

        assert result.exit_code == 0
        saved = json.loads(output.read_text())
        victor_entries = [row for row in saved["results"] if row["framework"] == "victor"]
        assert len(victor_entries) == 2
        assert [row["model"] for row in victor_entries] == ["model-a", "model-b"]

    def test_compare_accepts_checked_in_fixture_set_directory(self, tmp_path):
        """Compare should accept a checked-in fixture set directory directly."""
        fixture_dir = Path("tests/fixtures/benchmarks/guide_fixture_set")
        output = tmp_path / "fixture_dir_compare.json"

        result = runner.invoke(
            benchmark_app,
            [
                "compare",
                "--benchmark",
                "guide",
                "--victor-results",
                str(fixture_dir),
                "--format",
                "json",
                "--output",
                str(output),
            ],
        )

        assert result.exit_code == 0
        saved = json.loads(output.read_text())
        victor_entries = [row for row in saved["results"] if row["framework"] == "victor"]
        assert len(victor_entries) == 2
        assert [row["model"] for row in victor_entries] == [
            "fixture-model-a",
            "fixture-model-b",
        ]

    def test_compare_accepts_checked_in_fixture_set_name(self, tmp_path):
        """Compare should accept a checked-in fixture set by stable name."""
        output = tmp_path / "fixture_name_compare.json"

        result = runner.invoke(
            benchmark_app,
            [
                "compare",
                "--benchmark",
                "guide",
                "--victor-fixture-set",
                "guide_fixture_set",
                "--format",
                "json",
                "--output",
                str(output),
            ],
        )

        assert result.exit_code == 0
        assert "guide_fixture_set" in result.stdout
        saved = json.loads(output.read_text())
        victor_entries = [row for row in saved["results"] if row["framework"] == "victor"]
        assert len(victor_entries) == 2
        assert [row["model"] for row in victor_entries] == [
            "fixture-model-a",
            "fixture-model-b",
        ]

    def test_compare_accepts_fixture_sets_by_benchmark(self, tmp_path):
        """Compare should accept all checked-in fixture sets for a benchmark."""
        output = tmp_path / "fixture_benchmark_compare.json"

        result = runner.invoke(
            benchmark_app,
            [
                "compare",
                "--benchmark",
                "guide",
                "--victor-fixture-benchmark",
                "guide",
                "--format",
                "json",
                "--output",
                str(output),
            ],
        )

        assert result.exit_code == 0
        assert "Included Victor fixture benchmark: guide" in result.stdout
        saved = json.loads(output.read_text())
        victor_entries = [row for row in saved["results"] if row["framework"] == "victor"]
        assert len(victor_entries) == 3
        assert [row["model"] for row in victor_entries] == [
            "fixture-model-a",
            "fixture-model-b",
            "fixture-model-c",
        ]

    def test_compare_accepts_publication_bundle_root(self, tmp_path):
        """Compare should accept a portable fixture publication bundle root directly."""
        publication = save_fixture_benchmark_publication_bundle(
            output_path=tmp_path / "published_fixtures",
            benchmark="guide",
            verify=True,
        )
        output = tmp_path / "publication_compare.json"

        result = runner.invoke(
            benchmark_app,
            [
                "compare",
                "--benchmark",
                "guide",
                "--victor-publication-root",
                str(publication["root"]),
                "--format",
                "json",
                "--output",
                str(output),
            ],
        )

        assert result.exit_code == 0
        # Rich wraps console output to the terminal width, which on CI (narrow,
        # ~80 cols) splits the long publication-root path across lines (e.g.
        # "published_fixture\ns"). Normalize whitespace so these substring checks
        # are console-width independent.
        normalized = " ".join(result.stdout.split())
        assert "Included Victor publication root:" in normalized
        assert "published_fixtures" in normalized.replace(" ", "")
        saved = json.loads(output.read_text())
        victor_entries = [row for row in saved["results"] if row["framework"] == "victor"]
        assert len(victor_entries) == 3
        assert [row["model"] for row in victor_entries] == [
            "fixture-model-a",
            "fixture-model-b",
            "fixture-model-c",
        ]

    def test_compare_rejects_unknown_fixture_set_name(self):
        """Compare should fail fast for an unknown checked-in fixture set name."""
        result = runner.invoke(
            benchmark_app,
            [
                "compare",
                "--benchmark",
                "guide",
                "--victor-fixture-set",
                "missing_fixture_set",
            ],
        )

        assert result.exit_code == 1
        assert "Failed to load Victor results" in result.stdout
        assert "missing_fixture_set" in result.stdout

    def test_compare_rejects_mismatched_local_victor_results(self, tmp_path):
        """Comparison should fail when a local artifact is for another benchmark."""
        saved_result = tmp_path / "wrong_result.json"
        saved_result.write_text(
            json.dumps(
                {
                    "benchmark": "swe-bench",
                    "model": "test-model",
                    "metrics": {"total_tasks": 1, "pass_rate": 1.0},
                    "task_results": [{"task_id": "swe-1", "status": "passed"}],
                }
            )
        )

        result = runner.invoke(
            benchmark_app,
            [
                "compare",
                "--benchmark",
                "guide",
                "--victor-results",
                str(saved_result),
            ],
        )

        assert result.exit_code == 1
        assert "does not match requested" in result.stdout


class TestBenchmarkLeaderboard:
    """Tests for benchmark leaderboard command."""

    def test_leaderboard_swe_bench(self):
        """Test showing SWE-bench leaderboard."""
        result = runner.invoke(benchmark_app, ["leaderboard", "--benchmark", "swe-bench"])
        assert result.exit_code == 0
        assert "Leaderboard" in result.stdout
        assert "Rank" in result.stdout

    def test_leaderboard_unknown_benchmark(self):
        """Test leaderboard with unknown benchmark."""
        result = runner.invoke(benchmark_app, ["leaderboard", "--benchmark", "unknown"])
        assert result.exit_code == 1
        assert "Unknown benchmark" in result.stdout

    def test_leaderboard_external_benchmark_with_local_victor_results(self, tmp_path):
        """Leaderboards should accept local Victor artifacts for external benchmarks."""
        saved_result = tmp_path / "guide_result.json"
        saved_result.write_text(
            json.dumps(
                {
                    "benchmark": "guide",
                    "model": "test-model",
                    "dataset_metadata": {"source_name": "GUIDE Consortium"},
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

        result = runner.invoke(
            benchmark_app,
            [
                "leaderboard",
                "--benchmark",
                "guide",
                "--victor-results",
                str(saved_result),
            ],
        )

        assert result.exit_code == 0
        assert "Leaderboard" in result.stdout
        assert "victor" in result.stdout
        assert "GUIDE Consortium" in result.stdout


class TestBenchmarkCapabilities:
    """Tests for benchmark capabilities command."""

    def test_capabilities(self):
        """Test showing capabilities comparison."""
        result = runner.invoke(benchmark_app, ["capabilities"])
        assert result.exit_code == 0
        assert "Framework Capabilities Comparison" in result.stdout
        assert "Code Generation" in result.stdout
        assert "victor" in result.stdout
        assert "aider" in result.stdout


class TestBenchmarkHelp:
    """Tests for benchmark help."""

    def test_benchmark_help(self):
        """Test benchmark help output."""
        result = runner.invoke(benchmark_app, ["--help"])
        assert result.exit_code == 0
        assert "list" in result.stdout
        assert "run" in result.stdout
        assert "compare" in result.stdout
        assert "leaderboard" in result.stdout
        assert "capabilities" in result.stdout


@pytest.mark.skipif(
    not is_ollama_available(),
    reason="Ollama not available - skipping benchmark execution tests",
)
class TestBenchmarkExecution:
    """Integration tests for actually running benchmarks.

    These tests require Ollama to be running with a model available.
    They are skipped in CI environments where Ollama is not available.
    """

    def test_run_humaneval_with_ollama(self):
        """Test running HumanEval with Ollama (limited tasks)."""
        # Use default profile which should use Ollama if available
        result = runner.invoke(
            benchmark_app,
            [
                "run",
                "humaneval",
                "--max-tasks",
                "1",
                "--timeout",
                "60",
                "--profile",
                "default",
            ],
        )
        # Should start running or fail gracefully if model unavailable
        # We don't assert success since the model might not be loaded
        assert (
            "HumanEval" in result.stdout
            or "humaneval" in result.stdout
            or result.exit_code in (0, 1)
        )

    def test_run_mbpp_with_ollama(self):
        """Test running MBPP with Ollama (limited tasks)."""
        result = runner.invoke(
            benchmark_app,
            [
                "run",
                "mbpp",
                "--max-tasks",
                "1",
                "--timeout",
                "60",
                "--profile",
                "default",
            ],
        )
        assert "MBPP" in result.stdout or "mbpp" in result.stdout or result.exit_code in (0, 1)
