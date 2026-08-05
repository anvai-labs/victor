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

"""Unit tests for evaluation harness and metrics."""

import pytest

from victor.evaluation import (
    BenchmarkFailureCategory,
    BenchmarkType,
    CodeQualityMetrics,
    EvaluationConfig,
    EvaluationResult,
    TaskResult,
    TaskStatus,
    aggregate_planning_feedback,
    aggregate_team_feedback,
    extract_planning_events,
    extract_team_feedback_artifacts,
    pass_at_k,
    summarize_planning_feedback,
    summarize_team_feedback,
)
from victor.evaluation.code_quality import CodeQualityAnalyzer
from victor.evaluation.pass_at_k import (
    PassAtKResult,
    combinations,
    estimate_required_samples,
    generate_report,
)
from victor.evaluation.analyzers import AnalyzerRegistry
from victor.evaluation.protocol import FailureStage
from victor.evaluation.protocol import ConfidenceBucket


class TestPassAtK:
    """Tests for Pass@k metric calculation."""

    def test_pass_at_k_basic(self):
        """Test basic pass@k calculation."""
        # 50% success rate
        result = pass_at_k(n=100, c=50, k=1)
        assert result == pytest.approx(0.5, abs=0.01)

    def test_pass_at_k_high_k(self):
        """With high k, pass@k should approach 1 if any correct samples."""
        result = pass_at_k(n=100, c=50, k=100)
        assert result == pytest.approx(1.0, abs=0.01)

    def test_pass_at_k_zero_correct(self):
        """With no correct samples, pass@k should be 0."""
        result = pass_at_k(n=100, c=0, k=10)
        assert result == 0.0

    def test_pass_at_k_all_correct(self):
        """With all correct samples, pass@k should be 1."""
        result = pass_at_k(n=100, c=100, k=1)
        assert result == 1.0

    def test_pass_at_k_edge_cases(self):
        """Test edge cases."""
        assert pass_at_k(n=0, c=0, k=1) == 0.0
        assert pass_at_k(n=10, c=10, k=5) == 1.0
        assert pass_at_k(n=10, c=5, k=20) > 0.9  # k > n, should use k=n

    def test_pass_at_k_realistic_values(self):
        """Test with realistic benchmark values."""
        # HumanEval-like scenario: 50 samples, 20 correct
        result = pass_at_k(n=50, c=20, k=10)
        assert 0.9 < result < 1.0  # High probability with k=10

        # Low success rate scenario
        result = pass_at_k(n=100, c=5, k=1)
        assert result == pytest.approx(0.05, abs=0.01)


def test_public_api_exports_planning_feedback_helpers():
    """The package-level evaluation API should expose planning feedback helpers."""
    assert callable(extract_planning_events)
    assert callable(summarize_planning_feedback)
    assert callable(aggregate_planning_feedback)


def test_public_api_exports_team_feedback_helpers():
    """The package-level evaluation API should expose team feedback helpers."""
    assert callable(extract_team_feedback_artifacts)
    assert callable(summarize_team_feedback)
    assert callable(aggregate_team_feedback)

    def test_combinations(self):
        """Test combinations calculation."""
        assert combinations(5, 2) == pytest.approx(10, abs=0.01)
        assert combinations(10, 3) == pytest.approx(120, abs=0.01)
        assert combinations(10, 0) == 1
        assert combinations(10, 10) == 1
        assert combinations(5, 6) == 0  # k > n


class TestCodeQualityMetrics:
    """Tests for CodeQualityMetrics dataclass."""

    def test_default_values(self):
        """Test default metric values."""
        metrics = CodeQualityMetrics()
        assert metrics.syntax_valid is True
        assert metrics.lint_errors == 0
        assert metrics.style_score == 1.0
        assert metrics.maintainability_index == 100.0

    def test_overall_score_perfect(self):
        """Test overall score with perfect metrics."""
        metrics = CodeQualityMetrics(
            syntax_valid=True,
            style_score=1.0,
            cyclomatic_complexity=1.0,
            maintainability_index=100.0,
        )
        score = metrics.get_overall_score()
        assert score == pytest.approx(97.75, abs=1.0)

    def test_overall_score_syntax_error(self):
        """Test overall score with syntax error."""
        metrics = CodeQualityMetrics(syntax_valid=False)
        score = metrics.get_overall_score()
        assert score < 100  # Syntax invalid penalizes score

    def test_overall_score_high_complexity(self):
        """Test overall score with high complexity."""
        metrics = CodeQualityMetrics(cyclomatic_complexity=20.0)
        score = metrics.get_overall_score()
        assert score < 80  # High complexity should lower score


class TestCodeQualityAnalyzer:
    """Tests for CodeQualityAnalyzer."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        return CodeQualityAnalyzer(use_ruff=False, use_radon=False)

    @pytest.mark.asyncio
    async def test_valid_python_code(self, analyzer):
        """Test analysis of valid Python code."""
        code = '''
def hello_world():
    """Print hello world."""
    print("Hello, World!")
    return True
'''
        metrics = await analyzer.analyze(code, language="python")
        assert metrics.syntax_valid is True
        assert metrics.functions_count == 1
        assert metrics.lines_of_code > 0

    @pytest.mark.asyncio
    async def test_invalid_syntax(self, analyzer):
        """Test analysis of code with syntax errors."""
        code = "def broken(:"
        metrics = await analyzer.analyze(code, language="python")
        assert metrics.syntax_valid is False

    @pytest.mark.asyncio
    async def test_empty_code(self, analyzer):
        """Test analysis of empty code."""
        metrics = await analyzer.analyze("", language="python")
        assert metrics.syntax_valid is False

    @pytest.mark.asyncio
    async def test_type_coverage(self, analyzer):
        """Test type hint coverage detection."""
        typed_code = """
def add(a: int, b: int) -> int:
    return a + b
"""
        untyped_code = """
def add(a, b):
    return a + b
"""
        typed_metrics = await analyzer.analyze(typed_code, language="python")
        untyped_metrics = await analyzer.analyze(untyped_code, language="python")

        assert typed_metrics.type_coverage > untyped_metrics.type_coverage

    @pytest.mark.asyncio
    async def test_complexity_estimation(self, analyzer):
        """Test complexity estimation without radon."""
        simple_code = """
def simple():
    return 1
"""
        complex_code = """
def complex(x):
    if x > 0:
        for i in range(x):
            if i % 2 == 0:
                print(i)
            elif i % 3 == 0:
                continue
            else:
                break
"""
        simple_metrics = await analyzer.analyze(simple_code, language="python")
        complex_metrics = await analyzer.analyze(complex_code, language="python")

        assert complex_metrics.cyclomatic_complexity > simple_metrics.cyclomatic_complexity


class TestTaskResult:
    """Tests for TaskResult dataclass."""

    def test_test_pass_rate(self):
        """Test test pass rate calculation."""
        result = TaskResult(
            task_id="test1",
            status=TaskStatus.PASSED,
            tests_passed=8,
            tests_total=10,
        )
        assert result.test_pass_rate == 0.8

    def test_test_pass_rate_zero_total(self):
        """Test pass rate with zero total tests."""
        result = TaskResult(
            task_id="test1",
            status=TaskStatus.PASSED,
            tests_total=0,
        )
        assert result.test_pass_rate == 0.0

    def test_is_success(self):
        """Test is_success property."""
        passed = TaskResult(task_id="t1", status=TaskStatus.PASSED)
        failed = TaskResult(task_id="t2", status=TaskStatus.FAILED)

        assert passed.is_success is True
        assert failed.is_success is False

    def test_tokens_per_test(self):
        """Test token efficiency calculation."""
        result = TaskResult(
            task_id="test1",
            status=TaskStatus.PASSED,
            tokens_used=1000,
            tests_passed=10,
        )
        assert result.tokens_per_test == 100.0

    def test_tokens_per_test_zero_passed(self):
        """Test tokens per test with zero passed."""
        result = TaskResult(
            task_id="test1",
            status=TaskStatus.FAILED,
            tokens_used=1000,
            tests_passed=0,
        )
        assert result.tokens_per_test == float("inf")

    def test_cost_efficiency(self):
        """Test cost efficiency calculation."""
        result = TaskResult(
            task_id="test1",
            status=TaskStatus.PASSED,
            tokens_used=1000,
            completion_score=0.8,
        )
        assert result.cost_efficiency == pytest.approx(0.8, abs=0.01)

    def test_calculate_completion_score(self):
        """Test completion score calculation."""
        result = TaskResult(
            task_id="test1",
            status=TaskStatus.PASSED,
            tests_passed=8,
            tests_total=10,
            generated_code="def foo(): pass",
        )

        score = result.calculate_completion_score()
        assert 0.5 < score < 1.0  # Should have good score

    def test_code_intelligence_usage_flags(self):
        """TaskResult should expose code-search/graph usage flags."""
        result = TaskResult(
            task_id="test1",
            status=TaskStatus.PASSED,
            code_search_calls=2,
            graph_calls=1,
        )

        assert result.code_intelligence_calls == 3
        assert result.used_code_search is True
        assert result.used_graph is True
        assert result.used_code_intelligence is True

    def test_get_failure_diagnosis_derives_hierarchical_taxonomy(self):
        """Structured diagnosis should infer stage and subtype from failure details."""
        result = TaskResult(
            task_id="test1",
            status=TaskStatus.FAILED,
            failure_category=BenchmarkFailureCategory.TASK_COMPLETION,
            failure_details={
                "missing_actions": ["click"],
                "missing_answer_phrases": ["settings page"],
            },
        )

        diagnosis = result.get_failure_diagnosis()

        assert diagnosis is not None
        assert diagnosis.stage == FailureStage.ACTION
        assert diagnosis.subtype == "missing_required_actions"
        assert diagnosis.path == "action.task_completion.missing_required_actions"

    def test_get_confidence_assessment_high_for_supported_pass(self):
        """Successful results with strong evidence should report high confidence."""
        result = TaskResult(
            task_id="test1",
            status=TaskStatus.PASSED,
            tests_passed=4,
            tests_total=4,
            completion_score=1.0,
        )

        assessment = result.get_confidence_assessment()

        assert assessment.confidence_score > 0.95
        assert assessment.bucket == ConfidenceBucket.HIGH
        assert assessment.truth_aligned is True

    def test_get_confidence_assessment_penalizes_unsupported_claim_failures(self):
        """Unsupported-claim failures should remain low confidence even with high coverage."""
        result = TaskResult(
            task_id="test1",
            status=TaskStatus.FAILED,
            completion_score=1.0,
            failure_category=BenchmarkFailureCategory.UNSUPPORTED_CLAIM,
            failure_details={
                "claim_coverage": 1.0,
                "citation_coverage": 1.0,
                "forbidden_claim_hits": ["invented claim"],
            },
        )

        assessment = result.get_confidence_assessment()

        assert assessment.evidence_score == pytest.approx(1.0)
        assert assessment.confidence_score < 0.3
        assert assessment.bucket == ConfidenceBucket.LOW
        assert assessment.truth_aligned is True


class TestEvaluationResult:
    """Tests for EvaluationResult aggregate metrics."""

    def test_get_metrics_includes_code_intelligence_usage(self):
        """Aggregate metrics should include code-search/graph usage counts and coverage."""
        result = EvaluationResult(
            config=EvaluationConfig(
                benchmark=BenchmarkType.HUMAN_EVAL,
                model="test-model",
            ),
            task_results=[
                TaskResult(
                    task_id="task-1",
                    status=TaskStatus.PASSED,
                    tool_calls=5,
                    code_search_calls=2,
                    graph_calls=1,
                ),
                TaskResult(
                    task_id="task-2",
                    status=TaskStatus.FAILED,
                    tool_calls=1,
                    code_search_calls=0,
                    graph_calls=0,
                ),
            ],
        )

        metrics = result.get_metrics()

        assert metrics["total_tool_calls"] == 6
        assert metrics["total_code_search_calls"] == 2
        assert metrics["total_graph_calls"] == 1
        assert metrics["total_code_intelligence_calls"] == 3
        assert metrics["tasks_using_code_search"] == 1
        assert metrics["tasks_using_graph"] == 1
        assert metrics["tasks_using_code_intelligence"] == 1
        assert metrics["code_intelligence_task_coverage"] == pytest.approx(0.5)

    def test_get_metrics_includes_failure_taxonomy_breakdown(self):
        """Aggregate metrics should count failure stages and taxonomy paths."""
        result = EvaluationResult(
            config=EvaluationConfig(
                benchmark=BenchmarkType.GUIDE,
                model="test-model",
            ),
            task_results=[
                TaskResult(
                    task_id="task-1",
                    status=TaskStatus.FAILED,
                    failure_category=BenchmarkFailureCategory.TASK_COMPLETION,
                    failure_details={"missing_actions": ["click"]},
                ),
                TaskResult(
                    task_id="task-2",
                    status=TaskStatus.FAILED,
                    failure_category=BenchmarkFailureCategory.UNSUPPORTED_CLAIM,
                    failure_details={"forbidden_claim_hits": ["invented claim"]},
                ),
            ],
        )

        metrics = result.get_metrics()

        assert metrics["failure_categories"] == {
            "task_completion": 1,
            "unsupported_claim": 1,
        }
        assert metrics["failure_stages"] == {
            "action": 1,
            "grounding": 1,
        }
        assert metrics["failure_taxonomy"] == {
            "action.task_completion.missing_required_actions": 1,
            "grounding.unsupported_claim.forbidden_claim": 1,
        }

    def test_get_metrics_includes_confidence_calibration_summary(self):
        """Aggregate metrics should summarize confidence buckets and alignment."""
        passed = TaskResult(
            task_id="task-1",
            status=TaskStatus.PASSED,
            tests_passed=2,
            tests_total=2,
            completion_score=1.0,
        )
        failed = TaskResult(
            task_id="task-2",
            status=TaskStatus.FAILED,
            completion_score=1.0,
            failure_category=BenchmarkFailureCategory.UNSUPPORTED_CLAIM,
            failure_details={
                "claim_coverage": 1.0,
                "citation_coverage": 1.0,
                "forbidden_claim_hits": ["invented claim"],
            },
        )
        result = EvaluationResult(
            config=EvaluationConfig(
                benchmark=BenchmarkType.DR3_EVAL,
                model="test-model",
            ),
            task_results=[passed, failed],
        )

        metrics = result.get_metrics()

        expected_average = (
            passed.get_confidence_assessment().confidence_score
            + failed.get_confidence_assessment().confidence_score
        ) / 2
        assert metrics["avg_confidence_score"] == pytest.approx(expected_average)
        assert metrics["confidence_buckets"] == {
            "high": 1,
            "low": 1,
        }
        assert metrics["truth_alignment_rate"] == pytest.approx(1.0)

    def test_get_metrics_includes_topology_feedback_summary(self):
        """Aggregate metrics should expose topology coverage, mix, and reward."""
        result = EvaluationResult(
            config=EvaluationConfig(
                benchmark=BenchmarkType.CUSTOM,
                model="test-model",
            ),
            task_results=[
                TaskResult(
                    task_id="task-1",
                    status=TaskStatus.PASSED,
                    completion_score=1.0,
                    tool_calls=4,
                    turns=2,
                    metadata={
                        "topology_events": [
                            {
                                "action": "single_agent",
                                "topology": "single_agent",
                                "execution_mode": "single_agent",
                                "provider": "openai",
                                "confidence": 0.76,
                            }
                        ]
                    },
                ),
                TaskResult(
                    task_id="task-2",
                    status=TaskStatus.FAILED,
                    completion_score=0.25,
                    tool_calls=7,
                    turns=4,
                    metadata={
                        "topology_events": [
                            {
                                "action": "team_plan",
                                "topology": "team",
                                "execution_mode": "team_execution",
                                "formation": "parallel",
                                "provider": "anthropic",
                                "confidence": 0.86,
                                "fallback_action": "escalate_model",
                            }
                        ]
                    },
                ),
            ],
        )

        metrics = result.get_metrics()

        assert metrics["tasks_with_topology_feedback"] == 2
        assert metrics["topology_feedback_coverage"] == pytest.approx(1.0)
        assert metrics["topology_actions"] == {"single_agent": 1, "team_plan": 1}
        assert metrics["topology_kinds"] == {"single_agent": 1, "team": 1}
        assert metrics["topology_execution_modes"] == {
            "single_agent": 1,
            "team_execution": 1,
        }
        assert 0.0 <= metrics["avg_topology_reward"] <= 1.0
        assert metrics["topology_fallback_rate"] > 0.0

    def test_get_metrics_includes_planning_feedback_summary(self):
        """Aggregate metrics should expose planning policy coverage and deltas."""
        result = EvaluationResult(
            config=EvaluationConfig(
                benchmark=BenchmarkType.CUSTOM,
                model="test-model",
            ),
            task_results=[
                TaskResult(
                    task_id="task-1",
                    status=TaskStatus.FAILED,
                    completion_score=0.2,
                    metadata={
                        "planning_events": [
                            {
                                "selection_policy": "heuristic_fast_path",
                                "used_llm_planning": False,
                                "task_type": "action",
                            }
                        ]
                    },
                ),
                TaskResult(
                    task_id="task-2",
                    status=TaskStatus.PASSED,
                    completion_score=0.9,
                    metadata={
                        "planning_events": [
                            {
                                "selection_policy": "experiment_forced_slow_path",
                                "used_llm_planning": True,
                                "task_type": "action",
                                "force_reason": "experiment_constraints: tests_pass",
                                "constraint_tags": ["tests_pass"],
                                "experiment_support": 0.4,
                            }
                        ]
                    },
                ),
            ],
        )

        metrics = result.get_metrics()

        assert metrics["tasks_with_planning_feedback"] == 2
        assert metrics["planning_feedback_coverage"] == pytest.approx(1.0)
        assert metrics["planning_policy_counts"] == {
            "heuristic_fast_path": 1,
            "experiment_forced_slow_path": 1,
        }
        assert metrics["planning_force_reasons"] == {"experiment_constraints: tests_pass": 1}
        assert metrics["planning_used_llm_rate"] == pytest.approx(0.5)
        assert metrics["planning_forced_slow_path_completion_delta"] == pytest.approx(0.7)

    def test_get_metrics_includes_team_feedback_summary(self):
        """Aggregate metrics should expose team/worktree coverage and merge risk."""
        result = EvaluationResult(
            config=EvaluationConfig(
                benchmark=BenchmarkType.CUSTOM,
                model="test-model",
            ),
            task_results=[
                TaskResult(
                    task_id="task-1",
                    status=TaskStatus.PASSED,
                    metadata={
                        "worktree_plan": {
                            "formation": "parallel",
                            "assignments": [
                                {"member_id": "planner", "claimed_paths": ["src/auth"]},
                                {
                                    "member_id": "tester",
                                    "claimed_paths": ["tests/auth"],
                                },
                            ],
                        },
                        "worktree_session": {
                            "materialized": True,
                            "assignments": [
                                {"member_id": "planner", "materialized": True},
                                {"member_id": "tester", "materialized": True},
                            ],
                        },
                        "merge_analysis": {
                            "risk_level": "low",
                            "conflict_count": 0,
                            "member_changed_files": {
                                "planner": ["src/auth/service.py"],
                                "tester": ["tests/auth/test_service.py"],
                            },
                        },
                    },
                ),
                TaskResult(
                    task_id="task-2",
                    status=TaskStatus.FAILED,
                    metadata={
                        "worktree_plan": {
                            "formation": "parallel",
                            "assignments": [
                                {"member_id": "planner", "claimed_paths": ["src/auth"]},
                                {
                                    "member_id": "reviewer",
                                    "claimed_paths": ["src/auth"],
                                },
                            ],
                        },
                        "merge_analysis": {
                            "risk_level": "high",
                            "conflict_count": 1,
                            "overlapping_files": [{"path": "src/auth/service.py"}],
                            "member_changed_files": {
                                "planner": ["src/auth/service.py"],
                                "reviewer": ["src/auth/service.py"],
                            },
                            "readonly_violations": {"reviewer": ["docs/guide.md"]},
                        },
                        "worktree_cleanup": {
                            "removed": [],
                            "errors": ["cleanup failed"],
                        },
                    },
                ),
            ],
        )

        metrics = result.get_metrics()

        assert metrics["tasks_with_team_feedback"] == 2
        assert metrics["team_feedback_coverage"] == pytest.approx(1.0)
        assert metrics["team_formations"] == {"parallel": 2}
        assert metrics["team_merge_risk_levels"] == {"low": 1, "high": 1}
        assert metrics["team_worktree_materialized_count"] == 1
        assert metrics["team_high_risk_task_count"] == 1
        assert metrics["team_merge_conflict_count"] == 1
        assert metrics["team_readonly_violation_count"] == 1
        assert metrics["team_cleanup_error_task_count"] == 1
        assert metrics["team_worker_contract_task_count"] == 0
        assert metrics["team_merge_review_contract_task_count"] == 0

    def test_get_metrics_includes_merge_efficiency_and_first_edit_metrics(self):
        """Aggregate metrics should expose accepted-patch efficiency and first-edit timing."""
        result = EvaluationResult(
            config=EvaluationConfig(
                benchmark=BenchmarkType.CUSTOM,
                model="test-model",
            ),
            task_results=[
                TaskResult(
                    task_id="task-1",
                    status=TaskStatus.PASSED,
                    generated_patch="diff --git a/app.py b/app.py",
                    tokens_used=100,
                    cost_usd_micros=4000,
                    metadata={
                        "time_to_first_tool_call_seconds": 0.75,
                        "time_to_first_edit_seconds": 2.5,
                        "first_edit_tool_name": "edit",
                    },
                ),
                TaskResult(
                    task_id="task-2",
                    status=TaskStatus.PASSED,
                    generated_code="print('patched')",
                    tokens_used=50,
                    cost_usd_micros=2000,
                    metadata={
                        "accepted_patch": True,
                        "time_to_first_tool_call_seconds": 0.25,
                        "time_to_first_edit_seconds": 1.5,
                        "first_edit_tool_name": "write",
                    },
                ),
                TaskResult(
                    task_id="task-3",
                    status=TaskStatus.FAILED,
                    generated_patch="diff --git a/test.py b/test.py",
                    tokens_used=70,
                    cost_usd_micros=3000,
                    metadata={
                        "time_to_first_tool_call_seconds": 1.25,
                        "time_to_first_edit_seconds": 3.5,
                    },
                ),
            ],
        )

        metrics = result.get_metrics()

        assert metrics["accepted_patch_count"] == 2
        assert metrics["accepted_patch_rate"] == pytest.approx(2 / 3)
        assert metrics["tokens_to_merge_total"] == 150
        assert metrics["avg_tokens_to_merge"] == pytest.approx(75.0)
        assert metrics["cost_per_accepted_patch_usd"] == pytest.approx(0.003)
        assert metrics["tasks_with_time_to_first_tool_call"] == 3
        assert metrics["avg_time_to_first_tool_call_seconds"] == pytest.approx(0.75)
        assert metrics["tasks_with_time_to_first_edit"] == 3
        assert metrics["avg_time_to_first_edit_seconds"] == pytest.approx(2.5)

    def test_get_metrics_includes_tool_selection_effect_breakdown(self):
        """Aggregate metrics should distinguish code-intelligence and blind-navigation outcomes."""
        result = EvaluationResult(
            config=EvaluationConfig(
                benchmark=BenchmarkType.CUSTOM,
                model="test-model",
            ),
            task_results=[
                TaskResult(
                    task_id="task-1",
                    status=TaskStatus.PASSED,
                    tokens_used=120,
                    duration_seconds=8.0,
                    code_search_calls=2,
                    generated_patch="diff --git a/app.py b/app.py",
                    metadata={"accepted_patch": True},
                ),
                TaskResult(
                    task_id="task-2",
                    status=TaskStatus.FAILED,
                    tokens_used=60,
                    duration_seconds=4.0,
                    graph_calls=1,
                ),
                TaskResult(
                    task_id="task-3",
                    status=TaskStatus.PASSED,
                    tokens_used=30,
                    duration_seconds=2.0,
                ),
            ],
        )

        metrics = result.get_metrics()

        assert metrics["tasks_without_code_intelligence"] == 1
        assert metrics["code_intelligence_pass_rate"] == pytest.approx(0.5)
        assert metrics["non_code_intelligence_pass_rate"] == pytest.approx(1.0)
        assert metrics["code_intelligence_avg_tokens_per_task"] == pytest.approx(90.0)
        assert metrics["non_code_intelligence_avg_tokens_per_task"] == pytest.approx(30.0)
        assert metrics["code_intelligence_avg_duration_seconds"] == pytest.approx(6.0)
        assert metrics["non_code_intelligence_avg_duration_seconds"] == pytest.approx(2.0)
        assert metrics["code_intelligence_accepted_patch_rate"] == pytest.approx(0.5)
        assert metrics["non_code_intelligence_accepted_patch_rate"] == pytest.approx(0.0)


class TestPassAtKResult:
    """Tests for PassAtKResult."""

    def test_pass_rate(self):
        """Test pass rate calculation."""
        result = PassAtKResult(
            task_id="test1",
            total_samples=100,
            correct_samples=50,
            k_values=[1, 10],
            pass_at_k_scores={1: 0.5, 10: 0.99},
        )
        assert result.pass_rate == 0.5


class TestAnalyzerRegistry:
    """Tests for AnalyzerRegistry."""

    def setup_method(self):
        """Clear registry before each test."""
        AnalyzerRegistry.clear()

    def test_register_and_get(self):
        """Test registering and retrieving analyzers."""
        mock_analyzer = {"type": "mock"}
        AnalyzerRegistry.register("mock", mock_analyzer)

        retrieved = AnalyzerRegistry.get("mock")
        assert retrieved == mock_analyzer

    def test_get_nonexistent(self):
        """Test getting non-existent analyzer."""
        result = AnalyzerRegistry.get("nonexistent")
        assert result is None

    def test_register_factory(self):
        """Test lazy instantiation with factory."""
        factory_calls = [0]

        def factory():
            factory_calls[0] += 1
            return {"created": True}

        AnalyzerRegistry.register_factory("lazy", factory)

        # Factory not called yet
        assert factory_calls[0] == 0

        # Get instance triggers factory
        result = AnalyzerRegistry.get("lazy")
        assert factory_calls[0] == 1
        assert result == {"created": True}

        # Second get uses cached instance
        result2 = AnalyzerRegistry.get("lazy")
        assert factory_calls[0] == 1  # No second call
        assert result2 == result

    def test_get_code_quality_analyzer(self):
        """Test getting code quality analyzer singleton."""
        analyzer1 = AnalyzerRegistry.get_code_quality_analyzer()
        analyzer2 = AnalyzerRegistry.get_code_quality_analyzer()

        assert analyzer1 is analyzer2  # Same instance

    def test_clear(self):
        """Test clearing registry."""
        AnalyzerRegistry.register("test", {})
        assert AnalyzerRegistry.get("test") is not None

        AnalyzerRegistry.clear()
        assert AnalyzerRegistry.get("test") is None


class TestEstimateRequiredSamples:
    """Tests for sample estimation."""

    def test_estimate_basic(self):
        """Test basic sample estimation."""
        # To achieve 95% pass@10, we need some samples
        n = estimate_required_samples(target_pass_rate=0.95, k=10)
        assert n >= 10

    def test_estimate_returns_value(self):
        """Test estimation returns a reasonable value."""
        # The estimation searches for n where pass@k >= target
        n = estimate_required_samples(target_pass_rate=0.5, k=1)
        # With pass@1=0.5, we need c/n=0.5, so any n with c=n/2 works
        assert n >= 1


class TestGenerateReport:
    """Tests for report generation."""

    def test_generate_report(self):
        """Test report generation."""
        from victor.evaluation.pass_at_k import AggregatePassAtKResult

        result = AggregatePassAtKResult(
            total_tasks=10,
            k_values=[1, 10, 100],
            mean_pass_at_k={1: 0.3, 10: 0.8, 100: 0.99},
            task_results=[
                PassAtKResult(
                    task_id=f"task_{i}",
                    total_samples=100,
                    correct_samples=50,
                    k_values=[1, 10, 100],
                    pass_at_k_scores={1: 0.5, 10: 0.99, 100: 1.0},
                )
                for i in range(10)
            ],
        )

        report = generate_report(result)

        assert "PASS@K EVALUATION REPORT" in report
        assert "Pass@  1: 30.00%" in report
        assert "Pass@ 10: 80.00%" in report
        assert "Pass@100: 99.00%" in report
        assert "Total Tasks: 10" in report
