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

"""Tests for agentic benchmark harness."""

import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from victor.evaluation.agentic_harness import (
    AgenticExecutionTrace,
    AgenticMetrics,
    AgenticTaskResult,
    AgenticValidationType,
    AgenticValidator,
    FileEdit,
    FileEditValidator,
    GenerativeOptimizationAssessment,
    GenerativeOptimizationFeasibilityGate,
    PatchApplicationValidator,
    TestPassingValidator,
    EvalToolCall,
    ToolUsageValidator,
    generate_agentic_report,
)
from victor.evaluation.protocol import (
    BenchmarkFailureCategory,
    BenchmarkTask,
    BenchmarkType,
    TaskStatus,
)


class TestEvalToolCall:
    """Tests for EvalToolCall dataclass."""

    def test_default_values(self):
        """Test EvalToolCall with default values."""
        call = EvalToolCall(name="file_read", arguments={"path": "/test.py"})
        assert call.name == "file_read"
        assert call.arguments == {"path": "/test.py"}
        assert call.result is None
        assert call.success is True
        assert call.timestamp == 0.0

    def test_with_result(self):
        """Test EvalToolCall with result."""
        call = EvalToolCall(
            name="file_write",
            arguments={"path": "/test.py", "content": "print('hello')"},
            result="File written successfully",
            success=True,
            timestamp=time.time(),
        )
        assert call.name == "file_write"
        assert call.result == "File written successfully"
        assert call.success is True

    def test_failed_call(self):
        """Test failed EvalToolCall."""
        call = EvalToolCall(
            name="file_read",
            arguments={"path": "/nonexistent.py"},
            result="File not found",
            success=False,
        )
        assert call.success is False


class TestFileEdit:
    """Tests for FileEdit dataclass."""

    def test_create_action(self):
        """Test FileEdit with create action."""
        edit = FileEdit(
            path="src/new_file.py",
            action="create",
            after_content="print('new file')",
        )
        assert edit.action == "create"
        assert edit.before_content == ""
        assert edit.after_content == "print('new file')"

    def test_modify_action(self):
        """Test FileEdit with modify action."""
        edit = FileEdit(
            path="src/existing.py",
            action="modify",
            before_content="old content",
            after_content="new content",
            diff="--- a/src/existing.py\n+++ b/src/existing.py\n",
        )
        assert edit.action == "modify"
        assert edit.before_content == "old content"
        assert edit.after_content == "new content"
        assert "---" in edit.diff

    def test_delete_action(self):
        """Test FileEdit with delete action."""
        edit = FileEdit(
            path="src/deprecated.py",
            action="delete",
            before_content="content to delete",
        )
        assert edit.action == "delete"
        assert edit.after_content == ""


class TestAgenticExecutionTrace:
    """Tests for AgenticExecutionTrace dataclass."""

    def test_default_values(self):
        """Test trace with default values."""
        trace = AgenticExecutionTrace(task_id="test-001", start_time=100.0)
        assert trace.task_id == "test-001"
        assert trace.tool_calls == []
        assert trace.file_edits == []
        assert trace.messages == []
        assert trace.turns == 0
        assert trace.generated_patch == ""
        assert trace.validation_errors == {}
        assert trace.completion_signals == {}
        assert trace.topology_events == []

    def test_duration_calculation(self):
        """Test trace duration calculation."""
        trace = AgenticExecutionTrace(
            task_id="test-001",
            start_time=100.0,
            end_time=150.0,
        )
        assert trace.duration_seconds == 50.0

    def test_with_tool_calls(self):
        """Test trace with tool calls."""
        calls = [
            EvalToolCall(name="file_read", arguments={"path": "a.py"}),
            EvalToolCall(name="file_write", arguments={"path": "b.py"}),
        ]
        trace = AgenticExecutionTrace(task_id="test-001", start_time=0.0, tool_calls=calls)
        assert len(trace.tool_calls) == 2
        assert trace.total_tool_calls == 2

    def test_with_file_edits(self):
        """Test trace with file edits."""
        edits = [
            FileEdit(path="a.py", action="modify"),
            FileEdit(path="b.py", action="create"),
        ]
        trace = AgenticExecutionTrace(task_id="test-001", start_time=0.0, file_edits=edits)
        assert len(trace.file_edits) == 2
        assert trace.files_modified == ["a.py", "b.py"]

    def test_total_turns_property(self):
        """Test total_turns is an alias for turns."""
        trace = AgenticExecutionTrace(task_id="test-001", start_time=0.0, turns=5)
        assert trace.turns == 5
        assert trace.total_turns == 5
        assert trace.total_turns == trace.turns

    def test_to_dict(self):
        """Test to_dict serialization method."""
        calls = [
            EvalToolCall(
                name="file_read",
                arguments={"path": "a.py"},
                result="content",
                success=True,
            ),
        ]
        edits = [
            FileEdit(path="a.py", action="modify", before_content="old", after_content="new"),
        ]
        trace = AgenticExecutionTrace(
            task_id="test-001",
            start_time=100.0,
            end_time=150.0,
            turns=3,
            tool_calls=calls,
            file_edits=edits,
            generated_patch="--- a/a.py\n+++ b/a.py",
            validations={"patch_applies": True},
            validation_errors={"tests_pass": "Some tests failed"},
            topology_events=[
                {
                    "action": "team_plan",
                    "topology": "team",
                    "execution_mode": "team_execution",
                    "confidence": 0.83,
                }
            ],
        )
        result = trace.to_dict()

        assert result["task_id"] == "test-001"
        assert result["start_time"] == 100.0
        assert result["end_time"] == 150.0
        assert result["duration_seconds"] == 50.0
        assert result["turns"] == 3
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["name"] == "file_read"
        assert len(result["file_edits"]) == 1
        assert result["file_edits"][0]["path"] == "a.py"
        assert result["generated_patch"] == "--- a/a.py\n+++ b/a.py"
        assert result["validations"]["patch_applies"] is True
        assert "tests_pass" in result["validation_errors"]
        assert result["total_tool_calls"] == 1
        assert result["successful_tool_calls"] == 1
        assert result["topology_events"][0]["action"] == "team_plan"
        assert result["files_modified"] == ["a.py"]

    def test_to_dict_includes_benchmark_metadata(self):
        """Test trace export includes benchmark metadata for reporting."""
        trace = AgenticExecutionTrace(
            task_id="test-003",
            start_time=0.0,
            benchmark=BenchmarkType.CLAW_BENCH.value,
            benchmark_source="Research",
        )
        result = trace.to_dict()

        assert result["benchmark"] == "claw_bench"
        assert result["benchmark_source"] == "Research"

    def test_to_dict_empty_trace(self):
        """Test to_dict with minimal trace."""
        trace = AgenticExecutionTrace(task_id="test-002", start_time=0.0)
        result = trace.to_dict()

        assert result["task_id"] == "test-002"
        assert result["tool_calls"] == []
        assert result["file_edits"] == []
        assert result["generated_patch"] == ""
        assert result["total_tool_calls"] == 0


class TestAgenticTaskResult:
    """Tests for AgenticTaskResult dataclass."""

    def test_success_result(self):
        """Test successful task result."""
        trace = AgenticExecutionTrace(task_id="test-001", start_time=0.0)
        result = AgenticTaskResult(
            task_id="test-001",
            status=TaskStatus.PASSED,
            trace=trace,
            test_score=1.0,
        )
        assert result.is_success is True
        assert result.test_score == 1.0

    def test_failure_result(self):
        """Test failed task result."""
        trace = AgenticExecutionTrace(task_id="test-001", start_time=0.0)
        result = AgenticTaskResult(
            task_id="test-001",
            status=TaskStatus.FAILED,
            trace=trace,
            patch_score=0.0,
            error_message="Patch failed to apply",
        )
        assert result.is_success is False
        assert result.error_message == "Patch failed to apply"

    def test_calculate_overall_score(self):
        """Test overall score calculation."""
        trace = AgenticExecutionTrace(task_id="test-001", start_time=0.0)
        result = AgenticTaskResult(
            task_id="test-001",
            status=TaskStatus.PASSED,
            trace=trace,
            patch_score=1.0,
            test_score=0.8,
            edit_accuracy=0.9,
            tool_efficiency=0.7,
        )
        # Default weights: patch=0.3, test=0.4, edit=0.2, tool=0.1
        score = result.calculate_overall_score()
        expected = 1.0 * 0.3 + 0.8 * 0.4 + 0.9 * 0.2 + 0.7 * 0.1
        assert abs(score - expected) < 0.001

    def test_evaluate_generative_optimization_feasible_reward(self):
        """PR2 optimization assessment should produce feasible continuous reward."""
        trace = AgenticExecutionTrace(
            task_id="test-001",
            start_time=0.0,
            validations={
                AgenticValidationType.PATCH_APPLIES.value: True,
                AgenticValidationType.TESTS_PASS.value: True,
            },
        )
        result = AgenticTaskResult(
            task_id="test-001",
            status=TaskStatus.PASSED,
            trace=trace,
            overall_score=0.8,
            completion_precision=0.6,
        )

        assessment = result.evaluate_generative_optimization()

        assert isinstance(assessment, GenerativeOptimizationAssessment)
        assert assessment.feasible is True
        assert assessment.feasibility_failures == []
        assert assessment.reward == pytest.approx(0.77)
        assert assessment.reward_components["topology_reward"] == pytest.approx(0.8)

    def test_evaluate_generative_optimization_tracks_gate_failures(self):
        """Optimization feasibility should expose the exact hard-constraint failures."""
        trace = AgenticExecutionTrace(
            task_id="test-002",
            start_time=0.0,
            validations={
                AgenticValidationType.PATCH_APPLIES.value: True,
                AgenticValidationType.TESTS_PASS.value: False,
            },
        )
        result = AgenticTaskResult(
            task_id="test-002",
            status=TaskStatus.FAILED,
            trace=trace,
            overall_score=0.45,
            completion_precision=0.3,
            unsupported_claim_rate=0.4,
        )

        assessment = result.evaluate_generative_optimization(
            gate=GenerativeOptimizationFeasibilityGate(max_unsupported_claim_rate=0.25)
        )

        assert assessment.feasible is False
        assert "tests_pass" in assessment.feasibility_failures
        assert "unsupported_claim_rate" in assessment.feasibility_failures
        assert assessment.reward > 0.0


class TestAgenticMetrics:
    """Tests for AgenticMetrics dataclass."""

    def test_default_metrics(self):
        """Test default metric values."""
        metrics = AgenticMetrics()
        assert metrics.total_tasks == 0
        assert metrics.passed == 0
        assert metrics.failed == 0
        assert metrics.total_tool_calls == 0
        assert metrics.total_turns == 0
        assert metrics.pass_rate == 0.0
        assert metrics.avg_tool_calls == 0.0
        assert metrics.avg_turns == 0.0
        assert metrics.avg_completion_precision == 0.0
        assert metrics.avg_unsupported_claim_rate == 0.0

    def test_pass_rate(self):
        """Test pass rate calculation."""
        metrics = AgenticMetrics(total_tasks=10, passed=7, failed=3)
        assert metrics.pass_rate == 0.7

    def test_averages(self):
        """Test average calculations."""
        metrics = AgenticMetrics(
            total_tasks=4,
            total_tool_calls=20,
            total_turns=12,
        )
        assert metrics.avg_tool_calls == 5.0
        assert metrics.avg_turns == 3.0

    def test_to_dict(self):
        """Test metrics export to dict."""
        metrics = AgenticMetrics(
            total_tasks=2,
            passed=1,
            failed=1,
            avg_completion_precision=0.75,
            avg_unsupported_claim_rate=0.1,
        )
        d = metrics.to_dict()
        assert "summary" in d
        assert d["summary"]["total_tasks"] == 2
        assert d["summary"]["pass_rate"] == 0.5
        assert d["quality"]["avg_completion_precision"] == 0.75
        assert d["quality"]["avg_unsupported_claim_rate"] == 0.1

    def test_to_dict_includes_optimization_metrics(self):
        """Agentic metrics should export PR2 feasibility and reward summaries."""
        feasible_trace = AgenticExecutionTrace(
            task_id="test-007",
            start_time=0.0,
            validations={
                AgenticValidationType.PATCH_APPLIES.value: True,
                AgenticValidationType.TESTS_PASS.value: True,
            },
        )
        feasible_result = AgenticTaskResult(
            task_id="test-007",
            status=TaskStatus.PASSED,
            trace=feasible_trace,
            benchmark=BenchmarkType.SWE_BENCH,
            overall_score=0.8,
            completion_precision=0.6,
        )
        infeasible_trace = AgenticExecutionTrace(
            task_id="test-008",
            start_time=0.0,
            validations={
                AgenticValidationType.PATCH_APPLIES.value: True,
                AgenticValidationType.TESTS_PASS.value: False,
            },
        )
        infeasible_result = AgenticTaskResult(
            task_id="test-008",
            status=TaskStatus.FAILED,
            trace=infeasible_trace,
            benchmark=BenchmarkType.SWE_BENCH,
            overall_score=0.4,
            completion_precision=0.2,
            unsupported_claim_rate=0.4,
        )
        metrics = AgenticMetrics(
            total_tasks=2,
            passed=1,
            failed=1,
            task_results=[feasible_result, infeasible_result],
        )

        exported = metrics.to_dict()

        assert exported["summary"]["optimization_feasible_tasks"] == 1
        assert exported["summary"]["optimization_infeasible_tasks"] == 1
        assert exported["summary"]["optimization_feasibility_rate"] == 0.5
        assert exported["quality"]["avg_optimization_reward"] == pytest.approx(0.57)
        assert exported["optimization"]["gate_failures"] == {
            "tests_pass": 1,
            "unsupported_claim_rate": 1,
        }
        assert exported["tasks"][0]["optimization_summary"]["feasible"] is True
        assert exported["tasks"][1]["optimization_summary"]["feasible"] is False

    def test_to_dict_includes_external_benchmark_summary(self):
        """Test external benchmark coverage is surfaced in metrics export."""
        trace = AgenticExecutionTrace(
            task_id="test-004",
            start_time=0.0,
            benchmark=BenchmarkType.CLAW_BENCH.value,
        )
        result = AgenticTaskResult(
            task_id="test-004",
            status=TaskStatus.PASSED,
            trace=trace,
            benchmark=BenchmarkType.CLAW_BENCH,
        )
        metrics = AgenticMetrics(total_tasks=1, passed=1, task_results=[result])

        exported = metrics.to_dict()
        assert exported["summary"]["benchmarks_evaluated"] == ["claw_bench"]
        assert exported["summary"]["external_benchmark_tasks"] == 1
        assert exported["tasks"][0]["benchmark"] == "claw_bench"

    def test_to_dict_includes_failure_taxonomy(self):
        """Test normalized failure categories are aggregated in agentic metrics."""
        trace = AgenticExecutionTrace(
            task_id="test-005",
            start_time=0.0,
            benchmark=BenchmarkType.GUIDE.value,
        )
        result = AgenticTaskResult(
            task_id="test-005",
            status=TaskStatus.FAILED,
            trace=trace,
            benchmark=BenchmarkType.GUIDE,
            failure_category=BenchmarkFailureCategory.TEST_FAILURE,
        )
        metrics = AgenticMetrics(total_tasks=1, failed=1, task_results=[result])

        exported = metrics.to_dict()
        assert exported["summary"]["failure_categories"] == {"test_failure": 1}
        assert exported["tasks"][0]["failure_category"] == "test_failure"

    def test_to_dict_includes_topology_metrics(self):
        """Topology coverage and reward should be surfaced for agentic tasks."""
        trace = AgenticExecutionTrace(
            task_id="test-006",
            start_time=0.0,
            benchmark=BenchmarkType.SWE_BENCH.value,
            turns=3,
            tool_calls=[EvalToolCall(name="code_search", arguments={}, success=True)],
            topology_events=[
                {
                    "action": "team_plan",
                    "topology": "team",
                    "execution_mode": "team_execution",
                    "formation": "parallel",
                    "confidence": 0.84,
                }
            ],
        )
        result = AgenticTaskResult(
            task_id="test-006",
            status=TaskStatus.PASSED,
            trace=trace,
            benchmark=BenchmarkType.SWE_BENCH,
            overall_score=0.8,
        )
        metrics = AgenticMetrics(total_tasks=1, passed=1, task_results=[result])

        exported = metrics.to_dict()

        assert exported["summary"]["topology_feedback_coverage"] == 1.0
        assert exported["quality"]["avg_topology_reward"] >= 0.0
        assert exported["topology"]["topology_actions"] == {"team_plan": 1}
        assert exported["tasks"][0]["topology_summary"]["final_execution_mode"] == (
            "team_execution"
        )

    def test_generate_agentic_report_includes_optimization_summary(self):
        """Human-readable reports should surface PR2 optimization metrics."""
        trace = AgenticExecutionTrace(
            task_id="test-009",
            start_time=0.0,
            validations={
                AgenticValidationType.PATCH_APPLIES.value: True,
                AgenticValidationType.TESTS_PASS.value: True,
            },
        )
        result = AgenticTaskResult(
            task_id="test-009",
            status=TaskStatus.PASSED,
            trace=trace,
            benchmark=BenchmarkType.SWE_BENCH,
            overall_score=0.8,
            completion_precision=0.6,
        )
        metrics = AgenticMetrics(total_tasks=1, passed=1, task_results=[result])

        report = generate_agentic_report(metrics)

        assert "Feasibility:" in report
        assert "Opt Reward:" in report


class TestPatchApplicationValidator:
    """Tests for PatchApplicationValidator."""

    def test_validation_type(self):
        """Test validation type property."""
        validator = PatchApplicationValidator()
        assert validator.validation_type == AgenticValidationType.PATCH_APPLIES

    @pytest.mark.asyncio
    async def test_empty_patch_with_file_edits(self):
        """Test validation with empty patch but file edits present."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            trace = AgenticExecutionTrace(
                task_id="test-001",
                start_time=0.0,
                generated_patch="",
                file_edits=[FileEdit(path="test.py", action="modify")],
            )
            task = BenchmarkTask(
                task_id="test-001",
                benchmark=BenchmarkType.SWE_BENCH,
                description="Fix the function",
                prompt="Fix the function",
            )

            validator = PatchApplicationValidator()
            success, message, score = await validator.validate(task, trace, workspace)
            assert success is True
            assert score == 1.0

    @pytest.mark.asyncio
    async def test_empty_patch_no_edits(self):
        """Test validation with empty patch and no file edits."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            trace = AgenticExecutionTrace(task_id="test-001", start_time=0.0, generated_patch="")
            task = BenchmarkTask(
                task_id="test-001",
                benchmark=BenchmarkType.SWE_BENCH,
                description="Fix the function",
                prompt="Fix the function",
            )

            validator = PatchApplicationValidator()
            success, message, score = await validator.validate(task, trace, workspace)
            assert success is False
            assert score == 0.0
            assert "no patch" in message.lower()


class TestFileEditValidator:
    """Tests for FileEditValidator."""

    def test_validation_type(self):
        """Test validation type property."""
        validator = FileEditValidator()
        assert validator.validation_type == AgenticValidationType.FILE_EDITS

    @pytest.mark.asyncio
    async def test_single_file_edit(self):
        """Test validation with single file edit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)

            trace = AgenticExecutionTrace(
                task_id="test-001",
                start_time=0.0,
                file_edits=[FileEdit(path="test.py", action="modify", after_content="new content")],
            )
            task = BenchmarkTask(
                task_id="test-001",
                benchmark=BenchmarkType.SWE_BENCH,
                description="Fix the function",
                prompt="Fix the function",
            )

            validator = FileEditValidator()
            success, message, score = await validator.validate(task, trace, workspace)
            assert success is True
            assert score >= 0.5

    @pytest.mark.asyncio
    async def test_no_file_edits(self):
        """Test validation with no file edits."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            trace = AgenticExecutionTrace(task_id="test-001", start_time=0.0, file_edits=[])
            task = BenchmarkTask(
                task_id="test-001",
                benchmark=BenchmarkType.SWE_BENCH,
                description="Fix the function",
                prompt="Fix the function",
            )

            validator = FileEditValidator()
            success, message, score = await validator.validate(task, trace, workspace)
            assert success is False
            assert score == 0.0


class TestToolUsageValidator:
    """Tests for ToolUsageValidator."""

    def test_validation_type(self):
        """Test validation type property."""
        validator = ToolUsageValidator()
        assert validator.validation_type == AgenticValidationType.TOOL_USAGE

    def test_expected_tools_dict(self):
        """Test that expected tools dictionary exists."""
        assert hasattr(ToolUsageValidator, "EXPECTED_TOOLS")
        expected = ToolUsageValidator.EXPECTED_TOOLS
        assert "file_edit" in expected
        assert "code_search" in expected
        assert "test_run" in expected
        assert "git" in expected

    @pytest.mark.asyncio
    async def test_with_file_edit_tools(self):
        """Test validation when file edit tools are used."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            trace = AgenticExecutionTrace(
                task_id="test-001",
                start_time=0.0,
                tool_calls=[
                    EvalToolCall(name="file_write", arguments={"path": "test.py"}, success=True),
                    EvalToolCall(name="edit_file", arguments={"path": "test.py"}, success=True),
                ],
            )
            task = BenchmarkTask(
                task_id="test-001",
                benchmark=BenchmarkType.SWE_BENCH,
                description="Fix the function",
                prompt="Fix the function",
            )

            validator = ToolUsageValidator()
            success, message, score = await validator.validate(task, trace, workspace)
            # Should succeed - using expected file edit tools
            assert success is True
            assert score > 0.0

    @pytest.mark.asyncio
    async def test_with_no_tool_calls(self):
        """Test validation when no tools were called."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            trace = AgenticExecutionTrace(
                task_id="test-001",
                start_time=0.0,
                tool_calls=[],  # No tool calls
            )
            task = BenchmarkTask(
                task_id="test-001",
                benchmark=BenchmarkType.SWE_BENCH,
                description="Fix the function",
                prompt="Fix the function",
            )

            validator = ToolUsageValidator()
            success, message, score = await validator.validate(task, trace, workspace)
            # Score should reflect lack of tool usage
            assert score >= 0.0  # At least non-negative

    @pytest.mark.asyncio
    async def test_without_file_edit_tools(self):
        """Test validation when no file edit tools are used.

        The validator expects file edit tools for SWE-bench tasks.
        Using only code search tools should return success=False.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            trace = AgenticExecutionTrace(
                task_id="test-001",
                start_time=0.0,
                tool_calls=[
                    EvalToolCall(name="code_search", arguments={"query": "func"}, success=True),
                    EvalToolCall(name="grep", arguments={"pattern": "def"}, success=True),
                ],
            )
            task = BenchmarkTask(
                task_id="test-001",
                benchmark=BenchmarkType.SWE_BENCH,
                description="Find the function",
                prompt="Find the function",
            )

            validator = ToolUsageValidator()
            success, message, score = await validator.validate(task, trace, workspace)
            # Should fail because no file editing tools were used
            assert success is False
            assert "file edit" in message.lower()
            assert score >= 0.0  # Still gets partial score


class TestTestPassingValidator:
    """Tests for TestPassingValidator."""

    def test_validation_type(self):
        """Test validation type property."""
        validator = TestPassingValidator()
        assert validator.validation_type == AgenticValidationType.TESTS_PASS

    @pytest.mark.asyncio
    async def test_no_test_runner_detected(self):
        """Test validation when no test runner is detected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            # Create file but no test infrastructure
            (workspace / "main.py").write_text("print('hello')")

            trace = AgenticExecutionTrace(task_id="test-001", start_time=0.0)
            task = BenchmarkTask(
                task_id="test-001",
                benchmark=BenchmarkType.SWE_BENCH,
                description="Fix the function",
                prompt="Fix the function",
            )

            validator = TestPassingValidator()
            success, message, score = await validator.validate(task, trace, workspace)
            # Should succeed with default score when no tests
            assert success is True
            assert score >= 0.5

    @pytest.mark.asyncio
    async def test_with_mocked_test_runner(self):
        """Test validation with mocked test runner."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)

            trace = AgenticExecutionTrace(task_id="test-001", start_time=0.0)
            task = BenchmarkTask(
                task_id="test-001",
                benchmark=BenchmarkType.SWE_BENCH,
                description="Fix the function",
                prompt="Fix the function",
            )

            # Mock the test runner to avoid actual test execution
            validator = TestPassingValidator(timeout_seconds=10)
            with patch.object(validator, "_registry") as mock_registry:
                mock_runner = MagicMock()
                mock_results = MagicMock()
                mock_results.success_rate = 1.0
                mock_results.passed = 1
                mock_results.failed = 0
                mock_results.total = 1
                mock_results.all_passed = True
                mock_results.error_message = ""
                mock_results.skipped = 0
                mock_results.errors = 0
                mock_runner.run_tests = AsyncMock(return_value=mock_results)
                mock_runner.language = MagicMock()
                mock_registry.detect_runner.return_value = mock_runner

                success, message, score = await validator.validate(task, trace, workspace)
                assert success is True
                assert score == 1.0


class TestAgenticValidationType:
    """Tests for AgenticValidationType enum."""

    def test_validation_types(self):
        """Test all validation type values exist."""
        assert AgenticValidationType.PATCH_APPLIES.value == "patch_applies"
        assert AgenticValidationType.TESTS_PASS.value == "tests_pass"
        assert AgenticValidationType.FILE_EDITS.value == "file_edits"
        assert AgenticValidationType.TOOL_USAGE.value == "tool_usage"
        assert AgenticValidationType.SEMANTIC_MATCH.value == "semantic_match"
        assert AgenticValidationType.TASK_COMPLETE.value == "task_complete"


class TestValidatorAbstractClass:
    """Tests for AgenticValidator abstract base class."""

    def test_custom_validator(self):
        """Test creating a custom validator."""

        class CustomValidator(AgenticValidator):
            @property
            def validation_type(self) -> AgenticValidationType:
                return AgenticValidationType.TASK_COMPLETE

            async def validate(
                self, task: BenchmarkTask, trace: AgenticExecutionTrace, workspace: Path
            ) -> tuple[bool, str, float]:
                return True, "Custom validation passed", 1.0

        validator = CustomValidator()
        assert hasattr(validator, "validate")
        assert validator.validation_type == AgenticValidationType.TASK_COMPLETE

    @pytest.mark.asyncio
    async def test_custom_validator_execution(self):
        """Test executing a custom validator."""

        class CustomValidator(AgenticValidator):
            @property
            def validation_type(self) -> AgenticValidationType:
                return AgenticValidationType.TASK_COMPLETE

            async def validate(
                self, task: BenchmarkTask, trace: AgenticExecutionTrace, workspace: Path
            ) -> tuple[bool, str, float]:
                # Check if task ID matches trace
                if task.task_id == trace.task_id:
                    return True, "IDs match", 1.0
                return False, "ID mismatch", 0.0

        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            task = BenchmarkTask(
                task_id="test-001",
                benchmark=BenchmarkType.SWE_BENCH,
                description="Test",
                prompt="Test",
            )
            trace = AgenticExecutionTrace(task_id="test-001", start_time=0.0)

            validator = CustomValidator()
            success, message, score = await validator.validate(task, trace, workspace)
            assert success is True
            assert score == 1.0
