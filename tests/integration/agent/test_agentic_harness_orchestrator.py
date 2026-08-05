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

"""Integration tests for EvaluationOrchestrator.

These tests verify the end-to-end evaluation pipeline using mocked components
to avoid external dependencies (git, test runners, agent adapters).
"""

import asyncio
import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from victor.evaluation.agentic_harness import (
    AgenticExecutionTrace,
    FileEdit,
    EvalToolCall,
)
from victor.evaluation.baseline_validator import (
    BaselineStatus,
    TestBaseline,
)
from victor.evaluation.env_setup import SetupResult, SetupStrategy
from victor.evaluation.test_runners import Language
from victor.evaluation.evaluation_orchestrator import (
    EvaluationOrchestrator,
    EvaluationStage,
    EvaluationSummary,
    OrchestratorConfig,
    TaskProgress,
    run_swe_bench_evaluation,
)
from victor.evaluation.result_correlation import (
    CorrelationReport,
    SWEBenchScore,
)
from victor.evaluation.swe_bench_loader import SWEBenchInstance


# Helper to create test SWEBenchInstance
def create_test_instance(instance_id: str, repo: str = "test/repo") -> SWEBenchInstance:
    """Create a test SWE-bench instance."""
    return SWEBenchInstance(
        instance_id=instance_id,
        repo=repo,
        base_commit="abc123",
        problem_statement="Fix the bug in the function.",
        hints_text="Check the error handling.",
        patch="--- a/test.py\n+++ b/test.py\n@@ -1 +1 @@\n-old\n+new",
        test_patch="",
        fail_to_pass=["test_fix"],
        pass_to_pass=["test_existing"],
        created_at="2024-01-01T00:00:00Z",
        version="1.0",
    )


class TestOrchestratorConfig:
    """Tests for OrchestratorConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = OrchestratorConfig()
        assert config.dataset_path is None
        assert config.dataset_name == "princeton-nlp/SWE-bench_Lite"
        assert config.max_tasks == 0
        assert config.instance_ids == []
        assert config.repos == []
        assert config.agent_profile == "default"
        assert config.max_parallel == 1
        assert config.task_timeout == 1800
        assert config.continue_on_error is True
        assert config.save_traces is True
        assert config.save_patches is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = OrchestratorConfig(
            dataset_path=Path("/tmp/data.jsonl"),
            max_tasks=10,
            instance_ids=["django__django-12345"],
            repos=["django/django"],
            agent_profile="advanced",
            max_parallel=4,
            task_timeout=600,
            continue_on_error=False,
        )
        assert config.dataset_path == Path("/tmp/data.jsonl")
        assert config.max_tasks == 10
        assert config.instance_ids == ["django__django-12345"]
        assert config.agent_profile == "advanced"
        assert config.max_parallel == 4
        assert config.task_timeout == 600
        assert config.continue_on_error is False


class TestTaskProgress:
    """Tests for TaskProgress dataclass."""

    def test_default_progress(self):
        """Test default progress values."""
        progress = TaskProgress(instance_id="test-001")
        assert progress.instance_id == "test-001"
        assert progress.stage == EvaluationStage.LOADING
        assert progress.started_at is None
        assert progress.completed_at is None
        assert progress.error_message == ""
        assert progress.is_success is False
        assert progress.duration_seconds == 0.0

    def test_progress_duration(self):
        """Test duration calculation."""
        progress = TaskProgress(instance_id="test-001")
        progress.started_at = datetime(2024, 1, 1, 0, 0, 0)
        progress.completed_at = datetime(2024, 1, 1, 0, 0, 30)
        assert progress.duration_seconds == 30.0

    def test_progress_is_success(self):
        """Test success detection."""
        progress = TaskProgress(instance_id="test-001")
        assert progress.is_success is False

        progress.stage = EvaluationStage.COMPLETED
        assert progress.is_success is False  # No score yet

        progress.score = SWEBenchScore(
            instance_id="test-001",
            resolved=True,
            fail_to_pass_score=1.0,
            pass_to_pass_score=1.0,
        )
        assert progress.is_success is True

    def test_progress_to_dict(self):
        """Test progress serialization."""
        progress = TaskProgress(instance_id="test-001")
        progress.started_at = datetime(2024, 1, 1, 0, 0, 0)
        progress.stage = EvaluationStage.COMPLETED

        result = progress.to_dict()
        assert result["instance_id"] == "test-001"
        assert result["stage"] == "completed"
        assert result["started_at"] is not None
        assert "duration_seconds" in result


class TestEvaluationSummary:
    """Tests for EvaluationSummary dataclass."""

    def test_default_summary(self):
        """Test default summary values."""
        summary = EvaluationSummary()
        assert summary.total_tasks == 0
        assert summary.completed_tasks == 0
        assert summary.pass_rate == 0.0
        assert summary.avg_turns == 0.0

    def test_summary_duration(self):
        """Test summary duration calculation."""
        summary = EvaluationSummary()
        summary.started_at = datetime(2024, 1, 1, 0, 0, 0)
        summary.completed_at = datetime(2024, 1, 1, 0, 5, 0)
        assert summary.duration_seconds == 300.0

    def test_summary_to_dict(self):
        """Test summary serialization."""
        summary = EvaluationSummary(
            total_tasks=10,
            completed_tasks=8,
            successful_tasks=6,
            failed_tasks=2,
            pass_rate=0.75,
            avg_turns=5.0,
        )
        result = summary.to_dict()
        assert result["total_tasks"] == 10
        assert result["completed_tasks"] == 8
        assert result["pass_rate"] == 0.75

    def test_summary_to_text(self):
        """Test summary text generation."""
        summary = EvaluationSummary(
            total_tasks=10,
            completed_tasks=8,
            successful_tasks=6,
            failed_tasks=2,
            pass_rate=0.75,
        )
        text = summary.to_text()
        assert "Total Tasks: 10" in text
        assert "Pass Rate:" in text
        assert "75" in text


class TestEvaluationOrchestrator:
    """Tests for EvaluationOrchestrator class."""

    def test_orchestrator_initialization(self):
        """Test orchestrator initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = OrchestratorConfig(output_dir=Path(tmpdir))
            orchestrator = EvaluationOrchestrator(config)

            assert orchestrator.config == config
            assert orchestrator.progress_callback is None
            assert orchestrator._tasks == []
            assert orchestrator._progress == {}

    def test_orchestrator_with_callback(self):
        """Test orchestrator with progress callback."""
        callback_calls = []

        def callback(progress, stage):
            callback_calls.append((progress.instance_id, stage))

        with tempfile.TemporaryDirectory() as tmpdir:
            config = OrchestratorConfig(output_dir=Path(tmpdir))
            orchestrator = EvaluationOrchestrator(config, callback)

            assert orchestrator.progress_callback == callback

    @pytest.mark.asyncio
    async def test_run_single_task_success(self):
        """Test running a single task successfully with mocked components."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            config = OrchestratorConfig(
                output_dir=output_dir,
                max_tasks=1,
                save_traces=True,
                save_patches=True,
            )
            orchestrator = EvaluationOrchestrator(config)

            # Create test instance
            test_instance = create_test_instance("test-001")

            # Mock all the dependencies
            with patch.object(orchestrator, "_init_components"):
                # Mock loader
                orchestrator._loader = MagicMock()
                orchestrator._loader.load_instances_from_file.return_value = [test_instance]

                # Mock workspace manager
                mock_workspace = Path(tmpdir) / "workspace"
                mock_workspace.mkdir()
                orchestrator._workspace_manager = MagicMock()
                orchestrator._workspace_manager.setup_workspace = AsyncMock(
                    return_value=mock_workspace
                )
                orchestrator._workspace_manager.cleanup_workspace = AsyncMock()

                # Mock environment setup
                orchestrator._env_setup = MagicMock()
                orchestrator._env_setup.setup_environment = AsyncMock(
                    return_value=SetupResult(
                        success=True,
                        strategy=SetupStrategy.SYSTEM,
                        language=Language.PYTHON,
                    )
                )

                # Mock baseline validator
                mock_baseline = TestBaseline(
                    instance_id="test-001",
                    repo="test/repo",
                    base_commit="abc123",
                    fail_to_pass=["test_fix"],
                    pass_to_pass=["test_existing"],
                    status=BaselineStatus.VALID,
                )
                orchestrator._baseline_validator = MagicMock()
                orchestrator._baseline_validator.establish_baseline = AsyncMock(
                    return_value=mock_baseline
                )
                # Mock validate_changes to return a mock result
                mock_validation_result = MagicMock()
                mock_validation_result.instance_id = "test-001"
                mock_validation_result.baseline = mock_baseline
                mock_validation_result.status = BaselineStatus.VALID
                orchestrator._baseline_validator.validate_changes = AsyncMock(
                    return_value=mock_validation_result
                )

                # Mock test registry
                orchestrator._test_registry = MagicMock()

                # Mock correlator
                mock_score = SWEBenchScore(
                    instance_id="test-001",
                    resolved=True,
                    fail_to_pass_score=1.0,
                    pass_to_pass_score=1.0,
                )
                mock_report = CorrelationReport(
                    total_instances=1,
                    resolved_count=1,
                    scores=[mock_score],
                )
                orchestrator._correlator = MagicMock()
                orchestrator._correlator.compute_score.return_value = mock_score
                orchestrator._correlator.generate_report.return_value = mock_report

                # Mock agent adapter with patch
                mock_trace = AgenticExecutionTrace(
                    task_id="test-001",
                    start_time=0.0,
                    end_time=10.0,
                    turns=3,
                    tool_calls=[
                        EvalToolCall(
                            name="file_read",
                            arguments={"path": "test.py"},
                            success=True,
                        ),
                        EvalToolCall(
                            name="file_write",
                            arguments={"path": "test.py"},
                            success=True,
                        ),
                    ],
                    file_edits=[
                        FileEdit(path="test.py", action="modify", after_content="new content"),
                    ],
                    generated_patch="--- a/test.py\n+++ b/test.py\n@@ -1 +1 @@\n-old\n+new",
                )

                with patch(
                    "victor.evaluation.evaluation_orchestrator.VictorAgentAdapter"
                ) as MockAdapter:
                    mock_adapter = MagicMock()
                    mock_adapter.execute_task = AsyncMock(return_value=mock_trace)
                    MockAdapter.from_profile.return_value = mock_adapter

                    # Mock _apply_patch
                    with patch.object(
                        orchestrator, "_apply_patch", new_callable=AsyncMock
                    ) as mock_apply:
                        mock_apply.return_value = True
                        with patch.object(
                            orchestrator,
                            "_save_validated_session_feedback",
                        ) as mock_save_feedback:

                            # Set dataset path to avoid HuggingFace loading
                            orchestrator.config.dataset_path = Path(tmpdir) / "tasks.jsonl"

                            # Run single task
                            orchestrator._tasks = [test_instance]
                            orchestrator._progress["test-001"] = TaskProgress(
                                instance_id="test-001"
                            )

                            await orchestrator._run_single_task(test_instance)

                            # Verify task completed
                            progress = orchestrator._progress["test-001"]
                            assert progress.stage == EvaluationStage.COMPLETED
                            assert progress.score is not None
                            assert progress.execution_trace is not None
                            mock_save_feedback.assert_called_once()
                            save_args, save_kwargs = mock_save_feedback.call_args
                            assert save_args == ("test-001", mock_validation_result)
                            assert save_kwargs["score"] == mock_score
                            assert "metadata" in save_kwargs
                            assert "prompt_optimization" in save_kwargs["metadata"]

    @pytest.mark.asyncio
    async def test_run_task_environment_failure(self):
        """Test handling of environment setup failure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = OrchestratorConfig(output_dir=Path(tmpdir))
            orchestrator = EvaluationOrchestrator(config)

            test_instance = create_test_instance("test-001")

            with patch.object(orchestrator, "_init_components"):
                # Mock workspace manager
                mock_workspace = Path(tmpdir) / "workspace"
                mock_workspace.mkdir()
                orchestrator._workspace_manager = MagicMock()
                orchestrator._workspace_manager.setup_workspace = AsyncMock(
                    return_value=mock_workspace
                )
                orchestrator._workspace_manager.cleanup_workspace = AsyncMock()

                # Mock environment setup to fail
                orchestrator._env_setup = MagicMock()
                orchestrator._env_setup.setup_environment = AsyncMock(
                    return_value=SetupResult(
                        success=False,
                        strategy=SetupStrategy.SYSTEM,
                        error_message="Failed to install dependencies",
                    )
                )

                orchestrator._progress["test-001"] = TaskProgress(instance_id="test-001")

                with pytest.raises(RuntimeError, match="Environment setup failed"):
                    await orchestrator._run_single_task(test_instance)

                # Verify failure tracked
                assert orchestrator._summary.env_failures == 1

    @pytest.mark.asyncio
    async def test_run_task_baseline_failure(self):
        """Test handling of baseline establishment failure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = OrchestratorConfig(output_dir=Path(tmpdir))
            orchestrator = EvaluationOrchestrator(config)

            test_instance = create_test_instance("test-001")

            with patch.object(orchestrator, "_init_components"):
                # Mock workspace manager
                mock_workspace = Path(tmpdir) / "workspace"
                mock_workspace.mkdir()
                orchestrator._workspace_manager = MagicMock()
                orchestrator._workspace_manager.setup_workspace = AsyncMock(
                    return_value=mock_workspace
                )
                orchestrator._workspace_manager.cleanup_workspace = AsyncMock()

                # Mock environment setup success
                orchestrator._env_setup = MagicMock()
                orchestrator._env_setup.setup_environment = AsyncMock(
                    return_value=SetupResult(success=True, strategy=SetupStrategy.SYSTEM)
                )

                # Mock baseline validator to return invalid baseline
                mock_baseline = TestBaseline(
                    instance_id="test-001",
                    repo="test/repo",
                    base_commit="abc123",
                    fail_to_pass=["test_fix"],
                    pass_to_pass=["test_existing"],
                    status=BaselineStatus.INVALID,
                    error_message="Tests failed at baseline",
                )
                orchestrator._baseline_validator = MagicMock()
                orchestrator._baseline_validator.establish_baseline = AsyncMock(
                    return_value=mock_baseline
                )

                orchestrator._progress["test-001"] = TaskProgress(instance_id="test-001")

                with pytest.raises(RuntimeError, match="Baseline invalid"):
                    await orchestrator._run_single_task(test_instance)

                assert orchestrator._summary.baseline_failures == 1

    @pytest.mark.asyncio
    async def test_run_task_timeout(self):
        """Test handling of task timeout."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = OrchestratorConfig(output_dir=Path(tmpdir), task_timeout=1)
            orchestrator = EvaluationOrchestrator(config)

            test_instance = create_test_instance("test-001")

            with patch.object(orchestrator, "_init_components"):
                # Mock workspace manager
                mock_workspace = Path(tmpdir) / "workspace"
                mock_workspace.mkdir()
                orchestrator._workspace_manager = MagicMock()
                orchestrator._workspace_manager.setup_workspace = AsyncMock(
                    return_value=mock_workspace
                )
                orchestrator._workspace_manager.cleanup_workspace = AsyncMock()

                # Mock environment setup success
                orchestrator._env_setup = MagicMock()
                orchestrator._env_setup.setup_environment = AsyncMock(
                    return_value=SetupResult(success=True, strategy=SetupStrategy.SYSTEM)
                )

                # Mock baseline validator
                mock_baseline = TestBaseline(
                    instance_id="test-001",
                    repo="test/repo",
                    base_commit="abc123",
                    fail_to_pass=["test_fix"],
                    pass_to_pass=["test_existing"],
                    status=BaselineStatus.VALID,
                )
                orchestrator._baseline_validator = MagicMock()
                orchestrator._baseline_validator.establish_baseline = AsyncMock(
                    return_value=mock_baseline
                )

                # Mock agent adapter that times out
                async def slow_execute(*args, **kwargs):
                    await asyncio.sleep(10)  # Will timeout

                with patch(
                    "victor.evaluation.evaluation_orchestrator.VictorAgentAdapter"
                ) as MockAdapter:
                    mock_adapter = MagicMock()
                    mock_adapter.execute_task = slow_execute
                    MockAdapter.from_profile.return_value = mock_adapter

                    orchestrator._progress["test-001"] = TaskProgress(instance_id="test-001")

                    with pytest.raises(asyncio.TimeoutError):
                        await orchestrator._run_single_task(test_instance)

                    # Verify timeout tracked
                    progress = orchestrator._progress["test-001"]
                    assert "timed out" in progress.error_message.lower()
                    assert orchestrator._summary.execution_failures == 1

    @pytest.mark.asyncio
    async def test_run_sequential(self):
        """Test running tasks sequentially."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = OrchestratorConfig(
                output_dir=Path(tmpdir),
                max_parallel=1,
                continue_on_error=True,
            )
            orchestrator = EvaluationOrchestrator(config)

            instances = [
                create_test_instance("test-001"),
                create_test_instance("test-002"),
            ]
            orchestrator._tasks = instances

            # Track task execution order
            execution_order = []

            async def mock_run_single(task):
                execution_order.append(task.instance_id)
                # Initialize progress
                orchestrator._progress[task.instance_id] = TaskProgress(
                    instance_id=task.instance_id,
                    stage=EvaluationStage.COMPLETED,
                    started_at=datetime.now(),
                    completed_at=datetime.now(),
                )

            with patch.object(orchestrator, "_run_single_task", side_effect=mock_run_single):
                for task in instances:
                    orchestrator._progress[task.instance_id] = TaskProgress(
                        instance_id=task.instance_id
                    )

                await orchestrator._run_sequential()

                # Verify both tasks were executed
                assert len(execution_order) == 2
                assert execution_order == ["test-001", "test-002"]

    @pytest.mark.asyncio
    async def test_run_parallel(self):
        """Test running tasks in parallel."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = OrchestratorConfig(
                output_dir=Path(tmpdir),
                max_parallel=2,
            )
            orchestrator = EvaluationOrchestrator(config)

            instances = [
                create_test_instance("test-001"),
                create_test_instance("test-002"),
                create_test_instance("test-003"),
            ]
            orchestrator._tasks = instances

            execution_times = {}

            async def mock_run_single(task):
                execution_times[task.instance_id] = datetime.now()
                await asyncio.sleep(0.1)  # Simulate work
                orchestrator._progress[task.instance_id] = TaskProgress(
                    instance_id=task.instance_id,
                    stage=EvaluationStage.COMPLETED,
                )

            with patch.object(orchestrator, "_run_single_task", side_effect=mock_run_single):
                for task in instances:
                    orchestrator._progress[task.instance_id] = TaskProgress(
                        instance_id=task.instance_id
                    )

                await orchestrator._run_parallel()

                # All tasks should have been executed
                assert len(execution_times) == 3

    def test_get_progress(self):
        """Test getting task progress."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = OrchestratorConfig(output_dir=Path(tmpdir))
            orchestrator = EvaluationOrchestrator(config)

            orchestrator._progress["test-001"] = TaskProgress(
                instance_id="test-001",
                stage=EvaluationStage.AGENT_EXECUTION,
            )

            progress = orchestrator.get_progress("test-001")
            assert progress is not None
            assert progress.stage == EvaluationStage.AGENT_EXECUTION

            # Non-existent task
            assert orchestrator.get_progress("test-999") is None

    def test_get_summary(self):
        """Test getting evaluation summary."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = OrchestratorConfig(output_dir=Path(tmpdir))
            orchestrator = EvaluationOrchestrator(config)

            orchestrator._summary.total_tasks = 5
            orchestrator._summary.completed_tasks = 3

            summary = orchestrator.get_summary()
            assert summary.total_tasks == 5
            assert summary.completed_tasks == 3

    @pytest.mark.asyncio
    async def test_generate_report(self):
        """Test report generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = OrchestratorConfig(output_dir=Path(tmpdir))
            orchestrator = EvaluationOrchestrator(config)

            # Setup mock correlator
            mock_score = SWEBenchScore(
                instance_id="test-001",
                resolved=True,
                fail_to_pass_score=1.0,
                pass_to_pass_score=1.0,
            )
            mock_report = CorrelationReport(
                total_instances=1,
                resolved_count=1,
                scores=[mock_score],
            )
            orchestrator._correlator = MagicMock()
            orchestrator._correlator.generate_report.return_value = mock_report

            # Setup completed task
            trace = AgenticExecutionTrace(
                task_id="test-001",
                start_time=0.0,
                end_time=10.0,
                turns=5,
                tool_calls=[
                    EvalToolCall(name="file_read", arguments={}, success=True),
                    EvalToolCall(name="file_write", arguments={}, success=True),
                ],
            )

            orchestrator._progress["test-001"] = TaskProgress(
                instance_id="test-001",
                stage=EvaluationStage.COMPLETED,
                score=mock_score,
                execution_trace=trace,
            )

            report = orchestrator._generate_report()

            # Verify summary is updated
            assert orchestrator._summary.completed_tasks == 1
            assert orchestrator._summary.successful_tasks == 1
            assert report is not None

    @pytest.mark.asyncio
    async def test_save_trace_and_patch(self):
        """Test saving execution trace and patches."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            config = OrchestratorConfig(
                output_dir=output_dir,
                save_traces=True,
                save_patches=True,
            )
            orchestrator = EvaluationOrchestrator(config)
            output_dir.mkdir(exist_ok=True)

            trace = AgenticExecutionTrace(
                task_id="test-001",
                start_time=0.0,
                end_time=10.0,
                turns=3,
                generated_patch="--- a/test.py\n+++ b/test.py",
            )

            # Save trace
            orchestrator._save_trace("test-001", trace)
            trace_file = output_dir / "traces" / "test-001.json"
            assert trace_file.exists()
            trace_data = json.loads(trace_file.read_text())
            assert trace_data["task_id"] == "test-001"
            assert trace_data["turns"] == 3

            # Save patch
            orchestrator._save_patch("test-001", trace.generated_patch)
            patch_file = output_dir / "patches" / "test-001.patch"
            assert patch_file.exists()
            assert "--- a/test.py" in patch_file.read_text()

    @pytest.mark.asyncio
    async def test_save_summary(self):
        """Test saving evaluation summary."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            output_dir.mkdir(exist_ok=True)
            config = OrchestratorConfig(output_dir=output_dir)
            orchestrator = EvaluationOrchestrator(config)

            orchestrator._summary = EvaluationSummary(
                total_tasks=10,
                completed_tasks=8,
                successful_tasks=6,
                pass_rate=0.75,
            )

            orchestrator._save_summary()

            # Check JSON file
            summary_json = output_dir / "summary.json"
            assert summary_json.exists()
            data = json.loads(summary_json.read_text())
            assert data["total_tasks"] == 10
            assert data["pass_rate"] == 0.75

            # Check text file
            summary_txt = output_dir / "summary.txt"
            assert summary_txt.exists()
            text = summary_txt.read_text()
            assert "Total Tasks: 10" in text


class TestProgressCallback:
    """Tests for progress callback functionality."""

    @pytest.mark.asyncio
    async def test_callback_invocation(self):
        """Test that progress callback is invoked at each stage."""
        callback_calls = []

        def callback(progress, stage):
            callback_calls.append((progress.instance_id, stage.value))

        with tempfile.TemporaryDirectory() as tmpdir:
            config = OrchestratorConfig(output_dir=Path(tmpdir))
            orchestrator = EvaluationOrchestrator(config, callback)

            # Simulate notify progress
            progress = TaskProgress(instance_id="test-001")
            progress.stage = EvaluationStage.ENVIRONMENT_SETUP
            orchestrator._notify_progress(progress)

            progress.stage = EvaluationStage.AGENT_EXECUTION
            orchestrator._notify_progress(progress)

            assert len(callback_calls) == 2
            assert callback_calls[0] == ("test-001", "environment_setup")
            assert callback_calls[1] == ("test-001", "agent_execution")


class TestApplyPatch:
    """Tests for patch application."""

    @pytest.mark.asyncio
    async def test_apply_patch_creates_file(self):
        """Test that patch file is created and cleaned up."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            config = OrchestratorConfig(output_dir=workspace)
            orchestrator = EvaluationOrchestrator(config)

            patch_text = "--- a/test.py\n+++ b/test.py\n@@ -1 +1 @@\n-old\n+new"

            # Mock subprocess to avoid actual git apply
            with patch("asyncio.create_subprocess_exec") as mock_exec:
                mock_proc = MagicMock()
                mock_proc.communicate = AsyncMock(return_value=(b"", b""))
                mock_proc.returncode = 0
                mock_exec.return_value = mock_proc

                result = await orchestrator._apply_patch(workspace, patch_text)

                assert result is True
                # Verify patch file was cleaned up
                patch_file = workspace / ".agent_patch.diff"
                assert not patch_file.exists()


class TestEvaluationStage:
    """Tests for EvaluationStage enum."""

    def test_all_stages_defined(self):
        """Test all expected stages are defined."""
        stages = [
            EvaluationStage.LOADING,
            EvaluationStage.ENVIRONMENT_SETUP,
            EvaluationStage.BASELINE_ESTABLISHMENT,
            EvaluationStage.AGENT_EXECUTION,
            EvaluationStage.VALIDATION,
            EvaluationStage.CORRELATION,
            EvaluationStage.REPORTING,
            EvaluationStage.COMPLETED,
            EvaluationStage.FAILED,
        ]
        assert len(stages) == 9

    def test_stage_values(self):
        """Test stage string values."""
        assert EvaluationStage.LOADING.value == "loading"
        assert EvaluationStage.COMPLETED.value == "completed"
        assert EvaluationStage.FAILED.value == "failed"


class TestConvenienceFunction:
    """Tests for run_swe_bench_evaluation convenience function."""

    @pytest.mark.asyncio
    async def test_function_creates_orchestrator(self):
        """Test that convenience function creates and runs orchestrator."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Mock the orchestrator
            with patch(
                "victor.evaluation.evaluation_orchestrator.EvaluationOrchestrator"
            ) as MockOrch:
                mock_report = CorrelationReport(
                    total_instances=0,
                    resolved_count=0,
                    scores=[],
                )
                mock_instance = MagicMock()
                mock_instance.run_evaluation = AsyncMock(return_value=mock_report)
                MockOrch.return_value = mock_instance

                await run_swe_bench_evaluation(
                    dataset_path=Path(tmpdir) / "data.jsonl",
                    agent_profile="test",
                    output_dir=output_dir,
                    max_tasks=5,
                    max_parallel=2,
                )

                # Verify orchestrator was created with correct config
                MockOrch.assert_called_once()
                call_args = MockOrch.call_args
                config = call_args[0][0]
                assert config.agent_profile == "test"
                assert config.max_tasks == 5
                assert config.max_parallel == 2

                mock_instance.run_evaluation.assert_called_once()
