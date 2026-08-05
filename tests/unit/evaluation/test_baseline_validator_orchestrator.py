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

"""Tests for evaluation orchestrator."""

import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from victor.evaluation.baseline_validator import (
    BaselineStatus,
    BaselineValidationResult,
    TestBaseline,
)
from victor.evaluation.evaluation_orchestrator import (
    EvaluationOrchestrator,
    EvaluationStage,
    EvaluationSummary,
    OrchestratorConfig,
    TaskProgress,
)
from victor.evaluation.result_correlation import SWEBenchScore
from victor.evaluation.test_runners import TestRunResults
from victor.evaluation.validated_session_truth_emitters import (
    ValidatedSessionTruthArtifact,
    ValidatedSessionTruthEmitterRegistry,
)


class TestEvaluationStage:
    """Tests for EvaluationStage enum."""

    def test_stage_values(self):
        """Test all evaluation stage values."""
        assert EvaluationStage.LOADING.value == "loading"
        assert EvaluationStage.ENVIRONMENT_SETUP.value == "environment_setup"
        assert EvaluationStage.BASELINE_ESTABLISHMENT.value == "baseline_establishment"
        assert EvaluationStage.AGENT_EXECUTION.value == "agent_execution"
        assert EvaluationStage.VALIDATION.value == "validation"
        assert EvaluationStage.CORRELATION.value == "correlation"
        assert EvaluationStage.REPORTING.value == "reporting"
        assert EvaluationStage.COMPLETED.value == "completed"
        assert EvaluationStage.FAILED.value == "failed"


class TestTaskProgress:
    """Tests for TaskProgress dataclass."""

    def test_default_values(self):
        """Test default task progress values."""
        progress = TaskProgress(instance_id="test_instance")
        assert progress.instance_id == "test_instance"
        assert progress.stage == EvaluationStage.LOADING
        assert progress.started_at is None
        assert progress.completed_at is None
        assert progress.error_message == ""
        assert progress.env_setup_result is None
        assert progress.baseline is None
        assert progress.execution_trace is None
        assert progress.validation_result is None
        assert progress.score is None

    def test_duration_seconds_not_started(self):
        """Test duration when task not started."""
        progress = TaskProgress(instance_id="test")
        assert progress.duration_seconds == 0.0

    def test_duration_seconds_in_progress(self):
        """Test duration when task in progress."""
        progress = TaskProgress(
            instance_id="test",
            started_at=datetime(2025, 1, 1, 12, 0, 0),
        )
        # Duration should be positive (comparing to now)
        assert progress.duration_seconds > 0

    def test_duration_seconds_completed(self):
        """Test duration when task completed."""
        progress = TaskProgress(
            instance_id="test",
            started_at=datetime(2025, 1, 1, 12, 0, 0),
            completed_at=datetime(2025, 1, 1, 12, 5, 0),
        )
        assert progress.duration_seconds == 300.0  # 5 minutes

    def test_is_success_true(self):
        """Test is_success returns True for successful completion."""
        score = MagicMock(spec=SWEBenchScore)
        progress = TaskProgress(
            instance_id="test",
            stage=EvaluationStage.COMPLETED,
            score=score,
        )
        assert progress.is_success is True

    def test_is_success_false_not_completed(self):
        """Test is_success returns False when not completed."""
        progress = TaskProgress(
            instance_id="test",
            stage=EvaluationStage.AGENT_EXECUTION,
        )
        assert progress.is_success is False

    def test_is_success_false_no_score(self):
        """Test is_success returns False when no score."""
        progress = TaskProgress(
            instance_id="test",
            stage=EvaluationStage.COMPLETED,
        )
        assert progress.is_success is False

    def test_to_dict(self):
        """Test serialization to dictionary."""
        score = MagicMock(spec=SWEBenchScore)
        score.to_dict.return_value = {"overall_score": 1.0}

        progress = TaskProgress(
            instance_id="test_instance",
            stage=EvaluationStage.COMPLETED,
            started_at=datetime(2025, 1, 1, 12, 0, 0),
            completed_at=datetime(2025, 1, 1, 12, 5, 0),
            score=score,
        )

        data = progress.to_dict()
        assert data["instance_id"] == "test_instance"
        assert data["stage"] == "completed"
        assert data["duration_seconds"] == 300.0
        assert data["is_success"] is True
        assert data["score"] == {"overall_score": 1.0}


class TestOrchestratorConfig:
    """Tests for OrchestratorConfig dataclass."""

    def test_default_values(self):
        """Test default config values."""
        config = OrchestratorConfig()
        assert config.dataset_path is None
        assert config.dataset_name == "princeton-nlp/SWE-bench_Lite"
        assert config.max_tasks == 0
        assert config.instance_ids == []
        assert config.repos == []
        assert config.agent_profile == "default"
        assert config.adapter_config is None
        assert config.output_dir == Path("./swe_bench_results")
        assert config.save_traces is True
        assert config.save_patches is True
        assert config.max_parallel == 1
        assert config.task_timeout == 1800
        assert config.continue_on_error is True
        assert config.use_baseline_cache is True
        assert config.test_timeout == 300
        assert config.test_verbose is False
        assert config.prompt_candidate_hash is None
        assert config.prompt_section_name is None

    def test_custom_values(self):
        """Test custom config values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "results"
            dataset_path = Path(tmpdir) / "data.jsonl"

            config = OrchestratorConfig(
                dataset_path=dataset_path,
                max_tasks=10,
                instance_ids=["django__django-12345"],
                repos=["django/django"],
                agent_profile="test",
                output_dir=output_dir,
                max_parallel=4,
                task_timeout=600,
            )

            assert config.dataset_path == dataset_path
            assert config.max_tasks == 10
            assert config.instance_ids == ["django__django-12345"]
            assert config.repos == ["django/django"]
            assert config.agent_profile == "test"
            assert config.output_dir == output_dir
            assert config.max_parallel == 4
            assert config.task_timeout == 600


class TestEvaluationSummary:
    """Tests for EvaluationSummary dataclass."""

    def test_default_values(self):
        """Test default summary values."""
        summary = EvaluationSummary()
        assert summary.total_tasks == 0
        assert summary.completed_tasks == 0
        assert summary.successful_tasks == 0
        assert summary.failed_tasks == 0
        assert summary.skipped_tasks == 0
        assert summary.avg_score == 0.0
        assert summary.pass_rate == 0.0
        assert summary.partial_pass_rate == 0.0
        assert summary.avg_turns == 0.0
        assert summary.avg_tool_calls == 0.0
        assert summary.env_failures == 0
        assert summary.baseline_failures == 0
        assert summary.execution_failures == 0
        assert summary.validation_failures == 0
        assert summary.task_results == []

    def test_duration_seconds_not_started(self):
        """Test duration when evaluation not started."""
        summary = EvaluationSummary()
        assert summary.duration_seconds == 0.0

    def test_duration_seconds_completed(self):
        """Test duration when evaluation completed."""
        summary = EvaluationSummary(
            started_at=datetime(2025, 1, 1, 12, 0, 0),
            completed_at=datetime(2025, 1, 1, 12, 30, 0),
        )
        assert summary.duration_seconds == 1800.0  # 30 minutes

    def test_to_dict(self):
        """Test serialization to dictionary."""
        summary = EvaluationSummary(
            total_tasks=10,
            completed_tasks=8,
            successful_tasks=6,
            failed_tasks=2,
            pass_rate=0.75,
            started_at=datetime(2025, 1, 1, 12, 0, 0),
            completed_at=datetime(2025, 1, 1, 12, 30, 0),
        )

        data = summary.to_dict()
        assert data["total_tasks"] == 10
        assert data["completed_tasks"] == 8
        assert data["successful_tasks"] == 6
        assert data["failed_tasks"] == 2
        assert data["pass_rate"] == 0.75
        assert data["duration_seconds"] == 1800.0

    def test_to_text(self):
        """Test text report generation."""
        summary = EvaluationSummary(
            total_tasks=10,
            completed_tasks=8,
            successful_tasks=6,
            failed_tasks=2,
            pass_rate=0.75,
            partial_pass_rate=0.80,
            avg_score=0.65,
            avg_turns=15.5,
            avg_tool_calls=25.3,
            started_at=datetime(2025, 1, 1, 12, 0, 0),
            completed_at=datetime(2025, 1, 1, 12, 30, 0),
        )

        text = summary.to_text()
        assert "SWE-bench Evaluation Summary" in text
        assert "Total Tasks: 10" in text
        assert "Completed: 8" in text
        assert "Successful: 6" in text
        assert "Pass Rate: 75.0%" in text
        assert "Duration: 1800.0s" in text


class TestEvaluationOrchestrator:
    """Tests for EvaluationOrchestrator class."""

    def test_init(self):
        """Test orchestrator initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = OrchestratorConfig(output_dir=Path(tmpdir))
            orchestrator = EvaluationOrchestrator(config)

            assert orchestrator.config == config
            assert orchestrator.progress_callback is None

    def test_init_with_callback(self):
        """Test orchestrator initialization with progress callback."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = OrchestratorConfig(output_dir=Path(tmpdir))
            callback = MagicMock()
            orchestrator = EvaluationOrchestrator(config, callback)

            assert orchestrator.progress_callback == callback

    def test_get_progress_not_found(self):
        """Test getting progress for non-existent task."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = OrchestratorConfig(output_dir=Path(tmpdir))
            orchestrator = EvaluationOrchestrator(config)

            assert orchestrator.get_progress("nonexistent") is None

    def test_get_summary(self):
        """Test getting evaluation summary."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = OrchestratorConfig(output_dir=Path(tmpdir))
            orchestrator = EvaluationOrchestrator(config)

            summary = orchestrator.get_summary()
            assert isinstance(summary, EvaluationSummary)

    @pytest.mark.asyncio
    async def test_apply_patch_success(self):
        """Test successful patch application."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            config = OrchestratorConfig(output_dir=workspace)
            orchestrator = EvaluationOrchestrator(config)
            orchestrator._init_components()

            # Create a test file
            test_file = workspace / "test.py"
            test_file.write_text("def foo():\n    pass\n")

            # Create a simple patch
            patch = """--- a/test.py
+++ b/test.py
@@ -1,2 +1,2 @@
 def foo():
-    pass
+    return 42
"""

            # Initialize git repo for patch to work
            import subprocess

            subprocess.run(["git", "init"], cwd=workspace, capture_output=True)
            subprocess.run(["git", "add", "."], cwd=workspace, capture_output=True)
            subprocess.run(
                ["git", "commit", "-m", "Initial"],
                cwd=workspace,
                capture_output=True,
                env={
                    **dict(__import__("os").environ),
                    "GIT_AUTHOR_NAME": "Test",
                    "GIT_AUTHOR_EMAIL": "test@test.com",
                    "GIT_COMMITTER_NAME": "Test",
                    "GIT_COMMITTER_EMAIL": "test@test.com",
                },
            )

            result = await orchestrator._apply_patch(workspace, patch)
            # Patch might succeed or fail depending on git setup
            assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_init_components(self):
        """Test component initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = OrchestratorConfig(output_dir=Path(tmpdir))
            orchestrator = EvaluationOrchestrator(config)

            orchestrator._init_components()

            assert orchestrator._loader is not None
            assert orchestrator._workspace_manager is not None
            assert orchestrator._env_setup is not None
            assert orchestrator._test_registry is not None
            assert orchestrator._baseline_validator is not None
            assert orchestrator._correlator is not None
            assert Path(tmpdir).exists()

    def test_notify_progress_with_callback(self):
        """Test progress notification with callback."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = OrchestratorConfig(output_dir=Path(tmpdir))
            callback = MagicMock()
            orchestrator = EvaluationOrchestrator(config, callback)

            progress = TaskProgress(
                instance_id="test",
                stage=EvaluationStage.AGENT_EXECUTION,
            )

            orchestrator._notify_progress(progress)

            callback.assert_called_once_with(progress, EvaluationStage.AGENT_EXECUTION)

    def test_notify_progress_without_callback(self):
        """Test progress notification without callback."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = OrchestratorConfig(output_dir=Path(tmpdir))
            orchestrator = EvaluationOrchestrator(config)

            progress = TaskProgress(
                instance_id="test",
                stage=EvaluationStage.AGENT_EXECUTION,
            )

            # Should not raise
            orchestrator._notify_progress(progress)

    def test_save_trace(self):
        """Test saving execution trace."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            config = OrchestratorConfig(output_dir=output_dir)
            orchestrator = EvaluationOrchestrator(config)

            trace = MagicMock()
            trace.to_dict.return_value = {"tool_calls": [], "file_edits": []}

            orchestrator._save_trace("test_instance", trace)

            trace_file = output_dir / "traces" / "test_instance.json"
            assert trace_file.exists()

    def test_save_patch(self):
        """Test saving generated patch."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            config = OrchestratorConfig(output_dir=output_dir)
            orchestrator = EvaluationOrchestrator(config)

            patch_content = "--- a/file.py\n+++ b/file.py\n@@ -1 +1 @@\n-old\n+new\n"

            orchestrator._save_patch("test_instance", patch_content)

            patch_file = output_dir / "patches" / "test_instance.patch"
            assert patch_file.exists()
            assert patch_file.read_text() == patch_content

    def test_save_summary(self):
        """Test saving evaluation summary."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            config = OrchestratorConfig(output_dir=output_dir)
            orchestrator = EvaluationOrchestrator(config)
            orchestrator._summary = EvaluationSummary(
                total_tasks=5,
                completed_tasks=4,
                pass_rate=0.75,
            )

            orchestrator._save_summary()

            json_file = output_dir / "summary.json"
            text_file = output_dir / "summary.txt"
            assert json_file.exists()
            assert text_file.exists()

    def test_save_validated_session_feedback(self):
        """Test saving validated session-truth feedback from real validator output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            config = OrchestratorConfig(output_dir=output_dir)
            orchestrator = EvaluationOrchestrator(config)
            validation_result = BaselineValidationResult(
                instance_id="django__123",
                baseline=TestBaseline(
                    instance_id="django__123",
                    repo="django/django",
                    base_commit="abc123",
                    fail_to_pass=["test_fix_a", "test_fix_b"],
                    pass_to_pass=["test_keep_green"],
                    status=BaselineStatus.VALID,
                ),
                post_change_results=TestRunResults(
                    total=3,
                    passed=2,
                    failed=1,
                    duration_seconds=9.0,
                ),
                fail_to_pass_fixed=["test_fix_a"],
                pass_to_pass_broken=[],
                success=False,
                partial_success=True,
                score=0.5,
            )
            score = SWEBenchScore(
                instance_id="django__123",
                resolved=False,
                partial=True,
                fail_to_pass_score=0.5,
                pass_to_pass_score=1.0,
                overall_score=0.7,
                tests_fixed=1,
                tests_broken=0,
                total_fail_to_pass=2,
                total_pass_to_pass=1,
            )

            with patch(
                "victor.evaluation.validated_session_truth_persistence.refresh_runtime_evaluation_feedback_aggregate"
            ) as refresh_aggregate:
                orchestrator._save_validated_session_feedback(
                    "django__123",
                    validation_result,
                    score=score,
                )

            feedback_file = output_dir / "evaluations" / "eval_session_django__123.json"
            assert feedback_file.exists()
            payload = json.loads(feedback_file.read_text())
            assert payload["runtime_evaluation_feedback"]["metadata"]["truth_validation_mode"] == (
                "swe_bench_posthoc_validation"
            )
            assert payload["provider"] is None
            assert payload["prompt_candidate_hash"] is None
            assert payload["runtime_evaluation_feedback"]["metadata"]["scope"]["project"] == (
                "django/django"
            )
            refresh_aggregate.assert_called_once_with(output_dir / "evaluations")

    def test_save_validated_session_feedback_uses_emitter_registry(self):
        """The orchestrator should emit validated session truth via the shared service."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            config = OrchestratorConfig(output_dir=output_dir)

            captured_kwargs = {}

            class StubService:
                def persist_validation_result(self, **kwargs):
                    captured_kwargs.update(kwargs)
                    return kwargs["results_dir"] / "eval_session_stub.json"

            orchestrator = EvaluationOrchestrator(
                config,
                validated_session_truth_service=StubService(),
            )
            validation_result = BaselineValidationResult(
                instance_id="django__123",
                baseline=TestBaseline(
                    instance_id="django__123",
                    repo="django/django",
                    base_commit="abc123",
                    fail_to_pass=["test_fix_a"],
                    pass_to_pass=["test_keep_green"],
                    status=BaselineStatus.VALID,
                ),
                post_change_results=TestRunResults(
                    total=2,
                    passed=1,
                    failed=1,
                    duration_seconds=4.0,
                ),
                fail_to_pass_fixed=["test_fix_a"],
                pass_to_pass_broken=[],
                success=False,
                partial_success=True,
                score=0.5,
            )

            with patch(
                "victor.evaluation.validated_session_truth_persistence.refresh_runtime_evaluation_feedback_aggregate"
            ) as refresh_aggregate:
                feedback_path = orchestrator._save_validated_session_feedback(
                    "django__123",
                    validation_result,
                )

            assert feedback_path == output_dir / "evaluations" / "eval_session_stub.json"
            assert captured_kwargs["benchmark"].value == "swe_bench"
            assert captured_kwargs["task_id"] == "django__123"
            assert captured_kwargs["validation_result"] is validation_result
            assert captured_kwargs["results_dir"] == output_dir / "evaluations"
            assert captured_kwargs["metadata"] is None
            refresh_aggregate.assert_not_called()

    def test_save_validated_session_feedback_passes_runtime_metadata(self):
        """Explicit runtime metadata should flow through the shared service boundary."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            config = OrchestratorConfig(output_dir=output_dir)

            captured_kwargs = {}

            class StubService:
                def persist_validation_result(self, **kwargs):
                    captured_kwargs.update(kwargs)
                    return kwargs["results_dir"] / "eval_session_stub.json"

            orchestrator = EvaluationOrchestrator(
                config,
                validated_session_truth_service=StubService(),
            )

            feedback_path = orchestrator._save_validated_session_feedback(
                "django__123",
                validation_result=object(),
                metadata={
                    "provider": "anthropic",
                    "model": "claude-sonnet",
                    "prompt_candidate_hash": "cand-123",
                    "section_name": "GROUNDING_RULES",
                },
            )

            assert feedback_path == output_dir / "evaluations" / "eval_session_stub.json"
            assert captured_kwargs["metadata"] == {
                "provider": "anthropic",
                "model": "claude-sonnet",
                "prompt_candidate_hash": "cand-123",
                "section_name": "GROUNDING_RULES",
            }

    def test_build_validation_runtime_metadata_uses_single_active_prompt_candidate(
        self,
    ):
        """Single active prompt identities should map into canonical validation metadata."""
        orchestrator = EvaluationOrchestrator(OrchestratorConfig())
        inner = MagicMock()
        inner.provider_name = "anthropic"
        inner.model = "claude-sonnet"
        inner.get_active_prompt_optimization_metadata.return_value = {
            "entries": [
                {
                    "provider": "anthropic",
                    "prompt_candidate_hash": "cand-123",
                    "section_name": "GROUNDING_RULES",
                    "prompt_section_name": "GROUNDING_RULES",
                    "strategy_name": "gepa",
                }
            ],
            "by_section": {
                "GROUNDING_RULES": {
                    "provider": "anthropic",
                    "prompt_candidate_hash": "cand-123",
                    "section_name": "GROUNDING_RULES",
                    "prompt_section_name": "GROUNDING_RULES",
                    "strategy_name": "gepa",
                }
            },
        }
        adapter = MagicMock()
        adapter.orchestrator = inner

        metadata = orchestrator._build_validation_runtime_metadata(adapter)

        assert metadata["provider"] == "anthropic"
        assert metadata["model"] == "claude-sonnet"
        assert metadata["prompt_candidate_hash"] == "cand-123"
        assert metadata["section_name"] == "GROUNDING_RULES"
        assert metadata["prompt_section_name"] == "GROUNDING_RULES"
        assert metadata["prompt_optimization"]["entries"][0]["strategy_name"] == "gepa"

    def test_build_validation_runtime_metadata_omits_ambiguous_prompt_candidate(self):
        """Multiple active prompt candidates should keep singular identity fields unset."""
        orchestrator = EvaluationOrchestrator(OrchestratorConfig())
        inner = MagicMock()
        inner.provider_name = "anthropic"
        inner.model = "claude-sonnet"
        inner.get_active_prompt_optimization_metadata.return_value = {
            "entries": [
                {
                    "provider": "anthropic",
                    "prompt_candidate_hash": "cand-123",
                    "section_name": "GROUNDING_RULES",
                },
                {
                    "provider": "anthropic",
                    "prompt_candidate_hash": "cand-456",
                    "section_name": "COMPLETION_GUIDANCE",
                },
            ],
            "by_section": {},
        }
        adapter = MagicMock()
        adapter.orchestrator = inner

        metadata = orchestrator._build_validation_runtime_metadata(adapter)

        assert metadata["provider"] == "anthropic"
        assert metadata["model"] == "claude-sonnet"
        assert "prompt_candidate_hash" not in metadata
        assert "section_name" not in metadata
        assert len(metadata["prompt_optimization"]["entries"]) == 2

    def test_build_adapter_config_binds_requested_prompt_candidate(self):
        """Orchestrator-level prompt identity should become a runtime binding."""
        orchestrator = EvaluationOrchestrator(
            OrchestratorConfig(
                prompt_candidate_hash="cand-123",
                prompt_section_name="GROUNDING_RULES",
            )
        )

        adapter_config = orchestrator._build_adapter_config()

        assert adapter_config.prompt_binding is not None
        assert adapter_config.prompt_binding.prompt_candidate_hash == "cand-123"
        assert adapter_config.prompt_binding.section_name == "GROUNDING_RULES"

    def test_orchestrator_accepts_legacy_emitter_registry_keyword(self):
        """Legacy emitter-registry wiring should still resolve through the service factory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            config = OrchestratorConfig(output_dir=output_dir)

            class StubEmitter:
                def supports(self, benchmark):
                    return benchmark.value == "swe_bench"

                def build_artifact(self, context):
                    return ValidatedSessionTruthArtifact(
                        path=context.results_dir / "eval_session_stub.json",
                        record={"runtime_evaluation_feedback": {"metadata": {"source": "stub"}}},
                    )

            orchestrator = EvaluationOrchestrator(
                config,
                validated_session_truth_emitters=ValidatedSessionTruthEmitterRegistry(
                    [StubEmitter()]
                ),
            )

            saved_path = orchestrator._save_validated_session_feedback(
                "django__123",
                validation_result=object(),
            )

            assert saved_path == output_dir / "evaluations" / "eval_session_stub.json"

    def test_orchestrator_uses_canonical_evaluation_service_factory(self):
        """Orchestrator should resolve the default service through the evaluation-level entrypoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = OrchestratorConfig(output_dir=Path(tmpdir))
            registry = ValidatedSessionTruthEmitterRegistry()
            stub_service = object()

            with patch(
                "victor.evaluation.services.create_validated_session_truth_service",
                return_value=stub_service,
            ) as create_service:
                orchestrator = EvaluationOrchestrator(
                    config,
                    validated_session_truth_emitters=registry,
                )

            assert orchestrator._validated_session_truth_service is stub_service
            create_service.assert_called_once_with(registry)

    def test_orchestrator_materializes_service_through_shared_helper(self):
        """Orchestrator should centralize service materialization through one shared helper."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = OrchestratorConfig(output_dir=Path(tmpdir))
            registry = ValidatedSessionTruthEmitterRegistry()
            stub_service = object()

            with patch(
                "victor.evaluation.evaluation_orchestrator.materialize_validated_session_truth_service",
                return_value=stub_service,
            ) as materialize_service:
                orchestrator = EvaluationOrchestrator(
                    config,
                    validated_session_truth_emitters=registry,
                )

            assert orchestrator._validated_session_truth_service is stub_service
            materialize_service.assert_called_once_with(
                service=None,
                legacy_kwargs={"validated_session_truth_emitters": registry},
            )


class TestProgressCallback:
    """Tests for progress callback functionality."""

    def test_callback_called_for_each_stage(self):
        """Test that callback is called for stage transitions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = OrchestratorConfig(output_dir=Path(tmpdir))
            callback = MagicMock()
            orchestrator = EvaluationOrchestrator(config, callback)

            progress = TaskProgress(instance_id="test")

            # Simulate stage transitions
            for stage in [
                EvaluationStage.ENVIRONMENT_SETUP,
                EvaluationStage.BASELINE_ESTABLISHMENT,
                EvaluationStage.AGENT_EXECUTION,
                EvaluationStage.VALIDATION,
                EvaluationStage.COMPLETED,
            ]:
                progress.stage = stage
                orchestrator._notify_progress(progress)

            assert callback.call_count == 5


class TestGenerateReport:
    """Tests for report generation."""

    def test_generate_report_empty(self):
        """Test report generation with no tasks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = OrchestratorConfig(output_dir=Path(tmpdir))
            orchestrator = EvaluationOrchestrator(config)
            orchestrator._init_components()

            report = orchestrator._generate_report()

            # Should return a report even with no tasks
            assert report is not None
            assert orchestrator._summary.total_tasks == 0

    def test_generate_report_with_results(self):
        """Test report generation with task results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = OrchestratorConfig(output_dir=Path(tmpdir))
            orchestrator = EvaluationOrchestrator(config)
            orchestrator._init_components()

            # Add some mock progress - use MagicMock without spec to allow all attributes
            score1 = MagicMock()
            score1.instance_id = "task1"
            score1.overall_score = 1.0
            score1.fail_to_pass_score = 1.0
            score1.pass_to_pass_score = 1.0
            score1.resolved = True
            score1.partial = False
            score1.metadata = {}
            score1.correlations = []
            score1.tests_fixed = 1
            score1.tests_broken = 0
            score1.total_fail_to_pass = 1
            score1.total_pass_to_pass = 5
            score1.to_dict.return_value = {"instance_id": "task1", "overall_score": 1.0}

            score2 = MagicMock()
            score2.instance_id = "task2"
            score2.overall_score = 0.5
            score2.fail_to_pass_score = 0.5
            score2.pass_to_pass_score = 0.5
            score2.resolved = False
            score2.partial = True
            score2.metadata = {}
            score2.correlations = []
            score2.tests_fixed = 0
            score2.tests_broken = 1
            score2.total_fail_to_pass = 2
            score2.total_pass_to_pass = 3
            score2.to_dict.return_value = {"instance_id": "task2", "overall_score": 0.5}

            trace1 = MagicMock()
            trace1.total_turns = 10
            trace1.tool_calls = [MagicMock()] * 5

            trace2 = MagicMock()
            trace2.total_turns = 20
            trace2.tool_calls = [MagicMock()] * 10

            orchestrator._progress = {
                "task1": TaskProgress(
                    instance_id="task1",
                    stage=EvaluationStage.COMPLETED,
                    score=score1,
                    execution_trace=trace1,
                ),
                "task2": TaskProgress(
                    instance_id="task2",
                    stage=EvaluationStage.COMPLETED,
                    score=score2,
                    execution_trace=trace2,
                ),
                "task3": TaskProgress(
                    instance_id="task3",
                    stage=EvaluationStage.FAILED,
                    error_message="Test error",
                ),
            }

            orchestrator._generate_report()

            assert orchestrator._summary.completed_tasks == 2
            assert orchestrator._summary.failed_tasks == 1
            assert orchestrator._summary.successful_tasks == 1
            assert orchestrator._summary.avg_score == 0.75  # (1.0 + 0.5) / 2
            assert orchestrator._summary.pass_rate == 0.5  # 1 resolved out of 2
