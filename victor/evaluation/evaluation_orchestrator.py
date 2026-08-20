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

"""Master orchestrator for SWE-bench evaluation.

This module provides the EvaluationOrchestrator class which coordinates the
complete end-to-end SWE-bench evaluation flow:

1. Load SWE-bench tasks from dataset
2. Setup development environments
3. Establish test baselines at base commit
4. Run agent to generate patches
5. Validate changes against baseline
6. Correlate results and generate reports

Example usage:
    from victor.evaluation.evaluation_orchestrator import (
        EvaluationOrchestrator,
        OrchestratorConfig,
    )

    config = OrchestratorConfig(
        dataset_path="swe-bench-lite.jsonl",
        agent_profile="default",
        output_dir=Path("./results"),
        max_parallel=4,
    )

    orchestrator = EvaluationOrchestrator(config)
    report = await orchestrator.run_evaluation()
    print(report.to_text())
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

from victor.evaluation.agent_adapter import (
    AdapterConfig,
    PromptOptimizationBinding,
    VictorAgentAdapter,
)
from victor.evaluation.agentic_harness import AgenticExecutionTrace
from victor.evaluation.baseline_validator import (
    BaselineCache,
    BaselineValidationResult,
    BaselineValidator,
    TestBaseline,
)
from victor.evaluation.env_setup import EnvironmentConfig, EnvironmentSetup, SetupResult
from victor.evaluation.validated_session_truth_emitters import (
    ValidatedSessionTruthEmitterRegistry,
)
from victor.evaluation.services import (
    materialize_validated_session_truth_service,
    ValidatedSessionTruthServiceProtocol,
)
from victor.evaluation.result_correlation import (
    CorrelationReport,
    ResultCorrelator,
    SWEBenchScore,
)
from victor.evaluation.protocol import BenchmarkType
from victor.evaluation.swe_bench_loader import (
    SWEBenchConfig,
    SWEBenchInstance,
    SWEBenchLoader,
    SWEBenchWorkspaceManager,
)
from victor.evaluation.test_runners import (
    TestRunnerRegistry,
)

logger = logging.getLogger(__name__)


class EvaluationStage(Enum):
    """Stages of the evaluation pipeline."""

    LOADING = "loading"
    ENVIRONMENT_SETUP = "environment_setup"
    BASELINE_ESTABLISHMENT = "baseline_establishment"
    AGENT_EXECUTION = "agent_execution"
    VALIDATION = "validation"
    CORRELATION = "correlation"
    REPORTING = "reporting"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TaskProgress:
    """Progress tracking for a single task."""

    instance_id: str
    stage: EvaluationStage = EvaluationStage.LOADING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: str = ""

    # Stage results
    env_setup_result: Optional[SetupResult] = None
    baseline: Optional[TestBaseline] = None
    execution_trace: Optional[AgenticExecutionTrace] = None
    validation_result: Optional[BaselineValidationResult] = None
    score: Optional[SWEBenchScore] = None

    # Correction metrics (per-task)
    correction_metrics: Optional[dict[str, Any]] = None

    @property
    def duration_seconds(self) -> float:
        """Get total duration in seconds."""
        if not self.started_at:
            return 0.0
        end = self.completed_at or datetime.now()
        return (end - self.started_at).total_seconds()

    @property
    def is_success(self) -> bool:
        """Check if task completed successfully."""
        return self.stage == EvaluationStage.COMPLETED and self.score is not None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "instance_id": self.instance_id,
            "stage": self.stage.value,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": (self.completed_at.isoformat() if self.completed_at else None),
            "duration_seconds": self.duration_seconds,
            "error_message": self.error_message,
            "is_success": self.is_success,
            "score": self.score.to_dict() if self.score else None,
            "correction_metrics": self.correction_metrics,
        }


@dataclass
class OrchestratorConfig:
    """Configuration for the evaluation orchestrator."""

    # Dataset configuration
    dataset_path: Optional[Path] = None
    dataset_name: str = "princeton-nlp/SWE-bench_Lite"
    max_tasks: int = 0  # 0 = all tasks
    instance_ids: list[str] = field(default_factory=list)
    repos: list[str] = field(default_factory=list)

    # Agent configuration
    agent_profile: str = "default"
    adapter_config: Optional[AdapterConfig] = None
    prompt_candidate_hash: Optional[str] = None
    prompt_section_name: Optional[str] = None

    # Output configuration
    output_dir: Path = field(default_factory=lambda: Path("./swe_bench_results"))
    save_traces: bool = True
    save_patches: bool = True

    # Execution configuration
    max_parallel: int = 1
    task_timeout: int = 1800  # 30 minutes per task
    continue_on_error: bool = True

    # Environment configuration
    env_config: Optional[EnvironmentConfig] = None
    workspace_cache_dir: Optional[Path] = None

    # Baseline configuration
    use_baseline_cache: bool = True
    baseline_cache_dir: Optional[Path] = None

    # Test runner configuration
    test_timeout: int = 300  # 5 minutes
    test_verbose: bool = False


@dataclass
class EvaluationSummary:
    """Summary of the evaluation run."""

    total_tasks: int = 0
    completed_tasks: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    skipped_tasks: int = 0

    # Timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Aggregated metrics
    avg_score: float = 0.0
    pass_rate: float = 0.0
    partial_pass_rate: float = 0.0
    avg_turns: float = 0.0
    avg_tool_calls: float = 0.0

    # Correction metrics (aggregated)
    total_corrections: int = 0
    successful_corrections: int = 0
    correction_success_rate: float = 0.0
    avg_corrections_per_task: float = 0.0

    # Per-stage failures
    env_failures: int = 0
    baseline_failures: int = 0
    execution_failures: int = 0
    validation_failures: int = 0

    # Task details
    task_results: list[TaskProgress] = field(default_factory=list)

    @property
    def duration_seconds(self) -> float:
        """Get total duration in seconds."""
        if not self.started_at:
            return 0.0
        end = self.completed_at or datetime.now()
        return (end - self.started_at).total_seconds()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "total_tasks": self.total_tasks,
            "completed_tasks": self.completed_tasks,
            "successful_tasks": self.successful_tasks,
            "failed_tasks": self.failed_tasks,
            "skipped_tasks": self.skipped_tasks,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": (self.completed_at.isoformat() if self.completed_at else None),
            "duration_seconds": self.duration_seconds,
            "avg_score": self.avg_score,
            "pass_rate": self.pass_rate,
            "partial_pass_rate": self.partial_pass_rate,
            "avg_turns": self.avg_turns,
            "avg_tool_calls": self.avg_tool_calls,
            "total_corrections": self.total_corrections,
            "successful_corrections": self.successful_corrections,
            "correction_success_rate": self.correction_success_rate,
            "avg_corrections_per_task": self.avg_corrections_per_task,
            "env_failures": self.env_failures,
            "baseline_failures": self.baseline_failures,
            "execution_failures": self.execution_failures,
            "validation_failures": self.validation_failures,
            "task_results": [t.to_dict() for t in self.task_results],
        }

    def to_text(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 60,
            "SWE-bench Evaluation Summary",
            "=" * 60,
            "",
            f"Total Tasks: {self.total_tasks}",
            f"Completed: {self.completed_tasks}",
            f"Successful: {self.successful_tasks}",
            f"Failed: {self.failed_tasks}",
            f"Skipped: {self.skipped_tasks}",
            "",
            f"Pass Rate: {self.pass_rate:.1%}",
            f"Partial Pass Rate: {self.partial_pass_rate:.1%}",
            f"Average Score: {self.avg_score:.3f}",
            "",
            f"Average Turns: {self.avg_turns:.1f}",
            f"Average Tool Calls: {self.avg_tool_calls:.1f}",
            "",
            "Correction Metrics:",
            f"  Total Corrections: {self.total_corrections}",
            f"  Successful: {self.successful_corrections}",
            f"  Success Rate: {self.correction_success_rate:.1%}",
            f"  Avg per Task: {self.avg_corrections_per_task:.1f}",
            "",
            f"Duration: {self.duration_seconds:.1f}s",
            "",
            "Failures by Stage:",
            f"  Environment: {self.env_failures}",
            f"  Baseline: {self.baseline_failures}",
            f"  Execution: {self.execution_failures}",
            f"  Validation: {self.validation_failures}",
            "",
            "=" * 60,
        ]
        return "\n".join(lines)


# Progress callback type
ProgressCallback = Callable[[TaskProgress, EvaluationStage], None]


class EvaluationOrchestrator:
    """Master orchestrator for SWE-bench evaluation.

    This class coordinates the complete end-to-end evaluation flow,
    connecting all the individual components:
    - SWEBenchLoader: Load tasks from dataset
    - EnvironmentSetup: Setup development environments
    - BaselineValidator: Establish and validate test baselines
    - VictorAgentAdapter: Run agent to generate patches
    - ResultCorrelator: Correlate results and compute scores
    """

    def __init__(
        self,
        config: OrchestratorConfig,
        progress_callback: Optional[ProgressCallback] = None,
        validated_session_truth_service: Optional[ValidatedSessionTruthServiceProtocol] = None,
        **legacy_kwargs: Any,
    ):
        """Initialize the orchestrator.

        Args:
            config: Orchestrator configuration
            progress_callback: Optional callback for progress updates
            validated_session_truth_service: Service for validated session-truth orchestration
        """
        self.config = config
        self.progress_callback = progress_callback
        self._validated_session_truth_service = materialize_validated_session_truth_service(
            service=validated_session_truth_service,
            legacy_kwargs=legacy_kwargs,
        )

        # Initialize components
        self._loader: Optional[SWEBenchLoader] = None
        self._workspace_manager: Optional[SWEBenchWorkspaceManager] = None
        self._env_setup: Optional[EnvironmentSetup] = None
        self._baseline_validator: Optional[BaselineValidator] = None
        self._test_registry: Optional[TestRunnerRegistry] = None
        self._correlator: Optional[ResultCorrelator] = None

        # State
        self._tasks: list[SWEBenchInstance] = []
        self._progress: dict[str, TaskProgress] = {}
        self._summary: EvaluationSummary = EvaluationSummary()

    def _init_components(self) -> None:
        """Initialize evaluation components."""
        # SWE-bench loader
        swe_config = SWEBenchConfig(
            max_tasks=self.config.max_tasks,
            instance_ids=self.config.instance_ids,
            repos=self.config.repos,
        )
        self._loader = SWEBenchLoader(swe_config)

        # Workspace manager
        cache_dir = self.config.workspace_cache_dir or (self.config.output_dir / "workspace_cache")
        self._workspace_manager = SWEBenchWorkspaceManager(cache_dir=cache_dir)

        # Environment setup
        env_config = self.config.env_config or EnvironmentConfig(
            timeout_seconds=self.config.task_timeout,
            verbose=self.config.test_verbose,
        )
        self._env_setup = EnvironmentSetup(env_config)

        # Test runner registry
        self._test_registry = TestRunnerRegistry()

        # Baseline validator
        baseline_cache = None
        if self.config.use_baseline_cache:
            baseline_cache_dir = self.config.baseline_cache_dir or (
                self.config.output_dir / "baseline_cache"
            )
            baseline_cache = BaselineCache(baseline_cache_dir)
        self._baseline_validator = BaselineValidator(
            test_registry=self._test_registry,
            cache=baseline_cache,
            use_cache=self.config.use_baseline_cache,
        )

        # Result correlator
        self._correlator = ResultCorrelator()

        # Create output directory
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    async def run_evaluation(self) -> CorrelationReport:
        """Run the complete evaluation pipeline.

        Returns:
            CorrelationReport with aggregated results
        """
        self._summary = EvaluationSummary(started_at=datetime.now())

        try:
            # Initialize components
            self._init_components()

            # Load tasks
            await self._load_tasks()

            # Run tasks
            if self.config.max_parallel > 1:
                await self._run_parallel()
            else:
                await self._run_sequential()

            # Generate final report
            report = self._generate_report()

            self._summary.completed_at = datetime.now()
            self._save_summary()

            return report

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            self._summary.completed_at = datetime.now()
            raise

    async def _load_tasks(self) -> None:
        """Load tasks from dataset."""
        logger.info("Loading SWE-bench tasks...")

        if self.config.dataset_path:
            self._tasks = self._loader.load_instances_from_file(self.config.dataset_path)
        else:
            self._tasks = await self._loader.load_from_huggingface(self.config.dataset_name)

        # Filter by instance IDs if specified
        if self.config.instance_ids:
            self._tasks = [t for t in self._tasks if t.instance_id in self.config.instance_ids]

        # Filter by repos if specified
        if self.config.repos:
            self._tasks = [t for t in self._tasks if t.repo in self.config.repos]

        # Apply max_tasks limit
        if self.config.max_tasks > 0:
            self._tasks = self._tasks[: self.config.max_tasks]

        self._summary.total_tasks = len(self._tasks)

        # Initialize progress tracking
        for task in self._tasks:
            self._progress[task.instance_id] = TaskProgress(instance_id=task.instance_id)

        logger.info(f"Loaded {len(self._tasks)} tasks")

    async def _run_sequential(self) -> None:
        """Run tasks sequentially."""
        for task in self._tasks:
            try:
                await self._run_single_task(task)
            except Exception as e:
                logger.error(f"Task {task.instance_id} failed: {e}")
                progress = self._progress[task.instance_id]
                progress.stage = EvaluationStage.FAILED
                progress.error_message = str(e)
                progress.completed_at = datetime.now()

                if not self.config.continue_on_error:
                    raise

    async def _run_parallel(self) -> None:
        """Run tasks in parallel with concurrency limit."""
        semaphore = asyncio.Semaphore(self.config.max_parallel)

        async def run_with_semaphore(task: SWEBenchInstance) -> None:
            async with semaphore:
                try:
                    await self._run_single_task(task)
                except Exception as e:
                    logger.error(f"Task {task.instance_id} failed: {e}")
                    progress = self._progress[task.instance_id]
                    progress.stage = EvaluationStage.FAILED
                    progress.error_message = str(e)
                    progress.completed_at = datetime.now()

        await asyncio.gather(*[run_with_semaphore(task) for task in self._tasks])

    async def _run_single_task(self, task: SWEBenchInstance) -> None:
        """Run a single evaluation task through the full pipeline.

        Args:
            task: SWE-bench instance to evaluate
        """
        progress = self._progress[task.instance_id]
        progress.started_at = datetime.now()

        logger.info(f"Starting task {task.instance_id}")

        try:
            # Convert to BenchmarkTask first (needed for workspace setup)
            benchmark_task = task.to_benchmark_task()

            # Stage 1: Setup workspace
            progress.stage = EvaluationStage.ENVIRONMENT_SETUP
            self._notify_progress(progress)

            workspace = await self._workspace_manager.setup_workspace(
                task=benchmark_task,
                use_cache=True,
            )

            # Setup environment
            env_result = await self._env_setup.setup_environment(workspace)
            progress.env_setup_result = env_result

            if not env_result.success:
                raise RuntimeError(f"Environment setup failed: {env_result.error_message}")

            # Stage 2: Establish baseline
            progress.stage = EvaluationStage.BASELINE_ESTABLISHMENT
            self._notify_progress(progress)

            baseline = await self._baseline_validator.establish_baseline(
                instance_id=task.instance_id,
                repo=task.repo,
                base_commit=task.base_commit,
                workspace_dir=workspace,
                fail_to_pass=task.fail_to_pass,
                pass_to_pass=task.pass_to_pass,
            )
            progress.baseline = baseline

            if not baseline.is_valid():
                raise RuntimeError(f"Baseline invalid: {baseline.error_message}")

            # Stage 3: Run agent
            progress.stage = EvaluationStage.AGENT_EXECUTION
            self._notify_progress(progress)

            # Create agent adapter
            adapter_config = self._build_adapter_config()
            adapter = VictorAgentAdapter.from_profile(
                profile=self.config.agent_profile,
                config=adapter_config,
            )

            # Execute agent in repo subdirectory (workspace/repo contains the cloned code)
            repo_dir = workspace / "repo"
            trace = await asyncio.wait_for(
                adapter.execute_task(benchmark_task, repo_dir),
                timeout=self.config.task_timeout,
            )
            progress.execution_trace = trace

            # Save trace if configured
            if self.config.save_traces:
                self._save_trace(task.instance_id, trace)

            # Save patch if generated
            if self.config.save_patches and trace.generated_patch:
                self._save_patch(task.instance_id, trace.generated_patch)

            # Stage 4: Apply patch and validate
            progress.stage = EvaluationStage.VALIDATION
            self._notify_progress(progress)

            if trace.generated_patch:
                # Apply patch
                await self._apply_patch(workspace, trace.generated_patch)

            # Run validation
            validation_result = await self._baseline_validator.validate_changes(
                baseline=baseline,
                workspace_dir=workspace,
            )
            progress.validation_result = validation_result

            # Stage 5: Compute score
            progress.stage = EvaluationStage.CORRELATION
            self._notify_progress(progress)

            score = self._correlator.compute_score(
                validation_result=validation_result,
                instance_metadata={"instance_id": task.instance_id, "repo": task.repo},
            )
            progress.score = score
            validation_metadata = self._build_validation_runtime_metadata(adapter)
            self._save_validated_session_feedback(
                task.instance_id,
                validation_result,
                score=score,
                metadata=validation_metadata,
            )

            # Complete
            progress.stage = EvaluationStage.COMPLETED
            progress.completed_at = datetime.now()
            self._notify_progress(progress)

            logger.info(
                f"Completed task {task.instance_id}: "
                f"score={score.overall_score:.3f}, "
                f"f2p={score.fail_to_pass_score:.3f}, "
                f"p2p={score.pass_to_pass_score:.3f}"
            )

        except asyncio.TimeoutError:
            progress.stage = EvaluationStage.FAILED
            progress.error_message = f"Task timed out after {self.config.task_timeout}s"
            progress.completed_at = datetime.now()
            self._summary.execution_failures += 1
            raise

        except Exception as e:
            progress.error_message = str(e)
            progress.completed_at = datetime.now()

            # Categorize failure
            if progress.stage == EvaluationStage.ENVIRONMENT_SETUP:
                self._summary.env_failures += 1
            elif progress.stage == EvaluationStage.BASELINE_ESTABLISHMENT:
                self._summary.baseline_failures += 1
            elif progress.stage == EvaluationStage.AGENT_EXECUTION:
                self._summary.execution_failures += 1
            elif progress.stage == EvaluationStage.VALIDATION:
                self._summary.validation_failures += 1

            progress.stage = EvaluationStage.FAILED
            raise

        finally:
            # Cleanup workspace
            if self._workspace_manager:
                await self._workspace_manager.cleanup_workspace(workspace)

    async def _apply_patch(self, workspace: Path, patch: str) -> bool:
        """Apply a patch to the workspace.

        Args:
            workspace: Path to workspace
            patch: Unified diff patch content

        Returns:
            True if patch applied successfully
        """
        patch_file = workspace / ".agent_patch.diff"
        patch_file.write_text(patch)

        try:
            proc = await asyncio.create_subprocess_exec(
                "git",
                "apply",
                "--whitespace=fix",
                str(patch_file),
                cwd=workspace,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()

            if proc.returncode != 0:
                logger.warning(f"Patch application failed: {stderr.decode()}")
                return False

            return True

        finally:
            patch_file.unlink(missing_ok=True)

    def _generate_report(self) -> CorrelationReport:
        """Generate the final correlation report.

        Returns:
            CorrelationReport with aggregated results
        """
        # Collect all scores
        scores = []
        total_turns = 0
        total_tool_calls = 0

        for progress in self._progress.values():
            self._summary.task_results.append(progress)

            if progress.stage == EvaluationStage.COMPLETED:
                self._summary.completed_tasks += 1

                if progress.score:
                    scores.append(progress.score)

                    if progress.score.resolved:
                        self._summary.successful_tasks += 1

                if progress.execution_trace:
                    total_turns += progress.execution_trace.total_turns
                    total_tool_calls += len(progress.execution_trace.tool_calls)

            elif progress.stage == EvaluationStage.FAILED:
                self._summary.failed_tasks += 1
            else:
                self._summary.skipped_tasks += 1

        # Calculate aggregated metrics
        if scores:
            self._summary.avg_score = sum(s.overall_score for s in scores) / len(scores)
            self._summary.pass_rate = sum(1 for s in scores if s.resolved) / len(scores)
            self._summary.partial_pass_rate = sum(
                1 for s in scores if s.fail_to_pass_score > 0
            ) / len(scores)

        if self._summary.completed_tasks > 0:
            self._summary.avg_turns = total_turns / self._summary.completed_tasks
            self._summary.avg_tool_calls = total_tool_calls / self._summary.completed_tasks

        # Aggregate correction metrics from per-task results
        total_corrections = 0
        successful_corrections = 0
        for progress in self._progress.values():
            if progress.correction_metrics:
                metrics = progress.correction_metrics
                # Extract from summary section if available
                if "summary" in metrics:
                    total_corrections += metrics["summary"].get("total_corrections", 0)
                    successful_corrections += metrics["summary"].get("successful_corrections", 0)
                else:
                    # Direct access for flattened dict
                    total_corrections += metrics.get("total_corrections", 0)
                    successful_corrections += metrics.get("successful_corrections", 0)

        self._summary.total_corrections = total_corrections
        self._summary.successful_corrections = successful_corrections
        if total_corrections > 0:
            self._summary.correction_success_rate = successful_corrections / total_corrections
        if self._summary.completed_tasks > 0:
            self._summary.avg_corrections_per_task = (
                total_corrections / self._summary.completed_tasks
            )

        # Generate correlation report
        return self._correlator.generate_report(scores)

    def _notify_progress(self, progress: TaskProgress) -> None:
        """Notify progress callback if configured."""
        if self.progress_callback:
            self.progress_callback(progress, progress.stage)

    def _save_trace(self, instance_id: str, trace: AgenticExecutionTrace) -> None:
        """Save execution trace to file."""
        traces_dir = self.config.output_dir / "traces"
        traces_dir.mkdir(exist_ok=True)

        trace_file = traces_dir / f"{instance_id}.json"
        trace_file.write_text(json.dumps(trace.to_dict(), indent=2))

    def _save_patch(self, instance_id: str, patch: str) -> None:
        """Save generated patch to file."""
        patches_dir = self.config.output_dir / "patches"
        patches_dir.mkdir(exist_ok=True)

        patch_file = patches_dir / f"{instance_id}.patch"
        patch_file.write_text(patch)

    def _save_summary(self) -> None:
        """Save evaluation summary to file."""
        summary_file = self.config.output_dir / "summary.json"
        summary_file.write_text(json.dumps(self._summary.to_dict(), indent=2))

        # Also save human-readable version
        text_file = self.config.output_dir / "summary.txt"
        text_file.write_text(self._summary.to_text())

    def _save_validated_session_feedback(
        self,
        instance_id: str,
        validation_result: BaselineValidationResult,
        *,
        score: Optional[SWEBenchScore] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Optional[Path]:
        """Persist validated session-truth feedback from objective SWE-bench validation."""
        evaluations_dir = self.config.output_dir / "evaluations"
        return self._validated_session_truth_service.persist_validation_result(
            benchmark=BenchmarkType.SWE_BENCH,
            results_dir=evaluations_dir,
            task_id=instance_id,
            validation_result=validation_result,
            score=score,
            metadata=metadata,
        )

    @staticmethod
    def _canonical_prompt_optimization_identity(metadata: Any) -> dict[str, Any]:
        """Normalize live prompt-optimization metadata for persisted artifacts."""
        if not isinstance(metadata, dict):
            return {"entries": [], "by_section": {}}
        return {
            "entries": list(metadata.get("entries") or []),
            "by_section": dict(metadata.get("by_section") or {}),
        }

    @staticmethod
    def _optional_text(value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def _build_validation_runtime_metadata(
        self,
        adapter: Optional[VictorAgentAdapter] = None,
    ) -> dict[str, Any]:
        """Build canonical runtime metadata for validated session-truth artifacts."""
        if adapter is None:
            return {}

        orchestrator = getattr(adapter, "orchestrator", None)
        if orchestrator is None:
            return {}

        metadata: dict[str, Any] = {
            "provider": self._optional_text(getattr(orchestrator, "provider_name", None)),
            "model": self._optional_text(getattr(orchestrator, "model", None)),
        }

        prompt_metadata_getter = getattr(
            orchestrator, "get_active_prompt_optimization_metadata", None
        )
        if callable(prompt_metadata_getter):
            prompt_metadata = self._canonical_prompt_optimization_identity(prompt_metadata_getter())
            metadata["prompt_optimization"] = prompt_metadata

            candidate_entries = []
            for entry in prompt_metadata["entries"]:
                if not isinstance(entry, dict):
                    continue
                section_name = self._optional_text(
                    entry.get("section_name") or entry.get("prompt_section_name")
                )
                candidate_hash = self._optional_text(entry.get("prompt_candidate_hash"))
                if section_name and candidate_hash:
                    candidate_entries.append((section_name, candidate_hash, entry))

            if len(candidate_entries) == 1:
                section_name, candidate_hash, entry = candidate_entries[0]
                metadata["section_name"] = section_name
                metadata["prompt_section_name"] = section_name
                metadata["prompt_candidate_hash"] = candidate_hash
                metadata["provider"] = self._optional_text(entry.get("provider")) or metadata.get(
                    "provider"
                )

        return {key: value for key, value in metadata.items() if value is not None}

    def _build_adapter_config(self) -> AdapterConfig:
        """Resolve the adapter config, including any explicit prompt binding."""
        adapter_config = self.config.adapter_config or AdapterConfig()
        if not self.config.prompt_candidate_hash or not self.config.prompt_section_name:
            return adapter_config
        return replace(
            adapter_config,
            prompt_binding=PromptOptimizationBinding(
                section_name=self.config.prompt_section_name,
                prompt_candidate_hash=self.config.prompt_candidate_hash,
            ),
        )

    def get_progress(self, instance_id: str) -> Optional[TaskProgress]:
        """Get progress for a specific task.

        Args:
            instance_id: Task instance ID

        Returns:
            TaskProgress or None if not found
        """
        return self._progress.get(instance_id)

    def get_summary(self) -> EvaluationSummary:
        """Get current evaluation summary.

        Returns:
            EvaluationSummary with current state
        """
        return self._summary


async def run_swe_bench_evaluation(
    dataset_path: Optional[Path] = None,
    agent_profile: str = "default",
    output_dir: Path = Path("./swe_bench_results"),
    max_tasks: int = 0,
    max_parallel: int = 1,
    instance_ids: Optional[list[str]] = None,
    repos: Optional[list[str]] = None,
    verbose: bool = False,
) -> CorrelationReport:
    """Convenience function to run SWE-bench evaluation.

    Args:
        dataset_path: Path to JSONL dataset file
        agent_profile: Victor profile name
        output_dir: Directory for results
        max_tasks: Maximum tasks to run (0 = all)
        max_parallel: Maximum parallel tasks
        instance_ids: Specific instance IDs to run
        repos: Specific repos to filter
        verbose: Enable verbose output

    Returns:
        CorrelationReport with results
    """
    config = OrchestratorConfig(
        dataset_path=dataset_path,
        agent_profile=agent_profile,
        output_dir=output_dir,
        max_tasks=max_tasks,
        max_parallel=max_parallel,
        instance_ids=instance_ids or [],
        repos=repos or [],
    )

    def progress_callback(progress: TaskProgress, stage: EvaluationStage) -> None:
        if verbose:
            logger.info(f"[{progress.instance_id}] Stage: {stage.value}")

    orchestrator = EvaluationOrchestrator(config, progress_callback)
    return await orchestrator.run_evaluation()
