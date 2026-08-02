# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
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

"""Evaluation harness for benchmark testing.

This module provides infrastructure for evaluating coding agents against
standardized benchmarks like SWE-bench, HumanEval, MBPP, etc.

Example usage:
    from victor.evaluation import (
        EvaluationHarness,
        EvaluationConfig,
        BenchmarkType,
        SWEBenchRunner,
        get_harness,
    )
    from victor.core.async_utils import run_sync

    async def run_agent(task):
        '''Your agent implementation.'''
        # ... agent logic ...
        return agent_output

    async def evaluate():
        # Get harness and register runner
        harness = get_harness()
        harness.register_runner(SWEBenchRunner())

        # Configure evaluation
        config = EvaluationConfig(
            benchmark=BenchmarkType.SWE_BENCH,
            model="claude-3-opus",
            max_tasks=10,
            timeout_per_task=300,
        )

        # Run evaluation
        result = await harness.run_evaluation(config, run_agent)

        # Generate report
        print(harness.generate_report(result, format="text"))
        print(f"Pass rate: {result.pass_rate:.1%}")

    run_sync(evaluate())
"""

from victor.evaluation.protocol import (
    BenchmarkFailureCategory,
    ConfidenceAssessment,
    ConfidenceBucket,
    FailureDiagnosis,
    FailureStage,
    BenchmarkMetadata,
    BenchmarkTask,
    BenchmarkType,
    CodeQualityMetrics,
    EvaluationConfig,
    EvaluationMetric,
    EvaluationResult,
    LeaderboardEntry,
    TaskResult,
    TaskStatus,
)
from victor.evaluation.harness import (
    BaseBenchmarkRunner,
    BenchmarkRunner,
    EvaluationHarness,
    PromptCandidateEvaluationRun,
    PromptCandidateEvaluationSpec,
    PromptCandidateEvaluationSuiteResult,
    TaskEnvironment,
    bind_prompt_candidate_evaluation_config,
    load_prompt_candidate_evaluation_suite,
    get_harness,
)
from victor.evaluation.benchmarks import (
    BrowserTaskBenchmarkRunner,
    DeepResearchBenchmarkRunner,
    ExternalAgenticBenchmarkRunner,
    HumanEvalRunner,
    MBPPRunner,
    SWEBenchRunner,
)
from victor.evaluation.code_quality import (
    CodeQualityAnalyzer,
    BatchCodeAnalyzer,
    LintResult,
)
from victor.evaluation.pass_at_k import (
    PassAtKEvaluator,
    PassAtKResult,
    AggregatePassAtKResult,
    GreedyVsSamplingComparison,
    pass_at_k,
    generate_report as generate_pass_at_k_report,
)
from victor.evaluation.analyzers import (
    AnalyzerRegistry,
    get_code_quality_analyzer,
    get_pass_at_k_evaluator,
)
from victor.evaluation.services import (
    materialize_validated_session_truth_service,
    parse_validated_session_truth_legacy_kwargs,
    ValidatedSessionTruthServiceProtocol,
    ValidatedSessionTruthService,
    create_validated_session_truth_service,
    resolve_validated_session_truth_service,
)
from victor.evaluation.planning_feedback import (
    aggregate_planning_feedback,
    extract_planning_events,
    summarize_planning_feedback,
)
from victor.evaluation.degradation_feedback import (
    aggregate_degradation_feedback,
    extract_degradation_events,
    summarize_degradation_feedback,
)
from victor.evaluation.team_feedback import (
    aggregate_team_feedback,
    extract_team_feedback_artifacts,
    summarize_team_feedback,
)

# Code generation harness (HumanEval, MBPP - provider-only, no tools)
from victor.evaluation.code_generation_harness import (
    CodeGenMetrics,
    CodeGenResult,
    CodeGenerationRunner,
    CodeGenerationBenchmark,
    create_code_gen_runner,
)

# Agentic harness (SWE-bench, Aider Polyglot - with tools and file editing)
from victor.evaluation.agentic_harness import (
    AgenticValidationType,
    EvalToolCall,
    FileEdit,
    AgenticExecutionTrace,
    AgenticTaskResult,
    AgenticMetrics,
    AgenticValidator,
    PatchApplicationValidator,
    TestPassingValidator,
    FileEditValidator,
    ToolUsageValidator,
    AgenticBenchmarkRunner,
    generate_agentic_report,
)

# Trajectory-level evaluation (EVR-1, FEP-0008 Phase E)
from victor.evaluation.trajectory_eval import (
    TrajectoryDimension,
    DimensionScore,
    TrajectoryScore,
    DimensionInterval,
    IntervalStat,
    BatteryResult,
    DimensionScorer,
    EffectGroundingScorer,  # ADR-010 / EVR-4 — not in default_scorers() by design
    TrajectoryEvaluator,
    default_scorers,
    mean_confidence_interval,
)

# Regression-gated harness acceptance oracle + HTIR traces (EVR-5, ADR-012)
from victor.evaluation.htir import (
    Role,
    StepStatus,
    ArtifactEffect,
    ETCLOVGLayer,
    HTIRStep,
    HTIRTrace,
    normalize,
    attribute_failure_layer,
)
from victor.evaluation.acceptance_oracle import (
    HarnessConfig,
    AcceptanceVerdict,
    DimensionDelta,
    CharacterizationDelta,
    AcceptanceReport,
    HarnessAcceptanceOracle,
    characterization_from_signatures,
)

# Agent adapter (connects Victor orchestrator to agentic benchmarks)
from victor.evaluation.agent_adapter import (
    AdapterConfig,
    VictorAgentAdapter,
    create_victor_agent_callback,
    run_agentic_task,
)

# SWE-bench dataset loader and workspace management
from victor.evaluation.swe_bench_loader import (
    SWEBenchConfig,
    SWEBenchInstance,
    SWEBenchLoader,
    SWEBenchWorkspaceManager,
    load_swe_bench_tasks,
    setup_swe_bench_workspace,
    get_swe_bench_repos,
)

# Multi-language test runners for agentic benchmarks
from victor.evaluation.test_runners import (
    Language,
    TestRunnerConfig,
    TestResult,
    TestRunResults,
    BaseTestRunner,
    PythonTestRunner,
    JavaScriptTestRunner,
    GoTestRunner,
    RustTestRunner,
    JavaTestRunner,
    TestRunnerRegistry,
    detect_language,
    is_test_tool_available,
)

# Environment setup utilities
from victor.evaluation.env_setup import (
    SetupStrategy,
    SetupResult,
    EnvironmentConfig,
    EnvironmentSetup,
    validate_environment,
    quick_setup,
)

# Baseline test validation for SWE-bench
from victor.evaluation.baseline_validator import (
    BaselineStatus,
    TestBaseline,
    BaselineValidationResult,
    BaselineCache,
    BaselineValidator,
    quick_validate_baseline,
    check_environment_health,
    TestStatus,
    get_test_status,
)

# Test result correlation and scoring
from victor.evaluation.result_correlation import (
    FailureCategory,
    TestCorrelation,
    SWEBenchScore,
    CorrelationReport,
    ResultCorrelator,
    correlate_validation_results,
    analyze_failure_patterns,
    save_correlation_report,
)

# Evaluation orchestrator for end-to-end SWE-bench evaluation
from victor.evaluation.evaluation_orchestrator import (
    EvaluationStage,
    TaskProgress,
    OrchestratorConfig,
    EvaluationSummary,
    EvaluationOrchestrator,
    run_swe_bench_evaluation,
)

# Judge reliability gating (EVR-2, ADR-011)
from victor.evaluation.judge_calibration import (
    JudgeReliability,
    GateDecision,
    JudgeReliabilityGate,
    OrderSwapEnsembleJudge,
    cohens_kappa,
    krippendorff_alpha,
    evaluate_judge_agreement,
)

# Offline judge-calibration harness + verifiable corpus (EVR-2, ADR-011)
from victor.evaluation.judge_calibration_harness import (
    CalibrationReport,
    CalibrationSample,
    JudgeCalibrationHarness,
    PersistentLoopRunner,
    Transcript,
    TranscriptStep,
    VerifiableTask,
    alternating_scripted_executor,
    hard_scripted_executor,
    make_outcome_executor,
    make_scripted_executor,
)
from victor.evaluation.calibration_corpus import (
    default_corpus as default_calibration_corpus,
)
from victor.evaluation.calibration_agent_executor import (
    execute_verifiable_task,
    make_agent_executor,
    trace_to_transcript,
    verifiable_to_benchmark_task,
)
from victor.evaluation.calibration_rubric_judge import (
    JudgeCallStats,
    make_llm_rubric_judge,
    make_provider_complete_fn,
    make_rubric_judge,
    render_judged_content,
)

__all__ = [
    # Protocol types
    "BenchmarkFailureCategory",
    "ConfidenceAssessment",
    "ConfidenceBucket",
    "FailureDiagnosis",
    "FailureStage",
    "BenchmarkMetadata",
    "BenchmarkTask",
    "BenchmarkType",
    "CodeQualityMetrics",
    "EvaluationConfig",
    "EvaluationMetric",
    "EvaluationResult",
    "LeaderboardEntry",
    "TaskResult",
    "TaskStatus",
    # Harness
    "BaseBenchmarkRunner",
    "BenchmarkRunner",
    "EvaluationHarness",
    "PromptCandidateEvaluationRun",
    "PromptCandidateEvaluationSpec",
    "PromptCandidateEvaluationSuiteResult",
    "TaskEnvironment",
    "bind_prompt_candidate_evaluation_config",
    "load_prompt_candidate_evaluation_suite",
    "get_harness",
    # Benchmark runners
    "BrowserTaskBenchmarkRunner",
    "DeepResearchBenchmarkRunner",
    "ExternalAgenticBenchmarkRunner",
    "HumanEvalRunner",
    "MBPPRunner",
    "SWEBenchRunner",
    # Code quality
    "CodeQualityAnalyzer",
    "BatchCodeAnalyzer",
    "LintResult",
    # Pass@k
    "PassAtKEvaluator",
    "PassAtKResult",
    "AggregatePassAtKResult",
    "GreedyVsSamplingComparison",
    "pass_at_k",
    "generate_pass_at_k_report",
    # Registry
    "AnalyzerRegistry",
    "get_code_quality_analyzer",
    "get_pass_at_k_evaluator",
    "materialize_validated_session_truth_service",
    "parse_validated_session_truth_legacy_kwargs",
    "ValidatedSessionTruthServiceProtocol",
    "ValidatedSessionTruthService",
    "create_validated_session_truth_service",
    "resolve_validated_session_truth_service",
    # Planning feedback helpers
    "aggregate_planning_feedback",
    "extract_planning_events",
    "summarize_planning_feedback",
    "aggregate_team_feedback",
    "extract_team_feedback_artifacts",
    "summarize_team_feedback",
    "aggregate_degradation_feedback",
    "extract_degradation_events",
    "summarize_degradation_feedback",
    # Code generation harness (HumanEval, MBPP - provider-only)
    "CodeGenMetrics",
    "CodeGenResult",
    "CodeGenerationRunner",
    "CodeGenerationBenchmark",
    "create_code_gen_runner",
    # Agentic harness (SWE-bench, Aider Polyglot - with tools)
    "AgenticValidationType",
    "EvalToolCall",
    "FileEdit",
    "AgenticExecutionTrace",
    "AgenticTaskResult",
    "AgenticMetrics",
    "AgenticValidator",
    "PatchApplicationValidator",
    "TestPassingValidator",
    "FileEditValidator",
    "ToolUsageValidator",
    "AgenticBenchmarkRunner",
    "generate_agentic_report",
    # SWE-bench loader and workspace management
    "SWEBenchConfig",
    "SWEBenchInstance",
    "SWEBenchLoader",
    "SWEBenchWorkspaceManager",
    "load_swe_bench_tasks",
    "setup_swe_bench_workspace",
    "get_swe_bench_repos",
    # Multi-language test runners
    "Language",
    "TestRunnerConfig",
    "TestResult",
    "TestRunResults",
    "BaseTestRunner",
    "PythonTestRunner",
    "JavaScriptTestRunner",
    "GoTestRunner",
    "RustTestRunner",
    "JavaTestRunner",
    "TestRunnerRegistry",
    "detect_language",
    "is_test_tool_available",
    # Environment setup utilities
    "SetupStrategy",
    "SetupResult",
    "EnvironmentConfig",
    "EnvironmentSetup",
    "validate_environment",
    "quick_setup",
    # Baseline test validation
    "BaselineStatus",
    "TestBaseline",
    "BaselineValidationResult",
    "BaselineCache",
    "BaselineValidator",
    "quick_validate_baseline",
    "check_environment_health",
    "TestStatus",
    "get_test_status",
    # Test result correlation and scoring
    "FailureCategory",
    "TestCorrelation",
    "SWEBenchScore",
    "CorrelationReport",
    "ResultCorrelator",
    "correlate_validation_results",
    "analyze_failure_patterns",
    "save_correlation_report",
    # Evaluation orchestrator
    "EvaluationStage",
    "TaskProgress",
    "OrchestratorConfig",
    "EvaluationSummary",
    "EvaluationOrchestrator",
    "run_swe_bench_evaluation",
    # Trajectory-level evaluation (EVR-1)
    "TrajectoryDimension",
    "DimensionScore",
    "TrajectoryScore",
    "DimensionInterval",
    "IntervalStat",
    "BatteryResult",
    "DimensionScorer",
    "EffectGroundingScorer",
    "TrajectoryEvaluator",
    "default_scorers",
    "mean_confidence_interval",
    # Acceptance oracle + HTIR traces (EVR-5, ADR-012)
    "Role",
    "StepStatus",
    "ArtifactEffect",
    "ETCLOVGLayer",
    "HTIRStep",
    "HTIRTrace",
    "normalize",
    "attribute_failure_layer",
    "HarnessConfig",
    "AcceptanceVerdict",
    "DimensionDelta",
    "CharacterizationDelta",
    "AcceptanceReport",
    "HarnessAcceptanceOracle",
    "characterization_from_signatures",
    # Judge reliability gating (EVR-2)
    "JudgeReliability",
    "GateDecision",
    "JudgeReliabilityGate",
    "OrderSwapEnsembleJudge",
    "cohens_kappa",
    "krippendorff_alpha",
    "evaluate_judge_agreement",
    # Offline judge-calibration harness + corpus (EVR-2)
    "CalibrationReport",
    "CalibrationSample",
    "JudgeCalibrationHarness",
    "PersistentLoopRunner",
    "Transcript",
    "TranscriptStep",
    "VerifiableTask",
    "alternating_scripted_executor",
    "hard_scripted_executor",
    "make_outcome_executor",
    "make_scripted_executor",
    "default_calibration_corpus",
    "execute_verifiable_task",
    "make_agent_executor",
    "trace_to_transcript",
    "verifiable_to_benchmark_task",
    "JudgeCallStats",
    "make_llm_rubric_judge",
    "make_provider_complete_fn",
    "make_rubric_judge",
    "render_judged_content",
]
