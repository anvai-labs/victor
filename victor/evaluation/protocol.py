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

"""Evaluation protocol types for benchmark harnesses.

Defines data structures for running evaluations against
benchmarks like SWE-bench, HumanEval, MBPP, etc.
"""

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from victor.evaluation.topology_feedback import aggregate_topology_feedback
from victor.evaluation.planning_feedback import aggregate_planning_feedback
from victor.evaluation.degradation_feedback import aggregate_degradation_feedback
from victor.evaluation.team_feedback import aggregate_team_feedback


@dataclass
class TokenUsage:
    """Token usage metrics for evaluation tracking.

    Normalized across providers:
    - DeepSeek: prompt_cache_hit_tokens, prompt_cache_miss_tokens
    - OpenAI/xAI: prompt_tokens_details.cached_tokens, reasoning_tokens
    - Anthropic: cache_creation_input_tokens, cache_read_input_tokens
    - xAI: cost_in_usd_ticks (direct cost)

    Supports addition for aggregation across turns.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    # Cache tracking (normalized across providers)
    cached_tokens: int = 0
    cache_miss_tokens: int = 0
    # Reasoning/thinking tokens (xAI, OpenAI o-series)
    reasoning_tokens: int = 0
    # Direct cost tracking (xAI provides this)
    cost_usd_micros: int = 0  # Cost in millionths of USD (0 = not available)

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        """Add two TokenUsage instances for aggregation."""
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
            cache_miss_tokens=self.cache_miss_tokens + other.cache_miss_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
            cost_usd_micros=self.cost_usd_micros + other.cost_usd_micros,
        )

    def __iadd__(self, other: "TokenUsage") -> "TokenUsage":
        """In-place addition for aggregation."""
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.total_tokens += other.total_tokens
        self.cached_tokens += other.cached_tokens
        self.cache_miss_tokens += other.cache_miss_tokens
        self.reasoning_tokens += other.reasoning_tokens
        self.cost_usd_micros += other.cost_usd_micros
        return self

    @staticmethod
    def from_provider_usage(usage: dict, raw_usage: dict = None) -> "TokenUsage":
        """Create TokenUsage from provider-specific usage dict.

        Normalizes across DeepSeek, OpenAI, xAI, and Anthropic formats.
        """
        raw = raw_usage or {}
        prompt_details = raw.get("prompt_tokens_details", {}) or {}
        completion_details = raw.get("completion_tokens_details", {}) or {}

        return TokenUsage(
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            # Cache: OpenAI/xAI use prompt_tokens_details.cached_tokens
            # DeepSeek uses prompt_cache_hit_tokens
            # Anthropic uses cache_read_input_tokens
            cached_tokens=(
                prompt_details.get("cached_tokens", 0)
                or raw.get("prompt_cache_hit_tokens", 0)
                or raw.get("cache_read_input_tokens", 0)
            ),
            cache_miss_tokens=raw.get("prompt_cache_miss_tokens", 0),
            # Reasoning: xAI/OpenAI o-series
            reasoning_tokens=completion_details.get("reasoning_tokens", 0),
            # Direct cost: xAI provides cost_in_usd_ticks
            cost_usd_micros=raw.get("cost_in_usd_ticks", 0),
        )


class BenchmarkType(Enum):
    """Types of coding benchmarks."""

    SWE_BENCH = "swe_bench"  # Software engineering tasks
    HUMAN_EVAL = "human_eval"  # Code generation
    MBPP = "mbpp"  # Basic programming
    LIVE_CODE_BENCH = "live_code_bench"  # Live coding evaluation
    BIG_CODE_BENCH = "big_code_bench"  # Large-scale code tasks
    AIDER_POLYGLOT = "aider_polyglot"  # Multi-language tasks
    DR3_EVAL = "dr3_eval"  # Deep-research report evaluation
    CLAW_BENCH = "claw_bench"  # GUI/web computer-use evaluation
    GUIDE = "guide"  # Grounded multimodal UI agent evaluation
    VLAA_GUI = "vlaa_gui"  # Vision-language action benchmark for GUI tasks
    CUSTOM = "custom"  # User-defined benchmark


class TaskStatus(Enum):
    """Status of a benchmark task."""

    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"


class BenchmarkFailureCategory(Enum):
    """Normalized failure categories for benchmark reporting."""

    PATCH_APPLICATION = "patch_application"
    TEST_FAILURE = "test_failure"
    TOOL_USAGE = "tool_usage"
    TASK_COMPLETION = "task_completion"
    TIMEOUT = "timeout"
    EXECUTION_ERROR = "execution_error"
    ENVIRONMENT_ERROR = "environment_error"
    UNSUPPORTED_CLAIM = "unsupported_claim"
    UNKNOWN = "unknown"


class FailureStage(Enum):
    """Hierarchical failure stage aligned to agentic benchmark workflows."""

    PERCEPTION = "perception"
    ACTION = "action"
    GROUNDING = "grounding"
    VALIDATION = "validation"
    EXECUTION = "execution"
    ENVIRONMENT = "environment"
    COMPLETION = "completion"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class FailureDiagnosis:
    """Structured failure diagnosis preserving stage and subtype detail."""

    stage: FailureStage
    category: BenchmarkFailureCategory
    subtype: str = "generic"
    retryable: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def path(self) -> str:
        """Stable taxonomy path for reporting and aggregation."""
        return f"{self.stage.value}.{self.category.value}.{self.subtype}"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a report-friendly dictionary."""
        return {
            "stage": self.stage.value,
            "category": self.category.value,
            "subtype": self.subtype,
            "path": self.path,
            "retryable": self.retryable,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FailureDiagnosis":
        """Reconstruct a structured diagnosis from serialized data."""
        return cls(
            stage=FailureStage(payload["stage"]),
            category=BenchmarkFailureCategory(payload["category"]),
            subtype=str(payload.get("subtype") or "generic"),
            retryable=bool(payload.get("retryable", False)),
            metadata=dict(payload.get("metadata") or {}),
        )


class ConfidenceBucket(Enum):
    """Bucketized confidence for truth-aligned reporting."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True)
class ConfidenceAssessment:
    """Derived task confidence and uncertainty summary."""

    confidence_score: float
    uncertainty: float
    evidence_score: float
    bucket: ConfidenceBucket
    truth_aligned: bool
    signals: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to report-friendly output."""
        return {
            "confidence_score": self.confidence_score,
            "uncertainty": self.uncertainty,
            "evidence_score": self.evidence_score,
            "bucket": self.bucket.value,
            "truth_aligned": self.truth_aligned,
            "signals": dict(self.signals),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ConfidenceAssessment":
        """Reconstruct a confidence assessment from serialized data."""
        return cls(
            confidence_score=float(payload.get("confidence_score", 0.0)),
            uncertainty=float(payload.get("uncertainty", 1.0)),
            evidence_score=float(payload.get("evidence_score", 0.0)),
            bucket=ConfidenceBucket(payload.get("bucket", ConfidenceBucket.LOW.value)),
            truth_aligned=bool(payload.get("truth_aligned", False)),
            signals=dict(payload.get("signals") or {}),
        )


class EvaluationMetric(Enum):
    """Standard evaluation metrics."""

    PASS_AT_K = "pass@k"  # Pass rate at k attempts
    EXACT_MATCH = "exact_match"  # Exact solution match
    PARTIAL_MATCH = "partial_match"  # Partial solution match
    TEST_PASS_RATE = "test_pass_rate"  # Test passing rate
    EDIT_DISTANCE = "edit_distance"  # Code edit distance
    BLEU = "bleu"  # BLEU score for code
    CODE_BLEU = "code_bleu"  # CodeBLEU score
    SYNTAX_VALID = "syntax_valid"  # Syntactically valid
    SEMANTIC_EQUIV = "semantic_equiv"  # Semantically equivalent


@dataclass
class BenchmarkTask:
    """A single task in a benchmark."""

    task_id: str
    benchmark: BenchmarkType
    description: str
    language: str = "python"
    # Polyglot / multi-version runtime hints (consumed by the docker eval
    # backend). For SWE-bench the official per-instance image pins the runtime,
    # so language_version is informational there; for other benchmarks it is the
    # authoritative version (e.g. "3.9", "20", "1.22").
    language_version: Optional[str] = None
    docker_image: Optional[str] = None
    runtime_command: Optional[list[str]] = None

    # Input context
    prompt: str = ""
    context_code: str = ""
    test_code: str = ""
    # Read-only local inputs that must be staged into an ephemeral benchmark
    # workspace. Keys are workspace-relative destination paths; values are
    # validated source file paths. Unlike ``seed_files``, these may be binary
    # attachments (PDFs, images, videos), so they are copied rather than read
    # into memory as text.
    input_files: dict[str, str] = field(default_factory=dict)
    # SWE-bench FAIL_TO_PASS: the specific test node IDs that must pass after
    # the fix. The test runner uses these to build a targeted test command
    # (rather than scanning test_code), and the closed-loop verify gate (P1)
    # checks exactly these. Carried from SWEBenchInstance — previously dropped
    # in to_benchmark_task, so the runner's getattr(task, "fail_to_pass")
    # always returned None.
    fail_to_pass: list[str] = field(default_factory=list)
    seed_files: dict[str, str] = field(default_factory=dict)

    # Repository info (for SWE-bench)
    repo: Optional[str] = None
    base_commit: Optional[str] = None
    issue_text: Optional[str] = None
    hints: list[str] = field(default_factory=list)

    # Expected solution
    solution: Optional[str] = None
    patch: Optional[str] = None

    # Metadata
    difficulty: str = "medium"
    category: str = ""
    tags: list[str] = field(default_factory=list)
    timeout_seconds: int = 300

    # Task complexity override - if set, framework uses this instead of inference
    # Valid values: "simple", "medium", "complex", "generation", "action", "analysis"
    # If None, framework infers complexity from task description
    complexity_override: Optional[str] = None

    # Task type hint - if set, _infer_task_type() uses this instead of keyword inference.
    # Valid values match FrameworkTaskType names: "edit", "create", "search", "analyze",
    # "execute", "chat", "explore".  If None, keyword-based inference is used.
    task_type_hint: Optional[str] = None


@dataclass
class CodeQualityMetrics:
    """Code quality metrics for generated code."""

    # Syntax and style
    syntax_valid: bool = True
    lint_errors: int = 0
    lint_warnings: int = 0
    style_score: float = 1.0  # 0.0 to 1.0

    # Complexity
    cyclomatic_complexity: float = 0.0
    cognitive_complexity: float = 0.0
    maintainability_index: float = 100.0  # 0-100 scale

    # Structure
    lines_of_code: int = 0
    functions_count: int = 0
    classes_count: int = 0
    duplicate_lines: int = 0

    # Type safety
    type_coverage: float = 0.0  # For typed languages

    def get_overall_score(self) -> float:
        """Calculate overall code quality score (0-100)."""
        weights = {
            "syntax": 0.3,
            "style": 0.2,
            "complexity": 0.25,
            "maintainability": 0.25,
        }

        syntax_score = 100.0 if self.syntax_valid else 0.0
        style_score = self.style_score * 100
        complexity_score = max(0, 100 - (self.cyclomatic_complexity * 5))
        maint_score = self.maintainability_index

        return (
            weights["syntax"] * syntax_score
            + weights["style"] * style_score
            + weights["complexity"] * complexity_score
            + weights["maintainability"] * maint_score
        )


@dataclass
class TaskResult:
    """Result of running a single task."""

    task_id: str
    status: TaskStatus
    generated_code: str = ""
    generated_patch: str = ""

    # Test results
    tests_passed: int = 0
    tests_failed: int = 0
    tests_total: int = 0

    # Timing
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0

    # Agent metrics
    tokens_used: int = 0
    tokens_input: int = 0  # Input tokens (prompt)
    tokens_output: int = 0  # Output tokens (completion)
    cached_tokens: int = 0  # Prompt cache hits
    reasoning_tokens: int = 0  # Hidden reasoning/thinking tokens
    cost_usd_micros: int = 0  # Direct API cost (xAI)
    tool_calls: int = 0
    turns: int = 0
    code_search_calls: int = 0
    graph_calls: int = 0

    # Code quality metrics
    code_quality: Optional[CodeQualityMetrics] = None

    # Partial completion scoring (0.0 to 1.0)
    completion_score: float = 0.0
    partial_tests_weight: float = 0.0  # Weighted partial test success

    # Pass@k tracking
    attempts: int = 1
    successful_attempts: int = 0

    # Error info
    error_message: str = ""
    traceback: str = ""
    failure_category: Optional[BenchmarkFailureCategory] = None
    failure_details: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    failure_diagnosis: Optional[FailureDiagnosis] = None
    confidence_assessment: Optional[ConfidenceAssessment] = None

    # Polyglot / containerized-eval provenance (for per-language reporting,
    # debugging, and orphan-container cleanup).
    language: Optional[str] = None
    eval_backend: Optional[str] = None  # "local" | "docker"
    runtime_image: Optional[str] = None
    container_id: Optional[str] = None

    # Correlation spine + bounded execution trace (messages, tool calls, edits).
    # session_id joins this task's decisions (logged via log_decision) to its
    # outcome for the execution manifest → classifier training data.
    session_id: str = ""
    trace: dict[str, Any] = field(default_factory=dict)

    # Detailed output
    stdout: str = ""
    stderr: str = ""

    @property
    def test_pass_rate(self) -> float:
        """Calculate test pass rate."""
        if self.tests_total == 0:
            return 0.0
        return self.tests_passed / self.tests_total

    @property
    def is_success(self) -> bool:
        """Whether the task was successful."""
        return self.status == TaskStatus.PASSED

    @property
    def tokens_per_test(self) -> float:
        """Token efficiency: tokens used per test passed."""
        if self.tests_passed == 0:
            return float("inf")
        return self.tokens_used / self.tests_passed

    @property
    def cost_efficiency(self) -> float:
        """Cost efficiency score (higher is better)."""
        if self.tokens_used == 0:
            return 0.0
        # Score based on success per 1K tokens
        return (self.completion_score * 1000) / self.tokens_used

    @property
    def code_intelligence_calls(self) -> int:
        """Combined `code_search` and `graph` calls."""
        return self.code_search_calls + self.graph_calls

    @property
    def used_code_search(self) -> bool:
        """Whether the task used `code_search` at least once."""
        return self.code_search_calls > 0

    @property
    def used_graph(self) -> bool:
        """Whether the task used `graph` at least once."""
        return self.graph_calls > 0

    @property
    def used_code_intelligence(self) -> bool:
        """Whether the task used either `code_search` or `graph`."""
        return self.code_intelligence_calls > 0

    @property
    def has_patch_artifact(self) -> bool:
        """Whether the task produced code or patch output that could be merged."""
        return bool(self.generated_patch or self.generated_code)

    @property
    def accepted_patch(self) -> bool:
        """Whether the task produced an accepted patch outcome."""
        if "accepted_patch" in self.metadata:
            return bool(self.metadata.get("accepted_patch"))
        return self.status == TaskStatus.PASSED and self.has_patch_artifact

    def get_failure_diagnosis(self) -> Optional[FailureDiagnosis]:
        """Return a structured failure diagnosis, deriving one when needed."""
        if self.failure_diagnosis is not None:
            return self.failure_diagnosis
        if self.failure_category is None:
            return None
        return derive_failure_diagnosis(self)

    @property
    def failure_taxonomy_path(self) -> Optional[str]:
        """Stable hierarchical taxonomy path for this task failure."""
        diagnosis = self.get_failure_diagnosis()
        if diagnosis is None:
            return None
        return diagnosis.path

    def get_confidence_assessment(self) -> ConfidenceAssessment:
        """Return truth-aligned confidence and uncertainty for this task."""
        if self.confidence_assessment is not None:
            return self.confidence_assessment
        return derive_confidence_assessment(self)

    def calculate_completion_score(self) -> float:
        """Calculate partial completion score based on tests and quality."""
        score = 0.0

        # Test-based score (60% weight)
        if self.tests_total > 0:
            test_score = self.tests_passed / self.tests_total
            score += test_score * 0.6

        # Code quality score (20% weight)
        if self.code_quality:
            quality_score = self.code_quality.get_overall_score() / 100
            score += quality_score * 0.2

        # Basic completion (20% weight) - did it generate valid code?
        if self.generated_code and self.status != TaskStatus.ERROR:
            score += 0.2

        return min(score, 1.0)


@dataclass
class EvaluationConfig:
    """Configuration for an evaluation run.

    The optional prompt-optimization fields let benchmark artifacts identify the
    exact prompt candidate and section under evaluation. That keeps offline
    evaluation output reusable by frontier seeding instead of relying solely on
    runtime traces.
    """

    benchmark: BenchmarkType
    model: str
    provider: Optional[str] = None
    prompt_candidate_hash: Optional[str] = None
    prompt_section_name: Optional[str] = None
    max_tasks: Optional[int] = None
    timeout_per_task: int = 300
    max_retries: int = 1
    parallel_tasks: int = 1

    # Filtering
    task_ids: Optional[list[str]] = None
    languages: Optional[list[str]] = None
    categories: Optional[list[str]] = None
    difficulties: Optional[list[str]] = None

    # Agent config
    temperature: float = 0.0
    max_tokens: int = 4096
    max_turns: int = 10

    # Environment
    # use_docker is a back-compat alias: True ⇒ eval_backend "docker". Default
    # is False so the default backend stays "local" (host) unless the caller
    # opts into containers via use_docker=True or eval_backend="docker".
    use_docker: bool = False
    docker_image: str = "python:3.11"
    workspace_dir: Optional[Path] = None
    dataset_metadata: dict[str, Any] = field(default_factory=dict)

    # Containerized / polyglot eval backend. `eval_backend` is the modern
    # switch ("local" = host subprocess [default for back-compat], "docker" =
    # run each task in a container with the correct runtime). `use_docker=True`
    # is treated as `eval_backend="docker"` for back-compat. `swebench_image_*`
    # select the official per-instance SWE-bench images (the dataset carries no
    # python_version, so the official image is the source of truth).
    eval_backend: str = "local"
    runtime_version: Optional[str] = None
    docker_image_override: Optional[str] = None
    swebench_image_source: str = "official"  # official | build | skip
    swebench_image_registry: str = "docker.io/swebench"
    # Closed-loop verify-and-retry gate (PR2 #404). 0 = OFF (default — the gate
    # adds latency without benefit for current models; opt-in for stronger ones
    # that can capitalize on a retry). >0 = max times to feed FAIL_TO_PASS
    # failures back into the loop after the agent claims done.
    max_verify_retries: int = 0

    # Self-correction settings (generic iterative refinement)
    enable_self_correction: bool = False
    self_correction_max_iterations: int = 3
    auto_fix_imports: bool = True

    def to_artifact_config(self) -> dict[str, Any]:
        """Serialize stable evaluation identity for saved benchmark artifacts."""
        return {
            "benchmark": self.benchmark.value,
            "model": self.model,
            "provider": self.provider,
            "prompt_candidate_hash": self.prompt_candidate_hash,
            "section_name": self.prompt_section_name,
            "prompt_section_name": self.prompt_section_name,
            "max_tasks": self.max_tasks,
            "timeout_per_task": self.timeout_per_task,
            "max_turns": self.max_turns,
            "parallel_tasks": self.parallel_tasks,
            "dataset_metadata": dict(self.dataset_metadata),
        }


@dataclass
class EvaluationResult:
    """Result of a complete evaluation run."""

    config: EvaluationConfig
    task_results: list[TaskResult] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    # Self-correction metrics (populated when enable_self_correction=True)
    correction_metrics: Optional[dict[str, Any]] = None

    @property
    def total_tasks(self) -> int:
        """Total number of tasks evaluated."""
        return len(self.task_results)

    @property
    def passed_tasks(self) -> int:
        """Number of passed tasks."""
        return sum(1 for r in self.task_results if r.status == TaskStatus.PASSED)

    @property
    def failed_tasks(self) -> int:
        """Number of failed tasks."""
        return sum(1 for r in self.task_results if r.status == TaskStatus.FAILED)

    @property
    def error_tasks(self) -> int:
        """Number of tasks with errors."""
        return sum(1 for r in self.task_results if r.status == TaskStatus.ERROR)

    @property
    def timeout_tasks(self) -> int:
        """Number of timed out tasks."""
        return sum(1 for r in self.task_results if r.status == TaskStatus.TIMEOUT)

    @property
    def pass_rate(self) -> float:
        """Overall pass rate."""
        if self.total_tasks == 0:
            return 0.0
        return self.passed_tasks / self.total_tasks

    @property
    def duration_seconds(self) -> float:
        """Total evaluation duration."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return sum(r.duration_seconds for r in self.task_results)

    @property
    def total_tokens(self) -> int:
        """Total tokens used."""
        return sum(r.tokens_used for r in self.task_results)

    @property
    def total_tool_calls(self) -> int:
        """Total tool calls made."""
        return sum(r.tool_calls for r in self.task_results)

    @property
    def total_code_search_calls(self) -> int:
        """Total `code_search` calls across all tasks."""
        return sum(r.code_search_calls for r in self.task_results)

    @property
    def total_graph_calls(self) -> int:
        """Total `graph` calls across all tasks."""
        return sum(r.graph_calls for r in self.task_results)

    @property
    def tasks_using_code_search(self) -> int:
        """Number of tasks that used `code_search`."""
        return sum(1 for r in self.task_results if r.used_code_search)

    @property
    def tasks_using_graph(self) -> int:
        """Number of tasks that used `graph`."""
        return sum(1 for r in self.task_results if r.used_graph)

    @property
    def tasks_using_code_intelligence(self) -> int:
        """Number of tasks that used `code_search` or `graph`."""
        return sum(1 for r in self.task_results if r.used_code_intelligence)

    def get_by_status(self, status: TaskStatus) -> list[TaskResult]:
        """Get results by status."""
        return [r for r in self.task_results if r.status == status]

    def get_by_language(self, language: str) -> list[TaskResult]:
        """Get results by language (requires task metadata)."""
        # Would need task_id -> task mapping
        return self.task_results

    def get_metrics(self) -> dict[str, Any]:
        """Get summary metrics including extended token tracking."""
        cached = sum(r.cached_tokens for r in self.task_results)
        reasoning = sum(r.reasoning_tokens for r in self.task_results)
        cost_micros = sum(r.cost_usd_micros for r in self.task_results)
        code_search_calls = self.total_code_search_calls
        graph_calls = self.total_graph_calls
        code_intelligence_tasks = self.tasks_using_code_intelligence
        failure_categories = Counter()
        failure_stages = Counter()
        failure_taxonomy = Counter()
        confidence_buckets = Counter()
        confidence_scores: list[float] = []
        uncertainty_scores: list[float] = []
        truth_aligned_tasks = 0
        overconfident_failures = 0
        underconfident_passes = 0
        accepted_patch_results = [
            task_result for task_result in self.task_results if task_result.accepted_patch
        ]
        code_intelligence_results = [
            task_result for task_result in self.task_results if task_result.used_code_intelligence
        ]
        non_code_intelligence_results = [
            task_result
            for task_result in self.task_results
            if not task_result.used_code_intelligence
        ]
        time_to_first_tool_call_values: list[float] = []
        time_to_first_edit_values: list[float] = []
        for task_result in self.task_results:
            if task_result.failure_category is not None:
                failure_categories[task_result.failure_category.value] += 1
            diagnosis = task_result.get_failure_diagnosis()
            if diagnosis is not None:
                failure_stages[diagnosis.stage.value] += 1
                failure_taxonomy[diagnosis.path] += 1
            confidence = task_result.get_confidence_assessment()
            confidence_buckets[confidence.bucket.value] += 1
            confidence_scores.append(confidence.confidence_score)
            uncertainty_scores.append(confidence.uncertainty)
            if confidence.truth_aligned:
                truth_aligned_tasks += 1
            if (
                task_result.status != TaskStatus.PASSED
                and confidence.bucket == ConfidenceBucket.HIGH
            ):
                overconfident_failures += 1
            if (
                task_result.status == TaskStatus.PASSED
                and confidence.bucket == ConfidenceBucket.LOW
            ):
                underconfident_passes += 1
            tool_call_time = task_result.metadata.get("time_to_first_tool_call_seconds")
            try:
                if tool_call_time is not None:
                    time_to_first_tool_call_values.append(float(tool_call_time))
            except (TypeError, ValueError):
                pass
            edit_time = task_result.metadata.get("time_to_first_edit_seconds")
            try:
                if edit_time is not None:
                    time_to_first_edit_values.append(float(edit_time))
            except (TypeError, ValueError):
                pass
        accepted_patch_count = len(accepted_patch_results)
        accepted_patch_cost_micros = sum(
            result.cost_usd_micros for result in accepted_patch_results
        )

        def _pass_rate(results: list[TaskResult]) -> float:
            return sum(1 for result in results if result.status == TaskStatus.PASSED) / max(
                1, len(results)
            )

        def _avg_tokens(results: list[TaskResult]) -> float:
            return sum(result.tokens_used for result in results) / max(1, len(results))

        def _avg_duration(results: list[TaskResult]) -> float:
            return sum(result.duration_seconds for result in results) / max(1, len(results))

        def _accepted_patch_rate(results: list[TaskResult]) -> float:
            return sum(1 for result in results if result.accepted_patch) / max(1, len(results))

        metrics = {
            "total_tasks": self.total_tasks,
            "passed": self.passed_tasks,
            "failed": self.failed_tasks,
            "errors": self.error_tasks,
            "timeouts": self.timeout_tasks,
            "pass_rate": self.pass_rate,
            "duration_seconds": self.duration_seconds,
            "total_tokens": self.total_tokens,
            "total_tool_calls": self.total_tool_calls,
            "total_code_search_calls": code_search_calls,
            "total_graph_calls": graph_calls,
            "total_code_intelligence_calls": code_search_calls + graph_calls,
            "tasks_using_code_search": self.tasks_using_code_search,
            "tasks_using_graph": self.tasks_using_graph,
            "tasks_using_code_intelligence": code_intelligence_tasks,
            "tasks_without_code_intelligence": max(0, self.total_tasks - code_intelligence_tasks),
            "code_intelligence_task_coverage": code_intelligence_tasks / max(1, self.total_tasks),
            "code_intelligence_pass_rate": _pass_rate(code_intelligence_results),
            "non_code_intelligence_pass_rate": _pass_rate(non_code_intelligence_results),
            "code_intelligence_avg_tokens_per_task": _avg_tokens(code_intelligence_results),
            "non_code_intelligence_avg_tokens_per_task": _avg_tokens(non_code_intelligence_results),
            "code_intelligence_avg_duration_seconds": _avg_duration(code_intelligence_results),
            "non_code_intelligence_avg_duration_seconds": _avg_duration(
                non_code_intelligence_results
            ),
            "code_intelligence_accepted_patch_rate": _accepted_patch_rate(
                code_intelligence_results
            ),
            "non_code_intelligence_accepted_patch_rate": _accepted_patch_rate(
                non_code_intelligence_results
            ),
            "avg_tokens_per_task": self.total_tokens / max(1, self.total_tasks),
            "avg_duration_per_task": self.duration_seconds / max(1, self.total_tasks),
            # Extended token metrics
            "cached_tokens": cached,
            "reasoning_tokens": reasoning,
            "cost_usd_micros": cost_micros,
            "failure_categories": dict(failure_categories),
            "failure_stages": dict(failure_stages),
            "failure_taxonomy": dict(failure_taxonomy),
            "avg_confidence_score": sum(confidence_scores) / max(1, len(confidence_scores)),
            "avg_uncertainty": sum(uncertainty_scores) / max(1, len(uncertainty_scores)),
            "confidence_buckets": dict(confidence_buckets),
            "truth_alignment_rate": truth_aligned_tasks / max(1, self.total_tasks),
            "overconfidence_rate": overconfident_failures / max(1, self.total_tasks),
            "underconfidence_rate": underconfident_passes / max(1, self.total_tasks),
            "accepted_patch_count": accepted_patch_count,
            "accepted_patch_rate": accepted_patch_count / max(1, self.total_tasks),
            "tokens_to_merge_total": sum(result.tokens_used for result in accepted_patch_results),
            "avg_tokens_to_merge": (
                sum(result.tokens_used for result in accepted_patch_results) / accepted_patch_count
                if accepted_patch_count > 0
                else 0.0
            ),
            "accepted_patch_cost_usd_total": accepted_patch_cost_micros / 1_000_000,
            "cost_per_accepted_patch_usd": (
                (accepted_patch_cost_micros / 1_000_000) / accepted_patch_count
                if accepted_patch_count > 0
                else 0.0
            ),
            "tasks_with_time_to_first_tool_call": len(time_to_first_tool_call_values),
            "avg_time_to_first_tool_call_seconds": (
                sum(time_to_first_tool_call_values) / len(time_to_first_tool_call_values)
                if time_to_first_tool_call_values
                else 0.0
            ),
            "tasks_with_time_to_first_edit": len(time_to_first_edit_values),
            "avg_time_to_first_edit_seconds": (
                sum(time_to_first_edit_values) / len(time_to_first_edit_values)
                if time_to_first_edit_values
                else 0.0
            ),
        }
        metrics.update(aggregate_planning_feedback(self.task_results, total_tasks=self.total_tasks))
        metrics.update(aggregate_topology_feedback(self.task_results, total_tasks=self.total_tasks))
        metrics.update(
            aggregate_degradation_feedback(self.task_results, total_tasks=self.total_tasks)
        )
        metrics.update(aggregate_team_feedback(self.task_results, total_tasks=self.total_tasks))
        return metrics


def derive_failure_diagnosis(result: TaskResult) -> FailureDiagnosis:
    """Infer a structured failure diagnosis from a task result."""
    if result.failure_category is None:
        raise ValueError("Cannot derive failure diagnosis without a failure category")

    details = dict(result.failure_details)
    error_message = result.error_message.lower()

    def subset(*keys: str) -> dict[str, Any]:
        return {key: details[key] for key in keys if key in details and details[key]}

    if result.failure_category == BenchmarkFailureCategory.TOOL_USAGE:
        if details.get("forbidden_action_hits"):
            return FailureDiagnosis(
                stage=FailureStage.ACTION,
                category=result.failure_category,
                subtype="forbidden_action",
                retryable=False,
                metadata=subset("forbidden_action_hits"),
            )
        return FailureDiagnosis(
            stage=FailureStage.ACTION,
            category=result.failure_category,
            subtype="tool_policy_violation",
            retryable=True,
            metadata=dict(details),
        )

    if result.failure_category == BenchmarkFailureCategory.TASK_COMPLETION:
        if details.get("missing_actions"):
            return FailureDiagnosis(
                stage=FailureStage.ACTION,
                category=result.failure_category,
                subtype="missing_required_actions",
                retryable=True,
                metadata=subset("missing_actions"),
            )
        if details.get("missing_claims") or details.get("missing_citations"):
            return FailureDiagnosis(
                stage=FailureStage.GROUNDING,
                category=result.failure_category,
                subtype="missing_required_evidence",
                retryable=True,
                metadata=subset("missing_claims", "missing_citations"),
            )
        if details.get("missing_answer_phrases"):
            return FailureDiagnosis(
                stage=FailureStage.COMPLETION,
                category=result.failure_category,
                subtype="missing_answer_coverage",
                retryable=True,
                metadata=subset("missing_answer_phrases"),
            )
        if "no generated report" in error_message or "no generated output" in error_message:
            return FailureDiagnosis(
                stage=FailureStage.COMPLETION,
                category=result.failure_category,
                subtype="empty_output",
                retryable=True,
                metadata={},
            )
        return FailureDiagnosis(
            stage=FailureStage.COMPLETION,
            category=result.failure_category,
            subtype="incomplete_task",
            retryable=True,
            metadata=dict(details),
        )

    if result.failure_category == BenchmarkFailureCategory.UNSUPPORTED_CLAIM:
        if details.get("forbidden_claim_hits"):
            return FailureDiagnosis(
                stage=FailureStage.GROUNDING,
                category=result.failure_category,
                subtype="forbidden_claim",
                retryable=True,
                metadata=subset("forbidden_claim_hits"),
            )
        return FailureDiagnosis(
            stage=FailureStage.GROUNDING,
            category=result.failure_category,
            subtype="unsupported_claim",
            retryable=True,
            metadata=dict(details),
        )

    if result.failure_category == BenchmarkFailureCategory.PATCH_APPLICATION:
        return FailureDiagnosis(
            stage=FailureStage.EXECUTION,
            category=result.failure_category,
            subtype="patch_did_not_apply",
            retryable=True,
            metadata={},
        )

    if result.failure_category == BenchmarkFailureCategory.TEST_FAILURE:
        subtype = "partial_test_failure" if result.tests_passed > 0 else "all_tests_failed"
        return FailureDiagnosis(
            stage=FailureStage.VALIDATION,
            category=result.failure_category,
            subtype=subtype,
            retryable=True,
            metadata={
                "tests_passed": result.tests_passed,
                "tests_failed": result.tests_failed,
                "tests_total": result.tests_total,
            },
        )

    if result.failure_category == BenchmarkFailureCategory.TIMEOUT:
        subtype = "test_timeout" if "test" in error_message else "task_timeout"
        return FailureDiagnosis(
            stage=FailureStage.EXECUTION,
            category=result.failure_category,
            subtype=subtype,
            retryable=True,
            metadata={},
        )

    if result.failure_category == BenchmarkFailureCategory.EXECUTION_ERROR:
        return FailureDiagnosis(
            stage=FailureStage.EXECUTION,
            category=result.failure_category,
            subtype="runtime_exception",
            retryable=True,
            metadata={},
        )

    if result.failure_category == BenchmarkFailureCategory.ENVIRONMENT_ERROR:
        subtype = "test_runner_error" if "test runner" in error_message else "environment_setup"
        return FailureDiagnosis(
            stage=FailureStage.ENVIRONMENT,
            category=result.failure_category,
            subtype=subtype,
            retryable=True,
            metadata={},
        )

    return FailureDiagnosis(
        stage=FailureStage.UNKNOWN,
        category=result.failure_category,
        subtype="unknown",
        retryable=False,
        metadata=dict(details),
    )


def derive_confidence_assessment(result: TaskResult) -> ConfidenceAssessment:
    """Infer truth-aligned confidence from task evidence and failure state."""
    evidence_signals: list[float] = []

    if result.tests_total > 0:
        evidence_signals.append(result.tests_passed / max(1, result.tests_total))

    for key in (
        "claim_coverage",
        "citation_coverage",
        "action_coverage",
        "answer_coverage",
    ):
        value = result.failure_details.get(key)
        if isinstance(value, (int, float)):
            evidence_signals.append(float(value))

    if result.completion_score > 0:
        evidence_signals.append(float(result.completion_score))

    evidence_score = sum(evidence_signals) / max(1, len(evidence_signals))

    if result.status == TaskStatus.PASSED:
        confidence_score = min(0.99, 0.6 + (0.4 * evidence_score))
    else:
        failure_factor = {
            BenchmarkFailureCategory.UNSUPPORTED_CLAIM: 0.2,
            BenchmarkFailureCategory.TOOL_USAGE: 0.25,
            BenchmarkFailureCategory.TASK_COMPLETION: 0.35,
            BenchmarkFailureCategory.TEST_FAILURE: 0.45,
            BenchmarkFailureCategory.PATCH_APPLICATION: 0.3,
            BenchmarkFailureCategory.TIMEOUT: 0.25,
            BenchmarkFailureCategory.EXECUTION_ERROR: 0.15,
            BenchmarkFailureCategory.ENVIRONMENT_ERROR: 0.1,
            BenchmarkFailureCategory.UNKNOWN: 0.2,
        }.get(result.failure_category, 0.2)
        confidence_score = min(0.49, evidence_score * failure_factor)

    uncertainty = max(0.0, 1.0 - confidence_score)
    if confidence_score >= 0.67:
        bucket = ConfidenceBucket.HIGH
    elif confidence_score >= 0.34:
        bucket = ConfidenceBucket.MEDIUM
    else:
        bucket = ConfidenceBucket.LOW

    truth_aligned = (result.status == TaskStatus.PASSED and bucket != ConfidenceBucket.LOW) or (
        result.status != TaskStatus.PASSED and bucket != ConfidenceBucket.HIGH
    )

    return ConfidenceAssessment(
        confidence_score=round(confidence_score, 4),
        uncertainty=round(uncertainty, 4),
        evidence_score=round(evidence_score, 4),
        bucket=bucket,
        truth_aligned=truth_aligned,
        signals={
            "status": result.status.value,
            "failure_category": (
                result.failure_category.value if result.failure_category is not None else None
            ),
            "completion_score": result.completion_score,
            "tests_passed": result.tests_passed,
            "tests_total": result.tests_total,
        },
    )


@dataclass
class BenchmarkMetadata:
    """Metadata about a benchmark dataset."""

    name: str
    type: BenchmarkType
    version: str
    total_tasks: int
    languages: list[str]
    categories: list[str]
    description: str = ""
    source_name: str = ""
    source_url: str = ""
    paper_url: str = ""
    aliases: tuple[str, ...] = ()
    evaluation_mode: str = "code"
    runner_status: str = "implemented"

    def to_dict(self) -> dict[str, Any]:
        """Serialize benchmark metadata for reporting."""
        return {
            "name": self.name,
            "type": self.type.value,
            "version": self.version,
            "total_tasks": self.total_tasks,
            "languages": list(self.languages),
            "categories": list(self.categories),
            "description": self.description,
            "source_name": self.source_name,
            "source_url": self.source_url,
            "paper_url": self.paper_url,
            "aliases": list(self.aliases),
            "evaluation_mode": self.evaluation_mode,
            "runner_status": self.runner_status,
        }


@dataclass
class LeaderboardEntry:
    """Entry for evaluation leaderboard."""

    model: str
    benchmark: str
    pass_rate: float
    timestamp: datetime
    total_tasks: int
    passed_tasks: int
    avg_tokens: float
    avg_duration: float
    config: dict = field(default_factory=dict)


BENCHMARK_CATALOG: tuple[BenchmarkMetadata, ...] = (
    BenchmarkMetadata(
        name="swe-bench",
        type=BenchmarkType.SWE_BENCH,
        version="1.0",
        total_tasks=2300,
        languages=["python"],
        categories=["software-engineering", "agentic"],
        description="Real-world GitHub issue resolution benchmark.",
        source_name="Princeton NLP",
        aliases=("swe_bench",),
        evaluation_mode="agentic",
    ),
    BenchmarkMetadata(
        name="swe-bench-lite",
        type=BenchmarkType.SWE_BENCH,
        version="1.0",
        total_tasks=300,
        languages=["python"],
        categories=["software-engineering", "agentic"],
        description="Curated subset of SWE-bench for faster evaluation.",
        source_name="Princeton NLP",
        aliases=("swe_bench_lite",),
        evaluation_mode="agentic",
    ),
    BenchmarkMetadata(
        name="humaneval",
        type=BenchmarkType.HUMAN_EVAL,
        version="1.0",
        total_tasks=164,
        languages=["python"],
        categories=["code-generation"],
        description="Code generation from docstrings.",
        source_name="OpenAI",
        aliases=("human-eval", "human_eval"),
        evaluation_mode="code-generation",
    ),
    BenchmarkMetadata(
        name="mbpp",
        type=BenchmarkType.MBPP,
        version="1.0",
        total_tasks=974,
        languages=["python"],
        categories=["code-generation"],
        description="Mostly Basic Python Problems.",
        source_name="Google Research",
        evaluation_mode="code-generation",
    ),
    BenchmarkMetadata(
        name="mbpp-test",
        type=BenchmarkType.MBPP,
        version="1.0",
        total_tasks=500,
        languages=["python"],
        categories=["code-generation"],
        description="MBPP test split.",
        source_name="Google Research",
        aliases=("mbpp_test",),
        evaluation_mode="code-generation",
    ),
    BenchmarkMetadata(
        name="dr3-eval",
        type=BenchmarkType.DR3_EVAL,
        version="1.0",
        total_tasks=0,
        languages=["report"],
        categories=["agentic", "deep-research", "reporting"],
        description="Deep-research report benchmark with claim and citation checks.",
        source_name="DR3-Eval",
        aliases=("dr3_eval", "dr3"),
        evaluation_mode="deep-research",
    ),
    BenchmarkMetadata(
        name="clawbench",
        type=BenchmarkType.CLAW_BENCH,
        version="1.0",
        total_tasks=0,
        languages=["gui", "web"],
        categories=["agentic", "perception", "computer-use"],
        description="External GUI benchmark for computer-use agent evaluation.",
        source_name="Research",
        aliases=("claw-bench", "claw_bench"),
        evaluation_mode="agentic-perception",
        runner_status="implemented",
    ),
    BenchmarkMetadata(
        name="guide",
        type=BenchmarkType.GUIDE,
        version="1.0",
        total_tasks=0,
        languages=["gui", "web"],
        categories=["agentic", "perception", "grounding"],
        description="External grounded UI agent benchmark coverage.",
        source_name="Research",
        evaluation_mode="agentic-perception",
        runner_status="implemented",
    ),
    BenchmarkMetadata(
        name="vlaa-gui",
        type=BenchmarkType.VLAA_GUI,
        version="1.0",
        total_tasks=0,
        languages=["gui"],
        categories=["agentic", "perception", "gui-actions"],
        description="External GUI action benchmark for vision-language agents.",
        source_name="Research",
        aliases=("vlaa_gui",),
        evaluation_mode="agentic-perception",
        runner_status="implemented",
    ),
)

_BENCHMARK_NAME_INDEX = {
    alias: metadata
    for metadata in BENCHMARK_CATALOG
    for alias in (metadata.name, *metadata.aliases)
}

EXTERNAL_AGENTIC_BENCHMARK_TYPES = frozenset(
    {
        BenchmarkType.CLAW_BENCH,
        BenchmarkType.GUIDE,
        BenchmarkType.VLAA_GUI,
    }
)

BROWSER_TASK_BENCHMARK_TYPES = EXTERNAL_AGENTIC_BENCHMARK_TYPES
LOCAL_MANIFEST_BENCHMARK_TYPES = frozenset(
    {
        BenchmarkType.DR3_EVAL,
        *BROWSER_TASK_BENCHMARK_TYPES,
    }
)


def normalize_benchmark_name(name: str) -> str:
    """Normalize benchmark names for CLI and metadata lookup."""
    return name.strip().lower().replace("_", "-")


def get_benchmark_catalog() -> tuple[BenchmarkMetadata, ...]:
    """Return benchmark metadata in CLI display order."""
    return BENCHMARK_CATALOG


def get_benchmark_metadata(
    benchmark: BenchmarkType | str,
) -> Optional[BenchmarkMetadata]:
    """Resolve benchmark metadata from enum type or CLI name."""
    if isinstance(benchmark, BenchmarkType):
        return next((item for item in BENCHMARK_CATALOG if item.type == benchmark), None)
    return _BENCHMARK_NAME_INDEX.get(normalize_benchmark_name(benchmark))


def is_external_agentic_benchmark(benchmark: BenchmarkType | str) -> bool:
    """Return True when the benchmark is an external perception-heavy agentic task."""
    metadata = get_benchmark_metadata(benchmark)
    if metadata is None:
        return False
    return metadata.type in EXTERNAL_AGENTIC_BENCHMARK_TYPES


def is_browser_task_benchmark(benchmark: BenchmarkType | str) -> bool:
    """Return True for browser/web-task benchmarks with action-trace evaluation."""
    metadata = get_benchmark_metadata(benchmark)
    if metadata is None:
        return False
    return metadata.type in BROWSER_TASK_BENCHMARK_TYPES


def requires_local_manifest_benchmark(benchmark: BenchmarkType | str) -> bool:
    """Return True when a benchmark runner requires a local dataset manifest."""
    metadata = get_benchmark_metadata(benchmark)
    if metadata is None:
        return False
    return metadata.type in LOCAL_MANIFEST_BENCHMARK_TYPES
