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

"""Framework comparison benchmarks for AI coding assistants.

Provides standardized comparisons against other AI coding frameworks
including Aider, Claude Code, Cursor, and others.

This module defines:
- Standard metrics for framework comparison
- Leaderboard data structures
- Comparison report generation
"""

import json
import hashlib
import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from victor.evaluation.protocol import (
    BenchmarkType,
    EvaluationConfig,
    EvaluationResult,
    LeaderboardEntry,
    get_benchmark_catalog,
    get_benchmark_metadata,
    normalize_benchmark_name,
)
from victor.evaluation.team_feedback import aggregate_team_feedback

logger = logging.getLogger(__name__)

DEFAULT_FIXTURE_SET_ROOT = Path("tests/fixtures/benchmarks")


class Framework(Enum):
    """Known AI coding frameworks for comparison."""

    VICTOR = "victor"
    AIDER = "aider"
    CLAUDE_CODE = "claude_code"
    CURSOR = "cursor"
    GITHUB_COPILOT = "github_copilot"
    CODY = "cody"
    CONTINUE = "continue"
    TABBY = "tabby"
    CODEGPT = "codegpt"
    CUSTOM = "custom"


@dataclass
class FrameworkCapabilities:
    """Capabilities of a framework for comparison."""

    name: str
    framework: Framework

    # Core capabilities
    code_generation: bool = True
    code_editing: bool = True
    code_review: bool = False
    test_generation: bool = False
    documentation: bool = False
    refactoring: bool = False

    # Agent capabilities
    multi_file_editing: bool = False
    tool_use: bool = False
    autonomous_mode: bool = False
    planning: bool = False

    # Model support
    supported_models: list[str] = field(default_factory=list)
    default_model: str = ""

    # Infrastructure
    local_models: bool = False
    air_gapped: bool = False
    mcp_support: bool = False

    # Pricing/licensing
    open_source: bool = False
    free_tier: bool = False


# Known framework capabilities (based on public information)
FRAMEWORK_CAPABILITIES = {
    Framework.VICTOR: FrameworkCapabilities(
        name="Victor",
        framework=Framework.VICTOR,
        code_generation=True,
        code_editing=True,
        code_review=True,
        test_generation=True,
        documentation=True,
        refactoring=True,
        multi_file_editing=True,
        tool_use=True,
        autonomous_mode=True,
        planning=True,
        supported_models=["claude-3", "gpt-4", "gemini", "ollama/*", "lmstudio/*"],
        default_model="claude-3-sonnet",
        local_models=True,
        air_gapped=True,
        mcp_support=True,
        open_source=True,
        free_tier=True,
    ),
    Framework.AIDER: FrameworkCapabilities(
        name="Aider",
        framework=Framework.AIDER,
        code_generation=True,
        code_editing=True,
        code_review=False,
        test_generation=False,
        documentation=False,
        refactoring=True,
        multi_file_editing=True,
        tool_use=False,
        autonomous_mode=True,
        planning=False,
        supported_models=["claude-3", "gpt-4", "gemini", "ollama/*"],
        default_model="claude-3-sonnet",
        local_models=True,
        air_gapped=False,
        mcp_support=False,
        open_source=True,
        free_tier=True,
    ),
    Framework.CLAUDE_CODE: FrameworkCapabilities(
        name="Claude Code",
        framework=Framework.CLAUDE_CODE,
        code_generation=True,
        code_editing=True,
        code_review=True,
        test_generation=True,
        documentation=True,
        refactoring=True,
        multi_file_editing=True,
        tool_use=True,
        autonomous_mode=True,
        planning=True,
        supported_models=["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"],
        default_model="claude-3-sonnet",
        local_models=False,
        air_gapped=False,
        mcp_support=True,
        open_source=False,
        free_tier=False,
    ),
    Framework.CURSOR: FrameworkCapabilities(
        name="Cursor",
        framework=Framework.CURSOR,
        code_generation=True,
        code_editing=True,
        code_review=False,
        test_generation=False,
        documentation=False,
        refactoring=True,
        multi_file_editing=True,
        tool_use=False,
        autonomous_mode=False,
        planning=False,
        supported_models=["gpt-4", "claude-3"],
        default_model="gpt-4",
        local_models=False,
        air_gapped=False,
        mcp_support=False,
        open_source=False,
        free_tier=True,
    ),
}


@dataclass
class ComparisonMetrics:
    """Metrics for framework comparison."""

    # Performance
    pass_rate: float = 0.0  # Tasks passed / total
    avg_latency_ms: float = 0.0  # Average response time
    tokens_per_task: float = 0.0  # Average tokens used

    # Quality
    code_quality_score: float = 0.0  # 0-100 scale
    test_pass_rate: float = 0.0  # Test success rate
    partial_completion: float = 0.0  # Partial credit score

    # Efficiency
    cost_per_task: float = 0.0  # Estimated cost
    turns_per_task: float = 0.0  # Conversation turns
    tool_calls_per_task: float = 0.0  # Tool usage
    accepted_patch_rate: float = 0.0  # Accepted patches / total tasks
    tokens_to_merge: float = 0.0  # Average tokens for accepted patches
    cost_per_accepted_patch_usd: float = 0.0  # Cost per accepted patch
    avg_time_to_first_edit_seconds: float = 0.0  # Time to first edit
    avg_time_to_first_tool_call_seconds: float = 0.0  # Time to first tool call

    # Robustness
    error_rate: float = 0.0  # Tasks with errors
    timeout_rate: float = 0.0  # Tasks that timed out

    # Tool-selection effects
    code_intelligence_task_coverage: float = 0.0
    code_intelligence_pass_rate: float = 0.0
    non_code_intelligence_pass_rate: float = 0.0

    # Workspace-policy effects
    workspace_policy_task_coverage: float = 0.0
    workspace_policy_pass_rate: float = 0.0
    non_workspace_policy_pass_rate: float = 0.0
    workspace_policy_pass_delta: float = 0.0
    workspace_policy_materialize_rate: float = 0.0
    workspace_policy_dry_run_rate: float = 0.0
    workspace_policy_auto_merge_rate: float = 0.0
    workspace_policy_cleanup_disabled_rate: float = 0.0
    workspace_diagnostic_task_coverage: float = 0.0
    workspace_diagnostic_rate: float = 0.0


@dataclass
class FrameworkResult:
    """Result of evaluating a framework on a benchmark."""

    framework: Framework
    benchmark: BenchmarkType
    model: str
    metrics: ComparisonMetrics
    timestamp: datetime = field(default_factory=datetime.now)
    config: dict = field(default_factory=dict)

    # Raw results for detailed analysis
    task_results: list[dict] = field(default_factory=list)


@dataclass
class ComparisonReport:
    """Comparison report across multiple frameworks."""

    benchmark: BenchmarkType
    results: list[FrameworkResult] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    def get_leaderboard(self) -> list[tuple[Framework, float]]:
        """Get leaderboard sorted by pass rate."""
        return sorted(
            [(r.framework, r.metrics.pass_rate) for r in self.results],
            key=lambda x: x[1],
            reverse=True,
        )

    def get_winner(self) -> Optional[Framework]:
        """Get the framework with highest pass rate."""
        leaderboard = self.get_leaderboard()
        return leaderboard[0][0] if leaderboard else None

    def to_markdown(self) -> str:
        """Generate markdown comparison table."""
        lines = [
            f"# Framework Comparison: {self.benchmark.value}",
            "",
            f"Generated: {self.timestamp.isoformat()}",
            "",
            "## Leaderboard",
            "",
            "| Rank | Framework | Model | Pass Rate | Avg Latency | Tokens/Task | Cost/Task | Source |",
            "|------|-----------|-------|-----------|-------------|-------------|-----------|--------|",
        ]

        for i, result in enumerate(
            sorted(self.results, key=lambda r: r.metrics.pass_rate, reverse=True)
        ):
            lines.append(
                f"| {i+1} | {result.framework.value} | "
                f"{result.model} | "
                f"{result.metrics.pass_rate:.1%} | "
                f"{result.metrics.avg_latency_ms:.0f}ms | "
                f"{result.metrics.tokens_per_task:.0f} | "
                f"${result.metrics.cost_per_task:.4f} | "
                f"{result.config.get('source', '')} |"
            )

        lines.extend(
            [
                "",
                "## Detailed Metrics",
                "",
            ]
        )

        for result in self.results:
            m = result.metrics
            lines.extend(
                [
                    f"### {result.framework.value}",
                    "",
                    f"- **Model**: {result.model}",
                    f"- **Pass Rate**: {m.pass_rate:.1%}",
                    f"- **Accepted Patch Rate**: {m.accepted_patch_rate:.1%}",
                    f"- **Tokens to Merge**: {m.tokens_to_merge:.1f}",
                    f"- **Cost per Accepted Patch**: ${m.cost_per_accepted_patch_usd:.4f}",
                    f"- **Time to First Edit**: {m.avg_time_to_first_edit_seconds:.2f}s",
                    f"- **Time to First Tool Call**: {m.avg_time_to_first_tool_call_seconds:.2f}s",
                    (
                        f"- **Code-Intelligence Coverage**: "
                        f"{m.code_intelligence_task_coverage:.1%}"
                    ),
                    (
                        f"- **Code-Intelligence Pass Delta**: "
                        f"{(m.code_intelligence_pass_rate - m.non_code_intelligence_pass_rate):+.1%}"
                    ),
                    f"- **Workspace-Policy Coverage**: {m.workspace_policy_task_coverage:.1%}",
                    f"- **Workspace-Policy Pass Delta**: {m.workspace_policy_pass_delta:+.1%}",
                    (
                        f"- **Workspace Materialize/Dry-Run/Auto-Merge Rates**: "
                        f"{m.workspace_policy_materialize_rate:.1%} / "
                        f"{m.workspace_policy_dry_run_rate:.1%} / "
                        f"{m.workspace_policy_auto_merge_rate:.1%}"
                    ),
                    (
                        f"- **Workspace Diagnostic Coverage**: "
                        f"{m.workspace_diagnostic_task_coverage:.1%}"
                    ),
                    f"- **Code Quality**: {m.code_quality_score:.1f}/100",
                    f"- **Test Pass Rate**: {m.test_pass_rate:.1%}",
                    f"- **Error Rate**: {m.error_rate:.1%}",
                    f"- **Timeout Rate**: {m.timeout_rate:.1%}",
                    "",
                ]
            )

        return "\n".join(lines)

    def to_json(self) -> str:
        """Export report as JSON."""
        data = {
            "benchmark": self.benchmark.value,
            "timestamp": self.timestamp.isoformat(),
            "results": [
                {
                    "framework": r.framework.value,
                    "model": r.model,
                    "timestamp": r.timestamp.isoformat(),
                    "metrics": {
                        "pass_rate": r.metrics.pass_rate,
                        "avg_latency_ms": r.metrics.avg_latency_ms,
                        "tokens_per_task": r.metrics.tokens_per_task,
                        "code_quality_score": r.metrics.code_quality_score,
                        "cost_per_task": r.metrics.cost_per_task,
                        "accepted_patch_rate": r.metrics.accepted_patch_rate,
                        "tokens_to_merge": r.metrics.tokens_to_merge,
                        "cost_per_accepted_patch_usd": r.metrics.cost_per_accepted_patch_usd,
                        "avg_time_to_first_edit_seconds": (
                            r.metrics.avg_time_to_first_edit_seconds
                        ),
                        "avg_time_to_first_tool_call_seconds": (
                            r.metrics.avg_time_to_first_tool_call_seconds
                        ),
                        "test_pass_rate": r.metrics.test_pass_rate,
                        "partial_completion": r.metrics.partial_completion,
                        "error_rate": r.metrics.error_rate,
                        "timeout_rate": r.metrics.timeout_rate,
                        "code_intelligence_task_coverage": (
                            r.metrics.code_intelligence_task_coverage
                        ),
                        "code_intelligence_pass_rate": r.metrics.code_intelligence_pass_rate,
                        "non_code_intelligence_pass_rate": (
                            r.metrics.non_code_intelligence_pass_rate
                        ),
                        "workspace_policy_task_coverage": (
                            r.metrics.workspace_policy_task_coverage
                        ),
                        "workspace_policy_pass_rate": r.metrics.workspace_policy_pass_rate,
                        "non_workspace_policy_pass_rate": (
                            r.metrics.non_workspace_policy_pass_rate
                        ),
                        "workspace_policy_pass_delta": r.metrics.workspace_policy_pass_delta,
                        "workspace_policy_materialize_rate": (
                            r.metrics.workspace_policy_materialize_rate
                        ),
                        "workspace_policy_dry_run_rate": r.metrics.workspace_policy_dry_run_rate,
                        "workspace_policy_auto_merge_rate": (
                            r.metrics.workspace_policy_auto_merge_rate
                        ),
                        "workspace_policy_cleanup_disabled_rate": (
                            r.metrics.workspace_policy_cleanup_disabled_rate
                        ),
                        "workspace_diagnostic_task_coverage": (
                            r.metrics.workspace_diagnostic_task_coverage
                        ),
                        "workspace_diagnostic_rate": r.metrics.workspace_diagnostic_rate,
                    },
                    "config": r.config,
                }
                for r in self.results
            ],
        }
        return json.dumps(data, indent=2)


@dataclass(frozen=True)
class FixtureSetDescriptor:
    """Descriptor for a stable saved benchmark fixture set."""

    name: str
    benchmark: str
    manifest_path: Path
    artifact_count: int
    models: tuple[str, ...] = ()
    sources: tuple[str, ...] = ()


@dataclass(frozen=True)
class FixtureBenchmarkDescriptor:
    """Descriptor for a stable benchmark corpus built from checked-in fixture sets."""

    benchmark: str
    fixture_set_count: int
    artifact_count: int
    models: tuple[str, ...] = ()
    fixture_set_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class FixtureSetVerificationResult:
    """Verification result for a stable saved benchmark fixture set."""

    name: str
    benchmark: str
    manifest_path: Path
    artifact_count: int
    verified_artifact_count: int


def canonicalize_fixture_benchmark_name(name: str) -> str:
    """Canonicalize fixture benchmark identifiers through benchmark metadata aliases."""
    normalized_name = str(name).strip()
    if not normalized_name:
        return ""
    metadata = get_benchmark_metadata(normalized_name)
    if metadata is not None:
        return metadata.name
    return normalize_benchmark_name(normalized_name)


def fixture_benchmark_matches(candidate: str, benchmark: str) -> bool:
    """Return True when a fixture benchmark name matches a requested benchmark alias."""
    candidate_key = canonicalize_fixture_benchmark_name(candidate)
    benchmark_key = canonicalize_fixture_benchmark_name(benchmark)
    return bool(candidate_key and benchmark_key and candidate_key == benchmark_key)


def _build_fixture_benchmark_metadata_payload(benchmark: str) -> dict[str, Any]:
    """Build catalog metadata for a fixture benchmark when benchmark metadata exists."""
    metadata = get_benchmark_metadata(benchmark)
    if metadata is None:
        return {}
    payload: dict[str, Any] = {
        "catalog_name": metadata.name,
        "benchmark_source_name": metadata.source_name,
        "description": metadata.description,
        "evaluation_mode": metadata.evaluation_mode,
        "runner_status": metadata.runner_status,
        "languages": list(metadata.languages),
        "categories": list(metadata.categories),
    }
    aliases = [alias for alias in metadata.aliases if alias]
    if aliases:
        payload["aliases"] = aliases
    return payload


def _select_catalog_benchmark_metadata(benchmark: Optional[str]) -> list[Any]:
    """Return benchmark-catalog entries relevant to a fixture catalog query."""
    metadata_entries = list(get_benchmark_catalog())
    if benchmark is None:
        return metadata_entries
    return [
        metadata
        for metadata in metadata_entries
        if fixture_benchmark_matches(metadata.name, benchmark)
    ]


def compute_metrics_from_result(result: EvaluationResult) -> ComparisonMetrics:
    """Compute comparison metrics from an evaluation result."""
    metrics = ComparisonMetrics()

    if result.total_tasks == 0:
        return metrics

    summary_metrics = result.get_metrics()

    # Performance
    metrics.pass_rate = result.pass_rate
    metrics.avg_latency_ms = (result.duration_seconds * 1000) / result.total_tasks

    # Calculate average tokens
    total_tokens = sum(r.tokens_used for r in result.task_results)
    metrics.tokens_per_task = total_tokens / result.total_tasks

    # Quality - calculate from task results
    quality_scores = [
        r.code_quality.get_overall_score() for r in result.task_results if r.code_quality
    ]
    if quality_scores:
        metrics.code_quality_score = sum(quality_scores) / len(quality_scores)

    # Test pass rate across all tasks
    total_tests = sum(r.tests_total for r in result.task_results)
    passed_tests = sum(r.tests_passed for r in result.task_results)
    if total_tests > 0:
        metrics.test_pass_rate = passed_tests / total_tests

    # Partial completion
    completion_scores = [r.completion_score for r in result.task_results]
    if completion_scores:
        metrics.partial_completion = sum(completion_scores) / len(completion_scores)

    # Efficiency
    total_turns = sum(r.turns for r in result.task_results)
    metrics.turns_per_task = total_turns / result.total_tasks

    total_tool_calls = sum(r.tool_calls for r in result.task_results)
    metrics.tool_calls_per_task = total_tool_calls / result.total_tasks
    metrics.accepted_patch_rate = _safe_float(summary_metrics.get("accepted_patch_rate"))
    metrics.tokens_to_merge = _safe_float(summary_metrics.get("avg_tokens_to_merge"))
    metrics.cost_per_accepted_patch_usd = _safe_float(
        summary_metrics.get("cost_per_accepted_patch_usd")
    )
    metrics.avg_time_to_first_edit_seconds = _safe_float(
        summary_metrics.get("avg_time_to_first_edit_seconds")
    )
    metrics.avg_time_to_first_tool_call_seconds = _safe_float(
        summary_metrics.get("avg_time_to_first_tool_call_seconds")
    )
    metrics.code_intelligence_task_coverage = _safe_float(
        summary_metrics.get("code_intelligence_task_coverage")
    )
    metrics.code_intelligence_pass_rate = _safe_float(
        summary_metrics.get("code_intelligence_pass_rate")
    )
    metrics.non_code_intelligence_pass_rate = _safe_float(
        summary_metrics.get("non_code_intelligence_pass_rate")
    )
    _populate_workspace_policy_metrics(
        metrics,
        summary_metrics,
        task_results=[],
        total_tasks=result.total_tasks,
    )

    # Estimate cost (rough approximation based on tokens)
    # Assuming ~$0.01 per 1K tokens average
    metrics.cost_per_task = (metrics.tokens_per_task / 1000) * 0.01

    # Robustness
    metrics.error_rate = result.error_tasks / result.total_tasks
    metrics.timeout_rate = result.timeout_tasks / result.total_tasks

    return metrics


def _safe_int(value: Any) -> int:
    """Best-effort integer conversion for saved result ingestion."""
    try:
        if value is None:
            return 0
        return int(value)
    except (TypeError, ValueError):
        return 0


def _safe_float(value: Any) -> float:
    """Best-effort float conversion for saved result ingestion."""
    try:
        if value is None:
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _get_saved_task_results(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract task result records from supported saved result schemas."""
    tasks = data.get("task_results")
    if isinstance(tasks, list):
        return [task for task in tasks if isinstance(task, dict)]
    tasks = data.get("tasks")
    if isinstance(tasks, list):
        return [task for task in tasks if isinstance(task, dict)]
    return []


def _get_saved_dataset_metadata(data: dict[str, Any]) -> dict[str, Any]:
    """Extract dataset metadata from saved result payloads."""
    dataset_metadata = data.get("dataset_metadata")
    if isinstance(dataset_metadata, dict):
        return dict(dataset_metadata)
    config = data.get("config")
    if isinstance(config, dict) and isinstance(config.get("dataset_metadata"), dict):
        return dict(config["dataset_metadata"])
    return {}


def _get_task_team_feedback_summary(task: dict[str, Any]) -> dict[str, Any]:
    """Extract a per-task team-feedback summary from supported saved task shapes."""
    direct_summary = task.get("team_feedback_summary")
    if isinstance(direct_summary, dict):
        return direct_summary
    metadata = task.get("metadata")
    if isinstance(metadata, dict) and isinstance(metadata.get("team_feedback_summary"), dict):
        return metadata["team_feedback_summary"]
    return {}


def _task_has_workspace_policy(task: dict[str, Any]) -> bool:
    summary = _get_task_team_feedback_summary(task)
    return bool(
        summary.get("has_workspace_isolation_policy")
        or summary.get("workspace_policy_mode")
        or summary.get("workspace_policy_materialize_worktrees")
        or summary.get("workspace_policy_dry_run_worktrees")
        or summary.get("workspace_policy_auto_merge_worktrees")
    )


def _task_has_workspace_diagnostics(task: dict[str, Any]) -> bool:
    summary = _get_task_team_feedback_summary(task)
    return bool(
        summary.get("has_workspace_isolation_diagnostics")
        or _safe_int(summary.get("workspace_diagnostic_count")) > 0
    )


def _task_passed(task: dict[str, Any]) -> bool:
    return str(task.get("status", "")).lower() == "passed"


def _rate(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else 0.0


def _summary_has_workspace_policy_metrics(summary: dict[str, Any]) -> bool:
    return any(
        key in summary
        for key in (
            "team_workspace_policy_task_count",
            "workspace_policy_task_coverage",
            "workspace_policy_pass_rate",
            "team_workspace_diagnostic_task_count",
            "workspace_diagnostic_task_coverage",
        )
    )


def _populate_workspace_policy_metrics(
    metrics: ComparisonMetrics,
    summary: dict[str, Any],
    *,
    task_results: list[dict[str, Any]],
    total_tasks: int,
) -> None:
    """Populate workspace-policy comparison metrics from summaries or saved task records."""
    if total_tasks <= 0:
        return

    aggregate_summary = (
        dict(summary)
        if _summary_has_workspace_policy_metrics(summary)
        else aggregate_team_feedback(task_results, total_tasks=total_tasks)
    )

    policy_task_count = _safe_int(aggregate_summary.get("team_workspace_policy_task_count"))
    diagnostic_task_count = _safe_int(aggregate_summary.get("team_workspace_diagnostic_task_count"))
    diagnostic_count = _safe_int(aggregate_summary.get("team_workspace_diagnostic_count"))

    metrics.workspace_policy_task_coverage = _safe_float(
        aggregate_summary.get("workspace_policy_task_coverage")
    ) or _rate(policy_task_count, total_tasks)
    metrics.workspace_policy_materialize_rate = _safe_float(
        aggregate_summary.get("workspace_policy_materialize_rate")
    ) or _rate(
        _safe_int(aggregate_summary.get("team_workspace_policy_materialize_count")),
        total_tasks,
    )
    metrics.workspace_policy_dry_run_rate = _safe_float(
        aggregate_summary.get("workspace_policy_dry_run_rate")
    ) or _rate(
        _safe_int(aggregate_summary.get("team_workspace_policy_dry_run_count")),
        total_tasks,
    )
    metrics.workspace_policy_auto_merge_rate = _safe_float(
        aggregate_summary.get("workspace_policy_auto_merge_rate")
    ) or _rate(
        _safe_int(aggregate_summary.get("team_workspace_policy_auto_merge_count")),
        total_tasks,
    )
    metrics.workspace_policy_cleanup_disabled_rate = _safe_float(
        aggregate_summary.get("workspace_policy_cleanup_disabled_rate")
    ) or _rate(
        _safe_int(aggregate_summary.get("team_workspace_policy_cleanup_disabled_count")),
        total_tasks,
    )
    metrics.workspace_diagnostic_task_coverage = _safe_float(
        aggregate_summary.get("workspace_diagnostic_task_coverage")
    ) or _rate(diagnostic_task_count, total_tasks)
    metrics.workspace_diagnostic_rate = _safe_float(
        aggregate_summary.get("workspace_diagnostic_rate")
    ) or _rate(diagnostic_count, total_tasks)

    metrics.workspace_policy_pass_rate = _safe_float(
        aggregate_summary.get("workspace_policy_pass_rate")
    )
    metrics.non_workspace_policy_pass_rate = _safe_float(
        aggregate_summary.get("non_workspace_policy_pass_rate")
    )
    if task_results:
        policy_tasks = [task for task in task_results if _task_has_workspace_policy(task)]
        non_policy_tasks = [task for task in task_results if not _task_has_workspace_policy(task)]
        diagnostic_tasks = [task for task in task_results if _task_has_workspace_diagnostics(task)]
        if policy_tasks:
            metrics.workspace_policy_pass_rate = _rate(
                sum(1 for task in policy_tasks if _task_passed(task)),
                len(policy_tasks),
            )
        if non_policy_tasks:
            metrics.non_workspace_policy_pass_rate = _rate(
                sum(1 for task in non_policy_tasks if _task_passed(task)),
                len(non_policy_tasks),
            )
        if not diagnostic_task_count and diagnostic_tasks:
            metrics.workspace_diagnostic_task_coverage = _rate(len(diagnostic_tasks), total_tasks)

    metrics.workspace_policy_pass_delta = (
        _safe_float(aggregate_summary.get("workspace_policy_pass_delta"))
        if aggregate_summary.get("workspace_policy_pass_delta") is not None
        else metrics.workspace_policy_pass_rate - metrics.non_workspace_policy_pass_rate
    )


def _get_saved_prompt_binding(
    data: dict[str, Any],
) -> tuple[Optional[str], Optional[str]]:
    """Extract prompt/runtime variant metadata from saved result payloads."""
    config = data.get("config") if isinstance(data.get("config"), dict) else {}
    prompt_candidate_hash = data.get("prompt_candidate_hash") or config.get("prompt_candidate_hash")
    prompt_section_name = (
        data.get("prompt_section_name")
        or data.get("section_name")
        or config.get("prompt_section_name")
        or config.get("section_name")
    )
    candidate = str(prompt_candidate_hash).strip() if prompt_candidate_hash else None
    section = str(prompt_section_name).strip() if prompt_section_name else None
    return candidate or None, section or None


def _resolve_saved_benchmark_type(data: dict[str, Any]) -> BenchmarkType:
    """Resolve a benchmark enum from a saved result payload."""
    benchmark_name = data.get("benchmark")
    if benchmark_name is None:
        config = data.get("config")
        if isinstance(config, dict):
            benchmark_name = config.get("benchmark")
    if benchmark_name is None:
        raise ValueError("Saved benchmark result is missing benchmark metadata")
    metadata = get_benchmark_metadata(str(benchmark_name))
    if metadata is None:
        raise ValueError(f"Unknown benchmark in saved result: {benchmark_name}")
    return metadata.type


def compute_metrics_from_saved_result(data: dict[str, Any]) -> ComparisonMetrics:
    """Compute comparison metrics from a saved benchmark JSON artifact."""
    metrics = ComparisonMetrics()
    summary: dict[str, Any] = {}
    for key in ("summary", "metrics"):
        candidate = data.get(key)
        if isinstance(candidate, dict):
            summary.update(candidate)

    task_results = _get_saved_task_results(data)
    total_tasks = _safe_int(summary.get("total_tasks")) or len(task_results)
    if total_tasks <= 0:
        return metrics

    passed_tasks = sum(
        1 for task in task_results if str(task.get("status", "")).lower() == "passed"
    )
    error_tasks = sum(1 for task in task_results if str(task.get("status", "")).lower() == "error")
    timeout_tasks = sum(
        1 for task in task_results if str(task.get("status", "")).lower() == "timeout"
    )

    if summary.get("pass_rate") is not None:
        metrics.pass_rate = _safe_float(summary.get("pass_rate"))
    else:
        metrics.pass_rate = passed_tasks / total_tasks

    duration_seconds = _safe_float(summary.get("duration_seconds"))
    if duration_seconds > 0:
        metrics.avg_latency_ms = (duration_seconds * 1000) / total_tasks
    elif summary.get("avg_duration_per_task") is not None:
        metrics.avg_latency_ms = _safe_float(summary.get("avg_duration_per_task")) * 1000
    else:
        task_duration = sum(
            _safe_float(task.get("duration_seconds") or task.get("duration"))
            for task in task_results
        )
        metrics.avg_latency_ms = (task_duration * 1000) / total_tasks if task_duration else 0.0

    total_tokens = _safe_float(summary.get("total_tokens"))
    if total_tokens > 0:
        metrics.tokens_per_task = total_tokens / total_tasks
    elif summary.get("avg_tokens_per_task") is not None:
        metrics.tokens_per_task = _safe_float(summary.get("avg_tokens_per_task"))
    else:
        metrics.tokens_per_task = (
            sum(_safe_float(task.get("tokens_used")) for task in task_results) / total_tasks
        )

    total_tests = sum(_safe_int(task.get("tests_total")) for task in task_results)
    passed_tests = sum(_safe_int(task.get("tests_passed")) for task in task_results)
    if total_tests > 0:
        metrics.test_pass_rate = passed_tests / total_tests

    completion_scores = [
        _safe_float(task.get("completion_score"))
        for task in task_results
        if task.get("completion_score") is not None
    ]
    if completion_scores:
        metrics.partial_completion = sum(completion_scores) / len(completion_scores)

    quality_scores = []
    for task in task_results:
        code_quality = task.get("code_quality")
        if isinstance(code_quality, dict) and code_quality.get("overall_score") is not None:
            quality_scores.append(_safe_float(code_quality.get("overall_score")))
    if quality_scores:
        metrics.code_quality_score = sum(quality_scores) / len(quality_scores)

    total_turns = _safe_float(summary.get("total_turns"))
    if total_turns > 0:
        metrics.turns_per_task = total_turns / total_tasks
    else:
        metrics.turns_per_task = (
            sum(_safe_float(task.get("turns")) for task in task_results) / total_tasks
        )

    total_tool_calls = _safe_float(summary.get("total_tool_calls"))
    if total_tool_calls > 0:
        metrics.tool_calls_per_task = total_tool_calls / total_tasks
    else:
        metrics.tool_calls_per_task = (
            sum(_safe_float(task.get("tool_calls")) for task in task_results) / total_tasks
        )

    total_cost_micros = _safe_float(summary.get("cost_usd_micros"))
    if total_cost_micros > 0:
        metrics.cost_per_task = (total_cost_micros / 1_000_000) / total_tasks
    elif metrics.tokens_per_task > 0:
        metrics.cost_per_task = (metrics.tokens_per_task / 1000) * 0.01
    metrics.accepted_patch_rate = _safe_float(summary.get("accepted_patch_rate"))
    metrics.tokens_to_merge = _safe_float(summary.get("avg_tokens_to_merge"))
    metrics.cost_per_accepted_patch_usd = _safe_float(summary.get("cost_per_accepted_patch_usd"))
    metrics.avg_time_to_first_edit_seconds = _safe_float(
        summary.get("avg_time_to_first_edit_seconds")
    )
    metrics.avg_time_to_first_tool_call_seconds = _safe_float(
        summary.get("avg_time_to_first_tool_call_seconds")
    )
    metrics.code_intelligence_task_coverage = _safe_float(
        summary.get("code_intelligence_task_coverage")
    )
    metrics.code_intelligence_pass_rate = _safe_float(summary.get("code_intelligence_pass_rate"))
    metrics.non_code_intelligence_pass_rate = _safe_float(
        summary.get("non_code_intelligence_pass_rate")
    )
    _populate_workspace_policy_metrics(
        metrics,
        summary,
        task_results=task_results,
        total_tasks=total_tasks,
    )

    summary_errors = summary.get("errors")
    summary_timeouts = summary.get("timeouts")
    metrics.error_rate = (
        _safe_float(summary_errors) / total_tasks
        if summary_errors is not None
        else error_tasks / total_tasks
    )
    metrics.timeout_rate = (
        _safe_float(summary_timeouts) / total_tasks
        if summary_timeouts is not None
        else timeout_tasks / total_tasks
    )

    return metrics


def load_framework_result_from_file(
    path: Path,
    framework: Framework = Framework.VICTOR,
    model_override: Optional[str] = None,
) -> FrameworkResult:
    """Load a saved benchmark artifact into a FrameworkResult."""
    with open(path) as f:
        data = json.load(f)

    benchmark = _resolve_saved_benchmark_type(data)
    config = data.get("config") if isinstance(data.get("config"), dict) else {}
    dataset_metadata = _get_saved_dataset_metadata(data)
    resolved_model = model_override or data.get("model") or config.get("model") or "unknown"
    source_name = (
        dataset_metadata.get("source_name") or config.get("source") or f"Local result ({path.name})"
    )
    timestamp_value = data.get("timestamp") or data.get("end_time") or data.get("start_time")
    try:
        timestamp = (
            datetime.fromisoformat(str(timestamp_value)) if timestamp_value else datetime.now()
        )
    except ValueError:
        timestamp = datetime.now()

    framework_config: dict[str, Any] = {
        "artifact_path": str(path),
        "source": source_name,
        "dataset_metadata": dataset_metadata,
    }
    prompt_candidate_hash, prompt_section_name = _get_saved_prompt_binding(data)
    if config.get("max_tasks") is not None:
        framework_config["max_tasks"] = config.get("max_tasks")
    if prompt_candidate_hash is not None:
        framework_config["prompt_candidate_hash"] = prompt_candidate_hash
    if prompt_section_name is not None:
        framework_config["prompt_section_name"] = prompt_section_name

    return FrameworkResult(
        framework=framework,
        benchmark=benchmark,
        model=str(resolved_model),
        metrics=compute_metrics_from_saved_result(data),
        timestamp=timestamp,
        config=framework_config,
        task_results=_get_saved_task_results(data),
    )


# External benchmark results for comparison (from published data)
PUBLISHED_RESULTS: dict[BenchmarkType, dict[Framework, dict[str, Any]]] = {
    BenchmarkType.SWE_BENCH: {
        # SWE-bench Lite results as of late 2024
        Framework.CLAUDE_CODE: {
            "model": "claude-3-opus",
            "pass_rate": 0.49,  # ~49% on SWE-bench Lite
            "source": "Anthropic blog post, 2024",
        },
        Framework.AIDER: {
            "model": "claude-3-opus + gpt-4",
            "pass_rate": 0.268,  # 26.8% on SWE-bench Lite
            "source": "Aider leaderboard, 2024",
        },
    },
    BenchmarkType.HUMAN_EVAL: {
        # HumanEval pass@1 results
        Framework.CLAUDE_CODE: {
            "model": "claude-3-opus",
            "pass_rate": 0.846,
            "source": "Anthropic technical report",
        },
    },
}


def get_published_result(
    benchmark: BenchmarkType,
    framework: Framework,
) -> Optional[dict[str, Any]]:
    """Get published benchmark result for a framework."""
    if benchmark in PUBLISHED_RESULTS:
        if framework in PUBLISHED_RESULTS[benchmark]:
            return PUBLISHED_RESULTS[benchmark][framework]
    return None


def create_comparison_report(
    benchmark: BenchmarkType,
    victor_result: EvaluationResult,
    include_published: bool = True,
) -> ComparisonReport:
    """Create a comparison report with Victor results and published data."""
    report = ComparisonReport(benchmark=benchmark)

    # Add Victor result
    victor_metrics = compute_metrics_from_result(victor_result)
    victor_framework_result = FrameworkResult(
        framework=Framework.VICTOR,
        benchmark=benchmark,
        model=victor_result.config.model,
        metrics=victor_metrics,
        config={"max_tasks": victor_result.config.max_tasks},
    )
    report.results.append(victor_framework_result)

    # Add published results for comparison
    if include_published and benchmark in PUBLISHED_RESULTS:
        for framework, data in PUBLISHED_RESULTS[benchmark].items():
            metrics = ComparisonMetrics(pass_rate=data.get("pass_rate", 0.0))
            result = FrameworkResult(
                framework=framework,
                benchmark=benchmark,
                model=data.get("model", "unknown"),
                metrics=metrics,
                config={"source": data.get("source", "published")},
            )
            report.results.append(result)

    return report


def create_comparison_report_from_saved_result(
    path: Path,
    framework: Framework = Framework.VICTOR,
    include_published: bool = True,
) -> ComparisonReport:
    """Create a comparison report from a saved benchmark JSON artifact."""
    return create_comparison_report_from_saved_results(
        [path],
        framework=framework,
        include_published=include_published,
    )


def create_comparison_report_from_fixture_manifest(
    path: Path,
    framework: Framework = Framework.VICTOR,
    include_published: bool = True,
) -> ComparisonReport:
    """Create a comparison report from a saved fixture manifest bundle."""
    return create_comparison_report_from_saved_results(
        [path],
        framework=framework,
        include_published=include_published,
    )


def create_comparison_report_from_saved_results(
    paths: Sequence[Path],
    framework: Framework = Framework.VICTOR,
    include_published: bool = True,
) -> ComparisonReport:
    """Create a comparison report from one or more saved benchmark artifacts."""
    normalized_paths = _expand_saved_result_paths([Path(path) for path in paths])
    if not normalized_paths:
        raise ValueError("At least one saved benchmark result is required")

    framework_results = [
        load_framework_result_from_file(path, framework=framework) for path in normalized_paths
    ]
    benchmark = framework_results[0].benchmark
    for result in framework_results[1:]:
        if result.benchmark != benchmark:
            raise ValueError("All saved benchmark results must target the same benchmark type")

    report = ComparisonReport(benchmark=benchmark)
    report.results.extend(framework_results)

    if include_published and benchmark in PUBLISHED_RESULTS:
        for published_framework, data in PUBLISHED_RESULTS[benchmark].items():
            metrics = ComparisonMetrics(pass_rate=data.get("pass_rate", 0.0))
            result = FrameworkResult(
                framework=published_framework,
                benchmark=benchmark,
                model=data.get("model", "unknown"),
                metrics=metrics,
                config={"source": data.get("source", "published")},
            )
            report.results.append(result)

    return report


def build_comparison_report_summary(report: ComparisonReport) -> dict[str, Any]:
    """Build the machine-readable summary payload for a comparison report."""
    return {
        "benchmark": report.benchmark.value,
        "timestamp": report.timestamp.isoformat(),
        "winner": (report.get_winner().value if report.get_winner() is not None else None),
        "framework_count": len(report.results),
        "results": [
            {
                "framework": result.framework.value,
                "model": result.model,
                "pass_rate": result.metrics.pass_rate,
                "accepted_patch_rate": result.metrics.accepted_patch_rate,
                "tokens_to_merge": result.metrics.tokens_to_merge,
                "cost_per_accepted_patch_usd": result.metrics.cost_per_accepted_patch_usd,
                "avg_time_to_first_edit_seconds": (result.metrics.avg_time_to_first_edit_seconds),
                "avg_time_to_first_tool_call_seconds": (
                    result.metrics.avg_time_to_first_tool_call_seconds
                ),
                "code_intelligence_task_coverage": (result.metrics.code_intelligence_task_coverage),
                "code_intelligence_pass_rate": result.metrics.code_intelligence_pass_rate,
                "non_code_intelligence_pass_rate": (result.metrics.non_code_intelligence_pass_rate),
                "workspace_policy_task_coverage": (result.metrics.workspace_policy_task_coverage),
                "workspace_policy_pass_rate": result.metrics.workspace_policy_pass_rate,
                "non_workspace_policy_pass_rate": (result.metrics.non_workspace_policy_pass_rate),
                "workspace_policy_pass_delta": result.metrics.workspace_policy_pass_delta,
                "workspace_policy_materialize_rate": (
                    result.metrics.workspace_policy_materialize_rate
                ),
                "workspace_policy_dry_run_rate": result.metrics.workspace_policy_dry_run_rate,
                "workspace_policy_auto_merge_rate": (
                    result.metrics.workspace_policy_auto_merge_rate
                ),
                "workspace_policy_cleanup_disabled_rate": (
                    result.metrics.workspace_policy_cleanup_disabled_rate
                ),
                "workspace_diagnostic_task_coverage": (
                    result.metrics.workspace_diagnostic_task_coverage
                ),
                "workspace_diagnostic_rate": result.metrics.workspace_diagnostic_rate,
                "source": str((result.config or {}).get("source", "")),
                "artifact_path": str((result.config or {}).get("artifact_path", "")),
                "prompt_candidate_hash": str(
                    (result.config or {}).get("prompt_candidate_hash", "")
                ),
                "section_name": str((result.config or {}).get("prompt_section_name", "")),
            }
            for result in sorted(
                report.results,
                key=lambda item: item.metrics.pass_rate,
                reverse=True,
            )
        ],
    }


def build_publication_stable_run_summary(
    report: ComparisonReport,
    *,
    manifest_path: Optional[str] = None,
) -> dict[str, Any]:
    """Build public stable-run KPIs for a benchmark publication bundle."""
    comparison_summary = build_comparison_report_summary(report)
    results = list(comparison_summary.get("results") or [])
    best_result = results[0] if results else {}
    metadata = get_benchmark_metadata(report.benchmark)
    categories = set(metadata.categories if metadata is not None else [])

    issue_fix_success_rate = (
        _safe_float(best_result.get("pass_rate"))
        if _is_issue_fix_publication(report.benchmark, categories)
        else None
    )
    review_bug_catch_rate = (
        _safe_float(best_result.get("pass_rate"))
        if _is_review_bug_catch_publication(categories)
        else None
    )
    tokens_to_merge = _safe_float(best_result.get("tokens_to_merge"))
    time_to_first_edit = _safe_float(best_result.get("avg_time_to_first_edit_seconds"))
    cost_per_accepted_patch = _safe_float(best_result.get("cost_per_accepted_patch_usd"))

    required_public_kpis = {
        "issue_fix_success_rate": issue_fix_success_rate,
        "review_bug_catch_rate": review_bug_catch_rate,
        "tokens_to_merge": tokens_to_merge,
        "time_to_first_edit_seconds": time_to_first_edit,
        "cost_per_accepted_patch_usd": cost_per_accepted_patch,
    }
    kpi_availability = {
        key: _public_kpi_available(value) for key, value in required_public_kpis.items()
    }
    applicable_public_kpis = [
        key for key, value in required_public_kpis.items() if value is not None
    ]
    missing_public_kpis = [key for key in applicable_public_kpis if not kpi_availability[key]]
    corpus_readiness = build_publication_corpus_readiness(
        report,
        required_public_kpi_complete=not missing_public_kpis,
    )

    return {
        "benchmark": report.benchmark.value,
        "timestamp": report.timestamp.isoformat(),
        "manifest_path": manifest_path,
        "stable_run_artifact_count": len(results),
        "model_count": len(
            {
                str(result.get("model", "")).strip()
                for result in results
                if str(result.get("model", "")).strip()
            }
        ),
        "best_result": {
            "framework": str(best_result.get("framework", "")),
            "model": str(best_result.get("model", "")),
            "pass_rate": _safe_float(best_result.get("pass_rate")),
            "source": str(best_result.get("source", "")),
            "artifact_path": str(best_result.get("artifact_path", "")),
        },
        "required_public_kpis": required_public_kpis,
        "kpi_availability": kpi_availability,
        "applicable_public_kpis": applicable_public_kpis,
        "missing_public_kpis": missing_public_kpis,
        "required_public_kpi_complete": not missing_public_kpis,
        "corpus_readiness": corpus_readiness,
        "results": results,
    }


def _is_issue_fix_publication(benchmark: BenchmarkType, categories: set[str]) -> bool:
    return benchmark == BenchmarkType.SWE_BENCH or "software-engineering" in categories


def _is_review_bug_catch_publication(categories: set[str]) -> bool:
    return bool({"review", "bug-catch", "bug-catch-rate"} & categories)


def _public_kpi_available(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return float(value) > 0.0
    return bool(value)


def build_publication_corpus_readiness(
    report: ComparisonReport,
    *,
    required_public_kpi_complete: bool,
) -> dict[str, Any]:
    """Summarize whether a stable-run corpus is ready for public benchmark claims."""
    artifact_count = len(report.results)
    task_count = sum(len(result.task_results or []) for result in report.results)
    models = {
        result.model.strip()
        for result in report.results
        if isinstance(result.model, str) and result.model.strip()
    }
    sources = {
        str((result.config or {}).get("source", "")).strip()
        for result in report.results
        if str((result.config or {}).get("source", "")).strip()
    }

    missing_reasons: list[str] = []
    if artifact_count <= 0:
        missing_reasons.append("no_stable_run_artifacts")
    if task_count <= 0:
        missing_reasons.append("no_task_results")
    if not required_public_kpi_complete:
        missing_reasons.append("missing_public_kpis")

    return {
        "publishable": not missing_reasons,
        "artifact_count": artifact_count,
        "task_count": task_count,
        "source_count": len(sources),
        "model_count": len(models),
        "required_public_kpi_complete": required_public_kpi_complete,
        "missing_reasons": missing_reasons,
    }


def _stable_run_result_looks_like_fixture(result: FrameworkResult) -> bool:
    """Return True when a saved result appears to come from checked-in fixtures."""
    config = result.config or {}
    dataset_metadata = config.get("dataset_metadata")
    candidate_values: list[str] = [
        result.model,
        str(config.get("artifact_path", "")),
        str(config.get("source", "")),
    ]
    if isinstance(dataset_metadata, Mapping):
        for key in ("source_name", "name", "dataset", "fixture_set_name"):
            value = dataset_metadata.get(key)
            if value is not None:
                candidate_values.append(str(value))

    normalized_values = [value.lower().replace("\\", "/") for value in candidate_values]
    return any("fixture" in value or "tests/fixtures/" in value for value in normalized_values)


def _reject_fixture_stable_run_inputs(report: ComparisonReport) -> None:
    """Prevent stable real-run bundles from silently publishing fixture corpora."""
    fixture_artifacts = [
        str((result.config or {}).get("artifact_path", result.model))
        for result in report.results
        if result.framework == Framework.VICTOR and _stable_run_result_looks_like_fixture(result)
    ]
    if not fixture_artifacts:
        return
    preview = ", ".join(fixture_artifacts[:3])
    if len(fixture_artifacts) > 3:
        preview += f", +{len(fixture_artifacts) - 3} more"
    raise ValueError(
        "Stable real-run publication requires non-fixture saved artifacts. "
        "Use fixture benchmark publication for checked-in fixture corpora. "
        f"Offending artifact(s): {preview}"
    )


def build_comparison_report_fixture_manifest(
    report: ComparisonReport,
) -> dict[str, Any]:
    """Build a manifest describing the local result artifacts included in a comparison."""
    artifacts: list[dict[str, Any]] = []
    for result in report.results:
        config = result.config or {}
        artifact_path = str(config.get("artifact_path", "")).strip()
        if not artifact_path:
            continue
        artifact = {
            "framework": result.framework.value,
            "model": result.model,
            "artifact_path": artifact_path,
            "source": str(config.get("source", "")),
            "prompt_candidate_hash": str(config.get("prompt_candidate_hash", "")),
            "section_name": str(config.get("prompt_section_name", "")),
            "dataset_metadata": dict(config.get("dataset_metadata") or {}),
        }
        source_file = Path(artifact_path)
        if source_file.is_file():
            artifact["artifact_size_bytes"] = source_file.stat().st_size
            artifact["artifact_sha256"] = _compute_file_sha256(source_file)
        artifacts.append(artifact)
    return {
        "benchmark": report.benchmark.value,
        "timestamp": report.timestamp.isoformat(),
        "checksum_algorithm": "sha256",
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }


def _slugify_bundle_component(value: str) -> str:
    """Normalize a bundle path component for stable fixture filenames."""
    normalized = "".join(ch.lower() if ch.isalnum() else "-" for ch in value.strip())
    collapsed = "-".join(part for part in normalized.split("-") if part)
    return collapsed or "artifact"


def _compute_file_sha256(path: Path) -> str:
    """Return the SHA-256 digest for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_file(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def _load_fixture_manifest(path: Path) -> Optional[dict[str, Any]]:
    try:
        data = _read_json_file(path)
    except Exception:
        return None
    if not isinstance(data.get("artifacts"), list):
        return None
    if "artifact_count" not in data or "benchmark" not in data:
        return None
    return data


def _load_fixture_benchmark_publication_catalog(path: Path) -> Optional[dict[str, Any]]:
    try:
        data = _read_json_file(path)
    except Exception:
        return None
    if not isinstance(data.get("benchmarks"), list):
        return None
    if "benchmark_count" not in data:
        return None
    return data


def _resolve_fixture_benchmark_publication_catalog_path(path: Path) -> Optional[Path]:
    if path.is_file():
        return path if _load_fixture_benchmark_publication_catalog(path) is not None else None
    if not path.is_dir():
        return None
    for catalog_name in (
        "fixture_benchmark_publication_catalog.json",
        "stable_run_publication_catalog.json",
    ):
        publication_catalog = path / catalog_name
        if not publication_catalog.is_file():
            continue
        if _load_fixture_benchmark_publication_catalog(publication_catalog) is None:
            continue
        return publication_catalog
    return None


def _resolve_fixture_manifest_path(path: Path) -> Optional[Path]:
    if path.is_file():
        return path if _load_fixture_manifest(path) is not None else None
    if not path.is_dir():
        return None
    candidate_paths = sorted(path.glob("*_fixtures.json"))
    if len(candidate_paths) == 1:
        return candidate_paths[0]
    comparison_report_manifest = path / "comparison_report_fixtures.json"
    if comparison_report_manifest.is_file():
        return comparison_report_manifest
    if candidate_paths:
        raise ValueError(
            f"Fixture set directory {path} is ambiguous; expected one *_fixtures.json manifest"
        )
    return None


def discover_fixture_sets(root: Path) -> list[FixtureSetDescriptor]:
    """Discover saved benchmark fixture sets under a root directory."""
    root_path = Path(root)
    if not root_path.exists():
        return []

    candidate_manifests: list[tuple[str, Path]] = []
    resolved_root_manifest = _resolve_fixture_manifest_path(root_path)
    if resolved_root_manifest is not None:
        candidate_manifests.append((root_path.name, resolved_root_manifest))
    elif root_path.is_dir():
        for child in sorted(root_path.iterdir()):
            manifest_path = _resolve_fixture_manifest_path(child)
            if manifest_path is None:
                continue
            candidate_manifests.append((child.name, manifest_path))

    descriptors: list[FixtureSetDescriptor] = []
    for default_name, manifest_path in candidate_manifests:
        manifest = _load_fixture_manifest(manifest_path)
        if manifest is None:
            continue
        artifacts = list(manifest.get("artifacts", []))
        models = tuple(
            str(artifact.get("model", "")).strip()
            for artifact in artifacts
            if isinstance(artifact, dict) and str(artifact.get("model", "")).strip()
        )
        sources = tuple(
            str(artifact.get("source", "")).strip()
            for artifact in artifacts
            if isinstance(artifact, dict) and str(artifact.get("source", "")).strip()
        )
        descriptors.append(
            FixtureSetDescriptor(
                name=default_name,
                benchmark=str(manifest.get("benchmark", "")).strip(),
                manifest_path=manifest_path,
                artifact_count=int(manifest.get("artifact_count", len(artifacts)) or 0),
                models=tuple(dict.fromkeys(models)),
                sources=tuple(dict.fromkeys(sources)),
            )
        )

    return sorted(descriptors, key=lambda item: (item.benchmark, item.name))


def discover_fixture_benchmarks(
    root: Path = DEFAULT_FIXTURE_SET_ROOT,
) -> list[FixtureBenchmarkDescriptor]:
    """Discover benchmark-level checked-in fixture corpora under a root directory."""
    grouped: dict[str, list[FixtureSetDescriptor]] = {}
    for descriptor in discover_fixture_sets(root):
        grouped.setdefault(descriptor.benchmark, []).append(descriptor)

    benchmark_descriptors: list[FixtureBenchmarkDescriptor] = []
    for benchmark, descriptors in sorted(grouped.items()):
        models: list[str] = []
        fixture_set_names: list[str] = []
        artifact_count = 0
        for descriptor in descriptors:
            artifact_count += descriptor.artifact_count
            fixture_set_names.append(descriptor.name)
            models.extend(descriptor.models)
        benchmark_descriptors.append(
            FixtureBenchmarkDescriptor(
                benchmark=benchmark,
                fixture_set_count=len(descriptors),
                artifact_count=artifact_count,
                models=tuple(dict.fromkeys(models)),
                fixture_set_names=tuple(fixture_set_names),
            )
        )

    return benchmark_descriptors


def build_fixture_benchmark_catalog(
    *,
    root: Path = DEFAULT_FIXTURE_SET_ROOT,
    benchmark: Optional[str] = None,
    verify: bool = False,
) -> dict[str, Any]:
    """Build a machine-readable catalog for checked-in benchmark fixture corpora."""
    catalog_benchmark_metadata = _select_catalog_benchmark_metadata(benchmark)
    fixture_sets = discover_fixture_sets(root)
    grouped_fixture_sets: dict[str, list[FixtureSetDescriptor]] = {}
    for descriptor in fixture_sets:
        grouped_fixture_sets.setdefault(descriptor.benchmark, []).append(descriptor)
    descriptors = discover_fixture_benchmarks(root)
    if benchmark is not None:
        descriptors = [
            descriptor
            for descriptor in descriptors
            if fixture_benchmark_matches(descriptor.benchmark, benchmark)
        ]
    if not descriptors:
        raise ValueError(f"No fixture benchmarks found under {Path(root)}")

    verification_summary_by_benchmark: dict[str, tuple[int, int]] = {}
    if verify:
        for result in verify_fixture_sets(root=root, benchmark=benchmark):
            verified_sets, verified_artifacts = verification_summary_by_benchmark.get(
                result.benchmark,
                (0, 0),
            )
            verification_summary_by_benchmark[result.benchmark] = (
                verified_sets + 1,
                verified_artifacts + result.verified_artifact_count,
            )

    benchmarks: list[dict[str, Any]] = []
    for descriptor in descriptors:
        fixture_descriptors = list(grouped_fixture_sets.get(descriptor.benchmark, ()))
        fixture_sources: list[str] = []
        fixture_manifest_paths: list[str] = []
        for fixture_descriptor in fixture_descriptors:
            fixture_sources.extend(fixture_descriptor.sources)
            fixture_manifest_paths.append(str(fixture_descriptor.manifest_path))
        payload = {
            "benchmark": descriptor.benchmark,
            "fixture_set_count": descriptor.fixture_set_count,
            "artifact_count": descriptor.artifact_count,
            "models": list(descriptor.models),
            "fixture_set_names": list(descriptor.fixture_set_names),
            "fixture_sources": list(dict.fromkeys(fixture_sources)),
            "fixture_manifest_paths": fixture_manifest_paths,
        }
        payload.update(_build_fixture_benchmark_metadata_payload(descriptor.benchmark))
        if verify:
            verified_sets, verified_artifacts = verification_summary_by_benchmark.get(
                descriptor.benchmark,
                (0, 0),
            )
            payload["verified_fixture_set_count"] = verified_sets
            payload["verified_artifact_count"] = verified_artifacts
        benchmarks.append(payload)

    covered_catalog_benchmarks = [
        metadata
        for metadata in catalog_benchmark_metadata
        if any(
            fixture_benchmark_matches(descriptor.benchmark, metadata.name)
            for descriptor in descriptors
        )
    ]
    missing_catalog_benchmarks = [
        metadata.to_dict()
        for metadata in catalog_benchmark_metadata
        if not any(
            fixture_benchmark_matches(descriptor.benchmark, metadata.name)
            for descriptor in descriptors
        )
    ]
    catalog_benchmark_count = len(catalog_benchmark_metadata)
    covered_catalog_benchmark_count = len(covered_catalog_benchmarks)
    coverage_rate = round(
        covered_catalog_benchmark_count / max(1, catalog_benchmark_count),
        4,
    )
    has_full_catalog_coverage = covered_catalog_benchmark_count == catalog_benchmark_count

    return {
        "root": str(Path(root)),
        "verified": verify,
        "benchmark_count": len(descriptors),
        "fixture_set_count": sum(descriptor.fixture_set_count for descriptor in descriptors),
        "artifact_count": sum(descriptor.artifact_count for descriptor in descriptors),
        "verified_benchmark_count": (len(verification_summary_by_benchmark) if verify else 0),
        "catalog_benchmark_count": catalog_benchmark_count,
        "covered_catalog_benchmark_count": covered_catalog_benchmark_count,
        "catalog_benchmark_coverage_rate": coverage_rate,
        "has_full_catalog_coverage": has_full_catalog_coverage,
        "missing_catalog_benchmarks": missing_catalog_benchmarks,
        "benchmarks": benchmarks,
    }


def save_fixture_benchmark_catalog(
    *,
    output_path: Path,
    root: Path = DEFAULT_FIXTURE_SET_ROOT,
    benchmark: Optional[str] = None,
    verify: bool = False,
) -> Path:
    """Save a checked-in benchmark fixture catalog as machine-readable JSON."""
    catalog = build_fixture_benchmark_catalog(
        root=root,
        benchmark=benchmark,
        verify=verify,
    )
    normalized_output = Path(output_path)
    normalized_output.parent.mkdir(parents=True, exist_ok=True)
    normalized_output.write_text(json.dumps(catalog, indent=2) + "\n")
    return normalized_output


def save_fixture_benchmark_publication_bundle(
    *,
    output_path: Path,
    root: Path = DEFAULT_FIXTURE_SET_ROOT,
    benchmark: Optional[str] = None,
    verify: bool = False,
) -> dict[str, Any]:
    """Save portable benchmark-corpus publication bundles with direct-load manifests."""
    normalized_output = Path(output_path)
    normalized_output.mkdir(parents=True, exist_ok=True)

    catalog = build_fixture_benchmark_catalog(
        root=root,
        benchmark=benchmark,
        verify=verify,
    )
    catalog["publication_bundle_root"] = str(normalized_output)
    catalog["publication_generated_at"] = datetime.now().isoformat()

    fixture_sets = discover_fixture_sets(root)
    grouped_fixture_sets: dict[str, list[FixtureSetDescriptor]] = {}
    for descriptor in fixture_sets:
        grouped_fixture_sets.setdefault(descriptor.benchmark, []).append(descriptor)

    benchmark_manifest_paths: dict[str, Path] = {}
    for benchmark_payload in catalog.get("benchmarks", []):
        benchmark_name = str(benchmark_payload.get("benchmark", "")).strip()
        if not benchmark_name:
            continue

        bundle_dir_name = f"{_slugify_bundle_component(benchmark_name)}_fixture_bundle"
        bundle_dir = normalized_output / bundle_dir_name
        if bundle_dir.exists():
            shutil.rmtree(bundle_dir)
        bundle_dir.mkdir(parents=True, exist_ok=True)

        fixture_sets_dir = bundle_dir / "fixture_sets"
        fixture_sets_dir.mkdir(parents=True, exist_ok=True)

        combined_artifacts: list[dict[str, Any]] = []
        published_fixture_set_manifest_paths: list[str] = []
        checksum_algorithm = "sha256"

        for fixture_descriptor in grouped_fixture_sets.get(benchmark_name, []):
            source_set_dir = fixture_descriptor.manifest_path.parent
            destination_set_dir = fixture_sets_dir / fixture_descriptor.name
            shutil.copytree(source_set_dir, destination_set_dir)

            published_manifest_path = destination_set_dir / fixture_descriptor.manifest_path.name
            published_fixture_set_manifest_paths.append(
                str(published_manifest_path.relative_to(normalized_output))
            )

            source_manifest = _load_fixture_manifest(fixture_descriptor.manifest_path)
            if source_manifest is None:
                raise ValueError(
                    f"Invalid fixture manifest for benchmark publication: "
                    f"{fixture_descriptor.manifest_path}"
                )
            checksum_algorithm = str(source_manifest.get("checksum_algorithm", checksum_algorithm))
            resolved_artifact_paths = _resolve_fixture_manifest_artifact_paths(
                fixture_descriptor.manifest_path
            )
            source_artifacts = list(source_manifest.get("artifacts", []))
            if len(source_artifacts) != len(resolved_artifact_paths):
                raise ValueError(
                    f"Fixture manifest artifact count mismatch for {fixture_descriptor.manifest_path}"
                )

            for source_artifact, resolved_artifact_path in zip(
                source_artifacts,
                resolved_artifact_paths,
            ):
                copied_artifact = dict(source_artifact)
                copied_artifact["source_fixture_set"] = fixture_descriptor.name
                copied_artifact["source_fixture_manifest_path"] = str(
                    fixture_descriptor.manifest_path
                )
                copied_artifact["published_fixture_set_manifest_path"] = str(
                    published_manifest_path.relative_to(normalized_output)
                )

                bundled_relative = str(source_artifact.get("bundled_artifact_path", "")).strip()
                if bundled_relative:
                    published_bundled_path = (
                        Path("fixture_sets") / fixture_descriptor.name / bundled_relative
                    )
                else:
                    try:
                        relative_artifact_path = resolved_artifact_path.relative_to(source_set_dir)
                    except ValueError:
                        relative_artifact_path = Path(resolved_artifact_path.name)
                    published_bundled_path = (
                        Path("fixture_sets") / fixture_descriptor.name / relative_artifact_path
                    )

                copied_bundled_file = bundle_dir / published_bundled_path
                if not copied_bundled_file.is_file():
                    raise ValueError(
                        "Published fixture artifact copy is missing: " f"{copied_bundled_file}"
                    )
                copied_artifact["bundled_artifact_path"] = str(published_bundled_path)
                copied_artifact["bundled_artifact_size_bytes"] = copied_bundled_file.stat().st_size
                copied_artifact["bundled_artifact_sha256"] = _compute_file_sha256(
                    copied_bundled_file
                )
                combined_artifacts.append(copied_artifact)

        combined_manifest = {
            "benchmark": benchmark_name,
            "timestamp": datetime.now().isoformat(),
            "checksum_algorithm": checksum_algorithm,
            "artifact_count": len(combined_artifacts),
            "fixture_set_count": len(grouped_fixture_sets.get(benchmark_name, [])),
            "fixture_set_names": list(benchmark_payload.get("fixture_set_names", [])),
            "fixture_sources": list(benchmark_payload.get("fixture_sources", [])),
            "artifacts": combined_artifacts,
        }
        combined_manifest.update(_build_fixture_benchmark_metadata_payload(benchmark_name))

        combined_manifest_path = bundle_dir / "comparison_report_fixtures.json"
        combined_manifest_path.write_text(json.dumps(combined_manifest, indent=2) + "\n")

        relative_combined_manifest = str(combined_manifest_path.relative_to(normalized_output))
        stable_run_report = create_comparison_report_from_fixture_manifest(
            combined_manifest_path,
            include_published=False,
        )
        stable_run_summary = build_publication_stable_run_summary(
            stable_run_report,
            manifest_path=relative_combined_manifest,
        )
        stable_run_summary_path = bundle_dir / "stable_run_summary.json"
        stable_run_summary_path.write_text(json.dumps(stable_run_summary, indent=2) + "\n")
        relative_stable_run_summary = str(stable_run_summary_path.relative_to(normalized_output))
        benchmark_payload["published_bundle_dir"] = bundle_dir_name
        benchmark_payload["published_manifest_path"] = relative_combined_manifest
        benchmark_payload["published_fixture_set_manifest_paths"] = (
            published_fixture_set_manifest_paths
        )
        benchmark_payload["stable_run_summary_path"] = relative_stable_run_summary
        benchmark_payload["stable_run_summary"] = {
            "stable_run_artifact_count": stable_run_summary["stable_run_artifact_count"],
            "best_model": stable_run_summary["best_result"]["model"],
            "best_pass_rate": stable_run_summary["best_result"]["pass_rate"],
            "required_public_kpis": stable_run_summary["required_public_kpis"],
            "kpi_availability": stable_run_summary["kpi_availability"],
            "missing_public_kpis": stable_run_summary["missing_public_kpis"],
            "required_public_kpi_complete": stable_run_summary["required_public_kpi_complete"],
            "corpus_readiness": stable_run_summary["corpus_readiness"],
        }
        benchmark_manifest_paths[benchmark_name] = combined_manifest_path

    catalog_path = normalized_output / "fixture_benchmark_publication_catalog.json"
    catalog_path.write_text(json.dumps(catalog, indent=2) + "\n")
    return {
        "root": normalized_output,
        "catalog": catalog_path,
        "benchmark_manifests": benchmark_manifest_paths,
    }


def save_stable_run_publication_bundle(
    *,
    output_path: Path,
    result_paths: Sequence[Path],
    benchmark: Optional[str] = None,
) -> dict[str, Any]:
    """Save a publication bundle from real saved benchmark run artifacts."""
    normalized_output = Path(output_path)
    normalized_output.mkdir(parents=True, exist_ok=True)

    report = create_comparison_report_from_saved_results(
        [Path(path) for path in result_paths],
        include_published=False,
    )
    _reject_fixture_stable_run_inputs(report)
    if benchmark is not None:
        requested_metadata = get_benchmark_metadata(str(benchmark))
        if requested_metadata is None:
            raise ValueError(f"Unknown benchmark: {benchmark}")
        if report.benchmark != requested_metadata.type:
            raise ValueError(
                "Saved benchmark result artifacts target "
                f"'{report.benchmark.value}', not requested benchmark '{requested_metadata.name}'"
            )

    benchmark_name = report.benchmark.value
    bundle_dir_name = f"{_slugify_bundle_component(benchmark_name)}_stable_run_bundle"
    bundle_dir = normalized_output / bundle_dir_name
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    bundle_paths = save_comparison_report_bundle(
        report,
        bundle_dir / "comparison_report.json",
        primary_format="json",
    )
    manifest_path = bundle_paths["fixtures"]
    manifest = _read_json_file(manifest_path)
    manifest["publication_kind"] = "stable_run"
    manifest["artifact_provenance"] = "real_run"
    manifest["source_result_paths"] = [str(Path(path)) for path in result_paths]
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    relative_manifest_path = str(manifest_path.relative_to(normalized_output))
    stable_run_summary = build_publication_stable_run_summary(
        report,
        manifest_path=relative_manifest_path,
    )
    stable_run_summary["publication_kind"] = "stable_run"
    stable_run_summary["artifact_provenance"] = "real_run"
    stable_run_summary["source_result_paths"] = [str(Path(path)) for path in result_paths]

    stable_run_summary_path = bundle_dir / "stable_run_summary.json"
    stable_run_summary_path.write_text(json.dumps(stable_run_summary, indent=2) + "\n")
    relative_stable_run_summary_path = str(stable_run_summary_path.relative_to(normalized_output))

    catalog = {
        "publication_kind": "stable_run",
        "artifact_provenance": "real_run",
        "publication_bundle_root": str(normalized_output),
        "publication_generated_at": datetime.now().isoformat(),
        "benchmark_count": 1,
        "artifact_count": len(report.results),
        "benchmarks": [
            {
                "benchmark": benchmark_name,
                "published_bundle_dir": bundle_dir_name,
                "published_manifest_path": relative_manifest_path,
                "stable_run_summary_path": relative_stable_run_summary_path,
                "comparison_report_path": str(bundle_paths["json"].relative_to(normalized_output)),
                "comparison_summary_path": str(
                    bundle_paths["summary"].relative_to(normalized_output)
                ),
                "source_result_paths": [str(Path(path)) for path in result_paths],
                "stable_run_summary": {
                    "stable_run_artifact_count": stable_run_summary["stable_run_artifact_count"],
                    "best_model": stable_run_summary["best_result"]["model"],
                    "best_pass_rate": stable_run_summary["best_result"]["pass_rate"],
                    "required_public_kpis": stable_run_summary["required_public_kpis"],
                    "kpi_availability": stable_run_summary["kpi_availability"],
                    "missing_public_kpis": stable_run_summary["missing_public_kpis"],
                    "required_public_kpi_complete": stable_run_summary[
                        "required_public_kpi_complete"
                    ],
                    "corpus_readiness": stable_run_summary["corpus_readiness"],
                },
            }
        ],
    }
    catalog["benchmarks"][0].update(_build_fixture_benchmark_metadata_payload(benchmark_name))

    catalog_path = normalized_output / "stable_run_publication_catalog.json"
    catalog_path.write_text(json.dumps(catalog, indent=2) + "\n")
    return {
        "root": normalized_output,
        "catalog": catalog_path,
        "benchmark_manifests": {benchmark_name: manifest_path},
    }


def resolve_fixture_benchmark_publication_manifests(
    *,
    root: Path,
    benchmark: Optional[str] = None,
) -> list[Path]:
    """Resolve direct-load benchmark manifests from a publication bundle catalog."""
    catalog_path = _resolve_fixture_benchmark_publication_catalog_path(Path(root))
    if catalog_path is None:
        raise ValueError(f"No fixture benchmark publication catalog found under {Path(root)}")

    catalog = _load_fixture_benchmark_publication_catalog(catalog_path)
    if catalog is None:
        raise ValueError(f"{catalog_path} is not a valid fixture benchmark publication catalog")

    benchmark_payloads = [
        payload for payload in catalog.get("benchmarks", []) if isinstance(payload, dict)
    ]
    if benchmark is not None:
        benchmark_payloads = [
            payload
            for payload in benchmark_payloads
            if fixture_benchmark_matches(str(payload.get("benchmark", "")).strip(), benchmark)
        ]
        if not benchmark_payloads:
            available_benchmarks = (
                ", ".join(
                    sorted(
                        dict.fromkeys(
                            str(payload.get("benchmark", "")).strip()
                            for payload in catalog.get("benchmarks", [])
                            if isinstance(payload, dict)
                            and str(payload.get("benchmark", "")).strip()
                        )
                    )
                )
                or "(none)"
            )
            raise ValueError(
                f"Unknown published fixture benchmark '{benchmark}'. "
                f"Available publication benchmarks under {Path(root)}: {available_benchmarks}"
            )

    resolved_paths: list[Path] = []
    base_dir = catalog_path.parent
    for payload in benchmark_payloads:
        manifest_value = str(payload.get("published_manifest_path", "")).strip()
        if not manifest_value:
            continue
        manifest_path = Path(manifest_value)
        if not manifest_path.is_absolute():
            manifest_path = base_dir / manifest_path
        if _load_fixture_manifest(manifest_path) is None:
            raise ValueError(
                f"Published fixture manifest {manifest_path} referenced by {catalog_path} is invalid"
            )
        resolved_paths.append(manifest_path)

    if not resolved_paths:
        raise ValueError(
            f"Fixture benchmark publication catalog {catalog_path} does not include any manifests"
        )
    return resolved_paths


def resolve_fixture_set_names(
    names: Sequence[str],
    *,
    root: Path = DEFAULT_FIXTURE_SET_ROOT,
) -> list[Path]:
    """Resolve checked-in fixture-set names to manifest paths."""
    descriptors = discover_fixture_sets(root)
    by_name = {descriptor.name: descriptor for descriptor in descriptors}
    resolved_paths: list[Path] = []
    missing_names: list[str] = []

    for raw_name in names:
        normalized_name = str(raw_name).strip()
        if not normalized_name:
            continue
        descriptor = by_name.get(normalized_name)
        if descriptor is None:
            missing_names.append(normalized_name)
            continue
        resolved_paths.append(descriptor.manifest_path)

    if missing_names:
        available_names = ", ".join(sorted(by_name)) if by_name else "(none)"
        raise ValueError(
            "Unknown fixture set name(s): "
            + ", ".join(missing_names)
            + f". Available fixture sets under {Path(root)}: {available_names}"
        )

    return resolved_paths


def resolve_fixture_sets_for_benchmark(
    benchmark: str,
    *,
    root: Path = DEFAULT_FIXTURE_SET_ROOT,
) -> list[Path]:
    """Resolve all checked-in fixture-set manifests for a benchmark."""
    normalized_benchmark = str(benchmark).strip()
    if not normalized_benchmark:
        raise ValueError("Fixture benchmark name is required")

    descriptors = discover_fixture_sets(root)
    matching_descriptors = [
        descriptor
        for descriptor in descriptors
        if fixture_benchmark_matches(descriptor.benchmark, normalized_benchmark)
    ]
    if not matching_descriptors:
        available_benchmarks = (
            ", ".join(sorted(dict.fromkeys(descriptor.benchmark for descriptor in descriptors)))
            or "(none)"
        )
        raise ValueError(
            f"Unknown fixture benchmark '{normalized_benchmark}'. "
            f"Available benchmarks under {Path(root)}: {available_benchmarks}"
        )
    return [descriptor.manifest_path for descriptor in matching_descriptors]


def verify_fixture_sets(
    *,
    root: Path = DEFAULT_FIXTURE_SET_ROOT,
    benchmark: Optional[str] = None,
    names: Sequence[str] = (),
) -> list[FixtureSetVerificationResult]:
    """Verify checked-in fixture-set integrity and return validated descriptors."""
    descriptors = discover_fixture_sets(root)
    if benchmark is not None:
        descriptors = [
            descriptor
            for descriptor in descriptors
            if fixture_benchmark_matches(descriptor.benchmark, benchmark)
        ]
    if names:
        selected_names = {str(name).strip() for name in names if str(name).strip()}
        descriptors = [
            descriptor for descriptor in descriptors if descriptor.name in selected_names
        ]
    if not descriptors:
        raise ValueError(f"No fixture sets found under {Path(root)}")

    results: list[FixtureSetVerificationResult] = []
    for descriptor in descriptors:
        verified_paths = _resolve_fixture_manifest_artifact_paths(descriptor.manifest_path)
        results.append(
            FixtureSetVerificationResult(
                name=descriptor.name,
                benchmark=descriptor.benchmark,
                manifest_path=descriptor.manifest_path,
                artifact_count=descriptor.artifact_count,
                verified_artifact_count=len(verified_paths),
            )
        )

    return results


def _validate_fixture_artifact_file(
    path: Path,
    *,
    expected_size: Any,
    expected_checksum: Any,
    checksum_algorithm: Optional[str],
) -> None:
    if expected_size not in (None, ""):
        try:
            normalized_size = int(expected_size)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid fixture artifact size for {path}") from exc
        actual_size = path.stat().st_size
        if actual_size != normalized_size:
            raise ValueError(
                "Fixture artifact integrity mismatch for "
                f"{path}: expected size {normalized_size}, got {actual_size}"
            )
    if expected_checksum in (None, ""):
        return
    normalized_algorithm = (checksum_algorithm or "sha256").strip().lower()
    if normalized_algorithm != "sha256":
        raise ValueError(f"Unsupported fixture checksum algorithm: {normalized_algorithm}")
    actual_checksum = _compute_file_sha256(path)
    if actual_checksum != str(expected_checksum):
        raise ValueError(
            "Fixture artifact integrity mismatch for "
            f"{path}: expected checksum {expected_checksum}, got {actual_checksum}"
        )


def _resolve_fixture_manifest_artifact_paths(path: Path) -> list[Path]:
    manifest_path = _resolve_fixture_manifest_path(path) or path
    manifest = _load_fixture_manifest(manifest_path)
    if manifest is None:
        raise ValueError(f"{path} is not a valid comparison fixture manifest")

    checksum_algorithm = str(manifest.get("checksum_algorithm", "sha256"))
    resolved_paths: list[Path] = []
    for index, artifact in enumerate(manifest.get("artifacts", []), start=1):
        if not isinstance(artifact, dict):
            raise ValueError(f"Invalid artifact entry #{index} in fixture manifest {path}")

        candidate_specs: list[tuple[Path, Any, Any]] = []
        bundled_relative = str(artifact.get("bundled_artifact_path", "")).strip()
        if bundled_relative:
            candidate_specs.append(
                (
                    manifest_path.parent / bundled_relative,
                    artifact.get("bundled_artifact_size_bytes"),
                    artifact.get("bundled_artifact_sha256"),
                )
            )
        source_path = str(artifact.get("artifact_path", "")).strip()
        if source_path:
            candidate_specs.append(
                (
                    Path(source_path),
                    artifact.get("artifact_size_bytes"),
                    artifact.get("artifact_sha256"),
                )
            )

        resolved_path: Optional[Path] = None
        for candidate_path, expected_size, expected_checksum in candidate_specs:
            if not candidate_path.is_file():
                continue
            _validate_fixture_artifact_file(
                candidate_path,
                expected_size=expected_size,
                expected_checksum=expected_checksum,
                checksum_algorithm=checksum_algorithm,
            )
            resolved_path = candidate_path
            break

        if resolved_path is None:
            raise ValueError(
                f"Could not resolve fixture artifact #{index} from manifest {manifest_path}"
            )
        resolved_paths.append(resolved_path)

    if not resolved_paths:
        raise ValueError(
            f"Fixture manifest {manifest_path} does not include any saved benchmark artifacts"
        )
    return resolved_paths


def _expand_saved_result_paths(paths: Sequence[Path]) -> list[Path]:
    expanded_paths: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        manifest_path = _resolve_fixture_manifest_path(path)
        if manifest_path is not None:
            expanded_paths.extend(_resolve_fixture_manifest_artifact_paths(path))
            continue

        publication_catalog_path = _resolve_fixture_benchmark_publication_catalog_path(path)
        if publication_catalog_path is not None:
            for manifest_path in resolve_fixture_benchmark_publication_manifests(
                root=publication_catalog_path
            ):
                expanded_paths.extend(_resolve_fixture_manifest_artifact_paths(manifest_path))
            continue

        expanded_paths.append(path)
    return expanded_paths


def save_comparison_report_bundle(
    report: ComparisonReport,
    output_path: Path,
    *,
    primary_format: str = "markdown",
) -> dict[str, Path]:
    """Write markdown, json, and summary sidecars for a comparison report."""
    normalized_output = Path(output_path)
    if normalized_output.suffix:
        base_path = normalized_output.with_suffix("")
        normalized_output.parent.mkdir(parents=True, exist_ok=True)
    else:
        normalized_output.mkdir(parents=True, exist_ok=True)
        base_path = normalized_output / "comparison_report"

    markdown_path = base_path.with_suffix(".md")
    json_path = base_path.with_suffix(".json")
    summary_path = base_path.parent / f"{base_path.name}_summary.json"
    fixtures_path = base_path.parent / f"{base_path.name}_fixtures.json"
    fixture_dir = base_path.parent / f"{base_path.name}_fixtures"

    markdown_path.write_text(report.to_markdown())
    json_path.write_text(report.to_json())
    summary_path.write_text(json.dumps(build_comparison_report_summary(report), indent=2))

    if fixture_dir.exists():
        shutil.rmtree(fixture_dir)
    fixture_dir.mkdir(parents=True, exist_ok=True)

    fixture_manifest = build_comparison_report_fixture_manifest(report)
    bundled_artifacts: list[dict[str, Any]] = []
    for index, artifact in enumerate(fixture_manifest.get("artifacts", []), start=1):
        source_path = str(artifact.get("artifact_path", "")).strip()
        copied_artifact = dict(artifact)
        if source_path:
            source = Path(source_path)
            if source.is_file():
                framework_name = _slugify_bundle_component(str(artifact.get("framework", "")))
                model_name = _slugify_bundle_component(str(artifact.get("model", "")))
                bundled_name = f"{index:02d}_{framework_name}_{model_name}.json"
                bundled_path = fixture_dir / bundled_name
                shutil.copy2(source, bundled_path)
                copied_artifact["bundled_artifact_path"] = f"{fixture_dir.name}/{bundled_name}"
                copied_artifact["bundled_artifact_size_bytes"] = bundled_path.stat().st_size
                copied_artifact["bundled_artifact_sha256"] = _compute_file_sha256(bundled_path)
        bundled_artifacts.append(copied_artifact)
    fixture_manifest["artifacts"] = bundled_artifacts
    fixtures_path.write_text(json.dumps(fixture_manifest, indent=2))

    primary_format_normalized = primary_format.strip().lower()
    if primary_format_normalized == "json":
        primary_path = json_path
    else:
        primary_path = markdown_path

    return {
        "primary": primary_path,
        "markdown": markdown_path,
        "json": json_path,
        "summary": summary_path,
        "fixtures": fixtures_path,
        "fixture_dir": fixture_dir,
    }


def create_quick_comparison(
    benchmark: BenchmarkType = BenchmarkType.SWE_BENCH,
    victor_pass_rate: float = 0.0,
    victor_model: str = "claude-3-sonnet",
    include_published: bool = True,
) -> ComparisonReport:
    """Create a comparison report without running actual benchmarks.

    Useful for testing the comparison pipeline or generating reports
    from pre-computed pass rates.

    Args:
        benchmark: Benchmark type for the report
        victor_pass_rate: Victor's pass rate to use
        victor_model: Model name for Victor's entry
        include_published: Include published competitor data

    Returns:
        ComparisonReport with Victor and (optionally) published results
    """
    report = ComparisonReport(benchmark=benchmark)

    victor_metrics = ComparisonMetrics(pass_rate=victor_pass_rate)
    victor_result = FrameworkResult(
        framework=Framework.VICTOR,
        benchmark=benchmark,
        model=victor_model,
        metrics=victor_metrics,
    )
    report.results.append(victor_result)

    if include_published and benchmark in PUBLISHED_RESULTS:
        for framework, data in PUBLISHED_RESULTS[benchmark].items():
            metrics = ComparisonMetrics(pass_rate=data.get("pass_rate", 0.0))
            result = FrameworkResult(
                framework=framework,
                benchmark=benchmark,
                model=data.get("model", "unknown"),
                metrics=metrics,
                config={"source": data.get("source", "published")},
            )
            report.results.append(result)

    return report
