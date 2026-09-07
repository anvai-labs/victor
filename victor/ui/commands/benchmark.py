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

"""Benchmark CLI commands for Victor.

Provides CLI commands for running industry-standard AI coding benchmarks:
- SWE-bench: Real-world GitHub issue resolution
- HumanEval: Code generation from docstrings
- MBPP: Basic Python programming problems
- Framework comparison: Compare against other AI coding assistants

Usage:
    victor benchmark list                       # List available benchmarks
    victor benchmark run swe-bench --max-tasks 10
    victor benchmark run humaneval --model claude-3-sonnet
    victor benchmark run mbpp --output results.json
    victor benchmark compare --benchmark swe-bench
    victor benchmark leaderboard
"""

import asyncio
import json
import re
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from victor.config.settings import get_project_paths
from victor.core.async_utils import run_sync
from victor.ui.commands.utils import setup_logging

benchmark_app = typer.Typer(
    name="benchmark",
    help="Run AI coding benchmarks and compare against other frameworks.",
)
console = Console()

# Benchmarks whose runner executes ``agent_output + test_code`` as one Python
# file, so the agent's *source* is the artifact.
#
# Opt-in rather than opt-out on purpose: only these two have been read and
# confirmed to concatenate-and-run, so nothing else changes behaviour.
# Slugs, not the enum: BenchmarkType is imported lazily inside functions here,
# so a module-level reference to it would break the import.
_SOURCE_BENCHMARKS = frozenset({"mbpp", "human_eval"})

# Report benchmarks also consume direct output, but keeping them separate from
# executable-source benchmarks makes the routing contract explicit.
_REPORT_BENCHMARKS = frozenset({"dr3_eval"})


def _stage_benchmark_input_files(task: Any, work_dir: Path) -> None:
    """Copy validated binary/text task inputs into an ephemeral workspace."""
    import shutil

    workspace = work_dir.resolve()
    for relative_name, source_name in (getattr(task, "input_files", None) or {}).items():
        relative_path = Path(relative_name)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"Benchmark input destination escapes workspace: {relative_name}")
        destination = (workspace / relative_path).resolve()
        try:
            destination.relative_to(workspace)
        except ValueError as exc:
            raise ValueError(
                f"Benchmark input destination escapes workspace: {relative_name}"
            ) from exc

        source = Path(source_name).resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Benchmark input file not found: {source}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _get_global_evaluations_dir() -> Path:
    """Resolve the global benchmark evaluations directory."""
    return get_project_paths().global_victor_dir / "evaluations"


def _get_global_usage_logs_dir() -> Path:
    """Resolve the global usage-log directory for benchmark analytics."""
    return get_project_paths().global_logs_dir


def _get_prompt_section_baselines(section_names: list[str]) -> dict[str, str]:
    """Resolve canonical prompt-section fallback text from the shared registry."""
    from victor.agent.prompt_section_registry import build_fallback_map

    return build_fallback_map(section_names)


def _infer_provider_from_model_name(model_name: str) -> str:
    """Infer the provider family from a benchmark model name."""
    model_lower = (model_name or "").lower()
    for prefix, provider in [
        ("gpt", "openai"),
        ("grok", "xai"),
        ("deepseek", "deepseek"),
        ("haiku", "anthropic"),
        ("claude", "anthropic"),
    ]:
        if prefix in model_lower:
            return provider
    return "default"


def _auto_evolve_prompt_candidate(
    learner: Any,
    *,
    model_name: str,
    metrics: dict[str, Any],
) -> Optional[tuple[Any, str]]:
    """Seed one post-run prompt candidate from benchmark outcomes."""
    baseline_text = _get_prompt_section_baselines(["ASI_TOOL_EFFECTIVENESS_GUIDANCE"]).get(
        "ASI_TOOL_EFFECTIVENESS_GUIDANCE"
    )
    if not baseline_text:
        return None

    provider = _infer_provider_from_model_name(model_name)
    candidate = learner.evolve(
        "ASI_TOOL_EFFECTIVENESS_GUIDANCE",
        baseline_text,
        provider=provider,
    )
    if candidate is None:
        return None

    for _ in range(int(metrics.get("passed", 0) or 0)):
        candidate.update(True)
    for _ in range(int(metrics.get("failed", 0) or 0) + int(metrics.get("errors", 0) or 0)):
        candidate.update(False)
    learner._save_candidate(candidate)
    return candidate, provider


@dataclass(frozen=True)
class CodeIntelligencePrewarmResult:
    """Outcome of benchmark code-intelligence prewarm for a repo."""

    status: str
    message: str
    cached_hit: bool = False
    graph_nodes: int = 0
    graph_edges: int = 0


def _resolve_benchmark_target(benchmark: str, dataset_path: Optional[Path]):
    """Resolve a catalog benchmark into metadata, type, and runner instance."""
    from victor.evaluation.protocol import (
        BenchmarkType,
        get_benchmark_catalog,
        get_benchmark_metadata,
        normalize_benchmark_name,
        requires_local_manifest_benchmark,
    )

    benchmark_lower = normalize_benchmark_name(benchmark)
    metadata = get_benchmark_metadata(benchmark_lower)
    if metadata is None:
        available = ", ".join(item.name for item in get_benchmark_catalog())
        console.print(f"[bold red]Error:[/] Unknown benchmark: {benchmark}")
        console.print(f"Available: {available}")
        raise typer.Exit(1)

    from victor.evaluation.benchmarks import (
        BrowserTaskBenchmarkRunner,
        DeepResearchBenchmarkRunner,
        SWEBenchRunner,
        HumanEvalRunner,
        MBPPRunner,
    )

    if requires_local_manifest_benchmark(metadata.type) and dataset_path is None:
        console.print(
            f"[bold red]Error:[/] Benchmark '{metadata.name}' requires --dataset-path "
            "to load a local adapter manifest."
        )
        raise typer.Exit(1)

    benchmark_key = metadata.name
    benchmark_map = {
        "swe-bench": (BenchmarkType.SWE_BENCH, SWEBenchRunner),
        "swe-bench-lite": (
            BenchmarkType.SWE_BENCH,
            lambda: SWEBenchRunner(split="lite"),
        ),
        "humaneval": (BenchmarkType.HUMAN_EVAL, HumanEvalRunner),
        "mbpp": (BenchmarkType.MBPP, MBPPRunner),
        "mbpp-test": (BenchmarkType.MBPP, lambda: MBPPRunner(split="test")),
        "dr3-eval": (
            BenchmarkType.DR3_EVAL,
            lambda: DeepResearchBenchmarkRunner(dataset_path),
        ),
        "clawbench": (
            BenchmarkType.CLAW_BENCH,
            lambda: BrowserTaskBenchmarkRunner(BenchmarkType.CLAW_BENCH, dataset_path),
        ),
        "guide": (
            BenchmarkType.GUIDE,
            lambda: BrowserTaskBenchmarkRunner(BenchmarkType.GUIDE, dataset_path),
        ),
        "vlaa-gui": (
            BenchmarkType.VLAA_GUI,
            lambda: BrowserTaskBenchmarkRunner(BenchmarkType.VLAA_GUI, dataset_path),
        ),
    }

    if benchmark_key not in benchmark_map:
        console.print(
            f"[yellow]Benchmark '{metadata.name}' is cataloged, but the runner adapter "
            f"is not wired yet ({metadata.runner_status}).[/]"
        )
        raise typer.Exit(1)

    bench_type, runner_factory = benchmark_map[benchmark_key]

    # SWE-bench/HumanEval/MBPP/browser-task runners were extracted to the
    # optional `victor-coding` package and soft-load to None when it is absent
    # (victor.evaluation.benchmarks). Fail fast with an actionable message
    # instead of a NoneType/TypeError deep inside execution.
    _extracted_runner = {
        "swe-bench": SWEBenchRunner,
        "swe-bench-lite": SWEBenchRunner,
        "humaneval": HumanEvalRunner,
        "mbpp": MBPPRunner,
        "mbpp-test": MBPPRunner,
        "clawbench": BrowserTaskBenchmarkRunner,
        "guide": BrowserTaskBenchmarkRunner,
        "vlaa-gui": BrowserTaskBenchmarkRunner,
    }
    if benchmark_key in _extracted_runner and _extracted_runner[benchmark_key] is None:
        console.print(
            f"[bold red]Error:[/] Benchmark '{metadata.name}' requires the optional "
            "'victor-coding' package, which is not installed.\n"
            "  Install it with: [bold]pip install victor-coding[/]"
        )
        raise typer.Exit(1)

    runner = runner_factory() if callable(runner_factory) else runner_factory
    return metadata, bench_type, runner


def _resolve_account_selection(
    account: Optional[str],
    provider: Optional[str],
    model: Optional[str],
) -> tuple[Optional[str], Optional[str], Any]:
    """Resolve an account override into provider/model selection."""
    resolved_account = None
    if not account:
        return provider, model, resolved_account

    from victor.config.accounts import get_account_manager

    mgr = get_account_manager()
    resolved_account = mgr.get_account(account)
    if not resolved_account:
        console.print(f"[bold red]Error:[/] Account '{account}' not found")
        console.print("[dim]Run 'victor auth list' to see available accounts[/]")
        raise typer.Exit(1)

    provider = resolved_account.provider
    if not model:
        model = resolved_account.model
    console.print(f"[cyan]Using account '{account}' ({resolved_account.provider}/{model})[/]")
    return provider, model, resolved_account


# Maps a benchmark to the vertical whose capabilities it needs registered via
# AgentFactory (e.g. "coding" → code_search/graph). Benchmarks not listed use the
# default (no vertical) and rely on whatever tools the profile/session provides.
_BENCHMARK_VERTICAL: dict[str, str] = {
    "swe-bench": "coding",
    "swe-bench-lite": "coding",
    "humaneval": "coding",
    "mbpp": "coding",
    "mbpp-test": "coding",
    "dr3-eval": "research",
}


def _resolve_effective_model(profile: str, model: Optional[str]) -> str:
    """Resolve the effective benchmark model from CLI override or profile."""
    if model:
        return model

    from victor.config.settings import Settings

    settings = Settings()
    try:
        profiles = settings.load_profiles()
        profile_config = profiles.get(profile)
        if profile_config:
            return profile_config.model
        console.print(f"[yellow]Warning:[/] Profile '{profile}' not found, using default model")
        return "claude-3-sonnet"
    except Exception as e:
        console.print(f"[yellow]Warning:[/] Could not load profile: {e}")
        return "claude-3-sonnet"


def _attach_manifest_metadata(config, runner) -> None:
    """Copy dataset manifest metadata from the runner into the evaluation config."""
    manifest_metadata = getattr(runner, "manifest_metadata", None)
    if manifest_metadata is not None and hasattr(manifest_metadata, "to_dict"):
        config.dataset_metadata = manifest_metadata.to_dict()


def _print_benchmark_header(
    *,
    title: str,
    benchmark: str,
    config,
    max_tasks: Optional[int],
    dataset_path: Optional[Path],
    timeout: int,
) -> None:
    """Print the standard benchmark run header."""
    provider = getattr(config, "provider", None)
    prompt_section_name = getattr(config, "prompt_section_name", None)
    prompt_candidate_hash = getattr(config, "prompt_candidate_hash", None)
    dataset_metadata = getattr(config, "dataset_metadata", None) or {}

    console.print(f"\n[bold cyan]{title}[/]")
    console.print(f"Benchmark: {benchmark}")
    console.print(f"Model: {config.model}")
    if provider:
        console.print(f"Provider: {provider}")
    if prompt_section_name:
        console.print(f"Prompt Section: {prompt_section_name}")
    if prompt_candidate_hash:
        console.print(f"Prompt Candidate: {prompt_candidate_hash}")
    if max_tasks:
        console.print(f"Max tasks: {max_tasks}")
    if dataset_path is not None:
        console.print(f"Dataset: {dataset_path}")
    if dataset_metadata:
        source = dataset_metadata.get("source_name")
        version = dataset_metadata.get("version")
        languages = dataset_metadata.get("languages") or []
        if source:
            console.print(f"Source: {source}")
        if version:
            console.print(f"Manifest Version: {version}")
        if languages:
            console.print(f"Languages: {', '.join(languages)}")
    console.print(f"Timeout: {timeout}s per task")
    console.print()


def _print_prompt_candidate_suite_summary(suite) -> None:
    """Print a compact comparison table for a prompt candidate suite."""
    table = Table(title="Prompt Candidate Suite Summary")
    table.add_column("Candidate", style="cyan")
    table.add_column("Pass Rate", style="white")
    table.add_column("Passed", style="green")
    table.add_column("Failed", style="red")
    table.add_column("Duration", style="white")

    for run in suite.runs:
        metrics = run.result.get_metrics()
        table.add_row(
            run.label,
            f"{metrics['pass_rate']:.1%}",
            str(metrics["passed"]),
            str(metrics["failed"] + metrics["errors"] + metrics["timeouts"]),
            f"{metrics['duration_seconds']:.1f}s",
        )

    console.print(table)
    best = suite.best_run()
    if best is not None:
        console.print(
            f"[bold green]Best candidate:[/] {best.label} "
            f"({best.result.pass_rate:.1%} pass rate)"
        )


def _serialize_prompt_candidate_suite(
    benchmark: str,
    prompt_section: str,
    suite,
    prompt_optimizer_sync: Optional[object] = None,
    prompt_rollout: Optional[dict[str, Any]] = None,
    prompt_rollout_analysis: Optional[dict[str, Any]] = None,
    prompt_rollout_decision: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Serialize suite results to a stable JSON-friendly structure."""
    best = suite.best_run()
    payload = {
        "benchmark": benchmark,
        "model": suite.base_config.model,
        "provider": suite.base_config.provider,
        "section_name": prompt_section,
        "prompt_section_name": prompt_section,
        "config": suite.base_config.to_artifact_config(),
        "best_label": best.label if best is not None else None,
        "best_prompt_candidate_hash": (
            best.config.prompt_candidate_hash if best is not None else None
        ),
        "runs": [
            {
                "label": run.label,
                "provider": run.config.provider,
                "prompt_candidate_hash": run.config.prompt_candidate_hash,
                "section_name": run.config.prompt_section_name,
                "metrics": run.result.get_metrics(),
                "task_results": [
                    {
                        "task_id": task_result.task_id,
                        "status": task_result.status.value,
                        "tests_passed": task_result.tests_passed,
                        "tests_total": task_result.tests_total,
                        "duration": task_result.duration_seconds,
                        "tool_calls": task_result.tool_calls,
                        "code_search_calls": task_result.code_search_calls,
                        "graph_calls": task_result.graph_calls,
                        "failure_category": (
                            task_result.failure_category.value
                            if task_result.failure_category
                            else None
                        ),
                        "failure_details": task_result.failure_details,
                    }
                    for task_result in run.result.task_results
                ],
            }
            for run in suite.runs
        ],
    }
    if prompt_optimizer_sync is not None and hasattr(prompt_optimizer_sync, "to_dict"):
        payload["prompt_optimizer_sync"] = prompt_optimizer_sync.to_dict()
    if prompt_rollout is not None:
        payload["prompt_rollout"] = prompt_rollout
    if prompt_rollout_analysis is not None:
        payload["prompt_rollout_analysis"] = prompt_rollout_analysis
    if prompt_rollout_decision is not None:
        payload["prompt_rollout_decision"] = prompt_rollout_decision
    return payload


def _print_prompt_rollout_analysis_summary(report: dict[str, Any]) -> None:
    """Print rollout analysis for the approved suite winner."""
    console.print("\n[bold cyan]Prompt rollout analysis[/]")
    console.print(f"Experiment: {report.get('experiment_id', '-')}")
    console.print(f"Status: {report.get('status', '-')}")
    if not report.get("analysis_available", False):
        console.print(f"[yellow]{report.get('recommendation', 'Analysis unavailable')}[/]")
        return

    console.print(f"Recommendation: {report.get('recommendation', '-')}")
    console.print(f"Auto-apply action: {report.get('auto_action') or 'none'}")
    console.print(f"Significant: {'yes' if report.get('is_significant') else 'no'}")
    console.print(f"Treatment better: {'yes' if report.get('treatment_better') else 'no'}")
    console.print(f"Effect size: {float(report.get('effect_size', 0.0)):.1%}")
    console.print(f"P-value: {float(report.get('p_value', 1.0)):.4f}")


def _print_prompt_optimizer_sync_summary(sync_result) -> None:
    """Print the outcome of syncing a prompt suite into the optimizer state."""
    console.print("\n[bold cyan]Prompt optimizer benchmark sync[/]")
    if not getattr(sync_result, "decisions", None):
        console.print("[yellow]No prompt candidates were updated.[/]")
        return

    for decision in sync_result.decisions:
        status_parts = []
        if getattr(decision, "recorded", False):
            status_parts.append("recorded")
        else:
            status_parts.append("missing")
        if getattr(decision, "passed", False):
            status_parts.append("approved")
        if getattr(decision, "promoted", False):
            status_parts.append("promoted")
        # The paired result is the one that decides approval, so it leads. An
        # absolute pass rate says nothing about whether this beats the prompt we
        # already ship.
        contrast = getattr(decision, "paired_contrast", None)
        verdict = (
            f"vs baseline {contrast.summary()}"
            if contrast is not None
            else f"pass_rate={decision.score:.1%} [dim](no baseline arm)[/]"
        )
        console.print(
            "  "
            f"#{decision.rank} {decision.section_name}:{decision.provider}:{decision.prompt_candidate_hash} "
            f"{verdict} [{', '.join(status_parts)}]"
        )

    promoted_hash = getattr(sync_result, "promoted_prompt_candidate_hash", None)
    approved_hash = getattr(sync_result, "approved_prompt_candidate_hash", None)
    if promoted_hash:
        console.print(f"[bold green]Promoted best candidate:[/] {promoted_hash}")
    elif approved_hash:
        console.print(f"[bold green]Approved best candidate:[/] {approved_hash}")
    else:
        console.print("[yellow]No candidate met the benchmark approval threshold.[/]")


def _print_suite_verdict(verdict: Any, reason: str) -> None:
    """Print the classified verdict — color by trust, with a machine-parseable line.

    This is the gate signal a CI workflow consumes. It replaces the ambiguity
    where "no candidate met the threshold" read identically for a valid
    statistical loss and for a result that was never recorded: the verdict
    names whether the measurement was valid at all, and routes the red.
    """
    from victor.evaluation.suite_verdict import SuiteVerdict

    routing = {
        SuiteVerdict.GREEN: ("GREEN", "promote / merge"),
        SuiteVerdict.RED_REFUSED: ("RED_REFUSED", "re-evolve"),
        SuiteVerdict.RED_INCOMPLETE: ("RED_INCOMPLETE", "re-run"),
        SuiteVerdict.RED_THROTTLED: ("RED_THROTTLED", "re-run with quota"),
        SuiteVerdict.RED_UNRECORDED: ("RED_UNRECORDED", "fix infra"),
        SuiteVerdict.RED_CROSS_PROVIDER: ("RED_CROSS_PROVIDER", "re-run same provider"),
        SuiteVerdict.RED_NO_BASELINE: ("RED_NO_BASELINE", "add a baseline arm"),
    }
    label, action = routing[verdict]
    style = "bold green" if verdict is SuiteVerdict.GREEN else "bold red"
    console.print(f"\n[{style}]Suite verdict: {label} — {action}[/]")
    console.print(f"[dim]  reason: {reason}[/]")
    console.print(f"VERDICT={verdict.value}")


def _ensure_benchmark_runtime_tools(adapter) -> object:
    """Fail fast when the benchmark session is missing required core tools."""
    readiness = adapter.get_benchmark_tool_readiness()
    if readiness.ready:
        return readiness

    issues = []
    if readiness.missing_tools:
        issues.append(f"missing tools: {', '.join(readiness.missing_tools)}")
    if readiness.disabled_tools:
        issues.append(f"disabled tools: {', '.join(readiness.disabled_tools)}")

    raise RuntimeError(
        "Benchmark runtime is not ready for code-intelligence execution: "
        + "; ".join(issues)
        + ". Check tool discovery and tool configuration before running benchmarks."
    )


def _summarize_code_intelligence_diagnostics(
    result: Any,
    *,
    sample_limit: int = 5,
) -> dict[str, Any]:
    """Summarize which benchmark tasks skipped the code-intelligence path."""
    tasks = list(getattr(result, "task_results", []) or [])
    missing_code_intel = [
        task for task in tasks if not getattr(task, "used_code_intelligence", False)
    ]
    failed_missing = [
        task
        for task in missing_code_intel
        if getattr(getattr(task, "status", None), "value", None) in {"failed", "error", "timeout"}
    ]
    missing_graph = [task for task in tasks if not getattr(task, "used_graph", False)]

    def _task_ids(items: list[Any]) -> list[str]:
        return [str(getattr(item, "task_id", "")) for item in items if getattr(item, "task_id", "")]

    total_tasks = len(tasks)
    missing_ids = _task_ids(missing_code_intel)
    failed_missing_ids = _task_ids(failed_missing)
    missing_graph_ids = _task_ids(missing_graph)
    coverage = ((total_tasks - len(missing_ids)) / total_tasks) if total_tasks else 0.0

    return {
        "total_tasks": total_tasks,
        "code_intelligence_coverage": coverage,
        "tasks_without_code_intelligence": len(missing_ids),
        "task_ids_without_code_intelligence": missing_ids,
        "sample_task_ids_without_code_intelligence": missing_ids[:sample_limit],
        "failed_tasks_without_code_intelligence": len(failed_missing_ids),
        "failed_task_ids_without_code_intelligence": failed_missing_ids,
        "sample_failed_task_ids_without_code_intelligence": failed_missing_ids[:sample_limit],
        "tasks_without_graph": len(missing_graph_ids),
        "task_ids_without_graph": missing_graph_ids,
        "sample_task_ids_without_graph": missing_graph_ids[:sample_limit],
    }


def _summarize_failure_examples(
    result: Any,
    *,
    sample_limit: int = 3,
) -> dict[str, dict[str, Any]]:
    """Group representative benchmark failures by normalized category."""
    grouped: dict[str, dict[str, Any]] = {}
    for task in list(getattr(result, "task_results", []) or []):
        category = getattr(getattr(task, "failure_category", None), "value", None)
        if not category:
            continue
        entry = grouped.setdefault(
            category,
            {"count": 0, "sample_task_ids": [], "sample_errors": []},
        )
        entry["count"] += 1
        task_id = str(getattr(task, "task_id", "") or "")
        error_message = str(getattr(task, "error_message", "") or "")
        if task_id and len(entry["sample_task_ids"]) < sample_limit:
            entry["sample_task_ids"].append(task_id)
        if error_message and len(entry["sample_errors"]) < sample_limit:
            entry["sample_errors"].append(error_message[:200])
    return grouped


def _comparison_result_source(result: Any) -> str:
    """Format a human-readable source label for comparison rows."""
    config = getattr(result, "config", {}) or {}
    source = str(config.get("source") or "").strip()
    artifact_path = str(config.get("artifact_path") or "").strip()
    artifact_name = Path(artifact_path).name if artifact_path else ""
    if source and artifact_name:
        return f"{source} [{artifact_name}]"
    if source:
        return source
    if artifact_name:
        return f"Local result [{artifact_name}]"
    return ""


def _comparison_has_local_metrics(result: Any) -> bool:
    """Return True when extended competitive metrics came from a local artifact."""
    config = getattr(result, "config", {}) or {}
    if config.get("artifact_path"):
        return True
    task_results = getattr(result, "task_results", None)
    return bool(task_results)


def _format_comparison_percent(value: float, *, available: bool) -> str:
    if not available:
        return "-"
    return f"{value:.1%}"


def _format_comparison_seconds(value: float, *, available: bool) -> str:
    if not available:
        return "-"
    return f"{value:.2f}s"


async def _prewarm_code_intelligence_index(
    work_dir: Optional[Path],
    warmed_repos: Dict[str, CodeIntelligencePrewarmResult],
    *,
    timeout: float = 300.0,
) -> CodeIntelligencePrewarmResult:
    """Prewarm the shared code-search/graph index for a benchmark workspace."""
    if work_dir is None:
        return CodeIntelligencePrewarmResult(
            status="skipped",
            message="  [yellow]Code intelligence pre-warm skipped: workspace unavailable[/]",
        )

    repo_key = str(work_dir.resolve())
    cached = warmed_repos.get(repo_key)
    if cached is not None:
        if cached.status == "ready":
            message = "  Code search index pre-warmed (cached)"
        elif cached.status == "timeout":
            message = "  [yellow]Skipping index pre-warm after previous timeout[/]"
        else:
            message = "  [yellow]Skipping index pre-warm after previous failure[/]"
        return CodeIntelligencePrewarmResult(
            status=cached.status,
            message=message,
            cached_hit=True,
            graph_nodes=cached.graph_nodes,
            graph_edges=cached.graph_edges,
        )

    try:
        from victor.config.settings import load_settings

        from victor.core.utils.capability_loader import load_code_search_module

        _code_search_module = load_code_search_module()
        _get_or_build_index = _code_search_module._get_or_build_index

        prewarm_settings = load_settings()
        index, _rebuilt = await asyncio.wait_for(
            _get_or_build_index(work_dir, prewarm_settings, force_reindex=False),
            timeout=timeout,
        )

        graph_nodes = 0
        graph_edges = 0
        graph_store = getattr(index, "graph_store", None)
        if graph_store is not None:
            try:
                stats = await graph_store.stats()
                graph_nodes = int(stats.get("nodes", 0) or 0)
                graph_edges = int(stats.get("edges", 0) or 0)
            except Exception as exc:
                logger.debug("Graph stats unavailable during prewarm: %s", exc)

        message = "  Code search index pre-warmed"
        if graph_nodes or graph_edges:
            message += f" ({graph_nodes} graph nodes, {graph_edges} graph edges)"

        result = CodeIntelligencePrewarmResult(
            status="ready",
            message=message,
            graph_nodes=graph_nodes,
            graph_edges=graph_edges,
        )
        warmed_repos[repo_key] = result
        return result
    except asyncio.TimeoutError:
        logger.warning("Index pre-warm timed out after %.0fs, code_search may be slow", timeout)
        result = CodeIntelligencePrewarmResult(
            status="timeout",
            message="  [yellow]Code search index pre-warm timed out[/]",
        )
        warmed_repos[repo_key] = result
        return result
    except Exception as exc:
        logger.debug("Index pre-warm failed: %s", exc)
        result = CodeIntelligencePrewarmResult(
            status="failed",
            message=f"  [yellow]Code search index pre-warm skipped: {exc}[/]",
        )
        warmed_repos[repo_key] = result
        return result


def _configure_log_level(
    log_level: Optional[str],
    command: str = "benchmark",
    debug_modules: Optional[str] = None,
) -> None:
    """Configure logging for benchmark commands using centralized config.

    Uses the centralized logging config system with proper priority chain:
    1. CLI argument (log_level)
    2. Environment variable (VICTOR_LOG_LEVEL)
    3. User config (~/.victor/config.yaml)
    4. Command-specific override from package config
    5. Package defaults (WARNING console, INFO file)

    Args:
        log_level: CLI-provided log level (highest priority)
        command: Command name for command-specific config lookup
    """
    # Validate log level if provided
    if log_level is not None:
        log_level = log_level.upper()
        valid_levels = {"DEBUG", "INFO", "WARN", "WARNING", "ERROR", "CRITICAL"}

        if log_level not in valid_levels:
            console.print(
                f"[bold red]Error:[/] Invalid log level '{log_level}'. "
                f"Valid options: {', '.join(sorted(valid_levels))}"
            )
            raise typer.Exit(1)

        if log_level == "WARN":
            log_level = "WARNING"

    # Use centralized logging config
    setup_logging(
        command=command,
        cli_log_level=log_level,
        stream=sys.stderr,
        cli_debug_modules=debug_modules,
    )


@benchmark_app.command("list")
def list_benchmarks() -> None:
    """List available benchmarks and their descriptions."""
    from victor.evaluation.protocol import get_benchmark_catalog

    table = Table(title="Available Benchmarks")
    table.add_column("Benchmark", style="cyan")
    table.add_column("Description", style="white")
    table.add_column("Tasks", style="green")
    table.add_column("Source", style="dim")
    table.add_column("Status", style="yellow")

    for metadata in get_benchmark_catalog():
        task_count = str(metadata.total_tasks) if metadata.total_tasks > 0 else "TBD"
        source_name = metadata.source_name or "Custom"
        table.add_row(
            metadata.name,
            metadata.description,
            task_count,
            source_name,
            metadata.runner_status,
        )

    console.print(table)
    console.print("\n[dim]Run with: victor benchmark run <benchmark-name>[/]")


@benchmark_app.command("setup")
def setup_benchmark(
    benchmark: str = typer.Argument(..., help="Benchmark to setup: swe-bench, swe-bench-lite"),
    max_tasks: Optional[int] = typer.Option(
        None, "--max-tasks", "-n", help="Maximum number of repos to setup"
    ),
    force_reindex: bool = typer.Option(
        False, "--force", "-f", help="Force re-index even if already done"
    ),
    graph_rag: bool = typer.Option(
        False,
        "--graph-rag",
        help=(
            "Also build the Graph-RAG codegraph (CCG) per repo — the store "
            "the agent's marker-aware graph tooling reads; honors "
            ".victor/graph_backend (victor#907 slice 2)."
        ),
    ),
    log_level: Optional[str] = typer.Option(
        None, "--log-level", help="Logging level (DEBUG, INFO, WARNING, ERROR)"
    ),
) -> None:
    """Setup benchmark repos (clone + index). Run before 'run' for faster execution.

    This downloads and indexes the repositories needed for the benchmark.
    Indexes are cached in ~/.victor/swe_bench_cache/ and reused across runs.

    Example:
        victor benchmark setup swe-bench --max-tasks 5
        victor benchmark run swe-bench --max-tasks 5
    """
    _configure_log_level(log_level)

    # Only SWE-bench needs setup (repo cloning)
    benchmark_lower = benchmark.lower().replace("_", "-")
    if benchmark_lower not in ("swe-bench", "swe-bench-lite"):
        console.print(f"[yellow]Setup not needed for {benchmark}[/]")
        console.print("Only SWE-bench benchmarks require repo setup.")
        return
    run_sync(
        _setup_benchmark_async(
            benchmark=benchmark,
            benchmark_lower=benchmark_lower,
            max_tasks=max_tasks,
            force_reindex=force_reindex,
            # Typer replaces this with a bool on the CLI path.  Direct Python
            # callers that omit it receive the declaration's truthy
            # OptionInfo object, which must still mean the declared default.
            build_graph_rag=graph_rag is True,
        )
    )


async def _run_git_with_timeout(cmd, cwd, timeout=60):
    """Run a git command with timeout protection.

    Kills the process if it doesn't complete within timeout seconds.
    Prevents silent hangs from blocking the benchmark pipeline.
    """
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=cwd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        return proc.returncode, stdout, stderr
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        logger.warning("Git command timed out after %ds: %s", timeout, " ".join(cmd))
        raise


@benchmark_app.command("run")
def run_benchmark(
    benchmark: str = typer.Argument(
        ...,
        help="Benchmark to run: swe-bench, humaneval, mbpp, clawbench, guide, vlaa-gui",
    ),
    max_tasks: Optional[int] = typer.Option(
        None, "--max-tasks", "-n", help="Maximum number of tasks to run"
    ),
    start_task: int = typer.Option(
        0,
        "--start-task",
        help="Skip first N tasks (0-indexed, for targeting specific tasks)",
    ),
    shuffle: bool = typer.Option(
        False,
        "--shuffle",
        help="Shuffle task order for diverse repo sampling",
    ),
    frozen_prompts: bool = typer.Option(
        False,
        "--frozen-prompts",
        help=(
            "A/B hygiene: run on static baseline prompts (ignore evolved "
            "candidates) and skip post-run prompt evolution, so comparison "
            "arms see identical prompts."
        ),
    ),
    model: Optional[str] = typer.Option(
        None, "--model", "-m", help="Model to use (default: from profile)"
    ),
    profile: str = typer.Option("default", "--profile", "-p", help="Victor profile to use"),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output file for results (JSON)"
    ),
    dataset_path: Optional[Path] = typer.Option(
        None,
        "--dataset-path",
        help="Local JSON/JSONL dataset manifest for external benchmark adapters",
    ),
    timeout: int = typer.Option(420, "--timeout", "-t", help="Timeout per task in seconds"),
    max_turns: int = typer.Option(10, "--max-turns", help="Maximum conversation turns per task"),
    parallel: int = typer.Option(1, "--parallel", help="Number of parallel tasks"),
    use_docker: bool = typer.Option(
        False,
        "--docker",
        help="Run tests in Docker containers (requires per-repo images)",
    ),
    resume: bool = typer.Option(
        False,
        "--resume",
        "-r",
        help="Resume from checkpoint if previous run was interrupted",
    ),
    provider: Optional[str] = typer.Option(
        None, "--provider", help="Override provider (e.g., deepseek, openai, xai)"
    ),
    prompt_candidate_hash: Optional[str] = typer.Option(
        None,
        "--prompt-candidate-hash",
        help=(
            "Bind one exact prompt candidate hash into the live benchmark run and persist it "
            "in saved artifacts."
        ),
    ),
    prompt_section: Optional[str] = typer.Option(
        None,
        "--prompt-section",
        help=(
            "Bind one exact prompt section into the live benchmark run and persist it in "
            "saved artifacts."
        ),
    ),
    log_level: Optional[str] = typer.Option(
        None, "--log-level", help="Logging level (DEBUG, INFO, WARNING, ERROR)"
    ),
    debug_modules: Optional[str] = typer.Option(
        None,
        "--debug-modules",
        help="Comma-separated modules for DEBUG logging (e.g. code_search,agent_adapter)",
    ),
    no_edge_model: bool = typer.Option(
        False,
        "--no-edge-model",
        help="Disable edge model micro-decisions during benchmark",
    ),
    account: Optional[str] = typer.Option(
        None,
        "--account",
        "-a",
        help="Use a configured account (e.g., openai-oauth). Overrides --profile/--provider.",
    ),
    eval_backend: str = typer.Option(
        "local",
        "--eval-backend",
        help=(
            "Eval backend: 'local' (host subprocess, default) or 'docker' "
            "(each task runs in its correct runtime image — fixes Python "
            "version / C-ext / polyglot tasks)."
        ),
    ),
    runtime_version: Optional[str] = typer.Option(
        None,
        "--runtime-version",
        help="Runtime version for the docker image (e.g. 3.9, 20, 1.22). Informational for SWE-bench (image pins it).",
    ),
    docker_image_override: Optional[str] = typer.Option(
        None,
        "--docker-image",
        help="Override the docker image used for eval (skips per-task resolution).",
    ),
    swebench_image_source: str = typer.Option(
        "official",
        "--swebench-image-source",
        help="SWE-bench per-instance image source: official | build | skip (skip → host fallback).",
    ),
) -> None:
    """Run a benchmark evaluation."""
    if frozen_prompts:
        # Must be set before any agent/prompt machinery initializes.
        os.environ["VICTOR_FROZEN_PROMPTS"] = "1"
    _configure_log_level(log_level, debug_modules=debug_modules)

    # Disable edge model if requested
    if no_edge_model:
        from victor.core.feature_flags import get_feature_flag_manager, FeatureFlag

        get_feature_flag_manager().disable(FeatureFlag.USE_EDGE_MODEL)

    from victor.evaluation.protocol import EvaluationConfig

    _metadata, bench_type, runner = _resolve_benchmark_target(benchmark, dataset_path)
    provider, model, resolved_account = _resolve_account_selection(account, provider, model)
    effective_model = _resolve_effective_model(profile, model)

    # Build config — account for start_task offset in max_tasks
    effective_max_tasks = max_tasks
    # Resolve start_task — when called programmatically (not via Typer CLI),
    # the default may be a typer.OptionInfo rather than int.
    if not isinstance(start_task, int):
        start_task = 0
    if start_task > 0 and max_tasks is not None:
        effective_max_tasks = max_tasks + start_task

    config = EvaluationConfig(
        benchmark=bench_type,
        model=effective_model,
        provider=provider,
        prompt_candidate_hash=prompt_candidate_hash,
        prompt_section_name=prompt_section,
        max_tasks=effective_max_tasks,
        timeout_per_task=timeout,
        max_turns=max_turns,
        parallel_tasks=parallel,
        use_docker=use_docker,
        # Containerized / polyglot eval. `--docker` (use_docker) is a back-compat
        # alias for --eval-backend docker; honor whichever the user set.
        eval_backend=("docker" if use_docker else eval_backend),
        runtime_version=runtime_version,
        docker_image_override=docker_image_override,
        swebench_image_source=swebench_image_source,
    )
    _attach_manifest_metadata(config, runner)
    if benchmark.startswith("swe-bench") and config.eval_backend != "docker":
        # Host verification cannot reproduce the official harness: it depends on
        # the ambient environment and skips the per-instance runtime image.
        # --swebench-image-source only selects WHICH image once the docker
        # backend is on; it does not enable it.
        console.print(
            "[bold yellow]⚠ Host-fallback verification (eval backend: "
            f"{config.eval_backend}).[/bold yellow] SWE-bench pass rates from this "
            "mode are NOT comparable to the official harness and must not be "
            "published — run with [bold]--eval-backend docker[/bold] for "
            "in-container FAIL_TO_PASS verification."
        )
    _print_benchmark_header(
        title=f"Running {benchmark} benchmark",
        benchmark=benchmark,
        config=config,
        max_tasks=max_tasks,
        dataset_path=dataset_path,
        timeout=timeout,
    )
    result = run_sync(
        _run_benchmark_async(
            runner=runner,
            config=config,
            profile=profile,
            model=model,
            timeout=timeout,
            max_turns=max_turns,
            resume=resume,
            provider_override=provider,
            start_task=start_task,
            resolved_account=resolved_account,
        )
    )

    if result is None:
        raise typer.Exit(1)

    # Display results
    console.print("\n[bold green]Results[/]")

    metrics = result.get_metrics()
    results_table = Table()
    results_table.add_column("Metric", style="cyan")
    results_table.add_column("Value", style="white")

    results_table.add_row("Total Tasks", str(metrics["total_tasks"]))
    results_table.add_row("Passed", f"[green]{metrics['passed']}[/]")
    results_table.add_row("Failed", f"[red]{metrics['failed']}[/]")
    results_table.add_row("Errors", f"[yellow]{metrics['errors']}[/]")
    results_table.add_row("Pass Rate", f"[bold]{metrics['pass_rate']:.1%}[/]")
    results_table.add_row("Duration", f"{metrics['duration_seconds']:.1f}s")
    results_table.add_row("Total Tokens", f"{metrics['total_tokens']:,}")
    results_table.add_row("Tool Calls", f"{metrics['total_tool_calls']:,}")
    results_table.add_row("Code Search Calls", f"{metrics['total_code_search_calls']:,}")
    results_table.add_row("Graph Calls", f"{metrics['total_graph_calls']:,}")
    results_table.add_row(
        "Code Intel Coverage",
        f"{metrics['tasks_using_code_intelligence']}/{metrics['total_tasks']}"
        f" ({metrics['code_intelligence_task_coverage']:.1%})",
    )
    if metrics.get("failure_categories"):
        for category, count in sorted(metrics["failure_categories"].items()):
            results_table.add_row(f"Failure: {category}", str(count))

    # Extended token metrics (if available)
    cached = metrics.get("cached_tokens", 0)
    reasoning = metrics.get("reasoning_tokens", 0)
    cost_micros = metrics.get("cost_usd_micros", 0)
    if cached > 0:
        results_table.add_row("Cached Tokens", f"[dim]{cached:,}[/]")
    if reasoning > 0:
        results_table.add_row("Reasoning Tokens", f"[dim]{reasoning:,}[/]")
    if cost_micros > 0:
        cost_usd = cost_micros / 1_000_000
        results_table.add_row("API Cost", f"[dim]${cost_usd:.4f}[/]")

    console.print(results_table)

    code_intel_diagnostics = _summarize_code_intelligence_diagnostics(result)
    failure_examples = _summarize_failure_examples(result)
    failed_without_code_intel = code_intel_diagnostics["failed_task_ids_without_code_intelligence"]
    missing_code_intel = code_intel_diagnostics["task_ids_without_code_intelligence"]
    if failed_without_code_intel:
        console.print(
            "[yellow]Code intelligence missed on failed tasks:[/] "
            + ", ".join(failed_without_code_intel[:5])
        )
    elif missing_code_intel:
        console.print(
            "[dim]Tasks without code intelligence:[/] " + ", ".join(missing_code_intel[:5])
        )

    # Auto-evolve: run GEPA evolution after benchmark completes
    # This is the "post-session hook" from Memory Scaling — the agent
    # improves its prompts automatically after each benchmark run.
    if os.environ.get("VICTOR_FROZEN_PROMPTS"):
        logger.debug("Auto-evolve skipped: frozen prompts requested")
    elif metrics["total_tasks"] >= 5:  # Only evolve with enough data
        try:
            from victor.framework.rl.coordinator import get_rl_coordinator

            coordinator = get_rl_coordinator()
            learner = coordinator.get_learner("prompt_optimizer")
            if learner:
                evolved = _auto_evolve_prompt_candidate(
                    learner,
                    model_name=config.model or "",
                    metrics=metrics,
                )
                if evolved:
                    candidate, provider = evolved
                    console.print(
                        f"\n[dim]Auto-evolved prompt gen-{candidate.generation} "
                        f"for {provider} (mean={candidate.mean:.2f})[/]"
                    )
        except Exception as e:
            logger.debug("Auto-evolve skipped: %s", e)

    # Save results if output specified
    if output:
        artifact_config = config.to_artifact_config()
        output_data = {
            "benchmark": benchmark,
            "model": config.model,
            "provider": config.provider,
            "prompt_candidate_hash": config.prompt_candidate_hash,
            "section_name": config.prompt_section_name,
            "prompt_section_name": config.prompt_section_name,
            "config": artifact_config,
            "timestamp": datetime.now().isoformat(),
            "dataset_metadata": config.dataset_metadata,
            "metrics": metrics,
            "diagnostics": code_intel_diagnostics,
            "failure_examples": failure_examples,
            "task_results": [
                {
                    "task_id": r.task_id,
                    "status": r.status.value,
                    "tests_passed": r.tests_passed,
                    "tests_total": r.tests_total,
                    "duration": r.duration_seconds,
                    "tool_calls": r.tool_calls,
                    "code_search_calls": r.code_search_calls,
                    "graph_calls": r.graph_calls,
                    "failure_category": (r.failure_category.value if r.failure_category else None),
                    "failure_details": r.failure_details,
                }
                for r in result.task_results
            ],
        }
        output.write_text(json.dumps(output_data, indent=2))
        console.print(f"\n[dim]Results saved to {output}[/]")


async def _ensure_canonical_services(
    config,
    *,
    profile: str,
    vertical_hint: Optional[str],
) -> None:
    """Register ChatService and friends, if this process has not already.

    Building the agent is what populates the DI container with the canonical
    service graph. Skipped when ChatService is already registered, so an
    embedder that wired its own services keeps them.
    """
    try:
        from victor.agent.services.protocols import ChatServiceProtocol
        from victor.core import get_container

        if get_container().get_optional(ChatServiceProtocol) is not None:
            return
    except Exception:
        pass  # fall through and try to build it

    from victor.evaluation.agent_adapter import VictorAgentAdapter
    from victor.framework.session_config import SessionConfig

    session_config = SessionConfig.from_cli_flags(
        agent_profile=profile,
        provider_timeout=config.timeout_per_task,
    )
    await VictorAgentAdapter.create_from_session_config(
        session_config,
        profile=profile,
        vertical=vertical_hint,
        enable_observability=False,
    )


async def _run_real_benchmark_async(
    *,
    runner,
    config,
    output_dir: Optional[Path],
    resume: bool,
    profile: str,
    vertical_hint: Optional[str] = None,
):
    """Run a benchmark through the service-first ChatService real-run path."""
    from victor.evaluation.benchmarks.framework_comparison import Framework
    from victor.evaluation.real_run_runner import RealRunBenchmarkRunner, RealRunConfig

    # ChatService is registered as a side effect of building the agent: the
    # container only gains it once the conversation controller and streaming
    # coordinator exist, and nothing but agent construction creates those. A CLI
    # process that skipped this found no ChatService, and the runner used to
    # swallow that and return an empty answer per task — recording a full
    # artifact with 0 tool calls, 0 tokens and a test failure caused by the
    # empty answer, indistinguishable from genuine poor performance.
    #
    # The agent is built for its service graph only; execution still goes
    # through ChatService directly, which is the point of this path.
    await _ensure_canonical_services(config, profile=profile, vertical_hint=vertical_hint)

    real_runner = RealRunBenchmarkRunner(
        RealRunConfig(
            framework=Framework.VICTOR,
            model=config.model,
            benchmark=config.benchmark,
            max_tasks=config.max_tasks,
            timeout_per_task=config.timeout_per_task,
            parallel_tasks=config.parallel_tasks,
            output_dir=output_dir,
        )
    )
    eval_result, framework_result = await real_runner.execute_real_run(
        config,
        resume=resume,
        benchmark_runner=runner,
    )

    # The service-first path must emit the same joinable per-task artifact as
    # ``benchmark run``.  Without it, a successful real run cannot feed trace
    # mining or the independently labelled EVR-6 review-pack workflow.
    # Manifest persistence remains best-effort so an artifact failure never
    # invalidates the benchmark result itself.
    try:
        from victor.evaluation.manifest import emit_execution_manifest

        manifest_path = emit_execution_manifest(eval_result)
        if manifest_path is not None:
            console.print(f"[dim]Execution manifest: {manifest_path}[/]")
    except Exception as exc:  # never break the benchmark
        logger.debug("Execution manifest emission skipped: %s", exc)

    return eval_result, framework_result


@benchmark_app.command("run-real")
def run_real_benchmark(
    benchmark: str = typer.Argument(
        ...,
        help="Benchmark to run through the service-first ChatService path",
    ),
    max_tasks: Optional[int] = typer.Option(
        None, "--max-tasks", "-n", help="Maximum number of tasks to run"
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="Model to record for the run (default: from profile)",
    ),
    profile: str = typer.Option(
        "default", "--profile", "-p", help="Victor profile to resolve model"
    ),
    provider: Optional[str] = typer.Option(
        None, "--provider", help="Provider metadata to record for the run"
    ),
    dataset_path: Optional[Path] = typer.Option(
        None,
        "--dataset-path",
        help="Local JSON/JSONL dataset manifest for external benchmark adapters",
    ),
    timeout: int = typer.Option(420, "--timeout", "-t", help="Timeout per task in seconds"),
    max_turns: int = typer.Option(10, "--max-turns", help="Maximum conversation turns per task"),
    parallel: int = typer.Option(1, "--parallel", help="Number of parallel tasks"),
    use_docker: bool = typer.Option(
        False,
        "--docker",
        help="Run tests in Docker containers (requires per-repo images)",
    ),
    resume: bool = typer.Option(
        False,
        "--resume",
        "-r",
        help="Resume from checkpoint if previous run was interrupted",
    ),
    bundle_output: Optional[Path] = typer.Option(
        None,
        "--bundle-output",
        help="Optional stable real-run publication bundle output directory",
    ),
    log_level: Optional[str] = typer.Option(
        None, "--log-level", help="Logging level (DEBUG, INFO, WARNING, ERROR)"
    ),
) -> None:
    """Run a benchmark through the canonical service-first ChatService runner.

    This command is intentionally narrower than `benchmark run`: it exposes the
    workflow-level real-run path used for stable benchmark publication without
    routing through AgentOrchestrator adapter code.
    """
    _configure_log_level(log_level)

    from victor.evaluation.protocol import EvaluationConfig

    _metadata, bench_type, runner = _resolve_benchmark_target(benchmark, dataset_path)
    effective_model = _resolve_effective_model(profile, model)
    config = EvaluationConfig(
        benchmark=bench_type,
        model=effective_model,
        provider=provider,
        max_tasks=max_tasks,
        timeout_per_task=timeout,
        max_turns=max_turns,
        parallel_tasks=parallel,
    )
    _attach_manifest_metadata(config, runner)
    _print_benchmark_header(
        title=f"Running {benchmark} real benchmark",
        benchmark=benchmark,
        config=config,
        max_tasks=max_tasks,
        dataset_path=dataset_path,
        timeout=timeout,
    )

    try:
        eval_result, framework_result = run_sync(
            _run_real_benchmark_async(
                runner=runner,
                config=config,
                output_dir=bundle_output,
                resume=resume,
                profile=profile,
                vertical_hint=_BENCHMARK_VERTICAL.get(config.benchmark),
            )
        )
    except Exception as exc:
        console.print(f"[bold red]Error:[/] Real benchmark run failed: {exc}")
        raise typer.Exit(1)

    metrics = eval_result.get_metrics()
    console.print("\n[bold green]Real Run Results[/]")
    console.print(f"Tasks: {metrics['total_tasks']}")
    console.print(f"Pass Rate: {metrics['pass_rate']:.1%}")
    console.print(f"Total Tokens: {metrics['total_tokens']:,}")
    if bundle_output is not None:
        console.print(f"[dim]Stable real-run bundle output: {bundle_output}[/]")
    console.print(f"[dim]Framework result: {framework_result.framework.value}[/]")


@benchmark_app.command("run-prompt-suite")
def run_prompt_suite(
    benchmark: str = typer.Argument(
        ...,
        help="Benchmark to run: swe-bench, humaneval, mbpp, clawbench, guide, vlaa-gui",
    ),
    prompt_section: str = typer.Option(
        ...,
        "--prompt-section",
        help="Prompt section shared by all candidates in this suite.",
    ),
    candidate_hashes: list[str] = typer.Option(
        ...,
        "--candidate-hash",
        help="Prompt candidate hash to evaluate. Repeat once per candidate.",
    ),
    include_baseline: bool = typer.Option(
        False,
        "--include-baseline",
        help="Also run the section's SEED (baseline) prompt so the suite measures "
        "evolved-vs-seed. Auto-enabled when --create-rollout is set.",
    ),
    max_verify_retries: Optional[int] = typer.Option(
        None,
        "--max-verify-retries",
        help="Closed-loop verify-and-retry: after the agent claims done, verify its "
        "patch in-container and re-enter on failure (≤N times). Default 2 when "
        "--docker is set, else 0 (off). Essential for C-extension repos the host "
        "can't build (e.g. astropy).",
    ),
    max_tasks: Optional[int] = typer.Option(
        None, "--max-tasks", "-n", help="Maximum number of tasks to run"
    ),
    start_task: int = typer.Option(
        0,
        "--start-task",
        help="Skip first N tasks (0-indexed, for targeting specific tasks)",
    ),
    shuffle: bool = typer.Option(
        False,
        "--shuffle",
        help="Shuffle task order for diverse repo sampling",
    ),
    model: Optional[str] = typer.Option(
        None, "--model", "-m", help="Model to use (default: from profile)"
    ),
    profile: str = typer.Option("default", "--profile", "-p", help="Victor profile to use"),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output file for suite summary (JSON)"
    ),
    dataset_path: Optional[Path] = typer.Option(
        None,
        "--dataset-path",
        help="Local JSON/JSONL dataset manifest for external benchmark adapters",
    ),
    timeout: int = typer.Option(420, "--timeout", "-t", help="Timeout per task in seconds"),
    max_turns: int = typer.Option(10, "--max-turns", help="Maximum conversation turns per task"),
    parallel: int = typer.Option(1, "--parallel", help="Number of parallel tasks"),
    use_docker: bool = typer.Option(
        False,
        "--docker",
        help="Run tests in Docker containers (requires per-repo images)",
    ),
    resume: bool = typer.Option(
        False,
        "--resume",
        "-r",
        help="Resume from checkpoint if previous run was interrupted",
    ),
    provider: Optional[str] = typer.Option(
        None, "--provider", help="Override provider (e.g., deepseek, openai, xai)"
    ),
    log_level: Optional[str] = typer.Option(
        None, "--log-level", help="Logging level (DEBUG, INFO, WARNING, ERROR)"
    ),
    debug_modules: Optional[str] = typer.Option(
        None,
        "--debug-modules",
        help="Comma-separated modules for DEBUG logging (e.g. code_search,agent_adapter)",
    ),
    no_edge_model: bool = typer.Option(
        False,
        "--no-edge-model",
        help="Disable edge model micro-decisions during benchmark",
    ),
    account: Optional[str] = typer.Option(
        None,
        "--account",
        "-a",
        help="Use a configured account (e.g., openai-oauth). Overrides --profile/--provider.",
    ),
    record_benchmark_results: bool = typer.Option(
        False,
        "--record-benchmark-results",
        help="Record suite pass-rate evidence back into prompt-optimizer benchmark state.",
    ),
    promote_best: bool = typer.Option(
        False,
        "--promote-best",
        help="Promote the best benchmark-approved candidate after recording suite results.",
    ),
    create_rollout: bool = typer.Option(
        False,
        "--create-rollout",
        help="Create a prompt rollout experiment for the benchmark-approved suite winner.",
    ),
    rollout_control_hash: Optional[str] = typer.Option(
        None,
        "--rollout-control-hash",
        help="Optional control candidate hash to use for the rollout experiment.",
    ),
    rollout_traffic_split: float = typer.Option(
        0.1,
        "--rollout-traffic-split",
        help="Traffic split to use when creating a prompt rollout experiment.",
    ),
    rollout_min_samples_per_variant: int = typer.Option(
        100,
        "--rollout-min-samples-per-variant",
        help="Minimum samples per variant for the created prompt rollout experiment.",
    ),
    analyze_rollout: bool = typer.Option(
        False,
        "--analyze-rollout",
        help="Analyze the approved winner's prompt rollout experiment, if one exists.",
    ),
    apply_rollout_decision: bool = typer.Option(
        False,
        "--apply-rollout-decision",
        help="Apply the rollout analysis recommendation when it is clearly actionable.",
    ),
    rollout_decision_dry_run: bool = typer.Option(
        False,
        "--rollout-decision-dry-run",
        help="Report the rollout decision that would be applied without changing experiment state.",
    ),
    min_approval_pass_rate: float = typer.Option(
        0.5,
        "--min-approval-pass-rate",
        min=0.0,
        max=1.0,
        help="Minimum pass rate the suite winner must reach before benchmark approval.",
    ),
    ci_gate: bool = typer.Option(
        False,
        "--ci-gate",
        help=(
            "Exit non-zero unless the suite is GREEN (valid measurement + cleared the "
            "comparative gate). For CI: a classified red carries its routing reason "
            "(re-run / re-run-with-quota / fix-infra / re-evolve) instead of an ambiguous "
            "failure. Requires --record-benchmark-results."
        ),
    ),
) -> None:
    """Run one benchmark evaluation per prompt candidate and drive rollout decisions safely."""
    _configure_log_level(log_level, debug_modules=debug_modules)

    if ci_gate and not record_benchmark_results:
        console.print("[bold red]Error:[/] --ci-gate requires --record-benchmark-results")
        raise typer.Exit(1)
    if promote_best and not record_benchmark_results:
        console.print("[bold red]Error:[/] --promote-best requires --record-benchmark-results")
        raise typer.Exit(1)
    if create_rollout and not record_benchmark_results:
        console.print("[bold red]Error:[/] --create-rollout requires --record-benchmark-results")
        raise typer.Exit(1)
    if create_rollout and promote_best:
        console.print("[bold red]Error:[/] --create-rollout cannot be combined with --promote-best")
        raise typer.Exit(1)
    if analyze_rollout and not record_benchmark_results:
        console.print("[bold red]Error:[/] --analyze-rollout requires --record-benchmark-results")
        raise typer.Exit(1)
    if apply_rollout_decision and not analyze_rollout:
        console.print("[bold red]Error:[/] --apply-rollout-decision requires --analyze-rollout")
        raise typer.Exit(1)
    if apply_rollout_decision and promote_best:
        console.print(
            "[bold red]Error:[/] --apply-rollout-decision cannot be combined with --promote-best"
        )
        raise typer.Exit(1)
    if create_rollout and not 0.0 < rollout_traffic_split < 1.0:
        console.print("[bold red]Error:[/] --rollout-traffic-split must be between 0 and 1")
        raise typer.Exit(1)
    if create_rollout and rollout_min_samples_per_variant <= 0:
        console.print(
            "[bold red]Error:[/] --rollout-min-samples-per-variant must be greater than 0"
        )
        raise typer.Exit(1)

    if no_edge_model:
        from victor.core.feature_flags import get_feature_flag_manager, FeatureFlag

        get_feature_flag_manager().disable(FeatureFlag.USE_EDGE_MODEL)

    from victor.evaluation import EvaluationConfig, PromptCandidateEvaluationSpec

    from victor.agent.optimization_injector import BASELINE_CANDIDATE_HASH

    _metadata, bench_type, runner = _resolve_benchmark_target(benchmark, dataset_path)
    provider, model, resolved_account = _resolve_account_selection(account, provider, model)
    effective_model = _resolve_effective_model(profile, model)

    effective_max_tasks = max_tasks
    if not isinstance(start_task, int):
        start_task = 0
    if start_task > 0 and max_tasks is not None:
        effective_max_tasks = max_tasks + start_task

    # Verify-and-retry: default ON for docker SWE-bench (the agent can't verify
    # C-extension repos on the host; the closed-loop container verify lets it
    # iterate against the real env). Explicit --max-verify-retries overrides.
    effective_verify_retries = (
        max_verify_retries if max_verify_retries is not None else (2 if use_docker else 0)
    )

    base_config = EvaluationConfig(
        benchmark=bench_type,
        model=effective_model,
        provider=provider,
        max_tasks=effective_max_tasks,
        timeout_per_task=timeout,
        max_turns=max_turns,
        parallel_tasks=parallel,
        use_docker=use_docker,
        eval_backend=("docker" if use_docker else "local"),
        max_verify_retries=effective_verify_retries,
    )
    _attach_manifest_metadata(base_config, runner)

    _print_benchmark_header(
        title=f"Running {benchmark} prompt candidate suite",
        benchmark=benchmark,
        config=base_config,
        max_tasks=max_tasks,
        dataset_path=dataset_path,
        timeout=timeout,
    )
    console.print(f"Prompt Section: {prompt_section}")
    console.print(f"Candidate Count: {len(candidate_hashes)}")
    for candidate_hash in candidate_hashes:
        console.print(f"  - {candidate_hash}")
    console.print()

    candidate_specs = []
    # Baseline (seed) arm: auto-included with --create-rollout (a rollout without
    # a measured baseline is meaningless) or when --include-baseline is set.
    # Resolved by the injector to the section's seed text (no stored candidate).
    if include_baseline or create_rollout:
        candidate_specs.append(
            PromptCandidateEvaluationSpec(
                section_name=prompt_section,
                prompt_candidate_hash=BASELINE_CANDIDATE_HASH,
                provider=provider,
                label=f"{prompt_section}:baseline (seed)",
            )
        )
    candidate_specs.extend(
        PromptCandidateEvaluationSpec(
            section_name=prompt_section,
            prompt_candidate_hash=candidate_hash,
            provider=provider,
        )
        for candidate_hash in candidate_hashes
    )

    suite = run_sync(
        _run_prompt_candidate_suite_async(
            runner=runner,
            base_config=base_config,
            candidate_specs=candidate_specs,
            profile=profile,
            model=model,
            timeout=timeout,
            max_turns=max_turns,
            resume=resume,
            provider_override=provider,
            start_task=start_task,
            resolved_account=resolved_account,
        )
    )

    if suite is None:
        raise typer.Exit(1)

    _print_prompt_candidate_suite_summary(suite)
    prompt_optimizer_sync = None
    prompt_rollout: Optional[dict[str, Any]] = None
    prompt_rollout_analysis: Optional[dict[str, Any]] = None
    prompt_rollout_decision: Optional[dict[str, Any]] = None
    if record_benchmark_results:
        from victor.framework.rl import process_prompt_candidate_evaluation_suite

        try:
            workflow = process_prompt_candidate_evaluation_suite(
                suite,
                min_pass_rate=min_approval_pass_rate,
                promote_best=promote_best,
                create_rollout=create_rollout,
                rollout_control_hash=rollout_control_hash,
                rollout_traffic_split=rollout_traffic_split,
                rollout_min_samples_per_variant=rollout_min_samples_per_variant,
                analyze_rollout=analyze_rollout,
                apply_rollout_decision=apply_rollout_decision,
                rollout_decision_dry_run=rollout_decision_dry_run,
            )
        except ValueError as exc:
            console.print(f"[bold red]Error:[/] {exc}")
            raise typer.Exit(1) from exc
        prompt_optimizer_sync = workflow.prompt_optimizer_sync
        prompt_rollout = workflow.prompt_rollout
        prompt_rollout_analysis = workflow.prompt_rollout_analysis
        prompt_rollout_decision = workflow.prompt_rollout_decision
        _print_prompt_optimizer_sync_summary(prompt_optimizer_sync)

        # Classified gate signal: validity is asserted before any statistical
        # claim, so green/red is trustworthy enough to automate. See
        # victor.evaluation.suite_verdict.
        from victor.evaluation.suite_verdict import SuiteVerdict, verdict_from_sync

        suite_verdict, suite_verdict_reason = verdict_from_sync(
            suite, prompt_optimizer_sync, min_tasks=max_tasks or 0
        )
        _print_suite_verdict(suite_verdict, suite_verdict_reason)
        if ci_gate and suite_verdict is not SuiteVerdict.GREEN:
            raise typer.Exit(1)
        if prompt_rollout is not None:
            if prompt_rollout.get("created"):
                console.print(
                    "[bold green]Prompt rollout experiment started:[/] "
                    f"{prompt_rollout.get('experiment_id')}"
                )
            else:
                console.print(
                    "[yellow]Prompt rollout not created:[/] "
                    f"{prompt_rollout.get('error', 'unable to start prompt rollout experiment')}"
                )
        if prompt_rollout_analysis is not None:
            if prompt_rollout_analysis.get("analysis_available", False):
                _print_prompt_rollout_analysis_summary(prompt_rollout_analysis)
            else:
                console.print(
                    "[yellow]Prompt rollout analysis unavailable:[/] "
                    f"{prompt_rollout_analysis.get('recommendation', 'analysis unavailable')}"
                )
        if prompt_rollout_decision is not None:
            action = prompt_rollout_decision.get("action")
            if action and prompt_rollout_decision.get("applied"):
                console.print(f"[bold green]Prompt rollout decision applied:[/] {action}")
            elif action and prompt_rollout_decision.get("dry_run"):
                console.print(f"[cyan]Prompt rollout decision dry-run:[/] would apply {action}")
            else:
                console.print(
                    "[yellow]Prompt rollout decision not applied:[/] "
                    f"{prompt_rollout_decision.get('reason') or prompt_rollout_decision.get('error') or 'no actionable recommendation'}"
                )

    if output:
        output_data = _serialize_prompt_candidate_suite(
            benchmark,
            prompt_section,
            suite,
            prompt_optimizer_sync=prompt_optimizer_sync,
            prompt_rollout=prompt_rollout,
            prompt_rollout_analysis=prompt_rollout_analysis,
            prompt_rollout_decision=prompt_rollout_decision,
        )
        output_data["timestamp"] = datetime.now().isoformat()
        output.write_text(json.dumps(output_data, indent=2))
        console.print(f"\n[dim]Suite summary saved to {output}[/]")


async def _setup_benchmark_async(
    *,
    benchmark: str,
    benchmark_lower: str,
    max_tasks: Optional[int],
    force_reindex: bool,
    build_graph_rag: bool = False,
) -> None:
    from victor.evaluation.protocol import BenchmarkType, EvaluationConfig
    from victor.evaluation.benchmarks import SWEBenchRunner
    from victor.evaluation.swe_bench_loader import SWEBenchWorkspaceManager

    runner = (
        SWEBenchRunner(split="lite") if benchmark_lower == "swe-bench-lite" else SWEBenchRunner()
    )
    config = EvaluationConfig(
        benchmark=BenchmarkType.SWE_BENCH,
        model="setup-only",
        max_tasks=max_tasks,
    )

    console.print(f"[bold]Setting up {benchmark} benchmark...[/]")
    tasks = await runner.load_tasks(config)
    console.print(f"Found {len(tasks)} tasks to setup")

    workspace_manager = SWEBenchWorkspaceManager()
    repos_seen = set()
    unique_tasks = []
    for task in tasks:
        if task.repo and task.repo not in repos_seen:
            repos_seen.add(task.repo)
            unique_tasks.append(task)

    console.print(f"Unique repositories: {len(unique_tasks)}")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        setup_task = progress.add_task("Setting up repos...", total=len(unique_tasks))

        for i, task in enumerate(unique_tasks):
            repo_name = task.repo.split("/")[-1].replace(".git", "") if task.repo else task.task_id
            progress.update(setup_task, description=f"[{i+1}/{len(unique_tasks)}] {repo_name}")

            try:
                if (
                    not force_reindex
                    and workspace_manager.is_repo_indexed(task)
                    and not build_graph_rag
                ):
                    console.print(f"  [green]✓[/] {repo_name} (cached)")
                else:
                    await workspace_manager.setup_repo_with_indexes(
                        task,
                        force_reindex=force_reindex,
                        build_graph_rag=build_graph_rag,
                    )
                    console.print(f"  [green]✓[/] {repo_name} (indexed)")
            except Exception as e:
                console.print(f"  [red]✗[/] {repo_name}: {e}")

            progress.advance(setup_task)

    console.print("\n[bold green]Setup complete![/]")
    console.print(f"Cached repos: {workspace_manager.cache_dir}")
    console.print(f"\nNow run: victor benchmark run {benchmark}")


def _record_benchmark_task_outcome(_tr: Any) -> None:
    """Stamp one benchmark task's outcome into the decision_outcome junction.

    FEP-0012 Gap A fix: called from the per-task ``on_progress`` callback (so a
    ctrl+c / crash loses only the in-flight task, never completed ones) AND from
    the post-run safety-net loop (idempotent backstop). ``record_session_outcome``
    is DELETE-then-INSERT per session_id, so calling it twice is a no-op.

    Args:
        _tr: A TaskResult-like object exposing ``session_id``, ``tests_total``,
            and ``tests_passed``. Tasks without a session_id are skipped.
    """
    _sid = getattr(_tr, "session_id", "") or ""
    if not _sid:
        return
    _total = int(getattr(_tr, "tests_total", 0) or 0)
    _passed = int(getattr(_tr, "tests_passed", 0) or 0)
    _rate = (_passed / _total) if _total else 0.0
    from victor.agent.decisions.outcome import record_session_outcome

    record_session_outcome(
        _sid,
        success=(_passed == _total and _total > 0),
        quality_score=_rate,
    )


async def _run_benchmark_async(
    *,
    runner,
    config,
    profile: str,
    model: Optional[str],
    timeout: int,
    max_turns: int = 10,
    resume: bool,
    provider_override: Optional[str] = None,
    start_task: int = 0,
    resolved_account=None,
):
    from victor.evaluation.harness import EvaluationHarness
    from victor.evaluation.agent_adapter import VictorAgentAdapter
    from victor.evaluation.protocol import BenchmarkTask
    from victor.evaluation.swe_bench_loader import SWEBenchWorkspaceManager

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Loading tasks...", total=None)

        harness = EvaluationHarness()
        harness.register_runner(runner)

        tasks = await runner.load_tasks(config)

        # Skip tasks for targeted execution (--start-task)
        if start_task > 0 and start_task < len(tasks):
            tasks = tasks[start_task:]
            console.print(f"[dim]Skipped {start_task} tasks (starting from index {start_task})[/]")
            # Override runner.load_tasks to return filtered set
            # (harness calls load_tasks internally)
            _original_load = runner.load_tasks

            async def _filtered_load(cfg):
                return tasks

            runner.load_tasks = _filtered_load

        progress.update(task, description=f"Loaded {len(tasks)} tasks")

        if not tasks:
            console.print("[yellow]No tasks to run[/]")
            return None

        progress.update(task, description="Initializing agent...")
        try:
            from victor.evaluation.agent_adapter import (
                AdapterConfig,
                PromptOptimizationBinding,
            )
            from victor.framework.session_config import (
                DEFAULT_PROVIDER_MODELS,
                SessionConfig,
            )

            adapter_config = AdapterConfig(
                total_timeout=timeout,
                max_turns=max_turns,
                min_turn_timeout=max(240, timeout // max(max_turns, 1)),
                max_verify_retries=getattr(config, "max_verify_retries", 0),
                prompt_binding=(
                    PromptOptimizationBinding(
                        section_name=config.prompt_section_name,
                        prompt_candidate_hash=config.prompt_candidate_hash,
                    )
                    if config.prompt_candidate_hash and config.prompt_section_name
                    else None
                ),
            )
            # Bootstrap edge model decision service into the container
            # so tool selection, prompt focus, and stage detection can use it.
            from victor.core.feature_flags import get_feature_flag_manager, FeatureFlag

            _edge_enabled = get_feature_flag_manager().is_enabled(FeatureFlag.USE_EDGE_MODEL)
            if _edge_enabled:
                try:
                    from victor.agent.edge_model import (
                        EdgeModelConfig,
                        create_edge_decision_service,
                    )
                    from victor.core import get_container
                    from victor.agent.services.protocols.decision_service import (
                        LLMDecisionServiceProtocol,
                    )
                    from victor.core.container import ServiceLifetime

                    edge_service = create_edge_decision_service(EdgeModelConfig())
                    if edge_service:
                        container = get_container()
                        container.register(
                            LLMDecisionServiceProtocol,
                            lambda c: edge_service,
                            ServiceLifetime.SINGLETON,
                        )
                        console.print("[dim]Edge model enabled for micro-decisions[/]")
                except Exception as e:
                    console.print(f"[dim]Edge model unavailable: {e}[/]")
            else:
                console.print("[dim]Edge model disabled (--no-edge-model)[/]")

            # Benchmarks need their vertical's capabilities registered
            # (code_search/graph for coding) — Agent.create(vertical=...) via
            # AgentFactory does this. Previously the profile-only path called
            # from_profile (direct Orchestrator() construction), which bypassed
            # capability discovery, so the tools never registered and the
            # benchmark failed its readiness check. Both paths now go through
            # create_from_session_config (Agent.create).
            vertical = _BENCHMARK_VERTICAL.get(config.benchmark)
            if provider_override:
                effective_model = (
                    model
                    or DEFAULT_PROVIDER_MODELS.get(provider_override)
                    or _resolve_effective_model(profile, None)
                )
                session_config = SessionConfig.from_cli_flags(
                    agent_profile=profile,
                    provider=provider_override,
                    model=effective_model,
                    auth_mode=(
                        "oauth"
                        if resolved_account and resolved_account.auth.method == "oauth"
                        else None
                    ),
                    provider_timeout=timeout,
                )
                adapter = await VictorAgentAdapter.create_from_session_config(
                    session_config,
                    profile=profile,
                    vertical=vertical,
                    config=adapter_config,
                    enable_observability=False,
                )
            else:
                session_config = SessionConfig.from_cli_flags(
                    agent_profile=profile,
                    provider_timeout=timeout,
                )
                adapter = await VictorAgentAdapter.create_from_session_config(
                    session_config,
                    profile=profile,
                    vertical=vertical,
                    config=adapter_config,
                    enable_observability=False,
                )

            actual_provider = getattr(adapter.orchestrator, "provider_name", None) or getattr(
                getattr(adapter.orchestrator, "provider", None), "name", None
            )
            if actual_provider:
                config.provider = actual_provider
            readiness = _ensure_benchmark_runtime_tools(adapter)
            console.print(
                "  [dim]Benchmark tools ready: " + ", ".join(readiness.enabled_tools) + "[/]"
            )
            workspace_manager = SWEBenchWorkspaceManager()

            _warmed_repos: Dict[str, CodeIntelligencePrewarmResult] = {}

            async def agent_callback(benchmark_task: BenchmarkTask) -> dict:
                """Run agent on task and return generated code with metrics."""
                temp_work_dir = None
                cached_repo = workspace_manager.get_cached_repo_path(benchmark_task)
                if benchmark_task.repo:
                    if cached_repo and workspace_manager.is_repo_indexed(benchmark_task):
                        work_dir = cached_repo
                        console.print(f"  [dim]Using indexed repo: {cached_repo.name}[/]")
                    else:
                        console.print(
                            "  [dim]Setting up repo (run 'victor benchmark setup' for faster execution)...[/]"
                        )
                        await workspace_manager.setup_repo_with_indexes(benchmark_task)
                        work_dir = workspace_manager.get_cached_repo_path(benchmark_task)
                else:
                    import tempfile

                    temp_work_dir = Path(tempfile.mkdtemp(prefix="benchmark_task_"))
                    if benchmark_task.context_code:
                        (temp_work_dir / "solution.py").write_text(benchmark_task.context_code)
                    if benchmark_task.test_code:
                        (temp_work_dir / "test_solution.py").write_text(benchmark_task.test_code)
                    _stage_benchmark_input_files(benchmark_task, temp_work_dir)
                    work_dir = temp_work_dir
                    console.print(
                        f"  [dim]Using ephemeral workspace for {benchmark_task.task_id}[/]"
                    )

                if benchmark_task.base_commit and work_dir:
                    # Reset workspace to base_commit with clean working tree.
                    # Fetch the specific commit first (may be outside shallow depth).
                    for git_cmd in [
                        [
                            "git",
                            "fetch",
                            "--depth",
                            "1",
                            "origin",
                            benchmark_task.base_commit,
                        ],
                        ["git", "checkout", "--force", benchmark_task.base_commit],
                        ["git", "clean", "-fd", "-e", ".victor"],
                    ]:
                        try:
                            await _run_git_with_timeout(git_cmd, work_dir, timeout=60)
                        except asyncio.TimeoutError:
                            console.print(f"  [yellow]Git command timed out: {git_cmd[1]}[/]")

                # Pre-warm code_search/graph index BEFORE task timer starts.
                # This is a fixed cost — NOT deducted from the per-task timeout.
                prewarm_result = await _prewarm_code_intelligence_index(
                    work_dir,
                    _warmed_repos,
                    timeout=300.0,
                )
                if prewarm_result.message:
                    console.print(prewarm_result.message)

                original_cwd = os.getcwd()
                os.chdir(work_dir)
                try:
                    # FEP-0018 Phase 2: the framework verification hook on the
                    # agentic loop replaces the adapter's eval-specific verify_fn.
                    # The loop's DECIDE gate now handles verify-and-retry.
                    _persistent_container = None
                    if (
                        getattr(config, "max_verify_retries", 0) > 0
                        and getattr(config, "eval_backend", None) == "docker"
                        and getattr(benchmark_task, "repo", None)
                    ):
                        from victor.evaluation.benchmarks.swe_bench import (
                            ContainerTestVerifier,
                            SWEBenchRunner,
                        )
                        from victor.evaluation.container_eval import (
                            EvalContainer,
                            cleanup_stale_sandbox_containers,
                            resolve_swebench_image_exact,
                        )

                        # Sweep orphaned sandbox containers from prior crashed runs
                        # BEFORE starting the persistent one.
                        if not getattr(SWEBenchRunner, "_stale_containers_cleaned", False):
                            SWEBenchRunner._stale_containers_cleaned = True
                            try:
                                await cleanup_stale_sandbox_containers()
                            except Exception:
                                pass

                        # Start ONE persistent container for this task.
                        try:
                            _image = await resolve_swebench_image_exact(benchmark_task, config)
                            logger.info(
                                "Starting persistent verify container: image=%s",
                                _image,
                            )
                            _persistent_container = EvalContainer(
                                image=_image, workspace_host_path=str(work_dir)
                            )
                            await _persistent_container.start()
                            logger.info(
                                "Persistent verify container started: %s",
                                _persistent_container.name,
                            )
                        except Exception as exc:
                            logger.warning(
                                "Persistent verify container unavailable "
                                "(falling back to per-call creation): %s",
                                exc,
                            )
                            _persistent_container = None

                        # FEP-0018: set the verifier on the turn_executor —
                        # the loop's DECIDE gate handles verify-and-retry.
                        _turn_exec = getattr(adapter.orchestrator, "turn_executor", None)
                        if _turn_exec is not None:
                            _turn_exec._verifier = ContainerTestVerifier(
                                benchmark_task,
                                work_dir,
                                config,
                                container=_persistent_container,
                            )
                            _turn_exec._max_verify_retries = getattr(
                                config, "max_verify_retries", 0
                            )

                    trace = await adapter.execute_task(
                        benchmark_task, work_dir  # loop handles verification (FEP-0018)
                    )

                    # Unset the verifier after the task (avoid leaking to next task)
                    _turn_exec = getattr(adapter.orchestrator, "turn_executor", None)
                    if _turn_exec is not None:
                        _turn_exec._verifier = None
                except asyncio.CancelledError:
                    partial = adapter.get_partial_trace()
                    logger.info(
                        "Task cancelled - partial trace: tool_calls=%s, turns=%s, tokens=%s",
                        partial["tool_calls"],
                        partial["turns"],
                        partial["tokens_used"],
                    )
                    agent_callback._partial_data = {
                        "code": partial.get("code", ""),
                        "tokens_input": partial.get("tokens_input", 0),
                        "tokens_output": partial.get("tokens_output", 0),
                        "tokens_used": partial.get("tokens_used", 0),
                        "tool_calls": partial.get("tool_calls", 0),
                        "turns": partial.get("turns", 0),
                        "session_id": adapter._task_session_id,
                        "conversation_trace": adapter.get_conversation_trace(),
                        "prompt_identities": adapter.get_served_prompt_identities(),
                    }
                    raise
                finally:
                    if _persistent_container is not None:
                        await _persistent_container.stop()
                    os.chdir(original_cwd)
                    if temp_work_dir is not None:
                        import shutil

                        shutil.rmtree(temp_work_dir, ignore_errors=True)

                # Capture actual file changes as a git diff patch.
                # The agent edits files via the edit tool (modifies on disk),
                # but the evaluation harness needs a patch string to apply
                # on a fresh repo clone and run tests.
                patch = trace.generated_patch or trace.generated_code or ""
                if not patch and work_dir:
                    try:
                        diff_proc = await asyncio.create_subprocess_exec(
                            "git",
                            "diff",
                            cwd=work_dir,
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE,
                        )
                        diff_out, _ = await diff_proc.communicate()
                        if diff_out:
                            patch = diff_out.decode("utf-8", errors="replace")
                            logger.info(
                                "Captured git diff patch: %d bytes from workspace",
                                len(patch),
                            )
                    except Exception as e:
                        logger.debug("Failed to capture git diff: %s", e)

                # Patch or source, decided by what the runner will do with it.
                # SWE-bench applies this to a fresh clone, so it wants the diff.
                # Code-generation runners execute `agent_output + test_code`
                # directly, so a diff is a SyntaxError on line 3 — which is why
                # MBPP scored 0 on every one of the 135 real tasks it has run.
                # The patch stays as the fallback: an empty source capture should
                # degrade to the old behaviour, not to nothing.
                benchmark_slug = getattr(
                    benchmark_task.benchmark, "value", benchmark_task.benchmark
                )
                if benchmark_slug in _SOURCE_BENCHMARKS | _REPORT_BENCHMARKS:
                    solution = trace.generated_code or patch
                else:
                    solution = patch

                return {
                    "code": solution,
                    "tokens_input": trace.token_usage.input_tokens,
                    "tokens_output": trace.token_usage.output_tokens,
                    "tokens_used": trace.token_usage.total_tokens,
                    "tool_calls": len(trace.tool_calls),
                    "turns": trace.turns,
                    "code_search_calls": trace.code_search_calls,
                    "graph_calls": trace.graph_calls,
                    # Per-task correlation spine — joins this task's decisions
                    # (logged with get_session_id()) to its outcome for the
                    # execution manifest / classifier training data.
                    "session_id": trace.session_id,
                    # Bounded execution trace (messages + tool calls + edits).
                    # Persisted on TaskResult so the manifest + miner can
                    # reconstruct what the agent actually did.
                    "conversation_trace": adapter.get_conversation_trace(),
                    # Candidates actually served — lets this task's pass/fail
                    # become Pareto evidence without an explicit A/B flag.
                    "prompt_identities": adapter.get_served_prompt_identities(),
                }

        except Exception as e:
            console.print(f"[red]Error initializing agent:[/] {e}")
            import traceback

            traceback.print_exc()
            return None

        def on_progress(task_idx: int, total: int, result):
            progress.update(
                task,
                description=f"Task {task_idx + 1}/{total}: {result.status.value}",
            )
            # FEP-0012 Gap A fix: record this task's outcome the INSTANT it
            # completes — not deferred to after run_evaluation() returns. A
            # ctrl+c / crash / timeout during the benchmark then loses only the
            # in-flight task, never the completed ones. Best-effort.
            try:
                _record_benchmark_task_outcome(result)
            except Exception as exc:  # never break the benchmark
                logger.debug("per-task decision_outcome recording skipped: %s", exc)

        progress.update(
            task,
            description="Resuming evaluation..." if resume else "Running evaluation...",
        )
        eval_result = await harness.run_evaluation(
            config=config,
            agent_callback=agent_callback,
            progress_callback=on_progress,
            resume=resume,
        )

        # Emit the per-task execution manifest (closed-loop artifact): joins
        # each task's outcome (reward) + trace + logged decisions by session_id.
        # Best-effort — failure here never breaks the run. Feeds the classifier
        # miner (`victor benchmark mine` / python -m victor.ml.mining).
        try:
            from victor.evaluation.manifest import emit_execution_manifest

            manifest_path = emit_execution_manifest(eval_result)
            if manifest_path is not None:
                console.print(f"[dim]Execution manifest: {manifest_path}[/]")
        except Exception as exc:  # never break the benchmark
            logger.debug("Execution manifest emission skipped: %s", exc)

        # FEP-0012 safety net: re-stamp outcomes for every task now that the full
        # run completed. on_progress already recorded each as it finished (so an
        # interrupt loses nothing); this loop is idempotent and only backfills
        # any task whose on_progress callback did not fire (e.g. an edge on the
        # final task). Best-effort — never breaks the run.
        try:
            for _tr in getattr(eval_result, "task_results", []) or []:
                _record_benchmark_task_outcome(_tr)
        except Exception as exc:  # never break the benchmark
            logger.debug("decision_outcome safety-net recording skipped: %s", exc)

        # Deterministic teardown: stop any embedded server this run started
        # before returning. Servers are spawned setsid and survive their
        # parent, so a harness that finishes without releasing leaves them
        # running against the benchmark data dirs — where they keep rewriting
        # state and contaminate the next iteration (anvai-labs/victor#911).
        try:
            from victor.storage.proxima_runtime import (
                live_connection_count,
                release_all_connections,
            )

            if live_connection_count():
                await release_all_connections()
        except Exception as exc:  # pragma: no cover - teardown must not raise
            logger.debug("Embedded connection drain skipped: %s", exc)

        return eval_result


async def _run_prompt_candidate_suite_async(
    *,
    runner,
    base_config,
    candidate_specs,
    profile: str,
    model: Optional[str],
    timeout: int,
    max_turns: int,
    resume: bool,
    provider_override: Optional[str] = None,
    start_task: int = 0,
    resolved_account=None,
):
    """Run one benchmark evaluation per prompt candidate and return the suite result."""
    from victor.evaluation import (
        PromptCandidateEvaluationRun,
        PromptCandidateEvaluationSuiteResult,
        bind_prompt_candidate_evaluation_config,
    )

    runs = []
    total_candidates = len(candidate_specs)

    for idx, spec in enumerate(candidate_specs, start=1):
        bound_config = bind_prompt_candidate_evaluation_config(base_config, spec)
        label = spec.resolved_label(bound_config.provider)
        console.print(f"[bold cyan]Candidate {idx}/{total_candidates}:[/] {label}")
        result = await _run_benchmark_async(
            runner=runner,
            config=bound_config,
            profile=profile,
            model=model,
            timeout=timeout,
            max_turns=max_turns,
            resume=resume,
            provider_override=provider_override or spec.provider,
            start_task=start_task,
            resolved_account=resolved_account,
        )
        if result is None:
            return None
        runs.append(
            PromptCandidateEvaluationRun(
                spec=spec,
                config=bound_config,
                result=result,
                label=label,
            )
        )

    return PromptCandidateEvaluationSuiteResult(base_config=base_config, runs=runs)


@benchmark_app.command("compare")
def compare_frameworks(
    benchmark: str = typer.Option(
        "swe-bench", "--benchmark", "-b", help="Benchmark for comparison"
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output file for comparison report"
    ),
    format: str = typer.Option(
        "table", "--format", "-f", help="Output format: table, markdown, json"
    ),
    victor_results: list[Path] = typer.Option(
        [],
        "--victor-results",
        help="Saved Victor benchmark JSON artifact(s) to include in the comparison",
    ),
    victor_fixture_set: list[str] = typer.Option(
        [],
        "--victor-fixture-set",
        help="Checked-in Victor fixture set name(s) to include in the comparison",
    ),
    victor_fixture_benchmark: Optional[str] = typer.Option(
        None,
        "--victor-fixture-benchmark",
        help="Include all checked-in Victor fixture sets for the named benchmark",
    ),
    victor_publication_root: Optional[Path] = typer.Option(
        None,
        "--victor-publication-root",
        help="Portable Victor fixture publication bundle root or catalog to include for this benchmark",
    ),
    fixture_set_root: Path = typer.Option(
        Path("tests/fixtures/benchmarks"),
        "--fixture-set-root",
        help="Root directory containing checked-in Victor fixture sets",
    ),
) -> None:
    """Compare Victor against other AI coding frameworks."""
    from victor.evaluation.benchmarks import (
        ComparisonMetrics,
        ComparisonReport,
        FrameworkResult,
        PUBLISHED_RESULTS,
        create_comparison_report_from_saved_results,
        resolve_fixture_benchmark_publication_manifests,
        resolve_fixture_sets_for_benchmark,
        resolve_fixture_set_names,
        save_comparison_report_bundle,
    )
    from victor.evaluation.protocol import (
        get_benchmark_metadata,
        normalize_benchmark_name,
    )

    benchmark_lower = normalize_benchmark_name(benchmark)
    metadata = get_benchmark_metadata(benchmark_lower)
    if metadata is None:
        console.print(f"[bold red]Error:[/] Unknown benchmark: {benchmark}")
        raise typer.Exit(1)

    bench_type = metadata.type

    console.print(f"\n[bold cyan]Framework Comparison: {benchmark}[/]\n")

    resolved_victor_results = list(victor_results)
    if victor_fixture_set:
        try:
            resolved_victor_results.extend(
                resolve_fixture_set_names(victor_fixture_set, root=fixture_set_root)
            )
        except Exception as exc:
            console.print(f"[bold red]Error:[/] Failed to load Victor results: {exc}")
            raise typer.Exit(1)
    resolved_fixture_benchmark: Optional[str] = None
    if victor_fixture_benchmark is not None:
        fixture_benchmark_lower = normalize_benchmark_name(victor_fixture_benchmark)
        fixture_benchmark_metadata = get_benchmark_metadata(fixture_benchmark_lower)
        if fixture_benchmark_metadata is None:
            console.print(
                "[bold red]Error:[/] Failed to load Victor results: "
                f"Unknown fixture benchmark: {victor_fixture_benchmark}"
            )
            raise typer.Exit(1)
        resolved_fixture_benchmark = fixture_benchmark_metadata.type.value
        try:
            resolved_victor_results.extend(
                resolve_fixture_sets_for_benchmark(
                    resolved_fixture_benchmark,
                    root=fixture_set_root,
                )
            )
        except Exception as exc:
            console.print(f"[bold red]Error:[/] Failed to load Victor results: {exc}")
            raise typer.Exit(1)
    if victor_publication_root is not None:
        try:
            resolved_victor_results.extend(
                resolve_fixture_benchmark_publication_manifests(
                    root=victor_publication_root,
                    benchmark=metadata.name,
                )
            )
        except Exception as exc:
            console.print(f"[bold red]Error:[/] Failed to load Victor results: {exc}")
            raise typer.Exit(1)

    if resolved_victor_results:
        try:
            report = create_comparison_report_from_saved_results(
                resolved_victor_results,
                include_published=True,
            )
        except Exception as exc:
            console.print(f"[bold red]Error:[/] Failed to load Victor results: {exc}")
            raise typer.Exit(1)
        if report.benchmark != bench_type:
            console.print(
                "[bold red]Error:[/] Victor result artifact benchmark "
                f"'{report.benchmark.value}' does not match requested benchmark '{metadata.name}'"
            )
            raise typer.Exit(1)
    else:
        if bench_type not in PUBLISHED_RESULTS:
            console.print("[yellow]No published results available for this benchmark[/]")
            console.print("Run 'victor benchmark run' to generate Victor results")
            raise typer.Exit(0)

        report = ComparisonReport(benchmark=bench_type)
        for framework, data in PUBLISHED_RESULTS[bench_type].items():
            report.results.append(
                FrameworkResult(
                    framework=framework,
                    benchmark=bench_type,
                    model=data.get("model", "unknown"),
                    metrics=ComparisonMetrics(pass_rate=data.get("pass_rate", 0.0)),
                    config={"source": data.get("source", "published")},
                )
            )

    table = Table(title=f"Results: {benchmark}")
    table.add_column("Framework", style="cyan")
    table.add_column("Model", style="white")
    table.add_column("Pass Rate", style="green")
    table.add_column("Accepted Patch", style="magenta")
    table.add_column("Time to First Edit", style="yellow")
    table.add_column("Code-Intel", style="blue")
    table.add_column("Source", style="dim")

    for result in sorted(report.results, key=lambda item: item.metrics.pass_rate, reverse=True):
        has_local_metrics = _comparison_has_local_metrics(result)
        table.add_row(
            result.framework.value,
            result.model,
            f"{result.metrics.pass_rate:.1%}",
            _format_comparison_percent(
                result.metrics.accepted_patch_rate,
                available=has_local_metrics,
            ),
            _format_comparison_seconds(
                result.metrics.avg_time_to_first_edit_seconds,
                available=has_local_metrics,
            ),
            _format_comparison_percent(
                result.metrics.code_intelligence_task_coverage,
                available=has_local_metrics,
            ),
            _comparison_result_source(result),
        )

    console.print(table)

    if victor_fixture_set:
        console.print(
            "\n[dim]Included Victor fixture sets: " + ", ".join(victor_fixture_set) + "[/]"
        )
    if resolved_fixture_benchmark is not None:
        console.print(f"[dim]Included Victor fixture benchmark: {resolved_fixture_benchmark}[/]")
    if victor_publication_root is not None:
        console.print(f"[dim]Included Victor publication root: {victor_publication_root}[/]")
    if victor_results:
        included = ", ".join(str(path) for path in victor_results)
        console.print(f"[dim]Included local Victor results: {included}[/]")

    if output:
        bundle_paths = save_comparison_report_bundle(report, output, primary_format=format)
        console.print(f"\n[dim]Report saved to {bundle_paths['primary']}[/]")
        console.print(f"[dim]Summary saved to {bundle_paths['summary']}[/]")


@benchmark_app.command("fixture-sets")
def list_fixture_sets(
    benchmark: Optional[str] = typer.Option(
        None,
        "--benchmark",
        "-b",
        help="Filter fixture sets by benchmark",
    ),
    verify: bool = typer.Option(
        False,
        "--verify",
        help="Verify fixture artifact integrity for each listed set",
    ),
    root: Path = typer.Option(
        Path("tests/fixtures/benchmarks"),
        "--root",
        help="Root directory containing checked-in benchmark fixture sets",
    ),
) -> None:
    """List checked-in saved benchmark fixture sets."""
    from victor.evaluation.benchmarks import (
        discover_fixture_sets,
        fixture_benchmark_matches,
        verify_fixture_sets,
    )
    from victor.evaluation.protocol import (
        get_benchmark_metadata,
        normalize_benchmark_name,
    )

    descriptors = discover_fixture_sets(root)
    requested_benchmark: Optional[str] = None
    if benchmark is not None:
        benchmark_lower = normalize_benchmark_name(benchmark)
        metadata = get_benchmark_metadata(benchmark_lower)
        if metadata is None:
            console.print(f"[bold red]Error:[/] Unknown benchmark: {benchmark}")
            raise typer.Exit(1)
        requested_benchmark = metadata.name
        descriptors = [
            descriptor
            for descriptor in descriptors
            if fixture_benchmark_matches(descriptor.benchmark, requested_benchmark)
        ]

    if not descriptors:
        console.print(f"[yellow]No fixture sets found under {root}[/]")
        raise typer.Exit(0)

    table = Table(title="Checked-In Benchmark Fixture Sets")
    table.add_column("Name", style="cyan")
    table.add_column("Benchmark", style="green")
    table.add_column("Artifacts", justify="right")
    table.add_column("Models", style="white")
    table.add_column("Manifest", style="dim")

    for descriptor in descriptors:
        table.add_row(
            descriptor.name,
            descriptor.benchmark,
            str(descriptor.artifact_count),
            ", ".join(descriptor.models) or "-",
            str(descriptor.manifest_path),
        )

    console.print(table)
    console.print(
        "\n[dim]Available fixture sets: "
        + ", ".join(descriptor.name for descriptor in descriptors)
        + "[/]"
    )
    all_models = [model for descriptor in descriptors for model in descriptor.models]
    if all_models:
        console.print("[dim]Fixture models: " + ", ".join(dict.fromkeys(all_models)) + "[/]")
    if verify:
        try:
            verification_results = verify_fixture_sets(
                root=root,
                benchmark=requested_benchmark,
            )
        except Exception as exc:
            console.print(f"[bold red]Error:[/] Fixture verification failed: {exc}")
            raise typer.Exit(1)
        verified_artifact_count = sum(
            result.verified_artifact_count for result in verification_results
        )
        console.print(
            f"[dim]Verified fixture sets: {len(verification_results)} "
            f"({verified_artifact_count} artifacts)[/]"
        )
    console.print(
        "[dim]Use with: victor benchmark compare --victor-fixture-set <name> "
        "or --victor-fixture-benchmark <benchmark>[/]"
    )


@benchmark_app.command("fixture-benchmarks")
def list_fixture_benchmarks(
    benchmark: Optional[str] = typer.Option(
        None,
        "--benchmark",
        "-b",
        help="Filter benchmark corpora by benchmark name",
    ),
    verify: bool = typer.Option(
        False,
        "--verify",
        help="Verify fixture artifact integrity for each benchmark corpus",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Optional output path for a saved fixture benchmark catalog JSON",
    ),
    bundle_output: Optional[Path] = typer.Option(
        None,
        "--bundle-output",
        help="Optional output directory for portable fixture benchmark publication bundle(s)",
    ),
    root: Path = typer.Option(
        Path("tests/fixtures/benchmarks"),
        "--root",
        help="Root directory containing checked-in benchmark fixture sets",
    ),
) -> None:
    """List checked-in benchmark-level fixture corpora."""
    from victor.evaluation.benchmarks import (
        build_fixture_benchmark_catalog,
        discover_fixture_benchmarks,
        fixture_benchmark_matches,
        save_fixture_benchmark_catalog,
        save_fixture_benchmark_publication_bundle,
        verify_fixture_sets,
    )
    from victor.evaluation.protocol import (
        get_benchmark_metadata,
        normalize_benchmark_name,
    )

    descriptors = discover_fixture_benchmarks(root)
    requested_benchmark: Optional[str] = None
    if benchmark is not None:
        benchmark_lower = normalize_benchmark_name(benchmark)
        metadata = get_benchmark_metadata(benchmark_lower)
        if metadata is None:
            console.print(f"[bold red]Error:[/] Unknown benchmark: {benchmark}")
            raise typer.Exit(1)
        requested_benchmark = metadata.name
        descriptors = [
            descriptor
            for descriptor in descriptors
            if fixture_benchmark_matches(descriptor.benchmark, requested_benchmark)
        ]
    if not descriptors:
        console.print(f"[yellow]No fixture benchmarks found under {root}[/]")
        raise typer.Exit(0)
    coverage_catalog = build_fixture_benchmark_catalog(
        root=root,
        benchmark=requested_benchmark,
        verify=False,
    )

    verification_summary_by_benchmark: dict[str, tuple[int, int]] = {}
    if verify:
        try:
            verification_results = verify_fixture_sets(root=root, benchmark=requested_benchmark)
        except Exception as exc:
            console.print(f"[bold red]Error:[/] Fixture verification failed: {exc}")
            raise typer.Exit(1)
        for result in verification_results:
            verified_sets, verified_artifacts = verification_summary_by_benchmark.get(
                result.benchmark,
                (0, 0),
            )
            verification_summary_by_benchmark[result.benchmark] = (
                verified_sets + 1,
                verified_artifacts + result.verified_artifact_count,
            )

    table = Table(title="Checked-In Benchmark Fixture Corpora")
    table.add_column("Benchmark", style="green")
    table.add_column("Fixture Sets", justify="right")
    table.add_column("Artifacts", justify="right")
    table.add_column("Models", style="white")
    table.add_column("Fixture Sets", style="cyan")
    if verify:
        table.add_column("Verified", style="magenta")

    for descriptor in descriptors:
        row = [
            descriptor.benchmark,
            str(descriptor.fixture_set_count),
            str(descriptor.artifact_count),
            ", ".join(descriptor.models) or "-",
            ", ".join(descriptor.fixture_set_names) or "-",
        ]
        if verify:
            verified_sets, verified_artifacts = verification_summary_by_benchmark.get(
                descriptor.benchmark,
                (0, 0),
            )
            row.append(f"{verified_sets} sets / {verified_artifacts} artifacts")
        table.add_row(*row)

    console.print(table)
    console.print(
        "[dim]Fixture benchmark sets: "
        + "; ".join(
            f"{descriptor.benchmark}=" + ", ".join(descriptor.fixture_set_names)
            for descriptor in descriptors
        )
        + "[/]"
    )
    benchmark_models = [
        f"{descriptor.benchmark}=" + ", ".join(descriptor.models)
        for descriptor in descriptors
        if descriptor.models
    ]
    if benchmark_models:
        console.print(
            "[dim]Fixture benchmark models: " + "; ".join(benchmark_models) + "[/]",
            soft_wrap=True,
        )
    benchmark_publishers = []
    for descriptor in descriptors:
        metadata = get_benchmark_metadata(descriptor.benchmark)
        source_name = metadata.source_name if metadata is not None else ""
        if not source_name:
            continue
        benchmark_publishers.append(f"{descriptor.benchmark}={source_name}")
    if benchmark_publishers:
        console.print(
            "[dim]Fixture benchmark publishers: " + "; ".join(benchmark_publishers) + "[/]",
            soft_wrap=True,
        )
    console.print(
        "[dim]Fixture benchmark coverage: "
        + f"{coverage_catalog['covered_catalog_benchmark_count']}/"
        + f"{coverage_catalog['catalog_benchmark_count']} cataloged benchmarks "
        + f"({coverage_catalog['catalog_benchmark_coverage_rate']:.1%})[/]",
        soft_wrap=True,
    )
    missing_catalog_benchmarks = list(coverage_catalog.get("missing_catalog_benchmarks") or [])
    if coverage_catalog.get("has_full_catalog_coverage"):
        console.print(
            "[dim]All cataloged benchmarks have checked-in fixture coverage.[/]",
            soft_wrap=True,
        )
    elif missing_catalog_benchmarks:
        console.print(
            "[dim]Missing fixture benchmarks: "
            + ", ".join(
                str(item.get("name", "")).strip()
                for item in missing_catalog_benchmarks
                if str(item.get("name", "")).strip()
            )
            + "[/]",
            soft_wrap=True,
        )
    if verify:
        console.print(
            f"[dim]Verified fixture benchmarks: {len(verification_summary_by_benchmark)}[/]"
        )
    if output is not None:
        try:
            saved_path = save_fixture_benchmark_catalog(
                output_path=output,
                root=root,
                benchmark=requested_benchmark,
                verify=verify,
            )
        except Exception as exc:
            console.print(f"[bold red]Error:[/] Failed to save fixture benchmark catalog: {exc}")
            raise typer.Exit(1)
        console.print(f"[dim]Fixture benchmark catalog saved to {saved_path}[/]", soft_wrap=True)
    if bundle_output is not None:
        try:
            bundle_paths = save_fixture_benchmark_publication_bundle(
                output_path=bundle_output,
                root=root,
                benchmark=requested_benchmark,
                verify=verify,
            )
        except Exception as exc:
            console.print(
                f"[bold red]Error:[/] Failed to save fixture benchmark publication bundle: {exc}"
            )
            raise typer.Exit(1)
        console.print(
            f"[dim]Fixture benchmark publication bundle saved to {bundle_paths['root']}[/]",
            soft_wrap=True,
        )
        console.print(
            f"[dim]Publication catalog saved to {bundle_paths['catalog']}[/]",
            soft_wrap=True,
        )
    console.print(
        "[dim]Use with: victor benchmark compare --victor-fixture-benchmark <benchmark>[/]"
    )


@benchmark_app.command("stable-runs")
def publish_stable_runs(
    victor_results: list[Path] = typer.Option(
        [],
        "--victor-results",
        help="Saved Victor benchmark JSON artifact(s) from real runs to publish",
    ),
    benchmark: Optional[str] = typer.Option(
        None,
        "--benchmark",
        "-b",
        help="Optional benchmark name to validate against the saved run artifacts",
    ),
    bundle_output: Path = typer.Option(
        ...,
        "--bundle-output",
        help="Output directory for the stable real-run publication bundle",
    ),
    require_publishable: bool = typer.Option(
        False,
        "--require-publishable",
        help="Fail if the generated stable-run corpus is missing required public KPIs or tasks",
    ),
) -> None:
    """Publish stable benchmark outputs generated from saved real-run artifacts."""
    from victor.evaluation.benchmarks import save_stable_run_publication_bundle

    if not victor_results:
        console.print("[bold red]Error:[/] At least one --victor-results artifact is required")
        raise typer.Exit(1)

    try:
        publication = save_stable_run_publication_bundle(
            output_path=bundle_output,
            result_paths=victor_results,
            benchmark=benchmark,
        )
    except Exception as exc:
        console.print(f"[bold red]Error:[/] Failed to publish stable real runs: {exc}")
        raise typer.Exit(1)

    console.print(f"Stable real-run publication bundle saved to {publication['root']}")
    console.print(f"[dim]Catalog: {publication['catalog']}[/]")
    for benchmark_name, manifest in publication["benchmark_manifests"].items():
        console.print(f"[dim]{benchmark_name} manifest: {manifest}[/]")

    catalog = json.loads(Path(publication["catalog"]).read_text())
    for benchmark_payload in list(catalog.get("benchmarks") or []):
        if not isinstance(benchmark_payload, dict):
            continue
        readiness = (
            dict(benchmark_payload.get("stable_run_summary", {}).get("corpus_readiness") or {})
            if isinstance(benchmark_payload.get("stable_run_summary"), dict)
            else {}
        )
        if not readiness:
            continue
        benchmark_name = str(benchmark_payload.get("benchmark", "")).strip() or "benchmark"
        status = "publishable" if readiness.get("publishable") else "not publishable"
        console.print(
            f"[dim]{benchmark_name} corpus readiness: {status}; "
            f"tasks={readiness.get('task_count', 0)}, "
            f"artifacts={readiness.get('artifact_count', 0)}[/]"
        )
        if require_publishable and not readiness.get("publishable"):
            reasons = ", ".join(str(reason) for reason in readiness.get("missing_reasons", []))
            console.print(
                "[bold red]Error:[/] Stable real-run corpus is not publishable"
                + (f": {reasons}" if reasons else "")
            )
            raise typer.Exit(1)


@benchmark_app.command("leaderboard")
def show_leaderboard(
    benchmark: str = typer.Option("swe-bench", "--benchmark", "-b", help="Benchmark to show"),
    victor_results: Optional[Path] = typer.Option(
        None,
        "--victor-results",
        help="Saved Victor benchmark JSON artifact to include in the leaderboard",
    ),
) -> None:
    """Show the leaderboard for a benchmark."""
    from victor.evaluation.benchmarks import (
        PUBLISHED_RESULTS,
        create_comparison_report_from_saved_result,
    )
    from victor.evaluation.protocol import (
        get_benchmark_metadata,
        normalize_benchmark_name,
    )

    benchmark_lower = normalize_benchmark_name(benchmark)
    metadata = get_benchmark_metadata(benchmark_lower)
    if metadata is None:
        console.print(f"[bold red]Error:[/] Unknown benchmark: {benchmark}")
        raise typer.Exit(1)

    bench_type = metadata.type

    console.print(f"\n[bold cyan]Leaderboard: {benchmark}[/]\n")

    entries: list[tuple[str, float, str]] = []
    if bench_type in PUBLISHED_RESULTS:
        for framework, data in PUBLISHED_RESULTS[bench_type].items():
            entries.append(
                (
                    framework.value,
                    float(data.get("pass_rate", 0.0)),
                    str(data.get("source", "")),
                )
            )

    if victor_results is not None:
        try:
            report = create_comparison_report_from_saved_result(
                victor_results, include_published=False
            )
        except Exception as exc:
            console.print(f"[bold red]Error:[/] Failed to load Victor results: {exc}")
            raise typer.Exit(1)
        if report.benchmark != bench_type:
            console.print(
                "[bold red]Error:[/] Victor result artifact benchmark "
                f"'{report.benchmark.value}' does not match requested benchmark '{metadata.name}'"
            )
            raise typer.Exit(1)
        local_result = report.results[0]
        entries.append(
            (
                local_result.framework.value,
                float(local_result.metrics.pass_rate),
                _comparison_result_source(local_result),
            )
        )

    if not entries:
        console.print("[yellow]No results available[/]")
        raise typer.Exit(0)

    sorted_results = sorted(entries, key=lambda item: item[1], reverse=True)

    table = Table()
    table.add_column("Rank", style="bold")
    table.add_column("Framework", style="cyan")
    table.add_column("Pass Rate", style="green")
    table.add_column("Source", style="dim")

    for i, (framework_name, pass_rate, source) in enumerate(sorted_results, 1):
        medal = ""
        if i == 1:
            medal = "🥇 "
        elif i == 2:
            medal = "🥈 "
        elif i == 3:
            medal = "🥉 "

        table.add_row(
            f"{medal}{i}",
            framework_name,
            f"{pass_rate:.1%}",
            source,
        )

    console.print(table)
    if victor_results is not None:
        console.print(f"\n[dim]Includes local Victor result: {victor_results}[/]")
    else:
        console.print("\n[dim]Based on published benchmark results[/]")


@benchmark_app.command("capabilities")
def show_capabilities() -> None:
    """Show capabilities comparison across frameworks."""
    from victor.evaluation.benchmarks import Framework, FRAMEWORK_CAPABILITIES

    console.print("\n[bold cyan]Framework Capabilities Comparison[/]\n")

    table = Table()
    table.add_column("Capability", style="cyan")

    # Add framework columns
    frameworks = [
        Framework.VICTOR,
        Framework.AIDER,
        Framework.CLAUDE_CODE,
        Framework.CURSOR,
    ]
    for f in frameworks:
        table.add_column(f.value, justify="center")

    capabilities = [
        ("Code Generation", "code_generation"),
        ("Code Editing", "code_editing"),
        ("Multi-file Edit", "multi_file_editing"),
        ("Tool Use", "tool_use"),
        ("Autonomous Mode", "autonomous_mode"),
        ("Planning", "planning"),
        ("Local Models", "local_models"),
        ("Air-gapped", "air_gapped"),
        ("MCP Support", "mcp_support"),
        ("Open Source", "open_source"),
    ]

    for cap_name, cap_attr in capabilities:
        row = [cap_name]
        for f in frameworks:
            if f in FRAMEWORK_CAPABILITIES:
                cap = FRAMEWORK_CAPABILITIES[f]
                value = getattr(cap, cap_attr, False)
                row.append("[green]✓[/]" if value else "[red]✗[/]")
            else:
                row.append("[dim]?[/]")
        table.add_row(*row)

    console.print(table)


_SMALL_MODEL_RE = re.compile(r"[:\-](\d+(?:\.\d+)?)b\b", re.IGNORECASE)
UNDERSIZED_MUTATOR_B = 7.0


def _is_undersized_mutator(provider: Optional[str], model: Optional[str]) -> bool:
    """True when the reflect/mutate model is too small to rewrite a long prompt.

    Parameter count is parsed from the model name (``qwen3.5:2b`` -> 2.0), which
    is the only signal available locally. Purely advisory — a false negative is
    just a missing warning. Exists because a 2B local model silently rewrote
    every prompt for months and the symptom (a whitespace-only diff) looks like
    a strategy problem rather than a capacity one.
    """
    name = f"{provider or ''}/{model or ''}"
    match = _SMALL_MODEL_RE.search(name)
    if match is None:
        return False
    try:
        return float(match.group(1)) < UNDERSIZED_MUTATOR_B
    except ValueError:
        return False


def _build_mutator_rotation(profiles: Optional[str]):
    """Build a mutator rotation from a comma-separated profile list, or None."""
    if not profiles:
        return None
    from victor.framework.rl.mutator_rotation import build_rotation

    entries = []
    for name in (n.strip() for n in profiles.split(",")):
        if not name:
            continue
        provider, model = _resolve_mutator_spec(name, None)
        if provider and model:
            entries.append((provider, model, name))
        else:
            console.print(f"[yellow]Warning:[/] skipping unresolvable profile '{name}'")
    rotation = build_rotation(entries)
    if not rotation:
        console.print("[yellow]Warning:[/] no profiles in --profiles resolved; ignoring it")
        return None
    return rotation


def _resolve_mutator_spec(
    profile: Optional[str], model: Optional[str]
) -> tuple[Optional[str], Optional[str]]:
    """Resolve the (provider, model) that will perform reflection and mutation.

    Explicit --model wins; otherwise the profile's provider/model is used.
    Returns (None, None) when neither is given, leaving GEPA's existing
    resolution untouched so behaviour is unchanged for callers that do not ask.
    """
    if not profile and not model:
        return None, None
    resolved_provider = None
    resolved_model = model
    if profile:
        try:
            # The same two stores `victor auth show` consults, in its order.
            # SessionConfig.from_cli_flags only records the profile *name* and
            # Settings().load_profiles() does not carry these entries at all, so
            # neither resolves a provider — an earlier attempt through them
            # silently yielded nothing and the 2B fallback mutated anyway.
            from victor.config.accounts import get_account_manager
            from victor.framework.runtime_discovery import list_runtime_profiles

            entry = next((p for p in list_runtime_profiles() if p.name == profile), None)
            if entry is None:
                entry = next(
                    (a for a in get_account_manager().list_accounts() if a.name == profile),
                    None,
                )
            if entry is None:
                console.print(
                    f"[yellow]Warning:[/] profile '{profile}' does not resolve; "
                    "falling back to the session default model for reflect/mutate. "
                    "Run 'victor auth list' for valid names."
                )
            else:
                resolved_provider = getattr(entry, "provider", None)
                resolved_model = model or getattr(entry, "model", None)
        except Exception as exc:
            logger.debug("Could not resolve mutator profile %r: %s", profile, exc)
    return resolved_provider, resolved_model


_LEGACY_EVOLVE_PROVIDERS = ["openai", "xai", "deepseek", "anthropic"]


def _providers_present_in_traces(limit: int = 400) -> list[str]:
    """Providers that actually appear in the trace corpus, newest first.

    ``--provider all`` used to mean a hardcoded ["openai", "xai", "deepseek",
    "anthropic"]. Any provider outside that list could never be evolved, however
    much evidence it had: an overnight run on zai produced 656 tool events and
    123 graded tasks, and evolve created candidates for four providers that had
    not run while creating none for the one that had. The next benchmark phase
    then had nothing to serve, so the reward loop could not fire at all.

    Deriving the list from the traces means evolution follows the evidence.
    Falls back to the legacy list only when no provider can be identified, so a
    fresh install still does something sensible.
    """
    from victor.framework.rl.learners.prompt_optimizer import PromptOptimizerLearner

    counts: dict[str, int] = {}
    try:
        from victor.config.settings import get_project_paths

        logs_dir = Path(get_project_paths().global_logs_dir)
    except Exception:
        logs_dir = Path.home() / ".victor" / "logs"

    usage = logs_dir / "usage.jsonl"
    if usage.exists():
        try:
            with open(usage) as handle:
                for line in handle:
                    try:
                        data = json.loads(line).get("data") or {}
                    except (ValueError, AttributeError):
                        continue
                    if not isinstance(data, dict):
                        continue
                    label = PromptOptimizerLearner._normalize_provider_label(
                        data.get("provider", "")
                    )
                    if label:
                        counts[label] = counts.get(label, 0) + 1
        except OSError:
            pass

    if not counts:
        return list(_LEGACY_EVOLVE_PROVIDERS)
    return [name for name, _ in sorted(counts.items(), key=lambda kv: -kv[1])][:limit]


@benchmark_app.command("evolve")
def evolve_prompts(
    provider: str = typer.Option("all", "--provider", "-p", help="Provider to evolve (or 'all')"),
    section: str = typer.Option("all", "--section", "-s", help="Section to evolve (or 'all')"),
    compliance: bool = typer.Option(False, "--compliance", help="Show GEPA compliance scorecard"),
    profile: Optional[str] = typer.Option(
        None,
        "--profile",
        help="Profile whose model performs reflection/mutation (not the provider being evolved)",
    ),
    model: Optional[str] = typer.Option(
        None, "--model", "-m", help="Override the reflect/mutate model directly"
    ),
    profiles: Optional[str] = typer.Option(
        None,
        "--profiles",
        help=(
            "Comma-separated profiles to ROTATE the reflect/mutate model across sections, "
            "e.g. zai-glm52-openai,kimi,deepseek-v4pro-openai. Spreads calls so one "
            "provider's rate limit cannot stall the pass; a provider that 429s is benched."
        ),
    ),
) -> None:
    """Evolve prompts using GEPA + benchmark trace data.

    Reads execution traces from usage.jsonl and evaluation results,
    then evolves prompt sections using the configured strategy
    (GEPA, MIPROv2, CoT distillation).

    Examples:
        victor benchmark evolve                 # Evolve all
        victor benchmark evolve -p openai       # Evolve for OpenAI
        victor benchmark evolve --compliance    # Show compliance
    """
    from rich.table import Table

    try:
        from victor.framework.rl.coordinator import get_rl_coordinator

        coordinator = get_rl_coordinator()
        learner = coordinator.get_learner("prompt_optimizer")
        if learner is None:
            console.print("[yellow]Prompt optimizer not available[/]")
            return

        # Which model rewrites the prompt is the single biggest determinant of
        # whether evolution produces anything. GEPA defaults to use_main_model,
        # so with no session to inherit from this CLI silently fell through to
        # the local edge model — observed as ollama/qwen3.5:2b rewriting a
        # 1551-char structured prompt, which it echoed. --provider selects the
        # candidate namespace being evolved; it does NOT select the mutator.
        # The /prompt-optimize slash command pushes the live session model here;
        # the CLI had no equivalent until now.
        rotation = _build_mutator_rotation(profiles)
        if rotation:
            console.print(f"[dim]Reflect/mutate {rotation.summary()}[/]")
            mutator_provider = mutator_model = None
            # Spreading calls across sections is not enough on its own: a single
            # section makes several calls, so one throttled provider still costs
            # the whole section. Arming the failover lets an individual call move
            # on, and both paths share this rotation — a provider benched mid-call
            # drops out of the per-section round-robin too.
            if hasattr(learner, "set_mutator_rotation"):
                learner.set_mutator_rotation(rotation)
        else:
            mutator_provider, mutator_model = _resolve_mutator_spec(profile, model)
            if mutator_provider and hasattr(learner, "set_main_model_spec"):
                learner.set_main_model_spec(mutator_provider, mutator_model)
            console.print(
                f"[dim]Reflect/mutate model: {mutator_provider or 'session default'}"
                f"/{mutator_model or 'default'}[/]"
            )
        if _is_undersized_mutator(mutator_provider, mutator_model):
            console.print(
                "[yellow]Warning:[/] the reflect/mutate model looks small. Rewriting a "
                "1500+ character structured prompt needs a capable model; a tiny one "
                "tends to echo its input, which reads as 'no change' or a "
                "whitespace-only diff. Pass --profile/--model to choose."
            )

        from victor.framework.rl.learners.prompt_optimizer import PromptOptimizerLearner

        sections = (
            learner.get_evolvable_sections()
            if hasattr(learner, "get_evolvable_sections")
            else PromptOptimizerLearner.EVOLVABLE_SECTIONS
        )
        if section != "all":
            matched = [s for s in sections if section.upper() in s]
            sections = matched if matched else sections

        if provider != "all":
            providers = [provider]
        else:
            providers = _providers_present_in_traces()

        section_text = _get_prompt_section_baselines(list(sections))

        results = Table(title="GEPA Evolution Results")
        results.add_column("Provider", style="cyan")
        results.add_column("Section", style="dim")
        results.add_column("Gen", style="green")
        results.add_column("Mean", style="bold")
        results.add_column("Chars", style="dim")

        # Load ALL evaluation results (not just latest) to compute
        # aggregate pass rates per provider across all benchmark runs.
        import glob as _glob
        import json as _json

        eval_dir = _get_global_evaluations_dir()
        provider_agg = {}  # provider → {"pass": N, "fail": N}
        model_to_provider = {
            "gpt": "openai",
            "grok": "xai",
            "deepseek": "deepseek",
            "haiku": "anthropic",
            "claude": "anthropic",
        }
        for ef in sorted(_glob.glob(str(eval_dir / "eval_swe_bench_*.json"))):
            try:
                with open(ef) as _f:
                    edata = _json.load(_f)
                emodel = edata.get("config", {}).get("model", "")
                etasks = edata.get("tasks", [])
                # Map model to provider
                p_name = None
                for prefix, prov in model_to_provider.items():
                    if prefix in emodel.lower():
                        p_name = prov
                        break
                if not p_name or len(etasks) < 1:
                    continue
                if p_name not in provider_agg:
                    provider_agg[p_name] = {"pass": 0, "fail": 0}
                for t in etasks:
                    if t.get("status") == "passed":
                        provider_agg[p_name]["pass"] += 1
                    else:
                        provider_agg[p_name]["fail"] += 1
            except Exception:
                pass

        # Also collect trace-based success from usage.jsonl
        traces = learner._collect_traces(limit=200)
        trace_pass = sum(1 for t in traces if t.success)
        trace_fail = len(traces) - trace_pass
        console.print(
            f"[dim]Seeding from {len(list(_glob.glob(str(eval_dir / 'eval_*.json'))))} "
            f"eval files + {len(traces)} traces[/]"
        )

        # Default seed for providers without benchmark data
        default_seed = (max(trace_pass, 5), max(trace_fail, 3))

        for p in providers:
            for s in sections:
                current = section_text.get(s)
                if current is None:
                    results.add_row(p, s[:20], "-", "-", "[yellow]not registered[/]")
                    continue
                # Rotate the mutator per section. A pass makes several LLM calls
                # per section back-to-back; pointing them all at one provider
                # reliably earns a 429 partway through, and a 429 in mutate is
                # not a degraded result but no result at all.
                spec = None
                if rotation:
                    spec = rotation.next_spec()
                    if spec is None:
                        results.add_row(p, s[:20], "-", "-", "[red]all mutators rate-limited[/]")
                        continue
                    if hasattr(learner, "set_main_model_spec"):
                        learner.set_main_model_spec(
                            spec.provider, spec.model, base_url=spec.base_url
                        )
                try:
                    candidate = learner.evolve(s, current, provider=p)
                except Exception as exc:
                    # evolve() swallows most failures internally; this catches the
                    # rest so one throttled provider cannot abort the whole pass.
                    if rotation and spec is not None:
                        rotation.note_failure(spec, exc)
                    results.add_row(p, s[:20], "-", "-", f"[red]{str(exc)[:28]}[/]")
                    continue
                if candidate:
                    # Seed from aggregate benchmark data or defaults
                    agg = provider_agg.get(p)
                    if agg and (agg["pass"] + agg["fail"]) > 0:
                        # Scale to max 15 to avoid overwhelming priors
                        total = agg["pass"] + agg["fail"]
                        scale = min(15 / max(total, 1), 1.0)
                        passes = max(1, int(agg["pass"] * scale))
                        fails = max(1, int(agg["fail"] * scale))
                    else:
                        # No benchmark data — use trace-based defaults
                        passes, fails = default_seed
                    for _ in range(passes):
                        candidate.update(True)
                    for _ in range(fails):
                        candidate.update(False)
                    learner._save_candidate(candidate)
                    results.add_row(
                        p,
                        s[:20],
                        str(candidate.generation),
                        f"{candidate.mean:.2f}",
                        str(len(candidate.text)),
                    )
                else:
                    results.add_row(p, s[:20], "-", "-", "[dim]no change[/]")

        console.print(results)
        metrics = learner.export_metrics()
        console.print(f"\n[dim]Total candidates: {metrics['total_candidates']}[/]")

    except Exception as e:
        console.print(f"[red]Evolution failed:[/] {e}")
        import traceback

        console.print(traceback.format_exc())

    if compliance:
        _show_compliance_scorecard()


def _show_compliance_scorecard() -> None:
    """Show GEPA prompt compliance analysis from usage.jsonl traces."""
    import gzip
    import json
    from collections import defaultdict
    from rich.table import Table

    events = []
    logs_dir = _get_global_usage_logs_dir()
    for path in sorted(logs_dir.glob("usage.*.jsonl.gz")) + [logs_dir / "usage.jsonl"]:
        if not path.exists():
            continue
        try:
            opener = gzip.open if path.suffix == ".gz" else open
            mode = "rt" if path.suffix == ".gz" else "r"
            with opener(str(path), mode) as f:
                for line in f:
                    try:
                        events.append(json.loads(line))
                    except Exception:
                        pass
        except Exception:
            pass

    sessions = defaultdict(list)
    for e in events:
        if e.get("event_type") in ("tool_call", "tool_result"):
            sessions[e.get("session_id", "?")].append(e)

    # Rule checks
    total_sessions = len(sessions)
    search_first = sum(
        1
        for evts in sessions.values()
        if any(
            e.get("data", {}).get("tool_name") == "code_search"
            for e in evts
            if e.get("event_type") == "tool_call"
        )
        and next(
            (
                e.get("data", {}).get("tool_name")
                for e in evts
                if e.get("event_type") == "tool_call"
            ),
            None,
        )
        == "code_search"
    )

    shell_total = sum(
        1
        for e in events
        if e.get("event_type") == "tool_call" and e.get("data", {}).get("tool_name") == "shell"
    )
    shell_search = sum(
        1
        for e in events
        if e.get("event_type") == "tool_call"
        and e.get("data", {}).get("tool_name") == "shell"
        and any(
            s in str(e.get("data", {}).get("tool_args", {}).get("cmd", "")).lower()
            for s in ["grep ", "rg ", " ag "]
        )
    )

    total_reads = sum(
        1
        for e in events
        if e.get("event_type") == "tool_call" and e.get("data", {}).get("tool_name") == "read"
    )
    victor_reads = sum(
        1
        for e in events
        if e.get("event_type") == "tool_call"
        and e.get("data", {}).get("tool_name") == "read"
        and "victor" in str(e.get("data", {}).get("tool_args", {}).get("path", "")).lower()
        and "swe_bench" not in str(e.get("data", {}).get("tool_args", {}))
    )

    cs_total = sum(
        1
        for e in events
        if e.get("event_type") == "tool_call"
        and e.get("data", {}).get("tool_name") == "code_search"
    )
    cs_semantic = sum(
        1
        for e in events
        if e.get("event_type") == "tool_call"
        and e.get("data", {}).get("tool_name") == "code_search"
        and e.get("data", {}).get("tool_args", {}).get("mode", "semantic") == "semantic"
    )

    table = Table(title="GEPA Compliance Scorecard")
    table.add_column("Rule", style="cyan")
    table.add_column("Compliance", style="bold")
    table.add_column("Target", style="dim")

    rules = [
        ("Search first", search_first * 100 // max(total_sessions, 1), 50),
        (
            "No shell search",
            ((shell_total - shell_search) * 100 // max(shell_total, 1) if shell_total else 100),
            80,
        ),
        (
            "No workspace contamination",
            (total_reads - victor_reads) * 100 // max(total_reads, 1),
            95,
        ),
        (
            "Semantic search preferred",
            cs_semantic * 100 // max(cs_total, 1) if cs_total else 100,
            80,
        ),
    ]

    for name, pct, target in rules:
        status = (
            "[green]✓[/]"
            if pct >= target
            else "[yellow]⚠[/]" if pct >= target * 0.6 else "[red]✗[/]"
        )
        table.add_row(f"{status} {name}", f"{pct}%", f"{target}%")

    console.print("\n")
    console.print(table)

    # Efficiency scaling: tool calls per session over time
    session_tools = []
    for sid, evts in sessions.items():
        tc = sum(1 for e in evts if e.get("event_type") == "tool_call")
        if tc > 0:
            session_tools.append(tc)

    if len(session_tools) >= 4:
        first_half = session_tools[: len(session_tools) // 2]
        second_half = session_tools[len(session_tools) // 2 :]
        avg_first = sum(first_half) / len(first_half)
        avg_second = sum(second_half) / len(second_half)
        delta = avg_second - avg_first
        direction = "↓" if delta < 0 else "↑" if delta > 0 else "="
        console.print(
            f"[dim]Efficiency: {avg_first:.0f} → {avg_second:.0f} tools/session "
            f"({direction}{abs(delta):.0f}) across {len(session_tools)} sessions[/]"
        )

    console.print(f"[dim]Based on {len(events):,} events across {total_sessions} sessions[/]")
