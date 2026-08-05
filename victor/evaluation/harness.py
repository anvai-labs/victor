from __future__ import annotations

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

"""Evaluation harness for running benchmarks.

Provides infrastructure for evaluating coding agents against
standardized benchmarks like SWE-bench, HumanEval, etc.
"""

import asyncio
import json
import logging
import math
import os
from contextlib import contextmanager

from victor.observability.run_kind import RunKind, run_kind_scope
import shutil
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional, Protocol, runtime_checkable

from victor.evaluation.protocol import (
    BenchmarkFailureCategory,
    ConfidenceAssessment,
    FailureDiagnosis,
    BenchmarkTask,
    BenchmarkType,
    EvaluationConfig,
    EvaluationResult,
    TaskResult,
    TaskStatus,
)
from victor.evaluation.experiment_analyzer import analyze_evaluation_result
from victor.evaluation.experiment_memory import (
    ExperimentMemoryRecord,
    ExperimentMemoryStore,
)
from victor.evaluation.runtime_feedback import (
    build_runtime_evaluation_feedback_payload,
    derive_runtime_evaluation_feedback,
)
from victor.evaluation.validated_session_truth_emitters import (
    ValidatedSessionTruthEmitterRegistry,
)
from victor.evaluation.services import (
    materialize_validated_session_truth_service,
    ValidatedSessionTruthServiceProtocol,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PromptCandidateEvaluationSpec:
    """One exact prompt candidate to evaluate in isolation."""

    section_name: str
    prompt_candidate_hash: str
    provider: Optional[str] = None
    label: Optional[str] = None

    def resolved_label(self, default_provider: Optional[str] = None) -> str:
        """Return a stable human-readable label for reports and logs."""
        provider = (self.provider or default_provider or "").strip() or "default"
        return self.label or f"{self.section_name}:{provider}:{self.prompt_candidate_hash}"


@dataclass(frozen=True)
class PromptCandidateEvaluationRun:
    """One executed candidate-bound evaluation run."""

    spec: PromptCandidateEvaluationSpec
    config: EvaluationConfig
    result: EvaluationResult
    label: str


@dataclass(frozen=True)
class PairedContrast:
    """One variant measured against the baseline on the *same* tasks.

    A prompt candidate has never been compared to anything. It is scored on its
    own absolute pass rate, and the gate asks only whether that rate clears a
    fixed threshold — so a candidate can be approved while being worse than the
    prompt it replaces, or rejected while beating it on a hard benchmark. The
    number that matters is not "did it pass 50%" but "did it pass tasks the seed
    failed, more often than the reverse".

    Pairing is what makes that answerable at this sample size. Both arms run the
    identical task set, so task difficulty cancels: only the tasks where the two
    arms *disagree* carry information, and a handful of those says more than a
    difference of two percentage points across forty independent tasks.
    """

    n_paired: int
    variant_only_pass: int
    baseline_only_pass: int
    both_pass: int
    both_fail: int

    @property
    def discordant(self) -> int:
        """Tasks where exactly one arm passed — the only ones carrying signal."""
        return self.variant_only_pass + self.baseline_only_pass

    @property
    def effect(self) -> int:
        """Net tasks the variant gained. Negative means it lost ground."""
        return self.variant_only_pass - self.baseline_only_pass

    @property
    def noise_floor(self) -> float:
        """How large a lead a fair coin would produce on this many disagreements.

        Under the null the variant wins each discordant task with probability
        one half, so the effect has mean zero and standard deviation
        ``sqrt(discordant)``. A lead smaller than that is indistinguishable from
        chance however many tasks were run.

        This is why "positive effect over enough disagreements" was not a
        sufficient gate: a candidate cleared it at 8 versus 6 — an effect of two
        against a noise floor of 3.7, which the exact test scored p=0.79. The
        floor scales with the evidence, so more tasks demand a bigger lead in
        absolute terms while making a genuine one easier to reach.
        """
        return math.sqrt(self.discordant)

    def beats_noise(self) -> bool:
        """True when the lead exceeds what chance alone would produce."""
        return self.effect >= self.noise_floor and self.effect > 0

    @property
    def mcnemar_p(self) -> float:
        """Two-sided exact p for the discordant split under a fair coin.

        Exact rather than the chi-square approximation because the counts here
        are small by construction — with 40 tasks a typical split is 11 vs 3, and
        the approximation is unreliable below ~25 discordant pairs. `math.comb`
        keeps this dependency-free; scipy is not worth a wheel for one binomial.
        """
        n = self.discordant
        if n == 0:
            return 1.0
        extreme = max(self.variant_only_pass, self.baseline_only_pass)
        tail = sum(math.comb(n, i) for i in range(extreme, n + 1)) / (2**n)
        return min(1.0, 2.0 * tail)

    @classmethod
    def from_results(
        cls,
        baseline: "EvaluationResult",
        variant: "EvaluationResult",
    ) -> "PairedContrast":
        """Pair two arms on task id. Raises when they share no tasks.

        Silently returning an empty contrast would read downstream as "no
        difference" — indistinguishable from a real tie — when the truth is that
        the comparison never happened.
        """
        base_by_task = _passed_by_task(baseline)
        variant_by_task = _passed_by_task(variant)
        shared = sorted(set(base_by_task) & set(variant_by_task))
        if not shared:
            raise ValueError(
                "Paired contrast needs a shared task set; baseline ran "
                f"{len(base_by_task)} task(s) and the variant ran {len(variant_by_task)} "
                "with no overlap."
            )

        counts = {"bb": 0, "vv": 0, "both": 0, "neither": 0}
        for task_id in shared:
            b, v = base_by_task[task_id], variant_by_task[task_id]
            if v and not b:
                counts["vv"] += 1
            elif b and not v:
                counts["bb"] += 1
            elif b and v:
                counts["both"] += 1
            else:
                counts["neither"] += 1

        return cls(
            n_paired=len(shared),
            variant_only_pass=counts["vv"],
            baseline_only_pass=counts["bb"],
            both_pass=counts["both"],
            both_fail=counts["neither"],
        )

    def summary(self) -> str:
        """One line for the CLI: the effect, the split it rests on, and the p."""
        sign = "+" if self.effect >= 0 else ""
        return (
            f"{sign}{self.effect}/{self.n_paired} "
            f"(disc {self.variant_only_pass}/{self.baseline_only_pass}, "
            f"p={self.mcnemar_p:.2f})"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_paired": self.n_paired,
            "variant_only_pass": self.variant_only_pass,
            "baseline_only_pass": self.baseline_only_pass,
            "both_pass": self.both_pass,
            "both_fail": self.both_fail,
            "discordant": self.discordant,
            "effect": self.effect,
            "mcnemar_p": self.mcnemar_p,
        }


def was_throttled(task: Any) -> bool:
    """True when this task failed because the provider refused, not the prompt.

    A task the agent never got a model response for says nothing about the
    prompt under test. Left in, it reads as a failure caused by the candidate.
    """
    from victor.framework.rl.mutator_rotation import is_rate_limit

    if getattr(task, "status", None) == TaskStatus.PASSED:
        return False
    haystack = " ".join(
        str(part)
        for part in (
            getattr(task, "error_message", ""),
            getattr(task, "stderr", ""),
            getattr(task, "failure_details", {}),
        )
    )
    return is_rate_limit(haystack)


def _passed_by_task(result: "EvaluationResult") -> dict[str, bool]:
    """Map task id to whether it passed, for one arm.

    Throttled tasks are omitted, which drops the whole pair — a task only
    counts when *both* arms got a real attempt at it. Running three arms back to
    back exhausted one provider's quota and the last arm recorded 0/24 in 128
    seconds: ten explicit 429s and an agent that never received a response. That
    is an aborted arm, not a candidate that lost, and scoring it as a loss is
    the same defect as the mutator's invisible 429 — a quota failure wearing the
    costume of a quality signal.

    Later results win on duplicate ids: a resumed run appends rather than
    replaces, so the last attempt is the one that counts.
    """
    return {
        task.task_id: task.status == TaskStatus.PASSED
        for task in getattr(result, "task_results", []) or []
        if getattr(task, "task_id", "") and not was_throttled(task)
    }


@dataclass
class PromptCandidateEvaluationSuiteResult:
    """Collected results from evaluating multiple prompt candidates separately."""

    base_config: EvaluationConfig
    runs: list[PromptCandidateEvaluationRun]

    def best_run(self) -> Optional[PromptCandidateEvaluationRun]:
        """Return the strongest run by pass rate, then task count, then shortest duration."""
        if not self.runs:
            return None
        return max(
            self.runs,
            key=lambda run: (
                run.result.pass_rate,
                run.result.total_tasks,
                -run.result.duration_seconds,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the suite summary for reporting or persistence."""
        best = self.best_run()
        return {
            "benchmark": self.base_config.benchmark.value,
            "model": self.base_config.model,
            "provider": self.base_config.provider,
            "best_label": best.label if best is not None else None,
            "runs": [
                {
                    "label": run.label,
                    "provider": run.config.provider,
                    "prompt_candidate_hash": run.config.prompt_candidate_hash,
                    "section_name": run.config.prompt_section_name,
                    "metrics": run.result.get_metrics(),
                    # Per-task outcomes, not just the aggregate. A paired
                    # contrast needs to know *which* tasks each arm passed;
                    # ``from_dict`` has always read this key, and nothing wrote it.
                    "task_results": [
                        task_result_to_artifact(task) for task in (run.result.task_results or [])
                    ],
                }
                for run in self.runs
            ],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PromptCandidateEvaluationSuiteResult":
        """Reconstruct a suite result from a saved suite artifact payload."""
        if not isinstance(payload, dict):
            raise ValueError("suite payload must be a dictionary")

        config_payload = payload.get("config")
        config_dict = dict(config_payload) if isinstance(config_payload, dict) else {}
        config_dict.setdefault("benchmark", payload.get("benchmark"))
        config_dict.setdefault("model", payload.get("model"))
        config_dict.setdefault("provider", payload.get("provider"))
        config_dict.setdefault(
            "prompt_section_name",
            payload.get("prompt_section_name") or payload.get("section_name"),
        )
        base_config = evaluation_config_from_artifact_config(config_dict)

        runs: list[PromptCandidateEvaluationRun] = []
        for run_payload in payload.get("runs", []) or []:
            if not isinstance(run_payload, dict):
                continue

            provider = str(run_payload.get("provider") or base_config.provider or "") or None
            section_name = str(
                run_payload.get("section_name")
                or run_payload.get("prompt_section_name")
                or base_config.prompt_section_name
                or ""
            )
            prompt_candidate_hash = str(run_payload.get("prompt_candidate_hash") or "")
            run_config = replace(
                base_config,
                provider=provider,
                prompt_candidate_hash=prompt_candidate_hash or None,
                prompt_section_name=section_name or None,
            )
            spec = PromptCandidateEvaluationSpec(
                section_name=section_name,
                prompt_candidate_hash=prompt_candidate_hash,
                provider=provider,
                label=(
                    str(run_payload.get("label")) if run_payload.get("label") is not None else None
                ),
            )
            task_results = [
                task_result_from_artifact(task_payload)
                for task_payload in run_payload.get("task_results", []) or []
                if isinstance(task_payload, dict)
            ]
            result = EvaluationResult(config=run_config, task_results=task_results)
            label = (
                str(run_payload.get("label"))
                if run_payload.get("label") is not None
                else spec.resolved_label(run_config.provider)
            )
            runs.append(
                PromptCandidateEvaluationRun(
                    spec=spec,
                    config=run_config,
                    result=result,
                    label=label,
                )
            )

        return cls(base_config=base_config, runs=runs)


def evaluation_config_from_artifact_config(payload: dict[str, Any]) -> EvaluationConfig:
    """Reconstruct evaluation config from a saved artifact config payload."""
    benchmark_value = str(payload.get("benchmark") or BenchmarkType.CUSTOM.value)
    return EvaluationConfig(
        benchmark=BenchmarkType(benchmark_value),
        model=str(payload.get("model") or "unknown"),
        provider=(str(payload.get("provider")) if payload.get("provider") is not None else None),
        prompt_candidate_hash=(
            str(payload.get("prompt_candidate_hash"))
            if payload.get("prompt_candidate_hash") is not None
            else None
        ),
        prompt_section_name=(
            str(payload.get("prompt_section_name") or payload.get("section_name"))
            if (payload.get("prompt_section_name") or payload.get("section_name")) is not None
            else None
        ),
        max_tasks=payload.get("max_tasks"),
        timeout_per_task=int(payload.get("timeout_per_task", 300) or 300),
        parallel_tasks=int(payload.get("parallel_tasks", 1) or 1),
        max_turns=int(payload.get("max_turns", 10) or 10),
        dataset_metadata=dict(payload.get("dataset_metadata") or {}),
    )


def task_result_to_artifact(task_result: Any) -> dict[str, Any]:
    """Serialize a TaskResult into the shape ``task_result_from_artifact`` reads.

    Lives beside its inverse on purpose. The only writer used to be a private
    method on the real-run runner, so the suite serializer had no way to persist
    per-task outcomes and wrote aggregate metrics alone — while ``from_dict``
    read a ``task_results`` key nobody produced. A suite therefore round-tripped
    to zero tasks, which is fatal for a paired contrast: pairing needs to know
    *which* tasks each arm passed, and pass rates alone cannot say.
    """
    status = getattr(task_result, "status", None)
    failure_category = getattr(task_result, "failure_category", None)
    return {
        "task_id": getattr(task_result, "task_id", None),
        "status": getattr(status, "value", status),
        # Correlation spine — joins this task's decisions to its outcome.
        "session_id": getattr(task_result, "session_id", ""),
        "tests_passed": getattr(task_result, "tests_passed", 0),
        "tests_total": getattr(task_result, "tests_total", 0),
        "duration": getattr(task_result, "duration_seconds", 0.0),
        "duration_seconds": getattr(task_result, "duration_seconds", 0.0),
        "tokens_used": getattr(task_result, "tokens_used", 0),
        "tokens_input": getattr(task_result, "tokens_input", 0),
        "tokens_output": getattr(task_result, "tokens_output", 0),
        "cached_tokens": getattr(task_result, "cached_tokens", 0),
        "reasoning_tokens": getattr(task_result, "reasoning_tokens", 0),
        "cost_usd_micros": getattr(task_result, "cost_usd_micros", 0),
        "tool_calls": getattr(task_result, "tool_calls", 0),
        "turns": getattr(task_result, "turns", 0),
        "code_search_calls": getattr(task_result, "code_search_calls", 0),
        "graph_calls": getattr(task_result, "graph_calls", 0),
        "completion_score": getattr(task_result, "completion_score", 0.0),
        "failure_category": getattr(failure_category, "value", failure_category),
        "failure_details": dict(getattr(task_result, "failure_details", {}) or {}),
        "metadata": dict(getattr(task_result, "metadata", {}) or {}),
    }


def task_result_from_artifact(payload: dict[str, Any]) -> TaskResult:
    """Reconstruct a task result from a saved suite artifact entry."""
    failure_category_value = payload.get("failure_category")
    failure_category = (
        BenchmarkFailureCategory(str(failure_category_value)) if failure_category_value else None
    )
    tests_passed = int(payload.get("tests_passed", 0) or 0)
    tests_total = int(payload.get("tests_total", 0) or 0)
    failure_diagnosis_payload = payload.get("failure_diagnosis")
    confidence_payload = payload.get("confidence_assessment")
    return TaskResult(
        task_id=str(payload.get("task_id") or "unknown-task"),
        status=TaskStatus(str(payload.get("status") or TaskStatus.ERROR.value)),
        generated_code=str(payload.get("generated_code", "") or ""),
        generated_patch=str(payload.get("generated_patch", "") or ""),
        tests_passed=tests_passed,
        tests_failed=max(tests_total - tests_passed, 0),
        tests_total=tests_total,
        duration_seconds=float(
            payload.get("duration_seconds", payload.get("duration", 0.0)) or 0.0
        ),
        tokens_used=int(payload.get("tokens_used", 0) or 0),
        tokens_input=int(payload.get("tokens_input", 0) or 0),
        tokens_output=int(payload.get("tokens_output", 0) or 0),
        cached_tokens=int(payload.get("cached_tokens", 0) or 0),
        reasoning_tokens=int(payload.get("reasoning_tokens", 0) or 0),
        cost_usd_micros=int(payload.get("cost_usd_micros", 0) or 0),
        tool_calls=int(payload.get("tool_calls", 0) or 0),
        turns=int(payload.get("turns", 0) or 0),
        code_search_calls=int(payload.get("code_search_calls", 0) or 0),
        graph_calls=int(payload.get("graph_calls", 0) or 0),
        completion_score=float(payload.get("completion_score", 0.0) or 0.0),
        error_message=str(payload.get("error_message", "") or ""),
        failure_category=failure_category,
        failure_details=dict(payload.get("failure_details") or {}),
        metadata=dict(payload.get("metadata") or {}),
        failure_diagnosis=(
            FailureDiagnosis.from_dict(failure_diagnosis_payload)
            if isinstance(failure_diagnosis_payload, dict)
            else None
        ),
        confidence_assessment=(
            ConfidenceAssessment.from_dict(confidence_payload)
            if isinstance(confidence_payload, dict)
            else None
        ),
    )


def load_prompt_candidate_evaluation_suite(
    path: str | Path,
) -> PromptCandidateEvaluationSuiteResult:
    """Load a saved prompt-candidate suite artifact from disk."""
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError("suite artifact must contain a JSON object")
    return PromptCandidateEvaluationSuiteResult.from_dict(payload)


@contextmanager
def durable_evaluation_conversations(db_path: Optional[Path] = None):
    """Pin evaluation conversations to a durable, eval-owned database.

    Each task runs inside a ``tempfile.mkdtemp()`` workspace and the agent
    chdirs into it, while ``ConversationStore`` derives its path from the
    current working directory. Evaluation conversations were therefore written
    into a directory deleted moments later — or, when project paths had already
    been cached, into whichever repo the run started from, mixing benchmark
    sessions into a developer's own history. Neither outcome leaves the data
    where anyone can read it back.

    Pinning to ``{global}/evaluations/sessions.db`` makes eval conversations
    survive the workspace and keeps them out of project stores. An explicit
    ``VICTOR_CONVERSATION_DB`` already set by the caller is left alone.

    Restores the previous value on exit, so an in-process caller (the test
    suite, a notebook) is not left with a changed global.
    """
    from victor.agent.conversation.store import CONVERSATION_DB_ENV

    previous = os.environ.get(CONVERSATION_DB_ENV)
    if previous:
        yield Path(previous)
        return

    if db_path is None:
        try:
            from victor.config.settings import get_project_paths

            db_path = Path(get_project_paths().global_victor_dir) / "evaluations" / "sessions.db"
        except Exception:
            db_path = Path.home() / ".victor" / "evaluations" / "sessions.db"

    try:
        db_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        # A read-only or missing home should degrade to old behaviour, not
        # abort the benchmark run.
        logger.warning("Could not prepare durable evaluation store at %s: %s", db_path, exc)
        yield None
        return

    os.environ[CONVERSATION_DB_ENV] = str(db_path)
    logger.info("Evaluation conversations pinned to %s", db_path)
    try:
        yield db_path
    finally:
        if previous is None:
            os.environ.pop(CONVERSATION_DB_ENV, None)
        else:
            os.environ[CONVERSATION_DB_ENV] = previous


def _collect_observed_prompt_identities(result: "EvaluationResult") -> list[dict[str, Any]]:
    """Union the prompt identities served across every task in a run.

    An artifact previously named a candidate only when the operator passed
    ``--prompt-candidate-hash``, so ordinary benchmark runs produced ground-truth
    pass/fail that no candidate could ever be credited with. Recording what was
    actually served makes every run usable as Pareto evidence.

    Deduplicated on (section, provider, hash) and order-stable so artifacts diff
    cleanly across runs.
    """
    observed: dict[tuple[Any, Any, Any], dict[str, Any]] = {}
    for task_result in result.task_results:
        metadata = getattr(task_result, "metadata", None)
        if not isinstance(metadata, dict):
            continue
        for identity in metadata.get("prompt_identities") or []:
            if not isinstance(identity, dict):
                continue
            candidate_hash = identity.get("prompt_candidate_hash")
            if not candidate_hash:
                continue
            section = identity.get("prompt_section_name") or identity.get("section_name")
            observed.setdefault(
                (section, identity.get("provider"), candidate_hash),
                dict(identity),
            )
    return list(observed.values())


def bind_prompt_candidate_evaluation_config(
    base_config: EvaluationConfig,
    spec: PromptCandidateEvaluationSpec,
) -> EvaluationConfig:
    """Create a candidate-bound config from a benchmark base config."""
    return replace(
        base_config,
        provider=(spec.provider or base_config.provider),
        prompt_candidate_hash=spec.prompt_candidate_hash,
        prompt_section_name=spec.section_name,
    )


@runtime_checkable
class BenchmarkRunner(Protocol):
    """Protocol for benchmark runners."""

    @property
    def benchmark_type(self) -> BenchmarkType:
        """The type of benchmark this runner handles."""
        ...

    async def load_tasks(
        self,
        config: EvaluationConfig,
    ) -> list[BenchmarkTask]:
        """Load benchmark tasks."""
        ...

    async def run_task(
        self,
        task: BenchmarkTask,
        agent_output: str,
        config: EvaluationConfig,
    ) -> TaskResult:
        """Run a single task and evaluate the result."""
        ...


class BaseBenchmarkRunner(ABC):
    """Abstract base class for benchmark runners."""

    @property
    @abstractmethod
    def benchmark_type(self) -> BenchmarkType:
        """The type of benchmark this runner handles."""
        ...

    @abstractmethod
    async def load_tasks(
        self,
        config: EvaluationConfig,
    ) -> list[BenchmarkTask]:
        """Load benchmark tasks."""
        ...

    @abstractmethod
    async def run_task(
        self,
        task: BenchmarkTask,
        agent_output: str,
        config: EvaluationConfig,
    ) -> TaskResult:
        """Run a single task and evaluate the result."""
        ...

    def _filter_tasks(
        self,
        tasks: list[BenchmarkTask],
        config: EvaluationConfig,
    ) -> list[BenchmarkTask]:
        """Filter tasks based on config."""
        filtered = tasks

        if config.task_ids:
            filtered = [t for t in filtered if t.task_id in config.task_ids]

        if config.languages:
            filtered = [t for t in filtered if t.language in config.languages]

        if config.categories:
            filtered = [t for t in filtered if t.category in config.categories]

        if config.difficulties:
            filtered = [t for t in filtered if t.difficulty in config.difficulties]

        if config.max_tasks:
            filtered = filtered[: config.max_tasks]

        return filtered


class TaskEnvironment:
    """Isolated environment for running benchmark tasks."""

    def __init__(
        self,
        task: BenchmarkTask,
        workspace_dir: Optional[Path] = None,
        use_docker: bool = False,
        docker_image: str = "python:3.11",
    ):
        """Initialize the task environment.

        Args:
            task: The benchmark task
            workspace_dir: Base directory for workspaces
            use_docker: Whether to use Docker isolation
            docker_image: Docker image to use
        """
        self.task = task
        self.workspace_dir = workspace_dir or Path(tempfile.gettempdir())
        self.use_docker = use_docker
        self.docker_image = docker_image
        self._temp_dir: Optional[Path] = None

    async def setup(self) -> Path:
        """Set up the task environment.

        Returns:
            Path to the workspace directory
        """
        # Create temporary directory (sanitize task_id to be valid for directory names)
        safe_task_id = self.task.task_id.replace("/", "_").replace("\\", "_")
        self._temp_dir = Path(
            tempfile.mkdtemp(
                prefix=f"eval_{safe_task_id}_",
                dir=self.workspace_dir,
            )
        )

        # Clone repo if specified (for SWE-bench)
        if self.task.repo:
            await self._clone_repo()

        target_dir = (
            self._temp_dir / "repo" if (self._temp_dir / "repo").exists() else self._temp_dir
        )

        # Seed arbitrary workspace files before writing canonical solution/test entrypoints.
        if self.task.seed_files:
            self._write_seed_files(target_dir, self.task.seed_files)

        # Write context code
        if self.task.context_code:
            code_file = target_dir / "solution.py"
            code_file.write_text(self.task.context_code)

        # Write test code
        if self.task.test_code:
            test_file = target_dir / "test_solution.py"
            test_file.write_text(self.task.test_code)

        return self._temp_dir

    async def _clone_repo(self) -> None:
        """Clone the task's repository.

        For SWE-bench tasks, we need to clone the full repo (or enough history)
        to be able to checkout the specific base_commit.
        """
        if not self.task.repo or not self._temp_dir:
            return

        repo_dir = self._temp_dir / "repo"

        # Clone with enough history to access base_commit
        # Using --depth 100 as compromise between speed and history
        clone_cmd = ["git", "clone", "--depth", "100", self.task.repo, str(repo_dir)]

        try:
            logger.info(f"Cloning {self.task.repo}...")
            result = await asyncio.create_subprocess_exec(
                *clone_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await result.communicate()

            if result.returncode != 0:
                # Try full clone if shallow clone failed
                logger.warning(f"Shallow clone failed, trying full clone: {stderr.decode()}")
                clone_cmd = ["git", "clone", self.task.repo, str(repo_dir)]
                result = await asyncio.create_subprocess_exec(
                    *clone_cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                await result.communicate()

            # Checkout specific commit if provided
            if self.task.base_commit and repo_dir.exists():
                # Fetch the specific commit if not available
                fetch_cmd = [
                    "git",
                    "fetch",
                    "--depth",
                    "1",
                    "origin",
                    self.task.base_commit,
                ]
                result = await asyncio.create_subprocess_exec(
                    *fetch_cmd,
                    cwd=repo_dir,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                await result.communicate()

                # Checkout the commit
                checkout_cmd = ["git", "checkout", self.task.base_commit]
                result = await asyncio.create_subprocess_exec(
                    *checkout_cmd,
                    cwd=repo_dir,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await result.communicate()
                if result.returncode != 0:
                    logger.warning(f"Failed to checkout {self.task.base_commit}: {stderr.decode()}")
                else:
                    logger.info(f"Checked out {self.task.base_commit}")

        except Exception as e:
            logger.warning(f"Failed to clone repo: {e}")

    def _write_seed_files(self, base_dir: Path, files: dict[str, str]) -> None:
        """Write manifest-provided seed files into the task workspace."""
        base_dir = base_dir.resolve()
        for rel_path, content in files.items():
            target = (base_dir / rel_path).resolve()
            try:
                target.relative_to(base_dir)
            except ValueError as exc:
                raise ValueError(f"Unsafe seed file path: {rel_path}") from exc
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)

    async def apply_patch(self, patch: str) -> bool:
        """Apply a patch to the repository.

        Args:
            patch: The patch content

        Returns:
            True if patch applied successfully
        """
        if not self._temp_dir:
            return False

        patch_file = self._temp_dir / "solution.patch"
        patch_file.write_text(patch)

        try:
            result = await asyncio.create_subprocess_exec(
                "git",
                "apply",
                str(patch_file),
                cwd=(
                    self._temp_dir / "repo"
                    if (self._temp_dir / "repo").exists()
                    else self._temp_dir
                ),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await result.wait()
            return result.returncode == 0
        except Exception as e:
            logger.warning(f"Failed to apply patch: {e}")
            return False

    async def run_tests(self, timeout: int = 300) -> tuple[int, int, str, str]:
        """Run tests in the environment.

        Args:
            timeout: Timeout in seconds

        Returns:
            Tuple of (passed, total, stdout, stderr)
        """
        if not self._temp_dir:
            return 0, 0, "", "Environment not set up"

        test_dir = self._temp_dir / "repo" if (self._temp_dir / "repo").exists() else self._temp_dir

        if self.use_docker:
            return await self._run_tests_docker(test_dir, timeout)
        else:
            return await self._run_tests_local(test_dir, timeout)

    async def _install_task_repo(self, test_dir: Path, install_timeout: int) -> None:
        """Best-effort ``pip install -e .`` of the task repo.

        SWE-bench repos must be installed for their package to be importable
        during test collection; otherwise pytest collects 0 tests. Best-effort:
        if install fails or times out, the test run still proceeds (it may
        collect partially or 0).
        """
        if not any((test_dir / f).exists() for f in ("setup.py", "pyproject.toml", "setup.cfg")):
            return
        try:
            proc = await asyncio.create_subprocess_exec(
                "python",
                "-m",
                "pip",
                "install",
                "-e",
                ".",
                "--no-input",
                "--disable-pip-version-check",
                cwd=test_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**os.environ, "PIP_NO_BUILD_ISOLATION": "1"},
            )
            try:
                await asyncio.wait_for(proc.communicate(), timeout=install_timeout)
            except asyncio.TimeoutError:
                proc.kill()
        except Exception:
            pass  # best-effort

    async def _run_tests_local(
        self,
        test_dir: Path,
        timeout: int,
    ) -> tuple[int, int, str, str]:
        """Run tests locally."""
        try:
            # SWE-bench repos must be installed (pip install -e .) for their
            # package to be importable during test collection — otherwise pytest
            # collects 0 tests ("Tests not collected"). Best-effort: if install
            # fails, still attempt the test run.
            await self._install_task_repo(test_dir, install_timeout=max(60, timeout // 2))

            # Detect test framework
            if (test_dir / "pytest.ini").exists() or (test_dir / "setup.py").exists():
                cmd = ["python", "-m", "pytest", "-v", "--tb=short"]
            else:
                cmd = ["python", "-m", "unittest", "discover", "-v"]

            result = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=test_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    result.communicate(),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                result.kill()
                return 0, 0, "", "Timeout"

            stdout_str = stdout.decode()
            stderr_str = stderr.decode()

            # Parse test results
            passed, total = self._parse_test_output(stdout_str + stderr_str)

            return passed, total, stdout_str, stderr_str

        except Exception as e:
            return 0, 0, "", str(e)

    async def _run_tests_docker(
        self,
        test_dir: Path,
        timeout: int,
    ) -> tuple[int, int, str, str]:
        """Run tests in Docker container."""
        try:
            cmd = [
                "docker",
                "run",
                "--rm",
                "-v",
                f"{test_dir}:/workspace",
                "-w",
                "/workspace",
                "--network",
                "none",
                self.docker_image,
                "python",
                "-m",
                "pytest",
                "-v",
                "--tb=short",
            ]

            result = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    result.communicate(),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                # Kill container
                await asyncio.create_subprocess_exec(
                    "docker",
                    "kill",
                    str(result.pid),
                )
                return 0, 0, "", "Timeout"

            stdout_str = stdout.decode()
            stderr_str = stderr.decode()
            passed, total = self._parse_test_output(stdout_str + stderr_str)

            return passed, total, stdout_str, stderr_str

        except Exception as e:
            return 0, 0, "", str(e)

    def _parse_test_output(self, output: str) -> tuple[int, int]:
        """Parse test output to extract pass/fail counts."""
        import re

        # pytest format: "5 passed, 2 failed"
        pytest_match = re.search(r"(\d+) passed", output)
        pytest_failed = re.search(r"(\d+) failed", output)

        if pytest_match:
            passed = int(pytest_match.group(1))
            failed = int(pytest_failed.group(1)) if pytest_failed else 0
            return passed, passed + failed

        # unittest format: "Ran X tests"
        unittest_match = re.search(r"Ran (\d+) test", output)
        if unittest_match:
            total = int(unittest_match.group(1))
            failures = re.search(r"failures=(\d+)", output)
            errors = re.search(r"errors=(\d+)", output)
            failed = (int(failures.group(1)) if failures else 0) + (
                int(errors.group(1)) if errors else 0
            )
            return total - failed, total

        return 0, 0

    async def cleanup(self) -> None:
        """Clean up the task environment."""
        if self._temp_dir and self._temp_dir.exists():
            try:
                shutil.rmtree(self._temp_dir)
            except Exception as e:
                logger.warning(f"Failed to cleanup: {e}")


class EvaluationHarness:
    """Main harness for running evaluations."""

    def __init__(
        self,
        runners: Optional[dict[BenchmarkType, BaseBenchmarkRunner]] = None,
        checkpoint_dir: Optional[Path] = None,
        validated_session_truth_service: Optional[ValidatedSessionTruthServiceProtocol] = None,
        **legacy_kwargs: Any,
    ):
        """Initialize the harness.

        Args:
            runners: Dict mapping benchmark types to runners
            checkpoint_dir: Directory for checkpoint files (defaults to ~/.victor/checkpoints)
            validated_session_truth_service: Service for validated session-truth orchestration
        """
        self._runners = runners or {}
        self._validated_session_truth_service = materialize_validated_session_truth_service(
            service=validated_session_truth_service,
            legacy_kwargs=legacy_kwargs,
        )
        try:
            from victor.config.secure_paths import get_victor_dir

            victor_dir = get_victor_dir()
            self._results_dir = victor_dir / "evaluations"
            self._checkpoint_dir = checkpoint_dir or victor_dir / "checkpoints"
        except ImportError:
            self._results_dir = Path.home() / ".victor" / "evaluations"
            self._checkpoint_dir = checkpoint_dir or Path.home() / ".victor" / "checkpoints"
        try:
            self._results_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning(
                "Failed to prepare evaluation results directory %s: %s",
                self._results_dir,
                exc,
            )
        try:
            self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning(
                "Failed to prepare evaluation checkpoint directory %s: %s",
                self._checkpoint_dir,
                exc,
            )

    def register_runner(self, runner: BaseBenchmarkRunner) -> None:
        """Register a benchmark runner.

        Args:
            runner: The runner to register
        """
        self._runners[runner.benchmark_type] = runner

    def get_runner(self, benchmark_type: BenchmarkType) -> Optional[BaseBenchmarkRunner]:
        """Get a runner by benchmark type.

        Args:
            benchmark_type: The benchmark type

        Returns:
            The runner or None
        """
        return self._runners.get(benchmark_type)

    def _get_checkpoint_path(self, config: EvaluationConfig) -> Path:
        """Get checkpoint file path for a config.

        Creates a unique checkpoint ID based on benchmark, model, and task config.
        """
        import hashlib

        # Build unique ID from config
        config_str = f"{config.benchmark.value}_{config.model}_{config.max_tasks}"
        if config.provider:
            config_str += f"_{config.provider}"
        if config.prompt_section_name:
            config_str += f"_{config.prompt_section_name}"
        if config.prompt_candidate_hash:
            config_str += f"_{config.prompt_candidate_hash}"
        if config.task_ids:
            config_str += "_" + "_".join(sorted(config.task_ids))
        checkpoint_id = hashlib.md5(config_str.encode()).hexdigest()[:12]

        return self._checkpoint_dir / f"checkpoint_{config.benchmark.value}_{checkpoint_id}.json"

    def _save_checkpoint(
        self,
        checkpoint_path: Path,
        config: EvaluationConfig,
        completed_results: list[TaskResult],
        remaining_task_ids: list[str],
        start_time: datetime,
    ) -> None:
        """Save checkpoint to disk after each task completion."""
        import json

        checkpoint_data = {
            "config": config.to_artifact_config(),
            "start_time": start_time.isoformat(),
            "completed_task_ids": [r.task_id for r in completed_results],
            "remaining_task_ids": remaining_task_ids,
            "results": [
                {
                    "task_id": r.task_id,
                    "status": r.status.value,
                    "tests_passed": r.tests_passed,
                    "tests_failed": r.tests_failed,
                    "tests_total": r.tests_total,
                    "duration_seconds": r.duration_seconds,
                    "tokens_used": r.tokens_used,
                    "tokens_input": r.tokens_input,
                    "tokens_output": r.tokens_output,
                    "cached_tokens": r.cached_tokens,
                    "reasoning_tokens": r.reasoning_tokens,
                    "cost_usd_micros": r.cost_usd_micros,
                    "tool_calls": r.tool_calls,
                    "turns": r.turns,
                    "code_search_calls": r.code_search_calls,
                    "graph_calls": r.graph_calls,
                    "completion_score": r.completion_score,
                    "error_message": r.error_message,
                    "failure_category": (r.failure_category.value if r.failure_category else None),
                    "failure_details": r.failure_details,
                    "metadata": r.metadata,
                    "failure_diagnosis": (
                        r.get_failure_diagnosis().to_dict()
                        if r.get_failure_diagnosis() is not None
                        else None
                    ),
                    "confidence_assessment": r.get_confidence_assessment().to_dict(),
                    "generated_code": r.generated_code,
                }
                for r in completed_results
            ],
        }

        # Atomic write with temp file
        temp_path = checkpoint_path.with_suffix(".tmp")
        try:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            with open(temp_path, "w") as f:
                json.dump(checkpoint_data, f, indent=2)
            temp_path.rename(checkpoint_path)
        except OSError as exc:
            logger.warning("Failed to save checkpoint %s: %s", checkpoint_path, exc)
            return

        logger.debug(f"Checkpoint saved: {len(completed_results)} tasks completed")

    def _load_checkpoint(
        self,
        checkpoint_path: Path,
    ) -> Optional[tuple[list[TaskResult], list[str], datetime]]:
        """Load checkpoint from disk.

        Returns:
            Tuple of (completed_results, remaining_task_ids, start_time) or None
        """
        import json

        if not checkpoint_path.exists():
            return None

        try:
            with open(checkpoint_path) as f:
                data = json.load(f)

            # Reconstruct TaskResult objects
            completed_results = []
            for r in data.get("results", []):
                result = task_result_from_artifact(r)
                completed_results.append(result)

            remaining_task_ids = data.get("remaining_task_ids", [])
            start_time = datetime.fromisoformat(data["start_time"])

            logger.info(
                f"Loaded checkpoint: {len(completed_results)} completed, "
                f"{len(remaining_task_ids)} remaining"
            )

            return completed_results, remaining_task_ids, start_time

        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            return None

    def _clear_checkpoint(self, checkpoint_path: Path) -> None:
        """Remove checkpoint file after successful completion."""
        try:
            if checkpoint_path.exists():
                checkpoint_path.unlink()
                logger.debug("Checkpoint cleared")
        except OSError as exc:
            logger.warning("Failed to clear checkpoint %s: %s", checkpoint_path, exc)

    async def run_evaluation(
        self,
        config: EvaluationConfig,
        agent_callback: Any,  # Callable that takes task and returns output
        progress_callback: Optional[Any] = None,  # Callable(task_idx, total, TaskResult)
        retry_callback: Optional[Any] = None,  # Callable for self-correction retries
        resume: bool = False,  # Resume from checkpoint if available
    ) -> EvaluationResult:
        """Run a complete evaluation.

        Args:
            config: Evaluation configuration
            agent_callback: Async function that runs the agent on a task
            progress_callback: Optional callback called after each task completes.
                              Signature: (task_index: int, total_tasks: int, result: TaskResult) -> None
            retry_callback: Optional callback for self-correction retries.
                           Signature: (task, previous_code, feedback_prompt) -> str
                           Required if config.enable_self_correction is True.
            resume: If True, resume from checkpoint if one exists for this config

        Returns:
            EvaluationResult with all task results
        """
        runner = self.get_runner(config.benchmark)
        if runner is None:
            raise ValueError(f"No runner for benchmark: {config.benchmark}")

        with run_kind_scope(RunKind.EVALUATION), durable_evaluation_conversations():
            return await self._run_evaluation(
                config,
                runner,
                agent_callback,
                progress_callback,
                retry_callback,
                resume,
            )

    async def _run_evaluation(
        self,
        config: EvaluationConfig,
        runner: Any,
        agent_callback: Any,
        progress_callback: Optional[Any] = None,
        retry_callback: Optional[Any] = None,
        resume: bool = False,
    ) -> EvaluationResult:
        """Run the evaluation body (see ``run_evaluation`` for the contract)."""
        # Warn if self-correction enabled but no retry_callback
        if config.enable_self_correction and retry_callback is None:
            logger.warning(
                "Self-correction enabled but no retry_callback provided. "
                "Auto-fix will work but LLM retries will be skipped."
            )

        # Create metrics collector for self-correction tracking
        metrics_collector = None
        if config.enable_self_correction:
            from victor.evaluation.correction import CorrectionMetricsCollector

            metrics_collector = CorrectionMetricsCollector()

        result = EvaluationResult(config=config)

        # Check for checkpoint if resuming
        checkpoint_path = self._get_checkpoint_path(config)
        completed_results: list[TaskResult] = []
        completed_task_ids: set[str] = set()

        if resume:
            checkpoint_data = self._load_checkpoint(checkpoint_path)
            if checkpoint_data:
                completed_results, remaining_ids, result.start_time = checkpoint_data
                completed_task_ids = {r.task_id for r in completed_results}
                logger.info(
                    f"Resuming from checkpoint: {len(completed_results)} tasks already completed"
                )
            else:
                result.start_time = datetime.now()
                logger.info("No checkpoint found, starting fresh")
        else:
            result.start_time = datetime.now()

        # Load tasks
        tasks = await runner.load_tasks(config)
        total_tasks = len(tasks)
        logger.info(f"Loaded {total_tasks} tasks for {config.benchmark.value}")

        # Filter out already completed tasks
        if completed_task_ids:
            tasks = [t for t in tasks if t.task_id not in completed_task_ids]
            logger.info(f"Resuming with {len(tasks)} remaining tasks")

        # Run tasks
        if config.parallel_tasks > 1:
            new_results = await self._run_parallel(
                tasks,
                runner,
                agent_callback,
                config,
                progress_callback,
                retry_callback,
                metrics_collector,
                checkpoint_path=checkpoint_path,
                completed_results=completed_results,
                total_tasks=total_tasks,
            )
        else:
            new_results = await self._run_sequential(
                tasks,
                runner,
                agent_callback,
                config,
                progress_callback,
                retry_callback,
                metrics_collector,
                checkpoint_path=checkpoint_path,
                completed_results=completed_results,
                total_tasks=total_tasks,
            )

        # Combine completed results with new results
        result.task_results = completed_results + new_results
        result.end_time = datetime.now()

        # Clear checkpoint on successful completion
        self._clear_checkpoint(checkpoint_path)

        # Add correction metrics to result
        if metrics_collector:
            result.correction_metrics = metrics_collector.metrics.to_dict()
            logger.info(
                f"Self-correction metrics: {metrics_collector.metrics.correction_success_rate:.1%} success rate"
            )

        # Save results
        self._save_results(result)

        return result

    async def run_prompt_candidate_evaluation_suite(
        self,
        *,
        base_config: EvaluationConfig,
        candidate_specs: list[PromptCandidateEvaluationSpec],
        agent_callback_factory: Callable[[PromptCandidateEvaluationSpec, EvaluationConfig], Any],
        progress_callback: Optional[Any] = None,
        retry_callback: Optional[Any] = None,
        resume: bool = False,
    ) -> PromptCandidateEvaluationSuiteResult:
        """Run one benchmark evaluation per bound prompt candidate.

        Each candidate gets its own `EvaluationConfig`, which keeps checkpoints,
        saved artifacts, and reported metrics attributable to one exact prompt
        section/candidate pair.
        """
        runs: list[PromptCandidateEvaluationRun] = []
        for spec in candidate_specs:
            config = bind_prompt_candidate_evaluation_config(base_config, spec)
            callback = agent_callback_factory(spec, config)
            result = await self.run_evaluation(
                config=config,
                agent_callback=callback,
                progress_callback=progress_callback,
                retry_callback=retry_callback,
                resume=resume,
            )
            runs.append(
                PromptCandidateEvaluationRun(
                    spec=spec,
                    config=config,
                    result=result,
                    label=spec.resolved_label(config.provider),
                )
            )
        return PromptCandidateEvaluationSuiteResult(base_config=base_config, runs=runs)

    async def _run_sequential(
        self,
        tasks: list[BenchmarkTask],
        runner: BaseBenchmarkRunner,
        agent_callback: Any,
        config: EvaluationConfig,
        progress_callback: Optional[Any] = None,
        retry_callback: Optional[Any] = None,
        metrics_collector: Optional[Any] = None,
        checkpoint_path: Optional[Path] = None,
        completed_results: Optional[list[TaskResult]] = None,
        total_tasks: Optional[int] = None,
    ) -> list[TaskResult]:
        """Run tasks sequentially with checkpoint support."""
        results = []
        all_results = list(completed_results or [])
        effective_total = total_tasks or len(tasks)
        start_idx = len(all_results)

        for i, task in enumerate(tasks):
            effective_idx = start_idx + i
            logger.info(f"Running task {effective_idx + 1}/{effective_total}: {task.task_id}")

            try:
                task_result = await self._run_single_task(
                    task,
                    runner,
                    agent_callback,
                    config,
                    retry_callback,
                    metrics_collector,
                )
                results.append(task_result)
                all_results.append(task_result)

                if task_result.is_success:
                    logger.info(
                        "  Result: PASS (tests %d/%d)",
                        task_result.tests_passed,
                        task_result.tests_total,
                    )
                else:
                    # Surface WHY the task failed — the harness sets
                    # error_message to a categorized reason (partial pass,
                    # all tests failed, 0 collected / install issue, timeout,
                    # exception). Without this, "Result: FAIL" gave no clue.
                    logger.warning(
                        "  Result: FAIL (tests %d/%d) — reason: %s",
                        task_result.tests_passed,
                        task_result.tests_total,
                        task_result.error_message or "no reason recorded",
                    )

                # Save checkpoint after each task
                if checkpoint_path:
                    remaining_ids = [t.task_id for t in tasks[i + 1 :]]
                    self._save_checkpoint(
                        checkpoint_path,
                        config,
                        all_results,
                        remaining_ids,
                        datetime.now(),
                    )

                # Call progress callback if provided
                if progress_callback:
                    progress_callback(effective_idx, effective_total, task_result)

            except Exception as e:
                logger.error(f"  Error: {e}")
                error_result = TaskResult(
                    task_id=task.task_id,
                    status=TaskStatus.ERROR,
                    error_message=str(e),
                    failure_category=BenchmarkFailureCategory.EXECUTION_ERROR,
                )
                results.append(error_result)
                all_results.append(error_result)

                # Save checkpoint even on error
                if checkpoint_path:
                    remaining_ids = [t.task_id for t in tasks[i + 1 :]]
                    self._save_checkpoint(
                        checkpoint_path,
                        config,
                        all_results,
                        remaining_ids,
                        datetime.now(),
                    )

                # Call progress callback for errors too
                if progress_callback:
                    progress_callback(effective_idx, effective_total, error_result)

        return results

    async def _run_parallel(
        self,
        tasks: list[BenchmarkTask],
        runner: BaseBenchmarkRunner,
        agent_callback: Any,
        config: EvaluationConfig,
        progress_callback: Optional[Any] = None,
        retry_callback: Optional[Any] = None,
        metrics_collector: Optional[Any] = None,
        checkpoint_path: Optional[Path] = None,
        completed_results: Optional[list[TaskResult]] = None,
        total_tasks: Optional[int] = None,
    ) -> list[TaskResult]:
        """Run tasks in parallel with checkpoint support."""
        semaphore = asyncio.Semaphore(config.parallel_tasks)
        lock = asyncio.Lock()
        all_results = list(completed_results or [])
        new_results: list[TaskResult] = []
        effective_total = total_tasks or len(tasks)
        start_idx = len(all_results)
        completed_count = 0

        async def run_with_semaphore(idx: int, task: BenchmarkTask) -> TaskResult:
            nonlocal completed_count
            async with semaphore:
                try:
                    result = await self._run_single_task(
                        task,
                        runner,
                        agent_callback,
                        config,
                        retry_callback,
                        metrics_collector,
                    )
                except Exception as e:
                    result = TaskResult(
                        task_id=task.task_id,
                        status=TaskStatus.ERROR,
                        error_message=str(e),
                        failure_category=BenchmarkFailureCategory.EXECUTION_ERROR,
                    )

                # Save checkpoint and call progress callback with lock
                async with lock:
                    completed_count += 1
                    new_results.append(result)
                    all_results.append(result)

                    # Save checkpoint after each completion
                    if checkpoint_path:
                        completed_ids = {r.task_id for r in all_results}
                        remaining_ids = [t.task_id for t in tasks if t.task_id not in completed_ids]
                        self._save_checkpoint(
                            checkpoint_path,
                            config,
                            all_results,
                            remaining_ids,
                            datetime.now(),
                        )

                    if progress_callback:
                        effective_idx = start_idx + completed_count - 1
                        progress_callback(effective_idx, effective_total, result)

                return result

        await asyncio.gather(*[run_with_semaphore(i, task) for i, task in enumerate(tasks)])
        return new_results

    def _apply_agent_callback_payload(
        self,
        task_result: TaskResult,
        payload: dict[str, Any],
    ) -> str:
        """Apply dict-returning callback metrics to the canonical task result."""
        task_result.tokens_input = int(payload.get("tokens_input", 0) or 0)
        task_result.tokens_output = int(payload.get("tokens_output", 0) or 0)
        task_result.tokens_used = int(payload.get("tokens_used", 0) or 0)
        task_result.cached_tokens = int(payload.get("cached_tokens", 0) or 0)
        task_result.reasoning_tokens = int(payload.get("reasoning_tokens", 0) or 0)
        task_result.cost_usd_micros = int(payload.get("cost_usd_micros", 0) or 0)
        task_result.tool_calls = int(payload.get("tool_calls", 0) or 0)
        task_result.turns = int(payload.get("turns", 0) or 0)
        task_result.code_search_calls = int(payload.get("code_search_calls", 0) or 0)
        task_result.graph_calls = int(payload.get("graph_calls", 0) or 0)
        task_result.metadata = dict(payload.get("metadata", {}) or {})

        # Correlation spine + bounded execution trace. session_id joins this
        # task's logged decisions to its outcome; trace carries the messages /
        # tool calls / edits the manifest + miner need.
        sid = payload.get("session_id")
        if isinstance(sid, str) and sid:
            task_result.session_id = sid
        ct = payload.get("conversation_trace")
        if isinstance(ct, dict) and ct:
            task_result.trace = dict(ct)

        task_report = payload.get("task_report")
        if isinstance(task_report, dict):
            task_result.metadata["task_report"] = dict(task_report)

        for key in (
            "cache_hit_rate",
            "tool_schema_tokens",
            "compaction_saved_tokens",
            "compaction_messages_removed",
            "time_to_first_tool_call_seconds",
            "time_to_first_edit_seconds",
            "first_edit_tool_name",
        ):
            if payload.get(key) is not None:
                task_result.metadata[key] = payload.get(key)

        # Which prompt candidates produced this task's outcome. Aggregated into
        # the artifact by _save_results so seed_from_evaluations() can turn the
        # task's pass/fail into Pareto evidence for those candidates.
        prompt_identities = payload.get("prompt_identities")
        if prompt_identities:
            task_result.metadata["prompt_identities"] = list(prompt_identities)

        topology_events = payload.get("topology_events")
        if topology_events:
            task_result.metadata["topology_events"] = list(topology_events)
        planning_events = payload.get("planning_events")
        if planning_events:
            task_result.metadata["planning_events"] = list(planning_events)
        degradation_events = payload.get("degradation_events")
        if degradation_events:
            task_result.metadata["degradation_events"] = list(degradation_events)

        return str(payload.get("code", "") or "")

    async def _run_single_task(
        self,
        task: BenchmarkTask,
        runner: BaseBenchmarkRunner,
        agent_callback: Any,
        config: EvaluationConfig,
        retry_callback: Any = None,
        metrics_collector: Any = None,
    ) -> TaskResult:
        """Run a single task with optional self-correction.

        Args:
            task: The benchmark task
            runner: The benchmark runner
            agent_callback: Callback to generate initial output
            config: Evaluation configuration
            retry_callback: Optional callback for self-correction retries.
                           Signature: (task, previous_code, feedback_prompt) -> str
            metrics_collector: Optional CorrectionMetricsCollector for tracking metrics
        """
        start_time = datetime.now()
        task_result = TaskResult(
            task_id=task.task_id,
            status=TaskStatus.RUNNING,
            start_time=start_time,
        )

        try:
            # Run agent with timeout
            try:
                agent_output = await asyncio.wait_for(
                    agent_callback(task),
                    timeout=config.timeout_per_task,
                )
            except asyncio.TimeoutError:
                task_result.status = TaskStatus.TIMEOUT
                task_result.error_message = "Agent timeout"
                task_result.failure_category = BenchmarkFailureCategory.TIMEOUT
                # Check if callback stored partial data before cancellation
                partial_data = getattr(agent_callback, "_partial_data", None)
                if partial_data:
                    task_result.generated_code = self._apply_agent_callback_payload(
                        task_result,
                        partial_data,
                    )
                    logger.info(
                        f"Task timed out - partial metrics recovered: "
                        f"tool_calls={task_result.tool_calls}, turns={task_result.turns}"
                    )
                else:
                    logger.info("Task timed out - no partial metrics available")
                return task_result

            # Handle dict return type for token tracking (P1 fix)
            # Callbacks can return either:
            # - str: Just the generated code (legacy)
            # - dict: Code plus metrics {code, tokens_input, tokens_output, tokens_used, tool_calls, turns}
            if isinstance(agent_output, dict):
                agent_output = self._apply_agent_callback_payload(task_result, agent_output)

            # Self-correction loop (if enabled)
            if config.enable_self_correction:
                agent_output, task_result = await self._run_with_self_correction(
                    task=task,
                    runner=runner,
                    agent_output=agent_output,
                    config=config,
                    retry_callback=retry_callback,
                    task_result=task_result,
                    metrics_collector=metrics_collector,
                )
            else:
                # Standard flow without self-correction
                task_result.generated_code = agent_output

                # Analyze code quality
                try:
                    from victor.evaluation.analyzers import get_code_quality_analyzer

                    analyzer = get_code_quality_analyzer()
                    code_quality = await analyzer.analyze(
                        agent_output,
                        language=task.language,
                    )
                    task_result.code_quality = code_quality
                except Exception as e:
                    logger.warning(f"Code quality analysis failed: {e}")

                # Evaluate result
                eval_result = await runner.run_task(task, agent_output, config)

                # Merge results
                task_result.status = eval_result.status
                task_result.generated_patch = eval_result.generated_patch
                task_result.tests_passed = eval_result.tests_passed
                task_result.tests_failed = eval_result.tests_failed
                task_result.tests_total = eval_result.tests_total
                task_result.stdout = eval_result.stdout
                task_result.stderr = eval_result.stderr
                task_result.error_message = eval_result.error_message
                task_result.traceback = eval_result.traceback
                task_result.failure_category = eval_result.failure_category
                task_result.failure_details = dict(eval_result.failure_details)
                task_result.metadata = {
                    **task_result.metadata,
                    **dict(eval_result.metadata),
                }
                task_result.failure_diagnosis = eval_result.get_failure_diagnosis()
                task_result.confidence_assessment = eval_result.get_confidence_assessment()

                # Calculate completion score
                task_result.completion_score = task_result.calculate_completion_score()

        except Exception as e:
            task_result.status = TaskStatus.ERROR
            task_result.error_message = str(e)
            task_result.failure_category = BenchmarkFailureCategory.EXECUTION_ERROR
            import traceback

            task_result.traceback = traceback.format_exc()

        task_result.end_time = datetime.now()
        task_result.duration_seconds = (task_result.end_time - start_time).total_seconds()

        return task_result

    async def _run_with_self_correction(
        self,
        task: BenchmarkTask,
        runner: BaseBenchmarkRunner,
        agent_output: str,
        config: EvaluationConfig,
        retry_callback: Any,
        task_result: TaskResult,
        metrics_collector: Any = None,
    ) -> tuple[str, TaskResult]:
        """Run task with self-correction loop.

        This implements generic iterative refinement:
        1. Validate code (syntax, imports)
        2. Auto-fix common issues
        3. Run tests
        4. If failed, generate feedback and retry

        This is NOT task-specific - it works for any code generation.

        Args:
            metrics_collector: Optional CorrectionMetricsCollector for tracking metrics
        """
        from victor.evaluation.correction import (
            create_self_corrector,
            detect_language,
        )

        corrector = create_self_corrector(
            max_iterations=config.self_correction_max_iterations,
            auto_fix=config.auto_fix_imports,
        )

        # Detect language for metrics
        lang = detect_language(agent_output, filename=f"solution.{task.language}")

        best_output = agent_output
        best_result = None

        for iteration in range(config.self_correction_max_iterations):
            task_result.attempts = iteration + 1

            # Step 1: Validate and auto-fix
            fixed_code, validation = corrector.validate_and_fix(agent_output)
            agent_output = fixed_code

            # Record validation in metrics
            if metrics_collector:
                metrics_collector.record_validation(lang, validation)

            # Step 2: Analyze code quality
            code_quality = None
            try:
                from victor.evaluation.analyzers import get_code_quality_analyzer

                analyzer = get_code_quality_analyzer()
                code_quality = await analyzer.analyze(
                    agent_output,
                    language=task.language,
                )
            except Exception as e:
                logger.warning(f"Code quality analysis failed: {e}")

            # Step 3: Run tests
            eval_result = await runner.run_task(task, agent_output, config)

            # Track correction attempt in metrics
            if metrics_collector:
                with metrics_collector.track_correction(
                    task_id=task.task_id,
                    language=lang,
                    iteration=iteration + 1,
                    validation_before=validation,
                    test_passed_before=eval_result.tests_passed,
                    test_total=eval_result.tests_total,
                ) as tracker:
                    is_success = eval_result.status == TaskStatus.PASSED
                    tracker.set_result(
                        success=is_success,
                        validation_after=validation,
                        test_passed_after=eval_result.tests_passed,
                        auto_fixed=fixed_code != agent_output,
                    )

            # Track best result
            if best_result is None or eval_result.tests_passed > best_result.tests_passed:
                best_output = agent_output
                best_result = eval_result

            # Check if passed
            if eval_result.status == TaskStatus.PASSED:
                task_result.generated_code = agent_output
                task_result.code_quality = code_quality
                task_result.status = eval_result.status
                task_result.tests_passed = eval_result.tests_passed
                task_result.tests_failed = eval_result.tests_failed
                task_result.tests_total = eval_result.tests_total
                task_result.stdout = eval_result.stdout
                task_result.stderr = eval_result.stderr
                task_result.successful_attempts = 1
                task_result.completion_score = task_result.calculate_completion_score()
                logger.info(f"  Self-correction: PASSED on iteration {iteration + 1}")
                return agent_output, task_result

            # Step 4: Generate feedback for retry
            if iteration < config.self_correction_max_iterations - 1 and retry_callback:
                feedback = corrector.generate_feedback(
                    code=agent_output,
                    validation=validation,
                    test_stdout=eval_result.stdout,
                    test_stderr=eval_result.stderr,
                    test_passed=eval_result.tests_passed,
                    test_total=eval_result.tests_total,
                )

                if feedback.has_issues:
                    retry_prompt = corrector.build_retry_prompt(
                        original_prompt=task.prompt,
                        previous_code=agent_output,
                        feedback=feedback,
                        iteration=iteration + 1,
                    )

                    logger.info(
                        f"  Self-correction: Retry {iteration + 2}/{config.self_correction_max_iterations}"
                    )

                    try:
                        agent_output = await asyncio.wait_for(
                            retry_callback(task, agent_output, retry_prompt),
                            timeout=config.timeout_per_task,
                        )
                    except asyncio.TimeoutError:
                        logger.warning("  Self-correction: Retry timeout")
                        break
                    except Exception as e:
                        logger.warning(f"  Self-correction: Retry failed: {e}")
                        break

        # Use best result
        task_result.generated_code = best_output
        task_result.code_quality = code_quality
        if best_result:
            task_result.status = best_result.status
            task_result.tests_passed = best_result.tests_passed
            task_result.tests_failed = best_result.tests_failed
            task_result.tests_total = best_result.tests_total
            task_result.stdout = best_result.stdout
            task_result.stderr = best_result.stderr
        task_result.completion_score = task_result.calculate_completion_score()

        return best_output, task_result

    def _save_results(self, result: EvaluationResult) -> Path:
        """Save evaluation results to disk."""
        import json

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"eval_{result.config.benchmark.value}_{timestamp}.json"
        output_path = self._results_dir / filename
        summary = result.get_metrics()
        runtime_feedback = derive_runtime_evaluation_feedback(result)
        runtime_feedback_payload = build_runtime_evaluation_feedback_payload(
            runtime_feedback,
            source_result_path=output_path,
        )
        experiment_memory = analyze_evaluation_result(
            result,
            summary=summary,
            runtime_feedback=runtime_feedback_payload,
            source_result_path=output_path,
        )

        # Serialize result
        data = {
            "config": result.config.to_artifact_config(),
            "observed_prompt_identities": _collect_observed_prompt_identities(result),
            "summary": summary,
            "runtime_evaluation_feedback": runtime_feedback_payload,
            "experiment_memory": experiment_memory.to_dict(),
            "start_time": result.start_time.isoformat() if result.start_time else None,
            "end_time": result.end_time.isoformat() if result.end_time else None,
            "tasks": [
                {
                    "task_id": r.task_id,
                    # The correlation spine. TaskResult has carried session_id in
                    # memory (set from the agent payload) specifically to join this
                    # task's logged decisions to its outcome, but it was never
                    # serialized — so nothing downstream could match a usage.jsonl
                    # trace to the harness verdict that graded it, and prompt
                    # evolution scored eval sessions with a tool-failure proxy.
                    "session_id": r.session_id,
                    "status": r.status.value,
                    "tests_passed": r.tests_passed,
                    "tests_total": r.tests_total,
                    "duration_seconds": r.duration_seconds,
                    "tokens_used": r.tokens_used,
                    "tokens_input": r.tokens_input,
                    "tokens_output": r.tokens_output,
                    "cached_tokens": r.cached_tokens,
                    "reasoning_tokens": r.reasoning_tokens,
                    "cost_usd_micros": r.cost_usd_micros,
                    "tool_calls": r.tool_calls,
                    "turns": r.turns,
                    "code_search_calls": r.code_search_calls,
                    "graph_calls": r.graph_calls,
                    "completion_score": r.completion_score,
                    "failure_category": (r.failure_category.value if r.failure_category else None),
                    "failure_details": r.failure_details,
                    "metadata": r.metadata,
                    "failure_diagnosis": (
                        r.get_failure_diagnosis().to_dict()
                        if r.get_failure_diagnosis() is not None
                        else None
                    ),
                    "confidence_assessment": r.get_confidence_assessment().to_dict(),
                    "code_quality": (
                        {
                            "syntax_valid": r.code_quality.syntax_valid,
                            "lint_errors": r.code_quality.lint_errors,
                            "lint_warnings": r.code_quality.lint_warnings,
                            "style_score": r.code_quality.style_score,
                            "cyclomatic_complexity": r.code_quality.cyclomatic_complexity,
                            "maintainability_index": r.code_quality.maintainability_index,
                            "type_coverage": r.code_quality.type_coverage,
                            "overall_score": r.code_quality.get_overall_score(),
                        }
                        if r.code_quality
                        else None
                    ),
                    "error_message": r.error_message,
                }
                for r in result.task_results
            ],
        }

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(data, f, indent=2)
        except OSError as exc:
            logger.warning("Failed to save evaluation results to %s: %s", output_path, exc)
            return output_path

        self._persist_experiment_memory(experiment_memory)
        try:
            self._save_validated_session_feedbacks(
                result,
                source_result_path=output_path,
                summary=summary,
            )
        except Exception as exc:
            logger.warning("Validated session-truth persistence failed: %s", exc)

        logger.info(f"Results saved to: {output_path}")
        return output_path

    def _persist_experiment_memory(self, record: ExperimentMemoryRecord) -> None:
        """Append a reusable experiment-memory record for later retrieval."""
        try:
            store = ExperimentMemoryStore(
                persist_path=self._results_dir / "experiment_memory.json",
            )
            store.record(record)
        except Exception as exc:
            logger.warning("Experiment-memory persistence failed: %s", exc)

    def _save_validated_session_feedbacks(
        self,
        result: EvaluationResult,
        *,
        source_result_path: Path,
        summary: Optional[dict[str, Any]] = None,
    ) -> list[Path]:
        """Persist per-task validated session-truth artifacts for supported workflows."""
        return self._validated_session_truth_service.persist_evaluation_result(
            result,
            results_dir=self._results_dir,
            source_result_path=source_result_path,
            summary=summary,
            refresh_when_empty=True,
        )

    def load_results(self, path: Path) -> dict:
        """Load evaluation results from disk."""
        import json

        with open(path) as f:
            return json.load(f)

    def generate_report(
        self,
        result: EvaluationResult,
        format: str = "text",
    ) -> str:
        """Generate an evaluation report.

        Args:
            result: Evaluation result
            format: Report format (text, markdown, json)

        Returns:
            Report string
        """
        if format == "markdown":
            return self._generate_markdown_report(result)
        elif format == "json":
            import json

            return json.dumps(result.get_metrics(), indent=2)
        else:
            return self._generate_text_report(result)

    def _generate_text_report(self, result: EvaluationResult) -> str:
        """Generate text report."""
        lines = []

        lines.append("=" * 70)
        lines.append("EVALUATION REPORT")
        lines.append("=" * 70)
        lines.append("")

        lines.append("CONFIGURATION")
        lines.append("-" * 40)
        lines.append(f"Benchmark: {result.config.benchmark.value}")
        lines.append(f"Model: {result.config.model}")
        lines.append("")

        metrics = result.get_metrics()
        lines.append("RESULTS")
        lines.append("-" * 40)
        lines.append(f"Total Tasks:    {metrics['total_tasks']}")
        lines.append(f"Passed:         {metrics['passed']} ({metrics['pass_rate']:.1%})")
        lines.append(f"Failed:         {metrics['failed']}")
        lines.append(f"Errors:         {metrics['errors']}")
        lines.append(f"Timeouts:       {metrics['timeouts']}")
        lines.append("")

        lines.append("RESOURCES")
        lines.append("-" * 40)
        lines.append(f"Total Duration: {metrics['duration_seconds']:.1f}s")
        lines.append(f"Total Tokens:   {metrics['total_tokens']}")
        lines.append(f"Avg Tokens/Task: {metrics['avg_tokens_per_task']:.0f}")
        lines.append("")

        lines.append("=" * 70)

        return "\n".join(lines)

    def _generate_markdown_report(self, result: EvaluationResult) -> str:
        """Generate markdown report."""
        metrics = result.get_metrics()

        lines = []
        lines.append("# Evaluation Report")
        lines.append("")
        lines.append("## Configuration")
        lines.append("")
        lines.append(f"- **Benchmark:** {result.config.benchmark.value}")
        lines.append(f"- **Model:** {result.config.model}")
        lines.append("")

        lines.append("## Results")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Total Tasks | {metrics['total_tasks']} |")
        lines.append(f"| Pass Rate | {metrics['pass_rate']:.1%} |")
        lines.append(f"| Passed | {metrics['passed']} |")
        lines.append(f"| Failed | {metrics['failed']} |")
        lines.append(f"| Errors | {metrics['errors']} |")
        lines.append(f"| Timeouts | {metrics['timeouts']} |")
        lines.append("")

        lines.append("## Resource Usage")
        lines.append("")
        lines.append(f"- **Duration:** {metrics['duration_seconds']:.1f}s")
        lines.append(f"- **Tokens:** {metrics['total_tokens']}")
        lines.append(f"- **Tool Calls:** {metrics['total_tool_calls']}")
        lines.append("")

        return "\n".join(lines)


# Global harness instance
_harness: Optional[EvaluationHarness] = None


def get_harness() -> EvaluationHarness:
    """Get the global evaluation harness."""
    global _harness
    if _harness is None:
        _harness = EvaluationHarness()
    return _harness
