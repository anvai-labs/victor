# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
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

"""Memory A/B battery (FEP-0048 FOU-A2c; ADR-036 evidence).

Baseline = ``memory_enabled`` OFF, candidate = ON, over the same deterministic
seeded-memory task fixtures. The battery measures:

- **Retrieval usefulness**: on fixtures with a seeded preference/episode whose
  content matches the task query, did the injected memory block surface the
  seeded fact (and NOT surface facts outside the session's scope)? Fully
  deterministic — the wiring is exercised without any LLM.
- **Prompt-inflation cost**: the token estimate of the memory block per task
  (repo-standard chars/4), reported as mean/max per arm.

Verdict semantics (ADR-036): the battery REPORTS evidence; the default-on flip
stays as shipped (default-on with off-when-empty) unless the battery shows
prompt-inflation harm:

- ``KEEP_DEFAULT_ON`` — treatment surfaces seeded facts (usefulness >=
  ``min_usefulness``), zero false surfaces (cross-scope/empty-store fixtures
  inject nothing), and mean block tokens within the injection cap.
- ``ESCALATE_PROMPT_INFLATION`` — usefulness holds but the token cost exceeds
  the cap: escalate to the ADR-036 queue (possible cap/demotion review).
- ``HOLD_REVIEW`` — the injection does not reliably surface seeded facts (or
  false-surfaces): hold and investigate the read path.

**Why not ``flag_ab.py``?** That harness drives the real agent
(``VictorAgentAdapter`` + ``PersistentLoopRunner``) against a live provider —
minutes per task, nondeterministic, and it scores workspace effects, which says
nothing about whether the memory block was assembled, bounded, or
budget-registered. A2c's evidence gate is about the injection WIRING, so this
module mirrors flag_ab's reporting conventions instead (two JSON snapshots
``memory_enabled_{off,on}.json``, a verdict dict, a small CLI, env toggled per
arm and restored) with a deterministic executor. An LLM-run battery over the
flag can still be driven through ``flag_ab.py`` with ``--flag memory_enabled``
by adding it to ``FLAG_ENV`` once end-task effects are worth measuring.

    python -m victor.evaluation.memory_ab --out-dir output/memory_ab
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from victor.agent.memory_injection import (
    DEFAULT_MEMORY_INJECTION_MAX_TOKENS,
    MemoryBlockBuilder,
)
from victor.storage.memory.preference_episodic import MemoryScope

logger = logging.getLogger(__name__)

#: The A/B flag and its env var (mirrors flag_ab.FLAG_ENV conventions).
MEMORY_AB_FLAG = "memory_enabled"
MEMORY_AB_ENV = "VICTOR_MEMORY_ENABLED"


def memory_ab_flag_env(flag: str = MEMORY_AB_FLAG) -> str:
    """The env var a flag's resolver reads (flag_ab._flag_env_var convention)."""
    return MEMORY_AB_ENV if flag == MEMORY_AB_FLAG else f"VICTOR_{flag.upper()}"


@dataclass(frozen=True)
class MemoryABTask:
    """One deterministic fixture task.

    Attributes:
        task_id: Stable identifier reported in the snapshots.
        query: The session query the block is built for.
        seed_scope: Scope of the seeded preference/episode (None = no seed).
        seed_content: Content of the seeded memory (None = no seed).
        seed_kind: "preference" or "episode".
        session_scope: The session's declared memory scope context.
        expect_surface: Whether the seeded content MUST appear in the block.
    """

    task_id: str
    query: str
    seed_scope: Optional[MemoryScope] = None
    seed_content: Optional[str] = None
    seed_kind: str = "preference"
    session_scope: Optional[str] = None
    expect_surface: bool = True


def memory_ab_corpus(variants: int = 1) -> List[MemoryABTask]:
    """Deterministic seeded-memory fixture tasks (no LLM, no network).

    The corpus covers the four evidence cells: user-scope recall, project-scope
    recall, episode recall, and the two negatives (cross-scope suppression and
    empty-store no-op). ``variants`` repeats the families with distinct seeds
    (flag_ab convention; all variants stay deterministic).
    """
    tasks: List[MemoryABTask] = []
    for v in range(max(1, variants)):
        suffix = "" if v == 0 else f"_v{v + 1}"
        tasks.extend(
            [
                MemoryABTask(
                    task_id=f"user_preference_hit{suffix}",
                    query=f"packaging uv pip v{v}",
                    seed_scope=MemoryScope.USER,
                    seed_content=f"always use uv for packaging, never pip variant {v}",
                    session_scope="user",
                    expect_surface=True,
                ),
                MemoryABTask(
                    task_id=f"project_preference_hit{suffix}",
                    query=f"testing framework pytest preference v{v}",
                    seed_scope=MemoryScope.PROJECT,
                    seed_content=f"this repo uses pytest, never unittest variant {v}",
                    session_scope="project",
                    expect_surface=True,
                ),
                MemoryABTask(
                    task_id=f"episode_hit{suffix}",
                    query=f"renamed service checkout v{v}",
                    seed_scope=MemoryScope.PROJECT,
                    seed_content=f"the checkout service was renamed to cart variant {v}",
                    seed_kind="episode",
                    session_scope="project",
                    expect_surface=True,
                ),
                MemoryABTask(
                    task_id=f"cross_scope_suppressed{suffix}",
                    query=f"packaging uv pip v{v}",
                    seed_scope=MemoryScope.PROJECT,
                    seed_content=f"project-only fact must not reach user sessions v{v}",
                    session_scope="user",
                    expect_surface=False,
                ),
                MemoryABTask(
                    task_id=f"empty_store_noop{suffix}",
                    query=f"unrelated query with no seeds v{v}",
                    seed_scope=None,
                    session_scope="project",
                    expect_surface=False,
                ),
            ]
        )
    return tasks


@dataclass
class MemoryABArmResult:
    """Aggregated outcome of one arm (mirrors BatteryResult's role)."""

    enabled: bool
    usefulness_scores: List[float] = field(default_factory=list)
    token_costs: List[int] = field(default_factory=list)
    false_surfaces: int = 0
    injections: int = 0
    task_reports: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def usefulness_mean(self) -> float:
        if not self.usefulness_scores:
            return 0.0
        return sum(self.usefulness_scores) / len(self.usefulness_scores)

    @property
    def prompt_tokens_mean(self) -> float:
        if not self.token_costs:
            return 0.0
        return sum(self.token_costs) / len(self.token_costs)

    @property
    def prompt_tokens_max(self) -> int:
        return max(self.token_costs, default=0)

    def to_dict(self) -> Dict[str, Any]:
        """Snapshot dict (the JSON artifact's per-arm payload)."""
        return {
            "enabled": self.enabled,
            "task_count": len(self.usefulness_scores),
            "usefulness_mean": round(self.usefulness_mean, 4),
            "prompt_tokens_mean": round(self.prompt_tokens_mean, 2),
            "prompt_tokens_max": self.prompt_tokens_max,
            "false_surfaces": self.false_surfaces,
            "injections": self.injections,
            "tasks": self.task_reports,
        }


def _score_task(
    task: MemoryABTask,
    block: Any,
) -> Dict[str, Any]:
    """Score one task: did the block behave as the fixture demands?

    Usefulness is 1.0 when the block matches the expectation (surface when
    expected with the seeded content present, or correctly absent), else 0.0.
    A false surface (block present when none expected) is counted separately.
    """
    surfaced_seeds = False
    if block is not None and task.seed_content is not None:
        surfaced_seeds = task.seed_content in block.text

    if task.expect_surface:
        hit = block is not None and surfaced_seeds
        usefulness = 1.0 if hit else 0.0
        false_surface = False
    else:
        usefulness = 1.0 if block is None else 0.0
        false_surface = block is not None

    return {
        "task_id": task.task_id,
        "injected": block is not None,
        "token_estimate": block.token_estimate if block is not None else 0,
        "seed_surfaced": surfaced_seeds,
        "usefulness": usefulness,
        "false_surface": false_surface,
        "dropped_count": block.dropped_count if block is not None else 0,
    }


async def run_memory_ab_task(
    task: MemoryABTask,
    *,
    settings: Any = None,
    max_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """Run one fixture task through the injection builder and score it.

    The store is seeded on throwaway tmp-path databases (no network, no
    ``~/.victor`` writes); the builder reads ``VICTOR_MEMORY_ENABLED`` live,
    so the caller toggles the env for the arm.
    """
    from victor.core.database import (
        get_database,
        get_project_database,
        reset_database,
        reset_project_database,
    )
    from victor.storage.memory.adapters import create_preference_episodic_adapters
    from victor.storage.memory.preference_episodic import PreferenceEpisodicStore
    from victor.storage.memory.unified import UnifiedMemoryCoordinator

    with tempfile.TemporaryDirectory(prefix="victor-memory-ab-") as tmp:
        tmp_path = Path(tmp)
        # DatabaseManager is a process singleton; the reset/get pattern is the
        # established test convention for fixture-injecting tmp-path databases
        # (see tests/unit/storage/memory/test_preference_episodic.py).
        reset_database()
        reset_project_database(tmp_path)
        try:
            store = PreferenceEpisodicStore(
                global_db=get_database(tmp_path / "victor.db"),
                project_db=get_project_database(tmp_path),
            )
            if task.seed_scope is not None and task.seed_content is not None:
                if task.seed_kind == "episode":
                    store.record_episode(task.seed_content)
                else:
                    store.set_preference(task.seed_content, scope=task.seed_scope)

            coordinator = UnifiedMemoryCoordinator(enable_memory_evolution=False)
            for adapter in create_preference_episodic_adapters(store):
                coordinator.register_provider(adapter)

            builder = MemoryBlockBuilder(
                coordinator,
                settings=settings,
                max_tokens=max_tokens,
            )
            context = {"memory_scope": task.session_scope} if task.session_scope else None
            from victor.agent.memory_injection import memory_scopes_from_context

            block = await builder.build(task.query, memory_scopes_from_context(context))
            return _score_task(task, block)
        finally:
            reset_database()
            reset_project_database(tmp_path)


async def _run_arm(
    enabled: bool,
    *,
    variants: int,
    settings: Any,
    max_tokens: Optional[int],
    task_provider: Optional[Callable[[int], Sequence[MemoryABTask]]],
) -> MemoryABArmResult:
    """Run one arm with the env toggled (and restored), flag_ab-style."""
    previous = os.environ.get(MEMORY_AB_ENV)
    os.environ[MEMORY_AB_ENV] = "true" if enabled else "false"
    try:
        tasks = list((task_provider or memory_ab_corpus)(variants))
        result = MemoryABArmResult(enabled=enabled)
        for task in tasks:
            report = await run_memory_ab_task(task, settings=settings, max_tokens=max_tokens)
            result.task_reports.append(report)
            result.usefulness_scores.append(float(report["usefulness"]))
            result.token_costs.append(int(report["token_estimate"]))
            if report["injected"]:
                result.injections += 1
            if report["false_surface"]:
                result.false_surfaces += 1
        return result
    finally:
        if previous is None:
            os.environ.pop(MEMORY_AB_ENV, None)
        else:
            os.environ[MEMORY_AB_ENV] = previous


@dataclass(frozen=True)
class MemoryABVerdict:
    """ADR-036 evidence verdict for the memory_enabled flag.

    The verdict mechanism does NOT pre-judge the default: default-on ships
    (with off-when-empty semantics) and STAYS unless the battery shows
    prompt-inflation harm. ``should_keep_default_on`` is the machine-readable
    flip of that rule for the ADR-036 queue and the FOU ledger.
    """

    verdict: str
    should_keep_default_on: bool
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verdict": self.verdict,
            "should_keep_default_on": self.should_keep_default_on,
            "reasons": list(self.reasons),
        }


def assess_memory_ab(
    baseline: MemoryABArmResult,
    candidate: MemoryABArmResult,
    *,
    token_budget: int = DEFAULT_MEMORY_INJECTION_MAX_TOKENS,
    min_usefulness: float = 0.9,
) -> MemoryABVerdict:
    """Verdict per ADR-036: evidence-only; only inflation harm demotes."""
    reasons: List[str] = []
    usefulness_ok = candidate.usefulness_mean >= min_usefulness and candidate.false_surfaces == 0
    tokens_ok = candidate.prompt_tokens_mean <= token_budget

    if not usefulness_ok:
        reasons.append(
            f"candidate usefulness {candidate.usefulness_mean:.3f} < {min_usefulness} "
            f"or {candidate.false_surfaces} false surfaces"
        )
    else:
        reasons.append(
            f"candidate surfaces seeded facts (usefulness {candidate.usefulness_mean:.3f}, "
            f"no false surfaces)"
        )

    if not tokens_ok:
        reasons.append(
            f"prompt inflation: mean block {candidate.prompt_tokens_mean:.1f} tokens "
            f"exceeds the {token_budget}-token cap"
        )
    else:
        reasons.append(
            f"prompt cost bounded: mean block {candidate.prompt_tokens_mean:.1f} tokens "
            f"within the {token_budget}-token cap"
        )

    if usefulness_ok and tokens_ok:
        return MemoryABVerdict("KEEP_DEFAULT_ON", True, reasons)
    if usefulness_ok and not tokens_ok:
        return MemoryABVerdict("ESCALATE_PROMPT_INFLATION", False, reasons)
    return MemoryABVerdict("HOLD_REVIEW", False, reasons)


def _drive_arm(**kwargs: Any) -> "MemoryABArmResult":
    """Run one arm on a private event loop in a worker thread.

    The battery must be callable from sync CLI context AND from inside a
    running loop (pytest-asyncio auto mode keeps a loop live across tests),
    where plain ``asyncio.run`` would raise. The database managers use
    thread-local connections, so a worker thread is safe; arms run strictly
    sequentially so singleton DB state never overlaps.
    """
    import concurrent.futures

    def _run() -> "MemoryABArmResult":
        return asyncio.run(_run_arm(**kwargs))

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(_run).result()


def run_memory_ab(
    *,
    variants: int = 1,
    out_dir: str = ".",
    settings: Any = None,
    max_tokens: Optional[int] = None,
    token_budget: Optional[int] = None,
    min_usefulness: float = 0.9,
    task_provider: Optional[Callable[[int], Sequence[MemoryABTask]]] = None,
) -> Dict[str, Any]:
    """Run both arms (OFF baseline, ON candidate) and emit snapshots + verdict.

    Mirrors ``flag_ab.run_flag_ab``'s output contract: two JSON snapshots
    (``memory_enabled_off.json`` / ``memory_enabled_on.json``), both arms'
    dicts inline, and the verdict. Deterministic and offline — no provider,
    no agent loop, unit-testable runtime.

    Returns:
        Dict with snapshot paths, both arm dicts, and the ADR-036 verdict.
    """
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    baseline = _drive_arm(
        enabled=False,
        variants=variants,
        settings=settings,
        max_tokens=max_tokens,
        task_provider=task_provider,
    )
    candidate = _drive_arm(
        enabled=True,
        variants=variants,
        settings=settings,
        max_tokens=max_tokens,
        task_provider=task_provider,
    )

    off_path = out / f"{MEMORY_AB_FLAG}_off.json"
    on_path = out / f"{MEMORY_AB_FLAG}_on.json"
    off_path.write_text(json.dumps(baseline.to_dict(), indent=2) + "\n", encoding="utf-8")
    on_path.write_text(json.dumps(candidate.to_dict(), indent=2) + "\n", encoding="utf-8")

    verdict = assess_memory_ab(
        baseline,
        candidate,
        token_budget=(
            token_budget
            if token_budget is not None
            else (max_tokens if max_tokens is not None else DEFAULT_MEMORY_INJECTION_MAX_TOKENS)
        ),
        min_usefulness=min_usefulness,
    )

    return {
        "flag": MEMORY_AB_FLAG,
        "baseline_path": str(off_path),
        "candidate_path": str(on_path),
        "baseline_battery": baseline.to_dict(),
        "candidate_battery": candidate.to_dict(),
        "verdict": verdict.to_dict(),
    }


def _main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m victor.evaluation.memory_ab",
        description="Deterministic A/B for the FEP-0048 memory injection flag "
        "(OFF baseline vs ON candidate); emits snapshots + the ADR-036 verdict.",
    )
    parser.add_argument("--variants", type=int, default=1, help="Fixture repetitions")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Injection cap override (default: settings/500)",
    )
    parser.add_argument("--out-dir", default=".", help="Where to write the two snapshot JSONs")
    args = parser.parse_args(argv)

    result = run_memory_ab(variants=args.variants, max_tokens=args.max_tokens, out_dir=args.out_dir)
    print(json.dumps(result, indent=2))
    return 0 if result["verdict"]["should_keep_default_on"] else 1


if __name__ == "__main__":  # pragma: no cover
    import sys

    sys.exit(_main())
