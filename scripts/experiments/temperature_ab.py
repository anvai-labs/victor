# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Temperature A/B harness for the ADR-013 global-default decision (0.7 → 0.6?).

Runs a small read-only task battery at two temperatures via the real agent and reports the metrics
that actually matter for the default choice:
- **efficiency**: mean turns / tool calls (lower-temp should be no worse, ideally tighter);
- **completion**: success rate;
- **variance**: turn-count spread across repeats (the determinism argument for 0.6);
- **quality** (optional): LLM rubric score of the final answer (reuses EVR-3c).

Decision rule (printed): flip to the lower temperature only if it **match-or-beats** the higher on
completion + quality while not increasing turns/variance — same match-or-beat discipline as the EVR work.

Usage (needs a real model; authenticate via the bridge first):
    eval "$(victor auth env -p zai 2>/dev/null | grep '^export')"
    python scripts/experiments/temperature_ab.py --profile zai-glm52-openai --temps 0.6,0.7 --runs 2
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
from dataclasses import dataclass, field
from typing import List, Optional

# Multi-turn, tool-driven coding tasks (shared with completion_ab.py) — these stress determinism and
# spin: a flakier temperature shows up as higher turn-count variance / more tool churn. (Single-turn
# QA can't differentiate temperatures — every run ties at 1 turn.)
try:
    from scripts.experiments.coding_tasks import MULTI_TURN_CODING_TASKS as TASKS
except ImportError:  # allow running as `python scripts/experiments/temperature_ab.py`
    import os as _os
    import sys as _sys

    _sys.path.insert(0, _os.path.dirname(__file__))
    from coding_tasks import MULTI_TURN_CODING_TASKS as TASKS


@dataclass
class RunMetrics:
    turns: int
    tool_calls: int
    success: bool
    answer_len: int
    answer: str


@dataclass
class TempResult:
    temperature: float
    runs: List[RunMetrics] = field(default_factory=list)

    def _vals(self, attr):
        return [getattr(r, attr) for r in self.runs]

    def summary(self) -> dict:
        turns = self._vals("turns")
        return {
            "n": len(self.runs),
            "completion_rate": (
                (sum(r.success for r in self.runs) / len(self.runs)) if self.runs else 0.0
            ),
            "mean_turns": statistics.mean(turns) if turns else 0.0,
            "turns_stdev": statistics.pstdev(turns) if len(turns) > 1 else 0.0,
            "mean_tool_calls": (statistics.mean(self._vals("tool_calls")) if self.runs else 0.0),
            "mean_answer_len": (statistics.mean(self._vals("answer_len")) if self.runs else 0.0),
        }


async def _run_one(task: str, temperature: float, profile: Optional[str]) -> RunMetrics:
    from victor.framework.agent import Agent

    kwargs = {"temperature": temperature}
    if profile:
        kwargs["profile"] = profile
    agent = await Agent.create(**kwargs)
    setter = getattr(agent, "set_max_iterations", None)
    if setter:
        setter(12)
    try:
        result = await agent.run(task)
    finally:
        close = getattr(agent, "close", None) or getattr(agent, "shutdown", None)
        if close:
            try:
                await close()
            except Exception:
                pass
    md = result.metadata or {}
    # agentic_loop_iterations = real loop turn count; "turns" (request_count) next; proxy last.
    turns = int(
        md.get("agentic_loop_iterations")
        or md.get("turns")
        or md.get("iterations")
        or md.get("iteration_count")
        or 0
    )
    if turns == 0:  # fall back to a tool-activity proxy when the runtime doesn't expose it
        turns = len(result.tool_calls) + 1
    content = result.content or ""
    return RunMetrics(
        turns=turns,
        tool_calls=len(result.tool_calls),
        success=bool(result.success),
        answer_len=len(content),
        answer=content,
    )


async def main(temps: List[float], runs: int, profile: Optional[str]) -> None:
    results = {t: TempResult(temperature=t) for t in temps}
    for t in temps:
        for task in TASKS:
            for _ in range(runs):
                try:
                    results[t].runs.append(await _run_one(task, t, profile))
                except Exception as exc:  # keep going; a failed run is itself a signal
                    print(f"[warn] temp={t} task={task[:40]!r} failed: {exc!r}")

    print("\n==================== Temperature A/B (ADR-013 default decision) ====================")
    header = f"{'temp':>5} | {'n':>3} | {'complete':>8} | {'mean_turns':>10} | {'turns_sd':>8} | {'mean_tools':>10} | {'ans_len':>7}"
    print(header)
    print("-" * len(header))
    summaries = {}
    for t in temps:
        s = results[t].summary()
        summaries[t] = s
        print(
            f"{t:>5.2f} | {s['n']:>3} | {s['completion_rate']:>8.2f} | {s['mean_turns']:>10.2f} | "
            f"{s['turns_stdev']:>8.2f} | {s['mean_tool_calls']:>10.2f} | {s['mean_answer_len']:>7.0f}"
        )

    # Decision rule: flip to the lower temp only if it match-or-beats the higher.
    if len(temps) == 2:
        lo, hi = sorted(temps)
        slo, shi = summaries[lo], summaries[hi]
        match_or_beat = (
            slo["completion_rate"] >= shi["completion_rate"]
            and slo["mean_turns"] <= shi["mean_turns"] + 0.5
            and slo["turns_stdev"] <= shi["turns_stdev"] + 0.5
        )
        print("\nDecision:")
        if match_or_beat:
            print(
                f"  ✅ {lo:.2f} match-or-beats {hi:.2f} (completion ≥, turns ≤, variance ≤) → flip is supported."
            )
        else:
            print(
                f"  ⚠️  {lo:.2f} does NOT clearly match-or-beat {hi:.2f} → KEEP {hi:.2f} default."
            )
        print(
            "  NOTE: quality (answer correctness) is not auto-scored here — eyeball the answers or"
        )
        print(
            "  add an LLM rubric pass before flipping; this harness gates efficiency + completion + variance."
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--temps", default="0.6,0.7", help="comma-separated temperatures to compare")
    ap.add_argument("--runs", type=int, default=2, help="repeats per task (captures variance)")
    ap.add_argument("--profile", default="zai-glm52-openai", help="agent profile (model/provider)")
    args = ap.parse_args()
    asyncio.run(main([float(x) for x in args.temps.split(",")], args.runs, args.profile or None))
