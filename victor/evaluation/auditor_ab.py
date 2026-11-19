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

"""Per-turn-auditor A/B battery (FOU-S4; ADR-036 evidence for the EVR-6 flag).

Baseline = ``per_turn_auditor`` OFF, candidate = ON, over deterministic
TurnResult-shaped fixtures replayed through the production ``PerTurnAuditor``.
The battery measures the flag's DECIDABLE effects without any LLM:

- **Wiring correctness**: flag-ON consults the injected judge exactly once per
  turn whose upstream decision is COMPLETE and whose deterministic heuristic
  passed; the degenerate-COMPLETE heuristic alarm fires WITHOUT spending the
  judge and downgrades to RETRY; non-COMPLETE decisions and the flag-OFF arm
  are strict no-ops. A static wiring audit additionally guards the
  placement-not-existence class: both execution paths (buffered
  ``turn_execution_runtime`` and streaming ``chat_stream_executor``) must
  resolve the flag and inject the edge-model judge, and the loop must apply
  the auditor on the EVALUATE seam.
- **Cost model**: judge invocations per COMPLETE turn (max 1), the bounded
  edge-prompt char budget (500-char turn tail + 8×160-char prefix summaries),
  the 128-token edge output cap, the 2000 ms edge timeout, and zero main-loop
  token overhead (the judge rides the local Ollama edge model, token-free).

**What this battery cannot decide.** Whether the auditor *improves* outcomes
(catches premature completions without over-downgrading) is judge-quality
dependent and needs live-provider runs — see ``live_ab_protocol`` in the
verdict. Per ADR-036, graduation therefore maps to
``ESCALATE_LIVE_AB_REQUIRED`` with ``should_flip_default=False`` (the flag
stays at its current default, off, until that battery GRADUATEs).

**Why not ``flag_ab.py``?** That harness drives the real agent against a live
provider; its verdict (pass-rate delta through the EVR-5 acceptance oracle) is
exactly the live evidence this module queues, but it cannot certify wiring or
bounded cost. This module mirrors flag_ab's reporting conventions (two JSON
snapshots, a verdict dict, a small CLI, env toggled per arm and restored) with
a deterministic executor. The documented live protocol IS flag_ab:

    python -m victor.evaluation.flag_ab --flag per_turn_auditor \\
        --base-url http://<ollama-host>:11434 --model qwen3-coder-tools:30b \\
        --corpus effect-gate --variants 2 --out-dir output/auditor_ab_live

Measure: EVR-5 acceptance-oracle GRADUATE of flag-ON vs flag-OFF
(``victor/evaluation/flag_graduation.py`` ``assess_graduation()`` over the two
snapshots), plus the premature-completion catch rate and the false-downgrade
rate on the ``effect-gate`` corpus (a fair test: those tasks tempt premature
completion). Also note ``per_turn_auditor_max_alarms`` bounds downgrades per
run (annotate-and-allow), so a too-eager judge degrades to pass-through.

    python -m victor.evaluation.auditor_ab --out-dir output/auditor_ab
"""

from __future__ import annotations

import inspect
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from victor.framework.evaluation_nodes import EvaluationDecision, EvaluationResult
from victor.framework.per_turn_auditor import (
    AuditSignal,
    AuditVerdict,
    PerTurnAuditor,
    PerTurnAuditorConfig,
)

logger = logging.getLogger(__name__)

#: The A/B flag and its env var (mirrors flag_ab.FLAG_ENV conventions).
AUDITOR_AB_FLAG = "per_turn_auditor"
AUDITOR_AB_ENV = "VICTOR_PER_TURN_AUDITOR"

#: Documented cost bounds (kept in lockstep with edge_turn_judge / EdgeModelConfig).
EDGE_OUTPUT_TOKEN_CAP = 128
EDGE_TIMEOUT_MS = 2000


@dataclass(frozen=True)
class AuditorABTask:
    """One deterministic fixture task.

    Attributes:
        task_id: Stable identifier reported in the snapshots.
        content: Assistant content of the TurnResult-shaped result.
        tool_results: Tool-result dicts of the TurnResult-shaped result.
        decision: The upstream evaluation decision before the auditor applies.
        expect_judge_call: The documented contract for judge consultations.
        expect_downgrade: The documented contract for COMPLETE downgrades.
    """

    task_id: str
    content: str
    tool_results: tuple = ()
    decision: str = "complete"
    expect_judge_call: bool = False
    expect_downgrade: bool = False


def auditor_ab_corpus(variants: int = 1) -> List[AuditorABTask]:
    """Deterministic fixture tasks (no LLM, no network).

    The corpus covers the contract cells: a healthy COMPLETE turn (content
    present — judge consulted), a COMPLETE turn grounded only in a successful
    tool call (judge consulted), the degenerate COMPLETE (heuristic alarm,
    judge skipped, downgraded), and a non-COMPLETE pass-through. ``variants``
    repeats the families with distinct seeds (flag_ab convention; all
    variants stay deterministic).
    """
    tasks: List[AuditorABTask] = []
    for v in range(max(1, variants)):
        suffix = "" if v == 0 else f"_v{v + 1}"
        tasks.extend(
            [
                AuditorABTask(
                    task_id=f"healthy_content_complete{suffix}",
                    content=f"Refactored the module and all checks pass, variant {v}.",
                    expect_judge_call=True,
                ),
                AuditorABTask(
                    task_id=f"successful_tool_complete{suffix}",
                    content="",
                    tool_results=({"name": "write_file", "success": True},),
                    expect_judge_call=True,
                ),
                AuditorABTask(
                    task_id=f"degenerate_complete{suffix}",
                    content="",
                    tool_results=({"name": "read_file", "success": False},),
                    expect_judge_call=False,
                    expect_downgrade=True,
                ),
                AuditorABTask(
                    task_id=f"non_complete_pass_through{suffix}",
                    content="",
                    decision="retry",
                    expect_judge_call=False,
                ),
            ]
        )
    return tasks


class CountingJudge:
    """A deterministic ``TurnJudge`` double that counts consultations.

    Models the always-healthy edge model: every consultation returns CONTINUE,
    so any behavioral difference between arms is attributable to the auditor's
    own wiring, not the double's opinion.
    """

    def __init__(self) -> None:
        self.calls = 0

    def judge(self, action_result: Any, state: Optional[Dict[str, Any]] = None) -> AuditSignal:
        self.calls += 1
        return AuditSignal(AuditVerdict.CONTINUE)


@dataclass
class AuditorABArmResult:
    """Aggregated outcome of one arm (mirrors BatteryResult's role)."""

    enabled: bool
    judge_calls_total: int = 0
    complete_turns: int = 0
    downgrades: int = 0
    contract_violations: int = 0
    task_reports: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def contract_ok(self) -> bool:
        return self.contract_violations == 0

    def to_dict(self) -> Dict[str, Any]:
        """Snapshot dict (the JSON artifact's per-arm payload)."""
        return {
            "enabled": self.enabled,
            "task_count": len(self.task_reports),
            "judge_calls_total": self.judge_calls_total,
            "complete_turns": self.complete_turns,
            "downgrades": self.downgrades,
            "contract_violations": self.contract_violations,
            "tasks": self.task_reports,
        }


def _action_result_for(task: AuditorABTask) -> SimpleNamespace:
    """Adapt the fixture to the runtime's TurnResult-shaped auditor input."""
    return SimpleNamespace(content=task.content, tool_results=list(task.tool_results))


def _evaluation_for(task: AuditorABTask) -> EvaluationResult:
    return EvaluationResult(decision=EvaluationDecision(task.decision), score=0.9)


def _run_task(
    task: AuditorABTask,
    *,
    enabled: bool,
    judge: CountingJudge,
    max_alarms: int = 2,
) -> Dict[str, Any]:
    """Replay one fixture through the production auditor and record the outcome."""
    auditor = PerTurnAuditor(
        PerTurnAuditorConfig(enabled=enabled, max_alarms=max_alarms),
        judge=judge if enabled else None,
    )
    calls_before = judge.calls
    evaluation_before = _evaluation_for(task)
    evaluation = auditor.apply(evaluation_before, _action_result_for(task))
    final_decision = evaluation.decision.value.lower()
    downgraded = (
        evaluation.decision is not evaluation_before.decision
        and evaluation_before.decision is EvaluationDecision.COMPLETE
        and evaluation.decision is EvaluationDecision.RETRY
    )
    judge_calls = judge.calls - calls_before
    unchanged = evaluation is evaluation_before or (
        evaluation.decision is evaluation_before.decision and downgraded is False
    )

    return {
        "task_id": task.task_id,
        "judge_calls": judge_calls,
        "upstream_decision": task.decision,
        "final_decision": final_decision,
        "decision_unchanged": unchanged,
        "downgraded": downgraded,
    }


def _contract_violations(arm: AuditorABArmResult) -> List[str]:
    """Score an arm's per-task outcomes against the documented contract.

    OFF arm: strict no-op — zero judge calls, zero downgrades, nothing changed.
    ON arm: judge consulted exactly where the fixture says; downgrades only on
    the degenerate-COMPLETE cell.
    """
    violations: List[str] = []
    for report in arm.task_reports:
        if not arm.enabled:
            if (
                report["judge_calls"] != 0
                or report["downgraded"]
                or not report["decision_unchanged"]
            ):
                violations.append(f"{report['task_id']}: OFF arm is not a strict no-op")
            continue
        expect_calls = 1 if report["expect_judge_call"] else 0
        if report["judge_calls"] != expect_calls:
            violations.append(
                f"{report['task_id']}: judge_calls={report['judge_calls']}, contract expects "
                f"{expect_calls}"
            )
        if report["downgraded"] != report["expect_downgrade"]:
            violations.append(
                f"{report['task_id']}: downgraded={report['downgraded']}, contract expects "
                f"{report['expect_downgrade']}"
            )
    return violations


def _run_arm(enabled: bool, *, variants: int) -> AuditorABArmResult:
    """Run one arm with the flag env toggled (and restored), flag_ab-style.

    The env toggle exercises the same resolver the runtime uses
    (``resolve_per_turn_auditor_enabled``), so the arm distinction is the real
    flag path, not a parallel config branch.
    """
    previous = os.environ.get(AUDITOR_AB_ENV)
    os.environ[AUDITOR_AB_ENV] = "true" if enabled else "false"
    try:
        # Resolver smoke: the toggled env must resolve through the production seam.
        resolved = resolve_flag_for_arm()
        assert (
            resolved is enabled
        ), f"flag resolver disagrees with the arm toggle (resolved {resolved}, want {enabled})"
        result = AuditorABArmResult(enabled=enabled)
        judge = CountingJudge()
        for task in auditor_ab_corpus(variants):
            report = _run_task(task, enabled=enabled, judge=judge)
            report["expect_judge_call"] = task.expect_judge_call
            report["expect_downgrade"] = task.expect_downgrade
            result.task_reports.append(report)
            result.judge_calls_total += report["judge_calls"]
            if report["upstream_decision"] == "complete":
                result.complete_turns += 1
            if report["downgraded"]:
                result.downgrades += 1
        result.contract_violations = len(_contract_violations(result))
        return result
    finally:
        if previous is None:
            os.environ.pop(AUDITOR_AB_ENV, None)
        else:
            os.environ[AUDITOR_AB_ENV] = previous


def resolve_flag_for_arm(settings: Any = None) -> bool:
    """Resolve the flag through the production seam (env → settings → default off)."""
    from victor.framework.per_turn_auditor import resolve_per_turn_auditor_enabled as _resolve

    return _resolve(settings)


def auditor_wiring_audit() -> Dict[str, bool]:
    """Static placement audit: the flag must be wired at every execution path.

    Guards the placement-not-existence class: a flag whose resolver exists but
    whose build sites never consult it is dormant. Both the buffered
    (``turn_execution_runtime``) and streaming (``chat_stream_executor``) loop
    build sites must resolve the flag and inject the edge-model judge, and the
    loop must apply the auditor on the EVALUATE seam.
    """
    from victor.agent.services import chat_stream_executor, turn_execution_runtime
    from victor.framework import agentic_loop

    stream_src = inspect.getsource(chat_stream_executor)
    buffered_src = inspect.getsource(turn_execution_runtime)
    loop_src = inspect.getsource(agentic_loop)
    return {
        "streaming_path_resolves_flag": "resolve_per_turn_auditor_enabled" in stream_src,
        "streaming_path_injects_judge": "build_edge_turn_judge" in stream_src,
        "buffered_path_resolves_flag": "resolve_per_turn_auditor_enabled" in buffered_src,
        "buffered_path_injects_judge": "build_edge_turn_judge" in buffered_src,
        "loop_applies_auditor_on_evaluate_seam": "auditor.apply" in loop_src,
    }


def auditor_cost_model(
    baseline: AuditorABArmResult, candidate: AuditorABArmResult
) -> Dict[str, Any]:
    """The documented cost model, verified against the measured arms."""
    from victor.agent.edge_turn_judge import _PREFIX_STEPS, _PREFIX_SUMMARY_CHARS, _TAIL_CHARS

    prompt_char_budget = _TAIL_CHARS + _PREFIX_STEPS * _PREFIX_SUMMARY_CHARS
    return {
        "judge_invocations_per_complete_turn_max": 1,
        "heuristic_only_turns_skip_judge": True,
        "non_complete_turns_skip_judge": True,
        "edge_prompt_char_budget": prompt_char_budget,
        "edge_prompt_token_estimate": prompt_char_budget // 4,  # repo-standard chars/4
        "edge_output_token_cap": EDGE_OUTPUT_TOKEN_CAP,
        "edge_timeout_ms": EDGE_TIMEOUT_MS,
        "main_loop_token_overhead": 0,
        "off_arm_judge_calls": baseline.judge_calls_total,
        "on_arm_judge_calls": candidate.judge_calls_total,
        "on_arm_complete_turns": candidate.complete_turns,
    }


@dataclass(frozen=True)
class AuditorABVerdict:
    """ADR-036 evidence verdict for the ``per_turn_auditor`` flag.

    The deterministic battery certifies WIRING + COST only; judge quality
    (does auditing improve end-task acceptance without over-downgrading?) is
    undecidable offline, so a wiring-verified, cost-bounded result maps to
    ``ESCALATE_LIVE_AB_REQUIRED`` with the live protocol attached and the
    default left untouched.
    """

    verdict: str
    should_flip_default: bool
    wiring_verified: bool
    live_ab_protocol: str = ""
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verdict": self.verdict,
            "should_flip_default": self.should_flip_default,
            "wiring_verified": self.wiring_verified,
            "live_ab_protocol": self.live_ab_protocol,
            "reasons": list(self.reasons),
        }


#: The documented live A/B protocol (run it when quality graduation is pursued).
LIVE_AB_PROTOCOL = (
    "python -m victor.evaluation.flag_ab --flag per_turn_auditor "
    "--base-url http://<ollama-host>:11434 --model qwen3-coder-tools:30b "
    "--corpus effect-gate --variants 2 --out-dir output/auditor_ab_live; "
    "then assess_graduation() over the two snapshots (EVR-5 oracle) plus the "
    "premature-completion catch rate and false-downgrade rate on effect-gate"
)


def assess_auditor_ab(
    baseline: AuditorABArmResult,
    candidate: AuditorABArmResult,
    *,
    wiring: Dict[str, bool],
    cost_model: Optional[Dict[str, Any]] = None,
) -> AuditorABVerdict:
    """Verdict per ADR-036: wiring + cost certified; quality escalates to the live A/B."""
    reasons: List[str] = []
    wiring_gaps = [name for name, ok in wiring.items() if not ok]
    wiring_ok = not wiring_gaps and baseline.contract_ok and candidate.contract_ok

    if not wiring_ok:
        for gap in wiring_gaps:
            reasons.append(f"wiring gap: {gap}")
        if not baseline.contract_ok:
            reasons.append(
                f"OFF arm violated the no-op contract ({baseline.contract_violations} violations)"
            )
        if not candidate.contract_ok:
            reasons.append(
                f"ON arm violated the audit contract ({candidate.contract_violations} violations)"
            )
        return AuditorABVerdict("HOLD_WIRING_INCOMPLETE", False, False, LIVE_AB_PROTOCOL, reasons)

    reasons.append(
        "wiring verified: flag resolves and the judge is injected on both execution "
        "paths; the loop applies the auditor on the EVALUATE seam"
    )
    reasons.append(
        f"OFF arm is a strict no-op ({baseline.judge_calls_total} judge calls, "
        f"{baseline.downgrades} downgrades)"
    )
    reasons.append(
        f"ON arm consults the judge only where the contract says "
        f"({candidate.judge_calls_total} calls over {candidate.complete_turns} COMPLETE turns, "
        f"{candidate.downgrades} degenerate-COMPLETE downgrades, judge skipped on the heuristic "
        f"alarm)"
    )
    if cost_model is not None:
        reasons.append(
            "cost bounded: max 1 edge call per COMPLETE turn, prompt ≤ "
            f"{cost_model['edge_prompt_char_budget']} chars (~"
            f"{cost_model['edge_prompt_token_estimate']} tokens), output ≤ "
            f"{cost_model['edge_output_token_cap']} tokens, main-loop token overhead 0"
        )
    reasons.append(
        "judge quality (premature-completion catch vs false-downgrade) is undecidable "
        "deterministically — graduation requires the live flag_ab A/B per ADR-036"
    )
    return AuditorABVerdict("ESCALATE_LIVE_AB_REQUIRED", False, True, LIVE_AB_PROTOCOL, reasons)


def run_auditor_ab(
    *,
    variants: int = 1,
    out_dir: str = ".",
) -> Dict[str, Any]:
    """Run both arms (OFF baseline, ON candidate) and emit snapshots + verdict.

    Mirrors ``flag_ab.run_flag_ab``'s output contract: two JSON snapshots
    (``per_turn_auditor_off.json`` / ``per_turn_auditor_on.json``), both arms'
    dicts inline, the wiring audit, the cost model, and the ADR-036 verdict.
    Deterministic and offline — no provider, no agent loop, no edge model.

    Returns:
        Dict with snapshot paths, both arm dicts, wiring, cost model, verdict.
    """
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    wiring = auditor_wiring_audit()
    baseline = _run_arm(False, variants=variants)
    candidate = _run_arm(True, variants=variants)
    cost = auditor_cost_model(baseline, candidate)

    off_path = out / f"{AUDITOR_AB_FLAG}_off.json"
    on_path = out / f"{AUDITOR_AB_FLAG}_on.json"
    off_path.write_text(json.dumps(baseline.to_dict(), indent=2) + "\n", encoding="utf-8")
    on_path.write_text(json.dumps(candidate.to_dict(), indent=2) + "\n", encoding="utf-8")

    verdict = assess_auditor_ab(baseline, candidate, wiring=wiring, cost_model=cost)

    return {
        "flag": AUDITOR_AB_FLAG,
        "baseline_path": str(off_path),
        "candidate_path": str(on_path),
        "baseline_battery": baseline.to_dict(),
        "candidate_battery": candidate.to_dict(),
        "wiring": dict(wiring),
        "cost_model": cost,
        "verdict": verdict.to_dict(),
    }


def _main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m victor.evaluation.auditor_ab",
        description="Deterministic A/B for the EVR-6 per_turn_auditor flag "
        "(OFF baseline vs ON candidate); certifies wiring + cost and emits "
        "snapshots + the ADR-036 verdict.",
    )
    parser.add_argument("--variants", type=int, default=1, help="Fixture repetitions")
    parser.add_argument("--out-dir", default=".", help="Where to write the two snapshot JSONs")
    args = parser.parse_args(argv)

    result = run_auditor_ab(variants=args.variants, out_dir=args.out_dir)
    print(json.dumps(result, indent=2))
    # HOLD is the only failing exit: wiring must be fixed; ESCALATE queues work.
    return 0 if result["verdict"]["verdict"] != "HOLD_WIRING_INCOMPLETE" else 1


if __name__ == "__main__":  # pragma: no cover
    import sys

    sys.exit(_main())
