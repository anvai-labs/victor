# Flag Graduation Policy (TD-17)

**Status**: Proposed (criteria below are proposals; the owner ratifies per-flag) ·
**Date**: 2026-07-05 · **Tracks**: TD-17 in the
[tech-debt register](../tech-stack.md#technical-debt-register)

## Problem

Victor's quality/safety loop is largely opt-in: `USE_POLICY_ENGINE`,
`sandbox_enabled`, `completion_strategy=rubric`, and L1 reference-aware pruning all
default OFF, with no written criteria for when a flag graduates to default-on — or
gets deleted. Opt-in flags that never graduate are dead weight; flags that graduate
without evidence are risk. (Correction 2026-07: `USE_SMART_ROUTING` already defaults
**ON** — it is not in `is_opt_in_by_default()` — so it belongs to the inverse case: a
default-on flag with no gate yet. It needs a *retro-gate* on the current default, not
graduation-to-on. See its row below.)

## The policy

Every graduation-track flag must carry four things. A flag that cannot state them is not
a graduation candidate and should be scheduled for removal.

1. **A measurable claim** — what the flag improves, stated falsifiably.
2. **A gate** — the metric, threshold, and measurement procedure that decides
   graduation. The template is ADR-011's judge-reliability gate: an offline, repeatable
   measurement against trusted ground truth, with the evidence artifact (report JSON)
   linked from the graduation PR.
3. **A fallback contract** — what the system does when the flag's premise fails at
   runtime (provider down, judge uncalibrated, platform unsupported). Fail toward the
   pre-flag behavior, never toward silent degradation.
4. **A kill criterion** — the condition under which the flag is deleted instead of
   graduated (e.g. gate unmet after N attempts, superseded design).

Graduation PRs flip the default, link the evidence, and keep the flag for one release as
an opt-out before removal of the old path.

## Per-flag status and proposed gates

| Flag | Claim | Proposed gate | Fallback contract | Status |
|------|-------|---------------|-------------------|--------|
| `completion_strategy=rubric` | Rubric+LLM completion verdicts agree with ground truth better than `EnhancedCompletionEvaluator` | ADR-011: Krippendorff α ≥ 0.7 overall **and per family** vs verifier gold, n ≥ 16/family, integrity-clean, judge identity pinned; must hold on scripted, calibration-corpus real-agent, **and shipping-distribution** trajectories; verifier gold requires independent-overlay κ ≥ 0.8. Then Prong B (`completion_strategy_ab.py`) requires ≥24 paired verifier-backed tasks (≥4/family), task success match-or-beat, no false-positive increase, and no >10%/0.25-iteration latency increase. | Judge unavailable/uncalibrated → revert to `enhanced`; heuristic fallback (measured α=−0.092) must never gate alone | **NO-GO (2026-08-05): keep `enhanced` default.** Scripted/calibration-corpus checks and identity pin landed, but the in-container-verified SWE-bench-lite re-gate failed (llama3.3:70b α=0.26; gemma4:31b α=−0.52). Prong-B machinery is available for future re-evaluation, but a task-success pass cannot override the failed Prong-A production gate. |
| `USE_POLICY_ENGINE` | ALLOW/DENY/ASK verdicts enforce governance without blocking legitimate tool use | Zero false-DENY on a recorded corpus of accepted-tool-call traces; 100% DENY on the builtin policy violation suite; latency overhead < 5 ms/call | Engine error → fail per `governance.enabled` posture (deny-closed when governance on) | No gate corpus yet — build from HTIR traces (ADR-012 machinery) |
| `sandbox_enabled` | Subprocess/code tools run isolated with no capability loss for allowed operations | Full tool test suite green under bwrap (Linux CI) and seatbelt (macOS CI); documented escape-hatch list reviewed | Sandbox init failure → currently fail-open by design; graduation requires making fail-open an explicit, logged decision | MVP hardening gaps (seccomp/egress) noted in code; not gateable until CI runs the suite sandboxed |
| `USE_SMART_ROUTING` (already ON) | Cost/latency-aware routing cuts spend without quality loss | **Retro-gate** (flag already default-on): C0 cost-trace shows ≥ 20% cost or latency reduction on the benchmark suite with no drop in task success rate (same harness, A/B). If unmet → flip default OFF, don't graduate. | Router error → static profile routing | Ships ON today (not in opt-in set); no measurement yet; benchmark harness exists (`victor/evaluation/`) |
| L1 reference-aware pruning (`enable_reference_aware_pruning`) | Prunes tool results without losing referenced context | No answer-quality regression on the evaluation suite with ≥ 30% context reduction on long-trajectory tasks | Pruning error → unpruned context (fail-open, safe) | No measurement yet |
| `effect_gated_completion` (ADR-010 / EVR-4) | Downgrading COMPLETE-without-a-verifiable-effect to RETRY eliminates confident-but-empty completions without blocking legitimate ones | EFFECT_GROUNDING trajectory battery (`EffectGroundingScorer`): completion-without-effect rate → 0 on mutation-task batteries with **no** drop in task success rate or added completion latency beyond budget, flag-on vs flag-off A/B on the same harness; ADR-012 parity batteries stay byte-stable flag-off | Gate downgrade is bounded (`max_downgrades=2`, then annotate-and-allow `effect_gate_exhausted`); gate/ledger error → evidence recording is best-effort and never breaks the loop; disabled → strict no-op (pre-flag behavior) | Kill if the A/B shows false-block RETRY loops on subtle-effect tasks that leniency tuning can't fix, or the gate is superseded by the EVR-5 acceptance oracle. Shipped opt-in, default OFF (`victor/framework/effect_gate.py`) |
| `per_turn_auditor` (FEP-0008 Phase C / EVR-6) | A pinned prefix-only auditor detects trajectories needing intervention before the independently labelled decisive failure, without alarming on healthy trajectories or breaking the interactive latency budget | Offline prerequisite (`turn_auditor_eval.py`): ≥24 independently labelled HTIR traces, ≥8 alarm-positive and ≥8 healthy, ≥4 families with ≥4 traces/family and both polarities; precision and recall ≥0.80 overall, recall ≥0.80/family, false-alarm rate ≤0.05 overall and per family, p95 judge latency ≤2 s, integrity-clean and auditor identity pinned. A later flag-OFF/ON real-agent A/B must also match-or-beat task success before graduation. | Edge judge unavailable/error/non-LLM fallback → deterministic heuristic/existing spin guards; alarm downgrades are bounded (`max_alarms=2`) then annotate-and-allow; disabled → strict no-op | **MVP + offline gate shipped, default OFF; no real-distribution evidence artifact yet.** Kill or redesign after two independently reviewed real-distribution attempts miss the quality/latency gate; an offline PASS alone never flips the default. |

## Precedent

The `completion_strategy=rubric` row is the template working end-to-end: the strategy shipped
opt-in (ADR-009), the gate was defined before measurement (ADR-011), and calibration-corpus
success was challenged on the shipping distribution. That final gate stopped an unsafe default
flip. The lesson for every other flag is that a graduation checklist must include the production
distribution and must be allowed to end in a documented no-go, not merely accumulate positive
pilot evidence.
