---
fep: "0022"
title: "Measurement-driven framework self-consistency — the runtime reachability oracle (FEP-0021 Probe A substrate)"
type: Standards Track
status: Draft
created: 2026-07-19
modified: 2026-07-19
authors:
  - name: Vijaykumar Singh
    email: vijay@anvaiops.com
    github: vjsingh1984
reviewers: []
discussion: https://github.com/anvai-labs/victor/discussions/0022
---

# FEP-0022: Measurement-driven framework self-consistency

## Summary

FEP-0021 frames "close the inner loop" and ships Probes B (capability
invalidation) and C (generated flag inventory). It leaves **Probe A — the
reachability oracle for the FP-1 "registered/defined but never invoked" class —
as a proposal**, with a measured constraint: a single `Agent.create` bootstrap
resolves only **28/139** DI registrations, so a one-run census gate would emit
~110 false positives. This FEP specifies and lands the missing piece: a
**runtime reachability oracle** that accumulates "observed use" across diverse
runs into an *ever-observed* set, and a **ratcheting CI gate** that fails the
moment a registered artifact has zero observations across the corpus.

The substrate is **inert in the production hot path** — witnesses fire only
inside the evaluation/trajectory substrate. Phase 1–3 ship **DI reachability
only** (matches FEP-0021's container census); the same substrate generalizes to
feature-flag graduation in Phase 4 (the Probe-D bridge), proving one instrument
serves both the inner loop (dead-code reachability) and the outer loop
(graduation gating).

Honest framing: FEP-0021 already proposes the dynamic census and measured why a
single run fails. **FEP-0022 is the implementation spec for that path** — the
witness, the accumulator, the oracle, and the gate — not a new idea.

## Motivation

The FP-1 defect class (F-002, F-015, F-016a–f: code that is *imported and
registered but never invoked*) is currently closed **by hand** — the F-016
call-graph audit. Two facts make a static-only gate insufficient and a naive
runtime gate useless:

- **Static analysis can't see FP-1.** The dead modules *are* imported; vulture,
  import-linter, and the `.victor/project.db` call-graph (279k edges) all see
  them as reachable. They are simply never *called*. The existing
  `test_feature_flag_references.py` proves only that a flag is *referenced in
  source*, not that it is ever *evaluated* at runtime.
- **A single run is too narrow.** The 2026-07-19 spike (FEP-0021) showed 111 of
  139 registrations unresolved after one bootstrap — 47 `*Settings` (on-demand),
  50 `*Protocol` (lazy), 13 legitimately-conditional (tools/workflows/teams/
  recovery). A gate on one run = ~110 false positives. Only a **diverse corpus**
  shrinks the "never observed" set to where zero-observation is a real signal.

The missing piece is therefore not "should we measure?" (FEP-0021 answered yes)
but **the accumulation pipeline**: per-run witnesses → merged ever-observed
artifact → diff against the live registered set → ratcheting exempt-list → CI
gate. That pipeline is what this FEP specifies.

### Goals

1. Retire the manual F-016 call-graph lens with a standing, CI-enforced
   reachability check over DI registrations.
2. Deliver the substrate (witness + accumulator + oracle) independently of the
   gate, so an offline oracle is useful before the gate blocks.
3. Keep the production hot path zero-cost — witnesses fire only when armed.

### Non-Goals

- Replacing the static call-graph complement (FEP-0021 Phase 2) — it stays as a
  cheap first-pass; this FEP is the *oracle*, not the *triage*.
- Feature-flag graduation gating itself (Probe D) — only the *signal* it needs
  is produced here, in Phase 4.
- A new eval harness — this rides the funded EVR-5 trajectory substrate.

## Proposed Change

Four mechanisms, one substrate. Phases 1–3 are DI-only.

### 1. The witness (eval-only, per type-resolution)

A lightweight "observed" recorder, scoped to a run via `contextvars`, that a
witness fires into only when recording is armed (env `VICTOR_REACHABILITY_RECORD`
or an explicit eval context) — so production sessions pay zero overhead.

- **DI:** a `resolved` witness on `ServiceDescriptor`
  (`victor/core/container.py:104`), set inside the container's type-resolution
  path (the lookup that returns/constructs a registered instance). The observed
  set is diffable against `get_registered_types()` (`victor/core/container.py:520`).

Witnesses write to a run-local sidecar (`reachability-<run-id>.jsonl`), never to
the global or project DB. The pattern mirrors the existing `NativeMetrics` sink
(`victor/native/observability.py:203`).

### 2. The accumulator (N runs → ever-observed baseline)

`scripts/reachability_accumulate.py` merges the sidecar set from a diverse
corpus into a single committed artifact
`tests/fixtures/reachability-baseline.json` — the **ever-observed** set, keyed
by artifact class (`di:<service_key>`). The corpus is the EVR-5 trajectory
suite (`victor/evaluation/trajectory_eval.py`,
`agent_adapter.run_agentic_task`), which already exercises the conditional
paths the single bootstrap missed (turns, tools, workflows, teams, errors,
compaction, recovery).

### 3. The oracle (registered ⊖ ever_observed, minus exempt)

`registered ⊖ ever_observed − exempt_list = candidate-dead set`.

The exempt-list (`reachability-exempt.txt`) captures legitimately-conditional
patterns `dead_code_triage.py` already knows (lazy protocols, plugin/registry
extension points, settings resolved on demand). It is **ratcheting**: seeded
from the first corpus run as the baseline, it only ever *grows* by explicit
human addition with a stated reason.

### 4. The CI gate (ratcheting guard test)

`tests/unit/runtime/test_reachability_oracle.py` — a guard-family test in the
existing idiom (`test_singleton_guard.py`, `test_feature_flag_manifest_guard.py`):

- Every currently-registered DI type must appear in the ever-observed baseline
  **or** the exempt-list.
- A new registration in neither → **fail**. The fix is either (a) exercise it
  in the trajectory corpus (preferred — grows real coverage) or (b) add it to
  the exempt-list with a reason.
- **Ratchet:** the gate cannot retroactively fail `develop` — the initial
  baseline is the current ever-observed set.

### Generalization (Phase 4 — Probe A → Probe D bridge)

The same substrate answers a second reachability question with one extra
witness: a hook in `FeatureFlagManager.is_enabled()`
(`victor/core/feature_flags.py:309`) recording `(flag, evaluated=True)`. "This
flag was reached and evaluated across the corpus" is the necessary condition
Probe D needs for graduation — so one instrument serves the inner loop
(dead-flag reachability) and the outer loop (graduation gating) from one stream.

## Benefits

- **Retires a recurring manual audit.** The F-016 dead-code lens becomes a
  standing CI check; every new DI registration inherits reachability
  verification for free instead of waiting for the next human sweep.
- **Substrate lands before the gate.** Phases 1–2 deliver an offline oracle that
  produces a triage candidate list immediately — useful before Phase 3 blocks —
  so the investment pays before the gate ships.
- **One instrument, two loops.** The same observed-use stream serves dead-code
  reachability (inner loop) and, in Phase 4, flag-graduation gating (outer loop)
  — the framework's evaluation-centric thesis turned inward.
- **Reuses funded substrate.** Rides EVR-5, the existing guard-test idiom +
  `ci-success` gate, the `.victor/project.db` graph, and the `NativeMetrics`
  sink pattern — no parallel system.
- **Prevents future bug classes, not just FP-1.** Any future registered/defined
  artifact inherits the witness and the gate by construction.

## Implementation Plan

> **Status (2026-07-19):** Phase 1 shipped; Phase 2 tooling shipped (offline
> oracle usable now). Baseline seeding, triage, and the CI gate (Phase 2 tail +
> Phase 3) await the EVR-5 trajectory corpus.

### Phase 1 — Witness + recorder (foundation, DI-only) — ✅ SHIPPED

- [x] `ReachabilityRecorder` (contextvars-scoped, armed by env/eval context).
- [x] DI `resolved` witness on `ServiceDescriptor` in the container's
      type-resolution path.
- [x] Run-local sidecar flush; inert when disarmed.

**Deliverable:** the substrate, with unit tests proving zero hot-path cost when
disarmed. Shipped in `b9e823192` — `victor/runtime/reachability.py`,
`victor/core/container.py`, `tests/unit/runtime/test_reachability_recorder.py`
(13 tests incl. the disarm-overhead microbenchmark).

### Phase 2 — Accumulator + exempt-list — ✅ TOOLING SHIPPED

- [x] `scripts/reachability_accumulate.py` (`accumulate` + `report`) and the
      pure oracle in `victor/runtime/reachability.py` (`merge_sidecar_paths`,
      `write/load_baseline`, `load_exempt`, `candidate_dead`).
- [x] `reachability-exempt.txt` template seeded (`tests/fixtures/`).
- [ ] Seed `reachability-baseline.json` from a full corpus run — **awaits the
      EVR-5 trajectory corpus** (a bootstrap resolves only 28/139; the corpus is
      what makes the candidate list meaningful).
- [ ] Triage the residual candidate list against `dead_code_triage.py`
      heuristics and `comprehensive_graph_analysis.py:241 analyze_dead_code`.

**Deliverable:** an offline oracle producing a (noisy, triage-required)
candidate-dead list — useful immediately, no gate. Tooling shipped in
`667519b57` and verified end-to-end; baseline seeding + triage follow the first
corpus run.

### Phase 3 — CI gate (advisory → blocking)

- [ ] `tests/unit/runtime/test_reachability_oracle.py` (ratcheting, exempt-aware).
- [ ] Land **advisory** first (warn, don't fail); flip to blocking once the
      exempt-list stabilizes.
- [ ] Wire EVR-5 trajectory runs to arm the recorder in CI.

**Deliverable:** the standing reachability gate.

### Phase 4 — Generalize (flags + Probe D bridge)

- [ ] Flag-eval witness in `FeatureFlagManager.is_enabled()`.
- [ ] Expose the flag ever-evaluated signal to Probe D's graduation gates.
- [ ] Extend the witness to tool-path / provider-code reachability (FP-1 tail).

**Deliverable:** one substrate, two consumers; the inner/outer-loop unification.

### Testing Strategy

- **Unit:** witness disarm-overhead microbenchmark (sub-µs, no allocation on the
  disarmed path); accumulator determinism (same runs → byte-identical baseline);
  gate ratchet replay (baseline cannot fail `develop`).
- **Integration:** parity with `analyze_dead_code` on the known F-016 dead set
  (recall/precision vs the manual lens).
- **Backward compatibility:** a run with the recorder disarmed is byte-identical
  to a run without the FEP.

## Drawbacks and Alternatives

- **Static-only reachability (call-graph / vulture / import-linter).** Rejected
  as the oracle: it cannot see the FP-1 class because the dead modules *are*
  imported and therefore appear reachable. It is retained as a cheap *complement*
  (FEP-0021 Phase 2) for the cases it does catch, but precision on the real
  target is zero.
- **Single-run census gate.** Rejected: the 2026-07-19 spike measured ~110 false
  positives and zero true positives from one bootstrap, because 111 of 139
  registrations are legitimately lazy or conditional. Only a diverse corpus makes
  zero-observation a meaningful signal.
- **Hot-path overhead.** Mitigated, not eliminated: every resolution takes a
  guarded branch on an armed-flag. The guard is a single boolean read on the
  disarmed path; the sink runs only inside the eval job, never in production.
- **Baseline drift / gaming.** A registration could be "observed" by a trivial
  touch in the corpus. Mitigated by requiring the *trajectory* corpus (real
  agentic tasks, not arbitrary unit calls) and by human-reviewed exempt entries
  with a stated reason.

## Unresolved Questions

- Minimum corpus diversity/size before the Phase 3 gate may block (relates to
  EVR-1/EVR-5 funding — the same dependency FEP-0021 already accepts).
- Witness storage: sidecar JSONL vs a `reachability` table in
  `.victor/project.db`.
- Whether DI reachability and (Phase 4) flag reachability share one gate or two.

## Migration Path

Additive throughout; no public API changes and no behavior change for agents.
Phases 1–2 land as a new offline tool plus a committed baseline artifact
(`tests/fixtures/reachability-baseline.json`). Phase 3 lands as a new
guard-family test in the existing idiom (like the singleton/hotspot guards),
seeded with the current ever-observed set as the initial exempt-list so it
**cannot retroactively fail `develop`**. The gate ships **advisory** first
(warn, don't fail) and is flipped to blocking only once the exempt-list
stabilizes. Phase 3–4 blocking behavior is gated behind the EVR-5 corpus and
does not block until that corpus exists.

## Compatibility

Additive. No public API or behavior change for agents. Witnesses are inert
outside the armed eval context; the gate is a new CI signal that cannot
retroactively fail `develop` (ratcheting). No SDK, provider, or vertical impact.

## References

- Parent: [FEP-0021](fep-0021-close-the-inner-loop.md) (Probe A proposal + the
  28/139 spike that motivates the accumulation pipeline).
- Verification hook: [FEP-0018](fep-0018-framework-verification-hook.md).
- Existing guard idiom: `tests/unit/runtime/test_singleton_guard.py`,
  `tests/unit/runtime/test_feature_flag_manifest_guard.py`,
  `tests/unit/contracts/test_feature_flag_references.py`.
- Static complement: `scripts/comprehensive_graph_analysis.py:241 analyze_dead_code`,
  `scripts/dead_code_triage.py`.
- Substrate: `docs/architecture/evaluation-centric-runtime-backlog.md`
  (EVR-1/EVR-5), `victor/evaluation/trajectory_eval.py`,
  `victor/evaluation/agent_adapter.py`.

## Copyright

This FEP is licensed under the Apache License 2.0, same as the Victor project.
