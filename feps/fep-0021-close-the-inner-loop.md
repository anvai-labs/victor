---
fep: "0021"
title: "Close the inner loop — generation + invalidation + reachability for framework self-consistency"
type: Standards Track
status: Draft
created: 2026-07-19
modified: 2026-07-19
authors:
  - name: Vijaykumar Singh
    email: vijay@anvaiops.com
    github: vjsingh1984
reviewers: []
discussion: https://github.com/vjsingh1984/victor/discussions/0021
---

# FEP-0021: Close the inner loop

## Summary

Victor's north star (VISION / the Evaluation-Centric Runtime) is *"close the evaluation
loop; effect over assertion; one canonical loop."* The project applies this rigorously to
**agent outputs** — the *outer* loop — and not at all to **its own internal facts** — the
*inner* loop. This FEP makes the framework's internal facts self-consistent by the same
discipline: every derived internal artifact (a DI registration, a model-scoped capability,
a documented architecture fact) must be either **live-derived**, **invalidated on change**,
or **loop-closed by a CI/runtime invariant** — never asserted-and-cached.

The work is one initiative with four probes on a shared eval/CI substrate (the P0 EVR-1/EVR-5
harness). Each probe targets a distinct divergence class — dead registrations, stale
model-scoped capabilities, drifted architecture docs, and ungated feature-flag defaults — but
all four share the same remedy: make the fact self-checking rather than hand-maintained. Two
probes (capability invalidation and generated architecture facts) have already shipped as
Phase 1 with their motivating doc-truth corrections; this FEP frames the whole initiative and
specifies the two remaining probes (a reachability oracle and executable flag-graduation gates),
their sequencing behind the evaluation harness, and the honest constraints on each.

## Motivation

A first-principles reverification of the codebase self-analysis found four findings that look
unrelated but share **one defect shape** — *a value derived once and then asserted-and-cached,
silently diverging from the live source of truth it came from*:

- **FP-1 (dead code).** A **registration** asserts "this capability is wired"; nothing verifies
  it is ever *invoked*. F-002 (`credit_runtime` no-op), F-015 (orphaned hybrid-compaction,
  ~2,082 LOC), F-016a–f (dead subsystems / a never-called credit feedback loop) were all this
  class. Each was found by a **manual** call-graph audit (`CODEBASE_REVIEW_BACKLOG.md`), the
  "recommended lens for future sweeps" — i.e. a loop closed by hand.
- **FP-2 (stale capability).** `tool_calling_caps` cached at construction diverged from the
  provider's live caps after a mid-session switch (F-016m).
- **FP-3 (doc drift).** TD-9 asserted "streaming not integrated" long after the FEP-0007
  cutover integrated it — drift that **misled the self-analysis itself** into a wrong finding.
- **FP-4 (flag drift).** `USE_SMART_ROUTING` was documented "default OFF" in three prose sites
  while the code defaulted it **ON**; the flag count and a ghost flag were also wrong.

The project keeps closing these loops **by hand** — the F-016 manual audit, the pinned-count
doc gate, the singleton/size guards. That is a micro-optimization treadmill: each divergence is
patched individually and the *class* recurs. The co-optimization is to build the inner loop
once, riding the eval/CI substrate the roadmap already funds, so each probe makes a whole class
of divergence impossible-by-construction or caught-in-CI.

## Proposed Change

Four probes, one thesis. Each closes a divergence class:

### Probe A — Reachability oracle (closes FP-1) — *proposed*

Retire the manual F-016 lens by making "registered/constructed but never invoked" a standing
check. **Honest constraint:** this class is *not* catchable by static import-graph analysis —
the dead modules *are* imported; they are simply never called. Two viable engines, both
already-present assets:

- **Dynamic resolution census.** Add a lightweight "resolved" witness to `ServiceDescriptor`
  (`victor/core/container.py`); diff `get_registered_types()` against the observed resolved-set
  and flag zero-resolution registrations.

  **Measured (2026-07-19 spike):** why this must ride *diverse* runs, not a single bootstrap or
  the unit suite. Instrumenting the real container and bootstrapping an agent
  (`Agent.create`) resolves only **28/139 registrations (20%)** at construction time. The other
  **111 unresolved** decompose as **47 `*Settings`** (config, resolved on demand) + **50
  `*Protocol`** (lazy service interfaces) + **13 conditional services** — and all 13 are
  legitimately conditional (`ToolPipeline`/`ToolCacheManager` resolve during tool execution,
  `WorkflowCompilerImpl`/`WorkflowValidator`/`CompiledWorkflowExecutor` only on workflow runs,
  `OrchestratorPool` on teams, `IBudgetManager`/`IToolAccessController`/`IPathResolver` per turn).
  **Zero clean dead candidates from a single run** → a per-PR census gate on one bootstrap would
  be ~110 false positives / 0 true positives. The unit suite is no better a driver: it mostly
  *mocks* the container. Only a set of runs that exercises the conditional paths (turns, tools,
  workflows, teams, errors, compaction, recovery) shrinks the "legitimately never resolved" set
  to where zero-resolution is a real signal — which is exactly the **EVR-5** acceptance-oracle
  trajectory suite (`victor/evaluation/trajectory_eval.py`, `agent_adapter.run_agentic_task`).
  So the census **gate** is genuinely EVR-5-gated (now measured, not asserted); the same
  instrument that grades agent quality doubles as the liveness oracle (outer + inner loop, one
  instrument). An offline audit via the same technique is possible today but yields a noisy
  candidate list requiring per-service triage — a manual lens, not a gate.
- **Call-graph.** Promote the manual method to CI: reuse the `.victor/project.db` code graph
  (279k edges) + `scripts/comprehensive_graph_analysis.py::analyze_dead_code` +
  `scripts/dead_code_triage.py`'s DI-aware false-positive heuristics, wrapped as a guard-family
  test with a ratcheting exempt-list.

### Probe B — Capability invalidation seam (closes FP-2) — **DONE (Phase 1)**

A provider/model switch is an **event that invalidates model-scoped derived state**, not a
silent field copy. `ProviderManagementRuntime._resync_model_derived_state()` re-reads the
already-refreshed `provider_manager.capabilities` and re-pushes `tool_calling_caps` / `tool_budget`
/ tracker exploration via existing setters. Shipped in PR #558.

### Probe C — Architecture facts generated, not asserted (closes FP-3/FP-4 drift) — **DONE (Phase 1)**

Flag defaults are now a **generated** artifact (`docs/architecture/feature-flags.md` from
`get_flag_manifest()`), byte-compared in CI, and exposed via `victor capabilities --json`.
The prose class of drift that PR #555 fixed by hand can no longer recur. Shipped in PR #557.

### Probe D — Executable flag-graduation gates (closes FP-4 properly) — *proposed*

Generalize the one working template (`benchmarks/judge_calibration/run_offline_calibration.py`,
the rubric judge at α=0.929) to every graduation-track flag: each gate's report JSON wrapped in
a pass/fail CI assertion, with the generated flag table (Probe C) rendering live GREEN/RED per
flag. Most gate corpora *are* EVR eval sets (policy-engine needs HTIR traces; routing needs the
cost-trace A/B on `victor/evaluation/`), so Probe D is sequenced behind EVR-1/EVR-5.

## Benefits

- One discipline, four divergence classes retired — not four bespoke patches maintained forever.
- Reuses funded substrate rather than adding a parallel system: EVR-1/EVR-5 (P0), the existing
  guard-test idiom + `ci-success` gate, the `.victor/project.db` code graph, and the
  `check_docs_drift.py` derivation path. The marginal cost of each probe is small once the shared
  instrument exists.
- Turns the framework's own thesis inward — the outer loop (agent quality) and inner loop
  (framework truth) share the acceptance-oracle instrument, so one investment pays twice.
- Kills the recurring cost of hand-run audits (the F-016 manual call-graph sweep) and
  hand-maintained fact tables (TD-16/17), replacing "someone remembers to re-check" with a gate
  that fails the moment code and its documented/registered shadow disagree.
- Prevents whole *future* bug classes, not just the four found: any new model-derived field or
  DI registration inherits the invalidation seam and the reachability check for free.

## Drawbacks and Alternatives

- **Do nothing / keep manual audits.** Rejected: the F-016 class demonstrably recurs and each
  round is a full human audit; drift already misled the self-analysis (FP-3).
- **Static-only reachability (vulture/import-linter).** Rejected as sufficient: it cannot see the
  F-016 class (dead code is imported). Static analysis is a cheap *complement*, not the oracle.
- **Prose-scanning drift gate for flags.** Rejected: natural-language default claims are
  ambiguous ("wrongly listed as OFF", "defaults ON") → false positives on a gate that blocks
  every PR. Probe C uses a deterministic generated artifact instead.
- **Container census overhead.** The "resolved" witness is a single boolean write per resolve;
  the census runs only in the eval job, not the hot path.

## Unresolved Questions

- Probe A engine choice: dynamic census vs call-graph vs both. Dynamic is precise but needs a
  representative eval run; call-graph is always-on but needs the DI-aware exempt-list curated.
- How to seed Probe A's exempt-list without re-flagging the legitimate protocol/registry/plugin
  patterns `dead_code_triage.py` already knows about.
- Probe D: minimum gate-corpus size per flag before a graduation PR may flip a default; owner
  ratification workflow (ties into `flag-graduation-policy.md`).
- Services block for the generated manifest (deferred from Probe C): needs a canonical service
  registry first (TD-15) rather than a hardcoded tuple.

## Implementation Plan

- **Phase 1 — shipped.** Probe C (PR #557, generated flag inventory + regen guard) and Probe B
  (PR #558, capability invalidation seam). Doc-truth corrections that motivated them: PR #555.
- **Phase 2 — Probe A (call-graph guard).** Promote `analyze_dead_code` + `dead_code_triage`
  heuristics to a guard-family CI test with a ratcheting exempt-list. Cheap, always-on, no EVR
  dependency.
- **Phase 3 — Probe A (dynamic census).** Add the `ServiceDescriptor` resolved-witness and wire
  the census into the EVR-5 acceptance-oracle run. Depends on EVR-1/EVR-5.
- **Phase 4 — Probe D.** Wrap the rubric gate as a CI assertion (template), then extend per flag
  as EVR gate corpora land. Depends on EVR-1/EVR-5.

## Migration Path

Additive throughout. No public API changes; no behavior change for agents. Phase 2 lands as a new
guard test (like the existing singleton/hotspot guards) with the current dead-code baseline as the
initial exempt-list, so it cannot retroactively fail develop. Phases 3–4 are new CI signals gated
behind the EVR harness and do not block until their corpora exist.

## Compatibility

Backward compatible. Probes add CI/runtime invariants and generated artifacts; they do not change
framework public APIs, the agentic loop, provider adapters, or the SDK contract. The dynamic
census is inert outside the eval job.

## References

- Reverification findings: `docs/architecture/CODEBASE_REVIEW_BACKLOG.md` (F-002, F-015, F-016a–m)
- Tech-debt: `docs/tech-stack.md` (TD-9, TD-14, TD-15, TD-16, TD-17)
- Substrate: `docs/architecture/evaluation-centric-runtime-backlog.md` (EVR-1, EVR-5),
  `docs/architecture/adr/012-regression-gated-harness-acceptance.md`
- Phase 1 PRs: #555 (docs-truth), #557 (Probe C), #558 (Probe B)
- Generated inventory: `docs/architecture/feature-flags.md`;
  policy: `docs/architecture/flag-graduation-policy.md`
