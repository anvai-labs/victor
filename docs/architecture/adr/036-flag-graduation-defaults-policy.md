# ADR-036: Flag graduation to defaults — policy and current queue

## Metadata

- **Status**: Proposed
- **Date**: 2026-11-16
- **Decision Makers**: Vijaykumar Singh
- **Related Documents**: ADR-010 (effect-gated completion, opt-in), ADR-009/ADR-011 (completion strategies), ADR-012 (EVR-5 acceptance oracle), FEP-0047, `victor/evaluation/flag_graduation.py`, `victor/config/groups/agent_config.py`, `victor/core/feature_flags.py`

## Context

Victor's most advanced verification and learning machinery is flag-gated **off by default**:
`effect_gated_completion` (ADR-010; `AgentSettings.effect_gated_completion=False`),
`per_turn_auditor` (EVR-6/FEP-0008 Phase C; `AgentSettings.per_turn_auditor=False`), the
semantic response cache (`use_semantic_response_cache=False`), and the StateGraph agentic
executor (`USE_STATEGRAPH_AGENTIC_LOOP=False`). The ADR-010 index row itself records the
intended endgame: *"Shipped, opt-in … default off pending flag-graduation gate."* The
graduation harness exists (`victor/evaluation/flag_graduation.py` — `assess_graduation()`
comparing flag-ON vs flag-OFF batteries through the EVR-5 acceptance oracle, with a CLI and
Makefile target) but no codified policy says what graduates, when, or what evidence a default
flip requires.

The strategic cost is that "strong but opt-in" functions as "off" for every user who does
not read the docs: the capabilities that differentiate Victor (verifiable completion, online
auditing) are invisible in default behavior. A policy gap is also a consistency gap — flags
graduate (or stall) based on individual judgment rather than shared criteria.

## Decision

1. **Graduation is evidence-gated, not judgment-gated.** A flag graduates to default-on when
   `assess_graduation()` returns GRADUATE over a recorded battery: flag-ON matches-or-beats
   flag-OFF on the EVR-5 acceptance oracle across the standard battery, with baselines
   committed under `output/` (gitignored snapshots are evidence in the ledger, not in git).
2. **Every default-on flip ships with a documented opt-out and a negative test** pinning that
   the opt-out restores prior behavior byte-for-byte (env override via the existing
   `VICTOR_*` machinery; per-call override where the surface allows).
3. **A graduated flag may be demoted** by the same evidence standard: a regression battery or
   field telemetry showing flag-ON losing to flag-OFF reverts the default, with the demotion
   recorded in this ADR's queue table.
4. **The queue is public and current.** The table below is maintained in this ADR (the
   decision record); task-level evidence lives in the FOU ledger.
5. **Executor-parity exception**: a flag whose ON-path crashes on documented inputs (the
   StateGraph executor's `total_timeout`/verifier/streaming `NotImplementedError`s) is not
   graduation-eligible until parity lands — graduation never converts working runs into
   crashes.
6. **Default-on stateful graduation adds storage-boundary evidence.** A default-on StateGraph
   promotion must show that durable state belongs to a documented logical scope, does not
   silently add a physical database, is JSON-safe and runtime-handle-free, and writes only at
   semantically useful boundaries. For the agentic loop those boundaries are completed ACT and
   interrupt/approval, not every lightweight node.
7. **Streaming claims are scoped to observed chunks.** Turn-boundary StateGraph streaming is
   not token-level ACT streaming; graduation evidence and release notes must name the observed
   boundary rather than borrowing a stronger parity claim.

### Current queue

| Flag | Default today | Target | Prerequisite | Status |
|------|--------------|--------|--------------|--------|
| `effect_gated_completion` | off | on | graduation battery GRADUATE + negative test | queued (FOU-S4) |
| `per_turn_auditor` | off | on | graduation battery GRADUATE + negative test | queued (FOU-S4) |
| `USE_STATEGRAPH_AGENTIC_LOOP` | off | on | executor parity (FOU I-D5a/b/c) + parity suite both modes + durable ACT-boundary semantics + project-scoped checkpoints + selective writes + terminal policy propagation + sandbox graduation evidence + soak | blocked on I-D5 |
| `use_semantic_response_cache` | off | undecided | battery on cache-hit quality + staleness semantics | parked |
| `USE_DURABLE_GRAPH_CHECKPOINTS` (FEP-0047) | n/a | on-at-birth | ships default-on with opt-out from first merge | covered by FEP-0047 |

## Rationale

The graduation harness was built precisely to remove vibes from this decision; the missing
piece was a committed policy that (a) names the evidence standard, (b) demands the opt-out
and its negative test in the same PR, and (c) allows demotion so a default flip is falsifiable
rather than permanent. The executor-parity exception encodes the VAS program's core rule —
never equate local pass with live acceptance — at flag granularity.

## Consequences

- **Positive**: Victor's differentiating verification behavior becomes default behavior;
  default flips arrive with reproducible evidence; demotion gives a safe rollback path; the
  flag queue makes "when does X turn on" answerable.
- **Negative**: graduation requires battery runs per flag (compute + maintenance of the
  battery corpus); defaults become time-varying, so release notes must record every flip and
  demotion; some users will dislike behavior changes even with opt-outs.
- **Neutral**: flags that are genuinely experimental (e.g., rubric completion under
  calibrated-model constraints per ADR-011) stay opt-in indefinitely — the queue records that
  too, by omission or explicit parked status.

## Implementation

FOU-S4 (first application): run the graduation battery for `effect_gated_completion` and
`per_turn_auditor`; on GRADUATE, flip defaults in one PR each including the negative
opt-out test and battery evidence paths in the FOU ledger. If the verdict is HOLD, the flag
stays off and the blocking evidence gap is recorded in the queue above. No code changes are
required to the harness itself in v1; if battery coverage proves insufficient for a flag, the
gap is fixed before the flip, not after.

## Alternatives Considered

- **Flip everything on now without batteries**: rejected; it violates the repo's own
  evidence discipline and risks shipping regressions in the name of progress.
- **Keep everything opt-in indefinitely**: rejected; it concedes the observed-behavior gap
  this program exists to close (competitive review, Tier S rationale).
- **Automatic graduation on green CI**: rejected; the acceptance battery is a deliberate
  promotion gate, not a unit-test proxy.

## References

- `victor/evaluation/flag_graduation.py` — `assess_graduation()`, `GraduationVerdict`
- `victor/evaluation/acceptance_oracle.py` — EVR-5 oracle (ADR-012)
- `victor/config/groups/agent_config.py` — current flag defaults
- `docs/architecture/victor-foundations-plan.md` — FOU-S4 task evidence
- ADR-010, ADR-011, ADR-012 — prior opt-in decisions this policy graduates

## Revision History

| Date | Version | Changes | Author |
|------|---------|---------|--------|
| 2026-11-16 | 1.0 | Initial ADR | Vijaykumar Singh |
| 2026-11-18 | 1.1 | Added StateGraph storage-boundary, selective checkpoint, terminal-policy, sandbox, and scoped streaming graduation gates (FEP-0050) | Vijaykumar Singh |
