# ADR-023: Multi-Agent Team Durability — Checkpoint, Interrupt, Per-Member Streaming

## Metadata

- **Status**: Proposed
- **Date**: 2026-07-29
- **Decision Makers**: Vijaykumar Singh
- **Related ADRs**: 003 (workflow engine / StateGraph — the primitives this propagates), 021
  (terminal-native HITL — where team interrupts surface)
- **Work tracked by**: [TD-25](../../tech-stack.md#technical-debt-register)
- **Benchmark**: [competitive-benchmark-2026-07.md](../competitive-benchmark-2026-07.md) §4

## Context

Victor's multi-agent model is a genuine strength: `UnifiedTeamCoordinator` is used *directly* as a
StateGraph node (teams are formations — SEQUENTIAL/PARALLEL/HIERARCHICAL/PIPELINE — not a separate
graph abstraction), with `WorkspaceIsolation` per member. This is arguably cleaner than CrewAI's
role model or AutoGen's conversation model, and CLAUDE.md deliberately forbids a wrapper node type.

The StateGraph engine (ADR-003) already has **checkpointing and HITL-interrupt primitives**. The gap
is that a *team node* does not propagate them to its members' execution:

- **No team-run checkpoint/resume.** A long multi-member run cannot be durably checkpointed and
  replayed at member granularity; a crash mid-formation loses in-flight member state. LangGraph — the
  category bar — makes checkpoint + time-travel replay first-class.
- **HITL mid-run is web-only.** A member hitting an approval gate defers to Chainlit (see ADR-021);
  there is no terminal-native interrupt/resume for a paused team member.
- **Per-member streaming is incomplete.** The user sees aggregate progress, not each member's live
  stream — hard to follow a HIERARCHICAL or PARALLEL formation.

## Decision

Give team nodes a **durability contract** that propagates the StateGraph primitives to member
execution, without introducing a new multi-agent graph abstraction:

1. **Member-granular checkpoint/resume.** `UnifiedTeamCoordinator` checkpoints per-member state to
   the StateGraph checkpointer so a formation resumes at the last completed member (or mid-member
   step where the member itself checkpoints), not from the top.
2. **Interruptible members.** A member's approval/`ASK` gate raises a StateGraph `interrupt`, so the
   team run *pauses durably* and resumes on approval — surfaced terminal-natively via ADR-021 (not
   browser-bound).
3. **Per-member streaming.** Each member's `RenderAction` stream is tagged with member identity and
   fanned out so the TUI (ADR-020) can show per-member progress lanes.

This is a *contract on the team node* — the public shape of how a team participates in a graph —
so it is FEP-gated (below).

## Rationale

- **First principles.** Durability is the property that separates a demo multi-agent run from a
  production one: it must survive a crash, pause for a human, and be observable per participant.
  LangGraph set that bar; formations without it are behind regardless of how clean the topology model
  is.
- **Reuse, not reinvent.** The checkpointer and `interrupt` already exist at the graph layer; this
  ADR *threads them through* the team node rather than building a parallel mechanism — honoring the
  "teams are formations, not graphs" rule.
- **Co-design.** Depends on ADR-021 for the terminal interrupt surface and ADR-020 for per-member
  streaming lanes; the three specify one coherent multi-agent UX.

## Consequences

- **Positive**: crash-safe, pausable, observable team runs; parity with LangGraph on the durability
  contract while keeping the cleaner formation model.
- **Negative**: checkpoint volume grows (member-granular snapshots); the team-node contract change is
  a public surface ⇒ FEP + migration for existing team definitions.
- **Neutral**: single-agent runs and the formation taxonomy are unchanged.

## Implementation

- **Companion FEP** — the team-node durability contract (checkpoint identity, interrupt semantics,
  member-tagged streaming) is a `victor.framework` public surface: ratified in
  [FEP-0028](../../../feps/fep-0028-team-node-durability-contract.md). Increments:
  1. member-granular checkpoint/resume via the existing StateGraph checkpointer — **done**: SEQUENTIAL
     and PIPELINE (per-member), PARALLEL (lock-protected concurrent completed-set), and HIERARCHICAL
     (phase-granular — plan/specialists/synthesis snapshotted into `shared_state["__hier__"]`, resume
     restores up to the last completed phase). Concurrent mid-wave partial resume deferred;
  2. member `interrupt` — slice 2a (terminal-native member approval) **done**: a member's policy
     `ASK`-gated tool surfaces to the shared terminal approval modal (ADR-021), tagged with `member_id`,
     via a member-tagging wrapper published on a `ContextVar` during member-orchestrator construction
     (read by the policy-engine builder). Slice 2b-infra (durable pause/resume **mechanism**) **done**:
     a member reporting `metadata["awaiting_approval"]` stops the SEQUENTIAL formation (opt-in
     `pause_hook`), persists a pause checkpoint (paused member re-runs on resume), and returns a paused
     aggregate; resume re-runs the paused member via the pillar-1 checkpoint path. The real mid-member
     ASK→pause trigger is now **done**: a durable team run arms `current_member_durable_pause_enabled`,
     a member's `ASK` raises `MemberApprovalPause` (a `BaseException` that survives the runtime-core
     `except Exception` path), caught at `SubAgent.execute` and surfaced as the awaiting result.
     Concurrent durable pause is now **done for PARALLEL**: a wave collects every awaiting member into
     a multi-pause aggregate (`__awaiting_approvals__` + one batch pause checkpoint) and a resumed run
     re-runs exactly the paused set. HIERARCHICAL concurrent pause and the no-graph chat continuation
     (non-team single agent) are deferred;
  3. member-tagged `RenderAction` fan-out for ADR-020's per-member lanes — **done (SEQUENTIAL,
     FEP-0028 increment 4)**: a `MemberEventSink`/`ContextVar` teams→stream bridge in
     `stream_with_events` emits `member_start`/`member_completed`/`member_error` lanes; member
     tool/token streaming and concurrent formations deferred.

## Alternatives Considered

- **A dedicated multi-agent graph abstraction.** Rejected — explicitly forbidden (CLAUDE.md; teams
  are formations used directly as nodes). Durability must ride the existing StateGraph primitives.
- **Checkpoint only at formation boundaries (not per member).** Rejected: loses in-flight member
  work on crash; the whole point is member-granular resume.
- **Keep HITL web-only for teams.** Rejected: same terminal-first argument as ADR-021.

## References

- [ADR-003](003-workflow-engine.md), [ADR-020](020-interactive-terminal-tui.md),
  [ADR-021](021-terminal-native-hitl-and-loop-transparency.md)
- `victor/teams/unified_coordinator.py`, `victor/workflows/unified_compiler.py`,
  [TD-10](../../tech-stack.md#technical-debt-register) (workspace isolation rename)

## Revision History

| Date | Version | Changes | Author |
|------|---------|---------|--------|
| 2026-07-29 | 1.0 | Initial ADR — team durability contract (checkpoint/interrupt/per-member stream) | Vijaykumar Singh |
| 2026-07-30 | 1.1 | Increment 1 (member checkpoint/resume) and increment 4 (per-member streaming lanes) landed for SEQUENTIAL via FEP-0028 | Vijaykumar Singh |
| 2026-07-30 | 1.2 | Increment 3 slice 2a (terminal-native member approval: member ASK → shared modal, member_id-tagged) landed; durable pause/resume (2b) deferred | Vijaykumar Singh |
| 2026-07-31 | 1.3 | Increment 3 slice 2b-infra (durable member pause checkpoint + resume re-run at the teams layer) landed for SEQUENTIAL; real ASK trigger + chat continuation deferred | Vijaykumar Singh |
| 2026-07-31 | 1.4 | Increment 3 real mid-member ASK→durable-pause trigger landed (MemberApprovalPause BaseException, armed for durable team runs); non-team chat continuation still deferred | Vijaykumar Singh |
| 2026-07-31 | 1.5 | Concurrent-formation per-member streaming lanes (PARALLEL/HIERARCHICAL via a shared helper) + durable-pause arming gated to supports_durable_pause() (fixes a latent #740 abort); concurrent checkpoint/pause still deferred | Vijaykumar Singh |
| 2026-07-31 | 1.6 | PIPELINE full durability (checkpoint/resume/pause/lanes) via the shared sequential machinery — it's a sequential formation; CONSENSUS/REFLECTION + concurrent checkpoint/pause remain deferred | Vijaykumar Singh |
| 2026-07-31 | 1.7 | CONSENSUS + REFLECTION streaming lanes landed — per-member lanes now cover all six formations; concurrent durable checkpoint/pause is the last deferred item | Vijaykumar Singh |
| 2026-07-31 | 1.8 | Concurrent durable checkpoint/resume for PARALLEL landed (lock-protected cumulative-completed-set checkpoint; execution stays concurrent) — resolves the FEP's concurrent-resume open question; concurrent pause + HIERARCHICAL checkpoint deferred | Vijaykumar Singh |
| 2026-07-31 | 1.9 | Concurrent durable pause/resume for PARALLEL landed — a wave collects every awaiting member into a multi-pause aggregate (`__awaiting_approvals__` + one batch pause checkpoint); resume re-runs exactly the paused set; `PARALLEL.supports_durable_pause()` now True. HIERARCHICAL concurrent pause + non-team chat continuation deferred | Vijaykumar Singh |
| 2026-08-01 | 1.10 | HIERARCHICAL phase-granular checkpoint/resume landed — plan/specialists/synthesis each snapshotted into `shared_state["__hier__"]` (persisted by the existing member hook; pure formation-layer change); resume restores up to the last completed phase (completed plan restored not re-run, drives phase 2), fallback path ends at phase 2. Mid-wave partial resume + HIERARCHICAL pause deferred | Vijaykumar Singh |
