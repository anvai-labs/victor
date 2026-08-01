# ADR-028: Single-Agent Durable Chat Continuation

## Metadata

- **Status**: Proposed
- **Date**: 2026-08-01
- **Decision Makers**: Vijaykumar Singh
- **Related ADRs**: 023 (multi-agent team durability — the pause/resume primitives this generalizes),
  021 (terminal-native HITL — the inline modal this parks instead of blocking), 020 (interactive TUI —
  the paused-lane rendering reused for `AWAITING_APPROVAL`)
- **Companion FEP**: [FEP-0029](../../../feps/fep-0029-single-agent-durable-chat-continuation.md)

## Context

ADR-023 / FEP-0028 gave **team** runs durable pause/resume on a member's policy `ASK`: the run
persists a checkpoint, returns an `awaiting_approval` aggregate, and resumes on the same `thread_id`
once a human decides. ADR-023 explicitly deferred the **non-team single-agent chat continuation** —
"the big architectural epic".

For a single-agent run, a policy `ASK`-gated tool blocks **synchronously** on the terminal approval
modal (`victor/framework/policies/middleware.py` `_resolve_ask` → `HITLController.process_approval`).
That is correct for an interactive TTY but wrong for headless / API runs (no modal → hang or silent
deny), for **deferred** approval (a human who will approve later — the turn must be re-run from
scratch, re-sampling the model and possibly producing a *different* tool call), and for
crash/disconnect mid-`ASK` (the in-flight turn is lost).

The single agent is the top-level `AgentOrchestrator`/`AgenticLoop`; there is no `SubAgent` wrapper to
catch a pause signal, and there is no DB-backed checkpointer today (only `MemoryCheckpointer`).

## Decision

Give single-agent runs a **durable chat-continuation** capability that reuses the FEP-0028 pause
primitives rather than a parallel mechanism:

1. **Generalized pause signal.** `ApprovalPause(BaseException)` (with `MemberApprovalPause` as a
   subclass) is raised at the policy `ASK` handler when durable pause is armed, rides through the
   `except Exception` pipeline, and is caught at the **`AgenticLoop` turn boundary**.
2. **Lean durable pause record.** A `paused_run` row in **`project.db`** stores the pending assistant
   tool-call + `ApprovalRequest` + the conversation `session_id` — not the transcript, which is
   already durable via `ConversationStore`. Follows the `schema.py` versioned-migration pattern.
3. **Faithful resume.** `VictorClient.resume(run_id, decision)` rehydrates via the existing
   `resume_session(session_id)`, then executes the **persisted** gated tool call (approved) or skips it
   (rejected) and continues the loop — it does **not** re-sample the model, so the human approves the
   *exact* action the model proposed.
4. **Uniform surface.** A first-class `EventType.AWAITING_APPROVAL` + `TaskResult.status` +
   `run_id` across CLI, TUI, and API.

Opt-in via `SessionConfig.tool_approval.durable` (default `False`); the inline modal path is
byte-identical when disarmed. Defaulted on for the headless API surface.

## Rationale

- **Symmetry with teams.** One durable-pause model and one `AWAITING_APPROVAL` event across
  single-agent and multi-agent, reusing proven primitives.
- **Faithful approval.** Persisting and replaying the *pending* action (vs. re-sampling) is the only
  way to honor "approve this specific tool call".
- **Reuse, not reinvent.** The transcript is already durable (`ConversationStore`) and rehydration
  already exists (`resume_session`); the new state is just the single pause point.

## Consequences

- **Positive**: headless/API/deferred approval works; crash/disconnect mid-`ASK` is recoverable;
  consistent HITL UX across surfaces.
- **Negative**: two pause substrates (team `WorkflowCheckpoint` vs single-agent `paused_run`) until a
  unified persistent checkpointer converges them (a named follow-up); mid-turn re-entry at the loop
  boundary is the riskiest code path.
- **Neutral**: disarmed behavior (interactive TTY) is unchanged.

## Alternatives Considered

- **Persistent StateGraph checkpointer + reuse the team path.** Cleaner single substrate but requires
  building the missing DB checkpointer *and* modelling a single agent as a graph node. Kept as the
  convergence target, not the first step.
- **Re-run the whole turn on resume.** Simplest, but re-samples the model → approves a possibly
  different action. Rejected: breaks the approval contract.
- **Keep HITL inline-only.** Leaves headless/API broken. Rejected (same terminal-first argument as
  ADR-021, extended to the deferred case).

## Implementation

Phased per FEP-0029: (1) pause mechanism (in-memory), (2) durable `paused_run` persistence, (3) public
`resume` API + surfaces, (4) hardening (reject/timeout/expiry, chained pauses, GC).

## References

- [FEP-0029](../../../feps/fep-0029-single-agent-durable-chat-continuation.md),
  [ADR-023](023-multi-agent-team-durability.md), [ADR-021](021-terminal-native-hitl-and-loop-transparency.md),
  [ADR-020](020-interactive-terminal-tui.md)
- Anchors: `victor/framework/policies/middleware.py`, `victor/agent/tool_pipeline.py`,
  `victor/framework/agentic_loop.py`, `victor/framework/client.py` (`resume_session`),
  `victor/agent/member_approval_context.py`, `victor/core/schema.py`, `victor/agent/conversation/store.py`

## Revision History

| Date | Version | Changes | Author |
|------|---------|---------|--------|
| 2026-08-01 | 1.0 | Initial ADR — single-agent durable chat continuation (companion to FEP-0029) | Vijaykumar Singh |
