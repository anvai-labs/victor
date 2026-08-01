---
fep: 0029
title: "Single-Agent Durable Chat Continuation (pause/resume on approval)"
type: Standards Track
status: Draft
created: 2026-08-01
modified: 2026-08-01
authors:
  - name: Vijaykumar Singh
    email: singhvjd@gmail.com
    github: singhvjd
reviewers: []
discussion: https://github.com/anvai-labs/victor/discussions/0029
---

# FEP-0029: Single-Agent Durable Chat Continuation

## Table of Contents

1. [Summary](#summary)
2. [Motivation](#motivation)
3. [Proposed Change](#proposed-change)
4. [Benefits](#benefits)
5. [Drawbacks and Alternatives](#drawbacks-and-alternatives)
6. [Unresolved Questions](#unresolved-questions)
7. [Implementation Plan](#implementation-plan)
8. [Migration Path](#migration-path)
9. [Compatibility](#compatibility)
10. [References](#references)
11. [Review Process](#review-process)
12. [Acceptance Criteria](#acceptance-criteria)

---

## Summary

ADR-023 / FEP-0028 gave **team** runs durable pause/resume on a member's policy `ASK`: the run
persists a checkpoint, returns an `awaiting_approval` aggregate, and resumes on the same `thread_id`
once a human decides. **Non-team single-agent chat has no equivalent** — a policy `ASK` blocks
synchronously on the terminal modal (`resolve_policy_ask` → `HITLController.process_approval`), so a
headless, API, or otherwise human-absent turn either hangs or fails the tool.

This FEP proposes **durable chat continuation** for single-agent runs: when durable pause is armed,
a policy `ASK` raises an `ApprovalPause` control-flow signal (generalizing ADR-023's
`MemberApprovalPause`) that is caught at the `AgenticLoop` turn boundary, persists a **lean
paused-run record** (the pending assistant tool-call + `ApprovalRequest`, referencing the
already-durable conversation `session_id`), emits a first-class `AWAITING_APPROVAL` event, and
returns a `TaskResult` with `status="awaiting_approval"` + a `run_id`. A new public
`VictorClient.resume(run_id, decision)` rehydrates the conversation (reusing `resume_session`),
applies the human's decision to the gated tool call, executes or skips it, and continues the loop.
The change is **additive and opt-in** — with durable pause disarmed, the inline modal path is
byte-identical. It reuses the FEP-0028 pause primitives rather than inventing a parallel mechanism.

## Motivation

### Problem Statement

A single-agent turn that hits a policy `ASK`-gated tool blocks **inline** on the approval modal:

- `victor/framework/policies/middleware.py:169-175` — `before_tool_call` calls `_resolve_ask()` when
  the verdict `is_ask`.
- `resolve_policy_ask()` builds an `HITLController` and `await controller.process_approval(request.id)`
  — a **synchronous, in-process block** on the registered handler.
- `victor/agent/tool_pipeline.py:3012-3041` — if the handler rejects, the middleware returns
  `proceed=False` and the tool is skipped as `middleware_blocked`.

This is correct for an interactive TTY where a human is present. It is **wrong** for:

- **Headless / API runs** (`victor serve`, `victor/integrations/api/routes/chat_routes.py`): there is
  no terminal modal, so the ASK either falls back to deny (losing the action) or blocks a request
  thread indefinitely.
- **Deferred approval**: a human who is not at the keyboard *now* but will approve *later* — the turn
  cannot be parked and resumed; it must be re-run from scratch, re-invoking the LLM (which may produce
  a *different* tool call, breaking the "approve this specific action" contract).
- **Cross-process continuity**: a crash or client disconnect mid-ASK loses the in-flight turn entirely.

Teams already solved the analogous problem (FEP-0028). Single-agent — the *more common* path — did
not get it; FEP-0028 explicitly deferred "the non-team single-agent chat continuation (the big
architectural epic)".

### Goals

- **Durable pause on `ASK`** for single-agent runs: park the turn, persist it, and surface an
  `awaiting_approval` outcome + resume token instead of blocking or failing.
- **Faithful resume**: on approval, execute the **exact** gated tool call the model already produced
  (not a re-sampled one), then continue the loop; on rejection, skip it with a tool-error result and
  continue.
- **Reuse FEP-0028 primitives** (the `ApprovalPause` BaseException pattern, the checkpoint/decision
  plumbing) rather than a parallel mechanism.
- **Opt-in and byte-identical when off**: the inline modal path is unchanged unless durable pause is
  armed.
- **Surface uniformly** across CLI, TUI, and API via a first-class `AWAITING_APPROVAL` event + a
  `TaskResult` status.

### Non-Goals

- **Arbitrary mid-turn checkpointing.** Only the `ASK` approval boundary is a durable pause point in
  this FEP (not every tool call or token). General step-level replay is out of scope.
- **Changing team pause** (FEP-0028) — this generalizes the shared signal but leaves team semantics
  intact.
- **Distributed / multi-worker resume routing** (e.g. resuming on a different API worker than paused).
  The persistence makes this *possible*; the routing/coordination is a follow-up.
- **Approving a re-sampled plan.** We resume the *pending* action, not re-plan.

## Proposed Change

### High-Level Design

Five pieces, four of them small and reuse-first; the load-bearing one is the resume re-execution
semantics.

1. **Generalized pause signal.** Introduce `ApprovalPause(BaseException)` in
   `victor/framework/approval_pause.py` (moving the idea out of `member_approval_context.py`; it lives
   in the *framework* layer — not `victor/agent/` as first sketched — because the catch point is the
   framework turn boundary, so every import is a clean agent→framework or same-layer one).
   `MemberApprovalPause` becomes a subclass, so team behavior is unchanged. A single-agent run raises
   `ApprovalPause(request)` from the policy `ASK` handler when durable pause is armed — exactly as a
   member does — but there is no `SubAgent` wrapper to catch it, so it rides through the
   `except Exception` pipeline (it is a `BaseException`) up to the **`AgenticLoop` turn boundary**.

2. **Arming.** A `ContextVar` `current_durable_pause_enabled` (peer of
   `current_member_durable_pause_enabled`) is set for the duration of a turn when the session opts in
   via a new `SessionConfig.tool_approval.durable = True` (default `False`). The policy handler
   (`_build_...`/`PolicyApprovalHandler`) checks it and raises instead of blocking, identical to the
   member path. Disarmed ⇒ the existing inline modal runs, byte-identical.

3. **Catch + persist at the loop boundary.** `AgenticLoop.run` (and its streaming twin) wrap the turn
   in `except ApprovalPause as pause:` and hand off to a new `PauseService` (or `RecoveryService`
   extension) that writes a **lean paused-run record** and returns an `awaiting_approval` `LoopResult`.
   The record (see schema below) stores the **pending assistant message with its `tool_calls`**, the
   **index of the gated call**, the serialized `ApprovalRequest`, and the `session_id` — *not* the
   conversation (that is already durably persisted by `ConversationStore` in `project.db`).

4. **Surface.** A first-class `EventType.AWAITING_APPROVAL` is emitted on the stream before
   `STREAM_END`; `TaskResult` carries `status="awaiting_approval"`, `approval_request`, and `run_id`.
   `_StreamEvent` gains the same. CLI/TUI render a paused lane (reusing the ADR-020
   `MEMBER_AWAITING`/paused-lane rendering); the API returns `202`-style `awaiting_approval` JSON with
   the `run_id`.

5. **Resume API.** `VictorClient.resume(run_id, decision)` (the FEP-gated public surface):
   loads the paused-run record, **reuses `resume_session(session_id)`** to rehydrate conversation +
   orchestrator, then re-enters the loop at the tool boundary: it applies `decision` to the gated call
   (execute if approved, skip with a tool-error result if rejected), appends the tool results to the
   conversation, and continues `AgenticLoop` from there — the model sees the tool outcome and proceeds
   exactly as if the human had answered inline.

### Detailed Specification

#### The paused-run record

A lean record — the conversation itself already lives in `project.db` via `ConversationStore`, so the
paused-run row references it rather than duplicating it. Stored in **`project.db`** (a single-agent
conversation is project-scoped; the pause is a property of that conversation), following the
`schema.py` versioned-migration pattern (`Tables.PAUSED_RUN`, `Schema.PAUSED_RUN`,
`CURRENT_SCHEMA_VERSION` bump, idempotent `CREATE TABLE IF NOT EXISTS`):

```sql
CREATE TABLE IF NOT EXISTS paused_run (
    run_id            TEXT PRIMARY KEY,   -- opaque resume token
    session_id        TEXT NOT NULL,      -- FK to the durable conversation (ConversationStore)
    agent_id          TEXT,               -- profile/agent identity, for validation on resume
    paused_at         TEXT NOT NULL,
    resumed_at        TEXT,               -- NULL while pending
    status            TEXT NOT NULL,      -- 'awaiting_approval' | 'resumed' | 'expired' | 'cancelled'
    pending_message   TEXT NOT NULL,      -- JSON: assistant message incl. tool_calls
    gated_call_index  INTEGER NOT NULL,   -- which tool_call in pending_message is ASK-gated
    approval_request  TEXT NOT NULL,      -- JSON: ApprovalRequest.to_dict()
    metadata          TEXT,               -- JSON: turn index, model, decision on resume
    created_at        TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_paused_run_session ON paused_run(session_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_paused_run_pending ON paused_run(status) WHERE resumed_at IS NULL;
```

Access via a `PausedRunStore` repository (mirroring `ConversationStore`'s thread-local sqlite pattern),
not raw sqlite in the service.

> **Note — persistence & the missing DB checkpointer.** There is no DB-backed
> `CheckpointerProtocol` today (only `MemoryCheckpointer`). Rather than build one, this FEP persists
> the *single* pause point directly in `paused_run` and leans on the already-durable
> `ConversationStore` for the transcript. A general persistent checkpointer (which would also let
> **teams** resume across process restarts) is called out as a follow-up, not a prerequisite.

#### Resume re-execution (the load-bearing decision)

Resume executes the **persisted pending tool call**, it does **not** re-sample the turn. Re-running
the turn would re-invoke the LLM and could yield a *different* tool call, silently breaking the
"approve this action" contract. So resume:

1. Rehydrate conversation + orchestrator via `resume_session(session_id)`.
2. Append the persisted `pending_message` (assistant + tool_calls) if not already the tail.
3. For the gated call: if `decision.approved`, run it through `tool_pipeline` **with durable pause
   disarmed and the decision pre-seeded** (so the policy engine treats it as approved, no re-ASK); if
   rejected, synthesize a `tool_error` result (`"rejected by human"`).
4. Execute any *other* tool calls in the pending message normally.
5. Append tool results to the conversation and continue `AgenticLoop` from EVALUATE/DECIDE.

Idempotency: `run_id` is single-use; a second `resume` on a `resumed` record is a no-op error. If a
*second* `ASK` fires during the continued turn, it pauses again with a **new** `run_id` (chained
pauses).

#### API Changes (public — the FEP-gated surface)

```python
# victor/framework/client.py
class VictorClient:
    async def resume(
        self,
        run_id: str,
        decision: "ApprovalDecision",   # {approved: bool, response: str|None, responder: str|None}
    ) -> "TaskResult": ...

# TaskResult gains:
#   status: str                      # "ok" | "awaiting_approval"
#   run_id: Optional[str]            # present when status == "awaiting_approval"
#   approval_request: Optional[dict] # present when status == "awaiting_approval"

# victor/framework/events.py
class EventType(str, Enum):
    AWAITING_APPROVAL = "awaiting_approval"   # new, first-class (was CUSTOM for teams)
```

`ApprovalDecision` is a small public dataclass (mirrors the team `approval_decision` dict:
`{"approved": bool, ...}`), added to the HITL surface so callers construct it typed.

#### Configuration Changes

```python
# victor/framework/session_config.py — ToolApprovalConfig
@dataclass(frozen=True)
class ToolApprovalConfig:
    enabled: bool = ...
    ask_on_tools: ... = ...
    ask_fallback: str = "deny"
    durable: bool = False   # NEW: arm durable pause instead of inline modal on ASK
```

Surfaced as a CLI flag (`--durable-approval`) and defaulted **on** for the API server (headless), so
`victor serve` parks instead of blocking. Interactive TTY keeps `durable=False` (inline modal) unless
opted in.

## Benefits

### For Framework Users

- Headless/API runs no longer hang or silently deny on `ASK`; they return a resumable
  `awaiting_approval` result with a `run_id`.
- Deferred, faithful approval: approve the *exact* action later, from any surface, without re-sampling.
- Crash/disconnect mid-ASK is recoverable (the pause is persisted).

### For Vertical Developers

- The same `ASK` policy a vertical already declares now works in non-interactive contexts for free —
  no vertical code changes.

### For the Ecosystem

- Symmetry with teams (FEP-0028): one durable-pause model across single-agent and multi-agent, one
  `AWAITING_APPROVAL` event across surfaces.

## Drawbacks and Alternatives

- **Complexity at the loop boundary.** Catching `ApprovalPause` and re-entering mid-turn is the
  riskiest part. *Mitigation*: reuse the proven `BaseException`-rides-through pattern; phase the work
  (in-memory pause first, persistence second).
- **Two pause substrates** (team `WorkflowCheckpoint` vs single-agent `paused_run`). *Mitigation*:
  both surface the same `ApprovalRequest` + decision contract; a unified persistent checkpointer is a
  named follow-up that could subsume both.
- **Alternative A — persistent StateGraph checkpointer, reuse team path.** Cleaner in theory (one
  substrate) but requires building the missing DB checkpointer *and* modelling a single agent as a
  graph node. Rejected as the *first* step; kept as the convergence target.
- **Alternative B — re-run the whole turn on resume.** Simplest to implement but re-samples the LLM →
  approves a possibly-different action. Rejected: breaks the approval contract.
- **Alternative C — do nothing / keep inline modal.** Leaves headless/API broken. Rejected.

## Unresolved Questions

- **project.db vs global victor.db** for `paused_run`. This FEP recommends `project.db` (the pause is
  a property of a project conversation); teams persist runs globally. Should paused runs be global for
  cross-project listing?
- **Resume routing** for multi-worker API deployments (which worker resumes). Out of scope here;
  needs a follow-up if `victor serve` scales horizontally.
- **Timeout/expiry policy** for a pending `paused_run` (GC of never-resumed pauses).
- **Chained pauses** UX: a resumed turn that hits a *second* ASK returns a new `run_id` — is that
  surfaced clearly enough, or do we need a pause chain id?
- **Streaming resume**: does `resume` re-open a stream, or only return an aggregate `TaskResult`?

## Implementation Plan

Phased, each phase independently shippable and tested (mirroring how FEP-0028 landed):

- **Phase 1 — pause mechanism (in-memory). ✅ landed (#777).** `ApprovalPause` base +
  `MemberApprovalPause` subclass; `current_durable_pause_enabled` ContextVar; policy handler raises
  when armed (`resolve_policy_approval_handler` wraps only when `governance.durable`); the turn
  boundary (`execute_message`) arms + catches → an `awaiting_approval` `TaskResult`;
  `EventType.AWAITING_APPROVAL`; `TaskResult.status/run_id/approval_request`. In-memory
  `InMemoryPausedRunStore`. *No public `resume` yet* — validates the mechanism.
- **Phase 2 — durable persistence. ✅ landed.** `ProjectDbPausedRunStore` — a self-managing
  `paused_run` table in `project.db` (idempotent `CREATE TABLE IF NOT EXISTS`, thread-local
  connection, JSON columns), mirroring `ConversationStore` (no `schema.py` migration needed since the
  project DB self-creates its tables). Both backends implement `PausedRunStoreProtocol`; the process
  store defaults to the durable one (falling back to in-memory when no project DB is resolvable). A
  pause now survives a restart — a fresh store instance reads it back. (Full rehydration via
  `resume_session` is exercised in Phase 3.)
- **Phase 3a — public resume API + faithful replay. ✅ landed.** `VictorClient.resume(run_id,
  decision)` + `ApprovalDecision`. Loads the `PausedRun`, atomically claims it (`mark_resumed` —
  single-use), rehydrates via `resume_session`, then (runtime helper `resume_paused_run`) finds the
  gated `tool_call` in the paused assistant message (the one with no result yet), executes it under
  the decision **without re-sampling the model** — approve → `ToolService.execute_tool` (the raw
  executor, bypassing the ASK middleware since the human decided); reject → a tool-error result —
  appends the linked `role=tool` result, and drives `execute_turn` continuations (no spurious user
  message) to a final answer. Scope: **single gated tool** per paused turn.
- **Phase 3b (deferred) — surfaces + batches.** CLI `--durable-approval` + `victor chat --resume`;
  API `awaiting_approval` response + resume route; TUI paused lane; multi-tool batch partiality;
  streaming resume.
- **Phase 4 — hardening.** Reject/timeout/expiry, chained pauses, GC, docs, and a
  `victor chat --resume <run_id>` ergonomic.

### Testing Strategy

- Unit: armed `ASK` raises `ApprovalPause`; disarmed is byte-identical (inline modal); loop catches +
  emits `AWAITING_APPROVAL` + `run_id`; resume executes the persisted call (approved) / skips
  (rejected) and continues; the model is **not** re-sampled on resume; chained pause yields a new
  `run_id`. Persistence round-trips across a fresh process (new `PausedRunStore` + `resume_session`).
- Integration: `victor serve` chat that hits an `ASK` returns `awaiting_approval` + `run_id`, and a
  subsequent resume completes the turn.

### Rollout Plan

Opt-in via `ToolApprovalConfig.durable` (default `False`). Defaulted on only for the API/headless
surface. Interactive CLI/TUI keep the inline modal unless `--durable-approval`.

## Migration Path

Additive; no breaking changes. New table via idempotent migration (`CURRENT_SCHEMA_VERSION` bump).
Existing callers of `chat`/`stream` see the new `TaskResult.status` field default to `"ok"` — no
behavior change unless they arm durable pause. No deprecations.

## Compatibility

- **Backward**: inline-modal path unchanged when `durable=False`. `TaskResult` gains optional fields.
- **Version**: new `EventType.AWAITING_APPROVAL` is additive; consumers ignoring unknown events are
  unaffected.
- **Vertical**: no vertical changes; verticals import only `victor_contracts` and see no new required
  surface. `ApprovalDecision`/`resume` live in `victor/framework`, consistent with the client
  boundary.

## References

- ADR-023 (multi-agent team durability), FEP-0028 (team-node durability contract) — the reused
  pause/decision primitives.
- ADR-021 (terminal-native HITL) — the inline modal this FEP parks instead of blocking.
- ADR-020 (interactive TUI) — the paused-lane rendering reused for `AWAITING_APPROVAL`.
- Companion ADR: **ADR-028** (single-agent durable chat continuation) — to be added alongside.
- Anchors: `victor/framework/policies/middleware.py:169-175`, `victor/agent/tool_pipeline.py:3012-3041`,
  `victor/framework/agentic_loop.py` (turn boundary), `victor/framework/client.py:766-817`
  (`resume_session`), `victor/agent/member_approval_context.py` (the pattern generalized),
  `victor/core/schema.py` (migration pattern), `victor/agent/conversation/store.py` (durable transcript).

## Review Process

Status **Draft** — submitted for review. Open questions above are the decision points for reviewers
(esp. project.db vs global, and the resume re-execution semantics).

### Revision History

| Date | Version | Changes | Author |
|------|---------|---------|--------|
| 2026-08-01 | 0.1 | Initial draft — single-agent durable chat continuation | Vijaykumar Singh |
| 2026-08-01 | 0.2 | Phase 1 landed (#777): pause signal + arming + turn-boundary catch + AWAITING_APPROVAL + in-memory store. Records that `ApprovalPause` lives in `victor/framework/` (not `victor/agent/`) — the catch is at the framework turn boundary | Vijaykumar Singh |
| 2026-08-01 | 0.3 | Phase 2 landed: `ProjectDbPausedRunStore` — self-managing `project.db` `paused_run` table (mirrors ConversationStore; no schema.py migration); pauses survive a restart. `PausedRunStoreProtocol` with in-memory + project-db backends | Vijaykumar Singh |
| 2026-08-01 | 0.4 | Phase 3a landed: `VictorClient.resume(run_id, decision)` + `ApprovalDecision`; `resume_paused_run` faithfully replays the persisted gated call (approve → raw `execute_tool`, bypassing re-ASK; reject → tool-error) and drives `execute_turn` continuations without re-sampling or a spurious user message. Single gated tool per turn; batches/surfaces/streaming → Phase 3b | Vijaykumar Singh |

## Acceptance Criteria

### Must-Have

- Armed single-agent `ASK` durably pauses (does not block the modal or fail the tool), returns
  `status="awaiting_approval"` + `run_id` + `approval_request`, and is byte-identical when disarmed.
- `VictorClient.resume(run_id, decision)` executes the **persisted** gated call on approval (no
  re-sample) or skips it on rejection, then continues the turn to completion.
- Pause survives process restart (Phase 2): a fresh process resumes via the `paused_run` record +
  `resume_session`.
- `EventType.AWAITING_APPROVAL` emitted on the stream; CLI/TUI/API each surface the paused outcome.

### Should-Have

- Chained pauses (a second ASK during a resumed turn) yield a new `run_id`.
- CLI `--durable-approval` + `victor chat --resume <run_id>`; API resume route.

### Implementation Requirements

- Opt-in (`ToolApprovalConfig.durable`); reuse the `ApprovalPause`/decision primitives; `PausedRunStore`
  repository (no raw sqlite in services); migration via `CURRENT_SCHEMA_VERSION`.

### Validation

- Full unit + integration battery per phase; mypy strict; no vertical boundary violations; the
  headless `victor serve` ASK→resume path demonstrated end-to-end.
