---
fep: 0028
title: "Team-Node Durability Contract"
type: Standards Track
status: Draft
created: 2026-07-30
modified: 2026-07-31
authors:
  - name: Vijaykumar Singh
    email: singhvjd@gmail.com
    github: vijaydsingh
reviewers: []
discussion: https://github.com/anvai-labs/victor/discussions/0028
---

# FEP-0028: Team-Node Durability Contract

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
11. [Acceptance Criteria](#acceptance-criteria)

---

## Summary

`UnifiedTeamCoordinator` participates in a `StateGraph` directly as a node (teams are *formations*,
not a separate graph). Today that node is **opaque to durability**: member execution lives inside the
formation strategies, which return an aggregated `List[MemberResult]`, so a crash or pause mid-team
restarts the whole team, an approval gate defers to the web-only HITL path, and the TUI sees only
aggregate progress. This FEP defines the **team-node durability contract** — the public shape of how a
team node checkpoints, interrupts, and streams at *member* granularity — so the coordinator can
**propagate** the existing graph-layer primitives (`graph_checkpoint.py`, `InterruptHandler`,
`CompiledGraph.replay_from`) down to members rather than build new ones. The contract is **additive
and opt-in**: absent a checkpointer/interrupt/stream sink, team behavior is byte-identical. It affects
authors of multi-agent teams and the TUI surface (ADR-020/021). Implements ADR-023; tracked by TD-25.

## Motivation

### Problem Statement

- **No member-granular durability.** `victor/coordination/formations/*.py` run all members internally
  and return results; `UnifiedTeamCoordinator.__call__` writes a single `team_output`. A failure after
  member 3 of 5 re-runs members 1–3 (wasted work; non-idempotent side effects repeated).
- **Interrupt is web-only.** A member hitting an `ASK`/approval gate goes through
  `workflows/hitl_api.py` (a REST/SQLite store) with no bridge back to the graph `interrupt`, so a
  team run cannot *durably pause and resume* — and cannot be approved terminal-natively (ADR-021).
- **Streaming is opaque.** `RenderAction` / `_StreamEvent` carry no member identity, and member events
  never reach `VictorClient.stream` at all — so the TUI (ADR-020) cannot show per-member lanes.

LangGraph — the category bar — makes checkpoint + time-travel + interrupt first-class. Victor's
formation model is arguably cleaner, but lacks the durability contract.

### Goals

- A **member checkpoint identity + state shape** the coordinator writes after each member, resumable
  at member granularity, reusing `WorkflowCheckpoint`/`CheckpointerProtocol`.
- **Member interrupt semantics**: a member `ASK` raises a graph `interrupt`, durably pausing the run;
  resume on approval (terminal-native via ADR-021).
- An **additive member identity** on the event contract enabling per-member TUI lanes.
- **Zero behavior change** when the durability features are not configured.

### Non-Goals

- Replacing the StateGraph engine or the formation model (teams stay formations-as-nodes; CLAUDE.md).
- A new checkpoint/interrupt engine — this FEP *propagates* the graph-layer primitives.
- Distributed/multi-process team execution.

## Proposed Change

### 1. Member checkpoint/resume contract

After each member completes, the team node persists a `WorkflowCheckpoint`:

- `checkpoint_id = "{thread_id}:{team_node_id}:member:{index}"`
- `node_id = "{team_node_id}:member:{member_id}"`
- `state = { completed_member_ids: [...], member_results: [MemberResult.to_dict()...],
  shared_state: {...}, last_output, last_agent_id, next_member_index }`
- `metadata = { team_node_id, member_id, member_index, formation }`

Resume loads the latest checkpoint for `(thread_id, team_node_id)`, **skips** completed members,
restores `shared_state` and the chaining cursor, and merges prior results into the aggregate.

Wiring is **opt-in**: the coordinator is given a `CheckpointerProtocol` (constructor arg or
`with_checkpointer(...)`) and a stable `thread_id` (from the node context/state). Formations opt in by
reading two new `None`-default fields on `TeamContext`: `checkpoint_hook` (awaited after each member)
and `resume_completed` (seeds a resumed run). With neither set, the formation loop is unchanged.

### 2. Member interrupt contract

Split into two slices:

- **Slice 2a — terminal-native member approval (implemented).** A team member (SubAgent) shares the
  session's DI container (the global container, set by `bootstrap_container`'s finalize phase, that the
  client registers the ASK approval handler into), so a member's policy `ASK`-gated tool already
  resolves the session terminal handler. `SubAgent._build_member_approval_handler` wraps that session
  handler to stamp `ApprovalRequest.context["member_id"]`/`["member_role"]`, and publishes it on the
  `current_member_approval_handler` `ContextVar` (`victor/agent/member_approval_context.py`) for the
  duration of the member orchestrator's synchronous construction; `_maybe_add_policy_engine` reads that
  var and prefers it over the container-resolved handler (no constructor param threaded through the
  decomposed orchestrator hotspot). The shared terminal modal (`approval_modal.py`) shows which member
  is asking. In-process pause (the member's tool call blocks on the modal); reject → tool blocked.
  Opt-in via governance (`USE_POLICY_ENGINE` + `governance.enabled`); no member in scope ⇒
  byte-identical.
- **Slice 2b-infra — durable pause/resume mechanism (implemented, teams-layer).** When a member reports
  awaiting-approval (`MemberResult.metadata["awaiting_approval"]` + `["approval_request"]`) and a
  checkpointer is configured, the SEQUENTIAL formation **stops** at that member (via a new opt-in
  `TeamContext.pause_hook`, not appending the paused member so it re-runs), and the coordinator persists
  a **pause checkpoint** (`_make_member_pause_hook`: `completed_member_ids` excludes the paused member;
  carries `paused_member_id` + `approval_request` + `awaiting_approval`) and returns a **paused
  aggregate** (`status="awaiting_approval"`, `paused_member_id`, `approval_request`, `thread_id`).
  **Resume** needs no new entry point: re-running `execute_task` on the same `thread_id` (with an
  `approval_decision` in context, surfaced into `shared_state`) drives the pillar-1 `_load_member_resume`
  path — completed members are skipped, the paused member **re-runs** (member-granular re-run semantics).
  Reuses `WorkflowCheckpoint`/`CheckpointerProtocol`; no new engine. Opt-in via a checkpointer; no
  checkpointer ⇒ the flag is inert (byte-identical).
- **Slice 2b — real ASK trigger (implemented).** A durable team run (a checkpointer + thread_id, i.e.
  `TeamContext.pause_hook` set) arms `current_member_durable_pause_enabled` (a ContextVar,
  `member_approval_context.py`) around member execution. A member's policy `ASK` then raises
  `MemberApprovalPause` — deliberately a **`BaseException` subclass** so it rides *through* every
  `except Exception:` on the policy-ASK → tool-pipeline → orchestrator → `AgenticLoop` →
  `SubAgent._execute_with_retry` path untouched (audited: no `except BaseException` there) — caught only
  at `SubAgent.execute`, which returns an awaiting result; `execute_task` returns the awaiting **dict**
  that drives the shipped 2b-infra pause/checkpoint/resume. Without arming (no checkpointer, or non-team
  single-agent), a member ASK keeps slice-2a inline-modal approval (byte-identical). Nested
  sub-agents-within-a-member inherit the armed var → a nested ASK surfaces as the member's pause
  (member re-runs on resume) — a documented limitation.
- **Slice 2b — deferred pieces.** The **no-graph chat path** for a *non-team single agent*: a plain
  `victor chat` ASK still blocks inline (2a); durable pause there needs a continuation-token system
  (`paused_runs` table, a `VictorClient` resume API, `chat_service` paused-run detection, an
  `AWAITING_APPROVAL` event). Concurrent-formation arming (PARALLEL/…) also deferred. Not implemented.

### 3. Per-member streaming contract (implemented — increment 4, SEQUENTIAL)

An optional `member_id: Optional[str]` is added to `AgentExecutionEvent` (`events.py`), `_StreamEvent`
(`client.py`), `RenderAction` (`event_mapping.py`), and the v1 wire event (`wire_events.py`) —
additive, defaulting `None` (single-agent unchanged). Member lifecycle events (`member_start`,
`member_completed`, `member_error`) ride on `EventType.CUSTOM` with `metadata["custom_type"]` and are
fanned into the client stream by a **teams→stream bridge**: a per-turn `MemberEventSink`
(`victor/framework/member_event_sink.py`) is published on a `current_member_sink` `ContextVar` by the
stream funnel (`_internal.stream_with_events`); a running team reads it (via
`TeamContext.member_event_hook`, set by `UnifiedTeamCoordinator._execute_formation`) and emits, and the
funnel interleaves those events with the orchestrator's own chunks (`_merge_orchestrator_and_members`,
an `asyncio.wait(FIRST_COMPLETED)` race whose termination is orchestrator-driven; the sink is bounded
and lossy so a team emit never blocks). `map_event`/`map_wire_event` translate the member CUSTOM events
to new `RenderKind.MEMBER_START/MEMBER_END` actions; the TUI (`WireTimelineState`/`ConversationLog`)
renders per-member lane markers. Scope of this increment is **SEQUENTIAL** lifecycle events; per-member
**tool** and **token** streaming (needs wiring `SubAgent.stream_execute`) and concurrent formations are
deferred.

## Benefits

- Crash-safe, resumable team runs (no repeated member work / side effects).
- Durable, terminal-native pause-for-approval on team members (ADR-021 parity).
- Per-member visibility in the TUI (ADR-020 per-member lanes).
- Reuses existing, tested graph primitives — small, additive surface.

## Drawbacks and Alternatives

- **Checkpoint volume** grows (per-member snapshots). Mitigated by opt-in + a bounded/pluggable
  checkpointer (`MemoryCheckpointer` default; `RLCheckpointerAdapter`; a future `project.db` table).
- **Alternative: checkpoint only at formation boundaries** — rejected; loses in-flight member work,
  the whole point.
- **Alternative: a dedicated multi-agent graph abstraction** — rejected (CLAUDE.md: teams are
  formations used directly as nodes). This FEP threads existing primitives through the node.

## Unresolved Questions

- Concurrent-formation resume (PARALLEL/HIERARCHICAL): partial completion of concurrent members needs
  a well-defined "completed" set — SEQUENTIAL first, others follow.
- Where durable checkpoints live long-term (a `project.db` `team_member_checkpoints` table vs the
  injected checkpointer). Increment 1 uses the injected checkpointer only.

## Implementation Plan

- **Increment 1 (this FEP's first code, SEQUENTIAL):** `TeamContext.checkpoint_hook` +
  `resume_completed`; `MemberResult.from_dict`; coordinator opt-in `checkpointer` + per-member save +
  resume-skip; `SequentialFormation` reads the hook/resume. Tests with `MemoryCheckpointer`.
- **Increment 2:** checkpoint/resume for PARALLEL / HIERARCHICAL / PIPELINE / CONSENSUS / REFLECTION.
- **Increment 3:** member interrupt. Slice 2a (terminal-native member approval: member-tagging wrapper
  on a ContextVar + modal tag) **implemented**; slice 2b-infra (durable pause checkpoint + resume re-run
  at the teams layer: `MemberStatus.AWAITING_APPROVAL`, `TeamContext.pause_hook`,
  `_make_member_pause_hook`, paused aggregate) **implemented**; the real mid-member ASK trigger
  (`MemberApprovalPause` BaseException armed by `current_member_durable_pause_enabled`) **implemented**;
  the no-graph chat continuation for a non-team single agent deferred.
- **Increment 4 (implemented, SEQUENTIAL):** `member_id` event contract + `MemberEventSink`/`ContextVar`
  teams→stream bridge + `member_event_hook` producer + `map_event`/wire + TUI `MEMBER_START/END` lanes.
  Member tool/token streaming and concurrent formations deferred.
- **Increment 5:** durable `project.db` checkpointer.

## Migration Path

Fully backward compatible. Existing teams (no checkpointer/interrupt/stream sink) are unaffected. Team
authors opt in by passing a `CheckpointerProtocol` and a `thread_id`. When the `member_id` event field
lands, existing single-agent consumers ignore the `None` default.

## Compatibility

- **Public API:** additive only — new optional `TeamContext` fields, a coordinator `checkpointer`
  arg / `with_checkpointer`, `MemberResult.from_dict`, and (later) an optional `member_id` event
  field. No signatures change incompatibly.
- **Wire contract:** the `member_id` addition is an optional v1 field (forward/backward compatible).

## References

- ADR-023 (`docs/architecture/adr/023-multi-agent-team-durability.md`); ADR-003 (workflow engine);
  ADR-020/021 (TUI + terminal HITL); TD-25.
- `victor/framework/graph_checkpoint.py` (`WorkflowCheckpoint`, `CheckpointerProtocol`,
  `MemoryCheckpointer`), `graph_execution.py`/`graph_runtime.py` (interrupt + per-node checkpoint),
  `graph.py` (`replay_from`).
- `victor/teams/unified_coordinator.py`, `victor/coordination/formations/`, `victor/teams/types.py`.

## Acceptance Criteria

- Increment 1 merged: opt-in per-member checkpoint save + resume for SEQUENTIAL, `MemoryCheckpointer`
  tests green (save writes N checkpoints; resume skips completed members; opt-out is identical), team
  test gate green, no behavior change without a checkpointer.
- The contract (checkpoint identity/state, interrupt semantics, `member_id` field) is ratified here
  before the later increments implement pillars 2–3.
- Increment 4 merged: opt-in per-member streaming for SEQUENTIAL — a `MemberEventSink`/`ContextVar`
  teams→stream bridge in `stream_with_events` interleaves `member_start`/`member_completed`/
  `member_error` events (tagged `member_id`) with orchestrator chunks; `member_id` is additive on
  `AgentExecutionEvent`/`_StreamEvent`/`RenderAction`/wire (all default `None`); `map_event`/wire
  translate to `RenderKind.MEMBER_START/END` lanes. Verified green: the seam interleaves and terminates
  cleanly, a floods-past-capacity emit does not deadlock (bounded/lossy sink), and with no team the
  emitted event sequence is byte-identical (streaming-parity gate). Member tool/token streaming and
  concurrent formations remain deferred.
- Increment 3 slice 2a merged: opt-in terminal-native member approval — a member's policy `ASK`-gated
  tool resolves the session approval handler (shared global container), wrapped to tag
  `ApprovalRequest.context["member_id"]` and published on a `ContextVar` during member-orchestrator
  construction so `_maybe_add_policy_engine` prefers it; the modal shows the member tag. Verified green:
  the wrapper tags + delegates (approve/reject propagate), returns `None` with no handler registered
  (deny fallback unchanged), the member context handler beats container/console resolution, no member in
  scope is byte-identical. Durable pause/resume (2b) deferred.
- Increment 3 slice 2b-infra merged: opt-in durable member pause/resume — a member reporting
  `metadata["awaiting_approval"]` stops the SEQUENTIAL formation (opt-in `pause_hook`), the coordinator
  persists a pause checkpoint (paused member excluded from `completed_member_ids`; carries
  `approval_request`) and returns a paused aggregate; re-running on the same `thread_id` skips completed
  members and re-runs the paused one (pillar-1 resume path). Verified green: pauses + checkpoints at the
  awaiting member (m2 never runs), resume re-runs only the paused member and continues with the decision
  surfaced in `shared_state`, and no checkpointer is byte-identical. The real ASK trigger + chat
  continuation deferred.
- TUI awaiting-approval lane merged: the `member_awaiting_approval` event now renders as a distinct
  paused lane — a new `RenderKind.MEMBER_AWAITING` mapped by `map_event`/`map_wire_event` and drawn by
  `WireTimelineState` (`⏸ <member> awaiting approval <tool>`); the formation emits the tool/title from
  the pending `approval_request` as the lane detail. Verified via mapping + wire-parity + render tests.
- Increment 3 real ASK trigger merged: a durable team run arms `current_member_durable_pause_enabled`
  around member execution; a member's policy `ASK` raises `MemberApprovalPause` (a `BaseException`, so it
  survives the runtime-core `except Exception` path — audited), caught at `SubAgent.execute` and returned
  by `execute_task` as the awaiting dict that the shipped 2b-infra pauses on. Verified green: armed only
  with a checkpointer + thread_id (reset after), the wrapper raises instead of blocking (delegates when
  disarmed — slice-2a unchanged), and `execute`/`execute_task` convert the pause to the awaiting result.
  Non-team single-agent + no-checkpointer keep inline approval. Chat-path (non-team) continuation deferred.
- Concurrent-formation streaming lanes + pause gating merged: PARALLEL and HIERARCHICAL route each
  member through a shared `BaseFormationStrategy._execute_member_with_events` helper (reusing the
  coordinator's single `member_event_hook`; the sink `ContextVar` propagates into the `gather` tasks), so
  per-member start/completed/error/awaiting lanes render for concurrent teams. Durable-pause arming is
  now gated on `strategy.supports_durable_pause()` (SEQUENTIAL only) — fixing a latent #740 bug where a
  concurrent team with a checkpointer would silently abort an ASK-gated tool. Verified: concurrent
  members emit tagged lanes (order-independent); no hook ⇒ unchanged; PARALLEL does not arm durable pause,
  SEQUENTIAL does. Concurrent durable **checkpoint/pause** (race-unsafe / undefined completed-set) and
  CONSENSUS/REFLECTION/PIPELINE remain deferred.
- PIPELINE full durability merged: PIPELINE is sequential (stages chain output→input, stop on failure),
  so it reuses the shared `_execute_member_with_events` helper + the coordinator hooks to get the full
  SEQUENTIAL contract — checkpoint after each stage, resume by skipping completed stages, durable pause
  on an awaiting stage (`supports_durable_pause() == True`), and per-stage streaming lanes. Verified:
  checkpoints each stage + resumes (skips completed), pauses at an awaiting stage + resumes by re-running
  it, stops-on-failure but still checkpoints the failed stage, no checkpointer is byte-identical, and it
  emits per-stage lanes. Concurrent durable checkpoint/pause + CONSENSUS/REFLECTION remain deferred.
- CONSENSUS + REFLECTION streaming lanes merged — completing per-member lanes across the whole
  formation taxonomy. CONSENSUS reuses the shared `_execute_member_with_events` helper (its round
  members use `agent.execute(task, ctx)`; the round is the lane index); REFLECTION emits inline via the
  same `member_event_hook` around its generator/critic (their `execute(str, shared_state)` signature +
  single aggregate result don't fit the helper — same emit path, not a fork). Verified: CONSENSUS emits
  per-member start/completed, REFLECTION emits generator + critic lanes, no hook ⇒ unchanged. Streaming
  lanes now cover SEQUENTIAL / PIPELINE / PARALLEL / HIERARCHICAL / CONSENSUS / REFLECTION. Concurrent
  durable checkpoint/pause remains the last deferred pillar item.
