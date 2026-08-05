---
fep: 0028
title: "Team-Node Durability Contract"
type: Standards Track
status: Accepted
created: 2026-07-30
modified: 2026-08-01
authors:
  - name: Vijaykumar Singh
    email: vijay@anvaiops.com
    github: vijaydsingh
reviewers:
  - vijaydsingh
discussion: https://github.com/anvai-labs/victor/discussions/0028
---

# FEP-0028: Team-Node Durability Contract

> **Acceptance record (2026-08-01).** Accepted by Vijaykumar Singh (`vijaydsingh`), the project's
> sole FEP Maintainer — the single-maintainer analogue of FEP-0001's "2+ maintainer approvals"
> criterion, recorded here and in `reviewers:`. The contract was ratified incrementally as it
> shipped: every increment landed on `develop` across PRs #733–#752 with its acceptance-criteria
> bullet added in the same PR, so **Accepted describes merged reality, not intent**. The deferred
> remainder is recorded as explicit Non-Goals and Follow-ups below — nothing unresolved remains
> open against the contract.

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
- **Durable pause for the iterative formations (CONSENSUS/REFLECTION).** Their durable unit is the
  round/iteration, which is atomic — members race within a round and outputs only mean anything as a
  complete round — and the generator/critic path runs through `_MemberContextAgent.execute(str) ->
  str` (`unified_coordinator.py`), which has no awaiting-approval channel to surface a pause on.
  Both formations keep `supports_durable_pause()` `False`; a member `ASK` there keeps the slice-2a
  inline terminal modal.
- **Durable pause/continuation for a non-team single agent** (plain `victor chat`). It needs a
  continuation-token system (a `paused_runs` table, a `VictorClient` resume API, `chat_service`
  paused-run detection, an `AWAITING_APPROVAL` event) — a separate proposal outside the team-node
  contract, tracked independently.

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

The identity/state shape above is the *base* contract, written verbatim by the sequential
formations; each formation maps it onto its own **durable unit**:

#### Per-formation durability granularity

| Formation | Durable unit | Snapshot / resume semantics |
|-----------|--------------|-----------------------------|
| SEQUENTIAL | Member | Checkpoint after each member; resume skips completed members and restores `shared_state` + the chaining cursor. |
| PIPELINE | Member (stage) | Same sequential machinery; stage output→input chaining restored on resume; stop-on-failure still checkpoints the failed stage. |
| PARALLEL | Member (concurrent) | Each finishing member records + checkpoints the *cumulative* completed set under an `asyncio.Lock` (execution stays concurrent — only the completion handler serializes); resume seeds the set and re-runs only the rest, concurrently. |
| HIERARCHICAL | Phase (`plan` → `specialists` → `synthesis`, snapshotted in `shared_state["__hier__"]`) **plus** per-specialist within the wave | A completed plan is restored, not re-run (its `delegated_tasks` drive phase 2); the specialist wave runs through the shared concurrent runner, so mid-wave cumulative checkpoints — which embed the phase-1 snapshot — give per-specialist partial resume under the restored plan (no replan); synthesis is restored last. |
| CONSENSUS | Round (`shared_state["__consensus__"]`) | One snapshot per completed round; resume rebuilds the next round's task from the persisted loop state and continues at the first unfinished round. Members race *within* a round, so the round is atomic — a mid-round crash re-runs that round (documented contract). |
| REFLECTION | Iteration (`shared_state["__reflection__"]`) | One snapshot per generator+critic cycle, plus a terminal snapshot; resume rebuilds the refined generator prompt and continues at the first unfinished iteration; a resume after completion returns the aggregate without re-running. |

#### Reserved `shared_state` keys (contract-internal)

The contract reserves the dunder-named `shared_state` keys **`__hier__`**, **`__consensus__`**,
**`__reflection__`**, **`__awaiting_approval__`**, and **`__awaiting_approvals__`** for the
formation/coordinator machinery above. They are contract-internal: consumers (team authors, members,
tools) must **not** write them and must not depend on their internal shape — the public surface is
the paused/resumed aggregate (`status`, `paused_member_id`/`paused_member_ids`,
`awaiting_approvals`, `approval_request`, `thread_id`). Unknown or missing keys degrade to a fresh
(or coarser) resume, never a crash.

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
- **Slice 2b — pause coverage (final).** Durable-pause arming is gated on
  `strategy.supports_durable_pause()`, `True` for **SEQUENTIAL / PIPELINE / PARALLEL /
  HIERARCHICAL**. The sequential formations pause on a single awaiting member via
  `__awaiting_approval__`; PARALLEL and the HIERARCHICAL specialist wave collect *every* awaiting
  member of a concurrent wave into the **batch multi-pause aggregate** (`__awaiting_approvals__` +
  one batch pause checkpoint via `_make_member_batch_pause_hook`; resume re-runs exactly the paused
  set); HIERARCHICAL additionally pauses on an awaiting supervisor *plan* or *synthesis* via the
  singular aggregate (the paused phase is not snapshotted, so resume re-executes exactly it).
  CONSENSUS/REFLECTION durable pause and the non-team `victor chat` continuation are explicit
  **Non-Goals** (see [Motivation](#non-goals) for the recorded rationales).

### 3. Per-member streaming contract (implemented — lanes across all six formations)

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
renders per-member lane markers. Lanes cover **all six formations**: SEQUENTIAL/PIPELINE and the
concurrent formations route members through shared `BaseFormationStrategy` helpers
(`_execute_member_with_events` / `_execute_members_concurrently`), and CONSENSUS/REFLECTION emit via
the same `member_event_hook` path (round members through the helper; generator/critic inline).
Per-member **tool** and **token** streaming (needs wiring `SubAgent.stream_execute`) remains a
follow-up.

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

- ~~Concurrent-formation resume (PARALLEL/HIERARCHICAL): partial completion of concurrent members needs
  a well-defined "completed" set.~~ **Resolved:** the completed set is simply the members whose
  coroutines have recorded a result; each records + checkpoints the *cumulative* set under an
  `asyncio.Lock` (member execution stays concurrent — only the completion handler serializes), and
  `_load_member_resume`'s latest checkpoint is the largest completed set. Implemented for PARALLEL
  and the HIERARCHICAL specialist wave.
- ~~HIERARCHICAL resume: the supervisor runs twice (plan + synthesize) and the plan drives the
  specialist wave, so member-granular completion doesn't map cleanly ("which phase resumes").~~
  **Resolved:** resume is **phase-granular** — supervisor *plan* → concurrent *specialists* →
  supervisor *synthesis* are each snapshotted into `shared_state["__hier__"]` (persisted by the
  existing member checkpoint hook, no coordinator change), so a crash restores up to the last
  completed phase (a completed plan is restored, not re-run, and its `delegated_tasks` drive phase 2)
  and only the unreached phase(s) run. The specialist wave runs through the shared concurrent runner,
  whose mid-wave cumulative checkpoints embed the phase-1 snapshot — a crash *mid* wave resumes under
  the restored plan and re-runs only the unfinished specialists (per-specialist partial resume, no
  replan). HIERARCHICAL durable *pause* covers all three phases (plan/synthesis via the singular
  aggregate, the wave via the multi-pause aggregate).
- ~~Concurrent durable *pause*: a concurrent wave can have several members awaiting approval at once,
  so the single `__awaiting_approval__` shape can't express it.~~ **Resolved for PARALLEL:** members
  that come back awaiting are collected after the wave (not recorded as completed) into a **multi-pause
  aggregate** (`__awaiting_approvals__` — a list of `{member_id, approval_request}`) + a single pause
  checkpoint (`_make_member_batch_pause_hook`: `completed_member_ids` excludes the awaiting set; carries
  an `awaiting_approvals` list); the coordinator surfaces `status="awaiting_approval"` +
  `paused_member_ids` + `awaiting_approvals`, and a resumed run re-runs exactly the paused set. Arming is
  now gated true for PARALLEL and HIERARCHICAL (whose specialist wave reuses the same runner/aggregate,
  and whose supervisor plan/synthesis pause via the singular `__awaiting_approval__`). The non-team
  `victor chat` continuation remains open.
- ~~Where durable checkpoints live long-term (a `project.db` `team_member_checkpoints` table vs the
  injected checkpointer).~~ **Resolved (at acceptance):** the injected `CheckpointerProtocol` **is**
  the contract — the formation/coordinator machinery is storage-agnostic by design, and every
  increment shipped against it. A `project.db`-backed checkpointer (increment 5) is a follow-up
  *implementation* of that protocol, not a contract change (see Follow-ups below).

All questions raised while the increments landed were resolved before acceptance; nothing
unresolved remains open against this contract.

### Follow-ups (post-acceptance, non-blocking)

- A durable `project.db`-backed `CheckpointerProtocol` implementation (increment 5) — including
  `MemberResult.to_dict()` metadata JSON-purity, deferred to that work.
- Iterative-formation (CONSENSUS/REFLECTION) mid-loop *partial* resume — today a mid-round/iteration
  crash re-runs that round/iteration, the documented atomic contract.
- Per-member **tool/token** streaming (wiring `SubAgent.stream_execute` into the member lanes).
- The non-team single-agent chat continuation (a Non-Goal here; tracked as a separate proposal).

## Implementation Plan

- **Increment 1 (this FEP's first code, SEQUENTIAL):** `TeamContext.checkpoint_hook` +
  `resume_completed`; `MemberResult.from_dict`; coordinator opt-in `checkpointer` + per-member save +
  resume-skip; `SequentialFormation` reads the hook/resume. Tests with `MemoryCheckpointer`.
- **Increment 2:** checkpoint/resume for PARALLEL / HIERARCHICAL / PIPELINE / CONSENSUS / REFLECTION —
  **complete** (PARALLEL concurrent completed-set; PIPELINE per-stage; HIERARCHICAL per-phase **plus**
  per-specialist partial resume within the wave; CONSENSUS per-round; REFLECTION per-iteration).
  Iterative mid-loop *partial* resume is a follow-up (rounds/iterations are atomic).
- **Increment 3:** member interrupt — **complete for the pause-capable formations**. Slice 2a
  (terminal-native member approval: member-tagging wrapper on a ContextVar + modal tag)
  **implemented**; slice 2b-infra (durable pause checkpoint + resume re-run at the teams layer:
  `MemberStatus.AWAITING_APPROVAL`, `TeamContext.pause_hook`, `_make_member_pause_hook`, paused
  aggregate) **implemented**; the real mid-member ASK trigger (`MemberApprovalPause` BaseException
  armed by `current_member_durable_pause_enabled`) **implemented**; batch multi-pause for concurrent
  waves (`__awaiting_approvals__` + `_make_member_batch_pause_hook`) **implemented** for PARALLEL and
  HIERARCHICAL (whose plan/synthesis pause via the singular aggregate). Iterative-formation pause and
  the no-graph chat continuation are Non-Goals.
- **Increment 4 (implemented):** `member_id` event contract + `MemberEventSink`/`ContextVar`
  teams→stream bridge + `member_event_hook` producer + `map_event`/wire + TUI `MEMBER_START/END`
  lanes, covering all six formations. Member tool/token streaming is a follow-up.
- **Increment 5:** durable `project.db` checkpointer — a follow-up implementation of
  `CheckpointerProtocol` (post-acceptance; no contract change).

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
- Concurrent durable checkpoint/resume for PARALLEL merged: a shared
  `BaseFormationStrategy._execute_members_concurrently` runs members via `asyncio.gather` and, as each
  finishes, records + checkpoints the cumulative completed set under an `asyncio.Lock` (execution stays
  concurrent — only the completion handler serializes); resume seeds the completed set and re-runs the
  rest concurrently. Reuses the coordinator checkpoint hook + `_load_member_resume` + the
  `_execute_member_with_events` helper unchanged. Verified: a PARALLEL wave checkpoints every member (the
  latest holds the full set), resume skips the completed member and re-runs the rest, two members
  execute overlapping in-flight (the lock guards only completion, not execution), and no checkpointer is
  byte-identical. Concurrent durable **pause** (multi-pause aggregate) + HIERARCHICAL checkpoint remain
  deferred.
- Concurrent durable **pause/resume** for PARALLEL merged: a concurrent wave can have several members
  await approval at once, so the shared `_execute_members_concurrently` collects every member that comes
  back `awaiting_approval` (rather than recording it completed) and, after the wave, publishes a
  **multi-pause aggregate** — `__awaiting_approvals__` (a list of `{member_id, approval_request}`) plus a
  single pause checkpoint via a new `_make_member_batch_pause_hook` (`completed_member_ids` excludes the
  awaiting set; state carries an `awaiting_approvals` list). The coordinator surfaces
  `status="awaiting_approval"` + `paused_member_ids` + `awaiting_approvals`; a resumed run skips the
  completed members and re-runs **exactly** the paused set. `PARALLEL.supports_durable_pause()` now
  returns `True`, so the #742 arming gate arms it (a member ASK durably pauses instead of aborting the
  tool). Verified: a wave with two awaiting + two completed members surfaces both pending approvals and
  one pause checkpoint, a single awaiting member still uses the plural aggregate, resume re-runs only the
  paused set, and no checkpointer is byte-identical (awaiting is an inert non-success result). HIERARCHICAL
  concurrent pause + the non-team `victor chat` continuation remain deferred.
- HIERARCHICAL phase-granular checkpoint/resume merged: the supervisor runs twice (plan + synthesize) and
  the plan's `delegated_tasks` drive the specialist wave, so resume is **phase-granular**, not
  member-granular. Each phase — supervisor *plan*, concurrent *specialists*, supervisor *synthesis* — is
  snapshotted into `shared_state["__hier__"]` as it completes; because the existing member checkpoint hook
  already persists `shared_state`, this is a **pure formation-layer change** (no coordinator/PARALLEL
  edit). On resume, a completed plan is **restored, not re-run** (its `delegated_tasks` drive phase 2),
  completed specialists are restored, and only the unreached phase(s) execute; the no-delegation fallback
  path ends at phase 2 (no synthesis). `execute()` was refactored into `_resolve_supervisor` +
  `_run_specialists` (a pure extraction; specialist lanes via `_execute_specialist` unchanged). Verified: a
  full run snapshots phases 1/2/3, resume after a full run re-runs nothing, resume after a plan-only
  snapshot re-runs the specialist wave then synthesizes, the fallback path snapshots phases 1/2 and
  resumes, and no checkpointer is byte-identical. A crash *mid* specialist wave replans (per-specialist
  partial resume) and HIERARCHICAL durable **pause** remain deferred; `supports_durable_pause()` stays
  `False`.
- CONSENSUS + REFLECTION round/iteration-granular checkpoint/resume merged — completing checkpoint/resume
  across the whole formation taxonomy. Both are sequential *iterative* loops (each round/iteration builds
  its input from the previous one's output), so resume is **round/iteration-granular**: after each
  round/iteration the loop state is snapshotted into `shared_state["__consensus__"]` /
  `["__reflection__"]` (persisted by the existing member checkpoint hook — a **pure formation-layer
  change**), so a crash resumes at the next unfinished round/iteration and completed ones are restored,
  not re-run. CONSENSUS persists `{round_done, all_results, next_task_content, done, final}` and rebuilds
  the next round's task from `next_task_content`; REFLECTION persists `{iter_done, task_content, result,
  feedback, done}` and rebuilds the refined generator input, with a terminal snapshot so a resume after
  completion returns the aggregate without re-running (the final result build was factored into
  `_final_result`, shared by resume). Verified: each snapshots per round/iteration, resume skips completed
  rounds/iterations and re-runs only the rest, REFLECTION terminal resume re-runs nothing, and no
  checkpointer is byte-identical. Both keep `supports_durable_pause()` `False` (iterative-formation
  durable pause + the non-team `victor chat` continuation remain the deferred items).
- HIERARCHICAL durable pause + per-specialist partial resume merged: the specialist wave now runs through
  the shared concurrent runner (`_execute_members_concurrently`, extended with **additive** keyword params
  whose defaults keep PARALLEL byte-identical — `tasks` for per-specialist delegated tasks, `indices` for
  the 1-based specialist lane indices (supervisor is 0), and `resume_override` so the formation passes the
  coordinator payload filtered down to specialist ids). The wave therefore inherits the full concurrent
  contract: lock-protected mid-wave cumulative checkpoints — which embed the phase-1 `__hier__` snapshot,
  written into live `shared_state` *before* the wave even on the restored-plan path, so a mid-wave crash
  resumes under the restored plan and re-runs only the unfinished specialists (**no replan**) — and the
  multi-pause aggregate: awaiting specialists surface via `__awaiting_approvals__` + one batch pause
  checkpoint, phase 2 is not snapshotted, and the resumed run re-runs exactly the paused set. The
  supervisor **plan** and **synthesis** pause via the singular `__awaiting_approval__` aggregate
  (mirroring SEQUENTIAL): the paused phase is not snapshotted, so resume re-executes exactly it.
  `HIERARCHICAL.supports_durable_pause()` now returns `True` — safe because **all three phases** handle an
  awaiting result (the #740 arming hazard). Verified: a wave with two awaiting + one completed specialist
  surfaces both approvals in one batch pause checkpoint (no synthesis, phase 2 unsaved) and resume re-runs
  exactly the paused pair then synthesizes; a plan pause re-runs the plan on resume; a synthesis pause
  restores phases 1–2 and re-runs only the synthesis; a mid-wave crash resume skips the completed
  specialist without replanning; PARALLEL checkpoint/pause suites pass unmodified; old phase-only
  checkpoints resume unchanged; and no checkpointer is byte-identical (an awaiting specialist is an inert
  non-success result). The non-team `victor chat` continuation remains deferred.
- **Closure (2026-08-01):** every criterion above is merged and green on `develop` (PRs #733–#752).
  Checkpoint/resume and per-member streaming lanes cover **all six formations** at their recorded
  granularity (member / member-concurrent / phase+per-specialist / round / iteration); durable pause
  covers **SEQUENTIAL / PIPELINE / PARALLEL / HIERARCHICAL** (`supports_durable_pause()` `True`).
  The deferred remainder — iterative-formation durable pause, iterative mid-loop partial resume, the
  non-team chat continuation, member tool/token streaming, and the `project.db` checkpointer — is
  recorded as Non-Goals/Follow-ups, not open contract questions. FEP status: **Accepted**
  (acceptance record at the top); ADR-023 **Accepted** (rev 1.13); TD-25 **Done**.
