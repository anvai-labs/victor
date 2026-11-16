---
fep: "0047"
title: "Durable-by-Default Graph Checkpointing"
type: Standards Track
status: Draft
created: 2026-11-16
modified: 2026-11-16
authors:
  - name: Vijaykumar Singh
    email: vijay@anvaiops.com
    github: vjsingh1984
reviewers: []
discussion: https://github.com/anvai-labs/victor/discussions/0047
---

# FEP-0047: Durable-by-Default Graph Checkpointing

## Summary

StateGraph execution in Victor is durable in design but volatile in practice: the
`CheckpointerProtocol` (`victor/framework/graph_checkpoint.py`) supports save/load/list of
`WorkflowCheckpoint` objects with `next_node` resume positions (FEP-0032), yet the only
checkpointer that ships for general use is `MemoryCheckpointer`, which dies with the process.
A crash, container restart, or server redeploy mid-workflow loses all node-position and state
history, and the durable SQLite machinery that already exists
(`victor/storage/checkpoints/backends/sqlite_backend.py`, with parent-id fork tracking, tagged
metadata, schema versioning, compression, and retention cleanup) satisfies a *different*
protocol that no production call site wires into a compiled graph. This FEP makes durable
checkpointing the default for production graph runs by introducing `GraphSQLiteCheckpointer`
— a `CheckpointerProtocol` adapter over the existing `SQLiteCheckpointBackend`, owned by the
storage layer — and injecting it at the composition roots (`AgenticLoopGraphExecutor` and the
AgentFactory workflow path) while leaving `StateGraph.compile()` signatures untouched.
Checkpointing stays opt-out-able per deployment (`workflow_checkpoints_enabled`,
`VICTOR_WORKFLOW_CHECKPOINTS_ENABLED`), disk use stays bounded by per-thread retention, and
installations without the `aiosqlite` extra degrade loudly to the in-memory checkpointer
rather than silently losing durability. Who is affected: every user of the agentic-loop graph
executor and YAML-compiled workflows; backward compatibility is preserved because the
framework API surface does not change and the opt-out restores today's behavior exactly.

## Motivation

### Problem Statement

The industry expectation for stateful agent runtimes moved in 2025–2026: LangGraph 1.x made
durable state the centerpiece of its 1.0 release, and production users now treat "an agent
run survives a process restart" as table stakes rather than a premium feature. Victor already
has every ingredient — `WorkflowCheckpoint` with `next_node()` resume semantics (FEP-0032,
partially landed), a fork-aware SQLite checkpoint store with diffs and versioning, retention
cleanup, and compression — but the two halves were built against different protocols
(`CheckpointerProtocol`'s thread-keyed `save/load/list` vs the storage layer's richer
id-keyed `CheckpointManagerProtocol`) and nothing joins them. The result is that a user who
kills a long workflow ten minutes in loses everything, despite the repository containing a
tested SQLite backend that could have saved the run.

Callers *can* pass a custom checkpointer to `StateGraph.compile(checkpointer=...)`, but only
`MemoryCheckpointer` and an RL-store adapter (`RLCheckpointerAdapter`) ship — and the RL store
is per-learner key-value storage with no parent-id forks and no cleanup, currently mid-relocation
under FEP-0033. Expecting every application developer to hand-write a durable adapter is
exactly the kind of framework tax that pushes users toward competitors whose durability is
the default.

### Goals

1. A production graph run's checkpoints persist across process restarts by default, with no
   API changes for callers who never think about checkpointing.
2. Disk use per thread is bounded and predictable (retention on every save, compression on).
3. Opt-out is a single documented setting; behavior with opt-out is byte-identical to today.
4. A deployment missing the `aiosqlite` optional dependency degrades loudly and observably,
   never silently.

## Proposed Change

Introduce `GraphSQLiteCheckpointer` in a new module
`victor/storage/checkpoints/graph_adapter.py`, implementing `CheckpointerProtocol`
(`async save(checkpoint)`, `async load(thread_id)`, `async list(thread_id)`) over the existing
`SQLiteCheckpointBackend`. The adapter is owned by the storage layer and imports
`WorkflowCheckpoint`/`CheckpointerProtocol` from `victor.framework.graph_checkpoint`; this
upward Protocol/dataclass import is dependency inversion explicitly sanctioned by
`docs/architecture.md` (guarded by the layering ratchet, with precedent at
`victor/framework/agentic_loop.py:197`).

Translation rules:

- **Keyspacing**: `thread_id` maps to a namespaced checkpoint session `wf::{thread_id}` so
  workflow checkpoints never collide with conversation checkpoints in the same database file.
- **State**: `WorkflowCheckpoint.to_dict()` becomes the backend's `state_data`; a
  `state_schema` key inside the payload plus the backend's existing `version` column carry
  format evolution forward.
- **Resume metadata**: `next_node` and checkpoint-boundary fields ride in
  `CheckpointMetadata` and round-trip unchanged — no schema extension needed (FEP-0032
  chose metadata placement precisely so every backend persists it unchanged).
- **Forks**: `parent_id` is populated from checkpoint metadata, feeding the existing
  checkpoint tree/time-travel tooling.

Default injection happens at the composition roots, *not* in `StateGraph.compile()`: the
framework layer must not reach into infrastructure or settings. The two production compile
sites — `AgenticLoopGraphExecutor.__init__` (`victor/framework/agentic_graph/executor.py`)
and the AgentFactory workflow construction path — resolve a checkpointer through a lazy
factory `get_default_graph_checkpointer()` exposed by the adapter module. The factory returns
`None` (framework default, in-memory behavior) when `workflow_checkpoints_enabled` is false,
and a `MemoryCheckpointer` with a one-time warning when `aiosqlite` is absent, also surfaced
in the server `/system` info payload.

New settings on `CheckpointSettings` (`victor/config/checkpoint_settings.py`):
`workflow_checkpoints_enabled: bool = True` and
`workflow_checkpoint_max_per_thread: int = 20`. Both are registered in the settings alias map
so `VICTOR_WORKFLOW_CHECKPOINTS_ENABLED` and
`VICTOR_WORKFLOW_CHECKPOINT_MAX_PER_THREAD` environment overrides work with the existing
`VICTOR_` prefix machinery. Retention runs after every save
(`cleanup_old_checkpoints(keep_count=...)`), keeping disk use O(1) per thread; compression
reuses the existing `checkpoint_compression_enabled`/`checkpoint_compression_threshold` knobs.

## Benefits

- **Crash-resumable workflows out of the box**: the most common production failure mode
  (process death mid-run) changes from total state loss to resume-at-node. Combined with
  FEP-0032's already-landed `interrupted`/`next_node` fields, a paused run survives restarts.
- **Zero framework API churn**: `StateGraph.compile()` keeps its signature; third-party code
  passing explicit checkpointers is untouched; the change is invisible until it is needed.
- **One storage contract**: building on `SQLiteCheckpointBackend` instead of a new table keeps
  one retention policy, one compression path, one fork/time-travel consumer
  (`checkpoint_tree.py`), and one migration story alongside `ConversationCheckpointManager`.
- **Bounded and observable cost**: retention defaults (~20 per thread) plus compression mean
  predictable disk behavior, and degradation is loud (warning + `/system` surface) rather
  than a silent durability lie.

## Drawbacks and Alternatives

- **Write amplification**: every node boundary now deep-copies serialized state to disk.
  Large states multiply; the retention-on-save policy and default compression bound this, but
  very large state graphs will pay measurable per-node latency. Callers with extreme state
  sizes can opt out and wire their own checkpointer.
- **Behavior change**: FEP-0032's "re-invoking a completed thread executes nothing" becomes
  cross-restart-visible for the first time — a user who restarts the server between a
  completed run and a re-invoke now observes no-op semantics that were previously erased by
  process death. This is correct but surprising; the migration section addresses it.
- **Alternative considered — new dedicated table**: rejected because it forks the storage
  contract (second retention policy, second compression path, second migration story) for no
  capability the existing table lacks.
- **Alternative considered — flip `compile()` default**: rejected because the framework layer
  must remain settings-agnostic and dependency-free; composition roots are the sanctioned
  injection point and keep `victor-contracts` consumers unaffected.
- **Alternative considered — build on `RLCheckpointerAdapter`'s store**: rejected; per-learner
  KV shape without parent-id forks or cleanup, and FEP-0033 is actively relocating that
  subsystem — building durability on it would be a liability.

## Unresolved Questions

- Should `thread_id` default derive from the session id, or fall back to
  `project:{cwd-hash}` for sessions that lack one? (Implementation will use the session id
  when present; the fallback needs a product decision.)
- Should the SQLite database file live at `~/.victor/checkpoints.db` (backend default) or
  move under the project `.victor/` scope for project-local workflows? The global location is
  proposed for v1 since workflow threads are session-keyed.
- Does the wire/API surface need a "list resumable threads" endpoint in v1, or is that
  deferred to the paused-result surfacing work (FOU I-D4)?

## Implementation Plan

Delivered as FOU Tier-D increments I-D1 and I-D2 (tracked in
`docs/architecture/victor-foundations-plan.md`), one worktree/PR each:

1. **I-D1**: `GraphSQLiteCheckpointer` adapter + lazy export + WAL pragma/busy-timeout for
   concurrent sessions, opt-in only. RED test:
   `tests/unit/storage/checkpoints/test_graph_adapter.py` (save → fresh adapter → load
   deep-equal with `next_node` intact; ordered `list()`; missing-aiosqlite degradation;
   unknown thread → `None`; two-writer interleaving).
2. **I-D2**: composition-root injection + settings + retention + opt-out. RED tests: executor
   compiles with a non-memory checkpointer when enabled and memory when disabled
   (`tests/unit/framework/test_agentic_loop_executor.py`); settings alias round-trip.
3. Smoke (real server, per the VAS smoke rule): run a chat turn driving the agentic loop,
   assert a `wf::` row exists, kill and restart, resume without re-executing the first node;
   flip the env opt-out and assert no rows are written.

Each increment records RED/GREEN evidence and commit hashes in the FOU ledger; FEP status
moves to Implemented when the I-D2 PR merges.

## Migration Path

- **New installs**: durable from first run; nothing to do.
- **Existing installs on upgrade**: checkpointing turns on at next session. Users who pinned
  behavior via settings keep it (`workflow_checkpoints_enabled=false` restores today exactly);
  no data migration is required because rows are additive in the existing checkpoints table.
- **Opt-outs**: environment (`VICTOR_WORKFLOW_CHECKPOINTS_ENABLED=false`), settings file, or
  per-call explicit `checkpointer=` argument (highest precedence, unchanged semantics).
- **Communication**: release notes flag the cross-restart no-op re-invoke semantics; the
  `/system` info endpoint exposes durability status (backend kind, db path, retention).

## Compatibility

No framework public API signature changes. `CheckpointerProtocol`,
`WorkflowCheckpoint`, and `StateGraph.compile()` are unchanged; the adapter satisfies the
existing protocol. Third-party checkpointers remain first-class. The storage schema gains
namespaced rows only (`wf::` session keys, `version=1`, `state_schema` payload key); existing
conversation checkpoints are untouched. Python 3.12+ / aiosqlite-optional status is preserved
(degradation path). No new required dependency is introduced.

## References

- FEP-0032: Graph Interrupt/Resume Semantics (paused/resume-at model this builds on)
- FEP-0033: RL subsystem relocation (why the RL store was rejected as the backing store)
- `victor/framework/graph_checkpoint.py` — `CheckpointerProtocol`, `MemoryCheckpointer`
- `victor/storage/checkpoints/` — `SQLiteCheckpointBackend`, `CheckpointData`, tree tooling
- `docs/architecture/victor-foundations-program.md` — program context and tier ordering
- LangGraph checkpoint persistence documentation (competitive baseline for default durability)

## Review Process

Per FEP-0001: this Draft enters 14-day review on merge; acceptance requires two maintainer
approvals and no blocking objections. Implementation evidence (RED/GREEN, smoke) is tracked in
the FOU ledger, not here.

## Acceptance Criteria

- [ ] I-D1 adapter merged with the RED→GREEN unit suite including concurrency and degradation
- [ ] I-D2 composition-root default merged with opt-out verified by negative test
- [ ] Real-server smoke recorded (resume across restart; opt-out writes nothing)
- [ ] Retention keeps per-thread row count ≤ `workflow_checkpoint_max_per_thread`
- [ ] `/system` reports durability status; missing-aiosqlite warning fires exactly once
