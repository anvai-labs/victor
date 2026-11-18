---
fep: "0050"
title: "Durable Turn Records and Single-Turn Evaluation"
type: Standards Track
status: Draft
created: 2026-11-18
modified: 2026-11-18
authors:
  - name: Vijaykumar Singh
    email: vijay@anvaiops.com
    github: vjsingh1984
reviewers: []
discussion: https://github.com/anvai-labs/victor/discussions/0050
---

# FEP-0050: Durable Turn Records and Single-Turn Evaluation

## Summary

StateGraph execution needs one durable representation for a completed ACT turn and one owner for completed-turn evaluation. This FEP defines `victor.turn.record.v1`, a JSON-safe projection of `TurnResult`, and requires buffered, token-streamed, and StateGraph execution to converge on the same completed-turn evaluation pipeline. It also corrects FEP-0047's storage placement: workflow checkpoints default to the canonical project `.victor/project.db`, not another physical database. A path override is an explicit deployment exception for measured contention, cache lifetime, or portability.

The intended boundary is deliberately conservative. The agentic loop checkpoints completed ACT and interrupt/approval boundaries; PERCEIVE, PLAN, and PROMPT are not durable boundaries. EVALUATE is reconstructed from the latest completed ACT record. This keeps durability meaningful without turning every lightweight graph transition into a synchronous write amplifier. It also makes resume evidence comparable across execution modes and keeps the durable backup boundary aligned with the project rather than an unnoticed third database. The change finally gives runtime collaborators a clear injection boundary, so serialization policy no longer depends on discovering futures, streams, or service graphs hidden inside mutable state.

## Motivation

The legacy loop and StateGraph had drifted into separate evaluation paths. That made completion policy, disposition filtering, deadline handling, spin/ratchet evidence, and progress accounting difficult to compare. It also left ACT carrying a live `TurnResult`, which can contain provider-specific objects and run-local state that are unsuitable for serialization.

FEP-0047 then introduced durable checkpoints, but left database placement unresolved. The implementation selected `~/.victor/checkpoints.db` by default. That conflicts with Victor's canonical two logical database scopes: `~/.victor/victor.db` for global user state and `<project>/.victor/project.db` for project state. A third physical SQLite file must be an intentional exception, not an unnoticed side effect of constructing an agent.

### Goals

1. Preserve the canonical global/project database model.
2. Store workflow checkpoints in a versioned `workflow_checkpoint` table in `project.db` by default.
3. Keep `workflow_checkpoint_db_path` as an explicit contention or portability override.
4. Make completed ACT and interrupt/approval boundaries durable for the agentic loop.
5. Keep durable state strict JSON and free of runtime handles and raw provider material.
6. Make one pipeline own all evaluation after a completed turn.
7. Define StateGraph streaming as turn-boundary parity in this phase, without claiming token-level parity.

### Non-Goals

- Token-level ACT streaming inside StateGraph.
- Checkpointing every PERCEIVE, PLAN, PROMPT, or EVALUATE transition.
- Parallel team/workflow graph semantics; those graphs require their own durable observation contracts.
- General database consolidation for unrelated caches, undo logs, embeddings, or derived stores.

## Proposed Change

### State Representation Decision

The canonical agentic state is a frozen Pydantic value object, not a mutable dict, a dataclass used as a persistence object, or a DAO. This separation follows four ownership rules:

1. `AgenticLoopStateModel` owns typed durable workflow values. Nodes create transitions with `model_copy(update=...)`, and the model validates scalar boundaries while the checkpoint adapter enforces strict JSON at the persistence edge.
2. `AgenticLoopDependencies` and `RuntimeExecutionContext` own live collaborators such as runtime intelligence, planners, turn executors, evaluators, providers, and service accessors. They are resolved at node invocation and never stored in state.
3. Lifecycle callbacks own ephemeral observations. The ACT observer receives completed durable `TurnRecord` values and publishes them to a run-local channel; no channel or stream handle is placed in state.
4. `ProjectDatabaseGraphCheckpointer` is the persistence DAO. It maps a state value to the versioned `workflow_checkpoint` row and back, resolving only an already-loaded Pydantic model class. Persistence identity and SQL concerns stop at that boundary.

A plain dict remains the generic StateGraph compatibility format and the legacy migration input. A dataclass was rejected as the canonical representation because it provides fewer validation and JSON-mode guarantees. A DAO was rejected as live state because it would give workflow values persistence identity and hide service behavior inside state. Recursive runtime-handle-aware copying is therefore compatibility support for legacy dict graphs; canonical agentic execution uses shallow immutable value copies plus explicit collaborators.

This also avoids treating serialization as a reason to flatten the domain model. The durable projection is typed, versioned, and reconstructable, while non-JSON runtime behavior remains in collaborators that are rebuilt for the resumed run.

### Durable turn record

`TurnResult` remains the live runtime primitive. `TurnRecord` is its versioned durable projection:

```text
schema_version = "victor.turn.record.v1"
response = {
    content, role, tool_calls, stop_reason, usage, model, metadata
}
tool_results = [...]
follow_up_suggestions = [...]
has_tool_calls, tool_calls_count, all_tools_blocked, is_qa_response
disposition
tool_signatures = [...]
```

The projection supports only JSON-compatible primitives. Enums become values; sets become sorted lists; Pydantic values use JSON mode. Unknown non-JSON values are not silently embedded. Raw provider responses, credentials, refresh tokens, stream handles, live service graphs, execution contexts, and other runtime handles are excluded.

`TurnRecord.to_turn_result()` reconstructs the live evaluator input only at the EVALUATE boundary. A `TurnRecordChannel` may retain the latest completed record for one run or stream for timeout recovery. It is run-local, holds only the latest record, and is never serialized. Two concurrent runs must have separate channels; no module-global `ContextVar` is permitted.

### Single evaluation owner

`AgenticLoop.evaluate_completed_turn(...)` is the only completed-turn evaluation pipeline. It owns:

- spin and ratchet evidence;
- criteria-state mirroring;
- successful-tool evidence;
- evaluation invocation;
- progress-score accounting;
- turn-controller policy;
- disposition and backslide guards;
- run-owned deadline enforcement.

Buffered execution, token streaming, and StateGraph converge on this method. Callers must not apply a second disposition or backslide filter after the method returns. StateGraph uses a thin adapter to reconstruct `TurnResult` from `TurnRecord` and call the shared owner.

### Project checkpoint table

The default checkpointer writes to the canonical project database:

```sql
CREATE TABLE workflow_checkpoint (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    checkpoint_id TEXT NOT NULL UNIQUE,
    thread_id TEXT NOT NULL,
    schema_version INTEGER NOT NULL,
    completed_node TEXT,
    resume_target TEXT,
    state_type TEXT,
    durable_state TEXT NOT NULL,
    checkpoint_metadata TEXT NOT NULL,
    created_at REAL NOT NULL
);
```

Indexes cover `(thread_id, sequence)` and `(thread_id, created_at)`. `thread_id` is the durable run/thread identity. `schema_version` starts at 1. `resume_target` is a first-class column. Interrupt-before rows have no completed node and record the pending node in metadata with an explicit resume target. Typed state stores its already-loaded model identity only; loading never imports an arbitrary module named by stored data.

SQLite transactions are short, WAL-enabled, and serialized through a bounded worker so graph event loops do not execute blocking database work. Saving snapshots durable state. Loading and listing return copies. Retention keeps the newest `workflow_checkpoint_max_per_thread` rows per thread and optionally removes rows older than `workflow_checkpoint_retention_days`.

When `workflow_checkpoint_db_path` is set, the same table and semantics move to that physical file. The unset value means project `project.db`; it never means `~/.victor/checkpoints.db`.

### Selective graph boundaries

`CheckpointConfig.checkpoint_after_nodes` restricts ordinary completed-node checkpoints. `None` preserves generic StateGraph behavior: checkpoint after every completed node. An empty or nonempty set restricts ordinary writes to those nodes.

Interrupt-before and interrupt-after boundaries remain mandatory even when their node is excluded. Explicit start/end checkpoint controls, where used, retain their own semantics. The agentic executor configures `checkpoint_after_nodes={"act"}`. PROMPT, PERCEIVE, PLAN, and EVALUATE are not ordinary durable boundaries for this loop.

### Streaming semantics

This phase defines StateGraph streaming parity at completed-turn boundaries. A stream emits the durable result of a completed ACT turn; it does not imply that ACT model tokens are relayed one token at a time through the graph. Token-level StateGraph streaming requires a separate proposal and acceptance harness. Clients and release documentation must not describe this phase as token-level parity.

## Drawbacks and Alternatives

Writing to `project.db` places checkpoint traffic near conversations and code-graph writes. That is the correct default because durability and backup scope are more important than an unmeasured physical split. Deployments with measured contention may use the explicit path override. Selective ACT writes reduce the common-case cost before any split is justified. The same tradeoff also avoids serializing reconstructable perception and planning work while retaining an honest completed-action boundary for recovery, audit, and post-loop settlement.

A separate default checkpoint database was rejected because it created unnoticed state sprawl and complicated backup and project portability. Checkpointing every node was rejected because lightweight graph transitions are often reconstructable and their writes amplify latency without improving resume semantics.

## Migration Path

Fresh installs create the project table on first durable checkpoint. Existing `wf::` rows in an explicit `checkpoints.db` remain readable only when that path is explicitly configured; they are not silently imported. There is no automatic migration from a silent global default to project scope. Users who need old workflow history must retain and configure that path. No raw provider or credential material is expected to be migratable.

## Compatibility

The framework gains an additive compile option. Existing generic StateGraph checkpoint behavior remains unchanged when `checkpoint_after_nodes` is absent. Third-party checkpointers remain supported. The old `GraphSQLiteCheckpointer` remains an explicit adapter, but composition roots no longer select it by default. Durable initialization or write failures propagate rather than pretending that volatile memory satisfies durable-checkpoint policy.

## Acceptance Criteria

- [ ] Factory default resolves to `project.db`; no `checkpoints.db` is created implicitly.
- [ ] Explicit path override uses the same workflow table semantics.
- [ ] Table schema/version, newest-row load, ordered list, retention, and strict JSON rejection are tested.
- [ ] Typed state round-trips without importing arbitrary stored modules.
- [ ] Runtime handles and `_turn_record_channel` are absent from durable state.
- [ ] Agentic state is frozen and exposes no execution-context or channel fields.
- [ ] Service nodes receive execution context through graph dependencies.
- [ ] The ACT lifecycle observer is run-local and emits only durable turn records.
- [ ] Checkpoint binding copies immutable state instead of mutating its input.
- [ ] Recursive legacy-state copying is not used for canonical agentic transitions.
- [ ] Durable `TurnRecord` excludes raw provider response material.
- [ ] Two concurrent runs cannot observe one another's channel.
- [ ] EVALUATE reconstructs evaluator input after serialized record reload.
- [ ] Timeout after ACT preserves the latest completed record without dispatching another ACT.
- [ ] ACT is checkpointed; PROMPT/PERCEIVE/PLAN/EVALUATE are not.
- [ ] Interrupt-before and interrupt-after save regardless of `checkpoint_after_nodes`.
- [ ] Buffered, streamed, and StateGraph paths use the same completed-turn evaluation owner.
- [ ] Terminal policy errors propagate through node, runtime, executor, stream, and loop boundaries.
- [ ] Documentation claims turn-boundary streaming parity, not token-level parity.

## Benefits

- Buffered, token-streamed, and StateGraph execution share one completed-turn evaluation contract, reducing drift in terminal policy, disposition, progress, and recovery behavior.
- Durable state is typed, versioned, strict-JSON, and free of provider/runtime objects, making checkpoint failures explicit instead of discovery-by-pickle-error.
- Workflow checkpoints share the project backup and retention boundary by default; an override remains deliberate and observable.
- Selective ACT and interrupt writes reduce synchronous database work while preserving recovery at boundaries that cannot be reconstructed safely.
- Run-local observers support timeout recovery and streaming publication without module-global state or cross-run leakage.
- Explicit collaborators make resume deterministic: durable values are restored first, then services are rebuilt by the composition root for the resumed execution context.

## Unresolved Questions

- Whether future token-level StateGraph streaming should expose a formal event schema, or remain an executor adapter concern until another consumer requires graph-native token events.
- Which additional workflow families should adopt typed state models, and whether a shared state-model migration utility is warranted once more than the agentic loop migrates.
- What measured SQLite contention threshold justifies the explicit separate checkpoint path; the default should not move without that evidence.

## Implementation Plan

| Phase | Scope | Status |
|-------|-------|--------|
| 1 | Define `victor.turn.record.v1`, durable projection/reconstruction, and run-local channel | Implemented in this change |
| 2 | Consolidate completed-turn evaluation across buffered, streaming, and graph paths | Implemented in this change |
| 3 | Add project `workflow_checkpoint` schema, strict encoding, retention, and factory policy | Implemented in this change |
| 4 | Restrict agentic ordinary checkpoints to ACT and mandatory interrupt boundaries | Implemented in this change |
| 5 | Replace state-injected runtime handles with explicit collaborators and lifecycle observation | Implemented in this change |
| 6 | Token-level StateGraph streaming and additional typed workflow migrations | Separate proposal |


## References

- FEP-0032 — interrupt and resume semantics
- FEP-0047 — durable graph checkpointing and the corrected placement decision
- ADR-036 — evidence-gated flag graduation
- `victor/framework/agentic_graph/turn_record.py`
- `victor/storage/checkpoints/graph_adapter.py`
