---
fep: "0010"
title: "Shared protocol crate for edge + cloud (one contract surface)"
type: Standards Track
status: Draft
created: 2026-06-29
modified: 2026-06-29
authors:
  - name: Vijaykumar Singh
    email: vijay@anvaiops.com
    github: vjsingh1984
reviewers: []
discussion: https://github.com/anvai-labs/victor/discussions/0010
---

# FEP-0010: Shared protocol crate for edge + cloud

## Summary

Victor's Rust workspace has two independently-maintained contract surfaces that
ought to be one. `rust/crates/protocol` ("portable types") exists, but
`rust/crates/edge-runtime` redefines its own `agent`, `provider`, `config`, and
state types rather than consuming the shared protocol crate. Meanwhile the
Python orchestrator defines the same concepts (`Agent`, tool-call, message,
agent-loop) yet again. The result is three parallel definitions of the same
handful of core shapes that can silently drift.

This FEP proposes promoting `rust/crates/protocol` to the **single canonical
contract** for message, tool-call, and agent-loop step types, with both
`edge-runtime` and the Python bindings depending on it. The edge runtime stops
re-implementing types; the Python bindings re-export the same shapes so cloud
and edge behavior stay in lockstep. This is the "full-stack co-design" seam: one
contract, two runtimes.

> **Update (audit, 2026-06-29):** The Rust side of this is **already largely
> done.** `edge-runtime` already consumes `victor_protocol` for the core types
> (`Message`, `Role`, `ToolCall`, `ToolDefinition`, `Usage`,
> `CompletionResponse`) with **no local redefinition** — the original
> "edge-runtime redefines its own types" premise no longer holds. The protocol
> crate already exports those types plus `StreamChunk`. The **remaining** gap is
> narrower: (a) Python bindings do not yet expose these same shapes (no FFI
> shape parity between the Python orchestrator and the edge runtime), and
> (b) the protocol crate does not yet carry independent semver. Phases 1–3
> below are effectively complete; this FEP now scopes Phase 4 + versioning.

## Motivation

### Problem Statement

- **(Resolved on the Rust side.)** `edge-runtime` now `use`s
  `victor_protocol::{Message, Role, ToolCall, ToolDefinition, Usage,
  CompletionResponse}` and defines none of these locally.
- **(Remaining.)** The Python orchestrator still defines its own
  message/tool-call/stream-chunk shapes; there is no shared contract crossing
  the PyO3 boundary, so cloud (Python) and edge (Rust) agree on behavior only by
  convention, not by type.
- **(Remaining.)** `rust/crates/protocol` has no independent VERSION/semver, so
  consumers cannot pin the contract independently of `victor-ai`.

### Goals

1. One canonical Rust crate (`rust/crates/protocol`) owns message, tool-call,
   agent-loop step, and agent-config types.
2. `edge-runtime` consumes the shared crate instead of redefining types.
3. Python bindings expose the same contracts (`victor_native` types mirror the
   protocol crate), so the orchestrator and edge runtime agree on shape.
4. No behavioral change in this FEP — purely structural convergence.

### Non-Goals

- Unifying the Python `Agent` class hierarchy with the edge runtime's executor.
  This FEP is about **shared types/contracts**, not a shared execution engine.
- Merging `rust/crates/state` into `protocol` (state is runtime data, not
  contract — left as-is).
- Any change to the Python public `Agent` API.

## Proposed Change

### High-Level Design

```
                 ┌──────────────────────────────┐
                 │  rust/crates/protocol        │  ← canonical CONTRACTS
                 │  Message, ToolCall,          │     (frozen, versioned)
                 │  AgentStep, AgentConfig,     │
                 │  StreamChunk …               │
                 └──────────────┬───────────────┘
            ┌───────────────────┴───────────────────────┐
            ▼                                           ▼
  ┌─────────────────────┐                    ┌─────────────────────────┐
  │ rust/crates/        │                    │ rust/crates/            │
  │ edge-runtime        │                    │ python-bindings         │
  │ (consumes protocol; │                    │ (exposes protocol types │
  │  no own agent type) │                    │  via PyO3 → victor_native)│
  └─────────────────────┘                    └────────────┬────────────┘
                                                         ▼
                                              Python orchestrator
                                              (imports the same shapes)
```

### Detailed Specification

#### Phase 1 — Audit and extract

Inventory the overlapping types across `edge-runtime/src/{agent,provider,config}.rs`,
`rust/crates/protocol`, and the Python `Agent`/tool-call/message definitions.
Produce a table of: type, current locations, canonical home, migration action.

#### Phase 2 — Promote the protocol crate

Move the shared shapes into `rust/crates/protocol`:

```rust
// rust/crates/protocol/src/lib.rs (expanded)
pub mod message;      // Message { role, content, tool_calls, … }
pub mod tool_call;    // ToolCall { id, name, args }, ToolResult
pub mod agent_step;   // PERCEIVE/PLAN/ACT/EVALUATE/DECIDE step envelope
pub mod agent_config; // AgentConfig (provider/model/temperature hints)
pub mod stream;       // StreamChunk (mirrors Python StreamChunk)
```

All types `#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]` and
PyO3-compatible where they cross to Python.

#### Phase 3 — Edge runtime consumes it

`edge-runtime` drops its local `agent`/`provider`/`config` type definitions and
`use`s them from the protocol crate. Its executor logic stays; only the type
declarations move.

#### Phase 4 — Python bindings expose the same contracts

`python-bindings` wraps the protocol types with PyO3 so `victor_native.Message`,
`victor_native.ToolCall`, etc. exist and mirror `rust/crates/protocol`. The
Python orchestrator's existing dataclasses remain the public API; internal FFI
round-trips use the protocol types, eliminating the implicit shape assumptions
that today cause version-skew bugs.

### API Changes

- **New**: `rust/crates/protocol` gains message/tool-call/agent-step/config
  modules (additive; the crate already exists).
- **Modified**: `edge-runtime` types replaced by re-exports from `protocol`.
- **Python**: `victor_native` gains protocol mirrors (additive). No change to
  `victor.framework.Agent` or `victor.framework.events`.

### Dependencies

- No new external crates. `protocol` already depends on `serde`; PyO3 is added
  only where protocol types cross to Python (already a `python-bindings` dep).

## Benefits

### For the Framework

- Behavioral parity between cloud and edge becomes enforceable by the compiler:
  both depend on the same `protocol` types.
- Cuts the duplicated agent/provider/config code in `edge-runtime`.

### For the Ecosystem

- External Rust consumers (future standalone edge deployments) get a stable,
  documented contract crate to target.
- Reduces the version-skew class of bugs at the FFI boundary.

## Drawbacks and Alternatives

### Drawbacks

- One-time migration cost across `edge-runtime` and `python-bindings`.
  Mitigation: Phase 3/4 are mechanical `use` replacements behind the existing
  executor logic.
- Slightly widens the `protocol` crate's responsibility. Mitigation: only
  contract types move; runtime logic does not.

### Alternatives Considered

1. **Code-gen Python types from the Rust crate** (e.g., `pyo3` + schema emit).
   Pros: zero drift. Cons: build complexity, fights the existing hand-written
   Python dataclasses. Rejected for now; Phase 4 hand-mirroring is sufficient
   and reversible.
2. **Leave edge-runtime standalone.** Pros: zero work. Cons: drift risk remains
   (the status quo this FEP exists to fix).

## Unresolved Questions

- **Versioning**: should `protocol` carry its own semver independent of
  `victor-ai` (like `victor-contracts`)? Proposed answer: yes — it is a portable
  contract crate.
- **Streaming parity**: the Python `StreamingChatPipeline` is not yet integrated
  with `AgenticLoop` (per CLAUDE.md). Should the shared `StreamChunk` wait for
  that unification, or define the shape now? Proposed: define now; the loop
  unification consumes it later.

## Implementation Plan

> **Status (2026-06-29 audit):** Phases 1–3 are effectively **complete** —
> `edge-runtime` already consumes `victor_protocol` with no local redefinition,
> and the protocol crate already exports the core message/tool-call/stream
> types. Remaining work is Phase 4 (Python FFI shape parity) + protocol-crate
> versioning.

### Phase 1: Audit ✅ (done)

The type-overlap table is settled: `Role`, `Message`, `ToolCall`,
`CompletionResponse`, `Usage`, `ToolDefinition`, `StreamChunk` live in
`rust/crates/protocol` and are consumed by `edge-runtime` via `use
victor_protocol::{…}`.

### Phase 2: Promote protocol crate ✅ (done)

The shared shapes already live in `rust/crates/protocol/src/lib.rs` with serde
derives.

### Phase 3: Edge-runtime migration ✅ (done)

`edge-runtime` defines none of these types locally; it imports them from
`victor_protocol`. (Confirm via `grep -rn 'struct Message|struct ToolCall|enum
Role' rust/crates/edge-runtime/src/` — empty.)

### Phase 4: Python bindings (2–3 days) — REMAINING

- [ ] Wrap the protocol types with PyO3 so `victor_native.Message`,
      `victor_native.ToolCall`, `victor_native.StreamChunk` mirror
      `rust/crates/protocol`.
- [ ] Cross-runtime round-trip test: build a `Message` in Python, pass through
      the native layer, compare unchanged.
- [ ] Keep the Python public dataclasses as the public API; protocol types are
      the internal FFI agreement.

**Deliverable**: cloud↔edge shape parity at the FFI boundary.

### Phase 5: Protocol-crate versioning (0.5 day) — REMAINING

- [ ] Add a `VERSION` file + independent semver to `rust/crates/protocol`
      (mirroring `victor-contracts`).
- [ ] Document the crate as the canonical edge+cloud contract.

**Deliverable**: independently versionable contract crate.

### Testing Strategy

- Unit: protocol type serde round-trips.
- Integration: edge-runtime executor still passes its existing tests.
- Cross-runtime: a `Message`/`ToolCall` built in Python round-trips through the
  native layer and back unchanged.
- Backward compat: Python public API unchanged (assertion test).

## Migration Path

No breaking Python change. Rust consumers of `edge-runtime` types update imports
to `protocol`. The old re-exports can be kept for one release as deprecation
shims.

### Deprecation Timeline

- `v0.8.x`: protocol crate expanded; edge-runtime types re-export from it
  (deprecation notice in doc comments).
- `v0.9.x`: local edge-runtime type defs removed.

## Compatibility

- **Breaking change**: No (Python public API). Internal Rust import paths in
  `edge-runtime` change (consumers update `use` paths).
- **Minimum Python**: 3.10 (unchanged).

## References

- Transcript co-design audit (edge/cloud shared-contract gap).
- `rust/crates/protocol/`, `rust/crates/edge-runtime/`, `rust/crates/python-bindings/`.
- Related: FEP-0009 (SDK tool contract — also a "promote the contract" pattern).

## Review Process

- **Submitted by**: Vijaykumar Singh
- **Initial review period**: 14 days minimum.
- **PR**: TBD.

## Acceptance Criteria

### Must-Have

1. `edge-runtime` defines none of {message, tool-call, agent-config} locally —
   all `use`d from `protocol`. Verified by grep.
2. A cross-runtime round-trip test exists and passes.
3. No Python public-API change (guarded by existing boundary tests).

### Should-Have

1. `protocol` crate has independent semver + VERSION.
2. Docs page documenting the protocol crate as the canonical contract.

---

## Copyright

This FEP is licensed under the Apache License 2.0, same as the Victor project.
