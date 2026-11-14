---
fep: "0045"
title: "Opt-In Send Fan-Out Result Retention Budget"
type: "Standards Track"
status: "Draft"
created: 2026-11-14
modified: 2026-11-14
authors:
  - name: Vijaykumar Singh
    email: vijay@anvaiops.com
    github: vjsingh1984
reviewers: []
discussion: null
---

# FEP-0045: Opt-In Send Fan-Out Result Retention Budget

## Summary

Add an opt-in byte budget for results retained by `CompiledGraph._execute_parallel`.
The default remains unlimited so every admitted `Send` branch continues to reach the
state merger and existing callers keep their exact merge, ordering, and failure
semantics. When a caller sets the budget, Victor measures the retained branch-result
representation after each branch returns and before storing it in the fan-out result
list. Exceeding the budget stops further admission, cancels and joins unstarted sibling
workers, preserves diagnostic outcomes already observed, and raises a typed error rather
than silently truncating a result or invoking a merger with incomplete data. The error
identifies the configured limit, cumulative bytes, branch, node, and attempted size, and
explicitly warns that earlier branch effects may already have occurred.

This FEP is a contract proposal only. It does not change current default behavior and
does not claim durable spill, release qualification, or a native implementation. Those
are separate owners and gates.

## Motivation

`_execute_parallel` currently allocates one result slot per admitted branch and retains
each `(node, success, error, state_dict)` tuple until every branch finishes. Worker and
input-state admission are bounded, but result retention is not. A fan-out with 1,024
unique 64 KiB payloads therefore retains roughly 64 MiB even when its final merger
discards those payloads.

A representative local tracing probe on WSL2/Py3.12, after warm-up and with
`max_parallel_branches=16`, used unique 64 KiB branch payloads and a merger that
discarded them:

| Branches | Logical payload | Traced peak | Retained after merge |
| ---: | ---: | ---: | ---: |
| 128 | 8.0 MiB | 8.1 MiB | 0.0 MiB |
| 256 | 16.0 MiB | 16.2 MiB | 0.0 MiB |
| 512 | 32.0 MiB | 32.3 MiB | 0.1 MiB |
| 1024 | 64.0 MiB | 64.6 MiB | 0.1 MiB |

Peak retention tracks total payload size. This is local design evidence, not a release
benchmark or an installed-wheel/live-service measurement.

## Proposed Change

### Configuration

Add `ExecutionConfig.max_fanout_result_bytes: Optional[int] = None`. `None` preserves
the current retain-all contract. Positive integer values are the only accepted setting;
zero, booleans, floats, NaN, infinity, and negative values are rejected at construction,
`from_legacy`, and `compile(**kwargs)` boundaries.

Do not add an overflow mode in this phase. Truncation and silent substitution are
merger-contract changes. The first phase fails explicitly.

### Measured representation

The budget counts logical bytes that must remain available to the state merger, not
transient Python allocator overhead:

- UTF-8 bytes for strings, object keys, enum names, and error text;
- byte length for `bytes`, `bytearray`, and memoryview payloads (without copying them);
- canonical JSON-compatible scalar and container contents;
- shared object contents once per fan-out by object identity, so aliases do not count
  repeatedly;
- bounded cycle handling, with a cycle treated as a contract violation rather than an
  unbounded traversal.

The estimator runs after a branch result exists and before it is stored in `results`.
Transient model/tool output, an input state already discarded by a branch, serialization
buffers, and final merged output are not part of this accumulator. If an opt-in result
contains a value the estimator cannot measure deterministically, execution raises a typed
`FanoutResultUnmeasurable` error instead of falling back to pickle or guessing.

### Overflow and cancellation

When adding a branch result would exceed the configured budget:

1. stop admitting more branches;
2. cancel and join bounded sibling workers, following the existing owned-worker failure
   lifecycle;
3. retain already observed branch indexes, node names, success/error status, and
   accounting diagnostics without retaining their full result payloads;
4. raise `FanoutResultBudgetExceeded` with the configured limit, cumulative measured
   bytes, branch index, node name, and attempted result size;
5. do not invoke the state merger with a partial or truncated list.

The overflowing branch may already have produced an external effect. The error must say
that explicitly; retry semantics belong to the caller's idempotency/receipt contract, not
to this scheduler primitive.

### Observability

Emit one structured scheduler event when the budget is exceeded and expose cumulative
counters for measured bytes, admitted branches, observed branches, and the limiting
branch. The exception and event use the same fields. No generic warning that omits the
limit or branch identity is sufficient.

## Benefits

- Operators can bound the retained fan-out result set without limiting useful branch
  concurrency or reducing the explicit fan-out count budget.
- Merger inputs remain complete whenever a bounded run succeeds; there is no silent
  oldest/newest substitution whose meaning depends on the caller's merger.
- The accounting contract distinguishes logical retained bytes from interpreter
  allocator overhead and counts shared objects once, making measurements reproducible.
- Typed failure plus structured telemetry gives callers enough context to apply their own
  retry, receipt, or durable-spill policy.
- The default framework contract remains unchanged until maintainers accept the FEP.
  That separation lets bounded services adopt the field deliberately while ordinary
  library users continue to receive every admitted branch result.

## Drawbacks and Alternatives

- **Truncate the largest or oldest result:** rejected for phase one because the state
  merger may interpret absence, ordering, or payload bytes semantically.
- **Wait until all branches complete, then enforce the budget:** retains the exact
  unbounded peak this proposal prevents.
- **Count Python `sys.getsizeof`:** measures allocator implementation details, misses
  nested and shared objects, and varies by interpreter; it is unsuitable for a public
  contract.
- **Pickle arbitrary objects for measurement:** would silently expand deserialization
  and compatibility risk; unsupported values must be explicit opt-in failures.
- **Durable spill to a result store:** a useful later extension, but it requires ownership,
  retention, replay, eviction, and receipt contracts. It must not be smuggled into the
  first bounded-memory slice.

An opt-in budget can fail after some branch effects have occurred. That is unavoidable
when output size is unknown before execution, and explicit failure is safer than silent
data loss. Callers requiring all-or-nothing behavior must preallocate their own durable
work/receipt contract. Measurement also adds a bounded traversal to every result while
the feature is enabled.

## Unresolved Questions

- Should the first accepted implementation expose only bytes, or also a diagnostic
  `result_bytes` accumulator even when no limit is configured?
- Which structured scheduler event envelope and error code names should align with the
  unified run-event contract?
- Is a durable `ResultRef` spill a public contract or a deployment-specific adapter, and
  which store owner defines eviction and replay?
- Should a later accepted mode permit caller-selected truncation with an explicit
  marker, or remain fail-only permanently?
- What representative workload sizes and platforms become release-gate evidence rather
  than local design measurements?

## Implementation Plan

1. Specify the estimator as a typed framework utility with shared-reference and cycle
   tests; retain a Python implementation even if a native path is later measured.
2. Add strict configuration validation through all construction paths.
3. Extend the real `_execute_parallel` worker boundary with RED tests for linear growth,
   first overflow, sibling cancellation, observed-result diagnostics, unmeasurable values,
   caller cancellation, and legacy unlimited behavior.
4. Add structured event/error parity tests and representative memory measurements at the
   exact submitted head.
5. Keep `max_parallel_branches` and `max_fanout_branches` independent: one bounds work,
   one bounds expansion, and this proposal bounds retained output only after execution.
6. Record exact-head representative memory measurements in the canonical tracker before
   review; do not reuse this local design probe as release evidence.

## Compatibility

Default callers see no change. Existing compiled graphs retain every result and retain
their current merge, failure, and ordering semantics. No wire field or SDK method is
added until this FEP is accepted. Opt-in users accept typed failure after partial branch
execution.

## Migration Path

No caller action is required. Existing graphs continue to retain all branch results.
Callers that opt in must handle `FanoutResultBudgetExceeded` and
`FanoutResultUnmeasurable`; migration should start with workflows whose merger can
recover data from durable source-of-truth artifacts. Durable spill or truncation modes
require a separate accepted proposal.

## Acceptance Criteria

- Default fan-out behavior remains byte-unlimited and retains every result.
- Opt-in byte validation rejects invalid numeric types and values at every config path.
- Measurement is deterministic, counts shared values once, handles JSON-compatible and
  byte payloads, and refuses unsupported values without pickle fallback.
- Overflow stops admission, cancels/joins siblings, never merges partial data, and exposes
  complete typed diagnostics.
- Representative memory evidence demonstrates bounded retained peak at the exact head.
- Local pass, merged source, release artifact, and live acceptance remain distinct claims.

## References

- VAS-16b bounds fan-out admission and intentionally leaves result-byte retention open.
- FEP-0039 owns the unified agent-service run contract.
- `docs/architecture/native-acceleration-strategy.md` governs any later measured native
  optimization.
