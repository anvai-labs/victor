---
fep: "0044"
title: "Immutable run-policy snapshots across pause, resume and delegation"
type: Standards Track
status: Draft
created: 2026-11-13
modified: 2026-11-13
authors:
  - name: Vijaykumar Singh
    email: vijay@anvaiops.com
    github: vjsingh1984
reviewers: []
discussion: null
---

# FEP-0044: Immutable run-policy snapshots across pause, resume and delegation

## Summary

Admit one typed, immutable sandbox-policy snapshot for the lifetime of an agent
run. The snapshot records the effective policy, filesystem/network/namespace
restrictions and mount allow-list, plus a SHA-256 digest over those values. The
shared buffered and streaming message boundaries create it before model or tool
work, expose it on the per-run `RuntimeExecutionContext`, persist it in FEP-0029
paused-run metadata, and verify it before an approved action resumes. Delegated
or child execution receives the same object; it never rereads mutable global
settings to weaken or redefine an admitted run.

The design intentionally builds on existing owners instead of adding a parallel
control plane. The runtime execution context already travels through framework
message execution, the approval pause store already survives process restarts,
and resume already rechecks permissions and the exact action. This proposal adds
the missing admitted-policy identity at those same seams. In Phase 1 it is an
internal runtime contract: it adds no public HTTP field, SDK method,
caller-selectable policy or new authorization authority. Later transport and
delegation representations require separate accepted contracts.

## Motivation

Victor now fails closed when policy cannot be read, but several launch owners
still resolve policy at each effect. A long run can therefore observe different
policies after a settings reload. In particular, a paused run can wait for human
approval under `required`, then resume after the process reloads as `disabled`
or `preferred`; the approval was granted against a containment posture that no
longer necessarily applies. The reverse change is also unsafe for run semantics:
an immutable admission means the run must not silently tighten or otherwise
change mid-flight.

The existing pause record already binds the exact tool payload, scope, expiry
and ownership. It does not bind the containment policy under which that binding
was created. This proposal closes that evidence gap at the existing resume owner
rather than introducing a second approval store.

## Proposed Change

### Snapshot contract

`RunPolicySnapshot` is a frozen dataclass containing:

- schema version (`1`);
- effective policy (`disabled`, `preferred`, `required`);
- legacy `sandbox_enabled` flag;
- filesystem mode, namespace-restriction flag, network-isolation flag; and
- the immutable tuple form of the mount allow-list.

A canonical JSON encoding of those fields (sorted keys, no whitespace) is hashed
with SHA-256. Persistence includes the digest; restoration recomputes it and
uses a constant-time comparison. Missing or unexpected keys, unsupported types,
an unsupported schema or a changed value are verification failures. The only
valid successful `sandbox=None` interpretation remains the existing explicit
disabled policy.

The snapshot contains no credential, token, provider key or caller identity. It
is policy evidence, not authorization: every existing principal, resource and
approval check remains authoritative.

### Admission and propagation

`execute_message` and `stream_message_events` establish the snapshot before
resolving the chat runtime. If settings cannot be read or validated, the
buffered boundary returns a failed `TaskResult`; the streaming boundary yields a
non-recoverable error event. Neither reaches model or tool execution.

A successful run derives a context with `with_run_policy(snapshot)`, preserving
services and other runtime state. Existing legacy callers without a context are
not mutated; the pause still records the admitted snapshot. Phase 2 migrates
each launch owner to consume this attached snapshot so its behavior no longer
depends on when a global settings reload occurs. Until an owner is migrated, its
VAS-05q unreadable-policy refusal remains the floor.

### Pause and resume

New FEP-0029 pauses persist `metadata["run_policy"]`. Before resolving the
approval binding or dispatching an approved action, `resume_paused_run` restores
that snapshot, snapshots the current successfully loaded policy, and requires
exact equality. Any difference—loosening, tightening, changed mounts, malformed
metadata or unreadable current settings—raises `ResumeError` before an external
effect. The caller must start a new run and obtain a new approval if policy has
changed.

Pauses created before this field existed remain resumable. Legacy no-snapshot
records continue through the existing exact-action, current-policy and final
dispatch checks. Absence of the field is a compatibility fact, not proof that
the record was policy-bound.

### Delegation and child runs

A child or delegated execution inherits the parent snapshot by value. It may not
substitute global settings, client-supplied settings or a child profile for the
admitted parent policy. A delegated surface that cannot represent the snapshot
fails closed rather than spawning. Detailed background-agent and team-member
propagation is a separate implementation phase because their constructors and
recovery paths require their own real-boundary tests.

### Reload and revocation

This proposal defines run-level immutability, not operator-level emergency
revocation. A reload does not mutate an admitted run; a run carrying a snapshot
that no longer equals current policy must stop or, at a pause boundary, refuse
resume. A future emergency-stop mechanism may terminate owned work and record
the termination, but must not reinterpret the admitted policy as permission to
continue.

## Benefits

- Human approval and containment posture remain bound across disconnects and
  process restarts.
- Policy reload races become deterministic refusal instead of mixed behavior by
  launch site.
- The snapshot is small, type-checked, non-secret and independently verifiable.
- Existing launch owners can migrate incrementally without changing public APIs.
- Auditors can distinguish an old pause that predates policy evidence from a new
  pause whose admitted containment posture and approved payload are both bound.
- Tests can exercise one immutable value through the real message and resume
  boundaries instead of patching several launch sites and hoping their global
  reads stay synchronized.
- Operators get a predictable rule: a changed policy ends continuation of that
  admitted run and requires a new approval, rather than producing a partially
  changed execution depending on which branch happened to reread settings first.

## Drawbacks and Alternatives

- Persisting mount paths reveals operator configuration in the project database;
  they are already operator-controlled non-secret sandbox settings, and omitting
  them would let a reload change containment without changing the digest.
- Rejecting tightening changes can interrupt in-flight work after an operator
  hardens policy. Explicit termination is more truthful than silently changing
  the meaning of an admitted run.
- Rechecking current settings at resume still requires a successful read. This
  is intentional: a process that cannot establish current policy cannot prove
  the paused policy is still the admitted one.
- An alternative that rereads global settings at dispatch was rejected because it
  is exactly the reload race this proposal removes.
- Binding only the policy string was rejected because mounts and network/
  namespace restrictions are part of containment.

## Unresolved Questions

- Should long-running non-paused runs actively compare the snapshot on settings
  reload and terminate, or terminate only when the next governed launch observes
  the difference? Phase 2 must choose this per owned lifecycle boundary.
- What structured event/error names should each transport expose after FEP review?
- How should delegation represent a snapshot in serialized cross-process work
  units without reintroducing mutable settings objects?

## Implementation Plan

1. Add `victor.runtime.policy_snapshot` and the immutable per-run
   `RuntimeExecutionContext` derivative.
2. Establish the snapshot at both shared message-admission boundaries and
   persist it with new durable pauses.
3. Reject changed, malformed or unreadable snapshots in `resume_paused_run`
   before binding or dispatch.
4. Migrate canonical subprocess, workflow, terminal and git launch owners to the
   attached snapshot, one real-boundary PR per owner family.
5. Propagate to background agents, team members and durable delegation; add
   reload observations and explicit termination semantics.

Each launch-family migration must first write a RED test through its real route,
tool dispatch or workflow executor, including a zero-spawn negative when the
inherited snapshot cannot be enforced. The canonical subprocess owner comes
first because tools are the broadest effect surface; workflow, terminal and git
routes follow with their existing owner tests. Delegation is last because it
requires serialization and child-process identity decisions that should not be
guessed from in-memory test doubles alone. A phase is complete only when its
affected tests, type checks, collection check and changed-file selector pass at
the exact head submitted for review.

## References

- FEP-0029 defines the durable single-agent pause/resume record changed here.
- FEP-0039 owns the reviewed unified agent-service lifecycle and run contract.
- VAS-05i/05l/05q record the required-policy and unreadable-settings launch
  guarantees preserved during owner migration.

## Migration Path

No caller action is required. New paused records include the metadata field.
Old paused records remain valid and retain their current-policy checks. No
wire contract changes in Phase 1.

## Compatibility

Successful snapshots have no effect on ordinary runs in Phase 1. Launch owners
keep their VAS-05i/05q behavior until migrated. Legacy test doubles that do not
implement `with_run_policy` are not mutated. Existing public APIs, HTTP payloads
and generated SDKs are unchanged.

## Review Process

This draft is open for FEP review. It must not advance to Accepted until the
snapshot boundary, reload behavior and delegation representation have maintainer
review. Internal Phase-1 implementation may proceed only as a compatibility-safe
defect repair; later public-contract additions require FEP acceptance.

## Acceptance Criteria

- Snapshot creation copies policy values, ignores later mutation, round-trips
  metadata, and rejects malformed/tampered records.
- Both message admission boundaries establish policy before chat runtime use.
- New durable pauses retain the admitted snapshot.
- Resume refuses any snapshot difference before an approved external effect.
- Legacy pauses without a snapshot continue through existing checks.
- Each migrated launch owner proves zero-spawn refusal and inherited behavior
  through its real execution boundary.
- Delegation and background execution cannot substitute a different policy.
