---
fep: "0015"
title: "Trim internal-only symbols from the framework public API (step_handlers exports)"
type: Standards Track
status: Accepted
created: 2026-07-09
modified: 2026-07-09
authors:
  - name: Vijaykumar Singh
    email: vijay@anvaiops.com
    github: vjsingh1984
reviewers: []
discussion: https://github.com/anvai-labs/victor/discussions/0015
---

# FEP-0015: Trim internal-only symbols from the framework public API

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

`victor/framework/step_handlers.py` exports two symbols in its `__all__` —
`CapabilityConfigStepHandler` and `ExtensionHandler` — that no code outside that module
imports. Because `victor/framework` `__all__` **is** the framework's public API, these are
advertised as supported surface while being pure internal implementation detail:
`CapabilityConfigStepHandler` is instantiated only inside `StepHandlerRegistry.default()`,
and `ExtensionHandler` is used only within `ExtensionsStepHandler`/`ExtensionHandlerRegistry`
in the same file.

This FEP removes both from the public API: drop them from `__all__`, and (optionally) rename
`ExtensionHandler` to `_ExtensionHandler` to make its internal status explicit. Per the FEP
process, *any* change to the framework public API requires an FEP even when the symbol is
demonstrably unused; this FEP is the required vehicle. The change is expected to be a no-op
for all known consumers (zero external importers), with a one-release deprecation shim as a
safety net for out-of-tree code that may have imported them against the advertised API.

## Motivation

### Problem Statement

Verified (grep + AST of `__all__`, 2026-07; recorded as backlog finding **F-013**):

- `victor/framework/step_handlers.py:2629` `__all__` contains `"CapabilityConfigStepHandler"`
  and `"ExtensionHandler"`.
- `CapabilityConfigStepHandler` (class at L968): **zero importers** outside `step_handlers.py`;
  instantiated once internally in `StepHandlerRegistry.default()`.
- `ExtensionHandler` (dataclass at L2047): **zero importers** outside `step_handlers.py`;
  used internally at L2346–2361 (registry wiring). It is *not dead code* — it is live
  internal detail whose only defect is being exported.

A public API is a contract: every exported name implies support, stability, and a migration
burden if changed. Exporting implementation detail inflates that contract for no consumer
benefit and invites external code to couple to internals.

### Goals

1. Remove `CapabilityConfigStepHandler` and `ExtensionHandler` from the framework public API.
2. Make `ExtensionHandler`'s internal status explicit (naming).
3. Do so under the FEP process because it is a framework-public-API change.

### Non-Goals

- Deleting or refactoring the classes themselves (they remain internally used).
- Auditing the rest of `framework/__all__` (a broader public-surface review is separate).

## Proposed Change

### High-Level Design

```python
# victor/framework/step_handlers.py

# Before
__all__ = [
    ...,
    "CapabilityConfigStepHandler",   # internal-only
    "ExtensionHandler",              # internal-only
    ...,
]

# After
__all__ = [
    ...,   # both removed
]

# ExtensionHandler renamed to signal internal status:
@dataclass
class _ExtensionHandler:
    ...
```

### Detailed Specification

#### Removal from `__all__`
- Delete `"CapabilityConfigStepHandler"` and `"ExtensionHandler"` from
  `step_handlers.py:__all__`. The classes stay defined and internally used.

#### Optional rename
- Rename `ExtensionHandler` → `_ExtensionHandler` (and its ~6 internal instantiation sites).
  `CapabilityConfigStepHandler` may stay named as-is (it is a public-looking handler class
  used via the registry); removing it from `__all__` is sufficient. Naming choices finalized
  in review.

#### Deprecation shim (safety net)
- For one minor release, keep module-level names importable but emit `DeprecationWarning`
  on access via a module `__getattr__`, so any out-of-tree importer (relying on the
  previously-advertised API) gets a clear signal rather than an immediate `ImportError`.

```python
def __getattr__(name: str):
    if name in {"ExtensionHandler", "CapabilityConfigStepHandler"}:
        warnings.warn(
            f"{name} is internal to step_handlers and no longer part of the "
            "framework public API (FEP-0015).",
            DeprecationWarning, stacklevel=2,
        )
        return globals()[f"_{name}"] if name == "ExtensionHandler" else globals()[name]
    raise AttributeError(name)
```

## Benefits

### For Framework Users
- A smaller, more honest public API — exported names reflect actually-supported surface.

### For the Ecosystem
- Removes an accidental coupling point; internal refactors of step-handler wiring no longer
  risk being treated as breaking public changes.

## Drawbacks and Alternatives

### Drawbacks
- Any out-of-tree code that imported these against the advertised API breaks (at Phase 2).
  *Mitigation*: one-release `DeprecationWarning` shim; both symbols are internal-only in-tree.

### Alternatives Considered

1. **Leave in `__all__`, add a comment.**
   - Pros: zero change.
   - Cons: the false public contract persists; comments do not bind.
   - Why rejected: exporting implementation detail is the defect.

2. **Remove from `__all__` without an FEP.**
   - Pros: faster.
   - Cons: violates the FEP process — `framework/` public API changes require an FEP
     (CONTRIBUTING.md / FEP-0001), regardless of usage.
   - Why rejected: process compliance is the point of this document.

3. **Delete the classes.**
   - Pros: maximal cleanup.
   - Cons: they are live internal implementation.
   - Why rejected: not dead code; only their export is wrong.

## Unresolved Questions

- **Rename `CapabilityConfigStepHandler` too?** (Proposed: no — leave named, just unexport;
  it is a coherent handler class, not an obviously-private helper.)
- **Is the `__getattr__` shim worth it** given zero in-tree importers? (Proposed: yes, cheap
  insurance for out-of-tree consumers who trusted the advertised `__all__`.)

## Implementation Plan

### Phase 1: Unexport + shim (1 PR)
- [ ] Remove both names from `step_handlers.py:__all__`.
- [ ] Rename `ExtensionHandler` → `_ExtensionHandler` at definition + internal call sites.
- [ ] Add the `__getattr__` deprecation shim.
- [ ] Add a guard test asserting neither name is in `framework`'s re-exported public surface.

**Deliverable**: symbols removed from public API; shim warns; internal behavior unchanged.

### Phase 2: Remove shim (1 small PR — release-gated)

Gated on **one minor release** elapsing after Phase 1 (PR #454), so any out-of-tree
importer had its deprecation window. Then:
- [ ] Delete the `__getattr__` shim and the `_FEP_0015_DEPRECATED` map from `step_handlers.py`
      (also drop the now-unused `import warnings` if nothing else uses it).
- [ ] Tighten the guard test (`test_step_handlers_exports.py`): replace the "old name warns
      and resolves" case with an assertion that `from victor.framework.step_handlers import
      ExtensionHandler` now raises `ImportError`/`AttributeError` (fully internal).
- [ ] Re-grep `victor/ tests/ verticals/` for the old names immediately before merging; expect
      zero (unchanged from Phase 1) — abort/extend the window if any appeared.

**Deliverable**: `ExtensionHandler`/`CapabilityConfigStepHandler` are fully internal; no shim,
no deprecation surface. `_ExtensionHandler` remains the internal name.

### Testing Strategy
- Guard test: `CapabilityConfigStepHandler`/`ExtensionHandler` not in `step_handlers.__all__`
  nor re-exported by `victor.framework`.
- Existing step-handler/registry tests must pass unchanged (behavior invariant).
- A test that importing the old name emits `DeprecationWarning` during Phase 1.

## Migration Path

### For any external importer
```python
# Before (relied on advertised __all__)
from victor.framework.step_handlers import ExtensionHandler
# After
# No supported replacement — this was internal detail. Depend on the public
# StepHandlerRegistry / ExtensionsStepHandler surface instead.
```

### Deprecation Timeline
- vX.Y: unexported; `DeprecationWarning` on access.
- vX.Y+1: shim removed.

## Compatibility

### Backward Compatibility
- **Breaking change**: Only for out-of-tree importers, and only at Phase 2. In-tree: no-op.
- **Migration required**: No (no in-tree or built-in-vertical importers exist).
- **Deprecation period**: one minor release.

### Vertical Compatibility
- Built-in verticals: no references (verified). No change.
- External verticals: unaffected unless they imported internal detail; shim warns them.

## References

- Backlog: `docs/architecture/CODEBASE_REVIEW_BACKLOG.md` → **F-013**
- Process: [FEP-0001](fep-0001-fep-process.md) (framework public-API changes require an FEP)
- Sibling FEP: [FEP-0014](fep-0014-canonical-validation-metrics-contracts.md)

## Review Process

### Submission
- **Submitted by**: Vijaykumar Singh
- **Date**: 2026-07-09
- **Pull Request**: TBD

### Review Checklist
#### Technical Review
- [ ] Confirm zero in-tree importers at merge time (re-grep)
- [ ] Rename scope agreed (`ExtensionHandler` only vs both)
- [ ] Shim behavior reviewed
#### Community Review
- [ ] Deprecation messaging clear

### Decisions
- **Recommendation**: Accept
- **Decision date**: 2026-07-09
- **Approved by**: Vijaykumar Singh (repo owner)
- **Rationale**: Exporting two internal-only symbols inflates the framework public contract with zero consumers; unexporting behind a one-release deprecation shim is low-risk and correct. Implementation proceeds Phase 1 (unexport + `_ExtensionHandler` rename + shim + guard).

## Acceptance Criteria

### Must-Have Criteria
1. **Names removed from public API**: absent from `step_handlers.__all__` and not re-exported
   by `victor.framework`. Verification: guard test.
2. **No behavior change**: all step-handler/registry tests pass unchanged.
   Verification: CI green.
3. **Deprecation signal**: importing the old names during Phase 1 emits `DeprecationWarning`.
   Verification: unit test.

### Should-Have Criteria
1. **Explicit internal naming** (`_ExtensionHandler`) — Priority: Medium.

### Implementation Requirements
- [ ] Guard test preventing re-export
- [ ] Changelog entry
- [ ] `docs/api-reference` note (if these were documented)

---

## Copyright

This FEP is licensed under the Apache License 2.0, same as the Victor project.
