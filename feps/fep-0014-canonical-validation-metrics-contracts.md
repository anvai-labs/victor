---
fep: "0014"
title: "Canonical validation and metrics contracts (deduplicate ValidationSeverity / ValidationResult / MetricsCollector)"
type: Standards Track
status: Implemented
created: 2026-07-09
modified: 2026-07-09
authors:
  - name: Vijaykumar Singh
    email: vijay@anvaiops.com
    github: vjsingh1984
reviewers: []
discussion: https://github.com/anvai-labs/victor/discussions/0014
---

# FEP-0014: Canonical validation and metrics contracts

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

Three foundational concepts — validation severity, validation results, and metrics
collection — are each defined independently in 4–5 modules across the `core`,
`config`, `framework`, `tools`, `agent`, and `integrations` layers. The copies share
names but **diverge in shape and semantics**: one `ValidationSeverity` adds a
`CRITICAL` level the others lack, and the five `ValidationResult` classes carry
different field names (`is_valid` vs `status`, `errors` vs `issues` vs `error_message`).
Because there is no shared type or protocol, these cannot be passed across layer
boundaries or handled polymorphically, and a severity/result produced in one layer
silently means something different in another.

This FEP establishes **one canonical `ValidationSeverity` and `ValidationResult` in
`victor/core`**, and a **`MetricsCollectorProtocol` in `victor/core/protocols.py`**
that the five concrete collectors structurally satisfy. Existing types become thin
re-exports (or `Protocol`-conforming implementations) during a deprecation window,
then are removed. This is a Standards Track FEP because it touches `victor/framework`
and `victor/core` public API. It is source-compatible during the deprecation window;
the only breaking change is the eventual removal of the duplicate definitions.

## Motivation

### Problem Statement

Verified inventory (grep, 2026-07; recorded as backlog finding **F-012**):

**`ValidationSeverity` — 4 definitions, one divergent:**
- `victor/config/validation.py:26` — `{ERROR, WARNING, INFO}`
- `victor/core/validation.py:82` — `{ERROR, WARNING, INFO}` (a `str, Enum`)
- `victor/framework/middleware.py:89` — `{INFO, WARNING, ERROR, CRITICAL}` ← **adds `CRITICAL`**
- `victor/framework/capabilities/validation.py:53` — `{ERROR, WARNING, INFO}`

The `CRITICAL`-only variant is the sharp edge: any code that compares or maps severities
across the middleware boundary and another layer is comparing enums with different
member sets. A "highest severity" reduction means different things depending on which
`ValidationSeverity` a value happens to be.

**`ValidationResult` — 5 definitions, incompatible fields:**
- `victor/tools/tool_call_validator.py:17` — `(is_valid, errors)`
- `victor/config/connection_validation.py:54` — `(status, message, issues)`
- `victor/workflows/protocols.py:759` — nested class `(…)`
- `victor/framework/requirement_validator.py:88` — `(is_valid, issues, error_message)`
- `victor/framework/capabilities/validation.py:62` — `(is_valid, issues, severity)`

No module imports another's `ValidationResult`; each re-invented the same "bool + list of
problems" shape with a different vocabulary, so results cannot be forwarded or merged
across layers without translation glue.

**`MetricsCollector` — 5 definitions, no shared protocol:**
- `victor/observability/metrics.py:607`
- `victor/integrations/api/event_bridge.py:139`
- `victor/agent/metrics_collector.py:162`
- `victor/experiments/ab_testing/metrics.py:45`
- `victor/framework/observability/metrics.py:1124`

These are legitimately *different implementations* (event-based, sampling-based, SLO,
A/B, agent TTFT/cost), but they share method names (`record_metric`, `get_snapshot`)
with **no declared protocol**, so nothing can depend on "a metrics collector" abstractly.
`agent/metrics_collector.MetricsCollector` is also re-exported from `victor/agent/__init__.py`.

### Goals

1. Exactly one `ValidationSeverity` and one `ValidationResult`, owned by `victor/core`,
   used across `core`, `config`, `framework`, `tools`, `workflows`.
2. A `MetricsCollectorProtocol` (in `victor/core/protocols.py`) that the five concrete
   collectors satisfy structurally, enabling polymorphic use without merging their
   implementations.
3. No silent severity/result incompatibility: `CRITICAL` is either canonical or removed,
   decided once.
4. A deprecation window in which all existing import paths keep working.

### Non-Goals

- Merging the five `MetricsCollector` *implementations* into one class. They serve
  distinct domains; only a shared **protocol** is proposed.
- Changing validation *behavior* (which checks run, what fails). This is a type-contract
  consolidation, not a rules change.
- Touching provider capability types (covered by prior work) or unrelated `*Result`
  dataclasses that are not validation results.

## Proposed Change

### High-Level Design

Introduce canonical contracts in the lowest layer that all others already depend on
(`victor/core`), then re-point the duplicates at them.

```python
# victor/core/validation.py  (canonical home)

class ValidationSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"   # unified superset — see Unresolved Questions

@dataclass(frozen=True)
class ValidationResult:
    is_valid: bool
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity >= ValidationSeverity.ERROR]
```

```python
# victor/core/protocols.py

@runtime_checkable
class MetricsCollectorProtocol(Protocol):
    def record_metric(self, name: str, value: float, **tags: str) -> None: ...
    def get_snapshot(self) -> Mapping[str, Any]: ...
```

### Detailed Specification

#### Canonical `ValidationSeverity`

- Canonical definition lives in `victor/core/validation.py`, exported from
  `victor/core/__init__.py`.
- Superset membership `{INFO, WARNING, ERROR, CRITICAL}` (adopting the middleware
  variant's `CRITICAL`) so no existing code loses a level.
- The other three definitions become `from victor.core.validation import ValidationSeverity`
  re-exports during deprecation, then are deleted.

#### Canonical `ValidationResult`

- Canonical shape: `is_valid: bool` + `issues: list[ValidationIssue]`, with compatibility
  accessors (`errors`, `error_message`) covering the fields the five variants exposed.
- Each call site migrates field access (`.status` → `.is_valid`, `.message`/`.error_message`
  → derived accessor). Adapters provided for the deprecation window.

#### `MetricsCollectorProtocol`

- Structural `Protocol` capturing the common surface (`record_metric`, `get_snapshot`,
  and any method used across ≥2 collectors — to be finalized by reading all five).
- The five implementations are left in place but gain the protocol as documentation and,
  where helpful, a nominal base for type hints. No behavior change.
- Renaming the concrete classes to domain-specific names (`EventMetricsCollector`,
  `ExperimentMetricsCollector`, …) is **optional** and, because
  `agent/__init__.py` re-exports `MetricsCollector`, is itself an `__all__` change — see
  Compatibility.

### API Changes

```python
# Before (framework/capabilities/validation.py)
class ValidationSeverity(Enum): ERROR = ...; WARNING = ...; INFO = ...
class ValidationResult: is_valid: bool; issues: list; severity: ...

# After
from victor.core.validation import ValidationSeverity, ValidationResult
```

### Dependencies

None new. Consolidation only.

## Benefits

### For Framework Users
- A single, importable `ValidationResult`/`ValidationSeverity` — no guessing which layer's
  variant a value came from.

### For Vertical Developers
- Verticals can depend on one validation contract via documented public API instead of
  reaching into a specific internal module.

### For the Ecosystem
- Removes a class of silent cross-layer bugs (severity/result shape mismatch).
- `MetricsCollectorProtocol` enables writing collector-agnostic instrumentation.

## Drawbacks and Alternatives

### Drawbacks
- Touches many call sites across 6 subsystems (medium blast radius). *Mitigation*: phased,
  behind re-export shims, with a guard test that fails if a new duplicate `ValidationResult`
  is introduced.
- Adopting `CRITICAL` everywhere slightly widens the enum for layers that never emit it.
  *Mitigation*: harmless — those layers simply never produce `CRITICAL`.

### Alternatives Considered

1. **Leave as-is, document the divergence.**
   - Pros: zero code churn.
   - Cons: the `CRITICAL` incompatibility remains a live footgun; no polymorphism.
   - Why rejected: F-012 exists precisely because "documented but divergent" is unsafe.

2. **Canonical types in `victor_contracts` instead of `victor/core`.**
   - Pros: available to external packages by contract.
   - Cons: validation/metrics are runtime concerns, not definition-layer SDK types;
     would widen the contract SDK surface.
   - Why rejected: `core` is the correct owning layer; verticals already import documented
     `core`/`framework` public APIs for such contracts.

3. **Unify the five `MetricsCollector` classes into one.**
   - Pros: maximal dedup.
   - Cons: they are genuinely different implementations; a merged god-collector is worse.
   - Why rejected: a shared **protocol** captures the polymorphism without the coupling.

## Unresolved Questions

- **Is `CRITICAL` canonical or removed?** (Proposed: **keep it** as the superset so no
  caller loses a level; audit whether any comparison logic assumed a 3-member enum.)
- **Final `MetricsCollectorProtocol` surface** — which methods beyond `record_metric` /
  `get_snapshot` are shared by ≥2 collectors? (Resolve by reading all five before Phase 1.)
- **Do we rename the concrete `MetricsCollector` classes**, or only add the protocol?
  (Proposed: protocol-only first; renames as an optional follow-up to limit `__all__` churn.)

## Implementation Plan

### Phase 1: Canonical types (1 PR)
- [ ] Define canonical `ValidationSeverity` (superset) + `ValidationResult` in `victor/core/validation.py`; export from `core/__init__.py`.
- [ ] Add `MetricsCollectorProtocol` to `victor/core/protocols.py`.
- [ ] Add a guard test asserting no module outside `core` defines a class named `ValidationResult`/`ValidationSeverity`.

**Deliverable**: canonical contracts exist; guard in place (initially xfail-listing known dupes).

### Phase 2: Migrate consumers (detailed — informed by Phase 1 findings)

**Phase 1 discovery that reshapes this phase**: the five `ValidationResult` types are **not one concept** (same name, different meaning — verified by reading each). They split into "true duplicate → migrate" vs "different concept → rename". Blindly re-pointing all five at the canonical type would corrupt semantics. Each item below is its own small PR that also removes its entry from the guard allowlist.

**2a — `ValidationSeverity` (all genuinely the same concept → re-export):**
- [ ] `config/validation.py:26`, `framework/middleware.py:89`, `framework/capabilities/validation.py:53` → replace each local enum with `from victor.core.validation import ValidationSeverity`. Members are `{ERROR,WARNING,INFO}` (+ `CRITICAL` in middleware, already in canonical), so no member loss. Verify `severity_rank`-based comparisons where any code ordered severities.

**2b + 2c — `ValidationResult` disambiguation (DONE, PR #460). Revised from the original Tier-A/Tier-B split:**

Executing 2b revealed that the two "Tier A" candidates carry **domain-specific extra fields** the canonical `{is_valid, issues}` lacks, so migrating them would **lose data**. Ground truth: *all four* `ValidationResult`s are distinct concepts sharing a name (the F-011 / `ApprovalMode` lesson) — so each was **renamed** to a distinct name rather than merged. The canonical `ValidationResult` stays the sanctioned shape for new code; the F-012 goal (kill the name collision) is met by disambiguation.

- [x] `tools/tool_call_validator.py` — `{valid, errors, warnings, **normalized_args**}` → **`ToolCallValidationResult`** (merge would drop `normalized_args`).
- [x] `framework/capabilities/validation.py` — `{is_valid, error_message, **severity, details, validator_id**}` → **`CapabilityValidationResult`**.
- [x] `config/connection_validation.py` — `{**status: enum**, message, details}` → **`ConnectionValidationResult`**.
- [x] `framework/requirement_validator.py` — `{is_satisfied, **satisfaction_score**, …}` → **`RequirementResult`**.
- Note: `workflows/protocols.py:759` `ValidationResult` is a docstring-only mention; the AST guard never saw it.
- Guard allowlist `ValidationResult` set now empty (with the empty `ValidationSeverity` set from 2a, the validation-consolidation portion of Phase 2 is complete).

**2d — `MetricsCollector` → conform to `MetricsCollectorProtocol` via ADAPTER METHODS (ratified approach):**
Phase 1 found the five collectors share no public method today (`record` vs `record_tool_execution`/`record_first_token`; `get_snapshot` vs `get_summary`/`get_metrics`). Rather than narrow the protocol, add **thin adapter methods** to each so they satisfy `record_metric(name, value, **tags)` + `get_snapshot()`:
- [ ] `observability/metrics.py:607` — add `record_metric`/`get_snapshot` delegating to its existing record path + `get_summary`.
- [ ] `framework/observability/metrics.py:1124` — already has `get_snapshot`; add `record_metric` delegating to `record`.
- [ ] `agent/metrics_collector.py:162` — add generic `record_metric` + `get_snapshot` alongside its typed `record_*` methods.
- [ ] `integrations/api/event_bridge.py:139`, `experiments/ab_testing/metrics.py:45` — add the two adapter methods delegating to existing internals.
- [ ] Then assert `isinstance(x, MetricsCollectorProtocol)` for all five in a test; remove the Phase-1 "not-yet-conforming" baseline assertion.

**2e — Prep quick-win (unblocks the canonical name):**
- [ ] Rename `core/pickle_cache.py:ValidationResult{ok,delete,reason}` → `CacheValidation` (3 in-tree sites: `semantic_selector.py`, `collections.py`, `test_pickle_cache.py`). An unrelated collision introduced by PR #449; renaming removes that pragmatic guard-allowlist entry.

**Deliverable**: guard allowlist empties (the done-signal); every consumer on the canonical severity/result or an honestly-renamed distinct type; all five collectors conform to `MetricsCollectorProtocol`.

### Phase 3: Remove shims (1 PR, next minor)
- [ ] Delete the re-export shims and duplicate definitions.

**Deliverable**: single source of truth; guard enforces it going forward.

### Testing Strategy
- Unit tests for canonical types (severity ordering incl. `CRITICAL`, result accessors).
- Per-consumer regression tests unchanged (behavior must not move).
- Guard test prevents re-duplication.

### Rollout Plan
- No feature flag. Phased PRs; each independently green. Docs: update `docs/api-reference`
  validation section + add `MetricsCollectorProtocol`.

## Migration Path

### From duplicate to canonical

1. Replace local `class ValidationSeverity(...)` / `class ValidationResult(...)` with
   `from victor.core.validation import ValidationSeverity, ValidationResult`.
2. Update field access: `.status`→`.is_valid`, `.message`/`.error_message`→`.errors`/accessor.

```python
# Before
if result.status and not result.issues: ...
# After
if result.is_valid and not result.issues: ...
```

### Deprecation Timeline
- vX.Y: canonical types added; duplicates become deprecated re-exports (warn on import).
- vX.Y+1: duplicate definitions removed.

## Compatibility

### Backward Compatibility
- **Breaking change**: Only at Phase 3 (removal of duplicates). Phases 1–2 are additive.
- **Migration required**: Yes for direct importers of the removed duplicates (in-repo only;
  all identified consumers are first-party).
- **Deprecation period**: one minor release.

### Vertical Compatibility
- Built-in verticals: none import these duplicates today (verified). No change required.
- External verticals: should import the canonical `core` types; deprecation warnings guide them.

## References

- Backlog: `docs/architecture/CODEBASE_REVIEW_BACKLOG.md` → **F-012**
- Sibling FEP: [FEP-0015](fep-0015-trim-internal-framework-exports.md) (framework `__all__` hygiene)
- Prior art: provider capability-surface consolidation (PR #442) — same "one concept, one surface" principle.

## Review Process

### Submission
- **Submitted by**: Vijaykumar Singh
- **Date**: 2026-07-09
- **Pull Request**: TBD

### Review Checklist
#### Technical Review
- [ ] `MetricsCollectorProtocol` surface validated against all five implementations
- [ ] `CRITICAL` decision ratified
- [ ] Guard test design reviewed
#### Community Review
- [ ] Migration path clear for external verticals

### Decisions
- **Recommendation**: Accept
- **Decision date**: 2026-07-09
- **Approved by**: Vijaykumar Singh (repo owner)
- **Rationale**: The `CRITICAL`-only `ValidationSeverity` divergence and five incompatible `ValidationResult` shapes are a live cross-layer correctness hazard; the phased, shim-backed, guard-enforced plan removes it without behavior change. Implementation proceeds Phase 1 first (canonical types + protocol + guard).

## Acceptance Criteria

### Must-Have Criteria
1. **Single validation contract**: exactly one `ValidationSeverity` and one `ValidationResult`
   in the tree (guard-enforced). Verification: guard test green with empty allowlist.
2. **No behavior regression**: all existing validation tests pass unchanged.
   Verification: CI green across affected subsystems.
3. **Metrics protocol**: `isinstance(collector, MetricsCollectorProtocol)` holds for all five
   collectors. Verification: unit test per collector.

### Should-Have Criteria
1. **Domain-named collectors** (`EventMetricsCollector`, …) — Priority: Low.

### Implementation Requirements
- [ ] Guard test preventing re-duplication
- [ ] `docs/api-reference` updated
- [ ] Deprecation warnings on shim imports
- [ ] Changelog entry

---

## Copyright

This FEP is licensed under the Apache License 2.0, same as the Victor project.
