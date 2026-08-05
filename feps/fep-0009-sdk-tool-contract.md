---
fep: 0009
title: "SDK Tool Contract — promote tool metadata/traits into victor-contracts"
type: Standards Track
status: Draft
created: 2026-06-22
modified: 2026-06-22
authors:
  - name: Vijaykumar Singh
    email: vijay@anvaiops.com
    github: vjsingh1984
reviewers: []
discussion: https://github.com/anvai-labs/victor/discussions
---

# FEP-0009: SDK Tool Contract — promote tool metadata/traits into victor-contracts

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

Victor resolves a tool's metadata/traits through a single internal authority,
`victor.tools.contract.resolve_contract(tool)` (tool-supply P6, PR #234), which returns
the framework-internal `victor.tools.metadata.ToolMetadata` dataclass. The trait
vocabulary it depends on — `category`, `keywords`, `priority`, `access_mode`,
`danger_level`, `execution_category`, `task_types`, `stages`, `cost_tier`, `schema_level`
— lives entirely under `victor/tools/` (the dataclass in `metadata.py`, the enums in
`enums.py`). External, SDK-first verticals (`victor-coding`, `victor-devops`,
`victor-rag`, `victor-dataanalysis`, `victor-research`, `victor-invest`) must import only
from `victor_contracts`; today they have **no stable, importable way to declare these
traits** for the tools they contribute, so they either under-declare metadata or reach
into `victor.tools.*` (a boundary violation). This FEP promotes a minimal, frozen
**Tool Contract** — a data-only `ToolContract` plus the trait enums — into
`victor_contracts`, versioned with a `CapabilityContract`, and makes the existing
internal `ToolMetadata` a superset/adapter of it. The change is **additive**: it
introduces no breaking change for the framework or current tools; external packages opt
in. The audience is tool authors and the six external vertical maintainers.

## Motivation

### Problem Statement

P6 deliberately made `resolve_contract` the one fusion authority for tool metadata but
kept it **framework-internal** ("minimal cut … no new parallel contract type"). That was
the right call for an internal refactor, but it leaves a real gap at the SDK boundary:

- **No SDK trait surface.** `ToolMetadata` (`victor/tools/metadata.py:38`) and its enums
  (`CostTier`, `Priority`, `AccessMode`, `ExecutionCategory`, `DangerLevel`,
  `SchemaLevel` in `victor/tools/enums.py`) are all under `victor/tools/`. The SDK's
  tool surface (`victor_contracts/tool_runtime.py`) is a lazy re-export shim with no
  metadata vocabulary of its own.
- **External verticals can't declare traits cleanly.** The architecture mandate (see
  `CLAUDE.md`, "Extension System & SDK Contract") is that external verticals import only
  `victor_contracts` / `victor.framework.extensions`. A `victor-coding` tool that wants
  to set `access_mode=WRITE`, `danger_level=...`, or `category="git"` has no SDK type to
  declare against. The choices today are: (a) omit the traits and accept worse tool
  selection/safety defaults, or (b) import `victor.tools.enums` / `victor.tools.metadata`
  — a boundary violation that couples the vertical to root internals.
- **The autogen fallback can't see intent.** When traits are omitted, `resolve_contract`
  falls back to `ToolMetadata.generate_from_tool(...)`, which infers from name/description
  /parameters. Inference is necessarily weaker than a declared contract — e.g. it cannot
  know a tool mutates the filesystem (`access_mode`) or is network-bound
  (`execution_category`) from its name alone.

This matters now because the tool-supply track (P0–P8) made tool *selection, capping,
prompt-splitting, and parallel-execution* all read these traits. Tools that under-declare
get systematically worse treatment, and the packages most affected are exactly the ones
that can't declare — the external verticals.

### Goals

1. Provide a **stable, frozen, data-only** `ToolContract` and trait enums in
   `victor_contracts` that external tool authors can import and declare against.
2. Keep `resolve_contract` the single fusion authority; have it **bridge** an
   SDK-declared contract into the internal `ToolMetadata` with byte-stable precedence.
3. Version the contract explicitly via `CapabilityContract` so it can evolve
   independently and external packages can assert compatibility.
4. **Zero breaking change**: existing tools, `ToolMetadata`, and `@tool(...)` keep
   working unchanged; SDK declaration is purely opt-in.

### Non-Goals

- Not replacing `ToolMetadata`. The internal dataclass stays as the richer
  runtime-internal type (it carries selection/loop-detection fields that are not part of
  the public contract). `ToolContract` is the *declarable subset*.
- Not collapsing the execution-category axis with the organizational `ToolCategory`
  axis (those are intentionally distinct — see PR #238).
- Not changing tool *registration* or discovery mechanics, the `@tool` decorator's
  Python signature, or `BaseTool`.
- Not moving keyword/semantic-selection logic into the SDK; only the declarative traits
  move, not the selection engine.

## Proposed Change

### High-Level Design

```
External vertical tool (SDK-only imports)
    │  declares
    ▼
victor_contracts.tools.ToolContract  ◀── frozen dataclass + trait enums (NEW, SDK)
    │  consumed by
    ▼
victor.tools.contract.resolve_contract(tool)   ◀── single fusion authority (P6, exists)
    │  bridges SDK contract → internal
    ▼
victor.tools.metadata.ToolMetadata   ◀── internal superset (existing, unchanged shape)
    │  read by
    ▼
tool selection / capping / prompt-split / parallel-exec  (tool-supply P0–P8)
```

The SDK gains a *declaration* type; the framework keeps the *resolution* and *runtime*
types. `resolve_contract` is the only place the two meet.

### Detailed Specification

#### New SDK module: `victor_contracts/tools.py`

A new `victor_contracts.tools` module exposes:

```python
# victor_contracts/tools.py  (NEW — frozen, data-only, no framework imports)
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum

class ToolCategory(str, Enum):
    CORE = "core"; FILESYSTEM = "filesystem"; GIT = "git"; SEARCH = "search"
    WEB = "web"; DATABASE = "database"; DOCKER = "docker"; TESTING = "testing"
    REFACTORING = "refactoring"; DOCUMENTATION = "documentation"; ANALYSIS = "analysis"
    COMMUNICATION = "communication"; NOTEBOOK = "notebook"
    TASK_MANAGEMENT = "task_management"; VERIFICATION = "verification"; CUSTOM = "custom"

class AccessMode(str, Enum):
    READONLY = "readonly"; WRITE = "write"; EXECUTE = "execute"

class DangerLevel(str, Enum):
    SAFE = "safe"; CAUTION = "caution"; DANGEROUS = "dangerous"

class ExecutionCategory(str, Enum):
    READ_ONLY = "read_only"; WRITE = "write"; NETWORK = "network"; COMPUTE = "compute"

class CostTier(str, Enum):
    FREE = "free"; LOW = "low"; MEDIUM = "medium"; HIGH = "high"

@dataclass(frozen=True)
class ToolContract:
    """Declarable, stable subset of a tool's metadata (SDK boundary type).

    Carries only *declarable intent*: the safety/economics/identity traits that drive
    capping, parallel-exec, and prompt-split, plus the author's selection hints. It does
    NOT carry selection-*engine* knobs (priority/priority_hints, mandatory_keywords,
    signature_params) — those stay internal to ``ToolMetadata`` and are owned by the
    selection engine / RL, not the tool author. (Resolves Q2.)
    """
    category: ToolCategory | str = ToolCategory.CUSTOM
    access_mode: AccessMode = AccessMode.READONLY
    danger_level: DangerLevel = DangerLevel.SAFE
    execution_category: ExecutionCategory = ExecutionCategory.READ_ONLY
    cost_tier: CostTier = CostTier.LOW
    keywords: tuple[str, ...] = ()
    use_cases: tuple[str, ...] = ()
    task_types: tuple[str, ...] = ()
    stages: tuple[str, ...] = ()

# Versioned at the SDK floor where ToolContract first exists (resolves Q3). External
# packages assert this contract — not the raw victor-ai package range — for the feature.
CONTRACT = CapabilityContract(name="tools", version=1, min_sdk_version=">=0.7.1")
```

Notes:
- **`str`-valued enums** (like the framework's `SchemaLevel`) so values serialize/compare
  as plain strings across the package boundary and tolerate version skew.
- **Frozen + tuple fields** so the contract is hashable and immutable (safe to share).
- **Data-only**: no imports from `victor.*`. This is what lets external packages depend on
  it without pulling root internals.
- **`ToolCategory` is owned here (resolves Q1).** This is the same 16-name vocabulary from
  PR #238; `victor.framework.tools.ToolCategory` re-exports this SDK enum (adding only the
  legacy ``REFACTOR`` back-compat alias at the framework layer), and the existing
  ``test_tool_category_yaml_parity.py`` guard extends to assert SDK ≡ framework ≡ YAML.
- **No `Priority` in the contract (resolves Q2).** "Everyone declares HIGH" is a foot-gun;
  ranking stays an engine/RL concern.

#### Bridge in `resolve_contract` (framework side)

`resolve_contract` gains one branch, preserving its current precedence
(explicit → autogen) and byte-stability:

```python
def resolve_contract(tool: Any) -> ToolMetadata:
    explicit = getattr(tool, "metadata", None)
    if explicit is not None:
        return explicit
    sdk = getattr(tool, "contract", None)            # NEW: SDK-declared ToolContract
    if sdk is not None:
        return ToolMetadata.from_contract(sdk, tool) # NEW: bridge, then same shape
    return ToolMetadata.generate_from_tool(...)      # unchanged autogen fallback
```

`ToolMetadata.from_contract(...)` maps each SDK field onto the matching internal field and
enum (the internal `enums.py` values are unchanged), filling the remaining internal-only
fields (e.g. `signature_params`, `mandatory_keywords`) from autogen defaults. Tools that
set the existing `.metadata` continue to win — **no precedence change**.

#### Versioning

`victor_contracts.tools.CONTRACT` is a `CapabilityContract` (already in
`victor_contracts/core/capability_contract.py`). External packages assert
`CONTRACT.is_sdk_compatible(installed_sdk_version)` and the framework registers it in the
`CapabilityContractRegistry` so the existing `check_all()` health path covers it. The
contract version is independent of `CURRENT_API_VERSION` (manifest API) so the trait
vocabulary can evolve without a manifest bump.

### API Changes

```python
# Before — external vertical has no SDK option; must violate the boundary:
from victor.tools.enums import AccessMode          # ❌ root internal import
from victor.tools.metadata import ToolMetadata     # ❌

# After — declare against the SDK:
from victor_contracts.tools import ToolContract, AccessMode, ToolCategory

class DeployTool(BaseTool):
    contract = ToolContract(
        category=ToolCategory.DOCKER,
        access_mode=AccessMode.EXECUTE,
        danger_level=DangerLevel.CAUTION,
    )
```

No existing signatures change. `ToolMetadata` gains one classmethod (`from_contract`);
`resolve_contract` gains one branch.

### Dependencies

- No new third-party dependencies.
- `victor-contracts` additive patch bump (new module, backward-compatible): `0.7.0 → 0.7.1`.
- **No `victor-ai` dependency-range change required**: the current pin
  `victor-contracts>=0.6.0,<1.0` already admits `0.7.1`. The only requirement is that
  contracts `0.7.1` is released (and the `CONTRACT.min_sdk_version=">=0.7.1"` capability
  gate enforces the feature floor at runtime).

## Benefits

### For Framework Users

- Tools (internal or external) declare traits the same way; better selection/safety
  defaults because intent is declared, not inferred.
- Safer execution: a tool that declares `access_mode=WRITE` / `danger_level=DANGEROUS`
  is treated correctly by approval and parallel-execution paths instead of relying on
  name-based heuristics that can misclassify a mutating or network-bound tool as safe.

### For Vertical Developers

- A supported, import-clean way to declare tool traits — closes the boundary violation
  that is otherwise the only path. The six external verticals get parity with built-ins.
- One declaration, no churn: authors write a frozen `ToolContract` and never import
  `victor.tools.*`, so a root-internal refactor cannot break their package.

### For the Ecosystem

- A stable, versioned trait vocabulary other tooling (registry/marketplace, linters,
  docs) can read without depending on root internals.
- The `CapabilityContract` version makes the trait surface independently evolvable and
  machine-checkable, so consumers can detect incompatibility at install/validate time
  rather than failing at runtime.

## Drawbacks and Alternatives

### Drawbacks

- **Two trait types** (`ToolContract` SDK + `ToolMetadata` internal). Mitigation:
  `from_contract` is the only bridge; a guard test pins field/enum parity so they cannot
  drift, and `ToolContract` is explicitly the declarable *subset*, not a parallel runtime
  type.
- **Six-package lockstep** on the contracts bump. Mitigation: additive + version-gated;
  external packages opt in on their own schedule because the framework still accepts
  tools with no `contract`.

### Alternatives Considered

1. **Keep it framework-internal (status quo / P6 minimal cut)**
   - Pros: zero SDK surface to maintain.
   - Cons: external verticals keep under-declaring or violating the boundary. Rejected:
     the tool-supply track made traits matter for exactly those packages.
2. **Move `ToolMetadata` wholesale into the SDK**
   - Pros: one type.
   - Cons: drags runtime-internal fields (loop-detection `signature_params`, selection
     internals) and their churn into a stable public contract. Rejected: violates "stable
     SDK, evolving internals."
3. **TypedDict / plain dict contract**
   - Pros: ultra-light.
   - Cons: no enum safety, no frozen identity, weaker validation at the boundary.
     Rejected: trait correctness (access_mode/danger_level) is safety-relevant.

## Unresolved Questions

All three open questions from the initial draft (v1.0) are **resolved below** with a
recommended decision and rationale, and reflected in the spec above. They remain open for
reviewers to overturn during the review window, but the FEP now proposes a concrete answer
for each rather than leaving them undecided.

- **Q1: `ToolCategory` ownership across the boundary — RESOLVED: the SDK owns it.**
  `victor_contracts.tools.ToolCategory` is the single category-name authority;
  `victor.framework.tools.ToolCategory` re-exports it (adding only the legacy `REFACTOR`
  back-compat alias at the framework layer so `from victor.framework.tools import
  ToolCategory` keeps working), and `tests/unit/framework/test_tool_category_yaml_parity.py`
  extends to assert **SDK ≡ framework ≡ YAML**.
  *Rationale:* victor-contracts is the lower layer (victor-ai depends on it), so the
  identity authority established in PR #238 should live there; a re-export avoids a third
  parallel list and keeps every existing import path working. *Alternative (rejected):* a
  parallel framework list pinned equal by a guard — it duplicates the vocabulary and invites
  exactly the drift PR #238 removed.

- **Q2: Which fields belong in the contract — RESOLVED: declarable intent only.**
  Include `category`, `access_mode`, `danger_level`, `execution_category`, `cost_tier`
  (safety/economics/identity, which drive capping/parallel-exec/prompt-split) plus the
  author's selection *hints* (`keywords`, `use_cases`, `task_types`, `stages`). **Exclude
  `priority`** and the other engine-tuning fields (`priority_hints`, `mandatory_keywords`,
  `signature_params`) — they stay internal to `ToolMetadata`.
  *Rationale:* `keywords`/`use_cases` are genuine author knowledge (what triggers the tool);
  `priority`/`mandatory_keywords` are ranking knobs where author-declared values are a
  foot-gun ("everyone declares HIGH / mandatory") and are better owned by the selection
  engine and RL. `signature_params` is a loop-detection internal with no author meaning.

- **Q3: capability `min_sdk_version` — RESOLVED: pin to the contracts release that ships
  the module, package range unchanged.** The module ships additively in the `0.7.x` line
  (not a `0.8` minor), so `CONTRACT.min_sdk_version=">=0.7.1"`; `victor-ai` keeps
  `victor-contracts>=0.6.0,<1.0` (already admits `0.7.1`).
  *Rationale:* external packages should assert the **capability contract**, not the raw
  package range, so the feature floor and the dependency window are decoupled — the broad
  range avoids forcing a lockstep bump on packages that don't use the contract yet.

## Implementation Plan

### Phase 1: Contract in the SDK (1 PR, contracts repo)

- [ ] Add `victor_contracts/tools.py` (frozen `ToolContract` + trait enums + `CONTRACT`).
- [ ] Unit tests: enum value stability, frozen/hashable, default round-trip.
- [ ] Bump `victor-contracts/VERSION` → `0.7.1` (additive patch; keep the victor-ai floor
      broad — do NOT run `sync_version.py --sdk`, which would raise it).

**Deliverable**: importable, versioned SDK contract; contracts `0.7.1` released.

### Phase 2: Framework bridge (1 PR, root repo)

- [ ] `ToolMetadata.from_contract(contract, tool)`.
- [ ] `resolve_contract` SDK branch (precedence unchanged).
- [ ] Register `CONTRACT` in the `CapabilityContractRegistry` (no `victor-ai` dependency-
      range change — `>=0.6.0,<1.0` already admits `0.7.1`).
- [ ] Parity guard test: `ToolContract` fields/enums ⊆ `ToolMetadata`; bridge byte-stable
      against the current autogen for an undeclared tool.

**Deliverable**: tools may declare `contract = ToolContract(...)`; resolution honors it.

### Phase 3: Adopt + document (per external package, opt-in)

- [ ] Migrate one built-in tool and one external vertical tool as reference.
- [ ] Update the SDK/extension docs and the external-vertical compatibility matrix.

**Deliverable**: worked example + docs; external packages adopt on their own schedule.

### Testing Strategy

- Unit: contract immutability/enum stability; `from_contract` mapping; parity guard.
- Integration: a tool declaring only `contract` flows through selection/capping/parallel-
  exec identically to the same tool declaring `.metadata`.
- Backward compatibility: existing tools (no `contract`) produce byte-identical
  `ToolMetadata` to today (autogen path unchanged).

### Rollout Plan

- No feature flag (additive; absence of `contract` = today's behavior).
- Contracts `0.7.1` first, then the framework bridge, then opt-in adoption.
- External-vertical compatibility CI (the existing dispatch to the six repos) gates the
  lockstep.

## Migration Path

### From internal-only to SDK contract

1. External tool currently importing root internals (or under-declaring):
   ```python
   from victor.tools.enums import AccessMode  # boundary violation
   ```
2. Switch to the SDK contract:
   ```python
   from victor_contracts.tools import ToolContract, AccessMode
   class T(BaseTool):
       contract = ToolContract(access_mode=AccessMode.WRITE)
   ```
3. Built-in tools may stay on `.metadata`/`@tool(...)` indefinitely; optional convergence
   later.

### Deprecation Timeline

- `0.7.1`: `ToolContract` introduced (additive). No deprecations.
- Future (separate FEP, if ever): consider re-expressing `@tool(...)` trait kwargs in
  terms of `ToolContract`. Not in scope here; no removal is proposed.

## Compatibility

### Backward Compatibility

- **Breaking change**: No.
- **Migration required**: No (opt-in).
- **Deprecation period**: N/A (nothing deprecated).

### Version Compatibility

- Minimum Python: 3.10 (unchanged).
- `victor-contracts`: `0.7.1` (additive module in the 0.7.x line).
- `victor-ai`: keeps `victor-contracts>=0.6.0,<1.0` (already admits `0.7.1`); the feature
  floor is enforced by `CONTRACT.min_sdk_version=">=0.7.1"`, not the package range.

### Vertical Compatibility

- Built-in verticals: no change required.
- External verticals: opt-in; gain a supported declaration path. Lockstep handled by the
  existing external-vertical compatibility workflow.

## References

- PR #234 — `resolve_contract()` single metadata authority (tool-supply P6)
- PR #238 — ToolCategory ⇄ YAML single vocabulary authority (P6 follow-on)
- `victor/tools/metadata.py`, `victor/tools/enums.py` — internal metadata + trait enums
- `victor_contracts/core/capability_contract.py` — per-capability versioning
- `CLAUDE.md` — "Extension System & SDK Contract", "Plugin → Vertical → Extension"
- [FEP-0005](./fep-0005-policy-engine.md), [FEP-0006](./fep-0006-external-harness-executors.md)
  — prior SDK-boundary-touching FEPs

## Review Process

### Submission

- **Submitted by**: Vijaykumar Singh
- **Date**: 2026-06-22
- **Pull Request**: (this PR)

### Review Timeline

- **Initial review period**: 14 days minimum
- **Reviewers assigned**: TBD
- **Discussion thread**: TBD

### Review Checklist

#### Technical Review

- [ ] Specification is clear and complete
- [ ] API design follows Victor conventions (frozen, str-enums, data-only SDK type)
- [ ] Bridge precedence is byte-stable with today's `resolve_contract`
- [ ] Testing strategy is adequate (parity + backward-compat guards)
- [ ] Documentation plan is included

#### Community Review

- [ ] Use cases are well-understood (external verticals declaring traits)
- [ ] Benefits outweigh drawbacks (two types vs boundary cleanliness)
- [ ] Migration path is clear and opt-in
- [ ] Alternatives were considered
- [ ] Lockstep/compatibility addressed

### Decisions

- **Recommendation**: TBD
- **Decision date**: TBD
- **Approved by**: TBD
- **Rationale**: TBD

### Revision History

1. **v1.0** (2026-06-22): Initial submission.
2. **v1.1** (2026-06-22): Resolved the three open questions with recommendations (Q1 SDK
   owns `ToolCategory`; Q2 declarable-intent-only fields, drop `priority`; Q3 `>=0.8`
   capability gate, package range unchanged). Corrected the dependency claim — the existing
   `victor-contracts>=0.6.0,<1.0` already admits `0.8`, no range change needed.
3. **v1.2** (2026-06-22): **Implemented** (Phase 1 #245 contract module; Phase 2 #246
   duck-typed `resolve_contract` bridge) and **retargeted the shipping line to `0.7.1`**
   (additive patch in the 0.7.x line) instead of a `0.8` minor — per the maintainer's
   release decision. `CONTRACT.min_sdk_version` is now `>=0.7.1`; the victor-ai range stays
   `>=0.6.0,<1.0`. Pre-release dev uses editable installs (#247). Q1 (framework
   `ToolCategory` re-export) remains deferred — it needs a hard SDK import that would raise
   the victor-ai floor.

## Acceptance Criteria

### Must-Have Criteria

1. **Data-only SDK contract**: `victor_contracts.tools` imports nothing from `victor.*`.
   - Success metric: import-boundary test passes.
   - Verification: AST/import guard in `victor-contracts/tests`.
2. **Byte-stable resolution**: a tool with no `contract` yields the same `ToolMetadata`
   as today.
   - Success metric: parity test green.
   - Verification: golden compare against current autogen output.
3. **Field/enum parity**: every `ToolContract` field/enum maps onto an existing
   `ToolMetadata` field/enum (no orphan trait).
   - Success metric: parity guard test.
   - Verification: test enumerates contract fields against `ToolMetadata`.

### Should-Have Criteria

1. **Reference adoption**: one built-in and one external tool migrated.
   - Success metric: both resolve identically pre/post.
   - Priority: Medium.
2. **`ToolCategory` unification** across SDK and framework (per Q1: SDK owns it, framework
   re-exports).
   - Success metric: single vocabulary; `test_tool_category_yaml_parity.py` asserts
     SDK ≡ framework ≡ YAML.
   - Priority: Medium.

### Implementation Requirements

- [ ] Code implementation following the specification
- [ ] Comprehensive test coverage (>80% for new code)
- [ ] API documentation updated
- [ ] Migration guide (opt-in adoption) completed
- [ ] Changelog entries (both packages) added
- [ ] Backward compatibility verified (autogen byte-parity)
- [ ] External-vertical compatibility workflow green

### Validation Process

1. **Automated validation**: import-boundary + parity + backward-compat tests in CI.
2. **Manual review**: API review against SDK conventions.
3. **Community testing**: external-vertical maintainers trial the contract pre-`1.0`.
4. **Final approval**: maintainer sign-off after the 14-day window.

### Success Metrics

- Boundary violations (`victor.tools.*` imports in external verticals): target **0**.
- Tools declaring traits via the SDK contract: target **≥1 per external vertical** within
  one release after adoption docs land.

---

## Copyright

This FEP is licensed under the Apache License 2.0, same as the Victor project.
