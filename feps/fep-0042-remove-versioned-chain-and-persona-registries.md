---
fep: "0042"
title: "Remove the versioned chain registry and persona provider (F-019 Phase 2)"
type: Standards Track
status: Accepted
created: 2026-10-30
modified: 2026-10-30
authors:
  - name: Vijaykumar Singh
    email: vijay@anvaiops.com
    github: vjsingh1984
reviewers: []
discussion: null
---

# FEP-0042: Remove the versioned chain registry and persona provider

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

Victor has two registries for chains and two for personas. The flat registries
(`victor/framework/chain_registry.py`, `victor/framework/persona_registry.py`, sharing the
internal `namespaced_registry.py` base since PR #1608) are the only ones the runtime reads.
The host fills them from each vertical's `get_composed_chains()` / `get_chains()` /
`get_personas()` hook when the vertical is activated. The versioned registries
(`victor/framework/chains/`, `victor/framework/multi_agent/persona_provider.py`) and their
SDK bridges have no runtime readers. They were deprecated with victor-contracts 0.13.0,
released 2026-10-30, and in the victor-ai `[Unreleased]` changelog (backlog **F-019**).

This FEP is Phase 2 of that deprecation, and it is gated on releases. It removes the two host
modules and their tests, `victor_contracts.chain_runtime`, and
`victor_contracts.multi_agent.get_runtime_persona_provider()`. It also settles the persona
kernel's default-registry path. That path, `register_vertical_personas(..., provider=None)`,
is a silent no-op in production. The FEP recommends **removing import-time registration** and
making the activation hook the only writer. The repointing option is rejected. Under the
contract stability policy, the inert `RegistrationPolicy.provider_getter` field and the
default-registry path have not yet had a warning release. They are deprecated in 0.14.0 and
removed in 0.15.0 or later. Production behaviour does not change: nothing removed here is
read at runtime. External code that imported the deprecated surfaces must switch to the
vertical hooks.

## Motivation

### Problem Statement

These references were checked on `origin/develop` at `64c0dad86` on 2026-10-30.

**Readers use only the flat registries.**

- `victor capabilities` reads `victor.framework.persona_registry.get_persona_registry()` and
  `victor.framework.chain_registry.get_chain_registry()` at `victor/ui/commands/capabilities.py:150-160`
  and `:264-285`.
- `chain:` workflow handlers resolve through the flat `get_chain_registry()` at
  `victor/workflows/executors/compute.py:262,277`.
- Activation writes them. `FrameworkStepHandler.apply_chains` / `apply_personas` are at
  `victor/framework/step_handlers.py:1557-1562` and `:1606-1609`. They go through
  `FrameworkIntegrationRegistryService.register_chains` / `register_personas` at
  `victor/framework/framework_integration_registry_service.py:197-228`, which applies
  fingerprint idempotency and `replace=True`.

**The versioned registries have no readers.**

- `victor/framework/chains/` (470 lines) defines a second `ChainRegistry` /
  `get_chain_registry` (`chains/registry.py:92,401`). The names match the flat registry's
  public exports (`victor/framework/__init__.py:374-379`). Only tests and the deprecated
  `victor_contracts.chain_runtime` bridge (`chain_runtime.py:23-26`) import it.
- `FrameworkPersonaProvider` (`multi_agent/persona_provider.py:106`, 451 lines) repeats the
  same semver helpers (`_is_valid_semver`, `_semver_key`). Production code never constructs
  it. The only paths to it are the lazy, warning `__getattr__` in
  `victor/framework/multi_agent/__init__.py:111-121` and the deprecated
  `get_runtime_persona_provider()` (`victor_contracts/multi_agent.py:19-39`).

**The persona kernel's default path does nothing.**

- With `provider=None`, `register_vertical_personas` falls back to
  `get_default_persona_registry()` and returns `None` when that is unset
  (`victor_contracts/verticals/persona_kernel.py:303-315`).
- The only caller of `set_default_persona_registry` is `FrameworkPersonaProvider.__init__`
  (`persona_provider.py:156`). Production never constructs that class, so the default stays
  `None`.
- victor-coding, victor-devops and victor-research still call this path at import
  (`victor_coding/teams/personas.py:566,570`, `victor_devops/teams/personas.py:520,524`,
  `victor_research/teams/personas.py:562,566`). victor-dataanalysis calls it explicitly
  (`personas.py:551`). Each of these calls has no effect.
- `RegistrationPolicy.provider_getter` and `auto_register` (`persona_kernel.py:297-298`) are
  never read. Each vertical decides for itself whether to register at import.

**There is also a related gap that already exists.** On activation, only victor-coding delivers
personas. This was checked empirically with `ensure_runtime_vertical(V).get_personas()`. It
returns 8 personas for coding and 0 for devops, research and dataanalysis. Those three keep
their personas in `teams/personas.py`, which the `PersonasHandler` constant lookup
(`<VERTICAL>_PERSONAS` in the `teams` module) does not find. victor-dataanalysis also spells
the constant `DATA_ANALYSIS_PERSONAS`, not `DATAANALYSIS_PERSONAS`. As a result, these personas
are in **no** registry today.

### Goals

1. Delete the versioned registries, their SDK bridges, and their tests, so that one writer
   (activation) and one reader surface (the flat registries) remain.
2. Give the persona kernel's dormant default-registry path a deliberate outcome instead of a
   silent no-op.
3. Follow the victor-contracts deprecation policy (warn for at least one minor release, remove
   only in a minor release) and the release-order rule (SDK first, then victor-ai).

### Non-Goals

- Changing the flat registries, `NamespacedRegistry`, or the activation step handlers.
- Adding versioning to the flat registries. No consumer has asked for versioned lookup.
- Deciding whether the sibling `set_default_team_registry` hook should stay. It has a real
  writer and is out of scope.

## Proposed Change

### High-Level Design

```text
Before (0.13.0)                          After (this FEP)
vertical import ──► persona kernel ──► default registry (None) ──► no-op      (removed)
                                    └► FrameworkPersonaProvider (never built)  (removed)
chain_runtime ──► victor.framework.chains (no readers)                         (removed)
activation ──► step_handlers ──► chain_registry / persona_registry ◄── readers  (unchanged)
```

### Detailed Specification

#### Removed in victor-contracts 0.14.0 (deprecated in 0.13.0)

- The `victor_contracts.chain_runtime` module. Importing it raises `ModuleNotFoundError`, as
  with the 0.12.0 bridge removals.
- `victor_contracts.multi_agent.get_runtime_persona_provider` and its `__all__` entry.

#### Removed in victor-ai (the first release that requires victor-contracts>=0.14.0)

- The `victor/framework/chains/` package, `victor/framework/multi_agent/persona_provider.py`,
  the `FrameworkPersonaProvider` lazy `__getattr__` / `__dir__` entry and
  `PERSONA_PROVIDER_DEPRECATION` in `victor/framework/multi_agent/__init__.py`.
- Their tests: `tests/unit/framework/chains/`,
  `tests/unit/framework/multi_agent/test_persona_provider.py`,
  `tests/unit/framework/test_registry_sdk_interop.py`, and the deprecation cases in
  `tests/unit/framework/test_runtime_adapter_interop.py`.

#### Persona kernel default path: option (b) removes import-time registration

- **0.14.0 deprecates the default path.** `register_vertical_personas(..., provider=None)`
  emits a `DeprecationWarning` that points to `get_personas()`. Passing a non-default
  `RegistrationPolicy.provider_getter` or `auto_register` also warns, from `__post_init__`.
  `set_default_persona_registry` / `get_default_persona_registry` warn because no host writes
  them.
- **0.15.0 or later removes it.** `provider` becomes a required argument, and the two inert
  policy fields and the default-persona-registry hooks are deleted.
- **First-party verticals change in the same PR as the SDK deprecation.** The import-time
  `_register_*_personas()` calls and victor-dataanalysis's `register_data_analysis_personas()`
  are deleted. devops, research and dataanalysis get a `get_personas()` factory in their
  `teams` package. This closes the existing gap, so their personas reach the flat registry
  through activation just as coding's do. The devops alias
  `FrameworkPersonaProvider = PersonaRegistryProtocol` is removed.

### API Changes

```python
# Removed (victor-contracts 0.14.0)
from victor_contracts.chain_runtime import ChainRegistry, get_chain_registry
from victor_contracts.multi_agent import get_runtime_persona_provider
# Removed (victor-ai, release requiring contracts>=0.14.0)
from victor.framework.chains import ChainRegistry
from victor.framework.multi_agent import FrameworkPersonaProvider
# Deprecated in 0.14.0, removed >=0.15.0
register_vertical_personas(personas, policy)          # provider=None
RegistrationPolicy(..., provider_getter=..., auto_register=...)
victor_contracts.set_default_persona_registry / get_default_persona_registry
```

### Dependencies

victor-ai's `victor-contracts` floor moves to `>=0.14.0` in the release that removes the host
modules.

## Benefits

### For Framework Users

- There is one `ChainRegistry` / `get_chain_registry` name in the tree. Today the deprecated
  versioned registry shares those names with the flat public registry, so an import from the
  wrong module type-checks and then has no effect.
- About 920 lines of host code and their tests go away, along with two copies of the semver
  helpers that F-019's clone scan flagged.

### For Vertical Developers

- One documented way to ship chains and personas: return them from the vertical hook, and the
  host registers them, scoped to the vertical, on activation. Import-time side effects no
  longer write into a registry that may or may not exist.
- devops, research and dataanalysis personas become visible to `victor capabilities` and team
  tooling for the first time.

### For the Ecosystem

- The SDK no longer publishes bridges that resolve to dead host code, so external authors stop
  building on surfaces nothing reads.

## Drawbacks and Alternatives

### Drawbacks

- **External code that imported the deprecated surfaces breaks at removal.** Mitigation:
  `DeprecationWarning` since 0.13.0, a changelog entry, and the replacement table in
  `CONTRACT_STABILITY.md`.
- **The persona-kernel cleanup takes two SDK releases (0.14.0, then 0.15.0) instead of one.**
  Mitigation: this is what the policy requires, because `verticals/` and `registries.py` are
  Stable and those fields never warned.

### Alternatives Considered

1. **Option (a): point `set_default_persona_registry` at a flat-registry adapter at bootstrap.**
   - Description: bootstrap installs an adapter that implements `PersonaRegistryProtocol`
     over `persona_registry`, so the existing import-time registration starts doing something.
   - Pros: the vertical code stays the same, and the kernel default path gains a purpose.
   - Cons: (1) **Two writers for the same key.** Import-time registration would write
     `coding:architect` from `to_framework_traits()`, then activation would write it again
     from `to_dict()`. The two `PersonaSpec`s differ, and the adapter drops `version`,
     `category` and `author`, which `PersonaSpec` has no fields for. (2) **Order dependence.**
     A vertical whose `teams` module is imported before bootstrap still gets a no-op, which is
     the same dormancy this FEP removes. (3) **Bypasses activation.** The adapter would skip
     the registry service's fingerprint and `registration_version` checks. It would also
     register personas for verticals that are imported but never activated. (4) Activation
     already registers coding's personas, so the adapter adds nothing there.
   - Why rejected: it brings a dormant path back to life as a second, order-sensitive writer,
     when the fix for the real gap (three verticals with no hook) is a `get_personas()`
     factory.

2. **Remove everything in 0.14.0, including `provider_getter` and the default path.**
   - Pros: one release.
   - Cons: it breaks the "warn for at least one minor release" rule for Stable surfaces that
     never warned.
   - Why rejected: policy compliance.

3. **Keep the versioned registries and make the runtime read them.**
   - Why rejected: no consumer needs versioned lookup, and it would bring back the duplicated
     semver code that F-019 removes.

## Unresolved Questions

- **Should `set/get_default_persona_registry` be removed, or kept as a generic hook for
  third-party hosts?** Proposed: deprecate them in 0.14.0. No in-tree host writes them, and a
  third-party host can pass `provider=` explicitly.
- **Should `register_vertical_personas` survive at all once `provider` is required?**
  Proposed: keep it, because it is still a useful explicit-provider translator for external
  hosts. Revisit if there are still no callers after 0.15.0.
- **Should the dataanalysis constant naming mismatch (`DATA_ANALYSIS_PERSONAS`) be fixed in
  `PersonasHandler`?** Proposed: no. Prefer an explicit `get_personas()` factory over constant
  lookup.

## Implementation Plan

### Phase 1: Deprecation (done)

- [x] Warnings on host modules and SDK bridges; victor-contracts 0.13.0 (PR #1609).

**Deliverable**: deprecated surfaces warn; flat registries unchanged.

### Phase 2a: SDK removal and kernel deprecation (victor-contracts 0.14.0, one PR) — done (#1617, released 0.14.0)

- [x] Replace `chain_runtime.py` with an inert deprecated shim (no host import; deleted in
      0.15.0) and delete `get_runtime_persona_provider`; update `test_lazy.py`,
      `test_runtime_adapter_exports.py` and `test_multi_agent_contracts.py`.
- [x] Add the `provider=None`, `provider_getter`/`auto_register`, and default-registry-hook
      deprecation warnings, with tests.
- [x] In the same PR, update the verticals: drop the import-time calls and
      `register_data_analysis_personas` (the `get_personas()` factories for devops, research
      and dataanalysis already landed in PR #1615), remove the devops alias, and update the vertical persona
      tests (`victor-coding/tests/test_personas.py:127-138`,
      `victor-dataanalysis/tests/test_personas.py:134-136`).
- [x] Update the CHANGELOG `[0.14.0]` and the `CONTRACT_STABILITY.md` "Removed in 0.14.0" /
      "Deprecated" tables.

**Deliverable**: contracts 0.14.0 tagged (`sdk-v0.14.0`) and verified on PyPI.

### Phase 2b: Host removal (victor-ai, after 0.14.0 is on PyPI) — done (victor-ai 0.23.0)

- [x] Delete the host modules, the lazy export and the tests listed above. Raise the floor to
      `victor-contracts>=0.14.0`.
- [x] Update `docs/development/repository-map.md` (it links `victor/framework/chains`) and the
      F-019 backlog entry.

### Phase 3: Kernel removal (victor-contracts 0.15.0 or later)

Prerequisite (victor-ai 0.23.0): the host refuses first-party verticals older than 0.14.0
before importing them (`FIRST_PARTY_MIN_VERTICAL_VERSIONS` in the compatibility gate,
checked in the loader's entry-point preflight; editable installs skipped), and the
vertical extras require `>=0.14.0`. Owner decision 2026-10-30: gate old verticals, then
remove.

- [ ] Make `provider` required, delete `provider_getter`, `auto_register` and the
      default-persona-registry hooks, and update the persona-kernel module docstring.

### Testing Strategy

- Positive: an activation test for each first-party vertical shows that `get_personas()`
  returns a non-empty dict and the flat `PersonaRegistry` contains `<vertical>:<id>` after
  `apply_personas`. Chain activation is already covered (#1611).
- Negative: importing `victor_contracts.chain_runtime` / `victor.framework.chains` raises
  `ModuleNotFoundError`. `get_runtime_persona_provider` is absent. In 0.14.0,
  `register_vertical_personas(p, policy)` warns. Importing a vertical's `teams` package emits
  no `DeprecationWarning`, which proves the import-time call is gone.
- Guard: re-grep `victor/ tests/ verticals/ victor-contracts/` for the removed names before
  merge, and run `pytest tests/ --collect-only -q`.

### Rollout Plan

- No feature flag, because nothing removed is read at runtime. Follow the release-order rule:
  tag the SDK, verify on PyPI, then tag victor-ai.

## Migration Path

### From Old API to New API

1. **Chains**: return them from the vertical's `get_chains()` or `get_composed_chains()`. The
   host registers them as `<vertical>:<name>`. Read them with
   `victor.framework.chain_registry.get_chain_registry()`.
2. **Personas**: return a `{id: persona}` dict from `get_personas()`. A persona exposes
   `to_persona_spec()` or `to_dict()`, or is a `PersonaSpec`. Read with
   `victor.framework.persona_registry.get_persona_registry()`.
3. **Explicit registration (external hosts)**: pass `provider=` to
   `register_vertical_personas`.

```python
# Before
register_vertical_personas(MY_PERSONAS, POLICY)        # import time; silent no-op
# After (in the vertical's teams package)
def get_personas() -> dict[str, VerticalPersona]:
    return MY_PERSONAS
```

### Deprecation Timeline

- victor-contracts 0.13.0 (2026-10-30) and victor-ai `[Unreleased]`: the bridges and host
  modules warn.
- victor-contracts 0.14.0: the bridges are removed, and the kernel default path,
  `provider_getter`/`auto_register` and the default-persona-registry hooks warn.
- The victor-ai release that ships the deprecation and is followed by one requiring
  contracts>=0.14.0: the host modules are removed. The earliest is the minor release after
  the one that ships the `[Unreleased]` deprecation.
- victor-contracts 0.15.0 or later: the kernel default path and its fields are removed.

## Compatibility

### Backward Compatibility

- **Breaking change**: yes, but only for code that imports the deprecated surfaces. Runtime
  behaviour does not change.
- **Migration required**: only for external code that uses the removed names.
- **Deprecation period**: at least one minor release per surface (0.13.0 to 0.14.0 for the
  bridges, 0.14.0 to 0.15.0 for the kernel).

### Version Compatibility

- **victor-contracts 0.14.0 with an older victor-ai**: safe. The older host still has the
  modules, and the SDK no longer bridges to them.
- **The new victor-ai with victor-contracts 0.13.x**: not allowed. Accessing `chain_runtime`
  would fail with `ModuleNotFoundError`, so the floor is `>=0.14.0`.
- Minimum Python version: 3.12, unchanged.

### Vertical Compatibility

- **Built-in verticals**: updated in Phase 2a (see above). Coding's behaviour is unchanged;
  devops, research and dataanalysis gain activation-time personas.
- **External verticals**: anything that relied on import-time registration was already a
  no-op. Add a `get_personas()` hook to make the personas visible.

## References

- Backlog: `docs/architecture/CODEBASE_REVIEW_BACKLOG.md` → **F-019**
- `victor-contracts/CONTRACT_STABILITY.md` (Deprecated tier, Deprecation Policy);
  `victor-contracts/CHANGELOG.md` [0.13.0]; root `CHANGELOG.md` [Unreleased]
- PR #1608 (one namespaced-registry base for chains and personas), PR #1609 (deprecation,
  victor-contracts 0.13.0), PR #1611 (coding chain-activation tests)
- Precedent: [FEP-0015](fep-0015-trim-internal-framework-exports.md) (export trim with a
  deprecation phase)

## Review Process

### Submission

- **Submitted by**: Vijaykumar Singh
- **Date**: 2026-10-30
- **Pull Request**: TBD

### Review Checklist

#### Technical Review

- [ ] Re-grep confirms zero runtime readers at merge time
- [ ] Option (b) agreed; the gap closure for devops, research and dataanalysis is in scope
- [ ] Release order and the contracts floor are agreed

#### Community Review

- [ ] Migration table in `CONTRACT_STABILITY.md` is clear

### Decisions

- **Recommendation**: Accept
- **Decision date**: 2026-10-30
- **Approved by**: Vijaykumar Singh (maintainer)
- **Rationale**: Owner decisions recorded on review:
  1. Personas reach the host only through the activation hook (`get_personas()`), option (b):
     one writer through the registry service, no import-order dependence, and the same model
     chains already use. The bootstrap-adapter and dual-path options were rejected.
  2. The devops/research/dataanalysis persona gap is fixed immediately and separately from the
     removal, by adding `get_personas()` factories (PR #1615), so Phase 2a only deletes code.
  3. The inert Stable kernel surfaces (`provider_getter`, `auto_register`, the `provider=None`
     default path, `set/get_default_persona_registry`) follow CONTRACT_STABILITY to the letter:
     deprecated in 0.14.0, removed no earlier than 0.15.0.
  4. `chain_runtime` is **not** deleted in 0.14.0: the published victor-coding 0.13.0 imports it
     unguarded at module import (`composed_chains.py:734`), so 0.14.0 replaces the bridge with an
     inert, host-free shim (accepts `register_chain`, stores nothing, warns) and deletes it no
     earlier than 0.15.0, once victor-ai requires verticals >= 0.14.0. The published verticals also
     import `get_default_persona_registry` at module load (under 0.14.0 that only warns; the SDK
     CHANGELOG carries an upgrade note), so Phase 3 carries the same constraint.
  5. Delivery: Phase 2a ships in victor-contracts 0.14.0 together with victor-ai 0.22.0 in one
     promotion (SDK tag verified on PyPI before the victor-ai tag); Phase 2b ships in the next
     victor-ai release requiring `victor-contracts>=0.14.0`.

### Revision History

1. **v1.0** (2026-10-30): Initial draft
2. **v1.1** (2026-10-30): Accepted with the owner decisions above; the persona gap moved
   to a standalone fix (PR #1615) ahead of Phase 2a; `chain_runtime` becomes an inert shim in
   0.14.0 (published victor-coding 0.13.0 imports it at module load)

## Acceptance Criteria

### Must-Have Criteria

1. **Single source**: no `victor.framework.chains`, `persona_provider`, `chain_runtime` or
   `get_runtime_persona_provider` remains. Verified by negative import tests and a grep.
2. **No activation regression**: chains and personas for every first-party vertical are in the
   flat registries after activation. Verified by positive activation tests.
3. **Policy-compliant timeline**: each removed Stable surface warned for at least one minor
   release. Verified against the changelogs.

### Should-Have Criteria

1. **No import-time `DeprecationWarning`** from first-party verticals. Priority: High.

### Implementation Requirements

- [ ] Changelog entries (root and victor-contracts) and `CONTRACT_STABILITY.md` tables
- [ ] Repository map and F-019 backlog updated
- [ ] Release-order rule followed (SDK tag verified on PyPI before the victor tag)

---

## Copyright

This FEP is licensed under the Apache License 2.0, same as the Victor project.
