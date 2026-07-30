# ADR-007: Vertical Distribution Model and Contracts Boundary

## Metadata

- **Status**: Accepted (2026-07-02 — contract boundary CI-guarded, first-party verticals folded into monorepo under `verticals/`; was Proposed)
- **Date**: 2026-03-10
- **Decision Makers**: Vijaykumar Singh
- **Related ADRs**: ADR 001, ADR 004

## Context

Victor currently operates with a mixed vertical architecture:

1. `victor-ai` bundles contrib verticals directly in the main package and publishes
   them through the `victor.verticals` entry-point group.
2. The import resolver still prefers external wheel namespaces such as
   `victor_coding`, `victor_research`, and `victor_devops` before falling back to
   bundled contrib modules.
3. `victor-contracts` documents a zero-runtime-dependency vertical model, but current
   bundled verticals and the external vertical example still import framework/core
   implementation modules.
4. `victor.core.verticals.base.VerticalBase` still includes runtime creation logic
   via `create_agent()`, which imports `victor.framework.Agent`.

This creates architectural ambiguity in four places:

- **Source of truth ambiguity**: bundled contrib code vs. external wheel code
- **Contract ambiguity**: contract-only authoring vs. framework-coupled authoring
- **Packaging ambiguity**: bundled-by-default vs. independently released verticals
- **Runtime ambiguity**: declarative vertical definition vs. host-runtime-aware plugins

The current hybrid model increases drift risk, complicates documentation, and makes
it difficult to add guardrails such as forbidden-import checks or packaging CI.

## Decision

Victor should target an **contract-first extracted vertical architecture** with a single
authoritative implementation source for each vertical.

The target model is:

1. **`victor-contracts` owns stable contracts** for vertical definitions.
   This includes base protocols, shared types, canonical tool identifiers,
   capability identifiers, and serializable vertical definition structures.
2. **`victor-ai` owns runtime orchestration only**.
   Core/framework code loads, validates, and applies vertical definitions, but does
   not remain the long-term source of truth for domain vertical implementations.
3. **Each vertical has one authoritative package**.
   Bundled contrib copies must not coexist as peer implementations once extraction is
   complete. During transition, bundled contrib code may remain only as compatibility
   shims or temporary migration adapters.
4. **Vertical definition code must be contract-only**.
   Definition-layer modules such as `assistant.py` and package templates must not
   import `victor.framework`, `victor.core.verticals`, or framework tool registries.
5. **Runtime add-ons are separated from definition contracts**.
   Middleware, teams, workflows, safety adapters, enrichment services, and similar
   runtime-rich integrations may depend on `victor-ai`, but they must sit behind
   explicit extension points instead of being required by the definition layer.
6. **Runtime creation moves out of `VerticalBase`**.
   `VerticalBase.create_agent()` should be deprecated in favor of a host-owned
   factory/adapter layer.

Until this ADR is accepted, the working assumption for planning and documentation is
contract-first extraction, but no breaking packaging removals should be made without an
explicit acceptance step.

## Rationale

### Why this direction

- It resolves the current split-brain problem between bundled contrib modules and
  external package discovery paths.
- It aligns the implementation with `victor-contracts`'s published contract and examples.
- It makes dependency direction explicit: vertical definitions depend on the SDK,
  while the runtime depends on definitions and adapters.
- It gives the project a clean place to enforce import-boundary checks and packaging
  compatibility rules.

### Pros

- Clear single source of truth per vertical
- Stronger dependency inversion
- Cleaner external extension story
- Better fit for independent versioning and release cadence
- Easier CI validation for packaging, examples, and compatibility

### Cons

- More packaging and release coordination
- Temporary migration cost for existing contrib modules and tests
- Requires a deliberate split between declarative definition code and runtime add-ons

## Consequences

- **Positive**:
  - The architecture gets a single target state instead of competing narratives.
  - `victor-contracts` can become the actual public authoring surface, not just a partial one.
  - Future work on tool names, capabilities, examples, and templates can converge on
    one contract.
- **Negative**:
  - Existing bundled contrib code will need extraction or shim treatment.
  - Examples and documentation that currently teach framework-coupled authoring will
    need coordinated updates.
  - Some current convenience APIs, especially `VerticalBase.create_agent()`, will
    require deprecation planning.
- **Neutral**:
  - Runtime discovery remains lazy and cached.
  - Existing vertical protocols and extension loading infrastructure remain useful.

## Implementation

Implementation is tracked in:

- `docs/roadmap/vertical-platform-convergence-plan.md`

High-level phases:

1. Publish the architecture decision and transition rules.
2. Complete the SDK contract surface.
3. Move runtime creation and runtime-heavy helpers behind host-owned adapters.
4. Migrate definition-layer vertical modules to contract-only imports.
5. Converge packaging on one authoritative implementation per vertical.
6. Add guardrails, smoke tests, and compatibility checks.

## Alternatives Considered

### 1. Keep bundled contrib verticals as the permanent end state

Rejected for this ADR because it does not match the current contract-first ecosystem
story and leaves the import resolver, examples, and external package management in
an incoherent state unless extraction-oriented paths are removed.

### 2. Continue the hybrid model indefinitely

Rejected because it preserves the current ambiguity and guarantees drift between
documentation, examples, packaging, and runtime resolution behavior.

### 3. Extract only some verticals and permanently bundle others

Deferred. This could be reconsidered later, but only after a contract boundary is
fully defined and the project has clear rules for which vertical classes qualify for
bundling vs. extraction.

## References

- `pyproject.toml`
- `victor/core/verticals/base.py`
- `victor/core/verticals/import_resolver.py`
- `victor-contracts/README.md`
- `examples/external_vertical/src/victor_security/assistant.py`
- `docs/roadmap/vertical-platform-convergence-plan.md`

## Revision History

| Date | Version | Changes | Author |
|------|---------|---------|--------|
| 2026-03-10 | 1.0 | Initial ADR draft | Codex |
