# ADR-024: Abstraction Canonicalization and Import-Time Boundary Guard

## Metadata

- **Status**: Proposed
- **Date**: 2026-07-29
- **Decision Makers**: Vijaykumar Singh
- **Related ADRs**: 007 (contracts boundary — the layering this hardens), 002 (state management),
  019 (decomposition — a smaller runtime makes canonicalization tractable)
- **Work tracked by**: [TD-26](../../tech-stack.md#technical-debt-register)
- **Benchmark**: [competitive-benchmark-2026-07.md](../competitive-benchmark-2026-07.md) §4 (scoped state)

## Context

Two review findings, one principle:

1. **Duplicated abstractions per concern.** Responsibility for the same job is spread across several
   near-synonymous surfaces, so "which one is canonical?" is unclear at import time:
   - *Provider construction/lookup*: `victor/providers/factory.py` + `victor/providers/registry.py`
     + the runtime `ProviderService` + the orchestrator's `ProviderManager` component.
   - *State*: `GlobalStateManager` (`state/global_state_manager.py`) + `state/managers.py` +
     `state/factory.py`.
   - *Caching*: `cache_manager.py` + `query_cache.py` + `embedding_cache_manager.py`.
   Some of this is legitimate layering (a registry vs a service vs a facade); some is genuine
   duplication. Today nothing declares which surface new code should import.

2. **Boundaries are enforced post-hoc.** The architectural guards
   (`test_architectural_boundaries.py`, `test_core_vertical_import_boundary.py`,
   `test_service_layer_validation.py`) parse the AST *at test time*. A developer can violate a
   boundary locally and only learn at CI — after the bad import has spread.

## Decision

1. **One canonical surface per concern, declared and documented.** For each of provider
   construction, state management, and caching: name the single canonical entry point, document the
   role of every other surface (or delete it), and record the mapping in `architecture.md`'s
   Additional Subsystems section. Where two surfaces are genuinely one, collapse them; where they are
   distinct layers, the doc says so and points at the canonical one to import.
2. **Import-time boundary guard.** Add an opt-in import hook (a `sys.meta_path` finder or an
   equivalent pytest/conftest import assertion) that fails *at import* when a forbidden cross-layer
   import is executed — turning the existing AST rules into a fail-fast signal in dev, with the
   post-hoc AST tests retained as the CI backstop. The guard reuses the boundary rules the AST tests
   already encode (single source of truth for the rules).

This is a *canonicalization + enforcement* decision, not a rewrite: no behavior changes, only which
surface is authoritative and when violations surface.

## Rationale

- **First principles.** An abstraction earns its keep only if there is exactly one obvious way to
  reach a capability. Three registries for one concern is negative leverage — every reader pays the
  "which one?" tax and imports drift.
- **Fail fast.** A boundary caught at import is a boundary the developer fixes in seconds; one caught
  at CI is one that has already propagated. Same rules, earlier signal.
- **Co-design.** The import guard consumes the *same* rule set the AST tests use, so doc, dev-time
  guard, and CI guard cannot disagree.

## Consequences

- **Positive**: unambiguous imports; drift caught in the editor, not at CI; a cleaner target for
  ADR-019's decomposition (fewer places provider/state logic can hide).
- **Negative**: the import hook must be cheap and must not break lazy-import startup performance
  (the CLI relies on lazy heavy imports); mis-scoped rules could raise false positives — hence
  opt-in in dev, AST tests remain the authority in CI.
- **Neutral**: public APIs unchanged; verticals already constrained to `victor_contracts`.

## Implementation

Internal — **no companion FEP** (no public-API change):

1. Inventory each concern's surfaces; write the canonical-surface table into `architecture.md`;
   collapse true duplicates, document true layers.
2. Factor the boundary rules into a single shared rule module consumed by both the AST tests and a
   new import-time guard (conftest/`sys.meta_path`), opt-in via an env flag for dev.
3. Add a hygiene assertion that new code imports the canonical surface (extend
   `repo_hygiene_check.py` where cheap).

## Alternatives Considered

- **Leave duplication, rely on convention.** Rejected: convention is exactly what drifted here.
- **Hard import-time enforcement always-on in CI too.** Deferred: risk to lazy-import startup and
  false positives; keep AST tests as the CI authority, guard as the dev fast-path.
- **Full DI container.** Rejected: heavier than the problem; the six services + `ExecutionContext`
  already provide the seam — this ADR just makes the entry points singular and enforced.

## References

- [ADR-002](002-state-management.md), [ADR-007](007-vertical-distribution-and-sdk-boundary.md),
  [ADR-019](019-orchestrator-service-runtime-decomposition.md)
- `victor/providers/{factory,registry}.py`, `victor/state/{global_state_manager,managers,factory}.py`,
  `tests/unit/framework/test_architectural_boundaries.py`,
  `tests/unit/contracts/test_core_vertical_import_boundary.py`

## Revision History

| Date | Version | Changes | Author |
|------|---------|---------|--------|
| 2026-07-29 | 1.0 | Initial ADR — canonical surface per concern + import-time boundary guard | Vijaykumar Singh |
