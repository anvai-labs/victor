# Contract Stability Policy

This document classifies every public surface of the `victor_contracts` package
into a maturity tier and defines the deprecation policy that governs changes to
those surfaces. It is the reference for what external vertical and plugin
authors may depend on, and for how long.

The package follows semantic versioning and releases independently of
`victor-ai` via `sdk-v*` tags. See [CHANGELOG.md](CHANGELOG.md) for the release
history.

## Maturity Tiers

| Tier | Meaning | Host runtime (`victor-ai`) required? |
|------|---------|--------------------------------------|
| **Stable** | The definition layer. Safe to build external packages against; changes follow the deprecation policy below. | No |
| **Stable bridge — host required** | Lazy bridges that re-export runtime symbols from an installed `victor-ai` host. The module import itself is host-free; attribute access requires the host. | Yes (at attribute access) |
| **Deprecated** | Surfaces scheduled for removal. Do not use in new code; see the replacement pointers. | — |
| **Internal** | Support surfaces with no semver stability guarantee. May change in any release. | Varies |

### Stable — the definition layer

The contract-first authoring surface. External verticals should import only
from these modules (usually via the top-level `victor_contracts` namespace):

- `core/` — base types, `VerticalBase`, `VerticalDefinition`, stage and
  workflow metadata
- `verticals/` — vertical protocols and validation helpers
- `constants/` — canonical `ToolNames`, `CapabilityIds`, and related registries
- `tools.py` — the SDK Tool Contract (`ToolContract`, trait enums)
- `team_schema.py` — declarative team/formation schema
- `discovery.py` — entry-point discovery (`victor.plugins` group) and the
  global contract registry
- `protocols.py` — extension protocols
- `validation.py` — vertical package validation (backs the
  `victor-contracts check` CLI in `cli.py`)

The remaining definition-layer modules (`capabilities/`, `framework/`,
`providers/`, `safety/`, `skills/`, `conversation.py`, `multi_agent.py`,
`registries.py`, `rl.py`, `runtime_evaluation.py`, `safety_patterns.py`,
`safety_policy.py`, `workflows.py`, `utils/`) follow the same Stable policy.

Some Stable modules expose explicitly named host hooks — for example
`team_schema.get_runtime_team_registry()`,
`multi_agent.get_runtime_persona_provider()`, and
`capabilities.runtime` — which import the `victor-ai` host only when called.
Everything else in these modules works with `victor-contracts` alone.

### Stable bridge — host required

These modules use a `_LAZY_IMPORTS` table plus a module-level `__getattr__`
(PEP 562) to resolve symbols from the installed `victor-ai` host at first
attribute access. They exist so runtime-side code (including the first-party
verticals) can reach host functionality through the `victor_contracts`
namespace without a module-scope dependency on `victor.*`.

Importing a bridge module never requires the host; **accessing any attribute
does**. Without `victor-ai` installed, attribute access raises
`ModuleNotFoundError` for the underlying `victor.*` module.

The 15 supported bridge modules (each verified to have in-tree consumers):

| Bridge module | Resolves against |
|---------------|------------------|
| `capability_runtime` | host capability registry and config helpers |
| `chain_runtime` | host chain execution helpers |
| `database_runtime` | `victor.core.database` (project DB access) |
| `enrichment_runtime` | host enrichment services |
| `feature_flag_runtime` | host feature flags |
| `graph_rag_runtime` | host graph-RAG services |
| `indexing_runtime` | host code-index services |
| `init_runtime` | host initialization helpers |
| `lsp_runtime` | host LSP integration |
| `middleware_runtime` | host middleware surfaces |
| `processing_runtime` | host processing/chunking services |
| `provider_runtime` | `victor.providers` adapters |
| `rl_runtime` | host RL data services |
| `search_runtime` | host search services |
| `workflow_runtime` | host workflow definition/executor surfaces |

### Deprecated

The following bridge modules have **zero consumers** in the Victor monorepo
(core, first-party verticals, and examples) and duplicate surfaces available
elsewhere. They are classified Deprecated: warnings begin with the 0.9.0
release, and removal happens **no earlier than 0.10.0**:

| Deprecated module | Replacement |
|-------------------|-------------|
| `agent_spec_runtime` | Import from `victor.agent.specs.models` in host-runtime code (runtime-internal surface; not a contract) |
| `graph_runtime` | `victor.framework.graph` (`StateGraph`, `END`) — the framework public API |
| `handler_runtime` | `victor_contracts.workflow_runtime` for `register_compute_handler`; `victor.framework.handler_registry` (host) for registry helpers |
| `subagent_runtime` | Import from `victor.agent.subagents.protocols` in host-runtime code |
| `tool_runtime` | `victor.framework.tools` (`RuntimeToolSet`) |
| `workflow_executor_runtime` | `victor_contracts.workflow_runtime` (same surface, resolved against the current host workflow modules) |

> **Status note:** as of 0.9.0 these six modules emit `DeprecationWarning` on
> attribute access while continuing to resolve host symbols as before.

### Internal

No semver stability guarantee; may change in any release:

- `testing/` — drop-in contract test helpers for vertical test suites.
  Useful, but the helper API itself is best-effort and may change between
  minor releases.
- `victor_sdk` — the legacy alias namespace from before the
  `victor_sdk` → `victor_contracts` rename. It already emits a
  `DeprecationWarning` on import and exists only so pinned external plugins
  keep working. New code must import from `victor_contracts`.

## Deprecation Policy

1. **Warn before removal.** A deprecated surface emits `DeprecationWarning`
   (while continuing to work) for **at least one minor release** before it is
   removed.
2. **Removals only in minor bumps.** Surfaces are removed only in minor
   version bumps (`0.x.0`), never in patch releases.
3. **Announced in the changelog.** Every deprecation and every removal is
   recorded in [CHANGELOG.md](CHANGELOG.md), including the replacement path
   and the earliest removal version.

Example: a module deprecated in 0.9.0 can be removed in 0.10.0 at the
earliest, and the 0.9.0 and 0.10.0 changelog entries must both mention it.

## See Also

- [README.md](README.md) — package overview and the stable authoring surface
- [VERTICAL_DEVELOPMENT.md](VERTICAL_DEVELOPMENT.md) — vertical authoring guide
- [MIGRATION.md](MIGRATION.md) — migration guidance for older import paths
- [CHANGELOG.md](CHANGELOG.md) — release history
