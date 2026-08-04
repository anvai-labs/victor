# Deprecation Inventory (2026-03-03)

> **Historical inventory.** The bundled-contrib vertical entries below describe
> the March 2026 transition and do not identify current runtime packages. The
> authoritative vertical ownership model is [ADR-007](../architecture/adr/007-vertical-distribution-and-sdk-boundary.md): first-party domain
> verticals are authored once under the top-level `verticals/` packages and
> published from there. Do not use historical `victor/verticals/contrib/...`
> paths in this document as implementation guidance.

Owner: Verticals Lead (program owner), with role-level ownership per entry.

## Inventory

| Deprecated API/Symbol | Source Location | Replacement | Owner Role | Target Removal Version | Target Removal Date |
|---|---|---|---|---|---|
| `UnifiedWorkflowCompilerAdapter` legacy API surface | `victor/workflows/adapter.py` | Protocol-based compiler API (`victor.workflows.compiler`) | Architecture Lead | `v0.7.0` | `2026-06-30` |
| ~~`UnifiedWorkflowCompiler`~~ | `victor/workflows/unified_compiler.py` | **Removed from deprecation** — canonical compiler API; `create_compiler` was never implemented. See M4 update below. | Architecture Lead | N/A | N/A |
| `FrameworkShim` compatibility surface | `victor/framework/shim.py`, `victor/framework/__init__.py` | `Agent.create()` for public callers or `AgentFactory` / `AgentCreationFactory` for internal composition | Architecture Lead | `v1.0.0` | `2027-06-30` |
| `WorkflowGraph` alias from `victor.workflows.graph` | `victor/workflows/graph.py` | `BasicWorkflowGraph` or `victor.workflows.graph_dsl.WorkflowGraph` | Architecture Lead | `v0.8.0` | `2026-12-31` |
| `TeamNode*` workflow compatibility aliases (`TeamNode`, `TeamNodeConfig`, `TeamNodeWorkflow`, `TeamNodeExecutor`) | `victor/framework/workflows/nodes.py`, `victor/workflows/{__init__,definition.py}`, `victor/workflows/executors/{__init__,team.py}` | `TeamStep*` workflow names | Architecture Lead | `v0.9.0` | `2027-03-31` |
| Sync `AgentOrchestrator.switch_provider(...)` | `victor/agent/orchestrator.py` | Async `await orchestrator.switch_provider(...)` | Architecture Lead | `v0.8.0` | `2026-12-31` |
| `get_tiered_tools()` extension hook | `victor/core/verticals/extension_loader.py` | `get_tiered_tool_config()` | Verticals Lead | `v0.7.0` | `2026-06-30` |
| `TASK_TYPE_HINTS` and deprecated task-hint fallback path | `victor/agent/prompt_builder.py` | Vertical prompt contributors + `get_task_type_hint(..., prompt_contributors=[...])` | Architecture Lead | `v0.8.0` | `2026-12-31` |
| `CodingToolDependencyProvider` + deprecated module constants | `victor/verticals/contrib/coding/tool_dependencies.py` | `create_vertical_tool_dependency_provider("coding")` | Verticals Lead | `v0.8.0` | `2026-12-31` |
| `DataAnalysisToolDependencyProvider` + deprecated module constants | `victor/verticals/contrib/dataanalysis/tool_dependencies.py` | `create_vertical_tool_dependency_provider("dataanalysis")` | Verticals Lead | `v0.8.0` | `2026-12-31` |
| `ResearchToolDependencyProvider` + legacy constants | `victor/verticals/contrib/research/tool_dependencies.py` | `create_vertical_tool_dependency_provider("research")` | Verticals Lead | `v0.8.0` | `2026-12-31` |
| Deprecated DevOps dependency constants | `victor/verticals/contrib/devops/tool_dependencies.py` | `create_vertical_tool_dependency_provider("devops")` | Verticals Lead | `v0.8.0` | `2026-12-31` |
| `CodingTeamSpec` | `victor/verticals/contrib/coding/teams/specs.py` | `TeamSpec` from `victor.framework.team_schema` | Verticals Lead | `v0.8.0` | `2026-12-31` |
| `ResearchTeamSpec` | `victor/verticals/contrib/research/teams/__init__.py` | `TeamSpec` from `victor.framework.team_schema` | Verticals Lead | `v0.8.0` | `2026-12-31` |
| Fragmented/legacy event type names (module-level compatibility) | `victor/core/events/taxonomy.py` | `UnifiedEventType` values from `victor.core.events.taxonomy` | Observability Lead | `v0.8.0` | `2026-12-31` |
| `VerticalBase.create_agent()` convenience factory | `victor/core/verticals/base.py` | `Agent.create(vertical=MyVertical, ...)` | Verticals Lead | `v0.8.0` | `2026-12-31` |

## Notes

- Entries with explicit source-level targets were kept as-is (`v0.7.0` / `2026-06-30`).
- Entries without explicit source targets are assigned provisional `v0.8.0` / `2026-12-31` per policy and should be re-validated at each milestone cut.
- `FrameworkShim` remains a broad compatibility surface, so its removal is tracked
  on the first post-`0.x` stabilization milestone: `v1.0.0` / `2027-06-30`.
- `TeamNode*` workflow aliases were deprecated after `0.7.0` and therefore use
  the next compliant two-minor-release milestone: `v0.9.0` / `2027-03-31`.
- `VerticalBase.create_agent()` and the legacy config-only vertical activation shim
  share the same provisional removal milestone: `v0.8.0` / `2026-12-31`.
- Until that removal milestone lands, each release note set must include:
  - the replacement path `Agent.create(vertical=MyVertical, ...)`
  - confirmation that legacy config-only verticals still run through the runtime shim
  - a link to `victor-contracts/MIGRATION_GUIDE.md`

## M2 Update (2026-03-04)

Removed in `Unreleased` as part of deprecation-shim burn-down:

- `ResearchToolDependencyProvider` + legacy `RESEARCH_*` constants
- Deprecated DevOps `DEVOPS_*` constant exports + wrapper provider class
- `DataAnalysisToolDependencyProvider` + legacy `DATA_ANALYSIS_*` constants

Migration path for all three verticals:

```python
from victor.verticals.contrib.devops.tool_dependencies import get_provider
provider = get_provider()
```

## M3 Update (2026-03-10)

Removed 6 more items (9/13 total = 69%, exceeds 60% target):

1. **`UnifiedWorkflowCompilerAdapter` + `CompiledGraphAdapter` + `ExecutorResultAdapter`** (`victor/workflows/adapter.py`)
   - Module now raises `ImportError` with migration message
   - Migration: `victor.workflows.compiler_protocols.WorkflowCompilerProtocol` via DI container

2. **`get_tiered_tools()` extension hook** (`victor/core/verticals/extension_loader.py`)
   - Method and protocol declaration removed
   - Migration: override `get_tiered_tool_config()` instead

3. **`CodingToolDependencyProvider` class + 6 deprecated constants + `_LazyDeprecatedProperty` infra** (`victor/verticals/contrib/coding/tool_dependencies.py`)
   - Only `get_provider()` factory remains
   - Migration: `from victor.verticals.contrib.coding.tool_dependencies import get_provider`

4. **`CodingTeamSpec` class + `_FORMATION_TO_TOPOLOGY`** (`victor/verticals/contrib/coding/teams/specs.py`)
   - Migration: `TeamSpec` from `victor.framework.team_schema`

5. **`ResearchTeamSpec` class** (`victor/verticals/contrib/research/teams/__init__.py`)
   - Migration: `TeamSpec` from `victor.framework.team_schema`

6. **`TASK_TYPE_HINTS` fallback dict + `__getattr__`** (`victor/agent/prompt_builder.py`)
   - Migration: `get_task_type_hint(task_type, prompt_contributors=[...])` with vertical contributors

**Remaining** (3 items, deferred to v0.8.0):
- Sync `switch_provider()`
- `WorkflowGraph` alias
- Fragmented event type names

## Architecture Strengthening Update (2026-03-15)

Added to inventory:

| Deprecated API/Symbol | Replacement | Target Removal | Status |
|---|---|---|---|
| `victor.verticals.contrib.coding` | `victor-coding` package | `v0.7.0` | DeprecationWarning active |
| `victor.verticals.contrib.rag` | `victor-rag` package | `v0.7.0` | DeprecationWarning active |
| `victor.verticals.contrib.devops` | `victor-devops` package | `v0.7.0` | DeprecationWarning active |
| `victor.verticals.contrib.dataanalysis` | `victor-dataanalysis` package | `v0.7.0` | DeprecationWarning active |
| `victor.verticals.contrib.research` | `victor-research` package | `v0.7.0` | DeprecationWarning active |
| Settings flat-field access (e.g. `settings.default_provider`) | Nested groups (e.g. `settings.provider.default_provider`) | `v0.8.0` | DeprecationWarning active |
| `VerticalBase.create_agent()` | `Agent.create(vertical=MyVertical, ...)` | `v0.8.0` | DeprecationWarning active |

**Updated inventory**: 9/13 original items removed (69%). 7 new deprecations added with warnings active.
**E5 status**: Migration-note closure at 69% for original items. New contrib deprecations are fully documented with v0.7.0 target.

## M4 Update (2026-04-10)

**`UnifiedWorkflowCompiler` removed from deprecation inventory.**

The deprecation warning pointed to `create_compiler("yaml://", ...)` from `victor.workflows.create` — a module
that was never implemented. Analysis showed:

- `UnifiedWorkflowCompiler` is the **canonical compiler API**, used by the framework layer
  (`WorkflowEngine`, `YAMLCoordinator`, `BaseYAMLProvider`, `WorkflowScheduler`).
- The DI-facing `WorkflowCompiler` (`victor.workflows.compiler`) is a narrower, compile-only
  alternative — not a replacement. It lacks caching, multi-source support, and execution APIs.
- The `WorkflowCompilerRegistry` plugin infrastructure exists for third-party backends
  but the URI-based factory dispatcher was never built.

Actions taken:
1. Removed `warnings.warn()` and `.. deprecated::` from `UnifiedWorkflowCompiler.__init__`
2. Updated docstrings in `yaml_coordinator.py`, `base_yaml_provider.py`, `workflow_engine.py`,
   `scheduler.py`, and `bootstrap.py` to reflect canonical status
3. Removed from deprecation inventory table
