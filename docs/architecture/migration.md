# Victor Architecture Consolidation Migration Guide

**Status**: Historical consolidation guide with some still-useful framework examples
**Last Reviewed**: 2026-05-04
**Authoritative current agent runtime doc**: [System architecture](../architecture.md)

> Historical note: this guide mixes framework-side `StateGraph` migration
> examples with older `victor/agent` rollout guidance. Use it for background
> and selected framework patterns, not as the current source of truth for agent
> runtime ownership.

## Overview

Victor has undergone a significant architectural consolidation across the
framework and agent runtime layers. This guide captures the original migration
direction and still-useful examples, but some sections are now historical.

## What Changed

### Before: Two Parallel Architectures

1. **Framework Layer** (`victor/framework/`): Deterministic primitives
   - StateGraph, WorkflowEngine, Teams
   - Used by: Advanced users, workflow authors

2. **CLI/Chat Layer** (`victor/agent/`): Service-oriented, custom loops
   - AgentOrchestrator, AgenticLoop
   - Used by: CLI chat, API server

### After: Unified Architecture

- **Framework as foundation**: Workflow and team execution increasingly center on `StateGraph`
- **CLI/Chat runtime as service-first**: `AgentOrchestrator` composes canonical services
- **Service injection**: `ExecutionContext` remains useful for framework-side graph integrations
- **Team formations**: 5 formations (SEQUENTIAL, PARALLEL, HIERARCHICAL, PIPELINE, CONSENSUS)

## Migration Paths

### 1. AgenticLoop Users

**Before** (Custom while loop - now legacy):
```python
from victor.framework.agentic_loop import AgenticLoop

loop = AgenticLoop(orchestrator)
result = await loop.run("Write tests")
```

**After** (Optional framework-side `StateGraph` path during parity/benchmark work):
```python
# Enable the framework-side StateGraph path when explicitly validating it
export VICTOR_USE_STATEGRAPH_AGENTIC_LOOP=true

# Same API - no code changes needed
from victor.framework.agentic_loop import AgenticLoop

loop = AgenticLoop(orchestrator)
result = await loop.run("Write tests")
```

### 2. Coordinator Users

**Before** (Direct coordinator instantiation):
```python
from victor.agent.coordinators.exploration_state_passed import ExplorationStatePassedCoordinator

coordinator = ExplorationStatePassedCoordinator()
result = await coordinator.explore(context_snapshot, user_message="Find files")
```

**After** (StateGraph node):
```python
from victor.framework.agentic_graph.state import create_initial_state
from victor.framework.agentic_graph.coordinator_nodes import exploration_node

state = create_initial_state(query="Find files")
result = await exploration_node(state, exploration_coordinator=coordinator)
```

### 3. Service Access

**Current canonical runtime path**:
```python
chat_service = orchestrator.chat_service
response = await chat_service.chat(message="Hello")
```

**Framework-side graph integration option**:
```python
from victor.framework.agentic_graph.state import create_initial_state
from victor.framework.agentic_graph.service_nodes import chat_service_node

state = create_initial_state(query="Hello")
result = await chat_service_node(state, message="Hello")
response = result.context.get("last_response")
```

`AgenticLoopStateModel` is the canonical state type for these graph-facing
helpers. Passing a raw `dict` remains a compatibility path for direct graph
invocation, but new code should prefer `create_initial_state(...)` or an
explicit `AgenticLoopStateModel(...)`.

### 4. Team Formation Users

**Before** (Direct coordinator usage):
```python
from victor.teams import UnifiedTeamCoordinator, TeamFormation

coordinator = UnifiedTeamCoordinator(orchestrator)
coordinator.set_formation(TeamFormation.PARALLEL)
coordinator.add_member(agent1).add_member(agent2)
result = await coordinator.execute_task("Task", {})
```

**After** (Use `UnifiedTeamCoordinator` directly as a `StateGraph` step):
```python
from victor.framework import StateGraph
from victor.framework.agentic_graph.team_selector import select_formation
from victor.teams import (
    StateGraphNodeConfig,
    TeamFormation,
    UnifiedTeamCoordinator,
)

# Create coordinator
coordinator = UnifiedTeamCoordinator(orchestrator)
coordinator.set_formation(TeamFormation.PARALLEL)
coordinator.add_member(agent1).add_member(agent2)
coordinator.with_state_graph_config(
    StateGraphNodeConfig(formation_strategy=select_formation)
)

# Use directly in StateGraph
graph = StateGraph(AgentState)
graph.add_node("parallel_research", coordinator)
graph.add_edge("analyze", "parallel_research")
graph.add_edge("parallel_research", "synthesize")
```

**Why this layering matters:**
```python
# StateGraph owns control flow between steps.
# UnifiedTeamCoordinator owns collaboration inside one step.
# TeamStep is only the declarative workflow adapter surface.
```

## Terminology

- **Team**: a coordinated group of agents under one formation.
- **Formation**: the intra-team collaboration mode (`PARALLEL`, `HIERARCHICAL`, etc.).
- **Step**: the workflow surface that invokes a capability from the graph.
- **Node**: a generic graph runtime concept owned by `StateGraph`.

Prefer the term **team step** or **team invocation** for declarative workflow
surfaces. `UnifiedTeamCoordinator` is the primary runtime abstraction.

## Breaking Changes

### Historical Planning Note

The compatibility and removal targets below reflect the migration plan at the
time this guide was drafted. Verify current deprecations and ownership against
[System architecture](../architecture.md) and the seam-by-seam migration audit
before using them for new work.

### None During Migration

All changes are backward compatible. Compatibility aliases remain available
during migration, but legacy `TeamNode*` names now emit `DeprecationWarning`
to support a clean removal path in `v0.9.0` (`2027-03-31`).

### Future Deprecations

The following will be deprecated in future releases:

1. **Direct while-loop execution** in AgenticLoop
2. **Workflow-only wrappers as the primary team API**
   Use `UnifiedTeamCoordinator` directly for programmatic graphs.
3. **Global service access** (use ExecutionContext)
4. **Legacy `TeamNode*` workflow names**
   Use `TeamStep*` names during the deprecation window. Removal target:
   `v0.9.0` (`2027-03-31`).
5. **Legacy `WorkflowGraph` import from `victor.workflows.graph`**
   Use `BasicWorkflowGraph` for the simple container or
   `victor.workflows.graph_dsl.WorkflowGraph` for the typed DSL. Removal
   target: `v0.8.0` (`2026-12-31`).
6. **Legacy `FrameworkShim` compatibility surface**
   Use `Agent.create()` for public callers or `AgentFactory` /
   `AgentCreationFactory` for internal composition. The legacy
   `victor chat --legacy` path has been removed; chat now always uses the
   canonical framework client path. Removal target: `v1.0.0`
   (`2027-06-30`).

## Performance

Expected performance impact:
- **<5% overhead** from StateGraph execution
- **No degradation** in streaming performance
- **Improved** parallel team execution

## Testing

Use this when explicitly validating the framework-side `StateGraph`
agentic-loop path. It is not required for the default service-first
`victor/agent` runtime.

```bash
# Enable the optional StateGraph path
export VICTOR_USE_STATEGRAPH_AGENTIC_LOOP=true

# Run tests
pytest tests/
```

## Getting Help

- GitHub Issues: https://github.com/your-org/victor/issues
- Documentation: `CLAUDE.md` in repository
- Migration support: `migration.md` (this file)
