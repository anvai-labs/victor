# Multi-Agent System Migration Guide

## Overview

This guide helps you migrate your code from the old multi-agent system APIs to the consolidated, unified architecture.

**Consolidation Status**: Phases 1-3 Complete ✅
**Breaking Changes**: None - Full backward compatibility maintained
**Deprecations**: FrameworkTeamCoordinator is deprecated

## Migration Summary

| Old API | New API | Status |
|---------|---------|--------|
| `FrameworkTeamCoordinator` | `create_coordinator(lightweight=True)` | Deprecated (wrapper provided) |
| `victor.framework.team_coordinator` | `victor.teams` | Use new import path |
| Multiple coordinator implementations | Single `UnifiedTeamCoordinator` | Consolidated |
| Split type definitions | `victor.teams.types` | Unified |

## Quick Reference

### Before (Deprecated)

```python
from victor.framework.team_coordinator import FrameworkTeamCoordinator

coordinator = FrameworkTeamCoordinator()
coordinator.add_member(researcher).add_member(executor)
result = await coordinator.execute_task("Build feature", {})
```

### After (Recommended)

```python
from victor.teams import create_coordinator, TeamFormation

coordinator = create_coordinator(lightweight=True)
coordinator.add_member(researcher).add_member(executor)
result = await coordinator.execute_task("Build feature", {})
```

## Detailed Migration Guide

### 1. Import Changes

#### Old Import Paths
```python
# Old: Framework-specific imports
from victor.framework.team_coordinator import FrameworkTeamCoordinator
from victor.framework.agent_protocols import ITeamMember

# Old: Agent-level imports
from victor.agent.teams.coordinator import TeamCoordinator
```

#### New Import Paths
```python
# New: Unified imports
from victor.teams import (
    create_coordinator,
    TeamFormation,
    ITeamCoordinator,
    ITeamMember,
    UnifiedTeamCoordinator,
)

# New: Canonical protocols
from victor.protocols.team import ITeamMember, ITeamCoordinator
```

### 2. Coordinator Creation

#### FrameworkTeamCoordinator → UnifiedTeamCoordinator

**Old Code** (Deprecated):
```python
from victor.framework.team_coordinator import FrameworkTeamCoordinator

# Default coordinator
coordinator = FrameworkTeamCoordinator()
coordinator.add_member(agent1).add_member(agent2)
```

**New Code** (Recommended):
```python
from victor.teams import create_coordinator

# Lightweight coordinator (equivalent to old FrameworkTeamCoordinator)
coordinator = create_coordinator(lightweight=True)
coordinator.add_member(agent1).add_member(agent2)
```

**New Code** (Production):
```python
from victor.teams import create_coordinator

# Production coordinator with full features
coordinator = create_coordinator(
    orchestrator=my_orchestrator,
    enable_observability=True,
    enable_rl=True,
)
coordinator.add_member(agent1).add_member(agent2)
```

### 3. Formation Configuration

Formations are now configured using the `TeamFormation` enum:

```python
from victor.teams import TeamFormation, create_coordinator

coordinator = create_coordinator(lightweight=True)

# Set formation
coordinator.set_formation(TeamFormation.SEQUENTIAL)
coordinator.set_formation(TeamFormation.PARALLEL)
coordinator.set_formation(TeamFormation.HIERARCHICAL)
coordinator.set_formation(TeamFormation.PIPELINE)
coordinator.set_formation(TeamFormation.CONSENSUS)
```

### 4. Team Member Creation

Team members must implement the `ITeamMember` protocol:

```python
from victor.protocols.team import ITeamMember
from dataclasses import dataclass

@dataclass
class MyTeamMember:
    """Custom team member implementation."""

    id: str

    async def execute_task(self, task: str, context: dict) -> str:
        """Execute a task."""
        return f"Task completed: {task}"

    async def receive_message(self, message) -> Optional[AgentMessage]:
        """Receive and optionally respond to messages."""
        # Implement message handling
        pass
```

### 5. Context Passing

Context is now passed via `TeamContext` with `shared_state`:

```python
from victor.coordination.formations.base import TeamContext

# Create context with shared state
context = TeamContext(
    team_id="my_team",
    formation="sequential",
    shared_state={
        "project_root": "/path/to/project",
        "previous_output": None,
    },
)

# Execute with context
result = await coordinator.execute_task("Build feature", context.shared_state)
```

## Migration Examples

### Example 1: Sequential Team

**Before**:
```python
from victor.framework.team_coordinator import FrameworkTeamCoordinator

coordinator = FrameworkTeamCoordinator()
coordinator.add_member(researcher)
coordinator.add_member(executor)
coordinator.add_member(reviewer)
result = await coordinator.execute_task("Add authentication", {})
```

**After**:
```python
from victor.teams import create_coordinator, TeamFormation

coordinator = create_coordinator(lightweight=True)
coordinator.add_member(researcher)
coordinator.add_member(executor)
coordinator.add_member(reviewer)
coordinator.set_formation(TeamFormation.SEQUENTIAL)  # Explicit formation
result = await coordinator.execute_task("Add authentication", {})
```

### Example 2: Pipeline Team

**Before**:
```python
from victor.framework.team_coordinator import FrameworkTeamCoordinator

coordinator = FrameworkTeamCoordinator()
coordinator.add_member(researcher)
coordinator.add_member(executor)
# Implicit pipeline formation
result = await coordinator.execute_task("Implement feature", {})
```

**After**:
```python
from victor.teams import create_coordinator, TeamFormation

coordinator = create_coordinator(lightweight=True)
coordinator.add_member(researcher)
coordinator.add_member(executor)
coordinator.set_formation(TeamFormation.PIPELINE)
result = await coordinator.execute_task("Implement feature", {})
```

### Example 3: Hierarchical Team with Manager

**Before**:
```python
from victor.framework.team_coordinator import FrameworkTeamCoordinator

coordinator = FrameworkTeamCoordinator()
coordinator.set_manager(manager)
coordinator.add_member(worker1)
coordinator.add_member(worker2)
result = await coordinator.execute_task("Coordinate work", {})
```

**After**:
```python
from victor.teams import create_coordinator, TeamFormation

coordinator = create_coordinator(lightweight=True)
coordinator.add_member(manager)  # Manager added first
coordinator.add_member(worker1)
coordinator.add_member(worker2)
coordinator.set_formation(TeamFormation.HIERARCHICAL)
result = await coordinator.execute_task("Coordinate work", {})
```

**Note**: Hierarchical formation now auto-detects the manager by `DELEGATE` capability, so you don't need to call `set_manager()` explicitly.

### Example 4: Production Team with Observability

**Before**:
```python
from victor.framework.team_coordinator import FrameworkTeamCoordinator

coordinator = FrameworkTeamCoordinator()
# No observability support
```

**After**:
```python
from victor.teams import create_coordinator

# Production coordinator with full observability
coordinator = create_coordinator(
    orchestrator=my_orchestrator,
    enable_observability=True,
    enable_rl=True,
)
coordinator.add_member(agent1).add_member(agent2)
result = await coordinator.execute_task("Complex task", {})
```

## Deprecation Timeline

### Current Status (v0.4.x)
- ✅ All old APIs still work
- ⚠️ Deprecation warnings issued for FrameworkTeamCoordinator
- 📖 Migration guide available

### Future (v0.5.x)
- ⚠️ FrameworkTeamCoordinator remains as compatibility wrapper
- 📢 Encourage migration to new APIs
- 🔄 Update examples and documentation

### Future (v1.0.0)
- ❌ FrameworkTeamCoordinator may be removed
- ✅ Only new APIs supported
- 📚 Full migration to new architecture complete

## Backward Compatibility

### Wrapper Implementation

`FrameworkTeamCoordinator` is now a thin wrapper around `UnifiedTeamCoordinator`:

```python
class FrameworkTeamCoordinator(ITeamCoordinator):
    """Deprecated: Wrapper around UnifiedTeamCoordinator."""

    def __init__(self) -> None:
        warnings.warn(
            "FrameworkTeamCoordinator is deprecated. "
            "Use victor.teams.create_coordinator(lightweight=True) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        # Delegate to UnifiedTeamCoordinator
        self._unified = UnifiedTeamCoordinator(
            orchestrator=None,
            enable_observability=False,
            enable_rl=False,
            lightweight_mode=True,
        )

    def add_member(self, member: ITeamMember) -> "FrameworkTeamCoordinator":
        return self._unified.add_member(member)

    # ... all other methods delegate to _unified
```

### No Breaking Changes

All existing code continues to work:
- ✅ All public APIs maintained
- ✅ Same method signatures
- ✅ Same behavior
- ✅ Deprecation warnings guide migration

## New Features Available

By migrating to the new APIs, you gain access to:

1. **Unified Event System**: All agent events flow through EventBus
2. **Reinforcement Learning Integration**: RLMixin for team composition learning
3. **Enhanced Observability**: Better metrics and tracking
4. **Consensus Formation**: Multi-round consensus building
5. **Strategy Pattern**: Extensible formation strategies

## Testing Your Migration

### Unit Tests

```python
# Old test code
from victor.framework.team_coordinator import FrameworkTeamCoordinator

def test_team_execution():
    coordinator = FrameworkTeamCoordinator()
    # ... test code

# New test code
from victor.teams import create_coordinator

def test_team_execution():
    coordinator = create_coordinator(lightweight=True)
    # ... same test code
```

### Integration Tests

```python
# Test that old API still works (compatibility test)
import warnings
from victor.framework.team_coordinator import FrameworkTeamCoordinator

def test_backward_compatibility():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        coordinator = FrameworkTeamCoordinator()
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "deprecated" in str(w[0].message).lower()
```

## Common Migration Patterns

### Pattern 1: Simple Team

```python
# Old
coordinator = FrameworkTeamCoordinator()

# New
coordinator = create_coordinator(lightweight=True)
```

### Pattern 2: Team with Orchestrator

```python
# Old (not supported)
# FrameworkTeamCoordinator had no orchestrator support

# New
coordinator = create_coordinator(orchestrator=my_orchestrator)
```

### Pattern 3: Testing Without Dependencies

```python
# Old
coordinator = FrameworkTeamCoordinator()

# New
coordinator = create_coordinator(
    lightweight=True,
    enable_observability=False,
    enable_rl=False,
)
```

### Pattern 4: Production with All Features

```python
# Old
# Not possible with FrameworkTeamCoordinator

# New
coordinator = create_coordinator(
    orchestrator=my_orchestrator,
    enable_observability=True,
    enable_rl=True,
)
```

## Getting Help

### Documentation

- **Consolidation history**: See [completed and superseded work](https://github.com/anvai-labs/victor/blob/develop/docs/development/completed-work.md)
- **API Reference**: See `victor/teams/__init__.py`
- **Formation Strategies**: See `victor/coordination/formations/`

### Examples

See the test files for complete examples:
- `tests/integration/agents/test_team_execution.py`
- `tests/examples/test_multiagent_tdd_patterns.py`

### Support

If you encounter issues during migration:
1. Check this guide first
2. Review the test examples
3. Open an issue with your migration code and error message

## Checklist

Use this checklist to ensure complete migration:

- [ ] Update imports from `victor.framework.team_coordinator` to `victor.teams`
- [ ] Replace `FrameworkTeamCoordinator()` with `create_coordinator(lightweight=True)`
- [ ] Update coordinator creation to use `create_coordinator()` factory
- [ ] Add explicit `set_formation()` calls if you relied on implicit defaults
- [ ] Test with deprecation warnings enabled
- [ ] Update documentation and examples
- [ ] Remove old imports after migration

## Summary

The consolidated multi-agent system provides:
- ✅ **Simpler API**: Single `create_coordinator()` factory
- ✅ **More Features**: Observability, RL, enhanced formations
- ✅ **Better Design**: SOLID principles, strategy pattern
- ✅ **Full Compatibility**: Existing code continues to work

Migrate at your own pace. The old APIs will continue to work through the deprecation period.
