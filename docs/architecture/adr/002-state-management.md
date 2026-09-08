# ADR-002: State Management System

## Metadata

- **Status**: Accepted
- **Date**: 2025-02-26
- **Decision Makers**: Vijaykumar Singh
- **Related ADRs**: ADR-001 (Agent Orchestration)

## Context

Victor agents need to maintain state across:
- Multi-turn conversations
- Workflow execution
- Tool calls
- Checkpoint/resume operations
- Team coordination

The challenge is providing a unified state management system that:
- Works across different execution contexts
- Supports state persistence and resumption
- Enables copy-on-write optimization
- Provides clear state scoping

## Decision

We will implement a **Four-Scope State Management System** with unified state access through protocols.

### State Scopes

```
GLOBAL (Application-wide)
    ↓
TEAM (Multi-agent shared)
    ↓
WORKFLOW (Workflow execution)
    ↓
CONVERSATION (Single agent)
```

### Architecture

```python
class GlobalStateManager:
    """Unified state management across all scopes."""

    # Scope accessors
    def get_conversation_state(self, session_id: str) -> ConversationState
    def get_workflow_state(self, workflow_id: str) -> WorkflowState
    def get_team_state(self, team_id: str) -> TeamState
    def get_global_state(self) -> GlobalState

    # State operations
    async def save_checkpoint(self, scope: StateScope, state: dict)
    async def load_checkpoint(self, scope: StateScope) -> dict
```

### Key Features

1. **Unified Access**: Single entry point for all state operations
2. **Copy-on-Write**: Efficient state updates without full copies
3. **Checkpointing**: Save and restore state at any point
4. **Protocol-Based**: Type-safe state access through protocols
5. **Observable**: State changes emit events for monitoring

## Rationale

### Why Four Scopes?

**CONVERSATION** (Agent-level):
- Single agent's conversation history
- Message exchanges
- Tool call results
- **Lifetime**: Single agent session

**WORKFLOW** (Workflow-level):
- State shared across workflow nodes
- Intermediate results
- Node execution status
- **Lifetime**: Single workflow execution

**TEAM** (Multi-agent):
- State shared between agents
- Communication messages
- Shared context
- **Lifetime**: Team session

**GLOBAL** (Application):
- Application-wide settings
- Shared resources
- Caches
- **Lifetime**: Application lifetime

### Why Copy-on-Write?

**Pros:**
- Efficient for large state objects
- No need to deep-copy entire state
- Automatic optimization for updates
- Memory efficient

**Cons:**
- Slightly more complex implementation
- Need to track modified keys

### Why Protocol-Based Access?

**Pros:**
- Type-safe state access
- Clear interface contracts
- Easy to mock for testing
- Better IDE support

**Cons:**
- More boilerplate
- Need to define protocols

## Consequences

### Positive

- **Clear State Boundaries**: Each scope has well-defined responsibilities
- **Efficient**: Copy-on-write reduces memory overhead
- **Observable**: State changes emit events for debugging
- **Testable**: Each scope can be tested independently
- **Flexible**: Easy to add new state scopes if needed

### Negative

- **Complexity**: More complex than simple dict-based state
- **Learning Curve**: Developers need to understand scope hierarchy
- **Memory**: Some overhead from tracking state changes

### Neutral

- **Performance**: Copy-on-write is efficient for most use cases
- **API**: State access remains simple through protocols
- **Backward Compatible**: Existing code continues to work

## Implementation

### Phase 1: Core State Management (Completed)

- ✅ Four-scope state hierarchy
- ✅ Copy-on-write optimization
- ✅ State protocols for type-safe access
- ✅ Checkpointing system

### Phase 2: Enhanced Features (In Progress)

- 🔄 State change events
- 🔄 State validation
- 🔄 State migration between versions
- 🔄 State compression for storage

### Phase 3: Advanced Features (Planned)

- ⏳ Distributed state (multi-process)
- ⏳ State synchronization
- ⏳ State versioning

## Code Example

```python
from victor.state import GlobalStateManager

# Get state manager
manager = GlobalStateManager.get_instance()

# Access conversation state
conversation = manager.get_conversation_state("session-123")

# Update state (copy-on-write)
conversation.update({"messages": [...], "stage": "complete"})

# Save checkpoint
await manager.save_checkpoint(
    StateScope.CONVERSATION,
    conversation.to_dict()
)

# Later, restore
restored = await manager.load_checkpoint(
    StateScope.CONVERSATION,
    "session-123"
)
```

## Alternatives Considered

### 1. Single Global State

**Description**: One global state dictionary for everything

**Rejected Because**:
- No isolation between scopes
- Memory inefficient
- Hard to reason about state ownership

### 2. Immutable State

**Description**: State objects are immutable, always create new copies

**Rejected Because**:
- Memory inefficient
- Slower for frequent updates
- Unnecessary complexity

### 3. Database-Backed State

**Description**: All state stored in database

**Rejected Because**:
- Too slow for in-memory operations
- Adds dependency overhead
- Overkill for local state

## References

- [State Management Patterns](https://martinfowler.com/eaaDev/StateManagement.html)
- [Copy-on-Write Optimization](https://en.wikipedia.org/wiki/Copy-on-write)
- [Victor State Implementation](https://github.com/anvai-labs/victor/tree/develop/victor/state/)

## Revision History

| Date | Version | Changes | Author |
|------|---------|---------|--------|
| 2025-02-26 | 1.0 | Initial ADR | Vijaykumar Singh |
