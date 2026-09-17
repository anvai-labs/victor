# State-Passed Architecture for Victor Orchestrator

**Status**: Active selective pattern within the service-first runtime
**Last Reviewed**: 2026-05-04
**Priority**: Apply only where explicit snapshots and transitions improve correctness

---

## Overview

The state-passed architecture is a selective design pattern used in Victor for
decision, policy, and read-heavy seams that benefit from immutable snapshots
and explicit transitions.

It is not the canonical answer for every runtime concern. Effectful chat, tool,
provider, session, context, and recovery flows are service-owned. State-passed
is used where it improves correctness and testability without introducing a
second parallel runtime layer.

### Problem Statement

The current `AgentOrchestrator` still has too much composition and
compatibility logic:
1. **Large composition root**: `victor/agent/orchestrator.py` is still ~4,593 LOC
2. **Compatibility drag**: Deprecated coordinator shims still exist for some seams
3. **Testing difficulties**: Some decision-heavy flows are still easier to test via explicit snapshots than host-object reach-through
4. **Architecture risk**: Without discipline, state-passed and service patterns can become parallel layers instead of complementary ones

### Solution: Selective State-Passed Architecture

Use state-passed where the seam is fundamentally about analysis, classification,
policy, or recommended transitions:
- **Input**: Immutable `ContextSnapshot` of current state
- **Output**: `CoordinatorResult` with explicit `StateTransition` objects
- **No hidden mutation**: Decisions are explicit and testable
- **No blanket rewrite**: Effectful runtime flows remain service-owned

## Current Entry Points

Use the concrete migration surfaces that are already wired into the runtime.

| Need | Preferred surface | Notes |
|------|-------------------|-------|
| Chat execution | `ChatService` or `OrchestrationFacade.chat_service` | Canonical effectful runtime |
| Tool execution | `ToolService` or `OrchestrationFacade.tool_service` | Canonical effectful runtime |
| Session lifecycle | `SessionService` or `OrchestrationFacade.session_service` | Canonical effectful runtime |
| Context management | `ContextService` or `OrchestrationFacade.context_service` | Canonical effectful runtime |
| Provider management | `ProviderService` or `OrchestrationFacade.provider_service` | Canonical effectful runtime |
| Recovery and resilience | `RecoveryService` or `OrchestrationFacade.recovery_service` | Canonical effectful runtime |
| Read-only exploration helper | `victor.agent.coordinators.ExplorationCoordinator` | Legacy helper for direct exploration calls; use the state-passed wrapper for explicit transitions |
| State-passed exploration | `victor.agent.coordinators.ExplorationStatePassedCoordinator` | Snapshot/transition wrapper |
| State-passed system prompt logic | `victor.agent.coordinators.SystemPromptStatePassedCoordinator` | Prefer over direct shim imports |
| State-passed safety checks | `victor.agent.coordinators.SafetyStatePassedCoordinator` | Prefer over coordinator shim usage |
| State-passed coordination recommendation | `victor.agent.coordinators.CoordinationStatePassedCoordinator` | Prefer over legacy coordination heuristics |

Deprecated chat shims are compatibility-only. In particular,
`ChatCoordinator.stream_chat()` now forwards in this order:
1. bound `ChatService.stream_chat()`
2. orchestrator `_get_service_streaming_runtime()`
3. legacy `_stream_chat_runtime` hook

That legacy hook remains only as a fallback for older integrations and should
not be used as the primary runtime surface in new code.

```python
from victor.agent.coordinators import (
    ExplorationCoordinator,
    ExplorationStatePassedCoordinator,
    SafetyStatePassedCoordinator,
    SystemPromptStatePassedCoordinator,
)
from victor.agent.facades import OrchestrationFacade

facade = OrchestrationFacade(
    chat_service=chat_service,
    tool_service=tool_service,
    session_service=session_service,
    exploration_state_passed=ExplorationStatePassedCoordinator(),
    system_prompt_state_passed=SystemPromptStatePassedCoordinator(
        task_analyzer=task_analyzer
    ),
    safety_state_passed=SafetyStatePassedCoordinator(),
)

findings = await ExplorationCoordinator().explore_parallel(
    "Trace the failing import boundary",
    project_root=project_root,
)
```

---

## Core Components

### 1. ContextSnapshot

Immutable snapshot of orchestrator state at a point in time.

```python
@dataclass(frozen=True)
class ContextSnapshot:
    """Immutable snapshot of orchestrator state."""

    # Core conversation state
    messages: tuple[Message, ...]
    session_id: str
    conversation_stage: str

    # Configuration
    settings: Settings
    model: str
    provider: str
    max_tokens: int
    temperature: float

    # State (key-value store)
    conversation_state: Dict[str, Any]
    session_state: Dict[str, Any]

    # Metadata
    observed_files: tuple[str, ...]
    capabilities: Dict[str, Any]
```

**Key Features**:
- **Frozen**: Cannot be modified (prevents accidental mutation)
- **Tuples**: Collections are tuples (not lists) for immutability
- **Dict copies**: State dictionaries are copied (prevents shared mutation)

**Usage**:
```python
# Read from context
message_count = context.message_count
stage = context.conversation_stage
value = context.get_state("key", default=None)

# Check capabilities
if context.has_capability("planning_mode"):
    # ...
```

### 2. StateTransition

Encapsulates a single state change or side effect.

```python
@dataclass
class StateTransition:
    """Encapsulates a state change or side effect."""

    transition_type: TransitionType
    data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
```

**Transition Types**:
- `ADD_MESSAGE`: Add a message to conversation
- `UPDATE_STATE`: Update a key-value pair in state
- `DELETE_STATE`: Remove a key from state
- `EXECUTE_TOOL`: Request tool execution
- `UPDATE_CAPABILITY`: Update a capability flag
- `UPDATE_STAGE`: Update conversation stage
- `CREATE_SESSION` / `CLOSE_SESSION`: Session management

**Usage**:
```python
# Create transitions
transition1 = StateTransition(
    transition_type=TransitionType.UPDATE_STATE,
    data={"key": "plan_steps", "value": steps, "scope": "conversation"}
)

transition2 = StateTransition(
    transition_type=TransitionType.EXECUTE_TOOL,
    data={"tool_name": "list_files", "arguments": {"path": "."}}
)
```

### 3. TransitionBatch

Batches multiple transitions for atomic application.

```python
@dataclass
class TransitionBatch:
    """A batch of transitions to be applied atomically."""

    transitions: List[StateTransition] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
```

**Helper Methods**:
```python
batch = TransitionBatch()

# Method chaining for fluent API
batch.update_state("key", "value") \
     .add_message(message) \
     .execute_tool("tool_name", {"arg": "value"})

# Extend with other batches
batch.extend(other_batch)
```

### 4. CoordinatorResult

Combines transitions with metadata about coordinator decisions.

```python
@dataclass
class CoordinatorResult:
    """Result of a coordinator operation."""

    transitions: TransitionBatch
    reasoning: Optional[str] = None
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    should_continue: bool = True
    handoff_to: Optional[str] = None
```

**Factory Methods**:
```python
# No-op result
result = CoordinatorResult.no_op(reasoning="Nothing to do")

# Result with transitions
result = CoordinatorResult.transitions_only(
    transition1, transition2,
    reasoning="Applied updates"
)
```

### 5. TransitionApplier

Applies transitions to the actual orchestrator.

```python
class TransitionApplier:
    """Applies state transitions to an orchestrator."""

    async def apply(self, transition: StateTransition) -> None:
        """Apply a single transition."""

    async def apply_batch(self, batch: TransitionBatch) -> None:
        """Apply a batch of transitions atomically."""
```

### 6. create_snapshot()

Utility function to create snapshots from orchestrator.

```python
def create_snapshot(orchestrator: Any) -> ContextSnapshot:
    """Create a ContextSnapshot from an orchestrator."""
```

---

## Migration Pattern

### Before (Current Pattern)

```python
class ExampleCoordinator:
    def __init__(self, orchestrator):
        self._orchestrator = orchestrator  # Holds reference

    async def process(self, user_message):
        # Direct state access and mutation
        messages = self._orchestrator.messages
        self._orchestrator.conversation_state["key"] = "value"
        await self._orchestrator.execute_tool("tool", {})
```

**Issues**:
- Tight coupling to orchestrator
- Direct state mutation
- Hard to test (needs full orchestrator mock)

### After (State-Passed Pattern)

```python
class ExampleCoordinator:
    # No __init__ needed (or only config)

    async def process(self, context: ContextSnapshot, user_message: str) -> CoordinatorResult:
        # Pure function: read from context, return transitions
        messages = context.messages  # Read from immutable snapshot

        batch = TransitionBatch()
        batch.update_state("key", "value", scope="conversation")
        batch.execute_tool("tool", {})

        return CoordinatorResult(
            transitions=batch,
            reasoning="Updated state and executed tool",
        )
```

**Benefits**:
- No orchestrator reference
- No direct state mutation
- Easy to test (just create ContextSnapshot)

---

## Integration Example

### Orchestrator Integration

```python
class AgentOrchestrator:
    async def process_with_coordinator(self, user_message: str):
        from victor.agent.coordinators.state_context import (
            create_snapshot,
            TransitionApplier,
        )

        # 1. Create coordinator (no orchestrator reference)
        coordinator = ExampleStatePassedCoordinator()

        # 2. Create snapshot
        snapshot = create_snapshot(self)

        # 3. Call coordinator (pure function)
        result = await coordinator.process(snapshot, user_message)

        # 4. Apply transitions
        applier = TransitionApplier(self)
        await applier.apply_batch(result.transitions)

        # 5. Use result metadata
        if result.reasoning:
            logger.info(f"Coordinator: {result.reasoning}")

        if not result.should_continue:
            return  # Stop processing

        if result.handoff_to:
            # Hand off to next coordinator
            await self._handoff(result.handoff_to, snapshot, user_message)
```

---

## Testing Benefits

### Before (Needs Full Mock)

```python
def test_coordinator():
    # Need complex mock setup
    orchestrator = MagicMock()
    orchestrator.messages = [msg1, msg2]
    orchestrator.conversation_state = {}
    orchestrator.execute_tool = AsyncMock()

    coordinator = ExampleCoordinator(orchestrator)
    await coordinator.process("test")

    # Verify orchestrator was called
    orchestrator.execute_tool.assert_called_once()
```

### After (Simple Snapshot)

```python
def test_coordinator():
    # Just create a snapshot
    snapshot = ContextSnapshot(
        messages=(msg1, msg2),
        session_id="test",
        conversation_stage="initial",
        settings=Settings(),
        model="test",
        provider="test",
        max_tokens=4096,
        temperature=0.7,
        conversation_state={},
        session_state={},
        observed_files=(),
        capabilities={},
    )

    coordinator = ExampleStatePassedCoordinator()
    result = await coordinator.process(snapshot, "test")

    # Verify transitions (no orchestrator mock needed!)
    assert not result.transitions.is_empty()
    assert result.confidence > 0.5
```

---

## File Structure

```
victor/agent/coordinators/
├── state_context.py                      # Core abstractions
├── exploration_state_passed.py   # Implemented exploration coordinator
├── chat_coordinator.py                   # Deprecated shim over service-first runtime
├── planning_coordinator.py               # [TODO: Refactor]
├── execution_coordinator.py              # [TODO: Refactor]
└── sync_chat_coordinator.py              # Deprecated shim over service-first runtime

tests/unit/agent/coordinators/
└── test_state_context.py                 # 34 comprehensive tests
```

---

## Migration Checklist

### Phase 1: Foundation (✅ COMPLETE)
- [x] Create `state_context.py` with core abstractions
- [x] Selective exploration coordinator: `exploration_state_passed.py`
- [x] Create comprehensive unit tests (34 tests)
- [x] Document the pattern

### Phase 2: Gradual Migration (UPDATED 2026-04-27)
- [x] Collapse `chat_coordinator.py` into a deprecated shim over `ChatService`
  - Canonical runtime ownership moved to `ChatService`
  - Deprecated compatibility surface now delegates to service/orchestrator public entrypoints
  - Architecture tests guard against reintroducing local loop ownership
- [ ] Refactor `planning_coordinator.py`
- [ ] Refactor `execution_coordinator.py`
- [x] Collapse `sync_chat_coordinator.py` into a deprecated shim over `ChatService`
  - Canonical runtime ownership moved to `ChatService`
  - Planning path now routes through `ChatService.chat(..., use_planning=True)` or orchestrator public chat
- [ ] Update orchestrator integration points

### Phase 3: Validation (FUTURE)
- [ ] Run full test suite
- [ ] Integration testing
- [ ] Performance benchmarking
- [ ] Documentation updates

---

## Design Principles

### 1. Immutability
- Snapshots are frozen (cannot be modified)
- Collections are tuples (not lists)
- State dictionaries are copied

### 2. Explicit Transitions
- All state changes are explicit (no hidden mutations)
- Transitions are validated on creation
- Atomic batch application

### 3. Testability
- Coordinators are pure functions
- No orchestrator mocks needed
- Simple snapshot construction

### 4. Gradual Migration
- Old and new patterns can coexist
- No breaking changes required
- Incremental adoption possible

---

## FAQ

### Q: Why not just use protocols?
**A**: Protocols still allow direct state mutation. State-passed ensures coordinators are pure functions with no side effects.

### Q: Won't this be slower?
**A**: Negligible impact. Snapshot creation is fast (copies dicts, tuples references). The benefit in testability and maintainability far outweighs minimal overhead.

### Q: Do I need to refactor all coordinators at once?
**A**: No! The old and new patterns can coexist. Migrate incrementally, one coordinator at a time.

### Q: How do I handle async operations (tool execution)?
**A**: Return an `EXECUTE_TOOL` transition. The orchestrator will apply it and handle the async execution.

### Q: What about nested coordinator calls?
**A**: Pass the same snapshot. Each coordinator returns its own transitions. The orchestrator merges and applies them.

---

## References

- **Historical feedback**: The 3,915 LOC count belongs to the original review, not the current tree
- **Implementation**: `victor/agent/coordinators/state_context.py`
- **Example**: `victor/agent/coordinators/exploration_state_passed.py`
- **Tests**: `tests/unit/agent/coordinators/test_state_context.py`

---

## Status Summary

| Concern | Status | Notes |
|---------|--------|-------|
| Core abstractions | ✅ Complete | Context snapshot, transitions, and result types exist |
| Exploration coordinator | ✅ Implemented | `ExplorationStatePassedCoordinator` is a current selective example |
| Service-owned effectful runtime | ✅ Canonical | Chat, tool, session, context, provider, and recovery stay service-owned |
| Exploration / system prompt / safety | ✅ Canonical selective seams | State-passed is the preferred pattern here |
| Coordination recommendation | ✅ Canonical selective seam | `coordination_state_passed.py` |
| Blanket coordinator rewrites | 🚫 Not the goal | Do not create a second parallel runtime layer |
| Remaining orchestrator shrink work | ⏳ In progress | Size is ratcheted in `tests/unit/runtime/test_hotspot_size_guard.py` |

**Overall Progress**: State-passed is now an established selective pattern inside
the broader service-first runtime. Future work should target seams that still
benefit from explicit transitions, not re-migrate domains already cleanly owned
by services.

---

**Next Steps**: Keep service-owned domains service-owned, continue deleting dead
compatibility paths, and use `exploration_state_passed.py` as the
reference when a remaining decision seam genuinely benefits from explicit
snapshot/transition modeling.
