# Workflow Architecture Consolidation Plan

## Codex Claims Verification

### CONFIRMED Issues

| Claim | Status | Evidence |
|-------|--------|----------|
| **Dual StateGraph implementations** | ✅ CONFIRMED | `victor/framework/graph.py:793` and `victor/workflows/graph_dsl.py:278` |
| **Different compile() return types** | ✅ CONFIRMED | framework → `CompiledGraph`, workflows → `WorkflowDefinition` |
| **State model fragmentation** | ✅ CONFIRMED | 4 different state types across the codebase |
| **Facade split (not unified)** | ✅ CONFIRMED | `WorkflowEngine` uses `WorkflowExecutor`, `StateGraphExecutor` uses compiler |
| **Duplicate WorkflowState** | ✅ CONFIRMED | `adapters.py:63` and `yaml_to_graph_compiler.py:107` |

### Current Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           AUTHORING LAYER                                    │
├──────────────────────┬──────────────────────┬───────────────────────────────┤
│    YAML Files        │  graph_dsl.StateGraph │  framework.StateGraph        │
│    (*.yaml)          │  (dataclass State)    │  (TypedDict state)           │
└──────────┬───────────┴──────────┬───────────┴───────────────┬───────────────┘
           │                      │                           │
           ▼                      ▼                           ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                         INTERMEDIATE REPRESENTATION                           │
├──────────────────────────────────────────┬───────────────────────────────────┤
│          WorkflowDefinition              │         CompiledGraph             │
│          (victor/workflows/definition.py)│         (victor/framework/graph.py)│
└──────────────────┬───────────────────────┴───────────────┬───────────────────┘
                   │                                       │
    ┌──────────────┼───────────────────┐                   │
    │              │                   │                   │
    ▼              ▼                   ▼                   ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                           EXECUTION LAYER                                     │
├───────────────────┬───────────────────┬───────────────────┬──────────────────┤
│  WorkflowExecutor │ StateGraphExecutor│   WorkflowEngine  │ CompiledGraph    │
│  (executor.py)    │ (unified_exec.py) │  (workflow_engine)│   .invoke()      │
│                   │                   │                   │                  │
│  Uses:            │  Compiles to      │  Uses:            │  Native          │
│  WorkflowContext  │  CompiledGraph    │  WorkflowExecutor │  execution       │
│  (dataclass)      │  then executes    │  for YAML         │                  │
└───────────────────┴───────────────────┴───────────────────┴──────────────────┘
```

## SOLID Violations

### 1. Single Responsibility Principle (SRP) Violations
- `WorkflowExecutor` handles both DAG execution AND parallel node orchestration
- `StateGraph` in `graph_dsl.py` is both a builder AND compiles to different IR

### 2. Open/Closed Principle (OCP) Violations
- Adding new node types requires modifying multiple files (executor, compiler, handlers)
- Custom node execution is now extensible via `PluginContext.register_workflow_node_executor()`; remaining work is documenting and hardening the extension lifecycle

### 3. Liskov Substitution Principle (LSP) Violations
- Two `StateGraph` classes with same name but incompatible `compile()` return types
- Cannot substitute one for another despite similar APIs

### 4. Interface Segregation Principle (ISP) Violations
- `WorkflowExecutor` exposes methods for both simple and temporal execution
- Clients forced to depend on capabilities they don't use

### 5. Dependency Inversion Principle (DIP) Violations
- `WorkflowEngine` directly depends on concrete `WorkflowExecutor` instead of protocol
- `StateGraphExecutor` hardcodes `YAMLToStateGraphCompiler` dependency

## State Model Fragmentation

| Type | Location | Base | Purpose |
|------|----------|------|---------|
| `WorkflowContext` | executor.py:310 | dataclass | DAG execution state |
| `WorkflowState` | adapters.py:63 | TypedDict | Adapted workflow state |
| `WorkflowState` | yaml_to_graph_compiler.py:107 | TypedDict | Compiled YAML state |
| `StateType` | framework/graph.py | TypedDict | CompiledGraph state |

## Proposed Consolidation (SOLID-Aligned)

### Phase 1: Rename to Eliminate Confusion (LSP)

**Rationale:** Two classes named `StateGraph` with different behaviors violates LSP.

```python
# Before: victor/workflows/graph_dsl.py
class StateGraph(Generic[S]):  # Compiles to WorkflowDefinition
    def compile(self) -> WorkflowDefinition: ...

# After: Rename to avoid confusion
class WorkflowGraph(Generic[S]):  # Clear it produces WorkflowDefinition
    def compile(self) -> WorkflowDefinition: ...
```

**Files to modify:**
- `victor/workflows/graph_dsl.py` - Rename class
- `victor/workflows/__init__.py` - Update exports
- Tests and usages

### Phase 2: Unified Execution Context (DRY + SRP)

**Rationale:** Multiple state types violate DRY and make maintenance difficult.

```python
# New: victor/workflows/context.py
from typing import TypedDict, Any, Dict, Optional

class ExecutionContext(TypedDict, total=False):
    """Unified execution context for all workflow runtimes."""
    # Core data
    data: Dict[str, Any]

    # Execution metadata
    _workflow_id: str
    _current_node: str
    _node_results: Dict[str, Any]
    _error: Optional[str]

    # Temporal (optional)
    _as_of_date: Optional[str]
    _lookback_periods: Optional[int]
```

**Migration:**
- `WorkflowContext` → `ExecutionContext` adapter
- `WorkflowState` (both) → Deprecated, use `ExecutionContext`

### Phase 3: Node Execution Protocol (ISP + DIP)

**Rationale:** Extract common node execution behavior into a protocol.

```python
# New: victor/workflows/protocols.py
from typing import Protocol, Any, Dict

class NodeRunner(Protocol):
    """Protocol for node execution strategies."""

    async def execute(
        self,
        node_id: str,
        context: ExecutionContext,
        **kwargs: Any,
    ) -> ExecutionContext:
        """Execute a node and return updated context."""
        ...

class AgentNodeRunner:
    """Runs agent nodes via orchestrator."""
    ...

class ComputeNodeRunner:
    """Runs compute nodes via handlers."""
    ...

class TransformNodeRunner:
    """Runs transform nodes."""
    ...

class HITLNodeRunner:
    """Runs human-in-the-loop nodes."""
    ...
```

**Benefits:**
- `WorkflowExecutor` and `StateGraphExecutor` share `NodeRunner` implementations
- New node types added by implementing protocol (OCP)
- Clients depend on protocol, not concrete implementations (DIP)

### Phase 4: Single Execution Engine (SRP)

**Rationale:** Converge on `CompiledGraph` as the single execution engine.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           AUTHORING LAYER                                    │
├──────────────────────┬──────────────────────┬───────────────────────────────┤
│    YAML Files        │    WorkflowGraph     │    StateGraph                 │
│    (*.yaml)          │    (ex graph_dsl)    │    (framework)                │
└──────────┬───────────┴──────────┬───────────┴───────────────┬───────────────┘
           │                      │                           │
           ▼                      ▼                           │
┌──────────────────────────────────────────────────────────────────────────────┐
│                    COMPILATION LAYER (NEW)                                   │
├──────────────────────────────────────────┬───────────────────────────────────┤
│        YAMLToGraphCompiler               │    WorkflowGraphCompiler         │
│        (yaml_to_graph_compiler.py)       │    (new - from graph_dsl)        │
└──────────────────────────────────────────┴───────────────────────────────────┘
                           │                           │
                           ▼                           ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                    SINGLE IR: CompiledGraph                                  │
│                    (victor/framework/graph.py)                               │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                    UNIFIED EXECUTOR                                          │
│                    WorkflowEngine (facade)                                   │
│                    ├── Uses NodeRunner protocol                              │
│                    ├── Delegates to CompiledGraph.invoke()                   │
│                    └── Emits WorkflowEvent for observability                 │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Phase 5: Unified Observability (SRP)

**Rationale:** Streaming should emit rich events from all execution paths.

```python
# Extend CompiledGraph to emit events
class CompiledGraph(Generic[StateType]):
    async def invoke(
        self,
        state: StateType,
        event_callback: Optional[Callable[[WorkflowEvent], None]] = None,
    ) -> StateType:
        """Execute with optional event emission."""
        ...

    async def stream(
        self,
        state: StateType,
    ) -> AsyncIterator[WorkflowEvent]:  # Not just (node_id, state)
        """Stream rich events during execution."""
        ...
```

## Migration Strategy

### Backward Compatibility

1. **Deprecation aliases:**
   ```python
   # victor/workflows/graph_dsl.py
   StateGraph = WorkflowGraph  # Deprecated alias
   warnings.warn("StateGraph renamed to WorkflowGraph", DeprecationWarning)
   ```

2. **Context adapters:**
   ```python
   def to_execution_context(ctx: WorkflowContext) -> ExecutionContext:
       """Adapt legacy WorkflowContext to ExecutionContext."""
       ...
   ```

3. **Phased rollout:**
   - Phase 1-2: No breaking changes, only additions
   - Phase 3-4: Deprecation warnings
   - Phase 5: Remove deprecated code in next major version

## Files to Create/Modify

### New Files
- `victor/workflows/context.py` - Unified ExecutionContext
- `victor/workflows/node_runners.py` - NodeRunner implementations

### Modified Files
- `victor/workflows/graph_dsl.py` - Rename StateGraph → WorkflowGraph
- `victor/workflows/executor.py` - Use NodeRunner protocol
- `victor/workflows/unified_executor.py` - Consolidate with WorkflowEngine
- `victor/framework/workflow_engine.py` - Use CompiledGraph for all paths
- `victor/framework/graph.py` - Add event emission to streaming

### Deprecated (Remove in v1.0)
- `victor/workflows/adapters.py:WorkflowState` - Removed (dead code, co-design review item 25c); use ExecutionContext
- `victor/workflows/yaml_to_graph_compiler.py:WorkflowState` - Use ExecutionContext
- `victor/workflows/executor.py:WorkflowContext` - Use ExecutionContext

## Success Criteria

1. [ ] Single `StateGraph` class (rename DSL version to `WorkflowGraph`)
2. [ ] Single execution context type (`ExecutionContext`)
3. [ ] All execution paths through `CompiledGraph`
4. [ ] Unified streaming with `WorkflowEvent`
5. [ ] `NodeRunner` protocol used by all executors
6. [ ] No duplicate state types
7. [ ] All existing tests pass

## Risks

| Risk | Mitigation |
|------|------------|
| Breaking external integrations | Deprecation aliases + migration guide |
| Performance regression | Benchmark before/after |
| Test coverage gaps | Ensure 100% coverage on new code |
