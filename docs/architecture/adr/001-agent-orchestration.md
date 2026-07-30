# ADR-001: Agent Orchestrator Architecture

## Metadata

- **Status**: Superseded
- **Date**: 2025-02-26
- **Decision Makers**: Vijaykumar Singh
- **Related ADRs**: None

## Update (2026-05-04)

This ADR captures the original move away from a monolithic orchestrator, but it
no longer describes the current steady-state runtime.

Current runtime shape:

- `AgentOrchestrator` is the composition root, session boundary, and
  compatibility hotspot.
- Canonical effectful behavior lives in `ChatService`, `ToolService`,
  `SessionService`, `ContextService`, `ProviderService`, and
  `RecoveryService`.
- State-passed coordinators are used selectively for exploration, safety,
  system-prompt classification, and coordination recommendation.
- Facades and deprecated coordinators remain compatibility or grouping
  surfaces; they are not the canonical ownership layer for new behavior.

See [`docs/architecture.md`](../../architecture.md) (canonical system architecture) and
[ADR-019](019-orchestrator-service-runtime-decomposition.md) (the decomposition *target* the
service-first runtime is converging toward) for the authoritative current runtime architecture.

## Context

Victor needs a core orchestration system that can:
- Manage conversation flow between users and LLMs
- Coordinate tool execution
- Handle multi-turn conversations
- Support different LLM providers
- Enable extensibility through plugins and verticals

The challenge is balancing:
- Simplicity of the public API
- Flexibility for advanced use cases
- Performance for production workloads
- Maintainability over time

## Historical Decision

We will use a **Coordinator Pattern** with the Agent Orchestrator as the central coordinator, delegating specialized responsibilities to focused coordinator classes.

### Architecture

```
Agent (Public API)
    ↓
AgentOrchestrator (Central Coordinator)
    ↓
├── ConversationCoordinator (Message history, context management)
├── SafetyCoordinator (Safety rules, pattern matching)
├── ToolCoordinator (Tool execution, result caching)
├── MetricsCoordinator (Performance tracking, observability)
└── ProviderCoordinator (LLM provider abstraction)
```

### Key Components

1. **Agent**: Simple facade providing `run()`, `stream()`, `chat()` methods
2. **AgentOrchestrator**: Central coordinator managing the conversation loop
3. **Specialized Coordinators**: Each handles a specific aspect of agent behavior
4. **Provider Adapters**: Abstract LLM provider differences
5. **Tool Registry**: Dynamic tool discovery and execution

## Rationale

### Why Coordinator Pattern?

**Pros:**
- **Single Responsibility**: Each coordinator has one clear purpose
- **Testability**: Coordinators can be tested in isolation
- **Extensibility**: New capabilities added via new coordinators
- **Maintainability**: Changes localized to specific coordinators

**Cons:**
- **Complexity**: More classes to understand
- **Coordination overhead**: Need to manage coordinator interactions

### Why Not Alternatives?

**Monolithic Orchestrator**:
- ✗ Hard to test
- ✗ Difficult to extend
- ✗ Violates SRP

**Microservices**:
- ✗ Too complex for single-process framework
- ✗ Overhead of inter-service communication
- ✗ Deployment complexity

**Plugin Architecture**:
- ✗ Less type-safe
- ✗ Dynamic loading complexity
- ✗ Harder to debug

## Consequences

### Positive

- **Clear separation of concerns**: Each coordinator has a focused responsibility
- **Easy to extend**: Add new coordinators without modifying existing code
- **Testable**: Each coordinator can be unit tested independently
- **Observable**: MetricsCoordinator provides built-in observability
- **Safe**: SafetyCoordinator enforces security policies

### Negative

- **Learning curve**: Developers need to understand multiple coordinator classes
- **Overhead**: Coordination between coordinators adds some complexity
- **Initialization**: More setup required for agent creation

### Neutral

- **Public API**: Agent facade remains simple
- **Performance**: Minimal overhead from coordinator pattern
- **Compatibility**: Existing code continues to work

## Implementation

### Phase 1: Core Coordinators (Completed)

- ✅ ConversationCoordinator - Message and context management
- ✅ SafetyCoordinator - Safety rules and enforcement
- ✅ ToolCoordinator - Tool execution and caching
- ✅ MetricsCoordinator - Performance tracking

### Phase 2: Enhanced Coordinators (In Progress)

- 🔄 StateCoordinator - Enhanced state management
- 🔄 CacheCoordinator - Multi-level caching strategy
- 🔄 EventCoordinator - Event streaming and subscriptions

### Phase 3: Advanced Coordinators (Planned)

- ⏳ TeamCoordinator - Multi-agent coordination
- ⏳ WorkflowCoordinator - Workflow orchestration
- ⏳ VerticalCoordinator - Vertical-specific enhancements

## Code Example

```python
from victor import Agent

# Simple API (facade)
agent = Agent.create()

# Internally creates:
# - AgentOrchestrator
# - ConversationCoordinator
# - SafetyCoordinator
# - ToolCoordinator
# - MetricsCoordinator

result = await agent.run("Hello!")
```

## Alternatives Considered

### 1. Direct Orchestrator (Monolithic)

**Description**: Single orchestrator class handling all responsibilities

**Rejected Because**:
- Violates Single Responsibility Principle
- Hard to test
- Difficult to extend

### 2. Plugin Architecture

**Description**: Dynamic loading of capability plugins

**Rejected Because**:
- Less type-safe
- Dynamic loading complexity
- Harder to debug

### 3. Microservices

**Description**: Separate services for each concern

**Rejected Because**:
- Too complex for single-process framework
- Overhead of inter-service communication
- Deployment complexity

## References

- [SOLID Principles](https://en.wikipedia.org/wiki/SOLID)
- [Coordinator Pattern](https://martinfowler.com/eaaDev/Coordinator.html)
- [Victor AgentOrchestrator Implementation](../agent/orchestrator.py)

## Revision History

| Date | Version | Changes | Author |
|------|---------|---------|--------|
| 2025-02-26 | 1.0 | Initial ADR | Vijaykumar Singh |
| 2026-05-04 | 1.1 | Marked superseded; redirected to current service-first runtime docs | Vijaykumar Singh |
| 2026-07-29 | 1.2 | Reconciliation: fixed dead `CURRENT_STATE.md` pointer → `architecture.md`; linked ADR-019 as the decomposition target | Vijaykumar Singh |
