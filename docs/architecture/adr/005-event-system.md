# ADR-005: Event System Architecture

## Metadata

- **Status**: Accepted
- **Date**: 2025-02-26
- **Decision Makers**: Vijaykumar Singh
- **Related ADRs**: ADR-001 (Agent Orchestration), ADR-002 (State Management)

## Context

Victor needs an event system that:
- Provides observability into agent execution
- Enables real-time monitoring and debugging
- Supports event streaming to clients
- Allows middleware to intercept events
- Maintains performance for high-throughput scenarios

The challenge is balancing:
- Rich event data for observability
- Performance for production workloads
- Flexibility for different consumers
- Simplicity of the API

## Decision

We will implement a **Structured Event Model** with typed events, middleware pipeline, and subscription support.

### Architecture

```
Agent Execution
    ↓
Event Emission
    ↓
Middleware Pipeline (pre-processing)
    ↓
Event Bus (Pub/Sub)
    ↓
Subscribers (Metrics, Logging, Streaming, etc.)
```

### Event Types

```python
class EventType(str, Enum):
    # LLM Events
    THINKING = "thinking"           # Extended thinking
    CONTENT = "content"             # Text content

    # Tool Events
    TOOL_CALL = "tool_call"         # Tool invocation
    TOOL_RESULT = "tool_result"     # Tool response
    TOOL_ERROR = "tool_error"       # Tool failure

    # Agent Events
    STAGE_CHANGE = "stage_change"   # Conversation stage change
    ERROR = "error"                 # Agent error

    # System Events
    STREAM_START = "stream_start"   # Stream begin
    STREAM_END = "stream_end"       # Stream complete
    CHECKPOINT = "checkpoint"       # State saved
```

### Event Structure

```python
@dataclass
class AgentExecutionEvent:
    """Structured event from agent execution."""
    event_type: EventType
    timestamp: datetime
    correlation_id: str
    agent_id: str
    session_id: str
    content: Optional[str]
    metadata: Dict[str, Any]
```

### Middleware Pipeline

```python
class Middleware:
    """Base middleware for event processing."""

    async def pre_process(
        self,
        event: AgentExecutionEvent
    ) -> Optional[AgentExecutionEvent]:
        """Process event before emission."""
        pass

    async def post_process(
        self,
        event: AgentExecutionEvent
    ) -> Optional[AgentExecutionEvent]:
        """Process event after emission."""
        pass
```

## Rationale

### Why Structured Events?

**Benefits**:
- Type-safe event handling
- Self-documenting event schema
- Easy to serialize
- Better IDE support

**Trade-offs**:
- More boilerplate than unstructured events
- Need to define event types upfront

### Why Middleware Pipeline?

**Benefits**:
- Consistent event processing
- Composable behaviors
- Easy to add new middleware
- Order of operations controlled

**Trade-offs**:
- Slightly more complex than direct emission
- Need to manage middleware order

### Why Event Bus?

**Benefits**:
- Decoupled producers and consumers
- Multiple subscribers support
- Easy to add monitoring
- Natural streaming support

**Trade-offs**:
- More complex than direct callbacks
- Need to manage subscriptions

## Consequences

### Positive

- **Observable**: Rich event data for monitoring
- **Extensible**: Easy to add event consumers
- **Flexible**: Middleware provides customization
- **Performant**: Minimal overhead from event system
- **Debuggable**: Events provide execution trace

### Negative

- **Complexity**: More moving parts
- **Learning Curve**: Need to understand event types
- **Overhead**: Event emission has cost

### Neutral

- **API**: Event emission is automatic
- **Performance**: Optimized for low overhead
- **Compatibility**: Non-breaking to add events

## Implementation

### Phase 1: Core Events (Completed)

- ✅ Event type definitions
- ✅ Event structure
- ✅ Event emission from orchestrator
- ✅ Basic event bus

### Phase 2: Middleware (Completed)

- ✅ Logging middleware
- ✅ Metrics middleware
- ✅ Secret masking middleware
- ✅ Git safety middleware

### Phase 3: Advanced Features (In Progress)

- 🔄 Event filtering
- 🔄 Event aggregation
- 🔄 Event replay
- 🔄 Event archiving

## Code Example

### Event Emission

```python
from victor import Agent

agent = Agent.create()

# Events emitted automatically
async for event in agent.stream("Hello, World!"):
    if event.type == "content":
        print(event.content)
    elif event.type == "thinking":
        print("[Thinking...]")
```

### Custom Middleware

```python
from victor.framework.middleware import Middleware

class CustomMiddleware(Middleware):
    async def pre_process(self, event):
        print(f"Before: {event.event_type}")
        return event

    async def post_process(self, event):
        print(f"After: {event.event_type}")
        return event

# Add to agent
agent = Agent.create(middleware=[CustomMiddleware()])
```

### Event Subscription

```python
from victor.core.events import EventBus

bus = EventBus.get_instance()

async def my_subscriber(event):
    if event.event_type == "error":
        print(f"Error occurred: {event.content}")

# Subscribe
bus.subscribe(my_subscriber)
```

## Alternatives Considered

### 1. Callback-based

**Description**: Direct callbacks instead of events

**Rejected Because**:
- Tightly coupled
- Hard to extend
- No natural streaming

### 2. Unstructured Events

**Description**: Simple dict-based events

**Rejected Because**:
- No type safety
- Poor documentation
- Hard to maintain

### 3. No Events

**Description**: No built-in event system

**Rejected Because**:
- No observability
- Hard to debug
- No streaming support

## References

- [Event-Driven Architecture](https://martinfowler.com/eaaDev/EventDrivenArchitecture.html)
- [Observer Pattern](https://en.wikipedia.org/wiki/Observer_pattern)
- [Victor Events](https://github.com/anvai-labs/victor/blob/develop/victor/framework/events.py)

## Revision History

| Date | Version | Changes | Author |
|------|---------|---------|--------|
| 2025-02-26 | 1.0 | Initial ADR | Vijaykumar Singh |
