# State Machine Architecture

Victor uses a state machine to track conversation progress and manage tool availability during agent sessions. This document explains the state machine architecture, stages, transitions, and how to integrate with it.

## Overview

The state machine tracks the agent's workflow through discrete stages, from initial task receipt through completion. Each stage has:
- **Allowed tools**: Tools appropriate for that phase
- **Keywords**: Patterns that trigger stage detection
- **Valid transitions**: Which stages can follow

## Architecture

This diagram lists the conversation stage categories; it is not a fixed transition chain.
The [agentic-loop sequence](../architecture.md#agenticloop) describes execution authority.

```mermaid
---
title: Conversation stages are distinct from loop execution phases
---
%%{init: {"theme":"base","themeVariables":{"primaryColor":"#E8EFF7","primaryTextColor":"#17324D","primaryBorderColor":"#456987","lineColor":"#456987","fontFamily":"Arial"}}}%%
flowchart TB
  C["ConversationStateMachine"]
  I["INITIAL"]
  P["PLANNING"]
  R["READING"]
  A["ANALYSIS"]
  E["EXECUTION"]
  V["VERIFICATION"]
  D["COMPLETION"]
  C -->|"track conversation stage"| I & P & R & A & E & V & D
```

## Stages

### INITIAL
- **Purpose**: Starting point when conversation begins
- **Tools**: `read`, `search`, `code_search`, `overview`
- **Keywords**: Initial message, no prior context
- **Next stages**: PLANNING, READING, EXECUTION

### PLANNING
- **Purpose**: Agent is strategizing approach to task
- **Tools**: `read`, `search`, `code_search`, `overview`
- **Keywords**: "plan", "approach", "strategy", "how to"
- **Next stages**: READING, ANALYSIS, EXECUTION

### READING
- **Purpose**: Agent is gathering information from files
- **Tools**: `read`, `search`, `code_search`, `overview`, `web_search`
- **Keywords**: "read", "look at", "check", "examine"
- **Next stages**: ANALYSIS, EXECUTION, PLANNING

### ANALYSIS
- **Purpose**: Agent is analyzing gathered information
- **Tools**: `read`, `search`, `code_search`, `overview`
- **Keywords**: "analyze", "understand", "figure out"
- **Next stages**: EXECUTION, PLANNING

### EXECUTION
- **Purpose**: Agent is making changes or running commands
- **Tools**: `write`, `edit`, `shell`, `git_commit`, `refactor`
- **Keywords**: "write", "create", "implement", "change", "fix"
- **Next stages**: VERIFICATION, READING, COMPLETION

### VERIFICATION
- **Purpose**: Agent is validating changes work correctly
- **Tools**: `shell`, `read`, `search`, `test`
- **Keywords**: "verify", "test", "check", "confirm"
- **Next stages**: COMPLETION, EXECUTION

### COMPLETION
- **Purpose**: Task is done
- **Tools**: `read` (for final review)
- **Keywords**: "done", "complete", "finished"
- **Next stages**: (terminal state)

## Using the State Machine

### Framework API (Simplified)

```python
from victor.framework import Agent

agent = await Agent.create()

# Access current state
print(agent.state.stage)  # Stage.INITIAL

# Subscribe to state changes
def on_stage_change(old_state, new_state):
    print(f"Stage: {old_state.stage} -> {new_state.stage}")

agent.on_state_change(on_stage_change)

# Run task - state transitions automatically
result = await agent.run("Refactor the auth module")
print(agent.state.stage)  # Stage.COMPLETION
```

### Direct State Machine Access

```python
from victor.storage.state import (
    ConversationStateMachine,
    ConversationStage,
    STAGE_KEYWORDS,
)

# Create state machine
sm = ConversationStateMachine()

# Get current stage
stage = sm.get_stage()  # ConversationStage.INITIAL

# Get allowed tools for current stage
tools = sm.get_stage_tools()  # {"read", "search", ...}

# Manual transition (typically automatic)
sm.transition_to(ConversationStage.PLANNING, confidence=0.9)

# Check valid transitions
valid = sm.get_valid_transitions()  # [ConversationStage.READING, ...]
```

### Observability Integration

```python
from victor.core.events import get_observability_bus

async def watch_state_events():
    bus = get_observability_bus()

    async def on_state_event(event):
        print(event.topic, event.data)

    return await bus.subscribe("state.*", on_state_event)
```

The returned subscription handle can be passed to `await bus.unsubscribe(handle)`.
Event payload fields vary by producer; use the transition hooks below when you need
conversation-stage transitions specifically.

### Using StateHooks

```python
from victor.observability.hooks import StateHookManager

manager = StateHookManager()

@manager.on_transition
def log_all_transitions(old_stage, new_stage, context):
    print(f"{old_stage} -> {new_stage}")

@manager.on_enter("EXECUTION")
def on_execution_start(stage, context):
    print("Starting execution phase - changes incoming!")

@manager.on_exit("EXECUTION")
def on_execution_end(stage, context):
    print("Execution complete")

# Wire into state machine
sm = ConversationStateMachine(hooks=manager)
```

## Generic State Machine

For custom workflows, use the generic `StateMachine`:

```python
from victor.storage.state import StateMachine, StateConfig

# Define custom workflow
config = StateConfig(
    stages=["DRAFT", "REVIEW", "APPROVED", "PUBLISHED"],
    initial_stage="DRAFT",
    transitions={
        "DRAFT": ["REVIEW"],
        "REVIEW": ["DRAFT", "APPROVED"],
        "APPROVED": ["PUBLISHED"],
        "PUBLISHED": [],
    },
    stage_tools={
        "DRAFT": {"write", "edit"},
        "REVIEW": {"read", "comment"},
        "APPROVED": {"read"},
        "PUBLISHED": {"read"},
    },
)

machine = StateMachine(config)
machine.transition_to("REVIEW")
```

## Transition Validation

### Built-in Validators

```python
from victor.storage.state import StateMachine, StateConfig

config = StateConfig(
    stages=["A", "B", "C"],
    initial_stage="A",
    transitions={"A": ["B"], "B": ["C"], "C": []},
    cooldown_seconds=1.0,      # Prevent rapid transitions
    backward_threshold=0.9,    # Require high confidence for backward
)
```

### Custom Validators

```python
from victor.storage.state.protocols import TransitionValidatorProtocol

class RequireMinToolsValidator:
    """Require minimum tool calls before completion."""

    def validate(self, current, target, context):
        if target == "COMPLETION":
            tools_used = context.get("tool_calls", 0)
            if tools_used < 3:
                return False, "Need at least 3 tool calls"
        return True, None

machine = StateMachine(config, validators=[RequireMinToolsValidator()])
```

## Protocols

The state machine system uses protocols for extensibility:

```python
from victor.storage.state.protocols import (
    StateProtocol,           # Core state machine interface
    StageDetectorProtocol,   # Stage detection from context
    TransitionValidatorProtocol,  # Validate transitions
    StateObserverProtocol,   # Observe transitions
)

# Implement custom detector
class MyStageDetector:
    def detect_stage(self, context):
        if "urgent" in context.get("message", "").lower():
            return "EXECUTION", 0.95
        return None, 0.0
```

## Best Practices

1. **Let stages flow naturally** - Don't force transitions; let tool usage drive detection
2. **Use observability for debugging** - Subscribe to STATE events to trace issues
3. **Configure cooldowns for stability** - Prevent rapid back-and-forth transitions
4. **Use validators for invariants** - Enforce business rules on transitions
5. **Keep stage tools focused** - Each stage should have a distinct tool set

## Related Documentation

- [Verticals →](../reference/verticals/index.md) - Domain-specific stage configurations
- [Tool Catalog →](../reference/tools/catalog.md) - Complete tool reference
- [Development Guide →](../development/index.md) - Framework entrypoints and structure
