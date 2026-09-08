# StateGraph DSL Guide

Build stateful workflows with the public `StateGraph` builder. YAML workflow definitions
compile to the same graph engine; see the [workflow tutorial](../../tutorials/create-workflow.md).

## Core Concepts

A graph contains named nodes, edges, and an entry point. Nodes receive state and return
an updated state. A `TypedDict` can document that shape, or you can use a plain dictionary.
`Optional[T]` permits `None`; it does not make a TypedDict key optional or supply a default.
Initialize every key that your nodes read.

Synchronous functions suit pure computation; use asynchronous functions for asynchronous
I/O. Return a new mapping when practical so state transitions are easy to inspect.

## Quick Start

This complete example processes a task and returns a `GraphExecutionResult`. User data
is in `result.state`.

```python
import asyncio
from typing import TypedDict
from victor.framework import StateGraph, END

class TaskState(TypedDict):
    task: str
    result: str
    attempts: int

def process(state: TaskState) -> TaskState:
    return {**state, "result": f"Processed: {state['task']}", "attempts": state["attempts"] + 1}

async def main():
    graph = StateGraph(TaskState)
    graph.add_node("process", process)
    graph.set_entry_point("process")
    graph.add_edge("process", END)
    result = await graph.compile().invoke({"task": "Hello", "result": "", "attempts": 0})
    assert result.success, result.error
    print(result.state)

asyncio.run(main())
```

## Building Workflows

Use `add_node(name, function)`, `add_edge(source, target)`, and `set_entry_point(name)`.
`END` marks termination; `set_finish_point(name)` adds an edge to it. All referenced
nodes must exist before compilation.

## Conditional Branching

Use `add_conditional_edge` (singular). A synchronous router returns a branch name;
an explicit mapping converts that name into a destination. This bounded loop performs
two iterations and then stops.

```python
import asyncio
from victor.framework import StateGraph, END

async def main():
    graph = StateGraph(dict)
    graph.add_node("step", lambda state: {**state, "attempts": state["attempts"] + 1})
    graph.set_entry_point("step")
    graph.add_conditional_edge(
        "step",
        lambda state: "again" if state["attempts"] < 2 else "done",
        {"again": "step", "done": END},
    )
    result = await graph.compile(strict_edges=True).invoke({"attempts": 0})
    assert result.success, result.error
    assert result.state["attempts"] == 2

asyncio.run(main())
```

`strict_edges=True` turns an unmatched graph branch into an error. Explicit termination
conditions and iteration limits are needed for cyclic graphs.

## Streaming

`CompiledGraph.stream` yields `(node_id, state)` tuples, not framework `EventType`
objects. Use `aclosing` if a consumer may stop early, so node cancellation propagates
when the iterator is closed.

```python
from contextlib import aclosing

# compiled is a graph created with graph.compile().
async with aclosing(compiled.stream(initial_state)) as stream:
    async for node_id, state in stream:
        print(node_id, state)
```

For agent content/tool events, use `Agent.stream` as described in the
[Python API](../../reference/api/python-api.md).

## Checkpointing

Pass an implementation of the graph's `CheckpointerProtocol` to `compile(checkpointer=...)`.
Checkpoints preserve execution state and routing metadata supported by the chosen
path; they are not human approval responses. Completed-at versus resume-at semantics
remain path-dependent until FEP-0032. The general interrupt-after and paused-signal changes remain tracked in
[FEP-0032](https://github.com/anvai-labs/victor/blob/develop/feps/fep-0032-interrupt-resume-semantics.md).
See the [HITL guide](../HITL_WORKFLOWS.md) for the distinction and current limits.

## Integration with Tools

Application nodes can close over a configured Agent and call `await agent.run(...)`,
then store `result.content` in a user state key. Create the Agent with
`await Agent.create(...)` and manage its lifetime outside the graph; do not construct
an internal orchestrator inside each node.

## Multi-Agent Workflows

Create a team with `await Agent.create_team(...)`, then use `await team.run()` inside
an async node and store `result.final_output` in state. See the complete
[team quick start](../MULTI_AGENT_TEAMS.md) for provider and member configuration.
The lower-level `AgentTeam.create` requires an existing orchestrator.

## Advanced Patterns

For an application-level timeout, wrap invocation in Python's `asyncio.timeout`:

```python
async with asyncio.timeout(30):
    result = await compiled.invoke(initial_state)
```

Check `result.success` and `result.error` before consuming outputs. A failed node must
not be treated as a successful workflow merely because partial state is available.
For retries around external operations, use the [resilience guide](../RESILIENCE.md).

## API Reference

The [workflow API reference](../../api-reference/workflows.md) describes result surfaces
for raw graphs and the compatibility adapter. The former extended DSL catalog is preserved
in [page history](https://github.com/anvai-labs/victor/commits/develop/docs/guides/workflow-development/dsl.md).
