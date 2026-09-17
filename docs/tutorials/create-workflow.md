# Creating Workflows in Victor

Build a small workflow without configuring an LLM, then express the same computation in YAML.
Victor compiles workflow definitions to `CompiledGraph`; the retired breadth-first executor
is no longer a separate execution engine. See [ADR-030](../architecture/adr/030-single-graph-execution-engine.md).

## Prerequisites

Install Victor using the [installation guide](../getting-started/installation.md).
The examples below require Python 3.11 or later and no provider credentials.

## 1. Build a StateGraph

A node receives state and returns an updated mapping. `invoke()` returns a result object;
read its `state` field for the final data.

```python
import asyncio
from victor.framework import StateGraph, END

async def main():
    graph = StateGraph(dict)
    graph.add_node("greet", lambda state: {**state, "greeting": f"Hello, {state['name']}!"})
    graph.set_entry_point("greet")
    graph.add_edge("greet", END)
    result = await graph.compile().invoke({"name": "Victor"})
    assert result.success, result.error
    print(result.state["greeting"])

asyncio.run(main())
```

## 2. Write a YAML definition

Save this as `greeting.yaml`. YAML files contain a `workflows` mapping; edges are
specified with each node's `next` list. A terminal node has no successors.

```yaml
workflows:
  greeting:
    description: A deterministic greeting
    nodes:
      - id: greet
        type: transform
        transform: make_greeting
        next: []
```

## 3. Register the transform and execute

Registered functions provide the Python behavior referenced by YAML. They receive the
workflow state and return a mapping of state updates.

```python
import asyncio
from victor.workflows.unified_compiler import UnifiedWorkflowCompiler

def make_greeting(state):
    return {"greeting": f"Hello, {state['name']}!"}

async def main():
    compiler = UnifiedWorkflowCompiler()
    workflow = compiler.compile_yaml(
        "greeting.yaml",
        workflow_name="greeting",
        transform_registry={"make_greeting": make_greeting},
    )
    result = await workflow.invoke({"name": "Victor"})
    assert result.success, result.error
    print(result.state["greeting"])

asyncio.run(main())
```

## Branching and parallel work

For Python graphs use `add_conditional_edge` (singular) with a routing function and
an explicit branch mapping. YAML condition nodes use `branches`, while parallel nodes
name their children in `parallel_nodes`. See the [YAML syntax reference](../user-guide/yaml_workflow_syntax.md)
and [StateGraph DSL guide](../guides/workflow-development/dsl.md).

## Agent nodes and built-in workflows

Agent and team nodes need a configured runtime, including an orchestrator and provider.
The bare compiler in this tutorial supplies neither. For a configured vertical's named
workflow use `await agent.run_workflow("workflow_name", context={...})`; this method
resolves a built-in name, not a YAML file path. See the [Python API](../reference/api/python-api.md).

HITL nodes need application-managed human responses and checkpoint integration. General
graph interrupt-after/resume changes remain tracked in FEP-0032; this tutorial does
not assume that an arbitrary checkpoint supplies an approval response.

## Next steps

- [Workflow API reference](../api-reference/workflows.md)
- [Scheduling and versioning](../guides/workflow-development/scheduling.md)
- [Multi-agent teams](../guides/MULTI_AGENT_TEAMS.md)
