# Workflows API Reference

Victor has one graph execution engine. YAML and `WorkflowDefinition` inputs compile
through `NativeWorkflowGraphCompiler` into `CompiledGraph`; `StateGraphWorkflowExecutor`
provides compatibility for callers that consume `WorkflowResult`.

## StateGraph API

```python
from victor.framework import StateGraph, END

graph = StateGraph(dict)
graph.add_node("answer", lambda state: {**state, "answer": 42})
graph.set_entry_point("answer")
graph.add_edge("answer", END)
compiled = graph.compile()
```

Use `add_conditional_edge` (singular) for branch routing. `await compiled.invoke(state)`
returns a result object. `compiled.stream(state)` yields `(node_id, state)` tuples.
See the [DSL guide](../guides/workflow-development/dsl.md) and
[executable tutorial](../tutorials/create-workflow.md).

## UnifiedWorkflowCompiler

Import `UnifiedWorkflowCompiler` from `victor.workflows.unified_compiler`.
`compile_yaml(path, workflow_name=None, condition_registry=None, transform_registry=None)`
loads a file and returns `CachedCompiledGraph`. `compile_definition(definition)`
compiles an existing `WorkflowDefinition`. Agent/team nodes require the compiler's
runtime dependencies; creating a bare compiler does not configure a provider.

## BaseYAMLWorkflowProvider

Verticals expose named workflow definitions through workflow providers. Client code
should normally use `await agent.run_workflow(name, context={...})` on a configured
`Agent`; the name identifies a vertical workflow, not an arbitrary YAML path.
See [Python API](../reference/api/python-api.md).

## Node Types API

`victor.workflows.definition` defines `WorkflowDefinition`, `AgentNode`,
`ComputeNode`, `ConditionNode`, `ParallelNode`, `TransformNode`, and `TeamStepWorkflow`.
`HITLNode` is defined in `victor.workflows.hitl`. Nodes use `id` and `next_nodes`;
`WorkflowDefinition.nodes` is a dictionary keyed by ID. `WorkflowBuilder` provides
programmatic construction. See [YAML node syntax](../user-guide/yaml_workflow_syntax.md#node-types).

## State Management

Graph nodes receive workflow state; dictionary nodes return state mappings. Definitions
compiled from YAML also keep internal execution metadata. Application code should use
its own named state keys and avoid relying on underscore-prefixed engine metadata.

## Execution Results

| Execution API | Result | User data |
| --- | --- | --- |
| `CompiledGraph.invoke` / `CachedCompiledGraph.invoke` | `GraphExecutionResult` | `result.state` |
| `StateGraphExecutor.execute` | `ExecutorResult` | `result.state` |
| `StateGraphWorkflowExecutor.execute` | `WorkflowResult` | `result.context.data` / `result.final_state` |

All expose `success` and `error`. On `WorkflowResult`, per-node results are in
`result.context.node_results`; durations on node results use seconds. Do not index
a graph result directly as a dictionary or assume a `final_output` field exists on
`WorkflowResult`.

## Configuration

Execution limits belong to workflow/compiler configuration. Checkpoint settings depend
on the selected checkpointer. The remaining interrupt/resume changes are tracked by
FEP-0032 and must not be inferred from the presence of a checkpoint alone.

## Compatibility Aliases

`CompiledWorkflowExecutor` and `WorkflowExecutor` remain compatibility import names
for the StateGraph adapter. They no longer select the deleted BFS walker. Cache
infrastructure remains available independently; the retired executor's `execute_by_name`
and private traversal methods are not part of the adapter API.

See [ADR-030](../architecture/adr/030-single-graph-execution-engine.md) for the migration
record. The previous catalog is available in [page history](https://github.com/anvai-labs/victor/commits/develop/docs/api-reference/workflows.md).
