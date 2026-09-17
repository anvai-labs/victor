# YAML Workflow Syntax Reference

This reference describes the current `WorkflowDefinition` YAML loader. Start with the
[executable workflow tutorial](../tutorials/create-workflow.md), then use the
[workflow API reference](../api-reference/workflows.md) for Python integration.

## Basic structure

```yaml
workflows:
  greeting:
    description: A deterministic greeting
    start_node: greet
    nodes:
      - id: greet
        type: transform
        transform: make_greeting
        next: []
```

`workflows` maps names to definitions. `nodes` is a YAML list; the loader turns it into
a Python dictionary keyed by node ID. `start_node` defaults to the first node. Each
node's `next` list defines its successors; there is no separate top-level `edges` list.
A node with no successors terminates its path. There is no `type: end` node.

## Node Types

| Type | Configuration | Runtime requirement |
| --- | --- | --- |
| `agent` | `role`, `goal`, `tool_budget`, optional `input_mapping` and `output_key` | Configured sub-agent runtime and provider |
| `compute` | Tool or registered handler configuration | Corresponding tool or compute handler |
| `condition` | `condition`, `branches` | Registered condition or supported expression |
| `parallel` | `parallel_nodes`, `join_strategy` | Referenced child nodes |
| `transform` | `transform` | Registered function or supported transform expression |
| `hitl` | `hitl_type`, `prompt`, `fallback` | Application-managed human response and resume integration |
| `team` | Team configuration | Configured team runtime |

## Transform nodes

Prefer explicitly registered transforms for application logic. Pass a mapping such as
`transform_registry={"make_greeting": make_greeting}` to `compile_yaml`. A transform
receives the state mapping and returns a mapping of updates. It is not imported from
a `victor.workflows.transforms` module.

## Condition nodes

Register a synchronous function under the name used by `condition`. It returns a
branch name matching a `branches` key. Quote YAML boolean-like names if they are meant
to be strings. An unknown branch requires an explicit `default` route; otherwise it
fails rather than silently ending the workflow.

```yaml
- id: choose
  type: condition
  condition: select_route
  branches:
    ready: publish
    default: revise
```

The referenced `publish` and `revise` nodes must also be present in the definition.
Pass `condition_registry={"select_route": select_route}` when compiling.

## Parallel nodes

```yaml
- id: checks
  type: parallel
  parallel_nodes: [check_style, check_security]
  join_strategy: all
  next: [report]
```

Define each referenced child and successor node. Supported join strategies are `all`,
`any`, `first`, and `merge`; failure behavior depends on the selected strategy. This
node group is distinct from StateGraph `Send` fan-out. See the
[single-engine decision](../architecture/adr/030-single-graph-execution-engine.md).

## Execution limits

At workflow level, the loader accepts `max_execution_timeout_seconds`,
`default_node_timeout_seconds`, `max_iterations`, and `max_retries`. Scheduling and
workflow retry policy are described in the [scheduling guide](../guides/workflow-development/scheduling.md).

## Human-in-the-loop and checkpoints

HITL handler responses and graph execution checkpoints serve different purposes.
A saved graph checkpoint does not itself represent human approval. FEP-0032 tracks
the remaining general graph interrupt-after and paused-signal work; check the
[interrupt/resume design](https://github.com/anvai-labs/victor/blob/develop/feps/fep-0032-interrupt-resume-semantics.md)
and [HITL guide](../guides/HITL_WORKFLOWS.md) before relying on resume behavior.

## Execution results

The unified compiler returns a compiled graph wrapper. `await workflow.invoke(state)`
returns a graph result with `success`, `error`, and `state`. The legacy-compatible
workflow adapter instead returns `WorkflowResult`, whose data is `context.data`
(and its `final_state` compatibility property). Do not mix those result shapes.

Earlier examples are preserved in [page history](https://github.com/anvai-labs/victor/commits/develop/docs/user-guide/yaml_workflow_syntax.md).
