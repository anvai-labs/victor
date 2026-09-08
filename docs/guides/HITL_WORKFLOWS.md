# Human-in-the-Loop Workflows

Victor defines HITL nodes, human-input handlers, and graph checkpoints. These are
separate pieces: a checkpoint records execution progress, while a handler or application
supplies a human response. The remaining graph interrupt-after and paused-signal work
is tracked by FEP-0032; do not assume every execution surface supports a complete
approval-and-resume round trip.

## HITL Node Types

`victor.workflows.hitl.HITLNodeType` supports `approval`, `review`, `choice`, `input`,
and `confirmation`. There is no `override` node type.

## YAML configuration

This complete definition declares an approval point. It illustrates the loader's
schema, not a self-contained human-input application.

```yaml
workflows:
  approval_example:
    nodes:
      - id: approve
        type: hitl
        hitl_type: approval
        prompt: Approve the proposed change?
        context_keys: [changes_summary]
        timeout: 300
        fallback: abort
        next: []
```

Accepted fields also include `choices` and `default_value`. Fallback values are
`abort`, `continue`, `skip`, and `retry`. Use `fallback`, not `on_timeout`.
`next` is a successor list, not an approved/rejected routing dictionary. Add an explicit
condition node when your application needs response-dependent branches.

## Compiled execution

The native workflow compiler can configure `interrupt_before` for HITL nodes.
The resumed node executor reads the application's `_hitl_response` and records node
metadata. An explicitly rejected response with `fallback: abort` fails execution.
Supplying a checkpoint alone does not construct this response or prove human approval.

The presence of a `hitl_handler` argument on `WorkflowEngine` should not be treated
as proof that all YAML execution routes invoke that handler. Integrations must verify
their chosen execution and resume path. See the current
[interrupt/resume design](https://github.com/anvai-labs/victor/blob/develop/feps/fep-0032-interrupt-resume-semantics.md)
and [workflow API](../api-reference/workflows.md).

## Observability

Runtime instrumentation uses [topic-based events](observability/event-bus.md).
Application-facing streams use framework `EventType` values, a different event surface.

Earlier examples are preserved in [page history](https://github.com/anvai-labs/victor/commits/develop/docs/guides/HITL_WORKFLOWS.md).
