# Workflow Examples

Start with the [workflow tutorial](../../tutorials/create-workflow.md), the
[YAML syntax reference](../../user-guide/yaml_workflow_syntax.md), and the
[workflow API reference](../../api-reference/workflows.md).

Runnable repository examples and their setup instructions live in
[examples/workflows](https://github.com/anvai-labs/victor/tree/develop/examples/workflows).
Use the [scheduling guide](scheduling.md) for scheduled execution.

The earlier architecture research is preserved through the
[completed-work record](../../development/completed-work.md).
[ADR-030](../../architecture/adr/030-single-graph-execution-engine.md) owns the
completed engine migration; the BFS walker is deleted and streaming uses the
compiled graph engine. General interrupt/resume remains a separate FEP-0032 gate.
