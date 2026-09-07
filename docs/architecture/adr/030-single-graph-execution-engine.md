# ADR-030: Single Graph Execution Engine — retire the BFS WorkflowExecutor

## Metadata

- **Status**: Accepted
- **Date**: 2026-09-07
- **Decision Makers**: Vijaykumar Singh
- **Related ADRs/ FEPs**: FEP-0032 (graph interrupt/resume semantics), ADR-023 (multi-agent
  team durability — checkpoint/interrupt/streaming, the HITL features the surviving engine
  owns), FEP-0007 (Implemented — characterization-battery discipline)
- **Scope**: `victor/workflows/unified_executor.py` (the `CompiledWorkflowExecutor` BFS walker),
  its live call sites, and `batch_executor.py`. No YAML schema change; no
  `WorkflowDefinition` change.

## Context

U6-F1 (co-design review 2026-09-03) found **two live workflow engines** for the same
`WorkflowDefinition`: `CompiledWorkflowExecutor` (aliased `WorkflowExecutor`,
`unified_executor.py:456/:1095`, 1,095 lines) is a hand-rolled BFS DAG walker — own traversal
(`while to_execute:` :652), own ParallelNode gather (:849), own checkpoint shape — and is the
workflows package's canonical export; `StateGraphExecutor` (:144) compiles to `CompiledGraph`.
API routes, mode workflows, and the service provider use the BFS engine; chat UI and
`runtime.py` use StateGraph. This violates "StateGraph is always the execution engine"
(`docs/architecture.md:458`).

Verification on 2026-09-07 sharpened the finding in three ways:

1. **The engines share compilation but not execution.** The BFS walker already holds a
   `NativeWorkflowGraphCompiler` (:241-249) — it compiles identically, then executes with its
   own loop. And post-#1005, `unified_compiler.py` caches compiled graphs by (definition-content hash,
   node-callable fingerprint), so compilation cost — the engine split's original rationale — is
   cached away. What remains duplicated is the executor itself.
2. **`CompiledGraph` has already absorbed the hard features.** Interrupt/resume semantics
   (`interrupted`/`next_node`, FEP-0032), `InterruptConfig`, HITL, and checkpointing live on
   the CompiledGraph path (`victor/framework/graph.py:335-342`). The BFS walker's parallel
   gather still has the divergent failure semantics U6-F5 flagged.
3. **The live blast radius is three call sites plus one seam**, not an ecosystem:
   `workflow_routes.py:181` (`run_workflow`), `workflow_service_provider.py:479` (via the
   `compiled_executor.py` re-export shim), and the `runtime_executor_factory.py:38` seam;
   `streaming_executor.py:176` also reaches the walker via `create_legacy_workflow_executor`.
   (U6-F1's evidence listed `mode_workflows.py:164` and `specs/converter.py:66` as well, but
   both of those are docstring `Example:` blocks — the modules never import the executor; this
   ADR corrects the record.) The originally proposed remedy — make `CompiledWorkflowExecutor`
   a *facade* over CompiledGraph — is therefore more machinery than the situation needs.

## Decision

**Migrate the live call sites to the StateGraph/`CompiledGraph` execution path and delete the
BFS walker — no facade is built.** A facade over one engine preserves the facade's public name
while the engines still both exist in tree; deleting the walker actually reaches the
"StateGraph is always the execution engine" end state with less code.

Sequencing (each step landable, battery-gated):

1. **Parity gate first.** Extend the workflow test battery (the characterization discipline
   from FEP-0007) to run the live call sites' representative workflows through *both* engines
   and pin result equivalence — per-node results, parallel-branch merge outcomes, checkpoint
   shapes. Divergences found by this gate are fixed on the CompiledGraph side (this is where
   U6-F5's parallel-branch semantics converge, resolving that finding as a side effect).
2. **Migrate the live call sites** to the compiled path via the existing
   `runtime_executor_factory` seam. `batch_executor.py` follows.
3. **Delete the BFS walker** from `unified_executor.py` (keeping the module's `ExecutorResult`
   and HITL types, which FEP-0032 wires into) and re-point the workflows package's canonical
   export. `streaming_executor.py`'s BFS dependency moves to the compiled path or is retired
   with it, whichever its callers dictate.

## Consequences

- `unified_executor.py` shrinks from 1,095 lines to its result/type surface; the workflows
  package has exactly one execution engine, matching the architecture doc.
- The structural ratchet (#1024 pattern) gains a `unified_executor.py` line cap at the
  post-deletion size, may-only-shrink.
- Risk is concentrated in step 1's parity gate; the live call sites are all workflow-API
  surfaces (not the agent chat hot path), and the chat path already runs CompiledGraph.
- FEP-0032's interrupt/resume semantics land only on the surviving engine — no duplicate
  implementation across two walkers.
