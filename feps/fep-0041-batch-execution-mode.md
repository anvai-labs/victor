---
fep: "0041"
title: "Batch execution mode — the Code Mode insight adapted to victor"
type: Standards Track
status: Draft
created: 2026-10-27
modified: 2026-10-27
authors:
  - name: Vijaykumar Singh
    email: vijay@anvaiops.com
---

# FEP-0041: Batch Execution Mode

## Summary

Add an opt-in mode to the AgenticLoop where the PLAN step, upon identifying a well-defined sequential tool-call chain, emits a **batch plan** — a list of tool calls with their inputs and data-flow dependencies — that the ACT step executes **in one turn** without LLM round-trips between calls. The EVALUATE step runs after the batch completes, preserving error-catching and HITL gates. This adapts Cloudflare's Code Mode insight (blog.cloudflare.com/code-mode) to victor's architecture: reduce LLM round-trips for multi-step chains while keeping the agentic loop's safety properties (EVALUATE between batches, HITL gates, recovery). The key difference from Cloudflare's approach: victor uses structured plan output with input references rather than code generation in a sandbox, because victor's tools need real environment access and the existing tool pipeline already handles safety.

## Motivation

**The problem**: a 3-step chain (read → search → edit) currently costs 3 LLM calls, each with the full tool-schema payload and intermediate results passed through the model:

```
Turn 1: LLM sees tools + "read file A"    → tool_call(read, A)
Turn 2: LLM sees tools + result_A + "search B" → tool_call(search, B)
Turn 3: LLM sees tools + result_B + "edit C"   → tool_call(edit, C)
```

Each turn re-sends tool definitions (~1,600-2,300 tokens by default), re-processes intermediate results, and adds latency. For well-defined chains (the PLAN already knows what to do), this is pure overhead.

**What victor already has**:
- StateGraph workflows = "plans as code" that chain calls — but the AGENT LOOP doesn't opt into this for well-defined sub-tasks
- Tool pipeline with validation, retries, HITL gates — batch mode reuses it
- Per-turn tool selection with KV-cache-stable prefixes — batch mode preserves this (one stable prefix per batch, not per step)

**What we DON'T adopt from Code Mode**: V8 isolates (victor is Python-native), no-internet sandbox (victor's tools need real access), replacing the agent loop (EVALUATE/HITL are core features).

## Proposed Change

### The BatchPlan type

```python
@dataclass
class BatchStep:
    """One step in a batch plan."""
    tool_name: str
    inputs: Dict[str, Any]
    # Data-flow: input keys whose values come from a prior step's output
    # (e.g., {"file_path": "$step0.file_path"} means use step 0's output).
    input_refs: Dict[str, str] = field(default_factory=dict)
    description: str = ""  # for the audit log

@dataclass
class BatchPlan:
    """A PLAN-step output: a sequential chain of tool calls."""
    steps: List[BatchStep]
    rationale: str = ""  # why batching (for the audit log)
    stop_on_error: bool = True
```

### The flow

```
PLAN: identifies a chain → emits BatchPlan (or a normal plan)
  ↓
ACT: detects BatchPlan → executes all steps sequentially via ToolPipeline
     - resolves input_refs from prior step outputs
     - each step goes through the existing validation/HITL/retry pipeline
     - collects results + errors
  ↓
EVALUATE: sees the batch results (one evaluation, not N)
  ↓
DECIDE: continue / retry / complete (existing logic)
```

### Key design decisions

1. **Opt-in via plan structure, not a flag**: the PLAN step naturally emits either a normal plan (one action at a time) or a BatchPlan (chain of actions). The LLM chooses based on the task's shape.

2. **Input references use a simple substitution**: `$step0.key` syntax. No code generation, no sandboxing — the tool pipeline already handles safety.

3. **EVALUATE runs once per batch, not per step**: this is the win. But if a step fails, the batch stops (stop_on_error=True default), and EVALUATE sees the partial results + error.

4. **HITL gates work**: if any step in the batch requires approval, the batch pauses at that step, the approval happens, then the batch continues (or aborts).

5. **The tool schema payload is sent ONCE per batch**, not once per step: the LLM emits all steps in a single response.

## Benefits

- **Token savings**: tool schemas sent once per batch instead of once per step (proportional to chain length: 3-step chain saves ~2× tool-schema tokens)
- **Latency reduction**: N-step chain becomes 1 LLM turn instead of N (each ~1-3s saved)
- **Better tool handling**: the LLM sees the full chain upfront, reducing mid-chain confusion about what to do next
- **Preserves safety**: EVALUATE still runs (once per batch), HITL gates still work, recovery still applies
- **No breaking change**: opt-in via plan structure — existing single-step plans work unchanged

## Compatibility

- **Existing plans**: unchanged — a normal plan is just a BatchPlan with one step (or the existing single-action format)
- **Tool pipeline**: no changes to validation, retries, rate limiting, or HITL gates
- **EVALUATE/DECIDE**: batch results appear as a single evaluation unit (the loop's existing iteration model)
- **Recovery/spin guards**: unchanged — a batch is one iteration
- **Providers**: no provider changes needed (the batch is emitted by the PLAN step, not the provider)
- **MCP server**: no changes (batch mode is internal to the agent loop)

## Implementation Plan

### Phase 1: Types and plumbing (small)
- `victor/framework/batch_plan.py` — BatchStep, BatchPlan types
- Extend the PLAN step's output schema to allow BatchPlan
- AgenticLoop: detect BatchPlan in state["plan"], dispatch to batch executor

### Phase 2: Batch executor (medium)
- `victor/agent/services/batch_execution_runtime.py` — the executor:
  - Resolves input_refs from prior step outputs
  - Calls ToolPipeline._execute_single_call for each step
  - Handles stop_on_error, collects results
- Wire into the ACT step alongside the existing single-call executor

### Phase 3: Prompt guidance (small)
- System prompt addition: "When the task requires a well-defined sequential chain of tool calls, emit a batch plan"
- Few-shot examples in the prompt

## Drawbacks and Alternatives

1. **Full Code Mode (Cloudflare)**: generate TypeScript code, execute in a sandbox. REJECTED: victor is Python-native, tools need real environment access, and the EVALUATE step is a feature not overhead.

2. **StateGraph for everything**: use workflows for all tasks. REJECTED: workflows require explicit definition; batch mode is dynamic (the LLM decides per-task).

3. **Parallel tool calls**: some providers support multiple tool_calls per response. PARTIALLY ADOPTED: victor already supports this, but batch mode is different — it's SEQUENTIAL with data dependencies.

**Drawback of this approach**: the batch executor adds a new code path to the ACT step (complexity); the LLM must learn to emit BatchPlan (prompt engineering); error messages from a failed batch step may be less clear than per-step errors (mitigated by per-step audit logging).

## Migration Path

1. Land the types and executor (Phase 1-2) with NO prompt changes — batch mode is available but the LLM never triggers it
2. Add prompt guidance (Phase 3) behind a feature flag (`batch_execution_enabled`)
3. Monitor: track batch-plan emission rate, batch success rate, token savings
4. Graduate the flag when the data supports it

## References

- Cloudflare Code Mode: https://blog.cloudflare.com/code-mode (the inspiration; victor's approach differs in key ways)
- Victor's AgenticLoop: `victor/framework/agentic_loop.py` (the PERCEIVE → PLAN → ACT → EVALUATE → DECIDE cycle)
- Victor's ToolPipeline: `victor/agent/tool_pipeline.py` (validation, retries, HITL gates)
- Victor's StateGraph: `victor/framework/graph.py` (the existing "plans as code" that batch mode complements)
- Code Mode assessment in the strategic roadmap (2026-10-26 session)

## Unresolved Questions

1. **Should BatchPlan support conditional steps?** (e.g., "if step 0 returns empty, skip step 1") — deferred until the basic sequential version proves out.
2. **How to handle batch results in EVALUATE?** A batch produces N results but EVALUATE runs once — should it evaluate the chain outcome or each step? (Current design: the chain outcome, with per-step details in the audit log.)
3. **Interaction with tool selection**: batch mode sends tool schemas once, but per-turn tool selection narrows the surface — should the batch's tool schemas be narrowed too? (Current design: yes, narrowed to the tools referenced in the batch.)
4. **Provider-specific behavior**: do all providers handle the multi-step plan output format equally well? (Needs testing.)
