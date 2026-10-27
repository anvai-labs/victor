# FEP-0041: Batch Execution Mode

**Status:** Proposed
**Date:** 2026-10-27
**Author:** Session (Code Mode assessment, strategic roadmap Phase 2)

## Summary

Add an opt-in mode to the AgenticLoop where the PLAN step, upon identifying
a well-defined sequential tool-call chain, emits a **batch plan** — a list
of tool calls with their inputs and data-flow dependencies — that the ACT
step executes **in one turn** without LLM round-trips between calls. The
EVALUATE step runs after the batch completes, preserving error-catching
and HITL gates.

This adapts Cloudflare's Code Mode insight to victor's architecture: reduce
LLM round-trips for multi-step chains while keeping the agentic loop's
safety properties (EVALUATE between batches, HITL gates, recovery).

## Motivation

**The problem**: a 3-step chain (read → search → edit) currently costs 3
LLM calls, each with the full tool-schema payload and intermediate results
passed through the model:

```
Turn 1: LLM sees tools + "read file A"    → tool_call(read, A)
Turn 2: LLM sees tools + result_A + "search B" → tool_call(search, B)
Turn 3: LLM sees tools + result_B + "edit C"   → tool_call(edit, C)
```

Each turn re-sends tool definitions (~1,600-2,300 tokens by default),
re-processes intermediate results, and adds latency. For well-defined
chains (the PLAN already knows what to do), this is pure overhead.

**What victor already has**:
- StateGraph workflows = "plans as code" that chain calls — but the AGENT
  LOOP doesn't opt into this for well-defined sub-tasks
- Tool pipeline with validation, retries, HITL gates — batch mode reuses it
- Per-turn tool selection with KV-cache-stable prefixes — batch mode
  preserves this (one stable prefix per batch, not per step)

**What we DON'T adopt from Code Mode**: V8 isolates (victor is Python-
native), no-internet sandbox (victor's tools need real access), replacing
the agent loop (EVALUATE/HITL are core features).

## Design

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

1. **Opt-in via plan structure, not a flag**: the PLAN step naturally
   emits either a normal plan (one action at a time) or a BatchPlan
   (chain of actions). The LLM chooses based on the task's shape.

2. **Input references use a simple substitution**: `$step0.key` syntax.
   No code generation, no sandboxing — the tool pipeline already handles
   safety.

3. **EVALUATE runs once per batch, not per step**: this is the win. But
   if a step fails, the batch stops (stop_on_error=True default), and
   EVALUATE sees the partial results + error.

4. **HITL gates work**: if any step in the batch requires approval, the
   batch pauses at that step, the approval happens, then the batch
   continues (or aborts).

5. **The tool schema payload is sent ONCE per batch**, not once per
   step: the LLM emits all steps in a single response.

### What stays the same

- Tool pipeline validation, retries, rate limiting
- HITL gates and approval flows
- EVALUATE and DECIDE logic
- Recovery and spin guards
- Observability (each step is logged)

## Implementation plan

### Phase 1: Types and plumbing (small)

- `victor/framework/batch_plan.py` — BatchStep, BatchPlan types
- Extend the PLAN step's output schema to allow BatchPlan
- AgenticLoop: detect BatchPlan in state["plan"], dispatch to batch
  executor

### Phase 2: Batch executor (medium)

- `victor/agent/services/batch_execution_runtime.py` — the executor:
  - Resolves input_refs from prior step outputs
  - Calls ToolPipeline._execute_single_call for each step
  - Handles stop_on_error, collects results
- Wire into the ACT step alongside the existing single-call executor

### Phase 3: Prompt guidance (small)

- System prompt addition: "When the task requires a well-defined
  sequential chain of tool calls, emit a batch plan"
- Few-shot examples in the prompt

## Alternatives considered

1. **Full Code Mode (Cloudflare)**: generate TypeScript code, execute in
   a sandbox. REJECTED: victor is Python-native, tools need real
   environment access, and the EVALUATE step is a feature not overhead.

2. **StateGraph for everything**: use workflows for all tasks. REJECTED:
   workflows require explicit definition; batch mode is dynamic (the LLM
   decides per-task).

3. **Parallel tool calls**: some providers support multiple tool_calls
   per response. PARTIALLY ADOPTED: victor already supports this, but
   batch mode is different — it's SEQUENTIAL with data dependencies.

## Consequences

- The agent loop's ACT step has two modes: single-call (existing) and
  batch (new)
- The PLAN step's output schema expands to include BatchPlan
- EVALUATE sees batch results as a single evaluation unit
- Tool schemas are sent once per batch (token savings proportional to
  chain length)
- No change to the tool pipeline, HITL, or recovery
