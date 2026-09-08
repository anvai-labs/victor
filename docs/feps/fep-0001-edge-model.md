---
fep: "0001"
title: "Edge Model for Micro-Decisions and Token Reduction"
type: Standards Track
status: Draft
created: 2026-04-03
modified: 2026-04-03
authors:
  - name: "Vijaykumar Singh"
reviewers: []
discussion: null
implementation: null
---

# FEP-0001: Edge Model for Micro-Decisions and Token Reduction

> **Legacy proposal series.** This document retains the old `docs/feps` number 0001
> as historical context. It is distinct from canonical
> [FEP-0001](https://github.com/anvai-labs/victor/blob/develop/feps/fep-0001-fep-process.md) in the root `feps/` series.
> Its Draft metadata is preserved. See [FEP-0012](https://github.com/anvai-labs/victor/blob/develop/feps/fep-0012-shipped-edge-classifier.md)
> for the related current proposal or decision, and the [canonical FEP index](https://github.com/anvai-labs/victor/blob/develop/feps/README.md).


## Abstract

Introduce a lightweight "edge model" (50-100M parameters, running locally on CPU) that handles micro-decisions throughout the agent lifecycle — task classification, tool selection, completion detection, system prompt construction, and loop prevention. This eliminates hundreds of millions of tokens/year in cloud LLM costs while improving latency for sub-second decisions.

## Motivation

Victor currently makes 7-15 micro-decisions per conversation turn using either:
- **Regex/keyword heuristics** — fast but brittle (fail on complex prompts like SWE-bench issues)
- **Cloud LLM calls** — accurate but expensive (40-60 tokens per decision, 2-5s latency)

Neither is optimal. A local edge model running on CPU would be:
- **Faster than cloud LLM** (<100ms vs 2-5s)
- **Smarter than regex** (understands context, handles ambiguity)
- **Zero marginal cost** (no API tokens consumed)
- **Privacy-preserving** (no prompt data leaves the machine)

### Token Churn Analysis

| Decision Point | Calls/Turn | Current Tokens/Call | Turns/Session | Annual Est. |
|---------------|-----------|-------------------|--------------|------------|
| Tool schema broadcast (15 tools) | 1 | 3,750 | 10 | 200M+ tokens |
| System prompt | 1 | 2,500 | 10 | 150M+ tokens |
| Task type classification | 1-2 | 50 | 3 | 20M tokens |
| Completion detection | 1-3 | 40 | 8 | 50M tokens |
| Loop/stalling detection | 0-2 | 40 | 5 | 25M tokens |
| Error classification | 0-1 | 35 | 2 | 15M tokens |
| Continuation strategy | 0-2 | 45 | 4 | 20M tokens |
| Stage detection | 1 | 30 | 8 | 10M tokens |
| **Total** | | | | **~500M tokens/year** |

The largest savings come from **tool selection** (broadcasting 15 tool schemas = 3,750 tokens per request) and **system prompt** (2,500 tokens repeated every turn). An edge model that selects the right 5 tools and constructs a focused system prompt eliminates ~80% of per-turn token overhead.

## Design

### Architecture

```
User Request
    |
    v
[Edge Model] ──> Task Classification (fix/create/analyze/explain)
    |              Expected Deliverables (file_modified, analysis_provided)
    |              Complexity (simple/medium/complex)
    |
    +──> Tool Selection ──> Top 5 tools (from 48 available)
    |                        Filtered schemas sent to cloud model
    |
    +──> System Prompt Construction ──> Focused prompt (500 tokens vs 2,500)
    |                                    Task-specific guidance only
    |
    +──> During Execution:
         +──> Completion Detection (is task done?)
         +──> Loop Detection (is agent stuck?)
         +──> Error Classification (retry or abort?)
         +──> Stage Transition (reading → execution → verification)
         +──> Continuation Strategy (what to do next?)
    |
    v
[Cloud Model] ──> Receives: focused prompt + 5 tool schemas + user request
                   Saves: ~5,000 tokens per turn
```

### Edge Model Specification

#### Phase 1: Use Existing Small Models (configurable)

Use Ollama-served small models as the edge decision engine:

```yaml
# victor settings
edge_model:
  provider: ollama
  model: tinyllama:1.1b     # Default: TinyLlama 1.1B (1.1GB)
  fallback_model: qwen2.5-coder:1.5b  # Fallback: Qwen2.5 Coder 1.5B
  timeout_ms: 500            # Hard timeout — fall back to heuristic
  max_tokens: 30             # Micro-decisions need tiny responses
  temperature: 0.0           # Deterministic classification
  enabled: true              # Feature flag
  cache_ttl: 120             # Cache identical decisions for 2 minutes
```

#### Phase 2: Custom Victor Edge Model (future)

Train a purpose-built model optimized for Victor's decision vocabulary:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Architecture | Transformer decoder-only | Standard, well-supported |
| Parameters | 50-100M | CPU-viable, <500MB disk |
| Context window | 512 tokens | Micro-decisions need minimal context |
| Vocabulary | 8K tokens | Domain-specific (code + Victor concepts) |
| Training data | Victor decision logs | Real classification examples |
| Training method | SFT + CoT distillation | Chain-of-thought from large model |
| Quantization | INT4 (GGUF) | 25-50MB final model size |
| Inference | llama.cpp / ONNX Runtime | No GPU required |

### Decision Types

The edge model handles 10 decision types, each with a structured JSON output:

```
1. TASK_CLASSIFICATION    → {task_type, deliverables[], complexity, confidence}
2. TOOL_SELECTION         → {tools: ["read", "edit", "grep"], confidence}
3. COMPLETION_DETECTION   → {is_complete, phase, confidence}
4. LOOP_DETECTION         → {is_loop, loop_type, confidence}
5. ERROR_CLASSIFICATION   → {error_type, confidence}
6. STAGE_TRANSITION       → {stage, confidence}
7. CONTINUATION_ACTION    → {action, reason}
8. PROMPT_FOCUS           → {sections: ["grounding", "completion"], tools_guidance: bool}
9. INTENT_CLASSIFICATION  → {intent, confidence}
10. TOOL_CALL_VALIDATION  → {valid, corrections[]}
```

### Integration Points

#### 1. EdgeModelProvider (new component)

```python
# victor/agent/edge_model.py

class EdgeModelProvider:
    """Lightweight LLM provider for micro-decisions.
    
    Runs locally via Ollama or embedded inference runtime.
    Falls back to heuristic if unavailable or times out.
    """
    
    async def classify(
        self,
        decision_type: DecisionType,
        context: dict,
        timeout_ms: int = 500,
    ) -> DecisionResult:
        """Make a micro-decision with hard timeout.
        
        Priority chain:
        1. Cache hit → instant return
        2. Edge model call → <500ms
        3. Heuristic fallback → 0ms
        """

    def classify_sync(self, ...) -> DecisionResult:
        """Synchronous version for hot paths."""
```

#### 2. Replace LLMDecisionService Backend

The existing `LLMDecisionService` already has the right architecture. The change is minimal:

```python
# Current: uses cloud LLM for micro-decisions
decision_service = LLMDecisionService(provider=cloud_provider)

# New: uses edge model, falls back to cloud only if edge unavailable  
decision_service = LLMDecisionService(provider=edge_model_provider)
```

#### 3. Tool Selection Filter

```python
# Before cloud LLM call:
selected_tools = edge_model.classify(
    DecisionType.TOOL_SELECTION,
    context={
        "user_message": message[:200],
        "available_tools": tool_names,  # 48 names (not schemas!)
        "stage": current_stage,
        "recent_tools": last_3_tools,
    }
)
# Result: {"tools": ["read", "edit", "grep", "ls", "shell"], "confidence": 0.85}
# Only send these 5 tool schemas to cloud model (saves 3,000+ tokens)
```

#### 4. System Prompt Construction

```python
# Edge model decides which prompt sections are relevant:
prompt_focus = edge_model.classify(
    DecisionType.PROMPT_FOCUS,
    context={
        "user_message": message[:200],
        "task_type": classification.task_type,
        "stage": current_stage,
    }
)
# Result: {"sections": ["grounding", "tool_guidance"], "tools_guidance": true}
# Build focused prompt with only relevant sections (500 vs 2,500 tokens)
```

## Training Plan (Phase 2)

### Data Collection

1. **Decision logs**: Every `decide_sync()` call logs input context + cloud LLM output
2. **Tool selection traces**: Log which tools were actually used vs broadcast
3. **Completion accuracy**: Log when `should_stop()` was correct vs premature
4. **User corrections**: When user overrides agent behavior, log the correction

### Dataset Structure

```jsonl
{"input": "Classify: Fix the separability_matrix bug...", "output": {"task_type": "action", "deliverables": ["file_modified"], "confidence": 0.95}, "decision_type": "TASK_CLASSIFICATION"}
{"input": "Select tools for: Create a new cache manager", "output": {"tools": ["write", "ls", "shell"], "confidence": 0.88}, "decision_type": "TOOL_SELECTION"}
{"input": "Is complete? Response: **DONE**: Fixed auth.py", "output": {"is_complete": true, "phase": "done", "confidence": 0.97}, "decision_type": "COMPLETION_DETECTION"}
```

### Training Pipeline

```
Phase 1: Collect 10K decision examples from cloud LLM
Phase 2: Distill CoT reasoning from large model (Claude/GPT-4)
Phase 3: SFT on (input, output) pairs
Phase 4: RLHF/DPO with user correction data
Phase 5: Quantize to INT4 GGUF (25-50MB)
Phase 6: Evaluate on holdout set (target: 90%+ accuracy)
```

### CoT Distillation Example

```
Cloud model (teacher):
  Input: "Classify: Modeling's separability_matrix does not compute..."
  CoT: "This describes a bug where a function produces incorrect output.
        The user expects the code to be fixed. This requires modifying
        the existing separable.py file. Task type: action.
        Deliverable: file_modified. Complexity: complex (nested models)."
  Output: {"task_type": "action", "deliverables": ["file_modified"], ...}

Edge model (student):
  Learns to produce the same output without explicit CoT at inference time
  (CoT is used during training only for better generalization)
```

## Configuration

```yaml
# ~/.victor/settings.yaml
edge_model:
  enabled: true
  provider: ollama           # ollama | embedded | disabled
  model: tinyllama:1.1b      # Any Ollama model
  timeout_ms: 500            # Max latency before heuristic fallback
  max_tokens: 30             # Micro-response budget
  cache_ttl: 120             # Dedup identical decisions (seconds)
  confidence_threshold: 0.6  # Below this, use heuristic fallback
  
  # Decision-specific overrides
  decisions:
    tool_selection:
      enabled: true
      max_tools: 6            # Max tools to recommend
    prompt_focus:
      enabled: true
      max_sections: 4         # Max prompt sections to include
    completion_detection:
      enabled: true
    loop_detection:
      enabled: true
```

## Implementation Phases

### Phase 1: Configurable Edge Model via Ollama (2 weeks)

1. Add `EdgeModelProvider` class with Ollama backend
2. Wire into `LLMDecisionService` as provider option
3. Add `edge_model` settings section
4. Add tool selection filter (15 → 5 tools)
5. Test with `tinyllama:1.1b` and `qwen2.5-coder:1.5b`

**Deliverables**: Edge model runs all 10 decision types via Ollama, with heuristic fallback.

### Phase 2: System Prompt Optimization (1 week)

1. Implement `PROMPT_FOCUS` decision type
2. Build section-based prompt assembly (only include relevant sections)
3. Measure token savings vs accuracy impact

**Deliverables**: System prompts reduced from 2,500 to ~500 tokens.

### Phase 3: Data Collection Pipeline (1 week)

1. Add decision logging to all `decide_sync()` calls
2. Log tool selection traces (broadcast vs used)
3. Log completion accuracy (premature vs correct)
4. Export to training-ready JSONL format

**Deliverables**: Decision log pipeline producing 1K+ examples/day.

### Phase 4: Custom Model Training (4 weeks)

1. Collect 10K+ labeled decision examples
2. Distill CoT from Claude/GPT-4 for training
3. Fine-tune 50M param model (e.g., SmolLM-135M base)
4. Quantize to INT4 GGUF
5. Benchmark: accuracy, latency, model size
6. Ship as `victor-edge-model` package

**Deliverables**: Custom 25-50MB model with 90%+ accuracy on Victor decisions.

### Phase 5: Embedded Inference (2 weeks)

1. Integrate llama.cpp or ONNX Runtime for zero-dependency inference
2. Ship model as part of `victor-ai` package
3. Auto-download on first use (like sentence-transformers models)
4. Benchmark CPU inference latency (<100ms target)

**Deliverables**: Edge model runs without Ollama, pure Python/C++ inference.

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Token reduction per session | 40%+ | Before/after token logging |
| Decision accuracy | 90%+ | Holdout set evaluation |
| Decision latency | <100ms (p95) | Edge model inference timing |
| Model size | <50MB (INT4) | GGUF file size |
| CPU inference | No GPU required | Run on M1/x86 CPU |
| Fallback rate | <10% | Fraction of decisions falling to heuristic |

## Backwards Compatibility

- `edge_model.enabled: false` (default) preserves current behavior
- All decisions maintain heuristic fallback — no hard dependency on edge model
- Existing `LLMDecisionService` API unchanged — only the backend provider changes
- Feature flag: `USE_EDGE_MODEL` in `FeatureFlag` enum

## Security Considerations

- Edge model runs locally — no data exfiltration risk
- Model weights are open-source and auditable
- No internet access required for inference
- Decision outputs are structured JSON with Pydantic validation

## Rejected Alternatives

1. **Use cloud model for all decisions**: Current approach. Too expensive at scale.
2. **Pure regex/heuristic**: Already proven brittle for complex prompts (SWE-bench).
3. **Embedding-only classification**: Good for similarity but can't generate structured output.
4. **WASM-based inference**: Not mature enough for production; llama.cpp is better supported.
