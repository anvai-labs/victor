# Edge Provider Tool Strategy - Analysis and Recommendations

> **Historical design/review snapshot.** The findings, proposed APIs, diagrams, and rollout
> status below describe their original review context. For current implementation and remaining
> work, use [Architecture](../architecture.md) and the [Roadmap](../roadmap.md); this document
> is not an installation guide or a certification that every proposed feature shipped.

## Current State: Partial Coverage ✅

The tool strategy **does cover edge providers**, but with important limitations.

### What Works ✅

**1. Context Window Detection**
```python
# Edge models are correctly identified
qwen3.5:2b → 8192 tokens (default)
```

**2. Context Budget Enforcement**
```python
# HARD CONSTRAINT: Tools ≤ 25% of context window
max_tool_tokens = int(8192 * 0.25)  # 2048 tokens

if tool_tokens > max_tool_tokens:
    tools = self._demote_tools_to_fit(tools, max_tool_tokens, context_window)
```

**3. Automatic Demotion**
- When tools exceed budget, low-priority tools are demoted to STUB
- Ensures edge models never exceed context limits

### What's Missing ⚠️

**1. Single Tier Assignment**
```yaml
# Current: One set of tiers for ALL providers
FULL: [read, shell, ls, code_search, edit, write, ...]  # 10 tools
```

**Problem**: Edge models need different tiers than main models.

**2. Reactive vs Proactive**
```python
# Current: Reactive (demote at runtime)
if tool_tokens > max_tool_tokens:
    tools = self._demote_tools_to_fit(tools, max_tool_tokens)

# Better: Proactive (different tiers per provider)
edge_tiers = {FULL: [read, shell], STUB: "*"}  # Pre-optimized
main_tiers = {FULL: [read, shell, ls, code_search, ...], STUB: "*"}
```

## Validation Results

### Edge Model: qwen3.5:2b (8192 tokens)

```
✅ Context window detected: 8192 tokens
✅ Tier assignments exist: 10 FULL tools
⚠️  WARNING: Tool set (2090 tokens) exceeds budget (2048) by 42 tokens
```

**Analysis**:
- Current FULL tools: 10 × 125 = 1250 tokens
- System prompt overhead: ~840 tokens
- **Total**: 2090 tokens
- **Budget**: 2048 tokens (25% of 8192)
- **Over budget by**: 42 tokens (2%)

### Cloud Model: claude-sonnet-4 (200000 tokens)

```
✅ Context window: 200000 tokens
✅ Tool set (2090 tokens) fits within budget (50000)
✅ Context utilization: 4.2%
```

**Analysis**:
- Plenty of headroom
- All 10 FULL tools fit easily
- No demotion needed

## Recommended Improvements

### Option 1: Provider-Specific Tiers (Recommended)

```yaml
# victor/config/tool_tiers.yaml
provider_tiers:
  edge:  # Models with <16K context
    FULL: [read, shell]  # Only 2 core tools
    COMPACT: []
    STUB: "*"

  standard:  # Models with 16K-128K context
    FULL: [read, shell, ls, code_search, edit, write]
    COMPACT: [test, find, symbol]
    STUB: "*"

  large:  # Models with >128K context
    FULL: [read, shell, ls, code_search, edit, write, test, find, symbol, _get_directory_summaries]
    COMPACT: []
    STUB: "*"

# Fallback for backward compatibility
default_tier: large
```

**Benefits**:
- Proactive optimization (no runtime demotion)
- Edge models get minimal tool set (faster decisions)
- Large models get full tool set (maximum capability)

### Option 2: Context-Based Tier Selection

```python
def select_tiers_for_context(context_window: int) -> str:
    """Select tier set based on context window size."""
    if context_window < 16384:
        return "edge"  # <16K: Minimal tools
    elif context_window < 128000:
        return "standard"  # 16K-128K: Medium tools
    else:
        return "large"  # >128K: Full tools
```

### Option 3: Keep Current + Add Edge-Specific Analysis

```bash
# Generate separate tier assignments for edge models
python -m victor.scripts.analyze_tool_usage \
    --days 30 \
    --context-window-min 0 \
    --context-window-max 16384 \
    --output tool_tiers_edge.yaml
```

## Edge Model Usage Patterns

### Current Assumptions
Edge models are used for **micro-decisions**:
- Task classification (task_type)
- Tool selection (tool_necessity)
- Intent detection (user_intent)
- Confidence scoring

### Edge Model Tool Needs
Based on micro-decision patterns, edge models need:
1. **read** - Access config/files for context
2. **shell** - Execute simple commands
3. **Minimal other tools** - Micro-decisions are focused

### Recommended Edge Tiers
```yaml
edge:
  FULL: [read, shell]  # 2 tools × 125 = 250 tokens
  COMPACT: []
  STUB: "*"
```

**Token Budget**:
- Context window: 8192 tokens
- Tool budget: 2048 tokens (25%)
- Tool tokens: 250 tokens
- **Utilization**: 12.2% (plenty of headroom)

## Implementation Priority

### High Priority (Do Now)
1. **Add edge model to context_window() mappings**
   ```python
   CONTEXT_WINDOWS = {
       "qwen3.5:2b": 8192,
       # ... other edge models
   }
   ```

2. **Document edge model limitations**
   - Current tiers work but require runtime demotion
   - Edge models should use minimal tool sets

### Medium Priority (Next Sprint)
1. **Implement provider-specific tier selection**
   - Add `provider_tiers` to tool_tiers.yaml
   - Select tiers based on context window size
   - Validate with edge model testing

2. **Generate edge-specific tier assignments**
   - Analyze tool usage for edge decisions only
   - Create separate tier file for edge models
   - Test with real edge model workloads

### Low Priority (Future)
1. **Automatic tier optimization**
   - ML model to predict optimal tiers per provider
   - Continuous learning from usage patterns
   - Auto-generate provider-specific tiers

## Testing Strategy

### Test Current Implementation
```bash
# Test with edge model
export VICTOR_TOOL_STRATEGY_V2=true
victor chat --provider ollama --model qwen3.5:2b

# Monitor for:
- Context budget warnings
- Tool demotion events
- Decision latency
- Tool selection accuracy
```

### Test Provider-Specific Tiers (Future)
```bash
# After implementing provider-specific tiers
export VICTOR_TOOL_STRATEGY_V2=true
export VICTOR_USE_PROVIDER_SPECIFIC_TIERS=true

# Test edge model
victor chat --provider ollama --model qwen3.5:2b
# Should use: 2 FULL tools (read, shell)

# Test cloud model
victor chat --provider anthropic --model claude-sonnet-4-20250514
# Should use: 10 FULL tools (full set)
```

## Recommendations

### Short Term (Current Release)
✅ **Keep current implementation** - It works correctly
- Edge models get automatic demotion when needed
- Context constraints are enforced
- No breaking changes

### Medium Term (Next Release)
🔄 **Add provider-specific tiers** - Better optimization
- Separate tier sets for edge/standard/large models
- Proactive vs reactive optimization
- Better performance for edge models

### Long Term (Future)
🚀 **Automatic tier generation** - Continuous improvement
- ML-based tier optimization
- Per-workload tier customization
- Real-time tier adaptation

## Conclusion

**Current Status**: Edge providers are **partially covered** ✅
- Context window detection works
- Budget enforcement works
- Automatic demotion works
- **But**: Single tier assignment is not optimal

**Recommendation**: Keep current for now, add provider-specific tiers in next sprint.

The current implementation is **safe and correct** (edge models won't exceed context), but **not optimal** (edge models could use fewer tools proactively).
