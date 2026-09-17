# Edge Model Guide

**Fast, token-free micro-decisions using local LLMs.**

## Overview

The edge model system enables Victor to make lightweight decisions using a local LLM (via Ollama) instead of making expensive cloud API calls. This reduces both token costs and latency for micro-decisions like:

- **Task classification** — What type of task is this? (fix, create, analyze, explain)
- **Tool selection** — Which tools are most relevant for this request?
- **Prompt focus** — Which sections of the system prompt are needed?
- **Completion detection** — Is the task finished?
- **Error classification** — What type of error occurred?

## Benefits

| Benefit | Description |
|---------|-------------|
| **Token savings** | ~500M tokens/year saved by avoiding cloud LLM micro-decisions |
| **Lower latency** | <2s for edge decisions vs 2-5s for cloud API calls |
| **Privacy** | No prompt data leaves your machine |
| **Cost** | Zero marginal cost after initial model download |
| **Offline capable** | Works in air-gapped environments with local models |

## Quick Start

### 1. Install Ollama

```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh
```

### 2. Pull a Small Model

```bash
# Recommended: Qwen2.5 Coder 1.5B (1.1GB, fast and code-focused)
ollama pull qwen2.5-coder:1.5b

# Alternative: TinyLlama 1.1B (even smaller)
ollama pull tinyllama:1.1b
```

### 3. Enable Edge Model

```yaml
# ~/.victor/settings.yaml
edge_model:
  enabled: true
  provider: ollama
  model: qwen2.5-coder:1.5b
```

### 4. Start Victor

```bash
victor chat --provider ollama --model qwen2.5-coder:1.5b
```

Victor will now use the edge model for micro-decisions automatically.

## Configuration

### Full Configuration Options

```yaml
edge_model:
  # Master toggle
  enabled: true

  # Provider backend
  provider: ollama           # Currently only "ollama" or "disabled" supported
  model: qwen2.5-coder:1.5b  # Any Ollama-served model

  # Performance tuning
  timeout_ms: 2000           # Hard timeout before fallback (milliseconds)
  max_tokens: 50             # Max response tokens (micro-decisions need 30-60)
  cache_ttl: 120             # Cache identical decisions for 2 minutes

  # Fallback behavior
  confidence_threshold: 0.6  # Below this, use heuristic fallback

  # Ollama connection
  base_url: http://localhost:11434  # Ollama server URL

  # Feature-specific settings
  tool_selection_enabled: true   # Use edge model for tool ranking
  prompt_focus_enabled: true      # Use edge model for prompt trimming
  max_tools: 6                    # Max tools to recommend
```

### Recommended Models

| Model | Size | Speed | Best For | Download Command |
|-------|------|-------|----------|------------------|
| **qwen2.5-coder:1.5b** | 1.1GB | Fast | Code tasks | `ollama pull qwen2.5-coder:1.5b` |
| **tinyllama:1.1b** | 700MB | Very fast | General purpose | `ollama pull tinyllama:1.1b` |
| **qwen2.5-coder:3b** | 2GB | Medium | Better accuracy | `ollama pull qwen2.5-coder:3b` |
| **phi3:mini** | 2.3GB | Medium | Complex reasoning | `ollama pull phi3:mini` |

## Decision Types

The edge model handles 10 decision types, each with structured JSON output:

### 1. Task Type Classification

```python
# Input: User message excerpt
# Output: Task classification with deliverables
{
    "task_type": "action",  # action|create|analyze|explain
    "deliverables": ["file_modified"],
    "complexity": "medium",  # simple|medium|complex
    "confidence": 0.95
}
```

### 2. Tool Selection

```python
# Input: User request, available tools, stage, recent tools
# Output: Top 5-6 most relevant tools
{
    "tools": ["read", "edit", "grep", "ls", "shell"],
    "confidence": 0.88
}
```

**Token Savings**: Instead of broadcasting all 48 tool schemas (~8,000 tokens), only the selected 5-6 tools are sent to the cloud model (~900 tokens).

### 3. Prompt Focus

```python
# Input: Task type, user request excerpt
# Output: Which prompt sections to include
{
    "sections": ["grounding", "tool_guidance", "completion"],
    "confidence": 0.82
}
```

**Token Savings**: Reduces system prompt from 2,500 tokens to ~500 tokens by including only relevant sections.

### 4-10. Other Decision Types

- **Completion Detection** — Is the task finished?
- **Loop Detection** — Is the agent stuck in a loop?
- **Error Classification** — What type of error occurred?
- **Stage Transition** — Move to next execution stage?
- **Continuation Action** — What to do next?
- **Intent Classification** — User's primary intent?
- **Tool Call Validation** — Are the proposed tool calls valid?

## Usage Patterns

### For Framework Developers

```python
from victor.agent.edge_model import create_edge_decision_service
from victor.agent.services.protocols.decision_service import DecisionType

# Create edge decision service
service = create_edge_decision_service()
if service:
    # Make synchronous decision (fast path)
    result = service.decide_sync(
        DecisionType.TASK_TYPE_CLASSIFICATION,
        context={
            "message_excerpt": "Fix the authentication bug in auth.py",
            "stage": "planning",
        }
    )
    print(f"Task type: {result.task_type}")
    print(f"Confidence: {result.confidence}")
else:
    # Fall back to heuristic or cloud LLM
    result = heuristic_classify(message)
```

### For Vertical Developers

Edge model integration is automatic for verticals using the standard `VerticalBase` class. The edge model will:

1. Classify user requests into your vertical's task types
2. Select tools from your vertical's tool set
3. Optimize system prompts for your domain

```python
from victor_contracts import VerticalBase

class MyVertical(VerticalBase):
    name = "my_vertical"
    description = "My custom vertical"

    def get_tools(self):
        return ["tool1", "tool2", "tool3"]

    # Edge model automatically selects from these tools
    # based on user request context
```

## Performance Tuning

### Latency vs Accuracy

```yaml
# Faster but less accurate (for simple tasks)
edge_model:
  model: tinyllama:1.1b
  timeout_ms: 1000
  confidence_threshold: 0.5

# Slower but more accurate (for complex tasks)
edge_model:
  model: qwen2.5-coder:3b
  timeout_ms: 5000
  confidence_threshold: 0.7
```

### Cache Tuning

```yaml
# Longer cache = faster but stale decisions
edge_model:
  cache_ttl: 300  # 5 minutes (default: 120)

# Shorter cache = fresh but slower
edge_model:
  cache_ttl: 30   # 30 seconds
```

### Fallback Behavior

If the edge model is unavailable or times out, Victor automatically falls back to:

1. **Heuristic rules** — Fast regex/keyword based classification
2. **Cloud LLM** — As last resort for complex decisions

```yaml
# Adjust when fallback occurs
edge_model:
  timeout_ms: 2000              # Fallback after 2 seconds
  confidence_threshold: 0.6     # Fallback below 60% confidence
```

## Troubleshooting

### Edge Model Not Working

**Symptom**: Decisions still using cloud LLM

**Solutions**:
1. Check Ollama is running: `ollama list`
2. Verify model is pulled: `ollama pull qwen2.5-coder:1.5b`
3. Check logs: `victor chat --log-level debug`
4. Verify config: `cat ~/.victor/settings.yaml | grep edge_model`

### Model Not Found

**Symptom**: `Edge model 'qwen2.5-coder:1.5b' not available in Ollama`

**Solution**:
```bash
ollama pull qwen2.5-coder:1.5b
```

### Timeout Errors

**Symptom**: Edge model decisions timing out

**Solutions**:
1. Increase timeout: `timeout_ms: 5000`
2. Use smaller model: `model: tinyllama:1.1b`
3. Check Ollama is responsive: `ollama ps`

## Advanced Topics

### Custom Decision Prompts

The edge model uses structured prompts for each decision type. You can customize these in your vertical:

```python
from victor.agent.edge_model import TOOL_SELECTION_PROMPT

# Extend the default prompt
CUSTOM_TOOL_PROMPT = TOOL_SELECTION_PROMPT + """

Additional context for my vertical:
- Prefer tools that start with "my_"
- Avoid "shell" unless explicitly requested
"""
```

### Feature Flag Control

```python
from victor.core.feature_flags import FeatureFlag, feature_flags

# Check if edge model is enabled
if feature_flags.is_enabled(FeatureFlag.USE_EDGE_MODEL):
    # Use edge model
    pass
else:
    # Use fallback
    pass
```

### Monitoring and Metrics

The edge model logs decision accuracy and latency. Enable detailed logging:

```yaml
# ~/.victor/settings.yaml
logging:
  level: DEBUG
  modules:
    - victor.agent.edge_model
    - victor.agent.services.decision_service
```

## Best Practices

1. **Start with defaults** — The default configuration works well for most use cases
2. **Monitor fallback rate** — If >10% of decisions fall back, increase `timeout_ms` or use a larger model
3. **Profile before optimizing** — Measure token savings before tuning cache TTL
4. **Use appropriate model size** — 1.5B models are sufficient for most micro-decisions
5. **Keep Ollama updated** — New versions often include performance improvements

## Future Enhancements

- **Phase 2**: System prompt optimization (in progress)
- **Phase 3**: Data collection pipeline for custom training
- **Phase 4**: Custom Victor edge model (50M parameters)
- **Phase 5**: Embedded inference (no Ollama dependency)

See [FEP-0001](../feps/fep-0001-edge-model.md) for full roadmap.

## References

- [FEP-0001: Edge Model for Micro-Decisions](../feps/fep-0001-edge-model.md)
- [Feature Flags Documentation](./FEATURE_FLAGS.md)
- [Configuration Guide](../reference/configuration-options.md)
- [Local Models Guide](./development/local-models.md)
