# Smart Model Routing

## Overview

Smart Model Routing is an intelligent provider selection system that automatically switches between local and cloud providers based on multiple factors:

- **Health**: Provider availability, circuit breaker state, recent errors
- **Resources**: GPU availability for local providers, API quota for cloud
- **Cost**: Local (free) vs cloud (paid) preferences
- **Latency**: Speed vs accuracy tradeoffs
- **Performance**: Historical success rates and latency trends

## Features

### Automatic Local→Cloud Fallback
When local providers (Ollama, LMStudio, vLLM) fail or become unavailable, smart routing automatically switches to cloud providers (Anthropic, OpenAI, DeepSeek).

### Adaptive Learning
The system learns from provider performance over time, routing requests to the best-performing providers based on:
- Success rate (40% weight)
- Average latency (30% weight)
- Latency trends (30% weight)

### Cost Optimization
Different routing profiles optimize for different priorities:
- **balanced**: Balance cost and performance
- **cost-optimized**: Minimize API costs
- **performance**: Prioritize speed and quality
- **local-first**: Use local providers when available

## Quick Start

### Enable Smart Routing

```bash
# Enable with balanced profile (default)
victor chat --enable-smart-routing

# Use cost-optimized profile
victor chat --enable-smart-routing --routing-profile cost-optimized

# Use custom fallback chain
victor chat --enable-smart-routing --fallback-chain ollama,deepseek,anthropic
```

### Routing Profiles

#### balanced (default)
Balance cost and performance with fallback chain:
- Default: `ollama → anthropic → openai`
- Coding: `ollama → deepseek → anthropic`
- Chat: `ollama → groqcloud → anthropic`

#### cost-optimized
Minimize API costs with fallback chain:
- Default: `ollama → lmstudio → deepseek → groqcloud`

#### performance
Prioritize speed and quality with fallback chain:
- Default: `anthropic → openai → ollama`

#### local-first
Use local providers when available with fallback chain:
- Default: `ollama → lmstudio → vllm`

## Configuration

### Environment Variables

```bash
# Enable smart routing by default
export VICTOR_SMART_ROUTING_ENABLED=true

# Set default profile
export VICTOR_SMART_ROUTING_PROFILE=balanced

# Performance tracking window
export VICTOR_SMART_ROUTING_PERFORMANCE_WINDOW=100
```

### Settings File (~/.victor/settings.yaml)

```yaml
smart_routing_enabled: true
smart_routing_profile: balanced
smart_routing_fallback_chain: null  # Or: ["ollama", "anthropic", "openai"]
smart_routing_performance_window: 100
smart_routing_learning_enabled: true
smart_routing_resource_awareness: true
```

### Custom Routing Profiles

Create `~/.victor/routing_profiles.yaml`:

```yaml
profiles:
  - name: my-profile
    description: "My custom routing profile"
    fallback_chains:
      default:
        - ollama
        - deepseek
        - anthropic
      coding:
        - ollama
        - anthropic
    cost_preference: low
    latency_preference: normal
```

## How It Works

### Routing Decision Factors

The routing engine scores each provider on 5 factors:

1. **Health (30% weight)**: Provider availability, circuit breaker state
   - 1.0: Healthy
   - 0.3: Non-critical issues
   - 0.0: Critical failure (API key missing, auth errors)

2. **Resources (25% weight)**: Resource availability
   - Local providers: GPU available?
   - Cloud providers: Always 1.0

3. **Cost (15% weight)**: Cost preference alignment
   - Local + low cost preference: 1.0
   - Cloud + high cost preference: 1.0
   - Mismatches: 0.3-0.7

4. **Latency (15% weight)**: Latency preference alignment
   - Cloud + low latency preference: 1.0
   - Local + high latency preference: 1.0
   - Mismatches: 0.3-0.7

5. **Performance (15% weight)**: Historical performance
   - Success rate, average latency, trends
   - 0.0 to 1.0 composite score

### Provider Selection Flow

```
User Request
    ↓
SmartRoutingProvider
    ↓
RoutingDecisionEngine.decide()
    ↓
1. Get candidates (from profile or custom chain)
2. Score each provider (5 factors)
3. Sort by score (descending)
4. Select best provider
5. Build fallback chain (remaining providers)
    ↓
ResilientProvider (retry + circuit breaker)
    ↓
Execute Request
    ↓
Record Metrics (for adaptive learning)
```

## Examples

### Example 1: Local→Cloud Fallback

```bash
# Start with Ollama (local)
victor chat --enable-smart-routing

# If Ollama crashes or runs out of memory:
# → Automatically switches to Anthropic
# → Continues conversation seamlessly
```

### Example 2: Cost Optimization

```bash
# Minimize API costs
victor chat --enable-smart-routing --routing-profile cost-optimized

# Prefers: Ollama (free) → LMStudio (free) → DeepSeek (cheap) → Groc (cheap)
# Only uses expensive providers if all free/cheap options fail
```

### Example 3: Performance Priority

```bash
# Prioritize speed
victor chat --enable-smart-routing --routing-profile performance

# Prefers: Anthropic (fast) → OpenAI (fast) → Ollama (slow)
# Best for time-sensitive tasks
```

### Example 4: Custom Fallback Chain

```bash
# Custom provider order
victor chat --enable-smart-routing \
  --fallback-chain ollama,deepseek,groqcloud,anthropic

# Uses providers in exact order specified
```

## Metrics and Observability

### View Routing Statistics

```python
from victor.providers.smart_router import SmartRoutingProvider

provider = SmartRoutingProvider(...)
stats = provider.get_stats()

print(stats["performance"])
# {
#     "window_size": 100,
#     "providers_tracked": 3,
#     "provider_stats": {
#         "ollama": {
#             "total_requests": 50,
#             "success_rate": 0.92,
#             "average_latency_ms": 1250.0,
#             "latency_trend": "improving",
#             "score": 0.78
#         },
#         ...
#     }
# }
```

### Routing Decision Logs

Smart routing logs each decision with:
- Selected provider
- Confidence score
- Fallback chain
- Rationale
- Individual factor scores

```
INFO:victor.providers.smart_router:Routing decision: ollama (confidence=0.75, fallbacks=2)
INFO:victor.providers.smart_router:Selected ollama (confidence=0.75) based on health (1.00)
INFO:victor.providers.smart_router:Smart routing success: ollama (1250ms, rationale: Selected ollama (confidence=0.75) based on health (1.00))
```

## Performance Considerations

### Routing Decision Latency
- Target: <10ms per decision
- Cached GPU status (30s TTL)
- Efficient scoring algorithm

### Performance Tracking Overhead
- Target: <1ms per request
- Sliding window (default: 100 requests)
- In-memory metrics storage

### Total Overhead
- Additional latency: ~11ms per request (routing + tracking)
- Memory: ~1KB per tracked provider
- Network: No additional network calls

## Best Practices

### 1. Choose the Right Profile

**For development** (cost-conscious):
```bash
victor chat --enable-smart-routing --routing-profile cost-optimized
```

**For production** (performance-focused):
```bash
victor chat --enable-smart-routing --routing-profile performance
```

**For mixed workloads**:
```bash
victor chat --enable-smart-routing --routing-profile balanced
```

### 2. Configure Provider API Keys

Ensure all providers in your fallback chain have API keys configured:

```bash
# Set API keys
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
export DEEPSEE_API_KEY="sk-..."

# Verify configuration
victor doctor --providers
```

### 3. Monitor Performance

Check routing metrics to ensure optimal performance:

```python
# Get performance stats
stats = smart_provider.get_stats()

# Check each provider's score
for provider, stats in stats["performance"]["provider_stats"].items():
    print(f"{provider}: score={stats['score']:.2f}, "
          f"success_rate={stats['success_rate']:.2f}, "
          f"latency={stats['average_latency_ms']:.0f}ms")
```

### 4. Explicit Provider Choice

Smart routing NEVER overrides explicit `--provider` choice:

```bash
# Explicit provider always wins
victor chat --provider anthropic --enable-smart-routing

# Uses Anthropic regardless of routing profile
```

## Troubleshooting

### Smart Routing Not Working

**Check if enabled**:
```bash
# Verify flag is set
echo $VICTOR_SMART_ROUTING_ENABLED  # Should be "true"

# Or check in logs
grep "smart routing" ~/.victor/logs/victor.log
```

**Verify fallback chain**:
```bash
# Check routing profiles exist
cat ~/.victor/routing_profiles.yaml

# Test with custom chain
victor chat --enable-smart-routing \
  --fallback-chain anthropic,openai
```

### Providers Not Switching

**Check provider health**:
```bash
# Run diagnostics
victor doctor --providers

# Check for API keys
victor doctor --check anthropic
```

**Verify GPU detection**:
```bash
# Check GPU status
victor doctor --hardware

# Test without GPU
export VICTOR_SMART_ROUTING_RESOURCE_AWARENESS=false
```

### Performance Issues

**Reduce tracking overhead**:
```yaml
# ~/.victor/settings.yaml
smart_routing_performance_window: 50  # Reduce from 100
smart_routing_learning_enabled: false  # Disable learning
```

**Use simpler profiles**:
```bash
# Use cost-optimized (simpler logic)
victor chat --enable-smart-routing \
  --routing-profile cost-optimized
```

## Advanced Usage

### Programmatic Usage

```python
from victor.providers.factory import ManagedProviderFactory
from victor.providers.routing_config import SmartRoutingConfig

# Create smart routing provider
provider = await ManagedProviderFactory.create_with_smart_routing(
    provider_name="ollama",  # Suggestion
    model="qwen3-coder:30b",
    routing_profile="balanced",
    performance_window_size=100,
    learning_enabled=True,
)

# Use like any other provider
response = await provider.chat(messages, model=model)

# Check stats
stats = provider.get_stats()
print(stats)
```

### Custom Scoring Weights

Modify factor weights in `victor/providers/smart_router.py`:

```python
# In _score_provider method
total_score += 0.4 * health_score      # Increase from 0.3
total_score += 0.2 * resource_score   # Decrease from 0.25
total_score += 0.1 * cost_score       # Decrease from 0.15
total_score += 0.1 * latency_score    # Decrease from 0.15
total_score += 0.2 * perf_score       # Increase from 0.15
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     SmartRoutingProvider                     │
│  Wraps multiple providers, adds intelligent routing          │
└─────────────────────────────────────────────────────────────┘
                              ↓ uses
┌─────────────────────────────────────────────────────────────┐
│                   RoutingDecisionEngine                       │
│  1. Get candidates (profile/custom chain)                     │
│  2. Score each provider (5 factors)                           │
│  3. Sort by score (descending)                                │
│  4. Select best + build fallback chain                       │
└─────────────────────────────────────────────────────────────┘
                              ↓ uses
┌─────────────────────────────────────────────────────────────┐
│              Performance Tracking Layer                         │
│  - ProviderPerformanceTracker                                  │
│  - RequestMetric (success, latency, error)                    │
│  - Composite scoring (40% success + 30% latency + 30% trend)  │
└─────────────────────────────────────────────────────────────┘
                              ↓ uses
┌─────────────────────────────────────────────────────────────┐
│               Resource Detection Layer                          │
│  - ResourceAvailabilityDetector                                │
│  - GPUAvailability (NVIDIA, Apple Silicon, AMD)               │
│  - QuotaInfo (API quota for cloud providers)                  │
└─────────────────────────────────────────────────────────────┘
                              ↓ uses
┌─────────────────────────────────────────────────────────────┐
│                  Existing Resilience Layer                     │
│  - ResilientProvider (retry + circuit breaker)                │
│  - ProviderHealthChecker (pre-flight checks)                 │
└─────────────────────────────────────────────────────────────┘
```

## Future Enhancements

- [ ] Predictive routing (anticipate failures before they happen)
- [ ] A/B testing framework for routing strategies
- [ ] Geographic routing (select providers based on latency)
- [ ] Load balancing across multiple instances
- [ ] Custom scoring functions per use case
- [ ] Routing rules DSL (if/then logic)

## See Also

- [Provider Resilience](../guides/RESILIENCE.md)
- [Circuit Breakers](../guides/RESILIENCE.md)
- [Performance Monitoring](../guides/observability/metrics.md)
- [Configuration Guide](../reference/settings-reference.md)
