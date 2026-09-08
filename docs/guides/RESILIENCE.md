# Resilience Patterns Guide

This guide covers Victor's resilience infrastructure for building fault-tolerant agent applications.

## Overview

Victor provides production-grade resilience patterns:

- **Circuit Breaker**: Prevent cascade failures
- **Retry Strategies**: Exponential, linear, jittered backoff
- **Bulkhead**: Resource isolation
- **Rate Limiting**: Prevent overload
- **Fallback**: Graceful degradation

## Quick Start

```python
from victor.framework.resilience import (
    ResilientProvider,
    CircuitBreakerConfig,
    RetryConfig,
)

# Wrap a provider with resilience
resilient = ResilientProvider(
    provider=my_provider,
    circuit_breaker=CircuitBreakerConfig(
        failure_threshold=5,
        recovery_timeout=60.0,
    ),
    retry=RetryConfig(
        max_retries=3,
        backoff_strategy="exponential",
    ),
)

# Use normally - resilience is automatic
response = await resilient.chat(messages)
```

## Circuit Breaker

Prevents cascade failures by "tripping" after too many errors.

### States

```
    [Closed] ──(failures)─> [Open] ──(timeout)─> [Half-Open]
        ^                                            │
        └────────(success)───────────────────────────┘
```

| State | Behavior |
|-------|----------|
| **Closed** | Normal operation, requests pass through |
| **Open** | All requests fail immediately |
| **Half-Open** | Test requests allowed, success closes circuit |

### Configuration

```python
from victor.core.circuit_breaker import CircuitState
from victor.providers.circuit_breaker import CircuitBreaker

breaker = CircuitBreaker(
    name="anthropic_api",
    failure_threshold=5,      # Failures before opening
    recovery_timeout=60.0,    # Seconds before half-open
    success_threshold=2,      # Successes to close from half-open
    half_open_max_calls=3,    # Max test calls in half-open
)

# Use as context manager
async with breaker:
    result = await risky_operation()

# Or decorator
@breaker
async def api_call():
    return await provider.chat(messages)
```

### Monitoring State

```python
# Check current state
if breaker.state == CircuitState.OPEN:
    print("Circuit is open, using fallback")

# Get statistics
stats = breaker.get_stats()
print(f"Failures: {stats.failure_count}")
print(f"Success rate: {stats.success_rate}")
print(f"Last failure: {stats.last_failure_time}")

# Listen for state changes
breaker.on_state_change(lambda old, new:
    print(f"Circuit changed: {old} -> {new}")
)
```

### Circuit Breaker Registry

Manage multiple circuit breakers:

```python
from victor.framework.resilience import get_circuit_registry

registry = get_circuit_registry()

# Get or create a breaker
breaker = registry.get_or_create(
    "openai_api",
    failure_threshold=3,
    recovery_timeout=30.0
)

# List all breakers
for name, breaker in registry.list_all():
    print(f"{name}: {breaker.state}")

# Reset a specific breaker
registry.reset("openai_api")

# Reset all breakers
registry.reset_all()
```

## Retry Strategies

Automatic retries with configurable backoff.

### Backoff Strategies

```python
from victor.observability.resilience import (
    ExponentialBackoff,
    LinearBackoff,
    FixedBackoff,
    JitteredBackoff,
)

# Exponential: 1s, 2s, 4s, 8s, 16s...
exponential = ExponentialBackoff(
    base=1.0,
    multiplier=2.0,
    max_delay=60.0,
)

# Linear: 1s, 2s, 3s, 4s...
linear = LinearBackoff(
    initial=1.0,
    increment=1.0,
    max_delay=30.0,
)

# Fixed: 5s, 5s, 5s...
fixed = FixedBackoff(delay=5.0)

# Jittered: Adds randomness to prevent thundering herd
jittered = JitteredBackoff(
    base_strategy=exponential,
    jitter_factor=0.2,  # +/- 20%
)
```

### Using Retry

```python
from victor.framework.resilience import retry, RetryConfig

# As decorator
@retry(
    max_retries=3,
    backoff=ExponentialBackoff(),
    retry_on=[ConnectionError, TimeoutError],
)
async def api_call():
    return await provider.chat(messages)

# As context manager
async with retry(config=RetryConfig(max_retries=5)):
    result = await risky_operation()
```

### Retry Configuration

```python
config = RetryConfig(
    max_retries=3,
    backoff_strategy="exponential",  # or LinearBackoff()
    retry_on=[ConnectionError, TimeoutError],
    retry_if=lambda e: e.status_code >= 500,  # Custom condition
    on_retry=lambda attempt, error, delay:
        print(f"Retry {attempt} after {delay}s: {error}"),
    timeout=30.0,  # Total timeout for all retries
)
```

## Bulkhead Pattern

Isolate resources to prevent failures from spreading.

```python
from victor.observability.resilience import Bulkhead

# Limit concurrent operations
bulkhead = Bulkhead(
    name="api_calls",
    max_concurrent=10,
    max_queue=50,
    queue_timeout=5.0,
)

async with bulkhead:
    # Only 10 concurrent operations allowed
    result = await api_call()

# Check status
print(f"Active: {bulkhead.active_count}")
print(f"Queued: {bulkhead.queue_size}")
print(f"Rejected: {bulkhead.rejected_count}")
```

## Rate Limiting

Prevent API overload:

```python
from victor.observability.resilience import RateLimiter

limiter = RateLimiter(
    name="anthropic",
    requests_per_second=10,
    burst_size=20,  # Allow bursts up to 20
)

async with limiter:
    # Automatically throttled
    result = await api_call()

# Or check before calling
if limiter.try_acquire():
    result = await api_call()
else:
    print("Rate limited, try later")
```

## Fallback Strategies

Graceful degradation when primary fails:

```python
from victor.framework.resilience import with_fallback

# Static fallback
@with_fallback(default="Sorry, I couldn't process that.")
async def chat(message):
    return await provider.chat(message)

# Dynamic fallback
@with_fallback(fallback=lambda: use_backup_provider())
async def chat(message):
    return await primary_provider.chat(message)

# Chain of fallbacks
@with_fallback(
    fallbacks=[
        use_secondary_provider,
        use_cached_response,
        return_default_message,
    ]
)
async def chat(message):
    return await primary_provider.chat(message)
```

## ResilientProvider

Combines all patterns for LLM providers:

```python
from victor.framework.resilience import ResilientProvider

resilient = ResilientProvider(
    provider=anthropic_provider,

    # Circuit breaker config
    circuit_breaker=CircuitBreakerConfig(
        failure_threshold=5,
        recovery_timeout=60.0,
        success_threshold=2,
    ),

    # Retry config
    retry=RetryConfig(
        max_retries=3,
        backoff_strategy="exponential",
        retry_on=[RateLimitError, ConnectionError],
    ),

    # Bulkhead config
    bulkhead=BulkheadConfig(
        max_concurrent=20,
        max_queue=100,
    ),

    # Rate limit config
    rate_limit=RateLimitConfig(
        requests_per_second=10,
        burst_size=20,
    ),

    # Fallback
    fallback_provider=ollama_provider,  # Use local when API fails
)

# Use like normal provider
response = await resilient.chat(messages)
```

## Timeout Handling

Set timeouts at multiple levels:

```python
from victor.framework.resilience import with_timeout

# Operation timeout
@with_timeout(30.0)
async def long_operation():
    return await slow_api_call()

# With fallback on timeout
@with_timeout(10.0, fallback="Operation timed out")
async def risky_operation():
    return await api_call()

# Per-call timeout
result = await resilient.chat(
    messages,
    timeout=15.0,  # Override default timeout
)
```

## Health Checks

Monitor component health:

```python
from victor.framework.health import HealthChecker, HealthStatus

checker = HealthChecker()

# Register health checks
checker.register("anthropic_api", lambda: anthropic.is_healthy())
checker.register("database", lambda: db.ping())
checker.register("cache", lambda: cache.is_available())

# Run all checks
results = await checker.check_all()
for name, status in results.items():
    print(f"{name}: {status.value}")  # HEALTHY, DEGRADED, UNHEALTHY

# Get overall status
overall = checker.get_overall_status()
if overall == HealthStatus.UNHEALTHY:
    alert_ops_team()
```

## Best Practices

### 1. Configure Per-Provider

```python
# Different thresholds for different providers
anthropic_breaker = CircuitBreaker(
    failure_threshold=5,  # Lower tolerance
    recovery_timeout=60.0,
)

ollama_breaker = CircuitBreaker(
    failure_threshold=10,  # More tolerant for local
    recovery_timeout=10.0,
)
```

### 2. Use Jitter for Retries

```python
# Prevent thundering herd
config = RetryConfig(
    backoff_strategy=JitteredBackoff(
        base_strategy=ExponentialBackoff(),
        jitter_factor=0.3,
    )
)
```

### 3. Set Reasonable Timeouts

```python
# Different timeouts for different operations
quick_op_timeout = 5.0
normal_op_timeout = 30.0
long_op_timeout = 120.0
```

### 4. Monitor Circuit States

```python
# Alert when circuits open
breaker.on_state_change(lambda old, new:
    if new == CircuitState.OPEN:
        send_alert(f"Circuit {breaker.name} opened")
)
```

### 5. Log Retry Attempts

```python
config = RetryConfig(
    on_retry=lambda attempt, error, delay:
        logger.warning(f"Retry {attempt}: {error}, waiting {delay}s")
)
```

## Integration with EventBus

Resilience events are emitted for observability:

```python
from victor.observability.event_bus import get_event_bus, EventCategory

bus = get_event_bus()

bus.subscribe(EventCategory.ERROR, lambda e:
    if e.event_type == "circuit_opened":
        print(f"Circuit opened: {e.data['breaker_name']}")
)

# Events emitted:
# - circuit_opened
# - circuit_closed
# - circuit_half_open
# - retry_attempted
# - rate_limited
# - bulkhead_rejected
```

## Troubleshooting

### Circuit Stuck Open

1. Check recovery timeout setting
2. Verify half-open test succeeds
3. Manually reset if needed: `breaker.reset()`

### Too Many Retries

1. Add retry_if condition to filter retriable errors
2. Set max_retries lower
3. Add total timeout

### Rate Limiting Issues

1. Increase requests_per_second
2. Increase burst_size for spiky traffic
3. Distribute load across time

## Related Resources

- [Observability →](observability/index.md) - Resilience event monitoring
- [Provider Reference →](../reference/providers/index.md) - Provider configuration
- [User Guide →](../user-guide/index.md) - General usage
