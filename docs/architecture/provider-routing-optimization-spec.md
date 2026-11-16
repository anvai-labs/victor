# Provider Routing Optimization Spec

**Status:** Draft  
**Created:** 2026-11-15  
**Author:** Victor RL Team  
**Priority:** P2 (Medium)

---

## Problem Statement

Provider performance varies significantly based on RL outcome data from `~/.victor/victor.db`:

| Provider | Model | Success Rate | Avg Quality | Total Calls |
|----------|-------|-------------|-------------|-------------|
| deepseek | deepseek-chat | **98.3%** | 0.856 | 6,839 |
| zai | glm-5.1 | **96.5%** | 0.839 | 11,776 |
| zai | glm-5.2 | **96.3%** | 0.828 | 15,985 |
| moonshot | kimi-k3 | **97.6%** | 0.833 | 1,929 |
| anthropic | claude-sonnet-4-6 | **95.2%** | 0.846 | 668 |
| openai | qwen3.7-plus | **94.1%** | 0.826 | 6,713 |
| **inferflux** | **qwen3-coder-30b** | **81.6%** | **0.657** | 1,502 |

### The Issue

InferFlux `qwen3-coder-30b` significantly underperforms compared to other providers:
- **16.7% lower success rate** than the average of top providers
- **0.189 lower quality score** than the average of top providers
- Yet it's the default provider for many users

### Impact

1. **Suboptimal user experience:** Users get lower quality responses by default
2. **Wasted resources:** InferFlux calls that fail waste time and compute
3. **Missed opportunities:** Better providers aren't utilized
4. **No automatic failover:** System doesn't adapt to provider performance

---

## Current Architecture

### Provider Selection

Current provider selection is static:
1. User configures default provider in `profiles.yaml`
2. All requests go to that provider unless explicitly overridden
3. No automatic routing based on task type or provider performance
4. No failover on poor performance

### What We Have

- **Provider stats tracking:** `rl_provider_stat` table tracks success/latency per provider
- **Model selection learner:** `model_selector` learner exists but is underutilized
- **Task type detection:** System detects task type (code_generation, analysis, etc.)
- **Quality scoring:** System tracks quality scores per provider/model

### What We Lack

- **Dynamic routing:** No task-aware provider selection
- **Performance-based routing:** No automatic selection of best-performing providers
- **Failover logic:** No automatic switching on poor performance
- **Cost-aware routing:** No consideration of provider cost vs. quality tradeoff

---

## Proposed Solution

### Phase 1: Provider Performance Tracker (P1 — 3 days)

**Goal:** Build comprehensive provider performance tracking by task type.

#### 1.1 Enhanced Provider Stats

```python
@dataclass
class ProviderPerformanceStats:
    """Track provider performance by task type."""
    provider: str
    model: str
    task_type: str
    
    # Success metrics
    total_calls: int = 0
    successful_calls: int = 0
    success_rate: float = 0.0
    
    # Quality metrics
    avg_quality: float = 0.0
    quality_samples: int = 0
    
    # Latency metrics
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    latency_samples: list = field(default_factory=list)
    
    # Cost metrics (if available)
    avg_cost_per_call: float = 0.0
    cost_per_quality_point: float = 0.0
    
    # Error tracking
    error_counts: dict = field(default_factory=dict)  # error_type -> count
    
    # Recency
    last_updated: datetime = None
    
    def update(self, success: bool, quality: float, latency_ms: float, 
               error_type: str = None, cost: float = None):
        """Update stats with new outcome."""
        self.total_calls += 1
        if success:
            self.successful_calls += 1
        self.success_rate = self.successful_calls / self.total_calls
        
        # Update quality (EMA)
        if self.quality_samples == 0:
            self.avg_quality = quality
        else:
            alpha = 0.3  # EMA smoothing
            self.avg_quality = alpha * quality + (1 - alpha) * self.avg_quality
        self.quality_samples += 1
        
        # Update latency (EMA)
        if self.latency_samples:
            alpha = 0.3
            self.avg_latency_ms = alpha * latency_ms + (1 - alpha) * self.avg_latency_ms
        else:
            self.avg_latency_ms = latency_ms
        self.latency_samples.append(latency_ms)
        if len(self.latency_samples) > 100:
            self.latency_samples = self.latency_samples[-100:]
        self.p95_latency_ms = sorted(self.latency_samples)[int(0.95 * len(self.latency_samples))]
        
        # Update error counts
        if error_type:
            self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        # Update cost
        if cost:
            self.avg_cost_per_call = (
                (self.avg_cost_per_call * (self.total_calls - 1) + cost) / self.total_calls
            )
            if self.avg_quality > 0:
                self.cost_per_quality_point = self.avg_cost_per_call / self.avg_quality
        
        self.last_updated = datetime.now()
```

#### 1.2 Provider Performance Registry

```python
class ProviderPerformanceRegistry:
    """Registry for provider performance stats."""
    
    def __init__(self, db):
        self.db = db
        self._stats = {}  # (provider, model, task_type) -> ProviderPerformanceStats
    
    def record_outcome(self, provider: str, model: str, task_type: str,
                       success: bool, quality: float, latency_ms: float,
                       error_type: str = None, cost: float = None):
        """Record provider outcome."""
        key = (provider, model, task_type)
        
        if key not in self._stats:
            self._stats[key] = ProviderPerformanceStats(provider, model, task_type)
        
        self._stats[key].update(success, quality, latency_ms, error_type, cost)
        
        # Persist to database
        self._persist_stats(key)
    
    def get_best_provider(self, task_type: str, 
                          min_success_rate: float = 0.9,
                          max_latency_ms: float = None) -> Optional[tuple]:
        """Get best provider for task type."""
        candidates = [
            stats for key, stats in self._stats.items()
            if key[2] == task_type and stats.success_rate >= min_success_rate
        ]
        
        if max_latency_ms:
            candidates = [s for s in candidates if s.avg_latency_ms <= max_latency_ms]
        
        if not candidates:
            return None
        
        # Sort by quality (primary) and success rate (secondary)
        candidates.sort(key=lambda s: (s.avg_quality, s.success_rate), reverse=True)
        
        best = candidates[0]
        return (best.provider, best.model)
    
    def get_provider_ranking(self, task_type: str) -> list:
        """Get ranked list of providers for task type."""
        candidates = [
            stats for key, stats in self._stats.items()
            if key[2] == task_type
        ]
        
        candidates.sort(key=lambda s: (s.success_rate, s.avg_quality), reverse=True)
        
        return [
            {
                "provider": s.provider,
                "model": s.model,
                "success_rate": s.success_rate,
                "avg_quality": s.avg_quality,
                "total_calls": s.total_calls,
            }
            for s in candidates
        ]
```

### Phase 2: Task-Aware Provider Routing (P1 — 4 days)

**Goal:** Route requests to best provider based on task type.

#### 2.1 Provider Router

```python
class ProviderRouter:
    """Route requests to optimal provider based on task type and performance."""
    
    def __init__(self, registry: ProviderPerformanceRegistry, config: dict):
        self.registry = registry
        self.config = config
        self.min_calls_for_routing = config.get("min_calls_for_routing", 50)
        self.enable_dynamic_routing = config.get("enable_dynamic_routing", True)
    
    def select_provider(self, task_type: str, 
                       preferred_provider: str = None,
                       preferred_model: str = None) -> tuple:
        """Select optimal provider for task.
        
        Args:
            task_type: Type of task (code_generation, analysis, etc.)
            preferred_provider: User's preferred provider (if any)
            preferred_model: User's preferred model (if any)
        
        Returns:
            (provider, model) tuple
        """
        # If dynamic routing disabled, use preferred
        if not self.enable_dynamic_routing:
            return (preferred_provider, preferred_model)
        
        # Check if we have enough data for routing
        ranking = self.registry.get_provider_ranking(task_type)
        
        if not ranking or ranking[0]["total_calls"] < self.min_calls_for_routing:
            # Not enough data, use preferred
            return (preferred_provider, preferred_model)
        
        # Get best provider
        best = ranking[0]
        
        # Check if best is significantly better than preferred
        if preferred_provider:
            preferred_stats = next(
                (r for r in ranking if r["provider"] == preferred_provider),
                None
            )
            
            if preferred_stats:
                # Use best only if significantly better (>5% success rate or >0.1 quality)
                if (best["success_rate"] - preferred_stats["success_rate"] > 0.05 or
                    best["avg_quality"] - preferred_stats["avg_quality"] > 0.1):
                    logger.info(
                        f"ProviderRouter: Routing {task_type} from {preferred_provider} "
                        f"to {best['provider']} (better performance)"
                    )
                    return (best["provider"], best["model"])
        
        # Use best provider
        return (best["provider"], best["model"])
```

#### 2.2 Integration with Agentic Loop

```python
# In victor/framework/agentic_loop.py
async def _select_provider_for_task(self, task_type: str):
    """Select provider based on task type and performance."""
    from victor.framework.rl.provider_routing import ProviderRouter, ProviderPerformanceRegistry
    
    registry = ProviderPerformanceRegistry(self.db)
    router = ProviderRouter(registry, self.config)
    
    preferred_provider = self.profile.get("provider")
    preferred_model = self.profile.get("model")
    
    provider, model = router.select_provider(
        task_type=task_type,
        preferred_provider=preferred_provider,
        preferred_model=preferred_model,
    )
    
    return provider, model
```

### Phase 3: Automatic Failover (P2 — 3 days)

**Goal:** Automatically switch providers on poor performance.

#### 3.1 Failover Strategy

```python
class ProviderFailover:
    """Automatic failover on provider performance degradation."""
    
    def __init__(self, registry: ProviderPerformanceRegistry, config: dict):
        self.registry = registry
        self.config = config
        self.failure_threshold = config.get("failure_threshold", 0.5)  # 50% success rate
        self.min_calls_for_failover = config.get("min_calls_for_failover", 10)
        self.cooldown_period = config.get("cooldown_period", 300)  # 5 minutes
    
    def should_failover(self, provider: str, model: str, task_type: str) -> bool:
        """Check if we should failover from current provider."""
        stats = self.registry.get_stats(provider, model, task_type)
        
        if not stats or stats.total_calls < self.min_calls_for_failover:
            return False
        
        # Check if success rate below threshold
        if stats.success_rate < self.failure_threshold:
            logger.warning(
                f"ProviderFailover: {provider}/{model} success rate {stats.success_rate:.1%} "
                f"below threshold {self.failure_threshold:.1%} for {task_type}"
            )
            return True
        
        return False
    
    def get_failover_provider(self, task_type: str, 
                              exclude: list = None) -> Optional[tuple]:
        """Get failover provider."""
        ranking = self.registry.get_provider_ranking(task_type)
        
        exclude = exclude or []
        
        for candidate in ranking:
            if (candidate["provider"], candidate["model"]) not in exclude:
                if candidate["success_rate"] >= 0.9:  # Only failover to good providers
                    return (candidate["provider"], candidate["model"])
        
        return None
```

#### 3.2 Failover Integration

```python
# In provider adapter
async def call_provider(self, provider, model, messages, task_type):
    """Call provider with automatic failover."""
    failover = ProviderFailover(self.registry, self.config)
    
    try:
        result = await self._call_provider_impl(provider, model, messages)
        
        # Record success
        self.registry.record_outcome(
            provider=provider, model=model, task_type=task_type,
            success=True, quality=result.quality, latency_ms=result.latency_ms,
        )
        
        return result
        
    except Exception as e:
        # Record failure
        self.registry.record_outcome(
            provider=provider, model=model, task_type=task_type,
            success=False, quality=0.0, latency_ms=0.0,
            error_type=type(e).__name__,
        )
        
        # Check if we should failover
        if failover.should_failover(provider, model, task_type):
            failover_provider, failover_model = failover.get_failover_provider(
                task_type, exclude=[(provider, model)]
            )
            
            if failover_provider:
                logger.info(
                    f"Failing over from {provider}/{model} to {failover_provider}/{failover_model}"
                )
                return await self._call_provider_impl(
                    failover_provider, failover_model, messages
                )
        
        raise
```

### Phase 4: Cost-Aware Routing (P3 — 4 days)

**Goal:** Consider cost when routing to providers.

#### 4.1 Cost-Effectiveness Scoring

```python
def calculate_cost_effectiveness(stats: ProviderPerformanceStats) -> float:
    """Calculate cost-effectiveness score.
    
    Higher is better. Combines quality, success rate, and cost.
    """
    if stats.avg_cost_per_call == 0:
        # No cost data, use quality * success_rate
        return stats.avg_quality * stats.success_rate
    
    # Cost per quality point (lower is better)
    cost_per_quality = stats.avg_cost_per_call / max(stats.avg_quality, 0.01)
    
    # Invert so higher is better
    return 1.0 / cost_per_quality * stats.success_rate
```

#### 4.2 Budget-Aware Routing

```python
class BudgetAwareRouter(ProviderRouter):
    """Route providers considering budget constraints."""
    
    def __init__(self, registry, config, budget_per_day: float = None):
        super().__init__(registry, config)
        self.budget_per_day = budget_per_day
        self.daily_spend = 0.0
        self.last_reset = datetime.now().date()
    
    def select_provider(self, task_type, preferred_provider=None, preferred_model=None):
        """Select provider within budget."""
        # Reset daily spend if needed
        today = datetime.now().date()
        if today > self.last_reset:
            self.daily_spend = 0.0
            self.last_reset = today
        
        # Get candidates
        ranking = self.registry.get_provider_ranking(task_type)
        
        # Filter by budget
        if self.budget_per_day:
            remaining_budget = self.budget_per_day - self.daily_spend
            candidates = [
                r for r in ranking
                if r.get("avg_cost_per_call", 0) <= remaining_budget
            ]
        else:
            candidates = ranking
        
        if not candidates:
            # No candidates within budget, use cheapest
            candidates = sorted(ranking, key=lambda r: r.get("avg_cost_per_call", 0))
        
        if not candidates:
            return (preferred_provider, preferred_model)
        
        # Select best cost-effectiveness
        best = max(candidates, key=lambda r: calculate_cost_effectiveness(
            self.registry.get_stats(r["provider"], r["model"], task_type)
        ))
        
        return (best["provider"], best["model"])
```

### Phase 5: Monitoring and Alerts (P3 — 2 days)

**Goal:** Monitor provider performance and alert on degradation.

#### 5.1 Provider Health Dashboard

```python
def get_provider_health_dashboard() -> dict:
    """Get provider health summary."""
    registry = ProviderPerformanceRegistry(get_victor_db())
    
    # Get all task types
    task_types = registry.get_all_task_types()
    
    dashboard = {}
    
    for task_type in task_types:
        ranking = registry.get_provider_ranking(task_type)
        
        if not ranking:
            continue
        
        best = ranking[0]
        
        dashboard[task_type] = {
            "best_provider": best["provider"],
            "best_model": best["model"],
            "best_success_rate": best["success_rate"],
            "best_quality": best["avg_quality"],
            "total_providers": len(ranking),
            "providers": ranking[:5],  # Top 5
        }
    
    return dashboard
```

#### 5.2 Degradation Alerts

```python
def check_provider_degradation():
    """Check for provider performance degradation."""
    registry = ProviderPerformanceRegistry(get_victor_db())
    
    alerts = []
    
    for (provider, model, task_type), stats in registry._stats.items():
        # Check if success rate dropped significantly
        if stats.total_calls >= 50 and stats.success_rate < 0.8:
            alerts.append({
                "severity": "warning",
                "provider": provider,
                "model": model,
                "task_type": task_type,
                "message": f"{provider}/{model} success rate dropped to {stats.success_rate:.1%} for {task_type}",
            })
        
        # Check if latency increased
        if stats.p95_latency_ms > 10000:  # 10 seconds
            alerts.append({
                "severity": "info",
                "provider": provider,
                "model": model,
                "task_type": task_type,
                "message": f"{provider}/{model} p95 latency is {stats.p95_latency_ms:.0f}ms for {task_type}",
            })
    
    return alerts
```

---

## Implementation Plan

### Sprint 1: Performance Tracker (Weeks 1-2)

- [ ] Implement `ProviderPerformanceStats` dataclass
- [ ] Implement `ProviderPerformanceRegistry`
- [ ] Add persistence to database
- [ ] Add unit tests for stats tracking
- [ ] Migrate existing `rl_provider_stat` data

### Sprint 2: Task-Aware Routing (Weeks 3-4)

- [ ] Implement `ProviderRouter`
- [ ] Integrate with agentic loop
- [ ] Add configuration for dynamic routing
- [ ] Add unit tests for routing logic
- [ ] A/B test with real usage

### Sprint 3: Automatic Failover (Weeks 5-6)

- [ ] Implement `ProviderFailover`
- [ ] Integrate with provider adapters
- [ ] Add configuration for failover thresholds
- [ ] Add unit tests for failover logic
- [ ] Test with simulated failures

### Sprint 4: Cost-Aware Routing (Weeks 7-8)

- [ ] Implement cost-effectiveness scoring
- [ ] Implement `BudgetAwareRouter`
- [ ] Add cost tracking to provider calls
- [ ] Add unit tests for cost-aware routing
- [ ] A/B test with budget constraints

### Sprint 5: Monitoring (Week 9)

- [ ] Implement provider health dashboard
- [ ] Implement degradation alerts
- [ ] Add to `victor doctor`
- [ ] Add unit tests for monitoring
- [ ] Document monitoring approach

---

## Verification

### Provider Ranking

```bash
# Get provider ranking for task type
sqlite3 ~/.victor/victor.db "
SELECT provider, model, 
       ROUND(100.0 * SUM(success) / COUNT(*), 1) as success_pct,
       ROUND(AVG(latency_ms), 1) as avg_latency_ms
FROM rl_provider_stat
WHERE task_type = 'code_generation'
  AND created_at > datetime('now', '-7 days')
GROUP BY provider, model
ORDER BY success_pct DESC;
"
```

### Routing Decisions

```python
# Check routing decisions
router = ProviderRouter(registry, config)
provider, model = router.select_provider("code_generation")
print(f"Selected: {provider}/{model}")

# Check ranking
ranking = registry.get_provider_ranking("code_generation")
for r in ranking:
    print(f"{r['provider']}/{r['model']}: {r['success_rate']:.1%} success, {r['avg_quality']:.3f} quality")
```

### Failover Events

```bash
# Check failover events
sqlite3 ~/.victor/victor.db "
SELECT provider, model, task_type, COUNT(*) as failover_count
FROM rl_provider_failover_log
WHERE created_at > datetime('now', '-1 day')
GROUP BY provider, model, task_type;
"
```

---

## Acceptance Criteria

- [ ] Provider performance tracked by task type
- [ ] Dynamic routing selects best provider based on performance
- [ ] Automatic failover on poor performance (<50% success rate)
- [ ] Cost-aware routing considers cost vs. quality tradeoff
- [ ] Provider health dashboard available
- [ ] Degradation alerts configured
- [ ] InferFlux success rate improves (via routing to better providers)
- [ ] Overall system success rate improves by >5%
- [ ] Unit tests cover all new functionality
- [ ] No regression in existing provider functionality

---

## Open Questions

1. **Routing transparency:** Should we inform users when we route to a different provider?
   - Option: Log routing decisions, don't notify user
   - Option: Notify user with message like "Using {provider} for better performance"
   - Recommendation: Log decisions, don't interrupt user

2. **Provider preferences:** Should users be able to lock to specific providers?
   - Option: Add `lock_provider` config option
   - Risk: Users may lock to underperforming providers
   - Recommendation: Allow locking, but warn if locked provider underperforms

3. **Cost data source:** Where do we get provider cost data?
   - Option: Manual configuration in profiles
   - Option: Automatic from provider APIs
   - Recommendation: Manual config for now, automatic later

4. **Failover cooldown:** How long should we wait before trying failed provider again?
   - Option: 5 minutes (current)
   - Option: Exponential backoff (5min, 15min, 1hr)
   - Recommendation: Start with 5 minutes, add backoff if needed

---

## Follow-Up Work

1. **Provider A/B testing:** Framework for testing new providers
2. **Multi-provider aggregation:** Use multiple providers for different parts of task
3. **Provider-specific prompts:** Optimize prompts per provider
4. **Quality prediction:** Predict provider quality before calling

---

## References

- RL Data Analysis: `docs/analysis/rl-data-summary-2026-11-15.md`
- Smart Routing: `docs/architecture/smart_routing.md`
- Provider Agnostic Tiers: `docs/architecture/provider-agnostic-tiers.md`
- Model Selector Learner: `victor/framework/rl/learners/model_selector.py`
