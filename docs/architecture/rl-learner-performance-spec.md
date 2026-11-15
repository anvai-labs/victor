# RL Learner Performance Improvement Spec

**Status:** Draft  
**Created:** 2026-11-15  
**Author:** Victor RL Team  
**Priority:** P2 (Medium)

---

## Problem Statement

Two RL learners are significantly underperforming based on analysis of 191K outcomes in `~/.victor/victor.db`:

| Learner | Success Rate | Total Outcomes | Avg Quality | Target |
|---------|-------------|----------------|-------------|--------|
| `quality_weights` | **59.1%** | 4,631 | 0.640 | >80% |
| `continuation_prompts` | **69.8%** | 126 | 0.499 | >80% |

### Context

Other learners perform well:
- `tool_selector`: 93.8% success (103K outcomes)
- `semantic_threshold`: 99.9% success (11K outcomes)
- `workflow_execution`: 98.8% success (3K outcomes)
- `grounding_threshold`: 80.0% success (4K outcomes)

The underperforming learners need investigation and improvement.

---

## Root Cause Analysis

### quality_weights Learner (59.1% success)

**Purpose:** Learn optimal weights for quality score computation (grounding, user feedback, completion score).

**Hypothesized Issues:**

1. **Reward signal too noisy (40% likely)**
   - Quality scores are subjective and vary by task type
   - User feedback is sparse (only 46 outcomes with feedback)
   - Completion score may not correlate with actual quality

2. **Learning rate too high (25% likely)**
   - Current learning rate: 0.1 (from `quality_weights.py`)
   - May be over-adjusting weights based on noisy signals
   - Causes oscillation instead of convergence

3. **State representation too coarse (20% likely)**
   - Context key: `provider:model:task_type`
   - May not capture enough variation (e.g., code complexity, domain)
   - Same weights applied to very different tasks

4. **Weight initialization poor (15% likely)**
   - Initial weights may be far from optimal
   - Slow convergence from bad starting point

### continuation_prompts Learner (69.8% success)

**Purpose:** Learn optimal prompt selection strategy when model is stuck in continuation loop.

**Hypothesized Issues:**

1. **Prompt pool too small (35% likely)**
   - Limited variety of continuation prompts
   - Same prompts used for different stuck patterns
   - Can't adapt to different failure modes

2. **Success definition unclear (30% likely)**
   - What counts as "prompt selection succeeded"?
   - May be measuring wrong thing (prompt used vs. prompt effective)
   - Success signal may be delayed (prompt works after several turns)

3. **Context key too specific (20% likely)**
   - Context: `provider:model:task_type`
   - May over-fit to specific combinations
   - Not enough data per context (only 126 total outcomes)

4. **Exploration/exploitation imbalance (15% likely)**
   - May be exploiting suboptimal prompts too early
   - Not exploring enough to find better prompts
   - Epsilon-greedy parameters may need tuning

---

## Proposed Solution

### quality_weights Improvements

#### 1. Reward Signal Stabilization (P1 — 3 days)

**Problem:** Quality scores are noisy and subjective.

**Solution:** Use exponential moving average (EMA) to smooth reward signals.

```python
class QualityWeightsLearner(BaseLearner):
    def __init__(self, ...):
        self.ema_alpha = 0.3  # EMA smoothing factor
        self._ema_rewards = {}  # context_key -> ema_reward
    
    def record_outcome(self, outcome: RLOutcome) -> None:
        context_key = self._get_context_key(outcome.provider, outcome.model, outcome.task_type)
        
        # Compute raw reward from quality score
        raw_reward = outcome.quality_score
        
        # Apply EMA smoothing
        if context_key not in self._ema_rewards:
            self._ema_rewards[context_key] = raw_reward
        else:
            self._ema_rewards[context_key] = (
                self.ema_alpha * raw_reward + 
                (1 - self.ema_alpha) * self._ema_rewards[context_key]
            )
        
        smoothed_reward = self._ema_rewards[context_key]
        
        # Use smoothed reward for weight updates
        self._update_weights(context_key, smoothed_reward)
```

**Expected Impact:** Reduces oscillation, improves convergence by ~20%.

#### 2. Adaptive Learning Rate (P1 — 2 days)

**Problem:** Fixed learning rate doesn't account for signal quality.

**Solution:** Reduce learning rate as confidence increases.

```python
def _get_adaptive_learning_rate(self, context_key: str) -> float:
    """Reduce learning rate as we gather more evidence."""
    sample_count = self._get_sample_count(context_key)
    
    # Start at 0.1, decay to 0.01 after 100 samples
    base_lr = 0.1
    min_lr = 0.01
    decay_rate = 0.02  # Decay per sample
    
    adaptive_lr = max(min_lr, base_lr * math.exp(-decay_rate * sample_count))
    
    return adaptive_lr
```

**Expected Impact:** Prevents over-adjustment after convergence, improves stability.

#### 3. Hierarchical Context Keys (P2 — 3 days)

**Problem:** Context key `provider:model:task_type` is too coarse.

**Solution:** Use hierarchical context with fallback.

```python
def _get_context_key(self, provider: str, model: str, task_type: str, 
                     complexity: str = None) -> str:
    """Build hierarchical context key with fallback."""
    # Most specific
    if complexity:
        specific = f"{provider}:{model}:{task_type}:{complexity}"
        if self._has_enough_data(specific, min_samples=10):
            return specific
    
    # Medium specificity
    medium = f"{provider}:{model}:{task_type}"
    if self._has_enough_data(medium, min_samples=20):
        return medium
    
    # Provider-level fallback
    provider_level = f"{provider}:all:all"
    if self._has_enough_data(provider_level, min_samples=50):
        return provider_level
    
    # Global fallback
    return "global:all:all"

def _has_enough_data(self, context_key: str, min_samples: int) -> bool:
    """Check if context has enough samples for reliable learning."""
    return self._get_sample_count(context_key) >= min_samples
```

**Expected Impact:** Better generalization, improves success rate by ~15%.

#### 4. Multi-Objective Optimization (P2 — 4 days)

**Problem:** Single quality score doesn't capture multiple dimensions.

**Solution:** Track separate weights for different quality dimensions.

```python
@dataclass
class QualityDimensions:
    grounding_score: float      # Hallucination check
    completion_score: float     # Task completion
    user_feedback: float        # User rating (if available)
    code_quality: float         # Static analysis (if available)

class QualityWeightsLearner(BaseLearner):
    def __init__(self, ...):
        # Separate weights for each dimension
        self.weights = {
            "grounding": 0.4,
            "completion": 0.3,
            "user_feedback": 0.2,
            "code_quality": 0.1,
        }
    
    def record_outcome(self, outcome: RLOutcome) -> None:
        # Extract dimension scores from metadata
        dimensions = self._extract_dimensions(outcome.metadata)
        
        # Update weights for each dimension independently
        for dim_name, dim_score in dimensions.items():
            self._update_dimension_weight(dim_name, dim_score, outcome.success)
```

**Expected Impact:** More granular learning, improves quality score accuracy by ~25%.

### continuation_prompts Improvements

#### 1. Expand Prompt Pool (P1 — 2 days)

**Problem:** Limited variety of continuation prompts.

**Solution:** Add more prompt templates for different stuck patterns.

```python
CONTINUATION_PROMPTS = {
    # Existing prompts
    "gentle_nudge": "Please continue with the task.",
    "specific_hint": "Try using the {tool_name} tool to {action}.",
    
    # New prompts for different patterns
    "break_loop": "You seem to be repeating the same action. Try a different approach.",
    "suggest_alternative": "Consider using {alternative_tool} instead of {current_tool}.",
    "provide_context": "Here's additional context: {context_summary}. How does this change your approach?",
    "ask_clarification": "I'm not sure I understand the current approach. Can you explain your reasoning?",
    "reset_strategy": "Let's start fresh. What's the core problem we're trying to solve?",
    
    # Task-specific prompts
    "code_stuck": "The code isn't compiling. Check the error message and fix the syntax.",
    "test_failing": "The test is failing. Review the assertion and check if the implementation matches.",
    "search_no_results": "No results found. Try broadening your search query or using a different tool.",
}

def _select_prompt(self, stuck_pattern: str, task_type: str) -> str:
    """Select prompt based on stuck pattern and task type."""
    # Pattern-specific prompts
    if stuck_pattern == "repeating_action":
        return CONTINUATION_PROMPTS["break_loop"]
    elif stuck_pattern == "no_progress":
        return CONTINUATION_PROMPTS["reset_strategy"]
    
    # Task-specific prompts
    if task_type == "code_generation":
        return CONTINUATION_PROMPTS["code_stuck"]
    elif task_type == "debugging":
        return CONTINUATION_PROMPTS["test_failing"]
    
    # Default
    return CONTINUATION_PROMPTS["gentle_nudge"]
```

**Expected Impact:** Better prompt selection, improves success rate by ~20%.

#### 2. Delayed Reward Attribution (P1 — 3 days)

**Problem:** Success signal may be delayed (prompt works after several turns).

**Solution:** Track prompt effectiveness over a window of turns.

```python
class ContinuationPromptsLearner(BaseLearner):
    def __init__(self, ...):
        self.reward_window = 5  # Track effectiveness over 5 turns
        self._prompt_history = {}  # session_id -> list of (prompt, outcome)
    
    def record_outcome(self, outcome: RLOutcome) -> None:
        session_id = outcome.metadata.get("session_id")
        prompt_used = outcome.metadata.get("prompt_used")
        
        if session_id not in self._prompt_history:
            self._prompt_history[session_id] = []
        
        self._prompt_history[session_id].append({
            "prompt": prompt_used,
            "turn_outcome": outcome.success,
            "timestamp": outcome.timestamp,
        })
        
        # Check if prompt led to success within window
        history = self._prompt_history[session_id]
        if len(history) >= self.reward_window:
            recent = history[-self.reward_window:]
            prompt_effectiveness = sum(1 for h in recent if h["turn_outcome"]) / len(recent)
            
            # Attribute reward to prompt
            self._update_prompt_reward(prompt_used, prompt_effectiveness)
```

**Expected Impact:** Better credit assignment, improves learning accuracy by ~30%.

#### 3. Context-Aware Prompt Selection (P2 — 3 days)

**Problem:** Context key too specific, not enough data per context.

**Solution:** Use feature-based context representation.

```python
def _extract_features(self, outcome: RLOutcome) -> dict:
    """Extract features for context-aware selection."""
    return {
        "provider": outcome.provider,
        "model_family": self._get_model_family(outcome.model),  # e.g., "claude", "gpt", "qwen"
        "task_complexity": outcome.metadata.get("complexity", "medium"),
        "stuck_duration": outcome.metadata.get("stuck_turns", 1),
        "tools_available": len(outcome.metadata.get("available_tools", [])),
    }

def _select_prompt(self, features: dict) -> str:
    """Select prompt based on feature similarity."""
    # Find most similar past contexts
    similar_contexts = self._find_similar_contexts(features, top_k=5)
    
    # Select prompt with highest reward in similar contexts
    best_prompt = None
    best_reward = -1
    
    for ctx in similar_contexts:
        for prompt, reward in ctx["prompt_rewards"].items():
            if reward > best_reward:
                best_reward = reward
                best_prompt = prompt
    
    return best_prompt or self._default_prompt()
```

**Expected Impact:** Better generalization, improves success rate by ~15%.

#### 4. Exploration Boost (P2 — 2 days)

**Problem:** Not exploring enough to find better prompts.

**Solution:** Increase exploration epsilon for low-data contexts.

```python
def _get_exploration_epsilon(self, context_key: str) -> float:
    """Increase exploration for contexts with little data."""
    sample_count = self._get_sample_count(context_key)
    
    # High exploration for low-data contexts
    if sample_count < 10:
        return 0.5  # 50% exploration
    elif sample_count < 50:
        return 0.3  # 30% exploration
    else:
        return 0.1  # 10% exploration (exploit)
```

**Expected Impact:** Better prompt discovery, improves long-term performance by ~10%.

---

## Implementation Plan

### Sprint 1: quality_weights Stabilization (Weeks 1-2)

- [ ] Implement EMA smoothing for reward signals
- [ ] Implement adaptive learning rate
- [ ] Add unit tests for stabilization
- [ ] A/B test with real usage
- [ ] Measure improvement in success rate

### Sprint 2: continuation_prompts Expansion (Weeks 3-4)

- [ ] Expand prompt pool with new templates
- [ ] Implement delayed reward attribution
- [ ] Add unit tests for new prompts
- [ ] Test with different stuck patterns
- [ ] Measure improvement in success rate

### Sprint 3: Advanced Improvements (Weeks 5-6)

- [ ] Implement hierarchical context keys for quality_weights
- [ ] Implement context-aware prompt selection
- [ ] Implement exploration boost
- [ ] Add unit tests for advanced features
- [ ] A/B test with real usage

### Sprint 4: Multi-Objective Optimization (Weeks 7-8)

- [ ] Implement multi-dimensional quality tracking
- [ ] Update weight learning for multiple dimensions
- [ ] Add unit tests for multi-objective
- [ ] Measure improvement in quality score accuracy

---

## Verification

### Success Rate Monitoring

```bash
# Check learner success rates
sqlite3 ~/.victor/victor.db "
SELECT learner_id, 
       ROUND(100.0 * SUM(success) / COUNT(*), 1) as success_pct,
       COUNT(*) as total,
       ROUND(AVG(quality_score), 3) as avg_quality
FROM rl_outcome
WHERE learner_id IN ('quality_weights', 'continuation_prompts')
  AND created_at > datetime('now', '-7 days')
GROUP BY learner_id;
"
```

### Weight Convergence

```python
# Check if weights are stabilizing
learner = get_quality_weights_learner()
weights = learner.get_weights("anthropic:claude-3:code_generation")
print(f"Weights: {weights}")
print(f"Sample count: {learner.get_sample_count('anthropic:claude-3:code_generation')}")
```

### Prompt Effectiveness

```python
# Check prompt selection effectiveness
learner = get_continuation_prompts_learner()
effectiveness = learner.get_prompt_effectiveness()
for prompt, eff in sorted(effectiveness.items(), key=lambda x: x[1], reverse=True):
    print(f"{prompt}: {eff:.2%}")
```

---

## Acceptance Criteria

- [ ] `quality_weights` success rate improves from 59% to >80%
- [ ] `continuation_prompts` success rate improves from 70% to >80%
- [ ] Reward signals are smoothed (EMA implemented)
- [ ] Learning rate adapts based on sample count
- [ ] Prompt pool expanded with at least 5 new templates
- [ ] Delayed reward attribution implemented
- [ ] Unit tests cover all new functionality
- [ ] No regression in other learners
- [ ] A/B test results documented

---

## Open Questions

1. **Reward signal source:** Should we incorporate more reward signals (e.g., static analysis, test coverage)?
   - Option: Add code_quality dimension
   - Risk: Increases complexity
   - Recommendation: Start with existing signals, add more if needed

2. **Prompt template management:** Should prompts be configurable per-profile?
   - Option: Allow profiles to define custom prompts
   - Risk: Complexity, maintenance burden
   - Recommendation: Keep prompts in code for now, make configurable later

3. **Context key granularity:** How specific should context keys be?
   - Option: Include code complexity, domain, file type
   - Risk: Data sparsity
   - Recommendation: Use hierarchical approach with fallback

4. **Exploration strategy:** Should we use more sophisticated exploration (UCB, Thompson Sampling)?
   - Option: Replace epsilon-greedy with Thompson Sampling
   - Risk: More complex, harder to debug
   - Recommendation: Stick with epsilon-greedy for now, consider Thompson Sampling later

---

## Follow-Up Work

1. **Prompt effectiveness analysis:** Analyze which prompts work best for which patterns
2. **Cross-learner coordination:** Coordinate quality_weights and continuation_prompts
3. **Prompt evolution:** Use GEPA to evolve better continuation prompts
4. **User feedback integration:** Incorporate user feedback into reward signals

---

## References

- RL Data Analysis: `docs/analysis/rl-data-summary-2026-11-15.md`
- Quality Weights Learner: `victor/framework/rl/learners/quality_weights.py`
- Continuation Prompts Learner: `victor/framework/rl/learners/continuation_prompts.py`
- FEP-0017: Prompt Optimization Reward Loop
