# Bayesian Orchestration Architecture

This document describes the architecture of Victor's Bayes-consistent orchestration system, including design principles, component relationships, data flow, and extension points.

## Overview

The Bayesian orchestration system implements principled decision-making under uncertainty using Bayesian decision theory. It extends Victor's orchestration layer with:

1. **Belief State Tracking** — Maintain posterior distributions over task outcomes
2. **Observation Models** — Learn P(agent_message | task_outcome) for likelihoods
3. **Reliability Weighting** — Downweight noisy/biased agents using learned weights
4. **Value of Information** — Make intelligent querying decisions
5. **Bayesian Consensus** — Combine multi-agent opinions with reliability weighting

Based on: "Position: agentic AI orchestration should be Bayes-consistent" (arXiv:2605.00742, ICML 2026)

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     BayesianOrchestrationService                 │
│                        (Integration Layer)                      │
└─────────────────────────────────────────────────────────────────┘
           │                │                │                │
           ▼                ▼                ▼                ▼
┌──────────────────┐ ┌──────────────┐ ┌────────────┐ ┌──────────────┐
│  ObservationModel│ │AgentReliability│ │VoIController│ │  Consensus   │
│     Learner      │ │    Learner    │ │             │ │   Builder    │
│                  │ │              │ │             │ │              │
│ P(z|Y) with Beta │ │ α_i weights  │ │ VoI = E[ΔH] │ │ Weighted     │
│  distributions   │ │              │ │    - cost   │ │ pooling      │
└──────────────────┘ └──────────────┘ └────────────┘ └──────────────┘
           │                │                │                │
           └────────────────┴────────────────┴────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │ SQLite Database │
                    │                 │
                    │ rl_observation_model
                    │ rl_agent_reliability
                    │ rl_voi_history
                    │ rl_belief_history
                    │ rl_bayesian_consensus
                    └─────────────────┘
```

## Component Architecture

### 1. BayesianTaskAnalysis (Core Belief Representation)

**Purpose**: Extend TaskAnalysis with Bayesian belief states

**Responsibilities**:
- Maintain posterior distribution P(Y|D) over task outcomes
- Compute Shannon entropy H[Y|D] for uncertainty quantification
- Perform Bayesian posterior updates with conjugate priors
- Compute Value of Information for query decisions
- Track agent reliability weights

**Key Design Decisions**:
- **Extends TaskAnalysis**: Reuses existing task analysis infrastructure
- **Immutable belief updates**: Each update creates a new posterior (functional style)
- **Belief ID tracking**: Enables orchestration service to cache and retrieve belief states
- **Agent reliability caching**: Stores reliability weights for fast access

**Data Flow**:
```
Prior P(Y|D) → Observation P(z|Y) → Posterior P(Y|D,z)
     ↓              ↓                     ↓
  Entropy     Likelihood from         Updated Entropy
  H[Y|D]      ObservationModel        H[Y|D,z]
```

**Extension Points**:
- Custom prior distributions via `initial_belief` parameter
- Custom likelihood functions via overriding `compute_posterior`
- Custom entropy metrics via overriding `compute_entropy`

### 2. ObservationModelLearner (Likelihood Learning)

**Purpose**: Learn P(agent_message | task_outcome) using Bayesian inference

**Responsibilities**:
- Record observations (agent_message, actual_outcome) pairs
- Maintain Beta distributions for each (agent, outcome, category) triplet
- Categorize messages into semantic categories (affirm, deny, uncertain, error)
- Compute likelihoods using Thompson Sampling for exploration
- Track calibration metrics per agent

**Key Design Decisions**:
- **Beta conjugate prior**: Analytical posterior updates, no MCMC needed
- **Message categorization**: Reduces sparsity by grouping semantically similar messages
- **Thompson Sampling**: Balances exploration and exploitation in likelihood computation
- **Calibration tracking**: Monitors how well agent predictions match empirical frequencies

**Algorithm**:
```python
# Prior: Beta(α=1, β=1) — uniform
# Update with observation (message, outcome):
#   category = categorize(message)
#   if outcome == "success":
#       α_success[category] += 1
#   else:
#       β_failure[category] += 1

# Likelihood with Thompson Sampling:
#   sample α ~ Beta(α_success, β_failure)
#   return α
```

**Data Structure**:
```
(agent_id, outcome_type, message_category) → Beta(α, β)

Example:
("agent_a", "success", "affirm") → Beta(α=8, β=2)
("agent_a", "failure", "affirm") → Beta(α=3, β=7)
```

**Extension Points**:
- Custom message categorization via overriding `categorize_message`
- Custom prior distributions via database initialization
- Alternative exploration strategies via `use_thompson_sampling=False`

### 3. AgentReliabilityLearner (Reliability Weighting)

**Purpose**: Learn reliability weights α_i for each agent

**Responsibilities**:
- Record prediction results (was_correct, calibration_error)
- Maintain Beta distributions for reliability tracking
- Compute reliability weights in range [0.1, 5.0]
- Apply calibration-error-weighted updates
- Track reliability statistics per agent

**Key Design Decisions**:
- **Calibration-error weighting**: Well-calibrated agents get higher weights
- **Weight bounds**: Prevents extreme values that could dominate updates
- **Thompson Sampling**: Explores uncertain reliability estimates
- **Separate from observation model**: Reliability is about correctness, not message patterns

**Algorithm**:
```python
# Prior: Beta(α=1, β=1)
# Update with prediction result:
#   weight = 1.0 + (0.5 - calibration_error)
#   if was_correct:
#       α += weight
#   else:
#       β += weight

# Reliability weight with Thompson Sampling:
#   sample α ~ Beta(α_reliability, β_reliability)
#   return clip(α, min=0.1, max=5.0)
```

**Weight Interpretation**:
- α_i > 1.0: Upweight agent's evidence (more reliable than neutral)
- α_i = 1.0: Use evidence as-is (neutral reliability)
- α_i < 1.0: Downweight agent's evidence (less reliable)

**Extension Points**:
- Custom weight computation via overriding `compute_weight`
- Custom weight bounds via configuration
- Alternative reliability metrics (e.g., Brier score, log loss)

### 4. VoIController (Query Decision Making)

**Purpose**: Compute Expected Value of Information for intelligent querying

**Responsibilities**:
- Compute VoI = E[H[Y|D] - H[Y|D,z]] - cost
- Integrate observation and reliability learners
- Rank agents by expected information gain
- Track VoI predictions vs. actual outcomes
- Provide query recommendations

**Key Design Decisions**:
- **Entropy reduction**: VoI measures expected decrease in uncertainty
- **Cost-aware**: Subtracts query cost from expected benefit
- **Integration**: Combines observation model (likelihood) and reliability (weight)
- **Outcome tracking**: Monitors how well VoI predictions match actual gains

**Algorithm**:
```python
def compute_voi(task_analysis, agent_id, query_cost):
    current_entropy = task_analysis.belief_entropy

    # For each possible outcome, compute expected posterior entropy
    expected_posterior_entropy = 0
    for outcome in ["success", "failure"]:
        likelihood = observation_learner.get_likelihood(
            agent_id, simulated_message, outcome
        )
        reliability = reliability_learner.get_reliability_weight(agent_id)

        # Compute posterior entropy for this outcome
        posterior = task_analysis.compute_posterior(
            task_analysis.outcome_belief,
            {k: likelihood**reliability for k in task_analysis.outcome_belief}
        )
        posterior_entropy = task_analysis.compute_entropy(posterior)

        # Weight by prior probability of outcome
        expected_posterior_entropy += (
            task_analysis.outcome_belief[outcome] * posterior_entropy
        )

    # VoI = expected entropy reduction - cost
    voi = (current_entropy - expected_posterior_entropy) - query_cost
    return voi
```

**Decision Rule**:
```python
if voi > 0:
    query(agent)  # Expected benefit exceeds cost
else:
    skip(agent)  # Not worth querying
```

**Extension Points**:
- Custom VoI computation via overriding `compute_voi`
- Alternative decision rules (e.g., UCB, Thompson Sampling on VoI)
- Cost functions (e.g., non-linear cost, time-dependent cost)

### 5. BayesianOrchestrationService (Integration Layer)

**Purpose**: Coordinate all Bayesian components for end-to-end workflow

**Responsibilities**:
- Create and manage belief states
- Update beliefs with agent messages
- Make query decisions using VoI
- Select best agents to query
- Record task outcomes for learning
- Track belief state evolution

**Key Design Decisions**:
- **Service pattern**: Single entry point for Bayesian orchestration
- **In-memory caching**: Belief states cached for fast access
- **History tracking**: All belief updates persisted to database
- **Clean separation**: Delegates to specialized components (learners, controllers)

**Workflow**:
```
1. create_belief_state()
   → Initialize prior P(Y|D)

2. should_query_agent()
   → VoI > cost?
   → YES: select_best_agent_to_query()

3. update_belief_with_message()
   → P(Y|D,z) ∝ P(Y|D) * P(z|Y)^α_i

4. Execute task → Get outcome Y

5. record_task_outcome()
   → Update all learners (observation, reliability, VoI)
```

**Caching Strategy**:
```python
# In-memory cache
self._belief_cache: Dict[str, BayesianTaskAnalysis] = {}

# LRU eviction when cache exceeds size
def cleanup_belief_state(belief_id):
    del self._belief_cache[belief_id]
```

**Extension Points**:
- Custom belief initialization via overriding `create_belief_state`
- Custom update rules via overriding `update_belief_with_message`
- Custom outcome recording via overriding `record_task_outcome`

### 6. BayesianConsensusBuilder (Multi-Agent Consensus)

**Purpose**: Build consensus from multiple agent opinions with reliability weighting

**Responsibilities**:
- Analyze agent messages to extract votes
- Compute consensus using different strategies
- Update belief states with consensus
- Track consensus outcomes for learning
- Measure agreement levels among agents

**Key Design Decisions**:
- **Multiple strategies**: Majority vote (simple) and weighted Bayesian (sophisticated)
- **Reliability weighting**: More reliable agents have more influence
- **Agreement detection**: Identifies unanimous, partial, and divergent agreement
- **Belief integration**: Consensus updates belief state for downstream decisions

**Strategies**:

1. **Majority Vote**:
```python
def _majority_vote_consensus(agent_votes):
    success_votes = count(vote == "success" for vote in agent_votes)
    failure_votes = count(vote == "failure" for vote in agent_votes)

    if success_votes > failure_votes:
        return "success", success_votes / total_votes
    else:
        return "failure", failure_votes / total_votes
```

2. **Weighted Bayesian**:
```python
def _weighted_bayesian_consensus(belief_id, agent_votes, agent_messages):
    weighted_success_score = 0
    weighted_failure_score = 0

    for agent_id, vote in agent_votes:
        reliability = reliability_learner.get_reliability_weight(agent_id)

        if vote == "success":
            weighted_success_score += reliability
        else:
            weighted_failure_score += reliability

    if weighted_success_score > weighted_failure_score:
        return "success", weighted_success_score / total_weight
    else:
        return "failure", weighted_failure_score / total_weight
```

**Agreement Levels**:
```python
def _compute_agreement_level(agent_votes):
    success_ratio = success_votes / total_votes

    if success_ratio in [0.0, 1.0]:
        return "unanimous"  # All agents agree
    elif success_ratio >= 0.7 or success_ratio <= 0.3:
        return "partial"  # Strong majority
    else:
        return "divergent"  # Split decision
```

**Extension Points**:
- Custom consensus strategies via new methods
- Custom message analysis via overriding `_analyze_messages`
- Custom agreement metrics via overriding `_compute_agreement_level`

## Data Flow

### Single-Agent Workflow

```
User Request
    ↓
BayesianOrchestrationService.create_belief_state()
    ↓
Prior: P(Y|D) = {success: 0.5, failure: 0.5}
Entropy: H[Y|D] = 0.693 nats
    ↓
VoIController.should_query(agent_a, cost=0.1)
    ↓
Compute VoI = E[H[Y|D] - H[Y|D,z]] - cost
    ↓
VoI = 0.3 > 0 → Query agent_a
    ↓
Agent response: "Yes, this works"
    ↓
ObservationModelLearner.get_likelihood(agent_a, "Yes, this works", "success")
    ↓
Likelihood: P("Yes, this works" | success) = 0.8
    ↓
AgentReliabilityLearner.get_reliability_weight(agent_a)
    ↓
Reliability: α_a = 1.4
    ↓
BayesianTaskAnalysis.compute_posterior(prior, likelihood^α_a)
    ↓
Posterior: P(Y|D,z) = {success: 0.8, failure: 0.2}
Entropy: H[Y|D,z] = 0.5 nats
    ↓
Execute task → Outcome: success
    ↓
BayesianOrchestrationService.record_task_outcome()
    ↓
Update ObservationModelLearner: α_success[affirm] += 1
Update AgentReliabilityLearner: α_reliability += weight
Update VoIController: record_voi_outcome(predicted=0.3, actual=0.19)
```

### Multi-Agent Consensus Workflow

```
User Request
    ↓
BayesianOrchestrationService.create_belief_state()
    ↓
Prior: P(Y|D) = {success: 0.5, failure: 0.5}
    ↓
Query 3 agents:
    - agent_a: "Yes, this works" (α_a = 1.4)
    - agent_b: "This will work" (α_b = 1.2)
    - agent_c: "Agreed" (α_c = 0.8)
    ↓
BayesianConsensusBuilder.compute_consensus()
    ↓
Analyze messages → Extract votes
    - agent_a: success (reliability 1.4)
    - agent_b: success (reliability 1.2)
    - agent_c: success (reliability 0.8)
    ↓
Weighted scores:
    - success: 1.4 + 1.2 + 0.8 = 3.4
    - failure: 0
    ↓
Consensus: success (confidence: 1.0)
Agreement: unanimous
    ↓
BayesianConsensusBuilder.compute_consensus_and_update_belief()
    ↓
Posterior: P(Y|D, all_messages) = {success: 0.95, failure: 0.05}
    ↓
Execute task → Outcome: success
    ↓
BayesianConsensusBuilder.record_consensus_outcome()
    ↓
Update all learners with agent contributions
```

## Database Schema Design

### Schema Principles

1. **Conjugate Prior Storage**: Store α and β parameters for Beta distributions
2. **Time-Series Tracking**: All history tables include timestamp
3. **Agent-Centric**: Most tables indexed by agent_id
4. **Outcome Tracking**: Link predictions to actual outcomes for learning

### Table Relationships

```
rl_observation_model (P(z|Y) likelihoods)
    ├── agent_id → rl_agent_reliability (α_i weights)
    └── message_category → BayesianConsensusBuilder._analyze_messages()

rl_agent_reliability (α_i weights)
    ├── agent_id → rl_observation_model
    └── alpha_reliability → BayesianTaskAnalysis.agent_reliability

rl_voi_history (VoI tracking)
    ├── agent_id → rl_agent_reliability
    └── predicted_voi → VoIController.compute_voi()

rl_belief_history (belief evolution)
    ├── belief_id → BayesianOrchestrationService._belief_cache
    └── agent_id → rl_agent_reliability

rl_bayesian_consensus (consensus outcomes)
    ├── belief_id → rl_belief_history
    └── agent_contributions (JSON) → agent_id
```

### Index Design

```sql
-- Observation model: Fast lookup by agent
CREATE INDEX idx_observation_agent ON rl_observation_model(agent_id);

-- Reliability: Fast lookup by agent
CREATE INDEX idx_reliability_agent ON rl_agent_reliability(agent_id);

-- VoI history: Time-series queries
CREATE INDEX idx_voi_timestamp ON rl_voi_history(timestamp);

-- Belief history: Fast retrieval by belief_id
CREATE INDEX idx_belief_belief_id ON rl_belief_history(belief_id);

-- Consensus: Time-series queries
CREATE INDEX idx_consensus_timestamp ON rl_bayesian_consensus(timestamp);
```

## Concurrency and Consistency

### SQLite Concurrency

- **Write-ahead logging (WAL)**: Enables concurrent reads
- **Connection pooling**: Each learner has its own connection
- **Transaction isolation**: Each update in its own transaction

### Cache Coherence

```python
# In-memory cache + database persistence
self._belief_cache: Dict[str, BayesianTaskAnalysis] = {}

# Update path:
# 1. Update in-memory belief state
# 2. Persist to rl_belief_history
# 3. Cache remains coherent (single-threaded access)
```

### Race Conditions

- **Mitigation**: Single-threaded orchestration service
- **Future work**: Multi-agent concurrent updates with locks

## Performance Considerations

### Computational Complexity

| Component | Operation | Complexity |
|-----------|-----------|------------|
| BayesianTaskAnalysis | Posterior update | O(n_outcomes) |
| ObservationModelLearner | Likelihood query | O(1) with cache |
| AgentReliabilityLearner | Weight query | O(1) with cache |
| VoIController | VoI computation | O(n_outcomes^2) |
| BayesianConsensusBuilder | Consensus (N agents) | O(N) |

### Memory Usage

- **Belief state cache**: ~1 KB per belief state
- **Beta distribution cache**: ~100 bytes per (agent, outcome, category)
- **Recommended cache size**: 1000 belief states ~1 MB

### Database Size

- **Per observation**: ~200 bytes in rl_observation_model
- **Per prediction**: ~150 bytes in rl_agent_reliability
- **Per belief update**: ~250 bytes in rl_belief_history
- **1000 tasks**: ~600 KB total

## Extension Points

### Custom Prior Distributions

```python
# Override belief initialization
class CustomOrchestrationService(BayesianOrchestrationService):
    def create_belief_state(self, task_type, complexity, tool_budget, initial_belief):
        # Custom prior based on task complexity
        if complexity == TaskComplexity.SIMPLE:
            prior = {"success": 0.8, "failure": 0.2}
        else:
            prior = {"success": 0.5, "failure": 0.5}

        return super().create_belief_state(
            task_type, complexity, tool_budget, prior
        )
```

### Custom Likelihood Functions

```python
# Override observation model
class CustomObservationModel(ObservationModelLearner):
    def get_likelihood(self, agent_id, message, outcome):
        # Custom likelihood using semantic similarity
        message_embedding = self.embed(message)
        outcome_embedding = self.embed(outcome)

        similarity = cosine_similarity(message_embedding, outcome_embedding)
        return similarity
```

### Custom Consensus Strategies

```python
# Add new consensus strategy
class CustomConsensusBuilder(BayesianConsensusBuilder):
    def compute_consensus(self, belief_id, agent_messages, strategy="custom"):
        if strategy == "custom":
            return self._custom_consensus(belief_id, agent_messages)
        else:
            return super().compute_consensus(belief_id, agent_messages, strategy)

    def _custom_consensus(self, belief_id, agent_messages):
        # Custom consensus logic
        # e.g., Deweyan aggregation, Borda count, etc.
        pass
```

## Testing Strategy

### Unit Tests

- **BayesianTaskAnalysis**: Posterior updates, entropy, VoI
- **ObservationModelLearner**: Recording, likelihood, calibration
- **AgentReliabilityLearner**: Recording, weights, statistics
- **VoIController**: Computation, decisions, ranking
- **BayesianOrchestrationService**: Workflow, caching, history
- **BayesianConsensusBuilder**: Consensus, strategies, agreement

### Integration Tests

- **Full workflow**: Create belief → Query agents → Update → Record outcome
- **Multi-agent consensus**: 3+ agents with different reliabilities
- **Persistence**: Database recovery after restart

### Monte Carlo Tests

- **Convergence**: Do beliefs converge to true outcomes?
- **Calibration**: Are predicted probabilities well-calibrated?
- **VoI accuracy**: Do VoI predictions match actual gains?

## Future Enhancements

### Short-term

1. **CorrelationTracker**: Detect dependence between agent predictions
2. **Advanced message categorization**: Use embeddings instead of keywords
3. **Non-parametric priors**: Dirichlet processes for unknown outcome spaces

### Long-term

1. **Multi-armed bandit integration**: Thompson Sampling on agent selection
2. **Causal inference**: Detect and adjust for confounding variables
3. **Meta-learning**: Learn priors from similar tasks

## References

- **Paper**: "Position: agentic AI orchestration should be Bayes-consistent" (arXiv:2605.00742, ICML 2026)
- **Theory**: Bayesian decision theory, conjugate priors, Thompson sampling
- **Application**: Multi-agent systems, uncertainty quantification

## See Also

- [Bayesian Orchestrator Source](https://github.com/anvai-labs/victor/blob/develop/victor/framework/rl/orchestration/bayesian_orchestrator.py) — Implementation
- [Bayesian Configuration Source](https://github.com/anvai-labs/victor/blob/develop/victor/framework/bayesian_config.py) — Configuration definitions
- [Architecture Overview](../architecture.md) — Victor architecture
