# Feature Flags Guide

**Feature flags enable gradual rollout and experimentation in Victor.**

## Overview

Victor uses feature flags to control the availability of experimental features, phased rollouts, and architectural transitions. Feature flags allow:

- **Gradual rollout** — Enable features incrementally for users
- **A/B testing** — Compare different implementations
- **Quick rollback** — Disable features without deploying
- **Architecture migrations** — Transition between implementations safely

## Using Feature Flags

### Checking Feature Flags

```python
from victor.core.feature_flags import feature_flags, FeatureFlag

# Check if a feature is enabled
if feature_flags.is_enabled(FeatureFlag.USE_EDGE_MODEL):
    # Use edge model for micro-decisions
    service = create_edge_decision_service()
else:
    # Use fallback
    service = create_heuristic_service()

# Check with default value
enabled = feature_flags.is_enabled(
    FeatureFlag.EXPERIMENTAL_FEATURE,
    default=False
)
```

### Setting Feature Flags

Feature flags can be set via:

1. **Environment variables** (highest priority)
2. **Configuration file** (`~/.victor/settings.yaml`)
3. **Default values** (in code)

```bash
# Via environment variable
export VICTOR_USE_EDGE_MODEL=true
export VICTOR_USE_PROVIDER_POOLING=false
```

```yaml
# Via settings.yaml
feature_flags:
  use_edge_model: true
  use_provider_pooling: false
```

## Available Feature Flags

### AI/ML Features

#### `use_edge_model`

**Purpose**: Enable fast edge model for micro-decisions instead of cloud LLM

**Default**: `false`

**Configuration**:
```yaml
edge_model:
  enabled: true  # Master toggle for edge model
```

**Environment Variable**: `VICTOR_USE_EDGE_MODEL=true`

**See Also**: [Edge Model Guide](./EDGE_MODEL.md), [FEP-0001](../feps/fep-0001-edge-model.md)

**Impact**:
- Reduces token usage by ~500M tokens/year
- Improves latency for micro-decisions (<2s vs 2-5s)
- Requires Ollama with small model (1.1-3GB)

**Migration**: See [Edge Model Guide](./EDGE_MODEL.md) for setup instructions

---

> **Note**: `use_semantic_tool_selection` and `use_provider_pooling` are **Settings** keys
> (`tool_settings.use_semantic_tool_selection`, `providers.pooling_enabled`), not
> `FeatureFlag` enum members. See [Configuration Guide](../reference/configuration-options.md).

---

### Architecture & Refactoring

#### `use_composition_over_inheritance`

**Purpose**: Phase 4 — Use composition-based verticals instead of inheritance

**Default**: `false`

**Environment Variable**: `VICTOR_USE_COMPOSITION_OVER_INHERITANCE=true`

**Status**: 🔄 In Development

**Impact**:
- Changes vertical integration architecture
- Required for external vertical packages
- Breaking change for custom verticals

**Migration**: See [Vertical Migration Guide](../development/testing/vertical-migration-summary-2026-03-04.md)

---

#### `use_strategy_based_tool_registration`

**Purpose**: Phase 5 — Use strategy pattern for tool registration

**Default**: `false`

**Environment Variable**: `VICTOR_USE_STRATEGY_BASED_TOOL_REGISTRATION=true`

**Status**: ⏳ Planned

**Impact**:
- More flexible tool registration system
- Enables dynamic tool discovery
- Breaking change for tool providers

**Migration**: Not yet ready for production use

---

### Service Layer (Phase 3) — Complete ✅

> **Status**: All core agent services are now mandatory and bootstrapped
> through the canonical runtime/service bundle.

The following service layer flags have been removed as services are now always enabled:
- ~~`use_new_chat_service`~~ → ChatService is mandatory
- ~~`use_new_tool_service`~~ → ToolService is mandatory
- ~~`use_new_context_service`~~ → ContextService is mandatory
- ~~`use_new_provider_service`~~ → ProviderService is mandatory
- ~~`use_new_recovery_service`~~ → RecoveryService is mandatory
- ~~`use_new_session_service`~~ → SessionService is mandatory

**Impact**:
- Cleaner separation of concerns (completed)
- Better testing and mocking (completed)
- Foundation for future features (completed)

**Migration**: No action needed. Service-selection flags and their environment
variables are no longer supported; runtime wiring now resolves the canonical
service bundle automatically.

---

> **Note**: `use_emojis` and `use_mcp_tools` are **Settings** keys (`ui.use_emojis`,
> `mcp.enabled`), not `FeatureFlag` enum members.
> See [Configuration Guide](../reference/configuration-options.md).

---

### UI & Output

### Integration

## Feature Flag Lifecycle

### Phase 1: Development

```python
# Add new feature flag to enum
class FeatureFlag(str, Enum):
    MY_NEW_FEATURE = "use_my_new_feature"
```

### Phase 2: Testing

```yaml
# Enable for testing
feature_flags:
  use_my_new_feature: true
```

### Phase 3: Gradual Rollout

```bash
# Enable for specific users/groups
export VICTOR_USE_MY_NEW_FEATURE=true
```

### Phase 4: Default Enable

```python
# Change default in code
DEFAULT_FLAGS = {
    FeatureFlag.MY_NEW_FEATURE: True,  # Was False
}
```

### Phase 5: Removal

```python
# Remove flag after full rollout
# Delete old code path
# Remove flag from enum
```

## Best Practices

### 1. Default to Disabled

New features should default to `false` to avoid disrupting existing users:

```python
DEFAULT_FLAGS = {
    FeatureFlag.EXPERIMENTAL_FEATURE: False,  # Safe default
}
```

### 2. Document Dependencies

If a feature flag depends on other flags or configuration:

```python
"""
FeatureFlag.MY_FEATURE requires:
- FeatureFlag.DEPENDENCY_FEATURE enabled
- external_package installed
- API_KEY configured
"""
```

### 3. Provide Migration Path

Users need a clear path when flags are removed:

```python
# Deprecated: Remove in v1.0
if feature_flags.is_enabled(FeatureFlag.OLD_FEATURE):
    warnings.warn(
        "OLD_FEATURE is deprecated and will be removed in v1.0. "
        "Use NEW_FEATURE instead."
    )
```

### 4. Monitor Usage

Track feature flag usage to guide rollout decisions:

```python
if feature_flags.is_enabled(FeatureFlag.EXPERIMENTAL):
    analytics.track("experimental_feature_used")
    # ... feature code
```

### 5. Safe Fallback

Always provide a fallback when features are disabled:

```python
if feature_flags.is_enabled(FeatureFlag.NEW_FEATURE):
    result = new_implementation()
else:
    result = old_implementation()  # Fallback
```

## Troubleshooting

### Feature Flag Not Working

**Symptom**: Feature flag change has no effect

**Solutions**:
1. Check environment variable spelling: `VICTOR_USE_FEATURE_NAME`
2. Verify configuration file location: `~/.victor/settings.yaml`
3. Check for conflicting settings (env var > config file > default)
4. Enable debug logging: `victor chat --log-level debug`

### Multiple Flags Interacting

**Symptom**: Unexpected behavior when multiple flags enabled

**Solutions**:
1. Check flag dependencies in documentation
2. Test flags individually before combining
3. Review flag interaction matrix in docs

### Performance Issues

**Symptom**: Enabling a flag causes slowdown

**Solutions**:
1. Profile with and without flag: `victor chat --profile`
2. Check flag documentation for known performance impacts
3. Report issue if not documented

## References

- Feature flag implementation: `victor/core/feature_flags.py` (no dedicated ADR exists; graduation policy is tracked as TD-17 in the [tech-debt register](../tech-stack.md#technical-debt-register))
- [Edge Model Guide](./EDGE_MODEL.md)
- [Configuration Guide](../reference/configuration-options.md)
- [Architecture Decision Records](../architecture/adr/README.md)
