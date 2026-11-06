# Provider-Agnostic Decision Service Tiers

## Executive Summary

Refactored the decision service tier system to be provider-agnostic. Instead of hardcoding tiers to specific providers (edge→ollama, balanced→deepseek, performance→anthropic), implemented a system where each provider defines their own edge/balanced/performance models, and the decision service auto-detects the active provider and uses the appropriate model from that provider's lineup.

**Problem Solved**: If a user's main provider is OpenAI, compaction previously routed to Anthropic for performance tier (inconsistent, wasteful, requires multiple API keys). With provider-agnostic tiers, all decision service calls use the same provider as the main LLM.

## Key Design Principles

- **Provider consistency**: Decision service uses same provider as main LLM
- **Cost efficiency**: No cross-provider API key management
- **Flexibility**: Easy to add new providers and models
- **Backward compatibility**: Existing configs continue to work
- **Override capability**: Explicit tier/provider/model when needed

## Implementation Summary

### Phase 1: DecisionServiceSettings (victor/config/decision_settings.py)

**Changes**:
- Changed default `provider` from hardcoded values to `"auto"` for all tiers
- Changed default `model` from hardcoded values to `"auto"` for all tiers
- Added `provider_model_tiers` dictionary with mappings for 18 providers:
  - Cloud: anthropic, openai, google, xai, deepseek, mistral, zai, moonshot
  - Aggregators: groqcloud, cerebras, together, fireworks
  - Enterprise: azure, bedrock, vertex
  - Local: ollama, lmstudio, vllm
- Added `tier_overrides` dictionary for explicit override capability
- Added `validate_provider_tiers()` method for configuration validation

**Example Provider Mapping**:
```python
"anthropic": {
    "edge": "claude-haiku-4-5-20251001",
    "balanced": "claude-sonnet-4-20250514",
    "performance": "claude-opus-4-5-20251101",
},
"openai": {
    "edge": "gpt-4o-mini",
    "balanced": "gpt-4o",
    "performance": "o3-mini",
}
```

### Phase 2: Auto-Detection Logic (victor/agent/services/tiered_decision_service.py)

**Changes**:
- Added `_detect_active_provider()` method that checks multiple sources:
  1. Service container (ProviderService/ProviderManager)
  2. Settings.default_provider
  3. Environment variable VICTOR_PROVIDER
- Added provider detection caching for performance
- Modified `_create_service()` to resolve `provider="auto"` and `model="auto"`
- Added tier override support
- Enhanced logging to show resolved provider/model

**Auto-Detection Flow**:
```
_create_service(tier) →
  Check tier_overrides →
  Detect active provider →
  Resolve model from provider_model_tiers →
  Create LLMDecisionService
```

### Phase 3: Model Capabilities Configuration (victor/config/model_capabilities.yaml)

**Changes**:
- Added model definitions for newer models:
  - `claude-opus-4-5*` with reasoning capability
  - `gpt-4o-mini*` with high tool reliability
  - `o3-*` with reasoning capability
  - `gemini-2.5*` with thinking mode
  - `gemini-2.5-flash*`
- Enhanced model metadata for tier-appropriate models

### Phase 4: Configuration Validation (victor/config/decision_settings.py)

**Changes**:
- Added `validate_provider_tiers()` method that:
  - Checks all providers have required tiers (edge, balanced, performance)
  - Validates tier overrides reference valid tiers
  - Ensures overrides have provider and model keys
  - Returns validation result with errors/warnings

### Phase 5: Comprehensive Tests (tests/unit/agent/services/test_tiered_decision_service.py)

**Added Tests**:
- `test_default_model_specs_use_auto`: Verifies defaults use "auto"
- `test_provider_model_tiers_defined`: Checks all providers have 3 tiers
- `test_tier_override_capability`: Tests override functionality
- `test_validate_provider_tiers`: Tests validation method
- `test_auto_detection_from_settings`: Tests settings fallback
- `test_auto_detection_from_env_var`: Tests env var fallback
- `test_model_resolution_for_provider`: Tests model resolution
- `test_fallback_for_missing_tier`: Tests missing tier handling
- `test_tier_override_behavior`: Tests override application
- `test_provider_detection_caching`: Tests caching optimization

**Test Results**: 18/18 tests pass (100%)

## Migration Guide

### From Hardcoded Tiers to Provider-Agnostic

**Before (Hardcoded)**:
```python
# All users get same tiers regardless of their main provider
edge = DecisionModelSpec(provider="ollama", model="qwen3.5:2b")
balanced = DecisionModelSpec(provider="deepseek", model="deepseek-chat")
performance = DecisionModelSpec(provider="anthropic", model="claude-sonnet-4")
```

**After (Provider-Agnostic)**:
```python
# Each provider has their own tiers
edge = DecisionModelSpec(provider="auto", model="auto")
balanced = DecisionModelSpec(provider="auto", model="auto")
performance = DecisionModelSpec(provider="auto", model="auto")

# Provider model mappings
provider_model_tiers = {
    "anthropic": {
        "edge": "claude-3-5-haiku-20241022",
        "balanced": "claude-sonnet-4-20250514",
        "performance": "claude-opus-4-5-20251101",
    },
    "openai": {
        "edge": "gpt-4o-mini",
        "balanced": "gpt-4o",
        "performance": "o3-mini",
    },
}
```

### Backward Compatibility

**Existing Configs Continue to Work**:
```python
# Old-style explicit config still works
config = DecisionServiceSettings(
    edge=DecisionModelSpec(
        provider="ollama",  # Explicit provider (not "auto")
        model="qwen3.5:2b"
    )
)
# Behavior unchanged: uses Ollama for edge tier
```

**New Auto-Style Config**:
```python
# New-style auto config
config = DecisionServiceSettings()  # Defaults to "auto"
# Behavior: Auto-detects provider, uses appropriate model
```

## Usage Examples

### Example 1: Using with Anthropic

```python
from victor.config.decision_settings import DecisionServiceSettings
from victor.agent.services.tiered_decision_service import TieredDecisionService

# Uses default auto-detection
config = DecisionServiceSettings()
service = TieredDecisionService(config)

# If main LLM is Anthropic Claude Sonnet:
# - Edge tier → Claude Haiku
# - Balanced tier → Claude Sonnet
# - Performance tier → Claude Opus
```

### Example 2: Using with OpenAI

```python
# Same code, different provider
config = DecisionServiceSettings()
service = TieredDecisionService(config)

# If main LLM is OpenAI GPT-4o:
# - Edge tier → GPT-4o-mini
# - Balanced tier → GPT-4o
# - Performance tier → o3-mini
```

### Example 3: Using with Local Ollama

```python
# Same code, local provider
config = DecisionServiceSettings()
service = TieredDecisionService(config)

# If main LLM is Ollama:
# - Edge tier → qwen2.5:3b
# - Balanced tier → qwen2.5:7b
# - Performance tier → qwen2.5:32b
```

### Example 4: Override Edge Tier to Use Ollama

```python
# Force edge tier to use Ollama regardless of main provider
config = DecisionServiceSettings(
    tier_overrides={
        "edge": {"provider": "ollama", "model": "phi3:mini"}
    }
)
service = TieredDecisionService(config)

# Edge tier always uses Ollama phi3:mini
# Balanced/Performance use auto-detected provider
```

## Success Metrics

| Metric | Status | Notes |
|--------|--------|-------|
| Provider Consistency | ✅ | Decision service uses same provider as main LLM |
| Cost Efficiency | ✅ | Single provider API key management |
| Model Coverage | ✅ | 18 providers with 3 tiers each |
| Backward Compatibility | ✅ | Existing explicit configs work |
| Flexibility | ✅ | Easy to add new providers |
| Test Coverage | ✅ | 18/18 tests pass |

## Files Modified

| File | Changes | Lines Added |
|------|---------|-------------|
| `victor/config/decision_settings.py` | Added provider-agnostic tier system | ~200 |
| `victor/agent/services/tiered_decision_service.py` | Added auto-detection logic | ~180 |
| `victor/config/model_capabilities.yaml` | Added new model definitions | ~130 |
| `tests/unit/agent/services/test_tiered_decision_service.py` | Added comprehensive tests | ~170 |

## Benefits

1. **Provider Consistency**: Decision service now uses the same provider as the main LLM, eliminating inconsistencies
2. **Cost Efficiency**: No need to manage multiple provider API keys
3. **Flexibility**: Easy to add new providers - just add to `provider_model_tiers`
4. **Backward Compatibility**: Existing explicit configurations continue to work
5. **Better Resource Usage**: Leverages provider-specific strengths (e.g., xAI's 2M context window)

## Next Steps

1. ✅ Implementation complete
2. ✅ Tests passing
3. ✅ Documentation created
4. ⏭️ Integration testing with real providers
5. ⏭️ Performance benchmarking
6. ⏭️ User feedback collection
