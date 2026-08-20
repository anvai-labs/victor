---
fep: "0011"
title: "Characterized provider cache cost model (replace boolean cache flags)"
type: Standards Track
status: Draft
created: 2026-06-29
modified: 2026-06-29
authors:
  - name: Vijaykumar Singh
    email: vijay@anvaiops.com
    github: vjsingh1984
reviewers: []
discussion: https://github.com/anvai-labs/victor/discussions/0011
---

# FEP-0011: Characterized provider cache cost model

## Summary

Victor asks providers two boolean questions about caching —
`supports_prompt_caching()` (`victor/providers/base.py:411`) and
`supports_kv_prefix_caching()` (`base.py:433`) — and the prompt assembler
branches on those bits. A boolean collapses a rich physical reality (cache
discount %, TTL, minimum prefix granularity, write latency vs read latency,
max cacheable tokens) into one bit, so the assembler cannot optimize to the
actual numbers. It over-prunes for a provider with a 90% discount and 5-minute
TTL the same way it would for a 50% discount with a 30-second TTL.

This FEP proposes a characterized `CacheCostModel` that each provider
advertises, consumed by the prompt assembler to make pruning / stable-prefix /
dynamic-injection decisions against real values. The existing booleans become
**derived properties** for backward compatibility, so adoption is incremental
and non-breaking.

## Motivation

### Problem Statement

- **Lossy seam.** The cloud/local caching seam (API prompt caching for billing
  vs. KV prefix caching for latency) is the provider↔assembler co-design
  boundary. Branching on `bool` means the assembler picks one of two strategies
  instead of optimizing a continuous tradeoff (how hard to prune, whether to
  split stable/dynamic prefix, how large a prefix to freeze).
- **Co-design critique.** The transcript audit flagged this: "Co-design would
  have the provider advertise a cache cost model and let the prompt assembler
  optimize to the actual numbers rather than branch on True/False."
- **Hidden provider differences.** Anthropic (90% read discount, 5-min TTL),
  OpenAI (tiered, prefix-minimum), and local KV caches (no billing, latency-only)
  are all reduced to the same `True`.

### Goals

1. A `CacheCostModel` value object describing the cache economics of a provider.
2. `BaseProvider` exposes `cache_cost_model()`; the booleans derive from it.
3. The prompt assembler reads the model and tunes pruning/prefix strategy to the
   numbers (bounded by sane defaults so a sparse model still behaves well).
4. Non-breaking: providers that only implement the booleans keep working.

### Non-Goals

- Changing the KV-vs-API dichotomy itself (two independent capabilities remain).
- Provider billing/accounting. This is about informing prompt assembly, not
  metering.
- Removing the boolean methods (they stay as derived compat shims).

## Proposed Change

### High-Level Design

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class CacheCostModel:
    """Characterized caching economics for a provider capability."""
    supported: bool = False
    read_discount: float = 0.0     # 0.0–1.0 fraction saved on cached read
    write_overhead: float = 1.0    # cost multiplier for a cache write (>=1.0)
    ttl_seconds: float = 0.0       # 0 = no TTL / indefinite
    min_prefix_tokens: int = 0     # minimum stable prefix to be eligible
    max_cache_tokens: int = 0      # 0 = no published limit
    prefix_granularity: str = "token"  # "token" | "message" | "system_block"

    def effective(self) -> bool:
        return self.supported
```

### Detailed Specification

#### BaseProvider

```python
class BaseProvider(ABC):
    def cache_cost_model(self) -> CacheCostModel:
        """Override to advertise real cache economics. Default: derived from
        the legacy booleans so existing providers work unchanged."""
        return CacheCostModel(supported=self.supports_prompt_caching())

    # Legacy booleans — now derived, kept for backward compatibility.
    def supports_prompt_caching(self) -> bool:
        return self.cache_cost_model().supported
```

A provider that wants to participate overrides `cache_cost_model()`:

```python
class AnthropicProvider(BaseProvider):
    def cache_cost_model(self) -> CacheCostModel:
        return CacheCostModel(
            supported=True,
            read_discount=0.9,
            write_overhead=1.25,
            ttl_seconds=300,
            min_prefix_tokens=1024,
            max_cache_tokens=90_000,
            prefix_granularity="system_block",
        )
```

#### Prompt assembler consumption

The assembler (and KV optimization path) consult the model instead of the bit:

- **Pruning aggressiveness** scales inversely with `read_discount`: a 0.9
  discount justifies keeping a larger stable prefix; a 0.1 discount prunes hard.
- **Stable/dynamic split** is worthwhile when `ttl_seconds` is large enough to
  amortize the `write_overhead` across turns.
- **Prefix freeze size** respects `min_prefix_tokens` and `max_cache_tokens`.
- **Fallback**: a model with only `supported=True` (sparse) yields conservative
  defaults — current behavior — so unported providers are unaffected.

#### Settings

A `cache_optimization.aggressiveness` knob ("conservative" | "balanced" |
"aggressive") modulates how tightly the assembler follows the model, preserving
the existing `cache_optimization_enabled` / `kv_optimization_enabled` gates.

### API Changes

- **New**: `CacheCostModel`, `BaseProvider.cache_cost_model()`.
- **Modified**: `supports_prompt_caching()` / `supports_kv_prefix_caching()` become
  derived (default behavior unchanged for providers that don't override).
- **Deprecated** (later): direct branching on the booleans inside the assembler
  in favor of the model.

### Configuration Changes

```yaml
cache_optimization:
  enabled: true
  aggressiveness: balanced   # new
```

### Dependencies

None.

## Benefits

### For the Framework

- The provider↔assembler seam stops being a coin flip; it optimizes to measured
  economics. This is the characterized-vs-booleanized fix from the co-design
  audit.
- New providers (and local KV backends) can advertise real latency/discount
  curves instead of being bucketed.

### For Vertical Developers

- No change unless they ship a custom provider; if they do, overriding
  `cache_cost_model()` gives them precise control over how the assembler treats
  their cache.

## Drawbacks and Alternatives

### Drawbacks

- More parameters for provider authors to reason about. Mitigation: sparse
  defaults; unported providers behave exactly as today.
- Risk of over-tuning the assembler to advertised numbers that drift from
  reality. Mitigation: the existing adaptive-dispatch work (FEP-0010-adjacent
  Gap 3) is the template — measured feedback can later recalibrate the model.

### Alternatives Considered

1. **Keep booleans, add separate `cache_*` numeric methods.** Rejected: scatters
   the economics across methods; the model is a single value object precisely so
   the assembler reasons about one thing.
2. **Per-provider hard-coded assembler branches.** Rejected: that is the
   anti-pattern this FEP removes.

## Unresolved Questions

- Should the model include a `write_amortization_turns` hint, or is deriving it
  from `ttl_seconds` + `write_overhead` enough? Proposed: derive.
- Do we need a separate model for KV-prefix vs API-prompt caching, or one model
  with a `kind` field? Proposed: one model per capability (two methods:
  `cache_cost_model()` for API, `kv_cache_cost_model()` for KV), mirroring
  today's two-method split.

## Implementation Plan

### Phase 1: Model + base ✅ (done, 2026-06-29)

- [x] Add `CacheCostModel`; `BaseProvider.cache_cost_model()` /
      `kv_cache_cost_model()` derived from the booleans.
- [x] Unit tests: derivation correctness, frozen/immutability
      (`tests/unit/providers/test_cache_cost_model.py`).

**Deliverable**: non-breaking model in place; all providers behave as before.

### Phase 2: Port flagship providers ✅ (done, 2026-06-29)

- [x] Override `cache_cost_model()` for Anthropic (0.9 discount, 1.25x write,
      300s TTL, 1024 min-prefix, system-block) and OpenAI (0.9, 1024 min-prefix,
      token granularity); override `kv_cache_cost_model()` for llama.cpp
      (latency-only, no discount).
- [x] Document real numbers (from each provider's docstring).
- [x] Tests: `TestFlagshipProviderCharacterization`.

**Deliverable**: characterized economics for the main providers. Remaining
providers port incrementally (they inherit the default derived model).

### Phase 3: Assembler consumption ✅ (done, 2026-06-29)

- [x] `detect_provider_tier()` / new `detect_cache_economics()` in
      `victor/agent/prompt_pipeline.py` consume `cache_cost_model()` /
      `kv_cache_cost_model()`, exposing tier + real numbers (discount/TTL/
      min-prefix) + a derived `pruning_aggressiveness`.
- [x] MagicMock-safe: the cost model is trusted only when it returns a real
      `CacheCostModel` (isinstance), so test doubles and unported providers
      fall back to the booleans unchanged.
- [x] Regression: prompt-pipeline + KV + cache-contract suites green.

**Deliverable**: the assembler's tier detection is model-driven; the
characterized numbers + aggressiveness are a first-class consumable surface.
Remaining: wire `pruning_aggressiveness` into the ContentRouter placement
decisions (currently exposed; placement still tier-based) and add the
`aggressiveness` setting.

### Testing Strategy

- Unit: model derivation, assembler decision function for representative models.
- Integration: token-cost A/B (cached vs uncached) for a characterized provider.
- Backward compat: a provider returning only `True` behaves identically to today.

## Migration Path

1. Phase 1 lands (non-breaking). Providers continue to override booleans.
2. Providers migrate to `cache_cost_model()` opportunistically.
3. Phase 3 switches the assembler to the model; booleans remain as derived
   compat shims indefinitely.

### Deprecation Timeline

- `v0.8.x`: `CacheCostModel` introduced; assembler still reads booleans.
- `v0.9.x`: assembler reads the model; boolean branching deprecated internally.
- Booleans retained as public derived properties (no removal planned).

## Compatibility

- **Breaking change**: No. Default behavior identical for unported providers.
- **Minimum Python**: 3.10 (unchanged).

## References

- Transcript co-design audit (characterize-vs-booleanize provider seam).
- `victor/providers/base.py:411` (`supports_prompt_caching`),
  `base.py:433` (`supports_kv_prefix_caching`).
- CLAUDE.md "Provider Caching Architecture" section.

## Review Process

- **Submitted by**: Vijaykumar Singh
- **Initial review period**: 14 days minimum.
- **PR**: TBD.

## Acceptance Criteria

### Must-Have

1. `CacheCostModel` + `BaseProvider.cache_cost_model()` merged; booleans derived.
2. At least 3 providers (1 cloud-caching, 1 cloud-non-caching, 1 local-KV)
   override the model with documented numbers.
3. Assembler reads the model; a sparse-model provider behaves as today
   (regression test).

### Should-Have

1. Token-cost A/B shows measurable win for a characterized provider vs the
   boolean baseline.
2. Docs page on authoring a `CacheCostModel`.

---

## Copyright

This FEP is licensed under the Apache License 2.0, same as the Victor project.
