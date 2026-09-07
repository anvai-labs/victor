# ADR-022: Provider Gateway Feature Layer and Routing Performance

## Metadata

- **Status**: Proposed
- **Date**: 2026-07-29
- **Decision Makers**: Vijaykumar Singh
- **Related ADRs**: 006 (provider integration improvements), 018 (adopt `sandhi` usage gateway — the
  transport/metering layer this sits above)
- **Work tracked by**: [TD-24](../../tech-stack.md#technical-debt-register); depends on
  [TD-21](../../tech-stack.md#technical-debt-register) (typed provider runtime / sandhi)
- **Benchmark**: [competitive-benchmark-2026-07.md](../competitive-benchmark-2026-07.md) §3

## Context

Victor has 25 provider adapters behind `BaseProvider`, cost/latency `smart_router.py` (on by
default per TD-17's correction), resilience (circuit breaker + retry, `resilience.py`), and
prompt/KV-prefix caching. The typed *transport + usage-metering* layer is being homed in `sandhi`
(Rust; TD-21, ADR-018) — retries, HTTP/SSE, structured errors, and `UsageV2` move under FFI.

But the **feature/policy layer that gateways compete on** is thin or absent, and it is distinct from
transport:

- **Semantic caching: absent.** Only exact prompt-cache / KV-prefix. Portkey-class gateways cache on
  embedding similarity.
- **Budget guardrails: behind.** Cost is *tracked* (C0 metric) but not *enforced* — there is no
  hard stop when a session/user/team crosses a budget. Portkey ships virtual-key budgets.
- **Declarative fallback chains: behind.** Resilience retries a call; there is no user-declared
  model *ladder* ("opus → sonnet → local") as a first-class config. LiteLLM makes fallback lists a
  one-liner.
- **Throughput: at risk / unmeasured.** The router is Python. The field's own benchmarks show
  LiteLLM's Python proxy degrading sharply past ~500 RPS single-instance. Victor has no published
  routing-throughput number and a Rust hot-path pattern (`_NATIVE_AVAILABLE`) already exists for
  exactly this kind of acceleration.

These are *policy above transport*: sandhi decides *how* a call is made; this layer decides *whether,
which model, from cache or not, within budget*.

## Decision

Establish a **provider gateway feature layer** that sits above the sandhi transport runtime and owns
four policies, plus a routing-performance decision:

1. **Semantic response cache** — optional, embedding-similarity cache keyed on normalized request +
   model + policy fingerprint, with explicit staleness/TTL and a per-profile enable flag (default
   OFF, graduated per the flag-graduation policy / TD-17).
2. **Budget guardrails** — hard `enforce` mode layered on the existing C0 cost tracking: per
   session/user/team ceilings that fail closed (or downgrade to a cheaper fallback rung) at the
   limit, integrated with sandhi virtual keys where present.
3. **Declarative fallback chains** — a user-declared model ladder in profile/settings config,
   compiled into the router so `smart_router` selects down the ladder on error/budget/latency instead
   of ad-hoc retry.
4. **Routing performance** — treat the router as a hot path: publish a routing-throughput number,
   and if it confirms a Python ceiling, move the selection/scoring inner loop to the Rust
   `_NATIVE_AVAILABLE` pattern with a pure-Python fallback (consistent with the existing native
   crates).

Scope boundary: this layer does **not** re-implement transport/metering (that is sandhi/TD-21). It
is the decision plane above it.

## Rationale

- **First principles.** A gateway's value over a raw SDK is the *policy* it applies to every call —
  cache, budget, fallback, route. Transport parity (sandhi) is table stakes; the policy layer is
  where Victor is measured against Portkey/LiteLLM.
- **Reuse.** Cost tracking (C0), the router, resilience, the embedding stack, and the Rust
  fallback pattern all exist; this composes them into declared policies rather than building anew.
- **Co-design.** Every new policy defaults OFF and enters via the existing flag-graduation gate
  (TD-17), so it ships measured, not asserted.

## Consequences

- **Positive**: closes the §3 gaps (semantic cache, budgets, fallback, throughput); budgets make cost
  a hard constraint, not a post-hoc report.
- **Negative**: semantic cache adds a correctness risk (stale/near-miss hits) — must be conservative
  and off by default; a Rust router raises build/maintenance surface (guarded by the existing native
  fallback).
- **Neutral**: adapters and sandhi transport are unchanged; providers behind `BaseProvider` see no
  interface change.

## Implementation

- **Companion FEP likely required**: the fallback-chain config and budget-enforcement modes are
  public `victor.framework`/settings surface — land the config schema via a FEP, then implement.
- Sequence: (1) declarative fallback chains (config + router wiring); (2) budget enforce mode on C0;
  (3) semantic cache behind a graduated flag with a gate corpus; (4) route-perf benchmark → Rust
  inner loop only if the number justifies it.

## Alternatives Considered

- **Delegate everything to sandhi.** Rejected: sandhi is transport/metering; cache/budget/fallback
  *policy* is Victor's product surface and profile-driven.
- **Adopt LiteLLM/Portkey as the gateway.** Rejected: Victor already owns typed adapters + router +
  contracts; adopting an external Python proxy re-introduces the RPS ceiling and a second config
  plane. Borrow their *features*, not their runtime.
- **Skip semantic cache (correctness risk).** Partially adopted: it ships OFF-by-default and gated,
  not omitted.

## References

- [ADR-006](006-provider-integration-improvements.md), [ADR-018](018-adopt-sandhi-usage-gateway.md)
- [TD-21](../../tech-stack.md#technical-debt-register), [flag-graduation-policy.md](../flag-graduation-policy.md)
- `victor/providers/base.py`, `victor/providers/smart_router.py`, `victor/providers/resilience.py`,
  FEP-0020 (usage attribution)

## Revision History

| Date | Version | Changes | Author |
|------|---------|---------|--------|
| 2026-07-29 | 1.0 | Initial ADR — gateway feature layer above sandhi transport | Vijaykumar Singh |
