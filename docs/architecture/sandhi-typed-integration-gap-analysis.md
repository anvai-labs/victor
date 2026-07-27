# Sandhi Typed-vs-Raw Integration: Gap Analysis & Contract Proposal

**Date:** 2026-07-24 · **Status:** Analysis (grounded review of both repos at HEAD)
**Scope:** victor `develop` (post-#646) × sandhi `main` (post-TD-0003). All file:line
references verified at review time; re-verify before implementing.

## 1. The boundary principle (what is already right)

The division of responsibility is sound and should be preserved:

| Concern | Owner | Evidence |
|---|---|---|
| Wire transport, typed neutral contract v1 | **Sandhi** | `ChatRequestV1`/`ChatResponseV1`/`ChatStreamEventV1`/`UsageV2` (`sandhi-core/src/chat.rs`) |
| Measured usage (never estimated) | **Sandhi** | `Metered(Resilient(Adapter))`, exactly-one-event, Drop-guarded streams (`sandhi-providers/src/metering.rs:27-175`) |
| Capability **facts**, catalog **data** (no pricing, no volatile models) | **Sandhi** | `provider_descriptor()` (`catalog.rs:253`); "Sandhi must not manufacture volatile capabilities" (`catalog.rs:177-200`) |
| Pricing, budgets, model policy, OAuth acquisition | **Victor** | `config/metrics_capabilities.py`, `provider_metrics.yaml`; TD-0002:159-165 |
| Transport swap chokepoint | **Victor** | `registry.py:248-253` → `resolve_transport_class()`; no silent SDK fallback (billing-duplication safety, `sandhi_transport.py:76-99`) |

**Answer to "is provider swapping seamless?": at the transport level, yes, already.**
Every provider resolves through one registry chokepoint into a Sandhi-typed policy
shell; adding an OpenAI-compat provider is pure config (`CONFIG_KEY`, e.g.
`deepseek_provider.py:23-26`). The residual friction is *not* transport — it is
metering fidelity and hand-maintained routing/capability lists (below).

## 2. Gap ledger (ranked by impact on speed/cost/budget tracking + swap friction)

### G1 — Latency/speed is not metered in Sandhi at all ⚠ highest impact
`UsageEvent` (`sandhi-core/src/event.rs:22`) and `UsageV2` carry **no duration, no
time-to-first-token, no throughput** — only `occurred_at`. Victor measures TTFT and
duration client-side (`agent/stream_handler.py:131-149`), so FFI-mode numbers include
Victor overhead and proxy-mode consumers get nothing at all. "Speed properly tracked
in Sandhi" is currently **false by construction**.
**Contract:** additive v1 fields on `UsageEvent`: `duration_ms`, `time_to_first_token_ms`
(streams), measured at the adapter boundary; surfaced on `ChatResponseV1` metadata and
the final `Usage` stream event. Victor keeps interpreting (throughput, routing policy);
Sandhi measures (wire truth). Additive-only within v1 per TD-0002:33.

### G2 — Token-class fidelity loss → cost silently wrong for reasoning models
`UsageV2` parses `reasoning_tokens`, `audio_*`, `accepted/rejected_prediction_tokens`
(`chat.rs:201-210`, parsed `typed.rs:709-718`) but `UsageEvent` drops them
(`event.rs:46-51`), and Victor's usage dict (`base.py:111`, `usage_parsing.py:163-207`)
never sees them. o-series/reasoning bills these tokens; Victor's cost accounting
(`SessionCostTracker`) undercounts.
**Contract:** additive `UsageEvent` fields + `usage_dict_from_neutral` mapping +
`ProviderMetricsCapabilities.calculate_cost()` pricing for reasoning tokens.

### G3 — Pricing fragmentation inside Victor (4 tables, 2 unit systems)
Canonical: `config/metrics_capabilities.py:80-118` + `provider_metrics.yaml` (per-mtok).
Duplicates: `workflows/cost_router.py:96-194`, `agent/model_switcher.py:66-137`,
`framework/observability/metrics.py:729-736` (per-1k). Drift is guaranteed; Sandhi
correctly excludes pricing, but Victor never consolidated its side of the bargain.
**Contract:** one `PricingResolver` keyed `(provider, model)` joining Sandhi catalog
model ids; delete the three duplicates; guard test forbidding `*_cost_per_*` literals
outside the canonical config.

### G4 — Retry/completeness diagnostics recorded, then dropped
`sandhi_transport.py:267-289,446-447,501` attaches `metadata["sandhi_usage"]` =
`{attempts, completeness, outcome, upstream_request_id}` to every completion and final
stream chunk. **Zero consumers exist in Victor.** This is exactly the observability the
tool-reliability program needs (retry rates, usage completeness as a truth signal).
**Contract:** `metrics_service.finalize_stream_metrics()` persists these alongside
`RequestCost`; expose in session cost/telemetry reports.

### G5 — Hand-maintained routing lists (the coupling hot-spot)
`_SANDHI_VARIANTS` + `VICTOR_NATIVE_ONLY_PROVIDER_ALIASES`
(`sandhi_transport.py:48-63,736-744`) + per-slug branches in `usage_parsing.py:57-61`
must be kept in lockstep with the registry *by hand* ("Keep the aliases synchronized
with the registry"). Sandhi already owns the truth: `ProviderFamily::for_slug`
(`typed.rs:29`).
**Contract:** expose `provider_family(slug)` / routability through the FFI; Victor
derives Sandhi-vs-native routing from the descriptor instead of lists; CI guard
comparing registry coverage ↔ descriptor coverage.

### G6 — Capability provenance asymmetry
OpenAI-compat providers read capabilities from Sandhi's typed descriptor
(`openai_compat_model_policy.py:142-186`); Anthropic/Google/OpenAI hardcode
`supports_*` overrides; `BaseProvider.discover_capabilities()` (`base.py:890-907`)
exists but is config-only — the runtime-discovery contract is unwired.
**Contract:** descriptor-backed `discover_capabilities()` default for all
Sandhi-routed providers; hardcoded overrides become fallbacks.

### G7 — Native-codec residue (the true "gap #2" tail)
Transport-wise, `SandhiAnthropicProvider`/`SandhiGoogleProvider` exist and are wired
(`sandhi_transport.py:624-744`) — the memory's "Anthropic+Google migration not done"
is stale at the transport level. What remains: Anthropic wraps native Messages params
inside `extensions["anthropic"]` of an OpenAI-shaped neutral request
(`sandhi_transport.py:636-655`); Google ships a bare neutral payload with params under
`extensions["google"]` (`:566-592`) even though Sandhi has a Gemini-native typed codec
(`gemini_typed.rs`). Recurring native params ride untyped `serde_json::Value`
extensions.
**Contract:** promote recurring extension params to typed v1 fields (additive) or
per-family typed extension schemas; route Google through the Gemini-native codec.

### G8 — Response bloat crossing the FFI
Every typed response re-embeds the **entire provider-native body** under
`extensions["openai"]` (`typed.rs:647,669-671`) — payload cost on every call for a
debugging affordance hosts are told not to rely on.
**Contract:** request flag `include_native_response` (default off).

### G9 — Version machinery is policy, not mechanism
`ChatRequestV1::validate` checks only `schema_version == "1"` (`chat.rs:162-167`);
`wire_contract_version()` is hardcoded. Additive-only-within-v1 is documented
(TD-0002:33) but unenforced.
**Contract:** minor-version negotiation in the handshake + a Victor CI check pinning
the hash of consumed `schemas/*.schema.json` (schemas are already checked into sandhi).

### G10 — Smaller, known items
Node/Python FFI runtime glue is duplicated hand-written source (facade generator
covers models only); Bedrock has a usage parser but an empty transport module
(`lib.rs:72-75`); `FnProvider.stream()` hard-501s (`escape_hatch.rs:47-51`); proxy
budget reservation uses a bytes/4 heuristic (`proxy lib.rs:744-757`).

## 3. Sequencing (additive-only, cohesion-preserving)

- **Wave 1 — tracking correctness:** G1 + G2 (Sandhi additive fields) → G4 (Victor
  consumes diagnostics) → G3 (Victor-only pricing consolidation). After this wave,
  speed/cost/budget are *measured in Sandhi, priced in Victor* with no fidelity loss.
- **Wave 2 — decoupling:** G5 + G6 (descriptor-driven routing & capabilities). After
  this wave, adding/swapping a provider touches config + descriptor only.
- **Wave 3 — typed purity:** G7 + G8, then G9 machinery.

## 4. Corrections to prior working notes

- The typed-runtime spec in the sandhi repo is **TD-0002** (`docs/td/TD-0002-typed-
  provider-runtime.md`); sandhi has no TD-0004 — the transport-vs-catalog boundary
  work tracked as "TD-0004" lives on the Victor side.
- "Sandhi gap #2 (Anthropic+Google) not done" is stale: typed transport variants are
  wired; the remaining work is codec purity (G7), not transport migration.
