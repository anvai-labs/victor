# Foundations Strategy: P6 Residue + Sandhi Wave 3 (2026-07)

**Status:** Active. **Scope:** sequences all remaining post-UX-plan work
foundation-first, so parallel paths never build on unsettled boundaries.
**Method:** every claim below was re-verified against the develop tip on
2026-07-25 (victor `222c7346d`, sandhi `5df5c00`); premise corrections are
called out explicitly where the older plan/ledger text no longer matches the
code.

---

## 1. Premise corrections (evidence first)

### 1.1 P6 ("migrate Chat/Tool/Session coordinators into services") is already delivered

The three coordinators **no longer exist** — they were deleted, not
deprecated: `victor/agent/coordinators/__init__.py:31` ("ToolCoordinator,
ChatCoordinator, SessionCoordinator have been removed. Use ToolService,
ChatService, SessionService instead."). `ChatServiceAdapter` does not exist
either; `services/adapters/` holds only `ContextServiceAdapter`
(`adapters/__init__.py:21` records the removals). The services own their
logic with zero delegation back; remaining `coordinators/` content is the
**state-passed** family — the preferred pattern, not a migration target.

What genuinely remains of P6:

- **Residue**: `SessionFacadeProtocol` (`agent/facades/protocols.py:131`) has
  zero readers — dead. Stale coordinator references survive only in
  docstrings/comments (`chat_service.py:73,1446`, `tool_service.py:628,1150`,
  `session_service.py:65`, `orchestrator.py:2209`,
  `coordinators/__init__.py:79` mentions a `chat_compat` module that does not
  exist).
- **Ratchet slack**: `tool_selection/selector.py` cap 2882 vs actual 2765 —
  117 free lines of tightening. Several hotspots sit exactly at cap
  (`chat_service.py` 1740/1740, `planning_runtime.py` 3518/3518,
  `runtime_intelligence.py` 2864/2864, `context_compactor.py` 1827/1827);
  `orchestrator.py` is at 4702/4704.
- **The long game**: orchestrator slimming continues as opportunistic
  ratchet-down (guard: `tests/unit/runtime/test_hotspot_size_guard.py`,
  lower-only), constrained by
  `test_service_layer_validation.py` delegation-point floors (chat ≥3,
  tool ≥5, session ≥3, context ≥2, provider ≥2).

### 1.2 G8's blocker is two fields in one function

Victor's native-body (`extensions[slug]`) dependence, measured:

- The **only** response-side reader is `_completion_from_typed`
  (`victor/providers/sandhi_transport.py:503-543`). The streaming path
  (`_sandhi_stream`, `:545-623`) is fully typed and never touches native
  bodies.
- Neutral usage is already authoritative for every token count
  (`usage_parsing.py:163-214`); the native body contributes only a
  `total_tokens` cross-check (compute fallback exists) and a reasoning-text
  fallback (typed `extensions["reasoning"]` is preferred).
- The single load-bearing downstream consumer of `raw_response` is
  `services/turn_execution_runtime.py:1986-2006`, which reads four fields —
  two **redundant** with neutral usage (`cached_tokens`, `reasoning_tokens`)
  and two **genuinely native-only**: `prompt_cache_miss_tokens` and
  `cost_in_usd_ticks`. Cost from the provider body also contradicts the
  settled boundary (Victor owns pricing — G3, shipped in #651).

---

## 2. Dependency graph and sequencing

```
F0  this document (design gate for everything below)
F1  G8-blocker reduction (victor)      ── unblocks ──▶  W3a sandhi native-body gating (G8)
F2  P6 residue + ratchet tightening    (independent, foundational hygiene)
L1  vscode-victor streamChat v1 port   (leaf; depends only on shipped P1/P3 contract)
W3b sandhi typed-latency surface       (design §4.2; after F1 lands the consumption seam)
W3c sandhi minor-version negotiation   (design §4.3)
W3d G7 codec purity                    (design §4.4; last — largest cross-repo surface)
L2  Chainlit cross-visit resume        (FEP required; not scheduled here)
```

Impact/effort ranking for the parallelizable set (build order within tiers is
by this ratio, but **F-tier completes before W-tier starts** so cross-repo
work lands on settled Victor seams):

| Item | Impact | Effort | Ratio | Tier |
|---|---|---|---|---|
| F1 G8-blocker reduction | High (unblocks the whole native-body track; removes a pricing-boundary violation) | Low (1 function + transport diagnostics + tests) | ★★★ | Foundation |
| F2 P6 residue + ratchets | Medium (kills dead surfaces; locks in decomposition gains) | Low | ★★★ | Foundation |
| L1 vscode streamChat port | Medium (last mis-speaking wire consumer) | Low-Med (TS) | ★★ | Leaf |
| W3a G8 gating (sandhi) | High (bandwidth + decoupling for every client) | Low after F1 | ★★★ | Wave 3 |
| W3b typed latency | Medium (wire-truth latency, closes G1's contract half) | Med | ★★ | Wave 3 |
| W3c version negotiation | Medium (safe additive evolution) | Med | ★★ | Wave 3 |
| W3d G7 codec purity | Medium (codec honesty; Gemini native codec) | High (cross-repo contract design) | ★ | Wave 3 |

---

## 3. Foundation designs

### 3.1 F1 — make the native body strictly optional (Victor)

Goal: `extensions[slug]` becomes debug-only metadata; Victor behaves
identically when it is absent.

1. `turn_execution_runtime.py:1986-2006`: read `cached_tokens` and
   `reasoning_tokens` from `response.usage` (already populated from neutral
   usage) instead of `raw_response["usage"]`.
2. `prompt_cache_miss_tokens`: extract once at the transport boundary
   (`sandhi_transport.py` `_usage_diagnostics`, `:316-338`) into
   `metadata["sandhi_usage"]["cache_miss_tokens"]`; the runtime reads
   diagnostics, never the body. (Longer term this belongs as an additive
   neutral `UsageV2` field — W3 candidate, additive-only within v1 per
   TD-0002.)
3. `cost_in_usd_ticks`: stop reading entirely; cumulative cost comes from the
   canonical Victor pricing path (`get_metrics_capabilities().calculate_cost`,
   G3). Provider-reported cost, when present, may be *logged* as a diagnostic
   discrepancy signal, never consumed as truth.
4. `_completion_from_typed`: set `raw_response` to the native body only when
   present; document it as debug-only.
5. Tests: relax `test_sandhi_transport.py:253` and the consumed-contract pin
   (`test_sandhi_consumed_contract_pin.py:72`) so `extensions` is optional;
   add an explicit extensions-absent case asserting identical usage and
   cumulative tracking.

Acceptance: a synthetic typed response with `extensions=None` produces the
same `usage`, cumulative token/cost tracking, and metadata (minus debug body)
as one with the native body attached.

### 3.2 F2 — P6 residue + ratchet tightening (Victor)

1. Delete `SessionFacadeProtocol` (zero readers).
2. Purge stale coordinator/docstring references listed in §1.1 (docstrings
   only — no behavior).
3. Tighten `selector.py` cap 2882 → 2765 (current actual); take any other
   free slack found at implementation time (`tool_service.py` 3085 → 3079,
   `turn_execution_runtime.py` 2390 → post-F1 actual, `orchestrator.py`
   4704 → 4702).
4. Update `docs/ux-adoption-action-plan.md` §P6 with the as-delivered state
   (coordinators deleted; remaining work = ratchet game), mirroring the P3/P5
   premise-correction pattern.

Acceptance: hotspot guard passes with lowered caps; no dead
facade/coordinator symbols for Chat/Tool/Session remain outside the
state-passed family; service-layer validation floors untouched.

---

## 4. Wave 3 designs (Sandhi + Victor, cross-repo)

Rule of engagement (per TD-0002 and the co-design boundary): every change is
additive within v1; every new event/field gets a consumer-decision row; no
release cuts until the full feature set is promoted develop→main in both
repos (user gate).

### 4.1 W3a — native-body gating (G8)

After F1, no Victor behavior depends on `extensions[slug]`. Sandhi adds
`include_native_response: bool` (request option or transport config; default
**true** initially for compatibility). Victor flips its default to false
behind a setting once F1 ships; after one minor release of soak, sandhi flips
the default. The consumed-contract pin keeps `extensions` in the schema
(additive-optional) — consumers must tolerate absence, which F1's
extensions-absent test enforces.

### 4.2 W3b — typed-response latency surface ✅ SHIPPED (as-built correction)

As built (sandhi#97 + victor#676), the fields live on **`UsageV2`**
(`duration_ms` / `time_to_first_token_ms`, additive, skip-serialized) — not
the originally sketched `ChatResponseV1.metadata.latency` — because that one
struct covers both the response and the stream's terminal `usage` event, and
`ChatResponseV1` has no metadata field. Stamped at `ProviderHandle::
complete/stream` (the family-neutral seam; `MeteredProvider` is body-level
and unwired). Victor surfaces them into `metadata["sandhi_usage"]` and
`StreamMetrics` prefers wire truth over client wall-clock.

### 4.3 W3c — minor-version negotiation ✅ SHIPPED (as executed)

Sandhi (sandhi#99): `CHAT_CONTRACT_MINOR = 3` co-located with the major
(bump history: 1=#68, 2=#90, 3=#97), **machine-enforced** by a digest
ratchet test over the rendered contract schemas — any schema change without
a minor bump fails CI. Exported as `chat_contract_minor()` in both bindings.
Victor: the one-time wire handshake feature-detects the export (absent →
minor 0), caches `installed_chat_contract_minor()`, logs when the runtime is
ahead of `KNOWN_CONTRACT_MINOR`, and a conditional floor-pin test asserts
the installed binding meets Victor's floor once the export exists. No wire
change; consumers stay tolerant-absent by construction.

### 4.4 W3d — codec purity (G7) ✅ SHIPPED

Design addendum: `w3d-codec-purity-addendum.md`. As executed (sandhi PR-B +
victor PR-C): the evidence bar admitted exactly two params —
`reasoning_effort` and `thinking` — promoted to typed v1 fields
(`CHAT_CONTRACT_MINOR` 3→4, ratchet-forced), each with a per-encoder
consumer-decision row. Victor dual-writes into extensions until the runtime
speaks minor ≥ 4 (the W3c handshake is the migration tool). Two defects the
evidence surfaced were also fixed: victor stops re-labeling its openai-shaped
bucket onto native-encoder families (gemini/cohere/ollama — was shape
corruption), and internal orchestration kwargs (`execution_mode`, topology
hints) are stripped at the adapter boundary so they never reach a request
body. Google already had a native gemini request codec on the sandhi side —
the original "route Google through it" premise was stale.

---

## 5. Leaf work

### 5.1 L1 — vscode-victor `streamChat` v1 port

`vscode-victor/src/victorClient.ts:658-767` still speaks the pre-P1 protocol
(`{messages}` request, `data.type` discrimination, `[DONE]` terminator) at
`POST /chat/stream`, which now serves the v1 contract (`{message,
session_id}`, `event`-keyed, `stream_end`-terminated). Port the request
shape + parser; map `thinking`/`tool_result`(+`call_id`, `elapsed_ms`,
`truncated`)/`stream_end`; reuse the webview's existing ToolCall/Thinking
components. TypeScript; independent of everything above.

### 5.2 L2 — Chainlit cross-visit resume

Deferred pending FEP (Chainlit data layer + history-by-id API), per
`victor/ui/chat_app/app.py` docstring. Not scheduled in this round.

---

## 6. Execution order (PR train)

1. **PR-A (this doc)** — the design gate.
2. **PR-B = F1** (victor) — native-body optional; extensions-absent tests.
3. **PR-C = F2** (victor) — residue + ratchet tightening; plan §P6 update.
4. **PR-D = L1** (vscode-victor) — streamChat v1 port; can run parallel to B/C
   (different repo/surface, no shared files).
5. **PR-E = W3a** (sandhi) — `include_native_response` gate; victor default
   flip follows in a small PR-F.
6. **PR-G = W3b**, **PR-H = W3c** (sandhi + victor consumption).
7. **W3d** — separate design addendum first.

Each PR merges on the standard green-gate flow; develop→main promotion and
any release cut remain user-gated across both repos.
