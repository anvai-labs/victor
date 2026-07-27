# W3d Addendum — Codec Purity (G7), Evidence-Grounded Design

**Status:** Design gate for W3d implementation. Extends
`foundations-strategy-2026-07.md` §4.4. **Evidence:** parallel sweeps of
victor develop (`b3c52736f`) and sandhi develop (2026-07-26); every claim
carries file:line in the sections below.

---

## 1. What the evidence changed

The G7 ledger entry assumed a wide promotion surface ("recurring extensions
params") and a missing Gemini codec. Both premises are stale:

1. **The bucket is narrow.** Victor's payload builders emit exactly two
   non-reserved params that meet the promotion bar (≥2 families or
   load-bearing): `reasoning_effort` (OpenAI o-series/GPT-5 +
   Moonshot kimi-k3; load-bearing shape transform for the Responses
   protocol) and `thinking` (ZAI builder + provider-agnostic injection at
   three runtime sites). The folklore params — `top_p`, penalties,
   `do_sample`, `enable_thinking`, `prefix`, `top_k`, `logprobs` — are
   emitted by **zero** victor builders. No speculative fields.
2. **Sandhi's Gemini request codec already exists** and is a real native
   translator from typed fields (`gemini_typed.rs:54-165`: contents /
   systemInstruction / functionDeclarations / generationConfig). "Route
   Google through the gemini-native codec" is already true on the sandhi
   side. What's missing is narrower: `seed`/`response_format` and
   thinking-budget cannot reach it through typed fields, and the proxy's
   gemini *ingress* stash is asymmetric with the encoder's
   `extensions["gemini"]` expectation (pre-existing; sandhi-side follow-up).
3. **Two real defects found** (worse than the assumed "impurity"):
   - **Shape corruption, not just dropping**: victor's neutral mixin
     re-labels its openai-shaped bucket to the destination family's key
     (`sandhi_transport.py:816-823`), and every sandhi encoder clones its
     own family key **verbatim as the base body** (e.g.
     `gemini_typed.rs:54-165`). Openai-shaped keys can therefore land
     top-level inside native Gemini/Ollama bodies.
   - **Internal-key leak**: orchestration keys (`execution_mode`, topology
     hints) flow through unfiltered `**kwargs`
     (`orchestrator_protocol_adapter.py:160-168`,
     `turn_execution_runtime.py:1771-1785`) into `extensions["openai"]`
     and cross the wire.

## 2. Design decisions

### D1 — Promote exactly two typed fields (sandhi, additive, minor → 4)

```rust
// ChatRequestV1 (additive; skip-serialized when absent)
pub reasoning_effort: Option<String>,   // "low" | "medium" | "high" (openai vocabulary)
pub thinking: Option<ThinkingV1>,       // { enabled: bool, budget_tokens: Option<u64> }
```

Encoder mapping (consumer-decision rows — every encoder decides explicitly):

| Encoder | `reasoning_effort` | `thinking` |
|---|---|---|
| openai (chat) | `reasoning_effort` top-level | `thinking` object (ZAI/GLM shape rides openai-compat) |
| openai_responses | `reasoning: {effort}` **+ existing temperature-strip interaction** (`openai_responses_typed.rs:152`) | ignore (Responses uses `reasoning`) |
| anthropic | ignore (no effort vocabulary) | `thinking: {type: "enabled", budget_tokens}` |
| gemini | ignore | `generationConfig.thinkingConfig.thinkingBudget` |
| cohere / ollama | ignore | ignore |

Precedence stays the structural invariant sandhi already has: encoders clone
`extensions[<family>]` as base, then `insert` typed fields — **typed wins**
over any duplicate. Proxy ingress lifts the two params in each dialect
decoder so incoming native bodies populate the typed fields.
`CHAT_CONTRACT_MINOR` 3 → 4; the W3c digest ratchet forces the bump.

### D2 — Victor: dual-write until the runtime speaks minor ≥ 4

Victor emits the typed fields **and** keeps the extensions copies while
`installed_chat_contract_minor() < 4` (the pinned 0.1.4 runtime has no typed
fields; dropping the extensions copy early would lose the params). Once the
pin advances, the builder drops the extensions copies — gated by the W3c
handshake, no lock-step deployment. This is the exact migration pattern W3c
was built to enable.

### D3 — Victor: stop shape-corrupting non-openai families

The neutral mixin attaches its openai-shaped native bucket **only for
openai-compat families**. For gemini (and any non-openai family on the
neutral path), the bucket is dropped: promoted typed fields now carry
everything those encoders can honor, and openai-shaped keys stop landing in
native bodies. (Anthropic already builds a family-native bucket —
unaffected.)

### D4 — Victor: strip internal keys at the adapter boundary

`orchestrator_protocol_adapter.execute_turn` gains an explicit denylist strip
(`execution_mode`, `provider_hint`, `escalation_target`, `topology_action`,
`topology_kind`, `topology_metadata`) before forwarding kwargs to
`provider.chat()`, plus a defensive strip at the codec's bucket build. A
regression test asserts none of these keys can reach
`extensions["openai"]`. This is a bug fix independent of promotion.

### D5 — Deferred (recorded, not implemented here)

- Sandhi gemini ingress↔encoder stash asymmetry (proxy concern).
- Any additional param promotion — requires the same ≥2-family evidence bar,
  not folklore.

## 3. Sequencing

1. **PR-A (this addendum)** — design gate.
2. **PR-B (sandhi)**: `ThinkingV1` + two fields + encoder rows + ingress
   lifts + minor→4 + digest update + facade templates; tests per encoder row
   (honored vs explicitly-ignored) and typed-wins precedence.
3. **PR-C (victor)**: dual-write emission (D2), family-gated bucket (D3),
   internal-key strip + regression test (D4); conditional floor pin extends
   to minor 4 the same way W3c's does.
4. Extensions-copy removal rides the future pin bump (release chain,
   user-gated).
