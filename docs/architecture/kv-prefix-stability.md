# KV Prefix Stability Invariant

Status: Active (enforced by `tests/unit/framework/test_kv_prefix_invariant.py`)

## Why

Provider prefix caching (OpenAI-compatible automatic caching, KV-cache tiers)
only pays off when consecutive provider calls extend each other byte-for-byte.
Production measurement (2026-10-18, zai glm-5.3 via Sandhi): cache hits were
frozen at exactly the tools+system head (6,528 tokens) while input grew
2,685 → 56,762 across 24 calls of one agentic session — ~88% of every
request's prefill paid full price, and the same un-compacted history burned
81% of a 1M-token daily gateway budget in five minutes.

Chat-completions-style HTTP APIs are stateless: the complete conversation
payload is supplied on each call. An efficient request shape is an **append-only payload
within a user turn**. This preserves eligibility for provider prefix reuse;
actual hits depend on backend cache availability, granularity and routing.
OpenAI-compatible request syntax does not guarantee caching or a particular price.

## The invariant

Between consecutive provider calls of the same user turn, the canonical
message serialization must satisfy `payload[N+1].startswith(payload[N])`,
with a stable tool set. This is an application-level diagnostic, not a
byte-for-byte fingerprint of the entire HTTP body: the tracker excludes images
and metadata, and its tool fingerprint is order-insensitive. Live transport
qualification must also inspect provider-specific messages and tool schemas.
Divergence is sanctioned at the following boundaries; the backend determines
how much of the earlier prefix remains reusable:

| Boundary | Cause |
|---|---|
| `user_turn` | new genuine user message (anchor change) |
| `compaction_or_trim` | compaction at a turn boundary, or the rare history-count trim |
| `tool_set` | mid-session tool hydration / selection change |
| `system_refresh` | system-prompt signature change (mode/project context) |
| model switch / resume | history rehydration creates new message identities |

## Mechanisms (the "how")

- **Anchor identity**: the last *genuine* user message (`is_user_turn`
  provenance; hidden runtime messages never move it). All per-turn freezes
  key on `id()` of that message.
- **Frozen ledger render** (`conversation/assembler.py`): the `<SESSION_STATE>`
  block re-renders only on anchor change; intra-loop the ledger keeps
  live-updating internally but sent bytes are frozen. Cross-turn freshness is
  preserved; tool results ride the tail verbatim either way.
- **Frozen turn prefix** (`agent/turn_prefix_runtime.py`): dynamic guidance is
  composed once per user turn; mid-loop failure signals are **appended** as
  hidden user messages (`internal_prompt_kind=failure_hint`) instead of
  re-rendering the prefix. `apply_turn_prefix` targets the genuine anchor so
  hidden messages cannot steal placement.
- **Sticky selection** (`conversation/assembler.py`): the scored older-message
  set and semantic-augmentation renders freeze per turn; intra-loop budget
  pressure cannot drop previously-included messages. Hard-cap escape: if the
  assembled view exceeds `max_context_chars`, oldest-selected messages drop
  first and the event is logged.
- **Newest-stub dedup** (`agent/tool_result_deduplicator.py`): on a duplicate
  tool output, the *newer* arrival is stubbed ("see earlier message") and the
  already-sent older message stays full — byte-stable across re-assemblies,
  identical token savings, metadata preserved.
- **Boundary-gated compaction** (`agent/context_manager.py` monitor +
  `services/turn_execution_runtime.py` inline checks): compaction defers while
  `mid_user_turn` (activity after the anchor); the overflow trigger overrides
  (context overflow beats cache stability).
- **Unified history cap**: `DEFAULT_MAX_CONVERSATION_HISTORY = 2000`
  (`config/conversation_settings.py`) is the single source of truth for the
  Settings field, `MessageHistory` constructor/serialization defaults and both
  factories. A firing trim logs `[history-trim] sanctioned boundary`.

## Observability

`victor/agent/kv_prefix_observability.py` classifies every assembled payload:

```
[kv-cache] call=17 prefix=<sha12> tools=<sha12> extended=true boundary=none bytes=48211
[kv-cache] call=18 prefix=<sha12> tools=<sha12> extended=false boundary=user_turn bytes=51230 divergence=6531
```

- `extended=true` — this call's payload extends the previous call's bytes.
- `divergence=<offset>` — first differing byte vs the previous call; the
  mutating-message locator. Healthy sessions show divergence **only** on
  boundary lines; a repeating small offset on intra-turn lines means a
  mutator regressed.
- `VICTOR_KV_PREFIX_DEBUG=1` additionally logs a per-message hash chain, making
  the exact mutating message identifiable from logs alone.

## Verification

- **Invariant guard** (hard assert): `tests/unit/framework/test_kv_prefix_invariant.py`
  drives a multi-iteration tool loop with ledger updates, pending-action
  resolution, hidden-duplicate arrivals and selection pressure.
- **Live smoke**: run an actual agent with multiple tool iterations, verify its
  deliverable, and compare consecutive model-call prefixes within the same turn.
  Record sanctioned boundaries separately from unexpected divergence. Verify
  raw provider usage against the normalized Sandhi fields. For fresh-only
  `tokens_in`, the input cache ratio is
  `cache_read_tokens / (tokens_in + cache_read_tokens)`; for a provider total
  prompt count that already includes hits, divide cached tokens by that total.
  Keep missing cache fields unknown rather than silently replacing them with zero.
- **Backend outcomes**: record individual-call and aggregate ratios separately.
  A monotonic ratio or a 0.6 target can describe a particular workload, but is
  not a portable pass condition across OpenAI-compatible providers. Explicit
  zero reported hits do not establish absence of executed reuse. Pricing and
  weighted budget savings require their own evidence.
- **Reported ZAI result** (six-call handback, 2026-10-18): 22,290 fresh and
  84,160 cached input tokens, or **79.1% aggregate** input cache hits; the final
  call was **99.4%** cached. This alone does not establish a 9× daily-budget
  improvement or the result of replaying an earlier 813k-token workload.
