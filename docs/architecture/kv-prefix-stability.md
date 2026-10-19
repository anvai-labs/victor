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

Chat-completions-style HTTP APIs are stateless: full prefill per call is the
contract. The efficient version of that contract is an **append-only payload
within a user turn** — the provider serves everything it already saw from
cache and charges full price only for the new tail.

## The invariant

Between consecutive provider calls of the same user turn, the serialized
payload (tools array + system + all messages) must satisfy
`payload[N+1].startswith(payload[N])`. Divergence is sanctioned only at
boundaries, where a single cache miss is expected:

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
- **Live smoke**: run a multi-turn session (e.g. the zai coding profile) and
  check the gateway/ledger cache ratio — `cache_read_tokens / tokens_in`
  should be non-decreasing within each turn and ≥ ~0.6 by turn end
  (baseline before the fix: ~0.11).
