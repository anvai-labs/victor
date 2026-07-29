---
fep: "0027"
title: "Settle partial usage when a model stream aborts"
type: Standards Track
status: Draft
created: 2026-07-28
modified: 2026-07-28
authors:
  - name: Vijaykumar Singh
    email: singhvjd@gmail.com
    github: vjsingh1984
reviewers: []
discussion: https://github.com/anvai-labs/victor/discussions/0026
---

# FEP-0026: Settle partial usage when a model stream aborts

## Table of Contents

1. [Summary](#summary)
2. [Motivation](#motivation)
3. [Proposed Change](#proposed-change)
4. [Benefits](#benefits)
5. [Drawbacks and Alternatives](#drawbacks-and-alternatives)
6. [Unresolved Questions](#unresolved-questions)
7. [Implementation Plan](#implementation-plan)
8. [Migration Path](#migration-path)
9. [Compatibility](#compatibility)
10. [References](#references)
11. [Review Process](#review-process)
12. [Acceptance Criteria](#acceptance-criteria)

---

## Summary

FEP-0020 grounded streaming metering on one assumption (line 170): *"Streaming:
finalize counts from the terminal usage chunk (never estimate)."* That terminal
chunk is only produced when a stream completes cleanly. When a stream **aborts** —
client disconnect, task cancellation, or an early stop (stop sequence / max output) —
the terminal chunk is never yielded, the accumulated usage is discarded, and the
turn **meters as zero**. Until now that was a quiet edge case; Sandhi's
TD-0013 (landed 2026-07-28) changed the stakes: Anthropic and Gemini now emit **real
partial usage mid-stream** — input + the full cache split known before any content
byte — so genuinely recoverable usage is now being thrown away on every aborted
stream.

This FEP specifies how Victor settles the **last provider-reported cumulative
usage** on an aborted stream instead of zero, **without** weakening the
terminal-chunk contract for clean streams, **without** estimating tokens from
bytes, and **without** making the streaming runtime Sandhi-specific. The mechanism
carries the partial-vs-final usage distinction that Sandhi already emits end-to-end
into the runtime's per-turn accumulator, so the abort-finalize `finally` already in
place settles real numbers.

## Motivation

### Problem Statement

The loss has three reinforcing layers, all verified by reading the current code:

1. **Provider layer discards the held usage on abort.** `SandhiTypedProviderMixin.
   _sandhi_stream` (`victor/providers/sandhi_transport.py`) updates a local `usage`
   on each `usage` event but surfaces it **only** in the post-loop terminal
   `StreamChunk(usage=usage, is_final=True)` (~line 793). Its
   `except (CancelledError, KeyboardInterrupt, SystemExit): raise` (~line 801)
   re-raises and the held `usage` is dropped.

2. **The runtime accumulates by summing, but usage rides only the terminal chunk.**
   The streaming runtime folds each chunk's `usage` into `ctx.cumulative_usage` with
   `+=` (`victor/agent/services/chat_stream_helpers.py:1578`) and finalizes that
   dict in a `finally` (`victor/agent/services/chat_stream_runtime.py:415`). Because
   usage appears on no chunk before the terminal one, `cumulative_usage` is empty on
   abort and the finalize records zero. The `finally` is correct; the data feeding it
   is what's missing.

3. **TD-0013 made the lost value real and large.** Sandhi now emits
   `Usage{completeness: Partial}` mid-stream for Anthropic/Gemini. On the shipped
   Anthropic fixture, `message_start` announces 1024 input + 2048 cache-creation +
   4096 cache-read — **7168 billable tokens, known before a single content byte** —
   and a client that disconnects immediately after currently settles ~0. On
   cache-heavy workloads (the workload Sandhi is built for), `cache_read` dominates
   the bill, so this is an attribution hole, not a rounding error.

The same defect class — a turn that streamed real tokens metering as zero — was
closed in Sandhi's own gateway serve path in a separate, already-merged fix
([victor #716](https://github.com/anvai-labs/victor/pull/716): `gateway_proxy.py`
meters in a `finally`). This FEP closes the **in-process provider** path, which the
gateway fix does not touch.

### Goals

1. On any stream abort, settle the **last provider-reported cumulative usage** rather
   than zero.
2. Preserve the terminal-chunk contract for clean streams: no per-chunk usage flood,
   no double-counting.
3. Never **estimate** usage from byte counts — that is Sandhi's translation-plane
   concern (TD-0013 D4), not Victor's.
4. Keep the streaming runtime **provider-agnostic**: any provider that reports
   cumulative usage benefits; the seam is opt-in and feature-detected, not a
   Sandhi-specific branch in shared code (cf. the #92 "vendor differences are data,
   not branches" discipline).

### Non-Goals

- Changing per-chunk accumulation for the **clean-completion** path (today's
  sum-once behavior is correct and stays).
- Estimating tokens from bytes anywhere in Victor.
- The gateway serve-mode path (addressed by #716).
- Retry/reconnect semantics on abort — orthogonal.

## Proposed Change

### High-Level Design

Carry the **partial-vs-final** distinction that Sandhi's typed stream already emits
through to the runtime's per-turn accumulator, so usage is captured incrementally and
the existing abort-finalize `finally` settles it. Two mechanisms achieve this; review
selects one (see Unresolved Questions).

#### Why the obvious one-liners fail (falsified while drafting)

These were considered and rejected for concrete reasons, recorded so they are not
re-attempted:

1. **"Keep last-seen usage, emit it in a `finally`."** Rejected: Python forbids
   `yield` during `GeneratorExit`/`aclose` (it raises `RuntimeError: generator
   ignored GeneratorExit`), and on `CancelledError` there is no consumer left to
   resume the generator. A terminal chunk cannot be emitted during teardown.
2. **"Attach usage to every yielded chunk."** Rejected: the accumulator sums
   (`cumulative_usage[k] += chunk.usage[k]`) while provider usage is **cumulative**
   (each snapshot reflects total-so-far). Summing cumulative snapshots across chunks
   double-counts, badly, for any provider that reports more than once.
3. **"Read a provider-side `last_usage` at finalize."** Rejected as stated: a
   provider instance serves many concurrent calls, so a per-instance attribute is a
   race. The carrier must be **per-stream-iteration**, not per-instance.

The correct shape therefore either (a) rides usage on per-chunk data the runtime
already retains, or (b) introduces a per-stream carrier the runtime reads at
finalize — and in both cases uses **last-wins**, not sum, for cumulative snapshots.

### Detailed Specification

#### Mechanism A — partial-progress chunks + last-wins merge (recommended)

`_sandhi_stream` emits a usage-bearing `StreamChunk` whenever a `usage` event
arrives, tagged with the completeness Sandhi already provides (`partial` vs `final`),
in addition to the existing terminal `is_final` chunk. The per-chunk accumulator
switches from sum to **last-wins (replace)** for usage snapshots: because provider
usage is cumulative, the latest snapshot *is* the running total.

```python
# _sandhi_stream: today collapses every usage event into one local and emits
# only the terminal chunk. Proposed: forward each usage snapshot with its
# completeness, so the runtime sees usage incrementally.
elif kind == "usage":
    usage = usage_dict_from_neutral(event.get("usage"), None, slug=...)
    completeness = event.get("usage", {}).get("completeness", "final")
    yield StreamChunk(content="", usage=usage, metadata={"usage_completeness": completeness})
```

```python
# chat_stream_helpers accumulator: cumulative snapshots merge by last-wins, not sum.
if chunk.usage:
    completeness = (chunk.metadata or {}).get("usage_completeness", "final")
    if completeness == "partial" or stream_ctx.usage_is_cumulative:
        stream_ctx.cumulative_usage.update(chunk.usage)   # replace: last snapshot wins
    else:
        for k, v in chunk.usage.items():                  # legacy incremental path
            stream_ctx.cumulative_usage[k] += v
```

On a clean stream the final snapshot replaces the last partial → same result as
today. On abort, the last partial is already in `cumulative_usage` → the existing
`finally` settles it. No per-instance state, no teardown-time yield.

**Why `last-wins` is safe broadly:** providers that report usage once (OpenAI Chat,
Cohere, Ollama, OpenAI Responses — TD-0013's "terminal-only" cadence) emit exactly
one snapshot, so `replace`-once == `sum`-once. Only the incremental families
(Anthropic, Gemini) emit more than one, and their snapshots are cumulative by spec.

#### Mechanism B — per-stream carrier read at finalize (smaller blast radius)

`_sandhi_stream` writes its running `usage` to a **per-iteration** carrier (a small
mutable object created per `stream()` call, never on the shared provider instance).
The runtime's abort-finalize, when `ctx.cumulative_usage` holds no tokens for the
turn, folds in the carrier's last snapshot (last-wins). Narrower change (no
accumulator-semantics change) but introduces a new carrier type threaded from
provider to runtime, and a conditional fallback in the generic `finally`.

### API Changes

- **`StreamChunk`** (`victor/providers/base.py`): no field change required for
  Mechanism A — usage already rides `chunk.usage`; completeness rides `chunk.metadata`
  (additive). Mechanism B adds a per-stream carrier type.
- **Streaming runtime accumulator** (`chat_stream_helpers.py` / `chat_stream_runtime.py`):
  last-wins merge for cumulative usage snapshots (A), or a finalize-time fallback
  read (B). Provider-agnostic; gated on the presence of the signal, not on a
  Sandhi type check.

### Configuration Changes

None. Behavior is automatic for any provider that reports cumulative usage.

### Dependencies

None new. Relies on Sandhi ≥ the TD-0013 release emitting `Usage{Partial}` mid-stream
(already shipped on `develop`).

## Benefits

### For the Framework

- A turn that streamed real tokens no longer meters as zero on abort — the core
  "every call is counted and attributed" claim holds on the path users hit most
  (interactive streaming, where aborts are common: user cancels, stop sequences,
  max-output caps).
- Captures the cache split (`cache_creation` / `cache_read`) that dominates
  cache-heavy spend — exactly the attribution Sandhi exists to provide.

### For Vertical Developers

- No change required; any provider that reports cumulative usage benefits
  automatically.

### For the Ecosystem

- Aligns Victor's in-process path with the gateway-serve fix (#716) and with
  Sandhi's TD-0013, so metering trust is consistent across both deployment shapes.

## Drawbacks and Alternatives

### Drawbacks

- Mechanism A changes the generic accumulator's merge rule (sum → last-wins for
  cumulative snapshots). Mitigation: gated on the `usage_completeness` signal;
  providers without it keep today's sum behavior unchanged, so the blast radius is
  exactly the providers that already report cumulative usage.
- Mechanism B threads a new per-stream carrier. Mitigation: small, opaque type;
  feature-detected at finalize.

### Alternatives Considered

1. **Estimate output from bytes on abort.** Rejected: violates the non-goal; byte
   estimation is Sandhi's translation-plane concern (TD-0013 D4), and no byte count
   can recover the input/cache split that `message_start` already gave us for free.
2. **Clamp settle to the reserved budget ceiling.** Rejected: that would *recreate*
   the loss one layer down (a measured 7168 clamped to a ~47-token reservation) —
   the exact defect Sandhi's TD-0013 D6 rejected under test. Settle the truth,
   surface the overshoot separately.
3. **Document-only, no code.** Rejected as the final state: the gap is a real,
   measurable attribution loss now that partial usage is emitted. This FEP is
   document-first per governance, but defers only the *code*, not the intent.

## Unresolved Questions

- **Mechanism A vs B.** Proposed: A — it is the principled end-to-end answer
  (honors the partial/final distinction Sandhi already emits) and avoids per-instance
  state. B is the lower-blast-radius fallback if the accumulator-semantics change
  meets resistance in review.
- **Default for providers of unknown cadence.** Proposed: assume cumulative
  (last-wins) when `usage_completeness` is absent and more than one usage snapshot is
  seen in a turn; else sum. The terminal-only families are unaffected either way.
- **`reasoning_tokens` / latency diagnostics.** Out of scope here; the same carrier
  can later carry them if needed.

## Implementation Plan

Code is **deferred**; this FEP is document-first per the 14-day review requirement
for runtime changes.

### Phase 1: Agreement on mechanism (this FEP)

- [ ] Review selects Mechanism A or B.
- [ ] Confirm the `usage_completeness` signal contract with Sandhi (already emitted
      by TD-0013; verify it reaches `_sandhi_stream`'s `event["usage"]`).

**Deliverable**: an approved seam.

### Phase 2: Provider surfaces partial usage (Victor-local)

- [ ] `_sandhi_stream` forwards each usage snapshot with completeness (A), or writes
      the per-stream carrier (B).
- [ ] Unit test: an Anthropic fixture that emits a partial `message_start` usage then
      aborts → the snapshot is retained (fails before, passes after).

**Deliverable**: recoverable usage reachable on abort.

### Phase 3: Runtime settles it

- [ ] Accumulator last-wins for cumulative snapshots (A), or finalize-time fallback
      read (B).
- [ ] Integration test: a streamed turn aborted after `message_start` settles the
      real input+cache split, not zero.

**Deliverable**: end-to-end settlement on abort.

### Testing Strategy

- Unit: provider surfaces partial usage on abort (snapshot retained).
- Integration: runtime settles non-zero usage on an aborted streamed turn.
- Regression: clean-stream terminal usage unchanged (the existing streaming tests
  must pass unmodified).
- No double-count: a stream emitting multiple partials settles the final total, not a
  sum of partials.

### Rollout Plan

- No feature flag (correctness fix); lands behind the existing `sandhi` opt-in.
- Documentation: note the refinement to FEP-0020's "terminal chunk" assumption.

## Migration Path

No migration required — additive behavior. Providers that do not report cumulative
usage are unaffected.

## Compatibility

- **Breaking change:** No. Clean-stream behavior is unchanged; the change only adds
  settlement on the abort path.
- **Minimum Python:** 3.11 (unchanged).
- **Vertical compatibility:** none required; automatic for cumulative-usage providers.

## References

- [FEP-0020](./fep-0020-ai-gateway-usage-attribution.md) — the parent; its line-170
  "terminal usage chunk" assumption is what this FEP refines for the abort case.
- [victor #716](https://github.com/anvai-labs/victor/pull/716) — the same defect
  class closed in the gateway serve path (`gateway_proxy.py` meters in a `finally`).
- Sandhi **TD-0013** — streaming usage fidelity; emits `Usage{Partial}` mid-stream
  for Anthropic/Gemini (landed on Sandhi `develop` 2026-07-28).
- `victor/providers/sandhi_transport.py` (`_sandhi_stream`),
  `victor/agent/services/chat_stream_helpers.py` (accumulator),
  `victor/agent/services/chat_stream_runtime.py` (abort-finalize `finally`).

## Review Process

- **Submitted by:** Vijaykumar Singh
- **Initial review period:** 14 days minimum.
- **PR:** TBD.

---

## Copyright

This FEP is licensed under the Apache License 2.0, same as the Victor project.
