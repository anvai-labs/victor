# Handoff: consume InferFlux's shipped reasoning-separation + producer contract

**Date:** 2026-09-15 · **Status:** Not started on Victor's side — this is a from-scratch brief,
not a status update on in-flight work.

## Context

A three-way audit (2026-09-14, across InferFlux/Sandhi/Victor) scoped a co-design effort to
close gaps in the origin producer contract InferFlux exposes through Sandhi. InferFlux's side
is now **fully shipped and promoted to its `main`** (PRs #170, #171, #172, #174 — see
`docs/architecture/sandhi-typed-integration-gap-analysis.md` for the prior state this
supersedes, and Sandhi's `docs/td/TD-0027-three-way-origin-codesign.md` for the full
cross-repo picture and what Sandhi still owes). This document is the from-scratch starting
brief for the two Victor-side items the original plan scoped (Phase 0.3 and Phase 4) — nothing
here has landed in this repo yet; searched this repo's PR/commit history for any trace of the
three-way effort and found none past what Sandhi's own repo shows.

**What changed that matters to Victor:** InferFlux now separates model reasoning from
user-facing content on **both** the non-streaming and streaming paths, for **two** underlying
model formats — `<think>`-tag models (Qwen3, LFM2.5) and gpt-oss's harmony channel format —
with an identical wire shape for both, so nothing here needs format-specific handling:

- Non-streaming: `message.reasoning_content` (string) + `message.content` (thinking stripped)
- Streaming: `delta.reasoning_content` frames precede `delta.content` frames; never mixed in
  one frame
- Usage: `usage.completion_tokens_details.reasoning_tokens` (present whenever
  `reasoning_content` is non-empty, on both non-streaming bodies and the terminal streaming
  usage frame)

This flows through Sandhi's transparent plane unchanged (Sandhi doesn't rewrite content), so
whatever Victor's provider transport already receives from Sandhi should already carry these
fields for any InferFlux-backed model that reasons — the question is only whether Victor's own
event-mapping and pins are asserting on them yet.

## What Victor already has (starting point, not landed work)

`tests/unit/providers/test_sandhi_event_conformance.py` already pins Sandhi's
`reasoning_delta` event → `chunk.metadata["reasoning_content"]` mapping (line ~105), using a
hand-built synthetic event fixture — not a fixture sourced from a real InferFlux response, and
it does not currently assert on `completion_tokens_details.reasoning_tokens` at all (no
reference to `reasoning_tokens` anywhere in that file as of this writing). That's the file to
extend, not a new one to create.

## Lane V1 — extend the conformance pin (Phase 0.3 → Phase 4 in the original plan)

1. In `test_sandhi_event_conformance.py`, add a fixture case carrying
   `completion_tokens_details.reasoning_tokens` alongside the existing `reasoning_delta`
   event, asserting it surfaces wherever Victor's usage/cost tracking reads token counts
   today (find the call site that reads `usage.completion_tokens` and confirm reasoning
   tokens land somewhere sensible — either folded into the total or tracked separately;
   this repo's own cost-accounting design should decide which, not this handoff).
2. If any existing pin currently *tolerates the absence* of `reasoning_content`/
   `reasoning_tokens` (grep for something like a conditional skip or an "optional field" note
   around the reasoning fixtures — none was obviously present in the file as read, but check
   `tests/unit/providers/test_sandhi_transport.py` and `test_sandhi_transport_anthropic.py`
   too, since they share fixture patterns), flip it to assert presence — this was explicitly
   deferred in the original plan pending Phase 2 landing on InferFlux, which it now has.
3. Add (or confirm existing coverage for) an end-to-end fixture shaped like a real
   InferFlux streaming response with interleaved `delta.reasoning_content` then
   `delta.content` frames, to catch an ordering regression specifically — the synthetic
   fixture today tests the mapping, not the interleaving contract.

*Done when:* the conformance suite would fail if InferFlux started emitting
`reasoning_content` mixed into `content`, or dropped `reasoning_tokens` from the usage frame.

## Lane V2 — context-trimming policy decision (optional, only if reasoning content grows large)

The original plan flagged a possible `fep-0034-reasoning-context-trimming` follow-up: does
Victor count reasoning tokens toward context-window budgeting, or deliberately drop
`reasoning_content` from what gets fed back into subsequent turns? This is a genuine product
decision (reasoning transcripts can be long and aren't meant to be replayed as conversation
history per OpenAI's own harmony convention — see InferFlux's `RenderHarmony`, which never
round-trips reasoning content back into rendered history either, deliberately). Only worth a
FEP if Victor's session/context management doesn't already have an answer for "what happens to
a large `reasoning_content` field" — check current behavior first before assuming a FEP is
needed; this repo's FEP numbering isn't something this handoff should presume (check
`docs/development/fep-template.md` and the existing `docs/feps/` directory for the next
available number if one is warranted).

## Lane V3 — record

Once V1 lands, a thin consumer ADR under the ADR-022 umbrella (`docs/architecture/adr/022-
provider-gateway-feature-layer.md` — confirmed this exists) is the original plan's suggested
record, following this repo's own "one repo authoritative, others thin consumer ADRs" rule
(InferFlux/Sandhi own the contract; Victor's ADR just states what it consumes and pins).

## Verification

- This repo's existing test suite conventions (`pytest -m native_parity` and whatever runs
  `tests/unit/providers/` today) — should stay green throughout; V1 only adds/tightens
  assertions, shouldn't touch runtime code unless the mapping itself needs a fix.
- Manual e2e (once, not CI-gated): a real InferFlux reasoning-model call, through Sandhi, into
  a Victor session — confirm `reasoning_content` displays separately from the final answer in
  whatever UI surface renders it (grep found `test_stream_renderer.py`,
  `test_event_dispatcher.py`, `test_console_rendering_e2e.py` as the render-side tests most
  likely to need a look if the display side isn't already correct).
