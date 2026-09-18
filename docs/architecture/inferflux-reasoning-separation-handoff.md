# InferFlux reasoning-separation consumer contract

**Date:** 2026-09-17 · **Status:** Focused v0.9.5 consumer correction candidate; review,
CI, main promotion and publication pending. Context replay policy remains a separate product decision.

## Context

A three-way audit (2026-09-14, across InferFlux/Sandhi/Victor) scoped a co-design effort to
close gaps in the origin producer contract InferFlux exposes through Sandhi. InferFlux's side
is now **fully shipped and promoted to its `main`** (PRs #170, #171, #172, #174 — see
`docs/architecture/sandhi-typed-integration-gap-analysis.md` for the prior state this
supersedes, and Sandhi's `docs/td/TD-0027-three-way-origin-codesign.md` for the full
cross-repo picture and what Sandhi still owes). This document records the Victor-side consumer
contract from the original Phase 0.3 and Phase 4 scope. Victor pins Sandhi's reasoning event and
usage fields and verifies that reasoning frames remain separate from visible answer content.

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
Victor consumes the resulting typed events through Sandhi and keeps format-specific parsing out
of the framework and runtime layers.

## Current Victor behavior

The focused v0.9.5 candidate starts at protected main
`4e4e0d6405b4ff2a353aafa9adce1aac8aa4e993`. It carries only consumer usage/accounting
corrections and their tests from `f6776dc59`, `502a46355`, `6140260d2` and `730a47a30`,
plus aligned dependency pins and release records. It does not include the broader v0.10.0
develop cut, tool/session changes, FEP-0034, provider-default changes or new TUI/MCP behavior.

`pyproject.toml` and all three existing deployment snapshots pin `sandhi-gateway==0.7.0`;
the transport recognizes chat contract minor 8. The candidate preserves boolean
`reasoning_included`, folds separate reasoning into billable output per call before aggregation,
and carries the result through stream finalization and session pricing. Raw completion counts
remain separately observable. Explicit provider totals are retained; fallback totals include
separate reasoning. Duration and TTFT keep independent origin/boundary provenance.

Sandhi 0.7.0 is published and independently verified. Fresh validation of this focused source
with its published binding passed all 18 three-repository CPU/stub probes, using the earlier
reviewed InferFlux `3f45fed87` runtime binary. That binary is not a build of InferFlux's later
exact-main `12cf` revision, and these results establish wire behavior, not loaded-model or
publication readiness. Earlier candidate results remain historical. Detailed local evidence and
open gates are recorded in the [release checkpoint](../release-readiness-mvp.md#focused-v095-co-design-candidate).

`tests/unit/providers/test_sandhi_event_conformance.py` pins Sandhi's `reasoning_delta` event to
`chunk.metadata["reasoning_content"]`. Its InferFlux-shaped typed-event fixture verifies that
reasoning frames precede answer frames, never appear in visible chunk content, and preserve
`reasoning_tokens` in Victor's usage dictionary. The consumed-contract pin also requires
Sandhi's `reasoning_delta` variant and `UsageV2.reasoning_tokens` field.

## Lane V1 — conformance pin (implemented)

1. `test_sandhi_event_conformance.py` carries reasoning usage alongside `reasoning_delta` events
   and asserts that Victor surfaces it as `usage["reasoning_tokens"]`.
2. `test_sandhi_consumed_contract_pin.py` fails if the installed Sandhi schema removes the
   reasoning event or usage field that Victor consumes.
3. An InferFlux-shaped stream verifies two reasoning frames followed by two answer frames and a
   terminal usage event. It catches ordering, visible-content leakage and usage regressions at
   the Sandhi-to-Victor boundary.

This is a consumer-boundary guarantee. InferFlux owns its raw producer-frame tests; Victor's
suite fails if Sandhi delivers mixed reasoning/answer events or drops the consumed usage field.

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

ADR-022 (`docs/architecture/adr/022-provider-gateway-feature-layer.md`) remains the owning
provider-boundary decision. A separate ADR is warranted only if Victor changes the context replay
or trimming policy; InferFlux and Sandhi remain authoritative for the producer and wire contracts.

## Verification

- The original #1066 consumer-contract change passed 96 tests without changing runtime behavior.
  That is historical evidence; this v0.9.5 correction changes accounting and provenance handling.
- New regressions cover inclusion preservation, per-call accumulation, real production stream
  consumption/finalization, session pricing, reported/fallback totals and canonical timing labels.
- Still required for broader behavioral acceptance: a real InferFlux reasoning-model call, through Sandhi, into
  a Victor session — confirm `reasoning_content` displays separately from the final answer in
  whatever UI surface renders it (grep found `test_stream_renderer.py`,
  `test_event_dispatcher.py`, `test_console_rendering_e2e.py` as the render-side tests most
  likely to need a look if the display side isn't already correct).
