# InferFlux reasoning-separation consumer contract

**Updated:** 2026-09-19 · **Status:** Victor consumer correction published in v0.9.5.
Context replay policy remains a separate product decision.

## Context

A three-way audit (2026-09-14, across InferFlux/Sandhi/Victor) scoped a co-design effort to
close gaps in the origin producer contract InferFlux exposes through Sandhi. InferFlux's side
has source implementations (PRs #170, #171, #172, #174 — see
`docs/architecture/sandhi-typed-integration-gap-analysis.md` for the prior state this
supersedes, and Sandhi's `docs/td/TD-0027-three-way-origin-codesign.md` for the full
cross-repo picture and delivery gates). Source integration does not establish publication or
validate a later producer revision. This document records the Victor-side consumer
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

The focused v0.9.5 release was promoted to protected main at `e7bbb9427`. It carries the
consumer accounting correction, aligned Sandhi pins, tests and release records; it excludes
the broader 0.10.0 develop feature set. `pyproject.toml` and the three deployment locks pin
`sandhi-gateway==0.7.0`; the transport recognizes chat contract minor 8. The published Sandhi
wheel passed all 18 three-repository CPU/stub probes in an isolated review environment.

The release preserves boolean `reasoning_included` through response models, folds separate
reasoning into billable output per call before accumulation, and carries that numeric result
through stream finalization and session pricing. Raw completion counts remain separately
observable. Missing totals include separate reasoning; explicit provider totals are retained.
Duration and TTFT retain independent origin/boundary labels in stream metrics and the canonical
usage record.

Subsequent develop work corrected two interactions outside the focused release: caller-curated
tool sets take precedence over pruning and demand hydration, with finalized supply traces; member
session context is scoped to stream setup, advances and cleanup, restored before outward yields.
Timeout cleanup explicitly closes the member stream. These changes belong to the 0.10.0
development line and are not part of the published v0.9.5 artifacts.

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

The original plan flagged a possible reasoning-context-trimming follow-up: does
Victor count reasoning tokens toward context-window budgeting, or deliberately drop
`reasoning_content` from what gets fed back into subsequent turns? This is a genuine product
decision (reasoning transcripts can be long and aren't meant to be replayed as conversation
history per OpenAI's own harmony convention — see InferFlux's `RenderHarmony`, which never
round-trips reasoning content back into rendered history either, deliberately). Only worth a
FEP if Victor's session/context management doesn't already have an answer for "what happens to
a large `reasoning_content` field" — check current behavior first before assuming a FEP is
needed; this repo's FEP numbering isn't something this handoff should presume (check
`docs/development/fep-template.md` and the existing `docs/feps/` directory for the next
available number if one is warranted). FEP-0034 is already assigned to tool-supply consolidation;
it is not a reasoning-context policy.

## Lane V3 — record

ADR-022 (`docs/architecture/adr/022-provider-gateway-feature-layer.md`) remains the owning
provider-boundary decision. A separate ADR is warranted only if Victor changes the context replay
or trimming policy; InferFlux and Sandhi remain authoritative for the producer and wire contracts.

## Verification

- The release-review provider/accounting suite passed 1,585 tests (13 optional skips). Independent
  tool/session review passed 125 tests plus cancellation, interleaved-member, cross-task cleanup
  and early-close probes. The tag release workflow is the publication authority.
- Release regressions cover the production terminal-chunk consumer, turn accumulator,
  real metrics collector and session tracker in
  `tests/unit/agent/services/test_codesign_usage_pipeline.py`. They exercise folded, separate,
  mixed-call and legacy reasoning, totals, costs and latency provenance.
- The 2026-09-17 session-context fix passed 74 related subagent tests locally, including real
  ContextVar checks at every yield, early termination on a final chunk, cleanup in another task,
  lazy setup, the wrapper's timeout path and synchronous stream-factory failure cleanup.
  Curated-tool regressions exercise pruning both on and off, missing tools and trace emission;
  these broader develop checks are not v0.9.5 release contents.
- The original consumer-boundary change passed 96 focused contract/transport tests. That dated
  result predates the current accounting and session-context changes.
- Manual e2e (once, not CI-gated): **done 2026-09-16** against the production R9700 deployment
  (InferFlux `fix/victor-codesign-e2e` on WSL2/ROCm serving Qwen3-14B Q4_K_M): a real
  reasoning-model call through Sandhi into a Victor session returned `reasoning_content`
  (1396 chars) split from a clean final `content`, with
  `usage.completion_tokens_details.reasoning_tokens = 1` on the wire; the visible Victor answer
  contained zero `<think>` leakage. The coding-model leg of the same session (Qwen3-Coder-30B
  UD-Q4_K_XL) completed a full agentic write→write→bash task with structured `tool_calls` on
  both streaming and non-streaming paths after the InferFlux-side fixes below landed.

The historical loaded-model observation above is not validation of every published artifact.
The local accounting and context regressions are model-free. The separate pinned CPU/stub
three-repository probe validates protocol integration, not tokenizer accuracy, model quality,
GPU throughput, or production recovery. Re-run the appropriate checks against exact release
revisions before extending those claims.

## Follow-on co-design fixes found during that e2e (2026-09-16)

Source changes reported in that session, recorded here because the manual e2e surfaced them;
this list does not attest their release-artifact publication:

- **InferFlux**: non-streaming chat responses never emitted `message.tool_calls` (the inline
  choice builder dropped the detected call; `BuildChoice` was dead code). Extraction now
  detects **multiple** calls per completion (models chain write→write→run as one JSON object
  per line, which the old single-object parse rejected as "Extra data") and both response
  paths emit the full OpenAI array with `finish_reason: "tool_calls"`.
- **InferFlux**: the scheduler's slot-id clamp read a KV-sequence metric nothing ever wrote,
  so slot ids ran past the llama wrapper's `n_seq_max` and llama.cpp GGML_ASSERT-aborted the
  server on the third request of an agent session. Both backends now publish their sequence
  capacity, and the wrapper fails a request cleanly instead of aborting if an id is ever
  out of range.
- **Victor**: a broken optional embedding backend aborted stream preparation
  (`unified_task_tracker.detect_task_type` → task classifier) as a non-recoverable error.
  Classification now degrades loudly to `GENERAL` so the agent loop survives; the embeddings
  service also chains the real import failure instead of misreporting it as "sentence-transformers
  not installed".
- **Victor**: the inferflux policy tier carried a placeholder lineup (`llama3-8b`, 8192 ctx)
  and no tool-capability tier, so real deployments silently ran tool-less at the wrong context
  budget. The policy now pins the production lineup at 32768 (the serving pool's per-sequence
  context) and `model_capabilities.yaml` enables native tool calls provider-wide.
