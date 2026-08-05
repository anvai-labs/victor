---
fep: "0023"
title: "Activate the context-management subsystem (ledger-keystone, phased + gated)"
type: Standards Track
status: Draft
created: 2026-07-20
modified: 2026-07-20
authors:
  - name: Vijaykumar Singh
    email: vijay@anvaiops.com
    github: vjsingh1984
reviewers: []
discussion: https://github.com/anvai-labs/victor/discussions/0023
---

# FEP-0023: Activate the context-management subsystem

## Summary

Victor ships a **five-component "conversation context management overhaul"** — `SessionLedger`,
`TurnBoundaryContextAssembler`, `LedgerAwareCompactionSummarizer`, `ToolResultDeduplicator`, and
`ReferentialIntentResolver` — that is fully built and unit-tested, has an end-to-end integration
test specifying the intended data flow, yet is **inert in production**. The hub of the design, the
`SessionLedger`, is constructed and even *rendered* into the assembled context (as a
`<SESSION_STATE>` block) but is **never populated during a live turn**: no runtime code calls
`update_from_tool_result` or `update_from_assistant_response` on the orchestrator's ledger, so it
renders empty and every ledger-derived feature is a no-op. Two of the five components
(`ToolResultDeduplicator`, `ReferentialIntentResolver`) additionally have factory builders that
nobody calls. This FEP reframes the work from "wire two unwired components" to **activating one
coherent subsystem, keystoned on ledger population**, and lands it as three separable, dependency-
ordered, flag-gated, individually-measured phases so that each earns its place (or is dropped) on
evidence rather than assertion — the same discipline as FEP-0021/0022.

## Motivation

An adversarial review (F-018b) established three load-bearing facts:

1. **The ledger is never populated.** `orchestrator._session_ledger` is built
   (`component_assembler.py:181`) and rendered by the live assembler (`assembler.py:255`), but the
   only callers of its `record_*` / `update_from_*` methods are inside `session_ledger.py` itself
   and session restore. So `render()` → `""` and `get_recent_actionable_items()` → `[]` in every
   live turn. The whole subsystem's value is gated on this one missing connection.
2. **`ReferentialIntentResolver` is dead-on-arrival without it.** Its `enrich()` reads
   `get_recent_actionable_items(limit=5)`; against an empty ledger it returns the message
   unchanged. Wiring it before populating the ledger ships a no-op.
3. **`ReferentialIntentResolver` overlaps the assembler.** Both surface the *same* ledger items —
   the assembler as a passive `<SESSION_STATE>` block, the resolver as an explicit
   `[Context: the user is referring to these]` block appended to the user message. Once the ledger
   is populated, a capable model can resolve "do it" from `<SESSION_STATE>` alone, so the resolver
   must *measurably beat* passive state or it is redundant double-injection.

Two are genuine gaps (verified against every existing mechanism): `ToolResultDeduplicator` is the
only *history-level* repeated-read compactor (distinct from batch-dedup, cross-turn execution
caching, registry dedup, and the age/size compactor); `ReferentialIntentResolver` is the only
expander of *anaphoric user messages* (all existing "intent" code classifies *model* output). But
context management is exactly the **dominant cost term** (provider round-trips × context size) the
Evaluation-Centric Runtime targets, so activating it blindly — mutating history, double-injecting,
diverging streaming vs non-streaming — would reintroduce the very anti-patterns FEP-0021 removed.

## Proposed Change

Activate the subsystem as **one hub + staged consumers**, in three dependency-ordered phases, each
behind a `FeatureFlag` defaulting **OFF** (via `is_opt_in_by_default()`), gated at the **call site**
(the component-level `enabled` config defaults `True`, so it self-runs otherwise), and each with a
measurable graduation gate per `flag-graduation-policy.md`.

### Phase 1 — Ledger population (the keystone)

Populate `orchestrator._session_ledger` at a **single universal per-turn seam** covering
streaming, non-streaming, and planning turns (mirroring the FEP-0016m/F-016f teardown pattern), by:
- calling `update_from_tool_result(tool_name, args, result, turn_index)` as tool results are
  recorded, and
- calling `update_from_assistant_response(content, turn_index)` at the assistant-turn boundary.

This *also* activates the already-wired `<SESSION_STATE>` assembler injection (no extra work).
Flag `USE_SESSION_LEDGER` (default OFF). Highest leverage; unblocks Phases 2–3.

### Phase 2 — `ToolResultDeduplicator` as an assembler *view* stage

Wire dedup as a **stage inside `assemble()`**, operating on the windowed *copy* the assembler
already builds (`orchestrator.py:2063-2068` → `assembler.py`), **never mutating `self.messages`**
(the source-of-truth history). This co-locates it with the assembler's existing content-hash
`_deduplicate_semantic` helper so the two dedup notions unify rather than compete. Ledger-
independent, so it can ship and be measured on its own. Flag `USE_TOOL_RESULT_DEDUP` (default OFF).

### Phase 3 — `ReferentialIntentResolver` behind one shared input seam, gated on an A/B

Only after Phase 1. Introduce **one shared pre-add user-input transform** that both the
non-streaming (`turn_execution_runtime.py:512`) and streaming (`chat_stream_helpers.py:293`) paths
call, so enrichment can't drift between them. Ship only if an A/B shows it **beats populated
`<SESSION_STATE>` alone** on follow-up resolution; otherwise **delete** the resolver rather than
ship redundant double-injection. Flag `USE_REFERENTIAL_INTENT` (default OFF).

## Benefits

- **Turns ~700 LOC of tested-but-inert code into measured capability** — or deletes what can't earn
  its place, on evidence. Either outcome is a win over the current limbo.
- **Attacks the dominant cost term.** Ledger `<SESSION_STATE>` + read-dedup directly reduce
  context size (provider round-trips × tokens), the EVR cost-co-design target — with a token-savings
  metric, not a hope.
- **One coherent subsystem, not bolt-ons.** The ledger is the single source of truth; consumers are
  staged around it, so dedup, compaction, and `<SESSION_STATE>` cannot disagree about "current
  state."
- **Fixes a latent data-integrity bug before it ships** — the as-built `deduplicate_in_place` would
  mutate the persisted transcript; making dedup a view-stage keeps the source of truth intact.
- **Streaming/non-streaming parity by construction** — the shared input seam prevents the exact
  run/run_streaming drift FEP-0021 just deduped.
- **Every phase flag-gated + measured**, so the subsystem graduates (or is trimmed) the same
  evidence-first way as the rest of the framework — no assert-and-cache.

## Drawbacks and Alternatives

- **Do nothing / delete all five.** Rejected: two are genuine, tested gaps against the dominant
  cost term; deleting untested-in-prod-but-tested-in-unit capability wastes real work and the
  `<SESSION_STATE>` design is already half-wired.
- **Wire all five at once (the e2e-test flow verbatim).** Rejected: ships the resolver as a no-op
  (empty ledger), risks history mutation, and double-injects — the review's exact findings.
- **Keep dedup mutating `self.messages`** (as built). Rejected: irreversible corruption of the
  persisted transcript; a view-stage is strictly safer and regenerated each turn.
- **Ship the resolver on faith.** Rejected: it overlaps `<SESSION_STATE>`; additivity is unproven
  and double-injection is the default failure mode. Measure or delete.

## Unresolved Questions

- **Residual duplicate rate** for Phase 2: cross-turn/session read-dedup already blocks most
  duplicate reads from entering history — how many actually reach the deduplicator? Measure before
  graduating (it sets the ceiling on dedup's value).
- **Resolver vs `<SESSION_STATE>`:** does explicit "the user is referring to these" framing beat
  passive state enough to justify a second injection channel? Needs an anaphora A/B corpus.
- **Ledger population cost/noise:** `update_from_assistant_response` uses regex extraction of
  decisions/recommendations — precision/recall unknown; a noisy ledger degrades `<SESSION_STATE>`.
- **Config-driven magics:** resolver `limit=5` + 200-char gate and dedup 2000/500 fingerprint
  window are hardcoded; they're the A/B knobs and should move to config before graduation.

## Implementation Plan

- **Phase 1 (this FEP's first PR):** `USE_SESSION_LEDGER` flag (opt-in-default OFF); populate the
  ledger at the universal per-turn teardown seam; unit test that a read + assistant turn produces
  ledger entries and a non-empty `<SESSION_STATE>`; parity test (streaming + non-streaming both
  populate). No behavior change when OFF.
- **Phase 2:** move dedup into `assemble()` as a view-stage over the copy; guard test asserting
  `self.messages` is unchanged post-assembly; token-savings measurement harness; residual-duplicate
  census; `USE_TOOL_RESULT_DEDUP` flag + graduation gate.
- **Phase 3:** shared input-enrichment seam (both paths); anaphora A/B vs SESSION_STATE-alone;
  `USE_REFERENTIAL_INTENT` flag + gate, or deletion if the A/B fails.

Each PR is small, focused, and independently revertible; the flag inventory guard
(`tests/unit/runtime/test_feature_flag_manifest_guard.py`) picks up each new flag, and the
generated `docs/architecture/feature-flags.md` is regenerated in the same PR that adds the flag.

## Migration Path

Additive and OFF-by-default throughout. With all flags OFF, behavior is byte-identical to today
(the ledger stays empty/inert, dedup and resolver never run). Each phase is independently
revertible by flipping its flag. No public API, provider, or SDK changes; the generated flag
inventory (`docs/architecture/feature-flags.md`) picks up the new flags automatically from
`is_opt_in_by_default()`.

## Compatibility

Backward compatible. New flags default OFF; the components already exist; the only structural change
is (Phase 2) relocating dedup into the assembler's view path, which is internal to
`get_assembled_messages`. Session persistence/restore of the ledger (`to_dict`/`from_dict`) is
unaffected.

## References

- Review: F-018b in `docs/architecture/CODEBASE_REVIEW_BACKLOG.md`
- Hub: `victor/agent/session_ledger.py`; assembler `victor/agent/conversation/assembler.py`
  (`<SESSION_STATE>` at `:255`); build site `victor/agent/runtime/component_assembler.py:181,305`
- Components: `victor/agent/referential_intent_resolver.py`,
  `victor/agent/tool_result_deduplicator.py`; e2e spec
  `tests/integration/agent/test_context_management_e2e.py`
- Gating: `docs/architecture/feature-flags.md`, `docs/architecture/flag-graduation-policy.md`
- Cost thesis: `docs/architecture/evaluation-centric-runtime-backlog.md` (dominant cost term)
