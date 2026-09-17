---
fep: "0035"
title: "Shared transcripts and conversation-native team formations"
type: Standards Track
status: Draft
created: 2026-09-17
modified: 2026-09-17
authors:
  - name: Vijaykumar Singh
    email: vijay@anvaiops.com
    github: vjsingh1984
reviewers: []
discussion: https://github.com/anvai-labs/victor/discussions/0035
---

# FEP-0035: Shared transcripts and conversation-native team formations

## Summary

Add an opt-in, bounded team transcript and one conversation runner for GROUP_CHAT,
DEBATE, and HANDOFF. Existing formations keep their private histories and defaults.
The existing formation registry remains the only dispatch authority. Each new
formation receives its enum, registry entry, public preset, documentation,
coordinator-dispatch tests, and explicit durability statement.

## Motivation

A sequence of independent subagents cannot represent a shared conversation or a
peer-directed transfer. Handoff coverage gaps G14/G15 require a typed transcript,
speaker-selection contract, and bounded termination before these APIs are added.
Canonical coordination roles are member, subagent, supervisor, reviewer (one pass),
critic (iterative), judge (one-shot verdict), synthesizer, and router.

## Proposed Change

A `TeamTranscript` contains ordered `TranscriptEntry` records with `sequence`,
`member_id`, `content`, and optional `handoff_to`. Member IDs are the configured
IDs; sequence numbers are append indices, never parsed from text. Transcript
serialization is JSON and is returned in the team's shared context. Entries carry
short messages or artifact references, with a configurable character budget.
Exceeding the budget fails explicitly; no silent truncation or summarization.

Every speaking turn receives the original task, complete bounded transcript,
permitted member IDs, and the response contract as structured JSON. The required
response is `{ "content": string, "done": boolean, "handoff_to": string|null }`.
A few-shot example accompanies the contract. JSON objects with missing/extra keys,
wrong types, or unknown handoff destinations fail the member turn; no keyword,
fenced-text, or prose parsing fallback is permitted.

### Speaker selection

GROUP_CHAT defaults to round-robin. An optional programmatic `candidate_func`
filters configured member IDs for a turn; it must return a nonempty, unique subset.
An optional programmatic `selector_func` selects exactly one eligible ID from the
immutable transcript snapshot and eligible candidates. An optional router member
can instead select by returning `{ "speaker_id": string }`. Router and selector
are mutually exclusive. Invalid selection is an explicit failed run, never a
fallback to the first member. The router is excluded from speaking candidates.

DEBATE runs bounded round-robin contributions from members/critics, then one
configured judge. The judge returns `{ "selected_member_id": string,
"verdict": string }` and must select a participant that actually spoke. The judge
runs exactly once; its verdict is the final output. No regex-based winner parsing.

HANDOFF starts at a configured member (default first). Each member's typed
`handoff_to` chooses the next configured peer. Self-handoff, unknown destinations,
and missing destinations before `done=true` are errors. `done=true` with a
handoff destination is contradictory and rejected. This is peer control transfer
through the structured response contract; tool-call providers can return that
same object at their existing tool/response boundary. It does not introduce a
second coordinator or spawn registry.

### Termination and results

A positive `max_turns` bounds all conversations. `done=true` completes GROUP_CHAT
or HANDOFF; DEBATE completes after its contribution budget plus the judge. Reaching
the turn limit without completion returns an explicit incomplete outcome. An
optional programmatic termination predicate receives the transcript snapshot.
Predicate/selection exceptions fail with a warning diagnostic; they are not
interpreted as permission to continue.

Repeated turns by a member accumulate tool calls and duration into one
`MemberResult` keyed by its configured ID. The transcript retains individual turns.
A failed turn remains a failed aggregate; final-output selection cannot hide it.
Conversation metadata includes termination reason and transcript. Default formation
output assembly changes only when a conversation result explicitly supplies it.

### Durability

The three initial strategies return `supports_durable_pause() = False`; approvals
remain inline. A configured checkpoint/resume request is rejected before any member
runs. Durable transcript restoration, pending selector decisions, and pending peer
transfer are a separate follow-up; no partial replay is claimed here.

## Consumer Decisions

| Surface | Decision | Reason / verification |
|---|---|---|
| Formation registry and coordinator | Consume all three enums through the existing registry | Dispatch tests call the public coordinator |
| Framework presets | Construct canonical member/router/judge roles and resolve names to IDs once | Preset tests validate unknown/duplicate names |
| Shared team context / TeamResult | Consume serialized transcript and termination metadata | Round-trip and final-output tests |
| MemberEventSink | Emit additive `member_spoke` and `member_handoff` events with member IDs and compact metadata | Ordered event tests; no payload duplication |
| Framework stream bridge | Preserve metadata on both events | Bridge tests |
| v1 wire serializer | Admit the two event types; serialize transcript sequence and handoff destination | Wire contract tests |
| Existing chat/TUI member lanes | Intentionally ignore these additive kinds initially; lifecycle events remain authoritative | Ignore-path tests prevent false approval/complete state |
| Legacy clients | May ignore unknown event kinds | Existing lifecycle shape is unchanged |
| Durable team runner | Explicitly unsupported for conversation state in this increment | Reject-before-execution test and supports_durable_pause probes |
| Provider transport | Consume ordinary structured prompts through the existing member executor | No new provider or tool dispatch |

## Compatibility

All configuration is opt-in. The original formations and their defaults have no
transcript or new events. Unknown enum values and invalid new configuration fail
explicitly. No supervisor/manager dual keys or noncanonical coordination roles are
introduced. There is one transcript append authority and one session-ID derivation
in the existing subagent configuration.

## Validation

Table-driven tests cover each selector, bounded termination, malformed contracts,
unknown IDs, judge validation, cyclic handoffs bounded by max_turns, transcript
limits, member metric accumulation, consumer projections, and default isolation.
Every formation also has public coordinator-dispatch and preset coverage. Follow
WS-G's definition-of-done checklist in the multiagent handoff before marking its
coverage rows complete.

## Alternatives

Independent private histories cannot provide shared transcript semantics. A second
team runtime would duplicate dispatch, attribution, and streaming. Unbounded
transcripts and implicit fallback speakers violate the standards mandate.
