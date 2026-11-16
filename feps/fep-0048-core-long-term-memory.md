---
fep: "0048"
title: "Core Long-Term Memory — Preferences and Episodic Store Wired at Perceive"
type: Standards Track
status: Draft
created: 2026-11-16
modified: 2026-11-16
authors:
  - name: Vijaykumar Singh
    email: vijay@anvaiops.com
    github: vjsingh1984
reviewers: []
discussion: https://github.com/anvai-labs/victor/discussions/0048
---

# FEP-0048: Core Long-Term Memory — Preferences and Episodic Store Wired at Perceive

## Summary

Victor's memory today is split across three disconnected pieces: an entity/graph memory
subsystem (`victor/storage/memory/` — `UnifiedMemoryCoordinator`, entity extraction via
tree-sitter, conversation embedding store) oriented toward *code facts*, an RL learner layer
(`quality_weights.py`, prompt optimizers) that silently shapes prompts from outcomes, and an
external ProximaDB-backed conversational-memory integration that is flag-gated and documented
as experimental. What is missing is the capability every competing assistant shipped in
2025–2026 and users now assume: a durable, user-visible **preference and episodic memory** —
"the user prefers tables over prose", "this project uses pytest, never unittest", "we renamed
the service to X last week" — that persists across sessions, is readable and correctable by
the user, and is injected into the loop before planning. This FEP adds a core memory
subsystem with two scopes (user, project) and two content classes (preferences, episodes),
stored beside the existing `UnifiedMemoryCoordinator`, injected at the loop's PERCEIVE stage
via the existing `PerceptionIntegration` seam, editable through an explicit tool and a TUI/API
surface, and ranked into context with the existing relevance/recency strategies. Entity-graph
memory remains the code-intelligence tier; ProximaDB becomes the scale-out backend rather than
the only path. Affected users: every chat/agent session gains memory by default (opt-out per
scope); backward compatibility is clean because all wiring is additive and off-when-empty.

## Motivation

### Problem Statement

Agentic coding sessions lose the same lessons repeatedly. A user corrects style once
("always use `uv`, never pip") and re-corrects it next session; a project decision made on
Monday is re-litigated on Wednesday; the assistant re-reads the same onboarding facts every
run because nothing persists between runs except code indexes. Competing products made
durable memory a headline feature in 2025–2026 (assistant-level "memories" that users can
inspect and delete), and their absence is now felt as a gap rather than a nicety.

Victor's building blocks exist but do not compose into this capability. `MemoryType` in
`UnifiedMemoryCoordinator` has no user-preference or episodic types; ranking strategies
(relevance/recency/hybrid) and proactive hints exist but are fed only entity/code content;
the prompt builders consume RL-derived weights, not declarative facts; and the one
conversational-memory path requires the external ProximaDB deployment plus a feature flag,
documented as "In progress" in its own quickstart integration table. There is also no user
surface: nothing to view, edit, or forget a memory — which matters for correctness (stale
facts poison sessions) and for trust.

### Goals

1. Sessions recall durable user preferences and project facts without re-teaching, by default.
2. Memories are inspectable and correctable (`memory_list`/`memory_set`/`memory_forget`
   tool + TUI/API surfaces) — no write-only black boxes.
3. Injection happens before planning (PERCEIVE), bounded by the existing token-budget
   machinery, and never displaces higher-priority context.
4. Storage is local-first (project/global database scopes per the two-database model) with
   ProximaDB as the optional scale-out backend.

## Proposed Change

1. **New content classes**: extend the memory subsystem with `Preference` (stable user or
   project directives with scope, provenance, and confidence) and `Episode` (timestamped
   event summaries with session/thread linkage) stored in the project database
   (`.victor/project.db`) for project scope and the global database for user scope, consistent
   with the two-database model in `docs/architecture.md`. The store lives beside
   `victor/storage/memory/unified.py` as a `PreferenceEpisodicStore` behind the existing
   `MemoryProviderProtocol`, so `UnifiedMemoryCoordinator` ranking and transfer-hint machinery
   consume it without a parallel retrieval path.
2. **Write paths**: (a) an explicit LLM tool `memory_set`/`memory_forget` (registered in the
   tool registry, permission-gated as workspace-write) for user-directed capture; (b) an
   end-of-session consolidation hook in the Session service that proposes episodes from the
   session ledger and stores only human-confirmable facts by default (aggressive auto-capture
   is a setting, default off — wrong memories are worse than missing ones).
3. **Read path**: PERCEIVE-stage injection via `PerceptionIntegration` — a bounded memory
   block assembled by the coordinator's hybrid ranking, passed through the existing
   `BudgetManager` so memory competes for tokens by priority instead of free-riding.
   Injection respects an `effective_sandbox_policy`-style escape: `memory_enabled` (global
   and per-scope) with `VICTOR_MEMORY_ENABLED=false` opt-out.
4. **User surfaces**: slash command (`/memory`) and API routes (`/memory` list/set/delete)
   so memories are inspectable everywhere Victor runs; MCP exposure follows the existing
   server tool projection.
5. **Backends**: SQLite (project/global db) is the v1 store; the ProximaDB conversational
   backend plugs in behind the same protocol for embedding-scale recall, upgrading the
   existing flag-gated integration from parallel path to optional backend.

## Benefits

- **Cross-session competence**: corrected mistakes stay corrected; project conventions are
  recalled from run two onward. This is the single most-requested assistant capability and a
  visible parity gap against Claude-Memories-style features.
- **User trust through inspectability**: every memory has provenance and a delete path; the
  consolidation default (confirm before storing) prevents silent wrong-fact accumulation.
- **Composes with existing machinery**: ranking, hints, budgets, and the two-database model
  are reused, not duplicated; entity-graph memory remains untouched as the code tier, and
  the RL learner layer keeps operating on its own signals without a second retrieval path.
- **Local-first**: works air-gapped with the SQLite backend; ProximaDB remains optional —
  consistent with Victor's provider-sovereign positioning, and the existing flag-gated
  integration is upgraded from a parallel path to a pluggable backend rather than discarded.
- **Observable**: memories, verdicts, and injection decisions are inspectable through the
  standard surfaces, so failures are diagnosable rather than mysterious.

## Drawbacks and Alternatives

- **Stale-memory risk**: a wrong or outdated preference actively harms every later session.
  Mitigated by confirm-by-default capture, provenance + confidence fields, `/memory` editing,
  and decay/recency ranking; not eliminated.
- **Token budget competition**: memory blocks consume context that code and conversation may
  need. Mitigated by the existing budget/priority machinery and a hard injection cap.
- **Privacy surface**: memories persist facts the user may not want persisted (secrets,
  credentials mentioned in chat). Mitigated by reusing `victor/security/safety/secrets.py`
  scanning on the write path and refusing to store secret-shaped content.
- **Alternative considered — extend RL learners only**: rejected; RL weights shape *behavior
  scoring*, not declarative facts, and are neither user-visible nor editable.
- **Alternative considered — ProximaDB-only**: rejected as v1 default; it makes the core
  capability depend on an external service and contradicts local-first positioning.

## Unresolved Questions

- Forgetting policy: should episodes expire (e.g., 90-day recency decay) or persist until
  explicitly forgotten? (Proposal: episodes decay, preferences persist.)
- Multi-workspace projects sharing a user scope: how do conflicting project preferences
  resolve during ranking?
- Should `memory_set` require HITL approval under `read-only` permission mode?

## Implementation Plan

FOU Tier-A increments (tracked in `docs/architecture/victor-foundations-plan.md`), design
first per the large-epic flow:

1. **FOU-A2a**: `PreferenceEpisodicStore` + protocol integration + schema in the two
   databases; unit suite covering round-trip, scope isolation, secret-scan refusal, ranking
   interop.
2. **FOU-A2b**: write paths — `memory_set`/`memory_forget` tool with permission gating;
   consolidation hook behind confirm-by-default; negative tests for secret-shaped and
   oversized content.
3. **FOU-A2c**: PERCEIVE injection through `PerceptionIntegration` + budget integration;
   A/B harness run (existing `flag_ab.py` machinery) with `memory_enabled` off as baseline;
   graduation per ADR-036 policy.
4. **FOU-A2d**: `/memory` slash command + API routes; docs.

Each increment is one worktree/PR with the FOU ledger gates (RED→GREEN evidence, real-server
smoke, adversarial review before merge where shared state is touched). The A/B harness run in
A2c reuses the existing `flag_ab.py` machinery with `memory_enabled` off as baseline; the
default-on flip follows the ADR-036 evidence policy, and the consolidation hook stays
confirm-by-default until A2d's surfaces make confirmations ergonomic.

Status moves to Implemented when A2c merges with graduation evidence.

## Migration Path

- Additive schema in existing databases (new tables; no changes to existing tables).
- Memory is empty on upgrade: injection of an empty block is a no-op, so behavior before
  first capture is unchanged.
- Opt-out: `VICTOR_MEMORY_ENABLED=false` (global) or per-scope settings; existing ProximaDB
  users keep the current flag-gated path until it is re-pointed at the new protocol.
- The consolidation hook ships default-off for capture (confirm-by-default UI follows in
  A2d) so upgrades never silently start writing memories.

## Compatibility

New tool names register in `victor_contracts.constants.ToolNames` (additive; the contract
package releases independently). `MemoryProviderProtocol` gains optional methods with
default no-op implementations so existing providers compile unchanged. No changes to the
framework layer's public API surface beyond the PerceptionIntegration hook parameter, which
is additive and optional. Existing entity/graph memory, RL learners, and prompt builders are
unaffected; the ProximaDB integration keeps its current flag until re-pointed.

## References

- `victor/storage/memory/unified.py` — `UnifiedMemoryCoordinator`, `MemoryProviderProtocol`
- `victor/framework/perception_integration.py` — PERCEIVE seam
- `docs/architecture.md` — two-database model and layering rules
- `victor/security/safety/secrets.py` — secret-scan reuse on the write path
- `docs/guides/quickstart-proximadb-memory.md` — existing flag-gated integration status
- `docs/architecture/adr/036-flag-graduation-defaults-policy.md` — graduation policy for the
  injection flag

## Review Process

Per FEP-0001: 14-day review on merge; acceptance requires two maintainer approvals and no
blocking objections.

## Acceptance Criteria

- [ ] Preference and episode content classes land behind `MemoryProviderProtocol`
- [ ] Write path refuses secret-shaped content (negative tests pin this)
- [ ] PERCEIVE injection is budget-bounded and disabled-when-empty is a no-op
- [ ] `/memory` and API surfaces can list, set, and forget memories
- [ ] A/B evidence recorded for the injection flag per ADR-036 before default-on
