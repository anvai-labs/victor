# Victor — UX/UI & Adoption Action Plan

**Audience:** A medium-complexity agent (or engineer) executing this plan end-to-end.
**Scope:** UX/UI improvements across CLI (Typer/Rich), TUI (Textual), Web API (FastAPI) + Web UI, with supporting architecture fixes that the UX work depends on.
**Status:** Draft for execution. Each task is self-contained with acceptance criteria.

---

## 0. How to Use This Document

- Tasks are ordered by **execution priority**. Do them in order unless a task's
  *Dependencies* field says otherwise.
- Each task lists **Rationale**, **Alternatives considered**, and the
  **Recommended choice** so the executor can defend the decision in review.
- "Grounded evidence" cites verified file:line references. Re-verify before
  editing — the codebase moves.
- Run `make lint` and the smallest relevant test subset after each task.
  Run `make check-repo-hygiene` if you touch docs or workflow metadata.

### Dependency graph (read first)

```
P0-A (web boundary → VictorClient) ──┐
                                     ├──> P1 (typed event stream UX, all surfaces)
P0-B (session state → service) ──────┘         │
                                               ├──> P3 (web UI event-driven rendering)
P2 (CLI consolidation)          (independent) ─┤
P4 (first-run experience)       (depends P2) ──┤
P5 (TUI parity)                 (depends P1) ──┘
P6 (orchestrator decomposition) (parallel, de-risks P0 long-term)
```

**Critical path:** P0-A → P1 → P3. Everything UX-facing that matters hangs off
the web boundary honoring the canonical client seam and the typed event model.

---

## P0 — Foundational (blocks all other UX work)

### P0-A. Route the web server through `VictorClient`, not `AgentOrchestrator`

**Evidence:** `web/server/main.py:28` currently contains
`from victor.agent.orchestrator import AgentOrchestrator` — the exact import the
workspace rules forbid from the UI layer ("NEVER import AgentOrchestrator from
UI layer"). The mandated path is `VictorClient` + `SessionConfig` +
`Agent.create(session_config=...)`.

**Work:**
1. Replace all direct `AgentOrchestrator` construction/usage in
   `web/server/main.py` with `VictorClient` (`victor/framework/client.py`)
   created via `Agent.create(session_config=SessionConfig(...))`.
2. Map existing server settings (`settings.server_*`, CORS, session TTL) into
   `SessionConfig` where applicable; keep server-only concerns (CORS, API keys)
   in the web layer.
3. Add a boundary guard test mirroring
   `tests/unit/framework/test_architectural_boundaries.py` so
   `web/server/**` cannot re-import `victor.agent.orchestrator`.

**Acceptance criteria:**
- `grep -rn "AgentOrchestrator" web/server/` returns zero import sites.
- Existing web API endpoints (chat, session lifecycle, websocket) behave
  identically; manual smoke test of `/chat` + websocket round-trip passes.
- New guard test fails if the import is reintroduced.

**Rationale:** One execution path across all surfaces. Today the web layer
re-implements session/agent lifecycle that the framework already owns; every
framework improvement (retry, caching, events) must be manually ported to the
web server or it silently diverges. This is the single highest-leverage change
for adoption because the web UI is the first thing most evaluators see.

**Alternatives:**
- *A1 — Wrap orchestrator in a thin local adapter.* Cheapest (no behavior
  change), but entrenches the bypass and adds a third lifecycle owner.
  **Rejected.**
- *A2 — Rewrite web server as a pure client of `victor serve` (HTTP-to-HTTP).*
  Cleanest end-state, but doubles network hops and defers the fix.
  **Deferred to a later hardening phase.**
- **Recommended: P0-A as stated.** Uses the seam that already exists and is
  already enforced for CLI/TUI.

---

### P0-B. Move web session state out of module-level globals

**Evidence:** `web/server/main.py:77` `SESSION_AGENTS: Dict[str, Dict[str, Any]] = {}`
behind an `asyncio.Lock()` (line 78), with a token map at line 81
(`SESSION_TOKENS`). This is in-process, non-restartable, single-worker state.

**Work:**
1. Introduce a `SessionStore` protocol in the web layer (get/put/delete/touch,
   TTL-aware) with two implementations:
   - `InMemorySessionStore` (default; wraps the current dict behavior, so dev
     experience is unchanged), and
   - `ServiceBackedSessionStore` delegating persistence to the framework's
     canonical `SessionService` (sessions already persist — see the
     two-database architecture, project DB "Project sessions").
2. Make the store injectable via FastAPI `Depends`, configurable by env var
   (e.g. `VICTOR_WEB_SESSION_STORE=memory|service`).
3. Keep the existing TTL/idle cleanup semantics (lines 84-87 constants) but move
   them into the store.

**Acceptance criteria:**
- Server restart with `service` store preserves sessions; `memory` store
  behaves as today.
- No module-level mutable dicts remain in `web/server/main.py`.
- Unit tests for both store implementations.

**Rationale:** P0-A alone would still leave lifecycle state stranded in the web
process; P0-B makes sessions restart-safe and horizontally scalable later, and
it reuses persistence the framework already has instead of inventing a third
store.

**Alternatives:**
- *B1 — Redis/external store.* Real scalability, but adds an infra dependency
  for a surface that is currently local-first. **Deferred** — the protocol seam
  makes it a drop-in later.
- *B2 — Leave in-memory, document limitation.* Zero cost, but restart = lost
  sessions is an adoption-killer in demos. **Rejected.**
- **Recommended: protocol + two implementations.** Smallest step that removes
  the architectural smell without new infrastructure.

---

## P1 — Critical UX: one typed event stream, every surface

### P1. Promote the 6 framework event types to the universal UX contract

**Evidence:** The framework already defines structured events — THINKING,
TOOL_CALL, TOOL_RESULT, CONTENT, ERROR, STREAM_END — with correlation IDs
(`victor/framework/events.py`, per architecture docs). The CLI streams them;
the web layer manages raw message round-trips instead.

**Work:**
1. Define a single wire schema (JSON) for the 6 event types, versioned
   (`"v": 1`), in one shared module (framework-level, not web-local).
2. Web: expose the chat response as Server-Sent Events (or extend the existing
   WebSocket) emitting exactly these events, sourced from
   `VictorClient.stream()` — not re-shaped per endpoint.
3. CLI/TUI: refactor rendering to consume the *same* schema (it largely does;
   this step removes any local drift).
4. Document the schema in `docs/api-reference/` with one example per event.

**Acceptance criteria:**
- A single `curl -N` (SSE) or `websocat` session shows the full event stream:
  thinking → tool_call → tool_result → content → stream_end.
- Web UI and CLI render the same run with equivalent content.
- Contract test: a recorded event stream replays identically through both
  renderers.

**Rationale:** From first principles, an agent's UX quality = legibility of its
loop. Users trust what they can see. The events already exist — this is the
cheapest large UX win available, and it converts "is it stuck?" into visible
progress. It is also the prerequisite for the web UI (P3) and TUI parity (P5).

**Alternatives:**
- *C1 — Per-surface event models.* Matches each UI's idioms, but re-creates
  the divergence the runtime team already eliminated. **Rejected.**
- *C2 — GraphQL subscriptions.* More expressive querying, heavier stack;
  unjustified for 6 event types. **Rejected.**
- **Recommended: versioned JSON-over-SSE.** Boring, debuggable with curl,
  works through proxies.

**Dependencies:** P0-A (events must flow through VictorClient).

---

## P2 — High-impact terminal UX: CLI consolidation

### P2. Consolidate ~30 flat subcommands into ~6 verbs

**Evidence:** `victor/ui/cli.py:41-61` imports ~20+ subcommand apps
(`benchmark_app`, `bayesian_app`, `capabilities_app`, `chat_app`, `config_app`,
`dashboard_app`, `gateway_app`, `docs_app`, `embeddings_app`, `examples_app`,
`experiment_app`, `fep_app`, `graph_app`, `index_app`, `init_app`, `ml_app`,
`mcp_app`, plus lazy-loaded `keys_app`/`test_provider_app`). The skills CLI
already demonstrates the target pattern: `victor skill list|info|search|create|remove|preview|run`.

**Work:**
1. Group into top-level verbs (proposal): `chat`, `run` (workflow/skill/skill
   execution), `config` (config/keys/profiles), `index` (index/graph/embeddings),
   `observe` (observability/experiments/benchmark/metrics), `tool`
   (doctor/capabilities/examples).
2. Keep old commands as **hidden aliases** for one minor release with a
   deprecation warning pointing to the new verb (mirror the existing lazy-import
   pattern used for `keys`, `cli.py:65-76`).
3. Add `victor --help` grouping so the first screen shows ≤7 verbs.
4. Update `docs/getting-started/` and README quickstart to the new verbs.

**Acceptance criteria:**
- `victor --help` fits on one screen; each verb has `list/info/run`-style
  sub-verbs where sensible.
- Old invocations still work and print a deprecation hint.
- No command loses functionality; tests for alias redirects.

**Rationale:** The first 5 minutes determine adoption. 30 flat commands read as
a wall; 6 verbs read as a product. This is pure UX leverage — zero new
capability, large perceived-simplicity gain. Terminal UX matters as much as web
UX here because `victor` (bare command → interactive chat) is the primary
onboarding path.

**Alternatives:**
- *D1 — Leave commands, improve docs only.* Cheaper, but in-product
  discoverability (the actual friction) is unchanged. **Rejected.**
- *D2 — Interactive command picker (fuzzy TUI launcher).* Delightful, but it's
  an addition, not a fix — do after consolidation if wanted. **Deferred.**
- **Recommended: verb-grouping with hidden aliases.** Follows the repo's own
  precedent (lazy deprecated commands, skills CLI verbs).

**Dependencies:** None (independent of P0/P1). P4 builds on it.

---

## P3 — Web UI: event-driven rendering

### P3. Render the web chat surface from the P1 event stream

**Premise correction (2026-07-25):** this task originally said "reuse the
existing Vite project under `web/ui/`" — that project (and `ui/`) was removed
in PR #141 (June 2026, security consolidation); the web chat surface is the
pure-Python Chainlit app (`victor/ui/chat_app`, `victor ui`). The Chainlit
app already renders incrementally (thinking steps, per-call tool steps with
duration, streamed markdown, stop/retry actions — the chat-UX PR train), so
P3's remaining substance is contract work, not UI rebuilding.

**Work (as executed):**
1. ✅ Wire-contract hygiene: `tool_result.result` on the wire is always the
   human-readable output string — the internal tool-pipeline payload dict
   (original_result, follow_up_suggestions, was_pruned, …) no longer leaks
   through `to_wire_event`.
2. ✅ Additive v1 fields real renderers need: `tool_call.call_id`,
   `tool_result.call_id` (parallel-call correlation), `tool_result.elapsed_ms`
   (live-timeline durations), `tool_result.truncated`.
3. ✅ `map_wire_event()` in `victor/ui/chat_app/event_mapping.py`: pure
   translator from v1 wire events to the same `RenderAction` vocabulary the
   Chainlit app renders from — any Python surface consuming the contract
   (P5 TUI, remote SSE consumers) inherits the reference render semantics.
4. ✅ Renderer replay contract test
   (`tests/unit/ui/chat_app/test_wire_render_parity.py`): one golden stream,
   both mappers, pinned agreement on kinds/identity/correlation/outcome.

**Acceptance criteria:**
- Long tool runs show live progress with durations; user can cancel mid-run
  (already shipped in the Chainlit app: per-call steps + stop action).
- Only the P1 schema crosses the wire — no internal payload shapes (guarded
  by `test_tool_result_payload_dict_never_leaks_internal_keys`).
- A wire-fed renderer reproduces the in-process render semantics (parity test).

**Deferred / follow-ups:**
- Cross-visit session resume in Chainlit needs a Chainlit data layer +
  history-by-id API — explicitly deferred as a FEP (see `chat_app/app.py`
  docstring); best-effort same-process history replay already ships.
- `vscode-victor`'s `streamChat` still speaks the pre-P1 `type`/`[DONE]`
  protocol against `POST /chat/stream` — port to v1 (TypeScript, separate PR).

**Dependencies:** P1 (event schema), P0-B (session resume), transitively P0-A.

---

## P4 — Onboarding: first-run experience

### P4. Zero-friction first run on the terminal

**Work:**
1. `victor` bare (already interactive chat — keep) + detect first run: no
   configured provider → launch an interactive setup wizard (provider picker →
   key entry → 30-second smoke test → "you're ready" with 3 example prompts).
2. `victor doctor` (exists — `run_doctor` imported at `victor/ui/cli.py:49`)
   becomes the wizard's diagnoser: on any first-run failure, auto-suggest
   `victor doctor` output inline.
3. Ship `victor examples` (exists, `cli.py:55`) as a browsable,
   runnable-from-menu list post-setup.
4. Target metric: install → first successful agent response in < 3 minutes.

**Acceptance criteria:**
- Fresh venv: `pip install victor-ai` → `victor` → wizard → working chat with
  zero doc-reading.
- Wizard is skippable (`--no-setup`) and idempotent.
- Docs quickstart matches the wizard flow exactly.

**Rationale:** Adoption friction concentrates at setup (providers, keys,
profiles). The building blocks exist (`doctor`, `examples`, `init`); they are
just not sequenced into a journey. Sequencing is cheap; the payoff is the
entire top of funnel.

**Alternatives:**
- *F1 — Static "Getting Started" doc improvements only.* Necessary, not
  sufficient — users hit env issues docs can't see. **Complement, not
  substitute.**
- *F2 — Cloud-hosted sandbox first-run.* Removes local setup entirely; large
  infra commitment, off-mission for a local-first framework. **Rejected for now.**
- **Recommended: interactive wizard over existing doctor/examples/init
  primitives.**

**Dependencies:** P2 (wizard references the consolidated verbs).

---

## P5 — TUI parity

### P5. Bring `victor dashboard` (Textual) to event-stream parity

**Premise note (2026-07-25):** the Textual dashboard
(`victor/observability/dashboard`) is a passive observability viewer — it has
no chat pane and never calls `VictorClient`. Consistent with this task's own
alternatives note ("the dashboard serves a different job — monitoring"), P5
adds wire-stream *rendering* to the dashboard rather than turning it into a
second interactive chat client; interactive session/provider switching stays
with the chat surfaces (Chainlit `ChatSettings`, CLI flags).

**Work (as executed):**
1. ✅ `victor/ui/tui/wire_timeline.py`: pure `WireTimelineState` (RenderAction →
   Rich-markup lines, mirroring the Chainlit semantics: tool summary labels,
   duration suffix, ✓/✗ marks, truncation notes, buffered text paragraphs) +
   `WireTimeline` RichLog widget + `parse_wire_line` (accepts raw JSONL *and*
   SSE `data:` framing, so a `curl -N` capture of `POST /chat/stream` replays
   directly).
2. ✅ Dashboard gains an **Agent tab** hosting the timeline, with
   `victor observe dashboard --wire-log <path>` replay + live tail.
3. ✅ Boundary guard extended: `victor/observability/dashboard` is now scanned
   by `test_architectural_boundaries.py` (no `victor.agent.*`, no orchestrator
   imports) — it was previously outside the guard entirely.

**Acceptance criteria:**
- Same recorded contract stream renders equivalently: the TUI test's golden
  stream is the replay-parity stream from P3
  (`tests/unit/ui/tui/test_wire_timeline.py` ↔
  `tests/unit/ui/chat_app/test_wire_render_parity.py`).
- No `victor.agent.*` imports in the TUI layer (guard-enforced, previously
  unscanned).

**Deferred:** session picker / provider-model switcher in the dashboard — an
interactive-client concern; revisit only if the dashboard's job expands beyond
monitoring.

**Dependencies:** P1, P3 (`map_wire_event`).

---

## P6 — Parallel de-risking: orchestrator decomposition

### P6. Continue service-first migration (Chat/Tool/Session first)

**Evidence:** `AgentOrchestrator` is ~4,703 LOC, capped by the
`test_hotspot_size_guard` ratchet (per workspace docs). The 6 canonical
services are mandatory; coordinators are documented as compatibility-only.

**Work:**
1. Migrate the 3 highest-fan-in coordinators (Chat, Tool, Session) fully into
   their services; delete the compatibility shims behind them.
2. Lower the size ratchet as lines leave (ratchet down only, never up).
3. Do not route any new P0–P5 behavior through coordinators.

**Acceptance criteria:** Orchestrator LOC trending down per release; service
delegation tests (`tests/unit/runtime/test_service_layer_validation.py`) stay
green.

**Rationale:** P0-A removes the worst *consumer* of the orchestrator; P6
shrinks the orchestrator itself so future UX work stops negotiating with a god
object. Runs in parallel — it must not gate the UX critical path.

**Alternatives:** Accept the facade permanently. **Rejected** — it is the
stated anti-target, and every new feature routes around it, compounding
coupling.

---

## Summary Table

| # | Task | Class | Depends on | Effort | Primary payoff |
|---|------|-------|-----------|--------|----------------|
| P0-A | Web server → VictorClient | Foundational | — | M | One execution path; unblocks all web UX |
| P0-B | Session state → store protocol | Foundational | P0-A | M | Restart-safe sessions; scalability seam |
| P1 | Typed event stream contract | Critical UX | P0-A | M | Legible agent loop everywhere |
| P2 | CLI verb consolidation | Terminal UX | — | S | First-5-minutes simplicity |
| P3 | Web UI event-driven rendering | Critical UX | P1, P0-B | L | Demo-grade web experience |
| P4 | First-run wizard | Terminal UX | P2 | S-M | <3 min install→chat |
| P5 | TUI parity | Terminal UX | P1, P0-B | M | Three surfaces, one contract |
| P6 | Orchestrator decomposition | De-risking | — (parallel) | L (ongoing) | Removes long-term coupling |

**Execution order for a single agent:** P2 (quick, independent) → P0-A → P0-B → P1 → P4 → P3 → P5. P6 runs in background/parallel and must not block the UX path.

---

## Appendix — Verified Grounding

| Claim | Source |
|---|---|
| Web server imports orchestrator directly | `web/server/main.py:28` |
| In-process session state | `web/server/main.py:77-81` |
| Server CORS/settings pattern | `web/server/main.py:39-74` |
| CLI subcommand sprawl (~20 apps) | `victor/ui/cli.py:41-61` |
| Lazy-import pattern for deprecated commands | `victor/ui/cli.py:64-76` |
| `doctor`/`examples` commands exist | `victor/ui/cli.py:49`, `:55` |
| Package is Alpha, v0.8.0 | `pyproject.toml:43`, `:21` |
| AgenticLoop/ExecutionCoordinator layering | `victor/framework/agentic_loop.py:15-34` |
| 6 event types, SessionService, VictorClient seam, orchestrator size cap | Workspace architecture docs (CLAUDE.md / .victor/init.md) |

*Re-verify all line references before editing; the codebase moves. Items sourced from architecture docs rather than direct file reads are marked as such and should be confirmed during implementation.*