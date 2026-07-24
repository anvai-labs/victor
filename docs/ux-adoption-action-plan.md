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

### P3. Rebuild the web chat UI around the P1 event stream

**Work:**
1. Replace request/response chat rendering with incremental rendering per
   event: thinking (collapsible), tool_call (name + args, status pill),
   tool_result (expandable, syntax-highlighted), content (markdown stream),
   error (inline, with retry affordance), stream_end (finalize).
2. Show tool execution as a live timeline with elapsed time; add "cancel" wired
   to the session.
3. Surface session resume (P0-B) in the UI: "Continue previous session" list.
4. Keep the UI dependency-light; reuse the existing Vite project under `web/ui/`.

**Acceptance criteria:**
- Long tool runs show live progress; user can cancel mid-run.
- Refreshing the page with a `service` session store restores the conversation.
- No direct orchestrator-shaped payloads in the frontend — only the P1 schema.

**Rationale:** The web UI is the evaluation surface for most new users; today it
presents an opaque spinner while the framework's best asset (visible reasoning)
is discarded. This task converts P1 into perceived quality.

**Alternatives:**
- *E1 — Full app framework (Next.js) rewrite.* Nicer DX, but a large new stack
  for a single chat surface. **Rejected.**
- *E2 — Embed the Textual TUI via `textual-web`.* Zero new UI code and perfect
  parity with terminal UX, but limited styling control for a product surface.
  **Viable fallback** if web UI capacity is constrained.
- **Recommended: incremental rendering in the existing Vite app.**

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

**Work:**
1. Render the P1 event stream in the TUI with the same semantics as P3:
   collapsible thinking, tool timeline, streaming markdown content.
2. Add session picker (from P0-B store) and provider/model switcher.
3. Guard: TUI must consume events via `VictorClient.stream()` only — extend the
   boundary guard tests if not already covered.

**Acceptance criteria:**
- Same recorded contract-test stream renders equivalently in CLI, TUI, web.
- No `victor.agent.*` imports in the TUI layer.

**Rationale:** Three surfaces rendering one contract is the co-design end-state:
framework improvements appear everywhere simultaneously.

**Alternatives:** Merge TUI into CLI chat (one terminal surface). Simpler, but
the dashboard serves a different job (monitoring/multi-session). **Keep both,
share the renderer.**

**Dependencies:** P1, P0-B.

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