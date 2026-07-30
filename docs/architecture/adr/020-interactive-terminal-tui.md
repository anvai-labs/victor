# ADR-020: Interactive Terminal TUI (Textual) as a First-Class Surface

## Metadata

- **Status**: Accepted (2026-07-30 — v1 shipped: `victor tui` / `victor chat --tui`, opt-in,
  `victor/ui/tui/`; the dedicated diff pane and per-member team lanes remain deferred. Was Proposed.)
- **Date**: 2026-07-29
- **Decision Makers**: Vijaykumar Singh
- **Related ADRs**: 021 (terminal-native HITL & loop transparency — shares the surface), 005 (event
  system — the `RenderAction` vocabulary this reuses)
- **Work tracked by**: [TD-22](../../tech-stack.md#technical-debt-register)
- **Benchmark**: [competitive-benchmark-2026-07.md](../competitive-benchmark-2026-07.md) §2

## Context

Victor's rendering layer is genuinely strong — adaptive terminal-capability detection, safe-split
streaming markdown, per-tool-category colours, three themes — but there is **no full interactive
terminal UI**. The only Textual code, `victor/ui/tui/wire_timeline.py` (171 LOC), *replays* a
recorded JSONL wire stream into a `RichLog`; it is not a live driver. The `victor` command lands in a
prompt-toolkit REPL, and the only richer surface is the Chainlit **web** UI (`victor ui`).

For a product whose primary market is *terminal* coding agents, this is the single largest visible
gap: the two incumbents users compare against — Claude Code and Aider — both run fully in the
terminal with panels, live agent state, an inline diff view, and keyboard navigation. Victor makes a
terminal user open a browser to get the richer experience. `docs/architecture/ux-redesign-plan.md`
argues the same in prose; this ADR turns that into a tracked architectural decision.

The good news is the foundation exists: the unified `RenderAction` event vocabulary (ADR-005 event
system) already renders identically across CLI (Rich), web (Chainlit), and the wire-timeline
replay. A live TUI is a **new consumer of the same events**, not a new event model.

## Decision

Build a first-class interactive **Textual** app as a peer surface to the REPL and web UI, driven by
the existing `RenderAction` stream (no new event vocabulary). Target surface:

- **Layout:** conversation pane · collapsible tool/diff pane · agent-state sidebar (current phase,
  model, token/cost meter) · input line with slash-command completion.
- **Live agent state:** consumes the same events the wire-timeline replays, but from the live
  session runner instead of a JSONL file.
- **Inline diff view:** file edits render as an approve-in-place diff (feeds ADR-021's terminal-native
  HITL rather than deferring to Chainlit).
- **Keyboard-first:** navigation + actions are keyboard-driven and rebindable (see ADR-021's
  `keybindings.json`).
- **Graceful fallback:** the plain REPL remains for dumb terminals / pipes / CI; the TUI is selected
  when the terminal supports it (reuse `rendering/terminal_capabilities.py`), never forced.

Reuse over rebuild: extend `victor/ui/tui/` from the existing wire-timeline widgets; do **not**
fork a second rendering path or bypass `VictorClient`/`SessionConfig` (architectural boundary,
`test_architectural_boundaries.py`).

## Rationale

- **First principles.** A terminal-first agent's *primary* surface must be the terminal. Pushing the
  rich experience to a browser contradicts the product's own category.
- **Cheap because of prior investment.** The event model, capability detection, theming, and markdown
  renderer are done; a live TUI is mostly wiring live events into Textual widgets.
- **Co-design.** The TUI is specified *together with* ADR-021 (it is the surface that makes
  terminal-native approval and phase indicators visible) and inherits the boundary guard so it cannot
  regress the client→framework layering.

## Consequences

- **Positive**: closes the most-cited competitive gap; unlocks terminal HITL (ADR-021); a single
  event model now has three *interactive* consumers.
- **Negative**: a real surface to build and maintain; Textual version pinning; more UI test surface
  (snapshot/interaction tests needed).
- **Neutral**: REPL, web UI, and HTTP API are unaffected; no framework/runtime change.

## Implementation

Phased, behind capability detection:

1. **Skeleton** — Textual app shell + conversation pane driven by live `RenderAction`s; launch via
   `victor` when the terminal is capable, `--repl` to force the old path.
2. **Panes** — tool/diff pane + agent-state sidebar (phase/model/token/cost).
3. **Interaction** — slash-command palette, keyboard nav, inline diff approve (with ADR-021).
4. **Tests** — Textual snapshot + interaction tests; resize (SIGWINCH) handling.

No public-framework-API change ⇒ no companion FEP (client-layer only). If new client↔framework
event fields are required, add them to the ADR-005 event contract first.

## Alternatives Considered

- **Invest further in the Chainlit web UI instead.** Rejected: does not serve terminal-first users;
  keeps the browser dependency this ADR is removing.
- **Adopt a different TUI toolkit (e.g. urwid).** Rejected: Textual is already a dependency and the
  wire-timeline widgets are Textual; consistency + reuse win.
- **Enrich the prompt-toolkit REPL in place.** Rejected: REPL cannot express panels/sidebar/diff
  ergonomically; it remains as the fallback, not the ceiling.

## References

- [ADR-005](005-event-system.md), [ADR-021](021-terminal-native-hitl-and-loop-transparency.md)
- [ux-redesign-plan.md](../ux-redesign-plan.md)
- `victor/ui/tui/wire_timeline.py`, `victor/ui/rendering/terminal_capabilities.py`,
  `victor/ui/rendering/live_renderer.py`, `victor/framework/client.py`

## Revision History

| Date | Version | Changes | Author |
|------|---------|---------|--------|
| 2026-07-29 | 1.0 | Initial ADR — interactive Textual TUI as a first-class surface | Vijaykumar Singh |
