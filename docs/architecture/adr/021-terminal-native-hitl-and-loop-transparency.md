# ADR-021: Terminal-Native HITL and Agent-Loop Transparency

## Metadata

- **Status**: Accepted (2026-07-30 — v1 shipped in the TUI: terminal-native approval modal, stall
  watchdog, `/help` + palette, `~/.victor/keybindings.json`, Esc interrupt, and an *inferred* phase
  indicator. Exact phase-events (framework enhancement, FEP-gated) and REPL-surface approval parity
  remain deferred. Was Proposed.)
- **Date**: 2026-07-29
- **Decision Makers**: Vijaykumar Singh
- **Related ADRs**: 020 (interactive TUI — the surface these render on), 001 (service-first runtime —
  the loop being surfaced)
- **Work tracked by**: [TD-23](../../tech-stack.md#technical-debt-register)
- **Benchmark**: [competitive-benchmark-2026-07.md](../competitive-benchmark-2026-07.md) §2, §4

## Context

Two terminal-first expectations are unmet:

1. **Approvals are browser-bound.** Human-in-the-loop tool approval renders through Chainlit's
   `AskActionMessage` (Approve/Reject buttons) in the **web** UI. A user in `victor chat` who hits a
   policy `ASK` verdict on `bash`/`write_file`/`git_push` has no in-terminal way to approve — they
   must switch to the browser surface. Aider approves with a `y/n` prompt in the terminal; Claude
   Code confirms inline. Victor does not.
2. **The loop is opaque.** During execution the REPL shows a generic "running…" spinner. Victor runs
   a research-rooted **PERCEIVE→PLAN→ACT→EVALUATE→DECIDE** loop (`framework/agentic_loop.py`) and
   tracks per-turn cost (C0), but none of that phase/cost state is surfaced inline. Claude Code shows
   the active phase and token/cost in its status line.

Two smaller ergonomics gaps compound it: there is **no `/help`** overlay for the 26+ slash commands
(discoverability is docs-only), **no user keybindings** config, and streaming rendering can **block
on a stalled agent** — the Rich `Live` context waits on events with no watchdog, so a wedged loop
reads as a frozen terminal (cf. TD-20's wedged-loop failure mode).

## Decision

Make HITL and loop state **legible in the terminal**, reusing the framework's existing signals (no
new agentic loop, no new event vocabulary):

1. **Terminal-native approval.** Add a CLI/TUI approval renderer as a peer to the Chainlit one, both
   mapping the same framework approval request → `ApprovalStatus`. In the REPL: an inline prompt with
   a preview of the tool call (command / file diff / content) and `[a]pprove / [r]eject / [v]iew`.
   In the TUI (ADR-020): the inline diff pane's approve-in-place control. The Chainlit path stays for
   the web surface; the framework approval contract is surface-agnostic.
2. **Phase + cost indicator.** Surface the live PERCEIVE→PLAN→ACT→EVALUATE→DECIDE phase and the
   running token/cost from the existing per-turn tracker in the status line (TUI sidebar / REPL
   status suffix). Read-only projection of state the loop already emits.
3. **`/help` discoverability.** `slash/handler.py` already has `list_commands()`; wire a `/help`
   command and a first-turn hint. No new command infrastructure.
4. **Keybindings config.** Load `~/.victor/keybindings.json` (optional) to rebind TUI actions and
   slash shortcuts; ship sane defaults so the file is never required.
5. **Stall watchdog.** Wrap the streaming event wait with a timeout that flips the indicator to
   "agent unresponsive (Ns)" and offers interrupt, instead of a silent freeze. Pairs with the
   existing fail-safe approval timeout (120s → reject).

## Rationale

- **First principles.** Approval and progress are *control-plane* interactions; they must live where
  the user is (the terminal), not force a context switch. Opacity during long runs erodes trust more
  than latency does.
- **Reuse.** Every piece rides existing machinery: the policy engine's ASK verdicts, the surface-
  agnostic approval contract, the loop's phase state, `list_commands()`, the per-turn cost tracker.
  This is wiring, not new subsystems.
- **Co-design.** Specified with ADR-020 (its render surface) and consistent with TD-20's lesson that
  a wedged loop must be *visible and killable*, not silent.

## Consequences

- **Positive**: terminal users never leave the terminal; runs are legible; wedged loops surface
  instead of hang; commands are discoverable.
- **Negative**: two approval renderers to keep in lockstep behind one contract; watchdog timeouts
  need tuning to avoid false "unresponsive" on legitimately slow tools.
- **Neutral**: web/Chainlit approval and the HTTP HITL endpoints are unchanged.

## Implementation

Client-layer; **no companion FEP** *unless* the surface-agnostic approval contract needs a new field
on the public `victor.framework` surface — if so, that field lands via a small FEP first, then this
ADR consumes it.

1. Extract the approval-request→status mapping to a surface-agnostic helper; add the CLI/TUI renderer.
2. Add the phase/cost status projection (read-only) to REPL + TUI.
3. `/help` + first-turn hint; `keybindings.json` loader + defaults.
4. Stall watchdog around the streaming wait; wire interrupt.

## Alternatives Considered

- **Keep approvals web-only, document it.** Rejected: contradicts the terminal-first category; the
  §2 gap is the point.
- **Auto-approve in CLI to avoid the prompt.** Rejected: defeats the policy engine's safety purpose;
  the ASK verdict exists precisely to gate destructive tools.
- **Full progress dashboard in CLI.** Deferred: the phase/cost line is the high-value slice; a richer
  dashboard is the web/observability surface's job.

## References

- [ADR-020](020-interactive-terminal-tui.md), [ADR-005](005-event-system.md)
- `victor/framework/agentic_loop.py` (PPAED loop), `victor/ui/slash/handler.py`,
  `victor/ui/rendering/live_renderer.py`, `victor/ui/chat_app/approval.py` (Chainlit path)
- [TD-20](../../tech-stack.md#technical-debt-register) (wedged-loop / log-flood lesson)

## Revision History

| Date | Version | Changes | Author |
|------|---------|---------|--------|
| 2026-07-29 | 1.0 | Initial ADR — terminal-native HITL, phase/cost indicators, /help, keybindings, watchdog | Vijaykumar Singh |
