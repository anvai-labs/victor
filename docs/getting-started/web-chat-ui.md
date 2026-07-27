# Web Chat UI

Victor ships a pure-Python **web chat surface** built on [Chainlit](https://chainlit.io). It binds
to `VictorClient` (the single UI entry point — no orchestrator access), so the web UI streams the
same agent runtime as the CLI/TUI/API.

## Install & launch

```bash
pip install "victor-ai[chat-ui]"
victor ui                       # launches the Chainlit app in your browser
victor ui --profile zai-glm52-openai  # use a profile from ~/.victor/profiles.yaml
```

`victor ui` shells out to `chainlit run` on `victor/ui/chat_app/app.py`. The selected profile is
passed through to the per-session `VictorClient`.

## What you get

- **Live streaming** — assistant tokens and reasoning stream in as the agent works; a transient
  "Thinking…" placeholder shows until the first output arrives.
- **Natural per-iteration flow** — text and tool calls interleave like the terminal (text → tools →
  text …) instead of all tool steps piling at the end; parallel tool calls are grouped.
- **Per-call tool steps** — each tool call renders as its own collapsible step showing the
  arguments, the real output (syntax-highlighted), elapsed time, truncation notices, and any
  follow-up suggestions.
- **Informed approvals** — risky tools (shell, file writes/edits/deletes, git commit/push) prompt
  with a card showing the exact command/diff/content before you Approve or Deny.
- **Stop a running turn** — a Stop control cancels the in-flight turn cleanly (the provider stream
  is drained, not abandoned).
- **In-chat error recovery** — a failed turn shows a friendly message with a **Retry** action and a
  provider-switch hint.
- **ChatSettings** — switch **provider / model / profile** and toggle **tool approval** mid-session;
  the per-session client is rebuilt from the new settings (no restart).
- **Session-restore seam** — a reconnected session replays its prior turns (best-effort).
- **Per-turn cost footer** — tokens (in/out), round-trips, latency, $ and cache-hit rate for each
  turn, so the savings from context pruning and routing are visible.

## Approval behavior

Approval is on by default for state-mutating/code-executing tools and routes through Victor's
policy/approval handler. Read-only tools stay friction-free. Toggle it from ChatSettings or via
`SessionConfig` (`tool_approval_enabled`, `ask_on_tools`, `ask_fallback`).

## Notes

- The chat UI is one of several surfaces (CLI, TUI, web, HTTP API, MCP) — all go through
  `VictorClient`, so behavior matches across them.
- Full cross-visit session resume and a live plan panel are tracked as future enhancements (they
  require framework-level persistence/plan-event support).
