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

`victor ui` shells out to `chainlit run` on `victor/ui/chat_app/bootstrap.py`. The selected profile is
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
- **Stop a running turn** — the native Stop control cancels the UI turn and closes its
  owned iterator. Full downstream and origin cancellation remain under qualification.
- **In-chat error recovery** — a failed turn shows a friendly message with a **Retry** action and a
  provider-switch hint.
- **ChatSettings** — switch **provider / model / profile** and toggle **tool approval** mid-session;
  the per-session client is rebuilt from the new settings (no restart).
- **Session-restore seam** — a reconnected session replays its prior turns (best-effort).
- **Per-turn cost footer** — tokens (in/out), round-trips, latency, $ and cache-hit rate for each
  turn, so the savings from context pruning and routing are visible.

## Workspace appearance

Victor includes an AnvaiOps-aligned teal/slate palette, compact controls, a wide chat
layout and readable monospace code blocks. Use **Toggle theme** for light, dark or
system appearance. Keyboard focus remains visible and reduced-motion preferences
are respected. The stylesheet adds no remote font dependency and includes local system fallbacks.
Chainlit itself may still request its default external font and math stylesheets.

Packaged assets are served only under `/victor-assets/`; starting the UI does not
copy files into your repository. Explicit Chainlit `custom_css`, logo, avatar and
nondefault assistant-name settings remain in effect. Chainlit's generic
`Assistant` name and `default` layout become Victor and wide respectively. Agent profiles and credentials stay on
the server. This visual change does not add workflow authoring or plan-progress
streaming; those remain in the [service tracker](../architecture/victor-agent-service-plan.md).

## Approval behavior

Approval is on by default for state-mutating/code-executing tools and routes through Victor's
policy/approval handler. Read-only tools stay friction-free. Toggle it from ChatSettings or via
`SessionConfig` (`tool_approval_enabled`, `ask_on_tools`, `ask_fallback`).

## Notes

- The chat UI is one of several surfaces (CLI, TUI, web, HTTP API, MCP) — all go through
  `VictorClient`, so behavior matches across them.
- Full cross-visit session resume and a live plan panel are tracked as future enhancements (they
  require framework-level persistence/plan-event support).

## Relationship to `victor serve`

| Command | Current role | Execution path |
| --- | --- | --- |
| `victor ui` | Chainlit browser chat with its own server/session lifecycle | In-process `VictorClient` |
| `victor serve` | FastAPI for IDE/API clients and workflow routes | Server session runner → `VictorClient` |
| `web/server/main.py` | Separate web compatibility API | Its session store → `VictorClient` |

Both reuse the same runtime services. They do not currently share one server-owned
session/run lifecycle. Chainlit supports mounting under FastAPI, but that alone
would not unify authorization, approvals or recovery. The
[consolidation plan](../architecture/victor-agent-service-plan.md#web-chat-and-api-hosting-consolidation)
tracks a single optional host with shared service ownership and remote API parity.
Direct embedded Python usage remains supported.
