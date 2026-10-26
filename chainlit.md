# Victor — your coding agent in the browser

Ask me to write code, search the repo, run tools, or debug a failure. I stream my
reasoning and every tool call as I work, and risky tools ask before running.

## Tips

- **Stop** cancels a running turn at any point; **Retry** re-runs a failed one.
- The ⚙ **settings** panel switches provider, profile and model mid-conversation.
- Tool calls collapse into steps with their real output, duration and follow-ups.
- Sessions resume across visits — pick up a previous conversation from the greeting.

Running in remote mode? Set `VICTOR_UI_REMOTE_URL` to point this UI at a
`victor serve` instance and the same conversation runs against the hosted agent
service (approvals and session resume return when the durable approval flow
reaches the HTTP API).
