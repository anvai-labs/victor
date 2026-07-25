# Wire Events (v1) — the universal agent event contract

One versioned JSON schema for every surface: the framework's typed events
serialize to small wire documents that the web (SSE), CLI, and TUI all render
from. The agent loop stays legible everywhere — thinking → tool_call →
tool_result → content → stream_end — without per-surface reshaping.

Source of truth: `victor/framework/wire_events.py` (`WIRE_VERSION = 1`).
The contract is additive-only within a version: consumers must ignore unknown
keys and unknown event types.

## Event types

Every document carries `"v": 1` and an `"event"` discriminator.

### `thinking`

```json
{"v": 1, "event": "thinking", "content": "The user wants the failing test fixed…"}
```

### `tool_call`

```json
{"v": 1, "event": "tool_call", "tool": "read", "arguments": {"path": "victor/framework/agent.py"}}
```

Arguments are JSON-safe and size-bounded (2 KB per value).

### `tool_result`

```json
{"v": 1, "event": "tool_result", "tool": "read", "success": true, "result": "…file preview…"}
```

Failed tools (including framework `tool_error` events) arrive as
`"success": false`. Results are truncated at 16 KB — the stream carries
previews; full results belong to the conversation.

### `content`

```json
{"v": 1, "event": "content", "content": "The bug is in the retry loop: "}
```

Emitted incrementally; concatenate `content` fields to reconstruct the answer.

### `error`

```json
{"v": 1, "event": "error", "message": "provider timeout after 120s"}
```

In-stream failures are surfaced as an `error` event *before* the terminal
`stream_end` — a consumer never sees a silently truncated stream.

### `stream_end`

```json
{"v": 1, "event": "stream_end"}
```

Exactly one per stream, always last.

## Consuming over SSE

The web server exposes the stream at `POST /chat/stream`
(`web/server/main.py`), framed as Server-Sent Events:

```bash
curl -N -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "Find and fix the failing test"}'
```

```
data: {"v": 1, "event": "thinking", "content": "…"}

data: {"v": 1, "event": "tool_call", "tool": "shell", "arguments": {"cmd": "pytest -x"}}

data: {"v": 1, "event": "tool_result", "tool": "shell", "success": true, "result": "1 failed…"}

data: {"v": 1, "event": "content", "content": "The failure is caused by…"}

data: {"v": 1, "event": "stream_end"}
```

Pass `"session_id"` (from `POST /session/token`) to continue an existing
conversation; the response's `X-Session-Id` header identifies the session
serving the stream.

## Consuming in Python

```python
from victor.framework.client import VictorClient
from victor.framework.session_config import SessionConfig
from victor.framework.wire_events import stream_wire_events

client = VictorClient(SessionConfig())
async for wire in stream_wire_events(client, "Analyze this code"):
    print(wire["event"], wire.get("content", ""))
```

`stream_wire_events` guarantees the termination contract (one `stream_end`,
errors surfaced) regardless of how the underlying stream ends.
