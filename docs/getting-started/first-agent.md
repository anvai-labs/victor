# Your First Agent in 5 Minutes

Build and run a working agent from Python — a complete, runnable script with a
local-model path (Ollama, no API key) and a cloud path (Anthropic).

If you want the chat CLI instead of the Python API, see the
[Quickstart](quickstart.md).

## Prerequisites

- Victor installed: `pip install victor-ai` ([Installation Guide](installation.md))
- One of:
    - **Local**: [Ollama](https://ollama.com) running (`ollama serve`) with a model
      pulled (`ollama pull qwen2.5-coder:7b`) — free, private, no API key
    - **Cloud**: an Anthropic API key (`export ANTHROPIC_API_KEY=sk-ant-...`)

---

## The Complete Script

Save this as `first_agent.py`. It creates an agent, runs a task, streams a
second task token-by-token, and cleans up. The async context manager
(`async with`) guarantees the agent's provider connections and sessions are
closed even if an error occurs.

### Option A: Local model (Ollama, no API key)

```python
"""Your first Victor agent — local model via Ollama."""

import asyncio

from victor.framework import Agent, EventType


async def main() -> None:
    # Agent.create() is async — it must be awaited inside an async function.
    # `async with` closes the agent (connections, sessions) automatically.
    async with await Agent.create(
        provider="ollama",
        model="qwen2.5-coder:7b",
    ) as agent:
        # One-shot task: returns a TaskResult when the agent finishes.
        result = await agent.run("Summarize what this project does in 3 bullets.")
        print(result.content)

        # Streaming task: consume events as they arrive.
        async for event in agent.stream("Write a haiku about version control."):
            if event.type == EventType.CONTENT:
                print(event.content, end="", flush=True)
        print()


if __name__ == "__main__":
    asyncio.run(main())
```

Run it:

```bash
ollama serve            # if not already running
python first_agent.py
```

### Option B: Cloud model (Anthropic)

Same script — only the `Agent.create()` call changes:

```python
async with await Agent.create(
    provider="anthropic",
    model="claude-sonnet-4-5",
) as agent:
    ...
```

Run it:

```bash
export ANTHROPIC_API_KEY=sk-ant-your-key
python first_agent.py
```

Any other configured provider works the same way — pass `provider="openai"`,
`"google"`, `"groq"`, etc. If you omit `provider`, Victor uses your active
profile or default from `~/.victor/profiles.yaml`
([Configuration Guide](configuration.md)).

---

## What Just Happened

- `Agent.create(...)` builds a fully wired agent: provider connection, the
  default tool set (file reading, search, shell, and more), session state, and
  observability. It is async because it performs I/O (provider checks, tool
  registry setup).
- `agent.run(prompt)` executes the full agentic loop — the agent may call
  tools (read files, search code) before answering — and returns a
  `TaskResult`; `result.content` is the final text.
- `agent.stream(prompt)` yields typed `AgentExecutionEvent`s instead of
  blocking: filter on `EventType.CONTENT` for text, or also watch
  `EventType.TOOL_CALL` / `EventType.TOOL_RESULT` to display tool activity.
- `async with` calls `agent.close()` on exit. Without it, call
  `await agent.close()` yourself in a `finally` block.

## Common Variations

```python
# Multi-turn conversation (context carries across sends)
session = agent.chat("Let's review the auth module")
first = await session.send("Explain the auth flow")
followup = await session.send("Now suggest one improvement")

# Restrict tools or go read-only
from victor.framework import ToolSet
agent = await Agent.create(provider="ollama", tools=ToolSet.minimal())

# Domain-specialized agent via a vertical (curated tools + prompts)
agent = await Agent.create(provider="anthropic", vertical="coding")
```

Verticals can also be activated from the CLI (`victor chat --vertical coding`) —
both paths load the same vertical definition. See
[Two Ways to Activate a Vertical](../reference/verticals/index.md#two-ways-to-activate-a-vertical).

## Next Steps

1. **Pick your provider** — [Provider decision matrix](../user-guide/providers.md#provider-decision-matrix)
   and [Provider setup](../reference/providers/setup.md)
2. **Explore the tools** your agent can use — [Tool Catalog](../reference/tools/catalog.md)
3. **Go multi-step** — [StateGraph workflows](../guides/workflow-development/dsl.md)
   and [multi-agent teams](../user-guide/workflows.md)
4. **Tune configuration** — profiles, modes, project context:
   [Configuration Guide](configuration.md)

---

**Next**: [Configuration Guide](configuration.md) | [Basic Usage](basic-usage.md)
