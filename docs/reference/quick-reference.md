# Quick Reference

## Python API

```python
import asyncio
from victor.framework import Agent, EventType

async def main():
    async with await Agent.create(provider="ollama", model="llama3.1:8b") as agent:
        result = await agent.run("Explain this repository")
        print(result.content)
        async for event in agent.stream("Summarize the key points"):
            if event.type == EventType.CONTENT:
                print(event.content, end="")

asyncio.run(main())
```

`Agent.chat(prompt)` creates a chat session synchronously. Use `await session.send(...)`
for subsequent turns. `Agent.stream`, not `Agent.astream`, provides streamed events.
See [Python API](api/python-api.md) for session and result details.

## Workflows

Use [the workflow tutorial](../tutorials/create-workflow.md) for YAML files and StateGraph.
`await agent.run_workflow(name, context={...})` runs a named workflow supplied by the
configured vertical. It does not load a YAML filename.

## Teams

Use `await Agent.create_team(name=..., goal=..., members=...)`, then `await team.run()`.
See [multi-agent teams](../guides/MULTI_AGENT_TEAMS.md) for a complete example.

## CLI

```bash
victor --help
victor chat --help
victor workflow --help
victor scheduler --help
```

The installed command's help lists the available flags for your version. See the
[CLI reference](cli-commands.md), [installation](../getting-started/installation.md),
and [configuration](../getting-started/configuration.md) guides.

The former expanded cheatsheet is preserved in [page history](https://github.com/anvai-labs/victor/commits/develop/docs/reference/quick-reference.md).
