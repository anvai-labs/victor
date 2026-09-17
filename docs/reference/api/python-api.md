# Python API Reference

The canonical signatures are in the [generated API reference](../../api-reference/auto-api.md).
For a complete runnable example, start with [Your First Agent in 5 Minutes](../../getting-started/first-agent.md).
This page explains the public agent entry points without maintaining a second signature catalog.

## Create and close an agent

Use the asynchronous `Agent.create` factory. An async context manager closes provider
connections and sessions when the block exits:

```python
import asyncio

from victor.framework import Agent, EventType


async def main():
    async with await Agent.create(
        provider="ollama",
        model="qwen2.5-coder:7b",
    ) as agent:
        result = await agent.run("Describe this project in three sentences.")
        print(result.content)

        async for event in agent.stream("Write a short project summary."):
            if event.type == EventType.CONTENT:
                print(event.content, end="", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
```

The example requires Ollama running with the named model downloaded. See the
[installation guide](../../getting-started/installation.md) for setup and the
[provider reference](../providers/index.md) for other providers.

## Run a task

`await agent.run(prompt, context=...)` returns a `TaskResult`, whose `content` holds
response text. Its `success`, `error`, `tool_calls` and `metadata` fields describe the
outcome. Configure available tools when creating the agent; `run` does not accept a
`tools` argument. See the generated `Agent.create` documentation for supported options.

## Stream events

`agent.stream(prompt, context=...)` yields `AgentExecutionEvent` objects. Check
`event.type` against `EventType` before reading the corresponding payload, as the
example above does for content. This is an async iterator; iterate with `async for`.
There is no public `Agent.astream` method.

## Continue a conversation

`agent.chat(initial_prompt)` is a synchronous session factory. It returns a
`ChatSession`; sending a message is asynchronous and returns a `TaskResult`:

```python
# Inside the agent's async context:
session = agent.chat("Let's review the authentication module.")
response = await session.send("First, explain its current structure.")
print(response.content)
response = await session.send("Suggest focused regression tests.")
print(response.content)
```

For explicit session configuration, use `agent.create_session(...)`, which returns
the canonical `AgentSession` implementation. Both surfaces preserve conversation context.

## Related public APIs

- [Generated framework and client API](../../api-reference/auto-api.md): Agent,
  AgentBuilder, StateGraph, WorkflowEngine, VictorClient and SessionConfig.
- [Workflow API](../../api-reference/workflows.md) and
  [workflow development](../../guides/workflow-development/examples.md).
- [Tool API](../../api-reference/tools.md) and [tool catalog](../tools/catalog.md).
- [Provider API](../../api-reference/providers.md) and [protocols](../../api-reference/protocols.md).
- [Vertical reference](../verticals/index.md).

Client surfaces such as a CLI or HTTP service should use `VictorClient` and the
framework factories; see the [architectural boundary](../../architecture.md#layer-rules).

The previous hand-written catalog contained obsolete constructor, streaming and
chat examples. It is recoverable from [repository history](https://github.com/anvai-labs/victor/commits/develop/docs/reference/api/python-api.md).
