# MCP Client Integration

Victor's MCP client launches a server subprocess and communicates over stdio. Pass a
command list to `connect`; the first argument to `MCPClient` is a client name, not an
HTTP endpoint URL.

## Connect and discover tools

This example starts Victor's own MCP server as a subprocess. Use an absolute path to
the executable if it is not on the child process's PATH.

```python
import asyncio
from victor.integrations.mcp import MCPClient

async def main():
    client = MCPClient()
    try:
        if not await client.connect(["victor", "mcp"]):
            raise RuntimeError("MCP server connection failed")
        for tool in await client.refresh_tools():
            print(tool.name, tool.description)
    finally:
        await client.close()

asyncio.run(main())
```

## Call a tool

After checking the server's advertised schema, call
`await client.call_tool(tool_name, **arguments)`. Arguments are keyword arguments;
there is no positional argument dictionary after the tool name. Tool availability
depends on the connected server and installed dependencies.

## Server configuration

`MCPServerConfig` in `victor.integrations.mcp.registry` describes a registry-managed
server process. It is not an HTTP `MCPServer(host=..., port=...)` configuration.
For Victor's server command and client configuration, see the
[MCP server reference](../../reference/api/mcp-server.md).

## Observability and lifecycle

Close clients to terminate their subprocess and background health monitoring. Runtime
instrumentation uses [topic subscriptions](../observability/event-bus.md). The stdio
connection alone does not sandbox tools or grant remote HTTP access.

The earlier HTTP-style examples are preserved in [page history](https://github.com/anvai-labs/victor/commits/develop/docs/guides/integration/mcp-clients.md).
