# Victor as an MCP Server

Victor exposes discovered tools to MCP clients over stdio. The installed tool inventory
depends on available modules and optional dependencies; use MCP tool discovery to inspect
it instead of relying on a fixed count.

## Quick Start

```bash
pip install victor-ai
victor mcp --help
victor mcp
```

`victor mcp` defaults to stdio. Its current options are `--stdio/--no-stdio` and
`--log-level` (`-l`). Selecting `--no-stdio` reports that only stdio is supported.
There are no `--port`, `--tools`, or `--sandbox` options on this command.

## Integration with MCP clients

Configure a client that supports stdio servers to launch Victor:

```json
{
  "mcpServers": {
    "victor": {
      "command": "victor",
      "args": ["mcp"],
      "env": {}
    }
  }
}
```

Use an absolute executable path when the client's PATH differs from your shell.
The client must support this configuration shape; consult its own configuration UI
or documentation for the file location. For Python clients, see
[MCP client integration](../../guides/integration/mcp-clients.md).

## Available Tools

The CLI discovers decorated tools from the installed Victor tool modules. Missing
optional dependencies may prevent individual modules from loading. Ask the connected
client to list tools and inspect each tool's input schema before calling it.

## Configuration Options

The server process inherits its environment and working directory from the client.
Tool behavior follows the installed tools and runtime configuration. The MCP command
does not itself provide a filesystem jail, HTTP authentication gateway, or tool allowlist.
Use deployment controls appropriate to the tools you expose.

## Troubleshooting

```bash
victor mcp --log-level DEBUG 2>victor-mcp.log
```

Protocol traffic uses stdout and logs use stderr. If the client cannot start Victor,
check the executable path and its environment. If tools are missing, inspect stderr
for module import failures. Do not point an HTTP client at this stdio command.

## Advanced Usage

The Python `MCPServer` accepts a tool registry and can run with
`await server.start_stdio_server()`. Client applications should normally use the CLI
entry point above. There is no separately verified `victor-ai/mcp-server` image in
this guide; use the repository's [release process](../../development/releasing/publishing.md)
for published artifacts.

## Event Monitoring

See [observability events](../../guides/observability/event-bus.md) for the current
asynchronous topic-based API.

The former transport and deployment examples remain in [page history](https://github.com/anvai-labs/victor/commits/develop/docs/reference/api/mcp-server.md).
