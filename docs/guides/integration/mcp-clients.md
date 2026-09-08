# MCP Integration Guide

This guide covers Victor's Model Context Protocol (MCP) integration for tool interoperability.

## Overview

MCP (Model Context Protocol) enables:

- **Expose Victor tools** to external AI clients (Claude Desktop, etc.)
- **Consume external tools** from MCP-compatible servers
- **Sandboxed execution** for untrusted tool sources
- **Protocol bridging** between different tool formats

## Quick Start

### Run Victor as MCP Server

```bash
# Start MCP server mode
victor serve --mcp --port 8080

# With specific tools exposed
victor serve --mcp --tools read,write,search
```

External clients can now connect and use Victor's tools.

### Connect to External MCP Server

```python
from victor.integrations.mcp import MCPClient

# Connect to an MCP server
client = MCPClient("http://localhost:3000")
await client.connect()

# List available tools
tools = await client.list_tools()
for tool in tools:
    print(f"{tool.name}: {tool.description}")

# Call a tool
result = await client.call_tool("search", {"query": "authentication"})
```

## MCP Server Mode

### Configuration

```python
from victor.integrations.mcp import MCPServer, MCPServerConfig

config = MCPServerConfig(
    host="0.0.0.0",
    port=8080,
    tools=["read", "write", "search", "bash"],  # Tools to expose
    auth_required=True,
    api_key="your-secret-key",
)

server = MCPServer(config)
await server.start()
```

### Exposed Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/tools` | GET | List available tools |
| `/tools/{name}` | GET | Get tool schema |
| `/tools/{name}/call` | POST | Execute tool |
| `/health` | GET | Health check |

### Tool Schema Format

```json
{
  "name": "read_file",
  "description": "Read contents of a file",
  "parameters": {
    "type": "object",
    "properties": {
      "path": {
        "type": "string",
        "description": "Path to the file"
      }
    },
    "required": ["path"]
  }
}
```

## MCP Client Mode

### Connecting to Servers

```python
from victor.integrations.mcp import MCPClient, MCPClientConfig

config = MCPClientConfig(
    url="http://localhost:3000",
    timeout=30.0,
    retry_attempts=3,
)

client = MCPClient(config)
await client.connect()
```

### Discovering Tools

```python
# Get all tools
tools = await client.list_tools()

# Get specific tool schema
schema = await client.get_tool_schema("search")
print(schema.parameters)
```

### Calling Tools

```python
# Simple call
result = await client.call_tool("read_file", {"path": "/app/config.py"})
print(result.content)

# With timeout
result = await client.call_tool(
    "search",
    {"query": "pattern matching"},
    timeout=10.0
)

# Check for errors
if result.error:
    print(f"Tool error: {result.error}")
else:
    print(result.content)
```

## MCP Registry

Manage multiple MCP connections:

```python
from victor.integrations.mcp import MCPRegistry, get_mcp_registry

registry = get_mcp_registry()

# Register a server
registry.register(
    name="code_tools",
    url="http://localhost:3000",
    tools=["search", "analyze"],
)

# Register another server
registry.register(
    name="web_tools",
    url="http://localhost:3001",
    tools=["fetch", "scrape"],
)

# Get a client
client = registry.get_client("code_tools")
result = await client.call_tool("search", {"query": "bug"})

# List all registered servers
for name, info in registry.list_all():
    print(f"{name}: {info.url} ({len(info.tools)} tools)")
```

## MCP Bridge Tool

Use external MCP tools within Victor agents:

```python
from victor.integrations.mcp import MCPBridgeTool

# Create bridge to external tools
bridge = MCPBridgeTool(
    server_url="http://localhost:3000",
    tool_name="external_search",
)

# Register with agent
agent = await Agent.create(
    tools=ToolSet.default() + [bridge]
)

# Agent can now use external_search tool
result = await agent.run("Search for authentication bugs")
```

## Sandboxed Execution

Run untrusted MCP tools in isolation:

```python
from victor.integrations.mcp import MCPSandbox, SandboxConfig

config = SandboxConfig(
    timeout=30.0,
    memory_limit="512M",
    network_access=False,
    filesystem_readonly=True,
    allowed_paths=["/app/data"],
)

sandbox = MCPSandbox(config)

# Execute tool in sandbox
result = await sandbox.execute(
    tool_name="untrusted_tool",
    arguments={"input": "data"},
    server_url="http://external-server:3000",
)
```

### Sandbox Restrictions

| Restriction | Description | Default |
|-------------|-------------|---------|
| `timeout` | Max execution time | 30s |
| `memory_limit` | Max memory usage | 512M |
| `network_access` | Allow network calls | False |
| `filesystem_readonly` | Read-only filesystem | True |
| `allowed_paths` | Writable paths | [] |

## Protocol Translation

Convert between tool formats:

```python
from victor.integrations.mcp import ProtocolAdapter

adapter = ProtocolAdapter()

# Victor tool to MCP schema
victor_tool = registry.get_tool("read")
mcp_schema = adapter.to_mcp_schema(victor_tool)

# MCP schema to Victor tool
mcp_tool = external_client.get_tool_schema("search")
victor_tool = adapter.from_mcp_schema(mcp_tool)
```

## Authentication

### API Key Authentication

```python
# Server side
config = MCPServerConfig(
    auth_required=True,
    api_key="secret-key-123",
)

# Client side
client = MCPClient(
    url="http://localhost:8080",
    api_key="secret-key-123",
)
```

### OAuth Authentication

```python
from victor.integrations.mcp import OAuthConfig

oauth = OAuthConfig(
    client_id="your-client-id",
    client_secret="your-client-secret",
    token_url="https://auth.example.com/token",
)

client = MCPClient(
    url="http://localhost:8080",
    oauth=oauth,
)
```

## Event Integration

MCP events are emitted to EventBus:

```python
from victor.observability.event_bus import get_event_bus, EventCategory

bus = get_event_bus()

bus.subscribe(EventCategory.TOOL, lambda e:
    if e.data.get("source") == "mcp":
        print(f"MCP tool: {e.data.get('tool_name')}")
)

# Events emitted:
# - mcp_tool_called (source=mcp)
# - mcp_tool_completed
# - mcp_tool_error
# - mcp_server_connected
# - mcp_server_disconnected
```

## CLI Commands

```bash
# Start MCP server
victor mcp serve --port 8080

# List connected servers
victor mcp list

# Add a server
victor mcp add code_tools http://localhost:3000

# Remove a server
victor mcp remove code_tools

# Test a connection
victor mcp test code_tools

# List tools from server
victor mcp tools code_tools
```

## Configuration File

Configure MCP in `.victor/config.yaml`:

```yaml
mcp:
  server:
    enabled: true
    port: 8080
    tools:
      - read
      - write
      - search
    auth:
      required: true
      api_key_env: VICTOR_MCP_API_KEY

  clients:
    - name: code_tools
      url: http://localhost:3000
      tools:
        - analyze
        - refactor

    - name: web_tools
      url: http://localhost:3001
      sandbox: true
      sandbox_config:
        timeout: 10
        network_access: false
```

## Best Practices

### 1. Use Sandboxing for External Tools

```python
# Always sandbox untrusted sources
sandbox = MCPSandbox(SandboxConfig(
    network_access=False,
    filesystem_readonly=True,
))
```

### 2. Validate Tool Inputs

```python
# Validate before calling
schema = await client.get_tool_schema("search")
if not validate_input(args, schema.parameters):
    raise ValueError("Invalid arguments")
```

### 3. Handle Disconnections

```python
client.on_disconnect(lambda:
    logger.warning("MCP server disconnected, retrying...")
)

client.set_reconnect_policy(
    max_attempts=5,
    backoff="exponential",
)
```

### 4. Monitor Tool Usage

```python
bus.subscribe(EventCategory.TOOL, lambda e:
    if e.data.get("source") == "mcp":
        metrics.record_mcp_call(
            tool=e.data.get("tool_name"),
            duration=e.data.get("duration"),
        )
)
```

## Troubleshooting

### Connection Refused

1. Verify server is running: `curl http://localhost:8080/health`
2. Check firewall settings
3. Verify port is correct

### Authentication Failed

1. Check API key is set correctly
2. Verify OAuth credentials
3. Check token expiration

### Tool Not Found

1. List available tools: `victor mcp tools <server>`
2. Check tool name spelling
3. Verify server exposes the tool

### Timeout Errors

1. Increase timeout in config
2. Check server responsiveness
3. Consider using sandbox with higher limits

## Related Resources

- [Tool Catalog →](../../reference/tools/catalog.md) - Available Victor tools
- [Observability →](../observability/index.md) - MCP event monitoring
- [User Guide →](../../user-guide/index.md) - General usage
