# Victor as an MCP Server Toolkit

This guide covers how to run Victor as a Model Context Protocol (MCP) server, enabling external clients like Claude Desktop to use Victor's 34 tool modules across 9 verticals.

## Overview

Victor can expose its entire tool suite via the Model Context Protocol (MCP), allowing:

- **Claude Desktop integration**: Use Victor's tools directly from Claude Desktop
- **Multi-client support**: Any MCP-compatible client can connect
- **Tool filtering**: Expose only specific tools or verticals
- **Docker deployment**: Run as a containerized service
- **Secure execution**: Sandboxed tool execution options

## Quick Start

### Option 1: Direct Command (Recommended for Claude Desktop)

```bash
# Install Victor
pip install victor-ai

# Run as MCP server (stdio mode for Claude Desktop)
victor mcp
```

### Option 2: Python Module

```bash
python -m victor.integrations.mcp.server
```

### Option 3: Docker

```bash
docker run -it victor-ai/mcp-server
```

## Integration with Claude Desktop

### Step 1: Install Victor

```bash
# Install from PyPI
pip install victor-ai

# Or install from source
git clone https://github.com/victor-ai/victor.git
cd victor
pip install -e ./victor-contracts -e ".[dev]"
```

### Step 2: Configure Claude Desktop

Add Victor to your Claude Desktop configuration file:

**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
**Linux**: `~/.config/Claude/claude_desktop_config.json`

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

### Step 3: Restart Claude Desktop

Restart Claude Desktop to load the new MCP server configuration.

### Step 4: Verify Connection

In Claude Desktop, you should see Victor's tools available. Try asking:

> "Use Victor to list the files in the current directory"

## Available Tools

Victor exposes 34 tool modules across 9 verticals:

### Coding Tools
| Tool | Description |
|------|-------------|
| `read` | Read file contents |
| `write` | Write content to files |
| `edit` | Edit files with diff-based changes |
| `ls` | List directory contents |
| `git` | Git operations (status, diff, log, etc.) |
| `bash` | Execute shell commands |
| `grep` | Search file contents |
| `find_files` | Find files by pattern |
| `code_search` | Semantic code search |
| `code_review` | Automated code review |
| `test_generation` | Generate unit tests |
| `refactor` | Code refactoring suggestions |

### DevOps Tools
| Tool | Description |
|------|-------------|
| `docker` | Docker container operations |
| `terraform` | Infrastructure as Code |
| `cicd` | CI/CD pipeline operations |
| `kubernetes` | K8s cluster management |
| `aws` | AWS cloud operations |

### RAG Tools
| Tool | Description |
|------|-------------|
| `ingest_document` | Ingest documents for RAG |
| `vector_search` | Semantic vector search |
| `retrieve` | Retrieve relevant context |

### Data Analysis Tools
| Tool | Description |
|------|-------------|
| `pandas_query` | DataFrame operations |
| `visualize` | Create data visualizations |
| `statistics` | Statistical analysis |

### Research Tools
| Tool | Description |
|------|-------------|
| `web_search` | Search the web |
| `fetch_url` | Fetch URL contents |
| `summarize` | Summarize documents |
| `cite` | Citation management |

## Configuration Options

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `VICTOR_MCP_LOG_LEVEL` | Logging level (DEBUG, INFO, WARNING, ERROR) | WARNING |
| `VICTOR_MCP_TOOLS` | Comma-separated list of tools to expose | All tools |
| `VICTOR_MCP_VERTICALS` | Comma-separated list of verticals to expose | All verticals |
| `VICTOR_AIRGAPPED_MODE` | Disable network-dependent tools | false |

### Claude Desktop Configuration with Environment Variables

```json
{
  "mcpServers": {
    "victor": {
      "command": "victor",
      "args": ["mcp", "--log-level", "INFO"],
      "env": {
        "VICTOR_MCP_TOOLS": "read,write,edit,ls,git,bash",
        "VICTOR_MCP_LOG_LEVEL": "WARNING"
      }
    }
  }
}
```

### Exposing Only Specific Tools

```json
{
  "mcpServers": {
    "victor-coding": {
      "command": "victor",
      "args": ["mcp"],
      "env": {
        "VICTOR_MCP_VERTICALS": "coding"
      }
    }
  }
}
```

### Air-Gapped Mode (No Network Tools)

```json
{
  "mcpServers": {
    "victor-airgapped": {
      "command": "victor",
      "args": ["mcp"],
      "env": {
        "VICTOR_AIRGAPPED_MODE": "true"
      }
    }
  }
}
```

## Integration with Other MCP Clients

### Generic MCP Client Connection

Victor's MCP server uses stdio transport. Any MCP client supporting stdio can connect:

```python
import subprocess
import json

# Start Victor MCP server
process = subprocess.Popen(
    ["victor", "mcp"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)

# Send initialize message
init_msg = {
    "jsonrpc": "2.0",
    "id": "1",
    "method": "initialize",
    "params": {
        "clientInfo": {"name": "MyClient", "version": "1.0.0"}
    }
}
process.stdin.write(json.dumps(init_msg) + "\n")
process.stdin.flush()

# Read response
response = json.loads(process.stdout.readline())
print(f"Connected to: {response['result']['serverInfo']['name']}")
```

### VS Code Extension

For VS Code integration, add to your MCP extension settings:

```json
{
  "mcp.servers": {
    "victor": {
      "command": "victor",
      "args": ["mcp"],
      "transport": "stdio"
    }
  }
}
```

### Programmatic Client in Python

```python
import asyncio
from victor.integrations.mcp import MCPClient

async def main():
    # Connect to Victor MCP server
    client = MCPClient(transport="stdio", command=["victor", "mcp"])
    await client.connect()

    # List available tools
    tools = await client.list_tools()
    print(f"Available tools: {len(tools)}")

    # Call a tool
    result = await client.call_tool("ls", {"path": "."})
    print(result)

    await client.disconnect()

asyncio.run(main())
```

## Docker Deployment

### Using Pre-built Image

```bash
# Pull and run
docker run -it --rm \
  -v $(pwd):/workspace \
  victor-ai/mcp-server

# With environment variables
docker run -it --rm \
  -v $(pwd):/workspace \
  -e VICTOR_MCP_TOOLS=read,write,ls \
  victor-ai/mcp-server
```

### Building Custom Image

```bash
# Navigate to Victor repository
cd victor

# Build MCP server image
docker build -t victor-mcp-server -f docker/mcp-server/Dockerfile .

# Run
docker run -it --rm -v $(pwd):/workspace victor-mcp-server
```

### Docker Compose

```bash
# Start MCP server with docker-compose
cd docker/mcp-server
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

### Claude Desktop with Docker

```json
{
  "mcpServers": {
    "victor-docker": {
      "command": "docker",
      "args": [
        "run", "-i", "--rm",
        "-v", "${HOME}:/workspace",
        "victor-ai/mcp-server"
      ],
      "env": {}
    }
  }
}
```

## Security Considerations

### Tool Permissions

Victor tools have different permission levels:

| Cost Tier | Description | Examples |
|-----------|-------------|----------|
| `FREE` | Read-only, no side effects | `read`, `ls`, `grep` |
| `LOW` | Minimal side effects | `git status`, `code_search` |
| `MEDIUM` | Moderate changes | `write`, `edit`, `git commit` |
| `HIGH` | Significant changes | `bash`, `docker`, `terraform` |

### Restricting High-Risk Tools

```json
{
  "mcpServers": {
    "victor-safe": {
      "command": "victor",
      "args": ["mcp"],
      "env": {
        "VICTOR_MCP_TOOLS": "read,ls,grep,code_search,git"
      }
    }
  }
}
```

### Sandboxed Execution

For untrusted environments, use sandboxed mode:

```json
{
  "mcpServers": {
    "victor-sandboxed": {
      "command": "victor",
      "args": ["mcp"],
      "env": {
        "VICTOR_SANDBOX_MODE": "true",
        "VICTOR_SANDBOX_TIMEOUT": "30",
        "VICTOR_SANDBOX_MEMORY": "512M"
      }
    }
  }
}
```

### File System Restrictions

Limit file system access:

```json
{
  "mcpServers": {
    "victor-restricted": {
      "command": "victor",
      "args": ["mcp"],
      "env": {
        "VICTOR_ALLOWED_PATHS": "/home/user/projects,/tmp",
        "VICTOR_DENIED_PATHS": "/etc,/var,/usr"
      }
    }
  }
}
```

### Network Restrictions

Disable network-dependent tools:

```json
{
  "mcpServers": {
    "victor-offline": {
      "command": "victor",
      "args": ["mcp"],
      "env": {
        "VICTOR_AIRGAPPED_MODE": "true"
      }
    }
  }
}
```

## Troubleshooting

### Server Not Starting

1. **Check Victor installation**:
   ```bash
   victor --version
   ```

2. **Test MCP server directly**:
   ```bash
   victor mcp --log-level DEBUG
   ```

3. **Check for dependency issues**:
   ```bash
   pip install victor-ai --upgrade
   ```

### Tools Not Appearing in Claude Desktop

1. **Restart Claude Desktop** after configuration changes

2. **Check configuration file syntax**:
   ```bash
   cat ~/Library/Application\ Support/Claude/claude_desktop_config.json | python -m json.tool
   ```

3. **Verify Victor command path**:
   ```bash
   which victor
   ```

4. **Use absolute path if needed**:
   ```json
   {
     "mcpServers": {
       "victor": {
         "command": "/usr/local/bin/victor",
         "args": ["mcp"]
       }
     }
   }
   ```

### Connection Timeouts

Increase timeout in Victor settings:

```bash
export VICTOR_MCP_TIMEOUT=60
victor mcp
```

### Permission Errors

1. **File system permissions**: Ensure Victor has access to required directories

2. **Docker socket access** (for Docker tools):
   ```bash
   docker run -it --rm \
     -v /var/run/docker.sock:/var/run/docker.sock \
     -v $(pwd):/workspace \
     victor-ai/mcp-server
   ```

### Debugging

Enable verbose logging:

```bash
victor mcp --log-level DEBUG 2>victor-mcp.log
```

View MCP protocol messages:

```bash
# In another terminal
tail -f victor-mcp.log
```

## Advanced Usage

### Custom Tool Registration

Register additional tools with the MCP server:

```python
from victor.integrations.mcp import MCPServer
from victor.tools.base import ToolRegistry, tool

@tool(name="my_custom_tool", description="My custom tool")
def my_custom_tool(arg1: str) -> str:
    """Custom tool implementation."""
    return f"Result: {arg1}"

# Create registry with custom tools
registry = ToolRegistry()
registry.register(my_custom_tool)

# Create and run MCP server
server = MCPServer(
    name="Victor Custom MCP Server",
    tool_registry=registry
)

import asyncio
asyncio.run(server.start_stdio_server())
```

### Resource Registration

Expose files and data as MCP resources:

```python
from victor.integrations.mcp import MCPServer, MCPResource

server = MCPServer(name="Victor MCP Server")

# Register resources
server.register_resource(MCPResource(
    uri="file:///path/to/project/README.md",
    name="Project README",
    description="Project documentation",
    mime_type="text/markdown"
))

server.register_resource(MCPResource(
    uri="file:///path/to/project/src",
    name="Source Code",
    description="Project source code directory",
    mime_type="inode/directory"
))
```

### Event Monitoring

Subscribe to MCP events:

```python
from victor.observability.event_bus import get_event_bus, EventCategory

bus = get_event_bus()

def on_tool_call(event):
    if event.data.get("source") == "mcp":
        print(f"MCP tool called: {event.data.get('tool_name')}")

bus.subscribe(EventCategory.TOOL, on_tool_call)
```

## Related Documentation

- [MCP Integration →](../../guides/integration/mcp-clients.md) - Complete MCP integration documentation
- [Tool Catalog →](../../reference/tools/catalog.md) - Full list of available tools
- [Observability →](../../guides/observability/index.md) - Monitoring and logging
- [Local Models →](../../guides/development/local-models.md) - Using local LLM providers
