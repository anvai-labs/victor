# Reference Documentation

Complete reference documentation for Victor AI Assistant.

## Quick Links

| Reference | Description |
|-----------|-------------|
| [**Providers**](providers/) | All 25 LLM providers with setup, models, and configuration |
| [**Tools**](tools/) | 34 tool modules organized by category |
| [**Configuration**](configuration/) | Complete configuration reference (profiles, API keys, MCP) |
| [**API**](api/) | HTTP API, MCP server, Python package API |
| [**Verticals**](verticals/) | 9 domain-specific assistants |

---

## Provider Reference

Complete reference for all 24 supported LLM providers.

### Local Providers (No API Key Required)

| Provider | Description | Quick Start |
|----------|-------------|-------------|
| Ollama | Local model execution | `ollama pull qwen2.5-coder:7b` |
| LM Studio | Local model UI | Install LM Studio app |
| vLLM | Fast local inference | `vllm serve qwen2.5-coder:7b` |
| llama.cpp | Lightweight inference | `llama-cli model -p prompt` |

### Cloud Providers (API Key Required)

| Provider | Models | Pricing | Setup |
|----------|--------|--------|------|
| Anthropic | Claude 3.5, Opus, Haiku | Pay-per-use | `export ANTHROPIC_API_KEY=sk-...` |
| OpenAI | GPT-4, o1 | Pay-per-use | `export OPENAI_API_KEY=sk-...` |
| Google | Gemini 2.0 | Pay-per-use | `export GOOGLE_API_KEY=...` |
| Azure | Azure OpenAI | Pay-per-use | Azure portal setup |
| And 14 more... | See full list | See full list | See full list |

[**Full Provider Reference →**](providers/)

---

## Tool Reference

34 tool modules organized by category for code operations, testing, search, and more.

### Tool Categories

| Category | Tools | Use Case |
|----------|-------|----------|
| **File Operations** | 12 tools | Read, write, edit, search files |
| **Git** | 8 tools | Version control, commits, branches |
| **Testing** | 6 tools | Test execution, coverage, fixtures |
| **Search** | 7 tools | Code search, grep, semantic search |
| **Web** | 5 tools | HTTP requests, scraping, browsing |
| **Database** | 4 tools | SQL execution, schema inspection |
| **CI/CD** | 4 tools | GitHub Actions, Jenkins, GitLab CI |
| **Docker** | 3 tools | Container management, docker-compose |
| **And 8 more...** | See full list | See full list |

[**Tool Catalog →**](tools/catalog.md)

---

## Configuration Reference

Complete configuration for Victor's behavior and integrations.

| File | Purpose | Reference |
|------|---------|----------|
| `profiles.yaml` | Provider and model profiles | [profiles.yaml →](configuration/index.md#profilesyaml) |
| `config.yaml` | Global settings and options | [config.yaml →](configuration/index.md#configyaml) |
| `mcp.yaml` | MCP server configuration | [mcp.yaml →](configuration/index.md#mcpyaml) |
| `.victor.md` | Project context and instructions | [.victor.md →](../user-guide/index.md#5-project-context) |
| `CLAUDE.md` | AI assistant project instructions | [CLAUDE.md →](../../CLAUDE.md) |

### Quick Configuration

**Local model (default)**:
```bash
# No configuration needed
victor  # Automatically uses Ollama
```

**Cloud provider**:
```bash
export ANTHROPIC_API_KEY=sk-...
victor chat --provider anthropic
```

**Profiles** (`~/.victor/profiles.yaml`):
```yaml
profiles:
  development:
    provider: anthropic
    model: claude-sonnet-4-20250514
  production:
    provider: openai
    model: gpt-4
```

[**Full Configuration Reference →**](configuration/)

---

## API Reference

Programmatic access to Victor's capabilities.

### HTTP API

`victor serve` - REST API server for HTTP clients

- **Endpoints**: Chat, streaming, workflows, agents
- **Authentication**: API key or token-based
- **Documentation**: [HTTP API →](api/http-api.md)

### MCP Server

`victor mcp` - Model Context Protocol server

- **Protocol**: MCP stdio or SSE transport
- **Capabilities**: Tool exposure, prompt templates
- **Documentation**: [MCP Server →](api/mcp-server.md)

### Python API

```python
from victor import Agent

agent = await Agent.create(provider="anthropic")
result = await agent.run("Write tests for auth.py")
```

- **Documentation**: [Python API →](api/python-api.md)

---

## Vertical Reference

Domain-specific assistants for specialized tasks.

| Vertical | Description | Usage |
|----------|-------------|-------|
| **Coding** | Software development | `victor --vertical coding` |
| **DevOps** | DevOps and infrastructure | `victor --vertical devops` |
| **RAG** | Retrieval-augmented generation | `victor --vertical rag` |
| **Data Analysis** | Data science workflows | `victor --vertical dataanalysis` |
| **Research** | Research and analysis | `victor --vertical research` |

[**Full Vertical Reference →**](verticals/)

---

## Quick Reference Cards

### Provider Switching

```bash
# Start with Claude
victor chat --provider anthropic --model claude-sonnet-4-20250514

# Continue with GPT-4 (context preserved)
/provider openai --model gpt-4

# Finish with local model (context preserved)
/provider ollama --model qwen2.5-coder:7b
```

### Mode Selection

```bash
# BUILD mode: Real file edits
victor --mode build "Refactor this function"

# PLAN mode: Analysis without edits
victor --mode plan "Analyze this code"

# EXPLORE mode: Understanding only
victor --mode explore "How does this work?"
```

### Workflow Execution

```bash
# List workflows
victor workflow list

# Run workflow
victor workflow run code-review

# Schedule workflow
victor workflow schedule code-review --cron "0 9 * * 1"
```

---

## Common Tasks

| Task | Command | Documentation |
|------|---------|--------------|
| **Code Review** | `victor "Review this PR"` | [User Guide →](../user-guide/) |
| **Refactoring** | `victor "Refactor to use DI"` | [User Guide →](../user-guide/) |
| **Testing** | `victor "Write unit tests"` | [User Guide →](../user-guide/) |
| **Git Operations** | `victor "Create commit for changes"` | [Tool Catalog →](tools/catalog.md) |
| **CI/CD** | See guides overview | [Guides →](../guides/index.md) |

---

## Additional Resources

- **Troubleshooting**: [Troubleshooting Guide →](../user-guide/troubleshooting.md)
- **Development**: [Development Guide →](../development/)
- **Architecture**: [Architecture Deep Dive →](../development/architecture/deep-dive.md)
- **Community**: [Support →](../README.md#community--support)

---

<div align="center">

**[← Back to Documentation](../README.md)**

**Reference Documentation**

*Complete technical reference for Victor*

</div>
