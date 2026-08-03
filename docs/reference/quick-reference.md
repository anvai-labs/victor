# Victor Quick Reference

Essential commands, options, and configurations for the Victor agentic AI framework.

---

## Essential Commands

| Command | Description |
|---------|-------------|
| `pipx install victor-ai` | Install Victor (recommended) |
| `pip install victor-ai` | Install in virtual environment |
| `victor init` | Initialize Victor in current directory |
| `victor` | Start interactive CLI terminal mode |
| `victor chat "Your message"` | Start CLI chat mode |
| `victor chat --provider anthropic` | Use specific provider |
| `victor chat --provider openai --model gpt-4o` | Use specific model |
| `victor chat --mode plan` | Start in planning mode |
| `victor chat --mode explore` | Start in exploration mode |
| `victor serve --port 8080` | Start HTTP API server |
| `victor workflow validate path/to/workflow.yaml` | Validate YAML workflow |
| `victor workflow run my_workflow` | Execute a workflow |
| `victor keys --set anthropic --keyring` | Store API key securely |
| `victor profiles list` | List available profiles |
| `victor tools list` | List available tools |

---

## Common Options

| Option | Description | Example |
|--------|-------------|---------|
| `--provider, -p` | LLM provider | `--provider anthropic` |
| `--model, -m` | Model name | `--model claude-sonnet-4-5` |
| `--profile` | Saved profile | `--profile local` |
| `--mode` | Agent mode | `--mode plan` |
| `--airgapped` | Local providers only | `--airgapped` |
| `--stream/--no-stream` | Enable/disable streaming | `--no-stream` |
| `--tool-budget` | Max tool calls | `--tool-budget 20` |
| `--resume, -r` | Resume session | `--resume session-id` |
| `--log-level` | Logging verbosity | `--log-level DEBUG` |
| `--help` | Show help | `victor chat --help` |

### Agent Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `build` | Full edits allowed (default) | Active development |
| `plan` | 2.5x exploration, sandbox only | Research & analysis |
| `explore` | 3.0x exploration, no edits | Codebase understanding |

---

## Key Environment Variables

### API Keys

| Variable | Provider |
|----------|----------|
| `ANTHROPIC_API_KEY` | Anthropic (Claude) |
| `OPENAI_API_KEY` | OpenAI (GPT) |
| `GOOGLE_API_KEY` | Google (Gemini) |
| `XAI_API_KEY` | xAI (Grok) |
| `DEEPSEEK_API_KEY` | DeepSeek |
| `MISTRAL_API_KEY` | Mistral AI |
| `GROQ_API_KEY` | Groq Cloud |
| `TOGETHER_API_KEY` | Together AI |
| `OPENROUTER_API_KEY` | OpenRouter |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI |

### Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `VICTOR_DEFAULT_PROVIDER` | `ollama` | Default LLM provider |
| `VICTOR_DEFAULT_MODEL` | *auto* | Default model |
| `VICTOR_LOG_LEVEL` | `INFO` | Console logging |
| `VICTOR_PROFILE` | `default` | Default profile |
| `VICTOR_AIRGAPPED` | `unset` | Enable air-gapped mode |
| `VICTOR_SKIP_ENV_FILE` | `unset` | Skip .env loading |
| `OLLAMA_BASE_URL` | `localhost:11434` | Ollama server |
| `LMSTUDIO_BASE_URLS` | `127.0.0.1:1234` | LMStudio URLs |

---

## Quick Python API

### Basic Usage

```python
from victor.framework import Agent, EventType, ToolSet

# Create agent
agent = await Agent.create(provider="anthropic")

# Simple query
result = await agent.run("Explain this code")
print(result.content)

# Multi-turn conversation
session = agent.chat()
await session.send("What files are in src/?")
await session.send("Summarize the main module")
```

### Streaming

```python
from victor.framework import Agent

agent = await Agent.create(provider="openai", model="gpt-4o")

# Stream responses
async for event in agent.stream("Refactor this function"):
    if event.type == EventType.CONTENT:
        print(event.content, end="")
    elif event.type == EventType.TOOL_CALL:
        print(f"\n[Tool: {event.tool_name}]")
```

### With Tools

```python
from victor.framework import Agent, ToolSet

agent = await Agent.create(
    provider="anthropic",
    model="claude-sonnet-4-5",
    tools=ToolSet.default()  # or ToolSet.minimal(), ToolSet.full()
)

result = await agent.run("Analyze and test this code")
```

### Orchestrator API

```python
from victor.agent.orchestrator import AgentOrchestrator
from victor.config.settings import Settings

config = Settings(
    default_provider="anthropic",
    default_model="claude-sonnet-4-5"
)

orchestrator = AgentOrchestrator(config=config)
response = await orchestrator.process_message("Hello, Victor!")
```

### StateGraph Workflows

```python
from victor.framework import StateGraph, END
from typing import TypedDict

class WorkflowState(TypedDict):
    query: str
    research: str
    draft: str

graph = StateGraph(WorkflowState)
graph.add_node("research", research_fn)
graph.add_node("draft", draft_fn)
graph.add_edge("research", "draft")
graph.add_edge("draft", END)

compiled = graph.compile()
result = await compiled.invoke({"query": "AI trends"})
```

---

## Profile Locations

### Linux / macOS

```
~/.victor/
├── profiles.yaml       # Provider/model profiles
├── config.yaml         # Global settings
├── mcp.yaml            # MCP server config
├── api_keys.yaml       # API keys (use keyring instead)
├── cache/              # Cache storage
├── logs/               # Log files
└── conversations/      # Chat history
```

### Windows

```
C:\Users\<username>\.victor\
├── profiles.yaml
├── config.yaml
├── mcp.yaml
├── api_keys.yaml
├── cache\
├── logs\
└── conversations\
```

### Project Context

```
<project-root>/
├── .victor.md          # Project context (Victor reads this)
├── CLAUDE.md           # Also recognized
└── .victor/
    └── init.md         # Alternative context location
```

---

## Common Workflows

### Switch Providers Mid-Conversation

```
You: /provider openai --model gpt-4o
You: /provider anthropic
You: /provider ollama --model qwen2.5-coder:7b
```

### Run Workflow

```bash
# Validate workflow
victor workflow validate workflows/my_workflow.yaml

# Execute workflow
victor workflow run my_workflow

# Render workflow graph
victor workflow render workflows/my_workflow.yaml --format mermaid
```

### RAG Operations

```bash
# Ingest documentation
victor rag ingest ./docs --recursive --pattern "*.md"

# Search knowledge base
victor rag search "authentication" --top-k 5

# Query with LLM synthesis
victor rag query "How do I add a provider?" --synthesize
```

### Benchmark

```bash
# Run SWE-bench
victor benchmark run swe-bench --max-tasks 10

# Run HumanEval
victor benchmark run humaneval --profile default
```

---

## Provider Quick Setup

### Local (No API Key)

```bash
# Ollama (recommended for beginners)
brew install ollama  # macOS
curl -fsSL https://ollama.com/install.sh | sh  # Linux
ollama pull qwen2.5-coder:7b
victor chat

# LM Studio (GUI)
# Download from https://lmstudio.ai
# Enable local server in settings
victor chat --provider lmstudio
```

### Cloud (API Key Required)

```bash
# Anthropic (Claude)
export ANTHROPIC_API_KEY=sk-ant-...
victor chat --provider anthropic

# OpenAI (GPT)
export OPENAI_API_KEY=sk-proj-...
victor chat --provider openai --model gpt-4o

# Google (Gemini)
export GOOGLE_API_KEY=...
victor chat --provider google --model gemini-2.0-flash
```

### Secure Key Storage (Recommended)

```bash
# Store in system keyring
victor keys --set anthropic --keyring
victor keys --set openai --keyring

# Use without environment variables
victor chat --provider anthropic
```

---

## Slash Commands (In-Chat)

| Command | Description |
|---------|-------------|
| `/provider <name>` | Switch LLM provider |
| `/mode <mode>` | Change agent mode |
| `/profile <name>` | Load profile |
| `/prompt-optimize` | Run/inspect prompt evolution (`--status`, `--pareto`, `--tier`) — see the [Prompt Evolution Workflow](../development/prompt-evolution-workflow.md) |
| `/clear` | Clear conversation |
| `/help` | Show help |
| `/exit` | Exit session |

---

## Useful Links

| Resource | URL |
|----------|-----|
| **Full Documentation** | https://victor-ai.readthedocs.io |
| **GitHub Repository** | https://github.com/anvai-labs/victor |
| **PyPI Package** | https://pypi.org/project/victor-ai |
| **Docker Image** | https://ghcr.io/vjsingh1984/victor |
| **Issue Tracker** | https://github.com/anvai-labs/victor/issues |
| **Discussions** | https://github.com/anvai-labs/victor/discussions |

---

## Quick Troubleshooting

| Problem | Solution |
|---------|----------|
| `command not found: victor` | Run `pipx ensurepath` and restart terminal |
| `Could not build wheels` | Install build tools: `xcode-select --install` (macOS) |
| Ollama connection error | Run `ollama serve` before using Victor |
| API key not found | Set environment variable or use `victor keys --set` |
| Permission denied | Never use `sudo pip` - use pipx or virtual environment |

---

## Version & Info

```bash
victor --version              # Show version
victor providers list         # List all providers
victor tools list             # List all tools
victor profiles list          # List profiles
victor config show            # Show configuration
victor config path            # Show config file location
```

---

**Need more?** See [CLI Command Reference](./cli-commands.md) | [Configuration Guide](../getting-started/configuration.md) | [Environment Variables](./environment-variables.md)
