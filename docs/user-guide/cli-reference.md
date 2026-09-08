# CLI Command Reference

Complete reference for all Victor CLI commands.

## Quick Reference

| Command | Description |
|---------|-------------|
| `victor` | Start interactive CLI chat mode |
| `victor chat` | Start CLI chat mode |
| `victor init` | Initialize Victor in current directory |
| `victor serve` | Start HTTP API server |
| `victor mcp` | Run as MCP server |
| `victor keys` | Manage API keys |
| `victor providers list` | List available providers |
| `victor capabilities` | Inspect shared tools, teams, workflows, and recommendations |
| `victor profiles` | Manage provider profiles |
| `victor tools` | Manage and list tools |
| `victor models` | Manage Ollama models |
| `victor config` | View/edit configuration |
| `victor rag` | RAG operations |
| `victor benchmark` | Run benchmarks |
| `victor workflow` | Workflow management |
| `victor scheduler` | Workflow scheduling |
| `victor sessions` | Session management |
| `victor vertical` | Vertical package management |

---

## Core Commands

### victor (default)

Start Victor in interactive CLI chat mode.

```bash
victor                              # CLI mode with default profile
victor --profile local              # CLI with specific profile
victor --provider anthropic         # CLI with specific provider
victor --mode plan                  # CLI in planning mode
```

**Options:**

| Option | Description |
|--------|-------------|
| `--profile, -p` | Use a saved profile |
| `--provider` | Override provider |
| `--model, -m` | Override model |
| `--mode` | Agent mode: build, plan, review, delegate, explore |
| `--airgapped` | Enable air-gapped mode (local only) |
| `--version` | Show version |
| `--help` | Show help |

---

### victor chat

Start an interactive chat session in CLI mode.

```bash
victor chat                         # Default provider/model
victor chat --provider anthropic --model claude-sonnet-4-5
victor chat --stream                # Enable streaming (default)
victor chat --mode plan             # Start in planning mode
```

**Options:**

| Option | Description |
|--------|-------------|
| `--profile, -p` | Use a saved profile |
| `--provider` | LLM provider (anthropic, openai, ollama, etc.) |
| `--model, -m` | Model name |
| `--stream/--no-stream` | Enable/disable streaming |
| `--mode` | Agent mode: build, plan, review, delegate, explore |
| `--tool-budget` | Override tool call budget |
| `--resume, -r` | Resume session by ID |
| `--log-level` | Set logging level |

**Examples:**

```bash
# Start with Claude
victor chat --provider anthropic

# Local model with Ollama
victor chat --provider ollama --model qwen2.5-coder:7b

# Planning mode for research
victor chat --mode plan

# Resume a previous session
victor chat --resume myproj-9Kx7Z2
```

---

### victor init

Initialize Victor in the current directory. Creates `.victor/` directory with configuration files.

```bash
victor init                         # Interactive setup
victor init --force                 # Overwrite existing config
victor init --provider anthropic    # Pre-configure provider
victor init --model claude-3-5-sonnet
victor init --index                 # Build codebase index
```

**Options:**

| Option | Description |
|--------|-------------|
| `--force, -f` | Overwrite existing configuration |
| `--provider` | Pre-configure default provider |
| `--model` | Pre-configure default model |
| `--index/--no-index` | Build/skip codebase index |
| `--skip-git` | Skip git repository checks |
| `--quiet, -q` | Minimal output |

**Created Files:**

- `.victor/` - Configuration directory
- `.victor/init.md` - Project context file
- `.victor/config.yaml` - Local configuration

---

### victor serve

Start Victor as an HTTP API server.

```bash
victor serve                        # Default port 8765
victor serve --port 8080            # Custom port
victor serve --host 0.0.0.0         # Listen on all interfaces
victor serve --reload               # Auto-reload on changes
```

**Options:**

| Option | Description |
|--------|-------------|
| `--port` | Port number (default: 8765) |
| `--host` | Host address (default: 127.0.0.1) |
| `--reload` | Enable auto-reload for development |
| `--workers` | Number of worker processes |
| `--log-level` | Logging level |

**API Endpoints:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/chat` | POST | Send a message |
| `/capabilities` | GET | List available tools |
| `/capabilities/recommend` | GET | Recommend teams/workflows from shared catalogs |
| `/health` | GET | Health check |
| `/providers` | GET | List providers |

---

### victor capabilities

Inspect shared capability catalogs and ask the framework recommendation engine for
the best team/workflow matches for a task.

```bash
victor capabilities
victor capabilities --vertical coding
victor capabilities recommend feature high
victor capabilities recommend feature high --mode build --vertical coding --json
```

**Examples:**

```bash
# Show all discovered capability catalogs
victor capabilities

# Restrict inspection to one vertical
victor capabilities --vertical coding

# Ask for framework-backed coordination suggestions
victor capabilities recommend feature high --mode build
```

---

### victor mcp

Run Victor as an MCP (Model Context Protocol) server.

```bash
victor mcp                          # Run in stdio mode
victor mcp --stdio                  # Explicit stdio mode
victor mcp --log-level DEBUG        # Verbose logging
```

**Options:**

| Option | Description |
|--------|-------------|
| `--stdio/--no-stdio` | Run in stdio mode (default: true) |
| `--log-level, -l` | Set logging level |

**Usage with MCP Clients:**

Add to Claude Desktop `config.json`:
```json
{
  "mcpServers": {
    "victor": {
      "command": "victor",
      "args": ["mcp"]
    }
  }
}
```

---

## API Key Management

### victor keys

Manage API keys for cloud providers and services.

```bash
victor keys                         # List configured providers
victor keys --setup                 # Create template file
victor keys --set anthropic         # Set key (file storage)
victor keys --set anthropic --keyring  # Set key (secure keyring)
victor keys --migrate               # Migrate file to keyring
victor keys --services              # List external services
victor keys --set-service finnhub --keyring
```

**Options:**

| Option | Description |
|--------|-------------|
| `--setup, -s` | Create API keys template file |
| `--list, -l` | List configured providers |
| `--set <provider>` | Set API key for provider |
| `--set-service <svc>` | Set key for external service |
| `--keyring, -k` | Store in system keyring |
| `--migrate` | Migrate file keys to keyring |
| `--delete-keyring <prov>` | Remove provider from keyring |
| `--services` | List configured services |

**Supported Providers:**

- `anthropic` - Claude models
- `openai` - GPT models
- `google` - Gemini models
- `xai` - Grok models
- `deepseek` - DeepSeek models
- `mistral` - Mistral models
- `groqcloud` - Groq LPU
- `together` - Together AI
- `openrouter` - OpenRouter gateway
- `fireworks` - Fireworks AI
- And more...

**External Services:**

- `finnhub` - Stock data
- `fred` - Federal Reserve data
- `alphavantage` - Market data
- `newsapi` - News aggregation

---

## Provider Management

### victor providers list

List all available LLM providers.

```bash
victor providers list
```

Displays providers with status, features, and aliases.

---

### victor profiles

Manage saved provider/model profiles.

```bash
victor profiles                     # List profiles
victor profiles list                # Same as above
victor profiles show default        # Show profile details
victor profiles create fast --provider groqcloud --model llama3-70b
victor profiles delete old-profile
victor profiles set-default local
```

`victor profile ...` remains as a compatibility alias, but new scripts and
documentation should use `victor profiles ...`.

**Subcommands:**

| Subcommand | Description |
|------------|-------------|
| `list` | List all profiles |
| `show <name>` | Show profile details |
| `create <name>` | Create new profile |
| `delete <name>` | Delete profile |
| `set-default <name>` | Set default profile |

**Profile Configuration:**

Profiles are stored in `~/.victor/profiles.yaml`:

```yaml
profiles:
  default:
    provider: anthropic
    model: claude-sonnet-4-5
  local:
    provider: ollama
    model: qwen2.5-coder:7b
  fast:
    provider: groqcloud
    model: llama-3.1-70b-versatile
```

---

## Model Management

### victor models

Manage Ollama models.

```bash
victor models                       # List local models
victor models list                  # Same as above
victor models pull qwen2.5-coder:7b # Download model
victor models remove old-model      # Remove model
victor models info qwen2.5-coder:7b # Show model details
```

**Subcommands:**

| Subcommand | Description |
|------------|-------------|
| `list` | List downloaded models |
| `pull <model>` | Download a model |
| `remove <model>` | Delete a model |
| `info <model>` | Show model information |

---

## Tool Management

### victor tools

Manage and inspect available tools.

```bash
victor tools                        # List all tools
victor tools list                   # Same as above
victor tools list --enabled         # Only enabled tools
victor tools info read_file         # Tool details
victor tools enable tool-name       # Enable a tool
victor tools disable tool-name      # Disable a tool
```

**Subcommands:**

| Subcommand | Description |
|------------|-------------|
| `list` | List available tools |
| `info <tool>` | Show tool details |
| `enable <tool>` | Enable a disabled tool |
| `disable <tool>` | Disable a tool |

---

## Configuration

### victor config

View and edit configuration.

```bash
victor config                       # Show all config
victor config show                  # Same as above
victor config get default_provider  # Get specific value
victor config set default_provider anthropic
victor config edit                  # Open in editor
victor config path                  # Show config file path
```

**Subcommands:**

| Subcommand | Description |
|------------|-------------|
| `show` | Display current configuration |
| `get <key>` | Get a config value |
| `set <key> <value>` | Set a config value |
| `edit` | Open config in editor |
| `path` | Show config file path |
| `reset` | Reset to defaults |

**Configuration Options:**

| Key | Description |
|-----|-------------|
| `default_provider` | Default LLM provider |
| `default_model` | Default model |
| `tool_call_budget` | Max tool calls per turn |
| `airgapped_mode` | Enable air-gapped mode |
| `use_semantic_tool_selection` | Enable semantic tool selection |
| `unified_embedding_model` | Embedding model for RAG |

---

## RAG Operations

### victor rag

Retrieval-Augmented Generation operations.

```bash
# Ingest documents
victor rag ingest ./docs --recursive --pattern "*.md"
victor rag ingest https://example.com/docs
victor rag ingest ./README.md --type markdown

# Search and query
victor rag search "authentication"
victor rag search "error handling" --top-k 10
victor rag query "How do I add a provider?" --synthesize

# Management
victor rag list                     # List documents
victor rag stats                    # Show statistics
victor rag delete doc_abc123        # Delete document

# Demos
victor rag demo docs                # Ingest project docs
victor rag demo-sec --preset faang  # SEC filings demo
```

**Subcommands:**

| Subcommand | Description |
|------------|-------------|
| `ingest <source>` | Ingest file, directory, or URL |
| `search <query>` | Search knowledge base |
| `query <question>` | Query with optional synthesis |
| `list` | List all documents |
| `stats` | Show knowledge base stats |
| `delete <id>` | Delete a document |
| `demo` | Run documentation demo |
| `demo-sec` | SEC filings demo |

**Ingest Options:**

| Option | Description |
|--------|-------------|
| `--recursive, -r` | Recursively ingest directory |
| `--pattern, -p` | Glob pattern (e.g., "*.md") |
| `--type, -t` | Document type: auto, text, markdown, code, pdf, html |
| `--id` | Custom document ID |

**Query Options:**

| Option | Description |
|--------|-------------|
| `--synthesize, -S` | Use LLM to synthesize answer |
| `--provider, -p` | LLM provider for synthesis |
| `--model, -m` | Model for synthesis |
| `--top-k, -k` | Number of results |

---

## Benchmark Operations

### victor benchmark

Run AI coding benchmarks.

```bash
# List and setup
victor benchmark list               # List available benchmarks
victor benchmark setup swe-bench --max-tasks 10

# Run benchmarks
victor benchmark run swe-bench --max-tasks 10 --profile default
victor benchmark run humaneval --model claude-3-sonnet
victor benchmark run mbpp --output results.json

# Compare and leaderboard
victor benchmark compare --benchmark swe-bench
victor benchmark leaderboard --benchmark swe-bench
victor benchmark capabilities       # Framework comparison
```

**Subcommands:**

| Subcommand | Description |
|------------|-------------|
| `list` | List available benchmarks |
| `setup <bench>` | Setup benchmark repos |
| `run <bench>` | Run a benchmark |
| `compare` | Compare with other frameworks |
| `leaderboard` | Show benchmark leaderboard |
| `capabilities` | Compare framework capabilities |

**Available Benchmarks:**

| Benchmark | Description |
|-----------|-------------|
| `swe-bench` | Real-world GitHub issue resolution |
| `swe-bench-lite` | Curated SWE-bench subset |
| `humaneval` | Code generation from docstrings |
| `mbpp` | Mostly Basic Python Problems |

**Run Options:**

| Option | Description |
|--------|-------------|
| `--max-tasks, -n` | Maximum tasks to run |
| `--model, -m` | Model to use |
| `--profile, -p` | Victor profile |
| `--output, -o` | Output file (JSON) |
| `--timeout, -t` | Timeout per task (seconds) |
| `--max-turns` | Max conversation turns |
| `--parallel` | Parallel tasks |
| `--resume, -r` | Resume from checkpoint |

---

## Workflow Management

### victor workflow

Manage YAML workflows.

```bash
victor workflow validate path/to/workflow.yaml
victor workflow render workflow.yaml --format ascii
victor workflow list                # List workflows
victor workflow run my_workflow     # Execute workflow
```

**Subcommands:**

| Subcommand | Description |
|------------|-------------|
| `validate <path>` | Validate workflow YAML |
| `render <path>` | Render workflow graph |
| `list` | List available workflows |
| `run <name>` | Execute a workflow |

**Render Formats:**

- `ascii` - ASCII art diagram
- `mermaid` - Mermaid diagram
- `json` - JSON structure

---

### victor scheduler

Manage workflow scheduling.

```bash
# Service management
victor scheduler start              # Start in foreground
victor scheduler start --daemon     # Start as background daemon
victor scheduler stop               # Stop daemon
victor scheduler status             # Show status

# Schedule management
victor scheduler list               # List schedules
victor scheduler add daily_check --cron "0 9 * * *" --yaml workflow.yaml
victor scheduler remove schedule_id
victor scheduler history            # Show execution history

# Installation
victor scheduler install            # Generate systemd service
```

**Subcommands:**

| Subcommand | Description |
|------------|-------------|
| `start` | Start scheduler service |
| `stop` | Stop scheduler daemon |
| `status` | Show scheduler status |
| `list` | List scheduled workflows |
| `add` | Schedule a workflow |
| `remove` | Remove a schedule |
| `history` | Show execution history |
| `install` | Generate systemd service file |

**Cron Expression Examples:**

| Expression | Description |
|------------|-------------|
| `0 9 * * *` | Daily at 9 AM |
| `0 */2 * * *` | Every 2 hours |
| `0 0 * * MON` | Every Monday at midnight |
| `@hourly` | Every hour |
| `@daily` | Every day at midnight |

---

## Session Management

### victor sessions

Manage conversation sessions.

```bash
victor sessions list                # List sessions
victor sessions list --all          # List all sessions
victor sessions show session_id     # Show session details
victor sessions search "keyword"    # Search sessions
victor sessions delete session_id   # Delete session
victor sessions export              # Export to JSON
victor sessions clear               # Clear all sessions
```

**Subcommands:**

| Subcommand | Description |
|------------|-------------|
| `list` | List saved sessions |
| `show <id>` | Show session details |
| `search <query>` | Search sessions |
| `delete <id>` | Delete a session |
| `export` | Export sessions to JSON |
| `clear` | Clear all sessions |

**Options:**

| Option | Description |
|--------|-------------|
| `--limit, -n` | Maximum results |
| `--all` | Show all (no limit) |
| `--json` | Output as JSON |

---

## Vertical Management

### victor vertical

Manage vertical packages (plugins).

```bash
# Installation
victor vertical install victor-security
victor vertical install "victor-security>=1.0.0"
victor vertical install git+https://github.com/user/victor-security.git
victor vertical install ./local/path

# Management
victor vertical list                # List all verticals
victor vertical list --source installed
victor vertical search security     # Search verticals
victor vertical info security       # Show details
victor vertical uninstall victor-security

# Creation
victor vertical create security --description "Security analysis"
```

**Subcommands:**

| Subcommand | Description |
|------------|-------------|
| `install <pkg>` | Install a vertical |
| `uninstall <name>` | Uninstall a vertical |
| `list` | List verticals |
| `search <query>` | Search verticals |
| `info <name>` | Show vertical info |
| `create <name>` | Create a new contract-first vertical scaffold |

**List Filters:**

| Option | Description |
|--------|-------------|
| `--source` | Filter: all, installed, builtin, available |
| `--category, -c` | Filter by category |
| `--tags, -t` | Filter by tags |
| `--verbose, -v` | Show detailed info |

---

## Global Options

These options are available for most commands:

| Option | Description |
|--------|-------------|
| `--help` | Show help message |
| `--version` | Show version |
| `--log-level` | Set logging level (DEBUG, INFO, WARNING, ERROR) |
| `--debug` | Enable debug mode |
| `--quiet, -q` | Minimal output |

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `VICTOR_PROFILE` | Default profile name |
| `VICTOR_PROVIDER` | Default provider |
| `VICTOR_MODEL` | Default model |
| `VICTOR_LOG_LEVEL` | Logging level |
| `VICTOR_CONFIG_DIR` | Config directory override |
| `VICTOR_AIRGAPPED` | Enable air-gapped mode |
| `VICTOR_SKIP_ENV_FILE` | Skip .env file loading |

**Provider API Keys:**

| Variable | Provider |
|----------|----------|
| `ANTHROPIC_API_KEY` | Anthropic (Claude) |
| `OPENAI_API_KEY` | OpenAI |
| `GOOGLE_API_KEY` | Google (Gemini) |
| `XAI_API_KEY` | xAI (Grok) |
| `DEEPSEEK_API_KEY` | DeepSeek |
| `GROQ_API_KEY` | Groq |
| `TOGETHER_API_KEY` | Together AI |
| `OPENROUTER_API_KEY` | OpenRouter |

---

## See Also

- [CLI Mode Guide](./cli-reference.md) - Interactive prompt-toolkit CLI interface
- [Tool Catalog](../reference/tools/catalog.md) - Available tools
- [Provider Reference](../reference/providers/index.md) - Provider details
- [Workflow DSL](../guides/workflow-development/dsl.md) - Workflow syntax
