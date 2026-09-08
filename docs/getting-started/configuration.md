# Configuration Guide

This guide covers all configuration options for Victor, from basic setup to advanced customization.

## Configuration Overview

Victor uses a layered configuration system:

```
Priority (highest to lowest):
1. CLI flags              --provider anthropic --model claude-4
2. Environment variables  ANTHROPIC_API_KEY, VICTOR_LOG_LEVEL
3. Project context        .victor.md, CLAUDE.md
4. Profile config         ~/.victor/profiles.yaml
5. Global config          ~/.victor/config.yaml
6. Built-in defaults      Sensible defaults for all settings
```

---

## Quick Setup

### Minimal Configuration

**For local models** (no configuration needed):
```bash
# Victor auto-detects Ollama
victor chat
```

**For cloud providers**:
```bash
# Set API key
export ANTHROPIC_API_KEY=sk-ant-...

# Use Victor
victor chat --provider anthropic
```

**For persistent configuration** (recommended):
```bash
# Initialize Victor
victor init

# Edit profiles
nano ~/.victor/profiles.yaml
```

---

## Configuration Locations

### Directory Structure

```
~/.victor/                      # User configuration (global)
+-- profiles.yaml               # Provider/model profiles
+-- config.yaml                 # Global settings
+-- mcp.yaml                    # MCP server config
+-- api_keys.yaml               # API keys (use keyring instead)
+-- cache/                      # Cache storage
+-- logs/                       # Log files
+-- conversations/              # Saved conversation history

<your-project>/                 # Project-specific (local)
+-- .victor.md                  # Project context for Victor
+-- CLAUDE.md                   # AI instructions (also read by Victor)
+-- .victor/
    +-- init.md                 # Alternative project context location
```

### Finding Your Config

```bash
# Show current configuration
victor config show

# Show config file locations
ls -la ~/.victor/

# Show active profile
victor config profiles
```

---

## Environment Variables

### API Keys

Set API keys for cloud providers:

```bash
# Cloud providers
export ANTHROPIC_API_KEY=sk-ant-...      # Anthropic (Claude)
export OPENAI_API_KEY=sk-proj-...        # OpenAI (GPT-4)
export GOOGLE_API_KEY=...                # Google (Gemini)
export XAI_API_KEY=...                   # xAI (Grok)
export DEEPSEEK_API_KEY=...              # DeepSeek
export MISTRAL_API_KEY=...               # Mistral
export GROQ_API_KEY=...                  # Groq
export TOGETHER_API_KEY=...              # Together AI
export FIREWORKS_API_KEY=...             # Fireworks AI
export OPENROUTER_API_KEY=...            # OpenRouter
export CEREBRAS_API_KEY=...              # Cerebras
export MOONSHOT_API_KEY=...              # Moonshot

# Enterprise
export AZURE_OPENAI_API_KEY=...          # Azure OpenAI
export AZURE_OPENAI_ENDPOINT=https://... # Azure endpoint
export AWS_PROFILE=default               # AWS Bedrock
export GCP_PROJECT=...                   # Google Vertex AI
```

**Secure storage (recommended):**
```bash
# Store in system keyring (macOS Keychain, Windows Credential Manager, Linux Secret Service)
victor keys --set anthropic --keyring
victor keys --set openai --keyring

# List stored keys
victor keys --list
```

### Local Provider Hosts

Override default hosts for local providers:

```bash
export OLLAMA_HOST=127.0.0.1:11434      # Ollama (default)
export VICTOR_LM_STUDIO_HOST=127.0.0.1:1234   # LM Studio
export VICTOR_VLLM_HOST=127.0.0.1:8000        # vLLM
export VICTOR_LLAMACPP_HOST=127.0.0.1:8080    # llama.cpp
```

### Victor Settings

Configure Victor behavior:

```bash
export VICTOR_CONFIG_DIR=~/.victor       # Config directory
export VICTOR_PROFILE=development        # Default profile
export VICTOR_LOG_LEVEL=INFO             # Log level (DEBUG, INFO, WARNING, ERROR)
export VICTOR_LOG_FILE=/var/log/victor.log  # Log file path
export VICTOR_CACHE_ENABLED=true         # Enable caching
export VICTOR_TIMEOUT=30                 # Provider timeout (seconds)
```

### Shell Configuration

Add to your shell profile (`~/.bashrc`, `~/.zshrc`):

```bash
# Victor API Keys
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-proj-..."

# Victor defaults
export VICTOR_PROFILE="development"
export VICTOR_LOG_LEVEL="INFO"

# Optional: aliases
alias v="victor"
alias vc="victor chat"
```

---

## Profiles Configuration

Profiles let you save provider/model combinations for easy reuse.

### Basic Profile

**File**: `~/.victor/profiles.yaml`

```yaml
profiles:
  # Default profile (used when no --profile specified)
  default:
    provider: ollama
    model: qwen2.5-coder:7b

  # Cloud provider profile
  claude:
    provider: anthropic
    model: claude-sonnet-4-20250514
    temperature: 0.7
    max_tokens: 4096

  # Fast cloud profile
  fast:
    provider: groq
    model: llama-3.1-70b-versatile

  # Cost-effective profile
  cheap:
    provider: deepseek
    model: deepseek-coder

# Optional: set default profile
default_profile: default
```

### Using Profiles

```bash
# Use specific profile
victor --profile claude chat "Hello"

# Or set default profile
export VICTOR_PROFILE=claude
victor chat "Hello"

# List profiles
victor config profiles
```

### Profile Options

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `provider` | string | Provider name | `anthropic`, `openai`, `ollama` |
| `model` | string | Model name | `claude-sonnet-4-20250514` |
| `temperature` | float | Sampling temperature (0.0-2.0) | `0.7` |
| `max_tokens` | int | Maximum tokens to generate | `4096` |
| `top_p` | float | Nucleus sampling (0.0-1.0) | `0.9` |
| `top_k` | int | Top-k sampling | `40` |
| `api_key_env` | string | Environment variable for API key | `ANTHROPIC_API_KEY` |
| `base_url` | string | Custom API endpoint | `https://api.anthropic.com` |
| `timeout` | int | Request timeout (seconds) | `30` |

### System Prompt Policy Controls

Operators can now tune the fallback system prompt directly from `~/.victor/profiles.yaml` (or via `VICTOR_PROMPT_POLICY_*` environment variables) without editing code. Key fields:

| Field | Description |
|-------|-------------|
| `prompt_policy_identity` | Custom fallback identity text |
| `prompt_policy_guidelines` | Custom fallback guidelines block |
| `prompt_policy_operating_template` | Template for the operating preamble (`{stage}`, `{task_type}`, `{model}`, `{provider}`) |
| `prompt_policy_fallback_template` | Template when prompt assembly fails (`{task_type}`, `{message}`) |
| `prompt_policy_max_section_chars` | Character budget before low-priority sections are trimmed |
| `prompt_policy_protected_sections` | Sections that should never be trimmed |
| `prompt_policy_enforce_identity/guidelines/operating_preamble/unique_sections` | Booleans to toggle individual guards |

Example profile snippet:

```yaml
profiles:
  regulated:
    provider: anthropic
    model: claude-sonnet-4-20250514
    prompt_policy_identity: |
      You are Victor acting under the Regulated Banking SOP.
    prompt_policy_guidelines: |
      - Cite every data source.
      - Escalate if compliance steps are unclear.
    prompt_policy_protected_sections:
      - identity
      - guidelines
      - runbook
    prompt_policy_max_section_chars: 12000
```

To apply the same settings via environment variables:

```bash
export VICTOR_PROMPT_POLICY_IDENTITY="You are Victor running our regulated SOP."
export VICTOR_PROMPT_POLICY_MAX_SECTION_CHARS=12000
export VICTOR_PROMPT_POLICY_ENFORCE_IDENTITY=false  # Optional toggle
```

### Advanced Profile Examples

**Development profile with debugging:**
```yaml
profiles:
  development:
    provider: anthropic
    model: claude-sonnet-4-20250514
    temperature: 0.7
    max_tokens: 4096
```

**Production profile (more conservative):**
```yaml
profiles:
  production:
    provider: openai
    model: gpt-4o
    temperature: 0.3  # Lower for consistency
    max_tokens: 8192
```

**Local private profile:**
```yaml
profiles:
  private:
    provider: ollama
    model: qwen2.5-coder:14b
    base_url: http://127.0.0.1:11434
    # No API key needed
```

---

## Global Settings

### Basic Settings

**File**: `~/.victor/config.yaml`

```yaml
# Logging configuration
logging:
  level: INFO              # DEBUG, INFO, WARNING, ERROR, CRITICAL
  format: text             # text or json
  file: ~/.victor/logs/victor.log
  rotation: daily          # daily, weekly, or size
  retention: 30            # Days to keep logs

# Cache configuration
cache:
  enabled: true
  ttl: 3600                # Time-to-live in seconds
  max_size: 1000           # Maximum cached items
  backend: memory          # memory, disk, or redis

# Tool configuration
tools:
  lazy_load: true          # Load tools on-demand
  parallel_execution: true # Execute tools in parallel
  timeout: 30              # Tool execution timeout

# Conversation settings
conversation:
  max_history: 100         # Maximum messages in history
  save_history: true       # Save conversations to disk
  history_file: ~/.victor/conversations/

# Provider settings
providers:
  timeout: 30              # Request timeout
  retry_attempts: 3        # Retry on failure
  circuit_breaker:
    enabled: true          # Enable circuit breaker
    failure_threshold: 5   # Failures before opening
    recovery_timeout: 60   # Seconds before retry

# UI settings
ui:
  theme: dark              # dark or light
  syntax_highlighting: true
  line_numbers: true
```

---

## Project Context Files

Project context files teach Victor about your specific codebase.

### .victor.md

**Location**: `<project>/.victor.md`
**Purpose**: Project-specific context and instructions

```markdown
# Project Context

## Overview
This is a Django web application for e-commerce.

## Tech Stack
- Backend: Django 5.0, Python 3.11
- Frontend: React 18, TypeScript
- Database: PostgreSQL 15
- Cache: Redis 7

## Code Style
- Follow PEP 8 for Python
- Use type hints everywhere
- Max line length: 100 characters
- Google-style docstrings

## Testing
- Use pytest for all tests
- Minimum coverage: 80%
- Run `make test` before committing

## Important Directories
- `src/api/`: REST API endpoints
- `src/models/`: Database models
- `src/services/`: Business logic
- `tests/`: Test files

## Conventions
- All API endpoints use snake_case
- Database tables are singular (user, not users)
- Use environment variables for all secrets

## Victor Instructions
When modifying code:
1. Always add type hints
2. Update docstrings
3. Add or update tests
4. Run linter before finishing
```

### CLAUDE.md

**Location**: `<project>/CLAUDE.md`
**Purpose**: Instructions for AI assistants (Claude Code, Victor)

This file follows the same format as `.victor.md` and is automatically read by Victor.

### Loading Priority

Victor loads project context in this order (first found wins):
1. `.victor.md` in current directory
2. `CLAUDE.md` in current directory
3. `.victor/init.md` in current directory

---

## Modes Configuration

Victor supports three execution modes:

| Mode | File Edits | Use Case |
|------|------------|----------|
| **BUILD** | Yes | Implementation, refactoring, real changes |
| **PLAN** | Sandbox only | Analysis, planning, code review |
| **EXPLORE** | No | Understanding code, learning codebase |

### Using Modes

```bash
# Via CLI flag
victor chat --mode plan "Analyze this code"
victor chat --mode explore "How does auth work?"
victor chat --mode build "Implement feature X"

# Via in-chat command
/mode plan
/mode explore
/mode build
```

### Mode Budgets

Each mode has different tool exploration budgets:
- **BUILD**: 1x (default)
- **PLAN**: 2.5x (more exploration, sandbox edits)
- **EXPLORE**: 3x (maximum exploration, read-only)

---

## MCP Configuration

Configure Victor as an MCP server or connect to MCP servers.

### Victor as MCP Server

**File**: `~/.victor/mcp.yaml`

```yaml
server:
  host: 127.0.0.1
  port: 8080
  transport: stdio    # stdio or sse

  # Authentication (optional)
  auth:
    enabled: false
    api_key: your-api-key

# Expose specific tools
tools:
  expose_all: false
  allowed_tools:
    - read_file
    - write_file
    - search_files
    - run_tests

# Expose prompts
prompts:
  directory: ~/.victor/prompts/
  expose_prompts:
    - code_review
    - refactor
```

### Connecting to MCP Servers

```yaml
# ~/.victor/mcp.yaml
servers:
  - name: filesystem
    command: npx
    args: ["-y", "@anthropic/mcp-server-filesystem"]

  - name: github
    command: npx
    args: ["-y", "@anthropic/mcp-server-github"]
    env:
      GITHUB_TOKEN: ${GITHUB_TOKEN}
```

---

## Example Configurations

### Developer Setup

```yaml
# ~/.victor/profiles.yaml
profiles:
  default:
    provider: ollama
    model: qwen2.5-coder:7b

  claude:
    provider: anthropic
    model: claude-sonnet-4-20250514
    temperature: 0.7

  fast:
    provider: groq
    model: llama-3.1-70b-versatile

default_profile: default
```

```yaml
# ~/.victor/config.yaml
logging:
  level: INFO

cache:
  enabled: true
  backend: memory

tools:
  lazy_load: true
  parallel_execution: true
```

### Team Setup

```yaml
# ~/.victor/profiles.yaml
profiles:
  development:
    provider: anthropic
    model: claude-sonnet-4-20250514
    temperature: 0.7

  code-review:
    provider: anthropic
    model: claude-sonnet-4-20250514
    temperature: 0.3  # More consistent

  testing:
    provider: openai
    model: gpt-4o
```

```yaml
# ~/.victor/config.yaml
logging:
  level: INFO
  format: json
  file: /var/log/victor/team.log

cache:
  enabled: true
  backend: redis
  redis:
    host: redis.internal.example.com
    port: 6379
```

### Air-Gapped Setup

For secure, offline environments:

```yaml
# ~/.victor/profiles.yaml
profiles:
  default:
    provider: ollama
    model: qwen2.5-coder:14b
    base_url: http://localhost:11434
```

```yaml
# ~/.victor/config.yaml
airgapped_mode: true

# Only local providers allowed
providers:
  allowed:
    - ollama
    - lmstudio
    - vllm
    - llamacpp

# Disable web tools
tools:
  disabled:
    - web_search
    - fetch_url
```

---

## Validation and Testing

### Validate Configuration

```bash
# Validate all configuration files
victor config validate

# Check current configuration
victor config show

# Test a specific profile
victor --profile development chat "Test"
```

### Debug Configuration

```bash
# Enable debug logging
export VICTOR_LOG_LEVEL=DEBUG

# Run with debug output
victor --debug chat "Test"

# View configuration
victor config show

# Check logs
victor logs --tail 100
```

---

## Troubleshooting

### Configuration Not Loading

```bash
# Check file exists and has correct permissions
ls -la ~/.victor/profiles.yaml

# Validate YAML syntax
python -c "import yaml; yaml.safe_load(open('$HOME/.victor/profiles.yaml'))"

# Check for tab characters (should use spaces)
cat -A ~/.victor/profiles.yaml | grep '\t'
```

### Profile Not Found

```bash
# List available profiles
victor config profiles

# Check profile name in file
grep -A 5 "profiles:" ~/.victor/profiles.yaml
```

### API Key Not Found

```bash
# Check environment variable
echo $ANTHROPIC_API_KEY

# Check keyring
victor keys --list

# Set key
victor keys --set anthropic --keyring
```

---

## Best Practices

1. **Use profiles** for different contexts (development, testing, review)
2. **Store API keys in keyring**, not in config files
3. **Create project context** (`.victor.md`) for each project
4. **Use version control** for `.victor.md` (but not API keys)
5. **Start with PLAN mode** for unfamiliar codebases
6. **Use environment variables** for CI/CD

---

## Next Steps

- [User Guide](../user-guide/index.md) - Daily usage patterns
- [Provider Reference](../reference/providers/index.md) - All 24 providers
- [Tool Catalog](../reference/tools/catalog.md) - 34 tool modules
- [Workflow Guide](../guides/workflow-development/dsl.md) - Automation

---

**Need help?** See [Troubleshooting](../user-guide/troubleshooting.md) or [open an issue](https://github.com/anvai-labs/victor/issues).
