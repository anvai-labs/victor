# Configuration Reference

Complete reference for configuring Victor's behavior and integrations.

## Quick Links

| File | Purpose | Location | Reference |
|------|---------|----------|----------|
| **profiles.yaml** | Provider and model profiles | `~/.victor/profiles.yaml` | [#profilesyaml →](#profilesyaml) |
| **config.yaml** | Global settings and options | `~/.victor/config.yaml` | [#configyaml →](#configyaml) |
| **mcp.yaml** | MCP server configuration | `~/.victor/mcp.yaml` | [#mcpyaml →](#mcpyaml) |
| **.victor.md** | Project context | `<project>/.victor.md` | [.victor.md →](../../user-guide/index.md#5-project-context) |
| **CLAUDE.md** | AI instructions | `<project>/CLAUDE.md` | [Configuration architecture →](../../architecture.md#configuration-system) |

## Configuration Directory Structure

```
~/.victor/                      # User configuration (global)
├── profiles.yaml                # Provider/model profiles
├── config.yaml                  # Global settings
├── mcp.yaml                     # MCP server config
├── cache/                       # Cache storage
├── logs/                        # Log files
└── conversations/               # Conversation history

<project>/                       # Project-specific (local)
├── .victor.md                   # Project context
└── CLAUDE.md                    # AI instructions
```

## Configuration Priority

Victor loads configuration in this order (later overrides earlier):

1. **Default values** (built-in)
2. **Global config** (`~/.victor/config.yaml`)
3. **Profile config** (`~/.victor/profiles.yaml`)
4. **Project context** (`<project>/.victor.md`)
5. **Environment variables**
6. **CLI flags** (highest priority)

**Example**:
```bash
# Default: temperature=0.7
# config.yaml: temperature=0.5
# profiles.yaml: temperature=0.3
# Environment: VICTOR_TEMPERATURE=0.1
# CLI flag: --temperature 0.0
# Result: 0.0 (CLI flag wins)
```

## Quick Configuration

### Minimal Setup

**1. Local model** (no configuration needed):
```bash
# Victor auto-detects Ollama
victor chat "Hello!"
```

**2. Cloud provider** (environment variable):
```bash
# Set API key
export ANTHROPIC_API_KEY=sk-ant-...

# Use Victor
victor chat --provider anthropic "Hello!"
```

**3. Profile-based** (recommended):
```yaml
# ~/.victor/profiles.yaml
profiles:
  default:
    provider: anthropic
    model: claude-sonnet-4-20250514
```

```bash
# Use profile
victor chat "Hello!"  # Automatically uses profile
```

---

## profiles.yaml

Provider and model profile configuration.

**Location**: `~/.victor/profiles.yaml`
**Required**: No (defaults will be used)
**Purpose**: Define reusable provider/model configurations

### Basic Structure

```yaml
profiles:
  # Profile name (can be anything)
  default:
    # Provider selection
    provider: anthropic  # openai, google, ollama, etc.

    # Model selection
    model: claude-sonnet-4-20250514

    # Optional: Model parameters
    temperature: 0.7
    max_tokens: 4096
    top_p: 0.9
    top_k: 40

    # Optional: API key
    api_key_env: ANTHROPIC_API_KEY

    # Optional: API base URL (for custom endpoints)
    base_url: https://api.anthropic.com

    # Optional: Other settings
    timeout: 30
    retry_attempts: 3
```

### Profile Examples

**Development Profile**:
```yaml
profiles:
  development:
    provider: anthropic
    model: claude-sonnet-4-20250514
    temperature: 0.7
    max_tokens: 4096
    api_key_env: ANTHROPIC_API_KEY
```

**Production Profile**:
```yaml
profiles:
  production:
    provider: azure
    deployment: gpt-4o
    api_key_env: AZURE_OPENAI_API_KEY
    api_base_env: AZURE_OPENAI_ENDPOINT
    api_version: 2024-02-01
    max_tokens: 8192
    temperature: 0.3  # More deterministic
```

**Local Profile**:
```yaml
profiles:
  local:
    provider: ollama
    model: qwen2.5-coder:7b
    # No API key needed
    base_url: http://127.0.0.1:11434
```

**Cost-Optimized Profile**:
```yaml
profiles:
  cheap:
    provider: google
    model: gemini-2.0-flash-exp  # Free
    max_tokens: 4096
```

**Multi-Provider Profile** (with fallback):
```yaml
profiles:
  robust:
    provider: anthropic
    model: claude-sonnet-4-20250514
    api_key_env: ANTHROPIC_API_KEY
    # Fallback providers (if primary fails)
    fallback_providers:
      - provider: openai
        model: gpt-4o
      - provider: ollama
        model: qwen2.5-coder:7b
```

### Using Profiles

**Specify profile**:
```bash
victor --profile development chat "Hello"
```

**Set default profile**:
```yaml
# ~/.victor/profiles.yaml
default_profile: development
```

**List available profiles**:
```bash
victor config profiles
```

**Show current profile**:
```bash
victor config show
```

### Profile Reference

| Field | Type | Required | Description | Example |
|-------|------|----------|-------------|---------|
| **provider** | string | Yes | Provider name | `anthropic`, `openai`, `ollama` |
| **model** | string | Yes | Model name | `claude-sonnet-4-20250514` |
| **api_key_env** | string | No | Environment variable for API key | `ANTHROPIC_API_KEY` |
| **base_url** | string | No | Custom API base URL | `https://api.anthropic.com` |
| **temperature** | float | No | Sampling temperature (0.0-1.0) | `0.7` |
| **max_tokens** | int | No | Maximum tokens to generate | `4096` |
| **top_p** | float | No | Nucleus sampling (0.0-1.0) | `0.9` |
| **top_k** | int | No | Top-k sampling | `40` |
| **timeout** | int | No | Request timeout (seconds) | `30` |
| **retry_attempts** | int | No | Number of retries on failure | `3` |

[Full profiles.yaml Reference →](#profilesyaml)

---

## config.yaml

Global configuration for Victor's behavior.

**Location**: `~/.victor/config.yaml`
**Required**: No (defaults will be used)
**Purpose**: Global settings, caching, logging, etc.

### Basic Structure

```yaml
# Logging configuration
logging:
  level: INFO  # DEBUG, INFO, WARNING, ERROR, CRITICAL
  format: text  # text or json
  file: /var/log/victor.log
  rotation: daily
  retention: 30

# Cache configuration
cache:
  enabled: true
  ttl: 3600  # Time-to-live in seconds
  max_size: 1000  # Maximum number of cached items
  backend: memory  # memory, redis, disk

# Tool configuration
tools:
  lazy_load: true  # Load tools on-demand
  parallel_execution: true
  timeout: 30

# Conversation configuration
conversation:
  max_history: 100  # Maximum messages in history
  save_history: true
  history_file: ~/.victor/conversations/

# Provider configuration
providers:
  timeout: 30
  retry_attempts: 3
  circuit_breaker:
    enabled: true
    failure_threshold: 5
    recovery_timeout: 60

# Workflow configuration
workflows:
  directory: ~/.victor/workflows/
  auto_compile: true

# UI configuration
ui:
  theme: dark  # dark, light
  syntax_highlighting: true
  line_numbers: true
```

### Configuration Reference

#### Logging

```yaml
logging:
  level: INFO              # Log level
  format: text             # Output format (text/json)
  file: /var/log/victor.log # Log file path
  rotation: daily          # Log rotation (daily, weekly, size)
  retention: 30            # Days to keep logs
  console: true            # Also log to console
```

**Log Levels**:
- **DEBUG**: Detailed diagnostic information
- **INFO**: General informational messages
- **WARNING**: Warning messages
- **ERROR**: Error messages
- **CRITICAL**: Critical failures

#### Cache

```yaml
cache:
  enabled: true            # Enable caching
  ttl: 3600               # Time-to-live (seconds)
  max_size: 1000          # Max cached items
  backend: memory         # Backend (memory, redis, disk)

  # Disk backend config
  disk:
    path: ~/.victor/cache/
    max_size_mb: 1024

  # Redis backend config
  redis:
    host: 127.0.0.1
    port: 6379
    db: 0
    password: null
```

**Cache Strategies**:
- **Memory**: Fastest, lost on restart
- **Disk**: Persistent, slower
- **Redis**: Shared across instances, persistent

#### Tools

```yaml
tools:
  lazy_load: true          # Load tools on-demand (faster startup)
  parallel_execution: true # Execute tools in parallel when possible
  timeout: 30              # Tool execution timeout (seconds)
  max_concurrent: 5        # Max parallel tool executions
```

#### Conversation

```yaml
conversation:
  max_history: 100         # Max messages in conversation history
  save_history: true       # Save conversation to disk
  history_file: ~/.victor/conversations/
  compress_history: false  # Compress old messages
```

#### Providers

```yaml
providers:
  timeout: 30              # Request timeout (seconds)
  retry_attempts: 3        # Retry attempts on failure
  circuit_breaker:
    enabled: true          # Enable circuit breaker
    failure_threshold: 5   # Failures before opening circuit
    recovery_timeout: 60   # Seconds before retry
```

**Circuit Breaker**: Prevents cascading failures by temporarily disabling failing providers.

#### Workflows

```yaml
workflows:
  directory: ~/.victor/workflows/  # Workflow directory
  auto_compile: true                # Auto-compile YAML workflows
  cache_compiled: true              # Cache compiled workflows
  max_parallel_nodes: 10            # Max parallel workflow nodes
```

#### UI

```yaml
ui:
  theme: dark              # Theme (dark, light)
  syntax_highlighting: true
  line_numbers: true
  font_size: 14
  editor: vim              # Default editor
  pager: less              # Default pager
```

[Full config.yaml Reference →](#configyaml)

---

## mcp.yaml

Model Context Protocol (MCP) server configuration.

**Location**: `~/.victor/mcp.yaml`
**Required**: No (only if using MCP)
**Purpose**: Configure MCP server settings

### Basic Structure

```yaml
# MCP server configuration
server:
  host: 127.0.0.1
  port: 8080
  transport: stdio  # stdio or sse

  # Authentication
  auth:
    enabled: false
    api_key: your-api-key

# Tool exposure
tools:
  expose_all: false  # Expose all tools to MCP clients
  allowed_tools:
    - read_file
    - write_file
    - search_files
    - run_tests

# Prompt templates
prompts:
  directory: ~/.victor/prompts/
  expose_prompts:
    - code_review
    - refactor

# Resources
resources:
  expose_all: false
  allowed_resources:
    - file:///
    - git:///
```

### MCP Reference

**Transport Modes**:
- **stdio**: Standard input/output (default)
- **sse**: Server-sent events (HTTP)

**Tool Exposure**: Control which tools are available to MCP clients

**Prompts**: Expose prompt templates to MCP clients

**Resources**: Expose file system and git resources

[Full mcp.yaml Reference →](#mcpyaml)

---

## Environment Variables

Environment variables for configuration and secrets.

### Provider API Keys

```bash
# Cloud providers
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-proj-...
export GOOGLE_API_KEY=...
export XAI_API_KEY=...
export DEEPSEEK_API_KEY=...
export MISTRAL_API_KEY=...
export TOGETHER_API_KEY=...
export FIREWORKS_API_KEY=...
export OPENROUTER_API_KEY=...
export GROQ_API_KEY=...
export MOONSHOT_API_KEY=...
export CEREBRAS_API_KEY=...

# Enterprise
export AZURE_OPENAI_API_KEY=...
export AZURE_OPENAI_ENDPOINT=https://...
export AWS_PROFILE=default
export GCP_PROJECT=...

# Platforms
export HF_API_KEY=...
export REPLICATE_API_TOKEN=...
```

### Local Providers

```bash
# Optional: Override default hosts
export OLLAMA_HOST=127.0.0.1:11434
export VICTOR_LM_STUDIO_HOST=127.0.0.1:1234
export VICTOR_VLLM_HOST=127.0.0.1:8000
export VICTOR_LLAMACPP_HOST=127.0.0.1:8080
```

### Victor Configuration

```bash
# General
export VICTOR_CONFIG_DIR=~/.victor
export VICTOR_PROFILE=development
export VICTOR_LOG_LEVEL=INFO
export VICTOR_LOG_FILE=/var/log/victor.log

# Cache
export VICTOR_CACHE_ENABLED=true
export VICTOR_CACHE_TTL=3600

# Providers
export VICTOR_TIMEOUT=30
export VICTOR_RETRY_ATTEMPTS=3

# Tools
export VICTOR_TOOL_TIMEOUT=30
export VICTOR_MAX_CONCURRENT_TOOLS=5
```

### Proxy Settings

```bash
# HTTP proxy
export HTTP_PROXY=http://proxy.example.com:8080
export HTTPS_PROXY=http://proxy.example.com:8080
export NO_PROXY=localhost,127.0.0.1

# SOCKS proxy
export ALL_PROXY=socks5://proxy.example.com:1080
```

[Full Environment Variables →](keys.md)

---

## Project Context Files

Project-specific configuration for AI behavior.

### .victor.md

**Location**: `<project>/.victor.md`
**Purpose**: Project context and instructions for Victor

**Example**:
```markdown
# Project Context

## Project Overview
This is a Django web application with PostgreSQL backend.

## Tech Stack
- Backend: Django 5.0, Python 3.11
- Frontend: React 18, TypeScript
- Database: PostgreSQL 15
- Cache: Redis 7

## Code Style
- Follow PEP 8 for Python
- Use type hints (strict mode)
- Max line length: 100 characters
- Write docstrings (Google style)

## Testing
- Use pytest for unit tests
- Minimum coverage: 80%
- Run tests before committing

## Git Workflow
- Feature branches: `feature/xxx`
- Main branch: `main`
- PR required for all changes

## Victor Instructions
When refactoring code:
1. Add type hints
2. Update docstrings
3. Add unit tests
4. Run pytest and fix failures
```

**Usage**: Victor automatically reads `.victor.md` when working in a project directory.

[Full .victor.md Guide →](../../user-guide/index.md#5-project-context)

### CLAUDE.md

**Location**: `<project>/CLAUDE.md`
**Purpose**: Instructions for Claude AI (used by Claude Code and Victor)

**Example**:
```markdown
# Claude Instructions

## Project Structure
- `src/`: Source code
- `tests/`: Test files
- `docs/`: Documentation

## Development Commands
- `make test`: Run tests
- `make lint`: Run linter
- `make format`: Format code

## Conventions
- Use dependency injection
- Write async/await for I/O
- Add error handling
- Log important events

## Preferences
- Prefer composition over inheritance
- Use type hints
- Keep functions small (<50 lines)
- Write self-documenting code
```

[Victor Configuration Architecture →](../../architecture.md#configuration-system)

---

## Configuration Examples

### Example 1: Local Development

**~/.victor/profiles.yaml**:
```yaml
profiles:
  default:
    provider: ollama
    model: qwen2.5-coder:7b
```

**~/.victor/config.yaml**:
```yaml
logging:
  level: DEBUG

cache:
  enabled: true
  backend: memory

tools:
  lazy_load: true
```

**.victor.md**:
```markdown
# Local Development
Use Ollama for privacy and speed.
```

### Example 2: Team Development

**~/.victor/profiles.yaml**:
```yaml
profiles:
  development:
    provider: anthropic
    model: claude-sonnet-4-20250514
    temperature: 0.7

  production:
    provider: azure
    deployment: gpt-4o
    temperature: 0.3
```

**~/.victor/config.yaml**:
```yaml
logging:
  level: INFO
  file: /var/log/victor/team.log

cache:
  enabled: true
  backend: redis
  redis:
    host: redis.example.com
    port: 6379

tools:
  parallel_execution: true
  max_concurrent: 10
```

### Example 3: Enterprise

**~/.victor/profiles.yaml**:
```yaml
profiles:
  default:
    provider: bedrock
    model_id: anthropic.claude-3-sonnet-20240229-v1:0
    region_name: us-east-1

  fallback:
    provider: vertex
    model: gemini-2.0-flash-exp
    project: enterprise-project
```

**~/.victor/config.yaml**:
```yaml
logging:
  level: INFO
  format: json
  file: /var/log/victor/enterprise.log

providers:
  timeout: 60
  retry_attempts: 5
  circuit_breaker:
    enabled: true
    failure_threshold: 3

cache:
  enabled: true
  backend: redis
  redis:
    host: redis.internal.example.com
    port: 6379
    password: ${REDIS_PASSWORD}
```

---

## Validation and Testing

### Validate Configuration

```bash
# Validate all configuration files
victor config validate

# Check current configuration
victor config show

# Test profile
victor --profile development chat "Test"

# Test provider
victor chat --provider anthropic "Test"
```

### Debug Configuration

```bash
# Enable debug logging
export VICTOR_LOG_LEVEL=DEBUG

# View configuration
victor config show

# Check logs
victor logs --tail 100

# Trace configuration loading
victor --debug chat "Test"
```

---

## Troubleshooting

### Common Issues

**Configuration not loading**:
```bash
# Check file location
ls -la ~/.victor/profiles.yaml

# Validate YAML syntax
python -c "import yaml; yaml.safe_load(open('~/.victor/profiles.yaml'))"

# Check permissions
chmod 600 ~/.victor/profiles.yaml
```

**Profile not found**:
```bash
# List available profiles
victor config profiles

# Check profile name in profiles.yaml
grep -A 5 "profiles:" ~/.victor/profiles.yaml
```

**API key not found**:
```bash
# Check environment variable
echo $ANTHROPIC_API_KEY

# Set temporarily
export ANTHROPIC_API_KEY=sk-ant-...

# Add to shell profile
echo 'export ANTHROPIC_API_KEY=sk-ant-...' >> ~/.bashrc
source ~/.bashrc
```

[Full Troubleshooting Guide →](../../user-guide/troubleshooting.md)

---

## Best Practices

### 1. Use Profiles

**DO**:
```yaml
# Reusable profiles
profiles:
  development:
    provider: anthropic
    model: claude-sonnet-4-20250514

  production:
    provider: azure
    deployment: gpt-4o
```

**DON'T**:
```bash
# Hard-coded values
victor chat --provider anthropic --model claude-sonnet-4-20250514 --api-key sk-...
```

### 2. Secure API Keys

**DO**:
```bash
# Environment variables
export ANTHROPIC_API_KEY=sk-ant-...
```

**DON'T**:
```yaml
# API keys in config files
api_key: sk-ant-...  # NEVER commit this!
```

### 3. Version Control

**DO**:
```bash
# Add to .gitignore
echo ".victor/" >> .gitignore
echo "*.yaml" >> .gitignore
echo ".env" >> .gitignore
```

**DON'T**:
```bash
# Commit configuration files
git add ~/.victor/profiles.yaml  # May contain secrets!
```

### 4. Document Configuration

**DO**:
```yaml
# ~/.victor/profiles.yaml
profiles:
  # Development profile for local testing
  development:
    provider: anthropic
    model: claude-sonnet-4-20250514
    temperature: 0.7  # Higher temp for creativity
```

**DON'T**:
```yaml
# No comments
profiles:
  development:
    provider: anthropic
    model: claude-sonnet-4-20250514
    temperature: 0.7
```

### 5. Test Configuration

**DO**:
```bash
# Test before committing
victor config validate
victor --profile development chat "Test"
```

**DON'T**:
```bash
# Assume it works
victor chat "Important task"  # May fail!
```

---

## Additional Resources

- **profiles.yaml**: [Full Reference →](#profilesyaml)
- **config.yaml**: [Full Reference →](#configyaml)
- **mcp.yaml**: [Full Reference →](#mcpyaml)
- **API Keys**: [Key Management →](keys.md)
- **Troubleshooting**: [Troubleshooting Guide →](../../user-guide/troubleshooting.md)

---

**Next**: [profiles.yaml →](#profilesyaml) | [config.yaml →](#configyaml) | [API Keys →](keys.md) | [Troubleshooting →](../../user-guide/troubleshooting.md)
