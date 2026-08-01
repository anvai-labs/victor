# Quickstart Guide

Get Victor running and complete your first task in under 5 minutes.

## Prerequisites

Before starting, ensure you have:
- Victor installed ([Installation Guide](installation.md))
- Either a local model (Ollama) or cloud API key (Anthropic, OpenAI, etc.)

---

## Step 1: First Run

### Option A: Guided Setup (Recommended)

Just run Victor — on a fresh machine it launches the setup wizard
automatically:

```bash
victor
```

The wizard walks you through, in order:

1. **Environment detection** — finds running local providers (Ollama, LM Studio)
2. **Provider & model selection** — local or cloud (Anthropic, OpenAI, Google, ZhipuAI, …)
3. **Credentials** — API key entry (stored securely in your system keyring) or OAuth
4. **Connection test** — a live smoke test against the provider
5. **You're ready** — your default profile is applied and example prompts are shown, then chat starts

Skip it with `victor --no-setup`, or re-run it anytime with
`victor onboarding --force`. If anything fails, run `victor doctor` for a
full diagnosis.

### Option B: Manual Setup with Local Model (Ollama)

No API key required:

```bash
# Make sure Ollama is running
ollama serve

# Pull a model (if not already done)
ollama pull qwen2.5-coder:7b

# Start Victor
victor chat
```

### Option C: Manual Setup with Cloud Provider

For more powerful models:

```bash
# Set your API key (choose one)
export ANTHROPIC_API_KEY=sk-ant-your-key
export OPENAI_API_KEY=sk-proj-your-key

# Start Victor with your provider
victor chat --provider anthropic
# or
victor chat --provider openai
```

---

## Step 2: Your First Conversation

Once Victor starts, try these prompts:

### Explore Your Codebase
```
Summarize this repository and identify the main components
```

### Get Help with Code
```
Explain what the main function in src/app.py does
```

### Make Changes
```
Add error handling to the database connection in db.py
```

### Run Tests
```
Run the test suite and summarize any failures
```

---

## Step 3: Interface Modes

Victor offers multiple ways to interact:

### Interactive CLI (Default)

A rich terminal chat interface with syntax highlighting (the same surface as
`victor chat`; the separate Textual observability app is `victor observe dashboard`):

```bash
victor
# or
vic  # short alias
```

Features:
- Syntax-highlighted code blocks
- Tool execution status
- Conversation history
- Rich formatting

### CLI Mode

For quick queries and scripting:

```bash
# Interactive chat
victor chat

# One-shot task
victor "explain this file" src/main.py

# Pipe content
cat error.log | victor "diagnose this error"
```

### HTTP API Mode

For integration with other tools:

```bash
# Start API server
victor serve --port 8080

# Make requests
curl http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello, Victor!"}'
```

---

## Step 4: Essential Commands

Learn these commands to work efficiently:

### In-Chat Commands

| Command | Description |
|---------|-------------|
| `/help` | Show all available commands |
| `/shortcuts` | Show interactive CLI keyboard shortcuts |
| `/clear` | Clear conversation history |
| `/provider <name>` | Switch to a different LLM provider |
| `/mode <mode>` | Switch between BUILD/PLAN/EXPLORE |
| `/save "title"` | Save the current session |
| `/resume` | Restore a previous session |
| `/quit` or `/exit` | Exit Victor |

### Provider Switching

Switch models mid-conversation without losing context:

```
/provider anthropic --model claude-sonnet-4-5
/provider openai --model gpt-4o
/provider ollama --model qwen2.5-coder:7b
```

### Execution Modes

Control what Victor can do:

```bash
# BUILD mode (default) - can make file changes
victor chat --mode build

# PLAN mode - analyze and plan, no real edits
victor chat --mode plan

# EXPLORE mode - read-only exploration
victor chat --mode explore
```

---

## Step 5: Common Workflows

### Code Review

```bash
victor --mode plan "Review src/auth.py for security issues"
```

### Refactoring

```bash
victor "Refactor the User class to use dependency injection"
```

### Test Generation

```bash
victor "Write unit tests for the payment module"
```

### Git Operations

```bash
victor "Create a commit for the current changes with a descriptive message"
```

### Documentation

```bash
victor "Generate docstrings for all public functions in src/api/"
```

---

## Quick Reference Card

```
Installation:
  pipx install victor-ai          # Install Victor
  victor init                     # Initialize config
  victor --version                # Check version

Running Victor:
  victor                          # Interactive CLI (same as victor chat)
  victor chat                     # Interactive CLI
  victor observe dashboard        # Textual observability dashboard
  victor "task"                   # One-shot mode
  victor chat --provider X        # Use specific provider
  victor --profile local          # Use saved profile

In-Chat Commands:
  /help                           # Show help
  /shortcuts                      # Show keyboard shortcuts
  /provider <name>                # Switch provider
  /mode <mode>                    # Switch mode (build/plan/explore)
  /save "title"                   # Save session
  /resume                         # Restore session
  /clear                          # Clear history
  /quit                           # Exit

Interactive CLI Shortcuts:
  Tab                             # Complete commands and known command arguments
  F1                              # Show shortcut help
  Alt+Enter                       # Insert newline
  Ctrl+O                          # Expand last tool output

Modes:
  BUILD   - Full edits allowed (default)
  PLAN    - Analysis only, sandbox edits
  EXPLORE - Read-only exploration

Providers (examples):
  ollama    - Local (free, private)
  anthropic - Claude (powerful)
  openai    - GPT-4 (versatile)
  groq      - Fast inference
```

---

## Example Session

Here is a typical Victor session:

```bash
$ victor chat --provider anthropic

> Hello, Victor! Summarize this project.

Victor is analyzing your project...

This is a Python web application with the following structure:
- **src/**: Main application code
  - api/: REST API endpoints
  - models/: Database models
  - services/: Business logic
- **tests/**: Test suite
- **docs/**: Documentation

Key technologies: FastAPI, SQLAlchemy, PostgreSQL

> Add input validation to the user registration endpoint

I'll add Pydantic validation to the registration endpoint.

[Tool: read_file] Reading src/api/auth.py...
[Tool: edit_file] Adding validation...

Done! I've added:
1. Email format validation
2. Password strength requirements
3. Username length constraints

Would you like me to add tests for the validation?

> Yes, please

[Tool: write_file] Creating tests/test_auth_validation.py...
[Tool: run_tests] Running pytest...

All 5 new tests pass. The validation is working correctly.

> /provider ollama --model qwen2.5-coder:7b

Switched to ollama/qwen2.5-coder:7b. Context preserved.

> Continue with local model - add rate limiting

[Continuing with local model...]
```

---

## What to Try Next

Now that you have Victor running:

1. **Use Victor from Python** - Build your first agent in 5 minutes
   - [Your First Agent](first-agent.md)

2. **Configure profiles** - Create shortcuts for your favorite providers
   - [Configuration Guide](configuration.md)

3. **Learn the tools** - Explore Victor's 34 tool modules
   - [Tool Catalog](../reference/tools/catalog.md)

4. **Set up project context** - Teach Victor about your codebase
   - [Configuration - Project Context](configuration.md#project-context-files)

5. **Explore workflows** - Automate repetitive tasks
   - [Workflow Guide](../guides/workflow-development/)

6. **Try different providers** - Find the best model for your needs
   - [Provider decision matrix](../user-guide/providers.md#provider-decision-matrix)

---

## Getting Help

- **In-app help**: Type `/help` in Victor
- **Documentation**: [Full docs](../README.md)
- **Troubleshooting**: [Troubleshooting Guide](../user-guide/troubleshooting.md)
- **Community**: [GitHub Discussions](https://github.com/anvai-labs/victor/discussions)
- **Issues**: [Report bugs](https://github.com/anvai-labs/victor/issues)

---

**Next**: [Configuration Guide](configuration.md) - Customize Victor for your workflow
