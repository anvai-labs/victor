# Quick Start Guide

Get Victor up and running in **5 minutes**.

## Prerequisites

- Python 3.11+ installed
- One of the following:
  - **Local**: [Ollama](https://ollama.ai) for free local models
  - **Cloud**: API key for Anthropic, OpenAI, or Google

---

## 1. Install Victor (1 minute)

### Recommended: pipx
```bash
pipx install victor-ai
```

### Alternative: pip
```bash
pip install victor-ai
```

**Verify:**
```bash
victor --version
```

---

## 2. Choose Your Model (1 minute)

### Option A: Local Model (Free, No API Key)

```bash
# Install Ollama
brew install ollama  # macOS
# OR
curl -fsSL https://ollama.com/install.sh | sh  # Linux

# Start Ollama
ollama serve

# Pull a code-focused model
ollama pull qwen2.5-coder:7b
```

### Option B: Cloud Provider (API Key Required)

```bash
# Set your API key
export ANTHROPIC_API_KEY=sk-ant-your-key
# OR
export OPENAI_API_KEY=sk-proj-your-key
```

---

## 3. Start Victor (30 seconds)

```bash
# With local model
victor chat --provider ollama --model qwen2.5-coder:7b

# With cloud provider
victor chat --provider anthropic
```

---

## 4. Try It Out (2 minutes)

Once Victor starts, try these prompts:

### Explore Code
```
What does this project do?
```

### Get Help
```
Explain the main function in src/main.py
```

### Make Changes
```
Add error handling to the database connection
```

### Run Tests
```
Run tests and summarize any failures
```

---

## Quick Reference

| Command | Description |
|---------|-------------|
| `victor chat` | Start interactive chat |
| `victor "prompt"` | One-shot query |
| `victor --provider <name>` | Use specific provider |
| `/help` | Show in-chat commands |
| `/clear` | Clear conversation |
| `/quit` | Exit Victor |

---

## Common Providers

| Provider | Setup |
|----------|-------|
| **Ollama** (local) | `ollama pull qwen2.5-coder:7b` |
| **Anthropic** | `export ANTHROPIC_API_KEY=sk-...` |
| **OpenAI** | `export OPENAI_API_KEY=sk-proj-...` |
| **Google** | `export GOOGLE_API_KEY=...` |

---

## What's Next?

- **[Build Your First Agent](first-agent.md)** - Create custom agents
- **[Create Your First Workflow](first-workflow.md)** - Automate with YAML workflows
- **[Full Installation Guide](installation.md)** - Troubleshooting & details
- **[Configuration Guide](../getting-started/configuration.md)** - Customize settings

---

**Need help?** Type `/help` in Victor or see [Installation Troubleshooting](installation.md#troubleshooting)
