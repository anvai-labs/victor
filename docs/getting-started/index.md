# Getting Started with Victor

Welcome to Victor! This guide will help you install, configure, and run Victor in under 5 minutes.

## What is Victor?

**Victor** is an open-source agentic AI framework that supports 24 LLM providers (both local and cloud) through a unified CLI/TUI interface.

**Key Features:**
- **24 LLM Providers**: Cloud (Anthropic, OpenAI, Google, Vertex) and local (Ollama, LM Studio, vLLM)
- **Provider Switching**: Switch between any provider mid-conversation without losing context
- **No API Key Required**: Use local models by default for privacy and cost savings
- **34 Tool Modules**: File operations, code editing, git, testing, search, and more
- **9 Domain Verticals**: Coding, DevOps, RAG, Data Analysis, Research, Security, IaC, Classification, Benchmark
- **YAML Workflows**: Define multi-step automation with scheduling and versioning
- **Multi-Agent Teams**: Coordinate specialized AI agents for complex tasks

## Choose Your Path

Select the guide that matches your situation:

| Your Goal | Guide | Time |
|-----------|-------|------|
| **Quick start** | [Quickstart](quickstart.md) | 5 min |
| **First agent from Python** | [Your First Agent](first-agent.md) | 5 min |
| **Full installation options** | [Installation](installation.md) | 10 min |
| **Configure Victor** | [Configuration](configuration.md) | 10 min |
| **Troubleshoot issues** | [Troubleshooting](../user-guide/troubleshooting.md) | As needed |

## Fastest Path to Running Victor

### Local Model (No API Key)

```bash
# 1. Install Victor
pipx install victor-ai

# 2. Install and start Ollama
brew install ollama  # macOS
# or: curl -fsSL https://ollama.com/install.sh | sh  # Linux
ollama serve

# 3. Pull a model
ollama pull qwen2.5-coder:7b

# 4. Start Victor
victor chat "Hello, Victor!"
```

### Cloud Provider (API Key Required)

```bash
# 1. Install Victor
pipx install victor-ai

# 2. Set API key
export ANTHROPIC_API_KEY=sk-ant-your-key

# 3. Start Victor
victor chat --provider anthropic "Hello, Victor!"
```

### Docker

```bash
# Pull and run
docker pull ghcr.io/vjsingh1984/victor:latest
docker run -it \
  -v ~/.victor:/root/.victor \
  -v "$(pwd)":/workspace \
  -w /workspace \
  ghcr.io/vjsingh1984/victor:latest chat
```

## System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| **Python** | 3.10 | 3.11+ |
| **RAM** | 4 GB | 8 GB+ |
| **Disk** | 500 MB | 1 GB (local models: 8-32 GB) |
| **OS** | Linux, macOS, Windows (WSL2) | Linux or macOS |

## Quick Verification

After installation, verify Victor is working:

```bash
# Check version
victor --version

# List providers
victor providers

# Test chat
victor chat "Hello, Victor!"

# Open TUI mode
victor
```

## Getting Started Guides

### 1. [Installation Guide](installation.md)

Complete installation instructions including:
- System requirements
- Installation methods (pipx, pip, Docker, development)
- Optional dependencies
- Platform-specific instructions (macOS, Linux, Windows)
- Troubleshooting common issues

### 2. [Quickstart Guide](quickstart.md)

Get productive quickly:
- First run experience
- Interface modes (TUI, CLI, API)
- Essential commands
- Common workflows
- Quick reference card

### 3. [Configuration Guide](configuration.md)

Customize Victor for your needs:
- Environment variables and API keys
- Profiles configuration
- Global settings
- Project context files (.victor.md, CLAUDE.md)
- Modes (BUILD, PLAN, EXPLORE)

## What's Next?

After getting started:

1. **[User Guide](../user-guide/)** - Daily usage patterns and workflows
2. **[Provider Reference](../reference/providers/)** - All 24 providers detailed
3. **[Tool Catalog](../reference/tools/catalog.md)** - 34 tool modules
4. **[Workflow DSL](../guides/workflow-development/dsl.md)** - YAML workflow automation

## Need Help?

- **Troubleshooting**: [Troubleshooting Guide](../user-guide/troubleshooting.md)
- **Documentation**: [Full docs](../README.md)
- **Community**: [GitHub Discussions](https://github.com/anvai-labs/victor/discussions)
- **Issues**: [Report bugs](https://github.com/anvai-labs/victor/issues)

---

**Next**: [Installation Guide](installation.md) | [Quickstart](quickstart.md) | [Configuration](configuration.md)
