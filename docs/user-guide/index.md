# User Guide

Complete guide for daily usage of Victor AI Assistant.

## Overview

This guide covers everything you need to use Victor effectively, from basic conversations to advanced workflows and multi-agent teams.

**New to Victor?** Start with [Getting Started](../getting-started/).

## Quick Reference

| Topic | Documentation | Use Case |
|-------|--------------|----------|
| **Basic Usage** | [Basic Usage](../getting-started/basic-usage.md) | Your first conversation |
| **CLI Commands** | [CLI Reference](cli-reference.md) | All commands and options |
| **Tools** | [Tools Guide](tools.md) | Tool system, selection, execution |
| **Session Management** | [Session Management](session-management.md) | Save and restore sessions |
| **Providers** | [Provider Guide](providers.md) | 25 LLM providers setup and switching |
| **Workflows** | [Workflows Guide](workflows.md) | YAML-based automation |
| **Configuration** | [Configuration →](../reference/configuration/) | Profiles and settings |

## Common Tasks

### Daily Development

**Session Management**
```bash
/save "My Session Title"        # Save current conversation
/sessions                         # List saved sessions
/resume                          # Interactive session restore
/resume 20250107_153045          # Restore specific session
```
[Full Guide →](session-management.md)

**Code Review**
```bash
victor "Review this PR for bugs and improvements"
```
[Tools Guide →](tools.md#code-analysis)

**Refactoring**
```bash
victor --mode build "Refactor to use dependency injection"
```
[Tools Guide →](tools.md#file-operations)

**Testing**
```bash
victor "Write unit tests for auth.py"
```
[Tools Guide →](tools.md#shell-and-execution)

**Git Operations**
```bash
victor "Create commit for these changes"
```
[Tools Guide →](tools.md#git-operations)

### Advanced Features

**Provider Switching**
```bash
victor chat --provider anthropic
/provider openai --model gpt-4
/provider ollama --model qwen2.5-coder:7b
```
[Provider Guide →](providers.md)

**Workflow Execution**
```bash
victor workflow run code-review
```
[Workflows Guide →](workflows.md)

**Multi-Agent Teams**
```python
from victor.framework import Agent, AgentTeam

team = AgentTeam.hierarchical(
    lead="senior-developer",
    subagents=["frontend", "backend", "testing"]
)
```
[Learn More → Multi-Agent Teams](../guides/multi-agent/)

## Core Concepts

### 1. Provider Switching

Victor's key differentiator: **switch models without losing context**.

Start with Claude, continue with GPT-4, finish with a local model—all in one conversation.

- **Context Independence**: Conversation history managed separately from LLM
- **Instant Switching**: Use `/provider` command anytime
- **Full Support**: All 25 providers support context preservation

[Provider Guide →](providers.md)

### 2. Execution Modes

Three modes for different workflows:

| Mode | Purpose | File Edits | Use When |
|------|---------|------------|----------|
| **BUILD** | Real implementation | Yes | Making actual changes |
| **PLAN** | Analysis and planning | No | Understanding code |
| **EXPLORE** | Understanding only | No | Learning codebase |

[Learn More →](cli-reference.md#modes)

### 3. Tools System

34 tool modules organized by category:

- **File Operations** (12 tools): Read, write, edit, search files
- **Git** (8 tools): Version control, commits, branches
- **Testing** (6 tools): Test execution, coverage, fixtures
- **Search** (7 tools): Code search, grep, semantic search
- **Web** (5 tools): HTTP requests, scraping, browsing
- **And 8 more categories...**

[Tools Guide →](tools.md) | [Full Tool Catalog →](../reference/tools/catalog.md)

### 4. Workflows

YAML-based automation with scheduling and versioning:

```yaml
workflows:
  code_review:
    nodes:
      - id: analyze
        type: agent
        role: reviewer
        goal: "Analyze this PR for issues"
        tool_budget: 30
        next: [test]

      - id: test
        type: compute
        tools: [shell]
        inputs:
          command: pytest tests/
```

[Workflows Guide →](workflows.md) | [DSL Reference →](../guides/workflow-development/dsl.md)

### 5. Project Context

Configure project-specific instructions:

- **`.victor.md`**: Project context and instructions
- **`CLAUDE.md`**: AI assistant project instructions
- **Auto-discovery**: Victor finds these files automatically

[Learn More →](../reference/configuration/index.md#victormd)

## Configuration

### Quick Setup

**1. Local model** (default - no config needed):
```bash
victor  # Automatically uses Ollama
```

**2. Cloud provider**:
```bash
export ANTHROPIC_API_KEY=sk-...
victor chat --provider anthropic
```

**3. Profiles** (`~/.victor/profiles.yaml`):
```yaml
profiles:
  development:
    provider: anthropic
    model: claude-sonnet-4-20250514
  production:
    provider: openai
    model: gpt-4
```

[Full Configuration Guide →](../reference/configuration/)

## Troubleshooting

**Installation Issues**
- Victor not found after install? → [Getting Started →](../getting-started/installation.md)
- Permission errors? → [Troubleshooting →](troubleshooting.md)

**Provider Issues**
- API key errors? → [Provider Reference →](../reference/providers/)
- Connection timeouts? → [Troubleshooting →](troubleshooting.md)
- Model not found? → [Provider Reference →](../reference/providers/)

**Performance Issues**
- Slow responses? → [Performance Benchmarks →](../operations/performance/benchmarks.md)
- High memory usage? → [Performance Benchmarks →](../operations/performance/benchmarks.md)
- Tool execution errors? → [Troubleshooting →](troubleshooting.md)

[Full Troubleshooting Guide →](troubleshooting.md)

## Integration

### CI/CD Integration

**GitHub Actions**:
```yaml
- name: Code Review
  run: victor chat "Review this PR" --mode plan
```

[More Guides →](../guides/index.md)

### HTTP API

Start REST API server:
```bash
victor serve --port 8080
```

[API Reference →](../reference/api/http-api.md)

### MCP Server

Run as MCP server:
```bash
victor mcp --stdio
```

[MCP Server Reference →](../reference/api/mcp-server.md)

### VS Code Extension

Install from marketplace or build from source.

[VS Code Setup →](../../vscode-victor/README.md)

## Advanced Usage

### Verticals

Domain-specific assistants for specialized tasks:

| Vertical | Description | Usage |
|----------|-------------|-------|
| [**Coding**](../reference/verticals/index.md) | Software development | `victor --vertical coding` |
| [**DevOps**](../reference/verticals/index.md) | DevOps and infrastructure | `victor --vertical devops` |
| [**RAG**](../reference/verticals/index.md) | Retrieval-augmented generation | `victor --vertical rag` |
| [**Data Analysis**](../reference/verticals/index.md) | Data science workflows | `victor --vertical dataanalysis` |
| [**Research**](../reference/verticals/index.md) | Research and analysis | `victor --vertical research` |

### Multi-Agent Coordination

Coordinate specialized AI agents for complex tasks.

**Team Formations**:
- **Hierarchical**: Lead agent with sub-agents
- **Flat**: Peer agents with shared memory
- **Pipeline**: Sequential agent processing
- **Consensus**: Agents vote on decisions
- **Debate**: Agents debate to consensus

[Full Multi-Agent Guide →](../guides/multi-agent/)

### Observability

Monitor Victor's behavior and performance:

**Event Bus**:
```python
from victor.core.events import EventBus

def on_tool_execution(event):
    print(f"Tool {event.tool_name} executed")

EventBus.subscribe("tool.execution", on_tool_execution)
```

[Full Observability Guide →](../guides/observability/)

## Best Practices

### 1. Start with PLAN Mode

Use PLAN mode to understand before making changes:
```bash
victor --mode plan "Analyze this function"
```

### 2. Use Provider Switching

Leverage different models for different tasks:
- **Claude**: Complex reasoning and planning
- **GPT-4**: Code generation and refactoring
- **Local models**: Quick iterations and privacy

### 3. Configure Project Context

Create `.victor.md` with project-specific instructions:
```markdown
# Project Context

This is a Django project with PostgreSQL backend.
Follow Django conventions and use type hints.
```

### 4. Use Workflows for Repetitive Tasks

Define workflows for common operations:
- Code review
- Testing
- Documentation generation
- Deployment

[Workflows Guide →](workflows.md)

### 5. Leverage Tool Composition

Chain tools for complex operations:
```python
from victor.tools import pipe, parallel

# Sequential execution
pipe(read_file, analyze_code, write_report)

# Parallel execution
parallel(run_tests, run_linter, run_coverage)
```

[Tool Catalog →](../reference/tools/catalog.md)

## Examples

### Example 1: Code Review Workflow

```bash
# 1. Start review
victor workflow run code-review

# 2. Or manually
victor --mode plan "Review authentication.py for security issues"

# 3. Switch providers for different perspectives
/provider anthropic  # Claude's analysis
/provider openai     # GPT-4's analysis

# 4. Apply suggestions
victor --mode build "Fix the security issues identified"
```

### Example 2: Refactoring Session

```bash
# 1. Understand current code
victor --mode explore "How does user authentication work?"

# 2. Plan refactoring
victor --mode plan "Refactor auth to use dependency injection"

# 3. Implement changes
victor --mode build "Implement the DI refactoring"

# 4. Write tests
victor "Write unit tests for the refactored auth module"

# 5. Run tests
victor "Execute tests and fix any failures"
```

### Example 3: Multi-Agent Code Generation

```python
from victor.framework import Agent, AgentTeam

# Create specialized agents
frontend = Agent(role="Frontend developer", tools=["react", "typescript"])
backend = Agent(role="Backend developer", tools=["fastapi", "sqlalchemy"])
tester = Agent(role="QA engineer", tools=["pytest", "selenium"])

# Coordinate team
team = AgentTeam.hierarchical(
    lead="senior-developer",
    subagents=[frontend, backend, tester]
)

result = await team.run("Implement user registration feature")
```

## Additional Resources

- **Session Management**: [Session Management →](session-management.md)
- **Troubleshooting**: [Troubleshooting Guide →](troubleshooting.md)
- **Reference**: [Provider Reference →](../reference/providers/)
- **Reference**: [Tool Catalog →](../reference/tools/)
- **Workflows**: [Workflows Guide →](workflows.md)
- **Guides**: [Multi-Agent Teams →](../guides/multi-agent/)
- **Guides**: [Integration →](../guides/integration/)
- **Development**: [Contributing →](../../CONTRIBUTING.md)

---

**Next**: [Basic Usage →](../getting-started/basic-usage.md)
