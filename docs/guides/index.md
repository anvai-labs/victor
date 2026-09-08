# How-to Guides

Practical guides for specific Victor tasks and workflows.

## Overview

This section contains step-by-step guides for common tasks, from basic setup to advanced multi-agent coordination and observability.

**New to Victor?** Start with [Getting Started](../getting-started/index.md) or [User Guide](../user-guide/index.md).

## Quick Links

| Guide | Description | Difficulty |
|-------|-------------|------------|
| [**Workflow DSL**](workflow-development/dsl.md) | Define YAML workflows | Beginner |
| [**Multi-Agent Teams**](MULTI_AGENT_TEAMS.md) | Coordinate AI agents | Intermediate |
| [**Observability**](observability/index.md) | Events and metrics | Intermediate |
| [**MCP Clients**](integration/mcp-clients.md) | Using MCP servers | Intermediate |
| [**Codebase Verification**](CODEBASE_VERIFICATION.md) | Semantic validation & FP detection | Intermediate |
| [**Development Setup**](development/local-models.md) | Local model setup | Beginner |
| [**Performance**](BENCHMARKING.md) | Optimization and tuning | Advanced |
| [**Shell Safety Policy**](https://github.com/anvai-labs/victor/blob/develop/feps/fep-0013-shell-safety-policy.md) | Best practices | Intermediate |
| [**Resilience**](RESILIENCE.md) | Error handling | Advanced |
| [**HITL Workflows**](HITL_WORKFLOWS.md) | Human-in-the-loop patterns | Intermediate |

## Workflow Development

Complete guide to creating and using YAML-based workflows.

### Topics

- **[Workflow DSL](workflow-development/dsl.md)**: StateGraph YAML syntax and features
- **[Scheduling](workflow-development/scheduling.md)**: Cron-based workflow scheduling
- **[Examples](workflow-development/examples.md)**: Common workflow patterns

### Example Workflow

```yaml
workflow: CodeReview
description: Automated code review workflow

nodes:
  - id: analyze
    type: agent
    role: You are a code reviewer. Analyze this PR.
    goal: Identify bugs, security issues, and improvements.

  - id: test
    type: compute
    tools: [pytest, coverage]
    config:
      command: pytest tests/ -v

  - id: report
    type: transform
    input: analyze.output, test.output
    template: |
      # Review Report
      {{analyze.output}}
      ## Test Results
      {{test.output}}

edges:
  - source: analyze
    target: test
  - source: test
    target: report
```

[Full Workflow Guide →](workflow-development/dsl.md)

## Multi-Agent Coordination

Coordinate specialized AI agents for complex tasks.

### Topics

- **[Multi-Agent Teams](MULTI_AGENT_TEAMS.md)**: Team formations and patterns
- **[Quickstart](multi-agent-quickstart.md)**: Fast start for teams

### Example Multi-Agent Team

```python
from victor.framework import Agent, AgentTeam

# Create specialized agents
frontend = Agent(
    role="Frontend developer",
    tools=["react", "typescript", "tailwind"]
)

backend = Agent(
    role="Backend developer",
    tools=["fastapi", "sqlalchemy", "postgresql"]
)

tester = Agent(
    role="QA engineer",
    tools=["pytest", "selenium", "coverage"]
)

# Coordinate team
team = AgentTeam.hierarchical(
    lead="senior-developer",
    subagents=[frontend, backend, tester]
)

result = await team.run("Implement user registration feature")
```

[Full Multi-Agent Guide →](MULTI_AGENT_TEAMS.md)

## Observability

Monitor Victor's behavior and performance with the EventBus.

### Topics

- **[Event Bus](observability/event-bus.md)**: Event sourcing and subscriptions
- **[Metrics](observability/metrics.md)**: Metrics collection and analysis
- **[Overview](observability/index.md)**: Observability overview

### Example Event Monitoring

```python
from victor.core.events import EventBus, Event

def on_tool_execution(event: Event):
    print(f"Tool {event.data['tool_name']} executed")
    print(f"Duration: {event.data['duration_ms']}ms")
    print(f"Success: {event.data['success']}")

EventBus.subscribe("tool.execution", on_tool_execution)

# Now every tool execution will trigger this callback
```

[Full Observability Guide →](observability/index.md)

## Integration

Integrate Victor with other tools and platforms.

### Topics

- **[MCP Clients](integration/mcp-clients.md)**: Using Victor as MCP server
- **[MCP Server](VICTOR_AS_MCP_SERVER.md)**: Run Victor as MCP server

### Example: GitHub Actions

```yaml
name: Code Review
on:
  pull_request:
    types: [opened, synchronize]

jobs:
  review:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Install Victor
        run: pipx install victor-ai

      - name: Run Code Review
        run: victor chat "Review this PR" --mode plan

      - name: Comment on PR
        uses: actions/github-script@v6
        with:
          script: |
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: '${{ steps.review.outputs.result }}'
            })
```

[MCP Clients Guide →](integration/mcp-clients.md)

## Development Setup

Guides for setting up and configuring development environments.

### Topics

- **[Local Models](development/local-models.md)**: Ollama, LM Studio, vLLM setup
- **[Embeddings](development/embeddings.md)**: Vector database and embeddings setup
- **[Air-Gapped](development/AIRGAPPED.md)**: Offline operation setup

### Example: Local Model Setup

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a code-focused model
ollama pull qwen2.5-coder:7b

# Run Victor with local model
victor chat --provider ollama --model qwen2.5-coder:7b
```

[Full Development Setup →](../development/setup.md)

## Performance

Optimize Victor for performance and efficiency.

### Topics

- **Startup Time**: Lazy loading and initialization
- **Tool Execution**: Caching and optimization
- **Memory Usage**: Resource management
- **Provider Selection**: Choosing the right provider
- **Workflow Optimization**: Efficient workflow design

### Quick Tips

**1. Use Lazy Loading**
```python
from victor.tools import LazyToolRunnable

# Tools load on-demand
tool = LazyToolRunnable("expensive_tool.Tool")
```

**2. Cache Tool Results**
```yaml
# In workflow config
nodes:
  - id: expensive_operation
    cache:
      enabled: true
      ttl: 3600  # Cache for 1 hour
```

**3. Profile Workflows**
```bash
victor workflow run my-workflow --profile
```

[Performance Benchmarks →](BENCHMARKING.md)

## Security

Security best practices for using Victor.

### Topics

- **API Key Management**: Secure key storage
- **Code Review**: Security-focused code review
- **Data Privacy**: Handling sensitive data
- **Audit Logging**: Tracking operations
- **Compliance**: SOC2, GDPR considerations

### Best Practices

**1. Never commit API keys**
```bash
# Use environment variables
export ANTHROPIC_API_KEY=sk-...

# Or use ~/.victor/profiles.yaml
profiles:
  secure:
    provider: anthropic
    api_key_env: ANTHROPIC_API_KEY
```

**2. Use project context for security**
```markdown
# .victor.md

When reviewing authentication code:
- Check for SQL injection vulnerabilities
- Verify password hashing (bcrypt/argon2)
- Ensure CSRF protection
- Validate input sanitization
```

**3. Audit workflow execution**
```python
from victor.core.events import EventBus

def audit_workflow(event):
    log = {
        "workflow": event.data["workflow_id"],
        "user": event.data["user"],
        "timestamp": event.timestamp,
        "files_modified": event.data["files"]
    }
    write_audit_log(log)

EventBus.subscribe("workflow.complete", audit_workflow)
```

[Shell Safety Policy →](https://github.com/anvai-labs/victor/blob/develop/feps/fep-0013-shell-safety-policy.md)

## Resilience

Error handling patterns for reliable workflows.

### Topics

- **Retry Strategies**: Exponential backoff, circuit breakers
- **Error Recovery**: Graceful degradation
- **Fallback Mechanisms**: Alternative providers
- **Timeout Handling**: Managing long-running operations
- **Data Validation**: Input validation patterns

### Example: Resilient Workflow

```yaml
workflow: ResilientDataProcessing

nodes:
  - id: fetch_data
    type: compute
    retry:
      max_retries: 3
      backoff: exponential
      base_delay: 1.0

  - id: process_data
    type: agent
    fallback:
      node: process_data_simple

  - id: process_data_simple
    type: compute
    description: Fallback simple processing

edges:
  - source: fetch_data
    target: process_data
```

[Full Resilience Guide →](RESILIENCE.md)

## Human-in-the-Loop Workflows

Patterns for workflows that require human approval or input.

### Patterns

- **Approval Gates**: Require human approval before proceeding
- **Interactive Prompts**: Ask for user input during execution
- **Review and Revise**: Human reviews AI output
- **Exception Handling**: Human intervention on errors

### Example: Approval Workflow

```yaml
workflow: DeploymentApproval

nodes:
  - id: plan_deployment
    type: agent
    role: Plan deployment strategy

  - id: approve
    type: hitl
    prompt: Review the deployment plan and approve
    options:
      - "Approve and deploy"
      - "Request changes"
      - "Cancel deployment"

  - id: deploy
    type: compute
    condition: approve.choice == "Approve and deploy"
    tools: [kubectl, helm]

edges:
  - source: plan_deployment
    target: approve
  - source: approve
    target: deploy
```

[Full HITL Guide →](HITL_WORKFLOWS.md)

## Common Patterns

### 1. Sequential Tool Execution

```python
from victor.tools import pipe

result = pipe(
    read_file,
    analyze_code,
    suggest_improvements,
    write_report
)
```

### 2. Parallel Tool Execution

```python
from victor.tools import parallel

results = parallel(
    run_tests,
    run_linter,
    run_coverage,
    run_typecheck
)
```

### 3. Conditional Workflow

```yaml
workflow: ConditionalTest

nodes:
  - id: check_tests
    type: compute
    tools: [pytest]

  - id: run_coverage
    type: compute
    condition: check_tests.success
    tools: [coverage]

  - id: report_failure
    type: agent
    condition: not check_tests.success
    role: Analyze test failure

edges:
  - source: check_tests
    target: run_coverage
  - source: check_tests
    target: report_failure
```

### 4. Multi-Provider Fallback

```python
from victor.agent import Orchestrator

orchestrator = Orchestrator()

# Try Anthropic, fallback to OpenAI, then local
orchestrator.set_providers([
    "anthropic",
    "openai",
    "ollama"
])

result = await orchestrator.run("Analyze this code")
```

## Additional Resources

- **User Guide**: [Daily Usage →](../user-guide/index.md)
- **Reference**: [Providers →](../reference/providers/index.md)
- **Reference**: [Tools →](../reference/tools/catalog.md)
- **Reference**: [Configuration →](../reference/configuration/index.md)
- **Development**: [Contributing →](https://github.com/anvai-labs/victor/blob/develop/CONTRIBUTING.md)
- **Operations**: [Deployment →](../development/releasing/publishing.md)

---

**Next**: [Workflow DSL Guide →](workflow-development/dsl.md)
