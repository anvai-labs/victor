# Development Guide

**Everything you need to contribute to Victor.**

## Quick Start

```bash
# Clone and install
git clone https://github.com/anvai-labs/victor.git
cd victor
python -m venv .venv
source .venv/bin/activate
# victor-contracts first so victor-ai resolves the in-repo SDK, not PyPI
pip install -e ./victor-contracts -e ".[dev]"

# Run tests
make test
pytest tests/unit -v

# Format code
make format
```

## Documentation Map

```
docs/
├── README.md                          # Project overview
├── getting-started/                   # User onboarding
├── user-guide/                        # Daily usage
├── development/                       # Developer docs
│   ├── index.md                       # This file
│   └── testing/                       # Testing strategy
├── architecture/                      # System design (canonical)
│   ├── README.md                      # Architecture overview & index
│   ├── orchestrator_decomposition.md  # Service-first runtime
│   ├── data-flow-eventbus.md          # EventBus & data flow
│   └── state-machine.md               # Conversation stages
│   ├── extending/                     # Extension guides
│   │   ├── verticals.md               # Vertical development
│   │   └── plugins.md                 # Plugin system
│   ├── testing/                       # Testing strategy
│   ├── releasing/                     # Release process
│   └── PR_WORKFLOW.md                 # Pull request workflow guide
├── guides/                            # How-to guides
│   ├── vertical-quickstart.md         # Vertical quick reference
│   ├── tool-reference.md              # Tool catalog
│   ├── workflow-quickstart.md         # Workflow patterns
│   ├── multi-agent-quickstart.md      # Team coordination
│   ├── development/                   # Dev-specific guides
│   ├── observability/                 # Monitoring & debugging
│   ├── workflow-development/          # Workflow DSL
│   └── integration/                   # Integration guides
└── reference/                         # API & config reference
    ├── providers-comparison.md        # Provider matrix
    ├── api/                            # HTTP/MCP APIs
    ├── configuration/                  # Settings reference
    ├── providers/                      # Provider docs
    ├── tools/                          # Tool documentation
    └── verticals/                      # Built-in verticals
```

## Quick Reference

### Architecture

| Layer | Components | File |
|-------|------------|------|
| **Clients** | CLI, HTTP API, MCP | `victor/cli/`, `victor/integrations/` |
| **Orchestrator** | AgentOrchestrator, Controllers | `victor/agent/` |
| **Framework** | StateGraph, Workflows, Teams | `victor/framework/` |
| **Verticals** | 9 built-in + custom | `victor/{vertical}/` |
| **Providers** | 24 LLM providers | `victor/providers/` |
| **Tools** | 34 tool modules | `victor/tools/` |

### Verticals

| Vertical | Tools | Use Case |
|----------|-------|----------|
| **coding** | 30+ | Code analysis, refactoring, testing |
| **research** | 9 | Web search, synthesis, citations |
| **devops** | 13 | Docker, CI/CD, infrastructure |
| **data_analysis** | 11 | Pandas, visualization, statistics |
| **rag** | 10 | Document retrieval, vector search |

### Key Protocols

| Protocol | Purpose | Location |
|----------|---------|----------|
| `BaseProvider` | LLM abstraction | `victor/providers/base.py` |
| `BaseTool` | Tool interface | `victor/tools/base.py` |
| `VerticalBase` | Domain extension | `victor/core/verticals/base.py` |
| `CapabilityRegistryProtocol` | Capability discovery | `victor/framework/protocols.py` |

## Development Tasks

| Task | Command | Description |
|------|---------|-------------|
| **Run tests** | `make test` | Unit tests only |
| **Run all tests** | `make test-all` | Including integration |
| **Format code** | `make format` | Black + ruff |
| **Lint code** | `make lint` | Check formatting |
| **Type check** | `mypy victor` | Type validation |
| **Strict type gate** | `mypy --strict victor/config victor/storage/cache victor/telemetry victor/analytics victor/profiler victor/debug` | Mirrors CI blocking strict-package check |

## Extension Points

### Add a Provider

```python
from victor.providers.base import BaseProvider

class MyProvider(BaseProvider):
    @property
    def name(self) -> str:
        return "myprovider"

    async def chat(self, message: str, **kwargs) -> ChatResponse:
        # Implementation
        pass
```

[Provider Reference →](../reference/providers/)

### Add a Tool

```python
from victor.tools.base import BaseTool

class MyTool(BaseTool):
    name = "my_tool"
    description = "Does something useful"

    async def execute(self, **kwargs) -> ToolResult:
        # Implementation
        pass
```

**Tool Deduplication**: Native tools are automatically preferred over adapter tools (LangChain, MCP) when conflicts are detected. Tools are compared by normalized name (lowercase, separator normalization). Ensure your tool has a unique, descriptive name to avoid conflicts.

[Tool Catalog →](../reference/tools/catalog.md)
[Tool Deduplication →](../architecture/tool-deduplication.md)

### Create a Vertical

```python
from victor.core.verticals import VerticalBase

class MyVertical(VerticalBase):
    name = "my_vertical"

    @classmethod
    def get_tools(cls) -> list[str]:
        return ["read", "write", "grep"]

    @classmethod
    def get_system_prompt(cls) -> str:
        return "You are an expert in..."
```

[Vertical Quickstart →](../guides/vertical-quickstart.md)

### Create a Workflow

```yaml
workflows:
  my_workflow:
    nodes:
      - id: step1
        type: agent
        role: researcher
        goal: "Research the topic"
        next: [step2]

      - id: step2
        type: compute
        handler: summarize
        next: []
```

[Workflow Quickstart →](../guides/workflow-quickstart.md)

## Testing

| Test Type | Marker | Command |
|-----------|--------|---------|
| Unit | `@pytest.mark.unit` | `pytest -m unit` |
| Integration | `@pytest.mark.integration` | `pytest -m integration` |
| Slow | `@pytest.mark.slow` | `pytest -m "not slow"` |
| Workflow | `@pytest.mark.workflows` | `pytest -m workflows` |

[Testing Guide →](testing/strategy.md)

## Contributing

**IMPORTANT**: All changes must go through pull requests with CI/CD validation.

### PR Workflow

Victor uses a strict PR-based workflow to ensure code quality:

1. Create feature branch from `develop`
2. Make changes with tests
3. Run `make test && make lint`
4. Commit with conventional commits
5. Push and create PR: `feature` → `develop`
6. After review and merge to `develop`, create PR: `develop` → `main`
7. All status checks must pass before merging to `main`

[Detailed PR Workflow Guide →](PR_WORKFLOW.md)

[Contribution Workflow →](../../CONTRIBUTING.md)

## Resources

| Topic | Link |
|-------|------|
| Architecture Overview | [Architecture →](../architecture/README.md) |
| Service-First Runtime | [Orchestrator Decomposition →](../architecture/orchestrator_decomposition.md) |
| Provider Comparison | [Provider Matrix →](../reference/providers-comparison.md) |
| Tool Reference | [Tool Catalog →](../guides/tool-reference.md) |
| Multi-Agent Teams | [Team Quickstart →](../guides/multi-agent-quickstart.md) |
| Prompt Evolution | [Run & Promote Evolved Prompts →](prompt-evolution-workflow.md) |

---

**Next**: [Setup Guide →](setup.md)
