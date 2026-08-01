# Verticals - Domain-Specific Assistants

Verticals are pre-configured assistant templates optimized for specific domains. Each
vertical defines tool sets, stage configurations, system prompts, and evaluation
criteria tailored to its use case.

## Overview

Victor's runtime vertical system still exposes a **Template Method Pattern**
compatibility surface, but new external vertical packages should be authored against
`victor_contracts.VerticalBase` and published via `victor.plugins`.

> External authoring path: define the vertical in `victor-contracts`, decorate it with
> `@register_vertical`, and publish a thin `VictorPlugin` wrapper through
> `victor.plugins`. This page focuses on runtime consumption plus compatibility
> surfaces that still exist inside Victor. See `../guides/vertical-quickstart.md`
> for the canonical authoring flow.

```
┌─────────────────────────────────────────────────────────────────┐
│                      VerticalBase (Abstract)                     │
├─────────────────────────────────────────────────────────────────┤
│  get_tools()          → List of tool names                      │
│  get_system_prompt()  → Domain-specific instructions            │
│  get_stages()         → Custom stage definitions                │
│  get_config()         → Complete VerticalConfig                 │
└─────────────────────────────────────────────────────────────────┘
                              ▲
    ┌─────────────┬───────────┼───────────┬──────────────┐
    │             │           │           │              │
┌───┴───┐    ┌────┴────┐  ┌───┴───┐  ┌────┴────┐  ┌──────┴──────┐
│Coding │    │Research │  │DevOps │  │  Data   │  │   Custom    │
│       │    │         │  │       │  │Analysis │  │ (your own)  │
└───────┘    └─────────┘  └───────┘  └─────────┘  └─────────────┘
```

## Available Verticals

### CodingAssistant

The default Victor vertical, optimized for software development tasks.

```python
from victor.verticals import CodingAssistant

config = CodingAssistant.get_config()
# 30 tools: filesystem, git, shell, code analysis, web search
# 7 stages: INITIAL → PLANNING → READING → ANALYSIS → EXECUTION → VERIFICATION → COMPLETION
```

**Capabilities:**
- Code reading, writing, editing
- Git operations (commit, branch, diff)
- Shell command execution
- Code search (semantic and keyword)
- Web search for documentation
- Refactoring assistance

**System Prompt Focus:**
- Clean code principles
- Security awareness
- Test-driven development
- Documentation best practices

### ResearchAssistant

Optimized for web research and document analysis.

```python
from victor.verticals import ResearchAssistant

config = ResearchAssistant.get_config()
# 9 tools: web_search, web_fetch, read, write, grep, ls, etc.
# 4 stages: SEARCHING → READING → SYNTHESIZING → WRITING
```

**Capabilities:**
- Web search and content fetching
- Document reading and analysis
- Summary generation
- Report writing

**System Prompt Focus:**
- Source verification
- Balanced perspectives
- Citation practices
- Structured output

### DevOpsAssistant

Optimized for infrastructure and deployment tasks.

```python
from victor.verticals import DevOpsAssistant

config = DevOpsAssistant.get_config()
# 13 tools: docker, shell, git, test, web_search, web_fetch, etc.
# 8 stages: INITIAL → ASSESSMENT → PLANNING → IMPLEMENTATION → VALIDATION → DEPLOYMENT → MONITORING → COMPLETION
```

**Capabilities:**
- Docker container management
- CI/CD pipeline configuration
- Infrastructure as Code (Terraform, Ansible)
- Kubernetes deployments
- Monitoring setup

**System Prompt Focus:**
- Security-first approach
- Idempotent operations
- Infrastructure as Code
- No hardcoded secrets

### DataAnalysisAssistant

Optimized for data science and analysis tasks.

```python
from victor.verticals import DataAnalysisAssistant

config = DataAnalysisAssistant.get_config()
# 11 tools: read, write, shell, graph, grep, ls, overview, web_search, web_fetch, etc.
# Stages: LOADING → CLEANING → ANALYSIS → VISUALIZATION → COMPLETION
```

**Capabilities:**
- Data loading and examination
- Data cleaning and transformation
- Statistical analysis with pandas/numpy
- Visualization with matplotlib/seaborn
- Report generation

**System Prompt Focus:**
- Data quality and validation
- Statistical rigor
- Clear visualizations
- Reproducible analysis

### RAGAssistant

Optimized for Retrieval-Augmented Generation (RAG) workflows - document ingestion, vector search, and Q&A.

```python
from victor.verticals import RAGAssistant

config = RAGAssistant.get_config()
# 10 tools: rag_ingest, rag_search, rag_query, rag_list, rag_delete, rag_stats, read, ls, web_fetch, shell
# Stages: INITIAL → INGESTING → SEARCHING → QUERYING → SYNTHESIZING
```

**Capabilities:**
- Document ingestion from files (PDF, Markdown, Text, Code)
- URL/web content ingestion with HTML text extraction
- Directory batch ingestion with glob patterns
- LanceDB vector storage (embedded, no server required)
- Hybrid search combining vector + full-text
- Query with automatic context retrieval
- Source attribution and citations

**System Prompt Focus:**
- Always search before answering
- Cite sources with document references
- No hallucination - stay grounded in documents
- Clear distinction between indexed vs. unknown info

**Demo Scripts:**
- SEC 10-K/10-Q filing ingestion for FAANG stocks
- Project documentation ingestion

See `../examples/README.md` for usage examples.

## Two Ways to Activate a Vertical

There is one vertical system with two entry points — both resolve the same
registry (verticals discovered via the `victor.plugins` entry point, so the
vertical package must be installed either way):

| Path | How | When to use |
|------|-----|-------------|
| **CLI** | `victor chat --vertical coding "..."` (or `-V coding`) | Interactive/terminal sessions; the flag applies the vertical's tools, prompts, and stages to the chat session |
| **Python API** | `agent = await Agent.create(vertical="coding")` | Embedding Victor in applications; accepts a vertical name string or a `VerticalBase` class, and applies the identical configuration |

Same name, same configuration, same behavior — `--vertical coding` and
`Agent.create(vertical="coding")` load the same vertical definition. The
sections below detail each path.

## CLI Usage

The `victor chat` command supports verticals via the `--vertical` (or `-V`) flag:

### Basic CLI Usage

```bash
# Default behavior (no vertical, uses standard CodingAssistant behavior)
victor chat "Write a function to sort a list"

# Specify a vertical
victor chat --vertical coding "Write unit tests for auth module"
victor chat -V research "Research latest AI trends"
victor chat -V devops "Setup GitHub Actions CI/CD"

# List available verticals
victor chat --help  # Shows available verticals in help text
```

### Available Verticals via CLI

| Vertical | Flag | Description |
|----------|------|-------------|
| coding | `--vertical coding` | Software development (default behavior) |
| research | `--vertical research` | Web research and document analysis |
| devops | `--vertical devops` | Infrastructure and deployment |
| data_analysis | `--vertical data_analysis` | Data science and analysis |
| rag | `--vertical rag` | Retrieval-Augmented Generation (document Q&A) |

### Observability Options

```bash
# With observability (default)
victor chat --observability "Your prompt"

# Without observability
victor chat --no-observability "Quick question"
```

Victor chat now always uses the canonical framework client path. There is no separate
legacy CLI mode.

## Using Verticals in Python

### Basic Usage

```python
from victor.framework import Agent
from victor.verticals import CodingAssistant, ResearchAssistant

# Get vertical configuration
config = CodingAssistant.get_config()

# Create agent with vertical's tools
agent = await Agent.create(
    provider="anthropic",
    tools=config.tools,
)
```

### With Full Vertical Config

```python
from victor.verticals import DevOpsAssistant

config = DevOpsAssistant.get_config()

# Access all vertical settings
print(config.name)           # "devops"
print(config.tools)          # ["read", "write", "shell", "docker", ...]
print(config.system_prompt)  # Full DevOps system prompt
print(config.stages)         # {"ASSESSMENT": StageDefinition(...), ...}
print(config.metadata)       # {"supports_docker": True, ...}
```

### Using VerticalRegistry (Runtime/Compatibility)

`VerticalRegistry` is still useful for runtime inspection and compatibility flows
inside Victor. New external packages should not perform raw registry registration;
they should publish a `VictorPlugin` and call `context.register_vertical()` from the
plugin wrapper.

```python
from victor.verticals import VerticalRegistry

# List available verticals
names = VerticalRegistry.list_names()  # ["coding", "research", "devops", "data_analysis"]

# Get vertical by name
vertical = VerticalRegistry.get("research")
config = vertical.get_config()
```

## Creating Custom Verticals

### Minimal Vertical

```python
from victor_contracts import (
    PluginContext,
    StageDefinition,
    ToolRequirement,
    VictorPlugin,
    VerticalBase,
    register_vertical,
)


@register_vertical(
    name="mlops",
    version="1.0.0",
    min_framework_version=">=0.6.0",
    plugin_namespace="mlops",
)
class MLOpsAssistant(VerticalBase):
    """Vertical for ML operations tasks."""

    name = "mlops"
    description = "Machine learning operations assistant"
    version = "1.0.0"

    @classmethod
    def get_tool_requirements(cls) -> list[ToolRequirement]:
        return [
            ToolRequirement("read", purpose="inspect project state"),
            ToolRequirement("write", required=False, purpose="update configs"),
            ToolRequirement("shell", required=False, purpose="run local commands"),
            ToolRequirement("docker", required=False, purpose="build and deploy images"),
        ]

    @classmethod
    def get_tools(cls) -> list[str]:
        return [requirement.tool_name for requirement in cls.get_tool_requirements()]

    @classmethod
    def get_system_prompt(cls) -> str:
        return """You are an MLOps assistant.
        Focus on: model training, deployment, monitoring.
        Use Python with MLflow, Docker, Kubernetes."""

    @classmethod
    def get_stages(cls) -> dict[str, StageDefinition]:
        return {
            "EXPLORATION": StageDefinition(
                name="EXPLORATION",
                description="Exploring model requirements",
                required_tools=["read"],
                optional_tools=["shell"],
                keywords=["explore", "requirements", "data"],
                next_stages={"TRAINING", "DEPLOYMENT"},
            ),
            "TRAINING": StageDefinition(
                name="TRAINING",
                description="Training and evaluating models",
                required_tools=["read"],
                optional_tools=["write", "shell"],
                keywords=["train", "evaluate", "model", "metrics"],
                next_stages={"DEPLOYMENT"},
            ),
        }


class MLOpsPlugin(VictorPlugin):
    @property
    def name(self) -> str:
        return "mlops"

    def register(self, context: PluginContext) -> None:
        context.register_vertical(MLOpsAssistant)
```

### Extending Existing Runtime Vertical (Compatibility/Internal)

If you are customizing an in-process runtime vertical that is already available
through Victor, subclassing a compatibility shim is still possible. New external
packages should generally prefer a contract-first vertical plus a thin plugin wrapper
instead of subclassing runtime classes from `victor.verticals`.

```python
from victor.verticals import CodingAssistant

class SecurityAuditAssistant(CodingAssistant):
    """Coding assistant with security focus."""

    name = "security_audit"
    description = "Security-focused code review assistant"

    @classmethod
    def get_tools(cls):
        # Extend parent tools
        base_tools = super().get_tools()
        return base_tools + ["security_scan", "dependency_audit"]

    @classmethod
    def get_system_prompt(cls):
        return super().get_system_prompt() + """

        SECURITY FOCUS:
        - Check for OWASP Top 10 vulnerabilities
        - Verify input validation
        - Review authentication/authorization
        - Check for hardcoded secrets
        """

    @classmethod
    def get_evaluation_criteria(cls):
        return super().get_evaluation_criteria() + [
            "Security vulnerability detection",
            "Secret exposure prevention",
            "Dependency vulnerability awareness",
        ]
```

## StageDefinition

For external vertical authors, `StageDefinition` is an SDK contract type:

```python
from victor_contracts import StageDefinition

stage = StageDefinition(
    name="ANALYSIS",
    description="Analyzing code patterns",
    required_tools=["read", "search"],
    optional_tools=["code_search"],
    keywords=["analyze", "understand", "examine"],
    next_stages={"EXECUTION", "PLANNING"},
)

# Convert to dict for serialization
stage_dict = stage.to_dict()
```

## VerticalConfig

The complete definition-layer configuration uses the SDK `VerticalConfig` contract:

```python
from victor_contracts import StageDefinition, VerticalConfig

config = VerticalConfig(
    name="my_vertical",
    description="Custom vertical",
    tools=["read", "write", "shell"],
    system_prompt="You are a helpful assistant...",
    stages={
        "STAGE1": StageDefinition(
            name="STAGE1",
            description="Primary working stage",
            required_tools=["read"],
            optional_tools=["write", "shell"],
        )
    },
    metadata={"custom_key": "value"},
)

# Inspect normalized contract helpers
tool_names = config.get_tool_names()
stage_names = config.get_stage_names()
config = config.with_metadata(owner="ml-platform")
```

## Provider Hints

Verticals can specify provider preferences:

```python
@classmethod
def get_provider_hints(cls):
    return {
        "preferred_providers": ["anthropic", "openai"],
        "preferred_models": ["claude-sonnet-4-20250514", "gpt-4-turbo"],
        "min_context_window": 100000,
        "requires_tool_calling": True,
        "prefers_extended_thinking": True,  # For complex tasks
    }
```

## Evaluation Criteria

Define how to evaluate vertical performance:

```python
@classmethod
def get_evaluation_criteria(cls):
    return [
        "Code correctness and functionality",
        "Test coverage",
        "Documentation quality",
        "Security best practices",
        "Performance considerations",
    ]
```

## Best Practices

1. **Keep tools focused** - Only include tools relevant to the domain
2. **Design clear stages** - Each stage should have a distinct purpose
3. **Write specific prompts** - Domain expertise in system prompt
4. **Define evaluation criteria** - Enable quality measurement
5. **Use provider hints** - Guide model selection for optimal results

## API Reference

### VerticalBase Methods

| Method | Description |
|--------|-------------|
| `get_tools()` | Return list of tool names |
| `get_system_prompt()` | Return system prompt string |
| `get_stages()` | Return dict of stage definitions |
| `get_provider_hints()` | Return provider preferences |
| `get_evaluation_criteria()` | Return evaluation criteria list |
| `get_config()` | Return complete VerticalConfig |
| `customize_config(config)` | Modify config before return |

### VerticalRegistry Methods

| Method | Description |
|--------|-------------|
| `register(vertical)` | Register a vertical class |
| `unregister(name)` | Remove a vertical |
| `get(name)` | Get vertical by name |
| `list_all()` | Get all registered verticals |
| `list_names()` | Get all vertical names |

## Multi-Provider Testing

Verticals have been tested across multiple LLM providers. Results from testing (December 2025):

### Test Matrix

| Vertical | OpenAI GPT-4o | DeepSeek R1 | Notes |
|----------|--------------|-------------|-------|
| Coding | ✅ Pass | ⚠️ Stream errors | OpenAI reliable; DeepSeek has stream handling issues |
| Research | ✅ Pass | ✅ Pass | Both providers work well |
| DevOps | ⚠️ Edit tool error | ✅ Pass | OpenAI missed `ops` parameter (now fixed) |
| Data Analysis | ✅ Pass | ⏱️ Timeout | DeepSeek times out on complex analysis |

### Key Findings

1. **Edit Tool Parameter Validation**: LLMs sometimes call `edit()` without the required `ops` parameter. This is now handled gracefully with helpful error messages and examples.

2. **Mode Exploration Multipliers**: Plan mode (2.5x) and Explore mode (3.0x) multipliers allow more thorough exploration before forcing completion, critical for complex verticals.

3. **Sandbox Editing**: Plan and Explore modes restrict file edits to `.victor/sandbox/` directory, preventing accidental modifications during exploration.

### Test Commands

```bash
# Test coding vertical with different providers
victor chat --vertical coding --provider openai "Write a function to validate email"
victor chat --vertical coding --provider deepseek "Refactor the auth module"

# Test research vertical
victor chat -V research --provider openai "Research GraphQL best practices"

# Test DevOps vertical
victor chat -V devops --provider deepseek "Setup Docker Compose for development"

# Test data analysis vertical
victor chat -V data_analysis --provider openai "Analyze CSV data trends"

# Use plan mode for thorough exploration
victor chat --mode plan --vertical coding "Understand the caching system"
```

### Provider-Specific Notes

**OpenAI (GPT-4o)**:
- Reliable tool calling with consistent parameter formatting
- Good at following structured output requirements

**DeepSeek (R1)**:
- Supports thinking tags (`<think>...</think>`)
- May require longer timeouts for complex tasks
- Occasional stream handling issues (handled by provider adapter)

**Anthropic (Claude)**:
- Best overall tool calling support
- Recommended for complex multi-tool workflows

## Related Documentation

- [State Machine →](../../development/architecture/state-machine.md) - Stage architecture details
- [Tool Catalog →](../../reference/tools/catalog.md) - Complete tool reference
- [Development Guide →](../../development/) - Framework entrypoints and structure
