# ADR-004: Tool System Architecture

## Metadata

- **Status**: Accepted
- **Date**: 2025-02-26
- **Decision Makers**: Vijaykumar Singh
- **Related ADRs**: ADR-001 (Agent Orchestration)

## Context

Victor needs a tool system that:
- Provides 33+ built-in tools across 9 categories
- Enables easy tool discovery and selection
- Supports progressive disclosure (cost-based tiers)
- Allows custom tool creation
- Ensures tool safety through validation

The challenge is:
- Organizing tools across multiple categories
- Balancing power with safety
- Enabling extensibility without complexity
- Optimizing for performance

## Decision

We will implement a **Three-Phase Tool System** with metadata-driven discovery and progressive disclosure.

### Architecture

```
Tool Category Registry
    ↓
Tool Metadata Registry (Decorator-driven)
    ↓
Progressive Tools Registry (Cost tiers)
    ↓
Shared Tool Registry (Singleton instances)
```

### Tool Categories

1. **CORE**: Essential tools (read, write, edit)
2. **FILESYSTEM**: File operations (ls, grep, find)
3. **GIT**: Version control (git_status, git_commit)
4. **SEARCH**: Code search (code_search, overview)
5. **WEB**: Web access (web_search, web_fetch)
6. **DATABASE**: Database operations (query, execute)
7. **DOCKER**: Container management (docker_run, docker_exec)
8. **TESTING**: Test execution (pytest, test)
9. **REFACTORING**: Code refactoring (rename, extract)

### Cost Tiers (Progressive Disclosure)

- **LOW**: Safe, fast, free tools
- **MEDIUM**: Moderate cost, some restrictions
- **HIGH**: Expensive or dangerous tools
- **PROHIBITED**: Blocked by safety rules

### Tool Metadata

```python
@dataclass
class ToolMetadata:
    name: str
    category: ToolCategory
    description: str
    cost_tier: CostTier
    idempotent: bool
    keywords: List[str]
    use_cases: List[str]
    schema: Dict[str, Any]
```

## Rationale

### Why Three Registries?

**Tool Category Registry**:
- High-level organization
- Category-based tool selection
- Preset configurations (minimal, default, full)

**Tool Metadata Registry**:
- Decorator-driven registration
- Rich metadata for discovery
- Semantic search capabilities

**Progressive Tools Registry**:
- Cost-based tool disclosure
- Safety controls
- Resource management

### Why Progressive Disclosure?

**Benefits**:
- Safer for new users
- Cost control
- Gradual complexity
- Better UX

**Implementation**:
- Start with LOW tier tools
- Unlock higher tiers as needed
- Explicit opt-in for dangerous tools

### Why Metadata-Driven?

**Benefits**:
- Self-documenting tools
- Searchable by keywords
- Auto-generated documentation
- Type-safe validation

## Consequences

### Positive

- **Discoverable**: Rich metadata enables search and discovery
- **Safe**: Progressive disclosure prevents accidents
- **Extensible**: Easy to add custom tools
- **Organized**: Clear categorization
- **Observable**: Tool execution tracked

### Negative

- **Metadata Overhead**: Need to maintain tool metadata
- **Discovery Complexity**: Multiple registries can confuse
- **Registration**: Tools must be explicitly registered

### Neutral

- **Performance**: Metadata lookup is fast (O(1))
- **API**: Tool registration is simple decorator
- **Compatibility**: Existing tools continue to work

## Implementation

### Phase 1: Core Tool System (Completed)

- ✅ Base tool abstract class
- ✅ Tool metadata protocol
- ✅ Category registry
- ✅ 33+ built-in tools
- ✅ Progressive disclosure

### Phase 2: Enhanced Features (In Progress)

- 🔄 Tool validation
- 🔄 Tool aliases
- 🔄 Tool composition
- 🔄 Tool permissions

### Phase 3: Advanced Features (Planned)

- ⏳ Tool marketplace
- ⏳ Tool versioning
- ⏳ Tool dependencies
- ⏳ Tool sandboxing

## Code Example

### Creating a Tool

```python
from victor.tools.base import BaseTool, ToolMetadata, ToolCategory, CostTier

class MyCustomTool(BaseTool):
    """A custom tool example."""

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="my_tool",
            category=ToolCategory.CUSTOM,
            description="Description of my tool",
            cost_tier=CostTier.LOW,
            idempotent=True,
            keywords=["custom", "example"],
            use_cases=["Use case 1", "Use case 2"],
            schema={
                "type": "object",
                "properties": {
                    "input": {"type": "string"}
                }
            }
        )

    async def execute(self, **kwargs) -> str:
        """Execute the tool."""
        return f"Processed: {kwargs.get('input')}"
```

### Using Tool Presets

```python
from victor import Agent

# Minimal tools (safe only)
agent = Agent.create(tools="minimal")

# Default tools (safe filesystem)
agent = Agent.create(tools="default")

# Full tools (all available)
agent = Agent.create(tools="full")

# Custom selection
agent = Agent.create(tools=["read", "write", "grep"])
```

### Progressive Disclosure

```python
# Start with LOW tier tools
agent = Agent.create(
    tools=["read", "write"],
    max_cost_tier=CostTier.LOW
)

# User requests more capabilities
agent.add_tools(["shell", "docker"])
```

## Alternatives Considered

### 1. Flat Tool List

**Description**: Single list of all tools, no organization

**Rejected Because**:
- Hard to discover relevant tools
- No progressive disclosure
- Safety concerns

### 2. Plugin System

**Description**: Dynamic tool loading from plugins

**Rejected Because**:
- More complex
- Security concerns
- Slower discovery

### 3. No Metadata

**Description**: Tools registered without metadata

**Rejected Because**:
- Hard to discover
- No searchability
- Poor documentation

## References

- [Tool Design Patterns](https://martinfowler.com/bliki/ToolDefinition.html)
- [Progressive Disclosure](https://en.wikipedia.org/wiki/Progressive_disclosure)
- [Victor Tools](https://github.com/anvai-labs/victor/tree/develop/victor/tools/)

## Revision History

| Date | Version | Changes | Author |
|------|---------|---------|--------|
| 2025-02-26 | 1.0 | Initial ADR | Vijaykumar Singh |
