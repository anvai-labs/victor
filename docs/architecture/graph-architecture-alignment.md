# Graph Enhancements Architecture Alignment

> **Historical design/review snapshot.** The findings, proposed APIs, diagrams, and rollout
> status below describe their original review context. For current implementation and remaining
> work, use [Architecture](../architecture.md) and the [Roadmap](../roadmap.md); this document
> is not an installation guide or a certification that every proposed feature shipped.

## Current State Analysis

### 1. Tool Exposure Issue

**Problem**: The `graph_query_tool.py` file defines tools but they are NOT properly decorated with `@tool`, so they are NOT available to the LLM.

**Current Code**:
```python
# victor/tools/graph_query_tool.py
async def graph_query(...) -> Dict[str, Any]:  # Missing @tool decorator
    """Query codebase graph..."""
```

**Required Fix**:
```python
from victor.tools.decorators import tool

@tool(
    name="graph_query",
    category="analysis",
    keywords=["code graph", "impact analysis", "dependencies"],
    use_cases=["Understanding code structure", "Analyzing change impact"],
    priority=Priority.MEDIUM,
    access_mode=AccessMode.READONLY,
)
async def graph_query(...) -> Dict[str, Any]:
    """Query codebase graph..."""
```

### 2. Indexing Architecture: Core vs Vertical

**Current Situation**:
- Core has: `victor/contrib/codebase/indexer.py` (NullCodebaseIndexFactory stub)
- New work added: `victor/core/graph_rag/` and `victor/core/indexing/ccg_builder.py`
- External victor-coding expected to provide actual indexing

**Architectural Question**: Should graph indexing be in core or victor-coding?

### 3. Plugin Architecture Alignment

**Current Plugin Flow**:
```
victor-coding package:
  ├─ victor_plugin.py → VictorPlugin.register(context)
  ├─ CodingVertical → VerticalBase
  └─ get_tools() → ["read", "write", "search", ...]
```

## Recommended Architecture

### Principle: Core Provides Capabilities, Verticals Provide Domain Logic

```
┌─────────────────────────────────────────────────────────────┐
│                    victor-ai (core)                         │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Core Capabilities (reusable across domains)          │  │
│  │  - GraphStore (SQLite, Memory)                        │  │
│  │  - Graph protocol (GraphNode, GraphEdge)              │  │
│  │  - Edge types (EdgeType enum)                         │  │
│  │  - CCG Builder (language-agnostic)                    │  │
│  │  - Graph algorithms (NetworkX wrapper)                │  │
│  │  - Multi-hop retrieval framework                      │  │
│  │  - @tool decorators for registration                  │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ Uses
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                  victor-coding (external)                    │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Coding-Specific Logic                                │  │
│  │  - Python/JS/TS language specifics                     │  │
│  │  - Code patterns (API endpoints, classes)             │  │
│  │  - Testing patterns                                    │  │
│  │  - Refactoring heuristics                              │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                               │
│  Plugin Integration:                                         │
│  - register(CodingVertical)                                 │
│  - get_tools() includes "graph_query", "impact_analysis"    │
│  - get_extensions() provides coding-specific graph handlers  │
└─────────────────────────────────────────────────────────────┘
```

## Action Items

### 1. Fix Tool Registration (Core)

File: `victor/tools/graph_query_tool.py`

```python
from victor.tools.decorators import tool
from victor.tools.base import AccessMode, DangerLevel, Priority

@tool(
    name="graph_query",
    category="analysis",
    keywords=["graph", "dependency", "code structure", "impact"],
    use_cases=[
        "Understanding code dependencies",
        "Finding related code",
        "Analyzing change impact"
    ],
    priority=Priority.MEDIUM,
    access_mode=AccessMode.READONLY,
    danger_level=DangerLevel.SAFE,
    task_types=["analysis", "search"],
    execution_category="read_only",
)
async def graph_query(
    query: str,
    path: str = ".",
    mode: str = "semantic",
    max_hops: int = 2,
    max_results: int = 10,
) -> Dict[str, Any]:
    """Query codebase graph using natural language..."""

@tool(
    name="impact_analysis",
    category="analysis",
    keywords=["impact", "change", "break", "affect", "dependency"],
    use_cases=[
        "Analyzing change impact before refactoring",
        "Finding what depends on a function",
        "Understanding upstream dependencies"
    ],
    priority=Priority.HIGH,
    access_mode=AccessMode.READONLY,
    danger_level=DangerLevel.SAFE,
    task_types=["analysis"],
    execution_category="read_only",
)
async def impact_analysis(
    target: str,
    analysis_type: str = "forward",
    max_depth: int = 3,
    path: str = ".",
) -> Dict[str, Any]:
    """Analyze impact of code changes using CCG..."""
```

### 2. Register Tools in Tool Presets

File: `victor/tools/base.py` (ToolRegistry presets)

```python
@staticmethod
def with_graph_tools() -> "ToolRegistry":
    """Preset with graph analysis tools."""
    registry = ToolRegistry.default()
    # Tools are auto-registered via @tool decorator
    return registry
```

### 3. Update Coding Vertical to Include Graph Tools

File: `victor-coding/victor_coding/vertical.py` (external package)

```python
class CodingVertical(VerticalBase):
    @classmethod
    def get_tools(cls) -> List[str]:
        return [
            "read", "write", "search",
            "graph_query",      # NEW
            "impact_analysis",  # NEW
            # ... other coding tools
        ]
```

### 4. Core: Keep Language-Agnostic, Move Specifics to Vertical

**In Core** (`victor/core/indexing/ccg_builder.py`):
```python
class CodeContextGraphBuilder:
    """Language-agnostic CCG builder with pluggable parsers."""

    def __init__(self, graph_store, language: str):
        self.language = language
        self.parser = self._get_parser(language)  # Pluggable
```

**In victor-coding**:
```python
# victor-coding provides Python-specific enhancements
class PythonCCGExtension:
    """Python-specific CCG enhancements."""

    @staticmethod
    def enhance_cc_nodes(nodes, ast_root):
        """Add Python-specific context (decorators, etc.)."""
```

### 5. Plugin Registration Pattern

File: `victor-coding/victor_coding/__init__.py`

```python
from victor_contracts.core.plugins import VictorPlugin
from victor_contracts.verticals.protocols.base import VerticalBase

class CodingPlugin(VictorPlugin):
    """Plugin registration for victor-coding."""

    @classmethod
    def register(cls, context: PluginContext) -> None:
        # Register vertical
        context.register_vertical(CodingVertical)

        # Register graph tools (if not auto-registered via @tool)
        context.register_tool("graph_query", graph_query)
        context.register_tool("impact_analysis", impact_analysis)

        # Register CCG enhancements
        context.register_ccg_handler("python", PythonCCGHandler)
```

## Migration Path

### Phase 1: Fix Tool Registration (Immediate)
1. Add `@tool` decorators to `graph_query_tool.py`
2. Export tools in `victor/tools/__init__.py`
3. Verify tools appear in system prompt

### Phase 2: Clarify Core vs Vertical (Short-term)
1. Core keeps: GraphStore, protocol, edge types, base CCG builder
2. victor-coding adds: Language-specific handlers, coding patterns
3. Document separation clearly

### Phase 3: Plugin Integration (Medium-term)
1. victor-coding implements VictorPlugin
2. Registers tools and extensions via PluginContext
3. Core provides PluginContext hooks for graph capabilities

## Testing Checklist

- [ ] Tools appear in `victor tools list`
- [ ] Tools appear in system prompt for coding vertical
- [ ] LLM can successfully call `graph_query` tool
- [ ] LLM can successfully call `impact_analysis` tool
- [ ] victor-coding can extend core graph capabilities
- [ ] Plugin registration doesn't break existing functionality
