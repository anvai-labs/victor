# Graph Extension Guide

This guide explains how to extend Victor's graph capabilities with enhanced CCG (Code Context Graph) builders through external packages like victor-coding.

## Overview

Victor Core provides a generic graph storage and analysis framework. External packages can enhance these capabilities by registering language-specific CCG builders via the capability registry.

**Note**: This guide is about **Code Graph RAG** (multi-hop code symbol traversal), not **Document RAG** (victor-rag package for PDF/Markdown ingestion).

### Two RAG Systems in Victor

| Feature | Graph RAG (Core) | Document RAG (victor-rag) |
|---------|-----------------|---------------------------|
| **Location** | `victor/core/graph_rag/` | External package |
| **Purpose** | Code symbol relationships | Document chunks |
| **Data** | CFG, CDG, DDG graphs | Vector embeddings |
| **Retrieval** | Multi-hop traversal | Vector similarity |
| **Tools** | `graph_semantic_search`, `impact_analysis` | `rag_ingest`, `rag_search` |
| **Use Case** | Code analysis, impact detection | Knowledge management |

Code-graph retrieval and document RAG have different indexing inputs and domain policies.
Both consume storage interfaces; use the [canonical persistence diagram](../architecture.md#database-architecture)
for runtime ownership and the [graph retrieval guide](graph-rag-guide.md) for code context.
The external RAG vertical declares its capabilities through `victor_contracts`.

## Core Graph Components

### GraphStore Protocol

Located in `victor/storage/graph/protocol.py`:

```python
@runtime_checkable
class GraphStoreProtocol(Protocol):
    """Protocol for graph storage backends."""

    async def upsert_nodes(self, nodes: Iterable[Any]) -> None: ...

    async def upsert_edges(self, edges: Iterable[Any]) -> None: ...

    async def get_neighbors(self, node_id: str, edge_type: Optional[str] = None) -> List[Any]: ...

    async def find_nodes(self, **kwargs: Any) -> List[Any]: ...
```

### Edge Types

Located in `victor/storage/graph/edge_types.py`:

- **Legacy**: `CALLS`, `REFERENCES`, `CONTAINS`, `INHERITS`
- **CFG** (Control Flow): `CFG_SUCCESSOR`, `CFG_TRUE`, `CFG_FALSE`, `CFG_CASE`, `CFG_DEFAULT`
- **CDG** (Control Dependence): `CDG`, `CDG_LOOP`
- **DDG** (Data Dependence): `DDG_DEF_USE`, `DDG_RAW`, `DDG_WAR`, `DDG_WAW`
- **Requirement**: `SATISFIES`, `TESTS`, `DERIVES_FROM`
- **Semantic**: `SEMANTIC_SIMILAR`

### CCGBuilderProtocol

Located in `victor/framework/vertical_protocols.py`:

```python
@runtime_checkable
class CCGBuilderProtocol(Protocol):
    """Protocol for language-specific CCG builders."""

    async def build_ccg_for_file(
        self,
        file_path: Path,
        language: str | None = None,
    ) -> Tuple[List[Any], List[Any]]:
        """Build CCG for a file.

        Returns:
            Tuple of (nodes, edges) representing the CCG
        """
        ...

    def supports_language(self, language: str) -> bool:
        """Check if this builder supports the given language."""
        ...
```

## Extending with Enhanced CCG Builders

### Step 1: Implement CCGBuilderProtocol

Create a language-specific CCG builder in your external package:

```python
# victor_coding/ccg/python.py
from pathlib import Path
from typing import Any, List, Optional, Tuple

class PythonCCGBuilder:
    """Enhanced Python-specific CCG builder."""

    def __init__(self, graph_store=None):
        self.graph_store = graph_store
        self.language = "python"

    async def build_ccg_for_file(
        self,
        file_path: Path,
        language: Optional[str] = None,
    ) -> Tuple[List[Any], List[Any]]:
        """Build enhanced CCG for Python files.

        Enhancements over core:
        - Decorator-aware control flow
        - Type annotation tracking
        - Context manager analysis
        - Generator/async handling
        """
        # Parse with tree-sitter
        # Build CFG with Python-specific handling
        # Build CDG with post-dominance
        # Build DDG with type-aware def-use chains
        return nodes, edges

    def supports_language(self, language: str) -> bool:
        return language.lower() == "python"
```

### Step 2: Register via Plugin

```python
# victor_coding/plugin.py
from victor_contracts import VictorPlugin, PluginContext

class CodingPlugin(VictorPlugin):
    @property
    def name(self) -> str:
        return "coding"

    def register(self, context: PluginContext) -> None:
        from victor_coding.ccg.python import PythonCCGBuilder
        from victor_coding.ccg.javascript import JavaScriptCCGBuilder
        from victor_coding.ccg.typescript import TypeScriptCCGBuilder

        # Register enhanced CCG builders
        context.register_ccg_builder("python", PythonCCGBuilder())
        context.register_ccg_builder("javascript", JavaScriptCCGBuilder())
        context.register_ccg_builder("typescript", TypeScriptCCGBuilder())
```

### Step 3: Package Entry Point

Add to `pyproject.toml`:

```toml
[project.entry-points."victor.plugins"]
coding = "victor_coding:CodingPlugin"
```

## Core Fallback Behavior

The `CodeContextGraphBuilder` in core automatically:

1. Checks `CapabilityRegistry` for enhanced builders
2. Uses enhanced builder if available and language matches
3. Falls back to built-in implementation if:
   - No enhanced builder registered
   - Enhanced builder doesn't support the language
   - Enhanced builder raises an exception

```python
# victor/core/indexing/ccg_builder.py
class CodeContextGraphBuilder:
    def __init__(self, graph_store=None, language="python"):
        # Try to get enhanced builder from capability registry
        self._enhanced_builder = self._get_enhanced_builder(language)

    async def build_ccg_for_file(self, file_path, language=None):
        # Delegate to enhanced builder if available
        if self._enhanced_builder:
            try:
                return await self._enhanced_builder.build_ccg_for_file(file_path, language)
            except Exception as e:
                logger.warning(f"Enhanced CCG builder failed: {e}, falling back to built-in")

        # Use built-in implementation
        return await self._build_builtin_ccg(file_path, language)
```

## Available Extension Points

### 1. Graph Storage Extensions

Implement custom graph storage backends:

```python
from victor.storage.graph.protocol import GraphStoreProtocol

class CustomGraphStore:
    async def upsert_nodes(self, nodes): ...
    async def upsert_edges(self, edges): ...
    # ...
```

### 2. Graph Algorithm Extensions

Add custom graph traversal/analysis algorithms:

```python
from victor.processing.graph_algorithms import GraphAlgorithm

class CustomCentralityAlgorithm(GraphAlgorithm):
    def compute(self, graph): ...
```

### 3. Edge Type Extensions

Define custom edge types for domain-specific relationships:

```python
# In your package
from victor.storage.graph.edge_types import EdgeType

EdgeType.CUSTOM_RELATION = "CUSTOM_RELATION"
```

### 4. Graph Query Extensions

Create domain-specific graph query tools:

```python
from victor.tools import tool

@tool(name="custom_graph_query")
async def custom_graph_query(query: str, path: str = "."):
    """Custom domain-specific graph query."""
    # Query the graph
    return results
```

## Testing Extensions

### Unit Testing Enhanced Builders

```python
import pytest
from victor.core.capability_registry import CapabilityRegistry, CapabilityStatus
from victor.framework.vertical_protocols import CCGBuilderProtocol

@pytest.mark.asyncio
async def test_enhanced_builder():
    # Create enhanced builder
    class MyEnhancedBuilder:
        def supports_language(self, lang):
            return lang == "python"

        async def build_ccg_for_file(self, file_path, language=None):
            return [], []

    # Register
    CapabilityRegistry.reset()
    registry = CapabilityRegistry.get_instance()
    registry.register(CCGBuilderProtocol, MyEnhancedBuilder(), CapabilityStatus.ENHANCED)

    # Create CCG builder - should use enhanced
    from victor.core.indexing.ccg_builder import CodeContextGraphBuilder
    builder = CodeContextGraphBuilder(language="python")

    # Verify enhanced builder is used
    assert builder._enhanced_builder is not None

    # Cleanup
    CapabilityRegistry.reset()
```

### Integration Testing

```python
@pytest.mark.asyncio
async def test_full_graph_rag_pipeline():
    # 1. Register enhanced builder
    # 2. Create graph store
    # 3. Index codebase
    # 4. Query with graph RAG
    # 5. Verify results
```

## Best Practices

1. **Language Detection**: Use file extensions for auto-detection
2. **Error Handling**: Always fall back gracefully
3. **Performance**: Cache parsed ASTs when possible
4. **Testing**: Test both enhanced and fallback paths
5. **Documentation**: Document language-specific enhancements

## Backward Compatibility

- Core works without any external packages
- Enhanced builders are optional
- Fallback to built-in implementation is automatic
- Existing code continues to work unchanged

## References

- `victor/storage/graph/protocol.py` - Graph storage protocols
- `victor/storage/graph/edge_types.py` - Edge type definitions
- `victor/core/indexing/ccg_builder.py` - Core CCG builder
- `victor/core/graph_rag/` - Graph RAG pipeline
- `victor/processing/graph_algorithms.py` - Graph algorithms
