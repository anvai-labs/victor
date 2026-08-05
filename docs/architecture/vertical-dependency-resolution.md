# Vertical Dependency Resolution Architecture

**Status**: Implemented
**Version**: 1.0.0
**Created**: 2026-03-03
**Updated**: 2026-03-04
**Author**: Vijaykumar Singh <vijay@anvaiops.com>

## Executive Summary

This document addresses architectural issues caused by moving verticals to external packages (`victor-coding`, `victor-rag`, etc.). It provides design principles and a migration path to eliminate framework dependencies on external verticals while maintaining clean separation of concerns through protocol-based architecture and contrib packages.

## Key Architectural Principle: LSP (Language Server Protocol) Integration

The LSP integration demonstrates the core architectural pattern:

1. **Framework Protocol** (`victor.framework.vertical_protocols.LanguageServerProtocol`):
   - Defines the interface without any implementation
   - Zero dependencies on external packages
   - Provides stub types for LSP operations

2. **Contrib Package** (`victor.contrib.lsp.BasicLSPClient`):
   - Provides a default stub implementation
   - Returns empty results when no LSP server is available
   - Enables graceful degradation without errors

3. **Vertical Enhancement** (`victor-coding`):
   - Provides full LSP implementation via **Liskov Substitution Principle**
   - Can completely replace `BasicLSPClient` with advanced features
   - No framework code changes required

**This eliminates circular dependencies** - the framework never imports `victor-coding`, only protocols and contrib.

---

## Problem Statement

### Current Issues

1. **Framework Depends on External Verticals**: Framework code imports from `victor_coding`, creating circular dependencies when verticals are external packages.

2. **Tools Tightly Coupled to Verticals**: Tools like `file_editor_tool.py`, `lsp_tool.py`, `code_search_tool.py` import vertical-specific implementations.

3. **Graceful Degradation Only**: Current pattern uses try/except ImportError with error messages, but this is not a clean architectural solution.

4. **Duplicate Code Risk**: Code shared between framework and verticals must be maintained in two places.

### Affected Files (29 framework files reference victor_coding)

```
Framework Files with External Vertical Dependencies:
├── victor/tools/
│   ├── language_analyzer.py
│   ├── lsp_tool.py
│   ├── file_editor_tool.py
│   ├── graph_tool.py
│   ├── code_intelligence_tool.py
│   ├── code_search_tool.py
│   └── documentation_tool.py
├── victor/framework/
│   ├── agent.py
│   ├── prompt_builder.py
│   ├── agent_components.py
│   └── escape_hatch_registry.py
├── victor/core/
│   ├── verticals/extension_loader.py
│   ├── verticals/protocols/providers.py
│   ├── verticals/workflow_provider.py
│   └── bootstrap.py
├── victor/agent/
│   ├── service_provider.py
│   └── vertical_integration_adapter.py
└── victor/storage/memory/extractors/
    ├── tree_sitter_extractor.py
    └── code_extractor.py
```

---

## Design Principles

### 1. Dependency Direction Rule

**Rule**: Framework must NEVER import from external verticals.

```
┌─────────────────────────────────────────────────────────────────┐
│                     VALID DEPENDENCY FLOW                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   External Verticals ──import──▶ Framework Core                 │
│   (victor-coding)           (victor/)                           │
│        │                                                         │
│        └──import──▶ Contrib Packages                            │
│                   (victor.contrib.*)                            │
│                                                                  │
│   INVALID: Framework ──import──▶ External Vertical              │
└─────────────────────────────────────────────────────────────────┘
```

### 2. Three-Layer Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        VICTOR ARCHITECTURE                       │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ LAYER 1: Framework Core (victor/)                       │    │
│  │ • Essential: Agent, StateGraph, Tools, Events           │    │
│  │ • Protocol-based: All integration via protocols         │    │
│  │ • No external vertical dependencies                     │    │
│  └─────────────────────────────────────────────────────────┘    │
│                          ▲                                        │
│                          │                                        │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ LAYER 2: Contrib Packages (victor.contrib.*)            │    │
│  │ • Shared optional implementations                        │    │
│  │ • Eliminates code duplication                           │    │
│  │ • Verticals depend on contrib, not vice versa           │    │
│  │ • Example: victor.contrib.safety, victor.contrib.editing │    │
│  └─────────────────────────────────────────────────────────┘    │
│                          ▲                                        │
│                          │                                        │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ LAYER 3: External Verticals (victor-coding, etc.)       │    │
│  │ • Domain-specific implementations                        │    │
│  │ • Depend on Framework + Contrib                         │    │
│  │ • Discovered via entry points                           │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### 3. Decision Tree for Code Placement

```
┌──────────────────────────────────────────────────────────────────┐
│                      CODE PLACEMENT DECISION                      │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Is this code needed by the framework core?                      │
│     │                                                            │
│     ├─ YES ──▶ Does it require domain-specific logic?            │
│     │              │                                             │
│     │              ├─ YES ──▶ Define PROTOCOL in framework      │
│     │              │         Implement in vertical/contrib       │
│     │              │                                             │
│     │              └─ NO ──▶ Keep in framework core             │
│     │                                                            │
│     └─ NO ──▶ Is it shared by multiple verticals?               │
│                │                                                 │
│                ├─ YES ──▶ Put in victor.contrib.*               │
│                │                                                 │
│                └─ NO ──▶ Put in specific vertical               │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

---

## Migration Strategy

### Phase 1: Audit and Categorize Dependencies (Week 1)

**Action**: Create dependency matrix for all framework-to-vertical imports.

```markdown
| Framework File | Vertical Import | Category | Action |
|----------------|-----------------|----------|--------|
| file_editor_tool.py | victor_coding.editing.FileEditor | Implementation | Move to contrib.editing |
| lsp_tool.py | victor_coding.lsp.* | Protocol | Define LspProtocol in framework |
| code_extractor.py | victor_coding.codebase | Data | Move to contrib.codebase |
```

**Categories**:
1. **Protocol Definitions**: Interfaces - keep in framework
2. **Shared Implementations**: Common code - move to contrib
3. **Domain-Specific**: Vertical-specific - move to vertical
4. **Data Structures**: Shared types - move to framework or contrib

### Phase 2: Establish Contrib Packages (Week 2-3)

**New Contrib Packages**:

```python
# victor/contrib/editing/
├── __init__.py
├── base_editor.py        # Abstract base for file editing
├── diff_editor.py        # Diff-based editing implementation
└── protocols.py          # EditorProtocol, DiffResult

# victor/contrib/codebase/
├── __init__.py
├── base_analyzer.py      # Codebase analysis base
├── code_extractor.py     # Code extraction utilities
└── tree_sitter_parser.py # Tree-sitter parsing wrapper

# victor/contrib/lsp/
├── __init__.py
├── base_client.py        # LSP client base
├── protocol.py           # LspProtocol, CompletionProtocol
└── language_server.py    # Language server management
```

### Phase 3: Define Protocols in Framework (Week 3)

**Protocol-First Design**:

```python
# victor/framework/protocols/editing.py
from typing import Protocol, List, Optional
from dataclasses import dataclass

@runtime_checkable
class FileEditorProtocol(Protocol):
    """Protocol for file editing operations."""

    async def edit_file(
        self,
        file_path: str,
        edits: List["EditOperation"],
        preview: bool = False,
    ) -> "EditResult":
        """Apply edits to a file."""
        ...

    async def validate_edit(
        self,
        file_path: str,
        old_str: str,
        new_str: str,
    ) -> ValidationResult:
        """Validate an edit operation."""
        ...

@dataclass
class EditOperation:
    """Single edit operation."""
    old_str: str
    new_str: str
    start_line: Optional[int] = None
    end_line: Optional[int] = None

@dataclass
class EditResult:
    """Result of edit operation."""
    success: bool
    file_path: str
    edits_applied: int
    preview: Optional[str] = None
    error: Optional[str] = None
```

### Phase 4: Implement Lazy Loading with Graceful Fallback (Week 4)

**Current Pattern (Improved)**:

```python
# victor/tools/file_editor_tool.py
from victor.framework.protocols.editing import FileEditorProtocol
from victor.contrib.editing import DiffEditor  # Default implementation

class FileEditTool(BaseTool):
    """File editing tool with graceful fallback."""

    def __init__(self):
        super().__init__()
        self._editor: Optional[FileEditorProtocol] = None
        self._editor_available = False

    def _initialize_editor(self) -> None:
        """Lazy load editor with fallback."""
        if self._editor is not None:
            return

        # Try external vertical's editor
        try:
            from victor_coding.editing import FileEditor
            self._editor = FileEditor()
            self._editor_available = True
            return
        except ImportError:
            pass

        # Fall back to contrib implementation
        try:
            self._editor = DiffEditor()
            self._editor_available = True
            return
        except ImportError:
            pass

        # No editor available
        self._editor_available = False

    async def execute(
        self,
        file_path: str,
        edits: List[EditOperation],
        **kwargs
    ) -> ToolResult:
        """Execute file edit with graceful fallback."""
        self._initialize_editor()

        if not self._editor_available:
            return ToolResult(
                success=False,
                error="File editing requires victor-coding package. "
                      "Install with: pip install victor-coding"
            )

        return await self._editor.edit_file(file_path, edits)
```

### Phase 5: Migration Reference for Each Vertical (Week 5-8)

**For victor-coding**:

| Component | Current Location | Target Location |
|-----------|------------------|-----------------|
| FileEditor | victor_coding.editing | victor.contrib.editing (base) |
| CodebaseAnalyzer | victor_coding.codebase | victor.contrib.codebase (base) |
| LSPClient | victor_coding.lsp | victor.contrib.lsp (base) |
| TreeSitterParser | victor_coding.codebase | victor.contrib.codebase (parser) |
| CodingSafetyPatterns | victor_coding.safety | victor.contrib.safety (coding_patterns) |

**For victor-rag** (when externalized):

| Component | Current Location | Target Location |
|-----------|------------------|-----------------|
| VectorStore | victor.rag.stores | victor.contrib.vectorstores |
| EmbeddingService | victor.rag.embeddings | victor.contrib.embeddings |
| RetrievalPipeline | victor.rag.pipeline | victor.contrib.retrieval |

**For victor-devops** (when externalized):

| Component | Current Location | Target Location |
|-----------|------------------|-----------------|
| DockerManager | victor_devops.docker | victor.contrib.docker |
| KubernetesManager | victor_devops.k8s | victor.contrib.kubernetes |
| CI/CD Integrations | victor_devops.cicd | victor.contrib.cicd |

---

## Protocol Definition Guidelines

### When to Define a Protocol

Define a protocol in the framework when:

1. **Multiple verticals need the same capability** (e.g., file editing, LSP, vector storage)
2. **Framework needs to interact with the capability** (e.g., tool system needs editor protocol)
3. **The interface is stable** (not expected to change frequently)

### Protocol Template

```python
# victor/framework/protocols/<capability>.py
from __future__ import annotations
from typing import Protocol, runtime_checkable, Any, List, Optional
from dataclasses import dataclass

@runtime_checkable
class <Capability>Protocol(Protocol):
    """One-line description of the protocol.

    Longer description explaining:
    - What the protocol does
    - When to use it
    - Implementation requirements
    """

    async def <method_name>(self, <params>) -> <return_type>:
        """Brief method description."""
        ...

# Data classes for protocol
@dataclass
class <Request>:
    """Request structure."""
    ...

@dataclass
class <Response>:
    """Response structure."""
    ...
```

### Example: LSP Protocol

```python
# victor/framework/protocols/lsp.py
from __future__ import annotations
from typing import Protocol, runtime_checkable, List, Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path

@runtime_checkable
class LanguageServerProtocol(Protocol):
    """Protocol for Language Server Protocol (LSP) integration.

    LSP provides IDE-like features:
    - Code completion
    - Go-to-definition
    - Hover information
    - Diagnostics
    - Code actions
    """

    async def start_server(
        self,
        language: str,
        file_path: Path,
    ) -> bool:
        """Start language server for a file."""
        ...

    async def get_completions(
        self,
        file_path: Path,
        line: int,
        character: int,
    ) -> List["CompletionItem"]:
        """Get code completions at position."""
        ...

    async def get_definition(
        self,
        file_path: Path,
        line: int,
        character: int,
    ) -> Optional["Location"]:
        """Go to definition."""
        ...

@dataclass
class CompletionItem:
    """Code completion item."""
    label: str
    kind: int
    detail: Optional[str] = None
    documentation: Optional[str] = None
    insert_text: Optional[str] = None

@dataclass
class Location:
    """Source code location."""
    file_path: Path
    line: int
    character: int
```

---

## Contrib Package Guidelines

### When to Create a Contrib Package

Create a `victor.contrib.*` package when:

1. **Code is shared by 2+ verticals** (e.g., safety patterns, mode configs)
2. **Code is optional** (framework works without it)
3. **Code needs a default implementation** (verticals can override)

### Contrib Package Template

```python
# victor/contrib/<package>/
├── __init__.py           # Public API exports
├── base.py               # Abstract base class
├── default.py            # Default implementation
├── protocols.py          # Protocol definitions (if framework doesn't have them)
├── utils.py              # Helper utilities
└── tests/
    └── test_<package>.py
```

### Example: victor.contrib.editing

```python
# victor/contrib/editing/__init__.py
"""
File editing utilities for Victor verticals.

Provides base classes and default implementations for:
- Diff-based file editing
- Edit validation
- Edit preview and rollback
"""

from victor.contrib.editing.base import BaseEditor, EditResult
from victor.contrib.editing.default import DiffEditor
from victor.contrib.editing.protocols import EditorProtocol

__all__ = [
    "BaseEditor",
    "DiffEditor",
    "EditorProtocol",
    "EditResult",
]

# victor/contrib/editing/base.py
from abc import ABC, abstractmethod
from typing import List, Optional
from dataclasses import dataclass

@dataclass
class EditOperation:
    """Single edit operation."""
    old_str: str
    new_str: str

@dataclass
class EditResult:
    """Result of edit operation."""
    success: bool
    edits_applied: int
    preview: Optional[str] = None
    error: Optional[str] = None

class BaseEditor(ABC):
    """Abstract base for file editors."""

    @abstractmethod
    async def edit_file(
        self,
        file_path: str,
        edits: List[EditOperation],
        preview: bool = False,
    ) -> EditResult:
        """Apply edits to a file."""
        ...

    @abstractmethod
    def validate_edit(
        self,
        old_str: str,
        new_str: str,
    ) -> bool:
        """Validate edit operation."""
        ...

# victor/contrib/editing/default.py
from victor.contrib.editing.base import BaseEditor, EditResult, EditOperation

class DiffEditor(BaseEditor):
    """Default diff-based file editor."""

    async def edit_file(
        self,
        file_path: str,
        edits: List[EditOperation],
        preview: bool = False,
    ) -> EditResult:
        """Apply edits using diff-based approach."""
        # Implementation
        ...

    def validate_edit(
        self,
        old_str: str,
        new_str: str,
    ) -> bool:
        """Validate edit."""
        # Implementation
        ...
```

---

## Memory of Past Decisions

### Decision 1: External Verticals Architecture (2025-01-09)

**Context**: Verticals (coding, rag, devops, etc.) were moved from monolithic `victor/` to separate packages.

**Rationale**:
- Enable independent versioning
- Reduce framework package size
- Allow third-party verticals

**Consequence**: Framework code that imported vertical-specific implementations broke.

**Resolution**: Implemented try/except ImportError with graceful degradation.

**Future Prevention**: Framework should only depend on protocols, not implementations.

---

### Decision 2: Contrib Packages for Shared Code (2025-01-15)

**Context**: Multiple verticals were duplicating code (safety patterns, mode configs, conversation managers).

**Rationale**:
- DRY principle
- Consistent behavior across verticals
- Easier maintenance

**Resolution**: Created `victor.contrib.*` packages:
- victor.contrib.safety
- victor.contrib.conversation
- victor.contrib.mode_config
- victor.contrib.workflows
- victor.contrib.testing

**Future**: Continue expanding contrib for newly identified shared code.

---

### Decision 3: Protocol-Based Integration (2025-01-20)

**Context**: Framework needed to interact with vertical capabilities without tight coupling.

**Rationale**:
- Dependency Inversion Principle (DIP)
- Testability (mock protocols)
- Vertical independence

**Resolution**: Defined 13+ protocols in `victor/core/verticals/protocols/`:
- SafetyExtensionProtocol
- MiddlewareProtocol
- PromptContributorProtocol
- TeamSpecProviderProtocol
- WorkflowProviderProtocol
- etc.

**Future**: All vertical-framework interactions should use protocols.

---

## Implementation Checklist

### Framework Changes

- [ ] Audit all framework-to-vertical imports (29 files identified)
- [ ] Create protocol definitions for shared capabilities
- [ ] Move shared implementations to contrib packages
- [ ] Update tool imports to use protocols + contrib defaults
- [ ] Remove all direct victor_coding imports from framework

### Contrib Packages to Create

- [ ] `victor.contrib.editing` - File editing base and defaults
- [ ] `victor.contrib.codebase` - Codebase analysis utilities
- [ ] `victor.contrib.lsp` - Language server protocol base
- [ ] `victor.contrib.embeddings` - Embedding utilities
- [ ] `victor.contrib.vectorstores` - Vector storage abstractions

### victor-coding Changes

- [ ] Remove code moved to contrib
- [ ] Import from victor.contrib.* for shared code
- [ ] Keep only coding-specific implementations
- [ ] Update imports to use protocols where appropriate

### Documentation

- [ ] Update vertical development guide
- [ ] Document all protocols
- [ ] Create contrib package development guide
- [ ] Update migration guide for vertical authors

---

## Success Metrics

1. **Zero Framework-to-Vertical Imports**: Framework code never imports from external verticals
2. **All Shared Code in Contrib**: Code used by 2+ verticals in victor.contrib.*
3. **Protocol Coverage**: All framework-vertical interactions use protocols
4. **Graceful Fallback**: Tools work with deprecation warnings when verticals not installed
5. **Test Coverage**: Unit tests can mock protocols without importing verticals

---

## References

- [Framework & Vertical Integration Architecture](./framework-vertical-integration.md)
- [Vertical Package Specification](../../feps/vertical-package-spec.md)
- [Phase 3 Architecture Analysis](../../architecture-analysis-phase3.md)
- [Contrib Packages Documentation](../extending/contrib-packages.md)
