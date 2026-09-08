# Victor Protocols API Reference

This document provides comprehensive API reference documentation for Victor's protocol interfaces. Protocols in Victor follow Python's `Protocol` pattern (PEP 544) to enable structural subtyping, dependency inversion, and interface segregation.

## Table of Contents

- [Protocol Overview](#protocol-overview)
- [Core Protocols](#core-protocols)
  - [IProviderAdapter](#iprovideradapter)
  - [IGroundingStrategy](#igroundingstrategy)
  - [IQualityAssessor](#iqualityassessor)
  - [IModeController](#imodecontroller)
  - [IPathResolver](#ipathresolver)
- [Search Protocols](#search-protocols)
  - [ISemanticSearch](#isemanticsearch)
  - [IIndexable](#iindexable)
- [Team Protocols](#team-protocols)
  - [IAgent](#iagent)
  - [ITeamMember](#iteammember)
  - [ITeamCoordinator](#iteamcoordinator)
- [LSP Types](#lsp-types)
- [Tool Selection Protocols](#tool-selection-protocols)
- [Implementation Examples](#implementation-examples)

---

## Protocol Overview

### What are Protocols in Victor?

Victor uses Python's `Protocol` pattern (from `typing`) to define interfaces without requiring inheritance. This approach, known as structural subtyping or "duck typing with type hints," allows any class that implements the required methods to satisfy a protocol, regardless of its inheritance hierarchy.

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class IExample(Protocol):
    """Protocol interface definition."""

    def required_method(self, arg: str) -> int:
        """Method that implementations must provide."""
        ...
```

### ISP Compliance Design

Victor's protocols follow the Interface Segregation Principle (ISP):

1. **Small, Focused Interfaces**: Each protocol defines the minimum interface needed for a specific capability
2. **Composable Protocols**: Complex behaviors are achieved by implementing multiple simple protocols
3. **No Fat Interfaces**: Clients depend only on the methods they use

Example of protocol composition:
```python
# Small, focused protocols
class ITeamCoordinator(Protocol): ...
class IObservableCoordinator(Protocol): ...
class IRLCoordinator(Protocol): ...

# Composed protocol for full capabilities
class IEnhancedTeamCoordinator(
    ITeamCoordinator,
    IObservableCoordinator,
    IRLCoordinator,
    Protocol
): ...
```

### How to Implement Protocols

To implement a Victor protocol:

1. **Implicit Implementation**: Simply implement all required methods with matching signatures
2. **No Inheritance Required**: Your class doesn't need to explicitly inherit from the protocol
3. **Runtime Checking**: Use `isinstance()` with `@runtime_checkable` protocols

```python
from victor.protocols import IProviderAdapter, ProviderCapabilities

class MyProviderAdapter:
    """Custom provider adapter - implicitly implements IProviderAdapter."""

    @property
    def name(self) -> str:
        return "my_provider"

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(quality_threshold=0.75)

    def detect_continuation_needed(self, response: str) -> bool:
        return not response.strip()

    # ... implement other required methods
```

---

## Core Protocols

### IProviderAdapter

**Location**: `victor/protocols/provider_adapter.py`

**Import**: `from victor.protocols import IProviderAdapter`

The `IProviderAdapter` protocol defines the interface for adapting provider-specific behaviors. Each LLM provider (OpenAI, Anthropic, DeepSeek, etc.) has different response formats, tool calling conventions, and quality thresholds.

#### Protocol Definition

```python
@runtime_checkable
class IProviderAdapter(Protocol):
    """Interface for provider-specific behavior adaptation."""

    @property
    def name(self) -> str:
        """Return the provider name."""
        ...

    @property
    def capabilities(self) -> ProviderCapabilities:
        """Return provider capabilities configuration."""
        ...

    def detect_continuation_needed(self, response: str) -> bool:
        """Detect if response indicates continuation is needed.

        Args:
            response: The LLM response text

        Returns:
            True if the response appears incomplete
        """
        ...

    def extract_thinking_content(self, response: str) -> Tuple[str, str]:
        """Extract thinking tags and content separately.

        Args:
            response: The LLM response text

        Returns:
            Tuple of (thinking_content, main_content)
        """
        ...

    def normalize_tool_calls(self, raw_calls: List[Any]) -> List[ToolCall]:
        """Normalize tool calls to standard format.

        Args:
            raw_calls: Provider-specific tool call data

        Returns:
            List of normalized ToolCall objects
        """
        ...

    def should_retry(self, error: Exception) -> Tuple[bool, float]:
        """Determine if error is retryable and backoff time.

        Args:
            error: The exception that occurred

        Returns:
            Tuple of (is_retryable, backoff_seconds)
        """
        ...
```

#### Supporting Types

```python
class ToolCallFormat(Enum):
    """Tool call format variants across providers."""
    OPENAI = "openai"      # Standard OpenAI format
    ANTHROPIC = "anthropic" # Anthropic's content blocks
    NATIVE = "native"       # Provider's native format
    FALLBACK = "fallback"   # Text-based parsing fallback

@dataclass
class ProviderCapabilities:
    """Provider-specific capabilities and thresholds."""
    quality_threshold: float = 0.80
    supports_thinking_tags: bool = False
    thinking_tag_format: str = ""
    continuation_markers: List[str] = field(default_factory=list)
    max_continuation_attempts: int = 5
    tool_call_format: ToolCallFormat = ToolCallFormat.OPENAI
    output_deduplication: bool = False
    streaming_chunk_size: int = 1024
    supports_parallel_tools: bool = True
    grounding_required: bool = True
    grounding_strictness: float = 0.8
```

#### Usage

```python
from victor.protocols import get_provider_adapter

# Get adapter for a specific provider
adapter = get_provider_adapter("deepseek")

# Check capabilities
if adapter.capabilities.supports_thinking_tags:
    thinking, content = adapter.extract_thinking_content(response)

# Detect if continuation needed
if adapter.detect_continuation_needed(response):
    # Request continuation from LLM
    pass
```

---

### IGroundingStrategy

**Location**: `victor/protocols/grounding.py`

**Import**: `from victor.protocols import IGroundingStrategy`

The `IGroundingStrategy` protocol defines the interface for verifying claims in LLM responses against verifiable sources (files, symbols, content).

#### Protocol Definition

```python
@runtime_checkable
class IGroundingStrategy(Protocol):
    """Strategy interface for grounding verification."""

    @property
    def name(self) -> str:
        """Return strategy name."""
        ...

    @property
    def claim_types(self) -> List[GroundingClaimType]:
        """Return claim types this strategy can verify."""
        ...

    async def verify(
        self,
        claim: GroundingClaim,
        context: Dict[str, Any],
    ) -> VerificationResult:
        """Verify a claim against context.

        Args:
            claim: The claim to verify
            context: Additional context for verification

        Returns:
            Verification result with grounding status
        """
        ...

    def extract_claims(
        self,
        response: str,
        context: Dict[str, Any],
    ) -> List[GroundingClaim]:
        """Extract claims of this type from a response.

        Args:
            response: The response text to analyze
            context: Additional context

        Returns:
            List of claims found in the response
        """
        ...
```

#### Supporting Types

```python
class GroundingClaimType(str, Enum):
    """Types of claims that can be grounded."""
    FILE_EXISTS = "file_exists"
    FILE_NOT_EXISTS = "file_not_exists"
    SYMBOL_EXISTS = "symbol_exists"
    CONTENT_MATCH = "content_match"
    LINE_NUMBER = "line_number"
    DIRECTORY_EXISTS = "directory_exists"

@dataclass
class GroundingClaim:
    """A claim extracted from a response."""
    claim_type: GroundingClaimType
    value: str
    context: Dict[str, Any] = field(default_factory=dict)
    source_text: str = ""
    confidence: float = 1.0

@dataclass
class VerificationResult:
    """Result of verifying a single claim."""
    is_grounded: bool
    confidence: float = 0.0
    claim: Optional[GroundingClaim] = None
    evidence: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""
```

#### Built-in Strategies

- **FileExistenceStrategy**: Verifies file path references
- **SymbolReferenceStrategy**: Verifies code symbol references
- **ContentMatchStrategy**: Verifies quoted content matches source

#### Usage

```python
from victor.protocols import (
    CompositeGroundingVerifier,
    FileExistenceStrategy,
    SymbolReferenceStrategy,
)

# Create composite verifier
verifier = CompositeGroundingVerifier([
    FileExistenceStrategy(project_root),
    SymbolReferenceStrategy(symbol_table),
])

# Verify response
result = await verifier.verify(response, context)
if result.is_grounded:
    print(f"Verified {result.verified_claims}/{result.total_claims} claims")
```

---

### IQualityAssessor

**Location**: `victor/protocols/quality.py`

**Import**: `from victor.protocols import IQualityAssessor`

The `IQualityAssessor` protocol defines the interface for assessing response quality across multiple dimensions.

#### Protocol Definition

```python
@runtime_checkable
class IQualityAssessor(Protocol):
    """Interface for quality assessment."""

    def assess(
        self,
        response: str,
        context: Dict[str, Any],
    ) -> QualityScore:
        """Assess response quality.

        Args:
            response: The response text to assess
            context: Additional context (query, provider, etc.)

        Returns:
            Quality score with dimension breakdown
        """
        ...

    @property
    def dimensions(self) -> List[ProtocolQualityDimension]:
        """Return dimensions this assessor evaluates."""
        ...
```

#### Supporting Types

```python
class ProtocolQualityDimension(str, Enum):
    """Quality dimensions for response assessment."""
    GROUNDING = "grounding"      # Factual accuracy
    COVERAGE = "coverage"        # Query coverage
    CLARITY = "clarity"          # Response clarity
    CORRECTNESS = "correctness"  # Code correctness
    CONCISENESS = "conciseness"  # Appropriate brevity
    HELPFULNESS = "helpfulness"  # Task helpfulness
    SAFETY = "safety"            # Safety considerations

@dataclass
class DimensionScore:
    """Score for a single quality dimension."""
    dimension: ProtocolQualityDimension
    score: float  # 0.0-1.0
    weight: float = 1.0
    reason: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)

@dataclass
class QualityScore:
    """Overall quality assessment result."""
    score: float  # 0.0-1.0
    is_acceptable: bool
    threshold: float = 0.80
    provider: str = ""
    dimension_scores: Dict[ProtocolQualityDimension, DimensionScore]
    feedback: str = ""
    suggestions: List[str] = field(default_factory=list)
```

#### Built-in Assessors

- **SimpleQualityAssessor**: Heuristic-based assessment
- **ProviderAwareQualityAssessor**: Provider-specific adjustments
- **CompositeQualityAssessor**: Combines multiple assessors

#### Usage

```python
from victor.protocols import ProviderAwareQualityAssessor

assessor = ProviderAwareQualityAssessor(
    provider_name="anthropic",
    provider_threshold=0.85,
)

score = assessor.assess(response, {"query": user_query})
if score.is_acceptable:
    print(f"Quality: {score.score:.2%}")
else:
    print(f"Quality below threshold: {score.feedback}")
```

---

### IModeController

**Location**: `victor/protocols/mode_aware.py`

**Import**: `from victor.protocols import IModeController`

The `IModeController` protocol defines the interface for mode management (BUILD/PLAN/EXPLORE modes).

#### Protocol Definition

```python
@runtime_checkable
class IModeController(Protocol):
    """Protocol for mode controller access."""

    @property
    def current_mode(self) -> Any:
        """Get the current agent mode."""
        ...

    @property
    def config(self) -> Any:
        """Get the current mode configuration."""
        ...

    def is_tool_allowed(self, tool_name: str) -> bool:
        """Check if a tool is allowed in the current mode."""
        ...

    def get_tool_priority(self, tool_name: str) -> float:
        """Get priority adjustment for a tool in current mode."""
        ...
```

#### Supporting Types

```python
@dataclass
class ModeInfo:
    """Information about the current mode."""
    name: str = "BUILD"  # BUILD, PLAN, or EXPLORE
    allow_all_tools: bool = True
    exploration_multiplier: float = 1.0
    sandbox_dir: Optional[str] = None
    allowed_tools: Set[str] = None
    disallowed_tools: Set[str] = None
```

#### ModeAwareMixin

The `ModeAwareMixin` provides convenient mode-aware functionality:

```python
from victor.protocols import ModeAwareMixin

class MyComponent(ModeAwareMixin):
    def process(self) -> None:
        if self.is_build_mode:
            # Full capabilities
            pass
        elif self.is_plan_mode:
            # Read-only with sandbox
            pass
        elif self.is_explore_mode:
            # Read-only exploration
            pass
```

---

### IPathResolver

**Location**: `victor/protocols/path_resolver.py`

**Import**: `from victor.protocols import IPathResolver`

The `IPathResolver` protocol defines the interface for centralized path resolution and normalization.

#### Protocol Definition

```python
@runtime_checkable
class IPathResolver(Protocol):
    """Protocol for path resolution."""

    def resolve(self, path: str, must_exist: bool = True) -> PathResolution:
        """Resolve a path with normalization.

        Args:
            path: Path to resolve (relative or absolute)
            must_exist: If True, raises error if path doesn't exist

        Returns:
            PathResolution with resolved path and metadata
        """
        ...

    def resolve_file(self, path: str) -> PathResolution:
        """Resolve a file path.

        Args:
            path: File path to resolve

        Returns:
            PathResolution, raises if not a file
        """
        ...

    def resolve_directory(self, path: str) -> PathResolution:
        """Resolve a directory path.

        Args:
            path: Directory path to resolve

        Returns:
            PathResolution, raises if not a directory
        """
        ...

    def suggest_similar(self, path: str, limit: int = 5) -> List[str]:
        """Suggest similar paths that exist.

        Args:
            path: Non-existent path to find matches for
            limit: Maximum suggestions to return

        Returns:
            List of similar existing paths
        """
        ...
```

#### Supporting Types

```python
@dataclass
class PathResolution:
    """Result of path resolution."""
    original_path: str
    resolved_path: Path
    was_normalized: bool = False
    normalization_applied: Optional[str] = None
    exists: bool = False
    is_file: bool = False
    is_directory: bool = False
```

#### Usage

```python
from victor.protocols import create_path_resolver

resolver = create_path_resolver()

# Resolve with automatic normalization
result = resolver.resolve_file("project/utils/helper.py")
if result.was_normalized:
    print(f"Normalized: {result.original_path} -> {result.resolved_path}")

# Get suggestions for typos
suggestions = resolver.suggest_similar("modls/news.py")
```

---

## Search Protocols

### ISemanticSearch

**Location**: `victor/protocols/search.py`

**Import**: `from victor.protocols import ISemanticSearch`

The `ISemanticSearch` protocol defines the interface for semantic search implementations across all verticals.

#### Protocol Definition

```python
@runtime_checkable
class ISemanticSearch(Protocol):
    """Protocol for semantic search implementations."""

    @property
    def is_indexed(self) -> bool:
        """Whether the search provider has indexed content."""
        ...

    async def search(
        self,
        query: str,
        max_results: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[SearchHit]:
        """Execute semantic search on indexed content.

        Args:
            query: Natural language search query
            max_results: Maximum number of results to return
            filter_metadata: Optional metadata filters

        Returns:
            List of SearchHit objects ordered by relevance
        """
        ...
```

#### Usage

```python
from victor.protocols import ISemanticSearch

async def find_relevant_code(searcher: ISemanticSearch, query: str):
    if not searcher.is_indexed:
        return []

    results = await searcher.search(
        query="authentication error handling",
        max_results=5,
        filter_metadata={"file_type": "py"}
    )

    for hit in results:
        print(f"{hit.file_path}:{hit.line_number} - {hit.score:.2f}")
```

---

### IIndexable

**Location**: `victor/protocols/search.py`

**Import**: `from victor.protocols import IIndexable`

The `IIndexable` protocol defines the interface for content that can be indexed for semantic search.

#### Protocol Definition

```python
@runtime_checkable
class IIndexable(Protocol):
    """Protocol for indexable content sources."""

    async def index_document(
        self,
        file_path: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Index a document for semantic search.

        Args:
            file_path: Path or identifier for the document
            content: Text content to index
            metadata: Optional metadata to store
        """
        ...

    async def remove_document(self, file_path: str) -> bool:
        """Remove a document from the index.

        Args:
            file_path: Path or identifier for the document

        Returns:
            True if document was removed
        """
        ...

    async def clear_index(self) -> None:
        """Clear all indexed content."""
        ...
```

#### Combined Protocol

```python
@runtime_checkable
class ISemanticSearchWithIndexing(ISemanticSearch, IIndexable, Protocol):
    """Combined protocol for searchable and indexable implementations."""
    pass
```

---

## Team Protocols

### IAgent

**Location**: `victor/protocols/team.py`

**Import**: `from victor.protocols import IAgent`

The `IAgent` protocol is the unified base protocol for all agent implementations in Victor.

#### Protocol Definition

```python
@runtime_checkable
class IAgent(Protocol):
    """Unified protocol for all agent implementations."""

    @property
    def id(self) -> str:
        """Unique identifier for this agent."""
        ...

    @property
    def role(self) -> Any:
        """Role of this agent."""
        ...

    async def execute_task(self, task: str, context: Dict[str, Any]) -> Any:
        """Execute a task and return the result.

        Args:
            task: Task description
            context: Execution context with shared state

        Returns:
            Result of task execution
        """
        ...
```

---

### ITeamMember

**Location**: `victor/protocols/team.py`

**Import**: `from victor.protocols import ITeamMember`

The `ITeamMember` protocol extends `IAgent` with team coordination capabilities.

#### Protocol Definition

```python
@runtime_checkable
class ITeamMember(IAgent, Protocol):
    """Protocol for team members."""

    @property
    def persona(self) -> Optional[Any]:
        """Persona of this member."""
        ...

    async def execute_task(self, task: str, context: Dict[str, Any]) -> Any:
        """Execute a task and return the result."""
        ...

    async def receive_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Receive and optionally respond to a message.

        Args:
            message: Incoming message

        Returns:
            Optional response message
        """
        ...
```

---

### ITeamCoordinator

**Location**: `victor/protocols/team.py`

**Import**: `from victor.protocols import ITeamCoordinator`

The `ITeamCoordinator` protocol defines the base interface for team coordinators.

#### Protocol Definition

```python
@runtime_checkable
class ITeamCoordinator(Protocol):
    """Base protocol for team coordinators."""

    def add_member(self, member: ITeamMember) -> "ITeamCoordinator":
        """Add a member to the team.

        Args:
            member: Team member to add

        Returns:
            Self for fluent chaining
        """
        ...

    def set_formation(self, formation: TeamFormation) -> "ITeamCoordinator":
        """Set the team formation pattern.

        Args:
            formation: Formation to use

        Returns:
            Self for fluent chaining
        """
        ...

    async def execute_task(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task with the team.

        Args:
            task: Task description
            context: Execution context

        Returns:
            Result dictionary with success, member_results, final_output
        """
        ...

    async def broadcast(self, message: AgentMessage) -> List[Optional[AgentMessage]]:
        """Broadcast a message to all team members.

        Args:
            message: Message to broadcast

        Returns:
            List of responses from members
        """
        ...
```

#### Extended Protocols

Victor provides several extended coordinator protocols for additional capabilities:

```python
# Observability integration
class IObservableCoordinator(Protocol):
    def set_execution_context(self, task_type, complexity, vertical, trigger): ...
    def set_progress_callback(self, callback): ...

# Reinforcement learning integration
class IRLCoordinator(Protocol):
    def set_rl_coordinator(self, rl_coordinator): ...

# Message bus provider
class IMessageBusProvider(Protocol):
    @property
    def message_bus(self) -> Any: ...

# Shared memory provider
class ISharedMemoryProvider(Protocol):
    @property
    def shared_memory(self) -> Any: ...

# Combined enhanced coordinator
class IEnhancedTeamCoordinator(
    ITeamCoordinator,
    IObservableCoordinator,
    IRLCoordinator,
    IMessageBusProvider,
    ISharedMemoryProvider,
    Protocol,
): ...
```

---

## LSP Types

**Location**: `victor/protocols/lsp_types.py`

**Import**: `from victor.protocols import Position, Range, Diagnostic, DocumentSymbol`

Victor provides Language Server Protocol (LSP) standard types for cross-vertical document operations.

### Position

Represents a cursor position in a document (0-indexed).

```python
@dataclass
class Position:
    line: int       # Line position (0-indexed)
    character: int  # Character offset in line (0-indexed)

    def to_dict(self) -> Dict[str, int]: ...

    @classmethod
    def from_dict(cls, data: Dict[str, int]) -> "Position": ...
```

### Range

Represents a text span from start to end position.

```python
@dataclass
class Range:
    start: Position  # Start position (inclusive)
    end: Position    # End position (exclusive)

    def contains(self, position: Position) -> bool: ...
    def overlaps(self, other: "Range") -> bool: ...

    @property
    def is_empty(self) -> bool: ...
```

### Diagnostic

Represents a problem or suggestion in a document.

```python
@dataclass
class Diagnostic:
    range: Range
    message: str
    severity: DiagnosticSeverity = DiagnosticSeverity.ERROR
    source: Optional[str] = None
    code: Optional[Union[str, int]] = None
    tags: List[DiagnosticTag] = field(default_factory=list)
    related_information: List[DiagnosticRelatedInformation] = field(default_factory=list)

    @property
    def is_error(self) -> bool: ...

    @property
    def is_warning(self) -> bool: ...
```

### DocumentSymbol

Hierarchical representation of symbols (classes, functions, etc.).

```python
@dataclass
class DocumentSymbol:
    name: str
    kind: SymbolKind
    range: Range
    selection_range: Range
    detail: Optional[str] = None
    children: List["DocumentSymbol"] = field(default_factory=list)
    deprecated: bool = False
```

### Enumerations

```python
class DiagnosticSeverity(IntEnum):
    ERROR = 1
    WARNING = 2
    INFORMATION = 3
    HINT = 4

class SymbolKind(IntEnum):
    FILE = 1
    MODULE = 2
    CLASS = 5
    METHOD = 6
    FUNCTION = 12
    VARIABLE = 13
    # ... and more
```

### Usage in Code Analysis

```python
from victor.protocols import Position, Range, Diagnostic, DiagnosticSeverity

# Create a diagnostic for an undefined variable
diagnostic = Diagnostic(
    range=Range(
        start=Position(line=10, character=4),
        end=Position(line=10, character=15),
    ),
    message="Variable 'undefined_var' is not defined",
    severity=DiagnosticSeverity.ERROR,
    source="pylint",
    code="E0602",
)

# Check severity
if diagnostic.is_error:
    print(f"Error at line {diagnostic.range.start.line + 1}")
```

---

## Tool Selection Protocols

**Location**: `victor/protocols/tool_selector.py`

**Import**: `from victor.protocols import IToolSelector, ToolSelectionResult`

### IToolSelector

```python
@runtime_checkable
class IToolSelector(Protocol):
    """Protocol for tool selection implementations."""

    def select_tools(
        self,
        task: str,
        *,
        limit: int = 10,
        min_score: float = 0.0,
        context: Optional[ToolSelectionContext] = None,
    ) -> ToolSelectionResult:
        """Select relevant tools for a task.

        Args:
            task: Task description or query
            limit: Maximum number of tools to return
            min_score: Minimum relevance score threshold
            context: Optional additional context

        Returns:
            ToolSelectionResult with ranked tool names and scores
        """
        ...

    def get_tool_score(
        self,
        tool_name: str,
        task: str,
        *,
        context: Optional[ToolSelectionContext] = None,
    ) -> float:
        """Get relevance score for a specific tool."""
        ...

    @property
    def strategy(self) -> ToolSelectionStrategy:
        """Get the selection strategy used."""
        ...
```

### Supporting Types

```python
class ToolSelectionStrategy(Enum):
    KEYWORD = "keyword"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"

@dataclass
class ToolSelectionResult:
    tool_names: List[str]
    scores: Dict[str, float]
    strategy_used: ToolSelectionStrategy
    metadata: Dict[str, Any]

    @property
    def top_tool(self) -> Optional[str]: ...

    def filter_by_score(self, min_score: float) -> "ToolSelectionResult": ...

@dataclass
class ToolSelectionContext:
    task_description: str
    conversation_stage: Optional[str] = None
    previous_tools: List[str] = field(default_factory=list)
    failed_tools: Set[str] = field(default_factory=set)
    model_name: str = ""
    provider_name: str = ""
```

---

## Implementation Examples

### Custom Provider Adapter

```python
from typing import Any, List, Tuple
from victor.protocols import (
    IProviderAdapter,
    ProviderCapabilities,
    ToolCallFormat,
)
from victor.agent.tool_calling.base import ToolCall

class CustomProviderAdapter:
    """Custom provider adapter implementation."""

    @property
    def name(self) -> str:
        return "custom_provider"

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            quality_threshold=0.75,
            supports_thinking_tags=True,
            thinking_tag_format="<reasoning>...</reasoning>",
            continuation_markers=["...", "[CONTINUE]"],
            tool_call_format=ToolCallFormat.OPENAI,
            supports_parallel_tools=True,
        )

    def detect_continuation_needed(self, response: str) -> bool:
        if not response or not response.strip():
            return True
        for marker in self.capabilities.continuation_markers:
            if response.strip().endswith(marker):
                return True
        return False

    def extract_thinking_content(self, response: str) -> Tuple[str, str]:
        import re
        pattern = r"<reasoning>(.*?)</reasoning>"
        matches = re.findall(pattern, response, re.DOTALL)
        thinking = "\n".join(matches) if matches else ""
        content = re.sub(pattern, "", response, flags=re.DOTALL).strip()
        return (thinking, content)

    def normalize_tool_calls(self, raw_calls: List[Any]) -> List[ToolCall]:
        normalized = []
        for i, call in enumerate(raw_calls):
            if isinstance(call, dict):
                func = call.get("function", {})
                normalized.append(
                    ToolCall(
                        id=call.get("id", f"call_{i}"),
                        name=func.get("name", ""),
                        arguments=func.get("arguments", {}),
                        raw=call,
                    )
                )
        return normalized

    def should_retry(self, error: Exception) -> Tuple[bool, float]:
        error_str = str(error).lower()
        if "rate" in error_str and "limit" in error_str:
            return (True, 60.0)
        if "timeout" in error_str:
            return (True, 5.0)
        return (False, 0.0)
```

### Custom Grounding Strategy

```python
from typing import Any, Dict, List
from victor.protocols import (
    IGroundingStrategy,
    GroundingClaim,
    GroundingClaimType,
    VerificationResult,
)

class DatabaseReferenceStrategy:
    """Verify database table/column references."""

    def __init__(self, schema: Dict[str, List[str]]):
        self._schema = schema  # table_name -> [column_names]

    @property
    def name(self) -> str:
        return "database_reference"

    @property
    def claim_types(self) -> List[GroundingClaimType]:
        return [GroundingClaimType.SYMBOL_EXISTS]

    async def verify(
        self,
        claim: GroundingClaim,
        context: Dict[str, Any],
    ) -> VerificationResult:
        reference = claim.value

        # Check if it's a table.column reference
        if "." in reference:
            table, column = reference.split(".", 1)
            if table in self._schema:
                found = column in self._schema[table]
                return VerificationResult(
                    is_grounded=found,
                    confidence=0.95 if found else 0.0,
                    claim=claim,
                    reason=f"Column '{column}' {'exists' if found else 'not found'} in table '{table}'",
                )

        # Check if it's just a table name
        found = reference in self._schema
        return VerificationResult(
            is_grounded=found,
            confidence=0.9 if found else 0.0,
            claim=claim,
            reason=f"Table '{reference}' {'exists' if found else 'not found'}",
        )

    def extract_claims(
        self,
        response: str,
        context: Dict[str, Any],
    ) -> List[GroundingClaim]:
        import re
        claims = []

        # Find table.column patterns
        pattern = r"`([a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z_][a-zA-Z0-9_]*)`"
        for match in re.finditer(pattern, response):
            claims.append(
                GroundingClaim(
                    claim_type=GroundingClaimType.SYMBOL_EXISTS,
                    value=match.group(1),
                    source_text=match.group(0),
                    confidence=0.8,
                )
            )

        return claims
```

### Custom Team Member

```python
from typing import Any, Dict, Optional
from victor.protocols import ITeamMember
from victor.teams.types import AgentMessage

class SpecialistAgent:
    """A specialist agent for specific domain tasks."""

    def __init__(self, agent_id: str, specialty: str):
        self._id = agent_id
        self._specialty = specialty

    @property
    def id(self) -> str:
        return self._id

    @property
    def role(self) -> str:
        return f"{self._specialty}_specialist"

    @property
    def persona(self) -> Optional[str]:
        return f"I am a specialist in {self._specialty}."

    async def execute_task(self, task: str, context: Dict[str, Any]) -> str:
        # Implement task execution logic
        result = f"[{self.role}] Analyzed task: {task}"
        return result

    async def receive_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        # Process incoming message and optionally respond
        if self._specialty.lower() in message.content.lower():
            return AgentMessage(
                sender=self._id,
                content=f"I can help with {self._specialty} aspects.",
                message_type="response",
            )
        return None
```

### Custom Quality Assessor

```python
from typing import Any, Dict, List
from victor.protocols import (
    IQualityAssessor,
    QualityScore,
    DimensionScore,
    ProtocolQualityDimension,
)

class SecurityAwareQualityAssessor:
    """Quality assessor with security checks."""

    def __init__(self, threshold: float = 0.80):
        self._threshold = threshold

    @property
    def dimensions(self) -> List[ProtocolQualityDimension]:
        return [
            ProtocolQualityDimension.CORRECTNESS,
            ProtocolQualityDimension.SAFETY,
        ]

    def assess(
        self,
        response: str,
        context: Dict[str, Any],
    ) -> QualityScore:
        dimension_scores = {}

        # Assess correctness
        correctness_score = self._assess_correctness(response)
        dimension_scores[ProtocolQualityDimension.CORRECTNESS] = correctness_score

        # Assess safety
        safety_score = self._assess_safety(response)
        dimension_scores[ProtocolQualityDimension.SAFETY] = safety_score

        # Calculate overall score (safety weighted heavily)
        overall = (correctness_score.score * 0.4) + (safety_score.score * 0.6)

        return QualityScore(
            score=overall,
            is_acceptable=overall >= self._threshold,
            threshold=self._threshold,
            dimension_scores=dimension_scores,
        )

    def _assess_correctness(self, response: str) -> DimensionScore:
        # Implementation...
        return DimensionScore(
            dimension=ProtocolQualityDimension.CORRECTNESS,
            score=0.85,
            reason="Code syntax validated",
        )

    def _assess_safety(self, response: str) -> DimensionScore:
        dangerous_patterns = [
            "eval(", "exec(", "__import__",
            "rm -rf", "DROP TABLE", "DELETE FROM"
        ]

        for pattern in dangerous_patterns:
            if pattern in response:
                return DimensionScore(
                    dimension=ProtocolQualityDimension.SAFETY,
                    score=0.0,
                    reason=f"Dangerous pattern detected: {pattern}",
                )

        return DimensionScore(
            dimension=ProtocolQualityDimension.SAFETY,
            score=1.0,
            reason="No dangerous patterns detected",
        )
```

---

## See Also

- [Provider Development Guide](../user-guide/providers.md)
- [Tool Development Guide](../user-guide/tools.md)
- [Team Coordination Guide](../guides/MULTI_AGENT_TEAMS.md)
- [Vertical Development Guide](../development/extending/verticals.md)
