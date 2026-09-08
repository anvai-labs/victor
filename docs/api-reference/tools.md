# Victor Tools API Reference

This document provides comprehensive API documentation for creating and registering tools in Victor.

## Table of Contents

- [BaseTool Interface](#basetool-interface)
- [Tool Registration](#tool-registration)
- [Parameter Schema](#parameter-schema)
- [Cost Tiers](#cost-tiers)
- [Access Modes](#access-modes)
- [Tool Implementation Patterns](#tool-implementation-patterns)
- [Tool Decorator](#tool-decorator)
- [Advanced Features](#advanced-features)

---

## BaseTool Interface

All Victor tools inherit from the `BaseTool` abstract base class defined in `victor/tools/base.py`.

### Required Attributes

```python
from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseTool(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name - unique identifier used for tool calls."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description - explains what the tool does."""
        pass

    @property
    @abstractmethod
    def parameters(self) -> Dict[str, Any]:
        """JSON Schema for tool parameters."""
        pass
```

### Optional Attributes

```python
from victor.tools.enums import CostTier, AccessMode, DangerLevel, Priority, ExecutionCategory

class BaseTool(ABC):
    @property
    def cost_tier(self) -> CostTier:
        """Cost tier for budget-aware tool selection.
        Default: CostTier.FREE
        """
        return CostTier.FREE

    @property
    def is_idempotent(self) -> bool:
        """Whether tool execution is idempotent (same input = same output, no side effects).
        Enables optimizations: caching, safe retries, parallel execution.
        Default: False
        """
        return False

    @property
    def metadata(self) -> Optional[ToolMetadata]:
        """Semantic metadata for tool selection.
        Return None for auto-generation from tool properties.
        """
        return None
```

### The execute() Method

The core method every tool must implement:

```python
from victor.tools.base import ToolResult

@abstractmethod
async def execute(self, _exec_ctx: Dict[str, Any], **kwargs: Any) -> ToolResult:
    """Execute the tool.

    Args:
        _exec_ctx: Framework execution context (reserved name to avoid collision
                  with tool parameters). Contains shared resources like code_manager.
        **kwargs: Tool parameters as defined in the parameters schema.

    Returns:
        ToolResult with execution outcome.
    """
    pass
```

### ToolResult Return Type

All tools return a `ToolResult` object:

```python
from pydantic import BaseModel, Field
from typing import Any, Dict, Optional

class ToolResult(BaseModel):
    """Result from tool execution."""

    success: bool = Field(..., description="Whether execution succeeded")
    output: Any = Field(..., description="Tool output data")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")
```

**Example usage:**

```python
# Success case
return ToolResult(
    success=True,
    output={"files": ["main.py", "utils.py"]},
    metadata={"count": 2}
)

# Error case
return ToolResult(
    success=False,
    output=None,
    error="File not found: config.yaml",
    metadata={"exception": "FileNotFoundError"}
)
```

---

## Tool Registration

Tools are registered via the `ToolRegistry` class defined in `victor/tools/registry.py`.

### ToolRegistry Class

```python
from victor.tools.registry import ToolRegistry

# Get or create a registry
registry = ToolRegistry()
```

### register() Method

```python
def register(self, *args, enabled: bool = True, **kwargs) -> None:
    """Register a tool - supports both signatures for flexibility.

    Signatures:
        register(tool)           - Auto-extracts name from tool object (common case)
        register(key, value)     - Explicit key-value (LSP-compatible with BaseRegistry)

    Args:
        *args: Either (tool,) or (key, value)
        enabled: Whether the tool is enabled by default (default: True)

    Examples:
        registry.register(my_tool)                    # Auto-extract name
        registry.register("custom_name", my_tool)    # Explicit name
        registry.register(decorated_function)         # From @tool decorator
    """
```

### Tool State Management

```python
# Enable/disable tools
registry.enable_tool("git") -> bool       # Returns True if successful
registry.disable_tool("web_search") -> bool

# Check if tool is enabled
registry.is_tool_enabled("git") -> bool

# Batch state management
registry.set_tool_states({"git": True, "web_search": False})
registry.get_tool_states() -> Dict[str, bool]
```

### Retrieving Tools

```python
# Get single tool
tool = registry.get("git") -> Optional[BaseTool]

# List all tools
all_tools = registry.list_tools(only_enabled=True) -> List[BaseTool]

# Get JSON schemas for LLM
schemas = registry.get_tool_schemas(only_enabled=True) -> List[Dict[str, Any]]
```

### Cost-Based Filtering

```python
from victor.tools.enums import CostTier

# Get tools at or below a cost tier
cheap_tools = registry.get_tools_by_cost(max_tier=CostTier.LOW, only_enabled=True)

# Get cost summary
summary = registry.get_cost_summary(only_enabled=True)
# Returns: {"free": ["read", "ls", ...], "low": [...], "medium": [...], "high": [...]}
```

### Hook Registration

Hooks allow you to intercept tool execution for logging, validation, or modification:

```python
from victor.tools.registry import Hook

# Before-execution hook
def before_hook(tool_name: str, arguments: Dict[str, Any]) -> None:
    print(f"Executing {tool_name} with {arguments}")

registry.register_before_hook(before_hook)

# After-execution hook
def after_hook(result: ToolResult) -> None:
    if not result.success:
        log_error(result.error)

registry.register_after_hook(after_hook)

# Critical hooks (failure blocks execution)
registry.register_before_hook(
    validation_hook,
    critical=True,
    name="security_validation"
)
```

### Hook Class

```python
class Hook:
    """Tool execution hook with metadata."""

    def __init__(
        self,
        callback: Callable,
        name: str = "",
        critical: bool = False,
        description: str = "",
    ):
        """Initialize hook.

        Args:
            callback: The hook function to call
            name: Human-readable name for the hook
            critical: If True, hook failure blocks tool execution
            description: Description of what the hook does
        """
```

---

## Parameter Schema

Tool parameters are defined using JSON Schema format (Draft 7).

### Basic Schema Structure

```python
parameters = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": "File path to read"
        },
        "limit": {
            "type": "integer",
            "description": "Maximum lines to read",
            "default": 100
        },
        "recursive": {
            "type": "boolean",
            "description": "Search recursively",
            "default": False
        }
    },
    "required": ["path"],
    "additionalProperties": False  # Reject unknown/hallucinated arguments
}
```

### Type Definitions

| Python Type | JSON Schema Type |
|------------|------------------|
| `str` | `{"type": "string"}` |
| `int` | `{"type": "integer"}` |
| `float` | `{"type": "number"}` |
| `bool` | `{"type": "boolean"}` |
| `List[str]` | `{"type": "array", "items": {"type": "string"}}` |
| `Dict[str, Any]` | `{"type": "object"}` |
| `Optional[str]` | `{"type": "string"}` (not in required) |

### Enum Parameters

```python
"operation": {
    "type": "string",
    "description": "Git operation to perform",
    "enum": ["status", "diff", "stage", "commit", "log", "branch"]
}
```

### Nested Objects

```python
"options": {
    "type": "object",
    "properties": {
        "verbose": {"type": "boolean"},
        "format": {"type": "string", "enum": ["json", "text"]}
    }
}
```

### Array with Typed Items

```python
"files": {
    "type": "array",
    "items": {"type": "string"},
    "description": "List of file paths to process"
}
```

### Complete Example

```python
parameters = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": "Directory path to search"
        },
        "pattern": {
            "type": "string",
            "description": "Glob pattern (e.g., '*.py')",
            "default": "*"
        },
        "recursive": {
            "type": "boolean",
            "description": "Search recursively",
            "default": False
        },
        "limit": {
            "type": "integer",
            "description": "Maximum results",
            "default": 100
        },
        "types": {
            "type": "array",
            "items": {"type": "string", "enum": ["file", "dir"]},
            "description": "Filter by type"
        }
    },
    "required": ["path"],
    "additionalProperties": False
}
```

---

## Cost Tiers

The `CostTier` enum (from `victor/tools/enums.py`) enables cost-aware tool selection:

```python
from victor.tools.enums import CostTier

class CostTier(Enum):
    """Cost tier for tools - used for budget-aware selection."""

    FREE = "free"      # weight: 0.0
    LOW = "low"        # weight: 1.0
    MEDIUM = "medium"  # weight: 2.0
    HIGH = "high"      # weight: 3.0

    @property
    def weight(self) -> float:
        """Return numeric weight for cost comparison."""
```

### When to Use Each Tier

| Tier | Use Case | Examples |
|------|----------|----------|
| `FREE` | Local operations with no external costs | `read`, `ls`, `grep`, `git status`, `shell` |
| `LOW` | Compute-only operations (no API calls) | `code_review`, `refactor_analysis`, `test_generation` |
| `MEDIUM` | External API calls | `web_search`, `web_fetch`, `slack_post` |
| `HIGH` | Resource-intensive operations | `batch_process` (100+ files), `full_codebase_scan` |

### Budget Implications

- Tools are deprioritized based on cost tier when cheaper alternatives exist
- The `get_tools_by_cost()` method filters tools by maximum tier
- Agent modes may restrict available cost tiers (e.g., EXPLORE mode limits HIGH tier)

---

## Access Modes

The `AccessMode` enum defines what kind of system access a tool requires:

```python
from victor.tools.enums import AccessMode

class AccessMode(Enum):
    """Tool access mode for approval tracking and security policies."""

    READONLY = "readonly"   # Only reads data, no side effects
    WRITE = "write"         # Modifies files or state
    EXECUTE = "execute"     # Runs external commands or code
    NETWORK = "network"     # Makes external network calls
    MIXED = "mixed"         # Multiple access types in same operation
```

### Safety Properties

```python
access_mode = AccessMode.WRITE

# Check if requires approval
access_mode.requires_approval  # True for WRITE, EXECUTE, MIXED

# Check if safe (read-only)
access_mode.is_safe  # True only for READONLY

# Get approval level (higher = stricter)
access_mode.approval_level()
# 0: Auto-approve (readonly)
# 1: Log only (network)
# 2: Confirm on first use (write)
# 3: Always confirm (execute, mixed)
```

### Examples by Mode

| Mode | Examples | Approval |
|------|----------|----------|
| `READONLY` | `read`, `ls`, `grep`, `code_search` | Auto-approve |
| `WRITE` | `write`, `edit`, `create_file` | Confirm on first use |
| `EXECUTE` | `shell`, `sandbox`, `run_tests` | Always confirm |
| `NETWORK` | `web_search`, `web_fetch`, `api_call` | Log only |
| `MIXED` | `git` (read+write+network), `docker` | Always confirm |

### Mode Restrictions

In certain agent modes, tools with specific access modes may be restricted:

- **EXPLORE mode**: WRITE and EXECUTE tools restricted to sandbox
- **PLAN mode**: Only READONLY and NETWORK allowed
- **BUILD mode**: All access modes available

---

## Tool Implementation Patterns

### Pattern 1: Class-Based Tool (BaseTool)

For complex tools with state or multiple methods:

```python
from typing import Any, Dict, Optional
from victor.tools.base import BaseTool, ToolResult, ToolMetadata
from victor.tools.enums import CostTier, AccessMode, DangerLevel

class CustomTool(BaseTool):
    """A custom tool implementation."""

    @property
    def name(self) -> str:
        return "custom_tool"

    @property
    def description(self) -> str:
        return "Performs custom operations on the codebase."

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "description": "Operation to perform",
                    "enum": ["analyze", "transform", "validate"]
                },
                "target": {
                    "type": "string",
                    "description": "Target path or identifier"
                },
                "options": {
                    "type": "object",
                    "description": "Additional options"
                }
            },
            "required": ["operation", "target"],
            "additionalProperties": False
        }

    @property
    def cost_tier(self) -> CostTier:
        return CostTier.LOW

    @property
    def metadata(self) -> Optional[ToolMetadata]:
        """Provide explicit semantic metadata."""
        return ToolMetadata(
            category="analysis",
            keywords=["analyze", "transform", "validate", "custom"],
            use_cases=[
                "analyzing code structure",
                "transforming code patterns",
                "validating configurations"
            ],
            examples=[
                "analyze the utils module",
                "transform all deprecated calls",
                "validate the config file"
            ],
            priority_hints=["Use for custom analysis operations"]
        )

    async def execute(self, _exec_ctx: Dict[str, Any], **kwargs: Any) -> ToolResult:
        operation = kwargs.get("operation")
        target = kwargs.get("target")
        options = kwargs.get("options", {})

        try:
            if operation == "analyze":
                result = await self._analyze(target, options)
            elif operation == "transform":
                result = await self._transform(target, options)
            elif operation == "validate":
                result = await self._validate(target, options)
            else:
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Unknown operation: {operation}"
                )

            return ToolResult(
                success=True,
                output=result,
                metadata={"operation": operation}
            )

        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=str(e),
                metadata={"exception": type(e).__name__}
            )

    async def _analyze(self, target: str, options: Dict) -> Dict:
        # Implementation...
        pass

    async def _transform(self, target: str, options: Dict) -> Dict:
        # Implementation...
        pass

    async def _validate(self, target: str, options: Dict) -> Dict:
        # Implementation...
        pass
```

### Pattern 2: Decorator-Based Tool (@tool)

For simpler tools, use the `@tool` decorator:

```python
from typing import List, Optional, Dict, Any
from victor.tools.decorators import tool
from victor.tools.enums import CostTier, Priority, AccessMode, DangerLevel

@tool(
    name="find_files",
    cost_tier=CostTier.FREE,
    category="filesystem",
    priority=Priority.HIGH,
    access_mode=AccessMode.READONLY,
    danger_level=DangerLevel.SAFE,
    keywords=["find", "search", "locate", "files"],
    use_cases=[
        "finding files by name pattern",
        "locating specific files",
        "searching for files recursively"
    ],
    examples=[
        "find all Python files",
        "locate the config file",
        "find files matching *.test.js"
    ],
    stages=["initial", "planning", "reading"],
    task_types=["search", "analysis"],
    progress_params=["path", "pattern"],
)
async def find_files(
    pattern: str,
    path: str = ".",
    recursive: bool = True,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """Find files matching a pattern.

    Args:
        pattern: Glob pattern to match (e.g., '*.py', 'test_*.js')
        path: Directory to search in (default: current directory)
        recursive: Search recursively (default: True)
        limit: Maximum results to return (default: 100)

    Returns:
        List of matching files with path, type, and size.
    """
    from pathlib import Path
    import fnmatch

    results = []
    base = Path(path).resolve()

    for p in base.rglob("*") if recursive else base.glob("*"):
        if fnmatch.fnmatch(p.name, pattern):
            results.append({
                "path": str(p.relative_to(base)),
                "type": "directory" if p.is_dir() else "file",
                "size": p.stat().st_size if p.is_file() else 0
            })
            if len(results) >= limit:
                break

    return results
```

---

## Tool Decorator

The `@tool` decorator (from `victor/tools/decorators.py`) provides a concise way to define tools.

### Decorator Parameters

```python
@tool(
    # Naming
    name: Optional[str] = None,           # Explicit tool name (overrides function name)
    aliases: Optional[List[str]] = None,  # Backward compatibility names

    # Cost and Budget
    cost_tier: CostTier = CostTier.FREE,  # Cost tier for budget-aware selection

    # Semantic Metadata
    category: Optional[str] = None,       # Tool category for grouping
    keywords: Optional[List[str]] = None, # Keywords for semantic matching
    use_cases: Optional[List[str]] = None,# High-level use cases
    examples: Optional[List[str]] = None, # Example requests
    priority_hints: Optional[List[str]] = None,  # Usage hints

    # Selection and Approval
    priority: Priority = Priority.MEDIUM,      # Selection priority
    access_mode: AccessMode = AccessMode.READONLY,  # Access type
    danger_level: DangerLevel = DangerLevel.SAFE,   # Danger level

    # Stage and Task Affinity
    stages: Optional[List[str]] = None,   # Conversation stages where relevant
    task_types: Optional[List[str]] = None,  # Task types for classification

    # Advanced
    mandatory_keywords: Optional[List[str]] = None,  # Keywords that force inclusion
    progress_params: Optional[List[str]] = None,     # Params for loop detection
    execution_category: Optional[str] = None,        # Parallel execution category
    availability_check: Optional[Callable[[], bool]] = None,  # Dynamic availability
)
```

### Priority Levels

```python
from victor.tools.enums import Priority

class Priority(Enum):
    CRITICAL = 1   # Always available (read, ls, grep, shell)
    HIGH = 2       # Most tasks (write, edit, git)
    MEDIUM = 3     # Task-specific (docker, db, test)
    LOW = 4        # Specialized (batch, scaffold)
    CONTEXTUAL = 5 # Based on task classification
```

### Danger Levels

```python
from victor.tools.enums import DangerLevel

class DangerLevel(Enum):
    SAFE = "safe"         # No risk of data loss
    LOW = "low"           # Minor risk, easily reversible
    MEDIUM = "medium"     # Moderate risk, may require effort to reverse
    HIGH = "high"         # Significant risk, difficult to reverse
    CRITICAL = "critical" # Irreversible or system-wide impact
```

### Conversation Stages

Tools can specify which conversation stages they're most relevant for:

- `initial` - First interaction, exploring the request
- `planning` - Understanding scope, searching for files
- `reading` - Examining files, gathering context
- `analysis` - Reviewing code, analyzing structure
- `execution` - Making changes, running commands
- `verification` - Testing, validating changes
- `completion` - Summarizing, wrapping up

### Execution Categories

For parallel execution planning:

```python
from victor.tools.enums import ExecutionCategory

class ExecutionCategory(Enum):
    READ_ONLY = "read_only"  # Pure reads, safe to parallelize
    WRITE = "write"          # File modifications, may conflict
    COMPUTE = "compute"      # CPU-intensive but isolated
    NETWORK = "network"      # External calls, rate-limited
    EXECUTE = "execute"      # Shell commands, may have side effects
    MIXED = "mixed"          # Multiple categories
```

---

## Advanced Features

### ToolMetadata

Semantic metadata for tool selection and discovery:

```python
from victor.tools.metadata import ToolMetadata
from victor.tools.enums import Priority, AccessMode, DangerLevel, ExecutionCategory

metadata = ToolMetadata(
    category="filesystem",
    keywords=["read", "file", "content", "view"],
    use_cases=["reading source files", "viewing configuration"],
    examples=["read main.py", "show config.yaml contents"],
    priority_hints=["Use for TEXT files only", "Use ls first to check size"],
    priority=Priority.CRITICAL,
    access_mode=AccessMode.READONLY,
    danger_level=DangerLevel.SAFE,
    stages=["reading", "initial", "analysis"],
    mandatory_keywords=["read file", "show file"],
    task_types=["analysis", "search"],
    progress_params=["path", "offset", "limit"],
    execution_category=ExecutionCategory.READ_ONLY
)
```

### ToolMetadataRegistry

Centralized registry for tool metadata:

```python
from victor.tools.metadata_registry import ToolMetadataRegistry

# Get singleton instance
registry = ToolMetadataRegistry.get_instance()

# Refresh from tools (smart reindexing)
if registry.needs_reindex(tools):
    registry.refresh_from_tools(tools)

# Access metadata
metadata = registry.get_metadata("git")
tools_in_category = registry.get_tools_by_category("filesystem")
tools_with_keyword = registry.get_by_keyword("search")

# Search across all metadata
matches = registry.get_tools_matching_text("file")

# Get statistics
stats = registry.get_statistics()
# Returns: {
#   "total_tools": 55,
#   "tools_with_explicit_metadata": 42,
#   "tools_with_auto_metadata": 13,
#   "total_categories": 12,
#   "total_keywords": 234,
#   "categories": ["filesystem", "git", "web", ...]
# }
```

### Parameter Validation

Tools automatically validate parameters against their schema:

```python
# Simple validation (returns bool)
is_valid = tool.validate_parameters(path="/tmp/test.txt", limit=100)

# Detailed validation (returns ToolValidationResult)
result = tool.validate_parameters_detailed(path="/tmp/test.txt", limit="invalid")

if not result.valid:
    print(f"Errors: {result.errors}")
    # ["Parameter 'limit' has wrong type: expected integer, got str"]
    print(f"Invalid params: {result.invalid_params}")
    # {"limit": "type: expected integer"}
```

### Schema Levels

Control schema verbosity for LLM context management:

```python
from victor.tools.enums import SchemaLevel

# FULL: Complete schema (~100-150 tokens)
full_schema = tool.to_schema(SchemaLevel.FULL)

# COMPACT: All params, shorter descriptions (~60-80 tokens)
compact_schema = tool.to_schema(SchemaLevel.COMPACT)

# STUB: Required params only (~25-40 tokens)
stub_schema = tool.to_schema(SchemaLevel.STUB)
```

### Availability Checks

For tools requiring external configuration:

```python
def is_slack_configured() -> bool:
    """Check if Slack API token is available."""
    import os
    return bool(os.getenv("SLACK_API_TOKEN"))

@tool(
    name="slack_post",
    availability_check=is_slack_configured,
)
async def slack_post(channel: str, message: str):
    """Post a message to Slack."""
    # ...
```

Tools with availability checks are excluded from tool selection when unavailable.

### Tool Aliases

Support backward compatibility with renamed tools:

```python
@tool(
    name="shell",              # Canonical name
    aliases=["execute_bash"],  # Legacy names that still work
)
async def shell(command: str):
    """Execute a shell command."""
    # ...
```

---

## Example: Complete Tool Implementation

Here's a complete example showing all features:

```python
from typing import Any, Dict, List, Optional
from victor.tools.decorators import tool
from victor.tools.enums import (
    CostTier,
    Priority,
    AccessMode,
    DangerLevel,
    ExecutionCategory,
)

@tool(
    name="code_metrics",
    cost_tier=CostTier.LOW,
    category="analysis",
    priority=Priority.MEDIUM,
    access_mode=AccessMode.READONLY,
    danger_level=DangerLevel.SAFE,
    keywords=[
        "metrics",
        "complexity",
        "lines of code",
        "loc",
        "cyclomatic",
        "analyze",
    ],
    use_cases=[
        "analyzing code complexity",
        "measuring code metrics",
        "calculating lines of code",
        "assessing code quality",
    ],
    examples=[
        "show metrics for main.py",
        "calculate complexity of utils module",
        "how many lines of code in src/",
    ],
    priority_hints=[
        "Use for code quality assessment",
        "Supports Python, JavaScript, TypeScript",
    ],
    stages=["analysis", "verification"],
    task_types=["analysis"],
    progress_params=["path", "metric_type"],
    execution_category="compute",
    mandatory_keywords=["code metrics", "cyclomatic complexity"],
)
async def code_metrics(
    path: str,
    metric_type: str = "all",
    recursive: bool = True,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Calculate code metrics for files or directories.

    Computes various code quality metrics including lines of code,
    cyclomatic complexity, and maintainability index.

    Args:
        path: File or directory path to analyze.
        metric_type: Type of metrics to calculate.
            Options: "loc" (lines of code), "complexity", "all".
            Default: "all"
        recursive: Analyze subdirectories recursively. Default: True.
        context: Tool execution context (injected by framework).

    Returns:
        Dictionary containing:
        - files_analyzed: Number of files processed
        - total_lines: Total lines of code
        - complexity: Average cyclomatic complexity
        - metrics_by_file: Per-file breakdown
    """
    from pathlib import Path

    target = Path(path).resolve()

    if not target.exists():
        return {"error": f"Path not found: {path}"}

    files = []
    if target.is_file():
        files = [target]
    else:
        pattern = "**/*" if recursive else "*"
        files = [f for f in target.glob(pattern) if f.is_file()]

    results = {
        "files_analyzed": len(files),
        "total_lines": 0,
        "complexity": 0.0,
        "metrics_by_file": {}
    }

    for file in files:
        if file.suffix in {".py", ".js", ".ts"}:
            try:
                content = file.read_text()
                lines = len(content.splitlines())
                results["total_lines"] += lines
                results["metrics_by_file"][str(file)] = {
                    "lines": lines,
                    "complexity": 1.0  # Placeholder
                }
            except Exception:
                pass

    if results["metrics_by_file"]:
        avg_complexity = sum(
            m["complexity"] for m in results["metrics_by_file"].values()
        ) / len(results["metrics_by_file"])
        results["complexity"] = round(avg_complexity, 2)

    return results
```

---

## See Also

- [Tool architecture](../architecture.md#tool-system) - Tool registration and execution
- [victor/tools/base.py](https://github.com/anvai-labs/victor/blob/develop/victor/tools/base.py) - BaseTool implementation
- [victor/tools/registry.py](https://github.com/anvai-labs/victor/blob/develop/victor/tools/registry.py) - ToolRegistry implementation
- [victor/tools/decorators.py](https://github.com/anvai-labs/victor/blob/develop/victor/tools/decorators.py) - @tool decorator
- [victor/tools/enums.py](https://github.com/anvai-labs/victor/blob/develop/victor/tools/enums.py) - CostTier, AccessMode, Priority enums
- [victor/tools/metadata.py](https://github.com/anvai-labs/victor/blob/develop/victor/tools/metadata.py) - ToolMetadata and registry
