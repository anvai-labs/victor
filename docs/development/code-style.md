# Code Style Guide

Victor enforces consistent code style through automated tools. This guide covers formatting, linting, and style conventions.

## Quick Reference

```bash
# Format code (run before commits)
make format

# Check code style (CI runs this)
make lint

# Individual tools
black victor tests                    # Format with Black
ruff check --fix victor tests         # Lint and auto-fix with Ruff
mypy victor                           # Type check with MyPy
```

## Tools Overview

| Tool | Purpose | Configuration |
|------|---------|---------------|
| **Black** | Code formatting | `pyproject.toml` `[tool.black]` |
| **Ruff** | Linting (replaces flake8) | `pyproject.toml` `[tool.ruff]` |
| **MyPy** | Type checking | `pyproject.toml` `[tool.mypy]` |

## Black Formatting

Black is our code formatter with zero configuration decisions. It ensures consistent style across the codebase.

### Configuration

From `pyproject.toml`:

```toml
[tool.black]
line-length = 100
target-version = ['py310']
include = '\.pyi?$'
```

### Key Rules

- **Line length**: 100 characters (not the default 88)
- **Target version**: Python 3.11+
- **String quotes**: Double quotes preferred
- **Trailing commas**: Added in multi-line structures

### Running Black

```bash
# Format all code
black victor tests

# Check without modifying (CI mode)
black --check victor tests

# Show what would change
black --diff victor tests
```

### Examples

```python
# Before Black
def function(arg1,arg2,arg3):return arg1+arg2+arg3

# After Black
def function(arg1, arg2, arg3):
    return arg1 + arg2 + arg3
```

```python
# Long function calls are split with trailing commas
result = some_function(
    argument_one="value",
    argument_two="another_value",
    argument_three="third_value",  # Trailing comma
)
```

## Ruff Linting

Ruff is a fast Python linter that replaces flake8, isort, and other tools.

### Configuration

From `pyproject.toml`:

```toml
[tool.ruff]
line-length = 100
target-version = "py310"
extend-exclude = [
    "archive",
    "examples",
    "venv",
]

[tool.ruff.lint]
select = [
    "E",  # pycodestyle errors
    "W",  # pycodestyle warnings
    "F",  # pyflakes
    "B",  # flake8-bugbear
    "C4", # flake8-comprehensions
]
ignore = [
    "E501",  # line too long (handled by Black)
    "E402",  # module level import not at top
    "F401",  # unused imports (often intentional re-exports)
    "F811",  # redefined while unused
    "B007",  # loop control variable not used
    "B008",  # function calls in argument defaults
]

[tool.ruff.lint.per-file-ignores]
"__init__.py" = ["F401"]  # Allow unused imports in __init__.py
"tests/**/*.py" = ["B017", "B018", "E741", "F841"]
```

### Running Ruff

```bash
# Check for issues
ruff check victor tests

# Auto-fix issues
ruff check --fix victor tests

# Show all issues including fixable
ruff check --show-fixes victor tests
```

### Common Ruff Rules

| Rule | Description | Example |
|------|-------------|---------|
| `E` | pycodestyle errors | Missing whitespace, indentation |
| `W` | pycodestyle warnings | Trailing whitespace |
| `F` | pyflakes | Undefined names, unused imports |
| `B` | bugbear | Common bugs and design issues |
| `C4` | comprehensions | Unnecessary list/dict calls |

### Fixing Common Issues

```python
# F401: Unused import - remove or use
import os  # Remove if unused

# F841: Unused variable - prefix with underscore
_result = function()  # Explicitly unused

# B006: Mutable default argument
def bad(items=[]):  # Don't do this
    pass

def good(items=None):  # Do this instead
    items = items or []
```

## MyPy Type Checking

MyPy verifies type hints for type safety and better IDE support.

### Configuration

From `pyproject.toml`:

```toml
[tool.mypy]
python_version = "3.10"
strict = false                    # Gradual typing adoption
warn_return_any = false
warn_unused_configs = true
disallow_untyped_defs = false
ignore_missing_imports = true
no_implicit_optional = true
exclude = [
    "^archive/",
    "^tests/",
    "^scripts/",
]
files = ["victor"]

# Strictly typed modules
[[tool.mypy.overrides]]
module = [
    "victor.config.*",
    "victor.storage.cache.*",
    "victor.telemetry.*",
    "victor.analytics.*",
    "victor.profiler.*",
    "victor.debug.*",
]
strict = true
```

### Running MyPy

```bash
# Type check the codebase
mypy victor

# Run the strict-package gate (same command as CI)
mypy --strict \
  victor/config \
  victor/storage/cache \
  victor/telemetry \
  victor/analytics \
  victor/profiler \
  victor/debug

# Type check specific file
mypy victor/agent/orchestrator.py

# Show error codes
mypy victor --show-error-codes
```

### Type Hint Requirements

Type hints are required on all public APIs:

```python
from typing import List, Dict, Optional, Any, AsyncIterator

def public_function(
    name: str,
    count: int,
    options: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Public functions need type hints."""
    pass

async def async_function(
    items: List[str],
) -> AsyncIterator[str]:
    """Async functions too."""
    for item in items:
        yield item
```

### Common Type Patterns

```python
from typing import TYPE_CHECKING, Union, Callable, TypeVar

# Type-only imports (no runtime cost)
if TYPE_CHECKING:
    from victor.providers.base import BaseProvider

# Union types
def process(value: Union[str, int]) -> str:
    return str(value)

# Generic types
T = TypeVar("T")
def first(items: List[T]) -> Optional[T]:
    return items[0] if items else None

# Callable types
Handler = Callable[[str], None]
def register_handler(handler: Handler) -> None:
    pass
```

## Docstrings

Victor uses Google-style docstrings for documentation.

### Function Docstrings

```python
def complex_function(
    arg1: str,
    arg2: int,
    optional_arg: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Short one-line summary.

    Longer description explaining what the function does,
    why it exists, and any important details. This can span
    multiple lines.

    Args:
        arg1: Description of arg1. If the description is long,
            it can continue on the next line with indentation.
        arg2: Description of arg2.
        optional_arg: Description of optional argument.
            Defaults to None.

    Returns:
        Description of return value. For complex return types,
        describe the structure:
        - key1: Description of key1
        - key2: Description of key2

    Raises:
        ValueError: When arg1 is empty.
        TypeError: When arg2 is not an integer.

    Examples:
        >>> result = complex_function("test", 42)
        >>> result["status"]
        'success'
    """
    pass
```

### Class Docstrings

```python
class MyClass:
    """Short summary of the class.

    Longer description of the class, its purpose, and usage.
    Include any important notes about lifecycle or state.

    Attributes:
        name: The name of the instance.
        config: Configuration dictionary.
        _internal: Private attribute (not documented usually).

    Examples:
        >>> obj = MyClass("example")
        >>> obj.name
        'example'
    """

    def __init__(self, name: str, config: Optional[Dict] = None) -> None:
        """Initialize MyClass.

        Args:
            name: The name for this instance.
            config: Optional configuration dictionary.
        """
        self.name = name
        self.config = config or {}
```

### Module Docstrings

```python
"""Module providing XYZ functionality.

This module contains classes and functions for handling XYZ.
It is part of the larger ABC system.

Typical usage:
    from victor.module import MyClass

    instance = MyClass()
    result = instance.process()

See Also:
    victor.related_module: Related functionality
    victor.other_module: Other related module
"""
```

## Import Organization

Imports should be organized in groups:

```python
# Standard library imports
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Third-party imports
import httpx
from pydantic import BaseModel

# Local imports
from victor.config import Settings
from victor.tools.base import BaseTool
from victor.providers.base import BaseProvider
```

Ruff handles import sorting automatically with `--fix`.

## Naming Conventions

| Type | Convention | Example |
|------|------------|---------|
| Modules | lowercase_with_underscores | `tool_registry.py` |
| Classes | PascalCase | `AgentOrchestrator` |
| Functions | lowercase_with_underscores | `execute_tool()` |
| Constants | UPPERCASE_WITH_UNDERSCORES | `MAX_RETRIES` |
| Private | Leading underscore | `_internal_method()` |
| Type Variables | Single uppercase or PascalCase | `T`, `KeyType` |

### Examples

```python
# Constants
MAX_RETRIES = 3
DEFAULT_TIMEOUT = 30.0

# Classes
class MyToolExecutor:
    """Tool executor class."""

    def __init__(self) -> None:
        self._internal_state = {}  # Private attribute

    def execute_tool(self, tool_name: str) -> None:
        """Public method."""
        self._validate_tool(tool_name)

    def _validate_tool(self, name: str) -> None:
        """Private method."""
        pass
```

### Package & Directory Naming (Singular vs Plural)

Package and directory names follow one rule so that the same concept never
appears as both `foo/` and `foos/` siblings:

1. **Top-level packages and their `tests/` mirrors are plural** when they hold
   a collection of modules: `victor/tools`, `victor/providers`, `victor/protocols`,
   `victor/verticals`, mirrored by `tests/unit/tools`, `tests/unit/providers`, …
   A test directory mirrors its source package name **exactly**.
2. **Singular** is for a sub-package describing one concrete subsystem or a
   proper/mass noun: the `victor/benchmark` vertical, `mode_config`, `profiler`.
3. A **category** test directory that is *not* a package mirror uses a descriptive
   name, never the bare noun — `tests/performance`, not `tests/benchmark`.
4. **No parent holds both forms of one concept.** If a stem names two distinct
   things, disambiguate by renaming one (e.g. `agents/` meaning multi-agent
   coordination → `multi_agent/`).

```text
# Good — plural collection, test mirror matches source
victor/tools/              tests/unit/tools/

# Good — singular concrete subsystem (proper noun)
victor/benchmark/          tests/unit/benchmark/

# Bad — same concept as both singular and plural siblings (now folded)
tests/benchmark/  +  tests/benchmarks/        →  tests/benchmarks/
victor/integrations/protocol/  +  .../protocols/  →  query_enhancement moved to victor/protocols/
```

Cross-layer splits where singular and plural describe genuinely different things
at **different** layers are fine and documented, not folded — e.g.
`victor/agent/conversation` (the conversation runtime/store) vs
`victor/framework/conversations` (the coordination framework), and the
`benchmark` vertical vs `victor/evaluation/benchmarks` (eval-dataset suites).

This convention is **enforced in CI** by
`tests/unit/runtime/test_naming_singular_plural_guard.py`, which fails on any
new same-parent singular/plural directory collision. That guard test is the
authoritative statement of the rule; update it (and its allowlist) if the
convention changes.

## Async/Await Conventions

All I/O operations should be async:

```python
# Good: Async I/O
async def fetch_data(url: str) -> Dict[str, Any]:
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        return response.json()

# Good: Async file operations
async def read_file(path: Path) -> str:
    import aiofiles
    async with aiofiles.open(path) as f:
        return await f.read()

# Bad: Blocking I/O in async context
async def bad_fetch(url: str) -> Dict[str, Any]:
    import requests  # Don't use sync requests
    response = requests.get(url)  # This blocks!
    return response.json()
```

## Line Length

Maximum line length is 100 characters. Black handles wrapping automatically.

```python
# Good: Within 100 characters
result = process_data(input_value, config=settings)

# Good: Black wraps long lines
result = some_very_long_function_name(
    first_argument="value",
    second_argument="another_value",
    third_argument="yet_another_value",
)

# Good: String continuation
message = (
    "This is a very long message that needs to be split "
    "across multiple lines for readability."
)
```

## Pre-commit Integration

Victor supports pre-commit hooks. Create `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 24.4.0
    hooks:
      - id: black
        args: [--line-length=100]

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.4.0
    hooks:
      - id: ruff
        args: [--fix]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.10.0
    hooks:
      - id: mypy
        args: [--ignore-missing-imports]
        additional_dependencies: [types-pyyaml, types-aiofiles]
```

Install and run:

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

## CI/CD Checks

All pull requests must pass these checks:

```bash
# These commands run in CI
ruff check victor             # No lint errors
black --check victor          # Code is formatted
mypy victor --ignore-missing-imports  # Type check passes
```

If CI fails:

```bash
# Fix formatting
make format

# Check what's wrong
make lint
```

## Common Style Issues

### Issue: Long Import Lines

```python
# Bad
from victor.agent.orchestrator import AgentOrchestrator, ConversationController, ToolPipeline

# Good
from victor.agent.orchestrator import (
    AgentOrchestrator,
    ConversationController,
    ToolPipeline,
)
```

### Issue: Missing Type Hints on Public API

```python
# Bad
def get_tools(category):
    pass

# Good
def get_tools(category: str) -> List[BaseTool]:
    pass
```

### Issue: Mutable Default Arguments

```python
# Bad
def process(items=[]):
    items.append("new")
    return items

# Good
def process(items: Optional[List[str]] = None) -> List[str]:
    items = items or []
    items.append("new")
    return items
```

### Issue: Bare Except Clauses

```python
# Bad
try:
    risky_operation()
except:
    pass

# Good
try:
    risky_operation()
except Exception as e:
    logger.error(f"Operation failed: {e}")
```

## Next Steps

- [Setup Guide](setup.md) - Development environment setup
- [Testing Guide](testing.md) - Testing patterns and fixtures
- [Contributing Guide](../../CONTRIBUTING.md) - Pull request process

---

**Questions?** Open an issue on [GitHub](https://github.com/vijayksingh/victor/issues) or start a [discussion](https://github.com/vijayksingh/victor/discussions).
