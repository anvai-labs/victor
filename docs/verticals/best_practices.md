# Vertical Development Best Practices

**Version**: 1.0
**Date**: 2026-03-31
**Target Audience**: Vertical Developers

## Table of Contents

1. [Registration](#registration)
2. [Version Management](#version-management)
3. [Dependency Declaration](#dependency-declaration)
4. [Configuration](#configuration)
5. [Namespace Usage](#namespace-usage)
6. [Testing](#testing)
7. [Performance](#performance)
8. [Error Handling](#error-handling)
9. [Documentation](#documentation)
10. [Common Patterns](#common-patterns)

---

## Registration

Unless noted otherwise, the examples in this guide assume new external verticals are
defined against `victor-contracts`:

```python
from victor_contracts import VerticalBase, register_vertical
```

### DO: Use the @register_vertical Decorator

Always use the decorator for new verticals:

```python
@register_vertical(
    name="my_vertical",
    version="1.0.0",
)
class MyVertical(VerticalBase):
    # ...
```

**Why**:
- ✅ Declarative and self-documenting
- ✅ Type-safe metadata extraction
- ✅ Enables all new features
- ✅ Future-proof

### DON'T: Rely on Legacy Naming Patterns

Avoid relying on implicit naming:

```python
# Discouraged
class CodingAssistant(VerticalBase):  # Implicit "coding"
    # ...
```

**Why**:
- ❌ Fragile to refactoring
- ❌ Not self-documenting
- ❌ Deprecation warnings

---

## Version Management

### DO: Follow PEP 440

Use semantic versioning (MAJOR.MINOR.PATCH):

```python
@register_vertical(
    name="my_vertical",
    version="1.0.0",  # PEP 440 compliant
)
```

**Why**:
- ✅ Standard versioning
- ✅ Compatible with version constraints
- ✅ Clear compatibility semantics

**Versioning Rules**:
- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### DON'T: Use Non-Standard Versions

Avoid non-standard version formats:

```python
# Wrong
@register_vertical(
    name="my_vertical",
    version="1.0",  # Missing patch
    version="v1.0.0",  # Has prefix
    version="latest",  # Not a version
)
```

**Why**:
- ❌ Breaks version constraints
- ❌ Incompatible with PEP 440
- ❌ Unpredictable behavior

### DO: Set Framework Version Constraints

Declare minimum compatible framework version:

```python
@register_vertical(
    name="my_vertical",
    version="1.0.0",
    min_framework_version=">=0.6.0",  # Requires Victor 0.6.0+
)
```

**Why**:
- ✅ Prevents runtime incompatibilities
- ✅ Clear error messages
- ✅ Safe deployment

**Version Constraint Syntax** (PEP 440):
- `">=0.6.0"`: Version 0.6.0 or higher
- `">=0.6.0,<0.7.0"`: Version 0.6.x only
- `"~=0.6.0"`: Compatible release (0.6.0 ≤ v < 0.7.0)

---

## Dependency Declaration

### DO: Declare Extension Dependencies

Explicitly declare required extensions:

```python
@register_vertical(
    name="my_vertical",
    version="1.0.0",
    extension_dependencies=[
        "base_tools",  # Required
        "middleware",  # Required
    ],
)
```

**Why**:
- ✅ Ordered loading
- ✅ Dependency validation
- ✅ Clear error messages

### DO: Use Optional Dependencies Gracefully

Mark non-critical dependencies as optional:

```python
@register_vertical(
    name="my_vertical",
    version="1.0.0",
    extension_dependencies=[
        "base_tools",  # Required
    ],
)

# In code:
try:
    optional_feature = load_extension("optional_feature")
except ImportError:
    optional_feature = None  # Graceful degradation
```

**Why**:
- ✅ Graceful degradation
- ✅ Broader compatibility
- ✅ Better user experience

### DON'T: Over-Declare Dependencies

Only declare direct dependencies:

```python
# Wrong
extension_dependencies=[
    "base_tools",  # Direct dependency
    "middleware",  # Direct dependency
    "logging",  # Indirect (middleware already needs it)
]

# Correct
extension_dependencies=[
    "base_tools",  # Direct dependency only
    "middleware",  # Direct dependency only
]
```

**Why**:
- ❌ Unnecessary constraints
- ❌ Harder to maintain
- ❌ May conflict with other verticals

---

## Configuration

### DO: Configure Behavior Explicitly

Set behavior options explicitly:

```python
@register_vertical(
    name="my_vertical",
    version="1.0.0",
    canonicalize_tool_names=True,  # Auto-prefix tools
    tool_dependency_strategy="auto",  # Auto-detect dependencies
    strict_mode=False,  # Allow missing tools gracefully
)
```

**Why**:
- ✅ Predictable behavior
- ✅ Self-documenting
- ✅ Clear intent

### DO: Use Strict Mode for Critical Verticals

Enable strict mode for critical functionality:

```python
@register_vertical(
    name="security_critical",
    version="1.0.0",
    strict_mode=True,  # Fail on missing tools
)
```

**Why**:
- ✅ Fail-fast on errors
- ✅ Prevents silent failures
- ✅ Clear error messages

### DON'T: Override Default Behavior Unnecessarily

Only override defaults when needed:

```python
# Discouraged (redundant defaults)
@register_vertical(
    name="my_vertical",
    version="1.0.0",
    canonicalize_tool_names=True,  # Default is True
    strict_mode=False,  # Default is False
)

# Good (explicit only when different)
@register_vertical(
    name="my_vertical",
    version="1.0.0",
    canonicalize_tool_names=False,  # Override default
)
```

**Why**:
- ❌ Verbose
- ❌ Harder to maintain
- ❌ Default values may change

---

## Namespace Usage

### DO: Use Namespaces to Prevent Collisions

Register plugins in appropriate namespaces:

```python
# External verticals
@register_vertical(
    name="my_tool",
    version="1.0.0",
    plugin_namespace="external",  # External package
)
```

**Why**:
- ✅ Prevents naming collisions
- ✅ Clear ownership
- ✅ Priority-based resolution

**Namespace Priority**:
1. **external** (100): Third-party packages
2. **contrib** (50): Built-in contrib verticals
3. **experimental** (25): Experimental features
4. **default** (0): Fallback namespace

### DO: Use Custom Namespaces for Your Organization

Create organization-specific namespaces:

```python
@register_vertical(
    name="my_tool",
    version="1.0.0",
    plugin_namespace="my_company",  # Custom namespace
)

# Keep the canonical entry point group and namespace the plugin name if desired
# pyproject.toml:
# [project.entry-points."victor.plugins"]
# my_company.my_tool = "my_package.plugin:plugin"
```

**Why**:
- ✅ Organizational branding
- ✅ Isolated from other plugins
- ✅ Custom priority control

### DON'T: Use Default Namespace for Everything

Avoid the default namespace for external plugins:

```python
# Discouraged
@register_vertical(
    name="my_tool",
    version="1.0.0",
    # plugin_namespace defaults to "default"
)

# Recommended
@register_vertical(
    name="my_tool",
    version="1.0.0",
    plugin_namespace="external",  # Explicit
)
```

**Why**:
- ❌ No priority control
- ❌ May collide with other plugins
- ❌ Unclear ownership

---

## Testing

### DO: Write Comprehensive Tests

Start with SDK contract validation, then add full-runtime integration coverage only
when you need to verify activation inside `victor-ai`:

```python
from victor_contracts.validation import validate_vertical_package

def test_sdk_contracts():
    """Validate the published vertical package contract."""
    report = validate_vertical_package("my-vertical")
    assert report.is_valid
```

For full-runtime verification inside a Victor environment, add a separate integration
test:

```python
import pytest

from victor.core.verticals import VerticalLoader


class TestMyVerticalIntegration:
    def test_loading(self):
        """Test vertical loads successfully inside victor-ai."""
        loader = VerticalLoader()
        vertical = loader.load("my_vertical")
        assert vertical is not None

    def test_tools(self):
        """Test tool registration."""
        vertical = MyVertical()
        tools = vertical.get_tools()
        assert "required_tool" in tools

    def test_system_prompt(self):
        """Test system prompt."""
        vertical = MyVertical()
        prompt = vertical.get_system_prompt()
        assert prompt is not None
        assert len(prompt) > 0
```

**Why**:
- ✅ Validates registration
- ✅ Validates loading
- ✅ Validates functionality
- ✅ Catches regressions

### DO: Test Backward Compatibility

Ensure your vertical works with existing code:

```python
def test_backward_compatibility():
    """Test legacy patterns still work."""
    # Old code should still work
    vertical = MyVertical()
    tools = vertical.get_tools()
    prompt = vertical.get_system_prompt()
    assert tools is not None
    assert prompt is not None
```

**Why**:
- ✅ Ensures smooth migration
- ✅ Validates no breaking changes
- ✅ Consumer compatibility

### DO: Test Error Handling

Test error conditions:

```python
def test_missing_tools_graceful():
    """Test graceful degradation when tools missing."""
    vertical = MyVertical()

    # Should not crash, should degrade gracefully
    if not tool_available("advanced_tool"):
        result = vertical.use_basic_tool()
        assert result is not None
```

**Why**:
- ✅ Validates error handling
- ✅ Tests graceful degradation
- ✅ Prevents crashes

---

## Performance

### DO: Use Async Operations for I/O

Use async operations for slow I/O:

```python
@register_vertical(
    name="my_vertical",
    version="1.0.0",
)

class MyVertical(VerticalBase):
    @classmethod
    async def load_resources_async(cls):
        """Load resources asynchronously."""
        # Async I/O operations
        resources = await load_from_database_async()
        return resources
```

**Why**:
- ✅ Non-blocking
- ✅ Better performance
- ✅ Parallel loading support

### DO: Cache Expensive Operations

Prefer package-local lazy caching for definition-layer code. Core runtime caching,
telemetry, and DI are composed by Victor during activation.

```python
import asyncio

class MyVertical(VerticalBase):
    _heavy_object = None
    _heavy_object_lock = asyncio.Lock()

    @classmethod
    async def get_heavy_object(cls):
        """Get or create expensive object."""
        if cls._heavy_object is None:
            async with cls._heavy_object_lock:
                if cls._heavy_object is None:
                    cls._heavy_object = await build_heavy_object_async()
        return cls._heavy_object
```

**Why**:
- ✅ Avoid repeated expensive operations
- ✅ Thread-safe
- ✅ Parallel loading safe

### DON'T: Block on Slow Operations

Avoid blocking operations in loading:

```python
# Wrong
class MyVertical(VerticalBase):
    @classmethod
    def get_resources(cls):
        # Blocks for 5 seconds!
        result = slow_database_call()
        return result

# Correct
class MyVertical(VerticalBase):
    @classmethod
    async def get_resources_async(cls):
        # Non-blocking
        result = await slow_database_call_async()
        return result
```

**Why**:
- ❌ Blocks startup
- ❌ Poor performance
- ❌ Bad user experience

---

## Error Handling

### DO: Provide Clear Error Messages

Give users actionable error messages:

```python
class MyVertical(VerticalBase):
    @classmethod
    def get_tools(cls):
        tools = []

        required_tool = "critical_tool"
        if not tool_available(required_tool):
            raise RuntimeError(
                f"MyVertical requires '{required_tool}' but it is not available. "
                f"Please install the required package: pip install victor-{required_tool}"
            )

        return tools
```

**Why**:
- ✅ Actionable feedback
- ✅ Clear resolution path
- ✅ Better debugging

### DO: Use Appropriate Exception Types

Use specific exception types:

```python
# ImportError: Missing dependencies
if not dependency_available():
    raise ImportError(f"Required dependency '{dep}' is not available")

# ValueError: Invalid configuration
if not valid_config(config):
    raise ValueError(f"Invalid configuration: {config}")

# RuntimeError: Runtime failures
if operation_failed():
    raise RuntimeError(f"Operation failed: {reason}")
```

**Why**:
- ✅ Clear error categorization
- ✅ Easier error handling
- ✅ Better debugging

### DON'T: Catch All Exceptions Silently

Avoid broad exception catching:

```python
# Wrong
try:
    risky_operation()
except Exception:
    pass  # Silent failure

# Correct
try:
    risky_operation()
except SpecificError as e:
    logger.warning(f"Operation failed: {e}")
    # Graceful degradation
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    raise  # Re-raise unexpected errors
```

**Why**:
- ❌ Hides bugs
- ❌ Hard to debug
- ❌ Silent failures

---

## Documentation

### DO: Document Your Vertical

Provide comprehensive documentation:

```python
@register_vertical(
    name="my_vertical",
    version="1.0.0",
    description="Brief description of what this vertical does",
)
class MyVertical(VerticalBase):
    """My vertical for doing X.

    This vertical provides functionality for:
    - Feature 1: Description
    - Feature 2: Description
    - Feature 3: Description

    Example:
        >>> vertical = MyVertical()
        >>> result = vertical.do_something()
        >>> print(result)

    Requirements:
        - Python 3.11+
        - Victor 0.6.0+
        - External package: some-package>=1.0.0

    Configuration:
        Use canonicalize_tool_names=True for auto-prefixed tools.
    """

    @classmethod
    def get_description(cls) -> str:
        """Return human-readable description."""
        return "My vertical for doing X"
```

**Why**:
- ✅ Clear usage instructions
- ✅ Requirements documented
- ✅ Examples provided

### DO: Document Required Tools

List required tools:

```python
class MyVertical(VerticalBase):
    """My vertical.

    Required Tools:
        - read: File reading
        - write: File writing
        - search: Code search

    Optional Tools:
        - advanced_search: Enhanced search (requires index)
    """

    @classmethod
    def get_tools(cls) -> list[str]:
        return ["read", "write", "search"]
```

**Why**:
- ✅ Clear dependencies
- ✅ Installation instructions
- ✅ Feature discovery

---

## Common Patterns

### Pattern 1: Vertical with External Dependencies

```python
@register_vertical(
    name="my_vertical",
    version="1.0.0",
    min_framework_version=">=0.6.0",
    extension_dependencies=["external_package"],
    strict_mode=False,  # Graceful degradation
)
class MyVertical(VerticalBase):
    """Vertical with external dependencies."""

    @classmethod
    def get_tools(cls) -> list[str]:
        tools = ["base_tool"]

        # Add external tools if available
        if tool_available("external_tool"):
            tools.append("external_tool")

        return tools
```

### Pattern 2: Configurable Vertical

```python
@register_vertical(
    name="my_vertical",
    version="1.0.0",
    canonicalize_tool_names=False,  # Custom tool names
    tool_dependency_strategy="explicit",
)
class MyVertical(VerticalBase):
    """Configurable vertical."""

    @classmethod
    def get_config(cls) -> VerticalConfig:
        return VerticalConfig(
            name="my_vertical",
            tools=[
                ToolRequirement("custom_tool_1"),
                ToolRequirement("custom_tool_2"),
            ],
        )
```

### Pattern 3: Experimental Vertical

```python
@register_vertical(
    name="experimental_feature",
    version="0.1.0",
    min_framework_version=">=0.6.0",
    description="Experimental features (unstable)",
    load_priority=0,  # Load last
)
class ExperimentalVertical(VerticalBase):
    """Experimental vertical."""

    @classmethod
    def get_tools(cls) -> list[str]:
        logger.warning("ExperimentalVertical: Features may be unstable")
        return ["experimental_tool"]
```

### Pattern 4: Multi-Version Support

```python
# v1.py
@register_vertical(
    name="my_tool",
    version="1.0.0",
    min_framework_version=">=0.5.0",
)
class MyToolV1(VerticalBase):
    """Version 1.0.0 of my tool."""
    # ...

# v2.py
@register_vertical(
    name="my_tool",
    version="2.0.0",
    min_framework_version=">=0.6.0",
)
class MyToolV2(VerticalBase):
    """Version 2.0.0 of my tool."""
    # ...
```

---

## Checklist

Use this checklist when creating a new vertical:

### Registration
- [ ] Used `@register_vertical` decorator
- [ ] Set PEP 440 compliant version
- [ ] Set `min_framework_version` constraint
- [ ] Added description
- [ ] Set appropriate namespace

### Dependencies
- [ ] Declared all extension dependencies
- [ ] Marked optional dependencies appropriately
- [ ] Tested with and without dependencies

### Configuration
- [ ] Set `canonicalize_tool_names` appropriately
- [ ] Set `tool_dependency_strategy` appropriately
- [ ] Set `strict_mode` appropriately
- [ ] Set `load_priority` if needed

### Testing
- [ ] Unit tests for metadata
- [ ] Unit tests for loading
- [ ] Unit tests for functionality
- [ ] Tests for error conditions
- [ ] Tests for backward compatibility

### Documentation
- [ ] Docstring with description
- [ ] Docstring with requirements
- [ ] Docstring with examples
- [ ] Documented required tools
- [ ] Documented configuration options

### Performance
- [ ] Used async operations for I/O
- [ ] Cached expensive operations
- [ ] No blocking operations in loading
- [ ] Performance tested

---

For migration instructions, see [Migration Guide](migration_guide.md).
For API reference, see [API Reference](api_reference.md).
For architecture overview, see [Architecture Refactoring](architecture_refactoring.md).
