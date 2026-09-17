# Victor Verticals API Reference

**Version**: 1.0
**Date**: 2026-03-31

## Table of Contents

1. [Vertical Metadata](#vertical-metadata)
2. [Configuration Registry](#configuration-registry)
3. [Unified Entry Point Registry](#unified-entry-point-registry)
4. [Version Compatibility Matrix](#version-compatibility-matrix)
5. [Extension Dependency Graph](#extension-dependency-graph)
6. [Async-Safe Cache Manager](#async-safe-cache-manager)
7. [OpenTelemetry Integration](#opentelemetry-integration)
8. [Plugin Namespace Manager](#plugin-namespace-manager)
9. [Registration Decorator](#registration-decorator)

---

## Scope

This page mixes two kinds of APIs:

- **Public authoring contract**: external vertical packages should depend on
  `victor-contracts` and use `victor_contracts.VerticalBase`,
  `victor_contracts.register_vertical`, `VictorPlugin`, and
  `victor_contracts.validation.validate_vertical_package`.
- **Core runtime internals**: `victor.core.verticals.*` modules remain useful for
  framework contributors and advanced in-process integrations inside Victor, but
  they are not the preferred authoring surface for new external packages.

When in doubt, external package authors should start with the contract-first path and
use the core runtime APIs only when extending Victor itself.

---

## Vertical Metadata

### `VerticalMetadata`

Type-safe metadata extraction from vertical classes.

**Class**: `victor.core.verticals.vertical_metadata.VerticalMetadata`

**Attributes**:
- `name` (`str`): The vertical name
- `canonical_name` (`str`): Normalized lowercase name
- `display_name` (`str`): Human-readable name
- `version` (`str`): Vertical version
- `api_version` (`int`): Manifest API version
- `module_path` (`str`): Full Python module path
- `qualname` (`str`): Qualified class name
- `is_contrib` (`bool`): True if built-in contrib vertical
- `is_external` (`bool`): True if external vertical package

**Methods**:

##### `from_class(vertical_class: type) -> VerticalMetadata`

Extract metadata from a vertical class.

**Parameters**:
- `vertical_class` (`type`): The vertical class to extract metadata from

**Returns**: `VerticalMetadata` instance

**Example**:
```python
from victor.core.verticals.vertical_metadata import VerticalMetadata

metadata = VerticalMetadata.from_class(MyVertical)
print(metadata.canonical_name)  # "my_vertical"
print(metadata.version)  # "1.0.0"
print(metadata.is_external)  # True
```

---

### `VerticalNamingPattern`

Enum for supported naming patterns.

**Class**: `victor.core.verticals.vertical_metadata.VerticalNamingPattern`

**Values**:
- `ASSISTANT_SUFFIX`: "Assistant" suffix (e.g., `CodingAssistant`)
- `VERTICAL_SUFFIX`: "Vertical" suffix (e.g., `CodingVertical`)
- `EXPLICIT_NAME`: Uses `name` attribute directly

**Example**:
```python
from victor.core.verticals.vertical_metadata import VerticalNamingPattern

pattern = VerticalNamingPattern.ASSISTANT_SUFFIX
print(pattern.value)  # "Assistant"
```

---

## Configuration Registry

### `VerticalBehaviorConfig`

Configuration dataclass for vertical behavior.

**Class**: `victor.core.verticals.config_registry.VerticalBehaviorConfig`

**Attributes**:
- `canonicalize_tool_names` (`bool`): Auto-prefix tool names with vertical name
- `tool_dependency_strategy` (`str`): Strategy for tool dependencies ("auto", "explicit", "none")
- `strict_mode` (`bool`): Fail on missing tools/features
- `load_priority` (`int`): Load order priority (higher loads first)
- `allow_tool_override` (`bool`): Allow tools to override built-ins

**Example**:
```python
from victor.core.verticals.config_registry import VerticalBehaviorConfig

config = VerticalBehaviorConfig(
    canonicalize_tool_names=True,
    tool_dependency_strategy="auto",
    strict_mode=False,
    load_priority=50,
)
```

---

### `VerticalBehaviorConfigRegistry`

Singleton registry for vertical configurations.

**Class**: `victor.core.verticals.config_registry.VerticalBehaviorConfigRegistry`

**Methods**:

##### `get_instance() -> VerticalBehaviorConfigRegistry`

Get the singleton registry instance.

**Returns**: `VerticalBehaviorConfigRegistry` singleton

---

##### `register(vertical_name: str, config: VerticalBehaviorConfig) -> None`

Register configuration for a vertical.

**Parameters**:
- `vertical_name` (`str`): Name of the vertical
- `config` (`VerticalBehaviorConfig`): Configuration to register

**Example**:
```python
from victor.core.verticals.config_registry import (
    get_behavior_config_registry,
    VerticalBehaviorConfig,
)

registry = get_behavior_config_registry()
config = VerticalBehaviorConfig(canonicalize_tool_names=True)
registry.register("my_vertical", config)
```

---

##### `get(vertical_name: str) -> VerticalBehaviorConfig`

Get configuration for a vertical.

**Parameters**:
- `vertical_name` (`str`): Name of the vertical

**Returns**: `VerticalBehaviorConfig` or default if not found

**Example**:
```python
from victor.core.verticals.config_registry import get_behavior_config

config = get_behavior_config("my_vertical")
print(config.canonicalize_tool_names)  # True
```

---

## Unified Entry Point Registry

### `UnifiedEntryPointRegistry`

Singleton registry for entry point scanning with single-pass performance.

**Class**: `victor.framework.entry_point_registry.UnifiedEntryPointRegistry`

**Methods**:

##### `get_instance() -> UnifiedEntryPointRegistry`

Get the singleton registry instance.

**Returns**: `UnifiedEntryPointRegistry` singleton

---

##### `scan_all(force: bool = False) -> ScanMetrics`

Scan all entry point groups in a single pass.

**Parameters**:
- `force` (`bool`): If True, re-scan even if already scanned

**Returns**: `ScanMetrics` with timing and counts

**Example**:
```python
from victor.framework.entry_point_registry import get_entry_point_registry

registry = get_entry_point_registry()
metrics = registry.scan_all()

print(f"Scanned {metrics.total_entry_points} entry points in {metrics.scan_duration_ms:.2f}ms")
```

---

##### `get(group: str, name: str) -> Optional[Any]`

Get a specific entry point by group and name.

**Parameters**:
- `group` (`str`): Entry point group (e.g., `"victor.plugins"`)
- `name` (`str`): Entry point name (e.g., "coding")

**Returns**: Entry point object or None if not found

**Example**:
```python
from victor.framework.entry_point_registry import get_entry_point

coding_plugin = get_entry_point("victor.plugins", "coding")
if coding_plugin:
    print(coding_plugin.name)
```

---

##### `get_group(group: str) -> Optional[EntryPointGroup]`

Get all entry points in a group.

**Parameters**:
- `group` (`str`): Entry point group name

**Returns**: `EntryPointGroup` or None if not found

**Example**:
```python
from victor.framework.entry_point_registry import get_entry_point_group

group = get_entry_point_group("victor.plugins")
if group:
    for name, (ep, loaded) in group.entry_points.items():
        print(f"Found: {name}")
```

`victor.verticals` remains a legacy compatibility group for raw vertical class
entry points, but new packages should publish `VictorPlugin` objects through
`victor.plugins`.

---

### `ScanMetrics`

Metrics for entry point scanning.

**Class**: `victor.framework.entry_point_registry.ScanMetrics`

**Attributes**:
- `total_groups` (`int`): Number of groups discovered
- `total_entry_points` (`int`): Total entry points across all groups
- `scan_duration_ms` (`float`): Time taken to scan (milliseconds)
- `cache_hits` (`int`): Number of cache hits
- `cache_misses` (`int`): Number of cache misses

---

## Version Compatibility Matrix

### `VersionCompatibilityMatrix`

Version compatibility checker with PEP 440 support.

**Class**: `victor.core.verticals.version_matrix.VersionCompatibilityMatrix`

**Methods**:

##### `get_instance() -> VersionCompatibilityMatrix`

Get the singleton matrix instance.

**Returns**: `VersionCompatibilityMatrix` singleton

---

##### `check_vertical_compatibility(...) -> CompatibilityStatus`

Check if a vertical is compatible with the framework.

**Parameters**:
- `vertical_name` (`str`): Name of the vertical
- `vertical_version` (`str`): Version of the vertical
- `framework_version` (`str`): Framework version to check against

**Returns**: `CompatibilityStatus` enum value

**Example**:
```python
from victor.core.verticals.version_matrix import get_compatibility_matrix

matrix = get_compatibility_matrix()
status = matrix.check_vertical_compatibility(
    vertical_name="my_vertical",
    vertical_version="1.0.0",
    framework_version="0.6.0"
)

if status == CompatibilityStatus.COMPATIBLE:
    print("Compatible!")
elif status == CompatibilityStatus.INCOMPATIBLE:
    print("Incompatible!")
```

---

### `CompatibilityStatus`

Compatibility status enum.

**Class**: `victor.core.verticals.version_matrix.CompatibilityStatus`

**Values**:
- `COMPATIBLE`: Vertical can run on this framework version
- `INCOMPATIBLE`: Vertical cannot run on this framework version
- `DEGRADED`: Vertical can run but with reduced functionality
- `UNKNOWN`: Compatibility cannot be determined

---

## Extension Dependency Graph

### `ExtensionDependencyGraph`

Dependency graph for vertical extensions with topological sort.

**Class**: `victor.core.verticals.dependency_graph.ExtensionDependencyGraph`

**Methods**:

##### `add_vertical(vertical_name: str, version: str, manifest: Optional[ExtensionManifest] = None, load_priority: int = 0) -> None`

Add a vertical to the graph.

**Parameters**:
- `vertical_name` (`str`): Name of the vertical
- `version` (`str`): Version of the vertical
- `manifest` (`ExtensionManifest`, optional): Extension manifest
- `load_priority` (`int`): Load priority (higher loads first)

**Example**:
```python
from victor.core.verticals.dependency_graph import ExtensionDependencyGraph

graph = ExtensionDependencyGraph()
graph.add_vertical("my_vertical", version="1.0.0", load_priority=50)
```

---

##### `add_dependency(vertical_name: str, dependency_name: str, required: bool = True) -> None`

Add a dependency relationship.

**Parameters**:
- `vertical_name` (`str`): Name of the vertical that has the dependency
- `dependency_name` (`str`): Name of the vertical being depended on
- `required` (`bool`): Whether the dependency is required

**Example**:
```python
graph.add_dependency("my_vertical", "base_tools", required=True)
graph.add_dependency("my_vertical", "optional_feature", required=False)
```

---

##### `resolve_load_order() -> LoadOrder`

Resolve load order using topological sort.

**Returns**: `LoadOrder` object with ordered list and validation results

**Example**:
```python
load_order = graph.resolve_load_order()

if load_order.can_load:
    print("Load order:", load_order.order)
else:
    print("Missing dependencies:", load_order.missing_dependencies)
    print("Cycles:", load_order.cycles)
```

---

### `LoadOrder`

Result of dependency resolution.

**Class**: `victor.core.verticals.dependency_graph.LoadOrder`

**Attributes**:
- `order` (`List[str]`): List of vertical names in load order
- `missing_dependencies` (`Set[str]`): Required dependencies that are missing
- `missing_optional` (`Set[str]`): Optional dependencies that are missing
- `cycles` (`List[List[str]]`): List of detected cycles

**Properties**:
- `can_load` (`bool`): True if all required dependencies satisfied and no cycles

---

## Async-Safe Cache Manager

### `AsyncSafeCacheManager`

Thread-safe cache with lock-per-key for parallel loading.

**Class**: `victor.core.verticals.async_cache_manager.AsyncSafeCacheManager`

**Methods**:

##### `get_or_create(namespace: str, key: str, factory: Callable[[], Any]) -> Any`

Get cached value or create via factory (thread-safe).

**Parameters**:
- `namespace` (`str`): Cache namespace (typically vertical class path)
- `key` (`str`): Extension key (e.g., "middleware")
- `factory` (`Callable[[], Any]`): Zero-argument callable that creates the value

**Returns**: The cached or newly created value

**Example**:
```python
from victor.core.verticals.async_cache_manager import AsyncSafeCacheManager

cache = AsyncSafeCacheManager()

def create_middleware():
    return HeavyMiddleware()

middleware = cache.get_or_create("MyVertical:mod:qual", "middleware", create_middleware)
```

---

##### `async get_or_create_async(namespace: str, key: str, factory: Union[Awaitable, Callable]) -> Any`

Async version of get_or_create.

**Parameters**:
- `namespace` (`str`): Cache namespace
- `key` (`str`): Extension key
- `factory` (`Awaitable` or `Callable`): Async or sync factory

**Returns**: The cached or newly created value

**Example**:
```python
async def create_middleware_async():
    return await AsyncHeavyMiddleware.create()

middleware = await cache.get_or_create_async(
    "MyVertical:mod:qual",
    "middleware",
    create_middleware_async
)
```

---

##### `invalidate(namespace: Optional[str] = None, key: Optional[str] = None) -> int`

Invalidate cache entries.

**Parameters**:
- `namespace` (`str`, optional): If specified, only invalidate entries in this namespace
- `key` (`str`, optional): If specified with namespace, only invalidate this specific key

**Returns**: Number of entries invalidated

**Example**:
```python
# Invalidate all
count = cache.invalidate()

# Invalidate all for a namespace
count = cache.invalidate(namespace="MyVertical:mod:qual")

# Invalidate specific key
count = cache.invalidate(namespace="MyVertical:mod:qual", key="middleware")
```

---

##### `get_stats() -> Dict[str, int]`

Get cache statistics.

**Returns**: Dict with keys: cache_size, lock_count, hit_count, miss_count, hit_rate

**Example**:
```python
stats = cache.get_stats()
print(f"Cache size: {stats['cache_size']}")
print(f"Hit rate: {stats['hit_rate']:.2%}")
```

---

## OpenTelemetry Integration

### `vertical_load_span()`

Context manager for instrumenting vertical loading operations.

**Function**: `victor.core.verticals.telemetry.vertical_load_span`

**Parameters**:
- `vertical_name` (`str`): Name of the vertical being loaded
- `operation` (`str`): Operation type (e.g., "load", "activate", "discover")
- `emit_callback` (`Callable`, optional): Callback to emit span to telemetry backend

**Yields**: `VerticalLoadSpan` for recording operation details

**Example**:
```python
from victor.core.verticals.telemetry import vertical_load_span

with vertical_load_span("coding", "load") as span:
    try:
        vertical = load_vertical("coding")
        span.status = "success"
    except Exception as e:
        span.status = "error"
        span.error = str(e)
```

---

### `VerticalLoadSpan`

Context for a vertical loading operation.

**Class**: `victor.core.verticals.telemetry.VerticalLoadSpan`

**Attributes**:
- `vertical_name` (`str`): Name of the vertical
- `operation` (`str`): Operation type
- `start_time_ns` (`int`): Start time in nanoseconds
- `end_time_ns` (`int`, optional): End time in nanoseconds
- `status` (`str`): Status of operation (success, error, skipped)
- `error` (`str`, optional): Error message if status is error
- `metadata` (`Dict[str, Any]`): Additional structured metadata

**Properties**:
- `duration_ms` (`float`, optional): Duration in milliseconds
- `is_success` (`bool`): True if operation succeeded
- `is_error` (`bool`): True if operation failed

---

### `VerticalLoadTelemetry`

Telemetry recorder for vertical loading operations.

**Class**: `victor.core.verticals.telemetry.VerticalLoadTelemetry`

**Methods**:

##### `track_load(vertical_name: str, operation: str = "load")`

Context manager for tracking a vertical load operation.

**Parameters**:
- `vertical_name` (`str`): Name of the vertical
- `operation` (`str`): Operation type

**Yields**: `VerticalLoadSpan` for recording details

**Example**:
```python
from victor.core.verticals.telemetry import get_telemetry

telemetry = get_telemetry()

with telemetry.track_load("coding", "discover"):
    vertical = discover_vertical("coding")
```

---

##### `get_metrics() -> Dict[str, Any]`

Get aggregated metrics from all recorded operations.

**Returns**: Dict with metrics for operations, errors, slow operations

**Example**:
```python
metrics = telemetry.get_metrics()
print(f"Total operations: {metrics['total_operations']}")
print(f"Success rate: {metrics['success_rate']:.2%}")
print(f"Avg duration: {metrics['avg_duration_ms']:.2f}ms")
```

---

##### `get_slow_operations(threshold_ms: float = 500.0) -> List[Dict[str, Any]]`

Get list of slow operations.

**Parameters**:
- `threshold_ms` (`float`): Duration threshold for "slow" (default: 500ms)

**Returns**: List of slow operation details

**Example**:
```python
slow_ops = telemetry.get_slow_operations(threshold_ms=1000.0)

for op in slow_ops:
    print(f"{op['vertical']}.{op['operation']} took {op['duration_ms']:.2f}ms")
```

---

## Plugin Namespace Manager

### `PluginNamespaceManager`

Manager for plugin namespace isolation.

**Class**: `victor.core.verticals.namespace_manager.PluginNamespaceManager`

**Methods**:

##### `get_instance() -> PluginNamespaceManager`

Get the singleton namespace manager instance.

**Returns**: `PluginNamespaceManager` singleton

---

##### `make_key(namespace: str, plugin_name: str, version: Optional[str] = None) -> NamespacedPluginKey`

Create a namespaced plugin key.

**Parameters**:
- `namespace` (`str`): Namespace name (e.g., "external", "contrib")
- `plugin_name` (`str`): Name of the plugin
- `version` (`str`, optional): Optional version string

**Returns**: `NamespacedPluginKey` object

**Example**:
```python
from victor.core.verticals.namespace_manager import get_namespace_manager

manager = get_namespace_manager()
key = manager.make_key("external", "my_tool", "1.0.0")
print(key.full_key)  # "external:my_tool:1.0.0"
```

---

##### `register_plugin(namespace: str, plugin_name: str, plugin: Any, version: Optional[str] = None) -> None`

Register a plugin in a namespace.

**Parameters**:
- `namespace` (`str`): Namespace name
- `plugin_name` (`str`): Name of the plugin
- `plugin` (`Any`): The plugin object/function
- `version` (`str`, optional): Optional version string

**Example**:
```python
manager.register_plugin("external", "my_tool", my_tool_object, "1.0.0")
```

---

##### `resolve(plugin_name: str, available_namespaces: List[str], default_namespace: str = "default") -> Optional[Any]`

Resolve a plugin by name considering namespace priorities.

**Parameters**:
- `plugin_name` (`str`): Name of the plugin to resolve
- `available_namespaces` (`List[str]`): List of namespaces to search
- `default_namespace` (`str`): Fallback namespace if not in available_namespaces

**Returns**: The resolved plugin object, or None if not found

**Example**:
```python
# External (priority 100) > Contrib (50) > Default (0)
plugin = manager.resolve("my_tool", ["external", "contrib", "default"])
```

---

### `NamespacedPluginKey`

A namespaced plugin key that prevents collisions.

**Class**: `victor.core.verticals.namespace_manager.NamespacedPluginKey`

**Attributes**:
- `namespace` (`str`): Namespace name
- `plugin_name` (`str`): Name of the plugin
- `version` (`str`, optional): Optional version string
- `full_key` (`str`): Full composite key (namespace:name:version)

**Example**:
```python
key = NamespacedPluginKey("external", "my_tool", "1.0.0")
print(key.full_key)  # "external:my_tool:1.0.0"
```

---

### `NamespaceType`

Predefined namespace types with priority ordering.

**Class**: `victor.core.verticals.namespace_manager.NamespaceType`

**Values**:
- `DEFAULT` (priority 0): Fallback namespace
- `CONTRIBUTED` (priority 50): Built-in contrib verticals (deprecated)
- `EXTERNAL` (priority 100): Third-party vertical packages
- `EXPERIMENTAL` (priority 25): Experimental features

---

## Registration Decorator

### `@register_vertical`

Decorator for declarative vertical registration.

**Function**: `victor_contracts.verticals.registration.register_vertical`

**Parameters**:
- `name` (`str`): Vertical name/identifier
- `version` (`str`): Vertical version (PEP 440)
- `api_version` (`int`, optional): Manifest API version
- `min_framework_version` (`str`, optional): Minimum compatible framework version
- `requires` (`Set[ExtensionType]`, optional): Required extension types
- `provides` (`Set[ExtensionType]`, optional): Provided extension types
- `extension_dependencies` (`List[ExtensionDependency]`, optional): Required extension dependencies
- `requires_features` (`Set[str]`, optional): Required framework features
- `excludes_features` (`Set[str]`, optional): Incompatible framework features
- `canonicalize_tool_names` (`bool`, optional): Auto-prefix tool names
- `tool_dependency_strategy` (`str`, optional): Tool dependency strategy
- `strict_mode` (`bool`, optional): Fail on missing tools/features
- `load_priority` (`int`, optional): Load order priority
- `plugin_namespace` (`str`, optional): Plugin namespace
- `lazy_load` (`bool`, optional): Allow lazy runtime loading

**Example**:
```python
from victor_contracts import ToolRequirement, VerticalBase, register_vertical

@register_vertical(
    name="my_vertical",
    version="1.0.0",
    min_framework_version=">=0.6.0",
    plugin_namespace="my_company",
    canonicalize_tool_names=True,
)
class MyVertical(VerticalBase):
    """My vertical implementation."""

    @classmethod
    def get_tool_requirements(cls) -> list[ToolRequirement]:
        return [ToolRequirement("read"), ToolRequirement("write", required=False)]

    @classmethod
    def get_tools(cls) -> list[str]:
        return [requirement.tool_name for requirement in cls.get_tool_requirements()]
```

**Validation**:

```python
from victor_contracts.validation import validate_vertical_package

report = validate_vertical_package("my-vertical")
assert report.is_valid
```

---

## Type Aliases

### Common Type Aliases

```python
from typing import Any, Callable, Dict, List, Optional, Set, Union

# Factory functions
Factory = Callable[[], Any]
AsyncFactory = Union[Awaitable[Any], Callable[[], Any]]

# Plugin objects
Plugin = Any

# Version strings
Version = str

# Namespace names
Namespace = str

# Vertical names
VerticalName = str
```

---

## Exceptions

### Custom Exceptions

```python
# Dependency graph
from victor.core.verticals.dependency_graph import DependencyCycleError

# Version compatibility
from victor.core.verticals.version_matrix import VersionConflictError

# Capability negotiation
from victor.core.verticals.capability_negotiator import CapabilityMismatchError
```

---

## Helper Functions

### Convenience Functions

```python
# Metadata
from victor.core.verticals.vertical_metadata import VerticalMetadata
metadata = VerticalMetadata.from_class(MyVertical)

# Configuration
from victor.core.verticals.config_registry import get_behavior_config
config = get_behavior_config("my_vertical")

# Entry points
from victor.framework.entry_point_registry import (
    get_entry_point,
    get_entry_point_group,
    scan_all_entry_points,
)

# Namespace
from victor.core.verticals.namespace_manager import (
    get_namespace_manager,
    make_plugin_key,
)

# Telemetry
from victor.core.verticals.telemetry import get_telemetry
telemetry = get_telemetry()
```

---

For usage examples, see [Migration Guide](../architecture/EXTERNAL_VERTICALS_MIGRATION.md).
For best practices, see [Best Practices](best_practices.md).
For architecture overview, see [Architecture Refactoring](architecture_refactoring.md).
