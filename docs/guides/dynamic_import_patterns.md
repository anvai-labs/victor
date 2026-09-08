# Dynamic Import Patterns in Victor

This guide explains dynamic import patterns used in Victor verticals and how to properly analyze code for dependencies.

## Overview

Traditional static analysis (grep, AST) often misses Python's dynamic import patterns. Victor verticals use several dynamic loading mechanisms that must be accounted for when analyzing "dead code" or module dependencies.

## Common Dynamic Import Patterns

### 1. __init__.py Export Mappings

**Pattern**: Modules are re-exported through lazy import mappings.

```python
# victor_coding/__init__.py
_EXPORTS = {
    "CodingSafetyRules": ("victor_coding.safety_enhanced", "CodingSafetyRules"),
    "EnhancedCodingConversationManager": (
        "victor_coding.conversation_enhanced",
        "EnhancedCodingConversationManager",
    ),
}

def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target:
        module = import_module(module_name)
        return getattr(module, attribute_name)
```

**Detection**: Check for:
- `_EXPORTS` dictionary
- `__getattr__` implementations
- Lazy import patterns

**Tools**: 
```bash
grep -n "_EXPORTS\|__getattr__" victor_xxx/__init__.py
```

### 2. String-Based importlib.import_module()

**Pattern**: Module paths are strings loaded at runtime.

```python
# victor_coding/workflows/provider.py
class WorkflowProvider:
    def _get_escape_hatches_module(self) -> str:
        return "victor_coding.escape_hatches"
    
    def load_workflow(self, name: str):
        module_path = self._get_escape_hatches_module()
        escape_hatches = importlib.import_module(module_path)
```

**Detection**: Check for:
- `importlib.import_module()`
- `__import__()`
- Methods returning module path strings

**Tools**:
```bash
grep -rn "import_module\|__import__" victor_xxx/
```

### 3. Hook Methods

**Pattern**: Methods that return module/function paths for dynamic registration.

```python
# Common hook method names
_get_X_module()
_get_Y_path()
module_path()
```

**Examples**:
- `_get_escape_hatches_module()`
- `_get_capability_provider_module()`
- `_get_extension_provider_module()`

**Detection**: Check for methods with these naming patterns in:
- Provider classes
- Registry classes  
- Factory classes

### 4. Plugin Registration

**Pattern**: Classes/functions register themselves via decorators or explicit calls.

```python
@register_plugin("my-plugin")
class MyPlugin:
    pass
```

**Detection**: Check for:
- Decorator-based registration (`@register`, `@plugin`)
- Explicit registration calls
- Plugin manifests

### 5. YAML Workflow Conditions

**Pattern**: Functions referenced by string in YAML config files.

```yaml
# workflow.yaml
- id: check_quality
  type: condition
  condition: "retrieval_quality"  # References function in escape_hatches.py
```

```python
# escape_hatches.py
def retrieval_quality(ctx: Dict[str, Any]) -> str:
    return "sufficient"
```

**Detection**: Check for:
- YAML files with condition references
- Corresponding Python functions
- Workflow provider classes

## Analyzing Module Dependencies

### Static Analysis

For direct imports, static analysis works:
```bash
# Find direct imports
grep -rn "from victor_xxx import" victor_xxx/
grep -rn "import victor_xxx" victor_xxx/
```

### Dynamic Analysis

For dynamic imports, use the `DynamicImportTracker`:

```python
from victor.tools.graph_dynamic_import_tracker import DynamicImportTracker

tracker = DynamicImportTracker(root_path="/path/to/project")

# Scan all dynamic imports
imports = tracker.scan_all()

# Check if a module is used
analysis = tracker.augment_graph_analysis(
    static_callers=set(),  # From static analysis
    symbol_name="escape_hatches"
)

# Returns: {
#   "static_callers": [...],
#   "dynamic_importers": ["workflows/provider.py"],
#   "exported_from": [...],
#   "total_references": 1
# }
```

## Dead Code Detection Checklist

Before marking code as "dead", verify:

- [ ] Not in `__init__.py` `_EXPORTS` mapping
- [ ] Not loaded via `importlib.import_module()`
- [ ] Not a hook method (returns module path string)
- [ ] Not referenced in YAML/config files
- [ ] Not decorated with `@register` or similar
- [ ] No test imports/usage
- [ ] Not in public API (`__all__`)

## Examples

### Example 1: escape_hatches.py

**Static analysis shows**: 0 direct imports

**Dynamic analysis shows**:
```python
# workflows/provider.py
def _get_escape_hatches_module(self) -> str:
    return "victor_rag.escape_hatches"

# Later...
escape_hatches = importlib.import_module(self._get_escape_hatches_module())
```

**Conclusion**: NOT dead - dynamically loaded

### Example 2: safety_enhanced.py

**Static analysis shows**: 0 direct imports in victor_coding/

**Dynamic analysis shows**:
```python
# __init__.py
_EXPORTS = {
    "CodingSafetyRules": ("victor_coding.safety_enhanced", "CodingSafetyRules"),
}
```

**Conclusion**: NOT dead - exported as public API

## Tools

### DynamicImportTracker

Located in: `victor/tools/graph_dynamic_import_tracker.py`

```python
from victor.tools.graph_dynamic_import_tracker import DynamicImportTracker

tracker = DynamicImportTracker(root_path=".")

# Get all dynamic imports
imports = tracker.scan_all()

# Check reverse dependencies
importers = tracker.get_reverse_dynamic_dependencies("my_module")

# Check if file is a dynamic entrypoint
is_entrypoint, reason = tracker.is_dynamic_entrypoint("my_file.py")

# Augment graph analysis
analysis = tracker.augment_graph_analysis(
    static_callers={"main.py"},
    symbol_name="my_function"
)
```

## Best Practices

### For Code Authors

1. **Document dynamic imports**: Add comments explaining dynamic loading
2. **Use consistent patterns**: Follow Victor's conventions for dynamic imports
3. **Add type hints**: Help tools understand return types
4. **Register exports**: Always add to `__init__.py` for public APIs

### For Code Reviewers

1. **Check dynamic patterns**: Use this guide to verify "dead code"
2. **Ask for context**: When unsure, ask about intended usage
3. **Run tests**: Check if tests use the module
4. **Check __init__.py**: Look for export mappings

## Related Documentation

- [Graph Tool Modes](https://github.com/anvai-labs/victor/blob/ff9129a8de9eb8788a260c5fa20b27ef323ca7bc/victor/tools/graph_tool.py)
- [Codebase Indexing](https://github.com/anvai-labs/victor/blob/develop/victor/contrib/codebase/indexer.py)
- [Vertical Development Guide](https://github.com/anvai-labs/victor/blob/develop/victor-contracts/VERTICAL_DEVELOPMENT.md)
