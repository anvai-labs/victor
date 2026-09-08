# Framework Capabilities Guide

This guide covers the framework-level capabilities promoted from vertical-specific implementations to reusable components available across all verticals.

## Overview

Victor's framework capabilities provide common patterns and functionality that can be shared across verticals. These capabilities were previously duplicated in individual verticals and have been promoted to the framework level to:

- Reduce code duplication
- Ensure consistent behavior across verticals
- Make verticals easier to maintain and extend
- Provide a unified API for common operations

External vertical authors should consume these capabilities from contract-first verticals
that inherit from `victor_contracts.VerticalBase`. Core runtime shims remain supported for
compatibility, but new packages should not be authored against `victor.core.verticals`.

## Available Capabilities

### 1. Stage Builder Capability (`victor.framework.capabilities.stages`)

Provides a 7-stage generic workflow template that verticals can use and customize.

#### Features

- **Generic 7-Stage Workflow**: INITIAL → PLANNING → READING → ANALYSIS → EXECUTION → VERIFICATION → COMPLETION
- **Stage Validation**: Ensures required stages exist and transitions are valid
- **Stage-to-Tool Mappings**: Recommends tools for each stage
- **Prompt Hints**: Provides hints for each stage to include in system prompts

#### Usage

```python
from victor.framework.capabilities import StageBuilderCapability, StageBuilderPresets

# Use default stages
capability = StageBuilderCapability()
stages = capability.get_stages()

# Get stages for a specific vertical
coding_stages = StageBuilderPresets.coding()
stages = coding_stages.get_stages()

# Get prompt hints for all stages
hints = capability.get_prompt_hints()

# Get recommended tools for a stage
tools = capability.get_tools_for_stage("EXECUTION")

# Check if a transition is valid
is_valid = capability.is_valid_transition("PLANNING", "EXECUTION")
```

#### Presets

Pre-configured stage builders are available for common verticals:

- `StageBuilderPresets.coding()` - Optimized for coding vertical
- `StageBuilderPresets.devops()` - Optimized for DevOps vertical
- `StageBuilderPresets.research()` - Optimized for research vertical
- `StageBuilderPresets.data_analysis()` - Optimized for data analysis vertical

#### Integration in Verticals

```python
from victor_contracts import VerticalBase
from victor.framework.capabilities import StageBuilderPresets

class MyVertical(VerticalBase):
    def get_stages(self):
        capability = StageBuilderPresets.coding()
        return capability.get_stages()
```

---

### 2. Grounding Rules Capability (`victor.framework.capabilities.grounding_rules`)

Centralized grounding rules for common vertical constraints that ensure consistent, safe behavior.

#### Features

- **Base Rules**: Apply to all verticals (ground on tool output, acknowledge uncertainty)
- **File Safety**: Read before write, verify paths, preserve structure
- **Git Safety**: No force push to main, check branch status, review diffs
- **Test Requirements**: Verify after changes, run affected tests
- **Privacy**: No sensitive data exposure, redact secrets
- **Tool Usage**: Use appropriate tools, respect limits

#### Usage

```python
from victor.framework.capabilities import (
    GroundingRulesCapability,
    GroundingRulesPresets,
    RuleCategory,
)

# Get all grounding rules
capability = GroundingRulesCapability()
rules = capability.get_rules()

# Get rules for a specific category
file_rules = capability.get_rules(category=RuleCategory.FILE_SAFETY)

# Get rules for a specific vertical
coding_rules = capability.get_vertical_rules("coding")

# Get formatted text for prompts
rules_text = capability.get_rules_text(vertical="coding")

# Add custom rule
from victor.framework.capabilities import GroundingRule
custom_rule = GroundingRule(
    rule_id="custom_rule",
    category=RuleCategory.FILE_SAFETY,
    text="Always backup files before modifying",
    priority=90,
)
capability.add_rule(custom_rule)
```

#### Presets

Pre-configured grounding rules for common verticals:

- `GroundingRulesPresets.coding()` - Coding-specific rules
- `GroundingRulesPresets.devops()` - DevOps-specific rules
- `GroundingRulesPresets.research()` - Research-specific rules
- `GroundingRulesPresets.data_analysis()` - Data analysis-specific rules
- `GroundingRulesPresets.rag()` - RAG-specific rules

#### Integration in Verticals

```python
from victor_contracts import VerticalBase
from victor.framework.capabilities import GroundingRulesPresets

class MyVertical(VerticalBase):
    def get_system_prompt(self):
        capability = GroundingRulesPresets.coding()
        grounding_rules = capability.get_rules_text(vertical="coding")
        return f"{base_prompt}\n\n{grounding_rules}"
```

---

### 3. Validation Capability (`victor.framework.capabilities.validation`)

Pluggable validation system with common validators for tool pipelines and orchestrator.

#### Features

- **File Path Validation**: Validates paths, prevents traversal, checks allowed directories
- **Code Syntax Validation**: Validates Python, JavaScript, TypeScript, JSON, YAML syntax
- **Configuration Validation**: Validates required keys, value types, custom rules
- **Output Format Validation**: Validates length, required/forbidden patterns
- **Chain Validation**: Apply multiple validators in sequence

#### Usage

```python
from victor.framework.capabilities import ValidationCapabilityProvider

# Create validation provider
provider = ValidationCapabilityProvider()

# Validate file path
result = provider.validate("file_path", "/path/to/file.txt")
if not result.is_valid:
    print(f"Error: {result.error_message}")

# Validate code syntax
result = provider.validate("code_syntax_python", "print('hello')")

# Validate configuration
config = {"key": "value"}
result = provider.validate("configuration", config)

# Chain multiple validators
results = provider.validate_chain(
    ["file_path", "code_syntax_python"],
    value,
)

# Register custom validator
from victor.framework.capabilities import Validator, ValidationResult

class CustomValidator(Validator):
    @property
    def validator_id(self):
        return "custom"

    def validate(self, value, context=None):
        # Validation logic here
        return ValidationResult.success()

provider.register_validator(CustomValidator())
```

#### Built-in Validators

- `FilePathValidator` - File path validation
- `CodeSyntaxValidator` - Code syntax validation for multiple languages
- `ConfigurationValidator` - Configuration dictionary validation
- `OutputFormatValidator` - Output format validation

#### Integration Points

- **Tool Pipeline**: Validate tool arguments before execution
- **Orchestrator**: Validate configuration on startup
- **Verticals**: Register domain-specific validators

---

### 4. Safety Rules Capability (`victor.framework.capabilities.safety_rules`)

Reusable safety pattern definitions for dangerous operations across git, file, shell, docker, and network operations.

#### Features

- **Pattern Matching**: Regex-based pattern matching for dangerous commands
- **Action Types**: ALLOW, WARN, BLOCK, REQUIRE_CONFIRMATION
- **Severity Levels**: 1-10 severity rating for operations
- **Category-Based Rules**: Organized by operation type (git, file, shell, docker, network)

#### Usage

```python
from victor.framework.capabilities import SafetyRulesCapabilityProvider, SafetyRulesPresets

# Create safety rules provider
provider = SafetyRulesCapabilityProvider()

# Check if an operation is safe
is_safe, rules = provider.check_operation("git", ["push", "--force"])
if not is_safe:
    print("Operation blocked!")

# Get warnings for an operation
warnings = provider.get_warnings("git", ["reset", "--hard"])
for warning in warnings:
    print(f"Warning: {warning}")

# Get required confirmations
confirmations = provider.get_required_confirmations("shell", ["mkfs", "/dev/sda"])
if confirmations:
    print(f"Confirmation required: {confirmations[0]}")

# Add custom safety rule
from victor.framework.capabilities import SafetyRule, SafetyCategory, SafetyAction
custom_rule = SafetyRule(
    rule_id="custom_rule",
    category=SafetyCategory.SHELL,
    pattern=r"dangerous_command",
    description="Dangerous operation",
    action=SafetyAction.REQUIRE_CONFIRMATION,
    severity=9,
)
provider.add_rule(custom_rule)
```

#### Safety Categories

- **GIT**: Git operation safety (force push, branch deletion, etc.)
- **FILE**: File operation safety (recursive delete, overwrite, etc.)
- **SHELL**: Shell command safety (format, kill, system config, etc.)
- **DOCKER**: Docker operation safety (container removal, pruning, etc.)
- **NETWORK**: Network operation safety (port scanning, downloads, etc.)
- **SYSTEM**: System-level operations (kubectl, etc.)

#### Presets

- `SafetyRulesPresets.coding()` - Standard coding safety rules
- `SafetyRulesPresets.devops()` - DevOps safety rules (includes kubectl)
- `SafetyRulesPresets.research()` - Research safety rules
- `SafetyRulesPresets.local_mode()` - Less strict for local development
- `SafetyRulesPresets.production_mode()` - Very strict for production

---

### 5. Task Type Hints Capability (`victor.framework.capabilities.task_hints`)

Centralized task type hints for common task types with tool budget recommendations and priority tools.

#### Features

- **Task Type Detection**: Match tasks based on keywords
- **Tool Budget Recommendations**: Recommended tool call budgets per task type
- **Priority Tools**: Tools to prioritize for each task type
- **Prompt Hints**: Hint text for inclusion in system prompts

#### Usage

```python
from victor.framework.capabilities import TaskTypeHintCapabilityProvider, TaskTypeHintPresets

# Create task hints provider
provider = TaskTypeHintCapabilityProvider()

# Get hint for specific task type
hint = provider.get_hint("edit")
print(f"Hint: {hint.hint}")
print(f"Tool budget: {hint.tool_budget}")
print(f"Priority tools: {hint.priority_tools}")

# Get hint based on keywords
hint = provider.get_hint_for_keywords(["fix", "bug", "error"])

# Get tool budget for task
budget = provider.get_tool_budget("debug")

# Get priority tools for task
tools = provider.get_priority_tools("refactor")

# Get all prompt text formatted
prompt_text = provider.get_all_prompt_text()
```

#### Task Types

- `general` - General purpose tasks
- `search` - Searching for information
- `create` - Creating new files/content
- `edit` - Editing existing files
- `debug` - Finding and fixing bugs
- `refactor` - Improving code structure
- `test` - Writing and running tests
- `analyze` - Deep analysis
- `deploy` - Deployment operations
- `document` - Writing documentation

#### Presets

- `TaskTypeHintPresets.coding()` - Coding task hints (includes "review")
- `TaskTypeHintPresets.devops()` - DevOps task hints (includes "provision", "monitor")
- `TaskTypeHintPresets.research()` - Research task hints (includes "investigate", "synthesize")
- `TaskTypeHintPresets.data_analysis()` - Data analysis hints (includes "explore", "visualize")
- `TaskTypeHintPresets.rag()` - RAG task hints (includes "query", "answer")

---

### 6. Source Verification Capability (`victor.framework.capabilities.source_verification`)

Source verification for citations and references in research and RAG contexts.

#### Features

- **Citation Validation**: Validate citation formats
- **Reference Checking**: Check that references exist
- **Source Reliability Scoring**: Score sources by reliability
- **Web Search Integration**: Verify web search results

#### Usage

```python
from victor.framework.capabilities import SourceVerificationCapabilityProvider

# Create source verification provider
provider = SourceVerificationCapabilityProvider()

# Verify a citation
result = provider.verify_citation("https://example.com", "Title")
if not result.is_valid:
    print(f"Citation error: {result.error_message}")

# Check reference
result = provider.check_reference(source_id="abc123")

# Score source reliability
score = provider.score_reliability("https://trusted-source.com")

# Verify web search result
result = provider.verify_web_search_result({
    "url": "https://example.com",
    "title": "Example",
    "snippet": "Snippet"
})
```

---

## Performance Features

### HTTP Connection Pool (`victor.tools.http_pool`)

Connection pooling for HTTP-based tools to reduce request latency by 20-30%.

#### Usage

```python
from victor.tools.http_pool import get_http_pool, ConnectionPoolConfig

# Get connection pool
config = ConnectionPoolConfig(
    max_connections=100,
    max_per_host=10,
    timeout_seconds=30,
)
pool = await get_http_pool(config)

# Make pooled request
response = await pool.request("GET", "https://api.example.com/data")
```

#### Configuration

Enable via settings:

```python
VICTOR_HTTP_CONNECTION_POOL_ENABLED=true
VICTOR_HTTP_POOL_MAX_CONNECTIONS=100
VICTOR_HTTP_POOL_MAX_PER_HOST=10
```

### Preload Manager (`victor.framework.preload`)

Async preload manager for warm cache initialization to reduce first-request latency by 50-70%.

#### Usage

```python
from victor.framework.preload import PreloadManager, PreloadPriority

# Create preload manager
manager = PreloadManager(enable_parallel=True)

# Add preload tasks
manager.add_task("configuration", preload_configuration, priority=PreloadPriority.HIGH)
manager.add_task("tool_results", preload_tool_results, priority=PreloadPriority.MEDIUM)

# Execute all preloads
stats = await manager.preload_all()
print(f"Preloaded {stats.completed_tasks} tasks in {stats.total_duration_ms}ms")
```

#### Preload Tasks

Common preload tasks include:
- Configuration loading
- Tool results cache warming
- Embedding computation
- Model initialization

### Generic Result Cache (`victor.storage.cache.generic_result_cache`)

Extended caching for non-tool results with dependency tracking and invalidation.

#### Usage

```python
from victor.storage.cache import GenericResultCache

# Create cache
cache = GenericResultCache(max_size=1000, ttl_seconds=3600)

# Cache result
cache.put("key", {"result": "value"}, dependencies=["/path/to/file.py"])

# Get cached result
result = cache.get("key")

# Invalidate by dependency
cache.invalidate_by_dependency("/path/to/file.py")

# Get cache stats
stats = cache.get_stats()
```

---

## Observability

### Observability Manager (`victor.framework.observability`)

Unified observability manager aggregating metrics from all components.

#### Usage

```python
from victor.framework.observability import ObservabilityManager, ObservabilityConfig

# Create manager
config = ObservabilityConfig(
    max_history_size=100,
    collect_system_metrics=True,
)
manager = ObservabilityManager.get_instance(config=config)

# Register a metrics source
manager.register_source("my_capability", lambda: {
    "counter": 10,
    "gauge": 5.5,
})

# Collect metrics from all sources
metrics = manager.collect_metrics()

# Get dashboard data
dashboard_data = manager.get_dashboard_data()

# Get historical data
historical = manager.get_historical_data(limit=100)
```

#### Dashboard CLI

```bash
# Launch observability dashboard
victor observability dashboard

# Show metrics for specific source
victor observability metrics my_capability

# Show historical data
victor observability history --limit 50
```

---

## Best Practices

### 1. Use Presets When Available

Presets provide optimized configurations for common verticals:

```python
# Good: Use preset
capability = GroundingRulesPresets.coding()

# Avoid: Manually configure unless necessary
capability = GroundingRulesCapability(custom_rules=[...])
```

### 2. Extend Rather Than Replace

Add custom rules/hints rather than replacing defaults:

```python
# Good: Add custom rule to preset
capability = GroundingRulesPresets.coding()
capability.add_rule(custom_rule)

# Avoid: Replace entire capability
capability = GroundingRulesCapability(custom_rules=[...])  # Loses defaults
```

### 3. Register Custom Validators

Add domain-specific validators to the validation provider:

```python
provider = ValidationCapabilityProvider()
provider.register_validator(MyDomainValidator())
```

### 4. Use Safety Rules Consistently

Apply safety rules across all tools that handle dangerous operations:

```python
safety_provider = SafetyRulesCapabilityProvider()
is_safe, rules = safety_provider.check_operation(tool_name, args)
if not is_safe:
    # Handle blocked operation
```

### 5. Collect Metrics

Make your capabilities observable by implementing metrics collection:

```python
from victor.framework.observability import ObservabilityManager

manager = ObservabilityManager.get_instance()
manager.register_source("my_capability", self.get_metrics)
```

---

## Migration Guide

### For Existing Verticals

To migrate an existing vertical to use framework capabilities:

1. **Replace local stage definitions** with `StageBuilderPresets`
2. **Replace local grounding rules** with `GroundingRulesPresets`
3. **Replace local safety rules** with `SafetyRulesPresets`
4. **Replace local task hints** with `TaskTypeHintPresets`
5. **Add validation** using `ValidationCapabilityProvider`

Example migration:

```python
from victor_contracts import VerticalBase
from victor.framework.capabilities import StageBuilderPresets

# Before
class MyVertical(VerticalBase):
    def get_stages(self):
        return {
            "INITIAL": {...},
            "PLANNING": {...},
            # ... duplicated stage definitions
        }

# After
class MyVertical(VerticalBase):
    def get_stages(self):
        return StageBuilderPresets.coding().get_stages()
```

### For New Verticals

When creating a new vertical:

1. Start with the most relevant preset
2. Add only the customizations specific to your vertical
3. Use validation for any domain-specific constraints
4. Make your vertical observable with metrics

---

## Configuration

### Enable Performance Features

```yaml
# config/settings.yaml
performance:
  generic_result_cache_enabled: true
  http_connection_pool_enabled: true
  framework_preload_enabled: true
  framework_preload_parallel: true
```

### Observability Settings

```yaml
# config/settings.yaml
observability:
  enabled: true
  max_history_size: 100
  collect_system_metrics: true
  collection_interval_seconds: 60
```

---

## API Reference

See the [API Reference](FRAMEWORK_CAPABILITIES.md) for detailed API documentation.

---

## Troubleshooting

### Capability Not Found

If you get an import error:

```python
# Make sure capabilities are exported
from victor.framework.capabilities import (
    StageBuilderCapability,
    GroundingRulesCapability,
    # ...
)
```

### Custom Rules Not Applied

Make sure you're calling `add_rule()` on the capability instance, not the class:

```python
# Correct
capability = GroundingRulesCapability()
capability.add_rule(custom_rule)

# Incorrect
GroundingRulesCapability.add_rule(custom_rule)  # Class method doesn't exist
```

### Metrics Not Collected

Ensure you've registered your metrics source:

```python
manager = ObservabilityManager.get_instance()
manager.register_source("my_source", lambda: {...})
```

---

## Further Reading

- [Architecture Guide](../architecture.md)
- [Vertical Development Guide](./vertical-quickstart.md)
- [Observability Guide](./OBSERVABILITY.md)
- [Performance Optimization Guide](./BENCHMARKING.md)
