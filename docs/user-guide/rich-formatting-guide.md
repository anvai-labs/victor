# Rich Formatting Guide

## Overview

Victor now includes **Rich formatting** for tool outputs, providing color-coded, visually-structured output that makes it easier to scan and understand results from complex tool operations.

### What is Rich Formatting?

Rich formatting uses console markup (colors, bold, dim) to highlight important information in tool outputs:

- **Green** ✓ for success states (passed tests, successful operations)
- **Red** ✗ for failures and errors (failed tests, command failures)
- **Yellow** ⚠ for warnings and skipped items
- **Cyan** for headers and important information
- **Bold** for emphasis and key values
- **Dim** for subtle information and metadata

### Supported Tools

Rich formatting is available for **12 tool categories** covering **35+ tool names**:

#### Core Development Tools
- **Testing**: `test`, `pytest`, `run_tests` - Test results with pass/fail/skip counts
- **Code Search**: `code_search`, `semantic_code_search` - Search results with file paths and scores
- **Git**: `git` - Status, log, diff output with color-coded changes

#### Operations Tools
- **HTTP**: `http`, `https` - Response status, headers, and body formatting
- **Database**: `database`, `db`, `sql` - Query results in table format
- **Shell**: `shell`, `bash`, `exec` - Command execution with success/failure indicators
- **Filesystem**: `filesystem`, `ls`, `find`, `cat`, `read`, `overview` - File operations with type-specific colors
- **Network**: `network`, `ping`, `traceroute`, `dns` - Network diagnostics with latency color-coding
- **Build**: `build`, `make`, `cmake`, `cargo`, `npm`, `pip` - Build status with warnings and errors

#### Advanced Tools
- **Refactoring**: `refactor`, `refactoring` - Operation summaries with icons
- **Docker**: `docker` - Container states, images, volumes, services
- **Security**: `security`, `security_scan` - Severity levels with color coding

## Getting Started

### Default Behavior

Rich formatting is **enabled by default** for all supported tools. No configuration is required - just use Victor as normal and you'll see enhanced output automatically.

### Example Output

#### Before (Plain Text)
```
10 tests, 8 passed, 2 failed
Failed Tests:
- test_api_create_user
- test_auth_login_timeout
```

#### After (Rich Formatted)
```
✓ 8 passed ✗ 2 failed ○ 0 skipped • 10 total

Failed Tests:

  ✗ test_api_create_user
    tests/api/test_users.py
    AssertionError: Expected 201 but got 500

  ✗ test_auth_login_timeout
    tests/auth/test_login.py
    TimeoutError: Request timeout after 30s
```

## Configuration

### Enable/Disable Rich Formatting

#### Environment Variable
```bash
# Disable Rich formatting
export VICTOR_USE_RICH_FORMATTING=false

# Enable Rich formatting (default)
export VICTOR_USE_RICH_FORMATTING=true
```

#### Settings File
```yaml
# config/settings.yml
tool_settings:
  rich_formatting_enabled: true  # Master switch
  rich_formatting_tools:          # Tool whitelist
    - test
    - code_search
    - git
    # ... (35 tools total)
```

### Performance Tuning

```yaml
tool_settings:
  # Rich formatting limits
  rich_formatting_max_output_size: 1000000  # 1MB max
  rich_formatting_max_time_ms: 200          # 200ms timeout

  # Cache configuration
  rich_formatting_cache_enabled: true
  rich_formatting_cache_ttl: 300            # 5 minutes
  rich_formatting_cache_size: 100           # Max entries
```

### Customization Options

#### Disable Formatting for Specific Tools
```yaml
tool_settings:
  rich_formatting_tools:
    - test
    - code_search
    # Remove tools you don't want formatted
```

#### Adjust Output Limits
```yaml
tool_settings:
  rich_formatting_max_output_size: 2000000  # 2MB limit
  rich_formatting_max_time_ms: 500          # 500ms timeout
```

## Features

### Progressive Tool Output Streaming

Long-running tools (`shell`, `code_search`) can stream their stdout/stderr into a
live, updating terminal block **while they run**, instead of the UI waiting for
the final result. This makes long operations feel responsive rather than stalled.

**How it works (architecture):**

```mermaid
---
title: Tool progress reaches the active terminal renderer
---
%%{init: {"theme":"base","themeVariables":{"primaryColor":"#E8EFF7","primaryTextColor":"#17324D","primaryBorderColor":"#456987","lineColor":"#456987","fontFamily":"Arial"}}}%%
flowchart TB
  T["Running tool"]
  P["emit_tool_progress<br/>framework/tool_progress.py"]
  S["Process-global progress sink<br/>single active interactive turn"]
  R["renderer.on_tool_progress"]
  D["LiveDisplayRenderer<br/>throttled Rich panel"]
  T -->|"emit stdout / stderr progress"| P
  P -->|"forward if sink is registered"| S
  S -->|"deliver progress callback"| R
  R -->|"refresh live panel"| D
```

Key properties:

- **UI-ephemeral.** Progress is for human visibility only. It is **never** added
  to the conversation, persisted, or sent to the model, so it is fully decoupled
  from the LLM message stream and from `OutputPruner`.
- **Best-effort & safe.** When no sink is registered (API/server path, headless,
  tests) `emit_tool_progress` is a no-op; a failing renderer never breaks a tool.
- **The `shell` tool** streams via incremental subprocess reads only when a sink
  is active; otherwise it uses the cheaper single `communicate()` call, and
  cached read-only commands are unaffected.
- Non-live renderers (`BufferedRenderer`, `FormatterRenderer`) implement
  `on_tool_progress` as a no-op.

**Configuration** — `tool_progress_streaming_enabled` (default: `true`) under
tool settings. Set to `false` to disable the live block (tools still produce
their normal final output).

The corresponding event type is `EventType.TOOL_PROGRESS`
(`victor/framework/events.py`), carrying `metadata["tool_progress"]`.

### Color-Coded Status Indicators

- **✓ Green** - Success (tests passed, commands succeeded, hosts reachable)
- **✗ Red** - Failure (tests failed, commands failed, hosts down)
- **○ Yellow** - Skipped or warning
- **● Blue** - Information
- **⚠ Yellow** - Warning
- **‼ Red** - Critical

### Structured Formatting

#### Test Results
- Summary line with counts
- Grouped failures with file paths
- Truncated error messages (full errors available in logs)

#### Code Search
- Grouped by file with average scores
- Match count per file
- Semantic vs exact mode indication

#### Git Operations
- Color-coded file changes (modified: yellow, added: green, deleted: red)
- Branch highlighting (current branch: green)
- Diff formatting with line numbers

#### Shell Commands
- Success/failure indicators
- Execution time
- stderr highlighting
- Command display

### Performance Optimization

#### Intelligent Caching
- **Content-based caching**: SHA256 hash keys
- **5-minute TTL**: Balances freshness and performance
- **43x speedup**: Cache hits are dramatically faster
- **100-entry limit**: Bounded memory usage

#### Production Guards
- **Size limits**: 1MB max formatted output
- **Time limits**: 200ms max formatting time
- **Fallback**: Plain text if formatting fails

## Troubleshooting

### Rich Formatting Not Working

#### Check Feature Flag
```python
from victor.core.feature_flags import FeatureFlag
print(FeatureFlag.USE_RICH_FORMATTING)  # Should exist
```

#### Check Settings
```python
from victor.config.settings import get_settings
settings = get_settings()
print(settings.tool_settings.rich_formatting_enabled)  # Should be True
```

#### Check Tool Whitelist
```python
print(settings.tool_settings.rich_formatting_tools)
# Should contain 35 tools including your tool
```

### Performance Issues

#### Clear Cache
```python
from victor.tools.formatters.registry import clear_format_cache
clear_format_cache()
```

#### Check Cache Stats
```python
from victor.tools.formatters.registry import get_format_cache_stats
stats = get_format_cache_stats()
print(f"Hit rate: {stats['hit_rate']:.1%}")
print(f"Cache size: {stats['size']}/{stats['max_size']}")
```

### Disable for Testing
```bash
# Temporarily disable
export VICTOR_USE_RICH_FORMATTING=false
victor chat

# Re-enable
export VICTOR_USE_RICH_FORMATTING=true
```

## Advanced Usage

### Custom Formatters

Create custom formatters by extending the base class:

```python
from victor.tools.formatters.base import ToolFormatter, FormattedOutput

class MyCustomFormatter(ToolFormatter):
    def validate_input(self, data: dict) -> bool:
        return 'my_field' in data
    
    def format(self, data: dict, **kwargs) -> FormattedOutput:
        content = f"[bold]Custom Output:[/]\n{data['my_field']}"
        return FormattedOutput(
            content=content,
            format_type="rich",
            summary="Custom output",
            contains_markup=True
        )
```

### Cache Management

#### Monitor Cache Performance
```python
from victor.tools.formatters.registry import get_format_cache_stats

stats = get_format_cache_stats()
print(f"Hits: {stats['hits']}")
print(f"Misses: {stats['misses']}")
print(f"Hit rate: {stats['hit_rate']:.1%}")
print(f"Size: {stats['size']}/{stats['max_size']}")
```

#### Invalidate Specific Tool Cache
```python
from victor.tools.formatters.registry import invalidate_format_cache
invalidate_format_cache("test")  # Clear test formatter cache
```

#### Clear All Cache
```python
from victor.tools.formatters.registry import clear_format_cache
clear_format_cache()
```

## Migration Guide

### From Plain Text

No migration required! Rich formatting is backward compatible:

1. **Existing tools work unchanged** - Rich formatting is additive
2. **Raw data preserved** - Original output always available
3. **Feature flag control** - Disable instantly if needed

### For Tool Developers

#### Before (Custom Formatting)
```python
def _format_test_output(data):
    lines = []
    if data['passed']:
        lines.append(f"{data['passed']} tests passed")
    return '\n'.join(lines)
```

#### After (Using Rich Formatter)
```python
from victor.tools.formatters import format_test_results

result = format_test_results(data)
print(result.content)  # Rich formatted output
```

## Best Practices

### 1. Use Color Semantically
- **Green**: Success, completion, safe states
- **Red**: Errors, failures, dangerous states
- **Yellow**: Warnings, skipped, caution
- **Cyan**: Headers, metadata, information
- **Dim**: Subtle details, timestamps, metadata

### 2. Structure for Scanning
- Summary line first (key metrics)
- Group related information
- Use consistent formatting patterns

### 3. Handle Large Outputs
- Truncate with "more..." indicators
- Show most important information first
- Respect user's max_lines preference

### 4. Provide Fallbacks
- Always validate input before formatting
- Return plain text if formatting fails
- Log errors for debugging

## Performance Tips

### 1. Leverage Cache
- Same input → cached formatted output
- 43x faster on cache hits
- 5-minute TTL balances freshness/performance

### 2. Use Production Guards
- Size limits prevent memory issues
- Time limits prevent hanging
- Fallback ensures reliability

### 3. Monitor Metrics
- Track cache hit rate (target: >70%)
- Monitor formatting overhead (target: <200ms)
- Watch error rates (target: <1%)

## Support

### Issues and Feedback
- **GitHub Issues**: Report bugs and request features
- **Documentation**: Check latest updates in `/docs/user-guide/`
- **Examples**: See `/tests/unit/tools/formatters/` for usage examples

### Rollback
If you encounter issues, disable Rich formatting instantly:

```bash
export VICTOR_USE_RICH_FORMATTING=false
```

No code changes required - instant fallback to plain text.

## Summary

Rich formatting provides **improved readability** and **faster information scanning** for Victor tool outputs with:

- ✅ **Zero configuration** required (works out of the box)
- ✅ **Backward compatible** (no breaking changes)
- ✅ **Performance optimized** (43x cache speedup)
- ✅ **Production ready** (guards and fallbacks)
- ✅ **Instant rollback** (feature flag control)

Enjoy enhanced tool output with Rich formatting! 🎨