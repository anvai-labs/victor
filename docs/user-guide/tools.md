# Tools Guide

Victor provides 34 tool modules that enable the AI assistant to interact with your codebase, execute commands, search the web, and perform complex operations. This guide covers how tools work, how they are selected, and how to configure them.

## Overview

### What Are Tools?

Tools are functions that extend Victor's capabilities beyond conversation. When you ask Victor to "read the file main.py" or "search for authentication code", it uses tools to perform these actions.

Each tool has:
- **Name**: Unique identifier (e.g., `read`, `shell`, `git`)
- **Description**: What the tool does
- **Parameters**: JSON Schema defining required/optional inputs
- **Cost Tier**: Resource cost classification (FREE, LOW, MEDIUM, HIGH)
- **Access Mode**: Permission level (READONLY, WRITE, EXECUTE, NETWORK, MIXED)

### Rich Formatted Output

Victor includes a **Rich formatter system** that provides beautiful, color-coded output for many tools. This makes it easier to quickly scan and understand tool results.

**Tools with Rich Formatting**:
- ✅ **Testing** (`test`, `pytest`, `run_tests`) — Color-coded test results (✓ passed, ✗ failed, ○ skipped)
- ✅ **Code Search** (`code_search`, `semantic_code_search`) — Grouped file results with relevance scores
- ✅ **Git** (`git`) — Color-coded status (green=added, yellow=modified, red=deleted)
- ✅ **HTTP** (`http`, `https`) — Status code colors (2xx green, 3xx yellow, 4xx/5xx red)
- ✅ **Database** (`database`, `db`, `sql`) — Rich table formatting with row counts
- ✅ **Docker** (`docker`) — Container states (● running, ○ stopped, ◐ paused)
- ✅ **Refactoring** (`refactor`, `refactoring`) — Operation icons (↝ rename, ♢ extract, ♦ inline)
- ✅ **Security** (`security`, `security_scan`) — Severity levels (‼ critical, ⚠ high, ⚡ medium, ⚐ low)

**Benefits**:
- **Faster comprehension**: Color-coding highlights important information
- **Better UX**: Professional, readable output instead of plain text
- **Consistent format**: All tools follow the same formatting pattern
- **Zero overhead**: Formatters are <1ms overhead with production guards

**Example: Rich Test Output**
```
✓ 8 passed • 10 total

Failed Tests:

  ✗ test_authentication
    tests/auth.py
    AssertionError: Invalid token

  ✗ test_database_connection
    tests/db.py
    ConnectionError: Timeout
```

**Configuration**:
Rich formatting can be controlled via multiple methods:

**Method 1: Feature Flag (Master Switch)**
```bash
# Environment variable
export VICTOR_USE_RICH_FORMATTING=false

# YAML config (~/.victor/features.yaml)
features:
  use_rich_formatting: false
```

**Method 2: Settings (Per-Tool Control)**
```python
# In settings or profile
rich_formatting_enabled = true  # Master switch
rich_formatting_tools = ["test", "git", "code_search"]  # Tool whitelist
rich_formatting_max_output_size = 1_000_000  # 1MB limit
rich_formatting_max_time_ms = 200  # 200ms timeout
```

**Method 3: Runtime Override**
```python
from victor.core.feature_flags import get_feature_flag_manager, FeatureFlag

manager = get_feature_flag_manager()
manager.disable(FeatureFlag.USE_RICH_FORMATTING)
```

**Performance Guards**:
- Input size validation (1MB limit)
- Formatting timeout protection (200ms)
- Output size truncation
- Tool whitelist enforcement
- Graceful fallback to plain text

See `victor/config/tool_settings.py` for all available options.

### How Tools Work

1. **Selection**: Victor analyzes your request and selects relevant tools
2. **Execution**: The LLM generates tool calls with appropriate parameters
3. **Results**: Tool outputs are returned to the conversation
4. **Iteration**: Victor may chain multiple tools to complete complex tasks

## Tool Categories

Victor organizes tools into functional categories. Here are the main categories with their most important tools.

### File Operations

Tools for reading, writing, and navigating the filesystem.

| Tool | Description | Cost | Example Usage |
|------|-------------|------|---------------|
| `read` | Read text/code files with pagination | FREE | "Read the file src/main.py" |
| `write` | Create or overwrite files | FREE | "Create a new config.yaml file" |
| `edit` | Atomic multi-file edits with undo | FREE | "Replace DEBUG=True with DEBUG=False" |
| `ls` | List directory contents | FREE | "List files in the tests directory" |
| `find` | Find files by name pattern | FREE | "Find all *_test.py files" |
| `overview` | Project structure overview | FREE | "Show me the project structure" |

**Example: Reading a File**
```
User: Read the first 100 lines of utils.py

Victor calls: read(path="utils.py", limit=100)
```

**Example: Editing a File**
```
User: Change the timeout from 30 to 60 seconds in config.py

Victor calls: edit(ops=[{
  "type": "replace",
  "path": "config.py",
  "old_str": "timeout = 30",
  "new_str": "timeout = 60"
}])
```

### Code Analysis

Tools for understanding code structure and relationships.

| Tool | Description | Cost | Example Usage |
|------|-------------|------|---------------|
| `grep` | Search code by text pattern | FREE | "Find all imports of requests" |
| `code_search` | Semantic code search | FREE | "Find error handling patterns" |
| `refs` | Find all usages of a symbol | FREE | "Where is calculate_total called?" |
| `symbol` | Get full code of a function/class | FREE | "Show me the User class definition" |
| `graph` | Query codebase structure/relationships | FREE | "What modules depend on auth.py?" |

**Example: Semantic Search**
```
User: Find code related to authentication

Victor calls: code_search(query="authentication login user credentials", mode="semantic")
```

**Example: Finding References**
```
User: Where is the validate_email function used?

Victor calls: refs(name="validate_email")
```

### Search Tools

Tools for searching code and web content.

| Tool | Description | Cost | Example Usage |
|------|-------------|------|---------------|
| `code_search` | Semantic/literal code search | FREE | "Find similar error handling" |
| `grep` | Fast text search (like grep) | FREE | "Find TODO comments" |
| `web_search` | Search the web | MEDIUM | "Search for Python async best practices" |
| `web_fetch` | Fetch content from URL | MEDIUM | "Get the content from that documentation page" |

**Semantic vs Literal Search**

- **Semantic** (`mode="semantic"`): Best for concepts, patterns, and architectural questions
  - "Find error handling patterns"
  - "Show authentication flow"
  - "Classes implementing the Observer pattern"

- **Literal** (`mode="literal"`): Best for exact text matches
  - "Find all uses of BaseProvider"
  - "Search for TODO comments"
  - "Find deprecated function calls"

### Shell and Execution

Tools for running commands and code.

| Tool | Description | Cost | Example Usage |
|------|-------------|------|---------------|
| `shell` | Execute shell commands with safety controls | FREE | "Run pytest tests/" |
| `sandbox` | Run Python in isolated container | MEDIUM | "Execute this data analysis script" |
| `test` | Run pytest with structured output | FREE | "Run the unit tests" |

**Example: Running Tests**
```
User: Run the tests for the auth module

Victor calls: test(path="tests/unit/test_auth.py")
```

**Shell Safety**: The `shell` tool supports a `readonly=True` parameter for safe exploration (pwd, ls, cat, grep, git status, etc.) and blocks dangerous commands like `rm -rf /` unless explicitly allowed.

### Git Operations

Tools for version control.

| Tool | Description | Cost | Example Usage |
|------|-------------|------|---------------|
| `git` | Unified git operations (status, commit, log, diff, branch, commit_msg, conflicts) | FREE/LOW | "Show git status" |
| `pr` | Create GitHub pull requests | LOW | "Create a PR for this branch" |

**Example: Git Workflow**
```
User: Stage all changes and commit with a descriptive message

Victor calls: git(operation="stage")
Victor calls: git(operation="commit_msg")  # Generates message from diff
Victor calls: git(operation="commit", message="feat(auth): add OAuth2 support for SSO login")
```

### Web Tools

Tools for accessing web content (disabled in air-gapped mode).

| Tool | Description | Cost | Example Usage |
|------|-------------|------|---------------|
| `web_search` | Search using DuckDuckGo | MEDIUM | "Search for Django REST framework docs" |
| `web_fetch` | Fetch and extract web content | MEDIUM | "Get the content from that URL" |

**Example: Web Research**
```
User: Search for FastAPI authentication best practices

Victor calls: web_search(query="FastAPI authentication best practices", ai_summarize=true)
```

### Additional Categories

| Category | Tools | Purpose |
|----------|-------|---------|
| **Docker** | `docker`, `sandbox` | Container management, isolated execution |
| **Database** | `db` | SQL operations across SQLite/PostgreSQL/MySQL |
| **RAG** | `rag_ingest`, `rag_query`, `rag_search` | Document ingestion and retrieval |
| **DevOps** | `cicd`, `pipeline`, `iac` | CI/CD, infrastructure scanning |
| **Documentation** | `docs`, `docs_coverage` | Generate docs, check coverage |
| **Security** | `scan`, `audit` | Security scanning, compliance checks |
| **Refactoring** | `rename`, `extract`, `inline` | Safe code transformations |

## Tool Selection Strategies

Victor uses intelligent strategies to select the most relevant tools for each request.

### Selection Strategies

Victor supports three tool selection strategies:

| Strategy | Speed | Quality | Best For |
|----------|-------|---------|----------|
| **Keyword** | <1ms | Good | Simple, direct requests |
| **Semantic** | 10-50ms | Better | Complex, conceptual queries |
| **Hybrid** | 10-50ms | Best | General use (default) |

### Hybrid Selection (Default)

The hybrid strategy combines semantic and keyword approaches:

```
Final Score = (0.7 * Semantic Score) + (0.3 * Keyword Score)
```

This ensures:
- **High-quality selection** via semantic similarity
- **Reliable fallbacks** via keyword matching
- **Core tools always available** regardless of query

### How Selection Works

1. **Query Analysis**: Victor analyzes your request
2. **Category Detection**: Identifies relevant tool categories (git, file, search, etc.)
3. **Mandatory Tools**: Includes tools triggered by keywords (e.g., "diff" includes `git`)
4. **Semantic Matching**: Ranks tools by semantic similarity to your query
5. **Cost Optimization**: Deprioritizes expensive tools when cheaper alternatives exist
6. **Final Selection**: Returns top tools up to the configured limit

### Configuration

Configure tool selection in your profile (`~/.victor/profiles.yaml`):

```yaml
profiles:
  default:
    # Selection strategy: keyword, semantic, or hybrid
    tool_selection_strategy: hybrid

    # Maximum tools to consider per request
    max_tools_per_request: 15

    # Similarity threshold for semantic selection (0.0-1.0)
    semantic_similarity_threshold: 0.15

    # Enable cost-aware selection (deprioritize expensive tools)
    cost_aware_selection: true
```

## Tool Budgets and Cost Tiers

### Cost Tiers

Tools are classified by resource consumption:

| Tier | Cost | Examples | Notes |
|------|------|----------|-------|
| **FREE** | None | `read`, `ls`, `grep`, `git`, `shell` | Local operations only |
| **LOW** | Minimal | `commit_msg`, `code_review` | Compute-only, no external calls |
| **MEDIUM** | Moderate | `web_search`, `web_fetch` | External API calls |
| **HIGH** | Significant | `batch` (100+ files) | Resource-intensive operations |

### Tool Budgets

Victor tracks tool usage to prevent runaway operations:

```yaml
profiles:
  default:
    # Maximum tool calls per conversation turn
    tool_budget: 25

    # Maximum concurrent tool executions
    max_concurrent_tools: 5
```

When the budget is exhausted, Victor will inform you and wait for confirmation before continuing.

## Tool Execution

### Approval Modes

Control how tool executions are approved:

| Mode | Behavior | Use Case |
|------|----------|----------|
| **auto** | Execute automatically | Trusted environments |
| **ask** | Prompt for confirmation | Default, balanced safety |
| **deny** | Block execution | Review mode |

Configure per-tool or globally:

```yaml
profiles:
  development:
    # Global approval mode
    tool_approval_mode: auto

    # Per-tool overrides
    tool_approvals:
      shell: ask      # Always ask for shell commands
      write: auto     # Auto-approve file writes
      web_search: deny  # Block web searches
```

### Dry Run Mode

Preview tool effects without executing:

```bash
victor --dry-run "Refactor the auth module"
```

In dry run mode:
- Write operations show what would change
- Shell commands display without running
- Commit operations preview the message

### Error Handling

Tools provide detailed error information:

```json
{
  "success": false,
  "error": "File not found: config.yaml",
  "suggestion": "Did you mean config.yml?"
}
```

Victor automatically:
- Retries transient failures
- Suggests corrections for common errors
- Provides alternative approaches when tools fail

## Access Modes and Safety

### Access Mode Classification

| Mode | Description | Auto-Approve | Examples |
|------|-------------|--------------|----------|
| **READONLY** | Only reads data | Yes | `read`, `ls`, `grep` |
| **WRITE** | Modifies files | With consent | `write`, `edit` |
| **EXECUTE** | Runs external code | Cautious | `shell`, `sandbox` |
| **NETWORK** | External connections | Logged | `web_search`, `web_fetch` |
| **MIXED** | Multiple access types | Careful | `git`, `docker` |

### Sandbox Restrictions

In **EXPLORE** and **PLAN** modes, write operations are restricted to `.victor/sandbox/`:

```
[EXPLORE MODE] Cannot write to 'src/main.py'.
In EXPLORE mode, edits are restricted to: .victor/sandbox/
Use /mode build to switch to build mode for unrestricted access.
```

## Custom Tools

Victor supports custom tools through plugins and the MCP protocol.

### Quick Overview

Custom tools can be:
1. **Python plugins** registered via entry points
2. **MCP servers** providing tool definitions
3. **Inline tools** defined in workflows

### Plugin Example

```python
# my_victor_plugin/tools.py
from victor.tools.base import BaseTool, ToolResult, CostTier

class MyCustomTool(BaseTool):
    name = "my_tool"
    description = "Does something useful"
    parameters = {
        "type": "object",
        "properties": {
            "input": {"type": "string", "description": "Input data"}
        },
        "required": ["input"]
    }
    cost_tier = CostTier.LOW

    async def execute(self, _exec_ctx, **kwargs):
        result = process(kwargs["input"])
        return ToolResult(success=True, output=result)
```

Register in `pyproject.toml`:

```toml
[project.entry-points."victor.tools"]
my_tool = "my_victor_plugin.tools:MyCustomTool"
```

For detailed custom tool development, see:
- [Custom Tool Tutorial](../tutorials/build-custom-tool.md)
- [Plugin Development](../development/extending/plugins.md)

## Best Practices

### Effective Tool Usage

1. **Be Specific**: "Read lines 50-100 of auth.py" is better than "Read auth.py"
2. **Use Semantic Search for Concepts**: "Find error handling patterns" works better with semantic search
3. **Use Grep for Exact Matches**: "Find TODO comments" works better with literal search
4. **Chain Operations**: Victor can combine multiple tools for complex tasks

### Performance Tips

1. **Use pagination for large files**: `read(path, offset=0, limit=500)`
2. **Filter search results**: Use `file`, `symbol`, `lang` parameters
3. **Prefer specific tools**: `refs` is faster than `grep` for finding symbol usage
4. **Cache results**: Idempotent tools (read, grep, search) cache results

### Common Patterns

**Explore, Then Edit**
```
1. Use overview() to understand project structure
2. Use code_search() to find relevant code
3. Use read() to examine specific files
4. Use edit() to make changes
```

**Review and Commit**
```
1. Use git(operation="status") to see changes
2. Use git(operation="diff") to review modifications
3. Use git(operation="commit_msg") to generate message
4. Use git(operation="commit") to commit
```

**Research and Implement**
```
1. Use web_search() to find documentation
2. Use code_search() to find similar implementations
3. Use read() to understand existing code
4. Use edit() to implement changes
5. Use test() to verify
```

## Troubleshooting

### Tool Not Found

```
Error: Tool 'my_tool' not found
```

- Check if the tool is installed
- Verify it's enabled in your profile
- Some tools require optional dependencies

### Permission Denied

```
Error: Permission denied for shell execution
```

- Check your `tool_approval_mode` setting
- Use `/mode build` for write operations
- Some operations require elevated permissions

### Tool Timeout

```
Error: Tool execution timed out after 60 seconds
```

- Increase timeout in profile: `tool_timeout: 120`
- Break large operations into smaller chunks
- Check for infinite loops in shell commands

### Selection Issues

If wrong tools are being selected:

1. Check your query clarity
2. Try being more explicit about what you need
3. Use keywords that match tool descriptions
4. Consider adjusting `semantic_similarity_threshold`

## Reference

- [Full Tool Catalog](../reference/tools/catalog.md) - All 34 tool modules with parameters
- [Tool Calling Details](../reference/tools/tool-calling.md) - Provider-specific behavior
- [Configuration Reference](../reference/configuration/index.md) - All settings
- [Custom Tool Tutorial](../tutorials/build-custom-tool.md) - Build your own tools

---

**Next**: [CLI Reference](cli-reference.md) | [Session Management](session-management.md)
