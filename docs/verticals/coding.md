# Coding Vertical

The Coding vertical provides comprehensive code analysis, review, refactoring, and generation capabilities. It is designed to compete with GitHub Copilot Workspace, Cursor, and Sourcegraph Cody.

## Overview

The Coding vertical (`verticals/victor-coding/victor_coding/`) is Victor's core vertical for software development tasks. It provides intelligent code understanding through Tree-sitter AST parsing, LSP integration for real-time code intelligence, and semantic code search capabilities.

### Key Use Cases

- **Code Review**: Automated analysis of code changes with quality, security, and performance feedback
- **Feature Implementation**: End-to-end feature development with planning, implementation, and testing
- **Test-Driven Development (TDD)**: Red-green-refactor workflow with test generation
- **Refactoring**: Safe code restructuring with AST-aware transformations
- **Code Analysis**: Static analysis, dependency graphing, and codebase exploration

## Available Tools

The Coding vertical uses the following tools from `victor.tools.tool_names`:

| Tool | Description |
|------|-------------|
| `read` | Read file contents for analysis |
| `write` | Write new files or overwrite existing ones |
| `edit` | Make targeted edits to existing files |
| `ls` | List directory contents |
| `grep` | Keyword-based code search |
| `code_search` | Semantic code search using embeddings |
| `lsp` | Language-server diagnostics, definitions, references, hover, and related code intelligence |
| `symbol` | Targeted symbol lookup for functions, classes, and other named definitions |
| `refs` | Reference lookup for a known symbol or identifier |
| `project_overview` | Structure-aware project and directory overview before opening files |
| `overview` | Generate codebase overview and structure |
| `graph` | Code graph analysis (PageRank, dependencies) |
| `shell` | Execute shell commands for builds, tests, etc. |

## Default Code-Intelligence Workflow

Victor's default coding modes (`plan`, `build`, `review`, and `delegate`) are
code-intelligence-first. When the agent attempts to read a whole source file before narrowing
the target, the runtime steers that call through structure-aware tools first:

- Use `lsp(action="diagnostics", file_path=...)` to find concrete error or warning ranges.
- Use `symbol(file_path=..., symbol_name=...)` when the target symbol is known.
- Use `refs(symbol_name=..., search_path=...)` when references are the fastest way to find call sites.
- Use `project_overview(path=..., max_depth=...)` when the target file or symbol is still ambiguous.

After diagnostics, symbol lookup, references, or project overview produce a line or file hint,
follow-up broad reads are narrowed to a small `read(path, offset, limit)` range. Full-file reads
remain available for non-code files, explicitly targeted ranges, and advanced workflows that opt
into broader framework behavior.

## Default Coding-Agent Modes

`victor chat --mode` exposes a narrow coding-agent surface by default:

- `plan`: inspect the repository, narrow the target, and produce an implementation plan.
- `build`: make code changes, run focused validation, and report the result.
- `review`: inspect changes or architecture for defects, risks, and missing tests.
- `delegate`: route parallel coding work through workspace-isolated team execution and follow-up contracts.

`explore` remains available as an advanced opt-in compatibility mode for broader repository
discovery. Workflow DSL execution and delegate follow-up contract injection are also explicit
opt-ins through `victor workflow` or `victor chat --workflow ...`, not part of the default coding
mode surface.

## Available Workflows

### 1. Code Review (`code_review.yaml`)

Systematic code review workflow with multiple analysis phases:

```yaml
workflows:
  code_review:
    nodes:
      - understand_changes    # Analyze what changed
      - review_correctness    # Verify logic and behavior
      - review_quality        # Code style and best practices
      - review_security       # Security vulnerability check
      - review_performance    # Performance considerations
      - synthesize_review     # Combine all feedback
      - generate_report       # Create review document
```

**Key Features**:
- Parallel review of correctness, quality, security, and performance
- Configurable review depth (quick, standard, thorough)
- HITL approval gates for critical issues
- Structured output with actionable suggestions

### 2. Feature Implementation (`feature.yaml`)

End-to-end feature development workflow:

```yaml
workflows:
  feature:
    nodes:
      - understand_requirements  # Parse feature request
      - analyze_codebase        # Understand existing code
      - create_plan             # Design implementation approach
      - implement_changes       # Write the code
      - write_tests             # Create test coverage
      - verify_implementation   # Run tests and validate
      - finalize                # Clean up and document
```

**Key Features**:
- Requirement analysis with clarification requests
- Codebase exploration before implementation
- Incremental implementation with verification
- Automatic test generation

### 3. Test-Driven Development (`tdd.yaml`)

Red-green-refactor TDD workflow:

```yaml
workflows:
  tdd:
    nodes:
      - understand_feature     # What to build
      - write_failing_test     # RED: Create test first
      - run_tests_red          # Verify test fails
      - implement_minimum      # GREEN: Make it pass
      - run_tests_green        # Verify test passes
      - refactor               # REFACTOR: Improve code
      - run_tests_final        # Verify still passes
```

**Key Features**:
- Strict TDD cycle enforcement
- Automatic test execution at each phase
- Refactoring suggestions based on code smells
- Coverage tracking

### 4. Refactoring (`refactor.yaml`)

Safe code restructuring workflow:

```yaml
workflows:
  refactor:
    nodes:
      - analyze_code           # Understand current structure
      - identify_improvements  # Find refactoring opportunities
      - plan_refactoring       # Design safe transformation steps
      - apply_changes          # Execute refactoring
      - verify_behavior        # Ensure no regressions
      - update_tests           # Adjust tests if needed
```

**Key Features**:
- AST-aware transformations
- Behavior preservation verification
- Incremental changes with rollback capability
- Test suite validation

## Stage Definitions

The Coding vertical progresses through these stages:

| Stage | Description | Primary Tools |
|-------|-------------|---------------|
| `INITIAL` | Understanding the coding request | `read`, `ls`, `overview` |
| `UNDERSTANDING` | Analyzing existing code | `read`, `grep`, `code_search`, `graph` |
| `PLANNING` | Creating implementation plan | `read`, `overview` |
| `IMPLEMENTING` | Writing and modifying code | `write`, `edit`, `shell` |
| `TESTING` | Running and writing tests | `shell`, `write`, `edit` |
| `REVIEWING` | Verifying changes | `read`, `grep`, `shell` |
| `COMPLETION` | Finalizing deliverables | `write`, `read` |

## Key Features

### Tree-sitter Integration

The Coding vertical uses Tree-sitter for fast, accurate AST parsing across 20+ programming languages:

```python
from victor_coding.ast.tree_sitter_manager import TreeSitterManager

manager = TreeSitterManager()
tree = manager.parse_file("example.py")
# Access nodes, find definitions, analyze structure
```

**Supported Languages**: Python, JavaScript, TypeScript, Go, Rust, Java, C, C++, Ruby, PHP, and more.

### LSP Integration

Language Server Protocol support provides:
- Go-to-definition
- Find references
- Symbol search
- Diagnostics
- Hover information

```python
from victor_coding.lsp.manager import LSPManager

lsp = LSPManager()
await lsp.start_server("python")
definitions = await lsp.get_definitions(file_path, position)
```

### Semantic Code Search

Beyond keyword search, the Coding vertical supports semantic code search using embeddings:

```python
# Find code semantically similar to a query
results = await code_search.search("function that validates user input")
```

### Code Graph Analysis

Understand code relationships through graph analysis:
- Dependency graphs
- Call graphs
- Import relationships
- PageRank-based importance scoring

## Configuration Options

### Vertical Configuration

```python
from victor_coding.assistant import CodingAssistant

# Get system prompt for coding tasks
prompt = CodingAssistant.get_system_prompt()

# Get tiered tool configuration
tiered_tools = CodingAssistant.get_tiered_tool_config()

# Access capability provider
capabilities = CodingAssistant.get_capability_provider()
```

### Mode Configuration

The Coding vertical supports three agent modes:

| Mode | Description | Tool Access |
|------|-------------|-------------|
| `BUILD` | Full implementation (default) | All tools, full edits |
| `PLAN` | Planning and exploration | 2.5x exploration, sandbox only |
| `EXPLORE` | Analysis only | 3.0x exploration, no edits |

### Workflow Parameters

Common workflow configuration options:

```yaml
# In workflow YAML
llm_config:
  temperature: 0.2      # Lower for precise code generation
  model_hint: claude-sonnet  # Preferred model

tool_budget: 30         # Max tool calls per node
timeout: 300            # Node timeout in seconds
```

## Example Usage

Use the public factory with the `coding` vertical. Install the corresponding
`victor-coding` package if it is not already available in your environment.

```python
import asyncio
from victor.framework import Agent

async def main():
    async with await Agent.create(
        vertical="coding", provider="ollama", model="llama3.1:8b"
    ) as agent:
        result = await agent.run("Review the last commit for security issues")
        print(result.content)

asyncio.run(main())
```

For a named workflow supplied by the installed vertical, call
`await agent.run_workflow(workflow_name, context={...})` on the configured agent.
Use the installed package's workflow catalog to select a name and its expected input
keys. A bare workflow compiler does not create the agent runtime or provider.
See [Python API](../reference/api/python-api.md) and
[workflow execution](../tutorials/create-workflow.md).

## Integration with Other Verticals

The Coding vertical integrates with:

- **DevOps**: Deployment pipelines after code changes
- **RAG**: Documentation search for context
- **Research**: Technical documentation lookup

## File Structure

```
verticals/victor-coding/victor_coding/
├── assistant.py          # CodingAssistant vertical definition
├── capabilities.py       # Capability providers
├── mode_config.py        # Mode configurations
├── prompts.py            # Prompt templates
├── safety.py             # Safety checks for code operations
├── tool_dependencies.py  # Tool dependency configuration
├── ast/
│   ├── tree_sitter_manager.py  # Tree-sitter integration
│   └── ...
├── lsp/
│   ├── manager.py        # LSP client management
│   └── ...
├── codebase/
│   ├── analyzer.py       # Codebase analysis
│   └── ...
├── review.py             # Code review logic
├── test_generation.py    # Test generation utilities
├── workflows/
│   ├── code_review.yaml
│   ├── feature.yaml
│   ├── tdd.yaml
│   └── refactor.yaml
├── handlers.py           # Compute handlers for workflows
├── escape_hatches.py     # Complex condition logic
├── rl.py                 # Reinforcement learning config
└── teams.py              # Multi-agent team specs
```

## Best Practices

1. **Start with exploration**: Use `EXPLORE` mode to understand the codebase before making changes
2. **Use semantic search**: Leverage `code_search` for finding relevant code beyond keyword matching
3. **Validate incrementally**: Run tests after each significant change
4. **Review before commit**: Use the code review workflow for self-review
5. **Document changes**: Update documentation as part of the workflow
