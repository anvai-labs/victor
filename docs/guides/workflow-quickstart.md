# Workflow Quickstart

**Create multi-step workflows with YAML or Python.**

## Quick Start

```yaml
# workflows/my_workflow.yaml
workflows:
  my_task:
    nodes:
      - id: step1
        type: agent
        role: researcher
        goal: "Research the topic"
        next: [step2]

      - id: step2
        type: compute
        handler: summarize
        next: []
```

```bash
victor workflow render workflows/my_workflow.yaml
victor workflow execute my_workflow
```

## Node Types

| Type | Purpose | LLM | Example |
|------|---------|-----|---------|
| `agent` | LLM-powered tasks | ✅ | Research, analysis |
| `compute` | Function calls | ❌ | Statistics, transforms |
| `condition` | Branching logic | ❌ | Quality checks |
| `parallel` | Concurrent execution | ✅ | Multi-source |
| `hitl` | Human approval | ✅ | Code reviews |

## Workflow Structure

The example topology below illustrates authoring constructs. The
[single execution engine](../architecture.md#workflow-engine) compiles the definition.

```mermaid
---
title: Example workflow with conditional and parallel stages
---
%%{init: {"theme":"base","themeVariables":{"primaryColor":"#E8EFF7","primaryTextColor":"#17324D","primaryBorderColor":"#456987","lineColor":"#456987","fontFamily":"Arial"}}}%%
flowchart LR
  S["Start node"]
  C{"Condition node"}
  P["Parallel group<br/>child executors and join policy"]
  F["Fallback node"]
  E["End node"]
  S -->|"next node"| C
  C -->|"selected branch"| P
  C -->|"alternative branch"| F
  P -->|"join succeeds"| E
  F -->|"next node"| E
```

## Node Fields

| Field | Required | Description |
|-------|----------|-------------|
| `id` | ✅ | Unique identifier |
| `type` | ✅ | Node type |
| `role` | agent | Agent persona |
| `goal` | agent | Task description |
| `handler` | compute | Function name |
| `condition` | condition | Branch function |
| `next` | ✅ | Next node IDs |

## Example Workflows

### Code Review

```yaml
workflows:
  code_review:
    nodes:
      - id: analyze
        type: agent
        role: reviewer
        goal: "Review the code for issues"
        next: [check]

      - id: check
        type: condition
        condition: has_issues
        branches:
          "yes": fix
          "no": done

      - id: fix
        type: agent
        role: developer
        goal: "Fix identified issues"
        next: [done]

      - id: done
        type: agent
        role: summarizer
        goal: "Summarize the review"
        next: []
```

### Parallel Research

```yaml
workflows:
  deep_research:
    nodes:
      - id: parallel_search
        type: parallel
        nodes:
          - id: web
            type: agent
            goal: "Search the web"
          - id: docs
            type: agent
            goal: "Search documentation"
          - id: code
            type: agent
            goal: "Search codebase"
        join_strategy: merge_all
        next: [synthesize]

      - id: synthesize
        type: agent
        goal: "Combine all findings"
        next: []
```

## Escape Hatches

Python functions for complex logic:

```python
# victor/myvertical/escape_hatches.py

def has_issues(ctx: dict) -> str:
    """Check if code has issues."""
    issues = ctx.get("issues", [])
    return "yes" if issues else "no"

def merge_results(ctx: dict) -> dict:
    """Merge parallel results."""
    return {"merged": list(ctx.values())}

CONDITIONS = {"has_issues": has_issues}
TRANSFORMS = {"merge_results": merge_results}
```

## Scheduling

```bash
# Add scheduled workflow
victor scheduler add my_workflow --cron "0 9 * * *"

# List scheduled workflows
victor scheduler list

# Remove schedule
victor scheduler remove <id>
```

## Best Practices

| Practice | Description |
|----------|-------------|
| **Start simple** | Begin with linear workflows |
| **Use conditions** | Branch on quality/confidence |
| **Parallelize** | Run independent steps together |
| **Human in loop** | Add HITL for critical decisions |
| **Escape hatches** | Python for complex logic only |

## See Also

- [Workflow DSL](workflow-development/dsl.md) - Complete DSL reference
- [Workflow Scheduling](workflow-development/scheduling.md) - Cron scheduling
- [Vertical Workflows](../development/extending/verticals.md) - Vertical-specific workflows
