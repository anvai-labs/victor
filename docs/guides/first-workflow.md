# Creating Your First Workflow

Learn how to build multi-step workflows with Victor's StateGraph engine and YAML workflow DSL.

## What is a Workflow?

A **workflow** is a multi-step process that:
- Defines multiple stages of execution
- Can include conditional logic
- Supports parallel execution
- Maintains state across steps
- Can include human-in-the-loop approvals

---

## Two Ways to Create Workflows

Victor supports two approaches:

1. **YAML Workflows** - Declarative, easy to read and modify
2. **Python StateGraphs** - Programmatic, full flexibility

This guide covers both, starting with YAML (recommended for beginners).

---

## YAML Workflows

### Basic Workflow Structure

Create a file called `my_workflow.yaml`:

```yaml
workflow: "Greeting"
description: "A simple greeting workflow"

nodes:
  - id: greet
    type: agent
    role: You are a helpful assistant.
    goal: Greet the user warmly.

edges:
  - source: start
    target: greet
  - source: greet
    target: complete
```

Run the workflow:

```python
import asyncio
from victor.framework import Agent

async def main():
    agent = Agent()

    # Run the workflow
    result = await agent.run_workflow("my_workflow.yaml")

    print(result)

asyncio.run(main())
```

---

### Multi-Step Workflow

Create a workflow that processes input through multiple steps:

```yaml
workflow: "Content Processor"
description: "Analyze and summarize content"

nodes:
  # Step 1: Analyze input
  - id: analyze
    type: agent
    role: You are a content analyst.
    goal: |
      Analyze the following content:
      {{input}}

      Identify:
      1. Main topic
      2. Key points
      3. Sentiment

  # Step 2: Create summary
  - id: summarize
    type: agent
    role: You are a content summarizer.
    goal: |
      Based on this analysis:
      {{analyze.output}}

      Create a concise summary in 2-3 sentences.

  # Step 3: Extract keywords
  - id: keywords
    type: agent
    role: You are a keyword extractor.
    goal: |
      From the original content:
      {{input}}

      Extract the top 5 keywords.

edges:
  - source: start
    target: analyze
  - source: analyze
    target: summarize
  - source: summarize
    target: keywords
  - source: keywords
    target: complete
```

Run with input:

```python
result = await agent.run_workflow(
    "content_processor.yaml",
    input={"input": "Your content here..."}
)
```

---

### Conditional Workflows

Add conditional logic based on analysis:

```yaml
workflow: "Content Router"
description: "Route content based on type"

nodes:
  - id: classify
    type: agent
    role: Content classifier
    goal: |
      Classify this content as one of: [technical, creative, business]
      Content: {{input}}
      Respond with only the category name.

  - id: technical
    type: agent
    role: Technical expert
    goal: Analyze: {{input}}

  - id: creative
    type: agent
    role: Creative writer
    goal: Analyze: {{input}}

  - id: business
    type: agent
    role: Business analyst
    goal: Analyze: {{input}}

edges:
  - source: start
    target: classify

  # Conditional routing based on classify output
  - source: classify
    target: technical
    condition: "{{classify.content}} == 'technical'"

  - source: classify
    target: creative
    condition: "{{classify.content}} == 'creative'"

  - source: classify
    target: business
    condition: "{{classify.content}} == 'business'"

  # All paths lead to completion
  - source: technical
    target: complete
  - source: creative
    target: complete
  - source: business
    target: complete
```

---

### Workflows with Tools

Create a workflow that uses tools to analyze code:

```yaml
workflow: "Code Review"
description: "Review and analyze code"

nodes:
  - id: read_code
    type: compute
    tools:
      - read_file
    config:
      file_path: "{{file_path}}"

  - id: analyze
    type: agent
    role: Senior code reviewer
    tools:
      - grep
    goal: |
      Review this code:
      {{read_code.file_content}}

      Check for:
      - Potential bugs
      - Security issues
      - Code style violations
      - Performance problems

  - id: suggest
    type: agent
    role: Code improvement advisor
    goal: |
      Based on the analysis:
      {{analyze.content}}

      Provide specific improvement suggestions.

edges:
  - source: start
    target: read_code
  - source: read_code
    target: analyze
  - source: analyze
    target: suggest
  - source: suggest
    target: complete
```

---

### Parallel Execution

Run multiple steps in parallel:

```yaml
workflow: "Multi-Perspective Analysis"
description: "Analyze from multiple angles simultaneously"

nodes:
  - id: technical_review
    type: agent
    role: Technical analyst
    goal: Technical analysis of: {{input}}

  - id: business_review
    type: agent
    role: Business analyst
    goal: Business analysis of: {{input}}

  - id: user_review
    type: agent
    role: UX analyst
    goal: User experience analysis of: {{input}}

  - id: synthesize
    type: agent
    role: Senior analyst
    goal: |
      Synthesize these three perspectives:
      Technical: {{technical_review.content}}
      Business: {{business_review.content}}
      User: {{user_review.content}}

      Provide a comprehensive analysis.

edges:
  # Start all reviews in parallel
  - source: start
    target: technical_review
  - source: start
    target: business_review
  - source: start
    target: user_review

  # Wait for all to complete, then synthesize
  - source: technical_review
    target: synthesize
  - source: business_review
    target: synthesize
  - source: user_review
    target: synthesize

  - source: synthesize
    target: complete
```

---

## Python StateGraphs

For more complex workflows, use Python directly:

### Basic StateGraph

```python
import asyncio
from victor.framework import StateGraph
from victor.framework import Agent

async def analyze_node(state):
    """Analyze input."""
    prompt = f"Analyze this: {state['input']}"
    agent = Agent()
    result = await agent.run(prompt)
    return {"analysis": result.content}

async def summarize_node(state):
    """Summarize analysis."""
    prompt = f"Summarize: {state['analysis']}"
    agent = Agent()
    result = await agent.run(prompt)
    return {"summary": result.content}

# Create the workflow
workflow = StateGraph()

# Add nodes
workflow.add_node("analyze", analyze_node)
workflow.add_node("summarize", summarize_node)

# Add edges
workflow.set_entry_point("analyze")
workflow.add_edge("analyze", "summarize")
workflow.set_finish_point("summarize")

# Compile and run
compiled = workflow.compile()
result = await compiled.invoke({"input": "Your content here"})
print(result["summary"])
```

### Conditional Routing

```python
async def classify_node(state):
    """Classify content."""
    agent = Agent()
    result = await agent.run(
        f"Classify as [technical, creative, business]: {state['input']}"
    )
    category = result.content.strip().lower()
    return {"category": category}

async def route_function(state):
    """Route based on classification."""
    category = state.get("category", "business")
    return category

# Create workflow
workflow = StateGraph()
workflow.add_node("classify", classify_node)
workflow.add_node("technical", technical_node)
workflow.add_node("creative", creative_node)
workflow.add_node("business", business_node)

# Set entry point
workflow.set_entry_point("classify")

# Add conditional edges
workflow.add_conditional_edges(
    "classify",
    route_function,
    {
        "technical": "technical",
        "creative": "creative",
        "business": "business",
    }
)

# All paths finish
workflow.set_finish_point("technical")
workflow.set_finish_point("creative")
workflow.set_finish_point("business")
```

---

## Best Practices

### 1. Keep Nodes Focused

```yaml
# ❌ Bad: Too much responsibility
- id: do_everything
  type: agent
  goal: Analyze, summarize, translate, and format...

# ✅ Good: Single responsibility
- id: analyze
  type: agent
  goal: Analyze the input
- id: summarize
  type: agent
  goal: Summarize the analysis
```

### 2. Use Descriptive Names

```yaml
# ❌ Bad: Vague names
- id: step1
- id: step2

# ✅ Good: Descriptive names
- id: analyze_requirements
- id: design_solution
```

### 3. Document Your Workflow

```yaml
workflow: "Data Processing Pipeline"
description: |
  This workflow processes raw data through three stages:
  1. Validation - Check data quality
  2. Transformation - Convert to standard format
  3. Analysis - Generate insights

  Input: Raw data file path
  Output: Analysis report
```

---

## Common Workflow Patterns

### Pattern 1: Map-Reduce

```yaml
# Process items in parallel, then combine results
nodes:
  - id: map
    type: agent
    goal: Process: {{item}}
    items: "{{items}}"

  - id: reduce
    type: agent
    goal: Combine these results: {{map.outputs}}
```

### Pattern 2: Sequential Approval

```yaml
# Multiple approval stages
nodes:
  - id: manager_approval
    type: human
  - id: director_approval
    type: human
  - id: final_approval
    type: human
```

### Pattern 3: Retry with Backoff

```yaml
nodes:
  - id: attempt
    type: compute
    config:
      max_retries: 3
      backoff: exponential
```

---

## Putting It All Together

### Example: Blog Post Pipeline

```yaml
workflow: "Blog Post Creator"
description: "Create and review a blog post"

nodes:
  # Step 1: Generate outline
  - id: outline
    type: agent
    role: Blog content strategist
    goal: |
      Create a detailed outline for a blog post about: {{topic}}
      Include:
      - Catchy title
      - Section headings
      - Key points for each section

  # Step 2: Write content
  - id: write
    type: agent
    role: Blog writer
    goal: |
      Write the full blog post based on this outline:
      {{outline.content}}

      Make it engaging and informative.

  # Step 3: SEO optimization
  - id: seo
    type: agent
    role: SEO specialist
    goal: |
      Optimize this blog post for SEO:
      {{write.content}}

      Add:
      - Meta description
      - Keywords
      - Suggested title tags

  # Step 4: Human review
  - id: human_review
    type: human
    goal: |
      Review the blog post:
      {{write.content}}

      SEO suggestions:
      {{seo.content}}

      Approve for publication?

  # Step 5: Finalize
  - id: finalize
    type: agent
    role: Blog formatter
    goal: |
      Create the final HTML version:
      Content: {{write.content}}
      SEO: {{seo.content}}

edges:
  - from: start
    target: outline
  - from: outline
    target: write
  - from: write
    target: seo
  - from: seo
    target: human_review
  - from: human_review
    target: finalize
    condition: "{{human_review.approved}} == true"
  - from: human_review
    target: write
    condition: "{{human_review.approved}} == false"
  - from: finalize
    target: complete
```

---

## Troubleshooting

### Workflow Not Executing

**Check**: YAML syntax is valid
```bash
python -c "import yaml; print(yaml.safe_load(open('my_workflow.yaml')))"
```

### Nodes Not Connecting

**Check**: Edge node IDs match node IDs
```yaml
nodes:
  - id: "step1"
edges:
  - from: "start"
    to: "step1"  # Must match exactly
```

### Conditional Edges Not Working

**Check**: Condition syntax is correct
```yaml
condition: "{{previous_node.content}} == 'expected_value'"
```

---

## Next Steps

- 📖 [Workflow DSL Reference](workflow-development/dsl.md) - Complete YAML syntax
- 🔄 [Advanced Workflows](workflow-development/examples.md) - Complex patterns
- 🤖 [Building Agents](first-agent.md) - Create agents for workflows
- 📚 [StateGraph API](../api-reference/workflows.md) - Low-level Python API

---

## Quick Reference

```yaml
# Basic workflow structure
workflow: "Workflow Name"
nodes:
  - id: node1
    type: agent
    role: Your role
    goal: What to do

edges:
  - source: start
    target: node1
  - source: node1
    target: complete
```

```python
# Run workflow
result = await agent.run_workflow(
    "workflow.yaml",
    input={...}
)
```
