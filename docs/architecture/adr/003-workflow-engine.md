# ADR-003: Workflow Engine Architecture

## Metadata

- **Status**: Accepted
- **Date**: 2025-02-26
- **Decision Makers**: Vijaykumar Singh
- **Related ADRs**: ADR-001 (Agent Orchestration), ADR-002 (State Management)

## Context

Victor needs a workflow system that:
- Supports declarative workflow definitions (YAML)
- Enables programmatic workflows (Python)
- Provides consistent execution model
- Supports advanced features (checkpointing, HITL, streaming)
- Integrates with StateGraph for complex workflows

The challenge is balancing:
- Simplicity of YAML workflows
- Power of Python workflows
- Performance through caching
- Flexibility for different use cases

## Decision

We will implement a **Two-Level Architecture** with WorkflowEngine as facade and specialized coordinators for different concerns.

### Architecture

```
WorkflowEngine (Facade)
    ↓
├── YAMLWorkflowCoordinator (YAML parsing, validation)
├── GraphExecutionCoordinator (StateGraph execution)
├── HITLCoordinator (Human-in-the-loop)
└── CacheCoordinator (Two-level caching)
    ↓
StateGraph (Execution Engine)
    ↓
CompiledGraph (Optimized execution)
```

### Node Types

1. **agent**: Run LLM agent
2. **handler**: Execute tool or function
3. **human**: Human approval/input
4. **passthrough**: Pass data unchanged
5. **compute**: Execute Python expression

### Key Features

1. **Unified Execution**: Single path through `CompiledGraph.invoke()`
2. **Two-Level Caching**: Parsed workflows + execution results
3. **HITL Integration**: Built-in human-in-the-loop support
4. **Streaming**: Real-time progress updates
5. **Checkpointing**: Save and resume workflow execution

## Rationale

### Why Two-Level Architecture?

**Facade Layer (WorkflowEngine)**:
- Simple, unified API
- Hides complexity from users
- Easy to extend

**Coordinator Layer**:
- Separation of concerns
- Each coordinator has single responsibility
- Testable in isolation

### Why YAML + Python?

**YAML**:
- ✅ Declarative and readable
- ✅ Easy to version control
- ✅ Non-developers can write
- ✅ Good for common patterns

**Python**:
- ✅ Full programming power
- ✅ Dynamic workflow construction
- ✅ Complex logic easier
- ✅ Better for advanced users

### Why StateGraph?

**Benefits**:
- LangGraph-inspired execution
- Conditional edges
- Cyclic graphs
- Copy-on-write state
- Built-in checkpointing

## Consequences

### Positive

- **Flexible**: Both YAML and Python workflows supported
- **Performant**: Two-level caching reduces redundant work
- **Observable**: Streaming provides real-time feedback
- **Resumable**: Checkpointing enables long-running workflows
- **Extensible**: Easy to add new node types

### Negative

- **Complexity**: More layers to understand
- **Learning Curve**: Two syntaxes to learn (YAML + Python)
- **Overhead**: Coordinator layer adds some overhead

### Neutral

- **API**: WorkflowEngine provides simple facade
- **Performance**: Caching offsets overhead
- **Compatibility**: Both YAML and Python supported

## Implementation

### Phase 1: Core Engine (Completed)

- ✅ WorkflowEngine facade
- ✅ YAML workflow parser
- ✅ StateGraph integration
- ✅ Basic node types

### Phase 2: Advanced Features (Completed)

- ✅ Two-level caching
- ✅ Human-in-the-loop
- ✅ Streaming execution
- ✅ Checkpointing

### Phase 3: Enhanced Features (In Progress)

- 🔄 Workflow visualization
- 🔄 Workflow validation
- 🔄 Error recovery
- 🔄 Performance optimization

## Code Example

### YAML Workflow

```yaml
name: "Content Processor"
nodes:
  - id: "analyze"
    type: "agent"
    config:
      prompt: "Analyze: {{input}}"

  - id: "summarize"
    type: "agent"
    config:
      prompt: "Summarize: {{analyze.output}}"

edges:
  - from: "start"
    to: "analyze"
  - from: "analyze"
    to: "summarize"
  - from: "summarize"
    to: "complete"
```

### Python Workflow

```python
from victor.framework import StateGraph

async def analyze(state):
    result = await agent.run(f"Analyze: {state['input']}")
    return {"analysis": result.content}

workflow = StateGraph()
workflow.add_node("analyze", analyze)
workflow.set_entry_point("analyze")
workflow.set_finish_point("analyze")

compiled = workflow.compile()
result = await compiled.ainvoke({"input": "data"})
```

### Execution

```python
from victor import Agent

agent = Agent.create()

# Run YAML workflow
result = await agent.run_workflow("workflow.yaml", input={"topic": "AI"})

# Stream workflow
async for node_id, state in agent.stream_workflow("workflow.yaml"):
    print(f"Completed: {node_id}")
```

## Alternatives Considered

### 1. Pure YAML Workflows

**Description**: Only YAML-based workflows, no Python StateGraph

**Rejected Because**:
- Limited expressiveness
- Hard to do complex logic
- No dynamic workflow construction

### 2. Pure Python Workflows

**Description**: Only Python-based workflows, no YAML

**Rejected Because**:
- Too complex for simple use cases
- Non-developers can't write workflows
- Harder to version review

### 3. Airflow-like DAGs

**Description**: Directed Acyclic Graph execution like Airflow

**Rejected Because**:
- Too complex for LLM workflows
- Doesn't support cyclic graphs
- Overhead of DAG scheduling

## References

- [LangGraph](https://github.com/langchain-ai/langgraph)
- [Workflow Patterns](https://www.workflowpatterns.com)
- [Victor WorkflowEngine](https://github.com/anvai-labs/victor/blob/develop/victor/framework/workflow_engine.py)

## Revision History

| Date | Version | Changes | Author |
|------|---------|---------|--------|
| 2025-02-26 | 1.0 | Initial ADR | Vijaykumar Singh |
