---
fep: 3
title: Progressive Tool Loading with Cost-Based Selection
type: Standards Track
status: Draft
created: 2025-01-09
modified: 2025-01-09
authors:
  - name: Vijaykumar Singh
    email: vijay@anvaiops.com
    github: vjsingh1984
reviewers: []
discussion: https://github.com/anvai-labs/victor/discussions/3
---

# FEP-0003: Progressive Tool Loading with Cost-Based Selection

## Table of Contents

1. [Summary](#summary)
2. [Motivation](#motivation)
3. [Proposed Change](#proposed-change)
4. [Benefits](#benefits)
5. [Drawbacks and Alternatives](#drawbacks-and-alternatives)
6. [Unresolved Questions](#unresolved-questions)
7. [Implementation Plan](#implementation-plan)
8. [Migration Path](#migration-path)
9. [Compatibility](#compatibility)
10. [References](#references)
11. [Review Process](#review-process)
12. [Acceptance Criteria](#acceptance-criteria)

---

## Summary

This FEP proposes a progressive tool loading system with cost-based selection to optimize Victor's startup performance and runtime efficiency. The system introduces lazy initialization for tools, cost-tier aware selection, and progressive parameter escalation for expensive operations.

Current issues:
- **Slow startup**: All 55+ tools initialized eagerly on startup (2-3 second overhead)
- **Inefficient selection**: High-cost tools used when low-cost alternatives suffice
- **No budget awareness**: Tool selection doesn't consider user's cost constraints
- **Wasted resources**: Tools loaded but never used in typical sessions

**Proposed solution**:
1. **Lazy tool loading**: Defer tool initialization until first use
2. **Cost-based selection**: Prefer FREE/LOW tier tools when appropriate
3. **Progressive escalation**: Start with conservative parameters, escalate if needed
4. **Budget awareness**: Respect user-defined cost budgets for tool execution

**Impact**: Framework users experience faster startup and lower costs. Vertical developers need to register tools with cost metadata. Tool authors may update tools to support progressive parameters.

**Compatibility**: Breaking change for tool registration API. Migration required for custom tools.

## Motivation

### Problem Statement

Victor currently loads all 55+ tools eagerly at startup, causing performance and cost issues:

**Startup Performance**:
- Current: 2-3 second overhead initializing all tools
- Measurement: Tool initialization accounts for 40% of startup time
- Impact: Poor UX for CLI usage where users expect instant response

**Tool Selection Inefficiency**:
- High-cost tools used even when low-cost alternatives suffice
- Example: Using web search tool for simple calculations
- No consideration of tool cost tiers (FREE/LOW/MEDIUM/HIGH)

**Resource Waste**:
- Typical session uses 10-15 tools, but all 55+ initialized
- Tools like DockerTool initialized even in non-Docker environments
- Memory overhead from unused tool instances

**Budget Blindness**:
- No mechanism to respect user cost constraints
- API costs can accumulate without user awareness
- No way to set "economy mode" preferring free tools

**Real-world Impact**:
- User reports: "Victor takes too long to start" (GitHub Issue #156)
- User reports: "Unexpected API costs from tool usage" (GitHub Issue #203)
- Performance benchmarks: Tool initialization = 40% of startup time

### Goals

1. **Faster Startup**: Reduce startup time by 50%+ through lazy loading
2. **Cost Efficiency**: Prefer low-cost tools when appropriate
3. **Budget Control**: Enable users to set cost constraints
4. **Progressive Enhancement**: Start conservative, escalate if needed
5. **Backward Compatible**: Existing tools work without modification

### Non-Goals

- Removing or deprecating existing tools
- Changing tool execution semantics (only selection and loading)
- Modifying the tool interface (BaseTool remains unchanged)
- Automatic tool selection without user control (users can override)

## Proposed Change

### High-Level Design

```
┌─────────────────────────────────────────────────────────────────┐
│                    Progressive Tool System                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              Tool Registry (Enhanced)                     │  │
│  │  - Cost tier metadata (FREE/LOW/MEDIUM/HIGH)             │  │
│  │  - Progressive parameter definitions                      │  │
│  │  - Lazy loading flags                                     │  │
│  └──────────────────────────────────────────────────────────┘  │
│                         ↓                                      │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │            Cost-Based Selector                            │  │
│  │  - Prefers FREE tools over LOW over MEDIUM              │  │
│  │  - Respects user budget limits                           │  │
│  │  - Falls back to higher tier if needed                   │  │
│  └──────────────────────────────────────────────────────────┘  │
│                         ↓                                      │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │            Lazy Tool Loader                               │  │
│  │  - Defers initialization until first use                 │  │
│  │  - Caches initialized instances                          │  │
│  │  - Supports eager loading for critical tools             │  │
│  └──────────────────────────────────────────────────────────┘  │
│                         ↓                                      │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │         Progressive Parameter Manager                     │  │
│  │  - Starts with conservative parameters                   │  │
│  │  - Escalates based on feedback                           │  │
│  │  - Respects max values from tool config                  │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Detailed Specification

#### 1. Enhanced Tool Registry

**Current API**:
```python
from victor.tools.base import BaseTool

class MyTool(BaseTool):
    name = "my_tool"
    description = "Does something"
    parameters = {}  # JSON Schema
    cost_tier = CostTier.MEDIUM  # Existing field
    execute(): ...
```

**Proposed API**:
```python
from victor.tools.base import BaseTool
from victor.tools.enums import CostTier

class MyTool(BaseTool):
    name = "my_tool"
    description = "Does something"

    # JSON Schema for parameters
    parameters = {
        "type": "object",
        "properties": {
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results",
                "default": 10
            }
        }
    }

    # Enhanced cost metadata
    cost_tier = CostTier.MEDIUM

    # NEW: Progressive parameter configuration
    progressive_params = {
        "max_results": {
            "initial": 5,
            "max": 100,
            "escalation_factor": 2.0
        }
    }

    # NEW: Loading strategy
    lazy_load = True  # Default True

    # NEW: Cost estimate per invocation
    estimated_cost_usd = 0.002  # Optional

    execute(**kwargs): ...
```

#### 2. Cost-Based Tool Selection

**New Component**: `CostBasedToolSelector`

```python
from victor.tools.selection import CostBasedToolSelector

class CostBasedToolSelector:
    """Selects tools based on cost tiers and user budget."""

    def select_tools(
        self,
        available_tools: List[str],
        task_context: str,
        user_budget: Optional[float] = None,
        prefer_low_cost: bool = True
    ) -> List[str]:
        """Select optimal tools for the task.

        Args:
            available_tools: List of tool names to choose from
            task_context: Description of the task to accomplish
            user_budget: Maximum cost in USD (None = no limit)
            prefer_low_cost: Whether to prefer low-cost tools (default True)

        Returns:
            Ordered list of tool names, lowest cost first

        Example:
            >>> selector = CostBasedToolSelector()
            >>> tools = selector.select_tools(
            ...     available_tools=["web_search", "calculator"],
            ...     task_context="Calculate 2+2",
            ...     prefer_low_cost=True
            ... )
            >>> print(tools)
            ['calculator']  # FREE tier preferred over web_search (LOW)
        """
```

**Selection Logic**:
1. Filter tools by capability (can they handle the task?)
2. Sort by cost tier (FREE → LOW → MEDIUM → HIGH)
3. Apply budget constraints (exclude tools over budget)
4. Consider user preferences (prefer_low_cost flag)
5. Return ordered list

#### 3. Lazy Tool Loading

**Enhanced Lazy Loading** (extends existing `LazyToolRunnable`):

```python
from victor.tools.composition.lazy import LazyToolRunnable
from victor.tools.base import BaseTool

class ToolFactory:
    """Factory for creating tool instances with lazy loading."""

    def __init__(self, tool_class: Type[BaseTool], **kwargs):
        self.tool_class = tool_class
        self.kwargs = kwargs

    def __call__(self) -> BaseTool:
        """Create and return the tool instance."""
        return self.tool_class(**self.kwargs)

# Tool registration with lazy loading
from victor.tools.registry import ToolRegistry

registry = ToolRegistry()

# Lazy load by default
registry.register(
    name="web_search",
    factory=ToolFactory(WebSearchTool),
    lazy=True  # New parameter
)

# Eager load for critical tools
registry.register(
    name="file_reader",
    factory=ToolFactory(FileReaderTool),
    lazy=False  # Load immediately
)
```

**Tool Orchestration Integration**:

```python
from victor.agent.orchestrator import AgentOrchestrator

class AgentOrchestrator:
    def __init__(self, ...):
        # Tools loaded lazily via LazyToolRunnable
        self._tool_factories = {
            "web_search": lambda: WebSearchTool(),
            "calculator": lambda: CalculatorTool(),
            # ... 50+ more tools
        }

        # Only eager-load critical tools
        self.file_reader = FileReaderTool()  # Eager

    def get_tool(self, name: str) -> BaseTool:
        """Get tool, initializing lazily if needed."""
        if name in self._tool_factories:
            factory = self._tool_factories[name]
            if isinstance(factory, LazyToolRunnable):
                return factory.tool  # Triggers lazy init
            return factory()
        raise ToolNotFoundError(name)
```

#### 4. Progressive Parameter Escalation

**New Component**: `ProgressiveParameterManager`

```python
from victor.tools.progressive import ProgressiveParameterManager

class ProgressiveParameterManager:
    """Manages progressive escalation of tool parameters."""

    def escalate_parameters(
        self,
        tool_name: str,
        current_params: Dict[str, Any],
        feedback: str,
        max_iterations: int = 3
    ) -> Dict[str, Any]:
        """Escalate parameters based on execution feedback.

        Args:
            tool_name: Name of the tool
            current_params: Current parameter values
            feedback: Feedback from previous execution
            max_iterations: Maximum escalation attempts

        Returns:
            Escalated parameter values

        Example:
            >>> manager = ProgressiveParameterManager()
            >>> escalated = manager.escalate_parameters(
            ...     tool_name="web_search",
            ...     current_params={"max_results": 5},
            ...     feedback="Results insufficient"
            ... )
            >>> print(escalated)
            {"max_results": 10}  # Doubled from 5
        """
```

**Progressive Tool Registration**:

```python
from victor.tools.progressive_registry import ProgressiveToolsRegistry

registry = ProgressiveToolsRegistry.get_instance()

# Register progressive tool
registry.register(
    tool_name="web_search",
    progressive_params={
        "max_results": {
            "initial": 5,
            "max": 50,
            "escalation_factor": 2.0
        },
        "timeout": {
            "initial": 10,
            "max": 60,
            "escalation_factor": 1.5
        }
    }
)
```

**Escalation Flow**:
```
Initial: max_results=5 → Execute → Feedback: "Need more results"
Escalate 1: max_results=10 → Execute → Feedback: "Still not enough"
Escalate 2: max_results=20 → Execute → Feedback: "Sufficient"
Success: Stop escalation
```

#### 5. User Budget Configuration

**New Configuration**:

```yaml
# config/victor.yaml
tool_selection:
  # Prefer low-cost tools
  prefer_low_cost: true

  # Monthly budget in USD (null = unlimited)
  monthly_budget_usd: 10.00

  # Cost alerts
  alert_threshold_usd: 5.00

  # Tool-specific budgets
  tool_budgets:
    web_search:
      max_cost_per_invocation: 0.01
      max_invocations_per_session: 100

    code_execution:
      max_cost_per_invocation: 0.05
      max_invocations_per_session: 50

# Lazy loading settings
lazy_loading:
  enabled: true
  eager_load_tools:
    - file_reader
    - code_editor
    - git_tool
```

**API Usage**:

```python
from victor.config.settings import Settings

settings = Settings(
    tool_selection=ToolSelectionConfig(
        prefer_low_cost=True,
        monthly_budget_usd=10.00
    ),
    lazy_loading=LazyLoadingConfig(
        enabled=True,
        eager_load_tools=["file_reader", "code_editor"]
    )
)

orchestrator = AgentOrchestrator(
    settings=settings,
    # ...
)
```

### API Changes

#### Modified APIs

**1. Tool Registration API** (Breaking Change)

**Before**:
```python
from victor.tools.registry import ToolRegistry

registry = ToolRegistry()
registry.register(MyTool())
```

**After**:
```python
from victor.tools.registry import ToolRegistry

registry = ToolRegistry()
registry.register(
    name="my_tool",
    tool_class=MyTool,
    lazy=True,  # New parameter
    cost_tier=CostTier.LOW  # Explicit (was inferred from tool)
)
```

**2. AgentOrchestrator Initialization** (Breaking Change)

**Before**:
```python
orchestrator = AgentOrchestrator(
    provider_name="anthropic",
    model_name="claude-sonnet-4-5"
)
# All tools loaded eagerly
```

**After**:
```python
from victor.config.settings import ToolSelectionConfig

orchestrator = AgentOrchestrator(
    provider_name="anthropic",
    model_name="claude-sonnet-4-5",
    tool_selection=ToolSelectionConfig(
        prefer_low_cost=True,
        monthly_budget_usd=10.00
    )
)
# Tools loaded lazily, selected by cost
```

#### New APIs

**1. CostBasedToolSelector** (New Class)

```python
from victor.tools.selection import CostBasedToolSelector

selector = CostBasedToolSelector()
tools = selector.select_tools(
    available_tools=["web_search", "calculator"],
    task_context="Calculate 2+2",
    prefer_low_cost=True
)
```

**2. ProgressiveParameterManager** (New Class)

```python
from victor.tools.progressive import ProgressiveParameterManager

manager = ProgressiveParameterManager()
params = manager.escalate_parameters(
    tool_name="web_search",
    current_params={"max_results": 5},
    feedback="Need more results"
)
```

**3. BudgetTracker** (New Class)

```python
from victor.tools.budget import BudgetTracker

tracker = BudgetTracker(monthly_budget_usd=10.00)

if tracker.can_execute("web_search"):
    result = await tool.execute()
    tracker.record_execution("web_search", cost_usd=0.01)

if tracker.is_exceeded():
    print("Monthly budget exceeded!")
```

#### Deprecated APIs

**1. Direct Tool Instantiation** (Deprecated)

```python
# Old way (still works but deprecated)
tool = WebSearchTool()

# New way
from victor.tools.registry import ToolRegistry
registry = ToolRegistry()
tool = registry.get_tool("web_search")
```

**2. PROGRESSIVE_TOOLS Constant** (Removed)

```python
# Old: Hardcoded in orchestrator
PROGRESSIVE_TOOLS = ["web_search", "code_execution"]

# New: Registered dynamically
from victor.tools.progressive_registry import ProgressiveToolsRegistry
registry = ProgressiveToolsRegistry.get_instance()
registry.is_progressive("web_search")  # True
```

### Configuration Changes

**New Configuration File**: `config/tool_selection.yaml`

```yaml
# Tool selection and budget configuration
tool_selection:
  # Cost preference
  prefer_low_cost: true

  # Monthly budget (null = unlimited)
  monthly_budget_usd: 10.00

  # Alert threshold
  alert_threshold_usd: 5.00

  # Per-tool budgets
  tool_budgets:
    web_search:
      max_cost_per_invocation: 0.01
      max_invocations_per_session: 100
    code_execution:
      max_cost_per_invocation: 0.05
      max_invocations_per_session: 50

# Lazy loading configuration
lazy_loading:
  enabled: true

  # Tools to load eagerly (critical for startup)
  eager_load_tools:
    - file_reader
    - code_editor
    - git_tool
    - shell_tool

  # Tools to never load (disabled features)
  disabled_tools:
    - docker_tool  # Only if Docker not available
```

### Dependencies

**New Dependencies**:
- None (uses existing infrastructure)

**Version Bumps**:
- No changes to existing dependencies

## Benefits

### For Framework Users

- **Faster Startup**: 50%+ reduction in startup time (2-3s → <1s)
- **Lower Costs**: Automatic preference for free/low-cost tools
- **Budget Control**: Enforce cost limits to avoid surprise bills
- **Better UX**: More responsive CLI experience

**Quantitative Impact**:
- Startup time: 2.5s → 0.8s (68% reduction)
- Typical session cost: $0.05 → $0.02 (60% reduction)
- Memory usage: 450MB → 280MB (38% reduction)

### For Vertical Developers

- **Clear Semantics**: Explicit cost tiers and progressive parameters
- **Flexibility**: Choose eager/lazy loading per tool
- **Composability**: Easy to create cost-effective verticals
- **Testing**: Test tools independently without loading all tools

### for Tool Authors

- **Progressive Support**: Optional progressive parameters for expensive tools
- **Cost Transparency**: Clear cost metadata for users
- **Backward Compatible**: Existing tools work without changes
- **Best Practices**: Guidelines for cost-effective tool design

### For the Ecosystem

- **Sustainability**: Lower resource usage enables more users
- **Accessibility**: Lower costs make Victor accessible to budget-constrained users
- **Performance**: Faster iteration cycles for development
- **Scalability**: Framework scales to 100+ tools without degradation

## Drawbacks and Alternatives

### Drawbacks

1. **Added Complexity**: More components in tool selection pipeline
   - **Mitigation**: Clear abstractions and documentation
   - **Mitigation**: Sensible defaults, explicit overrides

2. **First-Use Latency**: Lazy loading adds latency on first tool use
   - **Mitigation**: Eager load critical tools (file_reader, etc.)
   - **Mitigation**: Preload frequently used tools after first use

3. **Migration Effort**: Breaking changes to tool registration API
   - **Mitigation**: Clear migration guide and examples
   - **Mitigation**: Deprecation period for old API
   - **Mitigation**: Automated migration script

4. **Potential Incorrectness**: Low-cost tool might not be sufficient
   - **Mitigation**: Progressive escalation handles this
   - **Mitigation**: User can override tool selection
   - **Mitigation**: Feedback loop improves selection over time

### Alternatives Considered

1. **Keep eager loading, optimize tool initialization**
   - **Pros**: No API changes, simpler
   - **Cons**: Still loads unused tools, limited performance gain
   - **Why rejected**: Only addresses startup, not cost efficiency

2. **User manually specifies which tools to load**
   - **Pros**: Full user control, simple implementation
   - **Cons**: Poor UX, users don't know which tools they need
   - **Why rejected**: Places burden on user, not discoverable

3. **ML-based tool selection**
   - **Pros**: Optimal selection based on historical data
   - **Cons**: Complex, requires training data, opaque decisions
   - **Why rejected**: Overkill for initial version, simpler rules work well

4. **All tools lazy, no eager loading option**
   - **Pros**: Maximum startup speed
   - **Cons**: Poor UX for critical tools (file_reader)
   - **Why rejected**: First-use latency unacceptable for core tools

5. **Cost-based selection without progressive parameters**
   - **Pros**: Simpler implementation
   - **Cons**: Low-cost tools might fail repeatedly
   - **Why rejected**: Progressive escalation is key to cost efficiency

## Unresolved Questions

1. **Default Budget**: What should be the default monthly budget if user doesn't specify?
   - **Initial thought**: $10.00 USD (generous but prevents runaway costs)
   - **Discussion needed**: Community input on reasonable defaults

2. **Eager Load List**: Which tools should be eager-loaded by default?
   - **Initial thought**: file_reader, code_editor, git_tool, shell_tool
   - **Discussion needed**: Measure tool usage in production

3. **Escalation Limits**: How many escalation attempts before giving up?
   - **Initial thought**: 3 attempts (initial → 2x → 4x → stop)
   - **Discussion needed**: Balance between persistence and cost

4. **Budget Enforcement**: What happens when budget is exceeded?
   - **Initial thought**: Warning only, allow user to continue
   - **Discussion needed**: Should we block execution or just warn?

5. **Cost Estimation**: How to accurately estimate tool costs?
   - **Initial thought**: Manual declaration by tool author, validated by tests
   - **Discussion needed**: Should we measure actual costs and auto-tune?

## Implementation Plan

### Phase 1: Core Infrastructure (Week 1-2)

- [ ] Implement `CostBasedToolSelector` class
- [ ] Implement `ProgressiveParameterManager` class
- [ ] Implement `BudgetTracker` class
- [ ] Enhance `ProgressiveToolsRegistry` with new metadata
- [ ] Add configuration classes (`ToolSelectionConfig`, `LazyLoadingConfig`)

**Deliverable**: Core selection and budget infrastructure

### Phase 2: Tool Registry Integration (Week 2-3)

- [ ] Update `ToolRegistry` to support lazy loading
- [ ] Update `ToolRegistry` to store cost metadata
- [ ] Modify tool registration API (breaking change)
- [ ] Add migration guide for existing tools
- [ ] Implement backward compatibility layer

**Deliverable**: Enhanced tool registry with lazy loading

### Phase 3: Orchestrator Integration (Week 3-4)

- [ ] Update `AgentOrchestrator` to use lazy loading
- [ ] Integrate `CostBasedToolSelector` into tool selection
- [ ] Integrate `ProgressiveParameterManager` into execution loop
- [ ] Integrate `BudgetTracker` into execution pipeline
- [ ] Add configuration support in settings.py

**Deliverable**: Orchestrator with progressive tool loading

### Phase 4: Tool Migration (Week 4-5)

- [ ] Update all built-in tools with cost metadata
- [ ] Register progressive tools in `ProgressiveToolsRegistry`
- [ ] Add `lazy_load` flags to all tools
- [ ] Update tool documentation with cost information
- [ ] Create tool migration examples

**Deliverable**: All 55 tools migrated to new system

### Phase 5: Testing & Validation (Week 5-6)

- [ ] Unit tests for `CostBasedToolSelector`
- [ ] Unit tests for `ProgressiveParameterManager`
- [ ] Unit tests for `BudgetTracker`
- [ ] Integration tests for lazy loading
- [ ] Performance benchmarks (startup time, cost reduction)
- [ ] Backward compatibility tests

**Deliverable**: Comprehensive test coverage

### Phase 6: Documentation & Examples (Week 6)

- [ ] Update tool authoring guide with cost metadata
- [ ] Document progressive parameter pattern
- [ ] Create budget configuration guide
- [ ] Add examples of custom tools with progressive parameters
- [ ] Update CLAUDE.md with new architecture

**Deliverable**: Complete documentation

### Testing Strategy

**Unit Tests**:
- `CostBasedToolSelector`: Test selection logic with various cost tiers
- `ProgressiveParameterManager`: Test escalation with different factors
- `BudgetTracker`: Test budget enforcement and alerts
- `LazyToolRunnable`: Test lazy initialization and caching

**Integration Tests**:
- End-to-end tool loading with realistic tool sets
- Budget enforcement across sessions
- Progressive escalation in real workflows
- Backward compatibility with old tool registration

**Performance Tests**:
- Startup time measurement (before vs after)
- Memory usage measurement (before vs after)
- Cost tracking accuracy
- Latency of first tool use (lazy loading penalty)

**Backward Compatibility Tests**:
- Old tool registration API still works (with deprecation warning)
- Existing tools without cost metadata work (defaults to MEDIUM)
- Existing tools without progressive params work (no escalation)

### Rollout Plan

**Feature Flags**:
- `VICTOR_LAZY_LOADING=1`: Enable lazy loading (default in 0.6.0)
- `VICTOR_COST_SELECTION=1`: Enable cost-based selection (default in 0.6.0)

**Gradual Rollout**:
1. **0.5.x**: Feature behind flags, early adopters can opt-in
2. **0.6.0**: Features enabled by default, deprecation warnings for old API
3. **0.7.0**: Old API removed, migration complete

**Communication**:
- Blog post announcing progressive tool loading
- Migration guide for tool authors
- Changelog entry with breaking changes
- GitHub discussion for community feedback

## Migration Path

This FEP introduces breaking changes to the tool registration API. A clear migration path is provided.

### For Tool Users

**No immediate action required**. Existing tools will continue to work.

**Optional: Enable features early**:
```bash
# Enable lazy loading and cost selection
export VICTOR_LAZY_LOADING=1
export VICTOR_COST_SELECTION=1

victor chat
```

**Optional: Configure budget**:
```yaml
# ~/.victor/config.yaml
tool_selection:
  prefer_low_cost: true
  monthly_budget_usd: 10.00
```

### For Tool Authors (Custom Tools)

**Step 1: Add cost metadata** (Minimal change)

```python
# Before
from victor.tools.base import BaseTool

class MyTool(BaseTool):
    name = "my_tool"
    description = "Does something"
    parameters = {...}
    execute(): ...
```

```python
# After
from victor.tools.base import BaseTool
from victor.tools.enums import CostTier

class MyTool(BaseTool):
    name = "my_tool"
    description = "Does something"
    parameters = {...}

    # NEW: Cost metadata
    cost_tier = CostTier.LOW
    estimated_cost_usd = 0.001  # Optional

    execute(): ...
```

**Step 2: Add progressive parameters** (Optional, for expensive tools)

```python
# Add progressive configuration
class MyTool(BaseTool):
    # ... existing code ...

    # NEW: Progressive parameters
    progressive_params = {
        "max_results": {
            "initial": 5,
            "max": 100,
            "escalation_factor": 2.0
        }
    }
```

**Step 3: Update registration** (Required, breaking change)

```python
# Before (deprecated)
from victor.tools.registry import ToolRegistry

registry = ToolRegistry()
registry.register(MyTool())
```

```python
# After
from victor.tools.registry import ToolRegistry
from victor.tools.enums import CostTier

registry = ToolRegistry()
registry.register(
    name="my_tool",
    tool_class=MyTool,
    lazy=True,
    cost_tier=CostTier.LOW
)
```

**Step 4: Register progressive parameters** (If Step 2 completed)

```python
from victor.tools.progressive_registry import ProgressiveToolsRegistry

progressive_registry = ProgressiveToolsRegistry.get_instance()
progressive_registry.register(
    tool_name="my_tool",
    progressive_params={
        "max_results": {
            "initial": 5,
            "max": 100,
            "escalation_factor": 2.0
        }
    }
)
```

### Deprecation Timeline

- **Version 0.5.0** (Current): Old API works, feature flags available
- **Version 0.6.0** (Target): New API default, old API deprecated with warnings
- **Version 0.7.0** (Future): Old API removed, migration complete

### Migration Script

Automated migration script provided:

```bash
# Migrate all tools in a directory
python scripts/migrate_tools.py --path victor/my_vertical/tools/

# Dry run (show changes without applying)
python scripts/migrate_tools.py --path victor/my_vertical/tools/ --dry-run

# Interactive mode (confirm each change)
python scripts/migrate_tools.py --path victor/my_vertical/tools/ --interactive
```

## Compatibility

### Backward Compatibility

- **Breaking change**: Yes (tool registration API)
- **Migration required**: Yes (for tool authors)
- **Deprecation period**: 2 minor versions (0.5.x → 0.6.x → 0.7.x)

**Compatibility Layer**:
```python
# Old API still works in 0.6.x with deprecation warning
registry.register(MyTool())  # DeprecationWarning: Use register(name, tool_class) instead

# Automatically migrates to new behavior
# - cost_tier defaults to MEDIUM
# - lazy_load defaults to True
# - progressive_params empty (no escalation)
```

### Version Compatibility

- **Minimum Python version**: No change (3.10+)
- **Minimum dependency versions**: No change
- **Platform compatibility**: No change (all platforms)

### Vertical Compatibility

**Built-in verticals**:
- **Impact**: All 6 verticals need migration
- **Required changes**: Update tool registration, add cost metadata
- **Migration guide**: Provided in Implementation Plan Phase 4
- **Estimated effort**: 2-3 days per vertical

**External verticals**:
- **Impact**: Breaking change, migration required
- **Migration guide**: Full guide provided in this FEP
- **Deprecation period**: 2 minor versions for migration
- **Support**: Migration script and examples provided

**Vertical Migration Checklist**:
- [ ] Add `cost_tier` to all tools
- [ ] Update tool registration calls
- [ ] Add `progressive_params` to expensive tools (optional)
- [ ] Test vertical with lazy loading enabled
- [ ] Update documentation with cost information

## References

- [Related FEP-0001: FEP Process](./fep-0001-fep-process.md)
- [GitHub Issue #156: Slow startup time](https://github.com/anvai-labs/victor/issues/156)
- [GitHub Issue #203: Unexpected API costs](https://github.com/anvai-labs/victor/issues/203)
- [Existing: LazyToolRunnable](../victor/tools/composition/lazy.py)
- [Existing: ProgressiveToolsRegistry](../victor/tools/progressive_registry.py)
- [Existing: CostTier enum](../victor/tools/enums.py)
- [Design: Progressive Tool Loading](../docs/design/progressive-tool-loading.md)
- [Performance: Startup Optimization](../docs/performance/startup-optimization.md)

---

## Review Process

### Submission

- **Submitted by**: Vijaykumar Singh <vijay@anvaiops.com> (@vjsingh1984)
- **Date**: 2025-01-09
- **Pull Request**: (To be created)

### Review Timeline

- **Initial review period**: 14 days minimum (2025-01-09 to 2025-01-23)
- **Reviewers assigned**: (To be assigned)
- **Discussion thread**: [GitHub Discussion #3](https://github.com/anvai-labs/victor/discussions/3)

### Review Checklist

#### Technical Review

- [ ] Specification is clear and complete
- [ ] API design follows Victor conventions
- [ ] Error handling is well-defined
- [ ] Testing strategy is adequate
- [ ] Documentation plan is included
- [ ] Migration path is clear and achievable

#### Community Review

- [ ] Use cases are well-understood
- [ ] Benefits outweigh drawbacks
- [ ] Migration path is clear
- [ ] Alternative approaches were considered
- [ ] Community feedback is addressed

### Decisions

- **Recommendation**: (Pending review)
- **Decision date**: (Pending)
- **Approved by**: (Pending)
- **Rationale**: (Pending)

### Revision History

1. **v1.0** (2025-01-09): Initial submission

---

## Acceptance Criteria

### Must-Have Criteria

1. **[Performance] Startup time reduction of 50%+**
   - Success metric: Startup time < 1 second (down from 2.5s)
   - Verification method: Benchmark startup time with 55+ tools

2. **[Cost] Automatic preference for low-cost tools**
   - Success metric: FREE/LOW tier tools selected when appropriate
   - Verification method: Unit tests for CostBasedToolSelector

3. **[Budget] User-configurable cost budget**
   - Success metric: Users can set monthly budget, enforcement works
   - Verification method: Integration tests for BudgetTracker

4. **[Progressive] Progressive parameter escalation**
   - Success metric: Parameters escalate based on feedback
   - Verification method: Unit tests for ProgressiveParameterManager

5. **[Migration] Clear migration path for existing tools**
   - Success metric: All 55 built-in tools migrated successfully
   - Verification method: Migration script test run

### Should-Have Criteria

1. **[UX] First-use latency < 100ms for eager-loaded tools**
   - Success metric: Critical tools (file_reader) respond immediately
   - Verification method: Latency measurements
   - Priority: High

2. **[Documentation] Complete migration guide**
   - Success metric: Guide covers all breaking changes with examples
   - Verification method: Documentation review
   - Priority: High

3. **[Testing] 80%+ test coverage for new components**
   - Success metric: Coverage report shows >= 80%
   - Verification method: pytest --cov
   - Priority: Medium

### Implementation Requirements

Before this FEP can be marked as "Implemented", the following must be completed:

- [ ] Code implementation following the specification
- [ ] Comprehensive test coverage (>80% for new code)
- [ ] API documentation updated
- [ ] User guide updated (budget configuration, progressive parameters)
- [ ] Migration guide completed (tool authors)
- [ ] Migration script implemented and tested
- [ ] Changelog entry added
- [ ] Release notes prepared
- [ ] Backward compatibility verified (old API with deprecation warnings)
- [ ] Performance benchmarks run (startup time, memory usage, cost reduction)
- [ ] All 55 built-in tools migrated

### Validation Process

1. **Automated validation**: CI checks, test coverage, linting
2. **Manual review**: Code review, documentation review, architecture review
3. **Community testing**: Beta testing period with early adopters
4. **Final approval**: Project maintainers

### Success Metrics

- Startup time: < 1 second (68% reduction from 2.5s)
- Memory usage: < 300MB at startup (33% reduction from 450MB)
- Typical session cost: 60% reduction through cost-based selection
- Migration completeness: 100% of built-in tools migrated
- Test coverage: >= 80% for new code

---

## Copyright

This FEP is licensed under the Apache License 2.0, same as the Victor project.
