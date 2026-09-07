# Victor Architecture

> **Single source of truth** for Victor system architecture.
> Supersedes: `ARCHITECTURE.md`, `docs/architecture/overview.md`, `docs/diagrams/`

**Version**: {{ victor_version }} | **Last Updated**: 2026-06 | **Status**: Canonical

---

## Table of Contents

- [System Overview](#system-overview)
- [Layer Architecture](#layer-architecture)
- [Service Layer](#service-layer)
- [Agent Runtime](#agent-runtime)
- [Provider System](#provider-system)
- [Tool System](#tool-system)
- [Workflow Engine](#workflow-engine)
- [Multi-Agent Teams](#multi-agent-teams)
- [State Management](#state-management)
- [Database Architecture](#database-architecture)
- [Configuration System](#configuration-system)
- [Extension System](#extension-system)
- [Rust Native Extensions](#rust-native-extensions)
- [Integration Points Map](#integration-points-map)

---

## System Overview

Victor is a contract-first agentic AI framework in Python 3.11+ providing a typed,
service-first runtime for building agents that reason, call tools, execute DAG
workflows, and coordinate multi-agent teams across 25 LLM providers.

```mermaid
flowchart TB
    subgraph Clients["CLIENT LAYER"]
        CLI["CLI / TUI"]
        API["HTTP API"]
        MCP["MCP Server"]
        VSC["VS Code"]
    end

    subgraph Framework["FRAMEWORK LAYER"]
        Agent["Agent API"]
        SG["StateGraph"]
        WE["WorkflowEngine"]
        Tools["Tool Registry"]
    end

    subgraph Services["SERVICE LAYER (6 canonical)"]
        CS["ChatService"]
        TS["ToolService"]
        SS["SessionService"]
        CX["ContextService"]
        PS["ProviderService"]
        RS["RecoveryService"]
    end

    subgraph Orchestrator["ORCHESTRATOR (Facade)"]
        ORC["AgentOrchestrator"]
        TP["ToolPipeline"]
        AL["AgenticLoop"]
    end

    subgraph Providers["PROVIDERS (24)"]
        P1["Anthropic"]
        P2["OpenAI"]
        P3["Gemini"]
        P4["Ollama"]
        P5["+ 20 more"]
    end

    subgraph ToolModules["TOOLS (34 modules)"]
        T1["Filesystem"]
        T2["Git"]
        T3["Shell"]
        T4["Web/Search"]
        T5["Analysis"]
    end

    subgraph StorageLayer["STORAGE"]
        GDB["Global DB\n~/.victor/victor.db"]
        PDB["Project DB\n./.victor/project.db"]
    end

    Clients --> Framework
    Framework --> Orchestrator
    Orchestrator --> Services
    Services --> Providers
    Services --> ToolModules
    Orchestrator --> StorageLayer

    style Clients fill:#e0e7ff,stroke:#4f46e5,color:#1e1b4b
    style Framework fill:#dbeafe,stroke:#3b82f6,color:#1e3a5f
    style Services fill:#d1fae5,stroke:#10b981,color:#064e3b
    style Orchestrator fill:#fef3c7,stroke:#f59e0b,color:#78350f
    style Providers fill:#fce7f3,stroke:#ec4899,color:#831843
    style ToolModules fill:#cffafe,stroke:#06b6d4,color:#164e63
    style StorageLayer fill:#f3e8ff,stroke:#a855f7,color:#3b0764
```

### Codebase Scale

| Metric | Value |
|--------|-------|
| Source files | 3,672 |
| Lines of code | 1,166,724 |
| Python packages | 294 |
| Provider adapters | 24 |
| Tool modules | 34 |
| Cargo crates | 5 |

### Layer Rules

| Rule | Description | Guard Test |
|------|-------------|------------|
| Clients use Framework only | UI never imports `victor.agent.*` | `test_architectural_boundaries.py` |
| Framework delegates to Runtime | `Agent.create()` goes through `AgentFactory` | Agent entry point |
| Runtime delegates to Services | Orchestrator is facade, services own logic | `test_service_layer_validation.py` |
| Services own infrastructure | Effectful behavior via `ExecutionContext.services` | Service accessor |
| External uses Contracts only | Verticals import `victor_contracts` | `test_core_vertical_import_boundary.py` |

### Data Flow

```mermaid
sequenceDiagram
    participant U as User
    participant C as Client (CLI/API/MCP)
    participant F as Framework Agent
    participant O as AgentOrchestrator
    participant S as Service Layer
    participant P as Provider (LLM)
    participant T as ToolPipeline

    U->>C: Prompt
    C->>F: agent.chat(prompt)
    F->>O: AgentFactory.create()
    O->>S: ChatService.process()
    S->>P: Provider.chat(messages)
    P-->>S: Response + tool_calls
    S->>T: Execute tools
    T-->>S: Tool results
    S->>P: Continue with results
    P-->>S: Final response
    S-->>O: Formatted result
    O-->>F: Agent response
    F-->>C: Stream events
    C-->>U: Output
```

---

## Layer Architecture

Victor follows a strict layered design. Each layer only depends on the layer
directly below it.

```mermaid
flowchart LR
    subgraph L1["L1: Client Surface"]
        direction TB
        C1["CLI (Typer)"]
        C2["TUI (Textual)"]
        C3["HTTP API (FastAPI)"]
        C4["MCP Server"]
        C5["VS Code Extension"]
    end

    subgraph L2["L2: Framework API"]
        direction TB
        F1["Agent"]
        F2["StateGraph"]
        F3["WorkflowEngine"]
        F4["Tool Registry"]
        F5["Skills"]
    end

    subgraph L3["L3: Runtime"]
        direction TB
        R1["AgentOrchestrator"]
        R2["Service Layer"]
        R3["AgenticLoop"]
        R4["ExecutionContext"]
    end

    subgraph L4["L4: Infrastructure"]
        direction TB
        I1["Providers (24)"]
        I2["Tools (34)"]
        I3["State Mgmt"]
        I4["Database"]
        I5["Config"]
    end

    L1 --> L2 --> L3 --> L4

    style L1 fill:#e0e7ff,stroke:#4f46e5,color:#1e1b4b
    style L2 fill:#dbeafe,stroke:#3b82f6,color:#1e3a5f
    style L3 fill:#d1fae5,stroke:#10b981,color:#064e3b
    style L4 fill:#fef3c7,stroke:#f59e0b,color:#78350f
```

| Layer | Module | Entry Point | Responsibility |
|-------|--------|-------------|----------------|
| **Client** | `victor/ui/` | `cli.py` | CLI, TUI, commands |
| **Client** | `victor/integrations/api/` | `server.py` | FastAPI REST server |
| **Client** | `victor/integrations/mcp/` | — | MCP protocol bridge |
| **Framework** | `victor/framework/` | `agent.py` | Public API surface |
| **Runtime** | `victor/agent/` | `orchestrator.py` | Orchestration facade |
| **Runtime** | `victor/agent/services/` | `chat_service.py` | 6 canonical services |
| **Infrastructure** | `victor/providers/` | `base.py` | LLM provider adapters |
| **Infrastructure** | `victor/tools/` | `base.py` | Tool modules |
| **Infrastructure** | `victor/state/` | `__init__.py` | 4-scope state management |
| **Infrastructure** | `victor/config/` | `settings.py` | Settings and profiles |
| **Infrastructure** | `victor/core/` | `database.py` | Event sourcing, CQRS, DI |

---

## Service Layer

The runtime is **service-first**. Six canonical services own all effectful
behavior. The orchestrator is a facade that delegates to these services.

```mermaid
flowchart TB
    ORC["AgentOrchestrator\nFacade only"]

    ORC --> CS["ChatService\nvictor/agent/services/chat_service.py\nOwns: chat loop, streaming, perception"]
    ORC --> TS["ToolService\nvictor/agent/services/tool_service.py\nOwns: tool registration, execution"]
    ORC --> SS["SessionService\nvictor/agent/services/session_service.py\nOwns: session lifecycle"]
    ORC --> CX["ContextService\nvictor/agent/services/context_service.py\nOwns: context assembly, pruning"]
    ORC --> PS["ProviderService\nvictor/agent/services/provider_service.py\nOwns: provider init, switching"]
    ORC --> RS["RecoveryService\nvictor/agent/services/recovery_service.py\nOwns: error recovery, retry"]

    style ORC fill:#fef3c7,stroke:#f59e0b,color:#78350f
    style CS fill:#d1fae5,stroke:#10b981,color:#064e3b
    style TS fill:#d1fae5,stroke:#10b981,color:#064e3b
    style SS fill:#d1fae5,stroke:#10b981,color:#064e3b
    style CX fill:#d1fae5,stroke:#10b981,color:#064e3b
    style PS fill:#d1fae5,stroke:#10b981,color:#064e3b
    style RS fill:#d1fae5,stroke:#10b981,color:#064e3b
```

**Access pattern** via `ExecutionContext`:

```python
from victor.runtime.context import ExecutionContext

ctx = ExecutionContext(settings=settings)
chat_svc = ctx.services.chat       # ChatService
tool_svc = ctx.services.tool       # ToolService
session_svc = ctx.services.session # SessionService
```

> **UI layer** must use `VictorClient` + `SessionConfig` — never import
> `AgentOrchestrator` or `AgentFactory` directly.

---

## Agent Runtime

### AgenticLoop

The `AgenticLoop` (`victor/framework/agentic_loop.py`) is the canonical execution
authority for chat. It runs: **PERCEIVE → PLAN → ACT → EVALUATE → DECIDE**.

```mermaid
flowchart TB
    START([AgenticLoop.run]) --> PERCEIVE["PERCEIVE\nPerceptionIntegration\nvictor/framework/perception_integration.py"]
    PERCEIVE --> PLAN["PLAN\nTaskAnalyzer\nvictor/agent/task_analyzer.py"]
    PLAN --> ACT["ACT\nAgenticLoop + TurnExecutor\nvictor/framework/agentic_loop.py"]
    ACT --> EVALUATE["EVALUATE\nEvaluationNode\nvictor/framework/evaluation_nodes.py"]
    EVALUATE --> DECIDE{"DECIDE\nFulfillmentDetector\nvictor/framework/fulfillment.py"}
    DECIDE -->|Complete| DONE([Return Result])
    DECIDE -->|Continue| PERCEIVE
    DECIDE -->|Retry| ACT

    style PERCEIVE fill:#dbeafe,stroke:#3b82f6
    style PLAN fill:#e0e7ff,stroke:#6366f1
    style ACT fill:#d1fae5,stroke:#10b981
    style EVALUATE fill:#fef3c7,stroke:#f59e0b
    style DECIDE fill:#fce7f3,stroke:#ec4899
```

**Entry point**: `TurnExecutor.execute_agentic_loop()` at
`victor/agent/services/turn_execution_runtime.py`.

### AgentFactory

`AgentFactory` (`victor/framework/agent_factory.py`) is the **single authority**
for all agent creation paths (CLI, API, `Agent.create()`). It validates config,
bootstraps the DI container, creates the orchestrator, and wires observability.

---

## Provider System

25 LLM provider adapters behind a unified interface with circuit breaker,
retry, and smart routing (multi-provider selection/fallback via
`victor/providers/smart_router.py`). The consolidated provider gateway
feature layer — semantic caching, budget guardrails, routing-performance
work — is **planned**, not shipped
([ADR-022](architecture/adr/022-provider-gateway-feature-layer.md), TD-24;
see the [roadmap](roadmap.md)).

```mermaid
flowchart TB
    subgraph Interface["Provider Interface"]
        BP["BaseProvider\nvictor/providers/base.py"]
    end

    subgraph Cloud["Cloud Providers"]
        AN["Anthropic\nanthropic_provider.py"]
        OA["OpenAI\nopenai_provider.py"]
        GG["Google Gemini\ngoogle_provider.py"]
        DS["DeepSeek"]
        BE["Bedrock"]
    end

    subgraph Local["Local Providers"]
        OL["Ollama\nollama_provider.py"]
        LM["LM Studio"]
        VL["vLLM"]
        ML["MLX"]
    end

    subgraph Resilience["Resilience Layer"]
        CB["Circuit Breaker"]
        RT["Retry Logic"]
        SR["Smart Routing"]
    end

    BP --> Cloud
    BP --> Local
    Cloud --> Resilience
    Local --> Resilience

    style Interface fill:#6366f1,color:#fff
    style Cloud fill:#dbeafe,stroke:#3b82f6
    style Local fill:#d1fae5,stroke:#10b981
    style Resilience fill:#fef3c7,stroke:#f59e0b
```

### Caching Architecture

Two independent caching capabilities per provider:

| Capability | Method | Cloud | Local |
|---|---|---|---|
| **API prompt caching** | `supports_prompt_caching()` | Billing discount | N/A |
| **KV prefix caching** | `supports_kv_prefix_caching()` | Stable prefix | Stable prefix |

**KV optimizations** (active for Ollama, LMStudio, vLLM, MLX):

- System prompt frozen after first build
- Tools sorted by name for prefix matching
- Dynamic content injected into user messages
- `Agent.warm_up()` primes KV cache

---

## Tool System

34 tool modules across 12 categories with semantic selection and budget enforcement.

```mermaid
flowchart TB
    subgraph Registry["Tool Registry\nvictor/framework/tools.py"]
        TR["ToolRegistrar\nvictor/agent/tool_registrar.py"]
    end

    subgraph Selection["Tool Selection"]
        KW["Keyword Match"]
        SM["Semantic (Embedding)"]
        HY["Hybrid 70/30"]
    end

    subgraph Categories["Tool Categories"]
        FS["Filesystem\nvictor/tools/filesystem/"]
        GT["Git\nvictor/tools/git/"]
        SH["Shell\nvictor/tools/shell/"]
        WB["Web/Search\nvictor/tools/web/"]
        AN["Analysis\nvictor/tools/analysis/"]
        DB["Database\nvictor/tools/database/"]
        DK["Docker\nvictor/tools/docker/"]
        TG["Testing\nvictor/tools/testing/"]
        RF["Refactoring\nvictor/tools/refactoring/"]
    end

    subgraph Pipeline["ToolPipeline\nvictor/agent/tool_pipeline.py"]
        VAL["Validation"]
        SEL["Selection"]
        EXE["Execution"]
        BUD["Budget Check"]
    end

    Registry --> Selection
    Selection --> Categories
    Registry --> Pipeline

    style Registry fill:#6366f1,color:#fff
    style Pipeline fill:#f59e0b,color:#fff
    style Categories fill:#cffafe,stroke:#06b6d4
```

### Tool Presets

| Preset | Description |
|--------|-------------|
| `default()` | Standard production set |
| `minimal()` | Read-only, safe operations |
| `full()` | All available tools |
| `airgapped()` | Local-only, no network |

---

## Workflow Engine

YAML-to-StateGraph compiler with typed state, conditional edges, checkpointing,
and human-in-the-loop.

```mermaid
flowchart LR
    YAML["YAML DSL\nvictor/workflows/"] --> COMP["UnifiedCompiler\nvictor/workflows/unified_compiler.py"]
    COMP --> SG["StateGraph\nvictor/framework/graph.py"]
    SG --> EXEC["WorkflowExecutor\nvictor/framework/workflow_engine.py"]

    subgraph NodeTypes["Node Types"]
        AGENT["Agent Node"]
        COMPUTE["Compute Node"]
        HANDLER["Handler Node"]
        PASSTHROUGH["Passthrough Node"]
    end

    SG --> NodeTypes

    style YAML fill:#fce7f3,stroke:#ec4899
    style COMP fill:#8b5cf6,color:#fff
    style SG fill:#10b981,color:#fff
    style EXEC fill:#3b82f6,color:#fff
```

### StateGraph Features

- **Typed state** — `TypedDict` state schemas
- **Conditional edges** — Route based on state values
- **Cyclic graphs** — Loopback edges for iteration
- **Checkpointing** — Persist and resume state
- **Copy-on-write** — Efficient state mutations
- **Human-in-the-loop** — Interrupt for approval

---

## Multi-Agent Teams

Teams are **formations** (coordination patterns), not separate graphs.
`StateGraph` is always the execution engine.

```mermaid
flowchart TB
    subgraph UTC["UnifiedTeamCoordinator\nvictor/teams/"]
        FM["Team Formation"]
    end

    FM --> SEQ["SEQUENTIAL\nChain of agents"]
    FM --> PAR["PARALLEL\nConcurrent agents"]
    FM --> HIER["HIERARCHICAL\nManager + workers"]
    FM --> PIPE["PIPELINE\nStage-based"]

    subgraph SG["StateGraph (always the engine)"]
        N1["Team as Node"]
    end

    UTC -->|"Direct usage as node"| SG

    style UTC fill:#6366f1,color:#fff
    style SG fill:#10b981,color:#fff
    style SEQ fill:#dbeafe,stroke:#3b82f6
    style PAR fill:#d1fae5,stroke:#10b981
    style HIER fill:#fef3c7,stroke:#f59e0b
    style PIPE fill:#fce7f3,stroke:#ec4899
```

### Correct Usage

```python
from victor.framework import StateGraph
from victor.teams import UnifiedTeamCoordinator, TeamFormation

coordinator = UnifiedTeamCoordinator(orchestrator)
coordinator.set_formation(TeamFormation.PARALLEL)
coordinator.add_member(agent1).add_member(agent2)

graph = StateGraph(AgentState)
graph.add_node("research_team", coordinator)  # Direct usage!
```

> **Do not** create wrapper nodes for each formation or separate "multi-agent graph" types.

---

## State Management

Unified state management across 4 scopes with the `GlobalStateManager` facade
providing a single entry point with copy-on-write optimization.

```mermaid
flowchart TB
    GSM["GlobalStateManager\nvictor/state/"]

    GSM --> WF["WORKFLOW Scope\nPer-workflow execution state"]
    GSM --> CONV["CONVERSATION Scope\nPer-conversation context"]
    GSM --> TM["TEAM Scope\nMulti-agent shared state"]
    GSM --> GLB["GLOBAL Scope\nCross-session persistent"]

    style GSM fill:#6366f1,color:#fff
    style WF fill:#dbeafe,stroke:#3b82f6
    style CONV fill:#d1fae5,stroke:#10b981
    style TM fill:#fef3c7,stroke:#f59e0b
    style GLB fill:#fce7f3,stroke:#ec4899
```

---

## Governance, Isolation & Cost

Cross-cutting runtime subsystems layered over tool execution and the provider path
(see [Features](features.md) for the user-facing summary):

- **Policy engine** (`victor/framework/policies/`) — evaluates **ALLOW / DENY / ASK** verdicts over
  tool calls across REQUEST and RESPONSE phases (streaming and non-streaming). ASK routes to a
  container-registered approval handler. Gated by `USE_POLICY_ENGINE` + `governance.enabled`.
- **Sandbox isolation** (`victor/tools/sandbox/`) — wraps subprocess/code-execution tools in an OS
  sandbox (bwrap on Linux, seatbelt on macOS), gated by `settings.sandbox.sandbox_enabled`
  (off by default, fail-open).
- **Cost co-design** — the dominant cost term (provider round-trips × context size) is measured and
  acted on: per-turn cost trace (**C0**, surfaced in the chat UI footer), reference-aware
  tool-result pruning (**L1**), per-task prompt-recompute caching (**L2**), and cost/latency-aware
  routing (**L4**, with `USE_SMART_ROUTING`).

## Additional Subsystems

Live packages under `victor/` that support the runtime but sit outside the core layer diagram
above:

| Package | Purpose |
|---------|---------|
| `victor/coordination/` | Multi-agent coordination — formation strategies for team execution. |
| `victor/classification/` | Unified task-type + complexity detection (consolidated pattern matching). |
| `victor/optimization/` | Workflow optimization algorithms (automated workflow tuning). |
| `victor/experiments/` | MLflow-like experiment tracking for workflow optimization. |
| `victor/analytics/` | Backward-compat namespace routing to `victor/observability/analytics/`. |
| `victor/benchmark/` | Benchmark vertical — high-level API for AI coding evaluations. |
| `victor/iac/` | IaC security scanner (Infrastructure-as-Code file scanning). |
| `victor/native/` | Re-exports of Rust/native processing hot paths (`victor/processing/native/`), with Python fallback. |

## Database Architecture

Victor uses a canonical two-database architecture (schema v7).

```mermaid
flowchart LR
    subgraph Global["Global DB\n~/.victor/victor.db"]
        GS["Settings & API keys"]
        RL["RL learning data"]
        TS["Team stats"]
        TP["TUI persistence"]
    end

    subgraph Project["Project DB\n./.victor/project.db"]
        GN["Graph nodes/edges"]
        CN["Conversations"]
        PS["Project sessions"]
        CT["Change tracking"]
    end

    style Global fill:#f3e8ff,stroke:#a855f7
    style Project fill:#e0e7ff,stroke:#4f46e5
```

| Database | Path | Contents |
|----------|------|----------|
| **Global** | `~/.victor/victor.db` | Settings, API keys, RL data, team stats, TUI sessions |
| **Project** | `./.victor/project.db` | Graph, conversations, sessions, cache |
| **Undo** | `./.victor/undo.db` | File-edit undo/redo history (change groups + file changes) |

**Access pattern:**

```python
from victor.core.database import get_database, get_project_database
from victor.core.undo_database import get_undo_database

global_db = get_database()    # ~/.victor/victor.db
project_db = get_project_database()  # ./.victor/project.db
undo_db = get_undo_database()  # ./.victor/undo.db
```

**Why undo.db is separate:** `project.db` is written continuously by the graph
indexer (reindex-on-save). SQLite serializes writers even under WAL, so the
tiny per-edit undo write kept losing the write-lock and failing with
`database is locked` — silently dropping undo history. A dedicated `undo.db`
gives the undo writer its own lock (never contends with the indexer) and lets
multiple sessions editing the same project record history concurrently. Undo
history is rebuildable/ephemeral; durable rollback is covered by file backups
in `.victor/backups/`.

**Direction — correlated graph + vector backend (partially shipped behind a per-repo flag):** the default Code Context Graph
(SQLite `graph_*`) and the LanceDB embedding index are hand-joined today
(`graph_node.embedding_ref` is unpopulated). The Proxima backend now writes one authoritative
ProximaRecord containing complete node properties + vector + staleness markers under a single
`oid`, with ORION as a rebuildable traversal projection; its local Tier-A/Tier-B boundary is also
implemented. SQLite remains default pending a buildable native wheel and live GA evidence. This is tracked as TD-11/TD-12/TD-13 on the
[roadmap](roadmap.md) — design in
[ProximaDB as the CCG Backend](architecture/proximadb-codegraph-backend.md).

---

## Configuration System

Settings cascade: `.env` → `~/.victor/profiles.yaml` → CLI flags.
Runtime overrides via immutable `SessionConfig`.

```mermaid
flowchart LR
    ENV[".env"] --> PROFILE["profiles.yaml"]
    PROFILE --> CLI["CLI Flags"]
    CLI --> SC["SessionConfig\n(immutable)"]
    SC --> SETTINGS["Settings\nvictor/config/settings.py"]

    style ENV fill:#d1fae5,stroke:#10b981
    style PROFILE fill:#dbeafe,stroke:#3b82f6
    style CLI fill:#fef3c7,stroke:#f59e0b
    style SC fill:#6366f1,color:#fff
    style SETTINGS fill:#f3e8ff,stroke:#a855f7
```

**Key config groups** (26+ nested groups in `victor/config/settings.py`):

| Group | Purpose |
|-------|---------|
| `ProviderSettings` | LLM provider configuration |
| `ToolSettings` | Tool registration and budgets |
| `SearchSettings` | Code search configuration |
| `ResilienceSettings` | Retry and circuit breaker |
| `SecuritySettings` | Safety and access control |
| `EventSettings` | Event sourcing configuration |
| `PipelineSettings` | Middleware pipeline |
| `PromptOptimizationSettings` | Runtime prompt evolution |

---

## Extension System

Three orthogonal integration mechanisms:

```mermaid
flowchart TB
    subgraph Plugin["Plugin (Bootstrap)"]
        VP["VictorPlugin\nvictor.plugins entry point"]
    end

    subgraph Vertical["Vertical (Config Template)"]
        VB["VerticalBase\nvictor_contracts.verticals.protocols"]
    end

    subgraph Extension["Extension (Runtime)"]
        VE["VerticalExtensions\nvictor_contracts.verticals.extensions"]
    end

    VP -->|"register(context)"| VB
    VB -->|"get_extensions()"| VE

    style Plugin fill:#6366f1,color:#fff
    style Vertical fill:#10b981,color:#fff
    style Extension fill:#f59e0b,color:#fff
```

| Concept | Role | SDK Type | Lifecycle |
|---------|------|----------|-----------|
| **Plugin** | Bootstrap registrar | `VictorPlugin` | Transient: `register()` called once |
| **Vertical** | Configuration template | `VerticalBase` | Class-level: classmethods called |
| **Extension** | Runtime service | `VerticalExtensions` | Object-level: lazy-loaded |

**External packages** should import from:
- `victor_contracts` — Protocol/contract definitions
- `victor.framework.extensions` — Extension surfaces
- Never import `victor.agent.*` from external packages

### Extension Points

| Extension | How | Location |
|-----------|-----|----------|
| **Providers** | `BaseProvider` subclass | `victor/providers/` |
| **Tools** | `BaseTool` subclass | `victor/tools/` |
| **Workflows** | YAML DSL or StateGraph | `victor/workflows/` |
| **Middleware** | Pre/post hooks | `victor/agent/tool_pipeline.py` |

---

## Rust Native Extensions

Optional PyO3 extensions in `rust/` for performance-critical hot paths.

```mermaid
flowchart TB
    subgraph Crates["5 Cargo Crates"]
        P["protocol\nPortable types"]
        S["state\nConversation/shared state"]
        T["tools\nRegistry"]
        E["edge-runtime\nStandalone binary"]
        B["python-bindings\ncdylib"]
    end

    subgraph HotPaths["Hot Paths"]
        TK["Tokenizer"]
        VS["Vector similarity"]
        CF["Context fitting"]
    end

    B --> HotPaths

    style Crates fill:#6366f1,color:#fff
    style HotPaths fill:#f59e0b,color:#fff
```

**Build**: `cd rust && maturin develop --release`

**Fallback pattern**: Every native path uses `_NATIVE_AVAILABLE` with graceful
Python fallback when Rust extensions are absent.

---

## Integration Points Map

Complete map of how all packages connect:

```mermaid
flowchart TB
    subgraph External["External Packages"]
        VC["victor-coding"]
        VD["victor-devops"]
        VR["victor-rag"]
        VA["victor-dataanalysis"]
        VRE["victor-research"]
        VI["victor-invest"]
        VRG["victor-registry"]
    end

    subgraph Contracts["victor-contracts"]
        CT["Protocols & Types"]
    end

    subgraph Core["victor (Core)"]
        FW["victor/framework/"]
        AG["victor/agent/"]
        PR["victor/providers/"]
        TL["victor/tools/"]
        TM["victor/teams/"]
        ST["victor/state/"]
        CF["victor/config/"]
        CR["victor/core/"]
    end

    External -->|"imports only"| Contracts
    Contracts -->|"imported by"| Core
    FW --> AG
    AG --> PR
    AG --> TL
    AG --> TM
    AG --> ST
    AG --> CF
    AG --> CR

    style External fill:#fce7f3,stroke:#ec4899
    style Contracts fill:#d1fae5,stroke:#10b981
    style Core fill:#e0e7ff,stroke:#4f46e5
```

### Key Entry Points Summary

| Component | Path | Role |
|-----------|------|------|
| `Agent` | `victor/framework/agent.py` | Public API — `run()`, `stream()`, `chat()` |
| `StateGraph` | `victor/framework/graph.py` | DAG workflow engine |
| `AgentOrchestrator` | `victor/agent/orchestrator.py` | Central facade |
| `ChatService` | `victor/agent/services/chat_service.py` | Primary chat entry |
| `ToolService` | `victor/agent/services/tool_service.py` | Tool registration/execution |
| `AgentFactory` | `victor/framework/agent_factory.py` | Single authority for agent creation |
| `VictorAPIServer` | `victor/integrations/api/server.py` | FastAPI REST endpoint |
| `VictorClient` | `victor/framework/client.py` | UI layer entry point |
