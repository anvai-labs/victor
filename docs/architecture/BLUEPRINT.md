# Victor — Master Blueprint

> **The starting point for understanding Victor.** A single-page, top-down and
> bottom-up orientation map of the entire codebase — grounded in real modules
> and integration points. All diagrams are Mermaid and render inline.

**Version**: {{ victor_version }} | **Status**: Canonical orientation document

---

## Table of Contents

- [What Victor Is](#what-victor-is)
- [Top-Down View — Layer Architecture](#top-down-view--layer-architecture)
- [Request Flow — A Single Chat Turn](#request-flow--a-single-chat-turn)
- [Bottom-Up View — Package & Module Map](#bottom-up-view--package--module-map)
- [The Six Canonical Services](#the-six-canonical-services)
- [Extension Surfaces](#extension-surfaces)
- [Data & State Model](#data--state-model)
- [Reading Order — Suggested Path](#reading-order--suggested-path)
- [Glossary](#glossary)

---

## What Victor Is

Victor (`victor-ai`) is a **contract-first, service-first agentic AI framework**
in Python 3.11+ for building autonomous agents that reason, call tools, execute
DAG workflows, and coordinate multi-agent teams across **24 LLM providers**.

Three packages compose the ecosystem:

| Package | Role | Direction |
|---------|------|-----------|
| `victor` (this repo) | Core runtime + framework | Depends on contracts |
| `victor-contracts` | Stable SDK protocols/types | Imported by everyone |
| `victor-coding` … `victor-invest` | External domain verticals | Import contracts only |

**In one line:**
`User → Client → Agent API → AgentOrchestrator (facade) → 6 Services → Providers + Tools → Storage`.

> Authoritative depth lives in the [System Architecture](../architecture.md).
> This blueprint is the map; that document is the territory.

---

## Top-Down View — Layer Architecture

Victor is a strict four-layer stack. Each layer only depends on the layer below.

```mermaid
flowchart TB
    subgraph L1["L1 — Client Surface"]
        C1["CLI / TUI<br/>victor/ui/"]
        C2["HTTP API<br/>victor/integrations/api/server.py"]
        C3["MCP Server<br/>victor/integrations/mcp/"]
        C4["VS Code<br/>vscode-victor/"]
    end
    subgraph L2["L2 — Framework API"]
        F1["Agent<br/>victor/framework/agent.py"]
        F2["StateGraph<br/>victor/framework/graph.py"]
        F3["WorkflowEngine<br/>victor/framework/workflows/"]
        F4["Tool Registry<br/>victor/framework/tools.py"]
    end
    subgraph L3["L3 — Runtime"]
        R1["AgentOrchestrator Facade<br/>victor/agent/orchestrator.py"]
        R2["6 Canonical Services<br/>victor/agent/services/"]
        R3["AgenticLoop<br/>victor/framework/agentic_loop.py"]
        R4["ExecutionContext<br/>victor/runtime/context.py"]
    end
    subgraph L4["L4 — Infrastructure"]
        I1["Providers x 24<br/>victor/providers/"]
        I2["Tools x 34 modules<br/>victor/tools/"]
        I3["Teams / State<br/>victor/teams/ · victor/state/"]
        I4["Database<br/>victor/core/database.py"]
        I5["Config<br/>victor/config/settings.py"]
    end
    subgraph CT["victor-contracts (SDK)"]
        CT1["Protocols &amp; Types<br/>victor_contracts/"]
    end
    L1 -->|"VictorClient + SessionConfig"| L2
    L2 -->|"AgentFactory.create()"| L3
    L3 -->|"ExecutionContext.services"| L4
    CT -.->|"imported by"| L2
    CT -.->|"imported by"| L4
    style L1 fill:#e0e7ff,stroke:#4f46e5,color:#1e1b4b
    style L2 fill:#dbeafe,stroke:#3b82f6,color:#1e3a5f
    style L3 fill:#d1fae5,stroke:#10b981,color:#064e3b
    style L4 fill:#fef3c7,stroke:#f59e0b,color:#78350f
    style CT fill:#f3e8ff,stroke:#a855f7,color:#3b0764
```

**Layer rules (enforced by CI):**

| Rule | Guard test |
|------|-----------|
| L1 uses L2 only — never imports `victor.agent.*` | `tests/unit/framework/test_architectural_boundaries.py` |
| L2 delegates creation to `AgentFactory` | `victor/framework/agent_factory.py` |
| L3 orchestrator is a facade — services own logic | `tests/unit/runtime/test_service_layer_validation.py` |
| External verticals import `victor_contracts` only | `tests/unit/sdk/test_core_vertical_import_boundary.py` |

---

## Request Flow — A Single Chat Turn

How one prompt travels through the system — the most important diagram to
internalize first.

```mermaid
sequenceDiagram
    autonumber
    actor U as User
    participant C as Client (CLI/API/MCP)
    participant A as Agent (Framework)
    participant F as AgentFactory
    participant O as AgentOrchestrator
    participant CS as ChatService
    participant AL as AgenticLoop
    participant PS as ProviderService to LLM
    participant TS as ToolService to ToolPipeline

    U->>C: prompt
    C->>A: VictorClient.chat(prompt)
    A->>F: Agent.create(session_config)
    F->>O: bootstrap DI container + services
    O-->>A: orchestrator ready
    A->>CS: ChatService.process(messages)
    CS->>AL: AgenticLoop.run()
    loop PERCEIVE then PLAN then ACT then EVALUATE then DECIDE
        AL->>PS: provider.chat(messages, tools)
        PS-->>AL: response + tool_calls
        AL->>TS: ToolPipeline.execute(tool_calls)
        TS-->>AL: tool results
        AL->>AL: FulfillmentDetector.decide()
    end
    AL-->>CS: final result
    CS-->>A: Agent response
    A-->>C: stream events
    C-->>U: rendered output
```

The `AgenticLoop` is the **canonical execution authority for chat** — it runs
unconditionally. Each cycle:

```mermaid
flowchart LR
    P([PERCEIVE]) --> PL([PLAN])
    PL --> AC([ACT<br/>AgenticLoop + TurnExecutor])
    AC --> EV([EVALUATE<br/>EvaluationNode])
    EV --> D{DECIDE<br/>FulfillmentDetector}
    D -->|complete| R([Return Result])
    D -->|continue| P
    D -->|retry| AC

    style P fill:#dbeafe,stroke:#3b82f6
    style PL fill:#e0e7ff,stroke:#6366f1
    style AC fill:#d1fae5,stroke:#10b981
    style EV fill:#fef3c7,stroke:#f59e0b
    style D fill:#fce7f3,stroke:#ec4899
    style R fill:#e0e7ff,stroke:#4f46e5
```

| Phase | Owner module |
|-------|-------------|
| PERCEIVE | `victor/framework/perception_integration.py` |
| PLAN | `victor/agent/task_analyzer.py` |
| ACT | `victor/framework/agentic_loop.py` |
| EVALUATE | `victor/framework/evaluation_nodes.py` |
| DECIDE | `victor/framework/fulfillment.py` |
| **Entry point** | `TurnExecutor.execute_agentic_loop()` — `victor/agent/services/turn_execution_runtime.py` |

---

## Bottom-Up View — Package & Module Map

Read this to find where any concern lives in the source tree.

```mermaid
flowchart TB
    subgraph FW["victor/framework/ — Public API"]
        agent["agent.py — run/stream/chat/run_workflow/run_team"]
        graphf["graph.py — StateGraph (DAG engine)"]
        agloop["agentic_loop.py — AgenticLoop"]
        af["agent_factory.py — AgentFactory (single authority)"]
        rl["rl/ — prompt optimization (GEPA, MiPROv2, CoT)"]
        extf["extensions.py — extension surfaces"]
    end
    subgraph AG["victor/agent/ — Runtime"]
        orc["orchestrator.py — Facade"]
        svcs["services/ — 6 canonical + runtime helpers"]
        conv["conversation/ — store, scoring, assembler"]
        coord["coordinators/ — extracted coordinators"]
        rt["runtime/ — ExecutionContext, boundary modules"]
    end
    subgraph INF["victor/ — Infrastructure"]
        prov["providers/ — 24 BaseProvider subclasses"]
        tools["tools/ — 34 modules · BaseTool"]
        teams["teams/ — UnifiedTeamCoordinator, formations"]
        statef["state/ — GlobalStateManager (4 scopes)"]
        config["config/settings.py — 26+ config groups"]
        core["core/ — DI container, database, CQRS, events"]
    end
    subgraph CT2["victor-contracts/ — SDK"]
        ct["verticals/protocols/base.py — VerticalBase"]
        plugins["core/plugins.py — VictorPlugin"]
        skills["SkillDefinition · SkillProvider"]
    end
    subgraph EXT["External Verticals (separate repos)"]
        ext1["victor-coding"]
        ext2["victor-devops"]
        ext3["victor-rag"]
        ext4["victor-dataanalysis"]
        ext5["victor-research"]
        ext6["victor-invest"]
    end
    FW --> AG
    AG --> INF
    CT2 -.->|"imported by"| FW
    CT2 -.->|"imported by"| AG
    EXT -->|"imports only"| CT2
    style FW fill:#dbeafe,stroke:#3b82f6,color:#1e3a5f
    style AG fill:#d1fae5,stroke:#10b981,color:#064e3b
    style INF fill:#fef3c7,stroke:#f59e0b,color:#78350f
    style CT2 fill:#f3e8ff,stroke:#a855f7,color:#3b0764
    style EXT fill:#fce7f3,stroke:#ec4899,color:#831843
```

### Key entry points

| Component | Path | Role |
|-----------|------|------|
| `Agent` | `victor/framework/agent.py` | Public API — `run()`, `stream()`, `chat()` |
| `StateGraph` | `victor/framework/graph.py` | DAG workflow engine |
| `AgentFactory` | `victor/framework/agent_factory.py` | Single authority for agent creation |
| `VictorClient` | `victor/framework/client.py` | UI layer entry point |
| `AgentOrchestrator` | `victor/agent/orchestrator.py` | Central facade (delegates, no logic) |
| `ChatService` | `victor/agent/services/chat_service.py` | Primary chat entry |
| `ToolService` | `victor/agent/services/tool_service.py` | Tool registration/execution |
| `BaseProvider` | `victor/providers/base.py` | Abstract base for all 24 providers |
| `BaseTool` | `victor/tools/base.py` | Foundation for all 55+ tools |
| `VerticalBase` | `victor_contracts/verticals/protocols/base.py` | Core abstraction for 101+ plugins |
| `VictorAPIServer` | `victor/integrations/api/server.py` | FastAPI REST endpoint |

---

## The Six Canonical Services

The orchestrator is a **facade** — all effectful behavior is owned by exactly
six mandatory services (no feature flags, no fallbacks).

```mermaid
flowchart TB
    O["AgentOrchestrator<br/>(facade — no logic)"]
    O --> CS["ChatService<br/>Primary chat authority"]
    O --> TS["ToolService<br/>Register · select · execute"]
    O --> SS["SessionService<br/>Sessions · persistence"]
    O --> CX["ContextService<br/>Context assembly · scoring"]
    O --> PS["ProviderService<br/>Provider lifecycle · routing"]
    O --> RS["RecoveryService<br/>Error recovery · retry"]
    CS --> SvcLayer["ExecutionContext.services<br/>(lazy accessor)"]
    TS --> SvcLayer
    SS --> SvcLayer
    CX --> SvcLayer
    PS --> SvcLayer
    RS --> SvcLayer
    style O fill:#fef3c7,stroke:#f59e0b,color:#78350f
    style CS fill:#d1fae5,stroke:#10b981,color:#064e3b
    style TS fill:#d1fae5,stroke:#10b981,color:#064e3b
    style SS fill:#d1fae5,stroke:#10b981,color:#064e3b
    style CX fill:#d1fae5,stroke:#10b981,color:#064e3b
    style PS fill:#d1fae5,stroke:#10b981,color:#064e3b
    style RS fill:#d1fae5,stroke:#10b981,color:#064e3b
    style SvcLayer fill:#e0e7ff,stroke:#4f46e5,color:#1e1b4b
```

| Service | Module | Owns |
|---------|--------|------|
| `ChatService` | `victor/agent/services/chat_service.py` | Turn execution, AgenticLoop wiring |
| `ToolService` | `victor/agent/services/tool_service.py` | Registration, semantic selection, budgets |
| `SessionService` | `victor/agent/services/session_service.py` | Session lifecycle, persistence (canonical) |
| `ContextService` | `victor/agent/services/context_service.py` | Context assembly, scoring, pruning |
| `ProviderService` | `victor/agent/services/provider_service.py` | Provider lifecycle, caching, switching |
| `RecoveryService` | `victor/agent/services/recovery_service.py` | Error recovery, loop-prevention |

**Access pattern:**

```python
from victor.runtime.context import ExecutionContext

ctx = ExecutionContext(settings=settings)
ctx.services.chat       # ChatService
ctx.services.tool       # ToolService
ctx.services.provider   # ProviderService
```

> **UI layer rule**: must go through `VictorClient` + `SessionConfig` — never
> import `AgentOrchestrator` or `AgentFactory` directly.

---

## Extension Surfaces

Three **orthogonal, complementary** mechanisms — not duplicates.

```mermaid
flowchart LR
    EP["victor.plugins entry point"] -->|"register(context)"| VP["VictorPlugin<br/>Bootstrap registrar"]
    VP -->|"context.register_vertical()"| VB["VerticalBase<br/>Config template (classmethods)"]
    VB -->|"get_extensions()"| VE["VerticalExtensions<br/>Runtime service (lazy)"]
    subgraph Also["Independent extension points"]
        BP["BaseProvider subclass"]
        BT["BaseTool subclass"]
        WF["YAML DSL / StateGraph"]
        MW["Pre/post middleware hooks"]
    end
    VP -.->|"can also register"| Also
    style VP fill:#6366f1,color:#fff
    style VB fill:#10b981,color:#fff
    style VE fill:#f59e0b,color:#fff
    style Also fill:#e0e7ff,stroke:#4f46e5,color:#1e1b4b
```

| Mechanism | Role | Location |
|-----------|------|----------|
| **Plugin** (`VictorPlugin`) | Bootstrap registrar — discovered via `victor.plugins` entry point | `victor_contracts/core/plugins.py` |
| **Vertical** (`VerticalBase`) | Configuration template — tools, prompts, stages | `victor_contracts/verticals/protocols/base.py` |
| **Extension** (`VerticalExtensions`) | Runtime service — middleware, safety, prompts | `victor_contracts/verticals/extensions.py` |
| **Provider** (`BaseProvider`) | LLM adapter | `victor/providers/` |
| **Tool** (`BaseTool`) | Tool implementation | `victor/tools/` |

**Rule:** External verticals import `victor_contracts` or
`victor.framework.extensions` only — never `victor.agent.*`.

---

## Data & State Model

Victor uses a **two-database architecture** plus a four-scope state model.

```mermaid
erDiagram
    USER ||--o{ PROJECT_DB : "owns many"
    USER ||--|| GLOBAL_DB : "has one"
    GLOBAL_DB {
        string settings
        string api_keys
        string rl_outcomes
        string rl_q_values
        string team_stats
        string tui_sessions
        string failed_signatures
    }
    PROJECT_DB {
        string graph_nodes
        string graph_edges
        string conversations
        string messages
        string sessions
        string entity_memory
        string change_tracking
    }
```

| Database | Path | Contents |
|----------|------|----------|
| **Global** | `~/.victor/victor.db` | Settings, API keys, RL data, team stats, TUI sessions |
| **Project** | `./.victor/project.db` | Graph, conversations, sessions, cache, change tracking |

```mermaid
flowchart TB
    GSM["GlobalStateManager<br/>victor/state/"]
    GSM --> WF["WORKFLOW<br/>per-execution"]
    GSM --> CONV["CONVERSATION<br/>per-conversation"]
    GSM --> TM["TEAM<br/>multi-agent shared"]
    GSM --> GLB["GLOBAL<br/>cross-session persistent"]
    style GSM fill:#6366f1,color:#fff
    style WF fill:#dbeafe,stroke:#3b82f6
    style CONV fill:#d1fae5,stroke:#10b981
    style TM fill:#fef3c7,stroke:#f59e0b
    style GLB fill:#fce7f3,stroke:#ec4899
```

---

## Reading Order — Suggested Path

New to the codebase? Read in this order:

```mermaid
flowchart LR
    A["1. This Blueprint"] --> B["2. System Architecture"]
    B --> C["3. Features Catalog"]
    C --> D["4. Tech Stack"]
    D --> E["5. Roadmap"]
    E --> F["6. Deep-dives"]
    style A fill:#e0e7ff,stroke:#4f46e5,color:#1e1b4b
    style B fill:#dbeafe,stroke:#3b82f6,color:#1e3a5f
    style C fill:#d1fae5,stroke:#10b981,color:#064e3b
    style D fill:#fef3c7,stroke:#f59e0b,color:#78350f
    style E fill:#fce7f3,stroke:#ec4899,color:#831843
    style F fill:#f3e8ff,stroke:#a855f7,color:#3b0764
```

| Step | Document | Purpose |
|------|----------|---------|
| 1 | [Master Blueprint](BLUEPRINT.md) *(this page)* | 10-minute mental model |
| 2 | [System Architecture](../architecture.md) | Canonical, full-depth reference |
| 3 | [Features Catalog](../features.md) | What's implemented, grounded in modules |
| 4 | [Tech Stack](../tech-stack.md) | Dependencies + technical debt register |
| 5 | [Roadmap](../roadmap.md) | Priorities, horizons, feature status |
| 6 | [Deep-dives](#) | [Orchestrator decomposition](orchestrator_decomposition.md), [SDK boundary](CONTRACTS_BOUNDARY.md), [State-passed architecture](state-passed-architecture.md), [Streaming pipeline](streaming-pipeline.md) |

---

## Glossary

| Term | Meaning |
|------|---------|
| **Facade** | `AgentOrchestrator` — delegates all requests; contains no business logic |
| **Canonical service** | One of the 6 mandatory effectful owners (Chat, Tool, Session, Context, Provider, Recovery) |
| **AgenticLoop** | The PERCEIVE→PLAN→ACT→EVALUATE→DECIDE cycle; canonical chat execution authority |
| **Vertical** | A domain configuration template (`VerticalBase`) — tools, prompts, stages |
| **Contract** | A stable type/protocol in `victor-contracts` (independent semver) |
| **StateGraph** | The DAG execution engine — also the foundation for multi-agent teams |
| **SessionConfig** | Immutable runtime-override object — the only sanctioned settings mutation point |
| **Global DB** | `~/.victor/victor.db` — user-wide data (settings, RL, team stats) |
| **Project DB** | `./.victor/project.db` — repo-scoped data (graph, conversations, sessions) |
| **KV caching** | Local-provider optimization: frozen system prompt + deterministic tool ordering |

---

> **Next**: Read the [System Architecture](../architecture.md) for full depth,
> or the [Features Catalog](../features.md) for what's implemented.
