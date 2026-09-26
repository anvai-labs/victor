# Victor AI Framework — Documentation

!!! abstract "Start here"

    Victor is a contract-first agent framework for tool use, compiled workflows and
    multi-agent coordination across local and cloud providers.

    **Published baseline:** Victor AI 0.9.5. **Development line:** 0.10.0 candidate.

**Version**: {{ victor_version }} | **License**: Apache-2.0 | **Python**: 3.12+

---

## Quick Start

```bash
pip install victor-ai
export ANTHROPIC_API_KEY=...
victor chat "Explain this codebase"
```

---

## Architecture at a Glance

```mermaid
---
title: Victor system overview
---
%%{init: {"theme":"base","themeVariables":{"primaryColor":"#E8EFF7","primaryTextColor":"#17324D","primaryBorderColor":"#456987","lineColor":"#456987","fontFamily":"Arial"}}}%%
flowchart TB
  subgraph C["Clients"]
    CS["CLI · TUI · HTTP · MCP · VS Code"]
  end
  subgraph F["Framework · public API"]
    VC["VictorClient<br/>application/session API"]
    AG["Agent · AgentFactory<br/>agent lifecycle"]
    WF["WorkflowEngine · StateGraph<br/>workflow authoring"]
  end
  subgraph R["Runtime · internal"]
    OR["AgentOrchestrator<br/>composition facade"]
    SV["Chat · tool · session services<br/>owned behavior"]
    WR["Workflow runtime<br/>compiler · executor · CompiledGraph"]
  end
  subgraph I["Infrastructure"]
    IN["providers · tools · storage · core"]
  end
  V["External vertical definitions"]
  S["victor_contracts<br/>portable definitions"]
  CS -->|"call"| VC
  CS -->|"create or embed"| AG
  CS -->|"submit workflows"| WF
  VC -->|"delegate"| OR
  AG -->|"construct and delegate"| OR
  WF -->|"compile and execute"| WR
  OR -->|"delegate behavior"| SV
  SV -->|"use"| IN
  WR -->|"use"| IN
  V -->|"import only"| S
  AG -.->|"consume contracts"| S
  WF -.->|"consume contracts"| S
```

**Start here** → [System Architecture](architecture.md) for the full picture.

---

## Delivery State

| Surface | v0.9.5 public release | Current `develop` |
| --- | --- | --- |
| Sandhi consumer accounting | Published | Published behavior plus later lifecycle fixes |
| Tool supply | Existing selection runtime | Unified turn pipeline and demand hydration integrated |
| Teams | Six core formations | Expanded formations, isolation and usage guards |
| Chat ownership | Earlier service slices | Turn frame, planning and stream controls migrated |
| Toolchain | Node 22.12 or 24 extension support | Node 24 minimum; Python 3.12/3.13 CI |

!!! note

    Pages deploys from `main`. Pull requests build a preview; the current `develop`
    documentation publishes with the next main promotion.

---

## Documentation Map

### Core Documents (Start Here)

| Document | Description |
|----------|-------------|
| **[System Architecture](architecture.md)** | **Single source of truth** — layers, services, providers, tools, state, extensions, diagrams |
| **[Features](features.md)** | Complete feature catalog grounded in implementation |
| **[Roadmap](roadmap.md)** | Current release and Stage C priorities, complete debt register |
| **[Tech Stack](tech-stack.md)** | Technology choices and dependency requirements |

### Architecture Deep-Dives

| Document | Description |
|----------|-------------|
| [Orchestrator Decomposition](architecture/orchestrator_decomposition.md) | Historical extraction record; current ownership links back here |
| [SDK Boundary](architecture/CONTRACTS_BOUNDARY.md) | Plugin/vertical/extension contracts and import rules |
| [State-Passed Architecture](architecture/state-passed-architecture.md) | Coordinator patterns, ContextSnapshot, CoordinatorResult |
| [Native Acceleration Strategy](architecture/native-acceleration-strategy.md) | Measured Python/Rust/FFI decision rules and platform plan |
| [Streaming Runtime](architecture/streaming-pipeline.md) | Current turn flow, capability ownership and lifecycle guards |
| [Smart Routing](architecture/smart_routing.md) | Provider routing and selection |
| [Edge Provider Strategy](architecture/edge-provider-tool-strategy.md) | Edge model decisions |
| [InferFlux Reasoning Contract](architecture/inferflux-reasoning-separation-handoff.md) | Reasoning stream and usage contract consumed through Sandhi |
| [ADR Index](architecture/adr/README.md) | Architecture Decision Records |

### User Guides

| Document | Description |
|----------|-------------|
| [CLI Reference](user-guide/cli-reference.md) | All `victor` CLI commands and flags |
| [Workflows](user-guide/workflows.md) | StateGraph and YAML workflow DSL |
| [Providers](user-guide/providers.md) | Provider configuration and switching |
| [Tools](user-guide/tools.md) | Built-in tool modules and usage |
| [Session Management](user-guide/session-management.md) | Sessions, context, state scopes |
| [Troubleshooting](user-guide/troubleshooting.md) | Common issues and solutions |

### Developer Guides

| Document | Description |
|----------|-------------|
| [Development Setup](development/setup.md) | Install, venv, pre-commit, editor config |
| [Testing Strategy](development/testing.md) | Unit/integration/benchmark, autouse fixtures |
| [Code Style](development/code-style.md) | Black, ruff, mypy, line length 100 |
| [Service Guide](architecture/orchestrator_decomposition.md) | Service layer development patterns |
| [Deprecation Policy](development/deprecation-policy.md) | How deprecations are managed |
| [PR Workflow](development/PR_WORKFLOW.md) | Branch hygiene, commit conventions, review process |
| [FEP Process](FEP_PROCESS.md) | Framework Enhancement Proposal workflow |
| [FEP Index](https://github.com/anvai-labs/victor/blob/develop/feps/README.md) | Canonical proposals in repository-root `feps/` |
| [Native Build](development/setup.md#native-extension-build) | Canonical native extension recipe |
| [Docs Build](development/setup.md#documentation-build) | Build and preview the documentation site |
| [Plugin Development](development/extending/plugins.md) | Creating Victor plugins |
| [Vertical Development](development/extending/verticals.md) | Building domain verticals |

### API Reference

| Document | Description |
|----------|-------------|
| [Providers API](api-reference/providers.md) | Provider adapter interface |
| [Tools API](api-reference/tools.md) | Tool registration and execution |
| [Workflows API](api-reference/workflows.md) | Workflow compiler and executor |
| [Protocols](api-reference/protocols.md) | Core protocols and interfaces |

### Configuration Reference

| Document | Description |
|----------|-------------|
| [Settings Reference](reference/settings-reference.md) | All 26+ config groups |
| [Configuration Options](reference/configuration-options.md) | Detailed config options |
| [Environment Variables](reference/environment-variables.md) | All env vars |
| [Provider Comparison](reference/providers-comparison.md) | Feature matrix for 25 providers |
| [CLI Commands](reference/cli-commands.md) | CLI command reference |
| [Skills](reference/skills.md) | Skill registry and YAML definitions |
| [Embeddings](reference/embeddings.md) | Embedding backends and configuration |

### Verticals

| Document | Description |
|----------|-------------|
| [Coding](verticals/coding.md) | Code review, editing, test generation |
| [DevOps](verticals/devops.md) | Infrastructure, CI/CD, containers |
| [RAG](verticals/rag.md) | Retrieval, ingestion, search |
| [Data Analysis](verticals/data-analysis.md) | Dataframes, statistics, visualization |
| [Research](verticals/research.md) | Source research, synthesis |
| [API Reference](verticals/api_reference.md) | Vertical API reference |

### Tutorials

| Document | Description |
|----------|-------------|
| [Build Custom Tool](tutorials/build-custom-tool.md) | Step-by-step tool creation |
| [Create Workflow](tutorials/create-workflow.md) | Workflow DSL tutorial |
| [Integrate Provider](tutorials/integrate-provider.md) | Provider adapter tutorial |

### Guides

| Document | Description |
|----------|-------------|
| [Observability](guides/observability/index.md) | Monitoring and tracing |
| [MCP Server](guides/VICTOR_AS_MCP_SERVER.md) | Using Victor as MCP server |
| [Task Completion](guides/task_completion.md) | Task fulfillment detection |
| [Workflows](guides/WORKFLOW_SCHEDULER.md) | Workflow scheduler guide |

---

## Document Governance

| Priority | Document | Role |
|----------|----------|------|
| **Canonical** | `docs/architecture.md` | System architecture (single source of truth) |
| **Canonical** | `docs/features.md` | Feature catalog |
| **Canonical** | `docs/roadmap.md` | Roadmap and tech debt |
| **Canonical** | `docs/tech-stack.md` | Technology stack |
| **Canonical** | `VISION.md` | Product vision |
| **Reference** | `docs/diagrams/` | Editable `.mmd` sources; canonical diagrams live inline in the master docs |

---

## Contributing

See [Development Setup](development/setup.md) and [PR Workflow](development/PR_WORKFLOW.md).
Contributions use conventional commits and the checks documented in the PR workflow.

## Current work and historical evidence

The [roadmap](roadmap.md) distinguishes the published 0.9.5 baseline from the 0.10.0
development candidate and remaining Stage C work.
The [September co-design review](reviews/2026-09-03-codesign/README.md) preserves dated findings;
merged design documents do not imply their implementation has shipped. See the
[documentation source index](https://github.com/anvai-labs/victor/blob/develop/docs/README.md) for canonical ownership and historical-record conventions.
