# Diagram refresh — September 2026

This records the D2 diagram dispositions after documentation consolidation and the ADR-030
single-engine migration. Current diagrams describe the execution paths after step 3;
FEP-0031, FEP-0032 and FEP-0033 diagrams are explicitly **TARGET** and retain Draft status.

## Canonical diagram map

| Subject | Canonical diagram | Source checked |
| --- | --- | --- |
| System layering and guards | [Architecture](../architecture.md#system-overview) | Framework entry points; client boundary tests; facade AST guard; hotspot guard; contracts boundary manifests |
| Unified agentic loop | [Streaming sequence](../architecture.md#agenticloop) | `chat_stream_runtime.py`, `chat_stream_executor.py`, `streaming_act_adapter.py`, `agentic_loop.py`; run/stream parity tests |
| Chat runtime inversion target | [FEP-0031](https://github.com/anvai-labs/victor/blob/develop/feps/fep-0031-chat-runtime-inversion.md#target-ownership-diagram) | Proposed Change and Acceptance Criteria; target names and measured baseline |
| One workflow engine | [Workflow execution](../architecture.md#workflow-engine) | Runtime factory, StateGraph adapter/executor, native compiler boundary, CompiledGraph; ADR-030 step 3 |
| Interrupt/resume target | [FEP-0032](https://github.com/anvai-labs/victor/blob/develop/feps/fep-0032-interrupt-resume-semantics.md#target-checkpoint-and-resume-flow) | Proposed checkpoint fields, paused results, input acknowledgement and unresolved HITL bridge |
| RL relocation target | [FEP-0033](https://github.com/anvai-labs/victor/blob/develop/feps/fep-0033-rl-subsystem-relocation.md#target-package-dependencies) | Contracts-first dependency inversion and one-window framework shim |
| SQLite worker and checkpoint boundaries | [Persistence](../architecture.md#database-architecture) | `sqlite_store.py`, compatibility executor failure wrapper, `graph_execution.py`, `graph_checkpoint.py` |
| Release trains | [Release overview](../development/releasing/publishing.md#release-process-overview) | Dependency pins; `release.yml`; `release-contracts.yml`; promotion procedure |
| CI aggregate and promotion matrix | [CI gate map](../development/PR_WORKFLOW.md#ci-gate-map) | `ci-fast.yml` required needs; `ci-test.yml` 3 Python versions × 12 shards; `ci-integration.yml` triggers |

The README and Pages landing page use the same compact layering view. Technology-stack
and duplicate integration maps link to the canonical diagrams rather than declaring a
second ownership model. Supplemental service, provider, tool, state, extension and native
maps use class/module names checked against source. Database initialization is not claimed
to be universally offloaded, and workflow `_error` propagation is separate from SQLite storage.

## Baseline candidate dispositions

The earlier inventory detected **131 candidates**: 36 Mermaid blocks and 95 possible ASCII
blocks. These are baseline locations, so line numbers and headings can differ after D1/D2.
Each row records an editorial disposition, not a claim that every retained document has
received a full API or behavioral audit. File trees, CLI output and mathematical notation
remain text. Existing ADR/FEP design illustrations retain their original context; a Draft
proposal is not evidence of a shipped implementation. Co-design unit reviews under
`docs/reviews/` are historical evidence and are unchanged.

The renamed historical blueprint and extraction records preserve their diagrams in place;
their banners identify the current source of truth. New source claims and changed diagrams
are verified independently of that retained historical material.

| Baseline file | Line | Kind | Nearest heading | Disposition |
| --- | --- | --- | --- | --- |
| `CONTRIBUTING.md` | 30 | Mermaid | Contribution Workflow | Refresh Mermaid or replace duplicate topology with a canonical diagram link |
| `CONTRIBUTING.md` | 233 | ASCII candidate | Vertical Package Structure | Retain file-tree or notation block; not a runtime topology |
| `docs/FEP_PROCESS.md` | 100 | Mermaid | FEP Lifecycle | Refresh Mermaid or replace duplicate topology with a canonical diagram link |
| `docs/README.md` | 16 | ASCII candidate | Document Hierarchy | D1 consolidation: old diagram removed or topic moved; use canonical links |
| `docs/architecture/BLUEPRINT.md` | 51 | Mermaid | Top-Down View — Layer Architecture | Retain historical diagram; page now explicitly directs current readers to canonical architecture |
| `docs/architecture/BLUEPRINT.md` | 109 | Mermaid | Request Flow — A Single Chat Turn | Retain historical diagram; page now explicitly directs current readers to canonical architecture |
| `docs/architecture/BLUEPRINT.md` | 145 | Mermaid | Request Flow — A Single Chat Turn | Retain historical diagram; page now explicitly directs current readers to canonical architecture |
| `docs/architecture/BLUEPRINT.md` | 178 | Mermaid | Bottom-Up View — Package & Module Map | Retain historical diagram; page now explicitly directs current readers to canonical architecture |
| `docs/architecture/BLUEPRINT.md` | 251 | Mermaid | The Six Canonical Services | Retain historical diagram; page now explicitly directs current readers to canonical architecture |
| `docs/architecture/BLUEPRINT.md` | 305 | Mermaid | Extension Surfaces | Retain historical diagram; page now explicitly directs current readers to canonical architecture |
| `docs/architecture/BLUEPRINT.md` | 340 | Mermaid | Data & State Model | Retain historical diagram; page now explicitly directs current readers to canonical architecture |
| `docs/architecture/BLUEPRINT.md` | 369 | Mermaid | Data & State Model | Retain historical diagram; page now explicitly directs current readers to canonical architecture |
| `docs/architecture/BLUEPRINT.md` | 389 | Mermaid | Reading Order — Suggested Path | Retain historical diagram; page now explicitly directs current readers to canonical architecture |
| `docs/architecture/CONTRACTS_BOUNDARY.md` | 12 | ASCII candidate | Overview | Refresh Mermaid or replace duplicate topology with a canonical diagram link |
| `docs/architecture/EXTERNAL_VERTICALS_MIGRATION.md` | 92 | ASCII candidate | 2.1 Directory Layout | Retain file-tree or notation block; not a runtime topology |
| `docs/architecture/adr/001-agent-orchestration.md` | 52 | ASCII candidate | Architecture | Retain proposal/decision illustration in its original design context; implementation follows document status |
| `docs/architecture/adr/003-workflow-engine.md` | 31 | ASCII candidate | Architecture | Retain proposal/decision illustration in its original design context; implementation follows document status |
| `docs/architecture/bayesian.md` | 19 | ASCII candidate | Architecture Diagram | Refresh Mermaid or replace duplicate topology with a canonical diagram link |
| `docs/architecture/bayesian.md` | 68 | ASCII candidate | 1. BayesianTaskAnalysis (Core Belief Representation) | Retain statistical notation, local algorithm sketch or schema relationship; refreshed service and request maps are canonical |
| `docs/architecture/bayesian.md` | 247 | ASCII candidate | 5. BayesianOrchestrationService (Integration Layer) | Retain statistical notation, local algorithm sketch or schema relationship; refreshed service and request maps are canonical |
| `docs/architecture/bayesian.md` | 352 | ASCII candidate | Single-Agent Workflow | Refresh Mermaid or replace duplicate topology with a canonical diagram link |
| `docs/architecture/bayesian.md` | 440 | ASCII candidate | Table Relationships | Retain statistical notation, local algorithm sketch or schema relationship; refreshed service and request maps are canonical |
| `docs/architecture/codegraph-v2-design.md` | 37 | ASCII candidate | Pipeline | Retain design/plan illustration; its document supplies scope and rollout status |
| `docs/architecture/debugging-profiling-guide.md` | 37 | ASCII candidate | DebugLogger: Runtime Logging | Retain code/example block; ASCII detector false positive |
| `docs/architecture/foundations-strategy-2026-07.md` | 67 | ASCII candidate | 2. Dependency graph and sequencing | Retain design/plan illustration; its document supplies scope and rollout status |
| `docs/architecture/framework-vertical-integration.md` | 9 | ASCII candidate | Overview | Refresh Mermaid or replace duplicate topology with a canonical diagram link |
| `docs/architecture/framework-vertical-integration.md` | 234 | ASCII candidate | Data Flow Summary | Refresh Mermaid or replace duplicate topology with a canonical diagram link |
| `docs/architecture/framework-vertical-integration.md` | 264 | ASCII candidate | Cancellation-Aware Tool Discovery | Refresh Mermaid or replace duplicate topology with a canonical diagram link |
| `docs/architecture/graph-architecture-alignment.md` | 44 | ASCII candidate | 3. Plugin Architecture Alignment | Retain design/plan illustration; its document supplies scope and rollout status |
| `docs/architecture/graph-architecture-alignment.md` | 55 | ASCII candidate | Principle: Core Provides Capabilities, Verticals Provide Domain Logic | Retain design/plan illustration; its document supplies scope and rollout status |
| `docs/architecture/graph-enhancements-spec.md` | 420 | ASCII candidate | 4.1 System Architecture | Retain design/plan illustration; its document supplies scope and rollout status |
| `docs/architecture/graph-enhancements-spec.md` | 705 | ASCII candidate | Query: "Who calls process_user and what will break?" | Retain design/plan illustration; its document supplies scope and rollout status |
| `docs/architecture/graph-extension-guide.md` | 22 | ASCII candidate | Two RAG Systems in Victor | Refresh Mermaid or replace duplicate topology with a canonical diagram link |
| `docs/architecture/graph-rag-guide.md` | 13 | ASCII candidate | Architecture | Refresh Mermaid or replace duplicate topology with a canonical diagram link |
| `docs/architecture/orchestrator_decomposition.md` | 9 | ASCII candidate | Component Topology | Retain historical diagram; page now explicitly directs current readers to canonical architecture |
| `docs/architecture/smart_routing.md` | 423 | ASCII candidate | Architecture | Refresh Mermaid or replace duplicate topology with a canonical diagram link |
| `docs/architecture/state-machine.md` | 14 | ASCII candidate | Architecture | Refresh Mermaid or replace duplicate topology with a canonical diagram link |
| `docs/architecture/state-passed-architecture.md` | 398 | ASCII candidate | File Structure | Retain file-tree or notation block; not a runtime topology |
| `docs/architecture/streaming-pipeline.md` | 28 | ASCII candidate | 2. Canonical Architecture | Refresh Mermaid or replace duplicate topology with a canonical diagram link |
| `docs/architecture/unified_prompt_architecture.md` | 30 | ASCII candidate | Solution Architecture | Refresh Mermaid or replace duplicate topology with a canonical diagram link |
| `docs/architecture/ux-redesign-plan.md` | 259 | ASCII candidate | Adaptation Helpers | Retain code/example block; ASCII detector false positive |
| `docs/architecture/vertical-dependency-resolution.md` | 50 | ASCII candidate | Affected Files (29 framework files reference victor_coding) | Retain historical diagram; page now explicitly directs current readers to canonical architecture |
| `docs/architecture/vertical-dependency-resolution.md` | 86 | ASCII candidate | 1. Dependency Direction Rule | Retain historical diagram; page now explicitly directs current readers to canonical architecture |
| `docs/architecture/vertical-dependency-resolution.md` | 103 | ASCII candidate | 2. Three-Layer Architecture | Retain historical diagram; page now explicitly directs current readers to canonical architecture |
| `docs/architecture/vertical-dependency-resolution.md` | 137 | ASCII candidate | 3. Decision Tree for Code Placement | Retain historical diagram; page now explicitly directs current readers to canonical architecture |
| `docs/architecture/vertical-dependency-resolution.md` | 186 | ASCII candidate | Phase 2: Establish Contrib Packages (Week 2-3) | Retain historical diagram; page now explicitly directs current readers to canonical architecture |
| `docs/architecture/vertical-dependency-resolution.md` | 468 | ASCII candidate | Contrib Package Template | Retain historical diagram; page now explicitly directs current readers to canonical architecture |
| `docs/architecture.md` | 35 | Mermaid | System Overview | Refresh Mermaid or replace duplicate topology with a canonical diagram link |
| `docs/architecture.md` | 126 | Mermaid | Data Flow | Refresh Mermaid or replace duplicate topology with a canonical diagram link |
| `docs/architecture.md` | 159 | Mermaid | Layer Architecture | Refresh Mermaid or replace duplicate topology with a canonical diagram link |
| `docs/architecture.md` | 225 | Mermaid | Service Layer | Refresh Mermaid or replace duplicate topology with a canonical diagram link |
| `docs/architecture.md` | 268 | Mermaid | AgenticLoop | Refresh Mermaid or replace duplicate topology with a canonical diagram link |
| `docs/architecture.md` | 307 | Mermaid | Provider System | Refresh Mermaid or replace duplicate topology with a canonical diagram link |
| `docs/architecture.md` | 367 | Mermaid | Tool System | Refresh Mermaid or replace duplicate topology with a canonical diagram link |
| `docs/architecture.md` | 423 | Mermaid | Workflow Engine | Refresh Mermaid or replace duplicate topology with a canonical diagram link |
| `docs/architecture.md` | 460 | Mermaid | Multi-Agent Teams | Refresh Mermaid or replace duplicate topology with a canonical diagram link |
| `docs/architecture.md` | 508 | Mermaid | State Management | Refresh Mermaid or replace duplicate topology with a canonical diagram link |
| `docs/architecture.md` | 562 | Mermaid | Database Architecture | Refresh Mermaid or replace duplicate topology with a canonical diagram link |
| `docs/architecture.md` | 624 | Mermaid | Configuration System | Refresh Mermaid or replace duplicate topology with a canonical diagram link |
| `docs/architecture.md` | 657 | Mermaid | Extension System | Refresh Mermaid or replace duplicate topology with a canonical diagram link |
| `docs/architecture.md` | 705 | Mermaid | Rust Native Extensions | Refresh Mermaid or replace duplicate topology with a canonical diagram link |
| `docs/architecture.md` | 738 | Mermaid | Integration Points Map | Refresh Mermaid or replace duplicate topology with a canonical diagram link |
| `docs/development/PR_WORKFLOW.md` | 26 | ASCII candidate | Branch Structure | D1 consolidation: old diagram removed or topic moved; use canonical links |
| `docs/development/extending/plugins.md` | 87 | ASCII candidate | Directory Layout | Retain file-tree or notation block; not a runtime topology |
| `docs/development/index.md` | 26 | ASCII candidate | Documentation Map | Retain file-tree or notation block; not a runtime topology |
| `docs/development/releasing/publishing.md` | 21 | Mermaid | Release Process Overview | Refresh Mermaid or replace duplicate topology with a canonical diagram link |
| `docs/development/setup.md` | 304 | ASCII candidate | Project Structure Overview | Retain file-tree illustration; not a runtime topology |
| `docs/development/testing/strategy.md` | 34 | ASCII candidate | Test Structure | Retain file-tree illustration; not a runtime topology |
| `docs/development/testing.md` | 9 | ASCII candidate | Test Structure | Retain file-tree illustration; not a runtime topology |
| `docs/feps/fep-0001-edge-model.md` | 53 | ASCII candidate | Architecture | Retain proposal/decision illustration in its original design context; implementation follows document status |
| `docs/feps/vertical-package-spec.md` | 30 | ASCII candidate | File Location | Retain proposal/decision illustration in its original design context; implementation follows document status |
| `docs/guides/OBSERVABILITY.md` | 421 | ASCII candidate | Architecture | Refresh Mermaid or replace duplicate topology with a canonical diagram link |
| `docs/guides/RESILIENCE.md` | 47 | ASCII candidate | States | Refresh Mermaid or replace duplicate topology with a canonical diagram link |
| `docs/guides/WORKFLOW_SCHEDULER.md` | 69 | ASCII candidate | Cron Expressions | Retain file-tree or notation block; not a runtime topology |
| `docs/guides/multi-agent-quickstart.md` | 30 | ASCII candidate | Team Topologies | Consolidate communication-pattern sketch into existing comparison table and canonical team execution link |
| `docs/guides/observability/event-bus.md` | 421 | ASCII candidate | Architecture | Refresh Mermaid or replace duplicate topology with a canonical diagram link |
| `docs/guides/observability/metrics.md` | 9 | ASCII candidate | Architecture | Refresh Mermaid or replace duplicate topology with a canonical diagram link |
| `docs/guides/observability/metrics.md` | 84 | ASCII candidate | View Real-Time Events | Retain command/output illustration; not a runtime topology |
| `docs/guides/task_completion.md` | 278 | ASCII candidate | Architecture | Refresh Mermaid or replace duplicate topology with a canonical diagram link |
| `docs/guides/vertical-quickstart.md` | 102 | Mermaid | Vertical Architecture | Refresh Mermaid or replace duplicate topology with a canonical diagram link |
| `docs/guides/vertical-quickstart.md` | 149 | Mermaid | Multi-Stage Vertical | Refresh Mermaid or replace duplicate topology with a canonical diagram link |
| `docs/guides/workflow-development/examples.md` | 17 | ASCII candidate | Current Architecture Diagram | D1 historical archive: original diagram retained in workflow-consolidation-plan-historical.md |
| `docs/guides/workflow-development/examples.md` | 174 | ASCII candidate | Phase 4: Single Execution Engine (SRP) | D1 historical archive: original diagram retained in workflow-consolidation-plan-historical.md |
| `docs/guides/workflow-development/scheduling.md` | 69 | ASCII candidate | Cron Expressions | Retain file-tree or notation block; not a runtime topology |
| `docs/guides/workflow-quickstart.md` | 41 | ASCII candidate | Workflow Structure | Refresh Mermaid or replace duplicate topology with a canonical diagram link |
| `docs/index.md` | 22 | Mermaid | Architecture at a Glance | Refresh Mermaid or replace duplicate topology with a canonical diagram link |
| `docs/reference/configuration/index.md` | 17 | ASCII candidate | Configuration Directory Structure | Retain file-tree or notation block; not a runtime topology |
| `docs/reference/configuration/keys.md` | 21 | ASCII candidate | Overview | Refresh Mermaid or replace duplicate topology with a canonical diagram link |
| `docs/reference/providers-comparison.md` | 21 | Mermaid | Decision Tree | Refresh Mermaid or replace duplicate topology with a canonical diagram link |
| `docs/reference/quick-reference.md` | 177 | ASCII candidate | Linux / macOS | Retain file-tree or notation block; not a runtime topology |
| `docs/reference/quick-reference.md` | 190 | ASCII candidate | Windows | Retain file-tree or notation block; not a runtime topology |
| `docs/reference/quick-reference.md` | 203 | ASCII candidate | Project Context | Retain file-tree or notation block; not a runtime topology |
| `docs/reference/verticals/index.md` | 19 | ASCII candidate | Overview | Refresh Mermaid or replace duplicate topology with a canonical diagram link |
| `docs/tech-stack.md` | 89 | Mermaid | Dependency Map | D1 consolidation: old diagram removed or topic moved; use canonical links |
| `docs/tech-stack.md` | 141 | Mermaid | Dependency Rules | D1 consolidation: old diagram removed or topic moved; use canonical links |
| `docs/tech-stack.md` | 193 | Mermaid | Database Schema | D1 consolidation: old diagram removed or topic moved; use canonical links |
| `docs/tech-stack.md` | 279 | Mermaid | Tech Debt Timeline | D1 consolidation: old diagram removed or topic moved; use canonical links |
| `docs/user-guide/rich-formatting-guide.md` | 138 | ASCII candidate | Progressive Tool Output Streaming | Refresh Mermaid or replace duplicate topology with a canonical diagram link |
| `docs/user-guide/session-management.md` | 100 | ASCII candidate | `/save` - Save Session | Retain command/output illustration; not a runtime topology |
| `docs/user-guide/session-management.md` | 127 | ASCII candidate | `/sessions` - List Sessions | Retain command/output illustration; not a runtime topology |
| `docs/user-guide/session-management.md` | 152 | ASCII candidate | `/resume` - Restore Session | Retain command/output illustration; not a runtime topology |
| `docs/user-guide/session-management.md` | 166 | ASCII candidate | `/resume` - Restore Session | Retain command/output illustration; not a runtime topology |
| `docs/user-guide/session-management.md` | 237 | ASCII candidate | `/compact` - Reduce Context Size | Retain command/output illustration; not a runtime topology |
| `docs/user-guide/session-management.md` | 270 | ASCII candidate | `victor sessions list` - List Sessions | Retain command/output illustration; not a runtime topology |
| `docs/user-guide/session-management.md` | 300 | ASCII candidate | `victor sessions show` - Show Session Details | Retain command/output illustration; not a runtime topology |
| `docs/user-guide/session-management.md` | 328 | ASCII candidate | `victor sessions search` - Search Sessions | Retain command/output illustration; not a runtime topology |
| `docs/user-guide/troubleshooting.md` | 717 | ASCII candidate | Check for syntax errors | Retain command/output illustration; not a runtime topology |
| `docs/ux-adoption-action-plan.md` | 22 | ASCII candidate | Dependency graph (read first) | Retain design/plan illustration; its document supplies scope and rollout status |
| `docs/verticals/architecture_refactoring.md` | 44 | ASCII candidate | After Refactoring | Retain historical diagram; page now explicitly directs current readers to canonical architecture |
| `docs/verticals/architecture_refactoring.md` | 383 | ASCII candidate | Vertical Loading Flow | Retain historical diagram; page now explicitly directs current readers to canonical architecture |
| `docs/verticals/coding.md` | 322 | ASCII candidate | File Structure | Retain file-tree or notation block; not a runtime topology |
| `docs/verticals/data-analysis.md` | 378 | ASCII candidate | File Structure | Retain file-tree or notation block; not a runtime topology |
| `docs/verticals/devops.md` | 295 | ASCII candidate | File Structure | Retain file-tree or notation block; not a runtime topology |
| `docs/verticals/rag.md` | 360 | ASCII candidate | File Structure | Retain file-tree or notation block; not a runtime topology |
| `docs/verticals/research.md` | 382 | ASCII candidate | File Structure | Retain file-tree or notation block; not a runtime topology |
| `feps/fep-0001-fep-process.md` | 85 | ASCII candidate | High-Level Design | Retain proposal/decision illustration in its original design context; implementation follows document status |
| `feps/fep-0001-fep-process.md` | 220 | ASCII candidate | Repository Structure | Retain proposal/decision illustration in its original design context; implementation follows document status |
| `feps/fep-0002-documentation-style-guide.md` | 86 | ASCII candidate | High-Level Design | Retain proposal/decision illustration in its original design context; implementation follows document status |
| `feps/fep-0002-documentation-style-guide.md` | 337 | ASCII candidate | Usage | Retain proposal/decision illustration in its original design context; implementation follows document status |
| `feps/fep-0002-documentation-style-guide.md` | 421 | Mermaid | Implementation | Retain proposal/decision illustration in its original design context; implementation follows document status |
| `feps/fep-0003-progressive-tool-loading.md` | 105 | ASCII candidate | High-Level Design | Retain proposal/decision illustration in its original design context; implementation follows document status |
| `feps/fep-0004-provider-oauth.md` | 32 | ASCII candidate | Provider OAuth Landscape (March 2026) | Retain proposal/decision illustration in its original design context; implementation follows document status |
| `feps/fep-0004-provider-oauth.md` | 68 | ASCII candidate | Architecture | Retain proposal/decision illustration in its original design context; implementation follows document status |
| `feps/fep-0004-provider-oauth.md` | 122 | ASCII candidate | Token Lifecycle | Retain proposal/decision illustration in its original design context; implementation follows document status |
| `feps/fep-0009-sdk-tool-contract.md` | 112 | ASCII candidate | High-Level Design | Retain proposal/decision illustration in its original design context; implementation follows document status |
| `feps/fep-0010-shared-protocol-crate.md` | 81 | ASCII candidate | High-Level Design | Retain proposal/decision illustration in its original design context; implementation follows document status |
| `feps/fep-0013-shell-safety-policy.md` | 181 | ASCII candidate | Layered safety model | Retain proposal/decision illustration in its original design context; implementation follows document status |
| `feps/fep-0013-shell-safety-policy.md` | 220 | ASCII candidate | Architecture: today vs. proposed | Retain proposal/decision illustration in its original design context; implementation follows document status |
| `feps/fep-0013-shell-safety-policy.md` | 248 | ASCII candidate | Decision flow | Retain proposal/decision illustration in its original design context; implementation follows document status |
| `feps/fep-0017-prompt-optimization-reward-loop.md` | 109 | ASCII candidate | Lifecycle (closed loop) | Retain proposal/decision illustration in its original design context; implementation follows document status |
| `feps/fep-0024-pluggable-code-correction.md` | 124 | ASCII candidate | High-Level Design | Retain proposal/decision illustration in its original design context; implementation follows document status |

## Rendering checks

All **38 added or changed Mermaid blocks** rendered successfully with Mermaid CLI **11.9.0**.
The final landing-page, streaming-loop, provider, tool, workflow and CI views were also rendered
as PNG previews for visual inspection. The common
theme uses a restrained blue palette, visible titles and labeled interactions. Source is
maintained in Markdown; rendered verification artifacts are temporary and are not committed.
Historical diagrams and unchanged illustrative examples are excluded from this rendering claim.
