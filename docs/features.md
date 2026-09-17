# Features

**Version**: {{ victor_version }} | **Last Updated**: 2026-08 | **Status**: Canonical

Victor is a contract-first, async-first agentic AI framework. It coordinates LLMs to reason, call
tools, run DAG workflows, and orchestrate multi-agent teams across **25 LLM providers**, with
**34 tool modules** and **9 verticals** (5 domain + 4 utility).

## Core abstractions

| Abstraction | What it does |
|-------------|--------------|
| **Agent** | Public API: `run()` (single-turn), `stream()` (real-time events), `chat()` (multi-turn), `run_workflow()`, `run_team()`. Built with the fluent `AgentBuilder`. |
| **StateGraph** | LangGraph-inspired execution engine — typed state, conditional/cyclic edges, checkpointing, copy-on-write, human-in-the-loop interrupts. |
| **WorkflowEngine** | Compiles YAML workflow definitions into StateGraphs (node types: agent, compute, handler, passthrough). |
| **Tools** | Registry with presets (`default()`, `minimal()`, `full()`, `airgapped()`) across categories: filesystem, git, search, web, database, docker, testing, refactoring, analysis. |
| **Events** | Structured event model (THINKING, TOOL_CALL, TOOL_RESULT, CONTENT, ERROR, STREAM_END) with correlation IDs. |

## Providers (24)

Unified `chat()` / `stream_chat()` across 25 LLM providers — cloud and local — with circuit
breaker, retry, and connection pooling. Cloud (Anthropic, OpenAI, Google, xAI, Azure OpenAI,
Bedrock, Vertex, DeepSeek, Mistral, Groq, Together, Fireworks, OpenRouter, Cerebras, Moonshot,
Z.AI, Replicate, HuggingFace) and local (Ollama, LM Studio, vLLM, llama.cpp, MLX). See the
[Provider Comparison](reference/providers-comparison.md).

**Provider caching** is two independent capabilities: API prompt caching (session-locks tools +
prompt sections for a billing discount on caching cloud providers) and KV prefix caching (freezes
the system prompt and sorts tools for latency savings on local runtimes).

## Tools (34 modules)

34 tool modules across categories (filesystem, git, shell/exec, web, code-search, analysis,
database, docker, testing). Tool selection is semantic + budget-enforced; results carry telemetry
(duration, pruning, follow-up suggestions). New tools subclass `BaseTool` with JSON-Schema params.

## Workflows

A YAML DSL compiles to StateGraphs (or build graphs programmatically). Supports conditional and
cyclic edges, checkpointing, and human-in-the-loop interrupts, with a streaming executor.

## Multi-agent teams

Teams are **formations over a StateGraph** (not a separate graph): `SEQUENTIAL`, `PARALLEL`,
`HIERARCHICAL`, `PIPELINE`, plus reflection. Use `UnifiedTeamCoordinator` directly as a StateGraph
node.

**Heterogeneous teams** — each member can run a different **provider / model / temperature /
reasoning_effort** (capability-gated, e.g. OpenAI o-series / GPT-5 reasoning effort), so a single
team can mix a high-reasoning model for planning with a fast model for execution. Built-in
**review** and **reflection** presets are included (reflection critiques against the original task
and judges satisfaction via a VERDICT).

## Web Chat UI (Chainlit, optional extra)

`victor ui` launches a pure-Python **Chainlit** web chat bound to `VictorClient` — streaming
tokens and reasoning, per-call tool steps with duration/output/pruning telemetry, **informed
approval** cards for risky tools, **stop/cancel** of a running turn, **ChatSettings** (switch
provider/model/profile and toggle approval live), a best-effort **session-restore** seam, and a
per-turn **cost/latency footer**. Install with `pip install victor-ai[chat-ui]`. See
[Web Chat UI](getting-started/web-chat-ui.md).

## Governance: policy engine

A pluggable **policy engine** (`victor/framework/policies/`) evaluates **ALLOW / DENY / ASK**
verdicts over tool calls across REQUEST and RESPONSE phases (non-streaming and streaming). ASK
routes to a container-registered approval handler (the chat UI registers an interactive one).
Gated by the `USE_POLICY_ENGINE` feature flag + `governance.enabled`.

## Sandbox isolation

Subprocess/code-execution tools can be wrapped in an OS sandbox (**bwrap** on Linux, **seatbelt**
on macOS), gated by `settings.sandbox.sandbox_enabled` (off by default, fail-open). Provides
filesystem/network confinement for tool execution without changing tool code.

## Cost & performance (co-design)

Victor measures and acts on the dominant cost term (provider round-trips × context size):

- **Per-turn cost trace (C0)** — accurate tokens/cost/latency per turn, surfaced in the chat UI footer.
- **Reference-aware pruning (L1)** — prunes old tool results no later turn cites, preserving cited output.
- **Prompt-recompute cache (L2)** — caches per-iteration prompt-optimization work within a task.
- **Cost/latency-aware routing (L4)** — biases routing toward cheaper/faster providers when recent turns run hot (with `USE_SMART_ROUTING`).

## Prompt optimization

Runtime evolution of system-prompt sections from execution traces — **GEPA** (default), **MIPROv2**
(few-shot mining), and **CoT distillation** — controlled entirely via settings. Three independent
prompt scopes (main agent, edge-model decisions, subagent).

## Verticals (9)

Complete domain applications on the framework: 5 domain-focused (**Coding, DevOps, RAG,
DataAnalysis, Research**) + 4 utility (**Security, IaC, Classification, Benchmark**). The domain
verticals are first-party source packages in this monorepo and are also separately installable
(`victor-coding`, `victor-devops`, `victor-rag`, `victor-dataanalysis`, `victor-research`). Their
compatibility suite is part of the blocking integration coverage. See
[ADR-007](architecture/adr/007-vertical-distribution-and-sdk-boundary.md) for the ownership and
distribution boundary.

## Surfaces

CLI (Typer), TUI (Textual), **web chat (Chainlit)**, HTTP API, and MCP server — all composing the
same framework through `VictorClient`.

### Release support levels

| Surface | Support level | Release evidence / boundary |
| --- | --- | --- |
| Python SDK and CLI | Core | Package tests run on Python 3.11/3.12; built-wheel import and CLI help smoke are blocking. |
| HTTP API and MCP | Core | Import/factory smoke is required for release; full deployment validation remains an integration concern. |
| First-party domain verticals | Core extension | Source is monorepo-owned; compatibility and discovery coverage are blocking. |
| Textual TUI | Supported optional surface | Selected with `victor tui` or `victor chat --tui`; terminal capability fallback remains available. |
| Chainlit web UI | Optional extra | Install with `victor-ai[chat-ui]`; validate an interactive session before claiming release support. |
| Docker image | Release candidate | Advertised, but a fresh image build/run smoke is required before the release tag. |
| VS Code extension and native Rust extensions | Preview | Their CI checks are currently advisory; do not represent them as release-blocking guarantees. |
| Observability dashboard/API | Experimental | Product support decision remains tracked by TD-5. |
