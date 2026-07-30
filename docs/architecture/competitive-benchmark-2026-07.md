# Competitive Benchmark — Victor vs. the 2026 Field

**Status:** Reference (non-canonical) · **Created:** 2026-07-29 · **Owner:** Vijaykumar Singh
**Companion:** [ADR index](adr/README.md) · [Technical Debt Register](../tech-stack.md#technical-debt-register) · [EVR backlog](evaluation-centric-runtime-backlog.md)

> This is a *reference* doc, not a canonical one: like the FEP specs it carries its own dated
> snapshot and is **not** version-stamped against `VERSION` (`scripts/ci/check_docs_drift.py` scans
> only the canonical set). It is refreshed per review cycle, not per release. Counts cited for Victor
> use the gated canon (**24 providers, 34 tool modules, 9 verticals**) so the doc stays consistent
> with the drift guard.

## Purpose & method

Victor operates in several markets at once — an agentic framework, a terminal coding agent, an LLM
provider gateway, a multi-agent orchestrator, and (its wager) an evaluation-centric runtime. This
doc benchmarks each against the current field so the improvement ADRs (019–026) are anchored in a
gap, not a hunch.

**Method — first principles per category.** For each market we state the *invariant a category
leader must satisfy* (derived from what the job actually requires, not from what any one product
ships), then score Victor **Leads / At par / Behind / Absent** against the best-in-class incumbent,
and link the ADR that closes the gap. This is deliberately unflattering: the point is to find where
Victor is *Behind* or *Absent* and spec the fix.

Scoring is a snapshot judgement from the 2026-07 review, not a benchmark run. Where a number matters
(e.g. gateway throughput) it is cited to a source below.

---

## 1. Agentic framework core

*Invariant:* a framework leader must give explicit, inspectable, **resumable** state; first-class
human-in-the-loop interrupts; streaming; typed tool/IO contracts; and a durable persistence story.

| Capability | Victor status | Best-in-class | Gap | ADR |
|---|---|---|---|---|
| Explicit state graph | **At par** — `StateGraph` + YAML→graph compiler (`workflows/unified_compiler.py`) | LangGraph | typed edges good; checkpoint story weaker | — |
| Checkpoint / resume | **Behind** — session resume exists; graph-level checkpoint/replay not first-class | LangGraph (checkpointers, time-travel) | no durable per-node checkpoint + replay | ADR-023 |
| HITL interrupts | **At par (web) / Behind (terminal)** — policy-gated approvals, but approval UI is Chainlit-only | LangGraph `interrupt()` | no terminal-native interrupt/resume | ADR-021, ADR-023 |
| Typed IO contracts | **Leads** — contract-first `victor-contracts` SDK + guard tests | Pydantic AI (type-safe) | — | — |
| Streaming | **At par** — one canonical PPAED loop drives streaming (FEP-0007) | LangGraph / Claude Agent SDK | per-node/per-member streaming incomplete | ADR-023 |
| Multi-provider | **Leads** — 24 adapters behind `BaseProvider` | most frameworks single-vendor-leaning | — | ADR-022 |
| Self-evaluation loop | **Leads (differentiator)** — see §5 | ~none built-in | — | ADR-025 |

*Field note:* AutoGen has moved to maintenance in favour of the Microsoft Agent Framework; CrewAI
remains the low-barrier role-based option; Pydantic AI V2 leads on type-safety; the Claude Agent SDK
leads Anthropic-native production agents (hierarchical subagents, fallback chains). Victor's edge is
**contract-first typing + a built-in evaluation loop**; its exposure is **durable graph state**.

## 2. Terminal coding agent / TUI

*Invariant:* the primary surface is the terminal — a leader must run *fully* in the terminal with
legible progress, in-terminal approvals, a repo/code memory, and keyboard-first ergonomics.

| Capability | Victor status | Best-in-class | Gap | ADR |
|---|---|---|---|---|
| Full interactive TUI | **Absent** — REPL + web only; `tui/wire_timeline.py` (171 LOC) replays recorded streams | Claude Code, Aider | no panels/agent-state/diff/keyboard-nav TUI | ADR-020 |
| In-terminal tool approval | **Behind** — HITL approval renders in Chainlit (browser) | Aider (`y/n` in terminal), Claude Code | CLI users must switch to a browser to approve | ADR-021 |
| Agent-loop transparency | **Behind** — a "running…" spinner; no PERCEIVE→PLAN→ACT→EVALUATE phase view | Claude Code (phase + token/cost in status) | no phase/cost surfaced inline | ADR-021 |
| Command discoverability | **Behind** — 26+ slash commands, no `/help` overlay on start | Aider (`/help`) | discoverability gap | ADR-021 |
| Keybindings config | **Absent** — no `~/.victor/keybindings.json` | many modern TUIs | not user-rebindable | ADR-021 |
| Streaming render quality | **Leads** — safe-split markdown, adaptive terminal caps, per-tool colours | Aider / Claude Code | — | — |
| Repo / code memory | **At par → Behind** — CodeGraph CPG index (phased) | Aider repo-map, Cursor | GA of correlated graph+vector memory pending | ADR-026 |
| Sandboxed execution | **At par** — policy engine + sandbox tiers (opt-in, TD-17) | Codex CLI (OS-level sandbox, Rust) | sandbox defaults OFF | — |

*Field note:* Gemini CLI is transitioning to Antigravity CLI; OpenCode leads open-source breadth;
Codex CLI leads sandbox safety; Aider leads git ergonomics. Victor's render quality is genuinely
strong; the glaring gap is that it has **no full terminal UI** and **pushes approvals to a browser** —
the two things a terminal-first user most expects.

## 3. LLM provider gateway

*Invariant:* a gateway leader must unify N providers, route on cost/latency, fail over on typed
fallback chains, cache (exact + semantic), meter/attribute usage, and enforce budgets — at
production throughput.

| Capability | Victor status | Best-in-class | Gap | ADR |
|---|---|---|---|---|
| Provider breadth | **At par** — 24 adapters | LiteLLM (100+) | fewer, but curated + typed | — |
| Smart routing | **Leads-ish** — cost/latency `smart_router.py` (default ON per TD-17) | Portkey, LiteLLM | — | ADR-022 |
| Typed fallback chains | **Behind** — resilience/retry present; declarative fallback chains not first-class | LiteLLM fallback lists, Portkey | no user-declared model fallback ladder | ADR-022 |
| Exact caching | **At par** — prompt-cache + KV-prefix aware | Portkey | — | — |
| Semantic caching | **Absent** | Portkey, GPTCache-style | no semantic cache layer | ADR-022 |
| Budget guardrails | **Behind** — cost tracked (C0); no hard budget stop | Portkey virtual keys/budgets | no enforce-at-limit | ADR-022 |
| Usage attribution | **In progress** — sandhi typed runtime, phases 1–3 (TD-21) | Portkey/LiteLLM proxy | per-user/team metering landing | — (TD-21, ADR-018) |
| Throughput ceiling | **Unknown/at-risk** — Python router | LiteLLM struggles >500 RPS single-instance | Python overhead; Rust router opportunity | ADR-022 |

*Field note:* the transport/metering layer is being homed in `sandhi` (Rust, TD-21/ADR-018). ADR-022
scopes the **policy/feature layer above transport** — semantic cache, budget enforcement, declarative
fallback — plus the case for a Rust-accelerated router given LiteLLM's documented Python RPS ceiling.

## 4. Multi-agent orchestration

*Invariant:* a leader must support multiple coordination topologies, share/scope state safely, allow
HITL mid-run, checkpoint/resume a run, and stream each member's progress.

| Capability | Victor status | Best-in-class | Gap | ADR |
|---|---|---|---|---|
| Topologies | **Leads** — SEQUENTIAL/PARALLEL/HIERARCHICAL/PIPELINE formations, teams-as-StateGraph-nodes | CrewAI (roles), AutoGen (conversations) | clean model, no wrapper graph | — |
| Scoped state | **At par** — 4 scopes incl. TEAM (`GlobalStateManager`) | LangGraph | — | ADR-024 |
| Workspace isolation | **At par** — `WorkspaceIsolation` (TD-10 rename in flight) | — | naming only | — |
| Checkpoint / resume | **Behind** — no team-run checkpoint/replay | LangGraph | durability gap | ADR-023 |
| HITL mid-run | **Behind** — web-only | LangGraph interrupt | terminal parity | ADR-021, ADR-023 |
| Per-member streaming | **Behind** | LangGraph/AutoGen | incomplete | ADR-023 |

*Field note:* Victor's *formation* model is arguably cleaner than CrewAI/AutoGen (no separate
multi-agent graph abstraction). The gap is **durability** — the LangGraph-style checkpoint/interrupt/
replay contract — which ADR-023 specs.

## 5. Evaluation / RL loop — Victor's wager

*Invariant (Victor's own thesis):* *Agent = Model + Harness*; a leader here makes completion
decisions **calibrated, multi-dimensional, effect-grounded, and judge-validated**, and gates every
harness/prompt edit on a **regression-aware acceptance oracle** reported at *(model, harness-config)*
granularity.

| Capability | Victor status | Field | Gap | ADR |
|---|---|---|---|---|
| Built-in completion evaluator | **Leads** — rubric (opt-in) + enhanced (default) | ~none ship this | default flip gated on parity | ADR-025 (EVR-3) |
| Effect-grounded completion | **Decided, not built** — ADR-010 Proposed | none | EVR-4 (P0) | ADR-025 |
| Regression-gated acceptance oracle | **Partial** — parity/characterization batteries exist | none | formalize (EVR-5, P0) | ADR-025 |
| Judge reliability (κ/α) | **Shipped, ungated** — `evaluation/judge_calibration.py`; κ/α not yet run vs human labels | none | EVR-2 | ADR-025 |
| Segment-level process reward | **Computed, unused** — `agent/credit_assignment.py` | none | EVR-7 | — |
| Prompt evolution (GEPA/MIPROv2) | **Leads** — shipped | rare | — | — |

*Field note:* this is the category where Victor is **most differentiated and least matched** — no
mainstream framework ships a graded acceptance oracle. The risk is not competitive; it is *execution*:
the decisions exist (ADR-009/010/011/012) but three loops are still open (see the vision doc). ADR-025
ratifies the P0 sequence rather than re-proposing it.

---

## Synthesis — where the ADRs aim

| Theme | Verdict | ADRs |
|---|---|---|
| **Terminal UX is the biggest visible gap** | Absent full TUI; browser-bound approvals | ADR-020, ADR-021 |
| **Framework/team durability** | Behind LangGraph on checkpoint/interrupt/replay | ADR-023 |
| **Gateway feature+perf layer** | Behind on semantic cache / budgets / fallback / RPS | ADR-022 |
| **Internal structure** | god-object + sprawl slow all of the above | ADR-019, ADR-024 |
| **Evaluation moat** | Lead exists; must be executed & defaulted | ADR-025, ADR-026 |

## Sources

- Turing — *A Detailed Comparison of Top 6 AI Agent Frameworks in 2026*
- Open.cx — *AI agent frameworks compared: LangGraph, CrewAI, AutoGen (2026)*
- amux.io — *Best Terminal AI Coding Agents in 2026 (Claude Code, Gemini CLI, Codex, OpenCode, Aider, Warp, Goose, Q)*
- DataCamp — *Gemini CLI vs. Claude Code (2026)*
- getmaxim.ai — *Top 5 LLM Gateways in 2026*; truefoundry.com — *Top 5 LiteLLM Alternatives in 2026*;
  pkgpulse.com — *Portkey vs LiteLLM vs OpenRouter (2026)*
- Internal: [architecture.md](../architecture.md), [vision-evaluation-centric-runtime.md](vision-evaluation-centric-runtime.md),
  [ux-redesign-plan.md](ux-redesign-plan.md), [tech-stack.md](../tech-stack.md)
