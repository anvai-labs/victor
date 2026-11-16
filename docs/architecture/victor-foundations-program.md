# Victor Foundations Program

Status: Active
Created: 2026-11-16
Tracker: [victor-foundations-plan.md](victor-foundations-plan.md) (owns tasks, status, evidence)
Related: FEP-0047, FEP-0048, FEP-0032, ADR-035, ADR-036, ADR-037

## 1. Why this program exists

A competitive review (2026-11-16, evidence-based survey of the full repo against
LangChain/LangGraph, CrewAI, AutoGen/Microsoft Agent Framework, Semantic Kernel, DSPy,
Claude Code, and Codex) placed Victor in the **deep-specialist quadrant**: the strongest
runtime engineering in the field attached to the weakest adoption surface.

Measured strengths (Victor leads the compared field): provider breadth and resilience
(25 adapters, dual circuit breakers, health-scored routing), eval/RL-from-trajectories
infrastructure (~25 learners, GEPA, judge calibration, SWE-bench harness), code intelligence
(victor-codegraph semantic/structural/SCIP tiers, native 81× CCG engine with parity CI),
and loop verification machinery (effect gate, per-turn auditor, backslide guard).

Measured gaps: workflow durability (graph checkpointing defaults to in-memory;
interrupt/resume partially landed), security defaults (sandbox fail-open/MVP backends; no
untrusted-content containment), memory (entity/graph only; no user/project preference store),
subagent surface (no LLM-callable spawn primitive), ecosystem/DX (single-maintainer signals,
example drift, contract-install tax), and the meta-gap that **the best machinery is
flag-gated off by default**.

The field consolidated in 2025–2026 (LangGraph 1.x durable execution; Microsoft Agent
Framework 1.0 merging Semantic Kernel + AutoGen; Codex default-sandbox branding). There is no
generic middle left to occupy. Victor's winning position:

> **The code-intelligent, provider-sovereign, verifiably self-correcting agent runtime —
> with its depth on by default.**

## 2. Principles

1. **Strong by default.** Verification, durability, and intelligence that exist in the repo
   must be observable in default behavior. Flags graduate to defaults on evidence
   (ADR-036), never by fiat.
2. **Safe by default.** Sandbox posture is fail-closed with an explicit, loud escape hatch
   (ADR-035). Untrusted content is contained structurally, with deterministic canary
   blocking as the floor (ADR-037).
3. **Durable by default.** Workflow state survives process death; paused runs resume at
   their node (FEP-0047, FEP-0032).
4. **Every flip ships with its opt-out and a negative test.** Defaults are falsifiable;
   demotion follows the same evidence standard as promotion.
5. **Reuse before build.** Each capability wires existing machinery (checkpoint store,
   graduation harness, memory coordinator, credibility/reputation signals) rather than
   adding parallel stacks.

## 3. Tier ladder

Implementation order (details and task IDs in the tracker):

- **Tier D — Durability** (first): `GraphSQLiteCheckpointer` adapter; durable-by-default
  injection with opt-out (FEP-0047); FEP-0032 phase 2 (loud input discard); paused-result
  surfacing on every client surface; StateGraph executor parity (timeout, verifier, token
  streaming); `USE_STATEGRAPH_AGENTIC_LOOP` default flip only after parity.
- **Tier S — Security**: fail-closed sandbox posture + backend hardening (ADR-035);
  container quality bar promoted to core; untrusted-content containment (ADR-037);
  flag graduation batteries for `effect_gated_completion` and `per_turn_auditor` (ADR-036).
- **Tier A — Adoption**: LLM-callable subagent-spawn primitive; core long-term memory
  (FEP-0048); SCIP precise tier auto-on + published impact-analysis benchmark; RBAC/SSO
  wired into the API surface; research stack (search API adapters, page-read, browser tool).
- **Tier C — Consolidation & governance** (continuous): single circuit breaker; services
  module map (continues Stage C); native-parity CI broadened; FEP reviewer panels activated
  (FEP-0036); external-committer rails.

Tier D goes first: durability is the foundation the other tiers stand on (paused results
need durable checkpoints; the executor flip needs parity; graduation evidence needs stable
runs).

## 4. What this program is not

- It does not duplicate the VAS agent-service plan (API/web/WS-B–I work) or the Stage C
  backlog; where items overlap, the FOU tracker cross-references and the existing ledger
  keeps ownership.
- It does not change the branch/PR model, the two-database model, teams-as-formations, or
  the contract-first boundary — it operates within them.
- It does not ship a default flip without its battery evidence, opt-out, and negative test.

## 5. Success criteria

A future re-run of the competitive review should show: workflow durability ≥ 9 (durable by
default, resume across restarts), security defaults ≥ 8.5 (fail-closed posture, containment
floor), DX ≥ 7 (subagent primitive, memory, SCIP on by default), with no regression on the
leading dimensions. Each flip is evidenced in the tracker; each FEP/ADR reaches Accepted or
Implemented status through the normal governance process.
