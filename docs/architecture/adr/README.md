# Architecture Decision Records (ADRs)

This directory contains the Architecture Decision Records for the Victor AI Framework.

## What are ADRs?

Architecture Decision Records (ADRs) document significant architectural decisions made during the development of Victor. Each ADR captures:

- **Context**: The problem or situation
- **Decision**: What was decided
- **Rationale**: Why the decision was made
- **Consequences**: Positive, negative, and neutral impacts

ADRs record *decisions*; the [Technical Debt Register](../../tech-stack.md#technical-debt-register)
records *work*. When an ADR's rollout is incomplete, the remaining work must have a TD-* or EVR-*
entry — the ADR itself never tracks tasks.

## ADR Index

Status is the governance state of the *decision*. Implementation is the observed state of the
*code* (verified 2026-07-29).

| ADR | Title | Status | Implementation | Date |
|-----|-------|--------|----------------|------|
| [ADR-001](001-agent-orchestration.md) | Agent Orchestrator Architecture | Superseded (by service-first runtime, see update in file) | Superseded | 2025-02-26 |
| [ADR-002](002-state-management.md) | State Management System | Accepted | Shipped | 2025-02-26 |
| [ADR-003](003-workflow-engine.md) | Workflow Engine Architecture | Accepted | Shipped | 2025-02-26 |
| [ADR-004](004-tool-system.md) | Tool System Architecture | Accepted | Shipped | 2025-02-26 |
| [ADR-005](005-event-system.md) | Event System Architecture | Accepted | Shipped | 2025-02-26 |
| [ADR-006](006-provider-integration-improvements.md) | Provider Integration Improvements for Non-Interactive Environments | Proposed | Partial | 2026-02-28 |
| [ADR-007](007-vertical-distribution-and-sdk-boundary.md) | Vertical Distribution Model and Contracts Boundary | Accepted | Shipped (CI-guarded boundary; verticals folded into monorepo) | 2026-03-10 |
| [ADR-008](008-registry-performance-optimization.md) | Tool Registry Performance Optimization | Accepted | Shipped | 2025-04-19 |
| [ADR-009](009-rubric-based-completion-evaluation.md) | Rubric-Based Completion Evaluation | Accepted | Shipped opt-in; production-distribution gate failed, so default remains `enhanced` | 2026-06-21 |
| [ADR-010](010-effect-grounded-completion.md) | Effect-Grounded Completion | Accepted | Shipped, opt-in (`victor/framework/effect_gate.py`; `effect_gated_completion` / `VICTOR_EFFECT_GATED_COMPLETION`, default off pending flag-graduation gate) | 2026-06-21 |
| [ADR-011](011-llm-judge-reliability-gating.md) | LLM-Judge Reliability Gating | Accepted | Shipped (`victor/evaluation/judge_calibration.py`, `trajectory_eval.py`); κ/α gate not yet run against human labels | 2026-06-21 |
| [ADR-012](012-regression-gated-harness-acceptance.md) | Regression-Gated Harness Acceptance | Accepted | Done (EVR-5: acceptance oracle + HTIR in `victor/evaluation/`, promotion-gated via `test_acceptance_oracle_gate.py`) | 2026-06-21 |
| [ADR-013](013-unified-temperature-policy.md) | Unified, Intent-Based Temperature Policy with Spin Ratchet | Accepted | Shipped (`victor/framework/temperature/`, default flip 0.7→0.6, scatter-guard test) | 2026-06-22 |
| [ADR-014](014-shared-codegraph-chunker-package.md) | Extract the code→CPG chunker into a shared `victor-codegraph` package | Accepted | Shipped (`victor-codegraph` 0.1.x released; `victor-coding` delegates) | 2026-06-26 |
| [ADR-015](015-victor-core-adopts-codegraph.md) | Victor Core adopts victor-codegraph as the foundational code parser (phased) | Implemented | Phases 0–3 shipped; one parser boundary across graph, chunking, native, memory, and utility consumers | 2026-06-26 |
| [ADR-016](016-distribution-packaging-strategy.md) | Distribution & Packaging: Docker image primary, pip dev; reject native single-binary | Proposed | Not started | 2026-07-02 |
| [ADR-017](017-rl-budget-calibration.md) | RL-Driven Tool-Budget Calibration | Proposed | Partial (3 modules shipped; wiring gated on FEP-0002 review) | 2026-07-08 |
| [ADR-018](018-adopt-sandhi-usage-gateway.md) | Adopt the `sandhi` OSS usage gateway (per-user/team attribution + shared-key metering) | Proposed | Not started (decision doc; see FEP-0020, AnvaiOps ADR-0047) | 2026-07-18 |
| [ADR-019](019-orchestrator-service-runtime-decomposition.md) | Orchestrator & service-runtime target decomposition | Proposed | Not started (records the target for TD-14/TD-15) | 2026-07-29 |
| [ADR-020](020-interactive-terminal-tui.md) | Interactive terminal TUI (Textual) as a first-class surface | Accepted | Shipped 2026-07-30 (`victor tui`, opt-in; TD-22) incl. diff pane + dark/light/high-contrast themes; per-member lanes via ADR-023 | 2026-07-29 |
| [ADR-021](021-terminal-native-hitl-and-loop-transparency.md) | Terminal-native HITL & agent-loop transparency | Accepted | Partial — v1 shipped 2026-07-30 in the TUI (TD-23); exact phase-events + REPL parity pending | 2026-07-29 |
| [ADR-022](022-provider-gateway-feature-layer.md) | Provider gateway feature layer & routing performance | Proposed | Not started (TD-24; depends TD-21) | 2026-07-29 |
| [ADR-023](023-multi-agent-team-durability.md) | Multi-agent team durability (checkpoint/interrupt/per-member stream) | Accepted | Shipped 2026-07-30→08-01 (revisions 1.1–1.13): member/phase/round/iteration-granular checkpoint+resume and per-member streaming lanes across all six formations; durable pause for SEQUENTIAL/PIPELINE/PARALLEL/HIERARCHICAL; FEP-0028 Accepted 2026-08-01; TD-25 Done (iterative-formation pause + non-team chat continuation deferred as FEP Non-Goals/Follow-ups) | 2026-07-29 |
| [ADR-024](024-abstraction-canonicalization-and-import-guard.md) | Abstraction canonicalization + import-time boundary guard | Proposed | Not started (TD-26) | 2026-07-29 |
| [ADR-025](025-ratify-evaluation-centric-p0-decisions.md) | Ratify the evaluation-centric P0 decisions | Proposed | In progress via EVR-2/3/4/5 (ratifies ADR-009/010/011/012) | 2026-07-29 |
| [ADR-026](026-durable-code-memory-ga.md) | Durable code memory GA (one `oid`, correlated graph+vector, tiered) | Proposed | Partial (records GA target for TD-11/12/13 + ADR-015 later phases) | 2026-07-29 |
| [ADR-027](027-prompt-optimization-strategy-fidelity.md) | Prompt-optimization strategy fidelity and honest naming | Accepted | Shipped (faithful CoT distillation; section-scoped PrefPO; honest docs; keys unchanged) | 2026-08-01 |
| [ADR-028](028-single-agent-durable-chat-continuation.md) | Single-agent durable chat continuation (pause/resume on approval) | Proposed | Not started (FEP-0029 Draft; generalizes ADR-023 pause to non-team single-agent runs) | 2026-08-01 |
| [ADR-029](029-provider-support-tiers.md) | Provider support tiers (Tier 1 / Community; docs + constant only, no deletions) | Accepted | Shipped (`TIER_1_PROVIDERS` in `victor/providers/registry.py`, `SUPPORT.md` policy; zero behavioral change) | 2026-08-02 |

## External ADR series (cross-repo)

Victor's code-memory direction is co-designed with two sibling repositories that keep their own ADR
series. References like "ADR-044" (and, in pre-2026-08 history, "ADR-029" for the chunker) in
commits, code comments, and `victor-codegraph/` belong to those series — they are **not** missing
Victor ADRs. Note the collision: Victor's own [ADR-029](029-provider-support-tiers.md) is provider
support tiers; the ProximaDB chunker decision must be cited as "ProximaDB ADR-029".

| Series | Repo / path | Numbering style | Referenced decisions |
|--------|-------------|-----------------|----------------------|
| ProximaDB | `proximaDB/docs/12-design/adr/` | `ADR-0NN` (2-digit, high numbers) | **ADR-029** shared codegraph chunker package (authoritative for the cross-repo chunker decision; pairs with Victor ADR-014/015) · **ADR-044** stable, line-independent symbol oid — the correlated-CPG join key (implemented in `victor-codegraph` 0.1.2) · ADR-028 index policy routing (unrelated to the chunker) |
| AnvaiOps | `anvaiops/docs/adr/` | `NNNN` (4-digit) | **0017** code-graph-as-a-service · **0018** consume shared codegraph chunker (SaaS consumer of `victor-codegraph`) · **0047** AI usage gateway open-core split (authoritative for the `sandhi` homing decision; pairs with Victor ADR-018/FEP-0020) |

Rules for cross-repo decisions:

1. One repo is named **authoritative** for each shared decision; the others hold thin consumer ADRs
   that link to it (chunker: ProximaDB ADR-029 authoritative, Victor ADR-014 owner-of-record for the
   package, AnvaiOps 0018 consumer).
2. Victor code and docs must cite external ADRs with their series name (e.g. "ProximaDB ADR-044"),
   never a bare number.
3. When an external ADR materially changes Victor behavior (as ADR-044 did for symbol identity),
   add or amend a Victor ADR that records the local consequence.

### ADR-016: Distribution & Packaging Strategy

**Decision**: Ship victor as a Docker image (`full` with all extras, `slim` core-only) as the primary packaged artifact; retain the pip-installable package for dev/extensible use; reject native single-binary (PyInstaller/Nuitka/PyOxidizer) as primary.

**Key Points**:
- Hosts run full-capability victor with only Docker — no dep provisioning, no venv pollution, no version drift.
- One consistent container story: `victor:full` agent + per-task eval images (correct runtime per task).
- Native single-binary rejected because victor's architecture fights freezing: dynamic plugin/entry-point discovery, optional native Rust extensions, heavy ML deps (torch ~2 GB), and the pip-based extensibility model.
- Dev experience unchanged (editable pip install remains first-class).

## Creating New ADRs

When making a significant architectural decision:

1. Copy the [template](000-template.md)
2. Fill in all sections
3. Use the next sequential number (next free: **ADR-030**)
4. Update this index (both tables if cross-repo)
5. Submit for review

Heading convention: `# ADR-0NN: Title` (hyphenated) with a `## Metadata` list. ADRs 006/007/008/014/015
predated this and were normalized to it on 2026-07-29 (heading hyphenation + `## Metadata` blocks on
006/008/016); new ADRs must follow it.

## ADR Lifecycle

```
Proposed → Accepted → Superseded/Deprecated
```

- **Proposed**: Initial draft for discussion
- **Accepted**: Decision made (implementation may still be phased/opt-in — record that in the
  Implementation column above, and file remaining rollout work as TD-*/EVR-* items)
- **Superseded**: Replaced by a newer decision (link to the new ADR from the old one)
- **Deprecated**: No longer applicable

Advance the status when reality changes: an ADR that shipped weeks ago must not still read
"Proposed" (this index and the file metadata must agree — the file is authoritative on conflict).

## Related Documentation

- [Victor Architecture](../../architecture.md) — canonical system architecture
- [Evaluation-Centric Runtime Vision](../vision-evaluation-centric-runtime.md) and
  [Backlog](../evaluation-centric-runtime-backlog.md) (EVR-* items)
- [ProximaDB as the CCG Backend](../proximadb-codegraph-backend.md) — live design behind ADR-014/015
  and ProximaDB ADR-044 (TD-11/12/13)
- [Technical Debt Register](../../tech-stack.md#technical-debt-register)
- [FEP Process](../../FEP_PROCESS.md) — FEPs govern framework API *changes*; ADRs record
  architectural *decisions* (a FEP usually yields one or more ADRs)

## References

- [Architecture Decision Records](https://adr.github.io/)
- [Markdown ADR Template](https://github.com/joelparkerhenderson/architecture_decision_record_template)
