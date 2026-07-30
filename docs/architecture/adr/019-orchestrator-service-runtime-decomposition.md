# ADR-019: Orchestrator and Service-Runtime Target Decomposition

## Metadata

- **Status**: Proposed
- **Date**: 2026-07-29
- **Decision Makers**: Vijaykumar Singh
- **Related ADRs**: 001 (superseded original orchestration decision — this ADR records the *target*
  the service-first runtime is converging toward)
- **Work tracked by**: [TD-14](../../tech-stack.md#technical-debt-register) (orchestrator regrowth),
  [TD-15](../../tech-stack.md#technical-debt-register) (services sprawl)
- **Benchmark**: [competitive-benchmark-2026-07.md](../competitive-benchmark-2026-07.md) §1

## Context

TD-14/TD-15 track *that* the runtime is too large; neither records the *target module boundaries* the
decomposition should reach. Per the ADR README ("ADRs record *decisions*; the register records
*work*"), that target is a decision and belongs here. The register items stay the work tracker; this
ADR is the destination they aim at.

Observed reality (2026-07-29):

- `victor/agent/orchestrator.py` is **4,690 LOC / 191 methods**. Its own docstring already declares
  "Facade Pattern" and lists nine extracted components (`ConversationController`, `ToolPipeline`,
  `StreamingController`, `StreamingCoordinator`, `LifecycleManager`, `TaskAnalyzer`, `ToolSelector`,
  `ToolRegistrar`, `ProviderManager`) — yet it regrew ~34% after TD-R1 declared it resolved at 3,510
  LOC. The facade *delegates* (delegation is real) but still *holds behavior*.
- `victor/agent/services/` holds ~55 files; four are 100k+ chars (`planning_runtime.py`,
  `runtime_intelligence.py`, `turn_execution_runtime.py`, `tool_service.py`) — far beyond the
  documented "six canonical services" (Chat, Tool, Session, Context, Provider, Recovery). The
  architecture story and the file tree disagree.
- The orchestrator reaches into service internals directly (`_chat_service`, `_tool_service`,
  `_metrics_coordinator`, …) rather than exclusively through `ExecutionContext.services` — the review
  counted 40+ such direct accesses. This is facade leakage: it defeats the swap-a-service seam the
  service layer exists to provide.

A ratchet guard already landed (`tests/unit/runtime/test_hotspot_size_guard.py`, 2026-07-02) so the
file cannot silently regrow a third time. What is missing is the *target shape* the ratchet lowers
toward.

## Decision

Adopt three invariants as the decomposition target, and lower the ratchet caps toward them:

1. **A facade holds no behavior.** `AgentOrchestrator` is a composition root + session boundary +
   compatibility surface only. Any method with branching domain logic moves to a service or a named
   component object; the orchestrator retains only wiring, high-level flow sequencing, and post-switch
   hooks. Target: **< 1,500 LOC**, reached by scheduled ratchet steps (not one big-bang PR).

2. **Six canonical services are the only ownership layer; the runtime modules are reconciled into
   it.** Each of `planning_runtime.py`, `runtime_intelligence.py`, `turn_execution_runtime.py` is
   either (a) folded under its owning canonical service as a private collaborator, or (b) promoted to
   a *named* seventh+ architectural element documented in `architecture.md` — but the tree must not
   silently carry undocumented mega-modules. No module in `services/` exceeds a declared size cap
   (start at current max, ratchet down).

3. **Non-composition-root code reaches services via `ExecutionContext.services`; nothing reaches
   service *private internals*.** *(Reconciled 2026-07-30.)* The original wording — "remove the
   orchestrator's direct `_service` reach-ins" — was wrong: the orchestrator is the **composition
   root**, so it legitimately holds and delegates to its services, and `test_service_layer_validation.py`
   in fact *requires* those `self._<x>_service` references. The real targets of the
   "`ExecutionContext.services`-only" rule are therefore (a) **non-orchestrator** code reaching
   services directly, and (b) any code (orchestrator included) reaching into a service's **private
   internals**. An access-boundary guard is deferred until it is scoped to those two cases; it is not
   part of increment 1.

This ADR does **not** introduce a new orchestration abstraction (explicitly a non-goal in the
evaluation-centric vision) — it finishes the service-first one ADR-001's update already names.

## Rationale

- **First principles.** A facade's whole value is that callers depend on a thin, stable surface and
  swap what's behind it. A 4,690-LOC facade with 40+ internal reach-ins is a god-object wearing a
  facade's name; the abstraction is nominal. The fix is to make the invariant *mechanically true*
  (size caps + access guard), not aspirational (a docstring).
- **Co-design.** The target caps are set *with* the ratchet guard that enforces them and *with* the
  TD-14/TD-15 work items that execute them — doc, test, and backlog move together.
- **Leverage.** Every other ADR here (TUI, gateway, team durability) adds surface to the runtime; a
  smaller, seam-clean core is the precondition that keeps those additions cheap.

## Consequences

- **Positive**: testable-in-isolation services; a real swap seam; the ratchet has a destination, so
  each decomposition PR can lower a concrete cap.
- **Negative**: a sustained multi-PR effort touching hot code paths; risk of regression in the chat
  loop — mitigated by the existing parity/characterization batteries (must stay green per each step).
- **Neutral**: public API (`Agent.create/run/stream`) is unchanged; this is internal.

## Implementation

Incremental, ratchet-gated (no separate FEP — internal runtime, no public-API change):

1. Add the access guard (extend `test_service_layer_validation.py`); freeze new reach-ins.
2. Per canonical service, pull orchestrator methods into the owning service; lower the orchestrator
   ratchet cap one step each PR.
3. Reconcile the three runtime mega-modules (fold or promote+document); add per-file size caps to the
   hotspot guard.
4. When the orchestrator clears < 1,500 LOC and no undocumented mega-module remains, mark TD-14/TD-15
   Resolved and flip this ADR to Accepted.

## Alternatives Considered

- **Leave as-is (ratchet only).** Rejected: a cap with no target just prevents *growth*, not the
  god-object itself.
- **Big-bang rewrite.** Rejected: unacceptable regression risk on the live chat loop; the batteries
  gate incremental change far more safely.
- **New orchestration abstraction / actor model.** Rejected: explicit non-goal (vision doc,
  "one canonical loop; no parallel abstractions").

## References

- [ADR-001](001-agent-orchestration.md) (superseded; its 2026-05-04 update names the service-first shape)
- [architecture.md](../../architecture.md) §Service Layer / §Agent Runtime
- `victor/agent/orchestrator.py`, `victor/agent/services/`, `tests/unit/runtime/test_hotspot_size_guard.py`,
  `tests/unit/framework/test_service_layer_validation.py`

## Revision History

| Date | Version | Changes | Author |
|------|---------|---------|--------|
| 2026-07-29 | 1.0 | Initial ADR — records the decomposition target for TD-14/TD-15 | Vijaykumar Singh |
| 2026-07-30 | 1.1 | Reconciled §3 (orchestrator is the composition root; access-boundary rule re-scoped, guard deferred). Increment 1 shipped: extracted the pure task-report metadata builders to `victor/agent/task_report_metadata.py`; orchestrator 4690→4600, ratchet lowered. | Vijaykumar Singh |
