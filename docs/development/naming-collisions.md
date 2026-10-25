# Naming collisions — tracked debt (TD-033)

**Status:** Rename queue CLEARED 2026-10-24. Guard: `tests/unit/framework/
test_agent_naming_validation.py::TestNoProductionSymbolShadowing` (general
AST scan; new top-level test classes shadowing production classes FAIL —
allowlist EMPTY; a genuine alias needs an owner-approved reason to be
added).

## Why this matters

Victor's code intelligence treats a bare symbol name as a first-class
handle (impact resolution, FTS symbol search, tier-3 join by name+scope).
A test double NAMED identically to the production symbol it stands in for
pollutes that namespace: `impact AgentOrchestrator` found 6 same-named
nodes and refused to guess. The doubles were forced into that name by a
stringly-typed gate (`type(orchestrator).__name__ == "AgentOrchestrator"`
in `Agent.__init__`) — fixed 2026-10-24 with a structural check, and the
five doubles renamed `AgentOrchestratorMock`.

## The rule

A test double is named for what it IS (`AgentOrchestratorMock`,
`FakeProvider`, `StubRepository`), never for what it REPLACES. Production
symbols keep their bare names findable.

## Completed renames (2026-10-24; guard allowlist now empty)

| Was | Files | Became |
|-----|-------|--------|
| `TestErrorHandling` | 13 | contextual pytest-group names (`TestAuditToolErrorHandling`, `TestWebSocketErrorHandling`, …) |
| `MockProvider` | 3 | `CircuitBreakerMockProvider` / `ToolStrategyMockProvider` / `SmartRoutingMockProvider` |
| `User` | 2 | `SampleUser` |
| `ModelFamily` | 1 | `MultiModelFamily` |
| `TestStatus` | 1 | `SwitcherStatusReporting` |

## Function-nested doubles (graph-name noise, not import shadows)

`Config`, `State`, `Agent`, `TaskType`, `ValidationResult`,
`SearchFilters`, `AnthropicProvider`, `RLConfigProvider` appear as classes
nested inside test functions. They cannot shadow imports; they DO create
graph nodes with colliding names. Handled at the UX layer
(`_resolve_target` prefers non-test definitions and lists candidates on
remaining ambiguity). Rename opportunistically when touching these files.

## Related stringly-typed gates (same bug class, no collision-forcing)

Sweep 2026-10-24 — the `Agent.__init__` gate was the ONLY one forcing
mock-naming. The rest are type-checks-by-string (brittle, queued as
hygiene, not collisions):

- `victor/tools/decorators.py:696` — `prop_type.__name__ == "Path"` (use
  `is`/isinstance)
- `victor/workflows/runtime.py:470` — `node.__class__.__name__ ==
  "HITLNode"` (isinstance)
- `victor/processing/native/context_fitter.py:147` — pyo3 exception by
  name (documented native-boundary constraint)
- `victor/providers/base.py` / `resilience.py` — exception classification
  by name against the failure-taxonomy registry (deliberate design: names
  are the cross-module contract)

## Finding worth its own follow-up

`OrchestratorProtocol` is aspirational: the concrete `AgentOrchestrator`
does NOT satisfy it at runtime (`model`, `provider_name`, `tool_budget`
are dynamic getattr surfaces, absent as class attributes). Either the
protocol shrinks to the real surface or the class grows the attributes —
until then `isinstance` gates against it would reject the real
orchestrator.
