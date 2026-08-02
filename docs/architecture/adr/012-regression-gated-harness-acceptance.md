# ADR-012: Regression-Gated Harness Acceptance

## Metadata

- **Status**: Accepted
- **Date**: 2026-06-21
- **Decision Makers**: Vijaykumar Singh
- **Related ADRs**: 009, 010, 011
- **Related**: [FEP-0008](../../../feps/fep-0008-evaluation-centric-completion.md), FEP-0007 (unified loop), [Vision](../vision-evaluation-centric-runtime.md)

## Context

FEP-0007 produced two offline batteries — `tests/integration/streaming/test_run_stream_parity.py`
(run≡stream) and `test_streaming_parity.py` (characterization) — used informally as the cutover gate.
The 2026 harness literature elevates exactly this mechanism to a first-class discipline:
Harness-Bench (`2605.27922`) frames capability as a property of the *(model, harness-config)* pair;
HarnessFix (`2606.06324`) and Self-Harness (`2606.09498`) show that **+15–50% / +21pp** gains come
*specifically* from making every harness/prompt edit pass a **regression-aware acceptance gate** on
held-out tasks. HarnessFix's ETCLOVG taxonomy (Execution / Tooling / Context-Memory /
Lifecycle-Orchestration / Observability / Verification / Governance) and HTIR trace normalization
give a vocabulary for attributing failures to a harness *layer*.

## Decision

Formalize Victor's batteries into a **regression-aware harness-edit acceptance oracle**: any change
to prompts (GEPA/MIPROv2 outputs), the loop, completion strategy, or recovery must pass the parity +
characterization batteries (byte-stable-or-justified) **plus** the trajectory-eval harness
(FEP-0008 Phase E) with **no unacceptable regression**, and eval results are reported at
*(model, harness-config)* granularity with confidence intervals. Adopt **HTIR-style trace
normalization** (Role / ExecutionStatus / artifact-effect) as the canonical input to recovery and
failure attribution, tagged by ETCLOVG layer.

## Rationale

- It is the load-bearing mechanism behind the literature's gains and behind FEP-0007's safe cutover —
  make it explicit and required rather than ad hoc.
- *(model, harness-config)* granularity prevents misattributing harness wins/losses to the base model
  (Harness-Bench's central finding; harness is model-specific per Self-Harness).
- HTIR + ETCLOVG give recovery and GEPA a structured failure vocabulary instead of blind retry.
- **Pros**: every loop/prompt edit is auto-gated; enables future Self-Harness-style self-improvement
  safely. **Cons**: battery runtime cost in CI (already ~2–4 min each; parallelize/shard).

## Consequences

- **Positive**: prompt/loop/completion evolution (FEP-0008, GEPA) cannot silently regress; failures
  are attributable to a harness layer; results are honestly scoped per (model, config).
- **Negative**: CI time and maintenance of the held-out trajectory set; need to keep "justified"
  characterization deltas disciplined (require written rationale, as FEP-0007 did).
- **Neutral**: runtime behavior is unchanged; this is an evaluation/CI discipline.

## Implementation

1. Promote the batteries to a named acceptance oracle invoked by GEPA/RL promotion and FEP changes.
2. Add HTIR trace normalization (Role/Status/effect, ETCLOVG layer) as the recovery + attribution
   input.
3. Report eval scores with confidence intervals at *(model, harness-config)* granularity; wire into CI
   as the promotion gate. Lays groundwork for a later Self-Harness-style self-improvement loop
   (vision-tier).

**Done (2026-08-01, EVR-5).** Landed as the acceptance-oracle + HTIR decision layer plus a
promotion-gated battery:

- Prong 1 — `victor/evaluation/acceptance_oracle.py::HarnessAcceptanceOracle` renders one
  accept/reject verdict at *(model, harness-config)* granularity, gating a candidate against a
  baseline on the trajectory battery (CI-significance regression test) and the FEP-0007
  characterization battery (byte-stable-or-justified).
- Prong 2 — `victor/evaluation/htir.py` normalizes `AgenticExecutionTrace` into Role/Status/
  artifact-effect steps tagged by ETCLOVG layer (`failures_by_layer()` is the input for
  layer-targeted recovery; wiring that into `RecoveryService` is a tracked follow-up).
- Prong 3 — `tests/integration/streaming/test_acceptance_oracle_gate.py` drives the scripted QA
  battery through the real loop and enforces the oracle in the develop→main promotion battery
  (`ci-integration.yml`); `make eval-accept` runs it, `EVR5_RECORD_BASELINE=1` re-records.

## Revision History

| Date | Version | Changes | Author |
|------|---------|---------|--------|
| 2026-06-21 | 1.0 | Initial ADR — regression-gated harness-edit acceptance oracle (Proposed) | Vijaykumar Singh |
| 2026-08-01 | 1.1 | EVR-5 landed (acceptance oracle + HTIR + promotion gate); Status Proposed→Accepted | Vijaykumar Singh |
