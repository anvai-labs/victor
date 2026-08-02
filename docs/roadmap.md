# Victor Roadmap

> **Canonical roadmap.** Referenced by `docs/index.md`, `docs/README.md`, and the root
> `README.md`. Restored to version control 2026-07-02 — the previous `docs/roadmap.md`
> existed only as an untracked local file and was lost.
>
> Companion documents: [Vision](../VISION.md) · [Tech-debt register](tech-stack.md#technical-debt-register) ·
> [EVR backlog](architecture/evaluation-centric-runtime-backlog.md) ·
> [Release-readiness MVP](release-readiness-mvp.md) · [Architecture](architecture.md)

**Operating principle** (from the evaluation-centric runtime vision): an agent is a *model +
harness*, and the harness is what we can engineer. The roadmap therefore prioritizes closing the
evaluation loop and gating every change on it over adding new capabilities.

---

## Now — 0.7.2 release train (July 2026)

Ship 0.7.2 with evidence, not posture. Source: [release-readiness MVP blockers](release-readiness-mvp.md).

1. Fix `make check-dist` (points at `Formula/victor.rb`; formula lives at `scripts/homebrew/victor.rb`).
2. Prove clean packaging: build + `twine check` + wheel install/CLI smoke on Python 3.11/3.12.
3. Make release-critical CI gates blocking (`packages.yml`, `ci-integration.yml` currently allow failure).
4. Finalize release notes; decide external-vertical support level (preview vs blocking).
5. Verify advertised surfaces: CLI, API/MCP import, `victor ui`, Docker.
6. Docs governance (TD-18): this file stays committed; hygiene check that pointer targets resolve.

## Next — Q3 2026: close the evaluation loop (EVR P0 sequence)

The gate for defaulting-on any judge-based completion is ADR-011's reliability threshold — no
graduation without measured κ/α against human labels.

| Order | Item | ADR | State |
|-------|------|-----|-------|
| 1 | EVR-1 trajectory-eval harness | — | Shipped (machinery) |
| 2 | EVR-2 LLM-judge reliability gate — run the κ/α validation | ADR-011 | **DONE.** Verifier gold validated (annotator↔verifier κ=1.0, run 12); llama3.3:70b graduates (scripted α=1.000 run 10; real α=0.878 refactor 1.000 run 13); gemma4:31b demoted (0.694 at n=96). Judge identity pinned in the flag wiring (`DEFAULT_CALIBRATED_JUDGE_MODELS`=llama3.3:70b). See [FINDINGS](../benchmarks/judge_calibration/FINDINGS.md) runs 10–14 + `docs/architecture/judge-independence-experiments.md` |
| 3 | EVR-3 rubric completion evaluator — must match-or-beat `EnhancedCompletionEvaluator` before becoming default | ADR-009 | **Prong A (judge quality) DONE**: on the run-12 real pack rubric-llm (llama3.3:70b) α=0.878 vs `enhanced` α=−0.837 — beats by Δα=1.72 ([evr3-parity-results](architecture/evr3-parity-results.md)). Prong B (end-to-end task-success A/B) remains before the default flip |
| 4 | EVR-4 effect-grounded completion gate | ADR-010 | Shipped opt-in (`victor/framework/effect_gate.py`; `effect_gated_completion` / `VICTOR_EFFECT_GATED_COMPLETION`, default off pending flag-graduation gate; A/B graduation battery not yet built) |
| 5 | EVR-5 regression-gated harness acceptance oracle | ADR-012 | **Shipped** (ADR-012 Accepted; `victor/evaluation/acceptance_oracle.py` + `htir.py`, promotion-gated via `tests/integration/streaming/test_acceptance_oracle_gate.py`) |
| 6 | EVR-6 online per-turn auditor (`TurnAuditor`, prefix-only CONTINUE/ALARM) | FEP-0008 Phase C | P1 — next after the P0 chain closes; measure offline against HTIR oracle traces before loop wiring |

In parallel, the high-priority debt band: TD-4 secrets, TD-7 onboarding, TD-1 API decomposition,
TD-6 SWE-bench publication, TD-14 orchestrator ratchet, TD-17 flag-graduation policy.

## Next — Q3/Q4 2026: durable code memory (correlated CPG)

Product bet #4 in [VISION.md](../VISION.md): one entity = relational row + graph node + vector,
addressed by a single stable oid.

- Foundation shipped: `victor-codegraph` extraction (ADR-014), phased core adoption (ADR-015,
  Phase 1 live), stable line-independent `symbol_oid` (ProximaDB ADR-044, victor-codegraph 0.1.2).
- Remaining: ADR-015 later phases; TD-11 `ProximaGraphStore`; TD-12 embedding↔node correlation by
  shared oid (retire `graph_node.embedding_ref` dual-write); TD-13 Tier-A/Tier-B CCG split.
  Design: [ProximaDB as the CCG Backend](architecture/proximadb-codegraph-backend.md).

## Later — directional horizons (from VISION.md)

- **3–6 months**: contract-first extension authoring; productize observability beyond EventBridge
  and the prototype dashboard (TD-5); published benchmark evidence (TD-6); EVR P1–P2 (online
  prefix auditing, judge-validation expansion, EVR-7 credit→learner loop).
- **6–12 months**: default open-source platform layer for typed, multi-provider, multi-surface
  agent systems; external-vertical ecosystem; operations-ready deployment patterns; multi-tenant
  code-memory service.

## Governance

- Completed quarters move to CHANGELOG.md; this file only carries live and future work.
- Every roadmap item cites its tracker (TD-*, EVR-*, ADR, or release-blocker list) — no orphan bullets.
- Update cadence: at each release cut and each quarter boundary, whichever comes first.
