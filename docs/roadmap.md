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

## Now — v0.9.0 release closeout (August 2026)

v0.9.0 shipped on 2026-08-20. The `develop` promotion passed the complete required matrix, and the
tag-triggered release workflow published the Python package, native wheels, binaries, VS Code
extension, checksums, SBOM, GitHub Release, and Docker image. Source:
[release-readiness closeout](release-readiness-mvp.md).

1. **DONE — release evidence:** promotion CI and the tag-triggered release workflow completed
   successfully; public PyPI and Docker registry entries were verified.
2. **DONE — support-level enforcement:** core tests, package/CLI smoke, integration, vertical
   compatibility, artifact builds, and publication block stable releases. TestPyPI is optional for
   stable tags; Trivy remains advisory.
3. **NEXT — security follow-up:** triage the v0.9.0 Trivy SARIF findings in GitHub Security and fix or
   explicitly accept each actionable critical/high dependency or image finding.
4. **DONE — public-install evidence:** clean Python 3.11 and 3.12 environments installed
   `victor-ai==0.9.0` from PyPI, imported `victor`/`Agent`, reported version 0.9.0, and rendered CLI
   help independently of the checkout and release artifacts.
5. **DONE — modular Rust distribution:** annotated tag `rust-v0.8.0` published
   `victor-protocol`, `victor-state`, `victor-tools`, and `victor-edge` to crates.io in dependency
   order. All four registry records and checksums were verified, docs.rs built successfully, and a
   clean `cargo install victor-edge --version 0.8.0 --locked` smoke passed.
6. **ONGOING — docs governance (TD-18):** keep this file committed and verify that canonical pointer
   targets resolve at every release cut.

## Next — Q3 2026: close the evaluation loop (EVR P0 sequence)

The gate for defaulting-on any judge-based completion is ADR-011's reliability threshold — no
graduation without measured κ/α against independent labels on the shipping distribution.

| Order | Item | ADR | State |
|-------|------|-----|-------|
| 1 | EVR-1 trajectory-eval harness | — | Shipped (machinery) |
| 2 | EVR-2 LLM-judge reliability gate — run the κ/α validation | ADR-011 | **DONE — gate produced a NO-GO.** Verifier gold was validated (annotator↔verifier κ=1.0, run 12) and llama3.3:70b passed scripted/calibration-corpus packs, but failed the later SWE-bench-lite shipping-distribution re-gate (α=0.26). The identity pin remains an opt-in safety/fallback guard; it does not authorize a default. See [FINDINGS](../benchmarks/judge_calibration/FINDINGS.md) and `docs/architecture/judge-independence-experiments.md`. |
| 3 | EVR-3 rubric completion evaluator — must match-or-beat `EnhancedCompletionEvaluator` before becoming default | ADR-009 | **NO-GO / default stays `enhanced`.** The calibration-corpus result was positive (llama3.3:70b α=0.878 vs enhanced −0.837), but the later in-container-verified SWE-bench-lite re-gate failed on the shipping distribution (llama α=0.26; gemma α=−0.52). Prong-B verifier-backed A/B machinery now exists (`victor/evaluation/completion_strategy_ab.py`) for future re-evaluation, but cannot override the failed reliability prerequisite. See [evr3-parity-results](architecture/evr3-parity-results.md) and [FINDINGS](../benchmarks/judge_calibration/FINDINGS.md). |
| 4 | EVR-4 effect-grounded completion gate | ADR-010 | Shipped opt-in (`victor/framework/effect_gate.py`; `effect_gated_completion` / `VICTOR_EFFECT_GATED_COMPLETION`, default off pending flag-graduation gate; A/B graduation battery not yet built) |
| 5 | EVR-5 regression-gated harness acceptance oracle | ADR-012 | **Shipped** (ADR-012 Accepted; `victor/evaluation/acceptance_oracle.py` + `htir.py`, promotion-gated via `tests/integration/streaming/test_acceptance_oracle_gate.py`) |
| 6 | EVR-6 online per-turn auditor (`TurnAuditor`, prefix-only CONTINUE/ALARM) | FEP-0008 Phase C | **Shipped opt-in; default OFF.** Deterministic + pinned edge-judge wiring exists (`per_turn_auditor.py`, `edge_turn_judge.py`). The [offline HTIR-oracle decision gate](architecture/evr6-auditor-gate.md) and digest-pinned evidence producer now exist (`turn_auditor_eval.py`, `turn_auditor_evidence.py`); next freeze an independently labelled real-distribution pack and run it before broader rollout. An offline PASS is necessary, not authority to flip the default. |

In parallel, the high-priority debt band: TD-4 secrets, TD-7 onboarding, TD-1 API decomposition,
TD-6 SWE-bench publication, TD-14 orchestrator ratchet, TD-17 flag-graduation policy.

## Next — Q3/Q4 2026: durable code memory (correlated CPG)

Product bet #4 in [VISION.md](../VISION.md): one entity = relational row + graph node + vector,
addressed by a single stable oid.

- Foundation shipped: `victor-codegraph` extraction (ADR-014), phased core adoption (ADR-015,
  Phase 1 live), stable line-independent `symbol_oid` (ProximaDB ADR-044, victor-codegraph 0.1.2).
- Shipped behind the per-repo flag: TD-12 one authoritative ProximaRecord for node props + vector +
  staleness under the shared oid, and TD-13's local Tier-A/Tier-B routing boundary.
- Remaining: ADR-015 later phases; TD-11 native-wheel/live-parity/default-graduation work; replacing
  Tier B's local fragment implementation with Proxima PAX/columnar fragments.
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
