---
fep: 0030
title: "Decoupled Completion Judge (session-model-independent, calibrated, cheap-resident)"
type: Standards Track
status: Draft
created: 2026-08-02
modified: 2026-08-03
authors:
  - name: Vijaykumar Singh
    email: vijay@anvaiops.com
    github: vjsingh1984
reviewers: []
discussion: https://github.com/anvai-labs/victor/discussions/0030
---

# FEP-0030: Decoupled Completion Judge

## Summary

Give the in-loop completion decision a judge that is **independent of the session
model**. Today the rubric completion judge *is* the session model
(`turn_execution_runtime._build_rubric_complete_fn` uses `provider_context.model`),
which means (a) it is a self-judge, and (b) after the ADR-011 pin
(`DEFAULT_CALIBRATED_JUDGE_MODELS`), `completion_strategy=rubric` engages **only**
when the session model is itself a calibrated judge (llama3.3:70b) and silently
falls back to `enhanced` for every other session. This FEP introduces a
`CompletionJudgeBackend` seam so any session — regardless of its chat model — can
gate completion with a calibrated judge: a cheap resident classifier
(ModernBERT), a side LLM judge, or the algorithmic `enhanced` fallback.

## Motivation

The judge-independence experiments (2026-08-02,
[judge-independence-experiments.md](../docs/architecture/judge-independence-experiments.md),
FINDINGS runs 12–14) measured every option on the same 96 real trajectories:

| Judge | Independence from agent | α (real pack) |
|---|---|---|
| `enhanced` (in-loop self-assessment) | none | −0.837 |
| Fresh-context **same model** (current rubric wiring) | context only | 0.469 |
| Cross-model gemma4:31b | context + weights | 0.694 (demoted) |
| Cross-model **llama3.3:70b** | context + weights | **0.878** |
| **ModernBERT judge** (real-pos + boundary-neg) | weights + grounded | **0.928** |

Two structural conclusions:

1. **Context isolation is necessary but insufficient.** Fresh context recovers
   1.3 α over in-loop self-assessment, but weight-level correlation still costs
   ~0.4 α versus an independent judge. A model asked to verify its own work
   shares its own blind spots.
2. **Weight independence is what graduates**, and it need not be expensive: a
   149M ModernBERT reaches α=0.928 — *better than the 70B* — at CPU cost, once
   trained on real-styled positives and boundary negatives.

Consequence: the value of `completion_strategy=rubric` is currently trapped.
The default-flip (program PR-8) is technically safe (the pin makes it a no-op
for non-calibrated sessions) but nearly worthless, because virtually no one runs
llama3.3:70b *as their agent*. The payoff requires decoupling the judge from the
session model.

## Proposed Change

### The seam

A `CompletionJudgeBackend` protocol — `judge(prompt, transcript, workspace) -> bool`
— resolved once per session from config, independent of the provider stack. Three
first-party backends:

- **`classifier`** — a resident local model (ModernBERT / the `victor/ml` linear
  head), loaded via the existing `calibration_classifier_judge` loaders. CPU,
  ~no marginal latency, no network. The EVR-6 per-turn auditor uses the *same*
  backend (this is what makes prefix-only per-turn auditing economically real).
- **`llm`** — any calibrated model on a side endpoint (e.g. llama3.3:70b via a
  judge-only Ollama handle), decoupled from the session provider. Reuses the
  rubric prompt (`render_judged_content`).
- **`enhanced`** — the current algorithmic evaluator, the always-available
  fallback.

The `edge_model.py` subsystem already establishes the decoupled-small-model
pattern (a separate model for micro-decisions); this FEP reuses that shape rather
than inventing a parallel one.

### Calibration gate generalizes

ADR-011's pin moves from "the *session model* is in the calibrated set" to "the
*configured judge backend* is calibrated." Any backend — classifier or side-LLM —
passes through the identical κ/α harness (`judge_calibration_harness`,
`--replay-pack` on a real pack) and is enabled only at α ≥ 0.7 overall and per
n≥16 family. An uncalibrated or unavailable backend falls back to `enhanced`,
loudly — the existing `resolve_completion_strategy` contract, retargeted from
"session model" to "judge backend."

### Where it wires

`turn_execution_runtime._build_rubric_complete_fn` and `chat_stream_executor`
resolve the configured judge backend (`agent.completion_judge`) instead of
reading `provider_context.model` directly. New setting `agent.completion_judge`
= `session-model` (default) | `enhanced` | `llm:<model>@<endpoint>` |
`classifier:<path>`. The `completion_strategy=rubric` default-flip (PR-8) then
becomes meaningful for *every* session, not just llama-as-agent sessions.

## Benefits

- **Rubric completion becomes usable for every session**, not only ones whose
  chat model is itself a calibrated judge — unlocking the measured Δα≈1.7
  advantage of rubric over `enhanced` on real trajectories (EVR-3).
- **Weight-independent verification.** A separate judge does not share the
  agent's blind spots (the fresh-context self-judge measured only 0.469).
- **Cheap resident option.** The classifier backend gates completion (and the
  EVR-6 per-turn auditor) at CPU cost, no network, no per-decision token spend.
- **One calibration discipline.** Every backend passes the same κ/α gate; the
  ADR-011 pin generalizes without a parallel mechanism.

## Drawbacks and Alternatives

- **Config surface.** A new `agent.completion_judge` value to reason about;
  mitigated by a default (`session-model`) that reproduces today's behavior.
- **Train/serve skew for the classifier.** A classifier calibrated on the
  synthetic corpus is *not* validated for live completion decisions, which
  render the actual project workspace (out-of-distribution). This is why the
  classifier backend is opt-in/bring-your-own until real-distribution
  calibration exists (the SWE-bench stratum).
- **Alternative — keep the self-judge and improve its prompt.** Rejected:
  weight-level correlation is structural (runs 12/14), not a prompting problem.
- **Alternative — always require a side LLM judge.** Rejected: it forces a
  second model / endpoint on every session; the resident classifier is the
  cheap path, and `enhanced` remains the zero-dependency fallback.

## Implementation Plan

1. **Protocol + config + gate retarget** (landed, #844). Define the backend seam,
   `agent.completion_judge` setting, retarget `resolve_completion_strategy` to the
   resolved judge. `session-model`/`enhanced` only — no behavior change.
2. **Side-LLM backend** (landed, #845). `llm:<model>@<endpoint>` builds a judge on
   a side provider; the pin checks the *judge* model. This is what lets an
   uncalibrated chat session gate rubric with a calibrated side judge.
   - **2b. Classifier backend.** `classifier:<path>` (resident ModernBERT/linear)
     via a direct-verdict loop seam. Gated on real-distribution calibration (the
     SWE-bench stratum, converter landed #848) before it is validated for live use.
3. **Default flip (absorbs PR-8).** With a calibrated decoupled judge available,
   `completion_strategy=rubric` becomes the default; the prong-B end-to-end
   task-success A/B runs against the *decoupled* judge (the meaningful A/B the
   coupled design could not support), with the kill-switch and streaming parity
   guards.
4. **EVR-6 convergence.** The per-turn `TurnAuditor` (FEP-0008 Phase C) consumes
   the same classifier backend for prefix-only CONTINUE/ALARM — one cheap
   resident judge serves both completion gating and per-turn auditing.

## Migration Path

No migration required. `agent.completion_judge` defaults to `session-model`,
which reproduces the pre-FEP resolution exactly; `completion_strategy` remains
`enhanced` by default. Operators adopt a decoupled judge by setting
`agent.completion_judge=llm:<model>@<endpoint>` (or, once validated,
`classifier:<path>`). The Phase 3 default flip is itself gated on a passing
decoupled judge and its own A/B, and ships with an env kill-switch.

## Compatibility

Default `agent.completion_judge=session-model` + default
`completion_strategy=enhanced` = today's behavior exactly. Every new path is
opt-in until the Phase 3 default flip, which is separately gated. The change is
internal to the runtime completion path (`victor/agent/services`), not a
public framework-API break; existing gate tests pass unchanged.

## Unresolved Questions

- **Model distribution.** Ship a calibrated ModernBERT judge as a release
  artifact (like `edge_classifier_v1.npz`), or train-on-first-use? The judge
  must be calibrated on *real* trajectories, and the current corpus (6 templates)
  is too narrow — the SWE-bench stratum is the prerequisite for a shippable
  general judge. Until then the classifier backend is opt-in/bring-your-own.
- **Live-input fidelity.** The classifier is calibrated on corpus renders; the
  live loop renders the real workspace. A live-loop fidelity validation (against
  the SWE-bench stratum) gates classifier live-wiring.
- **Threshold policy per backend.** The ModernBERT judge is probabilistic; the
  calibration gate scores binarized verdicts (the `calibration_classifier_judge`
  binarization contract) — the operating threshold may warrant per-backend tuning.

## References

- ADR-011 — LLM-judge reliability gating (the calibration pin this FEP generalizes).
- ADR-009 — Rubric-based completion evaluation.
- FEP-0008 — Evaluation-centric completion (EVR sequence; Phase C TurnAuditor).
- `docs/architecture/judge-independence-experiments.md` — the runs 12–14 evidence.
- `docs/architecture/evr3-parity-results.md` — rubric vs enhanced (Δα≈1.7).
- `benchmarks/judge_calibration/FINDINGS.md`,
  `benchmarks/judge_training/FINDINGS.md` — judge and trained-judge findings.
- PRs #844 (seam), #845 (side-LLM backend), #848 (SWE-bench stratum converter).
