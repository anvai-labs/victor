---
fep: 0030
title: "Decoupled Completion Judge (session-model-independent, calibrated, cheap-resident)"
type: Standards Track
status: Draft
created: 2026-08-02
modified: 2026-08-02
authors:
  - name: Vijaykumar Singh
    email: singhvjd@gmail.com
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

## Motivation — the evidence

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

## Design

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
stop reading `provider_context.model` and instead resolve the configured
`CompletionJudgeBackend`. New setting `agent.completion_judge` = `enhanced`
(default) | `classifier:<path>` | `llm:<model>@<endpoint>`. The
`completion_strategy=rubric` default-flip (PR-8) then becomes meaningful for
*every* session, not just llama-as-agent sessions.

## Phases

1. **Protocol + config + gate retarget.** Define `CompletionJudgeBackend`,
   `agent.completion_judge` setting, retarget `resolve_completion_strategy` to
   the backend. `enhanced` backend only — pure refactor, no behavior change
   (guard: streaming parity batteries green, default unchanged).
2. **Classifier + side-LLM backends.** Wire the `classifier` backend (resident
   ModernBERT/linear) and the `llm` backend (side endpoint). Each gated through
   the calibration harness before it may be selected.
3. **Default flip (absorbs PR-8).** With a calibrated decoupled judge available,
   `completion_strategy=rubric` becomes the default; prong-B end-to-end
   task-success A/B runs against the *decoupled* judge (the meaningful A/B the
   coupled design could not support), with the kill-switch and streaming parity
   guards.
4. **EVR-6 convergence.** The per-turn `TurnAuditor` (FEP-0008 Phase C) consumes
   the same classifier backend for prefix-only CONTINUE/ALARM — one cheap
   resident judge serves both completion gating and per-turn auditing.

## Open questions

- Model distribution: ship a calibrated ModernBERT judge as a release artifact
  (like `edge_classifier_v1.npz`), or train-on-first-use? The judge must be
  calibrated on *real* trajectories, and the current corpus (6 templates) is too
  narrow — the SWE-bench stratum is the prerequisite for a shippable general
  judge. Until then the classifier backend is opt-in/bring-your-own.
- Threshold policy per backend (the ModernBERT judge is probabilistic; the
  calibration gate scores binarized verdicts — see the
  `calibration_classifier_judge` binarization contract).

## Backwards compatibility

Default `agent.completion_judge=enhanced` + default `completion_strategy=enhanced`
= today's behavior exactly. Every new path is opt-in until Phase 3, which is
itself gated on a passing decoupled judge and its own A/B.
