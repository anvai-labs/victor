---
fep: "0012"
title: "Shipped RL-trained edge classifier (replace the Ollama edge default)"
type: Standards Track
status: Draft
created: 2026-06-30
modified: 2026-07-06
authors:
  - name: Vijaykumar Singh
    email: vijay@anvaiops.com
    github: vjsingh1984
reviewers: []
discussion: https://github.com/anvai-labs/victor/discussions/0012
---

# FEP-0012: Shipped RL-trained edge classifier

## Summary

FEP-0001's edge model defaults to `ollama/qwen3.5:2b` for ~14 micro-decisions
(task-type, completion, loop-detection, tool-necessity, prompt-focus, stage,
…). For client/edge installs that means a separate Ollama server + a ~2.7 GB
model pull just for micro-routing. It is a **soft** dependency (graceful
heuristic fallback exists), and observed logs show **~84 % of decisions already
run on heuristics** vs ~14 % on the LLM — so the Ollama edge model is barely
carrying weight.

This FEP replaces the Ollama edge **default** with a **shipped classic-ML
classifier** trained on the collected decisions + RL outcomes, hosted in the
native/numpy layer (no new heavy runtime dep). Ollama/LLM remains an
**optional auto-detected upgrade**. It removes the 2.7 GB client burden, likely
*improves* micro-decision quality (classic ML beats a 2B model on narrow
classification), and turns 20K+ logged decisions + RL outcomes into a real
asset.

The design ships a **universal pretrained baseline** for all users (solves
cold-start, works out of the box) plus an **optional per-project RL delta**
learned on-device from each project's own outcomes (never uploaded). It also
closes the **data-model gap** that today blocks reward-weighted training —
micro-decisions are logged without correlation ids, so they cannot be joined to
outcomes; this FEP stamps the correlation spine and adds the junction tables
that make reward-supervised learning possible.

## Motivation

### Problem Statement

- **Client/edge friction:** requiring Ollama + a 2.7 GB model for micro-routing
  is heavy for a client/edge install of Victor.
- **Low utilization:** the LLM edge model serves only ~14 % of decisions; the
  rest already fall back to keyword heuristics.
- **Noisy training data:** `decisions.jsonl` (20K+ records) is mostly
  heuristic-sourced with many null/0-confidence labels, and — critically — is
  **not joinable to outcomes** (no session/turn id), so it cannot today support
  reward-weighted training.

### Goals

1. A shipped, versioned classifier artifact (universal baseline) that runs
   offline with zero external dependency (numpy core; sklearn dev-only).
2. Reward-weighted training (learn from outcomes, not imitation of a noisy log).
3. Optional per-project RL personalization (on-device delta, never uploaded).
4. Non-breaking: Ollama/LLM edge remains an opt-in upgrade; graceful fallback to
   heuristics throughout.

### Non-Goals

- Generative decisions (`multi_skill_decomposition`, `continuation_action`,
  `compaction`) — inherently generative, stay LLM/heuristic.
- A distilled-transformer model (linear hashing is v1).
- Federated aggregation / uploading deltas across users (privacy + infra — later).
- Changing the **main** model default (the user's choice, not the framework's).

## Proposed Change

### Scoping: universal baseline + per-project RL delta

**Hybrid — not purely global, not per-user.**

- **Universal baseline** (released for **all** users, in the package): a
  pretrained classifier on *aggregate, cleaned* decision+outcome data. Solves
  cold-start.
- **Per-project RL delta** (optional, **on-device**): a small reward-weighted
  overlay learned from that project's own outcomes, stored in the **project DB**
  (`.victor/project.db`), never uploaded.
- **Not per-user-global:** a user's cross-project data is less useful for the
  prompt/project-sensitive decisions and raises privacy concerns.

**Decision taxonomy split** — which decisions the classifier owns:

| Universal (classifier-owned) | Project/prompt-sensitive (heuristic / +features) | Generative (out of scope) |
|---|---|---|
| task_type, task_completion, tool_necessity, loop_detection, error_classification, question_classification, intent_classification | prompt_focus, stage_detection, skill_selection, tool_selection | multi_skill_decomposition, continuation_action, compaction |

### Architecture

`caller → TieredDecisionService → local_classifier tier → LocalClassifierDecisionService`
implementing `LLMDecisionServiceProtocol` (`victor/agent/services/protocols/
decision_service.py`). The service: (1) feature-extracts the decision context
into a fixed-size hashed n-gram vector (hashing trick, `2^18` dims, versioned
`feature_spec_version`); (2) scores `W_universal·x + α·W_delta·x + b` per
DecisionType; (3) calibrates confidence; (4) if confidence < τ, returns the
heuristic result. Weights host: `victor_native` Rust (sub-ms) with a pure-numpy
fallback. The delta host is the project DB.

The LLM edge tier remains in the fallback chain — if Ollama is present and a
decision is below τ, the tiered service can still escalate to it (opt-in boost).

### Data model (the unblock)

The current blocker is that decision records carry no session/turn id. The
correlation spine already exists (`victor/core/context.py`:
`session_id`/`turn_id`/`request_id`); `rl_outcome` + `usage.jsonl` already
carry session/turn + success. This FEP closes the last mile:

- **Enrich decision records** (`chain.py`) with `decision_id`, `session_id`,
  `turn_id`, `turn_idx`, `trace_id`, `model_version`, `feature_spec_version`,
  `feature_digest`.
- **`decision_log` + `decision_outcome`** in the **global** DB (consistent with
  `rl_outcome`; enables cross-project training). The junction carries the
  session outcome (`success`, `quality_score`, `segment_rewards`) and the
  per-decision `attributed_reward` via existing credit assignment
  (`trace_to_credit`, `credit_assignment`).
- **`local_classifier_delta`** in the **project** DB: sparse reward-weighted
  overlay `(decision_type, feature_hash) → weight, samples, sum_reward`, top-K
  bounded per type, L2-decayed.
- **Feature provenance:** `feature_spec_version` pins the hasher config so a
  model trained on spec v1 only consumes v1 features; spec changes invalidate
  old deltas (migrated/zeroed).

### Training pipeline (offline, dev-side → universal artifact)

Clean (drop heuristic-noise labels) → join on `decision_outcome` → featurize
(versioned spec) → reward-supervised train (logistic / GBM) → Platt-calibrate →
A/B vs the LLM edge model (held-out, by-session split) → export compressed
sparse `W` + bias + label maps → `victor/models/edge_classifier_v{N}.npz`.
`sklearn`/`torch` are **dev-only** extras; runtime needs only numpy.

### RL personalization (online, per-project)

On session end / periodic flush: join the turn's decisions → session outcome →
reward `r`; reward-weighted logistic-SGD on the touched feature hashes writes
`local_classifier_delta` (bounded, L2-decayed so the universal model
re-asserts). Pure-numpy/Rust; local only.

## API Changes

- New `LocalClassifierDecisionService` implementing `LLMDecisionServiceProtocol`
  (drop-in for the edge model).
- New `local_classifier` tier in `TieredDecisionService`; becomes the default
  when the artifact is present.
- `DecisionResult.source` gains `local_classifier` / `local_classifier+delta`.
- Settings: `edge_classifier_version`, `feature_spec_version`,
  `USE_LOCAL_CLASSIFIER`, `local_learning_enabled`.

## Compatibility

- **Breaking:** No. Ollama/LLM edge remains available; heuristic fallback
  intact. The default *tier* shifts from `edge` to `local_classifier` when the
  artifact is present.
- **Minimum Python:** 3.10 (unchanged). **New runtime deps:** none (numpy
  already core).

## Benefits

### For framework users (client/edge installs)

- **No Ollama requirement for the edge layer:** micro-decisions resolve in
  sub-milliseconds from a bundled artifact instead of a 4 s timeout against a
  2.7 GB local model. Dramatically lowers the install/run footprint.
- **Faster, more consistent decisions:** a calibrated linear classifier is
  deterministic and sub-ms, eliminating the latency variance and timeout-driven
  auto-disable that plagues the Ollama edge path today.
- **Smooth degradation:** classifier → heuristic → (optional) LLM edge. Never
  worse than the status quo.

### For the ecosystem

- **A real ML asset from existing data:** 20K+ logged decisions + RL outcomes
  become supervised training data instead of a noisy log, continuously
  improving the shipped baseline.
- **Privacy-preserving personalization:** each project's RL delta stays local in
  `project.db`; nothing is uploaded. The universal baseline is the only
  universally-released artifact.
- **No new runtime dependency:** numpy is already core; sklearn/torch stay
  dev-only. The inference path also lands in the existing native Rust layer.

## Drawbacks and Alternatives

- **Drawback — a shipped baseline can be stale** relative to a fresh LLM, so it
  may lag new task patterns until the next artifact release. **Mitigation:**
  versioned artifacts with optional CDN updates between releases, plus the
  per-project RL delta that corrects locally and immediately from observed
  outcomes, and the retained LLM edge tier as an opt-in escalation for
  low-confidence decisions.
- **Drawback — linear hashing caps expressiveness** on decisions that depend on
  long-range structure (e.g. multi-step intent). **Mitigation:** scope the
  classifier to the narrow classification decisions where it wins; leave
  structural/generative decisions on the heuristic/LLM path.
- **Alternative — keep Ollama as the default:** rejected because the 2.7 GB
  client/edge burden and a separate server are unjustified when only ~14 % of
  decisions use the LLM today. Ollama stays as an optional upgrade, not the
  default.
- **Alternative — a distilled transformer (e.g. a few-MB fine-tuned model):**
  deferred to v2. It adds an inference runtime (ONNX) and training complexity
  for marginal gain on the narrow classification tasks; revisit only if the
  linear classifier underperforms on a specific decision type.
- **Alternative — retrain locally per project:** rejected because a single
  project rarely has enough data and it would require sklearn/torch at runtime.
  The reward-weighted delta overlay achieves personalization cheaply, in
  pure-numpy/Rust, without a heavy runtime.

## Unresolved Questions

- **α (delta blend weight) and τ (confidence gate) defaults:** tune via the A/B
  harness; proposed α=0.3, τ=0.6 to start.
- **CDN artifact distribution vs bundle-only:** proposed bundle-by-default,
  CDN as a later enhancement.

## Implementation Plan

1. **FEP-0012 doc** — this design, status Draft, 14-day review.
2. **Data model for RL (unblocks everything):** enrich decision records
   (`chain.py` stamping + `decision_log`) with the correlation spine and
   provenance; add `decision_outcome` (global) + `local_classifier_delta`
   (project) tables with a versioned migration; wire per-decision credit
   attribution by reusing `trace_to_credit` / `credit_assignment`.
3. **Training pipeline** (`scripts/train_edge_classifier.py` + `victor/ml/`,
   dev-only): clean → join on `decision_outcome` → featurize under a versioned
   feature spec → reward-supervised train (logistic / GBM) → Platt-calibrate →
   export the compressed artifact.
4. **Inference host** (`victor/processing/native/classifier.py` + Rust):
   `LocalEdgeClassifier` loads the artifact and scores the versioned hashed
   features, with a pure-numpy fallback.
5. **Service + default flip:** `LocalClassifierDecisionService` implementing
   `LLMDecisionServiceProtocol`; new `local_classifier` tier in
   `TieredDecisionService`; bootstrap factory; flip the default tier when the
   artifact is present (LLM edge becomes opt-in escalation).
6. **RL delta (online):** reward-weighted bandit overlay writing
   `local_classifier_delta` (bounded, L2-decayed), pure-numpy/Rust. **— Shipped
   (Phase 6).** On each resolved session (`record_session_outcome`),
   `victor/agent/decisions/local_delta.py` writes a per-label softmax-cross-entropy
   update toward the observed reward bucket (gradient scaled by where the
   universal model is wrong), L2-decays and top-K bounds the overlay, and the
   service blends `α·delta·x` into `predict()` via a TTL-cached `load_delta`.
   Local-only (project DB), gated by `local_learning_enabled`. Two refinements
   vs. the original scalar design: the table is **per-label** `(decision_type,
   feature_hash, label, feature_spec_version)` (schema v8→v9, DROP+CREATE — the
   table had never been written) because the shipped heads are multi-class
   (`task_completion` = fail/partial/pass), and it carries a `feature_spec_version`
   guard so a future hasher-spec bump can't blend stale hashes.
7. **Validation:** A/B the classifier vs the LLM edge model on held-out
   decisions; ship only at parity on the universal decision types.

> **task_completion consumption wiring (mapper + source gate) — landed behind a
> flag.** The shipped `task_completion` head predicts reward buckets
> (`pass/partial/fail`). Two earlier gaps made the classifier (and the Phase 6
> delta) a no-op for loop-stopping: (a) the service mapper expected labels
> `complete/true/yes` and (b) `task_completion.py` only consumed
> `decision.source == "llm"`. Both are fixed: the mapper is now reward-bucket
> aware (`pass`→complete, `fail`→incomplete, `partial`→defer; also accepts legacy
> labels), and both completion sites accept `source == "local_classifier"` when
> `DecisionServiceSettings.local_classifier_completion_signal` is on. **The flag
> defaults OFF** — trusting the classifier to stop the loop is a behavioral
> change that wants Phase-7 A/B validation before it defaults on. With it off,
> behavior is identical to pre-wiring (the mapper fix alone has no effect because
> the source gate still excludes `local_classifier`).

## Migration Path

This is a **non-breaking, additive** change with a staged default flip:

1. **Data model lands first** (this PR): decision records gain correlation ids;
   the three new tables + v8 migration ship. No behavior change — existing
   decisions.jsonl continues to accumulate, now with joinable fields.
2. **Artifact + inference land behind a flag:** `USE_LOCAL_CLASSIFIER` defaults
   off until the A/B parity gate passes. The Ollama edge model remains the
   active default during this window.
3. **Default flip:** once the classifier meets parity on the universal decision
   types, `USE_LOCAL_CLASSIFIER` defaults on (when the artifact is present) and
   the `local_classifier` tier becomes the default; the LLM edge tier stays
   available as opt-in escalation.
4. **Rollback:** disabling `USE_LOCAL_CLASSIFIER` (or removing the artifact)
   instantly restores the Ollama edge default; heuristic fallback is always
   present beneath both.

### Deprecation Timeline

- `v0.9.x`: data model + classifier behind `USE_LOCAL_CLASSIFIER` (off).
- `v1.0.x`: default flips to the shipped classifier; Ollama edge becomes opt-in
  (not removed — retained for users who want the LLM boost).

## Phase 7 Validation Run Plan

The parity gate (`victor ml validate`, GREEN) already ships the artifact at
held-out accuracy. Phase 7 validates the **loop-level** behavior — does trusting
the classifier (and its delta) in real agent runs help or hurt? Three A/B knobs
are now env-overridable so each run is a one-liner (no source edits):

| Knob | Env var | Default | What it tests |
|---|---|---|---|
| `decision_backend` | `VICTOR_DECISION_BACKEND` | `auto` | `local_classifier` vs `edge` (LLM) vs `heuristic` for micro-decisions |
| `local_learning_enabled` | `VICTOR_LOCAL_LEARNING_ENABLED` | `true` | on/off for the per-project RL delta |
| `local_classifier_completion_signal` | `VICTOR_LOCAL_CLASSIFIER_COMPLETION_SIGNAL` | `false` | whether the classifier's `task_completion` head may STOP the loop |

**Run pair (recommended minimum):** the same SWE-bench slice, two arms:

```bash
# Control: current defaults (classifier serves micro-decisions, delta on,
#          completion-signal OFF).
nohup script -q /dev/null bash -c 'victor benchmark run ...' > control.log 2>&1 &

# Treatment: flip the completion-signal on (the behavioral change under test).
VICTOR_LOCAL_CLASSIFIER_COMPLETION_SIGNAL=true \
  nohup script -q /dev/null bash -c 'victor benchmark run ...' > treatment.log 2>&1 &
```

(`script -q /dev/null` wraps a PTY so the keyring is reachable interactively —
plain `nohup` makes the process non-interactive and the API-key resolver skips
the keychain. Use the same model + slice + seed for both arms.)

**Decision criteria for flipping `local_classifier_completion_signal` default-on:**
1. Pass rate on the **un-starved subset** (`tool_calls > 0`) is ≥ control (no
   regression). Always read the un-starved subset — a rate-limited run starves
   tasks to 0 patches and corrupts the aggregate.
2. No measurable increase in **premature stops** (the risk: a confident-but-wrong
   "pass" stops the loop early). Track via `task_completion` decisions with
   `source=local_classifier`, `is_complete=True`, on tasks that ultimately failed.
3. Latency/cost unchanged or improved (the classifier is sub-ms vs a 4s LLM edge
   timeout).

If both hold, flip the default in `DecisionServiceSettings` and retire the
Ollama edge default. If (2) regresses, raise the head `threshold` (τ) or the
consumer `confidence ≥ 0.7` bar before re-running.

## References

- FEP-0001 (edge model system being replaced as default).
- `victor/agent/edge_model.py`, `victor/agent/services/decision_service.py`,
  `victor/agent/services/tiered_decision_service.py`, `victor/core/context.py`.
- Spec + ASCII diagrams: `/Users/vijaysingh/.claude/plans/lively-squishing-snowflake.md`.

## Review Process

- **Submitted by:** Vijaykumar Singh. **Initial review period:** 14 days.

## Acceptance Criteria

1. `victor chat` with no Ollama resolves micro-decisions via the shipped
   classifier (`source=local_classifier`); session completes.
2. A/B parity (or better) vs the LLM edge model on the universal decision types.
3. Per-project delta grows in `project.db` after a successful task; never
   uploaded.
4. No new runtime dependency; heuristic fallback intact when artifact absent.

---

## Copyright

This FEP is licensed under the Apache License 2.0, same as the Victor project.
