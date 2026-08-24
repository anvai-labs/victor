# EVR-6 Per-Turn Auditor: Offline HTIR Gate

**Status:** gate machinery shipped; real-distribution evidence pending; runtime default OFF
**Tracks:** FEP-0008 Phase C, EVR-6, flag-graduation policy

## Decision boundary

`victor/evaluation/turn_auditor_eval.py` measures a pinned prefix-only auditor against
independently labelled HTIR traces. It emits only `pass` or `hold`. A pass is an offline
prerequisite for broader rollout; it does not enable `per_turn_auditor`, change
`VICTOR_PER_TURN_AUDITOR`, or authorize a default flip. A later flag-OFF/ON real-agent A/B must
also match-or-beat task success.

`victor/evaluation/turn_auditor_evidence.py` is the corresponding evidence producer. It accepts
only a label pack without model observations, replays every HTIR prefix through the production
two-tier auditor, resolves the exact Ollama tag to a SHA-256 digest, disables decision caching,
and checks that the tag still has the same digest after the battery. This keeps oracle labelling
independent from the auditor under test.

`victor/evaluation/turn_auditor_review_pack.py` prepares that independent label pack from real-run
`eval_manifest_*.jsonl` files. Exported cases start in an explicit `pending` state; a null alarm
step is not a healthy label until a reviewer changes the state to `included` and records a
versioned `oracle_source`. Finalization refuses any pending case, permits explicit exclusions, and
strips task/tool reviewer context before emitting the evidence-producer input. It never invokes
the auditor.

## Evidence contract

Each case contains:

- a unique task id and task family;
- an HTIR trace with sequential step indices;
- an independent oracle source and the first step where intervention is justified (`null` for a
  healthy trace);
- one pinned-auditor observation for every HTIR prefix, in order, with verdict and latency.

The oracle label must not come from the auditor being evaluated. The evidence producer may use a
programmatic verifier, an independently reviewed annotation overlay, or another versioned source,
but must record that identity in `oracle_source`. The evaluated model/build is pinned in the
top-level `auditor_id`.

The label pack uses the same schema as the evidence input except that it must omit top-level
`auditor_id` and every case's `observations`. Each trace should retain source metadata sufficient
to locate the original real-agent run; labels derived from the auditor's own verdicts are invalid.

Legacy manifests often lack a benchmark/family field. Assign the real task family explicitly per
source rather than inferring it from outcomes or task text. The export contains bounded task,
tool-argument/result, and final-response context for local review, which may contain sensitive
repository data; keep the review pack local and use `--without-context` when reviewers will inspect
the source manifests directly.

```bash
python -m victor.evaluation.turn_auditor_review_pack export \
  --source code-fix=~/.victor/evaluations/eval_manifest_FIX.jsonl \
  --source code-gen=~/.victor/evaluations/eval_manifest_GEN.jsonl \
  --output artifacts/evr6/review-pack.json

# Independently review every case: status=included|excluded. Included cases require
# oracle_source and oracle_alarm_step (integer, or null only for a reviewed healthy trace).
python -m victor.evaluation.turn_auditor_review_pack finalize \
  artifacts/evr6/review-pack.json \
  --output artifacts/evr6/labels.json
```

Produce and assess a battery with the configured edge model:

```bash
python -m victor.evaluation.turn_auditor_evidence labels.json \
  --model qwen3.5:2b \
  --expected-digest 324d162be6ca5629ae4517c8710434d0bd2d665bc94dbad46e9af8fbf8a2f0df \
  --output evidence.json \
  --report report.json
```

The producer exits nonzero when the generated report is HOLD. That is an expected experimental
result, not permission to relax the pre-registered thresholds.

## Pre-registered gate

| Requirement | Threshold |
|---|---:|
| Total traces | ≥24 |
| Alarm-positive / healthy traces | ≥8 / ≥8 |
| Families | ≥4, with ≥4 traces and both polarities per family |
| Precision | ≥0.80 overall |
| Early-alarm recall | ≥0.80 overall and per family |
| Healthy-trace false-alarm rate | ≤0.05 overall and per family |
| Auditor p95 latency | ≤2,000 ms |
| Integrity | unique ids, full ordered prefix coverage, pinned identities |

An alarm is a true positive only when its first occurrence is at or before the oracle alarm step.
An alarm after that point is reported as late and counted as a false negative. This prevents a
post-failure detector from being described as early warning.

## Run

```bash
python -m victor.evaluation.turn_auditor_eval evidence.json \
  --output artifacts/evr6/turn-auditor-report.json
```

The command exits zero only on `pass`. Keep the report, source trace manifest, auditor identity,
and independent-label review together as the evidence artifact for any rollout PR.
