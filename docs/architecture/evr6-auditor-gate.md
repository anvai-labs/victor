# EVR-6 Per-Turn Auditor: Offline HTIR Gate

**Status:** gate machinery shipped; real-distribution evidence pending; runtime default OFF
**Tracks:** FEP-0008 Phase C, EVR-6, flag-graduation policy

## Decision boundary

`victor/evaluation/turn_auditor_eval.py` measures a pinned prefix-only auditor against
independently labelled HTIR traces. It emits only `pass` or `hold`. A pass is an offline
prerequisite for broader rollout; it does not enable `per_turn_auditor`, change
`VICTOR_PER_TURN_AUDITOR`, or authorize a default flip. A later flag-OFF/ON real-agent A/B must
also match-or-beat task success.

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
