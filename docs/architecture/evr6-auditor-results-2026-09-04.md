# EVR-6 real-distribution attempt 1

- **Date:** 2026-09-04
- **Verdict:** **HOLD**
- **Runtime decision:** keep `per_turn_auditor` default OFF

## Method and provenance

The run started from fetched `origin/develop` commit
`58f9a7c7f3e782cfeb3fd0aad13f2ac6a6a4626c`. The original pending review pack was
re-exported from the four source manifests below and reproduced its recorded SHA-256 exactly.
The independent reviewer's exact edits were recovered from the 2026-08-29 reviewer session and
applied to that byte-identical base. The resulting reviewed-pack SHA-256 matched the digest
recorded when the blinded review completed. No labels were inferred from aggregate counts, model
observations, or benchmark outcomes.

| Family | Source manifest | SHA-256 |
|--------|-----------------|---------|
| code-fix | `eval_manifest_6fb27261da29.jsonl` | `45c328d21be07dd91593642916526a6afcddcc58cfc246f58a007b71aebdeeaf` |
| code-gen-humaneval | `eval_manifest_770e78c17603.jsonl` | `696ea8d451e94d93a6982268d13e90d98cf26b688ca3d317e9a37bb03d7ee159` |
| code-gen-mbpp | `eval_manifest_4f3f8b0cf5f4.jsonl` | `989d0f63386a80155bf5d4532b4c8149ae05fc4df03d71c74c042838e3d3f01c` |
| deep-research-dr3 | `eval_manifest_86a0e7ed90e1.jsonl` | `960d8654759178722be8cc665583f6be18209307d2110c008b661e6ddffe6862` |

The versioned oracle source is `codex-independent-reviewer-v1/2026-08-29`. Finalization removed
all reviewer-only context and confirmed that the label pack contains no `auditor_id` or
`observations`.

## Frozen labels

| Family | Included | Alarm-positive | Healthy |
|--------|---------:|---------------:|--------:|
| code-fix | 10 | 4 | 6 |
| code-gen-humaneval | 6 | 4 | 2 |
| code-gen-mbpp | 6 | 5 | 1 |
| deep-research-dr3 | 6 | 6 | 0 |
| **Total** | **28** | **19** | **9** |

The preregistered family-coverage requirement is already unmet because DR3 has no healthy trace.
Cases were not replaced after observing this distribution.

## Replay integrity

- Auditor: `ollama:qwen3.5:2b@sha256:324d162be6ca5629ae4517c8710434d0bd2d665bc94dbad46e9af8fbf8a2f0df`.
- The tag resolved to that digest before replay and to the same digest after replay.
- Decision caching was disabled (`cache_ttl=0`; evidence metadata `cache_enabled=false`).
- All 28 task IDs are unique and all 245 HTIR prefixes have exactly one ordered observation.
- The evidence payload preserves the finalized label payload exactly, apart from adding the
  top-level producer/auditor identity and per-case observations.
- A separate `turn_auditor_eval` invocation produced a byte-identical report.

## Result

| Metric | Result | Gate | Outcome |
|--------|-------:|-----:|---------|
| Traces | 28 | ≥24 | pass |
| Alarm-positive / healthy | 19 / 9 | ≥8 / ≥8 | pass |
| Families | 4, each ≥4 | ≥4, each ≥4 | pass |
| Both polarities per family | DR3 has 6 / 0 | required | **fail** |
| Precision | 0.000 | ≥0.800 | **fail** |
| Early-alarm recall | 0.000 | ≥0.800 | **fail** |
| Per-family recall | 0.000 in all four | ≥0.800 | **fail** |
| Healthy false-alarm rate | 0.000 overall and per family | ≤0.050 | pass |
| p95 auditor latency | 1,639.2016 ms | ≤2,000 ms | pass |

The model produced zero early true positives, 19 false negatives (11 misses and 8 late alarms),
and zero false positives. The evaluator recorded these exact HOLD reasons:

1. `code-fix` recall was below 0.800.
2. `code-gen-humaneval` recall was below 0.800.
3. `code-gen-mbpp` recall was below 0.800.
4. `deep-research-dr3` lacked both positive and negative oracle traces.
5. Overall precision was below 0.800.
6. Overall recall was below 0.800.

This is an offline prerequisite result only. It does not authorize enabling the runtime flag or
running the graduation A/B. Any resampling must be declared and independently reviewed as a
separate attempt without changing these thresholds.

## Artifact custody

The complete bundle is retained in the maintainer evidence store at
`~/.victor/evaluations/evr6/attempt-1-2026-09-04/`. The raw manifests and context-bearing review
pack remain local as required by the review-pack confidentiality guidance. Digests below make the
bundle independently verifiable.

| Artifact | SHA-256 |
|----------|---------|
| `review-pack-original.json` | `64804d029f24ca190329f2d334ba9a122e5fc339046d5040de7dd287657f0eef` |
| `review-pack-reviewed.json` | `bccd89a4449b60c35de8945b7c98f151488f7864d9498c04dec3b6922681ae60` |
| `labels.json` | `661ddc959db109941d24d0e9eb564aa8f1782c884c8bd490aee81bf896400195` |
| `evidence.json` | `b66a1059125e64682aab7bc4c7715ee1dce048f90f36505ea1c3242d3db10d93` |
| `report.json` | `ae9107cf1bf5b9ab4af6daec2959aeb9e42935ff87c4f04124e5950c3446f41a` |
| `turn-auditor-report.json` | `ae9107cf1bf5b9ab4af6daec2959aeb9e42935ff87c4f04124e5950c3446f41a` |

The evidence producer additionally records the canonicalized label payload as
`sha256:170593f11cebc5c6e428c76a88cbef723159770db9a68a9d7da8eae3e38ffa05`.
