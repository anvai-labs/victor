# SWE-bench-lite trained-classifier re-gate (2026-08-05)

Committed evidence for the FINDINGS.md section *"Trained-classifier re-gate — the
other half fails too."*

A ModernBERT-base classifier trained on real SWE-bench trajectories (run-1, 30
instances / 5 positives) and tested on **unseen** run-2 instances (30 / 4 pos)
**collapses to the majority class** — it never predicts "resolved":

| file | judge | α | confusion |
|---|---|---|---|
| `modernbert_swebench_v1.json` | ModernBERT (trained on real) | **−0.05** | TP=0 FP=0 TN=26 FN=4 |

Held-out score range 0.017–0.117 (predicted-positive 0/30). With only 4 training
positives it cannot learn to discriminate. Together with the LLM re-gate
(`../swe-bench-lite-30/`, α 0.26 / −0.52, which *over*-credits), this shows both
substitute approaches fail in opposite directions at the real distribution — only
the in-container verifier discriminates.

## Regenerating (strata + model are `*.jsonl`/large-artifact, not committed)

```bash
# Two 30-instance docker runs (--start-task 0 and 30) → two manifests
victor benchmark run swe-bench-lite --max-tasks 30 --eval-backend docker --swebench-image-source official
victor benchmark run swe-bench-lite --start-task 30 --max-tasks 30 --eval-backend docker --swebench-image-source official
# Build train (run-1) / test (run-2) strata, train, then gate:
python benchmarks/judge_training/train_encoder.py --dataset <run1_split_dir> --out <model_dir> --epochs 4 --batch 8
# score the model over run-2's stratum with swe_bench_stratum.gate_stratum (see swe_bench_stratum.py)
```
