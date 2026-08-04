# SWE-bench-lite real-distribution judge re-gate (2026-08-04)

Committed evidence for the FINDINGS.md section *"SWE-bench-lite real-distribution
re-gate — the 'real' gate wasn't real enough."*

Both calibration-corpus gate-passers **over-credit** on the true shipping
distribution (in-container-verified SWE-bench-lite, 30 instances, 5 resolved /
25 not):

| file | judge | α (nominal) | confusion |
|---|---|---|---|
| `llama3_3.json` | llama3.3:70b | **0.263** | TP=5 FP=10 TN=15 FN=0 |
| `gemma4.json` | gemma4:31b | **−0.523** | TP=5 FP=23 TN=2 FN=0 |

Perfect recall (FN=0) but heavy false positives — the LLM judge cannot tell
"looks done" from "passes tests" at real distribution. See FINDINGS.md for the
full write-up and the n=5-positive caveat.

## Regenerating (the raw stratum jsonl is `*.jsonl`-gitignored, like the run12 packs)

```bash
# 1. Produce trajectories with in-container gold (both fixes: #868 image names, #869 patch persist)
victor benchmark run swe-bench-lite --max-tasks 30 --timeout 400 --max-turns 25 \
    --eval-backend docker --swebench-image-source official
# 2. Re-gate a judge over the run's manifest (auto-converts manifest → stratum)
python benchmarks/judge_calibration/gate_swe_bench_stratum.py \
    ~/.victor/evaluations/eval_manifest_<id>.jsonl \
    --judge llama3.3:70b --endpoint http://<ollama-host>:11434 \
    --out benchmarks/judge_calibration/labels/swe-bench-lite-30/llama3_3.json
```
