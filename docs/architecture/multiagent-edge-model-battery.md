# WS-F: paired local-model formation battery

Implementation/report: [PR #1118](https://github.com/anvai-labs/victor/pull/1118).

The six-formation battery is an artifact gate, not a claim that TeamResult.success
proves task completion. It uses the same harness and task on Qwen3-Coder-30B and
the locally available LFM2.5-8B-A1B Q4_K_M model, through Sandhi. Classifiers and
production formation behavior are unchanged.

## Reproduction and scope

Run scripts/validation/formation_model_battery.py from its linked worktree with
`.venv-codesign/bin/python`, supplying --model, --gateway-client, and --output-dir.
Each case creates fresh artifacts and two member sessions. Pass requires successful
member results, every required file, independent pytest, and the formation's
structured verdict/agreement contract. Failure evidence is retained.

The model store identifies LFM2.5-8B-A1B as a small MoE (4.9 GB quantized file).
It ran on llama_cpp_cpu; Qwen ran on llama_cpp_rocm. Both models were ready and
reported fallback=false. This is an additional local-model guard sample, not a
latency comparison, a dense 2B support claim, or a model ranking. Only one run per
model was measured. The fixed HIERARCHICAL case uses static member assignments;
it does not validate model-generated delegation/synthesis.

Tasks require a doubling function, a corresponding pytest file, actual execution,
and exact final JSON. REFLECTION instead requires the critic to inspect/test the
first member's code and emit a structured verdict plus review.json. CONSENSUS
must actually reach agreement. Bounded iterations=10, tool budget=12, timeout=240s;
the gateway's separate upstream timeout can fail a call sooner.


## Measured matrix (2026-09-18)

| Formation | Qwen / ROCm | LFM / CPU |
|---|---|---|
| SEQUENTIAL | Failed: missing test_first.py (30.85s) | Failed: missing second deliverables; test syntax error (232.33s) |
| PIPELINE | Failed: missing test_first.py (10.92s) | Failed: literal backslash-n in both tests (162.89s) |
| PARALLEL | Failed: missing test_second.py (11.09s) | Timeout; missing second deliverables (240.60s) |
| HIERARCHICAL | Failed: both test files missing (28.84s) | Timeout; no required artifacts (240.55s) |
| CONSENSUS | Passed: two member sessions, artifacts/tests and agreement (10.82s) | Timeout; no required artifacts (240.76s) |
| REFLECTION | Failed: invalid critic JSON and no review.json (14.46s) | Timeout; no required artifacts (240.55s) |

Result: Qwen 1/6; LFM 0/6. All twelve cases were attempted and retained. A partial
pytest pass does not compensate for missing member files. These results do not
replace the earlier successful live matrix: they use a different strict task and
record its failures rather than claiming universal formation/model reliability.

Evidence: [Qwen](evidence/ws-f-qwen-battery.json) and
[LFM](evidence/ws-f-edge-battery.json). Request-model assertions were checked against
every recorded request. Timeout cases have post-run independent pytest and artifact
checks as supplemental evidence; they remain failures regardless of late files.
The final harness adds an explicit nonzero failure exit and model-match gate; neither
changes the recorded model responses. Executed harness SHA256 is retained in evidence.

The edge model was loaded with default=false, then unloaded after the battery. Its
temporary Sandhi virtual key was revoked. The Qwen GPU model and ZAI gateway scopes
were retained. The local model-store README identifies this GGUF as a small MoE;
a dense 2B model remains unvalidated.

## Local verification and follow-ups

The formation/team suites passed 459 tests on the experiment base and 463 tests
on the final merged WS-E base. The additional
completion/intent suites passed 88 tests and failed one existing wall-clock guard:
TestIterationBoundHolds.test_never_complete_loop_stops_at_max_iterations. Its
iteration bound passed, but duration was 22.68s against a 20s assertion; an isolated
recheck failed at 23.38s. No timing threshold or classifier was relaxed. G33 tracks
isolating startup/plugin work from the bounded-loop test.

The unchanged classifier implementation is on the pre-WS-E experiment base; a CI-only
fast-forward occurred during the edge run. The report increment is based on merged
WS-E. Thus this report claims a measured baseline, not a live rerun of newly merged
usage-accounting changes. WS-E's separate mixed run provides that accounting evidence.

## Follow-up boundaries

- Missing or syntactically invalid files plus successful member status is a task
  completion gap, not passing evidence. LFM emitted literal backslash-n sequences
  into test source, and a member returned proposed shell commands as answer text.
- A malformed REFLECTION verdict is rejected explicitly. Preserve that contract;
  do not add prose parsing to make a weak model appear successful.
- CPU provider/gateway timeouts are transport/capacity observations. They do not
  identify a narration, intent, or refusal classifier defect by themselves.
- Any guard tuning needs the paired FEP-0025 experiment process, retained false
  positives/negatives, and independent artifact verification. This increment
  records follow-ups instead of silently changing shared completion heuristics.

Final environment check: Sandhi stop/start restored both permanent scoped keys;
one real Qwen completion and one real ZAI completion succeeded through the gateway.
