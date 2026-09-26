# Formation research evaluation — 2026-09-20

The immediate research contribution is stronger verification of member deliverables.
Learned routing is a later controlled experiment. None of the papers below establishes
that changing Victor's model or formation will fix its observed task-binding,
completion, accounting or lifecycle defects.

This audit extends the [formation handoff](multiagent-formation-coverage-handoff.md).
Its live denominators remain unchanged: nine original workstreams landed, ten of
fifteen new ZAI cases passed, no matched new Qwen matrix run, and current C5 mixed-team
acceptance failed. The original failed reports remain authoritative for those runs.

## Retrieval and evidence boundaries

Four semantic searches against the user-provided dataserver3 arXive API requested
five results each: coordination failures, formation/routing evaluation, heterogeneous
model routing, and executable coding-task verification. Rank-interleaving and
deduplication selected twelve papers for metadata/fulltext retrieval. Eleven returned
nonempty fulltext; Rethinking Scale lacked both abstract and fulltext in the LAN
index, so its public arXiv abstract was used. The video-recommendation survey was
excluded after abstract review. This is a targeted sample, not a systematic review.

The [sanitized retrieval manifest](evidence/formation-research-audit-2026-09-20.json)
records queries, search scores, request IDs, source URLs, indexed timestamps, response
hashes and review scope. Scores are retrieval similarities, not evidence quality.
Indexed metadata can lag public revisions. Authentication was fetched once and kept
in process memory; paper text and credentials are absent from committed evidence.

Repository assessment used develop `5a5777295`: identifier/title searches in
`docs`, `feps`, `victor` and `tests`, followed by inspection of matching code,
formation dispatch, matrix predicates and verification contracts. “No reproduction
found” below describes this audit, not proof that every historical experiment was
searched. Conceptual overlap, a citation, unit coverage and a paper benchmark
reproduction are separate levels of evidence.

## Papers and fit to the existing project

| Paper | Existing Victor evidence | Recommended disposition |
|---|---|---|
| [ExecCritic](https://arxiv.org/abs/2609.09133) | No citation or reproduction found. Existing PIPELINE/REFLECTION and FEP-0018 provide execution/verification seams. G36 still permits weak process acceptance. | Apply independent, qualified, frozen test-oracle principles to G42/G36. Do not add a formation or claim replication of its role-specific reinforcement learning. |
| [SWE-Bench Pro Verified](https://arxiv.org/abs/2609.08149) | No citation or reproduction found. The gateway matrix already retains provenance and failed reports, but member-authored tests are not independent correctness oracles. | Specify exact task domains and separate implementation-visible examples from runner-owned acceptance. Audit test tampering and leakage; the fifteen-case smoke matrix is not a SWE-bench result. |
| [AgentGate](https://arxiv.org/abs/2604.06696) | Already cited by `decision_trees.py` and `prompt_section_allocator.py`. Their heuristic behavior has unit coverage, but the paper's trained candidate-aware routing benchmark was not found. | Retain structured action/candidate constraints as inspiration. Evaluate an opt-in routing policy through the existing registry/dispatch before making quality or token-saving claims. |
| [ProgRouter](https://arxiv.org/abs/2608.25992) | No citation or reproduction found. Existing per-member model selection and adaptive/router formations are useful seams, not its learned progress predictor. | Later paired routing experiment after reliable task progress, G34 accounting and G36 verification. Compare against fixed routing on identical task contracts and budgets. |
| [Rethinking Scale](https://arxiv.org/abs/2604.19299) | No citation or reproduction found; assessed from public abstract only. Its study concerns models below 10B, not this Qwen3-Coder-30B deployment. | Include a single-agent tool-using control in a later quality/cost study. Do not infer that Qwen size explains these failures or that ZAI necessarily fixes them. |
| [Coordination as an Architectural Layer](https://arxiv.org/abs/2605.03310) | Concept overlaps Victor's existing single coordinator, formation strategies and FEP-0035 transcript design; no paper experiment reproduction found. | Use fixed inputs, comparable budgets and retained negative findings. No second coordination abstraction is needed. |
| [EvolveRouter](https://arxiv.org/abs/2604.05149) | No citation or reproduction found. FEP-0025 already describes controlled prompt comparison and its remaining gaps. | Defer learned routing/prompt co-evolution until a fixed-contract baseline exists. Its question-answering benchmarks do not validate repository artifact completion. |
| [Iterative Critique-and-Routing Controller](https://arxiv.org/abs/2605.08686) | Reflection and heterogeneous members overlap conceptually; no trained controller reproduction found. | Defer the reinforcement-learning controller. Its math-oriented evaluation and controller overhead require a separate experiment. |
| [TeamTR](https://arxiv.org/abs/2605.15207) | No citation or reproduction found; current work repairs inference-time execution. | Defer training-time trust-region coordination. It does not repair cache ownership, member identity or runtime verification. |
| [Beyond the Leaderboard](https://arxiv.org/abs/2607.05775) | Existing gap table already separates tool, coordination, lifecycle and measurement failures. No citation found. | Use its narrative failure taxonomy to check gap classification, not as an independent replication or a cross-model accuracy estimate. |
| [When Agents Implement Systems](https://arxiv.org/abs/2609.01985) | Existing workflow mandates reproduced defects and regression evidence. No citation found. | Retain measured counterexamples and scoped claims. Its single-session case study and gold-label retrieval substitution do not establish general agent performance. |
| [Multi-Agent Video Recommenders](https://arxiv.org/abs/2604.02211) | No citation found; abstract-only assessment. | Exclude from this implementation scope: recommendation-domain mechanisms do not address the measured formation defects. |

ExecCritic is especially relevant because a patch and its own generated test can
agree on the same wrong behavior. Its test-quality ablation also shows that adding
tests can reduce repair success when the tests are poor. Its reported training
improvements are not a compute-matched estimate for Victor. The proposed adoption
is therefore test qualification and ownership, not its numerical result or model.

AgentGate separates action choice from grounding that choice into executable
candidate-constrained outputs. Victor's two citing modules are deterministic
decision-tree and prompt-budget helpers. Their existing suites passed **43 tests**
using `.venv-codesign/bin/python -m pytest`; this tests those local heuristics.
Static import searches located test consumers but no production import sites for
these modules. That is a wiring question, not conclusive reachability analysis.
The allocator's “2–3x” token-reduction target is a stated target, not a measured
result established by these tests. Adding duplicate citation/presence tests would
not strengthen the evidence; no such tests were added.

## G42: a reproduced weakness in the matrix's correctness predicate

The new formation matrix asks members to write a doubling function and a test of
one input. Its independent pytest process reruns that member-authored test. At the
audited source, both the ordinary and ensemble scenarios in
[`formation_matrix_cases.py`](https://github.com/anvai-labs/victor/blob/develop/scripts/validation/formation_matrix_cases.py)
use the single check `f(4) == 8`.

An isolated offline counterexample supplied `def first(x): return 8` and the
requested test. Pytest exited zero, although `first(5)` returned 8 instead of 10.
The manifest records the candidate, predicate and counterexample. This is a new
**offline predicate finding with zero model calls**, not an actual-member replay,
proof that a live member wrote this mutant, or a claim that the complete matrix
accepted it. The live matrix's other accounting, session and artifact gates were
not exercised by this probe. Historical passes remain passes under their recorded
checks; they do not prove behavior outside that single example.

The corrective increment should establish a bounded numeric task domain before
execution, add independent runner-owned oracle checks, and reject known wrong
implementations through TDD. Keep member-authored tests as deliverables, but make
their success insufficient for final acceptance. Hash the frozen oracle and report
its process outcome and structured results; do not trust model-authored summaries.
Reject empty, failed, malformed, timed-out or modified oracle runs explicitly.
Use existing matrix predicate tests as the coverage owner and remove only tests
whose behavioral/branch coverage is demonstrably retained elsewhere.

## Ordered integration and evaluation

1. Finish the measured cache-owner correction and retain G41 until a complete live
   matrix establishes bounded resource behavior. Close G39's assignment/identity
   gap so rewritten tasks retain the correct member's deliverables without
   overriding native structured formation response contracts.
2. Fix G42's independent oracle and G36's structured process acceptance. Use the
   existing [FEP-0018 verification contract](https://github.com/anvai-labs/victor/blob/develop/feps/fep-0018-framework-verification-hook.md),
   with explicit process status, bounded cleanup and preserved failure diagnostics.
   G32 completion and G34 buffered accounting remain separate required work.
3. Freeze and version the corrected task contract, then run all fifteen cases on
   ZAI/Sandhi and the matched ready Qwen/InferFlux runtime. Record exact source,
   binary/model identity, offload/context settings, prompts, budgets and per-case
   outcomes. Preserve the 120-second buffered gateway deadline; timeouts fail even
   if a later retry succeeds. Require artifacts, independent pytest/oracles,
   distinct sessions and wire/SQLite/C4/dashboard conservation. These are new
   experiments; do not rewrite old verdicts or clear shared cache.
4. Run the corrected six-Qwen/one-ZAI C5 case and obtain review on InferFlux #184.
   Explicit zero reported cache does not prove zero executed reuse. Tokenizer,
   session-lease, origin-cancellation and full lifecycle acceptance remain bounded
   by the available origin evidence.
5. Only then compare routing policies: fixed Qwen, fixed ZAI and opt-in local/cloud
   routing, plus a single-agent control. Predeclare repeat counts, task split,
   correctness gates, budgets and comparison rule before calls. Report failures,
   latency and all invocation/token costs, including router/critic overhead.
   Neutral Sandhi token units are not currency pricing. Separate warm/cache effects
   through recorded observations; do not mutate shared cache to manufacture a
   comparison. Model-specific differences require repeated matched evidence.

The existing [FEP-0025 controlled-experiment design](https://github.com/anvai-labs/victor/blob/develop/feps/fep-0025-prompt-evolution-as-controlled-experiment.md)
is the reference for baseline arms and evidence-based promotion, not a claim that
all its phases are implemented. Any new public routing API requires its own reviewed
FEP scope. Keep one registry, one dispatch and canonical role/identifier derivation;
new behavior remains opt-in with unchanged defaults. None of these papers warrants
adding an unvalidated formation or bypassing the handoff's definition of done.
