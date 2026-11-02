# Similarity evaluation foundation

The [parity roadmap](graph-similarity-parity-and-roadmap.md) owns the improvement
order. This increment makes accuracy claims testable before adding retrieval,
compiler/effect analysis or developer gates. PR evidence owns validation and merge
state; source availability does not imply release or production qualification.

## Order and acceptance

1. Pin reviewed pairs and source snapshots; separate calibration from test projects.
2. Measure pair scores, abstentions, thresholds and hard negatives. Extend frontends
   only where observed failures justify a concrete, conservatively tested change.
3. Compare context/type features and bounded retrieval against these exact pair
   scores. Measure end-to-end CPU, wall time, RSS and cold/warm behavior before ANN.
4. Build family explanations and CI baseline/SARIF/changed-code applications on
   explicit evidence and stable identities. Source-copy gates and semantic review
   policies need separate acceptance criteria.

The first measured coverage repair recognizes Rust `f32`/`f64` literal suffixes.
The `portable-operations-v3-typed-floats` namespace preserves explicit float widths
and correct binary32 literal rounding. Decimal-to-binary64-to-binary32 conversion
can double-round; the implementation rounds the bounded exact decimal ratio with
integer ties-to-even arithmetic. Hexadecimal digits ending in `f32` remain integer
digits. Nonfinite literals are rejected. This does not prove expression types,
overflow compatibility, effects, ABI or behavioral equivalence. Older v2 evidence
retains its original namespace and cannot be used as v3 calibration.

## Corpus contract

`victor-similarity-corpus-v1` records projects, function references and reviewed
pair labels. Each repository project declares an immutable 40-character revision,
its repository identity and either `calibration` or `test`. Fixture projects are
explicitly marked `controlled_fixture`. Function references pin file SHA-256,
relative POSIX path, language, symbol and exact inclusive source lines.

A pair declares its intended relation (`copy`, `structure` or `behavior`), boolean
label, lineage/family group, input/observation domain, rationale and curator.
These are curator declarations, not authenticated independent review. Different
intentions need different labels: an in-memory helper and a persistent backend can
implement related interfaces without sharing operation structure or persistence
semantics. The caller must review transitive helper behavior and build assumptions
before labelling behavior. Compiler/runtime equivalence is not inferred here.

The validator rejects duplicate/conflicting labels, unknown endpoints, self-pairs,
unsafe paths, unreferenced units, and declared repository/family/file-hash leakage
between splits. Canonical source-function references also reject aliases under
new unit/project IDs for the same declared repository/revision/file/symbol/span;
renaming IDs cannot inflate class support or manufacture physical self-pairs.
It cannot automatically discover undeclared source-copy lineage,
repository aliases or semantic leakage. Curators must keep those groups together.
File/function identifiers are independent of list ordering; the canonical manifest
hash changes when the reviewed content changes. Corpus and selected threshold maps
are immutable after validation.

Source roots are explicit `--project ID=PATH` arguments. The runner opens each
root with the secure descriptor reader and verifies every requested file hash
before extracting a uniquely matching function. Only those listed files are read.
No project configuration, source code, compiler, build script or dynamic library
is executed. Existing source hashes are checked; the declared Git revision is
provenance provided by the curator, not independently authenticated by this runner.
Missing/stale/oversized/malformed source or unavailable grammars produce abstentions.
The initial parser profile requires intact files and ordinary source functions;
embedded-region, SCIP and CFG/DDG evidence are not evaluated by this profile.

## Scoring and calibration

Syntax scores are exact set-Jaccard over the production subtree fingerprints;
cross-language syntax namespaces are not compared. Portable scores compare
complete production operation projections and retain unsupported/small projections
as abstentions. Body scores reuse the existing local-only pinned model, complete
source windows, pooling and checksummed vector cache. Only requested vector pairs
are scored, in bounded batches; no all-pairs retrieval matrix is allocated.

Families are never expanded into artificial pair matches. For example, A–B=0.8
and B–C=0.8 can coexist with A–C=0.28. The body diagnostic path retains the latter
score instead of assigning the family's strongest edge to every member.

Thresholds are selected only from calibration projects, separately per relation
and unordered language pair. Both classes and the selected true positives must
meet `minimum_support`. Selection maximizes true positives subject to the requested
empirical precision, then chooses the more conservative threshold. An empty
prediction set cannot be a perfect-precision operating point. An unsupported
facet or unattainable precision produces no threshold. Evaluation revalidates the
frozen policy against its corpus, detector profile and calibration observations.
Test scores/labels cannot select that threshold.

Reports include confusion counts for decided pairs, conditional precision/recall,
scoring coverage, undecided/abstained counts and end-to-end recall over **all**
labelled positives. A missing score is never a true negative. Precision with no
positive predictions is undefined. Review precision@k is restricted to labelled
pairs within one facet; cutoff ties use expected uniform ordering. Unlabelled
pairs are not assumed negative. Wilson precision bounds show sampling uncertainty.
The Wilson bounds use nominal pair counts; correlated pairs/families are not
independent samples. Qualification needs more independent projects and grouped
uncertainty estimates. These metrics do not measure whole-repository retrieval
recall or family cohesion.

`production_quality_qualified` remains false. A small reviewed pilot, point
precision target, or successful protocol gate does not qualify a model for general
semantic deduplication. The default BGE profile remains uncalibrated in production.

## Commands and outputs

```bash
PYTHONPATH="$PWD:$PWD/victor-codegraph" python scripts/evaluate_graph_similarity.py \
  --corpus tests/fixtures/graph_similarity/controls.json \
  --project controls-calibration=tests/fixtures/graph_similarity/controls-calibration \
  --project controls-test=tests/fixtures/graph_similarity/controls-test \
  --channel portable --output /tmp/similarity-controls.json
```

Use `--channel body` with locally cached weights, optionally `--cache` pointing to
an exclusive derived SQLite cache. `--body-profile` accepts strict `BodyOptions`
JSON with explicit `model_id`, immutable `revision` and `dimension`; unknown fields
and incorrect scalar types are rejected. It never enables remote code or downloads.
The output/cache cannot replace corpus, profile, source or project graph inputs,
including SQLite WAL/SHM/rollback-journal sidecars and existing inode aliases.
Existing cache files must be regular, singly linked and contain only the body
cache schema; foreign databases cannot be repurposed as derived caches.
Reports are atomically published and contain hashes and scores, not source bodies.

Exit 0 means the requested protocol gates passed. Exit 1 means missing scoring
coverage, or failed optional acceptance gates. Exit 2 means invalid configuration,
manifest or a whole-run resource/IO error. Without extra gates, an unqualified
threshold policy is still a valid measurement report. `--require-decisions` fails
if any facet lacks a policy or any pair lacks evidence. Adding
`--minimum-test-precision 0.95` also requires a defined held-out precision at least
that value in every test facet and overall. Each test facet must also have the requested
minimum support in both classes. Both operate only on this labelled evaluation universe.

Reports embed the canonical corpus manifest and retain model/frontend/implementation
fingerprints, observations and source provenance plus wall/CPU time, labelled-pair throughput and process-lifetime peak
RSS. Timing covers source capture through report assembly and excludes process
startup and report-file serialization. Process peak is not incremental model memory. Do not compare profiles, corpus
sizes or cold/warm cache conditions as if they were matched workloads.

## Retained inputs and limitations

[Controlled sources and labels](../../tests/fixtures/graph_similarity/controls.json)
exercise Python/Python, Python/Rust, TypeScript/Rust and Java/C++. They are authored
protocol controls with related templates, **not a project-held-out accuracy set**.
The numeric-domain negatives retain identical operation shapes while changing the
allowed domain. JavaScript integer representation and checked Rust overflow are
specified by [ECMAScript](https://tc39.es/ecma262/2023/multipage/numbers-and-dates.html#sec-number.max_safe_integer)
and the [Rust reference](https://doc.rust-lang.org/reference/expressions/operator-expr.html#overflow).

The [repository pilot](../../tests/fixtures/graph_similarity/repository-pilot.json)
contains four Victor calibration pairs and four ProximaDB held-out pairs. Labels
come from source inspection of the pinned commits, with restricted structural
intent. No ProximaDB source is copied into this repository. Reproduce with explicit
roots containing the recorded bytes; later source edits correctly invalidate the
manifest. Its eight pairs are insufficient to estimate general production accuracy.

The first pilot exposed a high body-vector score for structurally different
`set_trace_tag` implementations and low scores for the held-out cosine patterns.
The report therefore provides a hard-negative regression opportunity, not a new
claim that semantic equivalence has been solved. The next data priority is more
independent projects, reviewed lineages and hard negatives, followed by typed
context ablation and measured retrieval. Unsupported macros/await, Windows source
handles, ABI verification, ANN/incremental indexing and SARIF/copy baselines remain
separate subsequent increments.

## Evidence and independent review

The [retained foundation observations](evidence/graph-similarity-foundation-2026-11-02.json)
include verified corpus, runner and detector hashes, all labelled pair observations,
calibration policies, per-facet metrics and cold/warm vector-cache timings. With the
same 40-file manifest and 234 eligible functions as the earlier audit, complete
portable projections increase from 167 to 169. The remaining 65 rejected units
include 53 macro invocations. This measures extraction coverage, not precision.

Reproduce the label-protocol load measurement separately:

```bash
PYTHONPATH="$PWD:$PWD/victor-codegraph" python scripts/benchmark_graph_similarity_evaluation.py \
  --pairs 50000 --runs 5 --output /tmp/similarity-protocol-scale.json
```

It includes manifest validation, threshold selection and report computation, and
excludes source parsing, model inference and retrieval. Synthetic score separation
is generated only to exercise protocol invariants. It is never an accuracy result.

Initial independent review found four issues, fixed before the first push:

| Finding | Correction and regression |
| --- | --- |
| New IDs alias the same source function and inflate class support | Canonical repository/revision/file/symbol/span uniqueness; both unit and project aliases rejected by `test_canonical_function_aliases_cannot_inflate_support` |
| Report output can replace SQLite WAL/SHM/journal files | Protect sidecar names, case aliases and inode aliases; `test_live_wal_is_not_replaced_by_a_report` verifies committed data remains readable |
| A cache hardlink can mutate a graph database | Reject linked/symlink cache files and foreign database schemas; `test_hardlink_cache_alias_cannot_mutate_graph_database` and body-cache regressions check unchanged bytes/schema |
| A large exponent on finite zero is treated as overflow | Handle finite signed zero before exponent magnitude checks; direct and Rust-extraction regressions retain `0e100f32` |

The numeric review additionally compared 11,872 nonzero cases with independent
libc `strtof`; threshold review checked 243 enumerated policy cases. Exact final
review, local checks and CI belong to the PR and commit-bound attestation. These
observations do not qualify a release, Windows source adapter, universal semantic
equivalence or safe automatic extraction.
