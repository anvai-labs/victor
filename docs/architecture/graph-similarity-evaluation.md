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

Body cache compatibility separates the input/model contract (`profile.identity`)
from the vector producer (`profile.producer.identity`). The latter records the
encoder/window/pooling, vector validation and cache serialization fingerprints,
installed NumPy/model-library versions, Python and platform metadata. A fresh standard encoder can reuse that
generation without importing the model stack or constructing model weights.
Library or implementation changes invalidate reuse. Legacy rows keyed only by
the input contract remain untouched and are recomputed under the producer key;
they are never relabelled as current-generation observations.

Encoder configuration is fixed at construction; supply custom models through
`model_factory`, rather than replacing `model`, `options` or `model_factory` on an
existing encoder. Generation metadata is checked again before cache publication
and returning results; an implementation change during inference fails instead
of publishing a mixture of generations. Model internals and weights must not be
mutated after loading. A new declaration is required for modified custom models.

Injected model factories, encoder subclasses and instance method overrides use
instance-scoped identities unless their caller explicitly supplies a stable
`producer_id`. The caller must
change that declaration when its custom implementation, configuration or weights
change. It is a provenance assertion, not authentication or model qualification.
Wrapped or unavailable framework components force instance scope even with a
custom declaration. Actual callable bytecode and nested constants accompany
source fingerprints, so decorator metadata cannot hide changed preprocessing.
Encoders without a declared producer cannot share another analysis's cached
vectors; pair evaluation requires producer provenance. Runtime metadata is a
compatibility boundary, not a promise of bitwise reproducibility on every CPU,
binary build or locally modified model installation. Cached data remains derived,
checksummed state, separate from indexed symbol-summary vectors.

`scripts/benchmark_body_cache.py` compares fresh-process cold/warm observations
on an immutable snapshot of 2–300 eligible functions. It records source and actual
imported implementation hashes, vector-byte hashes, every requested pair's score
digest, cache/load counts, wall/CPU time and lifetime peak RSS. Run the same script
and Python environment against each source revision with separate caches. Keep
`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`; no model download is required or allowed
by the encoder. This is an exact requested-pair cache experiment, not an all-pairs
retrieval scalability or accuracy benchmark.

The [2026-11-02 cache observations](evidence/body-cache-provenance-2026-11-02.json)
compare exported baseline `d5bf69043` with the producer-aware candidate using the
same 40 files from `7b4942c91`. All 234 functions, 233 unique vector inputs and
27,261 requested pair scores matched exactly in cold and warm runs, without
abstentions. Both cold runs inferred 274 windows; both warm runs had 233 cache
hits, zero inferred windows and zero model loads. Baseline/candidate cold wall
times were 15.377/16.155 seconds; warm times were 44/60 milliseconds. Warm process
peak RSS was 85,295,104/87,080,960 bytes. These are single sequential local ARM64
observations, with source collection outside the timer and lazy model imports
inside it. The additional compatibility checks have a cost; this increment fixes
cache correctness and makes no performance improvement claim. The artifact records
source hashes, actual imported implementation hashes, installed dependency
metadata, vector-byte/spans manifests and score digests. Reconstruct its snapshot
by exporting exactly the `source_manifest` paths from the workload revision; use
separate fresh derived caches and source roots outside the current working directory
for baseline and candidate invocations.

Families are never expanded into artificial pair matches. For example, A–B=0.8
and B–C=0.8 can coexist with A–C=0.28. The body diagnostic path retains the latter
score instead of assigning the family's strongest edge to every member.

### Family explanations

Body and indexed-summary vector families expose a representative, mean member-pair
cosine and a directly scored pair witness in JSON and terminal output. Member IDs
resolve each reference to its reported source location. These IDs identify members
of the current analysis; they are not persistent source-copy baseline fingerprints.

The representative maximizes mean cosine to the other members, with canonical ID
ordering and a 1e-12 tolerance on mean-score ties. The sum of normalized vectors
also yields the mean of all distinct member-pair cosines without enumerating a
quadratic matrix. Two farthest-member sweeps find an actual witness pair; they do
**not** certify the globally least-similar pair. The report explicitly retains
`pairwise_minimum_verified: false`. A representative can be close to every member
while another member pair falls below the retrieval threshold.

The scope is **all component members**, including pairs excluded by file/language
retrieval filters. Explanations use normalized float64 geometry and leave the
existing float32 retrieval, edge counts and thresholds unchanged. These numbers
describe the selected vector representation, not probability, behavior, safe
extraction or ABI compatibility. Summary vectors still omit function bodies.
Hybrid channels remain independent. Source-copy gates must use their own stable
evidence and error/coverage policy, never these semantic family counts.

Four O(n*d) passes use bounded float64 blocks (8 MiB by default), O(n+d) metadata
and bounded indexing temporaries. Inputs must remain immutable during analysis.
The [retained observations](evidence/vector-family-cohesion-2026-11-02.json) compare
merged baseline `28c13496e` with the candidate. Five observations per fresh process
preserve matching family/edge/score digests in all three cases. Median engine wall
times, baseline/candidate: 234 real vectors at synthetic locations 0.780/3.936 ms;
5,000 replicated real vectors 256.905/291.887 ms; seeded 5,000-vector control
130.949/182.541 ms. The replicated case retained 168 components and 89,513 edges;
peak process RSS was 145,408,000/150,487,040 bytes. These are added explanation
costs, not speed improvements. All locations are synthetic, including the real
vector fixture, so these family counts do not describe a source audit.

Reproduce with `scripts/benchmark_vector_families.py --count 5000 --runs 5` in
separate baseline/candidate processes using the same Python environment. The
optional `--fixture` accepts a bounded numeric `.npy` file without pickle and
records its hash. The retained fixture contains the 234 body vectors reconstructed
from the earlier cache observation's source/input/vector manifests; its source
hashes and pinned model profile are retained. Imports, fixture creation, source
collection and model inference are outside the engine timer; RSS is a lifetime
peak. Single-host diagnostic controls do not qualify independent-project accuracy,
full-project latency, the 200,000-symbol ceiling or cross-platform performance.

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

Local model construction and profile validation are attempted once per encoder.
A failed load is retained as a bounded error category/message; subsequent bodies
still report their individual failures without constructing the model again.
Create a fresh encoder after repairing the installation/profile. Cancellation
propagates and is not retained as a model failure. Warm vector-cache reads still
perform zero model loads. Body statistics expose load attempts and failure state.

The [failure-path measurement](evidence/graph-similarity-contracts-2026-11-02.json)
compares the same 234-function source manifest and intentionally incorrect pinned
model dimension. Construction count falls from 234 to one, with identical 234
errors and incomplete coverage. Single-process measurements are 20.426s versus
0.462s wall time and 18.823s versus 0.174s CPU. Imports/source collection are
outside the timer; RSS is process lifetime peak. These are failure-path results,
not successful inference throughput or accuracy. Reproduce with locally cached
weights and each candidate's source on `PYTHONPATH`:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=.:victor-codegraph \
  python scripts/benchmark_body_model_failure.py --path /path/to/matched-snapshot
```

JNI links additionally carry a versioned `declared_contract` diagnostic:
`matching_declaration`, `conflicting_declaration` or `unknown`. Under standard
JNI typedef spellings, known primitive argument/return differences and argument
counts produce conflicts. Static receivers and primitive arrays are inspected;
reference descriptors require fresh SCIP type identities, including imported or
qualified-looking platform types because Java names can be shadowed. The maps follow the
[JNI type specification](https://docs.oracle.com/en/java/javase/26/docs/specs/jni/types.html)
and [native argument convention](https://docs.oracle.com/en/java/javase/26/docs/specs/jni/design.html#native-method-arguments).
Unresolved types, C aliases/widths, differing reference spellings, variadics and
annotated native conventions abstain conservatively. Name links are retained
even when declarations conflict, and the CLI exposes their status separately
from duplicate families. Local/anonymous Java binary owners are not guessed;
their declarations increment `ffi_jni_unresolved_owners` and unresolved binding
counts. Ordinary member nested classes retain lexical owner support.
Preprocessing, library resolution, C++ linkage, platform
calling conventions and runtime behavior remain unverified: both `abi_verified`
and `runtime_verified` stay false. No compiled or live JNI acceptance is claimed.
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
