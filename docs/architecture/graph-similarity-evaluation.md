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

### Source-copy contract foundation

Source inventory and source reads share an owned directory descriptor. Git runs
from that descriptor through an isolated supervisor; filesystem fallback traverses
descriptor-relative directories. A replaced project pathname fails closed instead
of silently selecting a smaller, different tree. Git inventory includes tracked and
untracked supported sources, respects ignore rules, disables FSMonitor, pagers and
optional index writes, and resolves the executable before changing directory.
Inherited Git context overrides are removed. The effective worktree is bound to
the discovered metadata ancestor, preserving nested-root prefixes and ancestor
ignore rules. Trusted supervisor checks verify directory ancestry and metadata
identity before and after Git, before publishing status. The nearest marker
selection, bounded Git-file/`commondir` contents and their referenced directory
identities are also checked. Nonstandard marker types and selectors larger than
4 KiB fail closed. These are acquisition
boundary checks, not an atomic immutable Git/filesystem snapshot. Git failures do not silently switch
to a broader filesystem policy; only missing Git or an explicit non-repository
result without repository metadata permits fallback.

Inventory has a 10-second deadline, 16 MiB path/output limit and 256 MiB owned Git
process limit. Filesystem traversal also bounds examined entries (200,000),
directories (50,000), depth (128), and elapsed work even when no file has a source
suffix. Owned process supervision supports synchronous and cancellable async
consumers without replay, reserves the leader until cleanup, publishes child PID
and exit status atomically, and rejects stale status paths. Source acquisition
still requires qualified POSIX primitives; this does not add Windows support.

`TokenOptions(fingerprints=True)` requests a bounded additional native carrier.
The adapter correlates every result with the pinned JSON token report, including
source-relative paths, exact backend spans, kind, format and token count. Duplicate
physical signatures, conflicting identities, missing results and foreign roots
are rejected. `fingerprints_complete` retains source-coverage failures. No display
family is expanded into invented pair evidence.

The `jscpdCloneHash/v1` identifier is the pinned backend's opaque 64-bit value;
it is not authentication or proof of behavior. Real backend tests cover Python,
Rust, TypeScript, C++, Java, Unicode spans and special filenames. The pinned native
SARIF carrier emits **raw path strings**, including `%`, spaces, `#` and `?`;
these are correlated literally, without URI decoding. They must be escaped by a
future standards-compliant SARIF exporter. Moves and leading line shifts preserve
the fingerprint; adding another copy increases its count. Baseline and changed-code
applications will compare compatible fingerprint counts across complete scans,
not semantic family counts or arbitrary per-edge `isNew` assignment.

The [matched observations](evidence/copy-contracts-2026-11-02.json) retain identical
40-file manifests and three fragment records before/after this foundation, plus
three correlated fingerprint records when enabled. Single local wall observations:
baseline 0.348 s, hardened inventory 0.359 s, fingerprints 0.341 s. Parent peak RSS
was 48.8/49.9/49.3 MB; child memory is separately monitored. These are diagnostic
observations, not a speed improvement or accuracy claim. The changed-file Linux
gate installs the pinned backend only when its test owner is selected and requires
the actual protocol tests to run rather than silently skipping them.

## Copy-count baselines, changed-file gates and token SARIF

The application merged in [#1665](https://github.com/anvai-labs/victor/pull/1665) provides `--baseline`, `--write-baseline`,
`--changed-file`, `--fail-on-new` and `--sarif` in **token mode**. These are
source-copy count policies; semantic components and similarity scores are not CI
gate units. Baseline creation/update is explicit and separate from comparison.
Incomplete acquisition, an unaccounted native input, incompatible profiles or
malformed artifacts cannot create/update a baseline or produce a passing gate.

Profiles bind the pinned native version/schema, token options, configured scope,
test/docs/same-file flags, selected producer-source hashes and the loaded static
source-selection policy. A producer/options/policy change requires a reviewed new
baseline. Native opaque fingerprints are grouped by kind and format; counts are
actual reported pairs, not all pairs invented from a connected family. File moves
and leading line shifts preserve counts. A count increase is attributed to a
changed file if **any current occurrence in that fingerprint group** touches that
file. Assigning arbitrary physical edges as new could miss a newly inserted copy
when the backend changes pair order. Changed-file filtering does not narrow the
configured source scan or remove unchanged comparison partners.

Git inventory includes tracked files and nonignored untracked files. Ignore edits
cannot hide tracked/committed new copies, but untracked inclusion follows the
current ignore policy. Compatibility does not freeze mutable `.gitignore`, Git
configuration or filesystem state; review ignore-policy changes separately.
The acquired source manifest identifies the bytes actually inspected.

Physical-file accounting is distinct from `statistics.total.sources`: the native
counter omits below-threshold files and can count embedded-language units in
addition to their host file. An untruncated native file summary binds recognized
paths to acquired byte counts. Missing paths receive a bounded complexity-only
control at zero token/line thresholds; it performs no duplicate detection. Only
proven ineligible omissions count as covered. Possible eligible omissions,
unsupported extensions and uncertain embedded-host omissions remain incomplete.
Main and control calls share one timeout and the same owned-process RSS/artifact
bounds. Reports expose physical `backend_analyzed_files`,
`backend_reported_source_units` and `backend_control_files` separately.

Occurrence identities include source hashes, raw backend column spans, native
fingerprint, kind, format and token count. Freeze validation recomputes these
identities and rejects duplicate physical signatures independently of supplied
IDs. Native columns remain unqualified correlation data. SARIF uses valid
line-only regions, URI-encodes raw filenames once, preserves actual pair references
and exposes backend columns only in properties. Count deltas remain grouped;
results do not invent an edge-specific `baselineState`. Incomplete/workflow-error
reports declare unsuccessful execution; a successful analysis that finds a gated
count increase retains successful SARIF execution and exits 1.

Baseline artifacts use bounded strict JSON. Publication rejects foreign files,
source/database aliases, symlinks and hardlinks, and checks destination/directory
identity before atomic replacement. Filesystem identity checks cover NFC/NFD aliases on
normalization-insensitive filesystems. Publication settles once before cancellation
propagates; comparison never silently updates the baseline.

For a configured module gate (use the identical options on both commands):

```bash
victor graph similar --path /path/to/project --mode tokens \
  --jscpd-binary /path/to/jscpd --scope src/query/ --same-file \
  --min-tokens 300 --token-min-lines 3 \
  --write-baseline /path/to/copies.json --json
victor graph similar --path /path/to/project --mode tokens \
  --jscpd-binary /path/to/jscpd --scope src/query/ --same-file \
  --min-tokens 300 --token-min-lines 3 --baseline /path/to/copies.json \
  --changed-file src/query/sql_frontend/lowering.rs --fail-on-new --sarif
```

`--same-file` includes copies inside one file; the shared command otherwise
defaults to cross-file evidence. Omit `--scope` for the entire selected project
when it fits the stated resource bounds. Baseline/SARIF integration for other
channels remains pending; no output certifies behavioral or ABI equivalence.

The [retained qualification](evidence/copy-workflow-2026-11-02.json) uses 166 frozen
ProximaDB query files, each SHA-256 checked against Git blobs at
`9226274eea996b0554aada3825856a30002a2f01`. The merged foundation `f7d4b606e` and
candidate retain identical source hashes and three native copy records. Coverage
changes from incorrectly incomplete to 166 accounted paths: 127 main-summary
files plus 39 proven ineligible controls. Separate-process wall observations are
0.564/0.614 s; explicit baseline create/compare/update and SARIF take 0.014 s in
another process. These are local diagnostics, not statistical performance or
precision/recall claims. Installed dependency metadata is recorded separately
from source checkouts selected through `PYTHONPATH`.

The earlier whole-project observation reached the default 768 MiB native RSS
budget and failed explicitly; it did not create a baseline and does not qualify
whole-project capacity. Secure Windows acquisition remains unsupported. A
supported scoped run is not whole-project or release acceptance.

### Selected-project capacity continuation

The current capacity increment adds explicit `--token-memory-mib` (32–8192),
`--token-max-file-bytes` (1–134217728) and `--token-timeout` (0.1–300 seconds).
Defaults remain 768 MiB, 1,000,000 bytes per file and 60 seconds. Main detection
and physical-file controls share the remaining deadline and the same RSS bound.
RSS monitoring polls the owned process aggregate; the configured cap is neither
a measured peak nor a hard memory reservation. Existing total source/report
bounds remain. Oversized or selected symlink inputs still fail coverage.

`--token-exclude` is repeatable in token mode. Patterns are root-relative POSIX
globs: literal paths match exactly, `*`, `?` and character classes remain within
one directory segment, and a whole `**` segment matches zero or more directories.
Use `dir/**` for a subtree or `**/*.rs` at any depth. Absolute paths, empty, dot
and parent segments are rejected. Exclusions are normalized and recorded in
profile v2, so changing them requires an explicit reviewed baseline update.
Runtime budgets do not change copy identity. Exact v1 baseline bytes remain
readable and recognizable for explicit replacement; comparing them with v2
fails as incompatible rather than silently resetting counts.

The [retained observation](evidence/source-capacity-2026-11-02.json) accounts for
all 3,418 selected ProximaDB files: 2,733 main-summary paths and 685 proven
ineligible controls, with 65 native copy pairs. Every acquired hash was verified
against its Git blob at `9226274eea996b0554aada3825856a30002a2f01`. The selection
explicitly excludes four symlink paths, recorded with their targets, and includes
a 2,145,860-byte generated Rust file. Source manifests and all native fingerprint
records match the earlier isolated resource-override probe; there are no new
token matching semantics. Baseline create/compare/update succeeds, and its SARIF
passes the official OASIS schema including URI format checks.

The single macOS arm64 scan observed 6.969 seconds; application/schema work took
0.150 seconds. This is bounded capacity evidence for the stated policy and host,
not a statistical speed comparison, release qualification, clone accuracy
estimate or semantic-equivalence claim. Reproduce with pinned source bytes and
an explicitly installed jscpd 5.4.2:

```bash
python scripts/benchmark_graph_similarity_channels.py /path/to/project \
  --mode tokens --same-file --exclude-docs --min-tokens 300 --token-min-lines 3 \
  --token-fingerprints --token-memory-mib 2048 --token-max-file-bytes 8388608 \
  --token-timeout 120 --jscpd /path/to/jscpd \
  --token-exclude clients/go/codegen/openapi_31_to_30.py \
  --token-exclude clients/rust/codegen/openapi_31_to_30.py \
  --token-exclude examples/utilities/profiling/memory_profile.py \
  --token-exclude tools/demos/wal_search_achievement_demo.py \
  --output /path/to/observation.json
```

Reproduce source-channel observations with the same frozen input hashes,
interpreter and pinned backend, changing only the implementation checkout:

```bash
python scripts/benchmark_graph_similarity_channels.py /path/to/frozen/project \
  --mode tokens --scope src/query/ --same-file --exclude-docs \
  --min-tokens 300 --token-min-lines 3 --token-fingerprints \
  --jscpd /path/to/jscpd --output /outside/source/query-observation.json
```

Success and failure SARIF outputs were checked against the
[official OASIS schema](https://github.com/oasis-tcs/sarif-spec/blob/main/sarif-2.1/schema/sarif-schema-2.1.0.json),
including URI formats; its digest is retained in the qualification artifact.

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
handles, ABI verification, ANN/incremental indexing and broader-channel SARIF remain
separate subsequent increments; token-copy baselines and SARIF are merged in #1665.

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
