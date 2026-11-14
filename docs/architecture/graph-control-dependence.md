# Qualified control-dependence inputs

The builtin CCG builder previously labelled every reachable descendant of a
condition as control-dependent, including joins and subsequent statements. It
also derived statement IDs from line ranges, so calls sharing a line collided.
These are graph-construction errors; embedding thresholds cannot repair them.

## Graph calculation and limits

`victor/core/graph_rag/control_dependence.py` computes exit-path postdominance for
an explicit graph with integer bitsets and a worklist. A dependency exists when
a node postdominates some successor but does not strictly postdominate the
source; loop self-dependence is retained. Joins common to all branches are
excluded. The graph formulation follows
[Ferrante, Ottenstein and Warren](https://bears.ece.ucsb.edu/class/ece253/papers/ferrante87.pdf).
It quantifies exit-reaching paths and does not establish termination.

Callers can supply terminal exits or explicitly accept sinks inferred from the
provided graph. Every node must reach an exit; reverse-unreachable regions cause
a whole-graph abstention. Missing endpoints and nonterminal declared exits reject
as malformed. Budgets cover nodes, input edges, dataflow work and output pairs;
exhaustion returns no partial dependency set. These bounds apply to the graph
calculation, not the entire source parser/indexing pipeline.

The reference is checked against an independent vertex-deletion oracle over
4,368 small graphs, plus targeted branch, loop, early-return, malformed-input and
budget cases. Differential native tests compare the consumed graph facade with
the Python reference, including physical identities, metadata and structural
edges after leading blank lines and CRLF input.

## Source identity and provenance

Statement identity v2 hashes the path, AST kind and original parsed byte span.
File reads preserve CRLF bytes. Python line geometry uses byte offsets and LF
boundaries, avoiding the affected tree-sitter Point accessors. Direct helpers
need the full original source when the retained root text trims a prefix or
parser input is unavailable; split lines cannot recover that byte geometry.
Synthetic ASTs with explicit Point tuples remain supported as test inputs.

Builtin/native-facade node metadata records `cdg_algorithm`, `cdg_complete` and
`cdg_reason`; emitted edges record `postdominance-exit-paths-v1` and their exit
basis. Both retain `cfg_provenance=source_heuristic` and
`program_semantics_verified=false`. Correct postdominance over a supplied graph
does not qualify the heuristic source CFG: line adjacency, incomplete language
classification, exceptions and callable boundaries still need stronger
provenance and validation. Provider graphs remain provider-owned and are not
upgraded to compiler evidence by this normalization. Unversioned historical CDG
edges are not evidence of this algorithm.

Existing project graph data is derived state. Reindex after the identity change
with `victor graph index --force`; do not mix historical CCG identities with v2
records. Similarity channels continue to report their own evidence. No weighted
graph-context confidence or semantic-equivalence claim is introduced.

## Native and installation boundary

The existing Rust/PyO3 package advances to `victor-native` 0.26.0 and exports
`CCG_STATEMENT_SCHEMA=2`. Python declines old/incompatible statement schemas and
retains its typed fallback. Native batches return statement/CFG/DDG inputs;
`cdg_edges` is empty. The facade derives CDG once with the bounded Python
reference. The old quadratic reachability expansion is removed from Rust rather
than preserved as unused output. UTF-8 previews truncate Unicode characters,
matching Python slicing without splitting an encoded character.

The local portable release wheel and the native/reference differential suites
are separate from cross-platform acceptance. The affected-change workflow
`native-ccg-qualification.yml` builds release wheels with generic CPU settings and
installs them without core/optional extras for Python 3.12 and 3.13 on Linux x64
and ARM64, macOS ARM64, and Windows x64. macOS support is Apple Silicon only;
Intel macOS is not a release platform. Its standalone checker verifies
distribution/version/schema identity, byte-span identities, Unicode previews and
the raw CCG contract. All applicable PR checks must pass before merging; this
conditional workflow is additional to the required `CI Success` aggregate.

These wheel checks do not qualify secure Windows source scanning, JVM/FFI ABI,
runtime bindings or released artifacts. Until the new native version is
published, source checkouts can build/install the wheel locally or use the core
Python fallback; no package publication is performed by this change.

## Measurement

`scripts/benchmark_ccg_dependence.py` evaluates an explicit trusted immutable
snapshot with the same input paths/hashes for both implementations. It times
source reads, parsing and returned graph construction; imports, builder creation,
manifest hashing and report serialization are excluded. Run zero can include
lazy initialization. Native/fallback dispatch, duplicate node IDs, abstentions
and source-selection limits accompany timings. Process peak RSS is not an
incremental allocation measurement. A changed source or implementation snapshot
invalidates the report.

```bash
PYTHONPATH=.:victor-codegraph python scripts/benchmark_ccg_dependence.py \
  --root /path/to/immutable-snapshot --runs 5
PYTHONPATH=.:victor-codegraph python scripts/benchmark_ccg_dependence.py \
  --root /path/to/immutable-snapshot --scope victor --runs 1
```

The source snapshot and installed wheel must be pinned separately. Native and
Python legs can be selected explicitly; a requested native leg can still fall
back, so use the reported dispatch rather than assuming a wheel was used.
Graph shape changes in this repair; timings do not establish semantic accuracy
or equivalence of the original and corrected graphs.

The [2026-11-02 evidence](evidence/graph-control-dependence-2026-11-02.json)
pins the shared source snapshot, evaluated Python implementations, portable
native wheels and per-file observations. Five runs of the same seven-file
snapshot emit 950 statements. After excluding run zero, median returned-graph
wall time is 170.32ms before and 81.81ms after; process peak RSS is 332.02MB
versus 266.49MB. The original output has 885 unique IDs summed within files and
34,667 reachability-labelled CDG edges; the corrected output has 950 unique IDs
and 132 exit-path dependencies. These graph outputs have different meanings.

The larger candidate run covers 2,055 Python files / 24.93MB of source and
431,921 statements, using the native leg for every selected file. It takes
55.51s wall / 50.50s CPU with 295.75MB process peak RSS. Every file has distinct
statement IDs internally. `state_graph_adapter.py` abstains for non-exit-reaching
nodes. The explicit 64KiB file bound excludes 47 larger files; this is a bounded
repository workload, not a complete repository or semantic-quality audit.
