# Code similarity report verification and implementation

The subsequent [parity assessment and prioritized design](graph-similarity-parity-and-roadmap.md)
owns current maturity and follow-up recommendations. Historical measurements below
predate its `normalized-body-v2-parameter-roles` hardening; they are not refreshed
performance or quality certifications for that normalization schema.

**Date:** 2026-11-01. **Scope:** verify the supplied ProximaDB comparison and extend
Victor's structural scanner. At this historical checkpoint implementation was local on
`feature/codegraph-deduplication`, based on `095e981bf`; no index, source or vector
data in ProximaDB was changed. Source inspected at ProximaDB
`a4b762309f2079a9914b6ed34f37c9c6973e089f`.

## Corrections to the supplied report

| Claim | Verified result |
| --- | --- |
| 53 proven cross-file families of functionally equivalent code | Not established. The original raw result/configuration was not provided, and summary embeddings do not compare bodies. Families are connected components; their displayed similarity was the **maximum edge**, not every member pair's score. |
| `test_stats` family is replicated test logic | Counterexample: the geospatial test checks five entries and hash/point statistics; the CDC dedup test checks three calls, one unique event and two duplicates. Both indexed inputs are exactly `test_stats #[cfg(test)] ()`. Identical inputs explain identical vectors without equivalent code. |
| `persist_vector_segment` is one implementation copied four times | Counterexample: listed implementations include `unimplemented!("not used")`, `unimplemented!()`, `Ok(())`, and a recorder that pushes path/data into a mutex-protected vector. They serve different test-double roles. |
| `set_trace_tag` is an exact function copy | The catalog test implementation mutates an in-memory record through `with_trace_mut`; the runtime implementation takes an async mutation lock, reads a trace, edits it and persists it. Sharing an interface does not make these the same implementation. |
| Opposite SQL/Proxima conversions can be unified because similarity is 1 | Conversion direction and lossiness are part of the contract. Similar names/types and a high embedding score do not establish substitutability. |
| jscpd has no Rust AST similarity | False for the installed **jscpd 5.4.2**. Its help and a real scan confirm Rust AST support. |
| jscpd has no semantic option | False as a capability statement. The installed binary offers experimental **function-body** embeddings via a local model or HTTP provider. That detector was not benchmarked here; no model was downloaded. |
| jscpd's exact pairs are a strict subset of Victor's semantic families | Not demonstrated and not the same unit of measurement. Token blocks include same-file clones; Victor defaults to cross-file symbol components. A pair-to-family overlap study is required. |
| Victor is “best-in-class” at Type-4 equivalence | Unsupported without labeled positives/negatives, precision/recall and comparable inputs. Neither a cosine nor an AST score proves behavioral equivalence. |

The current partition-test summary vectors have cosine **0.912894** for
`partition_key_hash_id_is_none` versus `test_metadata_bounds_partition_key`.
The opposite conversion summaries have cosine **0.983905**. These are current
individual-pair measurements, not a reconstruction of the report's historical
family maxima. Review source bodies, effects, error paths and contracts before
extracting utilities or macros. No proposed ProximaDB refactors were applied.

## Reproduced measurements and index evidence

Commands used the cached jscpd 5.4.2 binary, `src/ crates/`, default configuration,
JSON reports written outside the repository, and no semantic model download:

| Configuration | Result |
| --- | --- |
| `--min-tokens 300 --format rust` | 281 exact pairs; 1,895 analyzed files; 1,389,382 lines; 7,801,165 tokens |
| Add `--mode weak` | 27 exact pairs; 1,852 analyzed files; license comment noise removed |
| `--min-tokens 100 --format rust --similarity 0.8 --kind ast` | **2,719 AST pairs in 4.70 s**; 2,027 analyzed files |

The last command's emitted AST units can be smaller than 100 tokens, so it must
not be described as a per-function 100-token cutoff. Different filters change
the file inventory. License classification also needs a declared rule: 262
default-result fragments contain the Apache license sentence, 258 of those span
a complete block comment. The report's exact “260 headers / 21 real clones” split
was not independently reproduced. `--mode weak` is the verified comment-removal
option; `--ignore-pattern "// Copyright"` does not cover every block header.

A stable APFS clone of the WAL-free project database was read after checking the
original's size/mtime and absence of a WAL before and after copying. It contained
**1,646,509 nodes, 5,016,634 edges, and 3,853 tracked file hashes**. Every tracked
file hash matched the current source. This does not prove coverage of new files
or completeness of partial parses. No indexed Git revision was recorded, so a
“54 commits stale” claim cannot be derived from this database.

The latest summary is **2026-11-01 14:30:01**, 3,857 processed files, **23 partial
files**, and precise indexing **disabled** for that pass. The watcher sidecar
says stopped. The Rust SCIP artifact is present: 2,587 documents, 2,542,045
occurrences and 380,274 definition-marked occurrences, including locals. These
are not the same metric as joined callable definitions or persisted compiler
edges; the older “176K edges / 4,639 definitions” run was not reconstructed.

The current Lance table has **69,714 rows**, 384 dimensions, and a validated
`BAAI/bge-small-en-v1.5` `symbol-summary-v1` profile. Embedding references in the
SQLite snapshot also number 69,714. The environments checked have LanceDB
0.38.0 and 0.30.2, not the claimed 0.41.0. Installed versions, index counts and
freshness must be attached to the environment and snapshot, not copied forward
from a previous audit.

## What jscpd's Rust AST detector does

The reviewed upstream source is commit
[`0b519608c47d287f59018ce4ad7f9939ac93e0c8`](https://github.com/kucherenko/jscpd/tree/0b519608c47d287f59018ce4ad7f9939ac93e0c8/rust/crates/cpd-similarity/src).
Its [syntax normalizer](https://github.com/kucherenko/jscpd/blob/0b519608c47d287f59018ce4ad7f9939ac93e0c8/rust/crates/cpd-similarity/src/syntax.rs)
uses grammar tables to identify callable bodies, calls, macros and operators.
Its [fingerprints](https://github.com/kucherenko/jscpd/blob/0b519608c47d287f59018ce4ad7f9939ac93e0c8/rust/crates/cpd-similarity/src/prints.rs)
preserve name repetition under renaming. Its
[join](https://github.com/kucherenko/jscpd/blob/0b519608c47d287f59018ce4ad7f9939ac93e0c8/rust/crates/cpd-similarity/src/join.rs)
uses rare-first prefixes and size bounds for an exact Jaccard threshold search,
grouping duplicate fingerprint sets. This review informed an independent Python
implementation using Victor's existing grammar dependencies and SCIP registry;
Victor does not shell out to jscpd or add its Rust crate/toolchain.

## Implemented in Victor

- Shared offline grammar loading for indexing and analysis; supported structural
  frontends cover Python, Rust, JavaScript, TypeScript/TSX, Go, Java, C/C++ and
  Ruby. Ordinary syntax tiers match within a language; the subsequent experimental
  portable tier described below can compare different languages.
- `--ast-metric subtree` adds parameterized subtree fingerprints, exact
  prefix-filtered Jaccard search, duplicate-set scoring and valid connecting
  forests. `--ast-metric shingle` retains existing approximate LSH behavior.
  The two metrics have different calibration; neither inherits the other's
  historical precision or recall figures.
- Per-language SCIP dumps are selected through existing providers. Full document
  text, when present, must match; otherwise mtime/range checks are conservative
  heuristics. UTF-8/16/32 coordinates are translated to byte columns. Missing,
  stale or unreadable compiler evidence falls back to syntax. Converted SCIP
  ranges are materialized one document at a time. Python now preserves full
  package and receiver-class identities too; the previous implementation could
  label different `self` method targets as normalized exact matches. Resolution
  counts include Python as well as the shared tree-sitter frontends.
- Common Rust inline test attributes/modules and trivial unimplemented stubs are
  handled. Indexing's universal vendor/build exclusions and a 1 MB source-file
  bound prevent bundled assets from silently dominating the audit.
- Missing grammars, unreadable/oversized files and parse errors are visible as
  incomplete coverage; the CLI retains results and exits nonzero. A byte-offset
  location path avoids the tree-sitter 0.26 Point accessor failure reproduced
  above line 256 on these ARM64 wheels.
- Semantic analysis remains **summary** analysis. It validates the embedding
  profile and finite/nonzero vectors, excludes root test paths and known Rust
  inline-test summaries by default, uses Arrow vector buffers, caps the corpus
  explicitly, and bounds each cosine tile to 32 MiB. Cross-file pair budgets are
  applied after same-file filtering. Components expose strongest/weakest matching
  edges and do not claim pairwise equivalence.

On the Rust-only repository scan, 18,218 eligible bodies across 2,353 parsed files
produced **94 normalized exact, 171 template, and 35 near families**. Five files
had parse errors and one exceeded the size bound; these are partial results, not
verified duplicate counts. Repeated runs before/after lazy SCIP conversion
produced identical serialized family results. Peak process RSS fell from
2,222,292,992 to **1,768,521,728 bytes** (20.4%); wall time was 59.36 versus 55.42 s,
CPU time 51.00 versus 46.08 s. These are paired local observations on ARM64 macOS,
not a cross-platform certification or an apples-to-apples speed claim against
jscpd's differently scoped scan. No native-language rewrite was needed.

## Reproduction and workflow

Use an environment containing this checkout, its current `victor-codegraph`, the
existing language extras, and optional `protobuf` for SCIP reading:

```bash
victor index graph similar --path /path/to/proximaDB --mode structural \
  --language rust --ast-metric subtree --near-threshold 0.8 --json
victor index graph similar --path /path/to/project --mode structural \
  --language typescript --language tsx --ast-metric subtree --json
python -m scripts.benchmark_graph_similarity /path/to/proximaDB \
  --language rust --metric subtree --output /tmp/structural-audit.json
jscpd --min-tokens 300 --format rust --mode weak src/ crates/
```

For a CI ratchet, pin jscpd's version/configuration and use its baseline/new-clone
features. Do not turn exploratory similarity counts into a CI refactoring gate
without labeled candidate review. If graph/embedding refresh is actually needed,
use `victor index graph index --path ... --embeddings --precise` and inspect the
partial/error/harvest summary. `victor index --force` targets the separate semantic
index entry point; a commit count or stopped watcher alone is not freshness proof.

Tests cover real grammars, Rust SCIP identity separation, Unicode coordinates,
stale-dump fallback, source exclusions, native locations above line 256, consistent
renaming, exact-join parity with exhaustive scoring, connecting-component parity,
coverage errors, profile rejection, test filtering and bounded cosine workspace.
The worktree retains the earlier Phase 1 graph-row repair and TUI changes; source
similarity candidates are never passed to the graph-row merger automatically.

Structural similarity reads source on demand and consumes existing SCIP artifacts;
it does not persist a second AST fingerprint index. `index --precise` builds the
compiler evidence. Incremental fingerprint persistence needs a separately measured
cache design if repeated scans become a demonstrated user-level bottleneck.

## Initial validation and continuation

At this initial checkpoint all implementation remained uncommitted in
`.worktrees/codegraph-deduplication` on `feature/codegraph-deduplication`.
No push, PR, merge, release or ProximaDB mutation is implied by these checks.

| Gate | Local result |
| --- | --- |
| Focused similarity, graph-row repair, CLI, Markdown/timeline and concurrency tests | **177 passed**, using Python 3.12.6 with protobuf and LanceDB 0.30.2 |
| Indexing/staleness/CLI, architecture/contracts and SQLite regression set | **412 passed, 1 skipped**, Python 3.12.16 with LanceDB 0.38.0 |
| Wider graph-RAG, graph storage, embeddings and affected CLI/UI set | **991 passed, 41 skipped**, one existing Mermaid SVG test failed because the isolated test HOME hid the installed browser cache |
| Isolated retry of that Mermaid SVG test | **1 passed** with `PUPPETEER_CACHE_DIR=/Users/vijaysingh/.cache/puppeteer` and browser-launch permission; no production or fixture change |
| Full repository collection | **37,470 tests collected** |
| Black / Ruff / MyPy / whitespace | All changed Python files format/lint clean; all 13 changed production/benchmark files type-check; `git diff --check` passes |

These are targeted/affected checks and collection, not execution of the entire
37,470-test suite. Optional SCIP tests are exercised in the protobuf environment;
they skip when that optional reader dependency is absent. No new wheel or native
toolchain was added; grammar behavior was measured on ARM64 macOS, not certified
across all supported operating systems.

The local adversarial pass added negative tests for mixed/unknown embedding
profiles, same-file budget noise, transitive family scores, cross-language
matching, distinct compiler targets, stale/Unicode SCIP ranges, incomplete parse
coverage and nonexistent roots. Exact Jaccard joins and compressed components
are checked against exhaustive reference results. The repository's independent
review and publication gates remain required before a first push.

From the worktree, with the protobuf-capable development environment activated,
the executable next verification step is:

```bash
export PYTHONPATH="$PWD:$PWD/victor-codegraph"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest --no-cov -q \
  tests/unit/core/graph_rag/test_ast_fingerprints.py \
  tests/unit/core/graph_rag/test_multilanguage_clones.py \
  tests/unit/core/graph_rag/test_semantic_clone_store.py \
  tests/unit/core/graph_rag/test_semantic_clones.py \
  tests/unit/core/graph_rag/test_structural_clones.py \
  tests/unit/ui/commands/test_graph_similar_cli.py \
  tests/integration/codegraph/test_deduplication.py \
  tests/unit/ui/commands/test_graph_dedup_cli.py \
  tests/unit/ui/rendering/test_markdown_tables.py \
  tests/unit/ui/tui/test_wire_timeline.py \
  tests/unit/storage/test_sqlite_graph_store_concurrency.py
```

## Polyglot / FFI follow-up

The subsequent request adds automatic extension routing from the **same registry
as precise indexing**, replacing a separate suffix inventory. Index fallback
definitions now cover C/C++, TSX and Ruby, including C/C++ declarator names rather
than return-type names. Conventional JNI declaration macros are handled in shared
parser input without changing source bytes or offsets. Uppercase suffixes are
recognized too. A language-specific parser is still necessary, but users do not
need to select it or enumerate languages.

```bash
victor index graph similar --mode structural --cross-language --json
```

The opt-in `portable-operations-v1` tier projects common operations across Python,
Rust, JS/TS/TSX, C/C++, Go, Java and Ruby. Positive examples include **TypeScript ↔
Rust, C ↔ C++, and Java ↔ C++**. Parameter declarations anchor first-use
fingerprints so `x-y` cannot become the same full fingerprint as `y-x` merely by
renaming. Operators, constants and supported control/evaluation structure survive.
The tier remains experimental: foreign types, ownership, overflow, library API
aliases, SIMD and arbitrary algorithm equivalence are not established by it.

FFI relationships are returned separately from similarity families:

- Java `native` declarations have no body. Conventional JNI declarations and
  C/C++ exports are connected as static binding candidates with JNI UTF-16 name
  escaping, overload argument descriptors and short-name precedence. Reference
  binary type identities can come from fresh Java SCIP evidence. Ambiguous
  exports/types remain unresolved. Runtime loading and dynamic `RegisterNatives`
  are not inferred. Rules follow the [JNI design specification](https://docs.oracle.com/en/java/javase/25/docs/specs/jni/design.html#resolving-native-method-names).
- Existing indexed FFI references are read only when both source content hashes
  match. Their provenance is explicitly `index_heuristic`. Thin wrappers remain
  visible even when they are below the body-clone size threshold.
- TypeScript/Rust body comparison does **not** imply automatic N-API binding
  recovery. No general N-API bridge resolver was added; existing indexed evidence
  can be consumed, but the current indexer's FFI producer remains repository-
  specific Python/Rust name matching.

### Measured scope and limitations

The portable schema is deliberately bounded, rather than silently deleting
unsupported operations. A controlled 1,000-function fixture (500 Python/Rust
translated pairs, 20 files, threshold 1.0, minimum 5 AST tokens) found all **500
cross-language families** with zero unsupported bodies. Three sequential local
runs in separate Python processes measured median wall **0.200 s without** versus
**0.317 s with** portable analysis, CPU **0.193 / 0.308 s**, and process RSS
**50,659,328 / 53,428,224 bytes**. These are local ARM64 observations, not accuracy
or throughput claims for arbitrary projects.

A read-only copied snapshot of Victor's actual accelerator sources (16 Python,
24 Rust files; 234 bodies meeting the normal 40-token threshold) measured
**0.425 / 0.429 s** wall, **0.415 / 0.422 s** CPU, approximately **56.2 MB** RSS.
Only **3 / 234 bodies** had a supported portable projection, and **no cross-language
families** were found. The other 231 were explicitly reported unsupported:
scoped identifiers, closures, `match`, context managers and other constructs.
This is a substantial coverage limitation, not evidence that these accelerator
implementations have no shared behavior. The existing ordinary AST tiers still
run, and partial portable coverage gives a nonzero CLI status with available JSON.

The copied corpus was every `*.py` directly in `victor/native/python` and
`victor/native/rust`, plus every `*.rs` directly in
`rust/crates/python-bindings/src`. It contained no copied SCIP or graph database.
The controlled fixture uses 500 indexed constants in this translated template:

```python
def f_i(x, y):
    total = x + y
    if total > i:
        return total * 2
    return total - 1
```

The Rust variant has equivalent assignments/branches with an implicit final
return; each language is split into ten files of fifty definitions. Reproduce
the scan with `python -m scripts.benchmark_graph_similarity FIXTURE --threshold 1
--min-nodes 5 --cross-language --output REPORT.json`; omit `--cross-language` for
the ordinary AST baseline. Wider native parity detection needs further operation
lowering, binding-specific evidence and labeled evaluation. It must not be
advertised as general Type-4 equivalence or justify removing fallback/native
implementations that intentionally coexist.

### Follow-up validation

The focused gate, including protobuf-backed Java reference-type resolution,
passes **214 tests** after the parameter-position and FFI size-filter corrections.
The wider graph-RAG/storage/CLI/UI and architecture set passed **1,249 tests**,
with 42 skips and the unchanged Mermaid browser integration test deselected; its
cache-isolation issue and earlier successful isolated retry are recorded above.
The final portable/FFI edge cases are covered by the subsequent focused gate.
Add `tests/unit/core/graph_rag/test_polyglot_clones.py` to the focused command above
when reproducing that checkpoint. The implementation was local/uncommitted on the same branch;
independent review and publication gates were pending.
Final Black/Ruff/whitespace checks pass, MyPy passes all **16** changed
production/benchmark files, and full collection passes with **37,507 tests**.
