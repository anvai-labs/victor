# Structural clone detection

`victor index graph similar --mode structural` finds **normalized syntax
candidates**, including functions with renamed locals, different literal values,
or edited statements. No score proves functional equivalence or safe extraction.
It complements the jscpd duplication ratchet (configured token copies)
and `--mode semantic` (embedding similarity of names, signatures and
docstrings, which pairs a Protocol with its implementation and misses copies
whose names drifted).

Implementation: `victor/core/graph_rag/structural_clones.py`,
`treesitter_clones.py`, `clone_bindings.py`, `ast_fingerprints.py` and the shared index grammar loader
`language_parsers.py`. See the [2026-11-01 verification and corrected comparison](code-similarity-audit-2026-11-01.md).

## How it works

1. **Normalise each function body** (Python `ast`, or installed tree-sitter wheels):
   - Parameter bindings receive placeholders in declaration order. Distinct lexical
     declarations retain separate identities through shadowing and captures
     (`normalized-body-v3-lexical-bindings`). Python uses compiler symbol tables,
     including inlined comprehensions; Rust handles closure parameters, match-arm
     patterns and initializer-before-binding order; JavaScript handles function
     captures, catch parameters and `var` hoisting. Named Python nested declarations
     are anchored before their call sites, and import targets survive normalization.
     Other frontends currently use a declaration subset, not complete language
     semantics. JSON `binding_support` names that distinction. Rust macros, unusual
     constant patterns and advanced destructuring still need compiler evidence.
   - Constants become their type.
   - Available `.victor/precise/<provider>.scip` dumps refine lexical names using the existing Python, TypeScript/JavaScript, Rust, Clang, Java, Ruby and Go provider registry. Different resolved callees remain distinct; absent or stale evidence falls back to lexical names.
   - Full resolved package and receiver-class identities survive in the `exact` tier, including Python `self.<member>`. Different targets can still be review candidates in the `template` tier. This corrects the earlier Python normalizer, which dropped those identities.
2. **Tiers**:
   - `exact`: identical **normalized** token streams, including renamed locals or changed constants. This is not Type-1 byte/token equality and does not justify merging.
   - `template`: normalized shapes match after global/attribute identities are erased. This suggests a possible abstraction, not proof of an injectable kernel.
   - `near`: MinHash/LSH over 5-token shingles, confirmed with exact Jaccard ≥ 0.8 (`--near-threshold`). **Edited copies.** Placeholders are unnumbered inside shingles: one inserted statement would otherwise renumber every later local and break every downstream shingle.
3. **Optional subtree metric:** `--ast-metric subtree` compares sets of normalized subtree fingerprints with exact Jaccard. Local-name repetition patterns survive alpha-renaming (`x+x` differs from `x+y`). A rare-first prefix index, length bounds and duplicate-set grouping avoid scoring every pair. A spanning forest retains the same candidate components without expanding all identical copies. `shingle` remains the default for compatibility; its LSH candidate generation is approximate and large skipped buckets are counted.
4. **Suppression:** a family whose members are *all* sanctioned is listed as suppressed with the reason, never dropped:
   - Protocol methods;
   - `@abstractmethod`;
   - literal-dense bodies (≥ 40% constants: data tables).
   - trivial Rust `todo!()` / `unimplemented!()` stubs.
5. **SCIP freshness and coordinates:**
   - Full document text, when supplied by SCIP, must match. Otherwise source mtimes and range plausibility are conservative heuristics, not content-hash certification of a dump.
   - UTF-8/16/32 range columns are translated to byte columns. Invalid ranges are omitted; sufficiently many mismatches mark the document stale. Ambiguous unspecified encodings on non-ASCII lines fall back.
   - **No dump:** name-only resolution, announced in the output.
6. **Coverage:** source frontends cover Python, Rust, JavaScript, TypeScript/TSX, Go, Java, C/C++, Ruby, Kotlin, Scala, C#, PHP, Swift, Lua and Bash. Installed grammar wheels are loaded offline through the same factory as indexing. Common Rust inline test modules/attributes are excluded unless `--include-tests` is given. The scan reuses indexing's universal build/vendor exclusions and a 1 MB file bound. Error-free tree-sitter functions can survive errors elsewhere in their file; their members carry `recovered_from_partial_file`, and the file still counts as partial. Malformed functions never become units. SCIP/FFI evidence from partially parsed files is disabled. Python can recover clean function fragments through its optional tree-sitter grammar; fragments requiring missing `nonlocal` context remain excluded. Missing grammars, compiler binding errors, unreadable/oversized files and parse errors produce `coverage_complete: false` and a nonzero CLI status while retaining available results.

Near families are transitive components. Their score is the strongest connecting
edge; it is not a lower bound on every member pair. JSON exposes the grouping,
score basis, weakest connecting edge and resolved-symbol counts. Source locations
are derived from byte offsets to avoid the affected tree-sitter 0.26 Point accessors.

## Automatic routing and experimental polyglot matching

No language list is required. Indexing and similarity share the optional
`victor-codegraph` package's extension inventory; SCIP providers only enrich
supported files. Kotlin and other AST frontends work without a matching SCIP
provider. Without the optional graph package, the base installation retains its
previous provider-derived inventory and Python analysis. Missing grammar wheels
are reported, never installed during analysis. Vue/Svelte/Astro script regions are selected automatically; Astro frontmatter uses TypeScript. Typed Markdown fences require `--include-docs`. Host line numbers and region kinds survive extraction; template expressions are excluded. External scripts, unknown script languages, ambiguous attributes, malformed blocks and region limits produce incomplete coverage. Virtual regions never inherit host SCIP ranges or runtime FFI facts. `--language` is only an optional scope filter. Syntax frontends remain
language-aware: file extensions select the appropriate AST and SCIP artifact.
Cross-language comparison then requires a shared representation rather than
comparing those different grammar node names directly.

```bash
victor index graph similar --mode structural --cross-language --json
victor index graph similar --mode structural --cross-language --ast-metric subtree
```

`--cross-language` adds the **experimental** `portable-operations-v2` schema and
a separate `cross_language` candidate tier. Parameters are anchored by position;
lexical binding identities, assignments, arithmetic/comparison operators, constants, calls,
branches, supported loops and returns use shared operation names. Version 2 adds
scoped/generic call identities, closures/captures, match and conditional patterns,
containers, destructuring, ranges, context managers, casts, borrows and error
propagation. Field order, inclusive bounds, discard positions and expression
values remain explicit; macros and other unknown constructs remain unsupported. Rust/Ruby tail
returns are represented explicitly. Subtree matching uses the same bounded,
exact Jaccard join, filtering same-language pairs before its candidate budget.
Type widths, overflow, ownership, library API equivalence and runtime effects
are not certified. Different SCIP producers' IDs are not assumed interchangeable;
the ordinary per-language tiers retain their full compiler identities.

Unsupported constructs reject the portable projection for that body and are
counted by reason. Available ordinary syntax results remain visible, and the CLI
returns nonzero for partial portable coverage. Current real accelerator code
has **167 supported / 234 eligible bodies** in the retained native subset; 67
remain explicitly unsupported, principally macros. This is representation coverage,
not accuracy or equivalence evidence for arbitrary translated algorithms.

FFI evidence is returned separately in `ffi_links`, including when there are
no duplicate families or wrappers are smaller than the clone-size threshold:

- Conventional JNI Java `native` declarations link to unique C/C++ definitions
  through JNI names, UTF-16 escaping and argument descriptors. Overloads,
  ambiguous exports and short-name precedence are handled conservatively.
  Primitive/array descriptors are lowered directly; reference binary names use
  fresh Java SCIP identities or explicit known platform types. Ambiguous package
  versus nested-class names and implicit reference types are not guessed.
  Known JNI declaration macros are blanked only in parser input, preserving byte
  offsets and original source, strings and comments.
- Existing graph FFI references can be attached when **both file hashes match**
  the scanned source. Their confidence remains `index_heuristic`; the existing
  indexer's Python/Rust FFI pass is repository-specific name matching.

These are static/index relationships with `runtime_verified: false`, not duplicate
bodies. Automatic N-API bridge resolution, dynamic JNI `RegisterNatives` tables,
runtime library loading and arbitrary ABI compatibility are not implemented.
TypeScript/Rust and Java/C++ **bodies** can still be compared without such a
binding; a Java native declaration itself has no body to compare. No similarity
candidate or binding relationship is sent to the graph-row merger.

Implementation owners are `portable_clones.py`, `polyglot_clones.py` and
`ffi_bindings.py`; positive/negative examples live in `test_polyglot_clones.py`.

The current [parity assessment and prioritized design](graph-similarity-parity-and-roadmap.md)
compares these capabilities with jscpd using retained local probes. Binding links
explicitly state `abi_verified: false`; indexed hashes and edges are read in one
SQLite snapshot. Coverage scope means selected supported frontends, and an empty
supported scope returns incomplete status rather than a clean scan.

## Historical Python feasibility experiments (2026-10-29)

These measurements predate the full SCIP identity correction and the subtree
metric; they are not validation of the current Rust or multi-language scanner.

These are retained historical observations, not a current Rust benchmark or
independent verification of precision/recall. The subtree metric has not inherited
the shingle metric's historical calibration.

| Approach | Result | Decision |
|---|---|---|
| SCIP-normalised hash | 15,428 functions in ~17 s; 70 exact + 85 template cross-file families. SCIP split 23 AST-only families whose copies call different functions; those became the `template` tier. | Shipped |
| MinHash/LSH near tier | 68 families at J ≥ 0.8 (125 at 0.7, noisier). numpy only. | Shipped, 0.8 default |
| Body embeddings (bge-small, GPU) | Reported 19K bodies in 179 s (106/s); candidate quality was not independently verified. | Model bake-off and labeled negatives required before adoption |
| Dependence-graph (CCG) isomorphism | Data exists (2.07M DDG edges). Little marginal value over the near tier on the misses below. | Deferred |

## Recall against past dedup PRs

**Labels:** functions (≥ 40 tokens) that a dedup PR deleted. **Hit:** the function was in a cross-file family on the PR's parent commit. AST-only, since there is no SCIP dump for historical commits.

| PRs | Removed functions flagged |
|---|---|
| #1554 #1555 #1556 #1561 #1564 #1565 #1567 #1571 | **39/56 = 70%** at J ≥ 0.8 (71% at 0.7, 75% at 0.6) |

The misses are:
- **diverged copies** whose logic genuinely differs: the per-vertical persona trait code that #1556 replaced with a table-driven kernel, and `TeamMemberSpec` copies with added fields;
- **same-file duplication**: the #1565 keyword tables. The scan is cross-file by default; `--same-file` adds same-file exact families.

Lowering the threshold buys little recall for much more noise, so 0.8 stays the default.

## Running it

```bash
victor index graph index --precise       # SCIP harvest requires explicit enablement
victor index graph similar --mode structural
victor index graph similar --mode structural --language rust --ast-metric subtree --json
victor index graph similar --mode structural --language typescript --language tsx --ast-metric subtree
victor index graph similar --mode structural --scope victor/agent/ --json
```


## Body vectors, token fragments and independent channels

The default `--mode semantic` retains its indexed `symbol-summary-v1` contract.
The following source modes add complementary evidence:

```bash
# Automatically selected script regions; documentation is explicit.
victor index graph similar --mode structural --include-docs --ast-metric subtree --json

# Complete function-source windows; cached pinned model weights must exist locally.
victor index graph similar --mode body --body-scope cross --json
victor index graph similar --mode body --body-cache .victor/analysis/body-vectors.sqlite --json

# Explicit locally installed jscpd 5.4.2; no npx/install/download during analysis.
victor index graph similar --mode tokens --jscpd-binary /path/to/jscpd --min-tokens 300 --json

# Independent syntax, body-vector and token channels; no AST prefilter for vectors.
victor index graph similar --mode hybrid --jscpd-binary /path/to/jscpd --json
```

Body analysis requires the graph and embeddings extras. `--body-model`,
`--body-revision`, `--body-dimension` and `--body-max-tokens` specify an immutable
model contract. The default pinned BGE model is a general text baseline, not a
claim of code-model accuracy; `quality_calibrated` is false. `--body-scope` selects
`all`, `same` or `cross` language edges **before** neighbor selection/grouping.
Existing summary vectors are never reused or overwritten.

`function-source-token-windows-v1` tokenizes the captured function declaration and body (preceding decorators,
enclosing module context and macro expansion are excluded), uses source-offset windows with overlap, checks the model's special-token budget,
and never silently truncates. Unit-normalized window vectors are averaged and
normalized again. JSON records source/input hashes, model revision, window token
spans and coverage. Unsupported window sizes fail explicitly. Cache entries use
input/profile keys and checksums; invalid inference never publishes partial
inputs. Cache hits avoid model loading. Body collection skips unused SCIP dumps
and subtree matching. Exact cosine joins remain tiled and quadratic, with a
5,000-function default bound and optional neighbor cap; no ANN scalability claim
is made.

Token analysis snapshots selected source files into a scratch directory, ignores
project jscpd configuration, validates returned paths/spans/kinds/counts, and emits
`literal_token_fragment` relations. `--rename-tokens` requests lexical identifier
normalization and labels the result as a pattern. File, total input, pair, report
and runtime budgets are explicit. RSS monitoring covers the owned process tree
where the OS permits it; restricted platforms report direct-process monitoring.
No compiler/build, source execution or project-index write occurs.

Hybrid JSON preserves channels separately with their score bases and failures.
It compares source manifests, marks additions/removals/content changes between
channels as conflicts, and never invents all-pairs matches from transitive
families. A failed model, unavailable token backend or exhausted AST pair budget
leaves the other channels visible and returns a nonzero incomplete status.
Source/representation coverage does not certify recall, behavior, ABI safety or
safe refactoring. The current [assessment and retained evidence](graph-similarity-parity-and-roadmap.md)
records measured coverage, cache behavior and unresolved work.
