# victor-codegraph

Shared **repository → Code Semantic Graph compiler**: symbol and relation extraction,
hard-bounded embeddable chunks, repository-wide resolution/deltas, diagnostics, and a
`ProximaRecord` projection. One compiler, three
consumers — Victor (owner), the ProximaDB SDK (`[codegraph]` extra), and AnvaiOps (SaaS
code-graph vertical).

> Design: ProximaDB `ADR-029` (authoritative) · Victor `ADR-014` (owner/donor) ·
> AnvaiOps `ADR-0018` (consumer). This package is the **TD-CG1** scaffold.

## Why

The same tree-sitter code→symbol+relation chunker existed twice (ProximaDB SDK `code.py`
and Victor `victor-coding`) and was about to be written a third time in AnvaiOps. This
package is the single neutral home. It merges the best of both donors and fixes their two
gaps:

- **Size-capping** — ProximaDB's `code.py` emitted one chunk per symbol with *no* size
  bound (a huge function became a huge chunk). Here, oversized symbols are body-split with
  overlap (LlamaIndex `CodeSplitter` discipline). See `sizing.py`.
- **Real JS/TS** — the donor JS/TS parser was a stub returning no symbols. Here JS/TS get a
  real tree-sitter extractor (functions, classes, methods, `const … = () =>`, imports).

## Install

Published on PyPI as [`victor-codegraph`](https://pypi.org/project/victor-codegraph/).
This source tree is version **0.10.2**; consumers should use a compatible `<1.0` pin.
It repairs nested Python definition/import shadowing: proven local definitions
retain their targets, while unmodeled conditional bindings stay unresolved.
Rebuild derived graphs when adopting the new parser; unchanged source hashes do
not invalidate old semantic results automatically. Publication is a separate gate.

```bash
# from PyPI
pip install victor-codegraph                 # Python-only (stdlib ast) path, zero native deps
pip install "victor-codegraph[treesitter]"   # + multi-language tree-sitter grammars

# dev (monorepo): editable, with tree-sitter grammars + test tooling
make -C victor-codegraph dev          # = pip install -e ../victor-contracts && pip install -e ".[dev]"
pip install -e ./victor-codegraph     # minimal editable install
```

### Releasing

CI: `.github/workflows/ci-codegraph.yml` runs the suite (editable install, grammars on) for every
PR touching `victor-codegraph/**`. Publishing: push a tag `victor-codegraph-v0.1.0` to trigger
`.github/workflows/release-codegraph.yml`, which builds and publishes via **PyPI Trusted Publishing**
(OIDC — no API token). Configure the publisher once on PyPI (owner `vjsingh1984`, repo `victor`,
workflow `release-codegraph.yml`, environments `pypi` / `testpypi`); see the header of that workflow.

## 0.9 consumer qualification and migration

| Surface | Contract / limit |
|---|---|
| Python references | Exact import aliases and dotted modules; lexical scope precedes arity. Unknown receivers, wildcard ambiguity and shadowed bindings remain explicit unresolved references. No runtime execution/type inference. |
| ES modules | Relative named/default/namespace imports, explicit named re-exports and exported arrow functions. Type-only imports produce dependencies, never runtime call evidence. Re-export traversal is cycle-safe and bounded to 64 bindings. Package/tsconfig aliases, star exports and dynamic exports remain unresolved. |
| Object-literal identity | Methods include structural object scopes, including anonymous callback ancestry. Whitespace/comments and checkout root do not affect IDs; inserting/reordering anonymous siblings can change structural ordinals. |
| Source evidence | Python call spans include UTF-8 byte offsets/lengths and end coordinates. New binding metadata is diagnostic evidence, not runtime authorization or a complete language semantic model. |
| Victor persistence | The canonical snapshot supplies symbols. CLI distinguishes attempted upserts from persisted totals, exposes partial parses and accepts exact opaque IDs; ambiguous display names are not chosen arbitrarily. Default file exclusions can differ from the package walker. |

Rebuild derived indexes after upgrading from older parser versions; an unchanged
source hash alone does not migrate older IDs/edges. Victor users can run
`victor graph index --force --no-ccg --no-embeddings`. Consumers must stage a new
complete generation and switch readers after validation; never mix old and new
symbol identities. The AnvaiOps adapter/generation migration remains consumer-owned.
The package adds no commercial runtime, account, entitlement or billing dependency.


## Use

```python
from victor_codegraph import chunk, parse, parse_repo, to_proxima_records, ChunkConfig

# Size-capped, embeddable chunks:
chunks = chunk(source, file_path="app/service.py", config=ChunkConfig(max_chunk_tokens=512))

# Symbols + relations:
parsed = parse(source, file_path="app/service.py")

# Project to the ProximaDB substrate-keystone record shape (one symbol = row+node+vector):
records = to_proxima_records(parsed, repo_graph_id="myrepo", branch_id="main",
                             embedder=my_embed_fn)  # embedder optional

# Repository-wide resolution + file/import/containment graph:
repository = parse_repo(".", repo_id="myrepo")
records = to_proxima_records(repository, repo_graph_id="myrepo")
```

## Design principles (the "best posture" this encodes)

1. Chunk at **symbol** granularity (not statement, not fixed-size).
2. **Newline-preferred and hard-bounded** — forced character splits are explicit; an optional
   tokenizer callback enforces the real model limit.
3. Extract a **Code Semantic Graph** (CALLS/EXTENDS/IMPLEMENTS/CONTAINS/IMPORTS). Full
   intra-procedural CFG/CDG/DDG is a separate optional layer, not implied here.
4. **One structural v2 ID** across symbols, chunks, relations, and records; line IDs remain aliases.
5. **Deletion-complete deltas** — incremental output can be checked against a full rebuild.
6. **Observable fallback** — every parse reports status, capability tier, and diagnostics.
7. **Explicit uncertainty** — unresolved targets become structured external-reference nodes.

## Status

`0.10.2` source version. Python (`ast`) is the full, offline frontend. JS/TS, Go, Rust,
Java, C, and C++ have conformance fixtures; other mapped tree-sitter languages report
symbols-tier or fallback status rather than implying full fidelity.

### Ambiguous declarations

Duplicate structural identities retain raw declarations in `ParsedRepository.files`
and report each conflicting source location. Repository projection emits only the
canonical FILE marker for that file, with typed diagnostics, declaration spans and
omitted symbol/relation counts in metadata. Calls into its declarations remain
unresolved; healthy imports can still point to its diagnostic marker. Repairing the
source restores normal projection through deletion-complete repository deltas.
Direct record projection rejects duplicate identities before invoking an embedder,
including when legacy OIDs are selected.

`identity_collisions` groups use the existing `symbol_id` with `resolution: ambiguous`.
Each declaration retains `name`, `qualified_name`, `symbol_type` and its full
`source_span`; these are inspectable evidence, not separately resolved graph nodes.
The record adapter forwards these groups, projection status and omitted counts only
when present, without copying arbitrary symbol metadata into exported properties.

### TypeScript grammar limitation

The current tree-sitter grammar reports a partial parse for inline import types
in generic calls, for example `load<typeof import('child_process')>()`, even though
TypeScript accepts that syntax. A named type alias (`type ChildProcessModule =
typeof import('child_process')`, then `load<ChildProcessModule>()`) is supported.
Partial status and syntax diagnostics remain visible; the parser does not rewrite
source or infer missing relationships.

### Rust graph fidelity

Rust indexing retains inline module scopes and discriminates same-named trait
implementations and conditional definitions in stable identities. Repository
resolution follows declared library modules and explicit top-level `use` paths,
including grouped imports, aliases, and local Cargo path dependencies. Hyphenated
Cargo dependency names are normalized to their Rust underscore names.

Binary-owned or ambiguous files, conditional items/imports, `#[path]` overrides,
workspace-inherited dependencies, re-export chains, macros and unknown receiver
types retain unresolved references. Glob imports retain a module dependency but
do not guess call targets. This is a conservative static graph, not rustc name
resolution or evidence that a call executes. Publish this package before a Victor
release that requires its exact repository graph contract.


### Embedding conformance (0.10.0)

The accepted FEP-0040 supplies store-neutral
`EmbeddingProfile`, `SymbolEmbedding`, `embed_symbols` and an optional
`SentenceTransformerEncoder`. The base compiler stays dependency-free. The
`embeddings` extra loads sentence-transformers only when explicitly selected;
private source is not sent to a remote model by default.

The contract uses canonical compiler IDs and `symbol-summary-v1` (name,
signature and documentation), with an immutable model revision, dimensions,
token limit, normalization, query prefix and separate input/source hashes.
Inference errors propagate; hash placeholders, zero vectors and malformed batches
are not semantic evidence. File/container symbols marked `embedding_excluded`
remain graph evidence only.

These APIs are introduced in 0.10.0 and are unavailable in 0.9.0. Install the
exact published `victor-codegraph[treesitter,embeddings]==0.10.0` when selecting
local model execution; the base package has no mandatory model dependency.
Current evidence covers
synthetic document-vector persistence parity and local LanceDB collection/query
profile enforcement. Labeled relevance and hosted lifecycle remain separate
acceptance gates. Victor binds a new LanceDB table to the complete profile in its
Arrow schema and rejects incompatible query encoders and mixed/orphan row markers.
Legacy tables require a fresh destination; they are never silently relabeled.
