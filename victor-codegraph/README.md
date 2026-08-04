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
This source tree is version **0.9.0**; consumers should use a compatible `<1.0` pin.

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

`0.9.0` source version. Python (`ast`) is the full, offline frontend. JS/TS, Go, Rust,
Java, C, and C++ have conformance fixtures; other mapped tree-sitter languages report
symbols-tier or fallback status rather than implying full fidelity.
