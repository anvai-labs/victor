# Quickstart: Durable Code Memory with ProximaDB

Give an agent a memory of your codebase that survives the session: index a repository
into [ProximaDB](https://github.com/anvai-labs/proximaDB) (a multi-model vector + graph +
document database, by the same author as Victor) and query it two ways —

- **Semantic recall** — "where do we validate JWT bearer tokens?"
- **Structural recall** — "who calls `parse_jwt`?"

This walkthrough uses only published artifacts: the `vjsingh1984/proximadb` Docker image
and the [`victor-codegraph`](https://pypi.org/project/victor-codegraph/) PyPI package —
the shared code→Code-Property-Graph chunker that Victor, the ProximaDB SDK, and AnvaiOps
all consume (see [ADR-014](architecture/adr/014-shared-codegraph-chunker-package.md)).
Everything below was run end-to-end against `vjsingh1984/proximadb:0.2.0`.

## Integration status (honest version)

| Piece | Status |
|---|---|
| `victor-codegraph` → `ProximaRecord` projection (this quickstart) | **Shipped** — PyPI `victor-codegraph>=0.1.2`, pure Python, no Victor install required |
| Victor's in-tree ProximaDB backends (`EmbeddingRegistry` providers `proximadb` / `proximadb_multi`, `create_graph_store("proxima")`, per-repo `.victor/graph_backend` marker) | **Implemented, flag-gated** — requires the `proximadb_sdk` Python SDK, which is not yet published to PyPI (install it from the ProximaDB repo, `clients/python`). SQLite + LanceDB remain Victor's defaults; nothing flips automatically. |
| Conversational-memory backend + multi-tenant service mode | **In progress** — see [ProximaDB as the CCG Backend](architecture/proximadb-codegraph-backend.md) and VISION.md bet 4 ("durable code memory") |

## Prerequisites

- Docker
- Python 3.11+
- ~10 minutes (most of it is the one-time embedding-model download, ~130 MB)

## 1. Start ProximaDB

```bash
docker run -d --rm --name proximadb -p 5678:5678 vjsingh1984/proximadb:0.2.0
curl -s http://localhost:5678/health
```

Use an explicit version tag — the image does not publish a `latest` tag. Port 5678 is
the REST API; the image also serves gRPC on 5679 if you map it.

## 2. Install the Python pieces

```bash
pip install "victor-codegraph[treesitter]" sentence-transformers httpx
```

`victor-codegraph[treesitter]` brings the multi-language grammars (Python, JS/TS, Go,
Rust, Java, ...). `sentence-transformers` supplies `BAAI/bge-small-en-v1.5` — the same
384-d embedding model Victor uses for its code memory.

## 3. Index a repository

Save as `index_code_memory.py`:

```python
"""Index a repository into ProximaDB as durable, queryable code memory."""

import os
from pathlib import Path

import httpx
from sentence_transformers import SentenceTransformer

from victor_codegraph import parse_repo, to_proxima_records

PROXIMADB = os.environ.get("PROXIMADB_URL", "http://localhost:5678")
REPO = Path(os.environ.get("REPO", ".")).resolve()
REPO_ID = REPO.name
COLLECTION = f"{REPO_ID}_codegraph"
DIM = 384

# Same embedding model Victor uses for its code memory (384-d).
model = SentenceTransformer("BAAI/bge-small-en-v1.5")


def embed(text: str) -> list[float]:
    return model.encode(text, normalize_embeddings=True).tolist()


client = httpx.Client(base_url=PROXIMADB, timeout=120.0)

# 1. Create the collection (idempotent: "already exists" is fine).
r = client.post("/api/v2/collections", json={"name": COLLECTION, "dimension": DIM})
assert r.status_code in (200, 201, 409) or "COLLECTION_EXISTS" in r.text, r.text

# 2. Parse the repo into symbols + call relations, projected to ProximaRecords.
repository = parse_repo(REPO, repo_id=REPO_ID)
records = to_proxima_records(repository, repo_graph_id=REPO_ID, embedder=embed)


# 3. Translate to the REST v2 wire shape ({id, vector, props}) and batch-upsert.
def to_wire(rec: dict) -> dict:
    props = {k: v for k, v in rec["props"].items() if v is not None}
    if "edge" in rec:  # call/contains relation: endpoints + type become props
        edge = rec["edge"]
        props.update(edge, record_type="edge")
        rid = f'{edge["from_oid"]}->{edge["edge_type"]}->{edge["to_oid"]}'
        return {"id": rid, "vector": [0.0] * DIM, "props": props}
    props["record_type"] = "symbol"
    vec = rec["embeddings"][0]["values"] if rec["embeddings"] else [0.0] * DIM
    return {"id": rec["oid"], "vector": vec, "props": props}


wire = [to_wire(rec) for rec in records]
for i in range(0, len(wire), 64):
    r = client.post(
        f"/api/v2/collections/{COLLECTION}/records/batch",
        json={"records": wire[i : i + 64], "upsert": True, "validate_schema": False},
    )
    assert r.status_code in (200, 201, 202, 207), r.text

n_symbols = sum(1 for w in wire if w["props"]["record_type"] == "symbol")
print(f"indexed {n_symbols} symbols + {len(wire) - n_symbols} edges into {COLLECTION}")


def prop(record: dict, key: str):
    """Unwrap ProximaDB's typed property values ({'type': ..., 'value': ...})."""
    v = (record.get("props") or record.get("metadata") or {}).get(key)
    return v.get("value") if isinstance(v, dict) and "value" in v else v


# 4. Semantic recall: ask a question about the codebase.
question = "where do we validate JWT bearer tokens?"
r = client.post(
    f"/api/v2/collections/{COLLECTION}/search",
    json={"vector": embed(question), "top_k": 3, "filter": {"record_type": "symbol"}},
)
r.raise_for_status()
print(f"\nQ: {question}")
for hit in r.json()["results"]:
    print(f'  {prop(hit, "fully_qualified_name")}  ({prop(hit, "file")}:{prop(hit, "line")})')

# 5. Structural recall: who calls parse_jwt?
r = client.post(
    f"/api/v2/collections/{COLLECTION}/records/scan",
    json={"filter": {"record_type": "symbol", "name": "parse_jwt"}},
)
r.raise_for_status()
target = r.json()["records"][0]
r = client.post(
    f"/api/v2/collections/{COLLECTION}/records/scan",
    json={"filter": {"record_type": "edge", "edge_type": "CALLS", "to_oid": target["id"]}},
)
r.raise_for_status()
print("\nWho calls parse_jwt?")
for edge in r.json()["records"]:
    caller = client.post(
        f"/api/v2/collections/{COLLECTION}/records/scan",
        json={"filter": {"record_type": "symbol", "stable_oid": prop(edge, "from_oid")}},
    ).json()["records"]
    if caller:
        print(f'  {prop(caller[0], "fully_qualified_name")}  (call at line {prop(edge, "line")})')
```

Run it against any repo (steps 4–5 assume the repo defines a `parse_jwt` — adapt the
question and symbol name to yours):

```bash
REPO=/path/to/your/repo python index_code_memory.py
```

Against a two-file demo repo (`auth.py` with `parse_jwt`/`validate_token`, `api.py`
calling them) this prints:

```
indexed 3 symbols + 9 edges into demo_repo_codegraph

Q: where do we validate JWT bearer tokens?
  auth.py::validate_token  (auth.py:13)
  auth.py::parse_jwt  (auth.py:6)
  api.py::handle_request  (api.py:5)

Who calls parse_jwt?
  auth.py::validate_token  (call at line 15)
```

## What just happened

- `victor_codegraph.parse` extracted **symbols** (functions/classes/methods with
  locations, signatures, docstrings) and **relations** (CALLS/CONTAINS/...) at AST
  granularity.
- `to_proxima_records` projected each symbol to the ProximaDB substrate-keystone shape:
  **one entity = one record** — a relational row (props), a graph node (oid + edges), and
  a vector — addressed by a single deterministic, line-independent `oid`. Re-running the
  script is an idempotent upsert: edit a file, re-index, and the same symbols keep the
  same identity.
- ProximaDB answered a semantic query (vector search over symbol embeddings) and a
  structural query (metadata scan over call edges) from the same collection.

An agent wired to this collection can answer "what is near this concept" and "what breaks
if I change this" without re-reading the repository each session — that is the durable
code memory bet (VISION.md, bet 4).

## Going deeper

- **Victor's in-tree backends** — Victor ships an embedded ProximaDB embedding provider
  and graph store (`victor/storage/vector_stores/proximadb_provider.py`,
  `victor/storage/graph/proxima_store.py`), selected via
  `create_graph_store("proxima", ...)` or a per-repo `.victor/graph_backend` marker.
  They are parity-tested against the SQLite defaults but flag-gated, and they need the
  `proximadb_sdk` package from the ProximaDB repo (`clients/python`) — not yet on PyPI.
  Design + status: [ProximaDB as the CCG Backend](architecture/proximadb-codegraph-backend.md).
- **Chunking for RAG** — `victor_codegraph.chunk_repo` emits size-capped, AST-aligned
  chunks (never split mid-statement) if you want plain retrieval instead of the graph
  projection. See the [victor-codegraph README](../victor-codegraph/README.md).
- **Managed / multi-tenant** — the same `victor-codegraph` seam powers the AnvaiOps
  managed code-graph service on top of ProximaDB, so the local path above scales to a
  hosted one without changing the record shape.
