# Graph RAG API Reference

Use the [graph quickstart](graph-quickstart.md) for executable indexing and retrieval examples.
The source modules below define the current signatures and defaults; this page avoids a second
hand-maintained catalog.

| Surface | Authoritative source |
| --- | --- |
| `GraphIndexConfig`, `RetrievalConfig` | [configuration](https://github.com/anvai-labs/victor/blob/develop/victor/core/graph_rag/config.py) |
| `GraphIndexingPipeline.index_repository` | [indexing](https://github.com/anvai-labs/victor/blob/develop/victor/core/graph_rag/indexing.py) |
| `MultiHopRetriever` | [retrieval](https://github.com/anvai-labs/victor/blob/develop/victor/core/graph_rag/retrieval.py) |
| `create_graph_store` | [store factory](https://github.com/anvai-labs/victor/blob/develop/victor/storage/graph/factory.py) |
| `GraphStoreProtocol`, graph node/edge/result types | [storage contracts](https://github.com/anvai-labs/victor/blob/develop/victor/storage/graph/protocol.py) |
| `EdgeType` and edge categories | [edge types](https://github.com/anvai-labs/victor/blob/develop/victor/storage/graph/edge_types.py) |

The former copied catalog included `GraphIndexingPipeline.index_file()` and
`build_embeddings()` as public methods and outdated configuration defaults. Those snippets
are not current APIs; the [original catalog](https://github.com/anvai-labs/victor/blob/b6b25a634/docs/architecture/graph-api-reference.md)
is preserved as historical context.
