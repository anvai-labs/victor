# Callee-context diagnostic controls

These five authored Python files contain 12 compared functions and eight requested
pairs: three positive, four negative and one explicitly unknown. The manifest's
restricted oracle, file hashes, exact spans and labels were frozen before scoring
(`c4a4d6e30e8ee8e43fdb1cae06d74869964c3f7e7be4e72454652a312794b26c`).
They are correlated controls from one small Python project, not independent-project
accuracy, retrieval recall, a trained threshold or safe-refactoring evidence.

`controlled_targets` supplies hand-authored function/method identities. It is a
separate leg from `compiler.scip`, the unmodified output of installed
`scip-python 0.6.8` on these exact files. That tool extends Pyright to produce SCIP;
it performs static indexing, not native compilation or runtime verification.
See the [upstream description](https://github.com/sourcegraph/scip-python).
`compiler-binding.json` records the raw index hash, source hashes and declared
producer/configuration. These provenance declarations are not authentication.

The captured index contains no document text and declares unspecified per-document
position encoding. These controls use ASCII identifiers, for which the coordinate
interpretations coincide. The benchmark also verifies an explicit source-hash
sidecar; timestamp plausibility is not its freshness criterion. A changed source,
foreign producer or index hash cannot silently reuse the recorded evidence.

Replay the frozen actual-index fixture with the repository's declared protobuf
runtime available. No indexer, model, source execution or download is needed:

```bash
PYTHONPATH=.:victor-codegraph python scripts/benchmark_callee_context.py \
  --root tests/fixtures/graph_context \
  --manifest tests/fixtures/graph_context/manifest.json \
  --index tests/fixtures/graph_context/compiler.scip \
  --binding tests/fixtures/graph_context/compiler-binding.json \
  --output /absolute/path/to/a-fresh-report.json
```

Omit both `--index` and `--binding` to measure syntax and curator-controlled
identities only. Requested missing/stale/unresolved compiler evidence is retained
as abstention and gives exit 1; invalid input or unsafe publication gives exit 2.
Publication currently requires POSIX descriptor-relative primitives and a fresh
canonical destination without symlinked ancestors. Existing files are preserved
and unsupported publication fails closed.

The shared reader test owner selects the declared protobuf runtime and
`VICTOR_REQUIRE_SCIP_READER_TESTS=1` in CI, making the actual-fixture checks mandatory.
Local minimal installations may skip those checks; their passes alone do not
qualify the actual-index leg. Tool generation, measurement boundaries and limits
are described in the [ablation protocol](../../../docs/architecture/graph-callee-context-ablation.md).
