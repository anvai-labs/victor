# Automatic compiler-index provider dispatch

With compiler indexing enabled, Victor selects registered providers by file
extension and discovers their installed tools automatically. Users do not supply
SCIP files or a separate command for every language. One producer runs per matched
provider, with unchanged artifacts reused from the per-provider source/tool cache.

Compilation executes project build logic. Run an explicitly authorized compiler
index in an isolated disposable copy with a reviewed environment and fixed timeout:

```bash
victor index graph index --path /isolated/source --precise --no-embeddings --no-ccg
```

`--precise` or `search.graph.enable_precise_index` enables the pass; the default is
off. `precise_index_tool=auto` uses each provider's discovery mechanism (usually
PATH; Ruby requires bundler and scip-ruby). Discovery does not download tools or
install project dependencies. Toolchain provisioning is a separate setup action.

| Registered producer | Eligible extensions |
| --- | --- |
| scip-python | .py |
| scip-typescript | .ts, .tsx, .js, .jsx, .mjs |
| rust-analyzer | .rs |
| scip-clang | .c, .h, .cc, .cpp, .cxx, .hpp, .hh, .hxx |
| scip-java | .java |
| scip-ruby | .rb, .rbi |
| scip-go | .go |

Dispatch uses files represented in the persisted syntax graph. It is not a census
of every tracked file. Source exclusions can remove an entire language before
provider selection: on Iceberg, default `**/site/**` exclusion hides its JavaScript
assets and one Python hook. Explicit source inventory inclusion and source-root
validation need separate integration. Scala/Kotlin/C# have no registered precise
adapter here; current scip-java does not restore Scala coverage. Missing language
variants, project roots or toolchains must not be called compiler-qualified.

Each selected producer reports a distinct outcome: successful edge count,
`toolchain-absent`, typed missing-build-marker skip, `dump-failed`, reader failure,
or `error:<exception type>`. An unexpected detection/dump/read exception affects
only that producer. Previously, a shared outer exception handler discarded
completed outcomes and stopped all remaining languages. The new inner boundary
preserves completed compiler edges and continues other providers. Cancellation
still propagates rather than becoming a successful or retried language job.

`tests/unit/core/graph_rag/test_precise_dispatch.py` exercises all seven registered
producers through the real harvest loop with synthetic SCIP outputs. TS/JS share
one invocation; absence, marker skip, returned failure and raised detection/dump/
reader errors preserve successful Python edges and later Go execution. Cancellation
stops continuation. These are orchestration regression tests, not real compilation
qualification for all seven toolchains. The separate
[Java/Iceberg audit](java-indexing.md) contains actual compiler measurements.
