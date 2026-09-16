# Victor Docker deployment

The root Dockerfile builds four targets from installed Victor wheels. See the
[canonical dependency and deployment guide](../docs/development/dependencies.md#container-targets)
for their package contents, lock regeneration and security validation.

| Target | Purpose | Default command |
| --- | --- | --- |
| `core` | HTTP and GraphQL API | `victor serve` on port 8765 |
| `mcp` | MCP over standard input/output | `victor mcp` |
| `native` | API dependencies and Rust acceleration | `victor --help` |
| `full` | CPU embeddings and cached BGE model | `bash` |

Build from the repository root:

```bash
docker build --target full -t victor-full:local .
docker run --rm --network none victor-full:local bash /app/docker/scripts/test-airgapped.sh
```

The full image caches `BAAI/bge-small-en-v1.5`. The command above requires that
cache and verifies embedding generation with networking disabled. LLM provider
models are separate: configure a reachable provider or provision a local model
server and its weights before moving to an isolated environment.

Tool embeddings are derived on demand from the installed tool registry. To
prepare them in a persistent volume for a particular workspace:

```bash
docker run --rm --network none -v victor-home:/home/victor/.victor \
  victor-full:local bash /app/docker/scripts/init-embeddings.sh
```

Profiles and credentials are user configuration; these helpers do not create or
replace them. Runtime images run as UID 1000 and contain no pip installer. Add
packages in a derived build and audit the resulting image. See the
[MCP guide](mcp-server/README.md) for stdio client configuration.
