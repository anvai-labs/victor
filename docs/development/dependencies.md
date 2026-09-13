# Dependency and deployment maintenance

`pyproject.toml` defines the supported package dependencies and optional features.
The root `requirements.txt` is a resolved **core runtime** snapshot for Python
3.12. It replaces the old workstation freeze, which mixed development packages,
optional backends, platform-specific libraries and private editable checkouts.
It does not install Victor itself or enable every optional feature.

## Choose a deployment

| Deployment | Installation | Additional dependency surface |
| --- | --- | --- |
| CLI, TUI, framework, MCP | `pip install victor-ai` | Core providers, terminal UI and generic tools |
| HTTP / GraphQL API | `pip install 'victor-ai[api]'` | FastAPI, Uvicorn, Strawberry |
| Local text embeddings | `pip install 'victor-ai[embeddings]'` | Sentence Transformers, Torch, LanceDB, Arrow |
| Google provider | `pip install 'victor-ai[google]'` | Google GenAI SDK |
| Browser tools | `pip install 'victor-ai[web]'` | Playwright, extraction libraries; browser installed separately |
| Docker tools | `pip install 'victor-ai[docker]'` | Docker SDK; Docker daemon is separate |
| Coding, DevOps, research definitions | Install the corresponding `victor-*` vertical | Contracts SDK; coding also needs codegraph. Base definitions do not require NumPy or the Victor runtime. |
| RAG / data analysis vertical | Install `victor-rag` / `victor-dataanalysis` | Embeddings stack / Pandas, respectively |
| Native acceleration | Build the root Rust wheel or the separate coding native wheel | Different PyO3 extensions; Python fallbacks remain available |
| Contributor tooling | `pip install -e '.[dev]'` | Tests, typing and development libraries |

The `all` extra includes development, documentation and build tooling. Use the
specific feature extras for deployment. The Apple Silicon extra is restricted to
Darwin ARM64. NumPy remains a core dependency because runtime inference and
similarity fallbacks use it. BeautifulSoup's `lxml` parser is an indirect runtime
consumer, so a zero direct-import count does not justify removing `lxml`.
Git operations use the system Git executable; GitPython is unnecessary. Coverage
uploads use the pinned GitHub Action, so the unused Python `codecov` CLI is
removed from the CI extra. RAG
package discovery keeps storage libraries lazy; requesting a document store,
chunker or RAG tool loads its backend.

## Regenerate and audit resolved dependencies

Use a fresh Python 3.12 environment, with `pip-tools==7.6.1`. Run from the
repository root; add `--upgrade` when refreshing existing pins:

```bash
python -m pip install --upgrade 'pip>=26.2.1'
python -m pip install 'pip-tools==7.6.1'
pip-compile --resolver=backtracking --strip-extras --allow-unsafe \
  --output-file=requirements.txt pyproject.toml
pip-compile --resolver=backtracking --strip-extras --allow-unsafe --extra=api \
  --output-file=requirements/api/requirements.txt pyproject.toml
pip-compile --resolver=backtracking --strip-extras --allow-unsafe --extra=embeddings \
  --extra-index-url=https://download.pytorch.org/whl/cpu \
  --output-file=requirements/embeddings-cpu/requirements.txt pyproject.toml
```

These files record resolved versions, not cross-platform lock guarantees. The
CPU embeddings snapshot targets Linux Python 3.12. Resolve separately on Python
3.11, macOS, or a GPU environment, then run the same feature tests and audit that
installed environment. Do not substitute an independent torchvision or torchaudio
version into a Torch installation; neither is required by Victor's text embedding
path. Install GPU Torch from the appropriate upstream index before resolving the
embeddings extra, and audit the result.

For each snapshot, install it into a clean environment, install the locally built
Victor and contracts wheels, run `pip check`, and smoke-test the corresponding
entrypoints. Run `pip-audit` over the complete installed environment, including
build tools. The filesystem gate requires reports for all committed runtime
snapshots; it does not replace installed-environment audits of optional features,
verticals, documentation and development dependencies.

Do not copy `pip freeze` from a working developer environment into a release
manifest. Preserve the scanner reports and dependency-resolution logs with the
release evidence. Follow the [CI batching policy](PR_WORKFLOW.md) before pushing.

## Container targets

One root `Dockerfile` owns the shared Python base, wheel build, runtime user and
Git/SSH packages. The obsolete split-package Dockerfiles have been removed.
Build from the repository root with BuildKit:

```bash
docker build --target core -t victor-core:local .
docker build --target mcp -t victor-mcp:local .
docker build --target native -t victor-native:local .
docker build --target full -t victor-full:local .
```

| Target | Contents and default command |
| --- | --- |
| `core` | Core + API dependencies; `victor serve --host 0.0.0.0 --port 8765` |
| `mcp` | Core dependencies; `victor mcp` over standard input/output |
| `native` | Core + API + compiled Rust wheel; `victor --help` |
| `full` (default) | Core + CPU embeddings + predownloaded BGE model; `bash` |

Runtime packages are installed wheels in `/opt/victor`, rather than editable
source trees copied from the builder. Rust/compiler tools and pip stay in build stages. Build a derived image to add
packages; runtime targets contain no package installer. The embeddings target
retains setuptools because Torch declares it as a dependency.
The full target copies the model cache directly into the runtime user's home.
It supports offline embeddings; cloud provider calls and network tools still
require their configured endpoints. Derived tool caches rebuild on demand.

Validate each final target: CLI/version, actual installed module paths, absence
of build-only dependencies, non-root execution, Git HTTPS/SSH, and the applicable
API, MCP or offline embedding behavior. Scan the **final image**, retaining its
immutable digest, package inventory and SBOM. A successful build is not security
approval: unaccepted critical/high findings continue to block publication.

## Persistent cache compatibility

The tiered cache now writes a private, versioned SQLite file containing tagged
JSON data. Supported persistent values are scalars, bytes, lists, tuples and
dictionaries of those values. Other objects can remain in memory; replacing a
persisted value with an unsupported value invalidates the old disk entry.
Legacy diskcache files are never deserialized or migrated. Cached data is derived
and can be rebuilt. The configured disk budget bounds encoded payload and key
bytes; SQLite file allocation and indexes add overhead.

This change removes the `diskcache` dependency and its security exception. Static embeddings, prompt corpus
embeddings, semantic usage statistics and usage analytics also use versioned
data-only files. Legacy pickle files are not loaded; known record types and
numeric arrays are validated before a restored snapshot replaces live state.
