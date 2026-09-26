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
repository root. Build the in-tree contracts wheel first when its required
version has not reached PyPI yet, then expose that temporary directory to the
resolver without emitting it into the portable lock files:

```bash
python -m pip install --upgrade 'pip>=26.2.1'
python -m pip install 'pip-tools==7.6.1'
python -m build --wheel --outdir /tmp/victor-contract-wheels victor-contracts
export PIP_FIND_LINKS=/tmp/victor-contract-wheels
pip-compile --upgrade --resolver=backtracking --strip-extras --allow-unsafe \
  --no-emit-find-links --constraint=constraints.txt \
  --output-file=requirements.txt pyproject.toml
pip-compile --upgrade --resolver=backtracking --strip-extras --allow-unsafe \
  --no-emit-find-links --constraint=constraints.txt --extra=api \
  --output-file=requirements/api/requirements.txt pyproject.toml
pip-compile --upgrade --resolver=backtracking --strip-extras --allow-unsafe \
  --no-emit-find-links --constraint=constraints.txt --extra=embeddings \
  --extra-index-url=https://download.pytorch.org/whl/cpu \
  --output-file=requirements/embeddings-cpu/requirements.txt pyproject.toml
unset PIP_FIND_LINKS
```

These files record resolved versions, not cross-platform lock guarantees. The
CPU embeddings snapshot targets Linux Python 3.12. Resolve separately on Python
3.13, macOS, or a GPU environment, then run the same feature tests and audit that
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
Git/SSH packages. The runtime uses digest-pinned Ubuntu 24.04 with its patched
Python 3.12 packages. The Rust builder uses a pinned Bookworm toolchain compatible
with Ubuntu's glibc. SDK wheel and dependency layers are independent of ordinary
application source edits, reducing repeated package downloads and builds.
The obsolete split-package Dockerfiles have been removed.
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

The `cvss` runtime dependency calculates standard CVSS v2/v3/v4 advisory base
scores and has no transitive dependencies. Keep it in all three resolved snapshots.

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

## Advisory cache compatibility

OSV queries follow continuation pages and reject malformed or conflicting records.
The runtime computes actual CVSS base scores; missing or unusable ratings remain
unknown and fail policy. Manifest parsing or lookup errors also fail policy.
Package-query snapshots include the exact version and distinguish missing coverage
from a completed clean query. Legacy score and versionless query caches refresh
online. Offline mode uses the same `cve.db`; missing, invalid or expired coverage
reports an incomplete scan, so an empty cache cannot establish a clean result.

## September 2026 dependency PR reconciliation

PRs #1145, #1146 and #1171 are consolidated into one tested maintenance candidate:
smallvec 1.16.1, Ruff 0.16.7, Vite 8.3.1 (manifest floor 8.3.0),
mkdocs-git-revision-date-localized-plugin 1.6.0 and pymdown-extensions 12.0.1.
The pymdown major upgrade requires a real documentation build and syntax-highlighting
check. The webview lock retains Linux glibc/musl selectors; regenerate from a clean
directory with compatible Node/npm versions rather than an older installed graph.

The remaining proposed upgrades are deliberately deferred, not represented as merged:

| Proposed migration | Reason to retain current constraints | Completion gate |
| --- | --- | --- |
| pydantic-core 2.49.0 | Pydantic 2.13.5 requires exactly 2.46.5; #1171 failed resolution | Upgrade Pydantic and core together; regenerate all three runtime snapshots and run validation/serialization suites |
| fsspec 2026.7 | Conflicts with the documented datasets bound and the CPU embeddings snapshot | Resolve datasets/RAG and embeddings together; regenerate all affected snapshots |
| peewee 4, OTel 1.44/0.65b0, wrapt 2 | Semgrep's current coupled toolchain uses peewee 3, OTel 1.37/0.58b0 and wrapt 1 | Upgrade or isolate the scanner toolchain; validate scanning plus all OTel packages as one family |
| NumPy 2.5 | Requires an explicit supported-range change across core/ML/embeddings and resolved snapshots | Differential numerical/native parity and supported-platform embedding validation |
| tree-sitter-language-pack 1.20 floor | Raises the package's minimum grammar bundle independently of this runtime repair | Validate supported grammars and codegraph packaging before raising the floor |
| TypeScript 7 and Node 26 types | #1146 failed peer resolution: typescript-eslint 8.70 requires TypeScript <6.1; the extension targets Node 24 | Upgrade the compiler/linter as one compatible set; keep type definitions aligned with the supported runtime |

The original bot CI failures are evidence of incompatibility, not reasons to disable
security alerts. Future routine updates still follow the existing grouped schedule;
reopening any deferred migration requires resolving its listed completion gate.
