# ADR-016: Distribution & Packaging Strategy

## Metadata

- **Status**: Proposed
- **Date**: 2026-07-02
- **Decision Makers**: Vijaykumar Singh
- **Related ADRs**: 007 (Vertical Distribution & Contracts Boundary), 014 (shared `victor-codegraph`)

## Context

Victor has a **heavy, mostly-optional dependency surface**: native Rust extensions (`rust/`, the `_NATIVE_AVAILABLE` PyO3 pattern), ML/embedding stacks (`sentence-transformers`, `torch`, `lancedb`), per-language tree-sitter grammars (`[lang-core]`/`[lang-all]` extras), the `swebench` eval harness, and 25 LLM provider adapters. None of these are required to *import* victor, but a host without them gets a degraded agent (no embeddings, no graph for non-Python, no containerized eval).

The recurring pain this causes:
- **Hosts without the right deps can't run victor fully** — installing the full matrix (torch + grammars + native ext + swebench) is heavy and version-fragile.
- **SWE-bench eval needs Docker anyway** (see the containerized-eval work — each task runs in its correct runtime image). So a Docker runtime is already a deployment assumption for serious eval.
- **Venv pollution**: in-process eval `pip install -e .`'s task repos into the host venv (the known SWE-bench pollution issue).

The question: *should victor ship as packaged software* — a self-contained artifact the host can run without provisioning all dependencies — analogous to how `bun` ships JS, `cargo` ships Rust, and `g++` produces native C/C++ binaries? What is Python's equivalent, and which fits victor?

## Decision

**Ship victor as a Docker image as the primary packaged artifact**, in two variants:
- **`victor:full`** — all extras baked in (ML/embeddings, all tree-sitter grammars, native Rust ext, `swebench`). The "host needs only Docker" artifact.
- **`victor:slim`** — core only (no torch/lancedb/grammars); smaller, for provider+tool+workflow use without code intelligence.

Retain the **pip-installable package** (`victor-ai` + `victor-contracts`) as the path for **dev / light / extensible** use (editable installs, adding custom verticals/tools, hosts that already have Python + select deps).

**Reject native single-binary packaging (PyInstaller / Nuitka / PyOxidizer) as the primary distribution** for victor. (Rationale below; documented so the trade-off isn't re-litigated.)

## Rationale

### Why Docker image
1. **Victor already containerizes.** The eval backend runs each task in a per-instance Docker image (ADR for containerized eval, in flight); `SandboxedExecutor` (`victor/workflows/sandbox_executor.py`) is real Docker. A `victor` image is the natural completion of "one Docker story" — the agent runtime AND the per-task eval envs.
2. **Bundles everything, host provisions nothing but Docker.** Python + all optional deps (native ext, ML, grammars, swebench) are baked into `victor:full`. This is the direct answer to "host doesn't have all dependencies."
3. **Solves both distribution *and* eval-env correctness together.** A victor-in-container process orchestrates per-task eval containers; no host venv, no pollution.
4. **Reproducible & versioned.** Pinned image ↔ pinned dep matrix; CI builds and publishes to GHCR on release tags (alongside the existing `make docker` / release workflow).
5. **Stays inside Python's strengths.** victor remains pip-installable and extensible; the image just freezes one known-good matrix.

### Why NOT native single-binary (PyInstaller/Nuitka/PyOxidizer) as primary
Victor's architecture actively fights frozen-binary packaging:
1. **Dynamic plugin / entry-point discovery.** `VictorPlugin.register(context)` + entry-point scanning (`victor.plugins`, `victor.safety_rules`, …) + `VerticalExtensionLoader` resolve modules at runtime. Freezers struggle to detect/import these; missed modules = broken plugins at runtime (silent, only in the shipped binary).
2. **Optional native Rust extensions + lazy imports.** The `_NATIVE_AVAILABLE` pattern (PyO3 cdylib + Python fallback) and lazy contracts-bridge imports assume a real install tree. Freezing a cdylib + its loader is brittle.
3. **Heavy ML deps bloat the binary enormously.** `torch` alone is ~2 GB; bundling it into a single executable yields a multi-GB artifact with slow startup — worse than a Docker layer that's pulled once and shared.
4. **The pluggable provider/tool/vertical model assumes the pip/Python ecosystem** for users to *add* capabilities. A frozen binary is closed; you'd lose the "install a vertical and it just works" property that is core to victor's design (ADR-007 vertical distribution).

Native single-binary remains a **possible future path for a stripped-down, closed CLI build** (e.g. a `victor-cli` with a fixed provider set, no ML), but it is explicitly **not** the primary distribution.

### Alternatives considered
- **PEX / shiv (`.pyz` zipapp):** self-extracting, but still requires a compatible Python interpreter on the host and doesn't reliably bundle C-extensions/native wheels. Good for Python-present hosts; doesn't meet "host has nothing."
- **`pipx` / `uv`:** excellent install DX, but they still *build/install* deps on the host (they don't bundle a pre-resolved matrix). Right answer for devs, not for "packaged software on a bare host."
- **PyOxidizer:** embeds a Python interpreter in a Rust binary; appealing given victor's Rust presence, but suffers the same plugin/ML/native-ext problems as PyInstaller with worse ecosystem support. Rejected for the same reasons.

## Consequences

**Positive**
- Hosts run full-capability victor with only Docker installed — no dep provisioning, no version drift, no venv pollution.
- One consistent container story: `victor:full` (agent) + per-task eval images (correct runtime per task).
- Reproducible, CI-published, version-tagged images; aligns with the existing release workflow (`make docker`, GHCR).
- Dev experience unchanged (editable pip install remains first-class).

**Negative / neutral**
- **Image size**: `victor:full` with torch/lancedb is large (GBs). Mitigated by the `slim`/`full` split and Docker layer caching (shared base, pulled once).
- **Iteration speed**: image rebuilds are slower than editable installs; devs keep using pip. The image targets deployment, CI, and under-provisioned hosts.
- **Docker dependency**: containerized eval and the packaged image both require Docker Desktop/runtime. Documented as a prerequisite; the `local` eval backend and pip install remain for Docker-less dev.
- **Native single-binary off the table (deliberately)**: anyone re-proposing PyInstaller/Nuitka should reference this ADR's rationale first.

## Implementation notes (follow-on, not this ADR)
- Multi-stage `Dockerfile`: builder stage installs `.[full]` (+ `[lang-all]`, `swebench`, native `maturin develop --release`); runtime stage copies the install + compiles native ext.
- CI: build + push `victor:full` and `victor:slim` to GHCR on release tags (extend `.github/workflows/`).
- The containerized eval backend (Phase 1) is the first concrete step: once eval runs in containers, shipping victor itself as a container is a small, consistent increment.
- Versioning: image tags mirror `victor-ai` VERSION; `latest` tracks the latest release.

## Relationship to other work
- **Containerized polyglot eval (Phase 1, in flight):** establishes the Docker-based execution model this ADR builds on for *victor itself*.
- **ADR-007 (vertical distribution):** the pip/Python extensibility model stays primary for verticals; the Docker image freezes a known-good matrix on top of it.
- **ADR-014 / ADR-015 (codegraph):** the `victor:full` image bundles `[lang-all]` grammars so polyglot indexing works out-of-the-box.
