# Distribution Strategy: Coherent Cross-Platform Packaging

**Date:** 2026-11-05 (refined 2026-11-06)  
**Status:** Design Proposal — end-user experience layer; channels are decided by [ADR-016](adr/016-distribution-packaging-strategy.md)  
**Owner:** Victor Core Team

## Relationship to ADR-016

[ADR-016](adr/016-distribution-packaging-strategy.md) decides the *channels*:
Docker `full`/`slim` images as the primary packaged artifact, pip/uv for CLI
and development, native single-binary explicitly rejected (with rationale that
must be cited before re-proposing it). This document does not re-decide any of
that. It covers the layer ADR-016 leaves open: how an install *feels* to the
end user once a channel is chosen — correct optional-dependency defaults,
drift prevention between the Python package and its native wheel, diagnosable
installations (`victor doctor`), and coherent feature naming across channels.

## Executive Summary

Victor currently ships as Python wheels, standalone binaries, Docker images, and a VS Code extension. While functional, the distribution surface has grown organically and lacks a coherent strategy for feature-gated builds, hardware acceleration, and vertical-specific packaging.

This document proposes a unified distribution architecture that:
1. Ships **one coherent unit** with features enabled via feature flags and optional dependencies
2. Provides **platform-specific binaries** (macOS, Windows, Linux) with native acceleration
3. Supports **vertical-specific distributions** (coding, data science, DevOps) without forking the codebase
4. Enables **hardware-accelerated builds** (CUDA, Metal, ROCm) as opt-in variants
5. Maintains **backward compatibility** with existing `pip install victor-ai` workflows

## Current State

### What We Ship Today

| Artifact | Format | Platform | Notes |
|----------|--------|----------|-------|
| **Python wheel** | `victor_ai-*.whl` | Cross-platform | Pure Python, no native extensions |
| **Standalone binaries** | `victor-{os}-{arch}` | macOS ARM64, Linux x64/ARM64, Windows x64 | PyO3-compiled, bundled Python runtime |
| **Docker image** | `ghcr.io/anvai-labs/victor:latest` | Linux x64/ARM64 | Multi-arch, includes optional deps |
| **VS Code extension** | `victor-*.vsix` | Cross-platform | Bundles CLI, activates in workspace |
| **Rust crates** | `victor-protocol`, `victor-state`, `victor-tools`, `victor-edge` | Cross-platform | Published to crates.io |
| **Homebrew formula** | `brew install vijayksingh/tap/victor` | macOS/Linux | Wraps standalone binary |

### Pain Points

1. **Feature sprawl:** Users install `victor-ai` and discover they need `victor-ai[embeddings]`, `victor-ai[providers-ollama]`, `victor-ai[native]` etc. No clear guidance on which extras to install.

2. **Binary discovery:** Standalone binaries exist but are not prominently documented. Users default to `pip install` even when binaries would be faster.

3. **Hardware acceleration:** Native extensions (Rust/PyO3) are opt-in via `victor-ai[native]`, but the build matrix doesn't produce CUDA/Metal/ROCm variants. Users must compile from source for GPU acceleration.

4. **Vertical packaging:** Verticals (`victor-coding`, `victor-datascience`, etc.) are separate packages with independent versioning. Users must manually install the right vertical for their use case.

5. **Platform fragmentation:** macOS users get Homebrew, Linux users get apt/yum (planned), Windows users get standalone binaries. No unified installer experience.

## Proposed Architecture

### 1. Unified Installer with Feature Detection

**Goal:** One command installs Victor with sensible defaults; advanced users opt into features.

```bash
# Default install (pure Python, works everywhere)
pip install victor-ai

# Auto-detect platform and install optimal variant
pip install victor-ai --prefer-binary  # picks native wheels if available

# Explicit feature selection
pip install victor-ai[full]            # all features, all providers, native extensions
pip install victor-ai[coding]          # coding vertical + dependencies
pip install victor-ai[gpu]             # CUDA/Metal/ROCm native extensions
```

**Implementation:**
- Use `pyproject.toml` `[project.optional-dependencies]` for feature groups
- Add `victor-ai[full]`, `victor-ai[coding]`, `victor-ai[gpu]`, `victor-ai[minimal]` meta-extras
- Document feature matrix in README and installation guide
- Add `victor doctor` CLI command to diagnose missing optional dependencies

### 2. Platform-Specific Binary Matrix

**Goal:** Pre-built binaries for every supported platform with native acceleration.

| Platform | Architecture | Variants | Notes |
|----------|--------------|----------|-------|
| **macOS** | ARM64 (Apple Silicon) | `victor-macos-arm64`, `victor-macos-arm64-metal` | Metal acceleration in `-metal` variant |
| **macOS** | x64 (Intel) | `victor-macos-x64` | No GPU acceleration |
| **Linux** | x64 | `victor-linux-x64`, `victor-linux-x64-cuda`, `victor-linux-x64-rocm` | CUDA 12.x, ROCm 6.x |
| **Linux** | ARM64 | `victor-linux-arm64` | No GPU acceleration (Jetson support planned) |
| **Windows** | x64 | `victor-windows-x64`, `victor-windows-x64-cuda` | CUDA 12.x |

**Build Matrix:**
- Extend `.github/workflows/release.yml` to produce GPU variants
- Use GitHub Actions runners with GPU support (or self-hosted runners)
- Publish GPU variants as separate assets in GitHub Release
- Add `victor install --gpu` installer script that auto-detects and downloads the right binary

**Distribution Channels:**
- **Standalone binaries:** GitHub Releases, Homebrew (macOS/Linux), WinGet (Windows)
- **Package managers:** apt/yum (Linux), Homebrew (macOS), WinGet/Chocolatey (Windows)
- **Container registries:** Docker Hub, GHCR (with GPU tags: `victor:latest-cuda`, `victor:latest-rocm`)
- **GitHub direct (OSS, delivered 2026-11):** the public repository is a
  first-class install source —
  `pip install "victor-ai[native] @ git+https://github.com/anvai-labs/victor.git@v0.26.2"`.
  Tag refs are version-coherent with the published `victor-native` wheel;
  `@develop` tracks bleeding edge (Python-only changes are safe — Rust changes
  need a source native rebuild). Documented in the installation guide;
  `victor doctor` is the post-install verification step.

### 3. Vertical-Specific Distributions

**Goal:** Verticals are opt-in modules, not separate packages. Users install the vertical they need.

**Current State:**
- `victor-coding`, `victor-datascience`, `victor-devops` are separate PyPI packages
- Independent versioning creates compatibility matrix complexity
- Users must manually discover and install verticals

**Proposed State:**
- Verticals become optional features within `victor-ai`: `victor-ai[coding]`, `victor-ai[datascience]`
- Vertical-specific CLI commands are registered via entry points
- Vertical packages remain for backward compatibility but are marked as deprecated
- `victor verticals list` shows installed verticals; `victor verticals install coding` installs the feature

**Migration Path:**
1. **Phase 1 (v0.27.0):** Add `victor-ai[coding]` etc. as meta-extras that install the corresponding vertical package
2. **Phase 2 (v0.28.0):** Move vertical code into `victor-ai` namespace (`victor.verticals.coding`)
3. **Phase 3 (v1.0.0):** Deprecate standalone vertical packages; `victor-ai[vertical]` is the only supported path

### 4. Hardware Acceleration Strategy

**Goal:** GPU acceleration is opt-in, platform-specific, and well-documented.

**Native Extension Layers:**

| Layer | Technology | Platform | Status |
|-------|-----------|----------|--------|
| **CPU acceleration** | Rust/PyO3 | Cross-platform | ✅ Implemented (opt-in via `victor-ai[native]`) |
| **Metal acceleration** | Rust + Metal | macOS ARM64 | 🟡 Planned |
| **CUDA acceleration** | Rust + CUDA | Linux/Windows x64 | 🟡 Planned |
| **ROCm acceleration** | Rust + ROCm | Linux x64 (AMD) | 🟡 Planned |

**Build Strategy:**
- **CPU variant:** Default native extension, no GPU dependencies
- **GPU variants:** Separate builds with CUDA/ROCm/Metal toolchains
- **Fallback:** Pure Python implementation if native extensions fail to load
- **Detection:** `victor doctor` reports available acceleration; `victor config` allows explicit selection

**Feature Flags:**
```python
# victor/__init__.py
from victor._acceleration import get_acceleration_backend

backend = get_acceleration_backend()  # auto-detects CPU/Metal/CUDA/ROCm
if backend == "cuda":
    from victor._native_cuda import fast_graph_traversal
elif backend == "metal":
    from victor._native_metal import fast_graph_traversal
else:
    from victor._native_cpu import fast_graph_traversal
```

### 5. Installer Experience

**Goal:** One-liner installation for new users; guided setup for advanced users.

**Unix (macOS/Linux):**
```bash
curl -fsSL https://get.victor.ai | sh
```

**Windows:**
```powershell
irm https://get.victor.ai/windows | iex
```

**Installer Behavior:**
1. Detect platform (OS, architecture, GPU)
2. Download appropriate binary or wheel
3. Install to `~/.local/bin` (Unix) or `%LOCALAPPDATA%\Victor\bin` (Windows)
4. Add to PATH
5. Run `victor doctor` to verify installation
6. Prompt for optional features: "Install GPU acceleration? [y/N]"

**Advanced Installation:**
```bash
# Install specific version
curl -fsSL https://get.victor.ai | sh -s -- --version 0.26.2

# Install with GPU support
curl -fsSL https://get.victor.ai | sh -s -- --gpu

# Install to custom location
curl -fsSL https://get.victor.ai | sh -s -- --prefix /opt/victor
```

### 6. Container Distribution

**Goal:** Production-ready containers with GPU support and minimal image size.

**Image Variants:**

| Tag | Base | Size | Use Case |
|-----|------|------|----------|
| `victor:latest` | Python 3.12 slim | ~500MB | Development, CI |
| `victor:latest-cuda` | NVIDIA CUDA 12.x | ~2GB | GPU workloads |
| `victor:latest-rocm` | ROCm 6.x | ~2.5GB | AMD GPU workloads |
| `victor:latest-minimal` | Alpine + Python 3.12 | ~200MB | Edge deployment |

**Dockerfile Strategy:**
- Multi-stage builds to minimize image size
- Separate `Dockerfile.cuda`, `Dockerfile.rocm` for GPU variants
- Include `victor doctor` in entrypoint to verify GPU access
- Document GPU passthrough for Docker Desktop, Podman, Kubernetes

**Example Usage:**
```bash
# CPU-only
docker run -it ghcr.io/anvai-labs/victor:latest victor analyze .

# GPU-accelerated (NVIDIA)
docker run --gpus all -it ghcr.io/anvai-labs/victor:latest-cuda victor analyze .

# Mount project directory
docker run -v $(pwd):/workspace -it ghcr.io/anvai-labs/victor:latest victor ui
```

## Delivered (2026-11): native wheel drift prevention

Motivating incident (2026-11-06): a long-lived venv silently ran
`victor_native 0.15.0` under `victor-ai 0.26.x`; the stale binary carried a
dedup livelock (fixed upstream months earlier) that pinned a chat session at
100% CPU. Nothing detected the drift because the `victor-ai[native]` extra is
only a floor for *fresh* installs — existing installs never upgrade the wheel.

- **Import-time drift classification + loud warning** in
  `victor/processing/native/_base.py` (`get_native_version_status()`; stale
  major.minor warns with the exact remediation command). PR #1700.
- **`victor doctor` native checks**: availability, version drift (stale =
  ERROR / exit 1, patch drift = WARNING, source builds = INFO), and the NumPy
  dispatch note. Remediation is printed, never auto-installed.
- **`make install-native`** pins `victor-native==$(VERSION)` and runs from
  `install-dev` (best-effort; `VICTOR_NATIVE_SKIP_SYNC=1` for Rust source
  builds; `victor doctor` remains the hard backstop).

## Implementation Roadmap

### Phase 1: Foundation (v0.27.0, Q1 2027)

**Goals:**
- Add feature-flagged extras to `pyproject.toml`
- Document feature matrix in README
- Add `victor doctor` CLI command
- Produce GPU-variant binaries in CI (CUDA, Metal)

**Deliverables:**
- [ ] `pyproject.toml` extras: `[full]`, `[coding]`, `[gpu]`, `[minimal]`
- [x] `victor doctor` diagnoses missing deps and reports the native/NumPy
      acceleration state, failing on native-wheel version drift (delivered
      2026-11; see "Delivered" above)
- [x] Native wheel drift prevention: import-time guard + `make install-native`
      sync from `install-dev` (delivered 2026-11)
- [ ] CI matrix: macOS ARM64 Metal, Linux x64 CUDA 12.x
- [ ] Installation guide rewrite (feature-first, not package-first)

### Phase 2: Installer & Package Managers (v0.28.0, Q2 2027)

**Goals:**
- Launch `get.victor.ai` installer
- Publish to Homebrew, WinGet, apt/yum
- Deprecate standalone vertical packages

**Deliverables:**
- [ ] `curl | sh` installer with GPU auto-detection
- [ ] Homebrew formula update (support `--with-cuda` option)
- [ ] WinGet manifest for Windows
- [ ] apt/yum repositories (Ubuntu 22.04+, RHEL 9+)
- [ ] Vertical migration guide (standalone → feature)

### Phase 3: GPU Acceleration (v0.29.0, Q3 2027)

**Goals:**
- Ship Metal, CUDA, ROCm native extensions
- Benchmark GPU vs CPU performance
- Document GPU setup for each platform

**Deliverables:**
- [ ] Rust/PyO3 Metal extension (macOS ARM64)
- [ ] Rust/PyO3 CUDA extension (Linux/Windows x64)
- [ ] Rust/PyO3 ROCm extension (Linux x64, AMD)
- [ ] Performance benchmarks (graph traversal, embedding generation)
- [ ] GPU troubleshooting guide

### Phase 4: Vertical Integration (v1.0.0, Q4 2027)

**Goals:**
- Move vertical code into `victor-ai` namespace
- Deprecate standalone vertical packages
- Launch vertical-specific installers (`victor install --coding`)

**Deliverables:**
- [ ] `victor.verticals.coding`, `victor.verticals.datascience` modules
- [ ] `victor verticals install <name>` CLI command
- [ ] Deprecation notices for `victor-coding`, `victor-datascience` packages
- [ ] Migration guide (standalone → integrated)

## Success Metrics

| Metric | Current | Target (v1.0.0) |
|--------|---------|-----------------|
| **Installation time** | ~30s (pip install) | <10s (binary installer) |
| **GPU acceleration adoption** | <5% (manual build) | >30% (auto-detected) |
| **Vertical discovery** | Manual (README, docs) | Automated (`victor verticals list`) |
| **Platform coverage** | 3/5 (macOS ARM64, Linux x64/ARM64) | 5/5 (+ macOS x64, Windows x64) |
| **Container GPU support** | Not documented | Documented for Docker, Podman, K8s |

## Risks and Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| **GPU build matrix explosion** | High (CI time, maintenance) | High | Start with CUDA (most common), add Metal/ROCm based on demand |
| **Vertical migration breaks users** | Medium (support burden) | Medium | 6-month deprecation window, migration guide, automated script |
| **Installer complexity** | Medium (user confusion) | Low | Keep installer simple (auto-detect), document manual installation |
| **Feature flag sprawl** | Medium (testing matrix) | Medium | Limit to 4-5 feature groups, test combinations in CI |

## Open Questions

1. **Should we ship a GUI installer?** (e.g., `.dmg` for macOS, `.msi` for Windows)
   - **Pros:** Familiar to non-developer users, can include uninstaller
   - **Cons:** Maintenance burden, platform-specific tooling
   - **Decision:** Defer to v1.1.0; focus on CLI installer first

2. **Should we offer a "Victor Pro" subscription with pre-built GPU binaries?**
   - **Pros:** Revenue stream, reduces CI costs
   - **Cons:** Community backlash, fragments user base
   - **Decision:** No; GPU binaries remain free, but we offer cloud-hosted Victor as a separate product

3. **Should we support ARM64 Windows?**
   - **Pros:** Future-proof (Snapdragon X laptops)
   - **Cons:** Low adoption today, testing infrastructure gaps
   - **Decision:** Monitor adoption; add in v1.1.0 if demand materializes

## References

- [Python Packaging User Guide](https://packaging.python.org/en/latest/)
- [PyO3 User Guide](https://pyo3.rs/)
- [CUDA Toolkit Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)
- [Metal Programming Guide](https://developer.apple.com/metal/)
- [ROCm Documentation](https://rocm.docs.amd.com/)

## Appendix: Feature Matrix

| Feature | Default | `[full]` | `[coding]` | `[gpu]` | `[minimal]` |
|---------|---------|----------|------------|---------|-------------|
| Core framework | ✅ | ✅ | ✅ | ✅ | ✅ |
| All providers | ❌ | ✅ | ❌ | ❌ | ❌ |
| Embeddings | ❌ | ✅ | ❌ | ✅ | ❌ |
| Native extensions (CPU) | ❌ | ✅ | ❌ | ✅ | ❌ |
| Native extensions (GPU) | ❌ | ❌ | ❌ | ✅ | ❌ |
| Coding vertical | ❌ | ✅ | ✅ | ❌ | ❌ |
| Data science vertical | ❌ | ✅ | ❌ | ❌ | ❌ |
| DevOps vertical | ❌ | ✅ | ❌ | ❌ | ❌ |
| CLI tools | ✅ | ✅ | ✅ | ✅ | ✅ |
| TUI interface | ✅ | ✅ | ✅ | ✅ | ❌ |
| Web UI (Chainlit) | ❌ | ✅ | ❌ | ❌ | ❌ |

---

**Next Steps:**
1. Review and approve this design document
2. Create GitHub issues for Phase 1 deliverables
3. Assign owners for each workstream
4. Begin implementation in v0.27.0 development cycle
