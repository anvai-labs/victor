# Phase 6: Packaging & Distribution -- Comprehensive Analysis & Plan
Victor's current packaging is a monolithic victor-ai wheel shipping ~1,900 Python files (~818K LOC) with ~20 core dependencies and ~30 optional extras. This document analyzes the full dependency surface, identifies redundancy and sprawl, evaluates Rust/PyO3 native wheel candidates, studies distribution strategies from comparable projects (Claude Code, Codex CLI, OpenCode), and recommends a phased packaging modernization plan.
---
| Package | Type | Dependencies | Size | Purpose |
|---------|------|-------------|------|---------|
| victor-ai | Main app | 20 core + 30 optional | ~818K LOC / 1,900 files | Full agentic AI framework |
| victor-contracts | SDK | 1 (typing-extensions) | ~15K LOC | Protocol definitions for plugins |
| victor_native | Rust PyO3 | Rust toolchain | ~5 crates | Performance-critical extensions |
| victor-codegraph | Utility | 0 core, 2 optional | ~10K LOC | Code CPG chunker (shared with ProximaDB) |
| victor-coding | Vertical | victor-contracts | External | Coding assistant vertical |
| victor-devops | Vertical | victor-contracts | External | DevOps vertical |
| victor-rag | Vertical | victor-contracts | External | RAG vertical |
| victor-dataanalysis | Vertical | victor-contracts | External | Data analysis vertical |
| victor-research | Vertical | victor-contracts | External | Research vertical |
Core dependencies (always installed): 20 packages
- pydantic, pydantic-settings, python-dotenv, typing-extensions, orjson
- packaging, numpy, httpx, aiofiles, aiohttp
- typer, rich, prompt-toolkit, textual
- anthropic, openai
- gitpython, jsonschema, tiktoken, pyyaml
Optional extras: 12 groups: docker, embeddings, proximadb, langchain, examples, chat-ui, ml, dev, ci, docs, build, all
Third-party import footprint (files referencing each lib, top 15):
- 94 pydantic, 85 rich, 50 typer, 49 yaml
- 33 httpx, 23 numpy, 21 fastapi, 19 fnmatch
- 16 argparse, 9 aiohttp, 6 keyring, 6 tomli, 6 tomllib, 6 contextvars
Long-tail dependencies (1-5 files each): ~100+ packages including:
- Cloud SDKs: boto3, botocore, google, kubernetes
- Databases: psycopg2, mysql, pyodbc, asyncpg, redis, duckdb
- Messaging: aio_pika, aiokafka
- Auth: ldap3, gssapi, spnego, pyotp, cryptography
- Observability: opentelemetry, prometheus_client
- ML: sentence_transformers, scipy, sklearn, statsmodels, faiss
- Vector stores: lancedb, proximadb_sdk
| Issue | Severity | Description |
|-------|----------|-------------|
| Monolithic core | High | All 1,900 files ship in one wheel; no lazy-load boundaries |
| Heavy optional deps | High | embeddings pulls sentence-transformers (500MB+), lancedb, pyarrow |
| Redundant HTTP clients | Medium | httpx + aiohttp + requests all present |
| Cloud SDK sprawl | Medium | boto3, google, kubernetes -- each 20-50MB installed |
| Unused imports | Low | fnmatch (19 files) -> pathlib; argparse (16) -> typer |
| Duplicate numpy pins | Low | numpy pinned in 3 places (core, embeddings, dev) |
| Rust build friction | Medium | Requires maturin + Rust toolchain; no pre-built wheels |
| No minimal install | High | No victor-ai[minimal] extra for constrained environments |
---
| Current | Replace With | Files | Effort |
|---------|-------------|-------|--------|
| fnmatch | pathlib.PurePath.match() | 19 | Low |
| argparse | typer (already dep) | 16 | Low |
| requests | httpx (already dep) | 5 | Low |
| aiofiles | anyio or asyncio builtins | 4 | Low |
| tomli/tomllib | stdlib tomllib (3.11+) | 6 | Low |
| json (stdlib) | orjson (already dep) | 297 | Medium |
| numpy (2 pins) | Single pin in core | 3 locations | Low |
| packaging | pkg_resources or inline | 4 | Low |
Move these from mandatory to optional:
- anthropic -> providers-anthropic (only needed for Claude)
- openai -> providers-openai (only needed for OpenAI)
- textual -> ui-full (TUI only; CLI works without)
- prompt-toolkit -> ui-full (only for interactive REPL)
- aiohttp -> optional (httpx covers 90% of use cases)
- numpy -> optional or victor_native (only for embeddings/ML)
Resulting minimal core: ~10 packages: pydantic, pydantic-settings, python-dotenv, typing-extensions, orjson, httpx, typer, rich, gitpython, pyyaml, jsonschema
---
| Crate | Purpose | Python Bridge | Speedup | Priority |
|-------|---------|---------------|---------|----------|
| protocol | Portable types | victor/native/protocols.py | 5-10x | P1 |
| state | Conversation/state | victor/native/ | 10-50x | P1 |
| tools | Tool registry | victor/native/ | 2-5x | P2 |
| edge-runtime | Standalone binary | N/A | N/A | P3 |
| python-bindings | cdylib entry | All above | Aggregate | P1 |
Python fallback modules (14 files in victor/processing/native/): All use the _NATIVE_AVAILABLE pattern.
Publish victor_native as a separate PyPI package with platform-specific wheels:
- Platforms: manylinux_2_28_x86_64, manylinux_2_28_aarch64, macosx_11_0_x86_64, macosx_11_0_arm64, win_amd64
- Build: maturin build --release --out dist
- Publish: maturin publish per release
Optional dependency in victor-ai:

---
| Project | Method | Size | Config | Plugin |
|---------|--------|------|--------|--------|
| Claude Code | npm package | ~50MB | .claude.json | None |
| Codex CLI | npm package | ~40MB | codex.json | None |
| OpenCode | pip install | ~30MB | .opencode.yaml | Entry points |
| Aider | pip install | ~100MB | .aider.conf.yml | None |
| Cline | VS Code ext | ~20MB | VS Code settings | None |
| Victor (current) | pip install | ~200MB+ | .victor/init.md | victor.plugins |
- Claude Code/Codex CLI prioritize zero-dependency install via bundled Node.js + esbuild
- OpenCode proves a pure-Python agent CLI is viable at ~30MB
- Victor's Rust extensions are a differentiator (performance) that should be optional
- All competitors use per-project config files (.claude.json, codex.json, .opencode.yaml)
- None have a plugin system -- Victor's victor.plugins is unique
---
| Tier | Format | Contents | Size | Install |
|------|--------|----------|------|---------|
| P0: Minimal | sdist + wheel | Core + CLI + 1 provider | ~15MB | pip install victor-ai |
| P1: Standard | sdist + wheel | Core + all providers + tools | ~30MB | pip install victor-ai[standard] |
| P2: Full | sdist + wheel | Everything + embeddings + ML | ~100MB | pip install victor-ai[all] |
| P3: Native | Platform wheels | victor_native Rust ext | ~5-10MB | pip install victor-ai[native] |
| P4: Standalone | PyInstaller | All-in-one binary | ~200MB | GitHub Releases |
| P5: Docker | Container | Full stack + API | ~500MB | docker pull victor-ai/server |
| Package | Contents | Dependencies | Size | Cadence |
|---------|----------|-------------|------|---------|
| victor-contracts | Protocol definitions | typing-extensions | ~15K LOC | Per SDK change |
| victor-ai-core | Framework core (no providers/UI) | 10 packages | ~300K LOC | Per release |
| victor-ai | Full CLI + providers + UI | victor-ai-core + extras | ~818K LOC | Per release |
| victor_native | Rust PyO3 wheels (separate) | None (binary) | ~5-10MB | Per release |
| Sub-phase | What | Effort | Dependencies |
|-----------|------|--------|-------------|
| 6a: Audit | Remove redundant imports | 2-3 days | None |
| 6b: Split core | Extract victor-ai-core | 3-5 days | 6a |
| 6c: Native wheels | Publish victor_native via CI | 2-3 days | maturin setup |
| 6d: Tiers | Restructure extras | 1-2 days | 6b |
| 6e: Standalone | PyInstaller bundle | 3-5 days | 6c |
| 6f: Homebrew | macOS formula | 1 day | 6e |
~/.victor/
  profiles.yaml       Provider configs (existing)
  settings.yaml       Global user preferences (new)
  skills/             User-defined skills (existing)
<project>/.victor/
  config.yaml         Per-project overrides (new, replaces init.md)
  init.md             Project context (existing, backward compat)
  project.db          Project database (existing)
| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Package split breaks installs | Low | High | Keep victor-ai as meta-package |
| Native wheel build fails | Medium | Medium | Python fallbacks always available |
| PyInstaller bundle too large | Medium | Low | Tree-shake unused modules |
| Homebrew formula maintenance | Low | Low | Automate via CI |
---
1. Audit and consolidate redundant imports (fnmatch, argparse, requests, aiofiles, tomli)
2. Move providers and UI to optional extras (anthropic, openai, textual, prompt-toolkit)
3. Publish victor_native as separate PyPI package with platform wheels
4. Split victor-ai-core from victor-ai for modular installs
5. Add minimal/standard/full extras for tiered installs
6. Add PyInstaller standalone for zero-Python environments
7. Add Homebrew formula for macOS users
8. Adopt .victor/config.yaml for per-project config (inspired by Claude Code/OpenCode)
9. Keep victor.plugins entry points -- unique differentiator
Estimated total effort: 12-19 days across 6 sub-phases
---
*This document is part of the UX Redesign worktree (.worktrees/ux-redesign) and supersedes the PLANNED Phase 6 section in ux-redesign-plan.md.*
