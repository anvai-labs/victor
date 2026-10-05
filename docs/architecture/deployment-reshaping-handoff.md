# Deployment Reshaping — Handoff for the Next Session

**Date:** 2026-10-04 · **From:** Claude Code session (deployment reshaping Phase 0+1 + release)
**To:** Codex / next Claude session · **Status:** Phase 0+1 COMPLETE, v0.12.0 RELEASED, all PRs merged

## What this session accomplished

Victor's core installation was reshaped: **numpy is no longer a dependency**,
the frozen-binary channel covers Linux/macOS/Windows with per-artifact smoke
tests, uv is the recommended CLI install, and a bare-install release gate
enforces numpy-absence at every publish. v0.12.0 shipped end-to-end on PyPI
(victor-ai + victor-native 0.12.0, 7 platform wheels) and the homebrew tap is
at 0.12.0.

### PRs merged (in landing order)

| PR | What |
|----|------|
| #1224 | dist-metrics gate; **click core-dep fix** (typer≥0.27 dropped click → bare installs crashed `--help`) |
| #1225 | numpy-optional similarity: victor_native → numpy accelerator → `victor/core/vecmath.py` |
| #1226 | uv-first install docs (Method 1), ADR-016 addendum, install.sh `--uv` |
| #1227 | numpy-free FEP-0012 runtime, `victor/ml/npz_reader.py`, 98 committed goldens, poisoned-subprocess guards |
| #1230 | numpy out of core deps; `bare-install-smoke` release gate |
| #1231 | Linux x64 PyInstaller binaries + per-artifact smoke (first-ever) |
| #1233 | frozen-binary dynamic imports (protocols/coordinators/providers) |
| #1234 | Windows-safe secure_paths (pwd was Unix-only), portable smoke, arm64 defer |
| #1238 | import-guard stderr surfacing + selector/orchestrator cycle break |
| #1239 | lazy ab_test patch target + bare-install artifact gate |
| #1242 | full lock refresh (replaced stale dependabot recreations) |
| #1243 | **textual → tui extra** (TUI consolidation: 3 stacks → 2) |
| #1247 | profiler numpy-free (stdlib statistics) — completes T5 sweep |

Non-PR releases: #1209 (durable actions, owner-merged), #1222/#1201 (dependabots),
#1223/#1232/#1240 (stale dependabots closed, superseded by #1242 lock refresh).

## Current state

- **develop tip** = main tip (v0.12.0 promoted via #1236, merge-only, 2-parent)
- **PyPI**: `victor-ai 0.12.0` (wheel + sdist) + `victor-native 0.12.0` (7 wheels)
- **Tap**: anvai-labs/homebrew-tap at 0.12.0 (livecheck all-at-latest)
- **Core deps**: 62 (was 64), **zero module-level numpy imports** in victor/
  (except dev-only `ml/trainer.py` gated by `[ml]` extra)
- **No open PRs from this session**

## The next session should

### 1. Immediate (bounded, testable)

- **Dependabot recreations**: #1240/#1232 were closed as stale (locks shifted).
  Dependabot will auto-recreate. Merge when green.
- **#1237** (`refactor(agent): bind provider retry policy to stream lifecycle`):
  someone else's PR — needs author review, not ours to merge.

### 2. Phase 2 — Sandhi-routed SDKs

The anthropic and openai SDKs are core deps (~24MB combined). Sandhi-gateway
(already a core dep) provides typed transport for all providers. The SDKs are
redundant on the gateway path.

**Pressure-test**: (a) verify every provider call routes through Sandhi in the
default config; (b) check for direct SDK usage that can't go through Sandhi;
(c) measure token/latency parity.

**Scope**: move anthropic+openai to extras; the sandhi-gateway dep provides
the transport. Provider fallback (non-Sandhi path) keeps the SDKs as optional.

### 3. Phase 2 — TUI consolidation (remaining)

textual moved to `tui` extra (#1243). The remaining overlap is
prompt-toolkit (interactive chat input, 7 function-local imports in
`chat.py`) vs rich (rendering). These serve different purposes; the
consolidation is **prompt-toolkit → lazy** (it's already function-local)
rather than removal. Lower priority than the SDK work.

### 4. Sandhi aarch64 wheel (upstream)

`sandhi-gateway==0.12.0` publishes no manylinux-aarch64 wheel. This blocks:
the linux-arm64 binary leg AND pip installs on that platform. Fix is in the
Sandhi repo: add `aarch64-unknown-linux-gnu` target to the maturin build
matrix (the `ubuntu-24.04-arm` runner already exists in victor's CI).

### 5. TestPyPI publisher (user action)

Register on test.pypi.org → victor-ai → Publishing:
owner `anvai-labs`, repo `victor`, workflow `release.yml`, environment
`testpypi`. This enables the TestPyPI rehearsal dispatch. Production-first
works without it (v0.12.0 proved that).

### 6. Gap-ledger residues (G60–G65)

The formation handoff ledger (`docs/architecture/multiagent-formation-
coverage-handoff.md`) tracks open gaps. The bounded ones: result-publication
follow-ups (G60 remainder), bounded durable HTTP settlement (G53 remainder,
Sandhi-side). #1209 shipped G62's local persistence piece.

## Critical context for pressure-testing

### The numpy-optional architecture

```
Core deps (62):  pydantic, orjson, sandhi-gateway==0.12.0, click, typer,
                 rich, prompt-toolkit, httpx, aiohttp, tiktoken, ...
NOT in core:     numpy (→ victor_native or vecmath fallback)
                 textual (→ tui extra)
                 torch/lancedb (→ embeddings extra)
                 fastapi/uvicorn (→ gateway extra)
```

Three enforcement layers:
1. `test_cli_import_survives_without_numpy` — poisons sys.modules, proves
   the CLI chain *functions* (not just that no import happened)
2. `test_cli_import_survives_without_textual` — same pattern for textual
3. `bare-install-smoke` release job — fresh venv, wheel-only install,
   numpy-absence assert, CLI chain probes, dist-metrics gates

### The synchronized versioning policy

`sync_version.py --ai` fans VERSION out to: pyproject.toml, rust/pyproject.toml,
rust/crates/python-bindings/Cargo.toml, rust/Cargo.lock (victor_native entry),
and the root `native` extra bound. `check_version_sync.py` FAILS the release
if any spot drifts. The native wheel version always equals the release version.

### Release pipeline mechanics

- **Tag required**: `release_contract.py plan` rejects non-tag refs for
  production. The tag must be `v{VERSION}` matching the VERSION file.
- **Merge-only promotion**: PR develop→main, `--merge --admin` (main requires
  1 review, enforce_admins=false). NEVER squash (ancestry severs).
- **Rehearsal**: `gh workflow run release.yml --ref develop -f publish_testpypi=true`.
  Requires the TestPyPI trusted publisher (not yet registered — user action).
- **If a release fails**: delete the GH release, delete the tag, re-tag at
  the fixed commit, push. (Done for v0.11.0→v0.12.0.)

### Gotchas (learned the hard way)

1. **typer ≥0.27 dropped click** — always declare click as a direct dep if
   you import it. (The 0.11.0 wheel shipped without click.)
2. **similarity.py imports `_NATIVE_AVAILABLE` BY VALUE** — monkeypatch
   `similarity._NATIVE_AVAILABLE`, not `_base`'s.
3. **`find_spec` raises ValueError** on None-poisoned sys.modules entries —
   the probe catches `(ImportError, ValueError)`.
4. **stacked-PR rebases onto squashed develop** need cherry-picks (rebase
   replays pre-squash duplicates). Or close+recreate from v2 branches.
5. **Shared CI runners are ~2× slower** than a dev machine — timing gates
   calibrated to 3000ms (the rehearsal tripped 1500ms with a 2030ms run).
6. **select_changed_tests is fail-closed** — changed source without a
   mirrored test blocks the PR. Map: `victor/<rel>/<name>.py` →
   `tests/unit/<rel>/test_<name>.py` (rglob fallback for stem match).
7. **diff-cover 80% fail-under on changed lines** — every changed line must
   be covered by the selected tests, including fallback branches.
8. **`pip install -e ./victor-contracts` FIRST** in CI — victor-contracts
   resolves the dependency; without it, "No matching distribution" error.
9. **macOS runners lack GNU `timeout`** — use portable patterns in smoke
   steps, or the job timeout.
10. **`sys.modules['numpy'] = None`** makes `import numpy` raise ImportError —
    this is the bare-install simulation. A membership check (`'numpy' in
    sys.modules`) can't distinguish eager from guarded imports; the poison
    proves function under absence.

### Key file paths

| What | Where |
|------|-------|
| Version sync | `scripts/sync_version.py`, `scripts/check_version_sync.py` |
| Dist metrics | `scripts/ci/dist_metrics.py` |
| Release contract | `scripts/ci/release_contract.py` |
| Binary builder | `scripts/build_binary.py` |
| Gap ledger | `docs/architecture/multiagent-formation-coverage-handoff.md` |
| Formation doc | `docs/architecture/multiagent-formations-inferflux.md` |
| vecmath fallback | `victor/core/vecmath.py` |
| npz reader | `victor/ml/npz_reader.py` |
| Golden tests | `tests/unit/ml/goldens/fep0012_predict_goldens.json` |
| Rust workspace | `rust/crates/{protocol,state,tools,edge-runtime,python-bindings}/` |
| TestPyPI publisher | test.pypi.org → victor-ai → Publishing (NOT YET REGISTERED) |

### Development environment

- **Worktree**: `.claude/worktrees/fix-inferflux-codesign` on `origin/develop`
- **Venv**: `.venv-codesign/bin/python` (NEVER bare `pytest`)
- **Local InferFlux**: `ssh vsingh@aiserver1`, port 8080 via SSH tunnel
  (`ssh -N -L 8080:127.0.0.1:8080 vsingh@aiserver1`), auth `dev-key-123`
- **Release**: `make release VERSION=x.y.z` or manual: write VERSION →
  `python scripts/sync_version.py` → `python scripts/check_version_sync.py` →
  commit → tag → push
