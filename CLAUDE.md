# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What Victor Is

A contract-first agentic AI framework (Python 3.11+): typed framework API, service-first agent runtime, StateGraph workflows, multi-agent teams, 25 LLM provider adapters, and 34 tool modules. Domain behavior lives in plugin packages, not the core.

## Common Commands

```bash
# Setup (contracts MUST be installed editable first, so victor-ai resolves
# the in-repo victor-contracts instead of the last PyPI release)
pip install -e ./victor-contracts && pip install -e ".[dev]"
make install-dev                 # same, plus docs/build extras + pre-commit
make install-verticals           # also editable-installs verticals/victor-*

# Tests
make test                        # unit tests (tests/unit)
make test-quick                  # unit tests, skipping -m slow
make test-all                    # everything including integration
pytest tests/unit/path/test_x.py::test_name -v   # single test
pytest -m "not slow"             # markers: slow, integration, workflows, agents, hitl, benchmark
make test-verticals              # per-vertical test suites

# Lint / format (line length 100)
make lint                        # ruff + black --check + mypy victor + repo hygiene
make format                      # black + ruff --fix
make check-repo-hygiene          # workflow/link/metadata drift guards
make check-vertical-boundaries   # verticals import only victor_contracts

# Run
victor                           # TUI
victor chat "..."                # CLI chat
make serve                       # HTTP API (victor serve)
```

Subprojects: `cd rust && cargo test` (PyO3 extensions; build with `maturin develop --release`), `npm --prefix vscode-victor run compile`.

Pytest config uses `--strict-markers` and `asyncio_mode = "auto"` (no `@pytest.mark.asyncio` needed).

## Architecture

Canonical doc: `docs/architecture.md`. Strict layering, each layer depends only on the one below, enforced by guard tests:

1. **Client** (`victor/ui/`, `victor/integrations/api/`, `victor/integrations/mcp/`, vscode-victor) — must use `VictorClient` + `SessionConfig` from `victor/framework/client.py`; never imports `victor.agent.*` (guard: `test_architectural_boundaries.py`).
2. **Framework** (`victor/framework/`) — the stable public API: `Agent` (agent.py), `StateGraph` (graph.py), `WorkflowEngine`, tool registry, extension surfaces. `AgentFactory` (`agent_factory.py`) is the single authority for all agent creation paths (CLI, API, `Agent.create()`).
3. **Runtime** (`victor/agent/`) — internal implementation. `AgentOrchestrator` (`orchestrator.py`) is a **facade only**; all effectful behavior belongs to the six canonical services in `victor/agent/services/`: Chat, Tool, Session, Context, Provider, Recovery (guard: `test_service_layer_validation.py`). Access via `ExecutionContext.services`. The `AgenticLoop` (`victor/framework/agentic_loop.py`) runs PERCEIVE → PLAN → ACT → EVALUATE → DECIDE.
4. **Infrastructure** — `victor/providers/` (adapters behind `BaseProvider` with circuit breaker/retry/routing), `victor/tools/` (auto-registered via SharedToolRegistry, executed through `victor/agent/tool_pipeline.py`), `victor/state/` (GlobalStateManager, 4 scopes), `victor/config/settings.py`, `victor/core/database.py`.

### Contract boundary (the monorepo discipline)

- `victor-contracts/` is the definition-layer SDK. External and vertical packages import **only** `victor_contracts`, `victor.framework.extensions`, or documented public APIs — never `victor.agent.*` or root runtime internals (guard: `test_core_vertical_import_boundary.py`, `make check-vertical-boundaries`).
- First-party verticals live in `verticals/victor-{coding,devops,research,rag,dataanalysis}` as separately installed editable packages, discovered via the `victor.plugins` entry point (`VictorPlugin.register(context)`).
- `victor-codegraph/` is the shared tree-sitter code→graph chunker (own package, own release tags).
- The root framework stays generic; domain-specific behavior belongs in verticals/plugins.

### Other load-bearing facts

- **Two-database model**: global `~/.victor/victor.db` (settings, API keys, RL data) vs. project `./.victor/project.db` (graph, conversations, sessions). Access via `get_database()` / `get_project_database()` in `victor/core/database.py`. Project code intelligence is derived, rebuildable state — never a source-of-truth artifact.
- **Teams are formations, not graphs**: `UnifiedTeamCoordinator` (`victor/teams/`) is used directly as a StateGraph node. Do not create wrapper node types or a separate multi-agent graph abstraction.
- **YAML workflows** compile to StateGraph via `victor/workflows/unified_compiler.py`.
- **Rust fallback pattern**: every native hot path checks `_NATIVE_AVAILABLE` and falls back to Python when the extension is absent.
- Settings cascade: `.env` → `~/.victor/profiles.yaml` → CLI flags → immutable `SessionConfig`.

## Development Workflow

Full doc: `docs/development/PR_WORKFLOW.md`.

- **Branch flow**: feature branches PR into `develop` (lightweight `ci-fast` gate: lint, guards, changed-file unit tests). The `develop` → `main` promotion PR runs the extensive battery (sharded unit, integration, build, security). `main` is strict — required checks with no admin bypass.
- **Worktree mandate**: all `feature/*`, `fix/*`, etc. work happens in a linked git worktree (`git worktree add ../victor-<task> -b <branch> develop`), never by checking the branch out in the main tree. Only `main`/`develop` live in the main tree.
- **Worktree hygiene — minimal open worktrees**: keep the fewest worktrees open that the work genuinely needs (default: one). Drive each to completion — spec → implement → tests → lint → push → PR → merge — and remove the worktree (`git worktree remove`, delete local+remote branch) as soon as its PR merges, instead of accumulating parallel checkouts. Open a second worktree only for a truly independent PR that must be developed simultaneously; never as a parking spot for half-finished work.
- **Adversarial review is part of done**: before merge, attack your own diff like a hostile reviewer — bypasses, regressions, cross-boundary effects — and pin every claim with a positive test (the fix works) AND a negative test (the old bug / malformed input is rejected). Timing scales with complexity: small mechanical PRs get the pass after CI goes green; anything touching security, concurrency, caching/shared state, or layer boundaries gets it BEFORE requesting review, while fixes are still cheap. Findings are fixed in the same PR with their negative tests.
- Before pushing, run `pytest tests/ --collect-only -q` — the same collect-check CI runs on every develop PR; it catches stale imports that would red the whole shard matrix at promotion.

### Minimize CI cycles (shared runner capacity)

- **Stop on local failure:** commit/push automation must stop if any required check fails or cannot run. Never separate validation from commit/push with an unconditional shell separator. After the final edit, run Black check, Ruff, affected tests, typing and collection as applicable; a test-fixture change is still a code change. Do not push a candidate known to fail or rely on a prior commit's green results.
- **Review before the first push:** complete independent adversarial review for security, concurrency, shared-state and layering changes locally, fix all findings together, and validate the resulting diff before triggering CI. Persist the tested commit, commands, outcomes and unresolved limitations in the PR evidence. Batch related capability/dependency changes into a coherent reviewed unit rather than triggering a full pipeline for each individual access replacement.
- Treat hosted runner capacity as shared across the organization. Finish the full local change, dependency resolution, targeted and affected tests, formatting, lint, typing, collection checks, and required adversarial review before the first CI-triggering push. Include related documentation and version changes in that validated candidate.
- Batch compatible fixes into one reviewable PR to `develop`; resolve all known failures from a CI run locally before one consolidated follow-up push. Aim for one successful candidate cycle per PR and one full promotion battery per release; this is an efficiency target, never permission to skip required verification or accept a failure.
- Do not use remote CI as an edit/test loop, create speculative promotion PRs, push cosmetic follow-ups while checks run, or restart queued jobs to chase capacity. Re-run only failed jobs when evidence supports a transient infrastructure failure; code changes require the new commit's applicable checks.
- Cancel only obsolete runs for this task when superseded; never cancel unrelated work. Reuse local build/scan artifacts and caches, record the tested commit, and preserve all required security and correctness gates. Batch routine dependency updates; prioritize urgent security fixes without waiting for unrelated feature work.
- Before promotion, close the batch, verify local preflight and independent review, and check runner demand. Avoid redundant dispatches or duplicate workflow triggers; optimize scheduling and path coverage only when required checks still run and their aggregate fails on errors, cancellation, missing reports, or unknown results.

## Conventions

- Changes to `victor/framework/` public APIs, protocols, or core architectural patterns require a **FEP** (Framework Enhancement Proposal, see `feps/` and CONTRIBUTING.md). New tools, providers, verticals, and bug fixes do not.
- Type hints required on public APIs (mypy-enforced); Google-style docstrings; async/await for I/O; `respx` for HTTP mocking in tests.
- Versioning: `VERSION` file is the source of truth; `make sync-version` / `make check-version` keep victor-ai and victor-contracts in sync. victor-contracts releases independently (`sdk-v*` tags).

### Performance and native-code choices

- Choose architecture from measured end-to-end latency, throughput, CPU, memory, and reliability on representative workloads. Profile first; do not infer a hot path from file size or rewrite Python merely because compiled code may be faster.
- Simplify the algorithm, data flow, allocation rate, cache behavior, concurrency, and dependency surface before changing languages. Keep orchestration and I/O in typed async Python.
- For stable CPU-bound batch work that still misses a measured target, extend the existing Rust workspace through PyO3. Batch inputs, minimize FFI crossings and copies, release the interpreter for measured multi-millisecond Rust-only work, and use portable release targets with runtime feature detection.
- Do not add Cython, Numba, a direct CPython C/C++ extension, or another native toolchain unless a benchmarked case cannot be served cleanly by Python, a vectorized dependency already in that deployment shape, or the existing Rust/PyO3 path. Record the exception and its build, wheel, debugging, and maintenance cost in the design review.
- Every native path must preserve a typed Python reference implementation or an explicit installation requirement, differential parity tests, production-size benchmarks, bounded failure semantics, and observability of native versus fallback dispatch. Never replay a side effect after an FFI failure.
- Treat packaging as part of correctness: validate supported Python versions, operating systems, architectures, baseline CPU features, wheel installation, and fallback behavior before merge. A local `target-cpu=native` result is not release evidence.
- A performance refactor is complete only when it improves the measured user-level target without weakening correctness, cancellation, security, portability, or maintainability. Remove experiments that fail that gate.
- Use [Native Acceleration Strategy](docs/architecture/native-acceleration-strategy.md) for the decision matrix and current audit priorities.
