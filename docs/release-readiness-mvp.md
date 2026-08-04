# Victor MVP Release Readiness

Date: 2026-08-03
Scope: repo-local review of architecture docs, packaging metadata, CI/release workflows, core runtime
layout, and focused guardrail tests.

## Decision

Status: not ready to cut a release tag yet.

The core Python framework is close: version files are aligned, canonical docs pass the drift guard,
the distribution checklist and a Python 3.12 wheel build/install smoke pass. The remaining work is
release discipline: repeat artifact verification on Python 3.11, make the advertised CI gates blocking
or scope their support level, and verify the optional advertised surfaces.

## Evidence Collected

Commands run from the repository root:

The artifact-specific results below are a dated 0.8.3 baseline. The source tree
has since advanced to 0.8.4, so those build, install, and smoke checks must be
repeated for the release candidate rather than treated as transferable proof.

| Check | Result |
| --- | --- |
| `make check-version` | Passed 2026-08-03: `victor-ai` is `0.8.3`; `victor-contracts` is `0.9.0`; the declared dependency range is compatible. |
| `python scripts/ci/check_docs_drift.py` | Passed 2026-08-03. Docs align to `version=0.8.3`, `providers=24`, `tool_modules=34`, `verticals=9`, and Python 3.11. |
| `python scripts/ci/repo_hygiene_check.py` | Passed. |
| Python import smoke | Superseded by the 2026-08-03 built-wheel smoke below; the old checkout smoke recorded 0.7.1 and is not release evidence. |
| `pytest tests/unit/contracts -q` | Passed: 94 passed, 6 skipped, 1 deprecation warning. |
| `pytest tests/unit/agent/services/test_service_layer_validation.py -q` | Passed: 13 passed. |
| selected import-boundary guard tests | Passed: 4 passed. |
| `python -m build --wheel --no-isolation` | Passed 2026-08-03 under Python 3.12; produced `victor_ai-0.8.3-py3-none-any.whl`. |
| `twine check dist/victor_ai-0.8.3-py3-none-any.whl` | Passed 2026-08-03. |
| Wheel install/import smoke outside the checkout | Passed 2026-08-03 under Python 3.11 and 3.12: installed `victor-ai 0.8.3` with its declared dependencies into fresh temporary environments, then imported `victor`, `Agent`, and the CLI application. |
| `make check-dist` | Passed 2026-08-03 after the target was corrected to check the tracked `scripts/homebrew/victor.rb` formula. |
| CLI help/version smoke | Passed 2026-08-03: `victor --version`, `victor --help`, and `victor ui --help` all start without network access. |
| API/MCP import smoke | Passed 2026-08-03: `create_fastapi_app()` creates the `Victor API` application and `victor.integrations.mcp` imports. |
| Docker image smoke | Blocked locally 2026-08-03: Docker BuildKit returns an immediate `EOF` before any Dockerfile stage is reported, including with `--load --progress=plain`. Treat Docker build/run smoke as a required CI release check. |

Full unit, integration, security, Docker CI, VS Code, native Rust, and release workflows still
require their release-run evidence.

## Design Document Reconciliation

| Document | Current Status | Release Interpretation |
| --- | --- | --- |
| `VISION.md` | Current and edited locally. Adds durable code memory / ProximaDB as a near-term product bet. | OK as strategic direction, but not an MVP promise. |
| `docs/architecture.md` | Canonical architecture. Service-first runtime, 6 canonical services, provider/tool/storage layers. Adds ProximaDB direction. | Mostly current. ProximaDB section must remain clearly "planned direction." |
| `docs/tech-stack.md` | Canonical stack and debt register. TD-11, TD-12, TD-13 added for ProximaDB CCG work. | Current if those items stay Planned. Good place to track release debt. |
| `docs/roadmap.md` | Canonical roadmap. Release train and EVR backlog are current. | Keep its release checklist synchronized with this document at each verification run. |
| `docs/features.md` | Canonical feature catalog. Claims 24 providers, 34 tools, 9 verticals, Chainlit UI, policy engine, sandbox. | Needs support-level tagging for optional/experimental features before release. |
| `docs/architecture/proximadb-codegraph-backend.md` | New untracked design-intent doc. | Good design record. Not part of MVP unless implemented behind `GraphStoreProtocol`. |
| FEP/ADR EVR docs | Evaluation-centric runtime design exists. | Q3 backlog, not MVP release criteria unless used as a release gate. |
| `docs/development/releasing/publishing.md` | Release process exists but includes old example versions and manual version-edit guidance. | Needs update to current process: `VERSION` + `sync_version`, clean build, twine check, CLI smoke. |

## Implemented MVP Surface

- Public Python API: `victor.framework.Agent`, `StateGraph`, `WorkflowEngine`, tools, events.
- Service-first runtime: `ChatService`, `ToolService`, `SessionService`, `ContextService`,
  `ProviderService`, `RecoveryService` are present and guard-tested.
- Provider layer: 24 provider adapter files are present and docs drift check derives that count.
- Tool layer: documented 34 tool-module canon is enforced by docs drift.
- Contract boundary: `victor-contracts` exists as an independently versioned package and contract tests pass.
- Packaging metadata: `pyproject.toml`, root `VERSION`, and `victor-contracts/VERSION` are internally aligned.
- Release automation: tag-triggered release workflow exists for PyPI, native wheels, binaries, Docker,
  checksums, and GitHub Release.

## MVP Release Blockers

1. Capture clean packaging proof in CI on the supported matrix.
   Run `python -m build`, `twine check dist/*`, install the built wheel with the built
   `victor-contracts` wheel, and run import plus CLI smoke under Python 3.11 and 3.12. Both Python
   versions passed locally in fresh environments on 2026-08-03; the release run still needs its CI
   artifacts and logs.

2. Make release-critical CI gates blocking.
   **Core package tests and built-wheel CLI smoke are blocking as of 2026-08-03.** Before tagging,
   decide whether the remaining advisory TestPyPI publishing, VS Code, and Rust checks are
   release-supported or explicitly experimental; the integration and vertical compatibility suites are
   already blocking.

3. Finalize release notes.
   The current package is `0.8.4`; the release cut still needs a final
   support-level pass: core runtime, contracts, CLI/API/MCP, optional chat UI, external verticals,
   native extensions, Docker/VS Code.

4. Verify optional surfaces that are advertised as MVP.
   At minimum: `victor --help`, `victor --version`, one no-network chat/help path, API server import,
   MCP server import, `victor ui --help` with `chat-ui` extra installed, and Docker image smoke.

## Not MVP

- ProximaDB CCG backend implementation (`ProximaGraphStore`, one-`oid` graph/vector/relational record).
- EVR Q3 evaluation-centric runtime backlog.
- Publishing benchmark superiority claims beyond whatever has a reproducible artifact.
- Full observability productization unless the release explicitly supports the dashboard/API surface.
- Workspace-isolation rename completion if current behavior is stable and documented.

## Recommended MVP Cut Checklist

1. Clean tree except intentional release docs.
2. Update `CHANGELOG.md` and roadmap status.
3. Run `make check-version`, `make check-repo-hygiene`, docs drift, format/lint/type gates.
4. Run focused guardrail tests plus full unit suite.
5. Run package build/install smoke from wheel on Python 3.11 and 3.12.
6. Run release workflow dry run or TestPyPI publish.
7. Decide and document support level for external verticals, Chainlit UI, Docker, native Rust wheels,
   VS Code extension, and observability.
8. Tag only after release artifacts and smoke tests are green in CI.
