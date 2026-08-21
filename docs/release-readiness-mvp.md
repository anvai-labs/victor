# Victor MVP Release Readiness

Date: 2026-08-20
Scope: v0.9.0 release closeout, including promotion CI, release automation, artifacts, and public
registry verification.

## Decision

Status: **v0.9.0 released.** The `develop` promotion passed the complete required matrix and merged to
`main` as `c9736e41ddab3014479f5eaeb24e10870053b4d8`. Annotated tag `v0.9.0` triggered the
[release workflow](https://github.com/anvai-labs/victor/actions/runs/32429452576), which completed
successfully. The [GitHub Release](https://github.com/anvai-labs/victor/releases/tag/v0.9.0), PyPI
wheel/sdist, native wheels, standalone binaries, VS Code extension, checksums, SBOM, and Docker image
were produced or published successfully.

Post-publication clean-environment smokes passed on Python 3.11 and 3.12: each environment installed
`victor-ai==0.9.0` from PyPI with its declared dependencies, imported `victor` and `Agent`, reported
version 0.9.0, and rendered `victor --help`. The base install emitted the expected notice that the
optional `victor-coding` plugin was not installed.

The Docker Trivy job reported vulnerability findings and uploaded SARIF to GitHub Security. That job
is intentionally advisory (`continue-on-error: true`) and did not block the release; triage of those
findings is the remaining post-release security follow-up. TestPyPI was intentionally skipped for the
stable tag.

## Historical Pre-release Evidence

Commands run from the repository root:

The artifact-specific results below are the dated 0.8.3 baseline that informed the release plan.
They are retained for provenance and are superseded by the v0.9.0 promotion and release runs above.

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

The v0.9.0 promotion and release workflows supplied the full unit, integration, security, Docker,
VS Code, native Rust, packaging, and publication evidence that this baseline lacked.

## Design Document Reconciliation

| Document | Current Status | Release Interpretation |
| --- | --- | --- |
| `VISION.md` | Current and edited locally. Adds durable code memory / ProximaDB as a near-term product bet. | OK as strategic direction, but not an MVP promise. |
| `docs/architecture.md` | Canonical architecture. Service-first runtime, 6 canonical services, provider/tool/storage layers. Adds ProximaDB direction. | Mostly current. ProximaDB section must remain clearly "planned direction." |
| `docs/tech-stack.md` | Canonical stack and debt register. TD-11, TD-12, TD-13 added for ProximaDB CCG work. | Current if those items stay Planned. Good place to track release debt. |
| `docs/roadmap.md` | Canonical roadmap. v0.9.0 closeout and the EVR backlog are current. | Keep its release checklist synchronized with this document at each verification run. |
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

## v0.9.0 Release Outcome

1. **Complete — packaging and promotion proof.** The promotion passed all 36 full-suite shards across
   Python 3.11, 3.12, and 3.13, plus the blocking integration, vertical, security, build, CLI smoke,
   and collection gates. The release run then passed version sync, package build, and smoke checks.

2. **Complete — release-critical gates.** Core tests, package/CLI smoke, integration, vertical
   compatibility, artifact builds, and publication gated the stable release. TestPyPI remains optional
   for stable tags. Trivy remains advisory and requires post-release finding triage.

3. **Complete — release notes and artifacts.** `CHANGELOG.md` contains the 0.9.0 entry and the GitHub
   Release includes the Python wheel/sdist, native wheels, binaries, checksums, and SBOM.

4. **Complete for the shipped release workflow.** CLI imports/help, VS Code packaging, native wheel
   builds, and Docker build/publish passed. API/MCP and UI support continue to be covered by their
   blocking promotion suites and documented extras.

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
