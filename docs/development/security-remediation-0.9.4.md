# Security remediation: 0.9.4 candidate

Status: prepared for integration; **not released**. This batch follows the
security gate changes in PR #1058. It contains security fixes and dependency
maintenance, so 0.9.4 is appropriate; the broader Stage C feature work remains
separate. The [dependency guide](dependencies.md) describes deployment shapes
and reproducible resolution, and [SECURITY.md](https://github.com/anvai-labs/victor/blob/develop/SECURITY.md) defines gates
and the exception process.

## Included changes

| Surface | Result |
| --- | --- |
| Persistent caches | Versioned data-only formats replace pickle/diskcache loading; legacy caches rebuild, and malformed snapshots cannot partially replace live state. |
| SQL and parsers | Bind values, quote identifiers, validate dynamic columns, bound condition evaluation, and disable XML entity expansion. |
| Runtime policy | Fail closed on safety/approval setup errors; preserve inherited tool restrictions; roll back temporary budgets and reject reuse after restoration failure. |
| Process and editor integration | Preserve configured MCP sandbox startup on reconnect, pass search patterns as data, avoid shell interpolation for backend/Git context commands, and confine editor changes to workspace paths. |
| Advisory coverage | Calculate CVSS base scores, follow OSV continuation pages, preserve unknown ratings/errors, and key validated query snapshots by exact version. |
| Integrity and validation | Keep SDK validation effective under optimized Python; use SHA-256 for undo integrity and verify legacy records against stored text. |
| Dependencies | Remove unused runtime/build dependencies, update Python/JavaScript/Rust packages, and keep RAG storage imports lazy. |
| Distribution | Consolidate four container targets, separate coding native wheel identity, and exclude development files from the VSIX. |

Mode and approval controls are application policy, not an OS security boundary.
Workspace path validation is not an atomic filesystem sandbox against concurrent
external path replacement. A lightweight workspace scan reports incomplete or
degraded results explicitly and is not a comprehensive security audit.

## Validation at candidate preparation

- Changed-file and added regression tests: **934 passed, 2 skipped**, with one
  slow test deselected. After moving the coding-only regression to its package
  suite, root collection contains **32,280 tests**, with no collection errors;
  the moved regression remains covered by the coding suite. The final image-gate
  regression cases bring root collection to **32,291 tests**, without errors.
  The image-gate suite passes all **61 tests** and independent review.
- Python formatting, production lint, version consistency, repository hygiene
  and the repository-configured global strict mypy gate passed. Separate checks
  outside that gate retain pre-existing typing errors in benchmark/SDK areas.
- PR #1060 at `4c5b09e0b` passed CI Success and all required checks. The newer
  advisory, MCP and Ubuntu-container changes are being validated locally before
  a consolidated push; those earlier green checks do not validate this new diff.
- The first PR CI cycle passed all security, dependency, typing, collection and
  architecture gates. Its only test failure was a coding-vertical regression
  placed in the core suite, whose environment does not install that vertical.
  The regression now runs in the coding package suite; its assertions remain
  unchanged.
- Extension unit tests: **50 passed**; TypeScript compilation passed. ESLint has
  nine existing warnings and no errors.
- Earlier native parity validation: **297 passed, 11 skipped**. The Rust
  workspace tests, doctests and four publishable crate archives passed after the
  version changes; PR native parity also passed.
- Earlier installed-environment audits covered core, API, CPU embeddings and an
  independently resolved compatible extras/verticals environment; JavaScript and
  Rust dependency audits found no advisories at that scan time. These results do
  not replace scans of the final commit and released artifacts.

- The extended security suite passed **460 tests, 1 skipped**. Separate MCP
  resource-limit, startup and workflow checks passed **34 tests**. Advisory/cache
  changes passed independent review, including **113 focused tests** and extra
  reproductions of conflicting records and malformed cached snapshots.

Independent review covered cache storage, SQL, parser conditions, SDK validation,
undo integrity, dependency discovery and runtime budget restoration. Final review
of editor/process changes and the wider native edge surface remains incomplete.
Do not interpret local test results as approval of those outstanding scopes.

## Open findings and release conditions

1. **Final container scans pass the existing publication threshold.** The old Debian core image at
   `4079eb623` reported 57 blocking OS findings. The Ubuntu 24.04 core candidate
   passes the unchanged report gate with **0 critical/high/unknown** findings,
   **45 medium and 4 low** OS findings, and **0 Python** findings. Offline runtime
   checks preserve Python 3.12, Git/SSH, UID 1000, AI 0.9.4 and SDK 0.9.2. The
   native candidate builds and imports native 0.8.1 successfully.
   Final core, MCP and native scans each retain **45 medium / 4 low** OS
   findings; full retains **55 medium / 4 low**. Every target has **0 blocking OS
   findings and 0 Python findings**. Final CLI/API, MCP, native import and offline
   embedding probes pass. Full inventories and CycloneDX SBOMs are retained.
   In particular, Expat CVE-2026-76957 remains unfixed and is rated medium by
   Ubuntu; changing a distribution's severity classification is not a fix.
   The exception manifest remains empty and no gate threshold has changed.
2. **Independent review remains required** for the outstanding scopes above
   before merge is requested. The review service could not complete those
   scopes; partial results are not accepted as approval.
3. **MCP compatibility and review:** the process backend now rejects unsupported
   filesystem/network/namespace/seccomp policies and fails child startup when
   rlimits or root demotion fail. Root demotion clears supplementary groups.
   It remains a resource-limit backend, not an OS isolation boundary. Use an
   external sandbox/container for stronger policies; see [SECURITY.md](https://github.com/anvai-labs/victor/blob/develop/SECURITY.md#mcp-process-limits).
   Local tests do not replace the outstanding independent process review.
4. **Refresh artifact evidence** after the final commit. The 0.5.1 VSIX builds
   with 14 files. The final follow-up collects **32,445 tests** without errors
   and passes **98 affected tests**, repository formatting/lint, global strict
   typing, hygiene, strict docs and built-site links (zero issues). Filesystem
   scanning of all committed snapshots reports zero findings. The final image
   results above cover the source candidate; CI must pass its eventual commit.

## Release order

The candidate versions are `victor-ai` **0.9.4**, `victor-contracts` **0.9.2**,
`victor-native` **0.8.1**, and the VS Code extension **0.5.1**. Coding native
metadata identifies the separate `victor-coding-native` **0.1.0** package.
Other independently versioned vertical packages need separate releases to deliver
their source changes to existing installations.

1. Finish review and required CI, then merge the batch into `develop`.
2. Publish SDK 0.9.2 through its independent release workflow. The AI package and
   snapshots require that SDK version so an upgrade receives its safety fixes.
   Container builds use the in-tree SDK wheel before publication.
3. Resolve the release conditions above, open the explicit `develop` → `main`
   promotion PR, and pass the full promotion battery without bypassing gates.
4. Merge the promotion, tag `v0.9.4` on its main commit, and verify publication of
   Python, native and container artifacts. Verify extension publication through
   its own distribution path; a changed manifest alone does not publish it.

Keep the batch closed while CI runs. Consolidate related fixes locally before a
follow-up push, and do not restart queued runs to seek runner capacity.
