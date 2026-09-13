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
| Integrity and validation | Keep SDK validation effective under optimized Python; use SHA-256 for undo integrity and verify legacy records against stored text. |
| Dependencies | Remove unused runtime/build dependencies, update Python/JavaScript/Rust packages, and keep RAG storage imports lazy. |
| Distribution | Consolidate four container targets, separate coding native wheel identity, and exclude development files from the VSIX. |

Mode and approval controls are application policy, not an OS security boundary.
Workspace path validation is not an atomic filesystem sandbox against concurrent
external path replacement. A lightweight workspace scan reports incomplete or
degraded results explicitly and is not a comprehensive security audit.

## Validation at candidate preparation

- Changed-file and added regression tests: **934 passed, 2 skipped**, with one
  slow test deselected. Collection: **32,281 tests**, no collection errors.
- Python formatting, production lint, version consistency and repository hygiene
  passed. Targeted typing passed for the runtime changes; other touched areas
  retain pre-existing typing errors, so this is not a claim of a clean full mypy run.
- Extension unit tests: **50 passed**; TypeScript compilation passed. ESLint has
  nine existing warnings and no errors.
- Earlier native parity validation: **297 passed, 11 skipped**. Rebuild native
  artifacts for the final version before publication.
- Earlier installed-environment audits covered core, API, CPU embeddings and an
  independently resolved compatible extras/verticals environment; JavaScript and
  Rust dependency audits found no advisories at that scan time. These results do
  not replace scans of the final commit and released artifacts.

Independent review covered cache storage, SQL, parser conditions, SDK validation,
undo integrity, dependency discovery and runtime budget restoration. Final review
of editor/process changes and the wider native edge surface remains incomplete.
Do not interpret local test results as approval of those outstanding scopes.

## Open findings and release conditions

1. **Container OS findings remain blocking.** Prior scans of all four candidate
   targets reported unaccepted high/critical OS advisories, including OpenSSH
   CVE-2026-60002. Updated Python dependency results do not resolve OS findings.
   Rebuild and scan final artifacts; fix findings or obtain narrowly scoped,
   owned, expiring exceptions through the documented process. The exception
   manifest is empty; no publication gate has been weakened.
2. **Independent review remains required** for the outstanding scopes above
   before merge is requested. The review service could not complete those
   scopes; partial results are not accepted as approval.
3. **Follow-up source work remains:** the runtime OSV adapter still assigns a
   fixed medium severity rather than deriving advisory severity. The wider MCP
   sandbox backend also needs a separate review of platform capability and
   resource/privilege enforcement. This batch does not claim either is resolved.
4. **Refresh artifact evidence** after the final commit. Existing image scans
   predate the latest source and package version changes.

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
