# Security remediation: 0.9.4 candidate

Status: #1058, #1060 and #1063 are merged on develop. The September 14 closeout adds
review-discovered fixes and refreshed scan evidence; **0.9.4 is not released**.
Latest verified AI GitHub release is 0.9.3. SDK 0.9.2 is not yet available on PyPI.

## Integrated security changes

The original batch replaced persisted Python-object caches with bounded data
formats, validated SQL/parser/editor inputs, preserved runtime policy failures,
made temporary overrides transactional, and corrected advisory scoring/pagination.
It removed unused dependencies, refreshed Python/JavaScript/Rust versions, made
optional RAG backends lazy and consolidated four container targets.
See [dependency/deployment guidance](dependencies.md) and
[security policy](https://github.com/anvai-labs/victor/blob/develop/SECURITY.md).

The closeout's independent review found an alternate MCP factory that retained
resource configuration outside the canonical reconnect path. The factory now
uses the configured MCPClient lifecycle for first connection, reconnection and
failed-initialization cleanup.

Native chunking could panic on Unicode or loop indefinitely on invalid parameters
and some valid overlapping inputs. Native and Python implementations now validate
parameters, advance monotonically and use consistent character offsets.
Cross-method offset checks, randomized Unicode comparisons and subprocess
deadlines exercise these failures against a rebuilt native 0.8.1 extension.

Semgrep now retains raw JSON/SARIF, logs and exit status from one scan. Operational
failures, missing/malformed reports and mismatched diagnostic inventories fail the
job. Findings remain advisory. PartialParsing warnings are explicit coverage gaps,
not silently interpreted as a complete parse. Only justified in-source findings
are filtered from the GitHub upload; the raw evidence remains available.

The [hash disposition audit](security-hash-dispositions.md) classifies 52 MD5/SHA-1
calls as non-security cache/deduplication/identifier uses. Explicit constructor
annotations preserve existing digests; B324 is removed from both the global
configuration and workflow skip arguments so future security misuse is visible.

Routine dependency PRs have a seven-day cooldown. Urgent security fixes continue
through separate triage. The [CI batching rule](PR_WORKFLOW.md#minimize-ci-cycles)
requires local validation and independent review before pushing, with no
unconditional commit/push after failed tests.

## September 15 runner portability follow-up

#1063 passed all 32 PR checks and merged at `35621ad08`. The automatic develop
run then exposed two environment assumptions on self-hosted overflow runners:
the security action invoked an undeclared Python interpreter, and test collection
loaded LanceDB's native extension during vector-provider registration, causing
an illegal-instruction crash on that runner.

The follow-up declares Python setup inside the shared security action and defers
the LanceDB import until provider initialization, before opening model resources.
Registry and search imports no longer execute that optional backend. Provider
unit tests mock the backend without importing its native extension. Import
deferral does not establish that the LanceDB binary supports the runner's CPU;
actual backend execution still requires a compatible build and host.

Independent review approved the changes. The affected suites passed 407 tests,
and full collection found 32,476 tests. The fresh-process regression reproduces
the eager native import on unchanged develop and passes with the deferred import.

The September 14 container digests below remain evidence for that audited tree;
release artifacts must be rebuilt and scanned from the final promoted commit.

## September 14 alert reconciliation

Baseline develop: `71a6f466d`; main: `6b6cbcc4c`.
GitHub reports **73 Dependabot alerts** (2 critical, 29 high, 36 medium, 6 low),
**81 code-scanning alerts**, and **zero open secret-scanning alerts**.
All 81 code-scanning instances refer to main. Counts must be refreshed after
promotion; source remediation does not itself close main's alerts.

| Findings | Current source evidence | Remaining verification |
| --- | --- | --- |
| 59 npm alerts | Locked versions match none of the reported vulnerable ranges; both fresh npm audits report zero vulnerabilities | Verify packaged extension delivery and main graph refresh |
| 5 Black / 3 PyO3 / 1 Pygments alerts | Patched manifest/lock versions on develop | Final installed environments and package publication |
| 4 core Python alerts | Affected packages removed from core; embeddings separately updated | All enabled deployment environments |
| 1 Trivy Action alert | Replaced by a pinned composite scan configuration using Trivy Action v0.35.0 and binary v0.74.0 | Historical exposure review below |
| 70 mutable Action findings | References replaced by commit/digest pins | Main Semgrep refresh |
| 4 SQL findings | Purge values bound; backup identifier accepts only an ASCII timestamp, with negative tests | Two safe identifier heuristics have narrow inline justification |
| 4 dynamic import findings | Current scan no longer reports these sites | Preserve current scan evidence |
| 3 Trivy code findings | Fresh develop filesystem report has zero findings | Final artifacts and main scanner refresh |

The final local Semgrep scan reports **1,078 files, zero unsuppressed findings,
34 in-source suppressions and 56 PartialParsing warnings**. These warnings are
primarily embedded shell/GitHub expression parsing and remain a documented limit.

## Validation and release conditions

Independent review approves the native, MCP and scanner-completeness changes.
The rebuilt native suite passed **333 tests, 11 skipped**; independent checks
covered **14,000 randomized Unicode cases** and **45 focused native/MCP cases**.
Semgrep report validation passed **33 tests**, including injected operational
errors. Targeted formatting, production Ruff and MyPy passed. Full collection
found **32,472 tests without errors** outside sandbox permission restrictions.
The complete affected security/gate suite passed **615 tests, 5 skipped**.
The full MCP client suite passed **58 tests**, and SDK tests passed **337 tests**
after correcting two stale assertions about the already-migrated executor class.
The hash-affected suites passed **865 tests**, with one runtime-intelligence
integration timeout also reproduced against the unchanged baseline module in
an isolated process. Six hash-policy tests guard against reintroducing a global
B324 exclusion. The timeout is a validation limitation, not a passing test.

Fresh pip-audit checks of the complete pinned core, API and CPU-embedding
inventories report zero known vulnerabilities. Both npm audits and the complete
filesystem Trivy scan report zero findings. These inventory checks are not
substitutes for installed-environment or final-container audits.

A freshly resolved compatible extras/all-vertical environment contains **351
installed packages**, passes `pip check` and a fresh OSV audit with **zero findings
and zero skipped packages**. The standalone installed docs environment similarly
audits **39 packages with zero findings or skips**. Strict docs build and local
site links pass. The compatibility selection excludes aggregate aliases and
separately built native/platform-specific shapes; container/native checks cover
the Linux distribution shapes. It does not certify macOS/GPU installations.

The optional investment package is external to this tree: a fresh public-index
resolution of `victor-invest>=0.5.0,<1.0.0` finds no distribution. Therefore the
`invest` and aggregate `verticals` extras cannot currently be certified as public
installations; use the individual available vertical packages. No security claim
is made for an unavailable/private investment distribution.

All four rebuilt targets pass offline runtime checks and the publication scan
gate: **zero Python findings and zero high/critical/unknown findings**.
Core/MCP/native retain **45 medium and 4 low OS findings** each; full retains
**55 medium and 4 low**. None has a vendor fix in the scanned database. These
advisories remain visible; no threshold or exception was changed to erase them.

| Rebuilt target | Local image content digest |
| --- | --- |
| core | `c2623be71dea0ca71bf70b32f0370bcd16b7ebefbf1cd709b1238ad2400494f8` |
| mcp | `116d2947c769f634f99af419a0c1b5bffdd8745d03f3c79946652d4ceb253cd1` |
| native | `0dd35b5a874de3785e2cec5757392f009744d30378f2a51566354a0617dc25a9` |
| full | `ddb3bb94c69553338f34de88a662a9361e946fb47ba4e68a70d23e2e3a4f28a0` |

Inventories and SBOMs accompany each report. These are local image IDs, not
registry manifest digests. Native checks exercise the fixed Unicode/progress
cases in the release wheel; full checks load the bundled embedding model offline.

## Historical Trivy exposure evidence

[GHSA-69fq-xp46-6x23](https://github.com/advisories/GHSA-69fq-xp46-6x23) concerns
compromised Trivy artifacts and Action references. Historical Victor workflows
referenced `aquasecurity/trivy-action@master` during the March 19–23 incident
period. Run 23309396211's security job 67792322004 started on March 19 at
18:04:38 UTC and failed. Its log now returns HTTP 410, preventing verification of
the resolved Action revision or whether malicious code executed.

This is an exposure uncertainty, not a confirmed compromise. Credential rotation
or incident-review evidence is still needed to close it. A current patched
dependency and zero open secret alerts cannot prove that historical credentials
were never exposed. Do not publish secret material in this record.

## Release sequence

Keep the prepared versions: AI **0.9.4**, contracts **0.9.2**, native **0.8.1**,
extension **0.5.1**. This closeout fixes defects; it adds no new feature scope.
Coding native retains its independent **0.1.0** identity.

1. Complete final local artifact checks and consolidate all residual fixes into
   one develop PR. Require review and CI Success on the final candidate.
2. Resolve the historical-risk evidence above and any unaccepted release blocker.
3. Publish contracts 0.9.2 via its independent release workflow and verify PyPI.
4. Open the explicit develop-to-main promotion; pass the full required battery,
   then merge with a merge commit.
5. Tag the validated main commit for AI 0.9.4 and verify Python/native/container
   publication. Deliver changed vertical and extension packages separately.
6. Refresh main's dependency graph and scanner categories, verify alert closure,
   and audit the downloaded artifacts. No bulk dismissal of stale alerts.

Do not count version metadata, an old green commit, or a queued publication as a
release. Keep raw reports, package inventories and image digests with the final
release evidence.
