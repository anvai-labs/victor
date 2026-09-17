# Documentation Audit — September 2026

This records the pre-consolidation documentation audit of 320 tracked files, including 260 Markdown files,
against the ADR-030 adapter worktree. It is a historical inventory, not a claim that every page
has received a complete semantic review. Links and selected status/index issues were repaired in
the consolidation PR after this baseline was recorded.

## Scope and method

- Enumerated all tracked `docs/**` and `feps/**`, plus root `README.md`, `SUPPORT.md` and
  `CONTRIBUTING.md`. The FEP process document is `docs/FEP_PROCESS.md`; no root copy exists.
- Scanned every file for local Markdown links, literal repository paths and diagram markers.
- Read targeted prose/excerpts for the handoff issues, indexes, duplicate topics and build setup.
- **CURRENT** means no verified defect was established by that review depth; an automated-only
  row is provisional and does not certify its APIs or claims against runtime behavior.
- **STALE-but-keep** includes useful pages needing repairs and intentionally historical records.
  Historical naming heuristics are marked explicitly; dates alone do not prove obsolete content.
- **DEAD** requires evidence justifying deletion. No document met that threshold.
- Preserve `docs/reviews/**` unchanged. Its findings describe the recorded co-design snapshot.

Baseline classification: **237 CURRENT**, **83 STALE-but-keep**, **0 DEAD**.

## Findings and disposition

- The baseline found 95 absent local Markdown destinations, seven site-root absolute Markdown
  paths, and ten heuristic anchor candidates. Five intentional template links were excluded.
  The consolidation repairs destinations to current docs/source or verified immutable history.
- All 53 explicit MkDocs navigation paths existed. 169 docs Markdown files were not listed in
  nav; omission from nav does not exclude a page from the build.
- The existing MkDocs Material site and Pages workflow are retained. Pipeline/renderer validation
  is separate from this source audit; the audit did not fetch external websites or Pages settings.
- A subsequent local MkDocs build verified the repaired page links and rendered heading anchors:
  no missing documentation destinations, unresolved relative paths, site-root source links, or
  missing anchors remain in its diagnostics. Repository-source links outside `docs/` now use GitHub
  URLs; all 27 distinct new destinations exist in the fetched `origin/develop` tree. Existing
  immutable historical URLs and the review documents were preserved. This validates repository
  destinations, not external HTTP availability.
- The build still reports the existing `docs/README.md`/`docs/index.md` output-name collision and
  revision-date fallback warnings for the two new, uncommitted audit/history pages. Navigation
  coverage and the README build exclusion belong to the Pages configuration follow-up.
- Root `feps/README.md` now indexes actual frontmatter statuses. Legacy `docs/feps` 0001/0002
  documents retain their identities with historical-series banners. FEP-0031–0033 remain Draft.
- ADR-030/031 index duplication is removed. ADR-030 records the compiled adapter in #1042 and
  the completed streaming migration/BFS deletion in #1043. Existing checkpoints do not complete FEP-0032.
- The actual streaming runtime class is **ServiceStreamingRuntime**. The earlier handoff label
  `ChatStreamRuntime` was incorrect; the canonical method path uses `run_unified` and
  `AgenticLoop.run_streaming`.
- Judge experiment documents retain their evidence. Their apparatus was removed by #1019
  (commit dated 2026-09-05); the September 6 status note links the immediate pre-deletion tree.
- The misplaced workflow consolidation plan is [preserved as history](../architecture/workflow-consolidation-plan-historical.md);
  its [former examples URL](../guides/workflow-development/examples.md) now points to maintained examples.
- Native build instructions belong in [developer setup](setup.md); root architecture/tech-stack
  should link there. Sphinx remains a separately documented build and is not declared dead.
- The embedding development guide and reference were byte-identical at baseline; consolidate
  them through the canonical reference while retaining the former URL.

- Follow-up semantic review found obsolete constructor/chat/streaming examples in the duplicate
  first-agent guide and hand-written Python catalog. These now point to the maintained tutorial
  and generated signature reference; the Python entry-point guide explains current result and
  session types. The superseded content remains recoverable from repository history.
- Optional browser, embedding, Docker and LangChain installation recipes moved from README
  into developer setup, including Playwright's separate browser install and the VS Code build.

## Pre-release refinement for 0.9.2

A second source-grounded pass reviewed the architecture/proposal group and the
user-facing guides, references and tutorials before promotion. It corrected
known stale active claims and replaced duplicate catalogs with maintained entry
points. Historical design and experiment records retain their original evidence
with status banners; the co-design review files remain unchanged.

The pass includes runnable README examples, current workflow/YAML and team
interfaces, contract-first vertical examples, event/metrics documentation, and
contribution instructions that install from the task worktree. Superseded guide
content remains available through linked repository history. Canonical links
are checked against a built site, and selected deterministic examples are run;
provider-dependent examples are source-checked without live API calls.

The inventory below remains the original baseline, including its provisional
classifications. It is not a claim that every example was executed or that every
external service integration was independently certified. Subsequent PR diffs
record the exact refinements after that inventory.

## Complete baseline inventory

“Automated” is a directory-purpose assumption, “Targeted” is excerpt review, “Frontmatter”
verifies proposal metadata only, and “Historical policy” preserves the user-requested record.
“Updated” means the file was touched by this consolidation when the inventory was written;
it does not expand the stated semantic review depth. Reasons describe the baseline.

| File | Baseline classification | Review basis | Baseline reason | D1 disposition |
|---|---|---|---|---|
| `CONTRIBUTING.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `README.md` | STALE-but-keep | Targeted | Embeds version-labelled Victor0.7 architecture diagram at line104 and primarily links repository docs rather than Pages. | Retain; follow-up review where provisional |
| `SUPPORT.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/.gitignore` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/FEP_PROCESS.md` | STALE-but-keep | Targeted | Process definitions overlap root feps/README | Retain; follow-up review where provisional |
| `docs/Makefile` | CURRENT | Targeted | Explicitly supported secondary Sphinx API-reference build | Retain; follow-up review where provisional |
| `docs/README.md` | STALE-but-keep | Targeted | Documentation index exists | Retain; follow-up review where provisional |
| `docs/analysis/2026-07-25-prompt-evolution-audit.md` | STALE-but-keep | Automated | Dated design/research/migration artifact inferred from path/title | Retain; follow-up review where provisional |
| `docs/analysis/2026-07-27-fep-0025-checkpoint.md` | STALE-but-keep | Automated | Dated design/research/migration artifact inferred from path/title | Retain; follow-up review where provisional |
| `docs/analysis/2026-07-28-fep-0025-n60-outcome.md` | STALE-but-keep | Automated | Dated design/research/migration artifact inferred from path/title | Retain; follow-up review where provisional |
| `docs/api-reference/auto-api.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/api-reference/protocols.md` | STALE-but-keep | Automated | Contains 4 local Markdown targets absent from the checkout | Updated |
| `docs/api-reference/providers.md` | CURRENT | Automated | Retained active documentation or supporting asset | Updated |
| `docs/api-reference/tools.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/api-reference/wire-events.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/api-reference/workflows.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/architecture.md` | STALE-but-keep | Targeted | Lines 458 and 471 assert StateGraph is always the engine | Retain; follow-up review where provisional |
| `docs/architecture/BLUEPRINT.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/architecture/CODEBASE_REVIEW_BACKLOG.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/architecture/CONTRACTS_BOUNDARY.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/architecture/EXTERNAL_VERTICALS_MIGRATION.md` | STALE-but-keep | Targeted | Already explicitly superseded by ADR007 | Retain; follow-up review where provisional |
| `docs/architecture/adr/000-template.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/architecture/adr/001-agent-orchestration.md` | STALE-but-keep | Automated | Contains 1 local Markdown targets absent from the checkout | Updated |
| `docs/architecture/adr/002-state-management.md` | STALE-but-keep | Automated | Contains 1 local Markdown targets absent from the checkout | Updated |
| `docs/architecture/adr/003-workflow-engine.md` | STALE-but-keep | Automated | Contains 1 local Markdown targets absent from the checkout | Updated |
| `docs/architecture/adr/004-tool-system.md` | STALE-but-keep | Automated | Contains 1 local Markdown targets absent from the checkout | Updated |
| `docs/architecture/adr/005-event-system.md` | STALE-but-keep | Automated | Contains 1 local Markdown targets absent from the checkout | Updated |
| `docs/architecture/adr/006-provider-integration-improvements.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/architecture/adr/007-vertical-distribution-and-sdk-boundary.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/architecture/adr/008-registry-performance-optimization.md` | STALE-but-keep | Automated | Contains 3 local Markdown targets absent from the checkout | Updated |
| `docs/architecture/adr/009-rubric-based-completion-evaluation.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/architecture/adr/010-effect-grounded-completion.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/architecture/adr/011-llm-judge-reliability-gating.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/architecture/adr/012-regression-gated-harness-acceptance.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/architecture/adr/013-unified-temperature-policy.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/architecture/adr/014-shared-codegraph-chunker-package.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/architecture/adr/015-victor-core-adopts-codegraph.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/architecture/adr/016-distribution-packaging-strategy.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/architecture/adr/017-rl-budget-calibration.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/architecture/adr/018-adopt-sandhi-usage-gateway.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/architecture/adr/019-orchestrator-service-runtime-decomposition.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/architecture/adr/020-interactive-terminal-tui.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/architecture/adr/021-terminal-native-hitl-and-loop-transparency.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/architecture/adr/022-provider-gateway-feature-layer.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/architecture/adr/023-multi-agent-team-durability.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/architecture/adr/024-abstraction-canonicalization-and-import-guard.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/architecture/adr/025-ratify-evaluation-centric-p0-decisions.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/architecture/adr/026-durable-code-memory-ga.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/architecture/adr/027-prompt-optimization-strategy-fidelity.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/architecture/adr/028-single-agent-durable-chat-continuation.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/architecture/adr/029-provider-support-tiers.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/architecture/adr/030-single-graph-execution-engine.md` | CURRENT | Automated | Retained active documentation or supporting asset | Updated |
| `docs/architecture/adr/031-vertical-template-bases-promotion.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/architecture/adr/README.md` | STALE-but-keep | Targeted | ADR030 and ADR031 each occur twice | Updated |
| `docs/architecture/bayesian.md` | STALE-but-keep | Automated | Contains 3 local Markdown targets absent from the checkout | Updated |
| `docs/architecture/codegraph-v2-design.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/architecture/competitive-benchmark-2026-07.md` | STALE-but-keep | Automated | Dated design/research/migration artifact inferred from path/title | Retain; follow-up review where provisional |
| `docs/architecture/data-flow-eventbus.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/architecture/debugging-profiling-guide.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/architecture/edge-provider-tool-strategy.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/architecture/evaluation-centric-runtime-backlog.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/architecture/evr2-human-validation-protocol.md` | STALE-but-keep | Targeted | Links deleted benchmarks/judge_calibration/FINDINGS.md and labels | Updated |
| `docs/architecture/evr3-parity-results.md` | STALE-but-keep | Targeted | References deleted benchmarks/judge_calibration/labels/run12 and FINDINGS.md without removal note. | Updated |
| `docs/architecture/evr6-auditor-gate.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/architecture/evr6-auditor-results-2026-09-04.md` | STALE-but-keep | Automated | Dated design/research/migration artifact inferred from path/title | Updated |
| `docs/architecture/extension_manifest.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/architecture/feature-flags.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/architecture/flag-graduation-policy.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/architecture/foundations-strategy-2026-07.md` | STALE-but-keep | Automated | Dated design/research/migration artifact inferred from path/title | Retain; follow-up review where provisional |
| `docs/architecture/framework-vertical-integration.md` | STALE-but-keep | Automated | Contains 17 local Markdown targets absent from the checkout | Updated |
| `docs/architecture/graph-api-reference.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/architecture/graph-architecture-alignment.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/architecture/graph-enhancements-spec.md` | STALE-but-keep | Targeted | Explicit Design Phase status dated2025-04-28 | Retain; follow-up review where provisional |
| `docs/architecture/graph-extension-guide.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/architecture/graph-migration-guide.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/architecture/graph-quickstart.md` | STALE-but-keep | Automated | Contains 1 local Markdown targets absent from the checkout | Updated |
| `docs/architecture/graph-rag-guide.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/architecture/graph-research-summary.md` | STALE-but-keep | Targeted | Research survey of external papers and proposed Victor applicability. | Retain; follow-up review where provisional |
| `docs/architecture/graph-tool-enhancements.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/architecture/judge-independence-experiments.md` | STALE-but-keep | Targeted | References deleted benchmarks/judge_calibration labels/reports/FINDINGS without removal note. | Updated |
| `docs/architecture/migration.md` | STALE-but-keep | Targeted | Already historical but names deleted docs/architecture/CURRENT_STATE.md as authoritative current runtime doc. | Updated |
| `docs/architecture/observability-axes.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/architecture/orchestrator_decomposition.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/architecture/phase6-packaging-plan.md` | STALE-but-keep | Targeted | Unbannered packaging plan presents historic package/LOC/dependency counts as current. | Updated |
| `docs/architecture/provider-agnostic-tiers.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/architecture/proximadb-codegraph-backend.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/architecture/sandhi-typed-integration-gap-analysis.md` | STALE-but-keep | Automated | Dated design/research/migration artifact inferred from path/title | Retain; follow-up review where provisional |
| `docs/architecture/smart_routing.md` | STALE-but-keep | Automated | Contains 4 local Markdown targets absent from the checkout | Updated |
| `docs/architecture/state-machine.md` | STALE-but-keep | Automated | Contains 3 local Markdown targets absent from the checkout | Updated |
| `docs/architecture/state-passed-architecture.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/architecture/streaming-pipeline.md` | STALE-but-keep | Targeted | Calls StreamingChatExecutor.run the canonical entry point | Retain; follow-up review where provisional |
| `docs/architecture/tool-output-condensation.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/architecture/unified_prompt_architecture.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/architecture/ux-redesign-plan.md` | STALE-but-keep | Automated | Dated design/research/migration artifact inferred from path/title | Retain; follow-up review where provisional |
| `docs/architecture/vertical-dependency-resolution.md` | STALE-but-keep | Automated | Contains 3 local Markdown targets absent from the checkout | Updated |
| `docs/architecture/vision-evaluation-centric-runtime.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/architecture/w3d-codec-purity-addendum.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/assets/.gitkeep` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/assets/diagrams/.gitkeep` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/assets/images/.gitkeep` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/assets/images/victor-logo-icon.svg` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/assets/screenshots/.gitkeep` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/assets/victor-banner.svg` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/conf.py` | CURRENT | Targeted | Explicitly supported secondary Sphinx API-reference build | Retain; follow-up review where provisional |
| `docs/development/PR_WORKFLOW.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/development/code-style.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/development/deprecation-inventory-2026-03-03.md` | STALE-but-keep | Automated | Dated design/research/migration artifact inferred from path/title | Retain; follow-up review where provisional |
| `docs/development/deprecation-policy.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/development/exception-handling.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/development/extending/plugins.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/development/extending/verticals.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/development/fep-template.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/development/index.md` | STALE-but-keep | Automated | Contains 2 local Markdown targets absent from the checkout | Updated |
| `docs/development/prompt-evolution-workflow.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/development/releasing/publishing.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/development/setup.md` | STALE-but-keep | Targeted | Developer setup remains the suitable canonical home, but native maturin build is currently scattered across architecture/tech-stack. | Retain; follow-up review where provisional |
| `docs/development/testing.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/development/testing/all-verticals-migration-plan.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/development/testing/category-5-analysis.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/development/testing/strategy.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/development/testing/vertical-migration-summary-2026-03-04.md` | STALE-but-keep | Automated | Dated design/research/migration artifact inferred from path/title | Retain; follow-up review where provisional |
| `docs/diagrams/README.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/diagrams/architecture/config-system.mmd` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/diagrams/architecture/config-system.svg` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/diagrams/architecture/multi-agent.mmd` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/diagrams/architecture/multi-agent.svg` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/diagrams/architecture/provider-system.mmd` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/diagrams/architecture/provider-system.svg` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/diagrams/architecture/system-overview.mmd` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/diagrams/architecture/system-overview.svg` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/diagrams/architecture/victor_0_7_architecture.mmd` | STALE-but-keep | Automated | Version-labelled0.7 diagram asset referenced by current README | Retain; follow-up review where provisional |
| `docs/diagrams/architecture/victor_0_7_architecture.svg` | STALE-but-keep | Automated | Version-labelled0.7 diagram asset referenced by current README | Retain; follow-up review where provisional |
| `docs/diagrams/architecture/victor_0_7_readme_architecture.svg` | STALE-but-keep | Automated | Version-labelled0.7 diagram asset referenced by current README | Retain; follow-up review where provisional |
| `docs/diagrams/sequences/provider-switch.mmd` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/diagrams/sequences/provider-switch.svg` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/diagrams/sequences/tool-execution.mmd` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/diagrams/sequences/tool-execution.svg` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/diagrams/sequences/workflow-execution.mmd` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/diagrams/sequences/workflow-execution.svg` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/diagrams/victor-guide/d01_landscape.mmd` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/diagrams/victor-guide/d01_landscape.svg` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/diagrams/victor-guide/d02_architecture.mmd` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/diagrams/victor-guide/d02_architecture.svg` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/diagrams/victor-guide/d03_api_levels.mmd` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/diagrams/victor-guide/d03_api_levels.svg` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/diagrams/victor-guide/d04_agentic_loop.mmd` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/diagrams/victor-guide/d04_agentic_loop.svg` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/diagrams/victor-guide/d05_tools.mmd` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/diagrams/victor-guide/d05_tools.svg` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/diagrams/victor-guide/d06_code_intel.mmd` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/diagrams/victor-guide/d06_code_intel.svg` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/diagrams/victor-guide/d07_graph_rag.mmd` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/diagrams/victor-guide/d07_graph_rag.svg` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/diagrams/victor-guide/d08_two_db.mmd` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/diagrams/victor-guide/d08_two_db.svg` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/diagrams/victor-guide/d09_teams.mmd` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/diagrams/victor-guide/d09_teams.svg` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/diagrams/victor-guide/d10_eval.mmd` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/diagrams/victor-guide/d10_eval.svg` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/diagrams/victor-guide/d11_prompt_opt.mmd` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/diagrams/victor-guide/d11_prompt_opt.svg` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/diagrams/victor-guide/d12_lifecycle.mmd` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/diagrams/victor-guide/d12_lifecycle.svg` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/features.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/feps/fep-0001-edge-model.md` | STALE-but-keep | Targeted | Legacy docs/feps numbering collides with root feps/fep-0001-fep-process.md | Updated |
| `docs/feps/fep-0002-rl-budget-calibration.md` | STALE-but-keep | Targeted | Legacy docs/feps numbering collides with root FEP0002 documentation style | Updated |
| `docs/feps/vertical-package-spec.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/getting-started/basic-usage.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/getting-started/configuration.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/getting-started/first-agent.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/getting-started/first-run.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/getting-started/index.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/getting-started/installation.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/getting-started/quickstart.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/getting-started/web-chat-ui.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/guides/BENCHMARKING.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/guides/CODEBASE_VERIFICATION.md` | STALE-but-keep | Automated | Contains 2 local Markdown targets absent from the checkout | Updated |
| `docs/guides/EDGE_MODEL.md` | STALE-but-keep | Automated | Contains 1 local Markdown targets absent from the checkout | Updated |
| `docs/guides/FEATURE_FLAGS.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/guides/FRAMEWORK_CAPABILITIES.md` | STALE-but-keep | Automated | Contains 2 local Markdown targets absent from the checkout | Updated |
| `docs/guides/HITL_WORKFLOWS.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/guides/MCP_INTEGRATION.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/guides/MULTI_AGENT_TEAMS.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/guides/OBSERVABILITY.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/guides/RESILIENCE.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/guides/TASK_PLANNER.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/guides/TROUBLESHOOTING.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/guides/VICTOR_AS_MCP_SERVER.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/guides/WORKFLOW_SCHEDULER.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/guides/development/AIRGAPPED.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/guides/development/ARCHITECTURE.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/guides/development/TOOL_CALLING_FORMATS.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/guides/development/TOOL_SELECTION.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/guides/development/embeddings.md` | STALE-but-keep | Targeted | Byte-identical to docs/reference/embeddings.md | Retain; follow-up review where provisional |
| `docs/guides/development/local-models.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/guides/dynamic_import_patterns.md` | STALE-but-keep | Automated | Contains 2 local Markdown targets absent from the checkout | Updated |
| `docs/guides/first-agent.md` | STALE-but-keep | Targeted | Parallel first-agent tutorial exists at getting-started/first-agent.md. | Updated |
| `docs/guides/first-workflow.md` | STALE-but-keep | Automated | Contains 1 local Markdown targets absent from the checkout | Updated |
| `docs/guides/index.md` | STALE-but-keep | Automated | Contains 6 local Markdown targets absent from the checkout | Updated |
| `docs/guides/installation.md` | STALE-but-keep | Targeted | Minimum requirements say Python3.10 despite pyproject.toml requiring >=3.11 | Updated |
| `docs/guides/integration/mcp-clients.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/guides/multi-agent-quickstart.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/guides/multi-provider-benchmarks.md` | STALE-but-keep | Automated | Contains 1 local Markdown targets absent from the checkout | Updated |
| `docs/guides/observability/event-bus.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/guides/observability/index.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/guides/observability/metrics.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/guides/quickstart.md` | STALE-but-keep | Targeted | Parallel quickstart exists at getting-started/quickstart.md | Retain; follow-up review where provisional |
| `docs/guides/task_completion.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/guides/tool-reference.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/guides/vertical-quickstart.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/guides/workflow-development/dsl.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/guides/workflow-development/examples.md` | STALE-but-keep | Targeted | Title is Workflow Architecture Consolidation Plan | Updated |
| `docs/guides/workflow-development/scheduling.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/guides/workflow-quickstart.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/index.md` | STALE-but-keep | Targeted | System diagram labels providers 24 while nearby prose says 25, and embeds a pre-Stage-C service/runtime layout. | Retain; follow-up review where provisional |
| `docs/javascripts/mermaid-init.js` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/make.bat` | CURRENT | Targeted | Explicitly supported secondary Sphinx API-reference build | Retain; follow-up review where provisional |
| `docs/quickstart-proximadb-memory.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/reference/api/http-api.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/reference/api/index.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/reference/api/mcp-server.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/reference/api/python-api.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/reference/cli-commands.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/reference/configuration-options.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/reference/configuration/index.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/reference/configuration/keys.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/reference/embeddings.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/reference/environment-variables.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/reference/index.md` | STALE-but-keep | Automated | Contains 1 local Markdown targets absent from the checkout | Updated |
| `docs/reference/provider-caching.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/reference/providers-comparison.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/reference/providers/comparison.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/reference/providers/index.md` | CURRENT | Automated | Retained active documentation or supporting asset | Updated |
| `docs/reference/providers/setup.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/reference/quick-reference.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/reference/settings-reference.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/reference/skills.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/reference/tools/catalog.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/reference/tools/migration.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/reference/tools/tool-calling.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/reference/verticals/index.md` | STALE-but-keep | Automated | Contains 1 local Markdown targets absent from the checkout | Updated |
| `docs/release-readiness-mvp.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/requirements.txt` | CURRENT | Targeted | Explicitly supported secondary Sphinx API-reference build | Retain; follow-up review where provisional |
| `docs/reviews/2026-09-03-codesign/README.md` | STALE-but-keep | Historical policy | Historical co-design review preserved verbatim by user instruction | Preserve unchanged |
| `docs/reviews/2026-09-03-codesign/U1-agentic-core.md` | STALE-but-keep | Historical policy | Historical co-design review preserved verbatim by user instruction | Preserve unchanged |
| `docs/reviews/2026-09-03-codesign/U10-evaluation-observability.md` | STALE-but-keep | Historical policy | Historical co-design review preserved verbatim by user instruction | Preserve unchanged |
| `docs/reviews/2026-09-03-codesign/U2-framework.md` | STALE-but-keep | Historical policy | Historical co-design review preserved verbatim by user instruction | Preserve unchanged |
| `docs/reviews/2026-09-03-codesign/U3-providers-config.md` | STALE-but-keep | Historical policy | Historical co-design review preserved verbatim by user instruction | Preserve unchanged |
| `docs/reviews/2026-09-03-codesign/U4-tools.md` | STALE-but-keep | Historical policy | Historical co-design review preserved verbatim by user instruction | Preserve unchanged |
| `docs/reviews/2026-09-03-codesign/U5-storage-core-codegraph.md` | STALE-but-keep | Historical policy | Historical co-design review preserved verbatim by user instruction | Preserve unchanged |
| `docs/reviews/2026-09-03-codesign/U6-workflows-teams.md` | STALE-but-keep | Historical policy | Historical co-design review preserved verbatim by user instruction | Preserve unchanged |
| `docs/reviews/2026-09-03-codesign/U7-client-surface.md` | STALE-but-keep | Historical policy | Historical co-design review preserved verbatim by user instruction | Preserve unchanged |
| `docs/reviews/2026-09-03-codesign/U8-context-processing-native.md` | STALE-but-keep | Historical policy | Historical co-design review preserved verbatim by user instruction | Preserve unchanged |
| `docs/reviews/2026-09-03-codesign/U9-contracts-verticals.md` | STALE-but-keep | Historical policy | Historical co-design review preserved verbatim by user instruction | Preserve unchanged |
| `docs/roadmap.md` | STALE-but-keep | Targeted | Now section is still v0.9.0 August2026 closeout while VERSION is0.9.1 and StageC is active. | Retain; follow-up review where provisional |
| `docs/tech-stack.md` | STALE-but-keep | Targeted | June2026 canonical stamp | Retain; follow-up review where provisional |
| `docs/tutorials/build-custom-tool.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/tutorials/create-workflow.md` | STALE-but-keep | Automated | Contains 6 local Markdown targets absent from the checkout | Updated |
| `docs/tutorials/integrate-provider.md` | STALE-but-keep | Automated | Contains 2 local Markdown targets absent from the checkout | Updated |
| `docs/tutorials/notebooks/00_introduction.ipynb` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/tutorials/notebooks/01_first_agent.ipynb` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/tutorials/notebooks/02_workflows.ipynb` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/tutorials/notebooks/03_tools.ipynb` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/tutorials/notebooks/04_verticals.ipynb` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/tutorials/notebooks/05_multi_agent.ipynb` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/tutorials/notebooks/06_production.ipynb` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/user-guide/cli-reference.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/user-guide/index.md` | STALE-but-keep | Automated | Contains 5 local Markdown targets absent from the checkout | Updated |
| `docs/user-guide/providers.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/user-guide/rich-formatting-guide.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/user-guide/session-management.md` | STALE-but-keep | Automated | Contains 1 local Markdown targets absent from the checkout | Updated |
| `docs/user-guide/tools.md` | STALE-but-keep | Automated | Contains 3 local Markdown targets absent from the checkout | Updated |
| `docs/user-guide/troubleshooting.md` | STALE-but-keep | Automated | Seven root-absolute links target Markdown source paths | Updated |
| `docs/user-guide/workflows.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/user-guide/yaml_workflow_examples.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/user-guide/yaml_workflow_syntax.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/ux-adoption-action-plan.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/verticals/api_reference.md` | STALE-but-keep | Automated | Contains 1 local Markdown targets absent from the checkout | Updated |
| `docs/verticals/architecture_refactoring.md` | STALE-but-keep | Automated | Contains 1 local Markdown targets absent from the checkout | Updated |
| `docs/verticals/best_practices.md` | STALE-but-keep | Automated | Contains 1 local Markdown targets absent from the checkout | Updated |
| `docs/verticals/coding.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/verticals/data-analysis.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/verticals/devops.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/verticals/rag.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `docs/verticals/research.md` | CURRENT | Automated | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `feps/README.md` | STALE-but-keep | Targeted | No numbered FEP status index exists, so FEP0007 Implemented and FEP0031-33 Draft are undiscoverable here | Updated |
| `feps/fep-0000-template.md` | CURRENT | Frontmatter | Canonical root FEP template | Retain; follow-up review where provisional |
| `feps/fep-0001-fep-process.md` | CURRENT | Frontmatter | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `feps/fep-0002-documentation-style-guide.md` | CURRENT | Frontmatter | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `feps/fep-0003-progressive-tool-loading.md` | STALE-but-keep | Frontmatter | Contains 2 local Markdown targets absent from the checkout | Updated |
| `feps/fep-0004-provider-oauth.md` | CURRENT | Frontmatter | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `feps/fep-0005-policy-engine.md` | CURRENT | Frontmatter | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `feps/fep-0006-external-harness-executors.md` | CURRENT | Frontmatter | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `feps/fep-0007-unified-agentic-loop.md` | CURRENT | Frontmatter | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `feps/fep-0008-evaluation-centric-completion.md` | CURRENT | Frontmatter | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `feps/fep-0009-sdk-tool-contract.md` | CURRENT | Frontmatter | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `feps/fep-0010-shared-protocol-crate.md` | CURRENT | Frontmatter | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `feps/fep-0011-provider-cache-cost-model.md` | CURRENT | Frontmatter | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `feps/fep-0012-shipped-edge-classifier.md` | CURRENT | Frontmatter | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `feps/fep-0013-shell-safety-policy.md` | CURRENT | Frontmatter | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `feps/fep-0014-canonical-validation-metrics-contracts.md` | CURRENT | Frontmatter | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `feps/fep-0015-trim-internal-framework-exports.md` | CURRENT | Frontmatter | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `feps/fep-0016-wire-initialization-phase-manager.md` | CURRENT | Frontmatter | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `feps/fep-0017-prompt-optimization-reward-loop.md` | CURRENT | Frontmatter | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `feps/fep-0018-framework-verification-hook.md` | CURRENT | Frontmatter | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `feps/fep-0019-lsp-integrated-verification.md` | CURRENT | Frontmatter | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `feps/fep-0020-ai-gateway-usage-attribution.md` | CURRENT | Frontmatter | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `feps/fep-0021-close-the-inner-loop.md` | CURRENT | Frontmatter | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `feps/fep-0022-measurement-driven-self-consistency.md` | CURRENT | Frontmatter | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `feps/fep-0023-context-management-subsystem-activation.md` | CURRENT | Frontmatter | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `feps/fep-0024-pluggable-code-correction.md` | CURRENT | Frontmatter | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `feps/fep-0025-prompt-evolution-as-controlled-experiment.md` | CURRENT | Frontmatter | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `feps/fep-0026-authenticated-control-plane-channel.md` | CURRENT | Frontmatter | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `feps/fep-0027-settle-stream-usage-on-abort.md` | CURRENT | Frontmatter | Retained active documentation or supporting asset | Updated |
| `feps/fep-0028-team-node-durability-contract.md` | CURRENT | Frontmatter | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `feps/fep-0029-single-agent-durable-chat-continuation.md` | CURRENT | Frontmatter | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `feps/fep-0030-decoupled-completion-judge.md` | CURRENT | Frontmatter | Retained active documentation or supporting asset | Updated |
| `feps/fep-0031-chat-runtime-inversion.md` | CURRENT | Frontmatter | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `feps/fep-0032-interrupt-resume-semantics.md` | CURRENT | Frontmatter | Retained active documentation or supporting asset | Retain; follow-up review where provisional |
| `feps/fep-0033-rl-subsystem-relocation.md` | CURRENT | Frontmatter | Retained active documentation or supporting asset | Retain; follow-up review where provisional |

The historical plan archive and this audit inventory were added by the consolidation and are
not part of the 320-file baseline. New source validation applies to the completed PR tree.
