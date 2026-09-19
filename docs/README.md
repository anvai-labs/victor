# Victor Documentation

Start with the [documentation map](index.md). This repository separates current user and
contributor guidance from dated experiments, design proposals and review evidence.

## Canonical guides

| Topic | Canonical document |
| --- | --- |
| System ownership and runtime behavior | [Architecture](architecture.md) |
| Technology choices and dependency requirements | [Tech stack](tech-stack.md) |
| Current execution plan and complete debt register | [Roadmap](roadmap.md) |
| Feature catalog | [Features](features.md) |
| Development environment | [Development Setup](development/setup.md) |
| Deployment dependencies and container targets | [Dependency maintenance](development/dependencies.md) |
| Security release evidence and retained findings | [0.9.4 release record](development/security-remediation-0.9.4.md) |
| Native extension build | [Native build recipe](development/setup.md#native-extension-build) |
| Native performance decisions | [Native acceleration strategy](architecture/native-acceleration-strategy.md) |
| Documentation build and preview | [Documentation build](development/setup.md#documentation-build) |
| Branches, verification and pull requests | [PR Workflow](development/PR_WORKFLOW.md) |
| Self-hosted runner preparation | [Runner environments](development/self-hosted-runners.md) |
| Architecture decisions | [ADR index](architecture/adr/README.md) |
| Enhancement proposals | [Repository FEP index](https://github.com/anvai-labs/victor/blob/develop/feps/README.md) |
| Proposal process | [FEP Process](FEP_PROCESS.md) |

The canonical proposal files live in repository-root `feps/`. Older `docs/feps/` pages are
historical pointers. A merged proposal can still be Draft; implementation status is stated
in the proposal and roadmap rather than inferred from the existence of a document.

## Historical records

The [September co-design review](reviews/2026-09-03-codesign/README.md) retains the original
findings and an execution ledger. Historical experiment reports preserve their measured
PASS, HOLD and NO-GO outcomes. Follow their status banners before using an old command or
file reference; removed research apparatus is recoverable from git history.

The [September documentation audit](development/docs-audit-2026-09.md) records consolidation
and archival decisions. Current recipes belong in their canonical guide; duplicate pages
point there instead of maintaining a second copy.

## Authoring conventions

Write for scanning first, then deeper reading:

| Order | Content |
| --- | --- |
| 1 | One summary or status admonition |
| 2 | One diagram, table or short checklist |
| 3 | Focused implementation details |
| 4 | Evidence and canonical links |

- Prefer Mermaid, tables and short bullets over repeated prose.
- Give each diagram one canonical home; link to it elsewhere.
- Label architecture **Current**, **Target** or **Historical**.
- Keep file names, links, versions and execution claims consistent with source.
- Update the owning guide in the same change as behavior.

`site/` is generated output: do not hand-edit or commit it. Use the canonical
[documentation build instructions](development/setup.md#documentation-build) for previewing changes.
