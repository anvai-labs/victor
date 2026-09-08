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
| Native extension build | [Native build recipe](development/setup.md#native-extension-build) |
| Documentation build and preview | [Documentation build](development/setup.md#documentation-build) |
| Branches, verification and pull requests | [PR Workflow](development/PR_WORKFLOW.md) |
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

Use Markdown and Mermaid for diagrams. Clearly label proposed architecture as a target until
its implementation lands. Keep file names, links, dependency requirements and execution claims
consistent with code, and update the owning guide when behavior changes.

`site/` is generated output: do not hand-edit or commit it. Use the canonical
[documentation build instructions](development/setup.md#documentation-build) for previewing changes.
