# Documentation publishing

The public site is [Victor on GitHub Pages](https://anvai-labs.github.io/victor/).
It is built from `docs/` by the existing
[Deploy Documentation workflow](https://github.com/anvai-labs/victor/blob/develop/.github/workflows/docs.yml).
MkDocs Material supplies navigation, search, API rendering and Mermaid diagrams.

| Deployment fact | Current value |
| --- | --- |
| Last verified successful Pages source | `main` `353d5e8871444bdc5ce3ce7729b129b1c2b630e5`, 2026-11-15; Victor 0.29.2 promotion |
| Workflow run | [43729659516](https://github.com/anvai-labs/victor/actions/runs/43729659516) succeeded |
| Develop preview | Built in pull-request CI; published after the next main promotion |
| Reviewed emergency hotfix | `hotfix/docs-0.29.2-pages`, pending reviewed main merge and Pages deployment; the verified row above remains authoritative until live hotfix content is checked |

!!! tip "Page shape"

    Lead with the outcome. Follow with a diagram, table or checklist. Put focused
    details and evidence below it. Link to canonical material instead of repeating it.

| Label | Use for |
| --- | --- |
| **Current** | Behavior verified in the source tree |
| **Target** | Proposed behavior with an owning FEP or ADR |
| **Historical** | Dated evidence retained for provenance |

## Build and review

From the repository root, in a virtual environment:

```bash
pip install -r mkdocs-requirements.txt
mkdocs build --strict
python scripts/ci/docs_link_check.py site
mkdocs serve
```

The pinned documentation dependencies are separate from Victor's runtime
dependencies. API pages use static source analysis. Pygments 2.20 or later is
required for the patched ADL lexer. The pinned toolchain builds API signatures
successfully; validate those pages whenever updating highlighting dependencies.
The former Pygments 2.18 workaround is no longer needed.

Review the build diagnostics and browse the generated site before opening a PR
against `develop`. The workflow builds documentation for relevant PRs targeting
both `develop` and `main`, including changes to the source-backed API pages,
version stamp and build hooks.

## Navigation and historical records

`mkdocs.yml` owns the site navigation; `docs/index.md` is its home page.
`docs/README.md` remains the repository documentation index and is explicitly
excluded from the build to avoid competing for the same output URL.

The Audits navigation section exposes the historical co-design review, current
repository map and compact completed-work record. Superseded interim notes live
in immutable Git history, with unresolved tasks transferred to current owners. Root `feps/` proposals are
linked directly to the repository so there is one canonical proposal copy;
legacy documents under `docs/feps/` retain their historical-series banners.

## Diagrams

Use fenced `mermaid` blocks with a title and an introducing sentence. Material's
[native Mermaid integration](https://squidfunk.github.io/mkdocs-material/reference/diagrams/)
handles initialization, instant navigation and light/dark mode. A second CDN
script or custom initializer is unnecessary and can race the built-in renderer.
Validate changed diagrams with Mermaid CLI, then inspect the built page.

## Link and anchor checks

The built-site checker validates local HTML destinations, assets and rendered
anchor IDs. Absolute URLs beneath the configured Pages project URL are resolved
against the artifact as well. It makes no network requests, so it does not
verify external URLs or the availability of the published site. Browser-generated
diagram anchors are outside its scope.

The checker returns a nonzero exit code for missing destinations or anchors, and
the workflow treats both the strict build and this check as deployment
prerequisites. A successful workflow therefore includes a clean built-site link
and anchor audit.

## Deploy to Pages

A push to `main` affecting the documentation inputs builds and deploys the site.
The repository's `github-pages` environment permits the `main` branch only.
Documentation changes merge through `develop` and publish with the normal
`develop` → `main` promotion; this keeps the site aligned with promoted code.

To rebuild the promoted documentation explicitly:

```bash
gh workflow run docs.yml --ref main
```

Manual runs on other branches can build but cannot deploy; PR runs never deploy.
The deploy job uses the `github-pages` environment and serializes deployments
through the `pages` concurrency group. Check both the build and deployment jobs,
then verify the published URL and a changed page against the promoted commit.

### Emergency docs-only main hotfix

The normal path remains a documentation PR to `develop`, followed by a normal
`develop` → `main` promotion. A direct docs-only PR to `main` is reserved for a
demonstrable public-site defect that misstates the promoted release identity,
installation contract, or security posture. The hotfix must branch from current
`main`; contain only affected documentation, documentation gates, and gate
tests; include the correction on `develop` where possible; pass exact-head
adversarial review and applicable main CI; and make no runtime, package,
artifact, credential, or release-version change.

After merge, verify the live home, roadmap, and publishing pages plus the Pages
workflow run. Record the verified content hotfix separately from the prior
normal deployment above and back-sync that record to `develop`. Do not describe
the hotfix as a new Victor release.
