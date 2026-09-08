# Documentation publishing

The public site is [Victor on GitHub Pages](https://anvai-labs.github.io/victor/).
It is built from `docs/` by the existing
[Deploy Documentation workflow](https://github.com/anvai-labs/victor/blob/develop/.github/workflows/docs.yml).
MkDocs Material supplies navigation, search, API rendering and Mermaid diagrams.

## Build and review

From the repository root, in a virtual environment:

```bash
pip install -r mkdocs-requirements.txt
mkdocs build
python scripts/ci/docs_link_check.py site
mkdocs serve
```

The pinned documentation dependencies are separate from Victor's runtime
dependencies. API pages use static source analysis. The existing Pygments 2.18
pin preserves the formatter workaround for the `filename=None` interaction
between signature highlighting and Pygments 2.19; update it only after validating
the generated API pages.

Review the build diagnostics and browse the generated site before opening a PR
against `develop`. The workflow builds documentation for relevant PRs targeting
both `develop` and `main`, including changes to the source-backed API pages,
version stamp and build hooks.

## Navigation and historical records

`mkdocs.yml` owns the site navigation; `docs/index.md` is its home page.
`docs/README.md` remains the repository documentation index and is explicitly
excluded from the build to avoid competing for the same output URL.

The Audits navigation section exposes the co-design review and documentation
inventory. Their historical status remains visible. Root `feps/` proposals are
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

The checker returns a nonzero exit code for missing destinations or anchors.
The workflow initially treats this step as advisory with `continue-on-error`,
while the site build remains required for deployment. Review any findings in the
Actions log; do not interpret a successful workflow as proof that advisory checks
passed. Once the existing-site baseline is consistently clean, a separate change
can make the link check blocking.

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
