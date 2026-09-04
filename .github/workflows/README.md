# GitHub Actions Workflows

This directory contains GitHub Actions workflows for automated validation, testing, and CI/CD.

## CI Gating Strategy (lightweight `develop`, extensive `develop` → `main`)

To keep the runner queue free and feature work merging fast, CI is split by the PR's
**target branch**:

| Trigger | Workflows that run | Intent |
|---------|-------------------|--------|
| **PR → `develop`** (and pushes to `develop`) | `ci-fast` plus always-present contracts, codegraph, and vertical summaries. Core Python changes run mirror tests with **80% changed-line coverage**; unmapped source and zero-test selections fail closed. | **Proportional** — focused core tests plus complete modular-package gates |
| **PR → `main`** (the `develop` → `main` promotion PR) | The full battery: all 36 Python-version/shard jobs, aggregate coverage, integration, builds, security, performance, validation, contracts, codegraph, verticals, Rust, and extension tests | **Extensive** — full verification before promoting to the protected branch |
| **Push to `main`** (the protected merge) | Delivery and lightweight verification workflows such as `build`, `security`, docs deployment, and `ci-fast` | **Post-merge** — publish/deploy from the merge SHA without repeating the identical multi-hour test tree |

> The core develop gate runs only unit tests **relevant to the PR's changed files**, not the whole suite: the full non-slow unit suite collects 29,834 tests and takes hours even sharded. A core source file without a mirrored test is rejected, and selected tests must cover at least 80% of changed lines. Complete suite coverage runs once at **develop → main**.
>
> On the **develop → main** promotion PR the changed-file unit job is **skipped** (`ci-fast.yml` → `quick-tests` `if: base_ref != 'main'`): there the diff spans the whole release delta and would map to the entire mirror-test set, blowing the job's timeout every time. Coverage isn't lost — the full sharded suite runs on that same PR via `ci-test`.

This is enforced by the `branches:` filter on each workflow's `push`/`pull_request`
triggers: heavy workflows target `[main]`. `main` remains the strict, protected branch;
administrators retain a recovery bypass but the operational rule is to merge only after every
required aggregate is green. To run a heavy workflow against a `develop` PR on demand, use its
**workflow_dispatch** entry.

`Test Summary` and `Coverage Gate` fail unless every `CI - Tests` shard succeeds. Their
`pull_request` trigger intentionally has no path filter. A skipped required
workflow remains expected forever, and a manual dispatch cannot satisfy the
PR-specific required contexts. The suite has no `push` trigger: protected
merge commits have the exact tree already tested on their promotion PR, so
repeating all 36 shards post-merge only consumes runner hours. Maintainers can
still use `workflow_dispatch` for an explicit on-demand rerun.

`ci-fast` deliberately has **no `branches:` filter on `pull_request`** — it runs
on every PR whatever its base. Restricting it to `[main, develop]` left stacked
PRs (base = another feature branch) with *no gate at all*: no Black, Ruff,
strict MyPy, guards, or `CI Success`. PRs #700–#702 showed 1–2 reported checks
instead of 24 and read as green while being essentially ungated, only getting a
real signal once retargeted to `develop` — after review had already happened.
`CI Success` is only *required* on `develop`/`main`, so reporting it on a
feature-base PR is informational and blocks nothing.

### Two things the develop gate does not do

**Branches must be tested against current `develop`.** Branch protection uses strict
up-to-date status checks. A stale PR must update/rebase and rerun the required summaries before
merge; GitHub's `CLEAN`/`MERGEABLE` alone is not treated as a test signal.

### Measured test and coverage gates

| Surface | Gate |
|---------|------|
| Core Python | 29,834 tests collected per supported Python; all 12 Python 3.11 coverage shards must be present; complete line coverage cannot fall below 40% |
| Changed core Python | A mirror test is mandatory and changed-line coverage must be at least 80% |
| victor-contracts | 337 tests across Python 3.10–3.13; line coverage cannot fall below 50%; wheel build/install smoke test |
| victor-codegraph | 106 tests across Python 3.10–3.12; line coverage cannot fall below 90% |
| Verticals | Isolated suites on Python 3.11/3.12, per-package collection-count floors, and measured line-coverage ratchets: coding 33%, devops 80%, RAG 20%, data-analysis 75%, research 80% |
| Rust workspace | 250 unit/doc tests, 68% line-coverage ratchet (69.26% measured), formatting, locked workspace test, and publishable-crate archive validation |
| VS Code extension | Measured Vitest/c8 ratchets (4% lines, 18% functions, 60% branches) plus blocking Electron/package integration in the main promotion gate |

Floors are ratchets, not targets. Raise them after adding coverage; lowering one requires an
explicitly reviewed change to the gate and its rationale.

**It does not run merged PRs against each other.** Each PR is verified against
the `develop` of its moment and nothing re-runs afterwards, so a set of
individually green PRs is first executed as a unit at `develop` → `main`.
`ci-develop-nightly.yml` runs the sharded unit suite against `develop` every
night to move that discovery from "at promotion, across a release delta" to
"within a day, across one day's merges". It is intentionally **not** a required
check — it is a signal, not a block.

## Validation Workflows

### FEP Validation (`fep-validation.yml`)

**Triggers:**
- Pull requests that modify files in `feps/` directory
- Manual workflow dispatch

**What it does:**
1. Finds all modified/added FEP markdown files in the PR
2. Runs `victor fep validate` on each FEP file
3. Validates YAML frontmatter syntax and required fields
4. Checks all required sections exist
5. Verifies section content quality (word counts)
6. Validates FEP numbering consistency (filename matches frontmatter)
7. Posts validation results as a PR comment

**Validation checks:**
- ✓ YAML frontmatter syntax
- ✓ Required metadata fields (fep, title, type, status, created, modified, authors)
- ✓ Required sections (Summary, Motivation, Proposed Change, etc.)
- ✓ Section content quality (minimum word counts)
- ✓ FEP numbering consistency

**Exit behavior:**
- Fails the CI check if any FEP fails validation
- Provides detailed error messages in the PR comment
- Uploads validation report as an artifact

### Vertical Package Validation (`vertical-validation.yml`)

**Triggers:**
- Pull requests that modify files in `victor/` directory
- Manual workflow dispatch

**What it does:**
1. Finds all vertical directories with `victor-vertical.toml` changes
2. Validates TOML metadata against `VerticalPackageMetadata` schema
3. Verifies vertical class exists and inherits from `VerticalBase`
4. Checks class can be imported and instantiated
5. Validates `pyproject.toml` entry points (if applicable)
6. Checks that provided tools exist in the tool registry

**Validation checks:**
- ✓ victor-vertical.toml schema validation
- ✓ Vertical class exists and inherits from VerticalBase
- ✓ Class can be imported and instantiated
- ✓ pyproject.toml entry points (for external packages)
- ✓ Provided tools validation

**Exit behavior:**
- Fails the CI check if validation fails
- Provides detailed error messages
- Uploads validation report as an artifact

### PR Comment Helper (`pr-comment.yml`)

**Triggers:**
- Pull requests (opened, synchronized, reopened)
- Can be called from other workflows

**What it does:**
1. Detects PR type (FEP, vertical, or code changes)
2. Posts helpful welcome comment on new PRs
3. Provides context-specific guidance based on PR content
4. Updates validation status on PR updates
5. Posts combined validation summaries from other workflows

**Comment sections:**
- Welcome message and next steps
- FEP-specific guidance (if FEP changes detected)
- Vertical-specific guidance (if vertical changes detected)
- Code change guidance (if code changes detected)
- Links to relevant documentation

## Other Workflows

### Packages CI (`packages.yml`)

Build and test modular packages:
- Test each package in isolation
- Build packages
- Integration tests
- Publish to TestPyPI (develop branch)
- Publish to PyPI (version tags)

**Triggers:** Changes to `packages/` directory

### Release (`release.yml`)

Automated release workflow:
- Create release notes
- Build and publish packages
- Create GitHub release

**Triggers:** Version tags (v*)

### Rust crates release (`publish-crates.yml`)

Publishes the modular Rust workspace to crates.io:
- Validate the `rust-vX.Y.Z` tag against the workspace version
- Test the complete locked workspace
- Publish `victor-protocol`, `victor-state`, `victor-tools`, and `victor-edge` in dependency order
- Wait for each registry record before publishing its dependents
- Install `victor-edge` back from crates.io as a public-consumer smoke test

**Triggers:** Rust version tags (`rust-v*`) or a manual dispatch for an existing tag

## Usage Examples

### Validating FEPs Locally

Before pushing, validate your FEP locally:

```bash
# Validate a single FEP
victor fep validate feps/fep-0002-my-feature.md

# Create a new FEP from template
victor fep create --title "My Feature" --type standards

# List all FEPs
victor fep list --status draft
```

### Validating Verticals Locally

Before pushing, validate your vertical metadata:

```bash
# Python script to validate vertical TOML
python - <<EOF
from pathlib import Path
from victor.core.verticals.package_schema import VerticalPackageMetadata

metadata = VerticalPackageMetadata.from_toml(
    Path("victor/coding/victor-vertical.toml")
)
print(f"✓ Valid: {metadata.name} v{metadata.version}")
EOF
```

### Testing Workflows Locally

Use [act](https://github.com/nektos/act) to test GitHub Actions locally:

```bash
# Install act
brew install act  # macOS

# Run FEP validation workflow
act -W .github/workflows/fep-validation.yml pull_request

# Run vertical validation workflow
act -W .github/workflows/vertical-validation.yml pull_request
```

## Workflow Features

### Caching

Workflows use GitHub Actions caching for faster runs:
- Python pip dependencies
- Cargo registry (Rust)
- Node modules (VS Code extension)

### Concurrency Control

All workflows use concurrency groups to cancel in-progress runs:
```yaml
concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true
```

This saves CI resources and provides faster feedback.

### Matrix Testing

Test across multiple Python versions:
```yaml
strategy:
  matrix:
    python-version: ["3.11", "3.12"]
```

### Artifact Upload

Validation reports are uploaded as artifacts with 30-day retention:
- `fep-validation-report`
- `vertical-validation-report`

### PR Comments

Workflows post detailed comments on PRs with:
- Validation results
- Error messages
- Next steps for contributors
- Links to documentation

## Adding New Workflows

When adding new workflows:

1. **Use standard naming**: `kebab-case.yml`
2. **Add concurrency control**: Prevent resource waste
3. **Cache dependencies**: Speed up runs
4. **Post helpful comments**: Keep contributors informed
5. **Use matrix testing**: Test across Python versions
6. **Set timeouts**: Prevent hanging jobs
7. **Upload artifacts**: Keep reports for debugging

### Workflow Template

```yaml
name: My Workflow

on:
  pull_request:
    paths:
      - 'relevant/path/**'
  workflow_dispatch:

concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

jobs:
  my-job:
    name: My Job
    runs-on: ubuntu-latest
    timeout-minutes: 10

    steps:
      - name: Checkout repository
        uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7.0.1
        with:
          fetch-depth: 0

      - name: Set up Python
        uses: actions/setup-python@5fda3b95a4ea91299a34e894583c3862153e4b97 # v7.0.0
        with:
          python-version: "3.11"
          cache: pip

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[dev]"

      - name: Run validation
        run: |
          # Your validation commands here

      - name: Post results
        if: always()
        uses: actions/github-script@3a2844b7e9c422d3c10d287c895573f7108da1b3 # v9.0.0
        with:
          script: |
            // Post PR comment with results
```

## Troubleshooting

### Workflow Not Triggering

Check path filters:
```yaml
on:
  pull_request:
    paths:
      - 'feps/**'  # Must match changed files
```

### Validation Fails Locally But Passes in CI

Check Python version:
```bash
# CI uses Python 3.11
python3.11 -m victor fep validate fep.md
```

### PR Comments Not Posting

Check permissions:
```yaml
permissions:
  pull-requests: write  # Required for posting comments
```

### Artifacts Not Downloading

Check artifact retention and names:
```yaml
- name: Upload artifact
  uses: actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a # v7.0.1
  with:
    name: my-artifact  # Must match download name
    path: path/to/files
    retention-days: 30
```

## Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Victor Documentation](https://docs.victor.dev)
- [FEP Process](../../feps/README.md)
- [Contributing Guide](../../CONTRIBUTING.md)
