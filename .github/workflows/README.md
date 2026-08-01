# GitHub Actions Workflows

This directory contains GitHub Actions workflows for automated validation, testing, and CI/CD.

## CI Gating Strategy (lightweight `develop`, extensive `develop` → `main`)

To keep the runner queue free and feature work merging fast, CI is split by the PR's
**target branch**:

| Trigger | Workflows that run | Intent |
|---------|-------------------|--------|
| **PR → `develop`** (and pushes to `develop`) | `ci-fast` only — Black, Ruff, repo-hygiene, MyPy (advisory + strict), import checks, guards, and **changed-file unit tests** (only the mirror tests for files the PR touches — see `scripts/ci/select_changed_tests.py`) + `fep-validation` if `feps/**` changed | **Lightweight** — fast feedback + a regression signal proportional to the change, unblock development |
| **PR → `main`** (the `develop` → `main` promotion PR) and pushes to `main` | The full battery: `ci`, `ci-test` (sharded units), `ci-integration`, `build`, `security`, `performance-tests`, `validation`, `vertical-validation`, `external-vertical-compat` | **Extensive** — full verification before promoting to the protected branch |

> The develop gate runs only the unit tests **relevant to the PR's changed files**, not the whole suite: the full non-slow unit suite is ~7h single-process (26.8k tests) and hours even sharded, so it can't gate every PR. Full coverage runs once at **develop → main**.
>
> On the **develop → main** promotion PR the changed-file unit job is **skipped** (`ci-fast.yml` → `quick-tests` `if: base_ref != 'main'`): there the diff spans the whole release delta and would map to the entire mirror-test set, blowing the job's timeout every time. Coverage isn't lost — the full sharded suite runs on that same PR via `ci-test`.

This is enforced by the `branches:` filter on each workflow's `push`/`pull_request`
triggers: heavy workflows target `[main]`. `main` remains the strict, protected branch
(required checks + admin enforcement), so the extensive suite always gates a release-bound
merge. To run a heavy workflow against a `develop` PR on demand, use its
**workflow_dispatch** entry.

`ci-fast` deliberately has **no `branches:` filter on `pull_request`** — it runs
on every PR whatever its base. Restricting it to `[main, develop]` left stacked
PRs (base = another feature branch) with *no gate at all*: no Black, Ruff,
strict MyPy, guards, or `CI Success`. PRs #700–#702 showed 1–2 reported checks
instead of 24 and read as green while being essentially ungated, only getting a
real signal once retargeted to `develop` — after review had already happened.
`CI Success` is only *required* on `develop`/`main`, so reporting it on a
feature-base PR is informational and blocks nothing.

### Two things the develop gate does not do

**It does not prove your branch works against current `develop`.** Branch
protection has `strict_up_to_date: false`, so a PR can merge with CI that ran
against an older `develop`. GitHub's `CLEAN`/`MERGEABLE` means "no merge
conflict", *not* "tested against the tip". Before merging anything non-trivial,
merge `develop` in and let CI re-run — especially if another PR touching the
same modules landed meanwhile.

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

### CI (`ci.yml`)

Main continuous integration workflow:
- Linting (Black, Ruff, MyPy)
- Unit tests across Python 3.11, 3.12, 3.13
- Security scanning (Gitleaks, pip-audit)
- Guards (legacy code protection, import checks)
- Rust native extensions build
- VS Code extension build
- Package build

**Triggers:** Push to `main`, pull requests targeting `main` (the `develop` → `main` promotion PR). See [CI Gating Strategy](#ci-gating-strategy-lightweight-develop-extensive-develop--main).

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
        uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Set up Python
        uses: actions/setup-python@v5
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
        uses: actions/github-script@v7
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
  uses: actions/upload-artifact@v4
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
