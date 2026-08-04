# Victor Development Workflow: PR-Based Process

## Overview

This document describes the new PR-based development workflow for the Victor project, designed to ensure code quality through CI/CD validation before merging to main.

## Why This Change?

**Previous State**: Commits were pushed directly to main, bypassing required status checks.
- ❌ No CI validation before production code
- ❌ No code review process
- ❌ Risk of breaking main branch

**New State**: All changes go through PRs with **branch-targeted** CI/CD validation.
- ✅ **PR → `develop`** runs a lightweight gate (`ci-fast`: Black, Ruff, repo-hygiene, MyPy,
  import checks, guards, plus **changed-file unit tests** via `scripts/ci/select_changed_tests.py`
  — only the mirror tests for the files the PR touches). Fast feedback that keeps the runner
  queue moving.
- ✅ **PR → `main`** (the `develop` → `main` promotion) runs the **extensive** battery: sharded
  unit tests, integration, build, security, performance, and validation workflows.
- ✅ `main` is strict — required checks **and `enforce_admins=true`** (no admin bypass); merges
  only when every required check passes. See `.github/workflows/README.md` for the full gating map.

## Branch Structure

```
main (protected)
  └── All production code, fully tested
  └── Requires PR + status checks to merge

develop (integration)
  └── Active development branch
  └── Forked from main, PRs into main
  └── Continuously rebased on main

feature/* (optional)
  └── Short-lived feature branches
  └── PR into develop for early integration
```

## Git Worktrees (Required for Feature Work)

**Mandate: all `feature/*`, `fix/*`, `refactor/*`, `docs/*`, and `chore/*` work
happens in a linked git worktree — NOT by checking the branch out in the main
working tree.** Only the integration branches (`main`, `develop`) live in the
main tree.

**Why.** The main working tree has a single `HEAD` shared by every session,
terminal, and agent operating in that directory. A `git checkout <feature>`
mid-task switches that shared `HEAD` and can silently revert another session's
in-flight work, or land a commit on the wrong branch. (This has happened: a
checkout moved the active branch while another session had uncommitted/committed
work, reverting files and misrouting a commit onto `develop`.) Linked worktrees
give each task its own working directory **and** its own `HEAD`, so a checkout in
one can never disturb another.

**Recommended layout** (one sibling worktree per task):

```bash
# from the main tree, on develop, up to date:
git checkout develop && git pull origin develop

# create a worktree + branch for the task (branch slashes -> dashes in the dir)
git worktree add ../victor-feature-my-feature -b feature/my-feature
cd ../victor-feature-my-feature

# ... work, commit, push, open PR ...
git push -u origin feature/my-feature

# when the PR merges, clean up:
cd /Users/vijaysingh/code/codingagent
git worktree remove ../victor-feature-my-feature
git branch -d feature/my-feature   # safe after merge
```

**Guard.** A `post-checkout` hook (`.githooks/post-checkout`) warns whenever a
feature branch is checked out in the main tree, reminding you to use a worktree.
Install it once per clone:

```bash
make hooks        # copies .githooks/* into .git/hooks/ (pre-commit hooks preserved)
```

The guard is advisory (it never blocks a checkout) and can be bypassed with
`VICTOR_SKIP_WORKTREE_GUARD=1`. Integration branches (`main`/`develop`) and
file-checkouts do not trigger it.

## Workflow

### 1. Start New Work

**Prefer a worktree** (see [Git Worktrees](#git-worktrees-required-for-feature-work) above):

```bash
# from the main tree, on develop:
git checkout develop && git pull origin develop
git worktree add ../victor-feature-your-feature -b feature/your-feature-name
cd ../victor-feature-your-feature
```

For a tiny change directly in the main tree (the guard will warn — bypass with
`VICTOR_SKIP_WORKTREE_GUARD=1`):

```bash
git checkout develop && git pull origin develop
git checkout -b feature/your-feature-name
```

### 2. Make Changes

```bash
# Make your changes
git add .
git commit -m "feat: your commit message"
```

### 3. Push to Feature Branch

```bash
# Push feature branch to origin
git push origin feature/your-feature-name
```

### 4. Create Pull Request

```bash
# Create PR: feature/your-feature-name -> develop
gh pr create \
  --title "feat: add your feature" \
  --body "Description of your changes" \
  --base develop \
  --head feature/your-feature-name
```

### 5. After Review: Merge to Develop

Once PR is approved and all checks pass:

```bash
# Squash merge to develop (keeps history clean)
gh pr merge --squash --delete-branch
```

### 6. Prepare Main Merge

When develop is ready for production:

```bash
# Update develop with latest main (resolve conflicts)
git checkout develop
git fetch origin main
git rebase origin/main
git push origin develop --force-with-lease
```

### 7. Create PR to Main

```bash
# Create PR: develop -> main
gh pr create \
  --title "Release: prepare for version X.Y.Z" \
  --body "Summary of changes in this release" \
  --base main \
  --head develop
```

### 8. Merge to Main

Once all status checks pass and PR is approved:

```bash
# Merge to main (requires status checks)
gh pr merge --squash
```

## Required Status Checks

Before any PR can merge to main, all checks must pass:

1. **Lint** - Code formatting and linting (ruff, black)
2. **Test (Python 3.11)** - Unit tests on Python 3.11
3. **Test (Python 3.12)** - Unit tests on Python 3.12
4. **Test (Python 3.12)** - Unit tests on Python 3.12
5. **Security Scan** - Security vulnerability scanning
6. **Build Package** - Package builds successfully

## Branch Protection Rules

### Main Branch

- ✅ Required status checks (strict mode)
  - All 6 checks must pass
  - Branch must be up-to-date with target branch
- ❌ Force pushes disabled
- ❌ Deletions disabled
- ⚠️ Admin enforcement: disabled (admins can bypass - be careful!)

### Develop Branch

- No protection rules (for development flexibility)

## Best Practices

### Commit Messages

Follow conventional commits format:

```bash
feat: add new feature
fix: fix bug
docs: update documentation
refactor: code refactoring
test: add tests
chore: maintenance tasks
perf: performance improvements
security: security fixes
```

### Branch Naming

```bash
feature/your-feature-name    # New features
fix/your-bug-fix            # Bug fixes
docs/your-doc-update        # Documentation
refactor/your-refactor      # Code refactoring
```

### PR Titles

```bash
feat: add observability dashboard
fix: resolve circular dependency
docs: add PR workflow guide
release: prepare for v0.6.0
```

### PR Descriptions

Use the PR template:

```markdown
## Summary
Brief description of changes

## Changes
- Change 1
- Change 2
- Change 3

## Testing
- Unit tests added/updated
- Manual testing performed
- All status checks passing

## Related Issues
Closes #123
Related to #456
```

## Common Scenarios

### Scenario 1: Quick Bug Fix

```bash
# Start from develop
git checkout develop
git pull origin develop

# Create feature branch
git checkout -b fix/urgent-bug

# Make fix
git add .
git commit -m "fix: resolve critical bug"
git push origin fix/urgent-bug

# Create PR to develop
gh pr create --base develop --head fix/urgent-bug

# After approval, merge to develop
gh pr merge --squash --delete-branch

# Then create PR from develop to main when ready
```

### Scenario 2: Large Feature

```bash
# Create feature branch from develop
git checkout develop
git pull origin develop
git checkout -b feature/large-feature

# Work on feature over multiple commits
git add .
git commit -m "feat: add first part of feature"
# ... more work ...
git add .
git commit -m "feat: add second part of feature"

# Push and create PR to develop
git push origin feature/large-feature
gh pr create --base develop

# Get feedback, iterate
# When approved, merge to develop
gh pr merge --squash

# Repeat for additional parts
```

### Scenario 3: Release to Main

```bash
# Ensure develop is ready
git checkout develop
git pull origin develop

# Rebase on main to get latest changes
git fetch origin main
git rebase origin/main
# Resolve any conflicts
git push origin develop --force-with-lease

# Create PR to main
gh pr create --base main --head develop

# All checks must pass before merging
```

## CI/CD Pipeline

### GitHub Actions Workflows

The repository uses GitHub Actions for CI/CD:

- `.github/workflows/lint.yml` - Linting
- `.github/workflows/test.yml` - Testing (Python 3.11, 3.12)
- `.github/workflows/security.yml` - Security scanning
- `.github/workflows/build.yml` - Package building

### Local Testing

Before pushing, run tests locally:

```bash
# Run linting
make lint

# Run tests
make test

# Run specific test file
pytest tests/unit/path/to/test_file.py -v

# Run with coverage
pytest --cov=victor tests/unit/
```

## Troubleshooting

### Status Checks Failing

1. **Lint Failed**
   ```bash
   make format  # Auto-format code
   make lint    # Check for remaining issues
   ```

2. **Tests Failed**
   ```bash
   make test  # Run tests locally
   # Fix failing tests
   ```

3. **Security Scan Failed**
   ```bash
   # Check for vulnerabilities
   pip install safety
   safety check
   ```

4. **Build Failed**
   ```bash
   make build  # Test build locally
   ```

### Merge Conflicts During Rebase

```bash
# During rebase, if conflicts occur:
git status  # See conflicts
# Edit conflicted files
git add conflict_file.py
git rebase --continue

# If too complex, abort and try merge instead
git rebase --abort
git merge origin/main  # Merge instead of rebase
# Resolve conflicts
git push origin develop
```

### Force Push Safety

Use `--force-with-lease` instead of `-f` to avoid destroying others' work:

```bash
# Good: checks if remote has new commits
git push --force-with-lease

# Bad: dangerous, can lose work
git push -f
```

## Current State

As of 2026-02-28:

- ✅ Main branch has strict protection rules
- ✅ Develop branch rebased on main
- ✅ All recent commits on both branches
- ✅ Ready for PR-based workflow

## Migration Summary

| Before | After |
|--------|-------|
| Direct commits to main | PRs from develop to main |
| No CI validation | All 6 status checks required |
| No code review | PR reviews required |
| Bypassed protections | Enforced protections |
| Linear history on main | Squash merge for clean history |

## Quick Reference

```bash
# Start new work
git checkout develop && git pull origin develop
git checkout -b feature/my-feature

# After commits, create PR
git push origin feature/my-feature
gh pr create --base develop

# After approval, merge to develop
gh pr merge --squash --delete-branch

# When ready for production
git checkout develop && git rebase origin/main && git push --force-with-lease
gh pr create --base main --head develop

# After checks pass, merge to main
gh pr merge --squash
```

---

**Version**: 1.0
**Created**: 2026-02-28
**Last Updated**: 2026-02-28
