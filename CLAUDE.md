# CLAUDE.md

Victor - enterprise AI coding assistant (Python). This file guides Claude Code when working in this repository.

# Branching & Releases

- **`develop`** is the integration branch: feature branches open PRs into `develop` (CI-gated).
- **Promotion is explicit**: `develop` -> `main` via a release PR (CI-gated); releases are tagged on `main`.
- `main` always reflects the last promoted, releasable state.